"""LLM orchestrator — picks a tool (or speaks plain text) given voice + optional screenshot.

Provider is set by FRIDAY_LLM in .env:
  gemini → gemini-3.1-flash-lite-preview via Google's OpenAI-compatible endpoint
  openai → GPT-4o
  claude → Claude Haiku 4.5

`tool_choice="auto"` so the LLM may answer with plain text — that becomes the
spoken response with no extra hop.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable, Optional

from friday import config

log = logging.getLogger(__name__)

_ROUTING_RULES = """\
You receive a voice transcript of what the user said. You may also receive a screenshot if one was captured.

Your job: decide which tool (if any) to call, or just answer directly.

Decision rules:
- If the user references something on their screen ("look at this", "what's on my screen", "this code", "this email", "read this") AND no screenshot is present → take_screenshot
- For current events, recent releases, prices, or facts you don't know → web_search
- If the user says "remember this", "keep in mind", "don't forget", or states a strong preference → save_memory
- If the user asks "do you remember", "what did we talk about" → memory_search
- If the user wants to find files by name or extension (list Python files, all markdown in a folder) AND they name or imply a specific directory → find_files with that path and a glob_pattern
- If the user wants to search inside files for text or code (grep-style: find a function name, TODO, string) AND they name or imply a specific directory → search_files with that path and a regex pattern
- If the user asks to open a specific file path or says to open one of the files you just found → open_file
- If they want to search their whole computer, "everywhere", or all files without naming a folder → do NOT use find_files or search_files. Respond in plain text and ask which folder to search (e.g. their project path or Desktop).
- For everything else (factual questions, explanations, reasoning, opinions, conversation) → respond with plain text. The text you write will be spoken aloud directly.

When a screenshot IS present:
- Use what you see to inform your answer
- Most of the time you'll just answer in plain text using the visual context

Style for plain-text answers (these get spoken aloud):
- Under 3 sentences for a natural voice feel
- Be direct and conversational
- No markdown, no bullet points, no URLs

For tools that have a `thinking` field, fill it with a natural, brief phrase that acknowledges what you're about to do. It will be spoken immediately while the tool runs."""


def _build_system_prompt(memory_context: str | None = None) -> str:
    parts = []
    if memory_context:
        parts.append(memory_context)
    if config.FILE_SEARCH_DEFAULT_ROOT is not None:
        parts.append(
            "When the user asks to search their project or files without naming a path, "
            f"use this default directory for find_files and search_files `path`: "
            f"{config.FILE_SEARCH_DEFAULT_ROOT}"
        )
    parts.append(_ROUTING_RULES)
    return "\n\n".join(parts)


def _truncate_tool_result(text: str, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 64] + "\n\n[Tool output truncated]"


def _assistant_message_for_history(msg: object, tool_calls: list | None) -> dict:
    out: dict = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


async def run_tool_loop(
    transcript: str,
    history: Optional[list[dict]] = None,
    memory_context: Optional[str] = None,
    dispatch_tool: Callable[[str, dict], Awaitable[str]] | None = None,
    speak_thinking: Callable[[str], Awaitable[None]] | None = None,
    capture_screenshot: Callable[[], Awaitable[str]] | None = None,
    max_steps: int = 6,
) -> tuple[str, list[dict]]:
    """Run assistant/tool loop until end-turn; return spoken text + new history messages."""
    from openai import AsyncOpenAI

    from friday.tools import TOOL_DEFINITIONS

    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    messages: list[dict] = [{"role": "system", "content": _build_system_prompt(memory_context)}]
    if history:
        messages.extend(history)
    user_msg = {"role": "user", "content": f"User said: {transcript}"}
    messages.append(user_msg)
    new_history: list[dict] = [user_msg]

    t0 = time.monotonic()
    log.debug(
        "Calling %s/%s (tool-loop, transcript=%r)",
        config.LLM_PROVIDER, cfg["model"], transcript,
    )

    for step in range(max_steps):
        response = await client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=1024,
        )
        log.info("%s responded in %.0f ms (step %d)", cfg["model"], (time.monotonic() - t0) * 1000, step + 1)

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        assistant_msg = _assistant_message_for_history(msg, tool_calls)
        messages.append(assistant_msg)
        new_history.append(assistant_msg)

        if not tool_calls:
            text = (getattr(msg, "content", "") or "").strip()
            log.info("Plain-text answer: %r", text[:120])
            return text, new_history

        if dispatch_tool is None:
            raise RuntimeError("dispatch_tool callback is required for tool loops")

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            thinking = arguments.get("thinking")
            log.info(
                "Tool: %s  Thinking: %r  Args: %s",
                tool_name, thinking, json.dumps(arguments, ensure_ascii=False)[:200],
            )
            if thinking and speak_thinking:
                await speak_thinking(thinking)

            if tool_name == "take_screenshot" and capture_screenshot is not None:
                b64 = await capture_screenshot()
                tool_result = "Screenshot captured."
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
                messages.append(tool_msg)
                new_history.append(tool_msg)
                image_msg = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Screenshot captured. Use it to answer the user."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
                messages.append(image_msg)
                # Keep history small: store marker, not full image payload.
                new_history.append({"role": "user", "content": "[screenshot provided]"})
                continue

            raw_result = await dispatch_tool(tool_name, arguments)
            tool_result = _truncate_tool_result(raw_result)
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            }
            messages.append(tool_msg)
            new_history.append(tool_msg)

    return "I ran into too many tool steps and stopped.", new_history


async def synthesize_response(user_query: str, tool_result: str) -> str:
    """Convert raw tool output (search results, memory hits) into a natural spoken sentence."""
    from openai import AsyncOpenAI

    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a voice assistant. Summarise the supplied results into a "
                "natural, conversational spoken response. 2-3 sentences max. "
                "No markdown, no bullets, no URLs."
            ),
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nResults:\n{tool_result}",
        },
    ]

    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        max_tokens=256,
    )
    log.info("Synthesis responded in %.0f ms", (time.monotonic() - t0) * 1000)
    return response.choices[0].message.content or "I couldn't summarise that."
