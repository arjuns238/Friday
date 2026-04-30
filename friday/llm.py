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
from typing import Optional

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
    parts.append(_ROUTING_RULES)
    return "\n\n".join(parts)


async def plan_tool_call(
    transcript: str,
    screenshot_b64: Optional[str] = None,
    history: Optional[list[dict]] = None,
    memory_context: Optional[str] = None,
) -> tuple[str, dict, Optional[str]]:
    """Call the LLM. Return (tool_name, arguments, thinking).

    If the LLM responds with plain text instead of a tool call, returns
    ("speak", {"answer": text}, None) so the caller speaks it directly.
    """
    from openai import AsyncOpenAI

    from friday.tools import TOOL_DEFINITIONS

    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    user_content: list[dict] = []
    if screenshot_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{screenshot_b64}",
                "detail": "high",
            },
        })
    user_content.append({"type": "text", "text": f"User said: {transcript}"})

    messages: list[dict] = [{"role": "system", "content": _build_system_prompt(memory_context)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    t0 = time.monotonic()
    log.debug(
        "Calling %s/%s (has_image=%s, transcript=%r)",
        config.LLM_PROVIDER, cfg["model"], bool(screenshot_b64), transcript,
    )

    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        max_tokens=1024,
    )
    log.info("%s responded in %.0f ms", cfg["model"], (time.monotonic() - t0) * 1000)

    msg = response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        tc = tool_calls[0]
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
        return tool_name, arguments, thinking

    text = (msg.content or "").strip()
    log.info("Plain-text answer: %r", text[:120])
    return "speak", {"answer": text}, None


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
