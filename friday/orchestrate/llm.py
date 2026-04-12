"""LLM orchestrator — routes voice + screenshot to the right tool.

Provider is controlled by FRIDAY_LLM in .env:
  FRIDAY_LLM=gemini   → gemini-3.1-flash-lite-preview via Google's OpenAI-compatible endpoint (default)
  FRIDAY_LLM=openai   → GPT-4o
  FRIDAY_LLM=claude   → Claude Haiku 4.5

All three use the OpenAI SDK — just different base_url / api_key / model.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from friday import config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are Friday, a voice-first AI assistant running on the user's Mac.
You receive a voice transcript of what the user said. You may also receive a screenshot if one was captured.

Your job: decide which tool to call based on what you hear (and see, if a screenshot is present).

Decision rules:
- If the user references something on their screen ("look at this", "what's on my screen", "this code", "this email", "read this", "check this out") AND no screenshot is present → take_screenshot
- If the user says "email", "write to", "reply to", "message" → draft_gmail
- Use web_search when you need to look something up or lack necessary info (current events, recent releases, prices, unknown facts). If you can answer confidently from general knowledge, use speak_answer instead.
- For everything else (factual questions, explanations, opinions) → speak_answer

When a screenshot IS present:
- Use visual context to inform your tool choice and arguments
- If the screenshot shows Gmail/Outlook → draft_gmail
- Otherwise analyze what you see and respond via speak_answer

When using speak_answer:
- Keep responses under 3 sentences for natural voice interaction
- Be direct and conversational — this will be spoken aloud

For tools that have a `thinking` field, always fill it with a natural, brief phrase that acknowledges what you're about to do. This will be spoken aloud immediately. Keep it contextual, not generic.

Always call exactly one tool. Do not respond with plain text."""


async def plan_tool_call(
    transcript: str,
    screenshot_b64: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> tuple[str, dict, Optional[str]]:
    """Call the LLM and return (tool_name, arguments, thinking) WITHOUT executing the tool.

    Callers should immediately speak `thinking` and dispatch the tool in parallel,
    so the acknowledgement phrase plays during tool execution rather than after.
    """
    from openai import AsyncOpenAI
    from friday.tools.base import TOOL_DEFINITIONS

    cfg = config.llm_config()
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

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

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)  # inject last N turns of conversation context
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
        tool_choice="required",
        max_tokens=1024,
    )

    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info("%s responded in %.0f ms", cfg["model"], elapsed_ms)

    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    thinking = arguments.get("thinking") if tool_name != "speak_answer" else None

    log.info("Tool: %s  Thinking: %r  Args: %s", tool_name, thinking, json.dumps(arguments, ensure_ascii=False)[:200])

    return tool_name, arguments, thinking


async def synthesize_response(user_query: str, tool_result: str) -> str:
    """Convert raw web search results into a natural spoken response.

    Uses the same configured LLM without tools — just text generation.
    """
    from openai import AsyncOpenAI

    cfg = config.llm_config()
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a voice assistant. The user asked a question and you searched the web. "
                "Summarize the search results into a natural, conversational spoken response. "
                "2-3 sentences max. Be direct and specific. No markdown, no bullet points, no URLs. "
                "Speak as if you're answering in conversation."
            ),
        },
        {
            "role": "user",
            "content": f"User asked: {user_query}\n\nSearch results:\n{tool_result}",
        },
    ]

    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        max_tokens=256,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info("Synthesis responded in %.0f ms", elapsed_ms)

    return response.choices[0].message.content or "I couldn't summarize that result."
