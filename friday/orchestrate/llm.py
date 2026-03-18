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
You receive a screenshot of what the user is currently looking at and a voice transcript of what they said.

Your job: decide which tool to call based on what you see and hear.

Decision rules:
- If the screenshot shows a terminal with Claude Code running, and the user is asking about code → inject_claude_code
- If the screenshot shows Gmail/Outlook, or the user says "email" / "write to" / "reply" → draft_gmail
- If the user asks "what is", "find", "look up", "search", "latest", "who is" → web_search
- For everything else (factual questions, explanations, time, math, opinions) → speak_answer

When using inject_claude_code:
- Formulate a precise, self-contained prompt for Claude Code
- Reference specific code/context visible in the screenshot
- Don't just relay what the user said — translate it into an effective Claude Code prompt

When using speak_answer:
- Keep responses under 3 sentences for natural voice interaction
- Be direct and conversational — this will be spoken aloud

Always call exactly one tool. Do not respond with plain text."""


async def orchestrate(
    transcript: str,
    screenshot_b64: Optional[str] = None,
) -> tuple[str, str]:
    """Call the configured LLM with transcript + optional screenshot.

    Returns (tool_name, result). If tool is speak_answer, result is the text to speak.
    """
    from openai import AsyncOpenAI
    from friday.tools.base import TOOL_DEFINITIONS, dispatch_tool

    cfg = config.llm_config()
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],  # None = default OpenAI endpoint
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

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

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

    log.info("Tool: %s  Args: %s", tool_name, json.dumps(arguments, ensure_ascii=False)[:200])

    result = await dispatch_tool(tool_name, arguments)
    return tool_name, result
