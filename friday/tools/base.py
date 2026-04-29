"""Orchestrator-level tools exposed to the deepagents main agent.

Subagent-level tools (desktop, research, email) live in `friday.agents.*`.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from friday.tools.desktop import open_file
from friday.memory.context import save_memory, memory_search

log = logging.getLogger(__name__)


@tool
def speak_answer(answer: str) -> str:
    """Speak a direct answer to the user with no side effects.

    Use for factual questions, explanations, opinions, or acknowledgements where
    no subagent and no side-effect tool is needed. Keep to 2-3 sentences for voice.
    """
    return answer


@tool
async def take_screenshot() -> str:
    """Capture a screenshot of the user's focused display.

    Use when the user references something visible on screen ("look at this",
    "what's on my screen", "this code"). The screenshot is saved to
    /state/last_screenshot.png; this tool returns a human-readable status line.
    """
    import asyncio
    from friday.capture.screenshot import capture_focused_display

    loop = asyncio.get_running_loop()
    b64 = await loop.run_in_executor(None, capture_focused_display)
    if not b64:
        return "Failed to capture screenshot."
    return (
        f"Screenshot captured ({len(b64)} chars base64). "
        "Saved to /state/last_screenshot.png. "
        "Describe what you need to see; the assistant can re-capture if needed."
    )


ORCH_TOOLS = [
    save_memory,
    memory_search,
    take_screenshot,
    open_file,
    speak_answer,
]
