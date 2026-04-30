"""Tool registry — small, flat list. The LLM picks one (or returns plain text).

Tools:
  take_screenshot — trigger second-pass routing with vision context
  web_search      — Tavily
  save_memory     — append a fact to MEMORY.md
  memory_search   — substring search over MEMORY.md
"""
from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": (
                "Capture a screenshot of what the user is currently looking at. "
                "Use this FIRST when the user references something visual on their screen — "
                "e.g. 'look at this', 'what's on my screen', 'this code', 'read this'. "
                "After calling, you'll receive the screenshot and decide what to do next."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": (
                            "Brief phrase to say aloud while capturing. "
                            "e.g. 'Let me take a look'. Keep under 8 words."
                        ),
                    },
                },
                "required": ["thinking"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use when you need current information you don't have — "
                "news, recent releases, prices, unknown facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Let me look that up'. Keep under 8 words.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                },
                "required": ["thinking", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save a fact or preference to long-term memory. "
                "Use when the user says 'remember', 'keep in mind', 'don't forget', "
                "or states a preference worth retaining across sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Noted'. Keep under 8 words.",
                    },
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written clearly.",
                    },
                },
                "required": ["thinking", "fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Search saved memories. Use when the user asks 'do you remember' "
                "or references something said earlier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Checking my notes'. Keep under 8 words.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                },
                "required": ["thinking", "query"],
            },
        },
    },
]


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call by name. Returns a string result."""
    arguments = {k: v for k, v in arguments.items() if k != "thinking"}

    if name == "take_screenshot":
        # Handled by the loop's two-pass logic; if we ever land here we just
        # capture and return raw b64 (loop will speak it as fallback).
        import asyncio

        from friday.capture.screenshot import capture_focused_display

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, capture_focused_display) or ""

    if name == "web_search":
        return await _web_search(arguments["query"])

    if name == "save_memory":
        from friday.memory import save_to_memory
        return save_to_memory(arguments["fact"])

    if name == "memory_search":
        from friday.memory import memory_search
        return memory_search(arguments["query"])

    # "speak" is the synthetic name we use when the LLM returned plain text;
    # the loop speaks `arguments["answer"]` directly.
    if name in ("speak", "speak_answer"):
        return arguments.get("answer", "")

    return f"Unknown tool: {name}"


# ── Web search ──────────────────────────────────────────────────────────────

async def _web_search(query: str) -> str:
    """Tavily search → formatted string. Synthesised into spoken form by the loop."""
    import asyncio
    import logging

    from friday import config

    log = logging.getLogger(__name__)
    log.info("Web search: %r", query)

    if not config.TAVILY_API_KEY:
        return f"Web search unavailable (no TAVILY_API_KEY). Query was: {query}"

    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: _tavily_sync(query)
        )
    except Exception as exc:
        log.error("Web search failed: %s", exc)
        return f"Web search failed: {exc}"


def _tavily_sync(query: str) -> str:
    from tavily import TavilyClient

    from friday import config

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    parts: list[str] = []
    if response.get("answer"):
        parts.append(response["answer"])
    for result in response.get("results", [])[:3]:
        title = result.get("title", "")
        content = result.get("content", "")[:300]
        url = result.get("url", "")
        parts.append(f"• {title}: {content} ({url})")
    return "\n".join(parts) if parts else "No results found."
