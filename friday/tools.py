"""Tool registry — small, flat list. The LLM picks one (or returns plain text).

Tools:
  take_screenshot — trigger second-pass routing with vision context
  web_search      — Tavily
  save_memory     — append a fact to MEMORY.md
  memory_search   — substring search over MEMORY.md
  find_files      — discover files by glob under a directory
  search_files    — search file contents with regex (ripgrep or Python fallback)
  open_file       — open/reveal a file path in Finder/default app
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

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
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "List files under a folder whose names match a glob pattern. "
                "Use for 'find all Python files', 'where are the markdown files', file discovery. "
                "Prefer the smallest directory path that answers the question (e.g. a project root, "
                "not the entire home folder unless the user clearly asked for that). "
                "If the user wants to search 'everywhere' or their whole computer without naming a "
                "folder, do NOT call this tool — answer in plain text and ask which directory to search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Let me find those'. Keep under 8 words.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute path to an existing directory to search under "
                            "(e.g. /Users/me/Projects/myapp)."
                        ),
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": (
                            "Glob for file names, e.g. '*.md', '**/*.py', or 'src/**/*.ts'. "
                            "Ripgrep semantics when ripgrep is installed."
                        ),
                    },
                    "head_limit": {
                        "type": "integer",
                        "description": "Max paths to return (default 80).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results after sorting by recency (default 0).",
                    },
                },
                "required": ["thinking", "path", "glob_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents under a directory using a regular expression (ripgrep syntax "
                "when rg is installed). Use for 'where is X defined', 'find TODO in my repo', "
                "grep-style questions. "
                "Prefer the smallest directory path that fits the question. "
                "If the user wants their whole disk or 'everywhere' without a specific folder, "
                "do NOT call this tool — answer in plain text and ask which directory to search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Searching your files'. Keep under 8 words.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to an existing directory to search under.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (ripgrep regex when rg is available).",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": (
                            "files_with_matches: list paths only (default). "
                            "content: matching lines with line numbers. "
                            "count: match counts per file."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob filter (e.g. '*.py') — only search matching paths.",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional ripgrep --type shorthand (e.g. py, rust, js). Ignored in Python-only fallback.",
                    },
                    "head_limit": {
                        "type": "integer",
                        "description": "Max lines or paths to return (default 80).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N lines or paths after sorting (default 0).",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive regex (default false).",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match (content mode only; rg only).",
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Allow patterns to span lines (default false).",
                    },
                },
                "required": ["thinking", "path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Open an existing file path with the default app, or reveal it in Finder. "
                "Use only when you already have a concrete file path from prior search results "
                "or from the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase. e.g. 'Opening that file'. Keep under 8 words.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to open.",
                    },
                    "reveal": {
                        "type": "boolean",
                        "description": "If true, reveal in Finder instead of opening (default false).",
                    },
                },
                "required": ["thinking", "path"],
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

    if name == "find_files":
        return await _find_files(arguments)

    if name == "search_files":
        return await _search_files(arguments)

    if name == "open_file":
        return await _open_file(arguments)

    # "speak" is the synthetic name we use when the LLM returned plain text;
    # the loop speaks `arguments["answer"]` directly.
    if name in ("speak", "speak_answer"):
        return arguments.get("answer", "")

    return f"Unknown tool: {name}"


async def _find_files(arguments: dict[str, Any]) -> str:
    from friday.file_search import find_files_sync, normalize_search_path

    raw_path = str(arguments.get("path", "")).strip()
    if raw_path:
        normalized = normalize_search_path(raw_path)
        if normalized != raw_path:
            log.info("find_files requested path normalized for dispatch: %r -> %r", raw_path, normalized)
        arguments = {**arguments, "path": normalized}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: find_files_sync(arguments))


async def _search_files(arguments: dict[str, Any]) -> str:
    from friday.file_search import normalize_search_path, search_files_sync

    raw_path = str(arguments.get("path", "")).strip()
    if raw_path:
        normalized = normalize_search_path(raw_path)
        if normalized != raw_path:
            log.info("search_files requested path normalized for dispatch: %r -> %r", raw_path, normalized)
        arguments = {**arguments, "path": normalized}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: search_files_sync(arguments))


async def _open_file(arguments: dict[str, Any]) -> str:
    from friday.file_search import open_file_sync

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: open_file_sync(arguments))


# ── Web search ──────────────────────────────────────────────────────────────

async def _web_search(query: str) -> str:
    """Tavily search → formatted string. Synthesised into spoken form by the loop."""
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
