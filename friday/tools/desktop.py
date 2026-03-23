"""Desktop intelligence subagent — ReAct loop over macOS data sources.

Phase 1: spotlight_search + file_metadata + open_file
Phase 2: reverse_geocode (GPS → place names)
Phase 3: classify_image (Vision framework)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import date, timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Subagent tool schemas (OpenAI function-calling format) ────────────────────

_SUBAGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "spotlight_search",
            "description": (
                "Search files via macOS Spotlight (mdfind). Fast, uses the system index. "
                "Returns up to 50 file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text Spotlight query. Can use kMDItem attributes for precision, "
                            "e.g. 'kMDItemContentType == public.image && kMDItemFSName == *beach*'"
                        ),
                    },
                    "content_type": {
                        "type": "string",
                        "description": (
                            "UTI content type filter. Common: public.image, public.pdf, "
                            "com.adobe.pdf, public.movie, public.audio, public.plain-text"
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD). Filters kMDItemContentCreationDate >=",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD). Filters kMDItemContentCreationDate <=",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Filename glob pattern, e.g. '*.pdf' or '*resume*'",
                    },
                    "search_dir": {
                        "type": "string",
                        "description": "Directory scope (defaults to home). Use ~/Downloads, ~/Desktop, etc.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_metadata",
            "description": (
                "Get rich metadata for one or more files via mdls. Returns dates, size, "
                "content type, GPS coordinates, authors, download source, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute file paths (max 20).",
                    },
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "Open a file with its default application, or reveal it in Finder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                    "reveal": {
                        "type": "boolean",
                        "description": "If true, reveal in Finder instead of opening.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────


async def spotlight_search(
    query: str = "*",
    content_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    name_pattern: Optional[str] = None,
    search_dir: Optional[str] = None,
) -> str:
    """Run mdfind with structured filters. Returns newline-separated paths."""
    from friday import config

    predicates: list[str] = []

    if content_type:
        predicates.append(f'kMDItemContentType == "{content_type}"')
    if date_from:
        predicates.append(f'kMDItemContentCreationDate >= $time.iso("{date_from}T00:00:00Z")')
    if date_to:
        predicates.append(f'kMDItemContentCreationDate <= $time.iso("{date_to}T23:59:59Z")')
    if name_pattern:
        predicates.append(f'kMDItemFSName == "{name_pattern}"wc')

    # Build mdfind command
    cmd = ["mdfind"]

    # Scope to allowed directories
    scope = search_dir or config.DESKTOP_SEARCH_DIRS[0]
    scope = os.path.expanduser(scope)
    cmd.extend(["-onlyin", scope])

    if predicates:
        full_query = " && ".join(predicates)
        if query and not any(query.startswith("kMDItem") for _ in [1]):
            # Combine free-text with predicates
            full_query = f'({full_query}) && ({query})'
        cmd.append(full_query)
    else:
        cmd.append(query)

    log.info("Spotlight: %s", " ".join(cmd))

    loop = asyncio.get_running_loop()
    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=10),
        )
    except subprocess.TimeoutExpired:
        return "Search timed out after 10 seconds."

    if proc.returncode != 0:
        return f"mdfind error: {proc.stderr.strip()}"

    paths = [p for p in proc.stdout.strip().split("\n") if p]
    total = len(paths)
    paths = paths[:50]  # cap results

    if not paths:
        return "No files found matching the search criteria."

    result = "\n".join(paths)
    if total > 50:
        result += f"\n\n({total} total results, showing first 50)"
    return result


async def file_metadata(paths: list[str]) -> str:
    """Run mdls on each path, parse into structured dicts."""
    paths = paths[:20]  # safety cap
    loop = asyncio.get_running_loop()
    results: list[dict[str, Any]] = []

    for path in paths:
        if not os.path.exists(path):
            results.append({"path": path, "error": "File not found"})
            continue

        try:
            proc = await loop.run_in_executor(
                None,
                lambda p=path: subprocess.run(
                    ["mdls", p], capture_output=True, text=True, timeout=5
                ),
            )
        except subprocess.TimeoutExpired:
            results.append({"path": path, "error": "mdls timed out"})
            continue

        meta = _parse_mdls(proc.stdout)
        meta["path"] = path
        meta["name"] = os.path.basename(path)
        results.append(meta)

    return json.dumps(results, indent=2, default=str)


def _parse_mdls(output: str) -> dict[str, Any]:
    """Parse mdls output into a dict of useful fields."""
    fields: dict[str, Any] = {}
    interesting = {
        "kMDItemContentType",
        "kMDItemContentCreationDate",
        "kMDItemContentModificationDate",
        "kMDItemLastUsedDate",
        "kMDItemFSSize",
        "kMDItemAuthors",
        "kMDItemWhereFroms",
        "kMDItemLatitude",
        "kMDItemLongitude",
        "kMDItemAltitude",
        "kMDItemPixelWidth",
        "kMDItemPixelHeight",
        "kMDItemDurationSeconds",
        "kMDItemTitle",
        "kMDItemDisplayName",
        "kMDItemKind",
        "kMDItemDownloadedDate",
    }

    lines = output.strip().split("\n")
    i = 0
    while i < len(lines):
        match = re.match(r"^(\w+)\s+=\s+(.+)$", lines[i].strip())
        if not match:
            i += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if key not in interesting:
            i += 1
            continue
        if value == "(null)":
            i += 1
            continue
        # Handle multi-line array values: ( ... )
        if value == "(":
            array_items: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != ")":
                item = lines[i].strip().strip(",").strip('"')
                if item:
                    array_items.append(item)
                i += 1
            fields[key] = array_items
        else:
            fields[key] = value.strip('"')
        i += 1

    # Rename to friendlier keys
    rename = {
        "kMDItemContentType": "content_type",
        "kMDItemContentCreationDate": "created",
        "kMDItemContentModificationDate": "modified",
        "kMDItemLastUsedDate": "last_used",
        "kMDItemFSSize": "size_bytes",
        "kMDItemAuthors": "authors",
        "kMDItemWhereFroms": "downloaded_from",
        "kMDItemLatitude": "gps_lat",
        "kMDItemLongitude": "gps_lon",
        "kMDItemAltitude": "gps_alt",
        "kMDItemPixelWidth": "width",
        "kMDItemPixelHeight": "height",
        "kMDItemDurationSeconds": "duration_sec",
        "kMDItemTitle": "title",
        "kMDItemDisplayName": "display_name",
        "kMDItemKind": "kind",
        "kMDItemDownloadedDate": "downloaded_date",
    }
    return {rename.get(k, k): v for k, v in fields.items()}


def open_file(path: str, reveal: bool = False) -> str:
    """Open a file with default app or reveal in Finder."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"File not found: {path}"

    cmd = ["open", "-R", path] if reveal else ["open", path]
    try:
        subprocess.run(cmd, check=True, timeout=5)
        action = "Revealed in Finder" if reveal else "Opened"
        return f"{action}: {os.path.basename(path)}"
    except subprocess.CalledProcessError as e:
        return f"Failed to open: {e}"
    except subprocess.TimeoutExpired:
        return "open command timed out"


# ── Query pre-processing ─────────────────────────────────────────────────────

_CONTENT_TYPE_MAP = {
    "picture": "public.image",
    "photo": "public.image",
    "image": "public.image",
    "screenshot": "public.png",
    "pdf": "com.adobe.pdf",
    "document": "public.content",
    "video": "public.movie",
    "movie": "public.movie",
    "audio": "public.audio",
    "music": "public.audio",
    "presentation": "org.openxmlformats.presentationml.presentation",
    "spreadsheet": "org.openxmlformats.spreadsheetml.sheet",
}

_TEMPORAL_PATTERNS = [
    (r"\btoday\b", lambda: (date.today(), date.today())),
    (r"\byesterday\b", lambda: (date.today() - timedelta(days=1), date.today() - timedelta(days=1))),
    (r"\bthis week\b", lambda: (date.today() - timedelta(days=date.today().weekday()), date.today())),
    (r"\blast week\b", lambda: (
        date.today() - timedelta(days=date.today().weekday() + 7),
        date.today() - timedelta(days=date.today().weekday() + 1),
    )),
    (r"\bthis month\b", lambda: (date.today().replace(day=1), date.today())),
    (r"\blast month\b", lambda: (
        (date.today().replace(day=1) - timedelta(days=1)).replace(day=1),
        date.today().replace(day=1) - timedelta(days=1),
    )),
    (r"\blast summer\b", lambda: (
        date(date.today().year - 1, 6, 1),
        date(date.today().year - 1, 8, 31),
    )),
    (r"\blast year\b", lambda: (
        date(date.today().year - 1, 1, 1),
        date(date.today().year - 1, 12, 31),
    )),
]

_DIR_PATTERNS = {
    r"\bon (?:my )?desktop\b": "~/Desktop",
    r"\bin (?:my )?downloads?\b": "~/Downloads",
    r"\bin (?:my )?documents?\b": "~/Documents",
}


def _parse_query(query: str) -> dict[str, Any]:
    """Extract temporal refs, content types, and location hints from query."""
    hints: dict[str, Any] = {}
    q_lower = query.lower()

    # Content type detection
    for keyword, uti in _CONTENT_TYPE_MAP.items():
        if keyword in q_lower:
            hints["content_type"] = uti
            break

    # Temporal detection
    for pattern, date_fn in _TEMPORAL_PATTERNS:
        if re.search(pattern, q_lower):
            date_from, date_to = date_fn()
            hints["date_from"] = date_from.isoformat()
            hints["date_to"] = date_to.isoformat()
            break

    # Directory detection
    for pattern, directory in _DIR_PATTERNS.items():
        if re.search(pattern, q_lower):
            hints["search_dir"] = directory
            break

    # GPS-related queries
    if any(word in q_lower for word in ("where did i go", "location", "gps", "travel")):
        hints["needs_gps"] = True

    return hints


# ── Desktop Subagent ──────────────────────────────────────────────────────────


class DesktopAgent:
    """ReAct agent for desktop intelligence queries.

    Has its own LLM context and tool set. Runs a reason→act→observe loop
    until it has enough information to answer the user's question.
    """

    MAX_ITERATIONS = 5

    SYSTEM_PROMPT = """You are a desktop intelligence agent with access to macOS data sources.
Your job is to answer questions about the user's files, photos, and local data by searching and reasoning over metadata.

Available tools:
- spotlight_search: Search files via macOS Spotlight (fast, indexed). Supports content type filters, date ranges, name patterns.
- file_metadata: Get rich metadata for files (GPS, dates, size, authors, download source, etc.)
- open_file: Open a file with its default app or reveal in Finder.

Strategy:
1. Break the query into what data you need
2. Start with spotlight_search to find relevant files
3. Use file_metadata to extract details (GPS, dates, etc.) if needed
4. Synthesize a natural spoken answer (2-3 sentences max)

Be efficient — you have a maximum of 5 tool calls. Give concise, spoken answers."""

    def __init__(self) -> None:
        self._client: Any = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            from friday.config import desktop_llm_config
            cfg = desktop_llm_config()
            self._client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            self._model = cfg["model"]
        return self._client

    async def run(self, query: str) -> str:
        """Execute ReAct loop. Returns natural language answer."""
        import time
        t0 = time.monotonic()

        client = await self._get_client()
        hints = _parse_query(query)

        # Build initial message with hints
        user_msg = query
        if hints:
            hint_lines = [f"- {k}: {v}" for k, v in hints.items()]
            user_msg = f"{query}\n\nPre-extracted hints:\n" + "\n".join(hint_lines)

        messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for i in range(self.MAX_ITERATIONS):
            log.debug("Desktop subagent iteration %d", i + 1)

            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_SUBAGENT_TOOLS,
                max_tokens=1024,
            )

            choice = response.choices[0]

            # No tool calls → final answer
            if not choice.message.tool_calls:
                answer = choice.message.content or "I searched but couldn't find a clear answer."
                log.info(
                    "Desktop subagent done in %d iterations (%.0fms): %s",
                    i + 1, (time.monotonic() - t0) * 1000, answer[:80],
                )
                return answer

            # Process tool calls — exclude_none avoids null fields that
            # Gemini's OpenAI-compatible endpoint rejects
            messages.append(choice.message.model_dump(exclude_none=True))

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                log.info("Desktop subagent tool: %s(%s)", fn_name, json.dumps(fn_args, ensure_ascii=False)[:200])
                result = await self._execute_tool(fn_name, fn_args)
                log.debug("Desktop tool result: %s", result[:300])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        log.warning("Desktop subagent hit max iterations")
        return "I searched through your files but couldn't find a definitive answer. Could you be more specific?"

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Dispatch a subagent tool call."""
        try:
            if name == "spotlight_search":
                return await spotlight_search(**args)
            elif name == "file_metadata":
                return await file_metadata(**args)
            elif name == "open_file":
                return open_file(**args)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            log.exception("Desktop tool %s failed: %s", name, e)
            return f"Tool error: {e}"
