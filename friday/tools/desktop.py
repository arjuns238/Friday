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
    {
        "type": "function",
        "function": {
            "name": "filesystem_search",
            "description": (
                "Search for files by name pattern using a raw filesystem walk — "
                "no Spotlight index needed. Finds files in excluded folders, external drives, "
                "hidden files, and recently created files that Spotlight misses. "
                "Uses fd if installed, falls back to find. Returns up to 50 file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename pattern to search for, e.g. 'resume' or '*.pdf'",
                    },
                    "search_dir": {
                        "type": "string",
                        "description": "Directory to search in (defaults to ~). Use ~/Downloads, ~/Desktop, etc.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "If true, also search hidden files and directories (starting with .)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "content_search",
            "description": (
                "Search inside file contents for a text pattern. "
                "Uses ripgrep (rg) if installed, falls back to grep -r. "
                "Returns up to 30 file paths that contain the pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for inside files (case-insensitive).",
                    },
                    "search_dir": {
                        "type": "string",
                        "description": "Directory to search in (defaults to ~).",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional glob to restrict file types, e.g. '*.py' or '*.md'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_files",
            "description": (
                "Find recently used or modified files. Combines Spotlight recency index "
                "with currently open Finder windows. Best for 'file I was just working on' queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours_back": {
                        "type": "integer",
                        "description": "How many hours back to look (default 24).",
                    },
                },
                "required": [],
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
        if query and not query.startswith("kMDItem"):
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


async def filesystem_search(
    pattern: str,
    search_dir: str = "~",
    include_hidden: bool = False,
) -> str:
    """Raw filesystem walk — no Spotlight index dependency. Uses fd if available, else find."""
    loop = asyncio.get_running_loop()

    # Strip leading/trailing wildcards — find -iname always wraps in *...*
    clean_pattern = pattern.strip("*")

    _PRUNE_DIRS = [
        "Library", "node_modules", ".venv", "venv", "__pycache__",
        ".git", ".Trash", "Caches", "Logs",
    ]

    # When searching home, try common dirs first — much faster than walking all of ~
    requested_dir = os.path.expanduser(search_dir)
    home = os.path.expanduser("~")
    if requested_dir == home:
        search_dirs = [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Documents"),
            os.path.join(home, "Downloads"),
            home,  # full scan only if not found above
        ]
    else:
        search_dirs = [requested_dir]

    async def _search_one(directory: str) -> list[str]:
        # Try fd first (fast, case-insensitive, stops at --max-results)
        try:
            fd_cmd = ["fd", "--type", "f", "--ignore-case", "--max-results", "50"]
            if include_hidden:
                fd_cmd.append("--hidden")
            fd_cmd.extend([clean_pattern, directory])
            log.info("filesystem_search (fd): %s", " ".join(fd_cmd))
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(fd_cmd, capture_output=True, text=True, timeout=8),
            )
            if proc.returncode == 0:
                return [p for p in proc.stdout.strip().split("\n") if p]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: find with directory pruning
        prune_expr: list[str] = []
        for d in _PRUNE_DIRS:
            prune_expr += ["-name", d, "-prune", "-o"]
        find_cmd = ["find", directory] + prune_expr
        if not include_hidden:
            find_cmd += ["(", "-not", "-path", "*/.*", ")"]
        find_cmd += ["-type", "f", "-iname", f"*{clean_pattern}*", "-print"]
        log.info("filesystem_search (find): %s", " ".join(find_cmd))
        try:
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(find_cmd, capture_output=True, text=True, timeout=8),
            )
            return [p for p in proc.stdout.strip().split("\n") if p]
        except subprocess.TimeoutExpired:
            log.warning("filesystem_search timed out in %s", directory)
            return []

    all_paths: list[str] = []
    seen: set[str] = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        results = await _search_one(d)
        for p in results:
            if p not in seen:
                seen.add(p)
                all_paths.append(p)
        if all_paths and d != home:
            # Found something in a common dir — no need for full home scan
            break

    all_paths = all_paths[:50]
    if not all_paths:
        return "No files found."
    return "\n".join(all_paths)


async def content_search(
    pattern: str,
    search_dir: str = "~",
    file_glob: Optional[str] = None,
) -> str:
    """Search inside file contents. Uses rg if available, else grep -r."""
    directory = os.path.expanduser(search_dir)
    loop = asyncio.get_running_loop()

    # Try ripgrep first
    try:
        rg_cmd = ["rg", "-l", "-i"]
        if file_glob:
            rg_cmd.extend(["--iglob", file_glob])
        rg_cmd.extend([pattern, directory])
        log.info("content_search (rg): %s", " ".join(rg_cmd))
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(rg_cmd, capture_output=True, text=True, timeout=30),
        )
        if proc.returncode in (0, 1):  # 1 = no matches (not an error)
            paths = [p for p in proc.stdout.strip().split("\n") if p]
            paths = paths[:30]
            if paths:
                return "\n".join(paths)
            return "No files found."
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # rg not installed — fall through to grep

    # Fallback: grep -r
    grep_cmd = ["grep", "-r", "-l", "-i"]
    if file_glob:
        grep_cmd.extend(["--include", file_glob])
    grep_cmd.extend([pattern, directory])
    log.info("content_search (grep): %s", " ".join(grep_cmd))

    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30),
        )
    except subprocess.TimeoutExpired:
        return "Content search timed out after 30 seconds."

    paths = [p for p in proc.stdout.strip().split("\n") if p]
    paths = paths[:30]
    if not paths:
        return "No files found."
    return "\n".join(paths)


async def recent_files(hours_back: int = 24) -> str:
    """Combine Spotlight recency + open Finder windows to find recently used files."""
    loop = asyncio.get_running_loop()
    found: list[str] = []

    # 1. mdfind: files used in the last N hours
    days = max(1, hours_back // 24)
    mdfind_cmd = ["mdfind", "-onlyin", os.path.expanduser("~"),
                  f"kMDItemLastUsedDate >= $time.today(-{days})"]
    log.info("recent_files mdfind: %s", " ".join(mdfind_cmd))
    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(mdfind_cmd, capture_output=True, text=True, timeout=10),
        )
        if proc.returncode == 0:
            found.extend(p for p in proc.stdout.strip().split("\n") if p)
    except subprocess.TimeoutExpired:
        log.warning("recent_files mdfind timed out")

    # 2. osascript: path of front Finder window (if open)
    osa_script = (
        'tell application "Finder"\n'
        '  if (count of windows) > 0 then\n'
        '    get POSIX path of (target of front window as alias)\n'
        '  end if\n'
        'end tell'
    )
    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["osascript", "-e", osa_script],
                capture_output=True, text=True, timeout=5,
            ),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            finder_dir = proc.stdout.strip()
            # List files in the Finder window directory
            for entry in os.scandir(finder_dir):
                if entry.is_file():
                    found.append(entry.path)
    except (subprocess.TimeoutExpired, PermissionError, OSError):
        pass  # Finder not open or inaccessible

    # Deduplicate, sort by mtime (most recent first), cap at 50
    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            unique.append(p)

    unique.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    unique = unique[:50]

    if not unique:
        return "No recently used files found."
    return "\n".join(unique)


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

    MAX_ITERATIONS = 8

    SYSTEM_PROMPT = """You are a desktop intelligence agent with access to macOS data sources.
Your job is to answer questions about the user's files, photos, and local data by searching and reasoning over metadata.

Available tools:
- spotlight_search: Search files via macOS Spotlight (fast, indexed). Supports content type filters, date ranges, name patterns.
- filesystem_search: Raw filesystem walk — no index needed. Finds files Spotlight misses (excluded folders, external drives, hidden files, recent files).
- content_search: Search inside file contents using ripgrep or grep. Returns files that contain the pattern.
- recent_files: Find recently used/modified files. Combines Spotlight recency with open Finder windows.
- file_metadata: Get rich metadata for files (GPS, dates, size, authors, download source, etc.)
- open_file: Open a file with its default app or reveal in Finder.

Strategy:
1. For "find file named X": if the folder is not obvious from the request, ask the user which folder it might be in before searching — e.g. "Do you know which folder it's in? Desktop, Documents, Downloads?"
2. If the user knows the folder, pass it as search_dir to spotlight_search or filesystem_search to avoid slow full-home scans.
3. If the folder is unknown, try spotlight_search first; if 0 results, use filesystem_search (no index needed).
4. For "file about X" / content queries: use content_search.
5. For "file I was just working on" / "recently opened": use recent_files.
6. Use file_metadata after finding paths to get dates, GPS, size.
7. Use open_file only when explicitly asked to open/reveal.
8. When you find a file, always include its full absolute path in your answer so it can be opened later.
   e.g. "Found your resume at /Users/asri/Desktop/Arjun_Sriram.pdf"
9. Synthesize a concise spoken answer (2-3 sentences).

Be efficient — you have a maximum of 8 tool calls. Give concise, spoken answers."""

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
            elif name == "filesystem_search":
                return await filesystem_search(**args)
            elif name == "content_search":
                return await content_search(**args)
            elif name == "recent_files":
                return await recent_files(**args)
            elif name == "file_metadata":
                return await file_metadata(**args)
            elif name == "open_file":
                return open_file(**args)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            log.exception("Desktop tool %s failed: %s", name, e)
            return f"Tool error: {e}"
