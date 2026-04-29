"""Desktop tools — macOS file search, metadata, reading, listing.

Each function is exposed to the Desktop SubAgent as a LangChain `@tool`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from typing import Any, Optional

from langchain_core.tools import tool

log = logging.getLogger(__name__)


# ── Tool: spotlight_search ────────────────────────────────────────────────────

@tool
async def spotlight_search(
    query: str = "*",
    content_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    name_pattern: Optional[str] = None,
    search_dir: Optional[str] = None,
) -> str:
    """Search files via macOS Spotlight (mdfind). Fast, uses the system index.

    Args:
        query: Free-text Spotlight query. Can also use kMDItem predicates.
        content_type: UTI filter, e.g. "public.image", "public.pdf", "com.adobe.pdf".
        date_from: ISO date (YYYY-MM-DD), filters kMDItemContentCreationDate >=.
        date_to: ISO date (YYYY-MM-DD), filters kMDItemContentCreationDate <=.
        name_pattern: Filename glob, e.g. "*.pdf" or "*resume*".
        search_dir: Directory scope. Defaults to the first configured search dir.

    Returns up to 50 file paths, newline-separated.
    """
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

    cmd = ["mdfind"]
    scope = search_dir or config.DESKTOP_SEARCH_DIRS[0]
    scope = os.path.expanduser(scope)
    cmd.extend(["-onlyin", scope])

    if predicates:
        full_query = " && ".join(predicates)
        if query and not query.startswith("kMDItem"):
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
    paths = paths[:50]

    if not paths:
        return "No files found matching the search criteria."

    result = "\n".join(paths)
    if total > 50:
        result += f"\n\n({total} total results, showing first 50)"
    return result


# ── Tool: file_metadata ───────────────────────────────────────────────────────

@tool
async def file_metadata(paths: list[str]) -> str:
    """Get rich metadata (dates, size, GPS, authors, download source) for up to 20 files via mdls."""
    paths = paths[:20]
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
    fields: dict[str, Any] = {}
    interesting = {
        "kMDItemContentType", "kMDItemContentCreationDate",
        "kMDItemContentModificationDate", "kMDItemLastUsedDate",
        "kMDItemFSSize", "kMDItemAuthors", "kMDItemWhereFroms",
        "kMDItemLatitude", "kMDItemLongitude", "kMDItemAltitude",
        "kMDItemPixelWidth", "kMDItemPixelHeight", "kMDItemDurationSeconds",
        "kMDItemTitle", "kMDItemDisplayName", "kMDItemKind", "kMDItemDownloadedDate",
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


# ── Tool: open_file ───────────────────────────────────────────────────────────

@tool
def open_file(path: str, reveal: bool = False) -> str:
    """Open a file with its default application, or reveal it in Finder (reveal=true)."""
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


# ── Tool: filesystem_search ───────────────────────────────────────────────────

@tool
async def filesystem_search(
    pattern: str,
    search_dir: str = "~",
    include_hidden: bool = False,
) -> str:
    """Raw filesystem walk for files by name pattern. No Spotlight index needed.

    Finds files Spotlight misses (excluded folders, external drives, hidden files).
    Uses fd if installed, falls back to find. Returns up to 50 file paths.
    """
    loop = asyncio.get_running_loop()

    clean_pattern = pattern.strip("*")

    _SEP_RE = r"[\s_\-.*]+"
    words = re.split(_SEP_RE, clean_pattern)
    words = [w for w in words if w]

    def _fuzzy_word(word: str) -> str:
        return "".join(re.escape(ch) + "+" if ch.isalpha() else re.escape(ch) for ch in word)

    fuzzy_re = r"[_\- .]*".join(_fuzzy_word(w) for w in words) if len(words) > 1 else (
        _fuzzy_word(words[0]) if words else None
    )

    _PRUNE_DIRS = [
        "Library", "node_modules", ".venv", "venv", "__pycache__",
        ".git", ".Trash", "Caches", "Logs",
    ]

    requested_dir = os.path.expanduser(search_dir)
    home = os.path.expanduser("~")
    if requested_dir == home:
        search_dirs = [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Documents"),
            os.path.join(home, "Downloads"),
            home,
        ]
    else:
        search_dirs = [requested_dir]

    async def _search_one(directory: str) -> list[str]:
        try:
            fd_cmd = ["fd", "--type", "f", "--ignore-case", "--max-results", "50"]
            if include_hidden:
                fd_cmd.append("--hidden")
            fd_cmd.extend(["--regex", fuzzy_re, directory])
            log.info("filesystem_search (fd): %s", " ".join(fd_cmd))
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(fd_cmd, capture_output=True, text=True, timeout=8),
            )
            if proc.returncode == 0:
                return [p for p in proc.stdout.strip().split("\n") if p]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        prune_expr: list[str] = []
        for d in _PRUNE_DIRS:
            prune_expr += ["-name", d, "-prune", "-o"]
        find_cmd = ["find", directory] + prune_expr
        if not include_hidden:
            find_cmd += ["(", "-not", "-path", "*/.*", ")"]
        find_cmd += ["-type", "f", "-iregex", f".*{fuzzy_re}.*", "-print"]
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
            break

    all_paths = all_paths[:50]
    if not all_paths:
        return "No files found."
    return "\n".join(all_paths)


# ── Tool: content_search ──────────────────────────────────────────────────────

@tool
async def content_search(
    pattern: str,
    search_dir: str = "~",
    file_glob: Optional[str] = None,
) -> str:
    """Search inside file contents for a text pattern. Uses ripgrep or grep -r. Returns up to 30 paths."""
    directory = os.path.expanduser(search_dir)
    loop = asyncio.get_running_loop()

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
        if proc.returncode in (0, 1):
            paths = [p for p in proc.stdout.strip().split("\n") if p]
            paths = paths[:30]
            if paths:
                return "\n".join(paths)
            return "No files found."
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

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


# ── Tool: recent_files ────────────────────────────────────────────────────────

@tool
async def recent_files(hours_back: int = 24) -> str:
    """Find recently used/modified files. Combines Spotlight recency with open Finder windows."""
    loop = asyncio.get_running_loop()
    found: list[str] = []

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
            for entry in os.scandir(finder_dir):
                if entry.is_file():
                    found.append(entry.path)
    except (subprocess.TimeoutExpired, PermissionError, OSError):
        pass

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


# ── Tool: read_file ───────────────────────────────────────────────────────────

_TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".csv",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash",
    ".zsh", ".html", ".css", ".xml", ".sql", ".r", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".swift", ".kt", ".lua",
    ".env", ".gitignore", ".dockerignore", ".makefile", ".cmake",
    ".log", ".rst", ".tex", ".bib",
}

_READ_CHAR_CAP = 8000


@tool
async def read_file(path: str) -> str:
    """Read file contents. Text files return content (~8K cap); PDFs extract text; binaries return metadata."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"File not found: {path}"
    if not os.path.isfile(path):
        return f"Not a file: {path}"

    ext = os.path.splitext(path)[1].lower()
    size = os.path.getsize(path)

    if ext == ".pdf":
        return await _read_pdf(path, size)

    if ext in _TEXT_EXTENSIONS:
        return _read_text(path, size)
    if ext == "" and _looks_like_text(path):
        return _read_text(path, size)

    return _read_binary_metadata(path, ext, size)


def _looks_like_text(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\x00" not in chunk
    except OSError:
        return False


def _read_text(path: str, size: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(_READ_CHAR_CAP)
    except Exception as e:
        return f"Error reading {path}: {e}"

    truncated = " (truncated)" if size > _READ_CHAR_CAP else ""
    name = os.path.basename(path)
    return f"── {name} ({_human_size(size)}{truncated}) ──\n{content}"


async def _read_pdf(path: str, size: int) -> str:
    loop = asyncio.get_running_loop()
    try:
        def _extract():
            import pdfplumber
            text_parts: list[str] = []
            total_chars = 0
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text_parts.append(f"[Page {i + 1}]\n{page_text}")
                    total_chars += len(page_text)
                    if total_chars >= _READ_CHAR_CAP:
                        break
                page_count = len(pdf.pages)
            return "\n\n".join(text_parts), page_count

        content, page_count = await loop.run_in_executor(None, _extract)
    except Exception as e:
        return f"Error reading PDF {path}: {e}"

    truncated = " (truncated)" if len(content) >= _READ_CHAR_CAP else ""
    name = os.path.basename(path)
    return f"── {name} ({_human_size(size)}, {page_count} pages{truncated}) ──\n{content[:_READ_CHAR_CAP]}"


def _read_binary_metadata(path: str, ext: str, size: int) -> str:
    import time as _time
    name = os.path.basename(path)
    mtime = os.path.getmtime(path)
    modified = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
    return f"{name}: {ext.lstrip('.')} file, {_human_size(size)}, modified {modified}. (Binary file — cannot display contents.)"


def _human_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}" if unit != "B" else f"{int(nbytes)}B"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


# ── Tool: list_directory ──────────────────────────────────────────────────────

@tool
async def list_directory(
    path: Optional[str] = None,
    recursive: bool = False,
) -> str:
    """List directory contents with name, type, size, modified date. Defaults to FILE_EXPLORER_DIR. Caps at 100 entries."""
    from friday import config

    dir_path = os.path.expanduser(path) if path else config.FILE_EXPLORER_DIR
    if not os.path.isdir(dir_path):
        return f"Not a directory: {dir_path}"

    import time as _time
    entries: list[str] = []
    cap = 100

    def _walk(current: str, depth: int) -> None:
        if len(entries) >= cap:
            return
        try:
            items = sorted(os.scandir(current), key=lambda e: e.name.lower())
        except PermissionError:
            entries.append(f"  [permission denied: {current}]")
            return

        for entry in items:
            if len(entries) >= cap:
                break
            if entry.name.startswith("."):
                continue
            indent = "  " * depth
            try:
                stat = entry.stat()
                mtime = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(stat.st_mtime))
                if entry.is_dir(follow_symlinks=False):
                    entries.append(f"{indent}{entry.name}/  (dir, modified {mtime})")
                    if recursive and depth < 3:
                        _walk(entry.path, depth + 1)
                else:
                    entries.append(f"{indent}{entry.name}  ({_human_size(stat.st_size)}, {mtime})")
            except OSError:
                entries.append(f"{indent}{entry.name}  (inaccessible)")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _walk, dir_path, 0)

    if not entries:
        return f"Directory is empty: {dir_path}"

    header = f"Contents of {dir_path} ({len(entries)} entries"
    if len(entries) >= cap:
        header += f", capped at {cap}"
    header += "):\n"
    return header + "\n".join(entries)
