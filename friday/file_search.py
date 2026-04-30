"""Read-only file search: glob discovery + content regex (ripgrep or Python fallback)."""
from __future__ import annotations

import fnmatch
import difflib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "eggs",
    ".gradle",
    "target",
})

RG_TIMEOUT_SEC = 20
MAX_OUTPUT_CHARS = 48_000
MAX_FILE_READ_BYTES = 512_000
DEFAULT_HEAD_LIMIT = 80
_HOME_ALIASES = {
    "downloads": "Downloads",
    "download": "Downloads",
    "documents": "Documents",
    "document": "Documents",
    "desktop": "Desktop",
    "home": "",
}
_COMMON_DOC_EXTS = (".pdf", ".docx", ".doc", ".txt", ".rtf")

# Extra --glob rules for ripgrep (junk dirs even if not in .gitignore)
_RG_EXCLUDE_GLOBS = (
    "!.git/**",
    "!.svn/**",
    "!.hg/**",
    "!.bzr/**",
    "!**/node_modules/**",
    "!**/__pycache__/**",
    "!**/.venv/**",
    "!**/venv/**",
    "!**/.mypy_cache/**",
    "!**/.pytest_cache/**",
    "!**/dist/**",
    "!**/build/**",
    "!**/.eggs/**",
)


def resolve_search_directory(path_arg: str) -> Path:
    """Expand user and common folder aliases, resolve, require existing directory."""
    normalized = normalize_search_path(path_arg)
    raw = Path(normalized).expanduser()
    try:
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"Invalid path: {exc}") from exc
    if not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {resolved}")
    return resolved


def normalize_search_path(path_arg: str) -> str:
    """Normalize user-friendly folder words to real absolute paths."""
    value = (path_arg or "").strip()
    if not value:
        return value
    lowered = value.lower().strip().strip("/").strip()
    home = Path.home()

    if lowered in _HOME_ALIASES:
        suffix = _HOME_ALIASES[lowered]
        return str(home / suffix) if suffix else str(home)

    if lowered.startswith(("downloads/", "documents/", "desktop/")):
        first, rest = lowered.split("/", 1)
        suffix = _HOME_ALIASES.get(first, first)
        return str(home / suffix / rest)

    # Recover from LLM hallucinating another username for obvious home folders.
    m = re.match(r"^/Users/[^/]+/(Downloads|Documents|Desktop)(?:/(.*))?$", value, re.IGNORECASE)
    if m:
        folder = m.group(1)
        tail = m.group(2)
        base = home / folder
        return str(base / tail) if tail else str(base)

    return value


def build_rg_search_argv(
    pattern: str,
    *,
    output_mode: str,
    glob: str | None,
    file_type: str | None,
    case_insensitive: bool,
    context_lines: int | None,
    multiline: bool,
) -> list[str]:
    """Build `rg` argv (pattern and path '.' added by caller with cwd=root)."""
    argv: list[str] = [
        "rg",
        "--max-columns",
        "500",
        "--hidden",
        "--no-heading",
    ]
    for g in _RG_EXCLUDE_GLOBS:
        argv.extend(["--glob", g])
    if glob:
        argv.extend(["--glob", glob])
    if file_type:
        argv.extend(["--type", file_type])
    if case_insensitive:
        argv.append("-i")
    if multiline:
        argv.extend(["-U", "--multiline", "--multiline-dotall"])

    if output_mode == "files_with_matches":
        argv.append("-l")
    elif output_mode == "count":
        argv.append("--count-matches")
    else:
        argv.append("-n")
        if context_lines is not None and context_lines > 0:
            argv.extend(["-C", str(context_lines)])

    if pattern.startswith("-"):
        argv.extend(["-e", pattern])
    else:
        argv.append(pattern)

    argv.append(".")
    return argv


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n\n[Output truncated for length.]"


def _paginate_lines(text: str, offset: int, head_limit: int) -> str:
    lines = text.splitlines()
    if offset:
        lines = lines[offset:]
    if head_limit > 0:
        lines = lines[:head_limit]
    return "\n".join(lines)


def _sort_paths_by_mtime(root: Path, rel_paths: list[str]) -> list[str]:
    scored: list[tuple[float, str]] = []
    for rel in rel_paths:
        try:
            st = (root / rel).stat()
            scored.append((-st.st_mtime, rel))
        except OSError:
            scored.append((0.0, rel))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [p for _, p in scored]


def _run_rg(argv: list[str], cwd: Path) -> str:
    rg = shutil.which("rg")
    if not rg:
        raise FileNotFoundError("ripgrep (rg) not found on PATH")
    argv = [rg, *argv[1:]]  # replace 'rg' placeholder with resolved binary
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RG_TIMEOUT_SEC,
    )
    if proc.returncode not in (0, 1):
        err = (proc.stderr or "").strip() or proc.stdout or "unknown error"
        raise RuntimeError(f"rg exited {proc.returncode}: {err[:500]}")
    return proc.stdout or ""


def search_files_rg(
    root: Path,
    pattern: str,
    *,
    output_mode: str,
    glob: str | None,
    file_type: str | None,
    case_insensitive: bool,
    head_limit: int,
    offset: int,
    context_lines: int | None,
    multiline: bool,
) -> str:
    argv = build_rg_search_argv(
        pattern,
        output_mode=output_mode,
        glob=glob,
        file_type=file_type,
        case_insensitive=case_insensitive,
        context_lines=context_lines,
        multiline=multiline,
    )
    out = _run_rg(argv, root)

    if output_mode == "files_with_matches":
        paths = [ln.strip() for ln in out.splitlines() if ln.strip()]
        paths = list(dict.fromkeys(paths))
        if len(paths) > 20_000:
            paths = paths[:20_000]
        paths = _sort_paths_by_mtime(root, paths)
        if offset:
            paths = paths[offset:]
        if head_limit > 0:
            paths = paths[:head_limit]
        return _truncate("\n".join(paths) if paths else "(no matches)")

    if output_mode == "count":
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if offset:
            lines = lines[offset:]
        if head_limit > 0:
            lines = lines[:head_limit]
        body = "\n".join(lines) if lines else "(no matches)"
        return _truncate(body)

    # content
    body = _paginate_lines(out, offset, head_limit)
    return _truncate(body if body.strip() else "(no matches)")


def find_files_rg(root: Path, glob_pattern: str, head_limit: int, offset: int) -> str:
    argv: list[str] = [
        "rg",
        "--files",
        "--hidden",
        "--glob",
        glob_pattern,
    ]
    for g in _RG_EXCLUDE_GLOBS:
        argv.extend(["--glob", g])
    argv.append(".")
    out = _run_rg(argv, root)
    paths = [ln.strip() for ln in out.splitlines() if ln.strip()]
    paths = list(dict.fromkeys(paths))
    paths = _sort_paths_by_mtime(root, paths)
    if offset:
        paths = paths[offset:]
    if head_limit > 0:
        paths = paths[:head_limit]
    return _truncate("\n".join(paths) if paths else "(no files found)")


def find_files_spotlight(root: Path, glob_pattern: str, head_limit: int, offset: int) -> str:
    """Fast filename search via Spotlight index, scoped to root."""
    mdfind = shutil.which("mdfind")
    if not mdfind:
        raise FileNotFoundError("mdfind not found on PATH")
    cmd = [mdfind, "-onlyin", str(root), f'kMDItemFSName == "{glob_pattern}"wc']
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=8,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "unknown error"
        raise RuntimeError(f"mdfind exited {proc.returncode}: {err[:200]}")
    paths = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    rel_paths: list[str] = []
    for p in paths:
        try:
            rel_paths.append(str(Path(p).resolve(strict=False).relative_to(root)))
        except Exception:
            continue
    rel_paths = list(dict.fromkeys(rel_paths))
    rel_paths = _sort_paths_by_mtime(root, rel_paths)
    if offset:
        rel_paths = rel_paths[offset:]
    if head_limit > 0:
        rel_paths = rel_paths[:head_limit]
    return _truncate("\n".join(rel_paths) if rel_paths else "(no files found)")


def _should_skip_dir(name: str) -> bool:
    return name in EXCLUDED_DIR_NAMES


def _read_text_limited(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            data = f.read(MAX_FILE_READ_BYTES)
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _path_matches_glob(rel: str, name: str, glob: str | None) -> bool:
    if not glob:
        return True
    if fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(rel, glob):
        return True
    if "**/" in glob:
        tail = glob.split("**/", 1)[-1]
        return fnmatch.fnmatch(name, tail) or fnmatch.fnmatch(rel, tail)
    return False


def search_files_python(
    root: Path,
    pattern: str,
    *,
    output_mode: str,
    glob: str | None,
    case_insensitive: bool,
    head_limit: int,
    offset: int,
    context_lines: int | None,
    multiline: bool,
) -> str:
    del context_lines  # optional future: context in Python fallback
    flags = re.IGNORECASE if case_insensitive else 0
    if multiline:
        flags |= re.DOTALL
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return f"Invalid regex: {exc}"

    rel_hits: list[str] = []
    count_rows: list[str] = []
    content_lines: list[str] = []

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    max_collect = max(head_limit + offset, 0) + 8000

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for name in filenames:
            full = Path(dirpath) / name
            r = rel(full)
            if not _path_matches_glob(r, name, glob):
                continue
            text = _read_text_limited(full)
            if text is None:
                continue
            if output_mode == "files_with_matches":
                if rx.search(text):
                    rel_hits.append(r)
                    if len(rel_hits) >= max_collect:
                        break
            elif output_mode == "count":
                n = len(rx.findall(text)) if multiline else sum(1 for _ in rx.finditer(text))
                if n:
                    count_rows.append(f"{r}:{n}")
                    if len(count_rows) >= max_collect:
                        break
            else:
                for m in rx.finditer(text):
                    if not multiline:
                        lineno = text.count("\n", 0, m.start()) + 1
                        line_start = text.rfind("\n", 0, m.start()) + 1
                        line_end = text.find("\n", m.end())
                        if line_end < 0:
                            line_end = len(text)
                        line = text[line_start:line_end].strip()[:500]
                        content_lines.append(f"{r}:{lineno}:{line}")
                    else:
                        content_lines.append(f"{r}:multiline-match")
                    if len(content_lines) >= max_collect:
                        break
        if output_mode == "files_with_matches" and len(rel_hits) >= max_collect:
            break
        if output_mode == "count" and len(count_rows) >= max_collect:
            break
        if output_mode == "content" and len(content_lines) >= max_collect:
            break

    if output_mode == "files_with_matches":
        rel_hits = list(dict.fromkeys(rel_hits))
        rel_hits = _sort_paths_by_mtime(root, rel_hits)
        if offset:
            rel_hits = rel_hits[offset:]
        if head_limit > 0:
            rel_hits = rel_hits[:head_limit]
        body = "\n".join(rel_hits) if rel_hits else "(no matches)"
        return _truncate(body)

    if output_mode == "count":
        if offset:
            count_rows = count_rows[offset:]
        if head_limit > 0:
            count_rows = count_rows[:head_limit]
        body = "\n".join(count_rows) if count_rows else "(no matches)"
        return _truncate(body)

    lines = content_lines[offset:] if offset else content_lines
    if head_limit > 0:
        lines = lines[:head_limit]
    body = "\n".join(lines) if lines else "(no matches)"
    return _truncate(body)


def find_files_python(root: Path, glob_pattern: str, head_limit: int, offset: int) -> str:
    matches: list[str] = []
    if "**/" in glob_pattern:
        left, right = glob_pattern.split("**/", 1)
        base = (root / left) if left else root
        if base.is_dir():
            for p in base.rglob(right):
                if p.is_file():
                    try:
                        matches.append(str(p.relative_to(root)))
                    except ValueError:
                        pass
    else:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            for name in filenames:
                full = Path(dirpath) / name
                r = str(full.relative_to(root))
                if fnmatch.fnmatch(name, glob_pattern) or fnmatch.fnmatch(r, glob_pattern):
                    matches.append(r)

    matches = list(dict.fromkeys(matches))
    matches = _sort_paths_by_mtime(root, matches)
    if offset:
        matches = matches[offset:]
    if head_limit > 0:
        matches = matches[:head_limit]
    body = "\n".join(matches) if matches else "(no files found)"
    return _truncate(body)


def search_files_sync(arguments: dict[str, Any]) -> str:
    path_arg = arguments.get("path") or ""
    pattern = arguments.get("pattern") or ""
    if not path_arg.strip():
        return "Error: path is required."
    if not pattern.strip():
        return "Error: pattern is required."

    try:
        normalized = normalize_search_path(str(path_arg))
        root = resolve_search_directory(str(path_arg))
        if normalized != str(path_arg).strip():
            log.info("search_files path normalized: %r -> %r", path_arg, normalized)
        log.info("search_files root resolved: %s", root)
    except ValueError as exc:
        return f"Error: {exc}"

    output_mode = arguments.get("output_mode") or "files_with_matches"
    if output_mode not in ("files_with_matches", "content", "count"):
        output_mode = "files_with_matches"

    glob = arguments.get("glob")
    file_type = arguments.get("file_type")
    case_insensitive = bool(arguments.get("case_insensitive"))
    head_limit = int(arguments.get("head_limit") or DEFAULT_HEAD_LIMIT)
    offset = int(arguments.get("offset") or 0)
    ctx = arguments.get("context_lines")
    context_lines = int(ctx) if ctx is not None else None
    multiline = bool(arguments.get("multiline"))

    if shutil.which("rg"):
        try:
            return search_files_rg(
                root,
                pattern,
                output_mode=output_mode,
                glob=glob,
                file_type=file_type,
                case_insensitive=case_insensitive,
                head_limit=head_limit,
                offset=offset,
                context_lines=context_lines,
                multiline=multiline,
            )
        except Exception as exc:
            log.warning("rg search failed, falling back to Python: %s", exc)

    return search_files_python(
        root,
        pattern,
        output_mode=output_mode,
        glob=glob,
        case_insensitive=case_insensitive,
        head_limit=head_limit,
        offset=offset,
        context_lines=context_lines,
        multiline=multiline,
    )


def find_files_sync(arguments: dict[str, Any]) -> str:
    path_arg = arguments.get("path") or ""
    glob_pattern = arguments.get("glob_pattern") or ""
    if not path_arg.strip():
        return "Error: path is required."
    if not glob_pattern.strip():
        return "Error: glob_pattern is required."

    try:
        normalized = normalize_search_path(str(path_arg))
        root = resolve_search_directory(str(path_arg))
        if normalized != str(path_arg).strip():
            log.info("find_files path normalized: %r -> %r", path_arg, normalized)
        log.info("find_files root resolved: %s", root)
    except ValueError as exc:
        return f"Error: {exc}"

    head_limit = int(arguments.get("head_limit") or DEFAULT_HEAD_LIMIT)
    offset = int(arguments.get("offset") or 0)

    patterns = _expand_glob_candidates(glob_pattern)

    # Prefer Spotlight for broad filename lookups on macOS.
    if shutil.which("mdfind"):
        for candidate in patterns:
            try:
                result = find_files_spotlight(root, candidate, head_limit, offset)
                if not _is_no_files_result(result):
                    return result
            except Exception as exc:
                log.warning("mdfind find_files failed for %r, falling back: %s", candidate, exc)

    if shutil.which("rg"):
        for candidate in patterns:
            try:
                result = find_files_rg(root, candidate, head_limit, offset)
                if not _is_no_files_result(result):
                    return result
            except Exception as exc:
                log.warning("rg find_files failed for %r, falling back to Python: %s", candidate, exc)

    for candidate in patterns:
        result = find_files_python(root, candidate, head_limit, offset)
        if not _is_no_files_result(result):
            return result
    fuzzy = _find_files_fuzzy(root, glob_pattern, head_limit, offset)
    if not _is_no_files_result(fuzzy):
        return fuzzy
    return "(no files found)"


def _is_no_files_result(text: str) -> bool:
    return text.strip().startswith("(no files found)")


def _expand_glob_candidates(glob_pattern: str) -> list[str]:
    """Generate tolerant filename patterns for voice inputs."""
    value = glob_pattern.strip()
    candidates: list[str] = []

    def add(p: str) -> None:
        if p and p not in candidates:
            candidates.append(p)

    add(value)

    core = value.strip("*")
    if core:
        add(f"*{core}*")
    if " " in core:
        add(f"*{core.replace(' ', '_')}*")
        add(f"*{core.replace(' ', '-')}*")
        add("*" + "*".join(core.split()) + "*")

    # If user didn't specify an extension, try common document extensions.
    if "." not in Path(core).name:
        seeds = list(candidates)
        for seed in seeds:
            seed_core = seed.rstrip("*")
            for ext in _COMMON_DOC_EXTS:
                add(f"{seed_core}{ext}")

    return candidates


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _query_from_glob(glob_pattern: str) -> str:
    q = glob_pattern.strip().strip("*")
    q = re.sub(r"\.pdf$|\.docx$|\.doc$|\.txt$|\.rtf$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpdf\b|\bdocx\b|\bdoc\b|\brtf\b|\btxt\b", "", q, flags=re.IGNORECASE)
    q = q.replace("_", " ").replace("-", " ").strip()
    return q


def _find_files_fuzzy(root: Path, glob_pattern: str, head_limit: int, offset: int) -> str:
    """Fallback fuzzy filename matching for minor transcription/name drift."""
    query = _query_from_glob(glob_pattern)
    query_norm = _norm_name(query)
    if not query_norm:
        return "(no files found)"

    candidates: list[tuple[float, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for name in filenames:
            base = Path(name).stem
            base_norm = _norm_name(base)
            if not base_norm:
                continue
            ratio = difflib.SequenceMatcher(None, query_norm, base_norm).ratio()
            # token overlap helps with separators/order shifts
            q_tokens = [t for t in re.split(r"[\s_\-]+", query.lower()) if t]
            b_tokens = [t for t in re.split(r"[\s_\-]+", base.lower()) if t]
            overlap = 0.0
            if q_tokens and b_tokens:
                overlap = len(set(q_tokens) & set(b_tokens)) / max(len(set(q_tokens)), 1)
            score = max(ratio, overlap)
            if score >= 0.72:
                full = Path(dirpath) / name
                rel = str(full.relative_to(root))
                candidates.append((score, rel))

    if not candidates:
        return "(no files found)"
    candidates.sort(key=lambda x: (-x[0], x[1]))
    rels = [r for _, r in candidates]
    rels = list(dict.fromkeys(rels))
    rels = _sort_paths_by_mtime(root, rels)
    if offset:
        rels = rels[offset:]
    if head_limit > 0:
        rels = rels[:head_limit]
    return _truncate("\n".join(rels) if rels else "(no files found)")


def open_file_sync(arguments: dict[str, Any]) -> str:
    """Open a file with default app, or reveal in Finder."""
    path_arg = arguments.get("path") or ""
    reveal = bool(arguments.get("reveal"))
    if not path_arg.strip():
        return "Error: path is required."

    normalized = normalize_search_path(str(path_arg))
    file_path = Path(normalized).expanduser().resolve(strict=False)
    if normalized != str(path_arg).strip():
        log.info("open_file path normalized: %r -> %r", path_arg, normalized)
    log.info("open_file target resolved: %s", file_path)

    if not file_path.exists():
        return f"Error: file does not exist: {file_path}"
    if not file_path.is_file():
        return f"Error: not a file: {file_path}"

    cmd = ["open", "-R", str(file_path)] if reveal else ["open", str(file_path)]
    try:
        subprocess.run(cmd, check=True, timeout=8)
    except subprocess.TimeoutExpired:
        return "Error: open command timed out."
    except subprocess.CalledProcessError as exc:
        return f"Error: open failed: {exc}"

    action = "Revealed in Finder" if reveal else "Opened"
    return f"{action}: {file_path}"
