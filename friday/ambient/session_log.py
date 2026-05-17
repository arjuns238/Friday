"""Raw session captures: append-only on disk (NDJSON) + in-memory active window.

Compression summarizes older in-memory rows into sessions/YYYY-MM-DD.md and trims
the in-memory buffer only — lines already on disk are never deleted (LCM-style).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from friday import config

log = logging.getLogger(__name__)

_ENTRY_KEYS = ("time", "activity", "app", "detail")
_RAW_TAIL_FOR_PROMPT = 10
_COMPRESS_KEEP_TAIL = 10
_PROMPT_MAX_CHARS = 4000


def _normalize_entry(raw: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in _ENTRY_KEYS:
        v = raw.get(k, "")
        out[k] = v if isinstance(v, str) else str(v)
    return out


class SessionLog:
    """Append-only raw log on disk; in-memory list is the active context window."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        state_path: Path | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._path = path if path is not None else config.SESSION_LOG_PATH
        self._state_path = (
            state_path
            if state_path is not None
            else (
                config.SESSION_LOG_STATE_PATH
                if self._path == config.SESSION_LOG_PATH
                else self._path.parent / "ambient_state.json"
            )
        )
        self._max = (
            max_entries if max_entries is not None else config.FRIDAY_SESSION_LOG_MAX
        )
        self._entries: list[dict[str, str]] = []
        self._last_compression_unix: float | None = None
        self._created_unix = time.time()
        self._lock = threading.Lock()
        self._load()

    def append(self, entry: dict[str, Any]) -> None:
        """Append one capture: always one new line on disk; trim in-memory window only."""
        row = _normalize_entry(entry)
        with self._lock:
            self._append_line_disk_unlocked(row)
            self._entries.append(row)
            cap = self._max * 2
            if len(self._entries) > cap:
                self._entries = self._entries[-self._max :]

    def get_recent(self, n: int) -> list[dict[str, str]]:
        if n <= 0:
            return []
        with self._lock:
            return [dict(e) for e in self._entries[-n:]]

    def get_compressible(self) -> list[dict[str, str]]:
        with self._lock:
            if len(self._entries) <= _COMPRESS_KEEP_TAIL:
                return []
            return [dict(e) for e in self._entries[: -_COMPRESS_KEEP_TAIL]]

    def remove_compressed(self, n: int) -> None:
        """Trim in-memory window after a successful compression. Disk log is unchanged."""
        if n <= 0:
            return
        with self._lock:
            keep = min(_COMPRESS_KEEP_TAIL, len(self._entries))
            max_drop = max(0, len(self._entries) - keep)
            n = min(n, max_drop)
            if n <= 0:
                return
            self._entries = self._entries[n:]

    def mark_compression_done(self) -> None:
        with self._lock:
            self._last_compression_unix = time.time()
            self._save_state_unlocked()

    def should_compress(self) -> bool:
        with self._lock:
            n = len(self._entries)
            last = self._last_compression_unix
        if n >= self._max:
            return True
        if n <= _COMPRESS_KEEP_TAIL:
            return False
        ref = last if last is not None else self._created_unix
        return (time.time() - ref) >= config.FRIDAY_COMPRESS_INTERVAL

    def get_full_context(self) -> str:
        with self._lock:
            entries = list(self._entries)
        return _format_entry_lines(entries)

    def get_prompt_context(self) -> str:
        from friday.memory import read_today_session_markdown_excerpt

        with self._lock:
            tail = [dict(e) for e in self._entries[-_RAW_TAIL_FOR_PROMPT:]]
        file_excerpt = read_today_session_markdown_excerpt(max_chars=2400)
        raw_block = _format_entry_lines(tail)
        parts: list[str] = []
        if file_excerpt.strip():
            parts.append("EARLIER TODAY (compressed session log):\n" + file_excerpt.strip())
        if raw_block.strip():
            parts.append("RECENT SCREEN OBSERVATIONS (raw):\n" + raw_block.strip())
        out = "\n\n".join(parts)
        if len(out) > _PROMPT_MAX_CHARS:
            out = out[: _PROMPT_MAX_CHARS - 20] + "\n[…truncated]"
        return out

    def seed_startup_from_now(self, now_body: str) -> None:
        text = (now_body or "").strip()
        if not text:
            return
        from datetime import datetime

        clip = text.replace("\n", " ").strip()
        if len(clip) > 800:
            clip = clip[:797] + "..."
        t = datetime.now().strftime("%H:%M")
        self.append(
            {
                "time": t,
                "activity": "startup",
                "app": "Friday",
                "detail": clip,
            }
        )

    def _append_line_disk_unlocked(self, row: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            log.warning("Could not append session log line (%s): %s", self._path, exc)

    def _save_state_unlocked(self) -> None:
        payload = {"last_compression_unix": self._last_compression_unix}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except OSError as exc:
            log.warning("Could not save session state (%s): %s", self._state_path, exc)

    def _load_state_unlocked(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not load session state (%s): %s", self._state_path, exc)
            return
        if isinstance(raw, dict) and isinstance(
            raw.get("last_compression_unix"), (int, float)
        ):
            self._last_compression_unix = float(raw["last_compression_unix"])

    def _maybe_migrate_legacy_root_log_unlocked(self) -> None:
        """Move ~/.friday/session_log.json (wrapped JSON) into sessions/ once (production path only)."""
        if self._path != config.SESSION_LOG_PATH:
            return
        legacy = config.LEGACY_SESSION_LOG_PATH
        if not legacy.exists() or self._path.exists():
            return
        try:
            raw = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items: list[Any] = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get("entries") if isinstance(raw.get("entries"), list) else []
            m = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
            if isinstance(m.get("last_compression_unix"), (int, float)):
                self._last_compression_unix = float(m["last_compression_unix"])
        else:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for it in items:
            if isinstance(it, dict):
                self._append_line_disk_unlocked(_normalize_entry(it))
        try:
            legacy.rename(legacy.with_suffix(legacy.suffix + ".migrated"))
        except OSError:
            log.warning("Could not rename legacy session log; leaving in place")
        log.info("Migrated legacy session log → %s", self._path)

    def _load_ndjson_unlocked(self) -> list[dict[str, str]]:
        loaded: list[dict[str, str]] = []
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return loaded
        if text.lstrip().startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return loaded
            if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
                for it in obj["entries"]:
                    if isinstance(it, dict):
                        loaded.append(_normalize_entry(it))
                m = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
                if isinstance(m.get("last_compression_unix"), (int, float)):
                    self._last_compression_unix = float(m["last_compression_unix"])
                self._rewrite_ndjson_from_entries_unlocked(loaded)
                log.info("Converted wrapped session log → NDJSON at %s", self._path)
                return loaded
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                loaded.append(_normalize_entry(obj))
        return loaded

    def _rewrite_ndjson_from_entries_unlocked(self, entries: list[dict[str, str]]) -> None:
        """One-time migration: same rows, NDJSON on disk."""
        tmp = self._path.with_suffix(self._path.suffix + ".rewrite.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for row in entries:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(self._path)
        except OSError as exc:
            log.warning("Could not rewrite session log as NDJSON: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _load(self) -> None:
        with self._lock:
            self._maybe_migrate_legacy_root_log_unlocked()
            self._load_state_unlocked()
            if not self._path.exists():
                return
            loaded = self._load_ndjson_unlocked()
            cap = self._max * 2
            if len(loaded) > cap:
                loaded = loaded[-cap:]
            self._entries = loaded
            log.info("Loaded %d session rows into memory from %s", len(self._entries), self._path)


def _format_entry_lines(entries: list[dict[str, str]]) -> str:
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries:
        lines.append(f"- [{e['time']}] {e['activity']} in {e['app']}: {e['detail']}")
    return "\n".join(lines)
