"""FTS5-based search across memory files and daily notes."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from friday import config

log = logging.getLogger(__name__)

_INDEX_PATH = config.FRIDAY_DIR / "memory_index.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_INDEX_PATH))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(source, date, content, tokenize='porter')"
    )
    return conn


def rebuild_index() -> None:
    """Re-index MEMORY.md and all daily notes into the FTS5 table."""
    conn = _get_conn()
    conn.execute("DELETE FROM memory_fts")

    # Index MEMORY.md
    if config.MEMORY_PATH.exists():
        text = config.MEMORY_PATH.read_text(encoding="utf-8").strip()
        if text:
            # Split into individual lines/facts for granular search
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("<!--"):
                    conn.execute(
                        "INSERT INTO memory_fts(source, date, content) VALUES (?, ?, ?)",
                        ("MEMORY.md", "", line),
                    )

    # Index daily notes
    for note_path in sorted(config.MEMORY_DIR.glob("*.md")):
        date_str = note_path.stem  # e.g. "2026-04-12"
        text = note_path.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    conn.execute(
                        "INSERT INTO memory_fts(source, date, content) VALUES (?, ?, ?)",
                        (note_path.name, date_str, line),
                    )

    conn.commit()
    conn.close()
    log.debug("Memory index rebuilt")


def search(query: str, limit: int = 5) -> list[dict]:
    """Search memory using FTS5 MATCH. Returns list of {source, date, snippet}."""
    rebuild_index()

    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT source, date, snippet(memory_fts, 2, '→', '←', '...', 40) "
            "FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        log.warning("FTS5 search error: %s", e)
        return []
    finally:
        conn.close()

    return [{"source": r[0], "date": r[1], "snippet": r[2]} for r in rows]
