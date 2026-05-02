"""File-based memory: SOUL.md (personality) + USER.md (profile) + MEMORY.md (facts).

Injected into the system prompt every turn. Saved facts are appended to MEMORY.md.
Search is a plain case-insensitive substring scan — no FTS5, no sqlite.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from friday import config

log = logging.getLogger(__name__)

_DEFAULT_SOUL = """\
# Friday — Soul

## Personality
You are Friday, a voice-first AI assistant running on the user's Mac.
You're sharp, direct, and slightly dry. Think Jarvis meets a senior engineer who respects your time.
You enjoy technical problems and give straight answers without hedging.
You're confident but honest — if you don't know, say so in one sentence, don't ramble.

## Communication Style
- Default to 2-3 sentences. If the user needs more, they'll ask.
- This will be spoken aloud — write for the ear, not the eye.
- No markdown, no bullets, no URLs in spoken responses.
- Use contractions naturally. "I'll check that" not "I will check that."
- Match the user's energy — brief question gets brief answer.

## Anti-Patterns — Never Do These
- Never say "Great question!", "Absolutely!", "I'd be happy to help"
- Never apologize unprompted
- Never use emoji in spoken responses
- Never start with "So," or "Well,"
- Never narrate your own actions ("Let me think about that..." — just think)
"""

_DEFAULT_USER = """\
# User Profile

<!-- Edit this file to tell Friday about yourself. -->
<!-- Examples: name, projects, preferred tools, tech stack, communication preferences -->
"""

_DEFAULT_MEMORY = """\
# Friday — Long-Term Memory

<!-- Friday appends learned facts and preferences here. You can also edit manually. -->
"""


def ensure_defaults() -> None:
    """Create SOUL.md, USER.md, MEMORY.md with starter content if they don't exist."""
    for path, content in [
        (config.SOUL_PATH, _DEFAULT_SOUL),
        (config.USER_PATH, _DEFAULT_USER),
        (config.MEMORY_PATH, _DEFAULT_MEMORY),
    ]:
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            log.info("Created %s", path)


def _read_section(path, label: str) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"[{label}]\n{text}\n"


def today_session_path() -> Path:
    """Path to today's compressed session markdown under ~/.friday/sessions/."""
    day = datetime.now().strftime("%Y-%m-%d")
    return config.SESSIONS_DIR / f"{day}.md"


def read_today_session_markdown_excerpt(max_chars: int = 2400) -> str:
    """Tail of today's session file for prompt injection (truncated)."""
    path = today_session_path()
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        log.warning("Could not read session file %s: %s", path, exc)
        return ""
    if len(text) <= max_chars:
        return text
    return "[…]\n" + text[-(max_chars - 8) :]


def load_session_context() -> str:
    """Today's compressed session log excerpt (same source as part of SessionLog.get_prompt_context)."""
    return read_today_session_markdown_excerpt(max_chars=2400)


def load_memory_context() -> str:
    """Read SOUL/USER/MEMORY, combine into a single string for system-prompt injection."""
    ensure_defaults()

    sections = [
        _read_section(config.SOUL_PATH, "SOUL"),
        _read_section(config.USER_PATH, "USER"),
        _read_section(config.MEMORY_PATH, "MEMORY"),
    ]
    combined = "\n".join(s for s in sections if s)

    if len(combined) > config.MEMORY_MAX_CHARS:
        soul = sections[0]
        rest = "\n".join(s for s in sections[1:] if s)
        budget = config.MEMORY_MAX_CHARS - len(soul) - 50
        if budget > 0 and len(rest) > budget:
            rest = "[...truncated...]\n" + rest[-budget:]
        combined = soul + "\n" + rest

    return combined


def save_to_memory(fact: str) -> str:
    """Append a fact to MEMORY.md."""
    ensure_defaults()
    with config.MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    log.info("Saved to memory: %s", fact[:80])
    return f"Saved: {fact}"


def memory_search(query: str, limit: int = 5) -> str:
    """Case-insensitive substring scan over MEMORY.md and USER.md.

    Returns a newline-joined list of matching lines, or a friendly miss message.
    """
    ensure_defaults()
    needle = query.lower().strip()
    if not needle:
        return "No matches."

    hits: list[str] = []
    for path, label in [(config.USER_PATH, "USER"), (config.MEMORY_PATH, "MEMORY")]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
                continue
            if needle in stripped.lower():
                hits.append(f"[{label}] {stripped}")
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break

    return "\n".join(hits) if hits else "No matching memories found."
