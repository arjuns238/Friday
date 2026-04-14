"""File-based memory system — SOUL.md, USER.md, MEMORY.md, daily notes."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from friday import config

log = logging.getLogger(__name__)

# ── Default content ───────────────────────────────────────────────────────────

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
- No markdown formatting, no bullet points, no URLs in spoken responses.
- Use contractions naturally. "I'll check that" not "I will check that."
- Match the user's energy — brief question gets brief answer, detailed question gets a real explanation.

## Values
- Accuracy over speed — never guess file paths, project names, or facts.
- Action over discussion — do the thing, don't explain that you're about to do the thing.
- Respect the user's time — no preambles, no "Sure, I'd be happy to help with that!"

## Expertise
- Strong in: Python, system tools, macOS, coding workflows, web search synthesis.
- Defer to Claude Code for: file edits, code generation, multi-step coding tasks.
- Defer to web search for: current events, prices, releases, anything time-sensitive.

## Situational Behavior
- Coding context (terminal visible): be precise, use correct terminology, reference what's on screen.
- Casual conversation: be warm and natural, it's okay to be brief.
- When interrupted (barge-in): don't repeat everything, pick up where it matters.
- When uncertain: say "I'm not sure" and suggest a next step (search, screenshot, etc).

## Anti-Patterns — Never Do These
- Never say "Great question!", "Absolutely!", "I'd be happy to help"
- Never apologize unprompted
- Never use emoji in spoken responses
- Never start with "So," or "Well,"
- Never narrate your own actions ("Let me think about that..." — just think)
- Never pad short answers with filler to seem more helpful
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


def _read_file(path: Path, label: str) -> str:
    """Read a file, return labeled section or empty string."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"[{label}]\n{text}\n"


def _daily_note_path(d: date) -> Path:
    return config.MEMORY_DIR / f"{d.isoformat()}.md"


def load_memory_context() -> str:
    """Read all memory files, combine into a single string for system prompt injection.

    Order: SOUL → USER → MEMORY → yesterday's note → today's note.
    Truncates from the middle (preserves SOUL priority and recent notes).
    """
    ensure_defaults()

    today = date.today()
    yesterday = today - timedelta(days=1)

    sections = [
        _read_file(config.SOUL_PATH, "SOUL"),
        _read_file(config.USER_PATH, "USER"),
        _read_file(config.MEMORY_PATH, "MEMORY"),
        _read_file(_daily_note_path(yesterday), f"NOTES {yesterday.isoformat()}"),
        _read_file(_daily_note_path(today), f"NOTES {today.isoformat()}"),
    ]

    combined = "\n".join(s for s in sections if s)

    if len(combined) > config.MEMORY_MAX_CHARS:
        # Keep SOUL (first section) and recent notes (last sections) — trim MEMORY from end
        soul = sections[0]
        rest = "\n".join(s for s in sections[1:] if s)
        budget = config.MEMORY_MAX_CHARS - len(soul) - 50  # 50 chars for truncation marker
        if budget > 0 and len(rest) > budget:
            rest = rest[-budget:]
            rest = "[...truncated...]\n" + rest
        combined = soul + "\n" + rest

    return combined


def append_daily_note(entry: str) -> None:
    """Append a timestamped line to today's daily note."""
    from datetime import datetime

    path = _daily_note_path(date.today())
    timestamp = datetime.now().strftime("%H:%M")
    line = f"- [{timestamp}] {entry}\n"

    if not path.exists():
        path.write_text(f"# {date.today().isoformat()}\n\n{line}", encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def save_to_memory(fact: str) -> str:
    """Append a fact to MEMORY.md. Returns confirmation string."""
    ensure_defaults()
    with config.MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    log.info("Saved to memory: %s", fact[:80])
    return f"Saved: {fact}"


def save_to_profile(fact: str) -> str:
    """Append a fact to USER.md. Returns confirmation string."""
    ensure_defaults()
    with config.USER_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    log.info("Saved to profile: %s", fact[:80])
    return f"Saved to profile: {fact}"
