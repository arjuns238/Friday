"""NOW.md — session bridge on shutdown; read at startup for seeding."""
from __future__ import annotations

import logging
import time
from datetime import datetime

from friday import config

log = logging.getLogger(__name__)


def read_now_md() -> str:
    if not config.NOW_PATH.exists():
        return ""
    try:
        return config.NOW_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read NOW.md: %s", exc)
        return ""


async def write_now_md(session_log) -> None:
    """Summarize last raw entries + session file tail into NOW.md."""
    from openai import AsyncOpenAI

    recent = session_log.get_recent(5)
    from friday.memory import read_today_session_markdown_excerpt

    tail = read_today_session_markdown_excerpt(max_chars=2000)
    lines = [
        f"[{e.get('time','')}] {e.get('activity','')} in {e.get('app','')}: {e.get('detail','')}"
        for e in recent
    ]
    user = (
        "Recent raw observations:\n"
        + "\n".join(lines)
        + "\n\nTail of today's compressed session log:\n"
        + (tail or "(none)")
    )
    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a concise NOW.md for the user's AI assistant to read on next launch. "
                    "Use this exact markdown structure:\n\n"
                    f"# Now — {stamp}\n\n"
                    "## Active work\n"
                    "(2-4 sentences)\n\n"
                    "## Open threads\n"
                    "(bullet list, short)\n\n"
                    "## Context to carry forward\n"
                    "(2-4 sentences)\n\n"
                    "No extra sections. Plain markdown only."
                ),
            },
            {"role": "user", "content": user},
        ],
        max_tokens=500,
    )
    body = (response.choices[0].message.content or "").strip()
    if not body:
        return
    try:
        config.NOW_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.NOW_PATH.write_text(body + "\n", encoding="utf-8")
        log.info("Wrote NOW.md (%d chars)", len(body))
    except OSError as exc:
        log.warning("Could not write NOW.md: %s", exc)
