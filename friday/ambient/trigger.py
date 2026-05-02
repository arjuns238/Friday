"""Ambient monitor: structured signal for the main agent (not final speech)."""
from __future__ import annotations

import json
import logging
import re
import threading
import time

from friday import config

log = logging.getLogger(__name__)

_TRIGGER_PROMPT = """You are the ambient background monitor for Friday, a Mac voice assistant.
You only see recent structured screen-activity lines (not a live screenshot in this message).

Activity log:
{entries}

Your job: produce a brief internal signal for the MAIN agent. The main agent has SOUL/personality,
full memory, session context, and conversation history — it will decide whether to speak and what to say.

Return a single JSON object only (no markdown code fences), exactly one of:

1) {{"surface": false}} — default; use when nothing is worth handing off.

2) {{"surface": true, "reason": "<why this might warrant an unprompted interruption>",
     "observation": "<concrete inference from the log>",
     "suggested_intervention": "<angle or fix to suggest — not scripted dialog>"}}

Be conservative. Prefer {{"surface": false}}."""


def format_activity_digest(entries: list[dict[str, str]]) -> str:
    lines = [
        f"[{e.get('time', '')}] {e.get('activity', '')} in {e.get('app', '')}: {e.get('detail', '')}"
        for e in entries
    ]
    return "\n".join(lines)


def _hard_rules_block(entries: list[dict[str, str]], mute_event: threading.Event) -> bool:
    """True → skip LLM; no signal."""
    if mute_event.is_set():
        return True
    if len(entries) >= 3:
        acts = {e.get("activity", "").lower() for e in entries[-3:]}
        if len(acts) == 1:
            return True
    return False


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def parse_proactive_trigger_payload(raw: str) -> dict[str, str] | None:
    """Parse ambient LLM output → reason/observation/suggested_intervention or None."""
    t = _strip_json_fence(raw)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        log.warning("Proactive trigger JSON parse failed: %r", raw[:240])
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("surface") is False:
        return None
    keys = ("reason", "observation", "suggested_intervention")
    if not all(k in obj for k in keys):
        return None
    out = {k: str(obj.get(k, "")).strip() for k in keys}
    if not all(out.values()):
        return None
    return out


async def evaluate_proactive_trigger(
    entries: list[dict[str, str]],
    mute_event: threading.Event,
) -> dict[str, str] | None:
    """Background LLM: structured explanation for main agent, or None."""
    if _hard_rules_block(entries, mute_event):
        return None

    from openai import AsyncOpenAI

    block = format_activity_digest(entries)
    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "user", "content": _TRIGGER_PROMPT.format(entries=block)},
        ],
        max_tokens=400,
    )
    raw = (response.choices[0].message.content or "").strip()
    log.info("Proactive trigger %.0f ms: %r", (time.monotonic() - t0) * 1000, raw[:280])
    return parse_proactive_trigger_payload(raw)
