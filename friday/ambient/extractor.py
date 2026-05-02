"""Vision LLM: screenshot base64 → structured activity / app / detail."""
from __future__ import annotations

import logging
import re
import time
from friday import config

log = logging.getLogger(__name__)

_SYSTEM = """Look at the screenshot. Reply with exactly three lines and nothing else:
activity: <one of: coding | reading | browsing | writing | terminal | meeting | other>
app: <short application name>
detail: <one concrete artifact: file name, URL, error line, or document title — max 120 chars>

No greetings, no explanation, no markdown."""


def _parse_lines(text: str) -> dict[str, str]:
    out = {"activity": "other", "app": "unknown", "detail": ""}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*activity:\s*(.+)\s*$", line, re.I)
        if m:
            out["activity"] = m.group(1).strip()[:64]
            continue
        m = re.match(r"^\s*app:\s*(.+)\s*$", line, re.I)
        if m:
            out["app"] = m.group(1).strip()[:120]
            continue
        m = re.match(r"^\s*detail:\s*(.+)\s*$", line, re.I)
        if m:
            out["detail"] = m.group(1).strip()[:200]
    return out


async def extract_context(screenshot_b64: str) -> dict[str, str]:
    """Return activity/app/detail dict. Uses configured FRIDAY_LLM provider."""
    if not screenshot_b64:
        return {"activity": "other", "app": "unknown", "detail": "no screenshot"}

    from openai import AsyncOpenAI

    cfg = config.llm_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screenshot_b64}",
                            "detail": "low",
                        },
                    }
                ],
            },
        ],
        max_tokens=100,
    )
    raw = (response.choices[0].message.content or "").strip()
    log.debug("Extractor LLM %.0f ms: %r", (time.monotonic() - t0) * 1000, raw[:200])
    return _parse_lines(raw)
