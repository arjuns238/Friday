"""Background ambient loop: capture → extract → log; compress; proactive trigger."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

from friday import config
from friday.ambient.extractor import extract_context
from friday.ambient.trigger import evaluate_proactive_trigger

if TYPE_CHECKING:
    from friday.ambient.session_log import SessionLog
    from friday.loop import Loop

log = logging.getLogger(__name__)

SIMULATE_SCENARIOS: dict[str, list[dict[str, str]]] = {
    "stuck_on_error": [
        {
            "time": "10:42",
            "activity": "terminal",
            "app": "iTerm",
            "detail": "pytest failing — AttributeError on session_log",
        },
        {
            "time": "10:43",
            "activity": "coding",
            "app": "VS Code",
            "detail": "editing ambient/session_log.py",
        },
        {
            "time": "10:44",
            "activity": "terminal",
            "app": "iTerm",
            "detail": "pytest failing — same error",
        },
        {
            "time": "10:45",
            "activity": "coding",
            "app": "VS Code",
            "detail": "editing ambient/session_log.py",
        },
        {
            "time": "10:46",
            "activity": "terminal",
            "app": "iTerm",
            "detail": "pytest failing — same error",
        },
    ],
    "angry_email": [
        {
            "time": "14:22",
            "activity": "writing",
            "app": "Mail",
            "detail": "composing email to manager",
        },
        {
            "time": "14:23",
            "activity": "reading",
            "app": "Mail",
            "detail": "re-reading thread before sending — third pass",
        },
        {
            "time": "14:24",
            "activity": "writing",
            "app": "Mail",
            "detail": "composing email to manager — heavy edits",
        },
        {
            "time": "14:25",
            "activity": "writing",
            "app": "Mail",
            "detail": "composing email to manager",
        },
        {
            "time": "14:26",
            "activity": "browsing",
            "app": "Safari",
            "detail": "HR policy page — workplace communications",
        },
    ],
    "repeated_azure_task": [
        {
            "time": "09:10",
            "activity": "browsing",
            "app": "Chrome",
            "detail": "Azure portal — App Service slot 1 env vars",
        },
        {
            "time": "09:12",
            "activity": "coding",
            "app": "VS Code",
            "detail": "pasting slot 1 values into infra/env.yaml",
        },
        {
            "time": "09:14",
            "activity": "browsing",
            "app": "Chrome",
            "detail": "Azure portal — App Service slot 2 env vars",
        },
        {
            "time": "09:16",
            "activity": "terminal",
            "app": "iTerm",
            "detail": "az webapp config appsettings list — comparing slots",
        },
        {
            "time": "09:18",
            "activity": "browsing",
            "app": "Chrome",
            "detail": "Azure portal — App Service slot 3 env vars",
        },
    ],
    "context_switch_after_meeting": [
        {
            "time": "15:00",
            "activity": "meeting",
            "app": "Zoom",
            "detail": "call ended",
        },
        {
            "time": "15:01",
            "activity": "coding",
            "app": "VS Code",
            "detail": "opened orchestrator.py",
        },
        {
            "time": "15:02",
            "activity": "terminal",
            "app": "iTerm",
            "detail": "tail -f logs — errors after deploy",
        },
        {
            "time": "15:03",
            "activity": "browsing",
            "app": "Chrome",
            "detail": "Stack Overflow — asyncio gather cancel",
        },
        {
            "time": "15:04",
            "activity": "coding",
            "app": "VS Code",
            "detail": "orchestrator.py — cursor idle ~8 min, no saves",
        },
    ],
}


class AmbientLoop:
    def __init__(
        self,
        session_log: SessionLog,
        stop_event: threading.Event,
        mute_event: threading.Event,
        voice_loop: Loop | None = None,
    ) -> None:
        self._session_log = session_log
        self._stop = stop_event
        self._mute = mute_event
        self._voice = voice_loop

    async def simulate_proactive(self, scenario_name: str) -> None:
        """Dev-only: run the same proactive path as _trigger_loop with fake get_recent entries."""
        from friday.speak.elevenlabs import speak

        log.info("[SIM] simulate_proactive start scenario=%r", scenario_name)
        if scenario_name not in SIMULATE_SCENARIOS:
            log.warning("[SIM] unknown scenario: %r", scenario_name)
            return
        if self._mute.is_set():
            log.info("[SIM] skipped (muted)")
            return
        if self._voice is not None and not self._voice.proactive_speech_permitted():
            log.info(
                "[SIM] skipped (proactive waits until after your first reactive reply plays)"
            )
            return

        entries_full = SIMULATE_SCENARIOS[scenario_name]
        orig_get_recent = self._session_log.get_recent

        def fake_get_recent(n: int) -> list[dict[str, str]]:
            if n <= 0:
                return []
            return [dict(e) for e in entries_full[-n:]]

        try:
            self._session_log.get_recent = fake_get_recent  # type: ignore[method-assign]
            entries = self._session_log.get_recent(5)
            log.info(
                "[SIM] evaluating proactive trigger (%d synthetic entries)",
                len(entries),
            )
            signal = await evaluate_proactive_trigger(entries, self._mute)
            if not signal:
                log.info("[SIM] no signal (hard rules or ambient LLM declined)")
                return
            if self._voice is None:
                log.warning("[SIM] signal but no voice loop — skipping main agent")
                return
            log.info("[SIM] main agent compose (interrupt_proactive)")
            msg = await self._voice.interrupt_proactive(signal)
            if msg and not self._stop.is_set():
                log.info("[SIM] speaking proactive message")
                await speak(msg)
            elif not msg:
                log.info("[SIM] main agent SKIP (no message)")
        finally:
            self._session_log.get_recent = orig_get_recent  # type: ignore[method-assign]
        log.info("[SIM] simulate_proactive end scenario=%r", scenario_name)

    async def run(self) -> None:
        log.info("Ambient loop started")
        try:
            await asyncio.gather(
                self._capture_loop(),
                self._compress_loop(),
                self._trigger_loop(),
            )
        except asyncio.CancelledError:
            log.info("Ambient loop cancelled")
            raise
        finally:
            log.info("Ambient loop stopped")

    async def _sleep_until(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.5, remaining))

    async def _capture_loop(self) -> None:
        from friday.capture.screenshot import capture_focused_display

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                b64 = await loop.run_in_executor(None, capture_focused_display)
                if b64:
                    snap = await extract_context(b64)
                    hhmm = datetime.now().strftime("%H:%M")
                    self._session_log.append(
                        {
                            "time": hhmm,
                            "activity": snap.get("activity", "other"),
                            "app": snap.get("app", "unknown"),
                            "detail": snap.get("detail", ""),
                        }
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Ambient capture/extract failed")
            elapsed = time.monotonic() - t0
            wait = max(1.0, config.FRIDAY_AMBIENT_INTERVAL - elapsed)
            await self._sleep_until(wait)

    async def _compress_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_until(30.0)
            if self._stop.is_set():
                break
            try:
                await self._maybe_compress()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Ambient compression failed")

    async def _maybe_compress(self) -> None:
        if not self._session_log.should_compress():
            return
        compressible = self._session_log.get_compressible()
        if not compressible:
            self._session_log.mark_compression_done()
            return
        summary = await self._llm_summarize_entries(compressible)
        if not summary:
            log.warning("Compression returned empty summary; keeping raw entries")
            return
        self._append_daily_session_file(summary)
        self._session_log.remove_compressed(len(compressible))
        self._session_log.mark_compression_done()

    async def _llm_summarize_entries(self, entries: list[dict[str, str]]) -> str:
        from openai import AsyncOpenAI

        lines = [
            f"[{e.get('time','')}] {e.get('activity','')} in {e.get('app','')}: {e.get('detail','')}"
            for e in entries
        ]
        block = "\n".join(lines)
        cfg = config.llm_config()
        client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        response = await client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize these screen-activity lines into 2-3 tight sentences for a day log. "
                        "Plain prose only — no bullets, no markdown headings."
                    ),
                },
                {"role": "user", "content": block},
            ],
            max_tokens=200,
        )
        return (response.choices[0].message.content or "").strip()

    def _append_daily_session_file(self, summary: str) -> None:
        if not summary:
            return
        day = datetime.now().strftime("%Y-%m-%d")
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.SESSIONS_DIR / f"{day}.md"
        block = f"## {datetime.now().strftime('%H:%M')}\n{summary}\n\n"
        try:
            if not path.exists():
                path.write_text(f"# Session — {day}\n\n", encoding="utf-8")
            with path.open("a", encoding="utf-8") as f:
                f.write(block)
        except OSError as exc:
            log.warning("Could not append session file: %s", exc)

    async def _trigger_loop(self) -> None:
        from friday.speak.elevenlabs import speak

        first_tick = True
        while not self._stop.is_set():
            if not first_tick:
                await self._sleep_until(float(config.FRIDAY_TRIGGER_INTERVAL))
            first_tick = False
            if self._stop.is_set():
                break
            if self._mute.is_set():
                continue
            if self._voice is not None and not self._voice.proactive_speech_permitted():
                log.debug("Proactive skipped: awaiting first reactive reply")
                continue
            try:
                entries = self._session_log.get_recent(5)
                signal = await evaluate_proactive_trigger(entries, self._mute)
                if not signal:
                    continue
                if self._voice is None:
                    log.warning("Proactive signal but no voice loop — skipping main agent")
                    continue
                msg = await self._voice.interrupt_proactive(signal)
                if msg and not self._stop.is_set():
                    await speak(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Proactive trigger failed")
