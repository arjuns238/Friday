"""Main invocation pipeline — orchestrates all stages from hotkey to spoken response.

Flow:
  hotkey press
    → screenshot (parallel with audio capture start)
    → audio capture (VAD auto-stop)
    → Deepgram transcription
    → GPT-4o vision + tool routing
    → tool execution
    → ElevenLabs TTS playback

All stages are instrumented with timing logs.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)


class Pipeline:
    """Manages one invocation cycle: hotkey → spoken response."""

    def __init__(self, on_state_change: Optional[Callable[[str], None]] = None) -> None:
        """
        Args:
            on_state_change: Optional callback called with state strings like
                'recording', 'processing', 'speaking', 'idle'.
                Used by the menu bar app to update the status indicator.
        """
        self._on_state = on_state_change or (lambda s: None)

    async def run(self) -> None:
        """Execute one full invocation. Called when hotkey is pressed."""
        t_total = time.monotonic()
        log.info("=== Friday invocation started ===")

        try:
            await self._run_inner()
        except Exception as exc:
            log.exception("Pipeline error: %s", exc)
            self._on_state("idle")
            # Speak an error so the user knows something went wrong
            from friday.speak.elevenlabs import speak
            await speak("Sorry, something went wrong. Check the logs.")

        elapsed = (time.monotonic() - t_total) * 1000
        log.info("=== Invocation complete in %.0f ms ===", elapsed)

    async def _run_inner(self) -> None:
        # ── Stage 1: Screenshot (fire immediately on hotkey) ─────────────────
        t0 = time.monotonic()
        screenshot_task = asyncio.create_task(self._take_screenshot())

        # ── Stage 2: Audio capture ─────────────────────────────────────────
        self._on_state("recording")
        from friday.capture.audio import record_audio

        audio_bytes = await record_audio()
        t_audio = time.monotonic()
        log.info("Stage audio: %.0f ms (%d bytes)", (t_audio - t0) * 1000, len(audio_bytes))

        if not audio_bytes:
            log.warning("No audio captured, aborting")
            self._on_state("idle")
            return

        # ── Stage 3: Transcription (parallel with screenshot finishing) ───────
        self._on_state("processing")
        from friday.transcribe.deepgram import transcribe

        transcript_task = asyncio.create_task(transcribe(audio_bytes))

        # Wait for both screenshot and transcript
        screenshot_b64, transcript = await asyncio.gather(
            screenshot_task, transcript_task
        )

        t_transcript = time.monotonic()
        log.info("Stage transcript: %.0f ms: %r", (t_transcript - t0) * 1000, transcript)

        if not transcript:
            log.warning("Empty transcript, aborting")
            self._on_state("idle")
            return

        # ── Stage 4: GPT-4o orchestration ────────────────────────────────────
        from friday.orchestrate.gpt4o import orchestrate

        tool_name, result = await orchestrate(transcript, screenshot_b64)
        t_orchestrate = time.monotonic()
        log.info(
            "Stage orchestrate: %.0f ms → tool=%s",
            (t_orchestrate - t0) * 1000,
            tool_name,
        )

        # ── Stage 5: Speak result ─────────────────────────────────────────────
        self._on_state("speaking")
        spoken_text = _result_to_speech(tool_name, result)

        from friday.speak.elevenlabs import speak
        await speak(spoken_text)

        t_end = time.monotonic()
        log.info("Stage speak: %.0f ms", (t_end - t_orchestrate) * 1000)
        log.info(
            "Timing breakdown — audio: %.0fms, transcript: %.0fms, "
            "orchestrate: %.0fms, speak: %.0fms, TOTAL: %.0fms",
            (t_audio - t0) * 1000,
            (t_transcript - t_audio) * 1000,
            (t_orchestrate - t_transcript) * 1000,
            (t_end - t_orchestrate) * 1000,
            (t_end - t0) * 1000,
        )

        self._on_state("idle")

    async def _take_screenshot(self) -> Optional[str]:
        """Take screenshot in executor to avoid blocking event loop."""
        from friday.capture.screenshot import capture_focused_display
        return await asyncio.get_event_loop().run_in_executor(
            None, capture_focused_display
        )


def _result_to_speech(tool_name: str, result: str) -> str:
    """Convert a tool result into a natural spoken response."""
    if tool_name == "speak_answer":
        return result

    elif tool_name == "inject_claude_code":
        if "failed" in result:
            return f"I couldn't inject into Claude Code. {result}"
        return "I've sent that to Claude Code."

    elif tool_name == "draft_gmail":
        return result  # Gmail tool returns its own spoken confirmation

    elif tool_name == "web_search":
        # result is raw search output — GPT-4o should synthesize this
        # For now, return first 300 chars as a spoken summary
        return result[:300] if result else "I couldn't find anything for that search."

    return result
