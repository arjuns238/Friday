"""Main invocation pipeline — orchestrates all stages from hotkey to spoken response.

Flow (always-on mode):
  hotkey press     → enter listening loop
  speech onset     → screenshot + record until silence
  silence offset   → transcribe → LLM → speak thinking phrase → speak response
  barge-in         → kill afplay → capture interruption → resume or new query
  "go on" / etc.  → re-speak same response
  new query        → process interruption audio as next request

All stages are instrumented with timing logs.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Resume intent: short phrase asking Friday to continue what it was saying
_RESUME_RE = re.compile(
    r"\b(go on|continue|keep going|keep talking|go ahead|resume|"
    r"carry on|please continue|finish|what else|what were you saying|"
    r"yeah go on|yes go on|ok go on|okay go on)\b",
    re.IGNORECASE,
)


def _is_resume_intent(transcript: str) -> bool:
    words = transcript.strip().split()
    return len(words) <= 6 and bool(_RESUME_RE.search(transcript))


class Pipeline:
    """Manages the always-on listen → process → speak loop."""

    def __init__(self, on_state_change: Optional[Callable[[str], None]] = None) -> None:
        self._on_state = on_state_change or (lambda _: None)

    async def run(self, stop_event: threading.Event, mute_event: threading.Event) -> None:
        """Run continuous listen → process loop until stop_event is set."""
        from friday.capture.audio import listen_for_speech
        from friday.speak.elevenlabs import speak_interruptible

        log.info("=== Friday always-on loop started ===")
        self._on_state("listening")

        pending_audio: Optional[bytes] = None  # audio captured during barge-in

        while not stop_event.is_set():
            # Get the next audio segment to process
            if pending_audio is not None:
                audio, pending_audio = pending_audio, None
            else:
                audio = await listen_for_speech(stop_event, mute_event)
                if stop_event.is_set() or audio is None:
                    break

            # Build response — interruptible: if user speaks during processing,
            # cancel and restart with their new audio.
            spoken_text, barge_during_build = await self._build_response_interruptible(
                audio, stop_event, mute_event
            )

            if barge_during_build is not None:
                log.info("Barge-in during processing — restarting with new audio")
                pending_audio = barge_during_build
                self._on_state("listening")
                continue

            if not spoken_text or stop_event.is_set():
                self._on_state("listening")
                continue

            # Inner speak loop: re-speak on resume, break on natural end or new query
            response_to_speak = spoken_text
            while response_to_speak and not stop_event.is_set():
                self._on_state("speaking")
                interruption = await speak_interruptible(response_to_speak, stop_event, mute_event)

                if interruption is None:
                    break  # natural completion

                self._on_state("processing")
                from friday.transcribe.deepgram import transcribe
                barge_transcript = await transcribe(interruption)

                if barge_transcript and _is_resume_intent(barge_transcript):
                    log.info("Resume intent: %r → re-speaking", barge_transcript)
                else:
                    log.info("New query from barge-in: %r", barge_transcript)
                    pending_audio = interruption
                    break

            if not stop_event.is_set():
                self._on_state("listening")

        self._on_state("idle")
        log.info("=== Friday always-on loop stopped ===")

    async def _build_response_interruptible(
        self,
        audio: bytes,
        stop_event: threading.Event,
        mute_event: threading.Event,
    ) -> tuple[Optional[str], Optional[bytes]]:
        """Run _build_response while watching for barge-in on the mic.

        Returns:
            (spoken_text, None)   — processing completed normally
            (None, barge_audio)   — user spoke during processing; reprocess their audio
            (None, None)          — stop_event set or unrecoverable error
        """
        from friday.capture.audio import _barge_in_sync

        loop = asyncio.get_event_loop()
        onset_event = threading.Event()
        cancel_event = threading.Event()
        barge_done_event = threading.Event()
        barge_audio_ref: list[Optional[bytes]] = [None]

        def _run_barge() -> None:
            barge_audio_ref[0] = _barge_in_sync(
                stop_event, mute_event, onset_event, cancel_event
            )
            barge_done_event.set()

        barge_thread = threading.Thread(target=_run_barge, daemon=True)
        barge_thread.start()

        build_task = asyncio.create_task(self._build_response(audio))

        try:
            # Poll every 30ms until build completes, user speaks, or stop is requested
            while not build_task.done() and not onset_event.is_set() and not stop_event.is_set():
                await asyncio.sleep(0.03)

            if onset_event.is_set():
                # User spoke during processing — abort build, wait for their full segment
                if not build_task.done():
                    build_task.cancel()
                    try:
                        await build_task
                    except asyncio.CancelledError:
                        pass
                await loop.run_in_executor(None, lambda: barge_done_event.wait(timeout=30))
                return None, barge_audio_ref[0]

            if stop_event.is_set():
                build_task.cancel()
                try:
                    await build_task
                except asyncio.CancelledError:
                    pass
                return None, None

            # Build completed normally
            try:
                return build_task.result(), None
            except Exception as exc:
                log.exception("Build response error: %s", exc)
                from friday.speak.elevenlabs import speak
                self._on_state("speaking")
                await speak("Sorry, something went wrong. Check the logs.")
                return None, None

        finally:
            # Always release the mic before the next phase opens its own stream
            cancel_event.set()
            await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))

    async def _build_response(self, audio: bytes) -> Optional[str]:
        """Transcribe → LLM plan → speak thinking + run tool in parallel → return spoken text."""
        t0 = time.monotonic()

        # Screenshot + transcription in parallel
        self._on_state("processing")
        from friday.transcribe.deepgram import transcribe

        screenshot_task = asyncio.create_task(self._take_screenshot())
        transcript_task = asyncio.create_task(transcribe(audio))
        screenshot_b64, transcript = await asyncio.gather(screenshot_task, transcript_task)

        log.info("Transcript (%.0fms): %r", (time.monotonic() - t0) * 1000, transcript)

        if not transcript:
            log.warning("Empty transcript, skipping")
            return None

        # LLM decides which tool + thinking phrase (does NOT run the tool yet)
        from friday.orchestrate.llm import plan_tool_call
        tool_name, arguments, thinking = await plan_tool_call(transcript, screenshot_b64)

        log.info("Plan (%.0fms): tool=%s thinking=%r", (time.monotonic() - t0) * 1000, tool_name, thinking)

        # Speak thinking phrase AND execute tool simultaneously
        from friday.tools.base import dispatch_tool
        from friday.speak.elevenlabs import speak

        if thinking:
            tool_result, _ = await asyncio.gather(
                dispatch_tool(tool_name, arguments),
                speak(thinking),
            )
        else:
            tool_result = await dispatch_tool(tool_name, arguments)

        # Build the final spoken response from the tool result
        spoken_text = await _build_spoken_response(transcript, tool_name, tool_result)

        log.info("Response ready (%.0fms): %r", (time.monotonic() - t0) * 1000, spoken_text[:80])
        return spoken_text

    async def _take_screenshot(self) -> Optional[str]:
        from friday.capture.screenshot import capture_focused_display
        return await asyncio.get_event_loop().run_in_executor(
            None, capture_focused_display
        )


async def _build_spoken_response(transcript: str, tool_name: str, result: str) -> str:
    if tool_name == "speak_answer":
        return result
    elif tool_name == "draft_gmail":
        return result
    elif tool_name == "web_search":
        from friday.orchestrate.llm import synthesize_response
        return await synthesize_response(transcript, result)
    return result
