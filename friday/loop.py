"""Friday's main async loop — replaces the old LangGraph state machine.

Plain Python: one coroutine, one in-memory history list, no checkpointer,
no node graph. Same UX as before:
  - always-on VAD
  - barge-in during build (cancel + restart with new audio)
  - barge-in during speak (resume on "go on" / treat as new query)
  - two-pass screenshot routing (text first, capture-and-re-route on take_screenshot)
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)

_HISTORY_MAX_TURNS = 12   # last N user/assistant pairs injected into LLM context

_RESUME_RE = re.compile(
    r"\b(go on|continue|keep going|keep talking|go ahead|resume|"
    r"carry on|please continue|finish|what else|what were you saying|"
    r"yeah go on|yes go on|ok go on|okay go on)\b",
    re.IGNORECASE,
)


def _is_resume_intent(transcript: str) -> bool:
    return len(transcript.strip().split()) <= 6 and bool(_RESUME_RE.search(transcript))


class Loop:
    """Always-on voice loop. One run() call drives the whole app."""

    def __init__(self, on_state_change: Optional[Callable[[str], None]] = None) -> None:
        self._on_state = on_state_change or (lambda _: None)
        self._history: list[dict] = []  # OpenAI-format {role, content} dicts

    async def run(self, stop_event: threading.Event, mute_event: threading.Event) -> None:
        from friday.capture.audio import listen_for_speech
        from friday.speak.elevenlabs import speak_interruptible
        from friday.transcribe.deepgram import transcribe

        log.info("=== Friday loop started ===")
        self._on_state("listening")

        pending_audio: Optional[bytes] = None

        while not stop_event.is_set():
            if pending_audio is not None:
                audio, pending_audio = pending_audio, None
            else:
                audio = await listen_for_speech(stop_event, mute_event)
                if stop_event.is_set() or audio is None:
                    break

            spoken_text, barge_during_build = await self._build_with_barge(
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

            response_to_speak = spoken_text
            while response_to_speak and not stop_event.is_set():
                self._on_state("speaking")
                interruption = await speak_interruptible(response_to_speak, stop_event, mute_event)

                if interruption is None:
                    break

                self._on_state("processing")
                barge_transcript = await transcribe(interruption)

                if barge_transcript and _is_resume_intent(barge_transcript):
                    log.info("Resume intent: %r → re-speaking", barge_transcript)
                    continue

                log.info("New query from barge-in: %r", barge_transcript)
                pending_audio = interruption
                break

            if not stop_event.is_set():
                self._on_state("listening")

        self._on_state("idle")
        log.info("=== Friday loop stopped ===")

    # ── build (transcribe → plan → tool) with parallel barge-in detection ──

    async def _build_with_barge(
        self,
        audio: bytes,
        stop_event: threading.Event,
        mute_event: threading.Event,
    ) -> tuple[Optional[str], Optional[bytes]]:
        """Run _build_response while a parallel thread watches for speech onset.

        Returns:
            (spoken_text, None)   — build completed, here is the response
            (None, barge_audio)   — user spoke during build, restart with their audio
            (None, None)          — stop_event fired or unrecoverable error
        """
        from friday.capture.audio import _barge_in_sync
        from friday.speak.elevenlabs import speak

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
            while not build_task.done() and not onset_event.is_set() and not stop_event.is_set():
                await asyncio.sleep(0.03)

            if onset_event.is_set():
                if not build_task.done():
                    build_task.cancel()
                    try:
                        await build_task
                    except asyncio.CancelledError:
                        pass
                await loop.run_in_executor(None, lambda: barge_done_event.wait(timeout=30))
                # Concatenate so the part spoken before the pause isn't lost
                barge = barge_audio_ref[0] or b""
                combined = (audio or b"") + barge
                log.info(
                    "Barge audio combined (%d+%d=%d bytes)",
                    len(audio or b""), len(barge), len(combined),
                )
                return None, combined

            if stop_event.is_set():
                build_task.cancel()
                try:
                    await build_task
                except asyncio.CancelledError:
                    pass
                return None, None

            try:
                return build_task.result(), None
            except Exception as exc:
                log.exception("Build response error: %s", exc)
                self._on_state("speaking")
                await speak("Sorry, something went wrong. Check the logs.")
                return None, None

        finally:
            cancel_event.set()
            await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))

    async def _build_response(self, audio: bytes) -> Optional[str]:
        """Transcribe → LLM plan → (optional screenshot re-plan) → dispatch tool → spoken text."""
        t0 = time.monotonic()

        from friday.capture.screenshot import capture_focused_display
        from friday.llm import plan_tool_call
        from friday.memory import load_memory_context
        from friday.speak.elevenlabs import speak
        from friday.tools import dispatch_tool
        from friday.transcribe.deepgram import transcribe

        self._on_state("processing")

        transcript = await transcribe(audio)
        log.info("Transcript (%.0fms): %r", (time.monotonic() - t0) * 1000, transcript)

        if not transcript:
            log.warning("Empty transcript, skipping")
            return None

        history = self._history[-_HISTORY_MAX_TURNS * 2:]  # role/content pairs
        memory_context = load_memory_context()

        # Pass 1: text-only routing
        tool_name, arguments, thinking = await plan_tool_call(
            transcript, screenshot_b64=None, history=history, memory_context=memory_context
        )
        log.info("Plan pass 1 (%.0fms): tool=%s", (time.monotonic() - t0) * 1000, tool_name)

        # Pass 2: if LLM wants vision, capture and re-plan
        if tool_name == "take_screenshot":
            if thinking:
                await speak(thinking)
            loop = asyncio.get_running_loop()
            screenshot_b64 = await loop.run_in_executor(None, capture_focused_display)
            tool_name, arguments, thinking = await plan_tool_call(
                transcript,
                screenshot_b64=screenshot_b64,
                history=history,
                memory_context=memory_context,
            )
            log.info("Plan pass 2 (%.0fms): tool=%s", (time.monotonic() - t0) * 1000, tool_name)

        # Speak thinking + run tool in parallel
        if thinking:
            tool_result, _ = await asyncio.gather(
                dispatch_tool(tool_name, arguments),
                speak(thinking),
            )
        else:
            tool_result = await dispatch_tool(tool_name, arguments)

        spoken_text = await _build_spoken_response(transcript, tool_name, tool_result)
        log.info("Response ready (%.0fms): %r", (time.monotonic() - t0) * 1000, spoken_text[:80])

        # Append to in-memory history (will be injected on next turn)
        self._history.append({"role": "user", "content": transcript})
        self._history.append({"role": "assistant", "content": spoken_text})
        # Keep history bounded
        if len(self._history) > _HISTORY_MAX_TURNS * 2:
            self._history = self._history[-_HISTORY_MAX_TURNS * 2:]

        return spoken_text


async def _build_spoken_response(transcript: str, tool_name: str, result: str) -> str:
    """Convert tool result into the actual text to speak."""
    if tool_name in ("speak", "speak_answer"):
        return result
    if tool_name == "web_search":
        from friday.llm import synthesize_response
        return await synthesize_response(transcript, result)
    if tool_name == "save_memory":
        return "Got it, I'll remember that."
    if tool_name == "memory_search":
        from friday.llm import synthesize_response
        return await synthesize_response(transcript, result)
    return result
