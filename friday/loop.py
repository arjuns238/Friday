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
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from friday.ambient.conversation_log import ConversationJsonlLog
    from friday.ambient.session_log import SessionLog

log = logging.getLogger(__name__)

_HISTORY_MAX_MESSAGES = 36

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

    def __init__(
        self,
        on_state_change: Optional[Callable[[str], None]] = None,
        session_log: Optional["SessionLog"] = None,
        conversation_log: Optional["ConversationJsonlLog"] = None,
    ) -> None:
        self._on_state = on_state_change or (lambda _: None)
        self._history: list[dict] = []  # OpenAI-format {role, content} dicts
        self._session_log = session_log
        self._conversation_log = conversation_log
        # Proactive TTS waits until the user has heard at least one reactive reply.
        self._proactive_tts_permitted = threading.Event()

    async def interrupt_proactive(self, trigger_output: dict[str, str]) -> Optional[str]:
        """Main agent: ambient structured signal → spoken line or None (SKIP)."""
        from friday.llm import compose_proactive_message
        from friday.memory import load_memory_context

        memory_context = load_memory_context()
        session_context = (
            self._session_log.get_prompt_context()
            if self._session_log is not None
            else ""
        )
        history = self._history[-_HISTORY_MAX_MESSAGES:]
        return await compose_proactive_message(
            memory_context=memory_context,
            session_context=session_context,
            history=history,
            trigger_output=trigger_output,
        )

    def proactive_speech_permitted(self) -> bool:
        """True after the first reactive reply has finished playing (not proactive TTS)."""
        return self._proactive_tts_permitted.is_set()

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
                    self._proactive_tts_permitted.set()
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
                self._proactive_tts_permitted.set()
                return None, None

        finally:
            cancel_event.set()
            await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))

    async def _build_response(self, audio: bytes) -> Optional[str]:
        """Transcribe → LLM plan → (optional screenshot re-plan) → dispatch tool → spoken text."""
        t0 = time.monotonic()

        from friday.capture.screenshot import capture_focused_display
        from friday.llm import run_tool_loop
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

        history = self._history[-_HISTORY_MAX_MESSAGES:]
        memory_context = load_memory_context()
        session_context = (
            self._session_log.get_prompt_context()
            if self._session_log is not None
            else ""
        )

        async def _capture_screenshot() -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, capture_focused_display) or ""

        spoken_text, new_messages = await run_tool_loop(
            transcript=transcript,
            history=history,
            memory_context=memory_context,
            session_context=session_context,
            dispatch_tool=dispatch_tool,
            speak_thinking=speak,
            capture_screenshot=_capture_screenshot,
        )
        log.info("Response ready (%.0fms): %r", (time.monotonic() - t0) * 1000, spoken_text[:80])

        # Persist full tool-loop messages for subsequent turns.
        self._history.extend(new_messages)
        if len(self._history) > _HISTORY_MAX_MESSAGES:
            self._history = self._history[-_HISTORY_MAX_MESSAGES:]

        if self._conversation_log is not None and spoken_text:
            self._conversation_log.append_turn(transcript, spoken_text)

        return spoken_text
