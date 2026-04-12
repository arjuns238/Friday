"""ElevenLabs TTS with streaming audio playback and barge-in support.

speak()              — non-interruptible (thinking phrases, errors)
speak_interruptible() — kills afplay if user speaks; returns their audio bytes
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from typing import Optional

from friday import config

log = logging.getLogger(__name__)


async def speak(text: str) -> None:
    """Convert text to speech and play it. Returns when playback is complete."""
    if not text or not text.strip():
        return

    log.info("Speaking: %r", text[:100])
    t0 = time.monotonic()

    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: _speak_sync(text))
    except Exception as exc:
        log.error("TTS playback failed: %s", exc)
        await _say_fallback(text)

    log.debug("Speech completed in %.0f ms", (time.monotonic() - t0) * 1000)


async def speak_interruptible(
    text: str,
    stop_event: threading.Event,
    mute_event: threading.Event,
) -> Optional[bytes]:
    """Speak text. If user speaks during playback, stop and return their audio.

    Returns:
        bytes — PCM audio of the interruption (to transcribe & classify)
        None  — playback completed naturally, or stop_event was set
    """
    if not text or not text.strip():
        return None

    from friday.capture.audio import _barge_in_sync

    loop = asyncio.get_event_loop()

    # Download TTS audio first (usually <200ms for Flash v2)
    try:
        mp3_bytes = await loop.run_in_executor(None, lambda: _get_tts_audio(text))
    except Exception as exc:
        log.error("TTS download failed: %s", exc)
        await _say_fallback(text)
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name

    onset_event = threading.Event()   # set by barge thread on speech onset
    cancel_event = threading.Event()  # set by us if afplay finishes first
    barge_result: list[Optional[bytes]] = [None]

    def _run_barge():
        barge_result[0] = _barge_in_sync(stop_event, mute_event, onset_event, cancel_event)

    barge_thread = threading.Thread(target=_run_barge, daemon=True)
    barge_thread.start()

    interrupted = False
    try:
        proc = await asyncio.create_subprocess_exec(
            "afplay", tmp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Poll every 30ms: did speech onset fire, or did afplay finish?
        while not stop_event.is_set():
            if onset_event.is_set():
                proc.kill()
                await proc.wait()
                interrupted = True
                break
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=0.03)
                break  # afplay completed naturally
            except asyncio.TimeoutError:
                continue

        if stop_event.is_set() and not interrupted:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass

    except Exception as exc:
        log.error("afplay failed: %s", exc)
    finally:
        if not interrupted:
            cancel_event.set()
        # Always wait for barge thread to release the mic before returning
        await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if interrupted:
        return barge_result[0]  # may be None if recording failed
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_tts_audio(text: str) -> bytes:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    audio_stream = client.text_to_speech.stream(
        voice_id=config.ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_flash_v2_5",
        optimize_streaming_latency=4,
        output_format="mp3_44100_128",
    )
    return b"".join(chunk for chunk in audio_stream if isinstance(chunk, bytes))


def _speak_sync(text: str) -> None:
    _play_via_afplay(_get_tts_audio(text))


def _play_via_afplay(mp3_bytes: bytes) -> None:
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name
    try:
        subprocess.run(["afplay", tmp_path], check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _say_fallback(text: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "say", "-v", "Samantha", text[:200],
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    log.warning("Used macOS say fallback for TTS")
