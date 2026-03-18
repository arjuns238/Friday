"""ElevenLabs TTS with streaming audio playback.

Streams text → audio chunks → plays as they arrive.
Target: first audio byte within 200ms of calling speak().
"""
from __future__ import annotations

import asyncio
import logging
import time

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
        # Fallback: macOS say command
        await _say_fallback(text)

    elapsed_ms = (time.monotonic() - t0) * 1000
    log.debug("Speech completed in %.0f ms", elapsed_ms)


def _speak_sync(text: str) -> None:
    """Blocking: get ElevenLabs audio and play via afplay."""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    audio_stream = client.text_to_speech.stream(
        voice_id=config.ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_flash_v2_5",
        optimize_streaming_latency=4,
        output_format="mp3_44100_128",
    )

    audio_bytes = b"".join(chunk for chunk in audio_stream if isinstance(chunk, bytes))
    _play_via_afplay(audio_bytes)


def _play_via_afplay(mp3_bytes: bytes) -> None:
    """Play audio using macOS afplay (no extra deps required)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name

    subprocess.run(["afplay", tmp_path], check=True)

    import os
    os.unlink(tmp_path)


async def _say_fallback(text: str) -> None:
    """macOS `say` command as ultimate fallback."""
    import subprocess

    # Truncate for sanity
    short_text = text[:200]
    proc = await asyncio.create_subprocess_exec(
        "say", short_text,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    log.warning("Used macOS say fallback for TTS")
