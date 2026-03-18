"""Audio capture — push-to-talk mode.

Records while the hotkey is held, stops when it's released.
A threading.Event signals the stop; the caller sets it on key release.
No VAD, no webrtcvad dependency.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

from friday import config

log = logging.getLogger(__name__)

_FRAME_MS = 30
_FRAME_SAMPLES = config.AUDIO_SAMPLE_RATE * _FRAME_MS // 1000  # 480 @ 16kHz


async def record_audio(stop_event: threading.Event) -> bytes:
    """Record until stop_event is set (hotkey released). Returns raw PCM bytes."""
    log.debug("Audio recording started (push-to-talk)")
    t0 = time.monotonic()

    frames = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _record_sync(stop_event)
    )

    elapsed = time.monotonic() - t0
    log.debug("Recorded %.2fs of audio (%d frames)", elapsed, len(frames))
    return b"".join(frames)


def _record_sync(stop_event: threading.Event) -> list[bytes]:
    """Blocking record loop. Exits when stop_event is set or max duration hit."""
    import sounddevice as sd

    frames: list[bytes] = []
    max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _FRAME_MS

    with sd.RawInputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype="int16",
        blocksize=_FRAME_SAMPLES,
    ) as stream:
        for _ in range(max_frames):
            if stop_event.is_set():
                break
            data, overflowed = stream.read(_FRAME_SAMPLES)
            if overflowed:
                log.warning("Audio input overflow")
            frames.append(bytes(data))

    return frames
