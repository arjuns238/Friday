"""Audio capture with VAD (Voice Activity Detection).

Starts recording on hotkey press, auto-stops after trailing silence.
Uses sounddevice for low-latency capture and webrtcvad for VAD.

Push-to-talk mode (hold hotkey) is also supported as a simpler alternative.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time

from friday import config

log = logging.getLogger(__name__)

# webrtcvad frame duration must be 10, 20, or 30 ms
_VAD_FRAME_MS = 30
_VAD_FRAME_SAMPLES = config.AUDIO_SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 480 @ 16kHz


class AudioCapture:
    """Records audio until trailing silence is detected or max duration reached.

    Usage::

        capture = AudioCapture()
        audio_bytes = await capture.record()
        # Returns raw PCM bytes (int16, mono, 16kHz)
    """

    def __init__(self) -> None:
        self._vad = _build_vad()

    async def record(self) -> bytes:
        """Record until VAD detects end of speech. Returns raw PCM bytes."""
        log.debug("Audio recording started")
        t0 = time.monotonic()
        frames = await asyncio.get_event_loop().run_in_executor(None, self._record_sync)
        elapsed = time.monotonic() - t0
        log.debug("Recorded %.2fs of audio (%d frames)", elapsed, len(frames))
        return b"".join(frames)

    def _record_sync(self) -> list[bytes]:
        """Blocking audio record. Run in executor to avoid blocking event loop."""
        import sounddevice as sd

        speech_frames: list[bytes] = []

        # Ring buffer of recent VAD results for trailing-silence detection
        trailing_silence_frames = config.VAD_TRAILING_SILENCE_MS // _VAD_FRAME_MS
        ring = collections.deque(maxlen=trailing_silence_frames)

        started_speaking = False
        max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _VAD_FRAME_MS

        with sd.RawInputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            blocksize=_VAD_FRAME_SAMPLES,
        ) as stream:
            for _ in range(max_frames):
                data, overflowed = stream.read(_VAD_FRAME_SAMPLES)
                if overflowed:
                    log.warning("Audio input overflow")

                frame = bytes(data)
                is_speech = self._is_speech(frame)
                ring.append(is_speech)

                if is_speech:
                    started_speaking = True

                if started_speaking:
                    speech_frames.append(frame)

                # Stop when we've seen enough trailing silence after speech
                if started_speaking and len(ring) == trailing_silence_frames:
                    if not any(ring):
                        log.debug("VAD: trailing silence detected, stopping")
                        break

        return speech_frames

    def _is_speech(self, frame: bytes) -> bool:
        """Return True if the frame contains speech."""
        if self._vad is None:
            # No VAD available — treat everything as speech
            return True
        try:
            return self._vad.is_speech(frame, config.AUDIO_SAMPLE_RATE)
        except Exception:
            return True


def _build_vad():
    try:
        import webrtcvad
        vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        return vad
    except ImportError:
        log.warning("webrtcvad not installed — VAD disabled, will use max duration")
        return None


async def record_audio() -> bytes:
    """Convenience function: create AudioCapture and record one utterance."""
    capture = AudioCapture()
    return await capture.record()
