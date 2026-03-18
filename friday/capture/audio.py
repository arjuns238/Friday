"""Audio capture — always-on VAD mode with mute support.

Listens continuously using energy-based VAD. Returns one speech segment at a
time. Supports a mute_event that silences mic input without stopping the loop.
"""
from __future__ import annotations

import asyncio
import logging
import math
import struct
import threading
from typing import Optional

from friday import config

log = logging.getLogger(__name__)

_FRAME_MS = 30
_FRAME_SAMPLES = config.AUDIO_SAMPLE_RATE * _FRAME_MS // 1000  # 480 @ 16kHz
_SILENCE_FRAME = bytes(_FRAME_SAMPLES * 2)  # all-zero int16 frame


def _rms(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"{n}h", data)
    return math.sqrt(sum(s * s for s in samples) / n)


def _barge_in_sync(
    stop_event: threading.Event,
    mute_event: threading.Event,
    onset_event: threading.Event,
    cancel_event: threading.Event,
) -> Optional[bytes]:
    """Barge-in VAD: waits for speech onset during TTS playback, then records until silence.

    Faster onset than the main VAD (3 frames ≈ 90ms) so interruptions feel instant.
    cancel_event: set by caller when afplay finishes naturally — exits before onset.
    onset_event:  set here when speech detected — caller kills afplay.
    """
    import sounddevice as sd

    pre_roll: list[bytes] = []
    recording: list[bytes] = []
    speech_frames = 0
    silence_frames = 0
    in_speech = False
    max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _FRAME_MS
    _BARGE_ONSET = 3  # frames: faster detection for natural interruption feel

    with sd.RawInputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype="int16",
        blocksize=_FRAME_SAMPLES,
    ) as stream:
        while not stop_event.is_set():
            if not in_speech and cancel_event.is_set():
                break

            data, overflowed = stream.read(_FRAME_SAMPLES)
            if overflowed:
                log.warning("Barge-in overflow")

            frame = _SILENCE_FRAME if mute_event.is_set() else bytes(data)
            rms = _rms(frame)
            is_speech = rms > config.VAD_SPEECH_THRESHOLD

            if not in_speech:
                pre_roll.append(frame)
                if len(pre_roll) > config.VAD_PRE_ROLL_FRAMES:
                    pre_roll.pop(0)

                if is_speech:
                    speech_frames += 1
                    if speech_frames >= _BARGE_ONSET:
                        in_speech = True
                        recording = pre_roll.copy()
                        pre_roll.clear()
                        speech_frames = 0
                        silence_frames = 0
                        onset_event.set()
                        log.debug("Barge-in onset (rms=%.0f)", rms)
                else:
                    speech_frames = max(0, speech_frames - 1)
            else:
                recording.append(frame)
                if len(recording) >= max_frames:
                    break
                if not is_speech:
                    silence_frames += 1
                    if silence_frames >= config.VAD_OFFSET_FRAMES:
                        log.debug("Barge-in segment complete")
                        break
                else:
                    silence_frames = 0

    if not recording:
        return None
    return b"".join(recording)


async def listen_for_speech(
    stop_event: threading.Event,
    mute_event: threading.Event,
) -> Optional[bytes]:
    """Block until speech is detected, record until silence, return PCM bytes.

    Returns None when stop_event is set before speech begins.
    When mute_event is set, all frames are treated as silence so the VAD never
    triggers — the loop still runs and stop_event is still respected.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _listen_sync(stop_event, mute_event)
    )


def _listen_sync(
    stop_event: threading.Event,
    mute_event: threading.Event,
) -> Optional[bytes]:
    import sounddevice as sd

    pre_roll: list[bytes] = []
    recording: list[bytes] = []
    speech_frames = 0
    silence_frames = 0
    in_speech = False
    max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _FRAME_MS

    with sd.RawInputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype="int16",
        blocksize=_FRAME_SAMPLES,
    ) as stream:
        while not stop_event.is_set():
            data, overflowed = stream.read(_FRAME_SAMPLES)
            if overflowed:
                log.warning("Audio input overflow")

            frame = _SILENCE_FRAME if mute_event.is_set() else bytes(data)
            rms = _rms(frame)
            is_speech = rms > config.VAD_SPEECH_THRESHOLD

            if not in_speech:
                # Maintain a rolling pre-roll buffer
                pre_roll.append(frame)
                if len(pre_roll) > config.VAD_PRE_ROLL_FRAMES:
                    pre_roll.pop(0)

                if is_speech:
                    speech_frames += 1
                    if speech_frames >= config.VAD_ONSET_FRAMES:
                        in_speech = True
                        recording = pre_roll.copy()
                        pre_roll.clear()
                        speech_frames = 0
                        silence_frames = 0
                        log.debug("Speech onset (rms=%.0f)", rms)
                else:
                    speech_frames = max(0, speech_frames - 1)

            else:
                recording.append(frame)

                if len(recording) >= max_frames:
                    log.warning("Max recording duration hit, forcing end of segment")
                    break

                if not is_speech:
                    silence_frames += 1
                    if silence_frames >= config.VAD_OFFSET_FRAMES:
                        log.debug("Silence offset detected, segment complete")
                        break
                else:
                    silence_frames = 0

    if not recording:
        return None

    log.debug("Speech segment: %.0fms (%d frames)", len(recording) * _FRAME_MS, len(recording))
    return b"".join(recording)
