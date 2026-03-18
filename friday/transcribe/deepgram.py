"""Deepgram transcription — SDK v6 API."""
from __future__ import annotations

import logging
import struct
import time

from friday import config

log = logging.getLogger(__name__)


async def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw PCM audio (int16, mono, 16kHz) via Deepgram Nova-3.

    Returns transcript string, or empty string on failure.
    """
    if not audio_bytes:
        return ""

    t0 = time.monotonic()
    try:
        from deepgram import AsyncDeepgramClient

        client = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)
        wav_bytes = _add_wav_header(audio_bytes)

        response = await client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
            language="en",
            smart_format=True,
            punctuate=True,
        )

        transcript = response.results.channels[0].alternatives[0].transcript.strip()

    except Exception as exc:
        log.error("Deepgram transcription failed: %s", exc)
        return ""

    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info("Transcription (%.0f ms): %r", elapsed_ms, transcript)
    return transcript


def _add_wav_header(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM (int16, mono, 16kHz) in a minimal WAV header."""
    sample_rate = config.AUDIO_SAMPLE_RATE
    num_channels = config.AUDIO_CHANNELS
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size,
        b"WAVE",
        b"fmt ", 16,
        1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm_bytes
