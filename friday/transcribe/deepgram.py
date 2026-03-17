"""Deepgram streaming transcription.

Takes raw PCM audio bytes and returns the final transcript string.
Uses Deepgram's pre-recorded audio endpoint for reliability;
streaming WebSocket is used when audio arrives in real-time chunks.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Optional

from friday import config

log = logging.getLogger(__name__)


async def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes via Deepgram.

    Args:
        audio_bytes: Raw PCM int16 mono 16kHz audio.

    Returns:
        Transcript string, or empty string on failure.
    """
    if not audio_bytes:
        return ""

    t0 = time.monotonic()
    try:
        transcript = await _transcribe_prerecorded(audio_bytes)
    except Exception as exc:
        log.error("Deepgram transcription failed: %s", exc)
        return ""

    elapsed_ms = (time.monotonic() - t0) * 1000
    log.info("Transcription (%.0f ms): %r", elapsed_ms, transcript)
    return transcript


async def _transcribe_prerecorded(audio_bytes: bytes) -> str:
    """Send audio to Deepgram pre-recorded endpoint and return transcript."""
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource

    client = DeepgramClient(config.DEEPGRAM_API_KEY)

    options = PrerecordedOptions(
        model="nova-2",
        language="en",
        smart_format=True,
        punctuate=True,
        # Useful for technical vocabulary (code, ML terms)
        keywords=["Claude", "Friday", "GPT", "LLM", "PyTorch", "TensorFlow", "numpy"],
    )

    payload: FileSource = {
        "buffer": audio_bytes,
        "mimetype": "audio/raw",
    }

    # Add WAV header so Deepgram can parse the raw PCM
    wav_bytes = _add_wav_header(audio_bytes)
    payload = {"buffer": wav_bytes, "mimetype": "audio/wav"}

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.listen.prerecorded.v("1").transcribe_file(payload, options),
    )

    # Navigate the response object safely
    try:
        transcript = (
            response.results.channels[0].alternatives[0].transcript
        )
        return transcript.strip()
    except (AttributeError, IndexError, KeyError):
        log.warning("Could not parse Deepgram response: %s", response)
        return ""


def _add_wav_header(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM (int16, mono, 16kHz) in a minimal WAV header."""
    import struct

    sample_rate = config.AUDIO_SAMPLE_RATE
    num_channels = config.AUDIO_CHANNELS
    bits_per_sample = 16
    num_samples = len(pcm_bytes) // (bits_per_sample // 8)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM subchunk size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes
