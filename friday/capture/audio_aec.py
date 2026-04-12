"""AVAudioEngine audio capture with hardware AEC (echo cancellation).

Uses the same kernel-level AEC path as Zoom, FaceTime, and Siri:
  inputNode.setVoiceProcessingEnabled(True)

This prevents Friday from hearing its own TTS output through the mic.

Exposes _listen_sync_aec and _barge_in_sync_aec as drop-in replacements
for the sounddevice equivalents in audio.py.
"""
from __future__ import annotations

import ctypes
import logging
import math
import queue
import struct
import threading
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from friday import config

log = logging.getLogger(__name__)

_FRAME_MS = 30
_FRAME_SAMPLES = config.AUDIO_SAMPLE_RATE * _FRAME_MS // 1000  # 480 @ 16kHz
_SILENCE_FRAME = bytes(_FRAME_SAMPLES * 2)  # all-zero int16 frame
_BARGE_ONSET = 3  # frames — fast onset for natural interruption feel

_engine = None
_engine_lock = threading.Lock()
_resample_up: int = 1    # updated in _get_engine(): up/down = TARGET_RATE/HW_RATE
_resample_down: int = 3  # e.g. 44100→16000: up=160, down=441
_frame_queue: queue.Queue = queue.Queue(maxsize=200)
_partial = np.empty(0, dtype=np.float32)  # tap-thread accumulator
_tap_call_count = 0  # for first-frame diagnostics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rms(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"{n}h", data)
    return math.sqrt(sum(s * s for s in samples) / n)


def _to_int16_bytes(arr: np.ndarray) -> bytes:
    """float32 frame → int16 little-endian bytes."""
    return (arr * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _drain_queue() -> None:
    """Discard queued frames before starting a fresh listen session."""
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# objc_msgSend setup — bypass pyobjc's objc.varlist wrapping for float**
# ---------------------------------------------------------------------------

_libobjc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
_libobjc.sel_registerName.restype = ctypes.c_void_p
_libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
_SEL_FLOAT_CHANNEL_DATA = ctypes.c_void_p(
    _libobjc.sel_registerName(b"floatChannelData")
)
# objc_msgSend signature set at call time (restype varies per call)


def _read_float32(buffer, n: int) -> Optional[np.ndarray]:
    """Read n float32 samples from AVAudioPCMBuffer channel 0.

    The problem: pyobjc wraps float** returns as objc.varlist, which can't
    be passed to ctypes.cast. Solution: use objc_msgSend with the raw ObjC
    id from buffer.__c_void_p__() to get float** as a plain c_void_p.
    """
    # A1: objc_msgSend — raw ObjC call that returns float** as c_void_p
    # buffer.__c_void_p__() gives the raw ObjC object pointer (pyobjc >= 9)
    try:
        buf_id = buffer.__c_void_p__()  # ctypes.c_void_p
        _libobjc.objc_msgSend.restype = ctypes.c_void_p
        _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        float_pp = _libobjc.objc_msgSend(buf_id, _SEL_FLOAT_CHANNEL_DATA)
        if float_pp:
            # Dereference float** → float* (channel 0 address)
            chan0 = ctypes.cast(float_pp, ctypes.POINTER(ctypes.c_void_p))[0]
            if chan0:
                return np.frombuffer(ctypes.string_at(chan0, n * 4), dtype=np.float32).copy()
    except Exception as e:
        log.warning("AEC read A1 (msgSend) failed: %s (%s)", e, type(e).__name__)

    # A2: np.frombuffer directly on ch[0] varlist (buffer protocol — fast if supported)
    try:
        ch = buffer.floatChannelData()
        if ch is not None:
            arr = np.frombuffer(ch[0], dtype=np.float32)[:n].copy()
            if len(arr) == n:
                return arr
    except Exception as e:
        log.warning("AEC read A2 (frombuffer varlist) failed: %s (%s)", e, type(e).__name__)

    # A3: list(ch[0][:n]) — varlist slice → Python list → numpy (medium speed)
    try:
        ch = buffer.floatChannelData()
        if ch is not None:
            arr = np.array(list(ch[0][:n]), dtype=np.float32)
            if len(arr) == n:
                return arr
    except Exception as e:
        log.warning("AEC read A3 (list varlist) failed: %s (%s)", e, type(e).__name__)

    # A4: iterate ch[0][i] — guaranteed if varlist is indexable (slowest fallback)
    try:
        ch = buffer.floatChannelData()
        if ch is not None:
            arr = np.fromiter((ch[0][i] for i in range(n)), dtype=np.float32, count=n)
            return arr.copy()
    except Exception as e:
        log.warning("AEC read A4 (iter varlist) failed: %s (%s)", e, type(e).__name__)

    return None


# ---------------------------------------------------------------------------
# CoreAudio tap callback — runs on CoreAudio's internal thread
# ---------------------------------------------------------------------------

def _tap_callback(buffer, when):
    """Receive AVAudioPCMBuffer. Must not block."""
    global _partial, _tap_call_count

    n = int(buffer.frameLength())
    if n == 0:
        return

    _tap_call_count += 1

    arr = _read_float32(buffer, n)
    if arr is None:
        log.warning(
            "AEC tap: all read approaches failed (call #%d, frameLength=%d, format=%s)",
            _tap_call_count, n, buffer.format(),
        )
        return

    if _tap_call_count == 1:
        log.info("AEC tap active — first frame received (n=%d samples)", n)

    # Resample hw_rate → 16kHz exactly (rational polyphase, anti-aliased)
    decimated = resample_poly(arr, _resample_up, _resample_down).astype(np.float32)
    combined = np.concatenate([_partial, decimated]) if _partial.size else decimated
    n_frames = len(combined) // _FRAME_SAMPLES
    _partial = combined[n_frames * _FRAME_SAMPLES :]

    for i in range(n_frames):
        chunk = combined[i * _FRAME_SAMPLES : (i + 1) * _FRAME_SAMPLES]
        try:
            _frame_queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop rather than block the audio thread


# ---------------------------------------------------------------------------
# Engine singleton — started once, never stopped (maintains AEC echo model)
# ---------------------------------------------------------------------------

def _get_engine():
    global _engine, _decimate

    with _engine_lock:
        if _engine is not None:
            return _engine

        from AVFoundation import AVAudioEngine, AVAudioFormat  # macOS only

        engine = AVAudioEngine.alloc().init()
        input_node = engine.inputNode()

        # Enable hardware AEC (Zoom / FaceTime / Siri path)
        try:
            result = input_node.setVoiceProcessingEnabled_error_(True, None)
            ok = result[0] if isinstance(result, (list, tuple)) else result
            if not ok:
                log.warning("AEC: setVoiceProcessingEnabled returned False — no echo cancellation")
        except Exception as e:
            log.warning("AEC: setVoiceProcessingEnabled failed: %s", e)

        try:
            input_node.setVoiceProcessingAGCEnabled_(True)
        except Exception:
            pass

        # With voice processing enabled, outputFormatForBus_(0) returns a 5-channel format.
        # Request an explicit mono float32 tap — AVAudioEngine will downmix automatically.
        import math
        hw_rate = int(input_node.outputFormatForBus_(0).sampleRate())
        gcd = math.gcd(config.AUDIO_SAMPLE_RATE, hw_rate)
        _resample_up = config.AUDIO_SAMPLE_RATE // gcd   # 16000/100 = 160
        _resample_down = hw_rate // gcd                  # 44100/100 = 441
        log.info("AEC engine: hw_rate=%d, resample %d/%d → %dHz",
                 hw_rate, _resample_up, _resample_down, config.AUDIO_SAMPLE_RATE)

        # AVAudioPCMFormatFloat32 = 1, mono (1 ch), non-interleaved
        mono_format = AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            1, float(hw_rate), 1, False
        )
        log.info("AEC tap format: %s", mono_format)

        input_node.installTapOnBus_bufferSize_format_block_(
            0, 512, mono_format, _tap_callback
        )

        try:
            result = engine.startAndReturnError_(None)
            success = result[0] if isinstance(result, (list, tuple)) else result
        except Exception as e:
            raise RuntimeError(f"AVAudioEngine.startAndReturnError_ failed: {e}") from e

        if not success:
            raise RuntimeError("AVAudioEngine failed to start")

        _engine = engine
        log.info("AVAudioEngine started with AEC (hw_rate=%dHz, resample %d/%d)",
                 hw_rate, _resample_up, _resample_down)

    return _engine


# ---------------------------------------------------------------------------
# Public VAD loops — same semantics as audio.py counterparts
# ---------------------------------------------------------------------------

def _listen_sync_aec(
    stop_event: threading.Event,
    mute_event: threading.Event,
) -> Optional[bytes]:
    """Block until speech detected, record until silence, return int16 PCM bytes."""
    _get_engine()
    _drain_queue()

    pre_roll: list[bytes] = []
    recording: list[bytes] = []
    speech_frames = 0
    silence_frames = 0
    in_speech = False
    max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _FRAME_MS
    idle_log_counter = 0

    while not stop_event.is_set():
        try:
            raw = _frame_queue.get(timeout=0.01)
        except queue.Empty:
            idle_log_counter += 1
            if idle_log_counter % 200 == 0:  # every ~10s
                log.info("AEC listen: queue empty — tap_calls=%d", _tap_call_count)
            continue

        frame = _SILENCE_FRAME if mute_event.is_set() else _to_int16_bytes(raw)
        rms = _rms(frame)
        is_speech = rms > config.VAD_SPEECH_THRESHOLD

        if not in_speech:
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
                    log.debug("AEC speech onset (rms=%.0f)", rms)
            else:
                speech_frames = max(0, speech_frames - 1)
        else:
            recording.append(frame)
            if len(recording) >= max_frames:
                log.warning("AEC: max recording duration hit, forcing end of segment")
                break
            if not is_speech:
                silence_frames += 1
                if silence_frames >= config.VAD_OFFSET_FRAMES:
                    log.debug("AEC silence offset, segment complete")
                    break
            else:
                silence_frames = 0

    if not recording:
        return None

    log.debug("AEC speech: %.0fms (%d frames)", len(recording) * _FRAME_MS, len(recording))
    return b"".join(recording)


def _barge_in_sync_aec(
    stop_event: threading.Event,
    mute_event: threading.Event,
    onset_event: threading.Event,
    cancel_event: threading.Event,
) -> Optional[bytes]:
    """Barge-in VAD: detect speech during TTS playback, record until silence."""
    _get_engine()

    pre_roll: list[bytes] = []
    recording: list[bytes] = []
    speech_frames = 0
    silence_frames = 0
    in_speech = False
    max_frames = (config.MAX_RECORDING_SECONDS * 1000) // _FRAME_MS

    while not stop_event.is_set():
        if not in_speech and cancel_event.is_set():
            break

        try:
            raw = _frame_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        frame = _SILENCE_FRAME if mute_event.is_set() else _to_int16_bytes(raw)
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
                    log.debug("AEC barge-in onset (rms=%.0f)", rms)
            else:
                speech_frames = max(0, speech_frames - 1)
        else:
            recording.append(frame)
            if len(recording) >= max_frames:
                break
            if not is_speech:
                silence_frames += 1
                if silence_frames >= config.VAD_OFFSET_FRAMES:
                    log.debug("AEC barge-in segment complete")
                    break
            else:
                silence_frames = 0

    return b"".join(recording) or None
