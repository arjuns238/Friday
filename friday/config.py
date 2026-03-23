"""Centralised configuration — reads .env, exposes typed settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (walk up from this file's location)
_repo_root = Path(__file__).parent.parent
load_dotenv(_repo_root / ".env", override=False)

# ── API keys ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

# ── Hotkey ────────────────────────────────────────────────────────────────────
# Parsed by app.py into pynput key combos
HOTKEY: str = os.environ.get("FRIDAY_HOTKEY", "ctrl+g")
# Mute toggle key — silences mic without stopping the always-on loop
MUTE_KEY: str = os.environ.get("FRIDAY_MUTE_KEY", "ctrl+shift+g")

# ── Voice / TTS ───────────────────────────────────────────────────────────────
# ElevenLabs voice ID — default: "Rachel" (21m00Tcm4TlvDq8ikWAM)
ELEVENLABS_VOICE_ID: str = os.environ.get(
    "FRIDAY_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"  # Bella (premade, free tier)
)

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1
# Maximum recording duration in seconds (safety cap)
MAX_RECORDING_SECONDS: int = 30
# VAD: RMS energy threshold for speech detection (int16 range 0–32768).
# Raise if false triggers on background noise; lower if quiet speech is missed.
VAD_SPEECH_THRESHOLD: int = int(os.environ.get("FRIDAY_VAD_THRESHOLD", "600"))
# Consecutive speech frames required to confirm onset (~150ms at 30ms/frame)
VAD_ONSET_FRAMES: int = int(os.environ.get("FRIDAY_VAD_ONSET_FRAMES", "5"))
# Consecutive silence frames required to end segment (~300ms)
VAD_OFFSET_FRAMES: int = int(os.environ.get("FRIDAY_VAD_OFFSET_FRAMES", "25"))
# Frames kept before speech onset (pre-roll, ~300ms)
VAD_PRE_ROLL_FRAMES: int = 10

# ── Screenshot ────────────────────────────────────────────────────────────────
SCREENSHOT_MAX_KB: int = int(os.environ.get("FRIDAY_SCREENSHOT_MAX_KB", "400"))
SCREENSHOT_JPEG_QUALITY: int = 80  # starting quality, reduced until under max KB

# ── LLM ──────────────────────────────────────────────────────────────────────
# Set FRIDAY_LLM to switch providers: "gemini" | "openai" | "claude"
# Gemini uses Google's OpenAI-compatible endpoint — no extra SDK needed.
LLM_PROVIDER: str = os.environ.get("FRIDAY_LLM", "gemini")

_LLM_CONFIGS: dict[str, dict] = {
    "gemini": {
        "model":    "gemini-3.1-flash-lite-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key":  lambda: GOOGLE_API_KEY,
    },
    "openai": {
        "model":    "gpt-4o",
        "base_url": None,  # default OpenAI endpoint
        "api_key":  lambda: OPENAI_API_KEY,
    },
    "claude": {
        "model":    "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com/v1/",
        "api_key":  lambda: os.environ.get("ANTHROPIC_API_KEY", ""),
    },
}

DESKTOP_LLM_PROVIDER: str = os.environ.get("FRIDAY_DESKTOP_LLM", "gemini")

# Directories the desktop subagent is allowed to search (Spotlight scope)
DESKTOP_SEARCH_DIRS: list[str] = [
    d.strip()
    for d in os.environ.get(
        "FRIDAY_DESKTOP_SEARCH_DIRS",
        str(Path.home()),
    ).split(",")
    if d.strip()
]


def desktop_llm_config() -> dict:
    """Return LLM config for the desktop subagent (separate from main LLM)."""
    cfg = _LLM_CONFIGS.get(DESKTOP_LLM_PROVIDER)
    if cfg is None:
        raise ValueError(
            f"Unknown FRIDAY_DESKTOP_LLM provider: {DESKTOP_LLM_PROVIDER!r}. "
            f"Choose: {list(_LLM_CONFIGS)}"
        )
    return {
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "api_key": cfg["api_key"](),
    }


def llm_config() -> dict:
    """Return the active LLM config dict: {model, base_url, api_key}."""
    cfg = _LLM_CONFIGS.get(LLM_PROVIDER)
    if cfg is None:
        raise ValueError(f"Unknown FRIDAY_LLM provider: {LLM_PROVIDER!r}. Choose: {list(_LLM_CONFIGS)}")
    return {
        "model":    cfg["model"],
        "base_url": cfg["base_url"],
        "api_key":  cfg["api_key"](),
    }

# ── Paths ─────────────────────────────────────────────────────────────────────
FRIDAY_DIR: Path = Path.home() / ".friday"
FRIDAY_DIR.mkdir(exist_ok=True)
GOOGLE_CREDS_PATH: Path = FRIDAY_DIR / "google_creds.json"
CLAUDE_PIPE_PATH: Path = FRIDAY_DIR / "claude_input.pipe"
DB_PATH: Path = FRIDAY_DIR / "memory.db"   # conversation history (AsyncSqliteSaver)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("FRIDAY_LOG_LEVEL", "INFO")


def validate_phase0() -> list[str]:
    """Return list of missing required keys for Phase 0."""
    missing = []

    # STT + TTS always required
    for name, val in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
    ]:
        if not val:
            missing.append(name)

    # LLM key depends on active provider
    cfg = _LLM_CONFIGS.get(LLM_PROVIDER, {})
    key_val = cfg.get("api_key", lambda: "")()
    key_names = {
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    if not key_val:
        missing.append(key_names.get(LLM_PROVIDER, "LLM_API_KEY"))

    return missing
