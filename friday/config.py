"""Centralised configuration — reads .env, exposes typed settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).parent.parent
load_dotenv(_repo_root / ".env", override=False)

# ── API keys ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

# ── Hotkey ────────────────────────────────────────────────────────────────────
MUTE_KEY: str = os.environ.get("FRIDAY_MUTE_KEY", "ctrl+m")

# ── Voice / TTS ───────────────────────────────────────────────────────────────
ELEVENLABS_VOICE_ID: str = os.environ.get(
    "FRIDAY_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2"  # Alice
)
TTS_VOLUME: float = float(os.environ.get("FRIDAY_TTS_VOLUME", "2"))

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1
MAX_RECORDING_SECONDS: int = 30
VAD_SPEECH_THRESHOLD: int = int(os.environ.get("FRIDAY_VAD_THRESHOLD", "600"))
VAD_ONSET_FRAMES: int = int(os.environ.get("FRIDAY_VAD_ONSET_FRAMES", "5"))
VAD_OFFSET_FRAMES: int = int(os.environ.get("FRIDAY_VAD_OFFSET_FRAMES", "25"))
VAD_PRE_ROLL_FRAMES: int = 10

# ── Screenshot ────────────────────────────────────────────────────────────────
SCREENSHOT_MAX_KB: int = int(os.environ.get("FRIDAY_SCREENSHOT_MAX_KB", "400"))
SCREENSHOT_JPEG_QUALITY: int = 80

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.environ.get("FRIDAY_LLM", "gemini")

_LLM_CONFIGS: dict[str, dict] = {
    "gemini": {
        "model":    "gemini-3.1-flash-lite-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key":  lambda: GOOGLE_API_KEY,
    },
    "openai": {
        "model":    "gpt-4o",
        "base_url": None,
        "api_key":  lambda: OPENAI_API_KEY,
    },
    "claude": {
        "model":    "claude-haiku-4-5-20251001",
        "base_url": "https://api.anthropic.com/v1/",
        "api_key":  lambda: ANTHROPIC_API_KEY,
    },
}


def llm_config() -> dict:
    """Return the active LLM config: {model, base_url, api_key}."""
    cfg = _LLM_CONFIGS.get(LLM_PROVIDER)
    if cfg is None:
        raise ValueError(
            f"Unknown FRIDAY_LLM provider: {LLM_PROVIDER!r}. "
            f"Choose: {list(_LLM_CONFIGS)}"
        )
    return {
        "model":    cfg["model"],
        "base_url": cfg["base_url"],
        "api_key":  cfg["api_key"](),
    }


# ── Paths ─────────────────────────────────────────────────────────────────────
FRIDAY_DIR: Path = Path.home() / ".friday"
FRIDAY_DIR.mkdir(exist_ok=True)
SOUL_PATH: Path = FRIDAY_DIR / "SOUL.md"
USER_PATH: Path = FRIDAY_DIR / "USER.md"
MEMORY_PATH: Path = FRIDAY_DIR / "MEMORY.md"
MEMORY_MAX_CHARS: int = int(os.environ.get("FRIDAY_MEMORY_MAX_CHARS", "8000"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("FRIDAY_LOG_LEVEL", "INFO")


def validate_keys() -> list[str]:
    """Return list of missing required keys for the current LLM provider."""
    missing = []
    for name, val in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
    ]:
        if not val:
            missing.append(name)

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
