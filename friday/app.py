"""macOS menu bar app — the top-level entry point.

Uses rumps for the menu bar and pynput for the global hotkey.
Hotkey triggers the async pipeline in a background thread.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import rumps

from friday import config
from friday.pipeline import Pipeline

log = logging.getLogger(__name__)

# Menu bar title characters for each state
_STATE_ICONS = {
    "idle": "🎙",
    "recording": "🔴",
    "processing": "⏳",
    "speaking": "🔊",
    "error": "⚠️",
}


class FridayApp(rumps.App):
    """macOS menu bar application."""

    def __init__(self) -> None:
        super().__init__(
            name="Friday",
            title=_STATE_ICONS["idle"],
            quit_button="Quit Friday",
        )
        self._pipeline = Pipeline(on_state_change=self._on_state_change)
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Start the asyncio event loop in a background thread
        self._loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self._loop_thread.start()

        # Register global hotkey
        self._hotkey_listener = _build_hotkey_listener(self._on_hotkey)
        self._hotkey_listener.start()

        log.info("Friday started — hotkey: %s", config.HOTKEY)

    # ── Menu items ────────────────────────────────────────────────────────────

    @rumps.clicked("About Friday")
    def about(self, _):
        rumps.alert(
            title="Friday",
            message=(
                f"Voice-first AI orchestrator\n"
                f"Hotkey: {config.HOTKEY}\n"
                f"Model: {config.OPENAI_MODEL}"
            ),
        )

    @rumps.clicked("Test Microphone")
    def test_mic(self, _):
        self._invoke_pipeline()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _on_hotkey(self) -> None:
        """Called from pynput thread when hotkey is pressed."""
        if self._running:
            log.debug("Hotkey pressed but pipeline already running — ignored")
            return
        self._invoke_pipeline()

    def _invoke_pipeline(self) -> None:
        """Submit pipeline coroutine to background event loop."""
        if self._loop is None:
            return
        self._running = True
        asyncio.run_coroutine_threadsafe(self._run_pipeline(), self._loop)

    async def _run_pipeline(self) -> None:
        try:
            await self._pipeline.run()
        finally:
            self._running = False

    def _on_state_change(self, state: str) -> None:
        """Update menu bar icon to reflect current pipeline state."""
        icon = _STATE_ICONS.get(state, _STATE_ICONS["idle"])
        self.title = icon


def _parse_hotkey(hotkey_str: str):
    """Parse hotkey string like 'cmd+option+space' into pynput Key combo."""
    from pynput import keyboard

    _KEY_MAP = {
        "cmd": keyboard.Key.cmd,
        "command": keyboard.Key.cmd,
        "ctrl": keyboard.Key.ctrl,
        "control": keyboard.Key.ctrl,
        "alt": keyboard.Key.alt,
        "option": keyboard.Key.alt,
        "shift": keyboard.Key.shift,
        "space": keyboard.Key.space,
        "enter": keyboard.Key.enter,
        "f1": keyboard.Key.f1,
        "f2": keyboard.Key.f2,
        "f3": keyboard.Key.f3,
        "f4": keyboard.Key.f4,
        "f5": keyboard.Key.f5,
    }

    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    keys = []
    for part in parts:
        if part in _KEY_MAP:
            keys.append(_KEY_MAP[part])
        elif len(part) == 1:
            keys.append(keyboard.KeyCode.from_char(part))
        else:
            log.warning("Unknown hotkey part: %r", part)

    return frozenset(keys)


def _build_hotkey_listener(callback):
    """Build a pynput GlobalHotKeys listener."""
    from pynput import keyboard

    hotkey_combo = _parse_hotkey(config.HOTKEY)

    current_keys: set = set()

    class _Listener(keyboard.Listener):
        def __init__(self):
            super().__init__(on_press=self._on_press, on_release=self._on_release)
            self._fired = False

        def _on_press(self, key):
            current_keys.add(key)
            if hotkey_combo.issubset(current_keys) and not self._fired:
                self._fired = True
                log.debug("Hotkey triggered")
                callback()

        def _on_release(self, key):
            current_keys.discard(key)
            if not hotkey_combo.issubset(current_keys):
                self._fired = False

    return _Listener()


def run() -> None:
    """Start the menu bar app. Blocks until quit."""
    # Validate config
    missing = config.validate_phase0()
    if missing:
        rumps.alert(
            title="Friday — Missing API Keys",
            message=(
                f"Missing required environment variables:\n"
                + "\n".join(f"  • {k}" for k in missing)
                + f"\n\nCopy .env.example to .env and fill in your keys."
            ),
        )

    app = FridayApp()
    app.run()
