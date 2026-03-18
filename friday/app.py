"""macOS menu bar app — the top-level entry point.

Uses rumps for the menu bar and pynput for the global hotkey.
Push-to-talk: hold hotkey to record, release to process.
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
        self._stop_event: Optional[threading.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self._loop_thread.start()

        self._hotkey_listener = _build_hotkey_listener(
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )
        self._hotkey_listener.start()

        log.info("Friday started — hold %s to speak", config.HOTKEY)

    @rumps.clicked("About Friday")
    def about(self, _):
        rumps.alert(
            title="Friday",
            message=(
                f"Voice-first AI orchestrator\n"
                f"Hold {config.HOTKEY} to speak\n"
                f"Model: {config.LLM_PROVIDER}"
            ),
        )

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _on_hotkey_press(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._run_pipeline(self._stop_event), self._loop
        )

    def _on_hotkey_release(self) -> None:
        if self._stop_event:
            self._stop_event.set()

    async def _run_pipeline(self, stop_event: threading.Event) -> None:
        try:
            await self._pipeline.run(stop_event)
        finally:
            self._running = False
            self._stop_event = None

    def _on_state_change(self, state: str) -> None:
        self.title = _STATE_ICONS.get(state, _STATE_ICONS["idle"])


def _parse_hotkey(hotkey_str: str) -> frozenset:
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


def _build_hotkey_listener(on_press, on_release):
    from pynput import keyboard

    hotkey_combo = _parse_hotkey(config.HOTKEY)
    current_keys: set = set()

    class _Listener(keyboard.Listener):
        def __init__(self):
            super().__init__(on_press=self._on_press, on_release=self._on_release)
            self._held = False

        def _on_press(self, key):
            current_keys.add(key)
            if hotkey_combo.issubset(current_keys) and not self._held:
                self._held = True
                on_press()

        def _on_release(self, key):
            if self._held and key in hotkey_combo:
                self._held = False
                on_release()
            current_keys.discard(key)

    return _Listener()


def run() -> None:
    missing = config.validate_phase0()
    if missing:
        rumps.alert(
            title="Friday — Missing API Keys",
            message=(
                "Missing required environment variables:\n"
                + "\n".join(f"  • {k}" for k in missing)
                + "\n\nCopy .env.example to .env and fill in your keys."
            ),
        )

    app = FridayApp()
    app.run()
