"""macOS menu bar app — the top-level entry point.

Uses rumps for the menu bar and pynput for the mute hotkey.
The always-on listening loop starts automatically on launch.
Mute key (ctrl+m): silences mic input without stopping the loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import rumps

from friday import config

log = logging.getLogger(__name__)

_STATE_ICONS = {
    "idle": "🎙",
    "listening": "👂",
    "muted": "🔇",
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
        self._stop_event = threading.Event()
        self._mute_event = threading.Event()   # set = muted
        self._muted = False
        self._pipeline_state = "idle"
        self._loop: asyncio.AbstractEventLoop | None = None

        self._loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self._loop_thread.start()

        self._mute_listener = _build_hotkey_listener(
            key_combo=config.MUTE_KEY,
            on_trigger=self._on_mute_toggle,
        )
        self._mute_listener.start()

        log.info("Friday started — %s to toggle mute", config.MUTE_KEY)

    @rumps.clicked("About Friday")
    def about(self, _):
        rumps.alert(
            title="Friday",
            message=(
                f"Voice-first AI orchestrator\n"
                f"Press {config.MUTE_KEY} to toggle mute\n"
                f"Model: {config.LLM_PROVIDER}"
            ),
        )

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Start the pipeline immediately
        self._loop.call_soon(
            lambda: asyncio.ensure_future(self._run_pipeline())
        )
        self._loop.run_forever()

    def _on_mute_toggle(self) -> None:
        if self._muted:
            self._mute_event.clear()
            self._muted = False
            log.info("Mic unmuted (state=%s)", self._pipeline_state)
        else:
            self._mute_event.set()
            self._muted = True
            log.info("Mic muted (state=%s)", self._pipeline_state)
        self._refresh_title()

    async def _run_pipeline(self) -> None:
        from friday.loop import Loop

        loop = Loop(on_state_change=self._on_state_change)
        # Restart the loop if it exits unexpectedly (transient error)
        while not self._stop_event.is_set():
            try:
                await loop.run(self._stop_event, self._mute_event)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.error("Loop exited unexpectedly: %s — restarting", exc)
                continue
            break

    def _on_state_change(self, state: str) -> None:
        self._pipeline_state = state
        self._refresh_title()

    def _refresh_title(self) -> None:
        if self._muted and self._pipeline_state in ("idle", "listening"):
            self.title = _STATE_ICONS["muted"]
        else:
            self.title = _STATE_ICONS.get(self._pipeline_state, _STATE_ICONS["idle"])


# macOS virtual key codes — physical key identity, unaffected by modifier keys
_MACOS_CHAR_VK: dict[str, int] = {
    'a': 0,  's': 1,  'd': 2,  'f': 3,  'h': 4,  'g': 5,
    'z': 6,  'x': 7,  'c': 8,  'v': 9,  'b': 11, 'q': 12,
    'w': 13, 'e': 14, 'r': 15, 'y': 16, 't': 17,
    '1': 18, '2': 19, '3': 20, '4': 21, '6': 22, '5': 23,
    '9': 25, '7': 26, '8': 28, '0': 29,
    'o': 31, 'u': 32, 'i': 34, 'p': 35,
    'l': 37, 'j': 38, 'k': 40, 'n': 45, 'm': 46,
}


def _build_hotkey_listener(key_combo: str, on_trigger):
    from pynput import keyboard

    _MOD_MAP = {
        "option": keyboard.Key.alt, "alt": keyboard.Key.alt,
        "cmd": keyboard.Key.cmd, "command": keyboard.Key.cmd,
        "ctrl": keyboard.Key.ctrl, "control": keyboard.Key.ctrl,
        "shift": keyboard.Key.shift,
    }
    _SPECIAL_KEY_MAP = {
        "space": keyboard.Key.space, "enter": keyboard.Key.enter,
        "f1": keyboard.Key.f1, "f2": keyboard.Key.f2,
        "f3": keyboard.Key.f3, "f4": keyboard.Key.f4, "f5": keyboard.Key.f5,
    }

    parts = [p.strip().lower() for p in key_combo.split("+")]
    modifier_keys: set = set()
    char_vks: set = set()
    special_keys: set = set()

    for part in parts:
        if part in _MOD_MAP:
            modifier_keys.add(_MOD_MAP[part])
        elif part in _SPECIAL_KEY_MAP:
            special_keys.add(_SPECIAL_KEY_MAP[part])
        elif part in _MACOS_CHAR_VK:
            char_vks.add(_MACOS_CHAR_VK[part])
        elif len(part) == 1:
            log.warning("No vk mapping for %r — falling back to char compare", part)
            special_keys.add(keyboard.KeyCode.from_char(part))
        else:
            log.warning("Unknown hotkey part: %r", part)

    log.info(
        "Hotkey %r: mods=%s vks=%s special=%s",
        key_combo, modifier_keys, char_vks, special_keys,
    )

    class _Listener(keyboard.Listener):
        def __init__(self):
            super().__init__(on_press=self._on_press, on_release=self._on_release)
            self._held = False
            self._current_mods: set = set()
            self._current_vks: set = set()
            self._current_special: set = set()

        def _on_press(self, key):
            if isinstance(key, keyboard.Key):
                self._current_mods.add(key)
            elif isinstance(key, keyboard.KeyCode):
                if key.vk is not None:
                    self._current_vks.add(key.vk)
                else:
                    self._current_special.add(key)

            if (modifier_keys == self._current_mods
                    and char_vks.issubset(self._current_vks)
                    and special_keys.issubset(self._current_special | self._current_mods)
                    and not self._held):
                self._held = True
                on_trigger()

        def _on_release(self, key):
            if isinstance(key, keyboard.Key):
                if self._held and key in modifier_keys:
                    self._held = False
                self._current_mods.discard(key)
            elif isinstance(key, keyboard.KeyCode):
                if key.vk is not None:
                    if self._held and key.vk in char_vks:
                        self._held = False
                    self._current_vks.discard(key.vk)
                else:
                    self._current_special.discard(key)

    return _Listener()


def run() -> None:
    missing = config.validate_keys()
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
