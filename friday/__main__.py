"""Entry point for `python -m friday` and the `friday` CLI command."""
from __future__ import annotations

import logging
import sys

from friday import config


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "test-pipeline":
            import asyncio
            import threading

            import tempfile
            from pathlib import Path

            from friday.ambient.conversation_log import ConversationJsonlLog, new_session_id
            from friday.ambient.loop import AmbientLoop
            from friday.ambient.session_log import SessionLog
            from friday.loop import Loop

            async def _test() -> None:
                stop = threading.Event()
                mute = threading.Event()
                asyncio.get_event_loop().call_later(120, stop.set)
                print("  Speak now... (auto-stops after 120s or press Ctrl+C)")
                session_log = SessionLog()
                with tempfile.TemporaryDirectory() as td:
                    conv = ConversationJsonlLog(
                        Path(td) / f"{new_session_id()}.jsonl"
                    )
                    voice = Loop(
                        on_state_change=lambda s: print(f"  state: {s}"),
                        session_log=session_log,
                        conversation_log=conv,
                    )
                    ambient = AmbientLoop(session_log, stop, mute, voice_loop=voice)
                    await asyncio.gather(ambient.run(), voice.run(stop, mute))

            print("Running Friday loop (test mode)...")
            asyncio.run(_test())
            return

        if cmd in ("-h", "--help", "help"):
            print(__doc_help__)
            return

        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc_help__)
        sys.exit(1)

    from friday.app import run
    run()


__doc_help__ = """
Friday — voice-first AI orchestrator

Usage:
  friday                 Start the menu bar app
  friday test-pipeline   Run one 120s loop session (no menu bar)
  friday help            Show this message
"""


if __name__ == "__main__":
    main()
