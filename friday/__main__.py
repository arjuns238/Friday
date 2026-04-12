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

    # Sub-commands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "setup-gmail":
            from friday.tools.gmail import setup_gmail_auth
            setup_gmail_auth()
            return

        if cmd == "test-pipeline":
            import asyncio
            import threading
            from datetime import date
            from friday.graph import build_graph
            from friday.speak.elevenlabs import speak
            from friday.tools.claude_code import init_coding_agent
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            async def _test():
                init_coding_agent(speak)
                stop = threading.Event()
                mute = threading.Event()
                asyncio.get_event_loop().call_later(120, stop.set)
                print("  Speak now... (auto-stops after 120s or press Ctrl+C)")
                async with AsyncSqliteSaver.from_conn_string(str(config.DB_PATH)) as checkpointer:
                    graph = build_graph(checkpointer)
                    await graph.ainvoke(
                        {"done": False},
                        config={
                            "configurable": {
                                "thread_id": date.today().isoformat(),
                                "stop_event": stop,
                                "mute_event": mute,
                                "on_state_change": lambda s: print(f"  state: {s}"),
                            }
                        },
                    )

            print("Running one pipeline invocation...")
            asyncio.run(_test())
            return

        if cmd in ("-h", "--help", "help"):
            print(__doc_help__)
            return

        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc_help__)
        sys.exit(1)

    # Default: run the menu bar app
    from friday.app import run
    run()


__doc_help__ = """
Friday — voice-first AI orchestrator

Usage:
  friday                 Start the menu bar app
  friday setup-gmail     Authenticate with Gmail (one-time OAuth2 setup)
  friday test-pipeline   Run one invocation cycle (no menu bar)
  friday help            Show this message
"""


if __name__ == "__main__":
    main()
