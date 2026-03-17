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

        if cmd == "start-claude":
            from friday.tools.claude_code import start_claude_via_pipe
            print("Starting Claude Code via named pipe...")
            proc = start_claude_via_pipe()
            print(f"Claude Code running (pid={proc.pid}). Press Ctrl+C to stop.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
            return

        if cmd == "test-pipeline":
            # Quick sanity check without the menu bar
            import asyncio
            from friday.pipeline import Pipeline

            async def _test():
                p = Pipeline(on_state_change=lambda s: print(f"  state: {s}"))
                await p.run()

            print("Running one pipeline invocation (speak after hotkey)...")
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
  friday start-claude    Start Claude Code connected to Friday via named pipe
  friday test-pipeline   Run one invocation cycle (no menu bar)
  friday help            Show this message
"""


if __name__ == "__main__":
    main()
