"""Claude Code CLI injection via osascript (AppleScript).

Sends a prompt to the active terminal session running `claude`.
Supports iTerm2 (preferred) and Terminal.app (new-tab fallback).

Phase 4 upgrade path: named pipe via ~/.friday/claude_input.pipe
for direct stdin injection without requiring a specific terminal emulator.
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

# Characters that need escaping in AppleScript quoted strings
_APPLESCRIPT_ESCAPE = str.maketrans({
    '"': '\\"',
    "\\": "\\\\",
})


def inject_into_claude_code(prompt: str) -> str:
    """Type a prompt into the active terminal window running Claude Code.

    Tries iTerm2 first (supports writing to existing session),
    then falls back to Terminal.app (opens new tab — less ideal).

    Returns a status string describing what happened.
    """
    escaped = prompt.translate(_APPLESCRIPT_ESCAPE)

    # ── iTerm2 (preferred) ────────────────────────────────────────────────────
    iterm_script = f'''
tell application "iTerm2"
    tell current session of current window
        write text "{escaped}"
    end tell
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", iterm_script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        log.info("Injected into iTerm2: %r", prompt[:80])
        return "injected via iTerm2"

    log.debug("iTerm2 injection failed (%s), trying Terminal.app", result.stderr.strip())

    # ── Terminal.app (fallback — opens new tab) ───────────────────────────────
    terminal_script = f'''
tell application "Terminal"
    do script "{escaped}" in front window
    activate
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", terminal_script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        log.info("Injected via Terminal.app (new tab): %r", prompt[:80])
        return "injected via Terminal.app (new tab — use iTerm2 for in-session injection)"

    log.error("Both injection methods failed: %s", result.stderr.strip())
    return f"injection failed: {result.stderr.strip()}"


# ── Named pipe injection (Phase 4) ────────────────────────────────────────────

def inject_via_pipe(prompt: str) -> str:
    """Write prompt to the named pipe connected to Claude Code's stdin.

    Requires Claude Code to be started via `friday start-claude` which
    sets up the pipe and launches: tail -f ~/.friday/claude_input.pipe | claude
    """
    from friday import config

    pipe_path = config.CLAUDE_PIPE_PATH
    if not pipe_path.exists():
        return "pipe not found — start Claude Code via `friday start-claude`"

    try:
        with open(pipe_path, "w") as f:
            f.write(prompt + "\n")
        log.info("Injected via named pipe: %r", prompt[:80])
        return "injected via named pipe"
    except OSError as exc:
        log.error("Pipe injection failed: %s", exc)
        return f"pipe injection failed: {exc}"


def start_claude_via_pipe() -> subprocess.Popen:
    """Launch Claude Code connected to the named pipe. Returns the Popen handle.

    Sets up ~/.friday/claude_input.pipe and starts:
        tail -f <pipe> | claude
    """
    import os
    from friday import config

    pipe_path = config.CLAUDE_PIPE_PATH
    if not pipe_path.exists():
        os.mkfifo(str(pipe_path))
        log.info("Created named pipe at %s", pipe_path)

    proc = subprocess.Popen(
        f"tail -f {pipe_path} | claude",
        shell=True,
    )
    log.info("Started Claude Code via pipe (pid=%d)", proc.pid)
    return proc
