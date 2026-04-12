"""Thin adapter: bridges Friday's tool dispatch → CodingAgent."""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from friday.agents.coding_agent import CodingAgent

_agent: CodingAgent | None = None


def init_coding_agent(speak_fn: Callable[[str], Awaitable[None]]) -> None:
    """Initialise the singleton CodingAgent. Call once at app startup."""
    global _agent
    _agent = CodingAgent(speak_fn)


def dispatch_claude_code(prompt: str, project_dir: str) -> str:
    """Non-blocking dispatch to CodingAgent. Returns immediate acknowledgement."""
    if _agent is None:
        return "CodingAgent not initialised — call init_coding_agent() at startup."
    task_id = str(uuid.uuid4())[:8]
    _agent.dispatch(task_id, prompt, project_dir)
    return f"Spinning up Claude Code in {project_dir}. I'll tell you when it's done."


def coding_agent_status() -> str:
    return _agent.status() if _agent else "CodingAgent not initialised."


def cancel_coding_task(task_id: str | None = None) -> str:
    return _agent.cancel(task_id) if _agent else "No agent."
