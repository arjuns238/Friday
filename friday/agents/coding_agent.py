"""
Programmer subagent — plans coding tasks, delegates execution to Claude Code.

Architecture (3 layers):
  Friday (voice) → Programmer (plans) → Claude Code (executes)

The Programmer reasons about the task and instructs Claude Code. If it needs
clarification, it speaks a question via TTS, stops, and waits. When the user
responds, Friday re-dispatches to the same project_dir — session resume gives
the Programmer full context of the prior conversation + its question.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, cast

if TYPE_CHECKING:
    from claude_agent_sdk import PermissionMode

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)

from friday import config

log = logging.getLogger(__name__)

_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions"}

# Prefix the Programmer uses when it needs user clarification
_QUESTION_PREFIX = "QUESTION:"

PROGRAMMER_SYSTEM_PROMPT = """\
You are the Programmer — a specialist coding subagent dispatched by Friday, \
a voice-first AI assistant.

## Your role
You plan and oversee coding tasks. You do NOT modify files directly — Claude Code \
does that. You receive a task from Friday, reason about it, explore the codebase \
to understand context, then instruct Claude Code with a clear, detailed spec of \
what to change.

## How you receive work
Friday sends you a task description based on what the user said and what was \
on their screen. Treat it as a brief from a product manager: understand the \
intent, figure out the right approach, then execute.

## How you work
1. Read and explore the relevant code first. Understand the codebase before \
   planning any changes.
2. Think through the approach. Consider edge cases, existing patterns, and \
   potential breakage.
3. Make focused, minimal changes that solve the task. Don't refactor unrelated code.
4. Run tests or linters if the project has them, to verify your changes work.

## Asking for clarification
If the task is genuinely ambiguous and you could go in very different directions, \
you may ask the user for clarification. To do this, prefix your ENTIRE final \
message with "QUESTION:" followed by a concise, spoken question. Then STOP — \
do not proceed with any code changes.

Example: "QUESTION: The auth module has both JWT and session-based auth. Should I \
refactor both, or just the JWT flow?"

Friday will speak your question aloud. When the user responds, you will receive \
their answer and can continue. Your session is preserved — you keep full context \
of what you already explored.

Only ask when truly necessary. If you can make a reasonable default choice, do that \
instead — the user is talking to a voice assistant and doesn't want to be peppered \
with questions.

## CRITICAL: Deletion policy
NEVER delete files, functions, classes, or significant blocks of code unless the \
task explicitly asks for deletion. When refactoring or moving code, keep the old \
code until the new code is verified working. If you must remove something, \
confirm it is truly unused first by searching for all references.

## What you report back
Your final message will be spoken aloud to the user via text-to-speech. Write it as \
a concise, natural spoken summary (2-3 sentences max):
- What you did
- Key files changed
- Whether tests pass (if applicable)

Bad: "I have completed the refactoring of the authentication module by extracting..."
Good: "Refactored the auth module into three files. Login, signup, and token refresh \
each have their own module now. All 12 tests pass."
"""


class CodingAgent:
    """Programmer subagent — plans tasks, delegates to Claude Code."""

    def __init__(self, speak_fn: Callable[[str], Awaitable[None]]) -> None:
        self._sessions: dict[str, str] = {}              # project_dir → session_id
        self._active_tasks: dict[str, asyncio.Task] = {} # task_id → Task
        self._speak = speak_fn                            # Friday's TTS function
        self._awaiting_response: dict[str, str] = {}     # project_dir → question text

    def dispatch(self, task_id: str, prompt: str, project_dir: str) -> None:
        """Fire-and-forget. Returns immediately. Task runs in background."""
        # If the Programmer was waiting for clarification on this project,
        # clear that state — the user's new prompt IS the answer.
        self._awaiting_response.pop(project_dir, None)

        task = asyncio.create_task(
            self._run(task_id, prompt, project_dir),
            name=f"programmer:{task_id}",
        )
        self._active_tasks[task_id] = task
        task.add_done_callback(lambda _: self._active_tasks.pop(task_id, None))

    async def _run(self, task_id: str, prompt: str, project_dir: str) -> None:
        prior_session = self._sessions.get(project_dir)

        permission_mode = config.CLAUDE_PERMISSION_MODE
        if permission_mode not in _PERMISSION_MODES:
            permission_mode = "acceptEdits"

        options = ClaudeAgentOptions(
            system_prompt=PROGRAMMER_SYSTEM_PROMPT,
            cwd=project_dir,
            allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            permission_mode=cast("PermissionMode", permission_mode),
            resume=prior_session,
        )

        tools_used: list[str] = []
        result_text = ""

        try:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            tools_used.append(block.name)
                elif isinstance(msg, ResultMessage):
                    result_text = msg.result or ""
                    if msg.session_id:
                        self._sessions[project_dir] = msg.session_id

            # Check if the Programmer is asking a question
            if result_text.strip().upper().startswith(_QUESTION_PREFIX):
                question = result_text.strip()[len(_QUESTION_PREFIX):].strip()
                self._awaiting_response[project_dir] = question
                log.info("Programmer asking for clarification: %s", question[:100])
                await self._speak(question)
            else:
                summary = result_text[:400] or f"Done. Used: {', '.join(set(tools_used))}."
                await self._speak(summary)

        except asyncio.CancelledError:
            log.info("Programmer task %s cancelled", task_id)
            await self._speak("Coding task cancelled.")
        except Exception as exc:
            log.exception("Programmer task %s error: %s", task_id, exc)
            await self._speak(f"Hit an error on the coding task: {str(exc)[:200]}")

    def status(self) -> str:
        parts: list[str] = []

        if self._awaiting_response:
            for proj, question in self._awaiting_response.items():
                parts.append(f"Waiting for your answer on {proj}: {question[:100]}")

        if self._active_tasks:
            tasks = ", ".join(self._active_tasks.keys())
            parts.append(f"{len(self._active_tasks)} task(s) running: {tasks}")

        return " ".join(parts) if parts else "No active coding tasks."

    def cancel(self, task_id: str | None = None) -> str:
        if task_id:
            t = self._active_tasks.get(task_id)
            if t:
                t.cancel()
                return f"Cancelled task {task_id}."
            return f"Task {task_id!r} not found."
        count = len(self._active_tasks)
        for t in list(self._active_tasks.values()):
            t.cancel()
        return f"Cancelled {count} task(s)."
