"""Email SubAgent — drafts Gmail messages from user intent. Never sends."""
from __future__ import annotations

from deepagents import SubAgent

from friday.agents._model import subagent_model
from friday.tools.gmail import draft_gmail

EMAIL_PROMPT = """You are Friday's email subagent. You turn natural-language intent into a Gmail draft.

Tool: draft_gmail(to, subject, body_instructions)

Rules:
- Call draft_gmail exactly once. Never send.
- If the recipient, subject, or body intent is unclear, return a clarification question as plain text instead.
- After drafting, write a JSON summary to `/state/last_draft.json` via `write_file`:
  {"to": "...", "subject": "...", "result": "<tool result>"}
- Return the tool's confirmation string (e.g. "Draft created in Gmail...") directly — that's what gets spoken.
"""

EMAIL_SA: SubAgent = {
    "name": "email",
    "description": "Draft an email. Never sends. Always leaves a Gmail draft for user review.",
    "system_prompt": EMAIL_PROMPT,
    "tools": [draft_gmail],
    "model": subagent_model(),
}
