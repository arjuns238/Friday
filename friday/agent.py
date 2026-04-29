"""Factory for the Friday deepagents orchestrator.

Architecture:
  Orchestrator (Claude Haiku 4.5, via ChatAnthropic)
    ├─ orchestrator tools: save_memory, memory_search, take_screenshot, open_file, speak_answer
    ├─ shared state: /state/*.json (ephemeral, per-thread, via StateBackend)
    ├─ durable memory files loaded via `memory=[...]` (SOUL/USER/MEMORY + today's note)
    └─ subagents: desktop, research, email (Gemini Flash Lite via ChatOpenAI)
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


ORCH_PROMPT = """You are Friday, a voice-first AI orchestrator running on the user's Mac. The user speaks to you via a hotkey-activated mic; your replies are spoken aloud via TTS.

# Communication
- Keep replies to 2-3 sentences, conversational, no markdown, no URLs, no bullet points.
- Use contractions. Don't preface with "Sure!" / "Absolutely!" / "Great question!".
- If no tool and no subagent is needed, use `speak_answer` to reply directly.

# Delegation
You have three subagents (call via the `task` tool):
- desktop: search, browse, read files on the user's Mac.
- research: web search + synthesis (current events, prices, releases, facts you don't know).
- email: draft Gmail messages from user intent. Never sends.

# CRITICAL — resolving references from shared state
Before delegating a follow-up that references a previously surfaced file ("that", "the certificate", "the third one", "that PDF"), ALWAYS first read `/state/last_listing.json` and `/state/recent_paths.json` via the `read_file` built-in tool. Then pass the resolved absolute path explicitly in your subagent call, so the subagent doesn't have to re-search.

Example:
  Turn 1 user: "What's in my Documents folder?"
    → task(subagent="desktop", query="list the contents of ~/Documents")
  Turn 2 user: "Read the certificate and tell me what course it was for"
    → read_file("/state/last_listing.json")  # find the certificate path
    → task(subagent="desktop", query="read /Users/.../certificate.pdf and summarize what course it's for")
  Turn 3 user: "Open that file"
    → read_file("/state/recent_paths.json")  # most-recent path
    → open_file(path="/Users/.../certificate.pdf")

# Direct tools (no subagent needed)
- open_file(path, reveal?) — open a known-path file or reveal in Finder.
- save_memory(fact, category) — persist a user preference or fact across sessions. "profile" for identity/tech-stack, "memory" for everything else.
- memory_search(query) — search prior memories and daily notes.
- take_screenshot() — capture the focused display when the user references visible content.

# Style
- Match the user's energy. Brief question → brief answer. Detailed question → real explanation.
- Accuracy over speed. Never guess file paths or project names.
- When uncertain, say "I'm not sure" and offer a next step.

Every user turn, respond with exactly one of: a direct `speak_answer`, a subagent `task` call, or a direct tool call. Do not emit empty output.
"""


def _today_note_path() -> Path:
    from friday import config
    return config.MEMORY_DIR / f"{date.today().isoformat()}.md"


def _ensure_today_note() -> Path:
    path = _today_note_path()
    if not path.exists():
        path.write_text(f"# {date.today().isoformat()}\n\n", encoding="utf-8")
    return path


def build_friday_agent(checkpointer=None):
    """Build and compile the Friday deepagents orchestrator."""
    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic

    from friday import config
    from friday.tools.base import ORCH_TOOLS
    from friday.memory.context import ensure_defaults
    from friday.agents import DESKTOP_SA, RESEARCH_SA, EMAIL_SA

    ensure_defaults()
    today_note = _ensure_today_note()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — the Friday orchestrator uses Claude Haiku 4.5."
        )

    orchestrator_model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=1024,
    )

    memory_files = [
        str(config.SOUL_PATH),
        str(config.USER_PATH),
        str(config.MEMORY_PATH),
        str(today_note),
    ]

    agent = create_deep_agent(
        model=orchestrator_model,
        tools=ORCH_TOOLS,
        system_prompt=ORCH_PROMPT,
        subagents=[DESKTOP_SA, RESEARCH_SA, EMAIL_SA],
        memory=memory_files,
        checkpointer=checkpointer,
    )

    log.info("Friday agent compiled (orchestrator=Haiku 4.5, subagents=3)")
    return agent
