"""Shared state file paths used by subagents + orchestrator.

Every file under `/state/` is ephemeral (StateBackend, per-thread checkpointed).
Files under `/memories/` are durable (StoreBackend).
"""
from __future__ import annotations

STATE_LAST_LISTING = "/state/last_listing.json"
STATE_RECENT_PATHS = "/state/recent_paths.json"
STATE_LAST_RESEARCH = "/state/last_research.json"
STATE_LAST_DRAFT = "/state/last_draft.json"
STATE_LAST_SCREENSHOT = "/state/last_screenshot.png"
