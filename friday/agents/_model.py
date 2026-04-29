"""Shared factory for the Gemini-backed subagent model.

Every subagent uses `desktop_llm_config()` (Gemini Flash Lite via Google's
OpenAI-compatible endpoint), wrapped in a LangChain `ChatOpenAI`.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def subagent_model():
    from langchain_openai import ChatOpenAI
    from friday.config import desktop_llm_config

    cfg = desktop_llm_config()
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        max_tokens=1024,
    )
