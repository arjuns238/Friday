"""Web search via Tavily API."""
from __future__ import annotations

import asyncio
import logging

from friday import config

log = logging.getLogger(__name__)


async def web_search(query: str) -> str:
    """Search the web and return a concise summary of results.

    Returns formatted results as a string for GPT-4o to synthesize
    into a spoken response, or directly speak if results are short.
    """
    log.info("Web search: %r", query)

    if not config.TAVILY_API_KEY:
        return f"Web search unavailable (no TAVILY_API_KEY). Query was: {query}"

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _search_sync(query)
        )
        return result
    except Exception as exc:
        log.error("Web search failed: %s", exc)
        return f"Web search failed: {exc}"


def _search_sync(query: str) -> str:
    """Blocking Tavily search — run in executor."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    parts = []

    # Direct answer if Tavily provides one
    if response.get("answer"):
        parts.append(response["answer"])

    # Top results
    for result in response.get("results", [])[:3]:
        title = result.get("title", "")
        content = result.get("content", "")[:300]
        url = result.get("url", "")
        parts.append(f"• {title}: {content} ({url})")

    return "\n".join(parts) if parts else "No results found."
