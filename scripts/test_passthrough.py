"""Headless text-only harness for the three-turn passthrough scenario.

Skips STT/TTS/mic — drives the deepagents orchestrator directly with typed
prompts on the same thread_id so /state/* scratchpad persists between turns.

Usage:
  .venv/bin/python scripts/test_passthrough.py
"""
from __future__ import annotations

import asyncio
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage


def _extract_text(result: dict) -> str:
    msgs = result.get("messages") or []
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                joined = "".join(parts).strip()
                if joined:
                    return joined
    return "(no text reply)"


def _summarize_tools(result: dict) -> list[str]:
    tools: list[str] = []
    for msg in result.get("messages") or []:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                tools.append(f"{name}({_short(args)})")
    return tools


def _short(args) -> str:
    import json
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    return s if len(s) < 120 else s[:117] + "..."


async def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")

    from friday import config
    from friday.agent import build_friday_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    missing = config.validate_phase0()
    if missing:
        print("Missing env vars:", missing)
        return

    thread_id = f"passthrough-{date.today().isoformat()}"
    invoke_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    prompts = [
        "What's in my Documents folder?",
        "Read the first PDF in that listing and tell me what it's about.",
        "Open that file.",
    ]

    async with AsyncSqliteSaver.from_conn_string(str(config.DB_PATH)) as cp:
        agent = build_friday_agent(cp)

        for i, prompt in enumerate(prompts, 1):
            print(f"\n━━━━━━━━━━━━━━━━━━━━ Turn {i} ━━━━━━━━━━━━━━━━━━━━")
            print(f"USER:      {prompt}")
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=invoke_config,
            )
            tools = _summarize_tools(result)
            for t in tools:
                print(f"  tool:    {t}")
            print(f"FRIDAY:    {_extract_text(result)}")


if __name__ == "__main__":
    asyncio.run(main())
