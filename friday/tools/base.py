"""Tool registry and base definitions for GPT-4o tool calling."""
from __future__ import annotations

from typing import Any

# OpenAI function-calling tool definitions
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "inject_claude_code",
            "description": (
                "Inject a prompt into the active Claude Code CLI session running in the terminal. "
                "Use when the screenshot shows a terminal running 'claude', VS Code terminal, "
                "or when the user's request is a coding task. "
                "Formulate a clear, self-contained prompt that Claude Code can act on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The exact prompt to inject into Claude Code. "
                            "Should be specific, actionable, and reference the code context visible in the screenshot."
                        ),
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_gmail",
            "description": (
                "Draft an email (never auto-send). Use when the screenshot shows Gmail or Outlook, "
                "or when the user says 'email', 'write to', 'reply to', 'message [person]'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address, extracted from context or inferred from user speech.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body_instructions": {
                        "type": "string",
                        "description": (
                            "Detailed instructions for drafting the email body. "
                            "Include tone, key points, length, and any context from the screenshot."
                        ),
                    },
                },
                "required": ["to", "subject", "body_instructions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web and return results. "
                "Use when the user asks 'what is', 'find', 'look up', 'search for', 'latest', 'who is'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string, optimized for web search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_answer",
            "description": (
                "Speak a direct answer to the user without invoking another tool. "
                "Use for factual questions, explanations, opinions, or when no tool is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The spoken response. Keep under 3 sentences for natural voice interaction.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call by name and return a string result."""
    if name == "inject_claude_code":
        from friday.tools.claude_code import inject_into_claude_code
        return inject_into_claude_code(arguments["prompt"])

    elif name == "draft_gmail":
        from friday.tools.gmail import draft_gmail
        return await draft_gmail(
            to=arguments["to"],
            subject=arguments["subject"],
            body_instructions=arguments["body_instructions"],
        )

    elif name == "web_search":
        from friday.tools.search import web_search
        return await web_search(arguments["query"])

    elif name == "speak_answer":
        # Just return the answer — pipeline.py handles TTS
        return arguments["answer"]

    else:
        return f"Unknown tool: {name}"
