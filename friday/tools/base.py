"""Tool registry and base definitions for GPT-4o tool calling."""
from __future__ import annotations

from typing import Any

# OpenAI function-calling tool definitions
TOOL_DEFINITIONS: list[dict] = [
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
                    "thinking": {
                        "type": "string",
                        "description": "Natural, brief phrase to say aloud before executing. e.g. 'Let me draft that email'. Keep under 8 words.",
                    },
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
                "required": ["thinking", "to", "subject", "body_instructions"],
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
                    "thinking": {
                        "type": "string",
                        "description": "Natural, brief phrase to say aloud before searching. e.g. 'Let me look that up', 'Let me check on that'. Keep under 8 words.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query string, optimized for web search.",
                    },
                },
                "required": ["thinking", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_query",
            "description": (
                "Search and reason over files, photos, and data on the user's Mac. "
                "Use for: finding files, answering questions about personal data, "
                "opening files, photo location queries, download history, etc. "
                "Examples: 'find my resume', 'screenshots from yesterday', "
                "'what did I download last week', 'pictures of the beach'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase to say aloud while searching. e.g. 'Let me look through your files'. Keep under 8 words.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The user's query about their local files/data.",
                    },
                },
                "required": ["thinking", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": (
                "Capture a screenshot of what the user is currently looking at. "
                "Use this FIRST when the user references something visual on their screen — "
                "e.g. 'look at this', 'what's on my screen', 'this code', 'this email', "
                "'read this', 'what do you see', 'check this out'. "
                "After calling this tool, you will receive the screenshot and be asked to "
                "decide which action to take based on what you see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": (
                            "Natural, brief phrase to say aloud while capturing. "
                            "e.g. 'Let me take a look', 'Checking your screen'. "
                            "Keep under 8 words."
                        ),
                    },
                },
                "required": ["thinking"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Open a file or reveal in Finder. Use when the file path is already known "
                "from a previous desktop_query result or conversation history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase to say aloud. e.g. 'Opening that now'. Keep under 8 words.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to open.",
                    },
                    "reveal": {
                        "type": "boolean",
                        "description": "If true, reveal in Finder instead of opening. Default false.",
                    },
                },
                "required": ["thinking", "path"],
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
    arguments = {k: v for k, v in arguments.items() if k != "thinking"}

    if name == "draft_gmail":
        from friday.tools.gmail import draft_gmail
        return await draft_gmail(
            to=arguments["to"],
            subject=arguments["subject"],
            body_instructions=arguments["body_instructions"],
        )

    elif name == "web_search":
        from friday.tools.search import web_search
        return await web_search(arguments["query"])

    elif name == "desktop_query":
        from friday.tools.desktop import DesktopAgent
        agent = DesktopAgent()
        return await agent.run(arguments["query"])

    elif name == "open_file":
        from friday.tools.desktop import open_file
        return open_file(arguments["path"], arguments.get("reveal", False))
    elif name == "take_screenshot":
        from friday.capture.screenshot import capture_focused_display
        import asyncio
        loop = asyncio.get_running_loop()
        screenshot_b64 = await loop.run_in_executor(None, capture_focused_display)
        return screenshot_b64 or ""

    elif name == "speak_answer":
        return arguments["answer"]

    else:
        return f"Unknown tool: {name}"
