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
                "Dispatch a coding task to Claude Code. "
                "Claude Code will work autonomously in the background and speak when done. "
                "This tool returns immediately — Friday does not wait. "
                "Use when the user's request is a coding task, refactor, bug fix, or "
                "when the screenshot shows a terminal, VS Code, or code editor. "
                "Formulate a clear, self-contained prompt that Claude Code can act on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": (
                            "Natural, brief phrase to say aloud before dispatching. "
                            "e.g. 'On it', 'Spinning that up', 'Let me get Claude on that'. "
                            "Keep under 8 words."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The exact prompt to send to Claude Code. "
                            "Should be specific, actionable, and reference the code context "
                            "visible in the screenshot."
                        ),
                    },
                    "project_dir": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project directory for Claude Code to work in. "
                            "Infer from the screenshot (terminal cwd, VS Code workspace, file paths). "
                            "Falls back to CLAUDE_DEFAULT_PROJECT_DIR if not determinable."
                        ),
                    },
                },
                "required": ["thinking", "prompt", "project_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "coding_agent_status",
            "description": (
                "Check what Claude Code is currently working on. "
                "Use when the user asks 'what's Claude doing', 'what's it working on', "
                "'is Claude done', 'what's the status'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase to say aloud. e.g. 'Let me check'. Keep under 8 words.",
                    },
                },
                "required": ["thinking"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_coding_task",
            "description": (
                "Cancel an active Claude Code task. "
                "Use when the user says 'stop', 'cancel', 'stop Claude', 'kill that task'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Brief phrase to say aloud. e.g. 'Stopping that'. Keep under 8 words.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": (
                            "Specific task ID to cancel (optional). "
                            "If omitted, cancels all active tasks."
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

    if name == "inject_claude_code":
        from friday import config
        from friday.tools.claude_code import dispatch_claude_code
        project_dir = arguments.get("project_dir") or config.CLAUDE_DEFAULT_PROJECT_DIR
        return dispatch_claude_code(arguments["prompt"], project_dir)

    elif name == "coding_agent_status":
        from friday.tools.claude_code import coding_agent_status
        return coding_agent_status()

    elif name == "cancel_coding_task":
        from friday.tools.claude_code import cancel_coding_task
        return cancel_coding_task(arguments.get("task_id"))

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

    elif name == "take_screenshot":
        from friday.capture.screenshot import capture_focused_display
        import asyncio
        loop = asyncio.get_running_loop()
        screenshot_b64 = await loop.run_in_executor(None, capture_focused_display)
        return screenshot_b64

    elif name == "speak_answer":
        return arguments["answer"]

    else:
        return f"Unknown tool: {name}"
