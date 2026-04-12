# Friday — Claude Code Guide

## What This Is

Voice-first AI orchestrator for macOS. Lives in the menu bar (rumps). Hotkey-invoked (default `ctrl+g`). User speaks → Friday sees screen + hears speech → routes to right tool → speaks back.

Primary use case: talk to your computer while coding. Say "add error handling to this function" → Friday dispatches to Claude Code in the background → speaks when done.

---

## Quick Start

```bash
cp .env.example .env   # fill in API keys (see below)
uv sync
python -m friday        # start menu bar app
```

Subcommands:
- `python -m friday` — start menu bar app
- `python -m friday test-pipeline` — single 120s test invocation (no menu bar)
- `python -m friday setup-gmail` — one-time OAuth2 setup for Gmail drafting

**macOS permissions required**: Accessibility, Screen Recording, Microphone (System Settings → Privacy).

---

## Project Structure

```
friday/
├── __main__.py          CLI entry point (subcommands)
├── app.py               rumps menu bar app, hotkey listeners, wires graph
├── config.py            all env vars, typed settings, llm_config()
├── graph.py             LangGraph state machine (4 nodes) ← CORE
├── pipeline.py          DEPRECATED — superseded by graph.py, safe to delete
├── agents/
│   └── coding_agent.py  CodingAgent wrapping claude_agent_sdk.query()
├── capture/
│   ├── audio.py         energy-based VAD + barge-in detection
│   ├── audio_aec.py     hardware AEC via AVAudioEngine (prevents self-interruption)
│   └── screenshot.py    Quartz CGDisplay capture → base64 JPEG
├── transcribe/
│   └── deepgram.py      Deepgram Nova-3 STT (raw PCM → WAV wrapper → API)
├── orchestrate/
│   └── llm.py           plan_tool_call(), synthesize_response()
├── speak/
│   └── elevenlabs.py    ElevenLabs Flash v2.5, interruptible via afplay
└── tools/
    ├── base.py          TOOL_DEFINITIONS (7 tools) + dispatch_tool()
    ├── claude_code.py   thin adapter → CodingAgent
    ├── gmail.py         Gemini body generation + Gmail OAuth draft
    └── search.py        Tavily web search
```

---

## State Machine (graph.py)

Four nodes in a continuous loop per hotkey activation:

```
listen → build → speak → interrupt → listen → ...
         ↕ (barge-in exits build, queues new audio)
```

### Nodes

**`listen_node`**: Wait for speech onset via VAD. If audio already queued (from barge-in), pass through immediately.

**`build_node`**: Transcribe → LLM route → dispatch tool. Two-pass:
1. Text-only routing (no screenshot)
2. If LLM calls `take_screenshot`, capture screen and re-route with visual context

Barge-in detector runs in a parallel thread during build. If user speaks, build is cancelled and new audio is queued.

**`speak_node`**: ElevenLabs TTS via `speak_interruptible()`. Parallel barge-in detection. Appends HumanMessage + AIMessage to conversation history when done.

**`interrupt_node`**: Transcribes barge audio, classifies as resume intent ("go on", "continue") or new query. Resume → re-speak. New query → queue for next cycle.

### State (FridayState TypedDict)

```python
audio: bytes | None           # segment to process
barge_audio: bytes | None     # captured during speak
transcript: str
tool_name: str
tool_args: dict
thinking: str | None          # spoken while tool runs
tool_result: str
response_text: str            # final TTS text
is_resume: bool               # barge was "go on" / "continue"
done: bool                    # stop_event fired → exit
messages: Annotated[list[BaseMessage], add_messages]  # conversation history
```

### Routing rules

- `listen` → `build` (or END if done)
- `build` → `speak` if response_text, else `listen` (barge-during-build or empty transcript), or END
- `speak` → `interrupt` if barge_audio, else `listen`
- `interrupt` → `speak` if is_resume, else `listen`

---

## Configuration (config.py + .env)

All settings live in `config.py`. `.env` overrides via `python-dotenv`.

### Required API keys

| Key | Purpose |
|-----|---------|
| `DEEPGRAM_API_KEY` | STT (always required) |
| `ELEVENLABS_API_KEY` | TTS (always required) |
| `GOOGLE_API_KEY` | Gemini (default LLM + Gmail body generation) |
| `OPENAI_API_KEY` | GPT-4o (if `FRIDAY_LLM=openai`) |
| `ANTHROPIC_API_KEY` | Claude (if `FRIDAY_LLM=claude`) |

### Optional

| Key | Default | Purpose |
|-----|---------|---------|
| `TAVILY_API_KEY` | — | Web search (tool disabled without it) |
| `FRIDAY_LLM` | `gemini` | LLM provider: `gemini\|openai\|claude` |
| `FRIDAY_HOTKEY` | `ctrl+g` | Toggle listening |
| `FRIDAY_MUTE_KEY` | `ctrl+shift+g` | Mute mic without stopping loop |
| `FRIDAY_VOICE_ID` | `EXAVITQu4vr4xnSDxMaL` (Bella) | ElevenLabs voice ID |
| `CLAUDE_DEFAULT_PROJECT_DIR` | `~` | Fallback project dir for Claude Code tasks |
| `CLAUDE_PERMISSION_MODE` | `acceptEdits` | `default\|acceptEdits\|plan\|bypassPermissions` |
| `FRIDAY_VAD_THRESHOLD` | `400` | RMS energy threshold (raise for noisy envs) |
| `FRIDAY_VAD_ONSET_FRAMES` | `5` | Frames (~150ms) to confirm speech onset |
| `FRIDAY_VAD_OFFSET_FRAMES` | `25` | Frames (~750ms) of silence to end segment |
| `FRIDAY_SCREENSHOT_MAX_KB` | `400` | Max screenshot size sent to LLM |
| `FRIDAY_LOG_LEVEL` | `INFO` | Logging level |

### LLM provider config

All three providers use the OpenAI SDK with different `base_url`/`model`:

```python
"gemini":  model="gemini-3.1-flash-lite-preview", base_url=Google OpenAI compat endpoint
"openai":  model="gpt-4o", base_url=None (default OpenAI)
"claude":  model="claude-haiku-4-5-20251001", base_url=Anthropic OpenAI compat endpoint
```

Switch via `FRIDAY_LLM=openai` (or gemini/claude). Active config returned by `config.llm_config()`.

---

## Tools (7 registered)

Defined as OpenAI function schemas in `tools/base.py`. Dispatched by `dispatch_tool(name, args)`.

| Tool | Handler | Description |
|------|---------|-------------|
| `speak_answer` | inline | Direct TTS response, no side effects |
| `take_screenshot` | `capture/screenshot.py` | Triggers 2-pass re-routing with visual context |
| `inject_claude_code` | `tools/claude_code.py` → `CodingAgent` | Fire-and-forget background coding task |
| `coding_agent_status` | `tools/claude_code.py` | Report active tasks / pending questions |
| `cancel_coding_task` | `tools/claude_code.py` | Cancel task by ID or all tasks |
| `draft_gmail` | `tools/gmail.py` | Gemini body generation + Gmail API draft (never sends) |
| `web_search` | `tools/search.py` → Tavily | Search + LLM synthesis into spoken response |

### Adding a new tool

1. Add function schema to `TOOL_DEFINITIONS` list in `tools/base.py`
2. Add handler function and wire in `dispatch_tool()` switch
3. Add spoken response handling in `_build_spoken_response()` in `graph.py` (if not returning directly)

---

## CodingAgent (agents/coding_agent.py)

3-layer architecture:
```
Friday (voice) → CodingAgent/Programmer (plans) → Claude Code (executes files)
```

Uses `claude_agent_sdk.query()` — NOT osascript/terminal injection. Runs in background.

### Key behaviors

- **Session resumption**: `session_id` stored per `project_dir`. Next dispatch to same dir resumes with full context.
- **Fire-and-forget**: `dispatch()` returns immediately. `_run()` is an async task. Speaks result when done.
- **QUESTION: protocol**: If result starts with `QUESTION:`, Programmer speaks the question and stops. Next `dispatch()` to same `project_dir` clears the pending question and resumes the session.
- **Tools granted**: `Read, Edit, Write, Bash, Glob, Grep`
- **Permission mode**: Configurable via `CLAUDE_PERMISSION_MODE` (default: `acceptEdits`)

### Programmer system prompt (summary)

- Plans tasks, explores codebase before making changes
- Does NOT directly modify files — delegates to Claude Code
- Must prefix with `QUESTION:` if clarification needed, then stop
- Final output is 2-3 sentence spoken summary
- NEVER deletes code unless explicitly asked

---

## Audio Pipeline

### VAD (audio.py)

Energy-based, no ML. Per 30ms frame (480 samples @ 16kHz):
1. Calculate RMS of frame
2. If RMS > `VAD_SPEECH_THRESHOLD` (400): increment speech frames counter
3. After `VAD_ONSET_FRAMES` (5) consecutive speech frames: speech onset confirmed
4. Record until `VAD_OFFSET_FRAMES` (25) consecutive silence frames
5. Includes 10-frame pre-roll buffer

Mute: when `mute_event.is_set()`, all frames treated as silence.

### Hardware AEC (audio_aec.py)

Uses `AVAudioEngine` with `inputNode.setVoiceProcessingEnabled(True)` — same kernel path as Zoom/FaceTime/Siri. Prevents Friday from hearing its own TTS through the mic. Resamples hw rate → 16kHz via scipy polyphase filter. Falls back to sounddevice VAD if AVFoundation unavailable.

### Barge-in detection

Runs in a background thread (`_barge_in_sync()`) in parallel to both `build_node` and `speak_node`. Sets `onset_event` when speech detected. Caller polls the event and cancels/interrupts accordingly.

---

## Screenshot Pipeline (capture/screenshot.py)

1. `capture_focused_display()` — main entry
2. `_capture_via_quartz()` — uses pyobjc/CoreGraphics, falls back to `screencapture` CLI
3. `_compress_to_b64()` — downsample if >1920px wide, JPEG quality starts at 80, decrements by 10 until under `SCREENSHOT_MAX_KB`

Returns base64-encoded JPEG string for LLM vision input.

---

## Conversation Memory

LangGraph `AsyncSqliteSaver` at `~/.friday/memory.db`. One thread per day: `thread_id = date.today().isoformat()`.

Last 12 turns injected into each LLM call as `history` (converted from LangChain messages to OpenAI-format dicts via `_messages_to_dicts()`).

`messages` field in `FridayState` uses `add_messages` reducer — automatically appends new messages without replacing history.

---

## Menu Bar App (app.py)

`FridayApp(rumps.App)` class:

- Icons: 🎙 idle, 👂 listening, 🔴 recording, ⚙️ processing, 🔊 speaking, 🔇 muted
- Hotkey parsing: `_build_hotkey_listener()` converts `"ctrl+g"` → pynput key combo
- `_run_pipeline()`: Opens AsyncSqliteSaver, compiles graph, calls `graph.ainvoke()`
- `_on_state_change()`: Icon update callback passed into graph config

Async loop runs in a background thread (rumps is synchronous). `asyncio.run_coroutine_threadsafe()` bridges hotkey callbacks → async tasks.

---

## Dependencies

Package manager: `uv`. Venv at `.venv/`.

Key packages: `rumps`, `pynput`, `sounddevice`, `numpy`, `scipy`, `deepgram-sdk`, `openai`, `elevenlabs`, `langgraph`, `langgraph-checkpoint-sqlite`, `claude-agent-sdk`, `tavily-python`, `pyobjc`, `Pillow`, `python-dotenv`, `httpx`, `google-generativeai`, `google-api-python-client`, `google-auth-oauthlib`

---

## Known Issues / TODOs

1. **`pipeline.py`** — Dead code, superseded by `graph.py`. Safe to delete.
3. **`claude-agent-sdk` package** — Verify PyPI package name matches import `claude_agent_sdk`. Run `pip show claude-agent-sdk` or check `.venv`.
4. **VAD threshold (400 RMS)** — May need tuning per microphone and environment. Raise if false triggers on background noise; lower if quiet speech is missed.
5. **macOS only** — Uses Quartz, `screencapture`, `afplay`, `AVAudioEngine`, `pynput` macOS backend. No Linux/Windows support.

---

## Paths

- `~/.friday/memory.db` — LangGraph conversation history (SQLite)
- `~/.friday/google_creds.json` — Gmail OAuth2 token (written by `setup-gmail`)
- `~/.friday/claude_input.pipe` — Legacy named pipe (unused in current branch)
- `.env` — API keys and config overrides (repo root)

## Working memory
For any non-trivial problem, maintain SCRATCHPAD.md in the root. 
Update it after each approach attempt, not just at session end. 
Include: problem statement, approaches tried + why each failed, 
current hypothesis, next steps, and ruled-out hypotheses.
