# Friday — Claude Code Guide

## What This Is

Voice-first ambient AI assistant for macOS. Lives in the menu bar (rumps).
Always-on listening loop. Speak → Friday hears + optionally sees screen → routes to a tool or just answers → speaks back.

But Friday is more than a voice assistant. The north star is ambient intelligence — Friday watches your screen continuously, accumulates context about what you're doing, and speaks up at the right moment without being asked. The reactive voice loop is the interface. The ambient context engine is the product.

Plain Python async loop, no orchestration framework. Roughly 250 lines of glue across `loop.py`, `llm.py`, `tools.py`, `memory.py`.

---

## Quick Start

```bash
cp .env.example .env       # fill in API keys
uv sync
python -m friday           # start menu bar app
```

Subcommands:
- `python -m friday` — start menu bar app
- `python -m friday test-pipeline` — single 120s loop session, no menu bar

**macOS permissions required**: Accessibility, Screen Recording, Microphone.

---

## Project Structure

```
friday/
├── __main__.py            CLI entry point
├── app.py                 rumps menu bar, mute hotkey, kicks off Loop
├── config.py              env vars + typed settings + llm_config()
├── loop.py                main async voice loop ← CORE
├── llm.py                 LLM orchestration (plan_tool_call + synthesize_response)
├── tools.py               tools + dispatch + Tavily web search
├── file_search.py         find_files / search_files (rg + Python fallback)
├── memory.py              loads all memory files into system prompt context
├── capture/
│   ├── audio.py           energy-based VAD + barge-in detection
│   ├── audio_aec.py       hardware AEC via AVAudioEngine
│   └── screenshot.py      Quartz CGDisplay capture → base64 JPEG
├── transcribe/
│   └── deepgram.py        Deepgram Nova-3 STT
├── speak/
│   └── elevenlabs.py      ElevenLabs Flash v2.5, interruptible via afplay
└── ambient/               ← NEW (see Ambient Loop section)
    ├── loop.py            background ambient loop (screenshot → extract → log)
    ├── extractor.py       screenshot → structured context snapshot via LLM
    ├── session_log.py     rolling raw log + compression into daily session file
    └── trigger.py         proactive trigger — decides when Friday speaks unprompted
```

---

## Memory System

Friday's memory lives in `~/.friday/` as plain Markdown files. No hidden state — the model only knows what's on disk. This is the same philosophy as OpenClaw: write-first, organize-later, never delete (only supersede).

### Files

```
~/.friday/
├── SOUL.md              personality, communication style, anti-patterns
├── USER.md              user profile (edit by hand)
├── MEMORY.md            long-term durable facts across all sessions
├── NOW.md               ← NEW: session lifeboat (see below)
└── sessions/
    └── YYYY-MM-DD.md    ← NEW: compressed daily session log
```

### Timescale separation — critical design principle

Each file serves a different timescale. Never mix them.

| File | Timescale | Written by | Loaded at |
|------|-----------|-----------|-----------|
| `SOUL.md` | Forever | Human | Every turn |
| `USER.md` | Forever | Human | Every turn |
| `MEMORY.md` | Cross-session durable facts | `save_memory` tool + end-of-session distillation | Every turn |
| `NOW.md` | Session bridge | Written at session end, overwritten each time | Startup only |
| `sessions/YYYY-MM-DD.md` | Today's compressed activity | Ambient loop compression pass | Startup + injected into system prompt |

### `NOW.md` — the session lifeboat

Written automatically when Friday shuts down (or when the ambient loop detects a natural stopping point). Read at startup so Friday wakes up knowing what you were doing.

Format:
```markdown
# Now — 2026-05-01 18:43

## Active work
Building the ambient context loop for Friday. Currently implementing SessionLog
in ambient/session_log.py. Last thing done: wrote the append() and get_recent() methods.

## Open threads
- Need to test ContextExtractor with a real screenshot
- ProactiveTrigger not started yet

## Context to carry forward
Was debugging an async issue where the ambient loop blocked the reactive loop.
Solution was to run AmbientLoop as a separate asyncio task, not a thread.
```

### `sessions/YYYY-MM-DD.md` — daily session log

Compressed summary of today's ambient activity. Written incrementally by the compression pass (not all at once at end of day). The raw session log entries get distilled into this file every 15-20 minutes.

Format:
```markdown
# Session — 2026-05-01

## 09:00–09:45
Read LangGraph state management docs in Chrome. Took notes in VS Code.

## 09:45–11:20
Implemented ambient/session_log.py. Worked through rolling window logic
and JSON persistence. Hit an issue with file locking on concurrent writes — solved
with a threading.Lock.

## 11:20–11:35
Switched to email. Drafted response to recruiter at Basis.
```

### What gets loaded into the system prompt

Every reactive turn loads:

```
[SOUL.md]                          ← always
[USER.md]                          ← always
[MEMORY.md]                        ← always
[sessions/today.md]                ← today's compressed session log
[SessionLog.get_recent(10)]        ← last ~10 minutes, raw entries
```

`NOW.md` is only read at startup to seed the session log — it is NOT injected every turn.

Total context overhead target: under 600 tokens. If today's session file grows beyond ~400 tokens, trigger a compression pass before loading.

---

## The Reactive Loop (`loop.py`) — unchanged

A single coroutine. No state machine, no nodes, no checkpointer.

```
listen → build → speak → (barge?) → listen → ...
```

Three phases per turn:

1. **Listen** — `listen_for_speech(stop_event, mute_event)` blocks on VAD until speech ends.
2. **Build** — runs `_build_response(audio)` while a parallel thread watches for new speech onset:
   - **Inner build**: transcribe (Deepgram) → `plan_tool_call` (LLM) → if LLM picks `take_screenshot`, capture screen and re-plan → `dispatch_tool` (parallel with speaking the "thinking" phrase) → format spoken response.
   - **If user speaks during build**: cancel the build task, concatenate original + new audio, queue it for the next iteration.
3. **Speak** — `speak_interruptible` plays TTS while another barge-in thread runs.
   - **Natural finish**: back to listen.
   - **Interrupted**: transcribe the barge audio. If it's a "go on" / "continue" phrase, re-speak. Otherwise, queue as the next user input.

Conversation history is `self._history: list[dict]` — last 12 turns are injected into each LLM call. In-memory only, lost on restart (intentional).

---

## LLM (`llm.py`)

`plan_tool_call(transcript, screenshot_b64, history, memory_context, session_context="")` returns `(tool_name, arguments, thinking)`.

- `tool_choice="auto"` — the LLM may answer with **plain text instead of a tool call**. Returns `("speak", {"answer": text}, None)`.
- System prompt = `memory_context` + `session_context` + routing rules.
- Provider switched via `FRIDAY_LLM` env: `gemini` (default) | `openai` | `claude`.

`session_context` is passed in from `loop.py` via `session_log.get_prompt_context()` — which returns today's session file + last 10 raw entries formatted as a single string.

`synthesize_response(query, tool_result)` converts raw tool output into a natural spoken sentence.

---

## Tools (`tools.py`)

| Tool | Purpose |
|------|---------|
| `take_screenshot` | Triggers two-pass routing with vision context |
| `web_search` | Tavily search |
| `save_memory` | Append a durable fact to `MEMORY.md` |
| `memory_search` | Substring scan over `MEMORY.md` + `USER.md` |
| `find_files` | Glob file discovery (`file_search.py`) |
| `search_files` | Regex content search (`file_search.py`) |

`dispatch_tool(name, arguments)` is a flat if/elif chain.

---

## Configuration (`config.py` + `.env`)

Required:

| Key | Purpose |
|-----|---------|
| `DEEPGRAM_API_KEY` | STT |
| `ELEVENLABS_API_KEY` | TTS |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | one, matching `FRIDAY_LLM` |

Optional:

| Key | Default | Purpose |
|-----|---------|---------|
| `TAVILY_API_KEY` | — | Web search |
| `FRIDAY_LLM` | `gemini` | `gemini` \| `openai` \| `claude` |
| `FRIDAY_MUTE_KEY` | `ctrl+m` | Mute mic |
| `FRIDAY_VOICE_ID` | Alice | ElevenLabs voice ID |
| `FRIDAY_VAD_THRESHOLD` | `600` | RMS energy threshold |
| `FRIDAY_VAD_ONSET_FRAMES` | `5` | Frames to confirm speech onset |
| `FRIDAY_VAD_OFFSET_FRAMES` | `25` | Frames of silence to end segment |
| `FRIDAY_SCREENSHOT_MAX_KB` | `400` | Max screenshot size sent to LLM |
| `FRIDAY_FILE_SEARCH_DEFAULT_ROOT` | — | Default path for file search |
| `FRIDAY_AMBIENT_INTERVAL` | `60` | Seconds between ambient captures |
| `FRIDAY_TRIGGER_INTERVAL` | `300` | Seconds between proactive trigger checks |
| `FRIDAY_SESSION_LOG_MAX` | `60` | Max raw entries before compression (rolling window) |
| `FRIDAY_COMPRESS_INTERVAL` | `900` | Seconds between session file compression passes (15 min) |
| `FRIDAY_LOG_LEVEL` | `INFO` | |

---

## Audio Pipeline

### VAD (`capture/audio.py`)
Energy-based, no ML. Per 30ms frame: RMS > threshold for `VAD_ONSET_FRAMES` → onset; `VAD_OFFSET_FRAMES` of silence → end.

### Hardware AEC (`capture/audio_aec.py`)
`AVAudioEngine` with voice processing. Prevents Friday hearing its own TTS. Falls back to plain sounddevice.

### Barge-in
`_barge_in_sync` in a background thread. Sets `onset_event` on speech; caller cancels current task and restarts.

---

## Ambient Loop — Active Workstream

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     AMBIENT LOOP                          │
│                                                           │
│  Screenshot every FRIDAY_AMBIENT_INTERVAL secs            │
│       ↓                                                   │
│  ContextExtractor (cheap LLM call, ~100 tokens max)       │
│       ↓                                                   │
│  SessionLog (raw rolling window, in memory + JSON)        │
│       ↓  (every FRIDAY_COMPRESS_INTERVAL secs)            │
│  Compression pass → sessions/YYYY-MM-DD.md                │
│       ↓  (every FRIDAY_TRIGGER_INTERVAL secs)             │
│  ProactiveTrigger → Friday speaks (or stays silent)       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│              REACTIVE LOOP (unchanged)                    │
│                                                           │
│  User speaks → plan_tool_call(session_context=...)        │
│             → response grounded in full session context   │
└──────────────────────────────────────────────────────────┘
```

### Component specs

#### `ambient/session_log.py` — build first

```python
class SessionLog:
    # Raw entries: list of dicts, capped at FRIDAY_SESSION_LOG_MAX
    # Persisted to ~/.friday/session_log.json (survives restart)

    def append(self, entry: dict): ...
    # entry = {"time": "10:42", "activity": "coding", "app": "VS Code",
    #          "detail": "editing ambient/session_log.py"}

    def get_recent(self, n: int) -> list[dict]: ...
    # Returns last n raw entries. Used by ProactiveTrigger.

    def get_prompt_context(self) -> str: ...
    # Returns: today's session file (compressed) + last 10 raw entries
    # formatted as a single string for system prompt injection.
    # Total must stay under ~300 tokens.

    def should_compress(self) -> bool: ...
    # True if raw entries >= FRIDAY_SESSION_LOG_MAX or
    # FRIDAY_COMPRESS_INTERVAL has elapsed since last compression.
```

#### `ambient/extractor.py` — build second

Single LLM call. Input: base64 screenshot (same format as `capture/screenshot.py`).
Output: 3-line structured string.

```
activity: <coding | reading | browsing | writing | terminal | meeting | other>
app: <application name>
detail: <one specific artifact — file name, URL, error, document title>
```

`max_tokens=100`. Use same provider as `FRIDAY_LLM`. Do NOT return prose or summaries.

#### `ambient/loop.py` — build third

Background asyncio task. Never blocks reactive loop. Respects `mute_event` — when muted, keep capturing and logging but skip `ProactiveTrigger`.

Three internal tasks running on their own intervals:
1. Screenshot → extract → append to SessionLog (every `FRIDAY_AMBIENT_INTERVAL`)
2. Compression pass → write to `sessions/YYYY-MM-DD.md` (every `FRIDAY_COMPRESS_INTERVAL`)
3. ProactiveTrigger check (every `FRIDAY_TRIGGER_INTERVAL`)

Also writes `NOW.md` on clean shutdown.

#### `ambient/trigger.py` — build last

Input: `session_log.get_recent(5)`
Output: `None` or 1-2 sentence string

Hard rules (no LLM call needed):
- Last 3 entries all same activity → `None`
- User spoke to Friday within 2 minutes → `None`
- Muted → `None`

LLM prompt (only reached if hard rules pass):
```
You are deciding whether an AI assistant should proactively say something.
Default is NO. Only say YES if ALL of:
1. Directly relevant to what they're doing right now
2. Something they likely haven't thought of
3. Actionable
4. Worth an interruption

Recent activity:
{entries}

Respond with exactly NO, or exactly what Friday should say (1-2 sentences, specific).
```

### Compression pass

Run by `AmbientLoop` every `FRIDAY_COMPRESS_INTERVAL` seconds. Takes the oldest raw entries (everything except the last 10), asks the LLM to summarize them into 2-3 sentences, appends that summary to `sessions/YYYY-MM-DD.md`, then removes those entries from the raw log. The raw log never grows beyond `FRIDAY_SESSION_LOG_MAX`. The session file grows slowly throughout the day.

```python
async def _compress(self):
    entries_to_compress = self.session_log.get_compressible()  # all but last 10
    if not entries_to_compress:
        return
    summary = await self._llm_compress(entries_to_compress)
    self._append_to_session_file(summary)
    self.session_log.remove_compressed(entries_to_compress)
```

### Session context injection into `llm.py`

```python
# Modified signature
async def plan_tool_call(transcript, screenshot_b64, history, memory_context, session_context=""):

# In system prompt
system = f"""{memory_context}

WHAT YOU HAVE BEEN OBSERVING THIS SESSION:
{session_context}

You have been watching the user's screen continuously. Use this context to give
grounded, specific responses. Do not ask them to explain what they're working on —
you already know. Reference specific details when relevant."""
```

### `NOW.md` writer

Called from `app.py` on shutdown (SIGTERM / menu bar quit). Reads the last 5 raw session log entries + last 2 session file summaries, makes a single LLM call to produce the `NOW.md` format shown in the Memory section above.

On startup, `app.py` reads `NOW.md` and seeds the session log with a single synthetic entry: `{"time": startup_time, "activity": "startup", "app": "Friday", "detail": now_md_content}`. This ensures the reactive loop immediately has continuity context without waiting for the first ambient capture.

### Implementation order

1. `SessionLog` — pure data structure, no LLM, test immediately
2. `ContextExtractor` — test in isolation with a real screenshot
3. `AmbientLoop` (capture + log only, no compression, no trigger yet)
4. **Session context injection into `llm.py`** ← validate here first. Friday should now respond as if it's been in the room with you. This is a shippable result on its own.
5. Compression pass + daily session file
6. `NOW.md` writer/reader — session continuity across restarts
7. `ProactiveTrigger` — last, tune carefully, high silence threshold

### Files to modify

- `app.py` — launch `AmbientLoop` as parallel asyncio task; write `NOW.md` on shutdown; seed session log from `NOW.md` on startup
- `llm.py` — add `session_context` param, inject into system prompt
- `loop.py` — pass `session_log.get_prompt_context()` into `plan_tool_call`
- `memory.py` — add `load_session_context()` that reads today's session file
- `config.py` — add new env vars listed above
- `capture/screenshot.py` — verify importable directly outside tool dispatch

### Success criteria

**Step 4**: Friday responds as if it's been watching you. When you say "what was I just doing?", it knows. When you ask about something on screen, it already has context. No re-explaining required.

**Step 6**: Friday restarts and picks up exactly where it left off. `NOW.md` is the bridge.

**Step 7**: Friday occasionally says something unprompted that is genuinely useful and well-timed. It stays silent 95% of the time. When it speaks, it's right.

---

## Adding a Tool

1. Add schema dict to `TOOL_DEFINITIONS` in `tools.py`.
2. Add `if name == "..."` branch in `dispatch_tool`.
3. If result needs prettifying before speaking, add branch in `_build_spoken_response` in `loop.py` to route through `synthesize_response`.

---

## Working Memory

Maintain `SCRATCHPAD.md` in repo root for any non-trivial problem. Update after each attempt. Include: problem statement, approaches tried + why they failed, current hypothesis, next steps, ruled-out paths.