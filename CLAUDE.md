# Friday — Claude Code Guide

## What This Is

Voice-first AI assistant for macOS. Lives in the menu bar (rumps).
Always-on listening loop. Speak → Friday hears + optionally sees screen → routes to a tool or just answers → speaks back.

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
├── memory.py              SOUL.md / USER.md / MEMORY.md
├── capture/
│   ├── audio.py           energy-based VAD + barge-in detection
│   ├── audio_aec.py       hardware AEC via AVAudioEngine
│   └── screenshot.py      Quartz CGDisplay capture → base64 JPEG
├── transcribe/
│   └── deepgram.py        Deepgram Nova-3 STT
└── speak/
    └── elevenlabs.py      ElevenLabs Flash v2.5, interruptible via afplay
```

---

## The Loop (`loop.py`)

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

Conversation history is `self._history: list[dict]` — last 12 turns are injected into each LLM call. In-memory only, lost on restart (intentional — no sqlite checkpointer).

---

## LLM (`llm.py`)

`plan_tool_call(transcript, screenshot_b64, history, memory_context)` returns `(tool_name, arguments, thinking)`.

- `tool_choice="auto"` — the LLM may answer with **plain text instead of a tool call**. In that case we return `("speak", {"answer": text}, None)` and the loop just speaks it. No `speak_answer` tool needed.
- System prompt = `memory_context` + routing rules.
- Provider switched via `FRIDAY_LLM` env: `gemini` (default) | `openai` | `claude`. All use the OpenAI SDK with different `base_url`/`model`.

`synthesize_response(query, tool_result)` is a follow-up text-only LLM call that converts raw search/memory output into a natural spoken sentence.

---

## Tools (`tools.py`)

OpenAI function-calling format:

| Tool | Purpose |
|------|---------|
| `take_screenshot` | Triggers two-pass routing with vision context |
| `web_search` | Tavily search (Tavily client embedded in `tools.py`) |
| `save_memory` | Append a fact line to `MEMORY.md` |
| `memory_search` | Case-insensitive substring scan over `MEMORY.md` + `USER.md` |
| `find_files` | Glob file discovery under a directory (`file_search.py`) |
| `search_files` | Regex content search under a directory (`file_search.py`) |

Plus the synthetic `speak` tool name used when the LLM returned plain text.

`dispatch_tool(name, arguments)` is a flat if/elif chain — about 30 lines.

---

## Memory (`memory.py`)

Three files in `~/.friday/`:

- `SOUL.md` — personality, communication style, anti-patterns
- `USER.md` — user profile (edit by hand)
- `MEMORY.md` — facts saved by `save_memory`

`load_memory_context()` concatenates all three (with a `MEMORY_MAX_CHARS` budget) into a single string injected into the system prompt every turn.

`memory_search(query)` — plain Python substring scan, no FTS5, no sqlite.

---

## Configuration (`config.py` + `.env`)

Required:

| Key | Purpose |
|-----|---------|
| `DEEPGRAM_API_KEY` | STT |
| `ELEVENLABS_API_KEY` | TTS |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | one of these, matching `FRIDAY_LLM` |

Optional:

| Key | Default | Purpose |
|-----|---------|---------|
| `TAVILY_API_KEY` | — | Web search (tool returns "unavailable" without it) |
| `FRIDAY_LLM` | `gemini` | `gemini` \| `openai` \| `claude` |
| `FRIDAY_MUTE_KEY` | `ctrl+m` | Mute mic without stopping the loop |
| `FRIDAY_VOICE_ID` | Alice | ElevenLabs voice ID |
| `FRIDAY_VAD_THRESHOLD` | `600` | RMS energy threshold |
| `FRIDAY_VAD_ONSET_FRAMES` | `5` | Frames (~150ms) to confirm speech onset |
| `FRIDAY_VAD_OFFSET_FRAMES` | `25` | Frames (~750ms) of silence to end segment |
| `FRIDAY_SCREENSHOT_MAX_KB` | `400` | Max screenshot size sent to LLM |
| `FRIDAY_FILE_SEARCH_DEFAULT_ROOT` | — | Default `path` for `find_files` / `search_files` when user omits it |
| `FRIDAY_LOG_LEVEL` | `INFO` | |

---

## Audio Pipeline

### VAD (`capture/audio.py`)

Energy-based, no ML. Per 30 ms frame (480 samples @ 16 kHz): RMS > threshold for `VAD_ONSET_FRAMES` consecutive frames → onset; record until `VAD_OFFSET_FRAMES` of silence.

Mute: when `mute_event.is_set()`, all frames are treated as silence — VAD never triggers.

### Hardware AEC (`capture/audio_aec.py`)

`AVAudioEngine` with voice processing enabled — same kernel path as Zoom/FaceTime/Siri. Prevents Friday hearing its own TTS. Falls back to plain sounddevice if AVFoundation is unavailable.

### Barge-in

`_barge_in_sync` runs in a background thread while either `_build_response` or `speak_interruptible` is active. Sets `onset_event` on speech onset; the caller cancels the current task and either restarts the build or stops TTS.

---

## Adding a Tool

1. Add a function schema dict to `TOOL_DEFINITIONS` in `tools.py`.
2. Add an `if name == "..."` branch in `dispatch_tool`.
3. If the result needs LLM-side prettifying before being spoken, add a branch in `_build_spoken_response` in `loop.py` to route through `synthesize_response`.

---

## Working memory

For any non-trivial problem, maintain `SCRATCHPAD.md` in the repo root. Update it after each approach attempt. Include: problem statement, approaches tried + why they failed, current hypothesis, next steps, and ruled-out hypotheses.
