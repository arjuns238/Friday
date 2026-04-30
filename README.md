# Friday

Voice-first AI assistant for macOS. Lives in your menu bar, listens always, speaks back.

Press a hotkey to mute. Say "remember that I prefer Python." Ask about what's on your screen. Ask a current-events question — Friday will search the web and tell you.

**Architecture:** plain Python async loop. No agent framework. ~250 lines of glue.

```
listen → STT → LLM (+ optional screenshot) → tool → synthesize → TTS
        ↑ barge-in detection runs in parallel during build & speak
```

---

## Setup

### 1. Install

```bash
uv sync
```

### 2. API keys

```bash
cp .env.example .env
```

Required:
- `DEEPGRAM_API_KEY` — STT
- `ELEVENLABS_API_KEY` — TTS
- One of `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (matching `FRIDAY_LLM`)

Optional:
- `TAVILY_API_KEY` — web search

### 3. macOS permissions

Granted on first launch:
- **Accessibility** — global mute hotkey
- **Screen Recording** — screenshot tool
- **Microphone** — voice input

### 4. Run

```bash
friday               # menu bar app
python -m friday     # same thing
```

A 🎙 icon appears in the menu bar. The always-on listening loop starts immediately.

---

## CLI

```bash
friday                  # menu bar app (default)
friday test-pipeline    # 120s loop session, no menu bar — for debugging
friday help             # this list
```

---

## Tools

The LLM can do four things, or just answer directly with plain text (the default path):

- `take_screenshot` — capture the focused display, then re-route with the screenshot in context
- `web_search` — Tavily search, summarised into a spoken sentence
- `save_memory` — append a fact to `~/.friday/MEMORY.md`
- `memory_search` — substring search over `MEMORY.md` and `USER.md`

Plain text from the LLM is spoken directly — no tool needed for "what's 2+2".

---

## Memory

Three files in `~/.friday/`:

- `SOUL.md` — Friday's personality, edited by you or untouched
- `USER.md` — your profile, edit by hand
- `MEMORY.md` — facts saved via `save_memory`, append-only

All three are concatenated into the system prompt every turn (capped at `FRIDAY_MEMORY_MAX_CHARS`, default 8000).

---

## LLM provider

Set `FRIDAY_LLM` in `.env`:

| Value | Model | API key needed |
|-------|-------|----------------|
| `gemini` (default) | `gemini-3.1-flash-lite-preview` | `GOOGLE_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `claude` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |

All three use the OpenAI SDK — Gemini and Claude are reached through their OpenAI-compatible endpoints.

---

## Constraints

- **macOS only** — uses `Quartz`, `AVAudioEngine`, `afplay`, rumps.
- The mute hotkey works system-wide; the listening loop is always on, but mute is always one key away.

---

## Cost (~50 queries/day)

| Service | Cost/day |
|---------|----------|
| Deepgram Nova-3 | ~$0.15 |
| LLM (Gemini Flash) | ~$0.10 |
| ElevenLabs Flash v2.5 | ~$0.30 |
| **Total** | **~$0.55** |

GPT-4o vision is more expensive (~$1.50/day) — switch to Gemini if cost matters.
