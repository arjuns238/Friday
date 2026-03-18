# Friday

Voice-first AI orchestrator for macOS. Lives in your menu bar.

Press a hotkey → speak → Friday sees your screen + hears you → routes to the right tool → speaks back.

**Core use case**: Working in Claude Code on ML training. See something on the web. Press hotkey, say "try adding a residual block." Friday sees your screen, formulates the right prompt, injects it into your active Claude Code session.

---

## Architecture

```
hotkey → screenshot + audio capture → Deepgram STT → GPT-4o vision → tool → ElevenLabs TTS
```

Target latency: **~700ms** hotkey to first spoken word.

**Tools GPT-4o can call:**
- `inject_claude_code` — types a prompt into the active iTerm2/Terminal Claude Code session
- `draft_gmail` — drafts an email via Gemini Flash + Gmail API (never auto-sends)
- `web_search` — Tavily web search
- `speak_answer` — direct spoken response

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

**macOS audio note**: `webrtcvad` requires Python < 3.12 or a patched version. If it fails:
```bash
pip install webrtcvad-wheels
```

**pydub for MP3 playback** (optional, falls back to `afplay`):
```bash
pip install pydub
brew install ffmpeg
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys:
# DEEPGRAM_API_KEY
# OPENAI_API_KEY
# ELEVENLABS_API_KEY
# GOOGLE_API_KEY       (for Gmail drafting, Phase 3)
# TAVILY_API_KEY       (for web search, optional)
```

### 3. macOS permissions

You'll need to grant these permissions once (macOS will prompt):
- **Accessibility** — for pynput global hotkey (`System Preferences → Privacy & Security → Accessibility`)
- **Screen Recording** — for screenshot capture (`System Preferences → Privacy & Security → Screen Recording`)
- **Microphone** — for audio capture

### 4. Run

```bash
# Start the menu bar app (default)
friday

# Or:
python -m friday
```

The 🎙 icon appears in your menu bar. Hold `F2` to speak, release to process.

---

## CLI Commands

```bash
friday                   # Start menu bar app
friday setup-gmail       # One-time Gmail OAuth2 auth
friday start-claude      # Start Claude Code via named pipe (Phase 4)
friday test-pipeline     # Single invocation test (no menu bar)
```

---

## Phases

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Hotkey → speak → hear back | 🏗 Implement |
| 1 | Screenshot → GPT-4o visual context | 🏗 Implement |
| 2 | Claude Code CLI injection | 🏗 Implement |
| 3 | Gmail draft integration | 🏗 Implement |
| 4 | Named pipe, multi-monitor, polish | 📋 Planned |

---

## Constraints

- **Requires iTerm2** for in-session Claude Code injection (Terminal.app opens a new tab instead). Set iTerm2 as your default terminal.
- **macOS only** — uses Apple-specific APIs (ScreenCaptureKit, osascript, rumps).
- **Gmail requires OAuth2 setup** — run `friday setup-gmail` once, then credentials persist at `~/.friday/google_creds.json`.

---

## Cost estimate (~50 queries/day)

| Service | Cost/day |
|---------|----------|
| Deepgram Nova-2 | ~$0.15 |
| GPT-4o vision | ~$1.50 |
| ElevenLabs Flash v2 | ~$0.30 |
| **Total** | **~$2.00** |
