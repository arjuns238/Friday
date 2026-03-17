# Jarvis: Voice-First AI Orchestrator — Architecture Plan

## Context

Not an overlay. Not a persistent UI. A macOS menu bar app that you invoke with a hotkey, speak at your screen, and it understands what you're looking at + what you said — then routes to the right tool (injects into Claude Code CLI, drafts email via Gmail API, etc.) and speaks back. Think J.A.R.V.I.S.: ambient, fast, voice-native, with eyes on your screen.

Core user flow: **I'm working in Claude Code on ML training. I look at something on the web. I press a hotkey and say "try adding a residual block." Jarvis sees my screen, understands the coding context, formulates the right Claude Code prompt, and injects it into my active CLI session.**

---

## Architecture Critique of the Original Plan

The original overlay + rolling context buffer + LangGraph approach was designed for a different product (a visual UI assistant). The new concept needs:

- **Voice-first** (not text-first with optional voice)
- **Screenshot at invocation** (not a rolling background buffer)
- **Menu bar / dock app** (not an always-on-top overlay window)
- **Tool dispatch** (not agent pipelines) — Jarvis is a thin orchestrator, not the executor
- **ElevenLabs TTS for voice response** (not text in a window)

The original routing complexity (semantic-router, FAISS, rolling buffer) was solving the wrong problem. With voice input, the intent signal is much richer — you don't need an ML classifier, you need an LLM that can see your screen and hear you talk.

---

## Voice Pipeline: Why Not GPT-4o Realtime

The user leaned toward GPT-4o Realtime (voice-native model). Here's the honest tradeoff:

| | GPT-4o Realtime | Deepgram → GPT-4o → ElevenLabs |
|---|---|---|
| Latency to first audio | ~200ms | ~600ms |
| Image input support | Limited (audio-primary) | Full (GPT-4o vision) |
| Tool calling with image context | Awkward | Clean |
| Lock-in | OpenAI only | Swappable per step |
| Voice quality | Good | Excellent (ElevenLabs) |

**Recommended: Deepgram → GPT-4o vision → ElevenLabs**

GPT-4o Realtime is optimized for audio-in, audio-out conversations without images. The moment you need "see my screen + hear me talk", the image injection into the Realtime API is clunky. The 3-hop pipeline at ~600ms is still very fast — the 400ms difference from Realtime is imperceptible in normal use. You get full vision support, better voice quality, and a modular pipeline you can tune each piece of.

Target end-to-end: **hotkey → first spoken word in ~700ms** for short queries.

---

## System Architecture

```
macOS Menu Bar App (Python + rumps)
│
├── Global hotkey listener (pynput)
│
│ [Hotkey pressed]
│

├─────────────────────────────────────────────────────────┐
│                   INVOCATION PIPELINE                    │
│                                                         │
│  1. Screenshot (ScreenCaptureKit via screenpipe/pyobjc) │
│     ~100ms, captures focused display                    │
│                                                         │
│  2. Audio capture (sounddevice/PyAudio)                 │
│     VAD (voice activity detection) to auto-stop         │
│     ~1-5s of user speech                                │
│                                                         │
│  3. Deepgram streaming transcription                    │
│     ~200ms from end of speech to transcript             │
│     Streams partial results while user speaks           │
│                                                         │
│  4. GPT-4o vision call                                  │
│     Input: screenshot + transcript                      │
│     Tools: inject_claude_code, draft_gmail,             │
│             web_search, speak_answer                    │
│     ~400ms to first token (streaming)                   │
│                                                         │
│  5. Tool execution (Python)                             │
│     OR direct text response → ElevenLabs TTS            │
│     ~150ms to first audio byte                          │
│                                                         │
│  Total p50: ~700ms hotkey to first spoken word          │
└─────────────────────────────────────────────────────────┘
```

### Tool Definitions (what GPT-4o can call)

```python
TOOLS = [
    {
        "name": "inject_claude_code",
        "description": "Inject a prompt into the active Claude Code CLI session in the terminal. Use when the screenshot shows VS Code or a terminal running 'claude', and the user's request is a coding task.",
        "parameters": {
            "prompt": "str — the exact prompt to inject into Claude Code"
        }
    },
    {
        "name": "draft_gmail",
        "description": "Draft and optionally send an email. Use when screenshot shows Gmail or user says 'email', 'write to', 'reply to'.",
        "parameters": {
            "to": "str — recipient email address (from context or user)",
            "subject": "str",
            "body_instructions": "str — detailed instructions for Gemini to draft the email body",
            "send": "bool — false by default, always confirm before sending"
        }
    },
    {
        "name": "web_search",
        "description": "Search the web and return results. Use when user asks 'what is', 'find', 'look up', 'search for'.",
        "parameters": {
            "query": "str"
        }
    },
    {
                "description": "Speak a direct answer back without using another tool. Use for factual questions, opinions, explanations.",
        "parameters": {
            "answer": "str"
        }
    }
]
```

---

## The Claude Code Injection (Most Critical Feature)

When the user says "try adding a residual block to this layer":

1. GPT-4o sees the screenshot (shows Claude Code CLI in terminal, maybe a browser with paper)
2. GPT-4o calls `inject_claude_code` with a formulated prompt like:
   > "In the current ML training script, add a residual block after the layer currently being discussed. Use the same architecture style as the existing layers. Run a quick test to confirm it works."
3. Python executes injection via osascript:

```python
# tools/claude_code_injector.py
import subprocess
import shlex

def inject_into_claude_code(prompt: str) -> str:
    """
    Types a prompt into the active terminal window running claude CLI.
    Detects iTerm2 or Terminal.app automatically.
    """
    escaped = prompt.replace('"', '\\"').replace("'", "\\'")

    # Try iTerm2 first
    iterm_script = f'''
    tell application "iTerm2"
        tell current session of current window
            write text "{escaped}"
        end tell
    end tell
    '''
    result = subprocess.run(
        ['osascript', '-e', iterm_script],
        capture_output=True
    )
    if result.returncode == 0:
        return "injected via iTerm2"

    # Fallback to Terminal.app
    terminal_script = f'''
    tell application "Terminal"
        do script "{escaped}" in front window
    end tell
    '''
    subprocess.run(['osascript', '-e', terminal_script])
    return "injected via Terminal"
```

**Limitation to call out**: `do script` in Terminal.app opens a *new* terminal tab, not the existing Claude Code session. For an existing interactive session, you need the user to be in iTerm2 (which supports `write text` to the current session). This is a real constraint — recommend iTerm2 as the supported terminal.


**Better alternative for production**: Run Claude Code via a Python subprocess with a named pipe, so Jarvis writes directly to its stdin:

```python
# Start Claude Code in a way Jarvis can control
import subprocess, os

pipe_path = os.path.expanduser("~/.jarvis/claude_input.pipe")
os.mkfifo(pipe_path)
claude_proc = subprocess.Popen(
    f"tail -f {pipe_path} | claude",
    shell=True,
    stdin=subprocess.PIPE
)

# Inject: just write to the pipe
with open(pipe_path, 'w') as f:
    f.write(prompt + '\n')
```

This is cleaner but requires the user to start Claude Code *through Jarvis*. Worth doing for V1.1.

---

## Gmail Integration

```python
# tools/gmail_tool.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import google.generativeai as genai

async def draft_gmail(to: str, subject: str, body_instructions: str, send: bool = False):
    # 1. Use Gemini to draft the email body
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(
        f"Draft a professional email with these instructions: {body_instructions}"
    )
    draft_body = response.text

    # 2. Create Gmail draft
    gmail = build('gmail', 'v1', credentials=get_credentials())
    message = create_message(to=to, subject=subject, body=draft_body)

    if send:
        # This requires explicit user confirmation — never auto-send
        gmail.users().messages().send(userId='me', body=message).execute()
        return f"Email sent to {to}"
    else:
        draft = gmail.users().drafts().create(userId='me', body={'message': message}).execute()
        return f"Draft created. Subject: {subject}. Open Gmail to review and send."
```

**Important**: Never auto-send. `draft_gmail` always creates a draft, not sends. The user reviews in Gmail. This is a safety constraint that should be hardcoded.

---

## Repository Structure


```
jarvis/
├── jarvis/                      # Main Python package
│   ├── __main__.py              # Entry point (menu bar app)
│   ├── app.py                   # rumps app + hotkey setup
│   ├── pipeline.py              # Main invocation pipeline
│   │
│   ├── capture/
│   │   ├── screenshot.py        # ScreenCaptureKit via pyobjc
│   │   └── audio.py             # sounddevice VAD + recording
│   │
│   ├── transcribe/
│   │   └── deepgram.py          # Streaming transcription
│   │
│   ├── orchestrate/
│   │   └── gpt4o.py             # GPT-4o vision call + tool routing
│   │
│   ├── tools/
│   │   ├── claude_code.py       # osascript / named pipe injection
│   │   ├── gmail.py             # Gmail API + Gemini draft
│   │   ├── search.py            # Tavily/Brave search
│   │   └── base.py              # Tool base class / registry
│   │
│   ├── speak/
│   │   └── elevenlabs.py        # TTS + audio playback
│   │
│   └── config.py                # API keys, settings
│
├── pyproject.toml               # uv project
├── .env.example
└── README.md
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Menu bar app | `rumps` (Python) | 50 lines to a macOS menu bar app, no Swift needed |
| Global hotkey | `pynput` | Cross-platform, reliable, Python-native |
| Screenshot | `pyobjc-framework-ScreenSaver` + ScreenCaptureKit | Native macOS, fastest API |
| Audio capture | `sounddevice` | Low latency, Python, VAD support via `webrtcvad` |
| STT | Deepgram streaming API | ~200ms from end of speech, best accuracy on technical terms |
| Orchestrator LLM | GPT-4o (with vision) | Multimodal, tool calling, good at understanding context |
| Email drafting | Gemini 2.0 Flash via `google-generativeai` | Fast, cheap, good at email tone |
| Claude Code injection | `osascript` (AppleScript) | Only reliable way to write to active terminal session |
| TTS | ElevenLabs (Flash v2) | ~150ms first audio byte, natural voice |
| Gmail | `google-api-python-client` | Official, OAuth2, full control |

**Dependencies to add later (not MVP):**
- Web search: Tavily Python client (simplest, good quality)
- Future: Claude Code named pipe integration (replaces osascript)

---

## MVP Phases
 ### Phase 0: Skeleton (Target: ~1 week)
**Goal**: Hotkey → speak → hears you → speaks back. No screen context yet.

- [ ] `rumps` menu bar app with hotkey (Cmd+Option+Space)
- [ ] `sounddevice` audio capture + `webrtcvad` for auto-stop on silence
- [ ] Deepgram streaming transcription (WebSocket)
- [ ] GPT-4o call with transcript only (no image yet)
- [ ] ElevenLabs TTS playback
- [ ] Mic-on indicator in menu bar during recording

**Proof point**: Say "what's 2+2" → Jarvis says "4" in ~700ms.

### Phase 1: Visual Context (Target: ~1 week)
**Goal**: Jarvis sees what you're looking at when you invoke.

- [ ] ScreenCaptureKit screenshot at hotkey press
- [ ] Compress + encode screenshot (JPEG, <500KB, base64)
- [ ] Pass to GPT-4o as `image_url` in the user message
- [ ] Test: open an article, ask "what's this about" → response reflects actual content

**Proof point**: Point at a Python error in your browser, say "what does this error mean" → correct explanation of that specific error.

### Phase 2: Claude Code Injection (Target: ~1 week)
**Goal**: The core use case — inject prompts into active Claude Code CLI.

- [ ] `inject_claude_code` tool via osascript (iTerm2 focus)
- [ ] GPT-4o system prompt tuned to recognize coding context from screenshot
- [ ] Test: running Claude Code in iTerm2, say "add a docstring to this function" → prompt appears in terminal

**Proof point**: The ML use case — Claude Code session open, browser with ML paper on screen, say "try the architecture from this paper" → appropriate Claude Code prompt is injected.

### Phase 3: Gmail Integration (Target: ~1 week)
**Goal**: Email draft without touching Gmail.

- [ ] Google OAuth2 setup (one-time, stored in `~/.jarvis/google_creds.json`)
- [ ] Gemini Flash email drafting
- [ ] Gmail draft creation (never auto-send)
- [ ] Spoken confirmation: "Draft created in Gmail, subject: [X]"

### Phase 4: Robustness + Polish (ongoing)
- [ ] Named pipe for Claude Code (replaces osascript, more reliable)
- [ ] Multi-monitor screenshot: capture only the focused display
- [ ] Whisper local fallback if Deepgram is down
- [ ] Tool result feedback: Jarvis speaks "I've injected the prompt into Claude Code"
- [ ] Conversation history (last N turns for context continuity)
- [ ] Settings UI via menu bar (API keys, hotkey config, voice selection)

---

## Hard Problems (What You're Underestimating)

### 1. osascript Terminal Injection is Fragile
`do script` in Terminal.app opens a new tab, doesn't type in the existing session. iTerm2's `write text` works for the current session but only if the window is focused. **The cleanest solution is the named pipe approach** — but it requires Claude Code to be launched through Jarvis. Plan for this in Phase 4. For MVP, document "requires iTerm2" as a constraint.

### 2. VAD (Voice Activity Detection) is Annoying to Tune
When to stop recording? Too eager and you cut off mid-sentence. Too slow and there's a 2-second delay after you stop talking. `webrtcvad` helps but the 300ms trailing-silence threshold needs tuning. Plan to iterate. Alternative: push-to-talk (hold hotkey while speaking, release to send) — simpler and more predictable UX, though less Jarvis-like.
                                                                                     
### 3. Screenshot → GPT-4o Context Quality Varies
Screenshots of terminals with small fonts, dark themes, or multiple panes give poor OCR quality. GPT-4o will misread code. Mitigation: run the screenshot through a preprocessing step (upscale to 2x, increase contrast). Also: capture the active window title and app name via AXUIElement to give GPT-4o a structured hint about what app is focused.

### 4. GPT-4o Vision API Latency on Large Screenshots
A full 4K screenshot at 4MB will slow down the API call. Always compress to JPEG at 1024x768 or similar before sending. Target: screenshot should be <300KB in the API call.

### 5. ElevenLabs Latency for Long Responses
ElevenLabs Flash v2 has ~150ms first audio byte for short strings. But if GPT-4o generates a 500-word response, you're streaming text → audio in chunks. Don't wait for the full response — stream sentences as they complete. First sentence playing within 600ms is achievable.

### 6. macOS Audio Session Management
macOS will fight you on audio routing: Deepgram input stream vs ElevenLabs output stream need to be on separate audio sessions, otherwise you get feedback loops or mic cutoff during playback. Use separate `sounddevice` streams with explicit device IDs.

### 7. API Cost at Daily Use Scale
At ~50 voice queries/day:
- Deepgram streaming: ~$0.15/day (50 × 5s audio × $0.0059/min)
- GPT-4o with vision: ~$1.50/day (50 × 1k image tokens + 500 input + 200 output tokens × $10/M)
- ElevenLabs Flash v2: ~$0.30/day (50 × 100 chars × $0.15/1000 chars)

Total: ~**$2/day** for personal use. Acceptable. Watch the GPT-4o vision cost — it's the biggest line item.

---

## Communication Diagram

```
pynput (hotkey)
      │
      ▼
pipeline.py (async orchestrator)
      ├── capture/screenshot.py  → base64 JPEG
      ├── capture/audio.py       → PCM audio stream
      │
      ├── transcribe/deepgram.py → text transcript   (WebSocket, async)
      │
      ├── orchestrate/gpt4o.py   → tool calls / text (HTTPS, streaming)
      │         │
      │         ├── tools/claude_code.py  → subprocess (osascript)
      │         ├── tools/gmail.py        → HTTPS (Google API)
      │         └── tools/search.py       → HTTPS (Tavily)
      │
      └── speak/elevenlabs.py    → audio bytes       (HTTPS + sounddevice playback)
```

No server process. No IPC. Pure Python async pipeline. The whole thing is one `asyncio` event loop.

---

## Verification

- **Phase 0**: Run `python -m jarvis`, press hotkey, say "what time is it in Tokyo" → Jarvis speaks the answer within 1 second
- **Phase 1**: Open an error message in browser, press hotkey, say "explain this error" → response references the actual error from the screenshot
- **Phase 2**: Open iTerm2 running `claude`, press hotkey, say "add logging to the main function" → text appears in Claude Code session
- **Phase 3**: Look at an email thread, press hotkey, say "draft a reply saying I'll follow up Friday" → Gmail draft appears with appropriate content
- **Latency check**: Add timing instrumentation to `pipeline.py` — log each stage's duration. Target: Deepgram transcript <300ms, GPT-4o first token <500ms, ElevenLabs first audio <200ms
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                