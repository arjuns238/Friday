"""LangGraph state machine — replaces pipeline.py.

Provides conversation history (AsyncSqliteSaver, thread_id = today's date)
and explicit typed state. Audio, TTS, and tool modules are unchanged.

Flow (one graph.ainvoke() call per hotkey activation — loops internally):
  listen → build → speak → [interrupt] → listen → ...

Barge-in during build:  build_node returns audio=barge_audio → listen passthrough → build
Barge-in during speak:  speak_node returns barge_audio → interrupt → speak (resume) or listen (new query)
Stop event:             any node sets done=True → END
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Annotated, Optional
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RunnableConfig

log = logging.getLogger(__name__)

_RESUME_RE = re.compile(
    r"\b(go on|continue|keep going|keep talking|go ahead|resume|"
    r"carry on|please continue|finish|what else|what were you saying|"
    r"yeah go on|yes go on|ok go on|okay go on)\b",
    re.IGNORECASE,
)


def _is_resume_intent(transcript: str) -> bool:
    return len(transcript.strip().split()) <= 6 and bool(_RESUME_RE.search(transcript))


def _on_state(config: dict, state_name: str) -> None:
    fn = config.get("configurable", {}).get("on_state_change")
    if fn:
        fn(state_name)


def _get_events(config: dict) -> tuple[threading.Event, threading.Event]:
    c = config.get("configurable", {})
    return c["stop_event"], c["mute_event"]


# ── State ──────────────────────────────────────────────────────────────────────

class FridayState(TypedDict, total=False):
    # Audio I/O (transient — cleared each cycle after use)
    audio: Optional[bytes]           # audio segment to process next
    barge_audio: Optional[bytes]     # audio captured during speak

    # Processing outputs (for observability / LangSmith tracing)
    transcript: str
    tool_name: str
    tool_args: dict
    thinking: Optional[str]
    tool_result: str
    response_text: str               # final spoken text

    # Control
    is_resume: bool                  # True if barge was a "go on" / "continue" phrase
    done: bool                       # True when stop_event fires → graph exits

    # Conversation history — add_messages reducer appends; last N injected into LLM
    messages: Annotated[list[BaseMessage], add_messages]


# ── Node: listen ───────────────────────────────────────────────────────────────

async def listen_node(state: FridayState, config: RunnableConfig) -> dict:
    """Wait for speech onset, or pass through if audio is already queued."""
    from friday.capture.audio import listen_for_speech

    stop_event, mute_event = _get_events(config)
    _on_state(config, "listening")

    if stop_event.is_set():
        return {"done": True}

    # Audio already queued (barge-in from build or interrupt node) — skip listening
    if state.get("audio"):
        return {"barge_audio": None, "is_resume": False, "response_text": ""}

    audio = await listen_for_speech(stop_event, mute_event)
    if audio is None or stop_event.is_set():
        return {"done": True, "audio": None}

    return {
        "audio": audio,
        "barge_audio": None,
        "is_resume": False,
        "transcript": "",
        "response_text": "",
        "done": False,
    }


# ── Node: build ────────────────────────────────────────────────────────────────

async def build_node(state: FridayState, config: RunnableConfig) -> dict:
    """Transcribe + LLM plan + (optional screenshot) + tool dispatch.

    Runs a barge-in detector in parallel. If the user speaks during processing,
    build is cancelled and their new audio is queued for the next cycle.
    """
    from friday.capture.audio import _barge_in_sync
    from friday.speak.elevenlabs import speak

    stop_event, mute_event = _get_events(config)
    _on_state(config, "processing")

    audio = state.get("audio")
    if not audio:
        return {"response_text": "", "audio": None}

    loop = asyncio.get_running_loop()
    onset_event = threading.Event()
    cancel_event = threading.Event()
    barge_done_event = threading.Event()
    barge_audio_ref: list[Optional[bytes]] = [None]

    def _run_barge() -> None:
        barge_audio_ref[0] = _barge_in_sync(stop_event, mute_event, onset_event, cancel_event)
        barge_done_event.set()

    barge_thread = threading.Thread(target=_run_barge, daemon=True)
    barge_thread.start()

    build_task = asyncio.create_task(
        _do_build(audio, list(state.get("messages") or []), speak)
    )
    try:
        while not build_task.done() and not onset_event.is_set() and not stop_event.is_set():
            await asyncio.sleep(0.03)

        if onset_event.is_set():
            if not build_task.done():
                build_task.cancel()
                try:
                    await build_task
                except asyncio.CancelledError:
                    pass
            await loop.run_in_executor(None, lambda: barge_done_event.wait(timeout=30))
            log.info("Barge-in during build — restarting with new audio")
            return {
                "audio": barge_audio_ref[0],  # queued: listen_node will passthrough
                "response_text": "",
                "transcript": "",
            }

        if stop_event.is_set():
            build_task.cancel()
            try:
                await build_task
            except asyncio.CancelledError:
                pass
            return {"done": True, "response_text": "", "audio": None}

        try:
            return build_task.result()
        except Exception as exc:
            log.exception("Build error: %s", exc)
            await speak("Sorry, something went wrong. Check the logs.")
            return {"response_text": "", "audio": None}

    finally:
        cancel_event.set()
        await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))


async def _do_build(audio: bytes, history: list[BaseMessage], speak_fn) -> dict:
    """Inner build (no barge detection): transcribe → plan → dispatch.

    Two-pass flow: transcribe first (no screenshot), then if the LLM calls
    take_screenshot, capture the screen and re-plan with visual context.
    """
    t0 = time.monotonic()
    from friday.capture.screenshot import capture_focused_display
    from friday.orchestrate.llm import plan_tool_call
    from friday.tools.base import dispatch_tool
    from friday.transcribe.deepgram import transcribe

    transcript = await transcribe(audio)

    log.info("Transcript (%.0fms): %r", (time.monotonic() - t0) * 1000, transcript)

    if not transcript:
        log.warning("Empty transcript, skipping")
        return {"response_text": "", "transcript": "", "audio": None}

    history_dicts = _messages_to_dicts(history[-12:])

    # First pass: text-only routing (no screenshot)
    tool_name, arguments, thinking = await plan_tool_call(
        transcript, screenshot_b64=None, history=history_dicts
    )
    log.info("Plan pass 1 (%.0fms): tool=%s", (time.monotonic() - t0) * 1000, tool_name)

    # If the LLM wants to see the screen, capture and re-route
    if tool_name == "take_screenshot":
        if thinking:
            await speak_fn(thinking)
        loop = asyncio.get_running_loop()
        screenshot_b64 = await loop.run_in_executor(None, capture_focused_display)
        tool_name, arguments, thinking = await plan_tool_call(
            transcript, screenshot_b64=screenshot_b64, history=history_dicts
        )
        log.info("Plan pass 2 (%.0fms): tool=%s", (time.monotonic() - t0) * 1000, tool_name)

    if thinking:
        tool_result, _ = await asyncio.gather(
            dispatch_tool(tool_name, arguments),
            speak_fn(thinking),
        )
    else:
        tool_result = await dispatch_tool(tool_name, arguments)

    response_text = await _build_spoken_response(transcript, tool_name, tool_result)
    log.info("Response ready (%.0fms): %r", (time.monotonic() - t0) * 1000, response_text[:80])

    return {
        "transcript": transcript,
        "tool_name": tool_name,
        "tool_args": arguments,
        "thinking": thinking,
        "tool_result": tool_result,
        "response_text": response_text,
        "audio": None,  # consumed
    }


# ── Node: speak ────────────────────────────────────────────────────────────────

async def speak_node(state: FridayState, config: RunnableConfig) -> dict:
    """Speak the response; capture barge-in audio if the user interrupts."""
    from friday.speak.elevenlabs import speak_interruptible

    stop_event, mute_event = _get_events(config)
    _on_state(config, "speaking")

    response_text = state.get("response_text", "")
    if not response_text or stop_event.is_set():
        return {"barge_audio": None}

    interruption = await speak_interruptible(response_text, stop_event, mute_event)

    # Append to conversation history after speaking
    new_msgs: list[BaseMessage] = []
    transcript = state.get("transcript", "")
    if transcript:
        new_msgs.append(HumanMessage(content=transcript))
    new_msgs.append(AIMessage(content=response_text))

    return {
        "barge_audio": interruption,
        "messages": new_msgs,  # add_messages reducer appends these
    }


# ── Node: interrupt ────────────────────────────────────────────────────────────

async def interrupt_node(state: FridayState, config: RunnableConfig) -> dict:
    """Transcribe barge audio; classify as resume intent or new query."""
    from friday.transcribe.deepgram import transcribe

    _on_state(config, "processing")
    barge_audio = state.get("barge_audio")
    if not barge_audio:
        return {"is_resume": False, "barge_audio": None}

    barge_transcript = await transcribe(barge_audio)

    if barge_transcript and _is_resume_intent(barge_transcript):
        log.info("Resume intent: %r → re-speaking", barge_transcript)
        return {"is_resume": True, "barge_audio": None}

    log.info("New query from barge-in: %r", barge_transcript)
    return {
        "is_resume": False,
        "audio": barge_audio,   # queue for listen_node passthrough → build
        "barge_audio": None,
    }


# ── Edge routing ───────────────────────────────────────────────────────────────

def _route_listen(state: FridayState) -> str:
    return END if state.get("done") else "build"


def _route_build(state: FridayState) -> str:
    if state.get("done"):
        return END
    if state.get("response_text"):
        return "speak"
    return "listen"  # barge-during-build (audio queued) or empty transcript


def _route_speak(state: FridayState) -> str:
    return "interrupt" if state.get("barge_audio") else "listen"


def _route_interrupt(state: FridayState) -> str:
    return "speak" if state.get("is_resume") else "listen"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Compile and return the Friday LangGraph state machine.

    Example usage (in app.py):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(config.DB_PATH)) as cp:
            graph = build_graph(cp)
            await graph.ainvoke(
                {"done": False},
                config={"configurable": {
                    "thread_id": date.today().isoformat(),
                    "stop_event": stop,
                    "mute_event": mute,
                    "on_state_change": callback,
                }},
            )
    """
    g = StateGraph(FridayState)

    g.add_node("listen", listen_node)
    g.add_node("build", build_node)
    g.add_node("speak", speak_node)
    g.add_node("interrupt", interrupt_node)

    g.set_entry_point("listen")

    g.add_conditional_edges("listen", _route_listen, {"build": "build", END: END})
    g.add_conditional_edges(
        "build", _route_build, {"speak": "speak", "listen": "listen", END: END}
    )
    g.add_conditional_edges(
        "speak", _route_speak, {"interrupt": "interrupt", "listen": "listen"}
    )
    g.add_conditional_edges(
        "interrupt", _route_interrupt, {"speak": "speak", "listen": "listen"}
    )

    return g.compile(checkpointer=checkpointer)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-format dicts for LLM context injection."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


async def _build_spoken_response(transcript: str, tool_name: str, result: str) -> str:
    if tool_name == "speak_answer":
        return result
    if tool_name == "draft_gmail":
        return result
    if tool_name == "web_search":
        from friday.orchestrate.llm import synthesize_response
        return await synthesize_response(transcript, result)
    if tool_name == "desktop_query":
        return result  # subagent already produces natural spoken answer
    if tool_name == "open_file":
        if "not found" in result.lower() or "failed" in result.lower():
            return f"I couldn't open that. {result}"
        return result
    return result
