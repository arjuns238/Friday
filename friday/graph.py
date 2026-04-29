"""LangGraph state machine — thin voice loop over the deepagents orchestrator.

Flow (one graph.ainvoke() call per session — loops internally):
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


def _get_agent(config: dict):
    agent = config.get("configurable", {}).get("friday_agent")
    if agent is None:
        raise RuntimeError("friday_agent missing from RunnableConfig.configurable")
    return agent


# ── State ──────────────────────────────────────────────────────────────────────

class FridayState(TypedDict, total=False):
    audio: Optional[bytes]
    barge_audio: Optional[bytes]
    transcript: str
    response_text: str
    is_resume: bool
    done: bool
    messages: Annotated[list[BaseMessage], add_messages]


# ── Node: listen ───────────────────────────────────────────────────────────────

async def listen_node(state: FridayState, config: RunnableConfig) -> dict:
    from friday.capture.audio import listen_for_speech

    stop_event, mute_event = _get_events(config)
    _on_state(config, "listening")

    if stop_event.is_set():
        return {"done": True}

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
    """Transcribe, then drive the deepagents orchestrator. Run barge-in in parallel."""
    from friday.capture.audio import _barge_in_sync

    stop_event, mute_event = _get_events(config)
    _on_state(config, "processing")

    audio = state.get("audio")
    if not audio:
        return {"response_text": "", "audio": None}

    agent = _get_agent(config)
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
        _do_build(agent, audio, config.get("configurable", {}).get("agent_thread_id"))
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
            barge = barge_audio_ref[0] or b""
            combined = (audio or b"") + barge
            log.info("Barge-in during build — restarting with combined audio (%d+%d=%d bytes)",
                     len(audio or b""), len(barge), len(combined))
            return {
                "audio": combined,
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
            from friday.speak.elevenlabs import speak
            await speak("Sorry, something went wrong. Check the logs.")
            return {"response_text": "", "audio": None}

    finally:
        cancel_event.set()
        await loop.run_in_executor(None, lambda: barge_thread.join(timeout=0.2))


async def _do_build(agent, audio: bytes, agent_thread_id: str | None) -> dict:
    """Transcribe audio, invoke the deepagents orchestrator, return response text."""
    t0 = time.monotonic()
    from friday.transcribe.deepgram import transcribe

    transcript = await transcribe(audio)
    log.info("Transcript (%.0fms): %r", (time.monotonic() - t0) * 1000, transcript)

    if not transcript:
        return {"response_text": "", "transcript": "", "audio": None}

    invoke_config = {
        "configurable": {"thread_id": agent_thread_id} if agent_thread_id else {},
        "recursion_limit": 50,
    }

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=transcript)]},
            config=invoke_config,
        )
    except Exception as exc:
        log.exception("Agent invocation failed: %s", exc)
        return {"response_text": "Sorry, something went wrong.", "transcript": transcript, "audio": None}

    response_text = _extract_response(result)
    log.info("Response ready (%.0fms): %r", (time.monotonic() - t0) * 1000, response_text[:120])

    return {
        "transcript": transcript,
        "response_text": response_text,
        "audio": None,
    }


def _extract_response(agent_result: dict) -> str:
    """Pull the last non-empty AIMessage content from the deepagents result."""
    msgs = agent_result.get("messages") or []
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                # Multimodal / structured content — pull text parts
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                joined = "".join(parts).strip()
                if joined:
                    return joined
    return ""


# ── Node: speak ────────────────────────────────────────────────────────────────

async def speak_node(state: FridayState, config: RunnableConfig) -> dict:
    from friday.speak.elevenlabs import speak_interruptible

    stop_event, mute_event = _get_events(config)
    _on_state(config, "speaking")

    response_text = state.get("response_text", "")
    if not response_text or stop_event.is_set():
        return {"barge_audio": None}

    interruption = await speak_interruptible(response_text, stop_event, mute_event)

    new_msgs: list[BaseMessage] = []
    transcript = state.get("transcript", "")
    if transcript:
        new_msgs.append(HumanMessage(content=transcript))
    new_msgs.append(AIMessage(content=response_text))

    try:
        from friday.memory.context import append_daily_note
        summary = f"Q: {transcript[:80]} → A: {response_text[:80]}"
        append_daily_note(summary)
    except Exception:
        log.debug("Daily note append failed", exc_info=True)

    return {
        "barge_audio": interruption,
        "messages": new_msgs,
    }


# ── Node: interrupt ────────────────────────────────────────────────────────────

async def interrupt_node(state: FridayState, config: RunnableConfig) -> dict:
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
        "audio": barge_audio,
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
    return "listen"


def _route_speak(state: FridayState) -> str:
    return "interrupt" if state.get("barge_audio") else "listen"


def _route_interrupt(state: FridayState) -> str:
    return "speak" if state.get("is_resume") else "listen"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
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
