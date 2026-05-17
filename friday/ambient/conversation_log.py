"""Append-only JSONL transcript for reactive voice turns (ground truth on disk)."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def new_session_id() -> str:
    """Unique id for one app run (one JSONL file per session)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


class ConversationJsonlLog:
    """One line per completed user turn: JSON with ts, user, friday."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append_turn(self, user: str, friday: str) -> None:
        """Append after a full reactive build+reply. Sparse vs ambient; no compression."""
        u = (user or "").strip()
        f = (friday or "").strip()
        if not u and not f:
            return
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "user": u,
            "friday": f,
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fp:
                    fp.write(line)
        except OSError as exc:
            log.warning("Could not append conversation log (%s): %s", self._path, exc)
