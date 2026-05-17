"""Conversation JSONL append-only log."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from friday.ambient.conversation_log import ConversationJsonlLog, new_session_id


class TestConversationJsonlLog(unittest.TestCase):
    def test_append_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            log = ConversationJsonlLog(p)
            log.append_turn("hello", "hi there")
            log.append_turn("bye", "see ya")
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            a = json.loads(lines[0])
            self.assertEqual(a["user"], "hello")
            self.assertEqual(a["friday"], "hi there")
            self.assertIn("ts", a)

    def test_skips_empty_both(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            log = ConversationJsonlLog(p)
            log.append_turn("  ", "")
            self.assertFalse(p.exists())

    def test_new_session_id_shape(self) -> None:
        sid = new_session_id()
        self.assertRegex(sid, r"^\d{8}-\d{6}-[0-9a-f]{8}$")
