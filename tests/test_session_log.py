"""SessionLog append-only NDJSON + in-memory window."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from friday.ambient.session_log import SessionLog


class TestSessionLog(unittest.TestCase):
    def test_safety_trim_when_over_2x_max(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session_log.json"
            log = SessionLog(path=p, max_entries=3)
            for i in range(7):
                log.append(
                    {
                        "time": f"10:{i:02d}",
                        "activity": "coding",
                        "app": "Editor",
                        "detail": f"file{i}.py",
                    }
                )
            recent = log.get_recent(20)
            self.assertEqual(len(recent), 3)
            self.assertEqual(recent[0]["detail"], "file4.py")
            self.assertEqual(recent[-1]["detail"], "file6.py")
            # Disk keeps all 7 lines (append-only)
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 7)

    def test_get_full_context_format(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(path=Path(d) / "sl.json", max_entries=60)
            log.append(
                {
                    "time": "10:42",
                    "activity": "coding",
                    "app": "VS Code",
                    "detail": "loop.py",
                }
            )
            ctx = log.get_full_context()
            self.assertIn("[10:42]", ctx)
            self.assertIn("coding", ctx)
            self.assertIn("VS Code", ctx)
            self.assertIn("loop.py", ctx)

    @patch("friday.memory.read_today_session_markdown_excerpt", return_value="Earlier summary.")
    def test_get_prompt_context_includes_file_and_raw(self, _mock: object) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(path=Path(d) / "sl.json", max_entries=60)
            log.append(
                {
                    "time": "11:00",
                    "activity": "reading",
                    "app": "Safari",
                    "detail": "example.com",
                }
            )
            ctx = log.get_prompt_context()
            self.assertIn("Earlier summary", ctx)
            self.assertIn("Safari", ctx)

    def test_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session_log.json"
            log1 = SessionLog(path=p, max_entries=60)
            log1.append(
                {
                    "time": "11:00",
                    "activity": "reading",
                    "app": "Safari",
                    "detail": "example.com",
                }
            )
            log2 = SessionLog(path=p, max_entries=60)
            self.assertEqual(len(log2.get_recent(5)), 1)
            self.assertEqual(log2.get_recent(1)[0]["app"], "Safari")

            lines = p.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["app"], "Safari")

    def test_remove_compressed_trims_memory_not_disk(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = SessionLog(path=Path(d) / "sl.json", max_entries=60)
            for i in range(15):
                log.append(
                    {
                        "time": f"12:{i:02d}",
                        "activity": "coding",
                        "app": "X",
                        "detail": str(i),
                    }
                )
            comp = log.get_compressible()
            self.assertEqual(len(comp), 5)
            log.remove_compressed(5)
            self.assertEqual(len(log.get_recent(20)), 10)
            disk_lines = (Path(d) / "sl.json").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(disk_lines), 15)
