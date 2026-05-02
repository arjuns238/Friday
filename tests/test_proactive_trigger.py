"""Parse ambient proactive trigger JSON."""
from __future__ import annotations

import unittest

from friday.ambient.trigger import parse_proactive_trigger_payload


class TestParseProactiveTrigger(unittest.TestCase):
    def test_surface_false(self) -> None:
        self.assertIsNone(parse_proactive_trigger_payload('{"surface": false}'))

    def test_full_signal(self) -> None:
        raw = (
            '{"surface": true, "reason": "stuck", '
            '"observation": "TypeError line 47", '
            '"suggested_intervention": "missing await"}'
        )
        out = parse_proactive_trigger_payload(raw)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["reason"], "stuck")
        self.assertEqual(out["observation"], "TypeError line 47")
        self.assertEqual(out["suggested_intervention"], "missing await")

    def test_strips_fence(self) -> None:
        raw = '```json\n{"surface": true, "reason": "r", "observation": "o", "suggested_intervention": "s"}\n```'
        out = parse_proactive_trigger_payload(raw)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["reason"], "r")

    def test_implicit_surface_three_keys(self) -> None:
        """Three non-empty strings without surface:true still accepted."""
        raw = '{"reason": "a", "observation": "b", "suggested_intervention": "c"}'
        out = parse_proactive_trigger_payload(raw)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["reason"], "a")

    def test_empty_field_rejected(self) -> None:
        raw = '{"reason": "", "observation": "b", "suggested_intervention": "c"}'
        self.assertIsNone(parse_proactive_trigger_payload(raw))
