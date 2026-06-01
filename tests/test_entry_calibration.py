from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.entry_calibration import evaluate_entry_calibration_gate, load_replay_trigger_reliability


class EntryCalibrationTests(unittest.TestCase):
    def test_gate_blocks_when_trigger_reliability_is_below_threshold(self) -> None:
        out = evaluate_entry_calibration_gate(
            calib_prob=0.86,
            calib_samples=80,
            trigger_reliability=0.42,
            min_calib_prob=0.55,
            min_calib_samples=10,
            min_trigger_reliability=0.55,
            min_entry_score=0.60,
            weight_calib=0.65,
            weight_trigger=0.35,
            require_samples=True,
        )
        self.assertFalse(bool(out.get("passed", True)))
        self.assertIn("trigger reliability too low", str(out.get("reason", "") or ""))

    def test_reliability_loader_prefers_latest_replay_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            openai_dir = os.path.join(td, "openai")
            os.makedirs(openai_dir, exist_ok=True)
            older = os.path.join(openai_dir, "stock_historical_replay_old.json")
            newer = os.path.join(openai_dir, "stock_historical_replay_new.json")
            with open(older, "w", encoding="utf-8") as f:
                json.dump({"trigger_match_pct": 55.0, "meta": {"test_trades": 22}}, f)
            with open(newer, "w", encoding="utf-8") as f:
                json.dump({"hybrid_test_metrics": {"trigger_match_pct": 70.0}, "meta": {"test_trades": 31}}, f)
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))

            out = load_replay_trigger_reliability(
                td,
                "stocks",
                patterns=["stock_historical_replay_*.json"],
            )
            self.assertAlmostEqual(float(out.get("value", 0.0) or 0.0), 0.70, places=6)
            self.assertEqual(int(out.get("samples", 0) or 0), 31)
            self.assertIn("stock_historical_replay_new.json", str(out.get("source", "") or ""))

    def test_reliability_loader_returns_neutral_default_when_no_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = load_replay_trigger_reliability(td, "forex")
            self.assertAlmostEqual(float(out.get("value", 0.0) or 0.0), 0.50, places=6)
            self.assertEqual(int(out.get("samples", 0) or 0), 0)


if __name__ == "__main__":
    unittest.main()
