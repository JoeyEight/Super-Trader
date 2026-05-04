from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.rejection_replay import (
    build_market_rejection_replay,
    build_rejection_replay_report,
    recommend_threshold_from_scores,
    replay_target_entries_for_market,
)


class TestRejectionReplay(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _write_jsonl(self, path: str, rows: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_market_replay_builds_scenarios_and_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "stocks", "stock_thinker_status.json"),
                {
                    "all_scores": [
                        {"symbol": "AAPL", "score": 0.52, "side": "long", "eligible_for_entry": True, "reason_logic": "trend"},
                        {"symbol": "MSFT", "score": 0.18, "side": "watch", "eligible_for_entry": False, "reason_logic": "weak"},
                        {"symbol": "TSLA", "score": -0.61, "side": "short", "eligible_for_entry": True, "reason_logic": "downtrend"},
                    ]
                },
            )
            self._write_jsonl(
                os.path.join(td, "stocks", "scanner_rankings.jsonl"),
                [
                    {"rejected": [{"symbol": "QQQ", "reason": "spread"}, {"symbol": "NVDA", "reason": "cooldown"}]},
                    {"rejected": [{"symbol": "AMZN", "reason": "cooldown"}]},
                ],
            )
            out = build_market_rejection_replay(
                td,
                "stocks",
                settings={"stock_score_threshold": 0.20, "replay_target_entries_stocks": 2},
            )
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertTrue(isinstance(out.get("scenarios", []), list))
            self.assertGreaterEqual(len(out.get("scenarios", [])), 4)
            rec = out.get("recommendation", {}) if isinstance(out.get("recommendation", {}), dict) else {}
            self.assertIn("recommended_threshold", rec)
            reasons = out.get("rejected_reason_breakdown", []) if isinstance(out.get("rejected_reason_breakdown", []), list) else []
            self.assertTrue(any(str((r or {}).get("reason", "")) == "cooldown" for r in reasons))

    def test_recommendation_helper_from_scores(self) -> None:
        rows = [
            {"symbol": "AAPL", "score": 0.52, "side": "long", "eligible_for_entry": True},
            {"symbol": "MSFT", "score": 0.18, "side": "watch", "eligible_for_entry": False},
            {"symbol": "TSLA", "score": -0.61, "side": "short", "eligible_for_entry": True},
        ]
        out = recommend_threshold_from_scores(rows, market="stocks", current_threshold=0.2, target_entries=2)
        rec = out.get("recommendation", {}) if isinstance(out.get("recommendation", {}), dict) else {}
        self.assertEqual(str(out.get("market", "")), "stocks")
        self.assertGreaterEqual(int(out.get("scored_rows", 0) or 0), 3)
        self.assertIn("recommended_threshold", rec)
        self.assertGreater(float(rec.get("recommended_threshold", 0.0) or 0.0), 0.0)

    def test_target_entries_helper_clamps(self) -> None:
        self.assertEqual(replay_target_entries_for_market({"replay_target_entries_stocks": 0}, "stocks"), 1)
        self.assertEqual(replay_target_entries_for_market({"replay_target_entries_forex": 999}, "forex"), 20)
        self.assertEqual(replay_target_entries_for_market({"replay_target_entries_crypto": 0}, "crypto"), 1)
        self.assertEqual(replay_target_entries_for_market({"replay_target_entries_crypto": 999}, "crypto"), 20)

    def test_replay_report_builds_both_markets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "stocks", "stock_thinker_status.json"),
                {"all_scores": [{"symbol": "SPY", "score": -0.31, "side": "watch"}]},
            )
            self._write_json(
                os.path.join(td, "forex", "forex_thinker_status.json"),
                {"all_scores": [{"pair": "EUR_USD", "score": 0.42, "side": "long", "eligible_for_entry": True}]},
            )
            self._write_jsonl(os.path.join(td, "stocks", "scanner_rankings.jsonl"), [{"rejected": []}])
            self._write_jsonl(os.path.join(td, "forex", "scanner_rankings.jsonl"), [{"rejected": [{"pair": "USD_JPY", "reason": "spread"}]}])
            out = build_rejection_replay_report(
                td,
                settings={
                    "stock_score_threshold": 0.2,
                    "forex_score_threshold": 0.2,
                    "replay_target_entries_stocks": 2,
                    "replay_target_entries_forex": 3,
                },
            )
            self.assertIn("stocks", out)
            self.assertIn("forex", out)
            self.assertEqual(str((out.get("stocks", {}) if isinstance(out.get("stocks", {}), dict) else {}).get("market", "")), "stocks")
            self.assertEqual(str((out.get("forex", {}) if isinstance(out.get("forex", {}), dict) else {}).get("market", "")), "forex")

    def test_crypto_market_replay_uses_dynamic_status_and_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {
                    "ranked": [
                        {"symbol": "BTC", "score": 1.42, "side": "long", "eligible_for_entry": True},
                        {"symbol": "ETH", "score": 0.64, "side": "watch", "eligible_for_entry": False},
                    ]
                },
            )
            self._write_jsonl(
                os.path.join(td, "crypto", "scanner_rankings.jsonl"),
                [
                    {"rejected": [{"symbol": "SOL", "reason": "cooldown"}]},
                    {"rejected": [{"symbol": "DOGE", "reason": "signal"}]},
                ],
            )
            out = build_market_rejection_replay(
                td,
                "crypto",
                settings={
                    "crypto_dynamic_min_projected_edge_pct": 0.20,
                    "replay_target_entries_crypto": 3,
                },
            )
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(str(out.get("market", "")), "crypto")
            self.assertTrue(isinstance(out.get("scenarios", []), list))
            rec = out.get("recommendation", {}) if isinstance(out.get("recommendation", {}), dict) else {}
            self.assertIn("recommended_threshold", rec)


if __name__ == "__main__":
    unittest.main()
