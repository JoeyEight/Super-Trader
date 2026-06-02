from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.model_quality_pass import build_closed_trades, build_replay_diagnostics, load_market_trade_events


class TestModelQualityPass(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _write_jsonl(self, path: str, rows: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_load_market_trade_events_infers_stock_qty_from_notional(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "stocks", "execution_audit.jsonl"),
                [
                    {
                        "ts": 1_700_000_000,
                        "event": "entry",
                        "symbol": "NVDA",
                        "notional": 200.0,
                        "price": 100.0,
                        "ok": True,
                    },
                    {
                        "ts": 1_700_000_100,
                        "event": "exit",
                        "symbol": "NVDA",
                        "qty": 2.0,
                        "price": 102.0,
                        "ok": True,
                    },
                ],
            )
            loaded = load_market_trade_events(td, "stocks")
            events = loaded.get("events", []) if isinstance(loaded.get("events", []), list) else []
            self.assertEqual(len(events), 2)
            self.assertAlmostEqual(float(events[0].get("qty", 0.0) or 0.0), 2.0, places=6)
            closed = build_closed_trades(events)
            rows = closed.get("closed_trades", []) if isinstance(closed.get("closed_trades", []), list) else []
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0].get("pnl_usd", 0.0) or 0.0), 0.0)

    def test_build_replay_diagnostics_outputs_confusion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            replay_path = os.path.join(td, "openai", "crypto_historical_replay_test.json")
            self._write_json(
                replay_path,
                {
                    "summary": "ok",
                    "hybrid_test_metrics": {
                        "directional_accuracy_pct": 50.0,
                        "trigger_match_pct": 50.0,
                        "pnl_trend_match_pct": 50.0,
                    },
                    "iterations": [
                        {
                            "test_predictions": [
                                {
                                    "symbol": "BTC-USD",
                                    "entry_price": 100.0,
                                    "actual_exit_price": 102.0,
                                    "predicted_exit_price": 103.0,
                                    "actual_direction": "up",
                                    "predicted_direction": "up",
                                    "actual_exit_trigger": "Trailing",
                                    "predicted_exit_trigger": "Trailing",
                                },
                                {
                                    "symbol": "ETH-USD",
                                    "entry_price": 100.0,
                                    "actual_exit_price": 99.0,
                                    "predicted_exit_price": 101.0,
                                    "actual_direction": "down",
                                    "predicted_direction": "up",
                                    "actual_exit_trigger": "Stale Alignment",
                                    "predicted_exit_trigger": "Trailing",
                                },
                            ]
                        }
                    ],
                },
            )
            out = build_replay_diagnostics(td, "crypto")
            self.assertEqual(str(out.get("state", "")), "READY")
            confusion = out.get("trigger_confusion_matrix", {}) if isinstance(out.get("trigger_confusion_matrix", {}), dict) else {}
            self.assertEqual(int((confusion.get("Trailing", {}) if isinstance(confusion.get("Trailing", {}), dict) else {}).get("Trailing", 0)), 1)
            self.assertEqual(int((confusion.get("Stale Alignment", {}) if isinstance(confusion.get("Stale Alignment", {}), dict) else {}).get("Trailing", 0)), 1)


if __name__ == "__main__":
    unittest.main()

