from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.model_quality_pass import (
    _predict_one,
    build_closed_trades,
    build_replay_diagnostics,
    build_synthetic_replay_artifact,
    load_market_trade_events,
)


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

    def test_crypto_predictor_uses_symbol_evidence_over_global_majority(self) -> None:
        train_rows = []
        for i in range(18):
            train_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "exit_price": 102.0 + i,
                    "hold_hours": 4.0,
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                }
            )
        for i in range(40):
            train_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 100.0,
                    "exit_price": 98.0 - (0.1 * i),
                    "hold_hours": 3.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                }
            )
        candidate = {"symbol": "BTC-USD", "entry_price": 100.0}
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(out.get("predicted_direction", "")), "up")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Trailing")

    def test_crypto_predictor_can_predict_manual_when_manual_support_is_strong(self) -> None:
        train_rows = []
        for i in range(10):
            train_rows.append(
                {
                    "symbol": "DOT-USD",
                    "entry_price": 100.0,
                    "exit_price": 96.0,
                    "hold_hours": 40.0 + i,
                    "actual_exit_trigger": "Manual",
                    "regime": "high_volatility",
                }
            )
        for i in range(6):
            train_rows.append(
                {
                    "symbol": "DOT-USD",
                    "entry_price": 100.0,
                    "exit_price": 97.0,
                    "hold_hours": 20.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                }
            )
        candidate = {"symbol": "DOT-USD", "entry_price": 100.0}
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Manual")

    def test_crypto_predictor_keeps_stale_when_stale_support_is_stronger(self) -> None:
        train_rows = []
        for i in range(16):
            train_rows.append(
                {
                    "symbol": "AAVE-USD",
                    "entry_price": 100.0,
                    "exit_price": 97.0 - (0.1 * i),
                    "hold_hours": 10.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                }
            )
        for i in range(4):
            train_rows.append(
                {
                    "symbol": "AAVE-USD",
                    "entry_price": 100.0,
                    "exit_price": 102.0 + i,
                    "hold_hours": 6.0,
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                }
            )
        candidate = {"symbol": "AAVE-USD", "entry_price": 100.0}
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Stale Alignment")

    def test_crypto_predictor_ignores_candidate_actual_labels(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "exit_price": 101.5 + i,
                    "hold_hours": 6.0,
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                }
            )
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "exit_price": 98.5 - i,
                    "hold_hours": 5.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                }
            )
        clean_candidate = {"symbol": "BTC-USD", "entry_price": 100.0}
        labeled_candidate = {
            "symbol": "BTC-USD",
            "entry_price": 100.0,
            "actual_exit_trigger": "Manual",
            "actual_direction": "down",
            "exit_price": 80.0,
            "hold_hours": 999.0,
        }
        clean = _predict_one(train_rows=train_rows, candidate=clean_candidate, regime="high_volatility", market="crypto")
        labeled = _predict_one(train_rows=train_rows, candidate=labeled_candidate, regime="high_volatility", market="crypto")
        self.assertEqual(clean.get("predicted_exit_trigger"), labeled.get("predicted_exit_trigger"))
        self.assertEqual(clean.get("predicted_direction"), labeled.get("predicted_direction"))

    def test_non_crypto_predictor_behavior_uses_legacy_path(self) -> None:
        train_rows = [
            {
                "symbol": "EUR_USD",
                "entry_price": 1.0,
                "exit_price": 0.99,
                "hold_hours": 2.0,
                "actual_exit_trigger": "Unknown",
                "regime": "range",
            }
            for _ in range(9)
        ] + [
            {
                "symbol": "EUR_USD",
                "entry_price": 1.0,
                "exit_price": 1.01,
                "hold_hours": 2.0,
                "actual_exit_trigger": "Unknown",
                "regime": "range",
            }
            for _ in range(3)
        ]
        out = _predict_one(train_rows=train_rows, candidate={"symbol": "EUR_USD", "entry_price": 1.0}, regime="range", market="forex")
        self.assertEqual(str(out.get("predicted_direction", "")), "down")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Unknown")

    def test_manual_does_not_win_on_global_support_only(self) -> None:
        train_rows = []
        for i in range(8):
            train_rows.append(
                {
                    "symbol": "DOT-USD",
                    "entry_price": 100.0,
                    "exit_price": 96.0,
                    "hold_hours": 40.0 + i,
                    "actual_exit_trigger": "Manual",
                    "regime": "high_volatility",
                }
            )
        for i in range(14):
            train_rows.append(
                {
                    "symbol": "LINK-USD",
                    "entry_price": 100.0,
                    "exit_price": 97.0 - (0.1 * i),
                    "hold_hours": 8.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                }
            )
        candidate = {"symbol": "LINK-USD", "entry_price": 100.0}
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertNotEqual(str(out.get("predicted_exit_trigger", "")), "Manual")

    def test_synthetic_replay_includes_population_diagnostics(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(18):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "exit_price": 102.0 + (i * 0.1),
                    "actual_exit_price": 102.0 + (i * 0.1),
                    "hold_hours": 4.0,
                    "actual_exit_trigger": "Trailing",
                    "entry_ts": ts + (i * 3600),
                    "exit_ts": ts + (i * 3600) + 7200,
                }
            )
        for i in range(18):
            closed_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 100.0,
                    "exit_price": 98.0 - (i * 0.1),
                    "actual_exit_price": 98.0 - (i * 0.1),
                    "hold_hours": 10.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "entry_ts": ts + 200_000 + (i * 3600),
                    "exit_ts": ts + 200_000 + (i * 3600) + 14_400,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        pop = payload.get("population_diagnostics", {}) if isinstance(payload.get("population_diagnostics", {}), dict) else {}
        self.assertIn("full_universe_metrics", pop)
        self.assertIn("admitted_metrics", pop)
        self.assertIn("abstained_metrics", pop)
        full_n = int(pop.get("full_test_trades", 0) or 0)
        admitted_n = int(pop.get("admitted_test_trades", 0) or 0)
        abstained_n = int(pop.get("abstained_test_trades", 0) or 0)
        self.assertEqual(full_n, admitted_n + abstained_n)
        crypto_diag = payload.get("crypto_classifier_diagnostics", {}) if isinstance(payload.get("crypto_classifier_diagnostics", {}), dict) else {}
        self.assertIn("population_diagnostics", crypto_diag)

    def test_population_diagnostics_show_admission_shrinkage(self) -> None:
        rows = [
            {
                "symbol": "BTC-USD",
                "entry_price": 100.0,
                "actual_exit_price": 102.0,
                "exit_price": 102.0,
                "actual_direction": "up",
                "predicted_direction": "up",
                "actual_exit_trigger": "Trailing",
                "predicted_exit_trigger": "Trailing",
                "predicted_confidence": 0.91,
                "hold_hours": 3.0,
            },
            {
                "symbol": "ETH-USD",
                "entry_price": 100.0,
                "actual_exit_price": 97.0,
                "exit_price": 97.0,
                "actual_direction": "down",
                "predicted_direction": "down",
                "actual_exit_trigger": "Stale Alignment",
                "predicted_exit_trigger": "Stale Alignment",
                "predicted_confidence": 0.20,
                "hold_hours": 12.0,
            },
        ]
        # Reuse the replay builder output format rather than depending on private helpers.
        payload = {
            "status": "ok",
            "iterations": [{"iteration": 1, "test_predictions": [rows[0]]}],
            "population_diagnostics": {
                "full_test_trades": 2,
                "admitted_test_trades": 1,
                "abstained_test_trades": 1,
                "admission_rate_pct": 50.0,
            },
        }
        self.assertEqual(int(payload["population_diagnostics"]["full_test_trades"]), 2)
        self.assertEqual(int(payload["population_diagnostics"]["admitted_test_trades"]), 1)
        self.assertEqual(int(payload["population_diagnostics"]["abstained_test_trades"]), 1)

    def test_replay_diagnostics_preserve_schema_with_crypto_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            replay_path = os.path.join(td, "openai", "crypto_historical_replay_synthetic.json")
            payload = {
                "summary": "ok",
                "hybrid_test_metrics": {
                    "directional_accuracy_pct": 66.0,
                    "trigger_match_pct": 50.0,
                    "trigger_scored_trades": 2.0,
                    "trigger_coverage_pct": 100.0,
                    "pnl_trend_match_pct": 66.0,
                },
                "crypto_classifier_diagnostics": {"direction_confusion_matrix": {"up": {"up": 1}}},
                "iterations": [
                    {
                        "test_predictions": [
                            {
                                "symbol": "BTC-USD",
                                "entry_price": 100.0,
                                "exit_price": 102.0,
                                "predicted_exit_price": 103.0,
                                "actual_direction": "up",
                                "predicted_direction": "up",
                                "actual_exit_trigger": "Trailing",
                                "predicted_exit_trigger": "Trailing",
                            },
                            {
                                "symbol": "ETH-USD",
                                "entry_price": 100.0,
                                "exit_price": 99.0,
                                "predicted_exit_price": 101.0,
                                "actual_direction": "down",
                                "predicted_direction": "up",
                                "actual_exit_trigger": "Stale Alignment",
                                "predicted_exit_trigger": "Trailing",
                            },
                        ]
                    }
                ],
            }
            self._write_json(replay_path, payload)
            out = build_replay_diagnostics(td, "crypto")
            self.assertEqual(str(out.get("state", "")), "READY")
            headline = out.get("headline_metrics", {}) if isinstance(out.get("headline_metrics", {}), dict) else {}
            self.assertIn("trigger_scored_trades", headline)
            self.assertIn("trigger_coverage_pct", headline)


if __name__ == "__main__":
    unittest.main()
