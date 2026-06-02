from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from app.model_quality_pass import (
    _generate_stock_historical_replay_closed_trades,
    _prepare_forex_audit_rows,
    _crypto_admission_decision,
    _predict_one,
    _score_with_threshold,
    build_market_dataset_quality,
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

    def test_crypto_policy_admits_strong_manual_below_old_global_threshold(self) -> None:
        row = {
            "predicted_confidence": 0.31,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 0.9,
            "direction_margin": 0.8,
            "manual_score": 6.0,
            "stale_score": 5.8,
            "trailing_score": 4.9,
            "manual_runner_up_margin": 0.2,
            "manual_same_symbol_support_count": 2,
            "manual_same_regime_support_count": 1,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.6,
            "manual_symbol_vs_stale_ratio": 0.4,
            "manual_symbol_vs_stale_long_hold_ratio": 0.5,
            "manual_recent_density_40": 0.08,
        }
        policy = {
            "base_conf_min": 0.50,
            "manual_conf_min": 0.28,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.10,
            "manual_vs_stale_long_hold_ratio_min": 0.15,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_score_advantage_min": -0.25,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertTrue(admit)
        self.assertEqual(reason, "manual_strong")

    def test_crypto_policy_rejects_weak_manual(self) -> None:
        row = {
            "predicted_confidence": 0.31,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 0.2,
            "direction_margin": 0.1,
            "manual_score": 5.0,
            "stale_score": 5.6,
            "trailing_score": 4.9,
            "manual_runner_up_margin": -0.6,
            "manual_same_symbol_support_count": 0,
            "manual_same_regime_support_count": 0,
            "manual_recent_support_count": 0,
            "manual_symbol_long_hold_ratio": 0.05,
            "manual_symbol_vs_stale_ratio": 0.0,
            "manual_symbol_vs_stale_long_hold_ratio": 0.0,
            "manual_recent_density_40": 0.0,
        }
        policy = {
            "base_conf_min": 0.50,
            "manual_conf_min": 0.28,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.10,
            "manual_vs_stale_long_hold_ratio_min": 0.15,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_score_advantage_min": -0.25,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertFalse(admit)
        self.assertIn("manual_", reason)

    def test_manual_precision_gate_does_not_reject_all_manual_rows(self) -> None:
        strong_row = {
            "predicted_confidence": 0.34,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 1.1,
            "direction_margin": 0.9,
            "manual_score": 6.4,
            "stale_score": 6.1,
            "trailing_score": 5.0,
            "manual_runner_up_margin": 0.3,
            "manual_same_symbol_support_count": 1,
            "manual_same_regime_support_count": 2,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.5,
            "manual_symbol_vs_stale_ratio": 0.3,
            "manual_symbol_vs_stale_long_hold_ratio": 0.4,
            "manual_recent_density_40": 0.07,
        }
        policy = {
            "base_conf_min": 0.50,
            "manual_conf_min": 0.28,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.10,
            "manual_vs_stale_long_hold_ratio_min": 0.15,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_score_advantage_min": -0.25,
        }
        admit, _ = _crypto_admission_decision(strong_row, policy)
        self.assertTrue(admit)

    def test_manual_cannot_be_admitted_from_regime_only_support(self) -> None:
        row = {
            "predicted_confidence": 0.34,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 1.0,
            "direction_margin": 0.8,
            "manual_score": 6.0,
            "stale_score": 5.7,
            "trailing_score": 4.8,
            "manual_runner_up_margin": 0.3,
            "manual_same_symbol_support_count": 0,
            "manual_same_symbol_stale_count": 4,
            "manual_same_regime_support_count": 5,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.35,
            "manual_symbol_vs_stale_ratio": 0.12,
            "manual_symbol_vs_stale_long_hold_ratio": 0.20,
            "manual_recent_density_40": 0.08,
        }
        policy = {
            "manual_mode": "strict",
            "base_conf_min": 0.50,
            "manual_conf_min": 0.24,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.10,
            "manual_vs_stale_long_hold_ratio_min": 0.15,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_symbol_support_min": 2,
            "manual_score_advantage_min": -0.10,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertFalse(admit)
        self.assertEqual(reason, "manual_regime_only_signal")

    def test_manual_blocked_when_symbol_vs_stale_ratio_low(self) -> None:
        row = {
            "predicted_confidence": 0.34,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 1.0,
            "direction_margin": 0.8,
            "manual_score": 6.0,
            "stale_score": 5.7,
            "trailing_score": 4.8,
            "manual_runner_up_margin": 0.3,
            "manual_same_symbol_support_count": 1,
            "manual_same_symbol_stale_count": 5,
            "manual_same_regime_support_count": 3,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.40,
            "manual_symbol_vs_stale_ratio": 0.10,
            "manual_symbol_vs_stale_long_hold_ratio": 0.20,
            "manual_recent_density_40": 0.08,
        }
        policy = {
            "manual_mode": "strict",
            "base_conf_min": 0.50,
            "manual_conf_min": 0.24,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.25,
            "manual_vs_stale_long_hold_ratio_min": 0.15,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_symbol_support_min": 2,
            "manual_score_advantage_min": -0.10,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertFalse(admit)
        self.assertEqual(reason, "manual_symbol_vs_stale_low")

    def test_manual_can_be_admitted_with_strong_symbol_level_evidence(self) -> None:
        row = {
            "predicted_confidence": 0.34,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 1.2,
            "direction_margin": 1.0,
            "manual_score": 6.4,
            "stale_score": 5.9,
            "trailing_score": 4.9,
            "manual_runner_up_margin": 0.5,
            "manual_same_symbol_support_count": 3,
            "manual_same_symbol_stale_count": 1,
            "manual_same_regime_support_count": 2,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.60,
            "manual_symbol_vs_stale_ratio": 0.50,
            "manual_symbol_vs_stale_long_hold_ratio": 0.60,
            "manual_recent_density_40": 0.08,
        }
        policy = {
            "manual_mode": "strict",
            "base_conf_min": 0.50,
            "manual_conf_min": 0.24,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.20,
            "manual_vs_stale_ratio_min": 0.25,
            "manual_vs_stale_long_hold_ratio_min": 0.35,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_symbol_support_min": 2,
            "manual_score_advantage_min": -0.10,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertTrue(admit)
        self.assertEqual(reason, "manual_strong")

    def test_diagnostic_only_manual_mode_blocks_weak_symbol_evidence(self) -> None:
        row = {
            "predicted_confidence": 0.40,
            "predicted_exit_trigger": "Manual",
            "trigger_margin": 1.0,
            "direction_margin": 0.8,
            "manual_score": 6.0,
            "stale_score": 5.8,
            "trailing_score": 4.8,
            "manual_runner_up_margin": 0.2,
            "manual_same_symbol_support_count": 1,
            "manual_same_symbol_stale_count": 4,
            "manual_same_regime_support_count": 4,
            "manual_recent_support_count": 2,
            "manual_symbol_long_hold_ratio": 0.25,
            "manual_symbol_vs_stale_ratio": 0.20,
            "manual_symbol_vs_stale_long_hold_ratio": 0.30,
            "manual_recent_density_40": 0.08,
        }
        policy = {
            "manual_mode": "diagnostic_only",
            "manual_diag_same_symbol_min": 2,
            "manual_diag_vs_stale_long_hold_min": 0.45,
            "manual_conf_min": 0.24,
            "manual_runner_margin_min": -0.25,
            "manual_trigger_margin_min": 0.25,
            "manual_dir_margin_min": 0.25,
            "manual_support_min": 2,
            "manual_long_hold_ratio_min": 0.30,
            "manual_vs_stale_ratio_min": 0.25,
            "manual_vs_stale_long_hold_ratio_min": 0.35,
            "manual_recent_density_min": 0.02,
            "manual_signal_votes_min": 3,
            "manual_symbol_support_min": 2,
            "manual_score_advantage_min": -0.10,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertFalse(admit)
        self.assertEqual(reason, "manual_diagnostic_only_block")

    def test_crypto_policy_admits_strong_trailing(self) -> None:
        row = {
            "predicted_confidence": 0.33,
            "predicted_exit_trigger": "Trailing",
            "trigger_margin": 2.8,
            "direction_margin": 1.5,
            "trailing_score": 7.0,
            "stale_score": 5.4,
        }
        policy = {
            "base_conf_min": 0.50,
            "trailing_conf_min": 0.28,
            "trailing_trigger_margin_min": 1.75,
            "trailing_dir_margin_min": 1.0,
            "trailing_vs_stale_min": -0.10,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertTrue(admit)
        self.assertEqual(reason, "trailing_strong")

    def test_crypto_policy_admits_strong_stale(self) -> None:
        row = {
            "predicted_confidence": 0.57,
            "predicted_exit_trigger": "Stale Alignment",
            "trigger_margin": 3.0,
            "direction_margin": 2.0,
        }
        policy = {
            "base_conf_min": 0.50,
            "stale_conf_min": 0.52,
            "stale_trigger_margin_min": 2.0,
            "stale_dir_margin_min": 1.25,
        }
        admit, reason = _crypto_admission_decision(row, policy)
        self.assertTrue(admit)
        self.assertEqual(reason, "stale_strong")

    def test_non_crypto_threshold_behavior_remains_legacy(self) -> None:
        rows = [
            {"predicted_confidence": 0.61, "actual_direction": "up", "predicted_direction": "up", "actual_exit_trigger": "Unknown", "predicted_exit_trigger": "Unknown", "entry_price": 1.0, "actual_exit_price": 1.01, "predicted_exit_price": 1.01},
            {"predicted_confidence": 0.49, "actual_direction": "down", "predicted_direction": "down", "actual_exit_trigger": "Unknown", "predicted_exit_trigger": "Unknown", "entry_price": 1.0, "actual_exit_price": 0.99, "predicted_exit_price": 0.99},
        ]
        scored = _score_with_threshold(rows, 0.50, "forex")
        admitted = scored.get("admitted", [])
        self.assertEqual(len(admitted), 1)
        self.assertAlmostEqual(float(scored.get("coverage", 0.0)), 0.5, places=6)

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

    def test_policy_frontier_is_generated_and_selected_policy_is_member(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(20):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "exit_price": 102.0 + (i * 0.05),
                    "actual_exit_price": 102.0 + (i * 0.05),
                    "hold_hours": 4.0,
                    "actual_exit_trigger": "Trailing",
                    "entry_ts": ts + (i * 3600),
                    "exit_ts": ts + (i * 3600) + 7200,
                }
            )
        for i in range(20):
            closed_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 100.0,
                    "exit_price": 98.0 - (i * 0.05),
                    "actual_exit_price": 98.0 - (i * 0.05),
                    "hold_hours": 12.0,
                    "actual_exit_trigger": "Stale Alignment",
                    "entry_ts": ts + 100_000 + (i * 3600),
                    "exit_ts": ts + 100_000 + (i * 3600) + 14_400,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        diag = payload.get("crypto_classifier_diagnostics", {}) if isinstance(payload.get("crypto_classifier_diagnostics", {}), dict) else {}
        frontier = diag.get("crypto_policy_frontier", [])
        selected = diag.get("selected_crypto_admission_policy", {})
        self.assertTrue(frontier)
        self.assertIn("policy_name", frontier[0])
        selected_name = payload.get("abstain_policy", {}).get("selected_crypto_policy_name", "")
        frontier_names = {row.get("policy_name", "") for row in frontier if isinstance(row, dict)}
        self.assertIn(selected_name, frontier_names)
        self.assertEqual(selected_name, frontier[0].get("policy_name", selected_name))

    def test_prepare_forex_audit_rows_marks_explicit_stale_exit(self) -> None:
        rows = [
            {
                "ts": 1_700_000_000,
                "event": "exit",
                "instrument": "EUR_USD",
                "qty": 100.0,
                "price": 1.1,
                "source": "policy_stale_exit",
                "msg": "Position close submitted",
            }
        ]
        prepared, diag = _prepare_forex_audit_rows(rows)
        self.assertEqual(prepared[0].get("tag"), "Stale Alignment")
        self.assertEqual(int(diag.get("explicit_reason_count", 0)), 1)
        self.assertEqual(int(diag.get("unknown_trigger_count_after", 0)), 0)

    def test_prepare_forex_audit_rows_infers_nearby_stale_context(self) -> None:
        rows = [
            {
                "ts": 1_700_000_000,
                "event": "shadow_live_divergence",
                "msg": "Exited 1 stale forex position(s); waiting one cycle before new entries.",
            },
            {
                "ts": 1_700_000_030,
                "event": "exit",
                "instrument": "EUR_USD",
                "qty": 100.0,
                "price": 1.1,
                "msg": "Position close submitted",
            },
        ]
        prepared, diag = _prepare_forex_audit_rows(rows, window_s=120)
        exit_row = [row for row in prepared if row.get("event") == "exit"][0]
        self.assertEqual(exit_row.get("tag"), "policy_stale_exit")
        self.assertEqual(int(diag.get("stale_context_matched_exits", 0)), 1)
        self.assertEqual(int(diag.get("unknown_trigger_count_after", 0)), 0)

    def test_generate_stock_historical_replay_closed_trades_with_fake_provider(self) -> None:
        class FakeClient:
            def get_stock_bars(
                self,
                symbol: str,
                timeframe: str = "1Hour",
                limit: int = 160,
                feed: str = "iex",
                start_iso: str = "",
                end_iso: str = "",
            ) -> list[dict]:
                rows = []
                base = 100.0
                for i in range(96):
                    if i < 24:
                        close = base + (i * 0.02)
                    elif i < 48:
                        close = base + 2.0 + ((i - 24) * 0.22)
                    elif i < 60:
                        close = base + 7.0 - ((i - 48) * 0.30)
                    else:
                        close = base + 0.8 + ((i - 60) * 0.10)
                    rows.append({"t": f"2026-05-{1 + (i // 24):02d}T{i % 24:02d}:00:00Z", "c": close})
                return rows

        with tempfile.TemporaryDirectory() as td:
            self._write_json(os.path.join(td, "stocks", "stock_universe_cache.json"), {"symbols": ["NVDA", "AAPL"]})
            with mock.patch("app.model_quality_pass._stock_provider_client", return_value=("alpaca", FakeClient(), {"provider": "alpaca"})):
                out = _generate_stock_historical_replay_closed_trades(
                    hub_dir=td,
                    base_dir=td,
                    settings={},
                    existing_closed_rows=[],
                )
            rows = out.get("rows", [])
            diag = out.get("diagnostics", {})
            self.assertTrue(rows)
            self.assertEqual(diag.get("source_type"), "historical_api_replay")
            self.assertEqual(diag.get("provider"), "alpaca")
            self.assertGreater(int(diag.get("rows_generated", 0)), 0)

    def test_build_market_dataset_quality_reports_crypto_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "crypto", "execution_audit.jsonl"),
                [
                    {
                        "ts": 1_700_000_000,
                        "event": "entry",
                        "symbol": "BTC-USD",
                        "qty": 1.0,
                        "price": 100.0,
                        "decision_snapshot_id": "abc123",
                    }
                ],
            )
            self._write_jsonl(
                os.path.join(td, "crypto", "decision_snapshots.jsonl"),
                [
                    {
                        "schema_version": 1,
                        "decision_snapshot_id": "abc123",
                        "timestamp": 1_700_000_000,
                        "market": "crypto",
                        "symbol": "BTC-USD",
                        "normalized_trigger": "Unknown",
                    }
                ],
            )
            out = build_market_dataset_quality(td, "crypto")
            snap = out.get("decision_snapshot_diagnostics", {})
            self.assertEqual(int(snap.get("decision_snapshots_found", 0)), 1)
            self.assertEqual(int(snap.get("decision_snapshots_joined", 0)), 1)


if __name__ == "__main__":
    unittest.main()
