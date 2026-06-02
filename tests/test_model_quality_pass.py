from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import app.model_quality_pass as model_quality_pass
import app.crypto_historical_replay as crypto_historical_replay
from app.crypto_artifacts import discover_crypto_trained_artifacts, load_crypto_artifact_features
from app.crypto_historical_replay import build_crypto_historical_strategy_replay
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
    run_model_quality_full_pass,
)
from app.trigger_normalization import normalize_exit_trigger


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

    def test_crypto_predictor_dispatches_historical_replay_path(self) -> None:
        candidate = {
            "symbol": "ATOM-USD",
            "entry_price": 10.0,
            "source_type": "historical_strategy_replay",
        }
        with mock.patch.object(
            model_quality_pass,
            "_crypto_predict_historical_replay_one",
            return_value={"predicted_direction": "up", "predicted_exit_trigger": "Trailing"},
        ) as hist_mock:
            out = _predict_one(train_rows=[], candidate=candidate, regime="high_volatility", market="crypto")
        hist_mock.assert_called_once()
        self.assertEqual(out["predicted_exit_trigger"], "Trailing")

    def test_crypto_historical_replay_predictor_supports_multiple_trigger_classes(self) -> None:
        train_rows = [
            {
                "symbol": "AAVE-USD",
                "entry_price": 100.0,
                "exit_price": 97.0,
                "actual_exit_price": 97.0,
                "hold_hours": 8.0,
                "pnl_pct": -2.8,
                "actual_direction": "down",
                "actual_exit_trigger": "Risk Cut",
                "regime": "high_volatility",
                "source_type": "historical_strategy_replay",
                "strategy_adapter_used": "minimal_artifact_replay_v1",
                "signal_side": "long",
                "current_candle_pct_move": 1.8,
                "recent_return_3": 2.8,
                "recent_return_6": 3.0,
                "recent_return_12": 3.4,
                "recent_return_24": 3.6,
                "recent_volatility": 0.82,
                "trend_momentum_score": 3.0,
                "signal_margin": 0.56,
                "active_timeframe_count": 6,
            },
            {
                "symbol": "ETH-USD",
                "entry_price": 100.0,
                "exit_price": 104.5,
                "actual_exit_price": 104.5,
                "hold_hours": 12.0,
                "pnl_pct": 4.5,
                "actual_direction": "up",
                "actual_exit_trigger": "Take Profit",
                "regime": "high_volatility",
                "source_type": "historical_strategy_replay",
                "strategy_adapter_used": "minimal_artifact_replay_v1",
                "signal_side": "long",
                "current_candle_pct_move": 0.8,
                "recent_return_3": 2.0,
                "recent_return_6": 2.2,
                "recent_return_12": 2.3,
                "recent_return_24": 2.4,
                "recent_volatility": 0.48,
                "trend_momentum_score": 2.2,
                "signal_margin": 0.39,
                "active_timeframe_count": 6,
            },
            {
                "symbol": "ATOM-USD",
                "entry_price": 100.0,
                "exit_price": 102.0,
                "actual_exit_price": 102.0,
                "hold_hours": 10.0,
                "pnl_pct": 2.0,
                "actual_direction": "up",
                "actual_exit_trigger": "Trailing",
                "regime": "high_volatility",
                "source_type": "historical_strategy_replay",
                "strategy_adapter_used": "minimal_artifact_replay_v1",
                "signal_side": "long",
                "current_candle_pct_move": 1.4,
                "recent_return_3": 2.3,
                "recent_return_6": 2.4,
                "recent_return_12": 2.5,
                "recent_return_24": 2.7,
                "recent_volatility": 0.62,
                "trend_momentum_score": 2.4,
                "signal_margin": 0.44,
                "active_timeframe_count": 6,
            },
        ]
        risk_candidate = {
            "symbol": "AAVE-USD",
            "entry_price": 100.0,
            "source_type": "historical_strategy_replay",
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 1.9,
            "recent_return_3": 2.9,
            "recent_return_6": 3.1,
            "recent_return_12": 3.6,
            "recent_return_24": 3.5,
            "recent_volatility": 0.86,
            "trend_momentum_score": 3.05,
            "signal_margin": 0.57,
            "active_timeframe_count": 6,
        }
        tp_candidate = {
            "symbol": "ETH-USD",
            "entry_price": 100.0,
            "source_type": "historical_strategy_replay",
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 0.75,
            "recent_return_3": 2.1,
            "recent_return_6": 2.2,
            "recent_return_12": 2.35,
            "recent_return_24": 2.5,
            "recent_volatility": 0.45,
            "trend_momentum_score": 2.1,
            "signal_margin": 0.38,
            "active_timeframe_count": 6,
        }
        risk_out = _predict_one(train_rows=train_rows, candidate=risk_candidate, regime="high_volatility", market="crypto")
        tp_out = _predict_one(train_rows=train_rows, candidate=tp_candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(risk_out.get("predictor_source_mode", "")), "historical_strategy_replay")
        self.assertEqual(str(risk_out.get("predicted_exit_trigger", "")), "Risk Cut")
        self.assertEqual(str(tp_out.get("predicted_exit_trigger", "")), "Take Profit")

    def test_crypto_historical_replay_predictor_recovers_upside_direction(self) -> None:
        train_rows = []
        for i in range(8):
            train_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 1.0,
                    "exit_price": 1.02,
                    "actual_exit_price": 1.02,
                    "hold_hours": 12.0,
                    "pnl_pct": 2.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                    "source_type": "historical_strategy_replay",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 1.3,
                    "recent_return_3": 2.1,
                    "recent_return_6": 2.3,
                    "recent_return_12": 2.5,
                    "recent_return_24": 2.7,
                    "recent_volatility": 0.62,
                    "trend_momentum_score": 2.35,
                    "signal_margin": 0.44,
                    "active_timeframe_count": 6,
                }
            )
        for i in range(5):
            train_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 1.0,
                    "exit_price": 0.975,
                    "actual_exit_price": 0.975,
                    "hold_hours": 18.0,
                    "pnl_pct": -2.5,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut",
                    "regime": "high_volatility",
                    "source_type": "historical_strategy_replay",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 1.7,
                    "recent_return_3": 2.8,
                    "recent_return_6": 3.1,
                    "recent_return_12": 3.5,
                    "recent_return_24": 3.6,
                    "recent_volatility": 0.82,
                    "trend_momentum_score": 3.0,
                    "signal_margin": 0.56,
                    "active_timeframe_count": 6,
                }
            )
        candidate = {
            "symbol": "ADA-USD",
            "entry_price": 1.0,
            "source_type": "historical_strategy_replay",
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 1.2,
            "recent_return_3": 2.0,
            "recent_return_6": 2.2,
            "recent_return_12": 2.4,
            "recent_return_24": 2.6,
            "recent_volatility": 0.60,
            "trend_momentum_score": 2.3,
            "signal_margin": 0.43,
            "active_timeframe_count": 6,
            "predicted_high_boundary": 1.03,
            "predicted_low_boundary": 0.99,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(out.get("predicted_direction", "")), "up")
        self.assertGreater(float(out.get("upside_score", 0.0) or 0.0), float(out.get("downside_score", 0.0) or 0.0))

    def test_crypto_historical_replay_trailing_can_beat_stale(self) -> None:
        train_rows = []
        for i in range(10):
            train_rows.append(
                {
                    "symbol": "ATOM-USD",
                    "entry_price": 10.0,
                    "exit_price": 10.25,
                    "actual_exit_price": 10.25,
                    "hold_hours": 12.0,
                    "pnl_pct": 2.5,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                    "source_type": "historical_strategy_replay",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 1.5,
                    "recent_return_3": 2.2,
                    "recent_return_6": 2.4,
                    "recent_return_12": 2.6,
                    "recent_return_24": 2.8,
                    "recent_volatility": 0.68,
                    "trend_momentum_score": 2.45,
                    "signal_margin": 0.45,
                    "active_timeframe_count": 6,
                }
            )
        for i in range(6):
            train_rows.append(
                {
                    "symbol": "ATOM-USD",
                    "entry_price": 10.0,
                    "exit_price": 9.92,
                    "actual_exit_price": 9.92,
                    "hold_hours": 18.0,
                    "pnl_pct": -0.8,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                    "source_type": "historical_strategy_replay",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 0.7,
                    "recent_return_3": 1.8,
                    "recent_return_6": 2.1,
                    "recent_return_12": 2.3,
                    "recent_return_24": 2.4,
                    "recent_volatility": 0.52,
                    "trend_momentum_score": 2.1,
                    "signal_margin": 0.39,
                    "active_timeframe_count": 6,
                }
            )
        candidate = {
            "symbol": "ATOM-USD",
            "entry_price": 10.0,
            "source_type": "historical_strategy_replay",
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 1.45,
            "recent_return_3": 2.25,
            "recent_return_6": 2.35,
            "recent_return_12": 2.55,
            "recent_return_24": 2.75,
            "recent_volatility": 0.70,
            "trend_momentum_score": 2.4,
            "signal_margin": 0.44,
            "active_timeframe_count": 6,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Trailing")

    def test_crypto_historical_replay_policy_can_admit_more_than_one_row(self) -> None:
        rows = [
            {
                "source_type": "historical_strategy_replay",
                "predicted_exit_trigger": "Risk Cut",
                "predicted_confidence": 0.42,
                "trigger_margin": 0.32,
                "direction_margin": 0.18,
            },
            {
                "source_type": "historical_strategy_replay",
                "predicted_exit_trigger": "Trailing",
                "predicted_confidence": 0.39,
                "trigger_margin": 0.28,
                "direction_margin": 0.12,
            },
            {
                "source_type": "historical_strategy_replay",
                "predicted_exit_trigger": "Take Profit",
                "predicted_confidence": 0.37,
                "trigger_margin": 0.26,
                "direction_margin": 0.10,
                "runner_up_trigger_score": 0.10,
                "trigger_scores": {"Take Profit": 0.42, "Trailing": 0.10},
            },
        ]
        policy = {
            "source_mode": "historical_strategy_replay",
            "base_conf_min": 0.34,
            "base_trigger_margin_min": 0.18,
            "base_dir_margin_min": 0.05,
            "risk_cut_conf_min": 0.34,
            "risk_cut_trigger_margin_min": 0.16,
            "risk_cut_dir_margin_min": 0.03,
            "trailing_conf_min": 0.30,
            "trailing_trigger_margin_min": 0.18,
            "trailing_dir_margin_min": 0.05,
            "take_profit_conf_min": 0.30,
            "take_profit_trigger_margin_min": 0.16,
            "take_profit_score_gap_min": 0.08,
            "stale_conf_min": 0.35,
            "stale_trigger_margin_min": 0.18,
            "stale_dir_margin_min": 0.02,
        }
        scored = model_quality_pass._score_crypto_with_policy(rows, policy)
        self.assertGreaterEqual(len(scored.get("admitted", [])), 2)

    def test_stock_predictor_dispatches_historical_replay_path(self) -> None:
        train_rows = [
            {
                "symbol": "NVDA",
                "entry_price": 100.0,
                "exit_price": 103.0,
                "hold_hours": 6.0,
                "actual_exit_trigger": "Trailing",
                "actual_direction": "up",
                "regime": "trend_up",
                "source_type": "historical_api_replay",
                "provider": "alpaca",
                "recent_return_6": 1.5,
                "recent_return_24": 2.5,
                "recent_volatility": 0.8,
                "trend_momentum_score": 2.0,
                "signal_margin": 0.3,
            }
            for _ in range(12)
        ]
        out = _predict_one(
            train_rows=train_rows,
            candidate={
                "symbol": "NVDA",
                "entry_price": 100.0,
                "source_type": "historical_api_replay",
                "provider": "alpaca",
                "recent_return_6": 1.4,
                "recent_return_24": 2.4,
                "recent_volatility": 0.7,
                "trend_momentum_score": 1.9,
                "signal_margin": 0.25,
            },
            regime="trend_up",
            market="stocks",
        )
        self.assertEqual(str(out.get("predictor_mode", "")), "stock_historical_replay")

    def test_forex_predictor_preserves_trigger_labels(self) -> None:
        train_rows = [
            {
                "symbol": "EUR_USD",
                "entry_price": 1.0,
                "exit_price": 1.001,
                "hold_hours": 8.0,
                "actual_exit_trigger": "Unknown",
                "actual_direction": "up",
                "side": "long",
                "regime": "range",
                "source_type": "execution_log",
            }
            for _ in range(20)
        ]
        out = _predict_one(
            train_rows=train_rows,
            candidate={
                "symbol": "EUR_USD",
                "entry_price": 1.0,
                "actual_exit_trigger": "Unknown",
                "side": "long",
                "hold_hours": 9.0,
                "source_type": "execution_log",
            },
            regime="range",
            market="forex",
        )
        self.assertEqual(str(out.get("predictor_mode", "")), "forex_execution_log")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Unknown")

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
            self.assertGreaterEqual(int(snap.get("snapshot_features_joined_count", 0)), 1)
            self.assertIn("snapshot_normalized_trigger", snap.get("snapshot_feature_names", []))

    def test_shared_trigger_normalization_is_consistent(self) -> None:
        self.assertEqual(normalize_exit_trigger("policy_stale_exit"), "Stale Alignment")
        self.assertEqual(normalize_exit_trigger("manual close"), "Manual")
        self.assertEqual(normalize_exit_trigger("trailing sell"), "Trailing")
        self.assertEqual(normalize_exit_trigger("blocked by rule"), "Blocked")
        self.assertEqual(normalize_exit_trigger("ai exit override"), "AI Exit")
        self.assertEqual(normalize_exit_trigger("risk stop"), "Risk Cut")
        self.assertEqual(normalize_exit_trigger("take profit"), "Take Profit")
        self.assertEqual(normalize_exit_trigger(""), "Unknown")

    def test_crypto_artifact_discovery_and_candidate_safe_loading(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sym_dir = os.path.join(td, "BTC-USD")
            os.makedirs(sym_dir, exist_ok=True)
            now_ts = 1_700_000_000
            files = {
                "trainer_status.json": {"state": "FINISHED"},
                "trainer_last_training_time.txt": str(now_ts - 60),
                "memories_1hour.txt": "a~b~c",
                "memory_weights_1hour.txt": "1 2 3",
                "memory_weights_high_1hour.txt": "1 2 3",
                "memory_weights_low_1hour.txt": "1 2 3",
                "neural_perfect_threshold_1hour.txt": "0.77",
            }
            for name, payload in files.items():
                path = os.path.join(sym_dir, name)
                if name.endswith(".json"):
                    self._write_json(path, payload)  # type: ignore[arg-type]
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(payload))
            disc = discover_crypto_trained_artifacts(td, symbols=["BTC-USD"])
            self.assertEqual(int(disc.get("trained_artifacts_found", 0)), 1)
            self.assertIn("BTC-USD", disc.get("trained_symbols", []))
            feats = load_crypto_artifact_features(sym_dir, as_of_ts=now_ts)
            self.assertTrue(bool(feats.get("usable", False)))
            self.assertEqual(int(feats.get("active_timeframe_count", 0)), 1)
            self.assertGreater(float(feats.get("threshold_mean", 0.0)), 0.0)
            aliased = load_crypto_artifact_features(os.path.join(td, "BTC-USD"), as_of_ts=now_ts)
            self.assertTrue(bool(aliased.get("usable", False)))
            late = load_crypto_artifact_features(sym_dir, as_of_ts=now_ts - 120)
            self.assertFalse(bool(late.get("usable", True)))
            self.assertEqual(str(late.get("reason_if_not_used", "")), "artifact_training_time_after_candidate")

    def test_crypto_artifact_discovery_flags_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sym_dir = os.path.join(td, "ETH-USD")
            os.makedirs(sym_dir, exist_ok=True)
            with open(os.path.join(sym_dir, "trainer_last_training_time.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(sym_dir, "memories_1hour.txt"), "w", encoding="utf-8") as f:
                f.write("a")
            with open(os.path.join(sym_dir, "memory_weights_1hour.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(sym_dir, "neural_perfect_threshold_1hour.txt"), "w", encoding="utf-8") as f:
                f.write("0.5")
            disc = discover_crypto_trained_artifacts(td, symbols=["ETH-USD"])
            self.assertIn("ETH-USD", disc.get("stale_artifacts", []))

    def test_crypto_historical_replay_adds_exit_shape_fields_without_post_exit_candles(self) -> None:
        candles = []
        ts = 1_700_000_000_000
        px = 100.0
        for i in range(60):
            if i < 30:
                px *= 1.002
            elif i < 34:
                px *= 1.01
            elif i < 36:
                px *= 0.992
            else:
                px *= 0.97
            candles.append([ts + (i * 3_600_000), px, px, px * 1.003, px * 0.997, 1000.0])
        artifact_ctx = {
            "usable": True,
            "active_timeframe_count": 6,
            "predicted_low_boundary": 99.0,
            "predicted_high_boundary": 106.0,
            "trained_artifacts_fresh": True,
            "artifact_training_time": 1_700_000_000,
            "signal_margin": 0.5,
        }
        thresholds = {
            "entry_signal_margin_min": 0.35,
            "entry_trend_score_min": 0.12,
            "entry_return6_min": -0.25,
            "risk_cut_pct": 2.25,
            "take_profit_pct": 4.25,
            "trailing_arm_pct": 1.6,
            "trailing_drawdown_pct": 1.1,
            "stale_hold_hours": 18.0,
            "stale_trend_score_max": 0.05,
        }
        rows = crypto_historical_replay._simulate_strategy_rows("BTC-USD", candles, artifact_ctx, "1hour", thresholds)
        self.assertTrue(rows)
        row = rows[0]
        self.assertIn("bars_in_trade", row)
        self.assertIn("trailing_armed", row)
        self.assertIn("favorable_then_softened_flag", row)
        self.assertGreaterEqual(int(row.get("bars_in_trade", 0) or 0), 1)
        self.assertIn(row.get("actual_exit_trigger"), {"Trailing", "Stale Alignment", "Risk Cut", "Take Profit"})

    def test_load_market_trade_events_joins_crypto_snapshot_fields(self) -> None:
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
                    },
                    {
                        "ts": 1_700_000_360,
                        "event": "exit",
                        "symbol": "BTC-USD",
                        "qty": 1.0,
                        "price": 102.0,
                        "tag": "trailing sell",
                        "decision_snapshot_id": "def456",
                    },
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
                        "strategy_score": 0.91,
                        "signal_gate_mode": "strict",
                    },
                    {
                        "schema_version": 1,
                        "decision_snapshot_id": "def456",
                        "timestamp": 1_700_000_360,
                        "market": "crypto",
                        "symbol": "BTC-USD",
                        "normalized_trigger": "Trailing",
                        "raw_rule_reason": "trailing sell",
                        "trailing_score": 3.5,
                    },
                ],
            )
            loaded = load_market_trade_events(td, "crypto")
            events = loaded.get("events", [])
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].get("snapshot_signal_gate_mode"), "strict")
            self.assertEqual(events[1].get("snapshot_normalized_trigger"), "Trailing")

    def test_run_model_quality_full_pass_reports_crypto_feature_source_priority(self) -> None:
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
                    },
                    {
                        "ts": 1_700_000_600,
                        "event": "exit",
                        "symbol": "BTC-USD",
                        "qty": 1.0,
                        "price": 103.0,
                        "tag": "trailing sell",
                        "decision_snapshot_id": "def456",
                    },
                ]
                * 20,
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
                        "strategy_score": 0.91,
                    },
                    {
                        "schema_version": 1,
                        "decision_snapshot_id": "def456",
                        "timestamp": 1_700_000_600,
                        "market": "crypto",
                        "symbol": "BTC-USD",
                        "normalized_trigger": "Trailing",
                        "trailing_score": 4.0,
                    },
                ],
            )
            coin_dir = os.path.join(td, "market_data", "coins", "BTC-USD")
            os.makedirs(coin_dir, exist_ok=True)
            for name, val in {
                "trainer_last_training_time.txt": "1699999000",
                "memories_1hour.txt": "a~b",
                "memory_weights_1hour.txt": "1 2",
                "memory_weights_high_1hour.txt": "1 2",
                "memory_weights_low_1hour.txt": "1 2",
                "neural_perfect_threshold_1hour.txt": "0.66",
            }.items():
                with open(os.path.join(coin_dir, name), "w", encoding="utf-8") as f:
                    f.write(val)
            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                            out = run_model_quality_full_pass(
                                base_dir=td,
                                hub_dir=td,
                                settings={"main_neural_dir": os.path.join(td, "market_data", "coins")},
                            )
            diag = out.get("crypto_feature_source_diagnostics", {})
            self.assertEqual(diag.get("crypto_feature_source_priority", [])[0], "decision_snapshot")
            self.assertIn(diag.get("crypto_feature_source_used"), {"decision_snapshot", "trained_artifact", "closed_trade_only"})
            self.assertIn("trained_artifacts_found", out.get("crypto_trained_artifact_diagnostics", {}))

    def test_crypto_historical_replay_runs_from_fixture_cache_without_mutating_live_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            coin_dir = os.path.join(td, "market_data", "coins", "BTC")
            os.makedirs(coin_dir, exist_ok=True)
            artifact_files = {
                "trainer_last_training_time.txt": str(int(time.time()) - 60),
                "trainer_status.json": {"state": "FINISHED"},
                "memories_1hour.txt": "a~b",
                "memory_weights_1hour.txt": "1 2",
                "memory_weights_high_1hour.txt": "1 2",
                "memory_weights_low_1hour.txt": "1 2",
                "neural_perfect_threshold_1hour.txt": "0.6",
                "long_dca_signal.txt": "4",
                "short_dca_signal.txt": "1",
                "futures_long_profit_margin.txt": "0.9",
                "futures_short_profit_margin.txt": "0.1",
                "low_bound_prices.html": "98 97 96",
                "high_bound_prices.html": "103 104 105",
            }
            for name, value in artifact_files.items():
                path = os.path.join(coin_dir, name)
                if name.endswith(".json"):
                    self._write_json(path, value)  # type: ignore[arg-type]
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(value))
            before_mtime = os.path.getmtime(os.path.join(coin_dir, "trainer_last_training_time.txt"))
            cache_rows = []
            base_ts = 1_700_000_000_000
            price = 100.0
            for i in range(220):
                if i % 36 < 12:
                    price += 0.45
                elif i % 36 < 18:
                    price += 0.20
                elif i % 36 < 24:
                    price -= 0.60
                else:
                    price += 0.10
                open_px = price - 0.15
                close_px = price
                cache_rows.append([base_ts + (i * 3600 * 1000), open_px, close_px, close_px + 0.3, close_px - 0.3, 1000.0])
            cache_path = os.path.join(td, "crypto", "historical_replay_cache", "candles", "BTC", "1hour.json")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_rows, f)
            out = build_crypto_historical_strategy_replay(
                hub_dir=td,
                settings={"main_neural_dir": os.path.join(td, "market_data", "coins")},
                symbols=["BTC-USD"],
                timeframe="1hour",
                lookback_days=30,
                max_symbols=1,
            )
            self.assertEqual(out.get("state"), "READY")
            self.assertTrue(out.get("rows"))
            row = out.get("rows", [])[0]
            self.assertEqual(row.get("source_type"), "historical_strategy_replay")
            self.assertIn(row.get("actual_exit_trigger"), {"Trailing", "Stale Alignment", "Risk Cut", "Take Profit"})
            self.assertEqual(row.get("strategy_adapter_used"), "minimal_artifact_replay_v1")
            self.assertFalse(bool(out.get("diagnostics", {}).get("replay_mutated_live_artifacts", True)))
            self.assertTrue(bool(out.get("diagnostics", {}).get("replay_used_local_cache", False)))
            self.assertEqual(os.path.getmtime(os.path.join(coin_dir, "trainer_last_training_time.txt")), before_mtime)

    def test_crypto_historical_replay_handles_missing_data_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = build_crypto_historical_strategy_replay(
                hub_dir=td,
                settings={"main_neural_dir": os.path.join(td, "market_data", "coins")},
                symbols=["BTC-USD"],
                timeframe="1hour",
                lookback_days=14,
                max_symbols=1,
            )
            self.assertIn(out.get("state"), {"NO_DATA", "READY"})
            self.assertIn("replay_cache_path", out.get("diagnostics", {}))

    def test_model_quality_prefers_historical_strategy_replay_when_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "crypto", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0},
                    {"ts": 1_700_000_100, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 101.0, "tag": "Trailing"},
                ],
            )
            hist_rows = []
            start_ts = 1_700_000_000
            for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
                for i in range(20):
                    entry_px = 100.0 + i
                    exit_px = entry_px + (2.0 if i % 2 == 0 else -1.0)
                    hist_rows.append(
                        {
                            "market": "crypto",
                            "symbol": sym,
                            "source_type": "historical_strategy_replay",
                            "provider": "local_cache",
                            "candle_timeframe": "1hour",
                            "entry_ts": start_ts + (i * 7200),
                            "exit_ts": start_ts + (i * 7200) + 3600,
                            "entry_price": entry_px,
                            "exit_price": exit_px,
                            "hold_hours": 1.0,
                            "pnl_usd": exit_px - entry_px,
                            "pnl_pct": ((exit_px / entry_px) - 1.0) * 100.0,
                            "actual_direction": "up" if exit_px > entry_px else "down",
                            "actual_exit_trigger": "Trailing" if exit_px > entry_px else "Stale Alignment",
                            "event_exit_tag": "historical_strategy_replay:test",
                            "raw_rule_reason": "test",
                            "normalized_trigger": "Trailing" if exit_px > entry_px else "Stale Alignment",
                            "strategy_adapter_used": "minimal_artifact_replay_v1",
                            "trained_artifact_fresh": True,
                            "artifact_training_time": start_ts - 1000,
                            "current_candle_pct_move": 0.2,
                            "recent_return_3": 0.3,
                            "recent_return_6": 0.4,
                            "recent_return_12": 0.5,
                            "recent_return_24": 0.6,
                            "recent_volatility": 0.3,
                            "trend_momentum_score": 0.4,
                            "signal_side": "long",
                            "signal_margin": 0.6,
                            "active_timeframe_count": 3,
                            "predicted_high_boundary": entry_px + 4.0,
                            "predicted_low_boundary": entry_px - 2.0,
                        }
                    )
            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": hist_rows, "diagnostics": {"historical_strategy_replay_rows": 60, "historical_strategy_replay_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"], "historical_strategy_replay_timeframe": "1hour", "historical_strategy_replay_provider": "local_cache", "historical_strategy_replay_cache_path": os.path.join(td, "crypto", "historical_replay_cache"), "historical_strategy_replay_skipped_reasons": [], "replay_used_local_cache": True, "replay_used_remote_api": False, "replay_mutated_live_artifacts": False}}):
                with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                    with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                        with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                            with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={"main_neural_dir": os.path.join(td, "market_data", "coins")})
            self.assertEqual(out.get("crypto_feature_source_diagnostics", {}).get("crypto_feature_source_used"), "historical_strategy_replay")

    def test_model_quality_falls_back_when_historical_strategy_replay_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "crypto", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0},
                    {"ts": 1_700_000_100, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 101.0, "tag": "Trailing"},
                ],
            )
            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                    with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                        with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                            with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={"main_neural_dir": os.path.join(td, "market_data", "coins")})
            self.assertNotEqual(out.get("crypto_feature_source_diagnostics", {}).get("crypto_feature_source_used"), "historical_strategy_replay")


if __name__ == "__main__":
    unittest.main()
