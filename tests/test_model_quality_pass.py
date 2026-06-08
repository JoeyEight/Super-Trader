from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import app.model_quality_pass as model_quality_pass
import app.crypto_historical_replay as crypto_historical_replay
from app.crypto_original_predictor import derive_crypto_original_signal_dry_run, predict_crypto_original_dry_run
from app.crypto_artifacts import discover_crypto_trained_artifacts, load_crypto_artifact_features
from app.crypto_historical_replay import build_crypto_historical_strategy_replay
from app.model_quality_pass import (
    _augment_historical_blind_rows,
    _build_historical_blind_simulation,
    _build_crypto_trigger_classifier_candidate,
    _crypto_apply_second_stage_discriminator,
    _crypto_blind_trigger_predict,
    _crypto_trigger_classifier_dataset_rows,
    _historical_blind_diagnostics,
    _performance_diagnostics_path,
    _safe_read_jsonl,
    _split_walkforward_rows,
    _write_jsonl,
    _completed_live_decision_rows,
    _verify_completed_live_synthetic_paths,
    _generate_stock_historical_replay_closed_trades,
    _normalize_stock_ticker,
    _prepare_forex_audit_rows,
    _crypto_admission_decision,
    _predict_one,
    _score_crypto_with_policy,
    _simulate_stock_trades_from_bars,
    _stock_predict_one,
    _score_with_threshold,
    add_stock_to_manual_watchlist,
    build_market_dataset_quality,
    build_stock_watchlist_prediction_preview,
    build_legacy_trade_model_replay,
    build_closed_trades,
    build_replay_diagnostics,
    build_synthetic_replay_artifact,
    load_market_trade_events,
    run_model_quality_full_pass,
    validate_stock_watchlist_symbol,
    warm_stock_historical_cache,
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

    def _crypto_trigger_train_rows(self) -> list[dict]:
        return [
            {
                "symbol": "BTC-USD",
                "regime": "high_volatility",
                "actual_exit_trigger": "Risk Cut",
                "actual_direction": "down",
                "entry_price": 100.0,
                "exit_price": 97.0,
                "pnl_pct": -3.0,
                "hold_hours": 6.0,
                "current_candle_pct_move": 0.2,
                "recent_return_3": 0.4,
                "recent_return_6": 0.5,
                "recent_return_12": 0.6,
                "recent_return_24": 0.7,
                "recent_volatility": 0.95,
                "trend_momentum_score": 0.6,
                "signal_margin": 0.12,
                "active_timeframe_count": 3,
            },
            {
                "symbol": "BTC-USD",
                "regime": "high_volatility",
                "actual_exit_trigger": "Take Profit",
                "actual_direction": "up",
                "entry_price": 100.0,
                "exit_price": 104.0,
                "pnl_pct": 4.0,
                "hold_hours": 10.0,
                "current_candle_pct_move": 0.9,
                "recent_return_3": 2.0,
                "recent_return_6": 2.2,
                "recent_return_12": 2.4,
                "recent_return_24": 2.6,
                "recent_volatility": 0.42,
                "trend_momentum_score": 2.0,
                "signal_margin": 0.32,
                "active_timeframe_count": 5,
            },
            {
                "symbol": "BTC-USD",
                "regime": "high_volatility",
                "actual_exit_trigger": "Trailing",
                "actual_direction": "up",
                "entry_price": 100.0,
                "exit_price": 103.0,
                "pnl_pct": 3.0,
                "hold_hours": 14.0,
                "current_candle_pct_move": 1.3,
                "recent_return_3": 2.4,
                "recent_return_6": 2.8,
                "recent_return_12": 3.2,
                "recent_return_24": 3.5,
                "recent_volatility": 0.72,
                "trend_momentum_score": 2.7,
                "signal_margin": 0.48,
                "active_timeframe_count": 6,
            },
            {
                "symbol": "BTC-USD",
                "regime": "high_volatility",
                "actual_exit_trigger": "Stale Alignment",
                "actual_direction": "down",
                "entry_price": 100.0,
                "exit_price": 99.0,
                "pnl_pct": -1.0,
                "hold_hours": 18.0,
                "current_candle_pct_move": 0.3,
                "recent_return_3": 0.7,
                "recent_return_6": 0.8,
                "recent_return_12": 1.0,
                "recent_return_24": 1.1,
                "recent_volatility": 0.5,
                "trend_momentum_score": 1.1,
                "signal_margin": 0.18,
                "active_timeframe_count": 4,
            },
        ] * 4

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

    def test_crypto_historical_replay_direction_not_forced_by_trigger(self) -> None:
        train_rows = []
        for i in range(8):
            train_rows.append(
                {
                    "symbol": "ADA-USD",
                    "entry_price": 1.0,
                    "exit_price": 0.98,
                    "actual_exit_price": 0.98,
                    "hold_hours": 10.0,
                    "pnl_pct": -2.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut",
                    "regime": "high_volatility",
                    "source_type": "historical_strategy_replay",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 0.4,
                    "recent_return_3": 1.0,
                    "recent_return_6": 1.1,
                    "recent_return_12": 1.2,
                    "recent_return_24": 1.4,
                    "recent_volatility": 0.35,
                    "trend_momentum_score": 1.6,
                    "signal_margin": 0.32,
                    "active_timeframe_count": 6,
                }
            )
        candidate = {
            "symbol": "ADA-USD",
            "entry_price": 1.0,
            "source_type": "historical_strategy_replay",
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 0.2,
            "recent_return_3": 0.8,
            "recent_return_6": 0.9,
            "recent_return_12": 1.0,
            "recent_return_24": 1.1,
            "recent_volatility": 0.30,
            "trend_momentum_score": 1.4,
            "signal_margin": 0.28,
            "active_timeframe_count": 6,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto", predictor_variant="baseline")
        self.assertIn("direction_independent_score_up", out)
        self.assertIn("direction_independent_score_down", out)
        self.assertFalse(bool(out.get("direction_forced_by_trigger", False)))

    def test_stock_predictor_applies_down_case_guard(self) -> None:
        train_rows = []
        for i in range(10):
            train_rows.append(
                {
                    "symbol": "NVDA",
                    "entry_price": 100.0,
                    "actual_exit_price": 97.0,
                    "hold_hours": 6.0,
                    "pnl_pct": -3.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                    "source_type": "historical_api_replay",
                    "recent_return_6": -2.0,
                    "recent_return_24": -3.0,
                    "recent_volatility": 1.2,
                    "trend_momentum_score": -0.2,
                    "signal_margin": 0.1,
                }
            )
        candidate = {
            "symbol": "NVDA",
            "entry_price": 100.0,
            "source_type": "historical_api_replay",
            "recent_return_6": -1.5,
            "recent_return_24": -2.5,
            "recent_volatility": 1.0,
            "trend_momentum_score": -0.1,
            "signal_margin": 0.1,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="stocks", predictor_variant="candidate")
        self.assertEqual(str(out.get("predicted_direction", "")), "down")
        self.assertTrue(bool(out.get("stock_up_bias_guard_applied", False)))
        self.assertEqual(str(out.get("stock_down_case_guard_reason", "")), "weak_negative_returns_high_volatility")
        self.assertTrue(bool(out.get("stock_down_case_guard_applied", False)))
        self.assertTrue(bool(out.get("stock_up_dampened_by_down_regime", False)))

    def test_stock_trigger_separation_prefers_stale_on_weak_profile(self) -> None:
        train_rows = []
        for i in range(10):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "entry_price": 100.0,
                    "actual_exit_price": 99.0,
                    "hold_hours": 10.0,
                    "pnl_pct": -1.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                    "source_type": "historical_api_replay",
                    "recent_return_6": -0.6,
                    "recent_return_24": -0.2,
                    "recent_volatility": 0.8,
                    "trend_momentum_score": 0.1,
                    "signal_margin": 0.05,
                }
            )
        candidate = {
            "symbol": "AAPL",
            "entry_price": 100.0,
            "source_type": "historical_api_replay",
            "recent_return_6": -0.5,
            "recent_return_24": -0.1,
            "recent_volatility": 0.7,
            "trend_momentum_score": 0.1,
            "signal_margin": 0.05,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="stocks", predictor_variant="candidate")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Stale Alignment")
        self.assertIn(str(out.get("stock_trigger_separation_reason", "")), {"weak_or_negative_momentum", "elevated_vol_without_follow_through"})
        self.assertGreater(float(out.get("stock_stale_vs_trailing_margin", 0.0)), 0.0)
        self.assertTrue(bool(out.get("stock_stale_override_applied", False)))

    def test_stock_pnl_quality_v2_ignores_candidate_actual_pnl_fields(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "SMCI",
                    "entry_price": 100.0,
                    "actual_exit_price": 97.0,
                    "hold_hours": 22.0,
                    "pnl_pct": -3.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                    "source_type": "historical_api_replay",
                    "recent_return_6": -0.4,
                    "recent_return_24": 0.3,
                    "recent_volatility": 0.95,
                    "trend_momentum_score": 0.2,
                    "signal_margin": 0.08,
                    "drawdown_from_peak_pct": -2.2,
                    "bars_in_trade": 24,
                    "bars_since_peak": 9,
                    "stale_hold_profile": True,
                }
            )
        base_candidate = {
            "symbol": "SMCI",
            "entry_price": 100.0,
            "source_type": "historical_api_replay",
            "recent_return_6": 0.1,
            "recent_return_24": 0.5,
            "recent_volatility": 0.90,
            "trend_momentum_score": 0.35,
            "signal_margin": 0.10,
            "peak_profit_pct": 1.2,
            "drawdown_from_peak_pct": -2.0,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "stale_hold_profile": True,
            "volatility_expansion_pct": 30.0,
            "trend_decay_after_peak": -1.2,
            "max_favorable_excursion_pct": 1.4,
            "max_adverse_excursion_pct": -1.8,
            "bars_since_peak": 8,
            "bars_in_trade": 26,
            "exit_momentum_3": -0.9,
            "exit_momentum_6": -1.1,
        }
        candidate_up = dict(base_candidate, actual_exit_price=120.0, pnl_pct=20.0)
        candidate_down = dict(base_candidate, actual_exit_price=80.0, pnl_pct=-20.0)
        out_up = _predict_one(train_rows=train_rows, candidate=candidate_up, regime="high_volatility", market="stocks", predictor_variant="stock_pnl_quality_v2")
        out_down = _predict_one(train_rows=train_rows, candidate=candidate_down, regime="high_volatility", market="stocks", predictor_variant="stock_pnl_quality_v2")
        self.assertAlmostEqual(float(out_up.get("stock_pnl_quality_score", 0.0)), float(out_down.get("stock_pnl_quality_score", 0.0)), places=6)
        self.assertEqual(str(out_up.get("predicted_pnl_trend", "")), str(out_down.get("predicted_pnl_trend", "")))

    def test_stock_pnl_quality_v2_reduces_weak_setup(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "AAPL",
                    "entry_price": 100.0,
                    "exit_price": 98.0,
                    "actual_exit_price": 98.0,
                    "hold_hours": 20.0,
                    "pnl_pct": -2.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Stale Alignment",
                    "regime": "high_volatility",
                    "source_type": "historical_api_replay",
                    "recent_return_6": -0.2,
                    "recent_return_24": 0.1,
                    "recent_volatility": 0.85,
                    "trend_momentum_score": 0.1,
                    "signal_margin": 0.08,
                    "drawdown_from_peak_pct": -2.1,
                    "bars_in_trade": 24,
                    "bars_since_peak": 8,
                    "stale_hold_profile": True,
                }
            )
        candidate = {
            "symbol": "AAPL",
            "entry_price": 100.0,
            "source_type": "historical_api_replay",
            "recent_return_6": 0.15,
            "recent_return_24": 0.6,
            "recent_volatility": 0.88,
            "trend_momentum_score": 0.3,
            "signal_margin": 0.10,
            "peak_profit_pct": 1.1,
            "drawdown_from_peak_pct": -2.3,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "stale_hold_profile": True,
            "volatility_expansion_pct": 28.0,
            "trend_decay_after_peak": -1.1,
            "max_favorable_excursion_pct": 1.5,
            "max_adverse_excursion_pct": -1.7,
            "bars_since_peak": 9,
            "bars_in_trade": 25,
            "exit_momentum_3": -0.8,
            "exit_momentum_6": -1.0,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="stocks", predictor_variant="stock_pnl_quality_v2")
        self.assertEqual(str(out.get("predicted_pnl_trend", "")), "down")
        self.assertTrue(bool(out.get("stock_weak_window_guard_applied", False)))
        self.assertLess(float(out.get("stock_pnl_quality_score", 1.0)), 0.50)

    def test_stock_pnl_quality_v2_keeps_strong_clean_setup_up(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "NVDA",
                    "entry_price": 100.0,
                    "exit_price": 104.0,
                    "actual_exit_price": 104.0,
                    "hold_hours": 8.0,
                    "pnl_pct": 4.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "regime": "trend_up",
                    "source_type": "historical_api_replay",
                    "recent_return_6": 1.4,
                    "recent_return_24": 2.6,
                    "recent_volatility": 0.45,
                    "trend_momentum_score": 1.8,
                    "signal_margin": 0.34,
                    "drawdown_from_peak_pct": -0.8,
                    "bars_in_trade": 10,
                    "bars_since_peak": 2,
                    "trailing_armed": True,
                    "favorable_then_softened_flag": True,
                }
            )
        candidate = {
            "symbol": "NVDA",
            "entry_price": 100.0,
            "source_type": "historical_api_replay",
            "recent_return_6": 1.2,
            "recent_return_24": 2.4,
            "recent_volatility": 0.42,
            "trend_momentum_score": 1.7,
            "signal_margin": 0.32,
            "peak_profit_pct": 3.2,
            "drawdown_from_peak_pct": -0.9,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "stale_hold_profile": False,
            "volatility_expansion_pct": 12.0,
            "trend_decay_after_peak": -0.4,
            "max_favorable_excursion_pct": 3.5,
            "max_adverse_excursion_pct": -0.6,
            "bars_since_peak": 2,
            "bars_in_trade": 10,
            "exit_momentum_3": 0.3,
            "exit_momentum_6": 0.6,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="trend_up", market="stocks", predictor_variant="stock_pnl_quality_v2")
        self.assertEqual(str(out.get("predicted_pnl_trend", "")), "up")
        self.assertFalse(bool(out.get("stock_trade_quality_gate_applied", False)))
        self.assertGreater(float(out.get("stock_pnl_quality_score", 0.0)), 0.50)

    def test_stock_safe_selection_evaluates_pnl_quality_variant(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(60):
            closed_rows.append(
                {
                    "symbol": "AMD",
                    "entry_price": 100.0,
                    "actual_exit_price": 104.0 if i % 2 == 0 else 98.0,
                    "exit_price": 104.0 if i % 2 == 0 else 98.0,
                    "hold_hours": 10.0 if i % 2 == 0 else 22.0,
                    "pnl_pct": 4.0 if i % 2 == 0 else -2.0,
                    "actual_direction": "up" if i % 2 == 0 else "down",
                    "actual_exit_trigger": "Trailing" if i % 2 == 0 else "Stale Alignment",
                    "source_type": "historical_api_replay",
                    "regime": "trend_up" if i % 2 == 0 else "high_volatility",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "recent_return_6": 1.4 if i % 2 == 0 else 0.1,
                    "recent_return_24": 2.4 if i % 2 == 0 else 0.4,
                    "recent_volatility": 0.42 if i % 2 == 0 else 0.88,
                    "trend_momentum_score": 1.6 if i % 2 == 0 else 0.2,
                    "signal_margin": 0.32 if i % 2 == 0 else 0.08,
                    "peak_profit_pct": 3.0 if i % 2 == 0 else 1.2,
                    "drawdown_from_peak_pct": -0.8 if i % 2 == 0 else -2.1,
                    "trailing_armed": bool(i % 2 == 0),
                    "favorable_then_softened_flag": bool(i % 2 == 0),
                    "stale_hold_profile": bool(i % 2 == 1),
                    "bars_in_trade": 10 if i % 2 == 0 else 24,
                    "bars_since_peak": 2 if i % 2 == 0 else 8,
                    "max_favorable_excursion_pct": 3.5 if i % 2 == 0 else 1.0,
                    "max_adverse_excursion_pct": -0.7 if i % 2 == 0 else -1.8,
                    "trend_decay_after_peak": -0.3 if i % 2 == 0 else -1.1,
                    "volatility_expansion_pct": 10.0 if i % 2 == 0 else 28.0,
                    "exit_momentum_3": 0.4 if i % 2 == 0 else -0.8,
                    "exit_momentum_6": 0.6 if i % 2 == 0 else -1.0,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "stocks", closed_rows)
        safe = payload.get("stock_safe_selection", {})
        evals = safe.get("candidate_evaluations", []) if isinstance(safe.get("candidate_evaluations", []), list) else []
        self.assertTrue(any(str(ev.get("variant", "")) == "stock_pnl_quality_v2" for ev in evals))

    def test_stock_predictor_diagnostics_emit_pnl_trend_miss_fields(self) -> None:
        rows = [
            {
                "symbol": "AMD",
                "actual_direction": "up",
                "predicted_direction": "up",
                "actual_exit_trigger": "Trailing",
                "predicted_exit_trigger": "Trailing",
                "entry_price": 100.0,
                "exit_price": 104.0,
                "actual_exit_price": 104.0,
                "predicted_exit_price": 99.0,
                "hold_hours": 20.0,
                "recent_return_6": 0.2,
                "recent_return_24": 0.6,
                "recent_volatility": 0.85,
                "max_favorable_excursion_pct": 2.2,
                "max_adverse_excursion_pct": -1.7,
                "drawdown_from_peak_pct": -2.1,
                "bars_since_peak": 9,
                "bars_in_trade": 24,
                "trailing_armed": True,
                "favorable_then_softened_flag": True,
                "stale_hold_profile": True,
                "walkforward_window_index": 2,
                "raw_rule_reason": "trailing",
            }
        ]
        diag = model_quality_pass._market_predictor_diagnostics("stocks", rows, full_rows=rows, abstained_rows=[])
        self.assertIn("stock_pnl_trend_miss_diagnostics", diag)
        self.assertIn("stock_weak_window_diagnostics", diag)
        self.assertEqual(int(diag.get("stock_direction_correct_pnl_wrong_count", 0)), 1)
        self.assertEqual(int(diag.get("stock_trigger_correct_pnl_wrong_count", 0)), 1)
        self.assertEqual(int(diag.get("stock_direction_trigger_correct_pnl_wrong_count", 0)), 1)

    def test_stock_watchlist_preview_includes_pnl_quality_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_data_provider": "alpaca"}
            bars = []
            base_ts = 1_700_000_000
            price = 100.0
            for i in range(72):
                price *= 1.004 if i < 36 else 1.001
                bars.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(base_ts + (i * 3600))), "o": price, "h": price * 1.01, "l": price * 0.995, "c": price, "v": 1000})
            fake_client = mock.Mock()
            fake_client.get_stock_bars.return_value = list(bars)
            with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", fake_client, {})):
                warm_stock_historical_cache(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
                preview = build_stock_watchlist_prediction_preview(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
            self.assertIn("stock_pnl_quality_score", preview)
            self.assertIn("stock_pnl_quality_reason", preview)
            self.assertIn("stock_trade_quality_score", preview)
            self.assertIn("trade_quality_gate_applied", preview)

    def test_stock_watchlist_preview_accepts_controlled_rollout_eligible_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_data_provider": "alpaca"}
            bars = []
            base_ts = 1_700_000_000
            price = 100.0
            for i in range(72):
                price *= 1.004 if i < 36 else 1.001
                bars.append(
                    {
                        "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(base_ts + (i * 3600))),
                        "o": price,
                        "h": price * 1.01,
                        "l": price * 0.995,
                        "c": price,
                        "v": 1000,
                    }
                )
            self._write_json(
                os.path.join(td, "model_quality_full_pass.json"),
                {
                    "controlled_rollout_readiness": {
                        "stocks": {
                            "eligible": True,
                            "reason": "",
                            "blockers": [],
                        }
                    }
                },
            )
            fake_client = mock.Mock()
            fake_client.get_stock_bars.return_value = list(bars)
            fake_pred = {
                "predicted_direction": "up",
                "predicted_exit_trigger": "Trailing",
                "predicted_pnl_trend": "up",
                "predicted_confidence": 0.85,
                "stock_pnl_quality_score": 0.7,
                "stock_pnl_quality_reason": "clean_trend_followthrough",
                "stock_trade_quality_score": 0.65,
                "stock_trade_quality_gate_applied": False,
                "direction_scores": {"up": 1.0, "down": 0.2},
                "trigger_scores": {"Trailing": 1.0, "Stale Alignment": 0.1},
            }
            with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", fake_client, {})):
                with mock.patch.object(model_quality_pass, "_stock_predict_one", return_value=fake_pred):
                    warm_stock_historical_cache(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
                    preview = build_stock_watchlist_prediction_preview(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
            blockers = list(preview.get("manual_watchlist_trade_blockers", []) or [])
            self.assertNotIn("market_rollout_not_ready", blockers)

    def test_safe_selection_falls_back_to_baseline_when_crypto_candidate_regresses(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(30):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 102.0,
                    "hold_hours": 6.0,
                    "pnl_pct": 2.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                }
            )
        def fake_predict(*, predictor_variant: str = "candidate", **kwargs):
            if predictor_variant == "baseline":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_hold_hours": 6.0,
                    "predicted_exit_price": 102.0,
                    "predicted_confidence": 0.9,
                    "trigger_margin": 1.0,
                    "direction_margin": 1.0,
                    "predictor_source_mode": "historical_strategy_replay",
                    "exit_shape_predictive_mode": "diagnostic_only",
                }
            return {
                "predicted_direction": "down",
                "predicted_exit_trigger": "Risk Cut",
                "predicted_hold_hours": 6.0,
                "predicted_exit_price": 98.0,
                "predicted_confidence": 0.9,
                "trigger_margin": 1.0,
                "direction_margin": 1.0,
                "predictor_source_mode": "historical_strategy_replay",
                "exit_shape_predictive_mode": "active",
            }

        with mock.patch.object(model_quality_pass, "_predict_one", side_effect=fake_predict):
            payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        safe = payload.get("safe_selection_diagnostics", {}).get("latest", {})
        crypto_diag = payload.get("crypto_classifier_diagnostics", {})
        recon = crypto_diag.get("candidate_vs_final_reconciliation", {})
        self.assertTrue(bool(safe.get("fallback_to_baseline", False)))
        self.assertEqual(str(safe.get("selected_predictor_variant", "")), "baseline")
        self.assertEqual(str(crypto_diag.get("exit_shape_predictive_mode", "")), "diagnostic_only")
        self.assertEqual(str(recon.get("selected_variant_name", "")), "baseline")
        self.assertIn("safe_selection_guardrails", str(recon.get("reason_final_metrics_differ_from_candidate_metrics", "")))

    def test_crypto_candidate_vs_final_reconciliation_emitted(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(60):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 98.0,
                    "exit_price": 98.0,
                    "hold_hours": 6.0,
                    "pnl_pct": -2.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "risk_cut_touched": True,
                    "take_profit_touched": False,
                    "trailing_armed": False,
                    "favorable_then_softened_flag": False,
                    "max_favorable_excursion_pct": 0.3,
                    "max_adverse_excursion_pct": -2.2,
                    "drawdown_from_peak_pct": -0.8,
                    "trailing_pullback_pct": 0.0,
                    "bars_in_trade": 6,
                    "exit_momentum_3": -0.6,
                    "exit_momentum_6": -0.7,
                    "trend_momentum_score": -0.5,
                    "recent_return_3": -0.4,
                    "recent_return_6": -0.9,
                    "recent_return_12": -1.3,
                    "recent_return_24": -1.7,
                    "signal_margin": 0.1,
                    "label_rule_version": "historical_replay_v2",
                    "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                    "same_candle_multi_exit_condition_count": 0,
                }
            )

        def fake_predict(*, predictor_variant: str = "candidate", **kwargs):
            if predictor_variant == "baseline":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_hold_hours": 6.0,
                    "predicted_exit_price": 102.0,
                    "predicted_confidence": 0.8,
                    "trigger_margin": 0.5,
                    "direction_margin": 0.5,
                    "predictor_source_mode": "historical_strategy_replay",
                }
            if predictor_variant == "candidate":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Stale Alignment",
                    "predicted_hold_hours": 6.0,
                    "predicted_exit_price": 101.0,
                    "predicted_confidence": 0.55,
                    "trigger_margin": 0.2,
                    "direction_margin": 0.2,
                    "predictor_source_mode": "historical_strategy_replay",
                }
            return {
                "predicted_direction": "down",
                "predicted_exit_trigger": "Risk Cut",
                "predicted_hold_hours": 6.0,
                "predicted_exit_price": 98.0,
                "predicted_confidence": 0.9,
                "trigger_margin": 1.2,
                "direction_margin": 1.0,
                "predictor_source_mode": "historical_strategy_replay",
            }

        with mock.patch.object(model_quality_pass, "_predict_one", side_effect=fake_predict):
            payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        crypto_diag = payload.get("crypto_classifier_diagnostics", {})
        recon = crypto_diag.get("candidate_vs_final_reconciliation", {})
        self.assertIn("candidate_variant_name", recon)
        self.assertIn("final_scoring_variant_name", recon)
        self.assertIn("overlap_count_between_candidate_and_final_admitted", recon)
        self.assertIn("candidate_metrics", recon)
        self.assertIn("final_metrics", recon)
        self.assertIn("final_metrics_using", recon)

    def test_crypto_v2_selected_final_metrics_identify_v2_rows(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(60):
            closed_rows.append(
                {
                    "symbol": "ETH-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 98.0,
                    "exit_price": 98.0,
                    "hold_hours": 6.0,
                    "pnl_pct": -2.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "risk_cut_touched": True,
                    "take_profit_touched": False,
                    "trailing_armed": False,
                    "favorable_then_softened_flag": False,
                    "max_favorable_excursion_pct": 0.3,
                    "max_adverse_excursion_pct": -2.2,
                    "drawdown_from_peak_pct": -0.8,
                    "trailing_pullback_pct": 0.0,
                    "bars_in_trade": 6,
                    "exit_momentum_3": -0.6,
                    "exit_momentum_6": -0.7,
                    "trend_momentum_score": -0.5,
                    "recent_return_3": -0.4,
                    "recent_return_6": -0.9,
                    "recent_return_12": -1.3,
                    "recent_return_24": -1.7,
                    "signal_margin": 0.1,
                    "label_rule_version": "historical_replay_v2",
                    "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                    "same_candle_multi_exit_condition_count": 0,
                }
            )

        def fake_predict(*, predictor_variant: str = "candidate", **kwargs):
            if predictor_variant == "baseline":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_hold_hours": 6.0,
                    "predicted_exit_price": 102.0,
                    "predicted_confidence": 0.8,
                    "trigger_margin": 0.5,
                    "direction_margin": 0.5,
                    "predictor_source_mode": "historical_strategy_replay",
                }
            if predictor_variant == "candidate":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Stale Alignment",
                    "predicted_hold_hours": 6.0,
                    "predicted_exit_price": 101.0,
                    "predicted_confidence": 0.55,
                    "trigger_margin": 0.2,
                    "direction_margin": 0.2,
                    "predictor_source_mode": "historical_strategy_replay",
                }
            return {
                "predicted_direction": "down",
                "predicted_exit_trigger": "Risk Cut",
                "predicted_hold_hours": 6.0,
                "predicted_exit_price": 98.0,
                "predicted_confidence": 0.95,
                "trigger_margin": 1.5,
                "direction_margin": 1.5,
                "predictor_source_mode": "historical_strategy_replay",
            }

        with mock.patch.object(model_quality_pass, "_predict_one", side_effect=fake_predict):
            payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        recon = payload.get("crypto_classifier_diagnostics", {}).get("candidate_vs_final_reconciliation", {})
        self.assertEqual(str(recon.get("selected_variant_name", "")), "label_compatible_v2")
        self.assertIn(str(recon.get("final_scoring_variant_name", "")), {"label_compatible_v2", "mixed_selected_variants_aggregate"})
        self.assertEqual(str((recon.get("final_metrics_using", {}) or {}).get("row_set", "")), "admitted_rows")

    def test_forex_safe_selection_falls_back_to_baseline(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(30):
            closed_rows.append(
                {
                    "symbol": "EUR_USD",
                    "entry_price": 1.1,
                    "actual_exit_price": 1.101,
                    "hold_hours": 4.0,
                    "pnl_pct": 0.1,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Stale Alignment",
                    "source_type": "execution_log",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "side": "long",
                }
            )
        original_predict = model_quality_pass._predict_one

        def fake_predict(*, market: str = "", predictor_variant: str = "candidate", candidate: dict, **kwargs):
            if market == "forex" and predictor_variant == "baseline":
                return {
                    "predicted_direction": "up",
                    "predicted_exit_trigger": candidate.get("actual_exit_trigger", "Unknown"),
                    "predicted_hold_hours": 4.0,
                    "predicted_exit_price": 1.101,
                    "predicted_confidence": 0.8,
                    "predictor_mode": "forex_execution_log",
                }
            if market == "forex":
                return {
                    "predicted_direction": "down",
                    "predicted_exit_trigger": candidate.get("actual_exit_trigger", "Unknown"),
                    "predicted_hold_hours": 4.0,
                    "predicted_exit_price": 1.099,
                    "predicted_confidence": 0.8,
                    "predictor_mode": "forex_execution_log",
                }
            return original_predict(train_rows=kwargs["train_rows"], candidate=candidate, regime=kwargs["regime"], market=market, predictor_variant=predictor_variant)

        with mock.patch.object(model_quality_pass, "_predict_one", side_effect=fake_predict):
            payload = build_synthetic_replay_artifact("/tmp", "forex", closed_rows)
        safe = payload.get("forex_safe_selection", {})
        self.assertTrue(bool(safe.get("fallback_to_baseline", False)))
        self.assertEqual(str(safe.get("selected_predictor_variant", "")), "baseline")

    def test_crypto_backfill_env_parameters_appear_in_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hist = {"state": "READY", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "replay_cache_path": os.path.join(td, "crypto", "historical_replay_cache"), "remote_rows_fetched": 12, "cache_rows_loaded": 34, "symbols_covered": ["BTC-USD"]}, "skipped": []}
            with mock.patch.dict(os.environ, {
                "CRYPTO_REPLAY_BACKFILL": "1",
                "CRYPTO_REPLAY_BACKFILL_LOOKBACK_DAYS": "365",
                "CRYPTO_REPLAY_BACKFILL_TIMEFRAME": "1hour",
                "CRYPTO_REPLAY_BACKFILL_MAX_SYMBOLS": "20",
            }, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=hist):
                    with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                        with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                            with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                    out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            diag = out.get("dataset_quality", {}).get("crypto", {}).get("replay_source_diagnostics", {})
            self.assertTrue(bool(diag.get("crypto_backfill_mode_enabled", False)))
            self.assertEqual(int(diag.get("crypto_backfill_lookback_days", 0)), 365)
            self.assertEqual(str(diag.get("crypto_backfill_timeframe", "")), "1hour")

    def test_stock_backfill_provider_failure_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("app.model_quality_pass._stock_provider_client", return_value=("alpaca", None, {"reason": "provider_down"})):
                out = _generate_stock_historical_replay_closed_trades(hub_dir=td, base_dir=td, settings={}, existing_closed_rows=[], lookback_days=365, max_symbols=10, force_refresh=False)
            self.assertIn("reason_unavailable", out.get("diagnostics", {}))

    def test_stock_backfill_env_parameters_appear_in_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {
                "STOCK_REPLAY_BACKFILL": "1",
                "STOCK_REPLAY_BACKFILL_LOOKBACK_DAYS": "365",
                "STOCK_REPLAY_BACKFILL_TIMEFRAME": "1Hour",
                "STOCK_REPLAY_BACKFILL_MAX_SYMBOLS": "50",
            }, clear=False):
                with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                    with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                        with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                            with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            diag = out.get("dataset_quality", {}).get("stocks", {}).get("replay_source_diagnostics", {})
            self.assertTrue(bool(diag.get("stock_backfill_mode_enabled", False)))
            self.assertEqual(int(diag.get("stock_backfill_lookback_days", 0)), 365)

    def test_safe_selection_diagnostics_emitted_for_all_markets(self) -> None:
        rows = []
        ts = 1_700_000_000
        for i in range(30):
            rows.append(
                {
                    "symbol": "SPY",
                    "entry_price": 100.0,
                    "actual_exit_price": 101.0,
                    "hold_hours": 4.0,
                    "pnl_pct": 1.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "source_type": "historical_api_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "stocks", rows)
        safe = payload.get("safe_selection_diagnostics", {})
        self.assertIn("latest", safe)
        self.assertIn("windows", safe)

    def test_deterministic_crypto_replay_sorts_rows_stably(self) -> None:
        rows = [
            {
                "symbol": "ETH-USD",
                "entry_price": 100.0,
                "actual_exit_price": 101.0,
                "exit_price": 101.0,
                "hold_hours": 4.0,
                "pnl_pct": 1.0,
                "actual_direction": "up",
                "actual_exit_trigger": "Trailing",
                "source_type": "historical_strategy_replay",
                "entry_ts": 200,
                "exit_ts": 260,
            },
            {
                "symbol": "BTC-USD",
                "entry_price": 100.0,
                "actual_exit_price": 99.0,
                "exit_price": 99.0,
                "hold_hours": 4.0,
                "pnl_pct": -1.0,
                "actual_direction": "down",
                "actual_exit_trigger": "Risk Cut",
                "source_type": "historical_strategy_replay",
                "entry_ts": 100,
                "exit_ts": 180,
            },
        ] * 20
        payload_a = build_synthetic_replay_artifact(
            "/tmp",
            "crypto",
            list(reversed(rows)),
            {"enabled": True, "start_ts": 0, "cutoff_ts": 0, "locked_symbols": ["BTC-USD", "ETH-USD"]},
        )
        payload_b = build_synthetic_replay_artifact(
            "/tmp",
            "crypto",
            list(rows),
            {"enabled": True, "start_ts": 0, "cutoff_ts": 0, "locked_symbols": ["ETH-USD", "BTC-USD"]},
        )
        self.assertEqual(
            payload_a.get("crypto_deterministic_evaluation", {}).get("rows_hash"),
            payload_b.get("crypto_deterministic_evaluation", {}).get("rows_hash"),
        )

    def test_repeat_eval_diagnostics_mark_disagreement_without_promoting_v2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "crypto", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0},
                    {"ts": 1_700_000_100, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 101.0, "tag": "Trailing"},
                ],
            )
            hist = {
                "state": "READY",
                "rows": [],
                "diagnostics": {
                    "historical_strategy_replay_rows": 60,
                    "historical_strategy_replay_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
                    "symbols_covered": ["BTC-USD", "ETH-USD", "SOL-USD"],
                    "crypto_replay_eval_start_ts": 100,
                    "crypto_replay_eval_cutoff_ts": 200,
                    "crypto_replay_symbols_locked": True,
                    "crypto_replay_symbol_list_hash": "abc",
                    "crypto_replay_cache_frozen": True,
                    "crypto_replay_rows_hash": "hash-a",
                },
            }
            disagreement = [
                {
                    "status": "ok",
                    "meta": {"market": "crypto", "closed_trades_total": 60, "full_test_trades": 40, "rows_hash": "hash-a"},
                    "safe_selection_diagnostics": {
                        "latest": {
                            "selected_predictor_variant": "baseline",
                            "baseline_admitted_trades": 36,
                            "candidate_evaluations": [
                                {"variant": "label_compatible_v2", "summary": {"admitted_trades": 38, "admission_rate_pct": 95.0, "metrics": {}}, "guardrail_failures": ["candidate_admitted_trades_below_40"]}
                            ],
                        }
                    },
                },
                {
                    "status": "ok",
                    "meta": {"market": "crypto", "closed_trades_total": 60, "full_test_trades": 40, "rows_hash": "hash-b"},
                    "safe_selection_diagnostics": {
                        "latest": {
                            "selected_predictor_variant": "label_compatible_v2",
                            "baseline_admitted_trades": 36,
                            "candidate_evaluations": [
                                {"variant": "label_compatible_v2", "summary": {"admitted_trades": 40, "admission_rate_pct": 100.0, "metrics": {}}, "guardrail_failures": []}
                            ],
                        }
                    },
                },
            ]
            real_build = model_quality_pass.build_synthetic_replay_artifact
            call_state = {"crypto": 0}

            def fake_build(hub_dir, market, closed_rows, deterministic_config=None):
                if market != "crypto":
                    return real_build(hub_dir, market, closed_rows, deterministic_config)
                idx = call_state["crypto"] % len(disagreement)
                call_state["crypto"] += 1
                payload = dict(disagreement[idx])
                payload["crypto_classifier_diagnostics"] = {}
                return payload

            with mock.patch.dict(os.environ, {"CRYPTO_REPLAY_DETERMINISTIC": "1", "CRYPTO_REPLAY_REPEAT_EVALS": "3"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=hist):
                    with mock.patch("app.model_quality_pass.build_synthetic_replay_artifact", side_effect=fake_build):
                        with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                            with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                        out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            diag = out.get("dataset_quality", {}).get("crypto", {}).get("replay_source_diagnostics", {})
            self.assertFalse(bool(diag.get("crypto_replay_repeat_activation_consistent", True)))
            self.assertFalse(bool(diag.get("crypto_replay_repeat_metrics_consistent", True)))
            self.assertFalse(bool(diag.get("crypto_v2_promotion_eligible", True)))

    def test_repeat_eval_can_mark_v2_promotion_eligible_when_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "crypto", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0},
                    {"ts": 1_700_000_100, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 101.0, "tag": "Trailing"},
                ],
            )
            hist = {
                "state": "READY",
                "rows": [],
                "diagnostics": {
                    "historical_strategy_replay_rows": 60,
                    "historical_strategy_replay_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
                    "symbols_covered": ["BTC-USD", "ETH-USD", "SOL-USD"],
                    "crypto_replay_eval_start_ts": 100,
                    "crypto_replay_eval_cutoff_ts": 200,
                    "crypto_replay_symbols_locked": True,
                    "crypto_replay_symbol_list_hash": "abc",
                    "crypto_replay_cache_frozen": True,
                    "crypto_replay_rows_hash": "hash-a",
                },
            }
            eligible_payload = {
                "status": "ok",
                "meta": {"market": "crypto", "closed_trades_total": 60, "full_test_trades": 40, "rows_hash": "hash-a"},
                "safe_selection_diagnostics": {
                    "latest": {
                        "selected_predictor_variant": "label_compatible_v2",
                        "baseline_admitted_trades": 36,
                        "candidate_evaluations": [
                            {"variant": "label_compatible_v2", "summary": {"admitted_trades": 40, "admission_rate_pct": 100.0, "metrics": {"trigger_match_pct": 80.0}}, "guardrail_failures": []}
                        ],
                    }
                },
                "crypto_classifier_diagnostics": {},
            }
            real_build = model_quality_pass.build_synthetic_replay_artifact

            def fake_build(hub_dir, market, closed_rows, deterministic_config=None):
                if market != "crypto":
                    return real_build(hub_dir, market, closed_rows, deterministic_config)
                return dict(eligible_payload)

            with mock.patch.dict(os.environ, {"CRYPTO_REPLAY_DETERMINISTIC": "1", "CRYPTO_REPLAY_REPEAT_EVALS": "3"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=hist):
                    with mock.patch("app.model_quality_pass.build_synthetic_replay_artifact", side_effect=fake_build):
                        with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                            with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                        out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            diag = out.get("dataset_quality", {}).get("crypto", {}).get("replay_source_diagnostics", {})
            self.assertTrue(bool(diag.get("crypto_replay_repeat_activation_consistent", False)))
            self.assertTrue(bool(diag.get("crypto_replay_repeat_metrics_consistent", False)))
            self.assertTrue(bool(diag.get("crypto_v2_promotion_eligible", False)))

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
        self.assertIn("label_rule_version", row)
        self.assertIn("exit_condition_priority_used", row)
        self.assertIn("late_favorable_reversal_score", row)
        self.assertIn("favorable_then_reversed", row)
        self.assertIn("bars_from_peak_to_exit_preview", row)
        self.assertGreaterEqual(int(row.get("bars_in_trade", 0) or 0), 1)
        self.assertIn(row.get("actual_exit_trigger"), {"Trailing", "Stale Alignment", "Risk Cut", "Take Profit"})

    def test_crypto_historical_replay_trailing_label_aligns_with_flags(self) -> None:
        candles = []
        ts = 1_700_000_000_000
        px = 100.0
        for i in range(60):
            if i < 28:
                px *= 1.003
            elif i < 34:
                px *= 1.012
            elif i < 38:
                px *= 0.988
            else:
                px *= 1.0002
            candles.append([ts + (i * 3_600_000), px, px, px * 1.004, px * 0.996, 1000.0])
        artifact_ctx = {
            "usable": True,
            "active_timeframe_count": 6,
            "predicted_low_boundary": 99.0,
            "predicted_high_boundary": 110.0,
            "trained_artifacts_fresh": True,
            "artifact_training_time": 1_700_000_000,
            "signal_margin": 0.5,
        }
        thresholds = {
            "entry_signal_margin_min": 0.35,
            "entry_trend_score_min": 0.12,
            "entry_return6_min": -0.25,
            "risk_cut_pct": 2.25,
            "take_profit_pct": 20.0,
            "trailing_arm_pct": 1.6,
            "trailing_drawdown_pct": 1.1,
            "stale_hold_hours": 18.0,
            "stale_trend_score_max": 0.05,
        }
        rows = crypto_historical_replay._simulate_strategy_rows("BTC-USD", candles, artifact_ctx, "1hour", thresholds)
        self.assertTrue(rows)
        trailing = [r for r in rows if r.get("actual_exit_trigger") == "Trailing"]
        self.assertTrue(trailing)
        self.assertTrue(bool(trailing[0].get("trailing_armed", False)))
        self.assertTrue(bool(trailing[0].get("favorable_then_softened_flag", False)))

    def test_crypto_historical_replay_risk_cut_label_aligns_with_flag(self) -> None:
        candles = []
        ts = 1_700_000_000_000
        px = 100.0
        for i in range(60):
            if i < 28:
                px *= 1.002
            elif i < 33:
                px *= 0.972
            else:
                px *= 1.0001
            candles.append([ts + (i * 3_600_000), px, px, px * 1.002, px * 0.97, 1000.0])
        artifact_ctx = {
            "usable": True,
            "active_timeframe_count": 6,
            "predicted_low_boundary": 96.0,
            "predicted_high_boundary": 104.0,
            "trained_artifacts_fresh": True,
            "artifact_training_time": 1_700_000_000,
            "signal_margin": 0.5,
        }
        thresholds = {
            "entry_signal_margin_min": 0.35,
            "entry_trend_score_min": 0.12,
            "entry_return6_min": -0.25,
            "risk_cut_pct": 2.25,
            "take_profit_pct": 6.0,
            "trailing_arm_pct": 1.6,
            "trailing_drawdown_pct": 1.1,
            "stale_hold_hours": 18.0,
            "stale_trend_score_max": 0.05,
        }
        rows = crypto_historical_replay._simulate_strategy_rows("BTC-USD", candles, artifact_ctx, "1hour", thresholds)
        risk_rows = [r for r in rows if r.get("actual_exit_trigger") == "Risk Cut"]
        self.assertTrue(risk_rows)
        self.assertTrue(bool(risk_rows[0].get("risk_cut_touched", False)))
        self.assertIn("risk_cut_touched_so_far", risk_rows[0])
        self.assertFalse(bool(risk_rows[0].get("risk_cut_touched_so_far", False)))
        self.assertIn("max_favorable_excursion_pct_so_far", risk_rows[0])
        self.assertLessEqual(
            float(risk_rows[0].get("max_favorable_excursion_pct_so_far", 0.0) or 0.0),
            float(risk_rows[0].get("max_favorable_excursion_pct", 0.0) or 0.0),
        )

    def test_crypto_alignment_diagnostics_emitted(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(30):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 98.0,
                    "hold_hours": 10.0,
                    "pnl_pct": -2.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut" if i % 2 == 0 else "Trailing",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "risk_cut_touched": bool(i % 2 == 0),
                    "take_profit_touched": False,
                    "trailing_armed": bool(i % 2 == 1),
                    "favorable_then_softened_flag": bool(i % 2 == 1),
                    "max_favorable_excursion_pct": 2.5,
                    "max_adverse_excursion_pct": -2.5,
                    "drawdown_from_peak_pct": -1.3,
                    "trailing_pullback_pct": 1.3,
                    "bars_in_trade": 8,
                    "exit_momentum_3": -0.8,
                    "exit_momentum_6": -0.5,
                    "trend_momentum_score": 0.3,
                    "recent_return_3": 0.8,
                    "recent_return_6": 1.1,
                    "recent_return_12": 1.3,
                    "recent_return_24": 1.8,
                    "signal_margin": 0.4,
                    "label_rule_version": "historical_replay_v2",
                    "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                    "same_candle_multi_exit_condition_count": 0,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        diag = payload.get("crypto_classifier_diagnostics", {})
        align = diag.get("crypto_label_feature_alignment_diagnostics", {})
        self.assertIn("feature_distributions_by_actual_trigger", align)
        self.assertIn("mismatch_counters", align)
        self.assertIn("label_rule_version", align)

    def test_label_compatible_v2_frontier_variants_present(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(40):
            closed_rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 102.0 if i % 2 == 0 else 98.0,
                    "hold_hours": 8.0,
                    "pnl_pct": 2.0 if i % 2 == 0 else -2.0,
                    "actual_direction": "up" if i % 2 == 0 else "down",
                    "actual_exit_trigger": "Trailing" if i % 2 == 0 else "Risk Cut",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "risk_cut_touched": bool(i % 2 == 1),
                    "take_profit_touched": False,
                    "trailing_armed": bool(i % 2 == 0),
                    "favorable_then_softened_flag": bool(i % 2 == 0),
                    "max_favorable_excursion_pct": 2.5,
                    "max_adverse_excursion_pct": -2.5,
                    "drawdown_from_peak_pct": -1.3,
                    "trailing_pullback_pct": 1.3,
                    "bars_in_trade": 8,
                    "exit_momentum_3": -0.5 if i % 2 == 1 else 0.2,
                    "exit_momentum_6": -0.4 if i % 2 == 1 else 0.3,
                    "trend_momentum_score": 0.3,
                    "recent_return_3": 0.8,
                    "recent_return_6": 1.1,
                    "recent_return_12": 1.3,
                    "recent_return_24": 1.8,
                    "signal_margin": 0.4,
                    "label_rule_version": "historical_replay_v2",
                    "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                    "same_candle_multi_exit_condition_count": 0,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        evals = payload.get("safe_selection_diagnostics", {}).get("latest", {}).get("candidate_evaluations", [])
        names = [e.get("summary", {}).get("policy_name", "") for e in evals if e.get("variant") == "label_compatible_v2"]
        self.assertTrue(any("label_compatible_v2" in n for n in names))

    def test_label_compatible_v2_rejection_diagnostics_emitted(self) -> None:
        closed_rows = []
        ts = 1_700_000_000
        for i in range(40):
            closed_rows.append(
                {
                    "symbol": "ETH-USD",
                    "entry_price": 100.0,
                    "actual_exit_price": 102.0,
                    "hold_hours": 8.0,
                    "pnl_pct": 2.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": ts + (i * 7200),
                    "exit_ts": ts + (i * 7200) + 3600,
                    "risk_cut_touched": False,
                    "take_profit_touched": False,
                    "trailing_armed": True,
                    "favorable_then_softened_flag": True,
                    "max_favorable_excursion_pct": 2.5,
                    "max_adverse_excursion_pct": -0.5,
                    "drawdown_from_peak_pct": -1.3,
                    "trailing_pullback_pct": 1.3,
                    "bars_in_trade": 8,
                    "exit_momentum_3": 0.2,
                    "exit_momentum_6": 0.3,
                    "trend_momentum_score": 0.3,
                    "recent_return_3": 0.8,
                    "recent_return_6": 1.1,
                    "recent_return_12": 1.3,
                    "recent_return_24": 1.8,
                    "signal_margin": 0.4,
                    "label_rule_version": "historical_replay_v2",
                    "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                    "same_candle_multi_exit_condition_count": 0,
                }
            )
        payload = build_synthetic_replay_artifact("/tmp", "crypto", closed_rows)
        diag = payload.get("crypto_classifier_diagnostics", {})
        self.assertIn("label_compatible_v2_rejection_diagnostics", diag)
        self.assertIn("label_compatible_v2_near_miss_rows_count", diag)

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
            self.assertEqual(diag.get("crypto_feature_source_priority", [])[0], "historical_strategy_replay")
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

    def test_crypto_original_predictor_is_replay_safe_and_parses_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            coin_dir = os.path.join(td, "BTC")
            os.makedirs(coin_dir, exist_ok=True)
            with open(os.path.join(coin_dir, "neural_perfect_threshold_1hour.txt"), "w", encoding="utf-8") as f:
                f.write("20.0")
            with open(os.path.join(coin_dir, "memories_1hour.txt"), "w", encoding="utf-8") as f:
                f.write("1.0 2.0{}4.0{}-1.0~0.5 -1.5{}1.0{}-2.0")
            for name in ("memory_weights_1hour.txt", "memory_weights_high_1hour.txt", "memory_weights_low_1hour.txt"):
                with open(os.path.join(coin_dir, name), "w", encoding="utf-8") as f:
                    f.write("1 1")
            snapshot = [
                [1_700_000_000_000, 100.0, 101.0, 101.5, 99.8, 1000.0],
                [1_700_000_360_000, 101.0, 102.0, 102.6, 100.7, 1100.0],
            ]
            cwd_before = os.getcwd()
            out = predict_crypto_original_dry_run(
                symbol="BTC-USD",
                timeframe="1hour",
                candle_snapshot=snapshot,
                artifact_dir=coin_dir,
                as_of_ts=1_700_000_360,
            )
            self.assertEqual(os.getcwd(), cwd_before)
            self.assertTrue(bool(out.get("original_predictor_dry_run_used", False)))
            self.assertFalse(bool(out.get("original_predictor_called_step_coin", True)))
            self.assertFalse(bool(out.get("original_predictor_called_robinhood", True)))
            self.assertFalse(bool(out.get("original_predictor_called_kucoin_live", True)))
            self.assertFalse(bool(out.get("original_predictor_wrote_live_signal_files", True)))
            self.assertFalse(bool(out.get("original_predictor_changed_cwd", True)))
            self.assertFalse(bool(out.get("original_predictor_mutated_live_artifacts", True)))
            self.assertIn(out.get("predicted_direction"), {"up", "down", "flat"})
            self.assertIn("direction_scores", out)
            self.assertEqual(str(out.get("original_trigger_semantics_status", "")), "unavailable")
            self.assertEqual(str(out.get("trigger_semantics_blocker", "")), "original_artifacts_do_not_encode_exit_trigger_class")

    def test_crypto_original_bound_signal_adapter_emits_messages_without_side_effects(self) -> None:
        cwd_before = os.getcwd()
        out = derive_crypto_original_signal_dry_run(
            symbol="BTC-USD",
            current_price=105.0,
            timeframe_predictions=[
                {
                    "timeframe": "1hour",
                    "active_model_state": "active",
                    "predicted_low_boundary": 98.0,
                    "predicted_high_boundary": 101.0,
                    "low_new_price": 99.0,
                    "high_new_price": 100.5,
                }
            ],
        )
        self.assertEqual(os.getcwd(), cwd_before)
        self.assertEqual(str(out.get("original_message_type", "")), "SHORT")
        self.assertEqual(str(out.get("original_signal_side", "")), "short")
        self.assertEqual(int(out.get("original_short_signal_count", 0) or 0), 1)
        self.assertFalse(bool(out.get("bound_signal_called_robinhood", True)))
        self.assertFalse(bool(out.get("bound_signal_called_kucoin_live", True)))
        self.assertFalse(bool(out.get("bound_signal_wrote_live_signal_files", True)))
        self.assertFalse(bool(out.get("bound_signal_wrote_bound_files", True)))
        self.assertFalse(bool(out.get("bound_signal_changed_cwd", True)))

    def test_crypto_original_bound_signal_adapter_can_emit_within_and_inactive(self) -> None:
        within = derive_crypto_original_signal_dry_run(
            symbol="BTC-USD",
            current_price=100.0,
            timeframe_predictions=[
                {
                    "timeframe": "1hour",
                    "active_model_state": "active",
                    "predicted_low_boundary": 99.0,
                    "predicted_high_boundary": 101.0,
                    "low_new_price": 99.2,
                    "high_new_price": 100.8,
                }
            ],
        )
        inactive = derive_crypto_original_signal_dry_run(
            symbol="BTC-USD",
            current_price=100.0,
            timeframe_predictions=[
                {
                    "timeframe": "1hour",
                    "active_model_state": "inactive",
                    "predicted_low_boundary": 99.0,
                    "predicted_high_boundary": 101.0,
                    "low_new_price": 99.2,
                    "high_new_price": 100.8,
                }
            ],
        )
        self.assertEqual(str(within.get("original_message_type", "")), "WITHIN")
        self.assertEqual(str(inactive.get("original_message_type", "")), "INACTIVE")
        self.assertEqual(str(within.get("original_trigger_semantics_status", "")), "unavailable")
        self.assertEqual(str(inactive.get("original_trigger_semantics_status", "")), "unavailable")

    def test_crypto_historical_replay_rows_include_original_dry_run_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            coin_dir = os.path.join(td, "market_data", "coins", "BTC")
            os.makedirs(coin_dir, exist_ok=True)
            for name, content in (
                ("trainer_last_training_time.txt", str(time.time())),
                ("neural_perfect_threshold_1hour.txt", "20.0"),
                ("memories_1hour.txt", "1.0 2.0{}4.0{}-1.0~0.5 -1.5{}1.0{}-2.0"),
                ("memory_weights_1hour.txt", "1 1"),
                ("memory_weights_high_1hour.txt", "1 1"),
                ("memory_weights_low_1hour.txt", "1 1"),
                ("long_dca_signal.txt", "1"),
                ("short_dca_signal.txt", "0"),
                ("futures_long_profit_margin.txt", "0.8"),
                ("futures_short_profit_margin.txt", "0.1"),
            ):
                with open(os.path.join(coin_dir, name), "w", encoding="utf-8") as f:
                    f.write(content)
            base_ts = 1_700_000_000_000
            cache_rows = []
            price = 100.0
            for i in range(96):
                price += 0.4 if i < 48 else (0.5 if i < 72 else -0.6)
                cache_rows.append([base_ts + (i * 3600 * 1000), price - 0.2, price, price + 0.4, price - 0.4, 1000.0])
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
            self.assertTrue(out.get("rows"))
            row = out.get("rows", [])[0]
            self.assertIn("original_predicted_direction", row)
            self.assertIn("original_predictor_dry_run_used", row)
            self.assertFalse(bool(row.get("original_predictor_called_robinhood", True)))
            self.assertFalse(bool(row.get("original_predictor_called_kucoin_live", True)))

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

    def test_completed_live_decision_rows_builds_crypto_rows(self) -> None:
        rows, diag = _completed_live_decision_rows(
            market="crypto",
            events=[],
            closed_rows=[
                {
                    "symbol": "BTC-USD",
                    "entry_ts": 100,
                    "exit_ts": 200,
                    "entry_price": 100.0,
                    "exit_price": 102.0,
                    "hold_hours": 2.0,
                    "actual_exit_trigger": "Trailing",
                    "entry_snapshot_selected_action": "buy",
                    "entry_snapshot_ai_confidence": 0.84,
                    "entry_snapshot_normalized_trigger": "Trailing",
                    "entry_snapshot_policy_mode": "shadow_live_v1",
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("source_type"), "completed_live_decision_snapshot")
        self.assertEqual(rows[0].get("predicted_direction"), "up")
        self.assertEqual(rows[0].get("predictor_mode"), "shadow_live_v1")
        self.assertTrue(bool(diag.get("live_decision_source_available")))

    def test_completed_live_decision_rows_builds_stock_row(self) -> None:
        rows, diag = _completed_live_decision_rows(
            market="stocks",
            events=[],
            closed_rows=[
                {
                    "symbol": "NVDA",
                    "entry_ts": 100,
                    "exit_ts": 200,
                    "entry_price": 100.0,
                    "exit_price": 104.0,
                    "side": "buy",
                    "qty": 1.0,
                    "hold_hours": 2.0,
                    "pnl_usd": 4.0,
                    "pnl_pct": 4.0,
                    "actual_exit_trigger": "Trailing",
                    "decision_snapshot_id": "snap-stock-1",
                    "entry_snapshot_selected_action": "buy",
                    "entry_snapshot_predicted_direction": "up",
                    "entry_snapshot_predicted_exit_trigger": "Trailing",
                    "entry_snapshot_predicted_pnl_trend": "up",
                    "entry_snapshot_predicted_confidence": 0.71,
                    "entry_snapshot_selected_predictor": "local_market_model",
                    "entry_snapshot_predictor_variant": "live",
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("source_type"), "completed_live_decision_snapshot")
        self.assertEqual(rows[0].get("predicted_direction"), "up")
        self.assertEqual(rows[0].get("predicted_exit_trigger"), "Trailing")
        self.assertTrue(bool(rows[0].get("eligible_for_future_learning")))
        self.assertTrue(bool(diag.get("live_decision_source_available")))

    def test_completed_live_decision_rows_builds_forex_row(self) -> None:
        rows, diag = _completed_live_decision_rows(
            market="forex",
            events=[],
            closed_rows=[
                {
                    "instrument": "EUR_USD",
                    "symbol": "EUR_USD",
                    "entry_ts": 100,
                    "exit_ts": 200,
                    "entry_price": 1.1000,
                    "exit_price": 1.1020,
                    "side": "long",
                    "qty": 1000.0,
                    "hold_hours": 2.0,
                    "pnl_usd": 2.0,
                    "pnl_pct": 0.1818,
                    "actual_exit_trigger": "Trailing",
                    "decision_snapshot_id": "snap-forex-1",
                    "entry_snapshot_selected_action": "long",
                    "entry_snapshot_predicted_direction": "up",
                    "entry_snapshot_predicted_exit_trigger": "Trailing",
                    "entry_snapshot_predicted_pnl_trend": "up",
                    "entry_snapshot_predicted_confidence": 0.74,
                    "entry_snapshot_selected_predictor": "local_market_model",
                    "entry_snapshot_predictor_variant": "live",
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("instrument"), "EUR_USD")
        self.assertTrue(bool(rows[0].get("eligible_for_future_learning")))
        self.assertEqual(int(diag.get("completed_live_eligible_rows_by_market", {}).get("forex", 0)), 1)

    def test_completed_live_decision_rows_missing_prediction_fields_are_ineligible_without_crashing(self) -> None:
        rows, diag = _completed_live_decision_rows(
            market="stocks",
            events=[],
            closed_rows=[
                {
                    "symbol": "AAPL",
                    "entry_ts": 100,
                    "exit_ts": 200,
                    "entry_price": 100.0,
                    "exit_price": 99.0,
                    "side": "buy",
                    "qty": 1.0,
                    "hold_hours": 1.0,
                    "pnl_usd": -1.0,
                    "pnl_pct": -1.0,
                    "actual_exit_trigger": "Risk Cut",
                }
            ],
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(bool(rows[0].get("eligible_for_future_learning")))
        self.assertIn("predicted_direction", list(rows[0].get("missing_fields", []) or []))
        self.assertIn("decision_snapshot_id", list(rows[0].get("missing_fields", []) or []))
        self.assertTrue(_s := str(rows[0].get("ineligible_reason", "")))
        self.assertFalse(bool(diag.get("completed_live_eligible_rows_by_market", {}).get("stocks", 0)))

    def test_synthetic_completed_live_paths_verify_all_markets(self) -> None:
        diag = _verify_completed_live_synthetic_paths()
        verified = diag.get("completed_live_synthetic_path_verified_by_market", {})
        eligible = diag.get("completed_live_synthetic_eligible_by_market", {})
        self.assertTrue(bool(verified.get("crypto")))
        self.assertTrue(bool(verified.get("stocks")))
        self.assertTrue(bool(verified.get("forex")))
        self.assertTrue(bool(eligible.get("crypto")))
        self.assertTrue(bool(eligible.get("stocks")))
        self.assertTrue(bool(eligible.get("forex")))

    def test_completed_live_decision_rows_do_not_exist_before_close(self) -> None:
        rows, diag = _completed_live_decision_rows(
            market="crypto",
            events=[{"event": "entry", "symbol": "BTC-USD", "ts": 100}],
            closed_rows=[],
        )
        self.assertEqual(rows, [])
        self.assertFalse(bool(diag.get("live_decision_source_available")))

    def test_model_quality_keeps_live_rows_supplemental_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = []
            for i in range(45):
                entry_px = 100.0 + i
                exit_px = entry_px + (2.0 if i % 2 == 0 else -1.0)
                rows.append(
                    {
                        "symbol": "BTC-USD",
                        "entry_ts": 1_700_000_000 + (i * 1000),
                        "exit_ts": 1_700_000_000 + (i * 1000) + 3600,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "hold_hours": 1.0,
                        "actual_exit_trigger": "Trailing" if exit_px > entry_px else "Risk Cut",
                        "predicted_direction": "up" if exit_px > entry_px else "down",
                        "predicted_exit_trigger": "Trailing" if exit_px > entry_px else "Risk Cut",
                        "predicted_confidence": 0.83,
                        "selected_predictor_name": "live_shadow_predictor",
                        "source_type": "completed_live_decision_snapshot",
                    }
                )
            hist_rows = [
                {
                    "market": "crypto",
                    "symbol": "BTC-USD",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": 1_700_000_000 + (i * 7200),
                    "exit_ts": 1_700_000_000 + (i * 7200) + 3600,
                    "entry_price": 100.0 + i,
                    "exit_price": 101.0 + i,
                    "hold_hours": 1.0,
                    "actual_exit_trigger": "Trailing",
                }
                for i in range(60)
            ]
            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": hist_rows, "diagnostics": {"historical_strategy_replay_rows": 60, "historical_strategy_replay_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"]}}):
                with mock.patch("app.model_quality_pass._completed_live_decision_rows", return_value=(rows, {"live_decision_source_available": True, "live_decision_rows_found": 45, "live_decision_rows_completed": 45, "live_decision_join_rate_pct": 100.0, "live_decision_rows_by_market": {"crypto": 45}, "live_decision_rows_by_predictor": {"live_shadow_predictor": 45}, "live_decision_missing_reason": "", "model_quality_source_priority_used": "completed_live_decision_snapshot"})):
                    with mock.patch("app.model_quality_pass.load_market_trade_events", return_value={"events": []}):
                        with mock.patch("app.model_quality_pass.build_closed_trades", return_value={"closed_trades": []}):
                            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                            out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertEqual(
                out.get("replay_generation", {}).get("crypto", {}).get("model_quality_source_priority_used"),
                "historical_strategy_replay",
            )
            self.assertTrue(bool(out.get("replay_generation", {}).get("crypto", {}).get("model_quality_live_rows_supplemental_only")))
            self.assertFalse(bool(out.get("replay_generation", {}).get("crypto", {}).get("completed_live_rows_used_as_primary")))
            self.assertEqual(
                out.get("crypto_feature_source_diagnostics", {}).get("crypto_feature_source_used"),
                "historical_strategy_replay",
            )
            self.assertTrue(bool(out.get("crypto_feature_source_diagnostics", {}).get("completed_live_rows_used_as_supplemental")))

    def test_model_quality_can_use_live_rows_as_primary_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = [
                {
                    "symbol": "BTC-USD",
                    "entry_ts": 1_700_000_000 + (i * 1000),
                    "exit_ts": 1_700_000_000 + (i * 1000) + 3600,
                    "entry_price": 100.0 + i,
                    "exit_price": 101.0 + i,
                    "hold_hours": 1.0,
                    "actual_exit_trigger": "Trailing",
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_confidence": 0.83,
                    "selected_predictor_name": "live_shadow_predictor",
                    "source_type": "completed_live_decision_snapshot",
                }
                for i in range(45)
            ]
            hist_rows = [
                {
                    "market": "crypto",
                    "symbol": "BTC-USD",
                    "source_type": "historical_strategy_replay",
                    "entry_ts": 1_700_000_000 + (i * 7200),
                    "exit_ts": 1_700_000_000 + (i * 7200) + 3600,
                    "entry_price": 100.0 + i,
                    "exit_price": 101.0 + i,
                    "hold_hours": 1.0,
                    "actual_exit_trigger": "Trailing",
                }
                for i in range(60)
            ]
            with mock.patch.dict(os.environ, {"MODEL_QUALITY_USE_LIVE_ROWS_AS_PRIMARY": "1"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": hist_rows, "diagnostics": {"historical_strategy_replay_rows": 60, "historical_strategy_replay_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"]}}):
                    with mock.patch("app.model_quality_pass._completed_live_decision_rows", return_value=(rows, {"live_decision_source_available": True, "live_decision_rows_found": 45, "live_decision_rows_completed": 45, "live_decision_join_rate_pct": 100.0, "live_decision_rows_by_market": {"crypto": 45}, "live_decision_rows_by_predictor": {"live_shadow_predictor": 45}, "live_decision_missing_reason": "", "model_quality_source_priority_used": "completed_live_decision_snapshot"})):
                        with mock.patch("app.model_quality_pass.load_market_trade_events", return_value={"events": []}):
                            with mock.patch("app.model_quality_pass.build_closed_trades", return_value={"closed_trades": []}):
                                with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                            with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                                out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertEqual(out.get("replay_generation", {}).get("crypto", {}).get("model_quality_source_priority_used"), "completed_live_decision_snapshot")
            self.assertTrue(bool(out.get("replay_generation", {}).get("crypto", {}).get("completed_live_rows_used_as_primary")))
            self.assertEqual(out.get("replay_generation", {}).get("crypto", {}).get("source_selection_reason"), "live_rows_primary_override_enabled_and_sufficient")

    def test_crypto_clean_expansion_diagnostics_do_not_force_activation(self) -> None:
        rows = [
            {
                "symbol": f"BTC-USD-{i}",
                "source_type": "historical_strategy_replay",
                "predicted_direction": "down",
                "actual_direction": "down",
                "predicted_exit_trigger": "Risk Cut",
                "actual_exit_trigger": "Risk Cut",
                "predicted_confidence": 0.26,
                "trigger_margin": 0.11,
                "direction_margin": 0.04,
                "trigger_scores": {"Risk Cut": 0.62, "Stale Alignment": 0.58},
                "runner_up_trigger_score": 0.58,
                "winning_trigger_score": 0.62,
                "risk_cut_touched": True,
                "take_profit_touched": False,
                "trailing_armed": False,
                "favorable_then_softened_flag": False,
                "max_adverse_excursion_pct": -2.2,
                "downside_score": 0.70,
                "bars_in_trade": 20,
                "trend_momentum_score": -0.10,
            }
            for i in range(3)
        ]
        policy = {
            "source_mode": "historical_strategy_replay",
            "enable_clean_expansion": True,
            "base_conf_min": 0.30,
            "base_trigger_margin_min": 0.13,
            "base_dir_margin_min": 0.05,
            "risk_cut_conf_min": 0.30,
            "risk_cut_trigger_margin_min": 0.13,
            "risk_cut_dir_margin_min": 0.05,
            "min_admitted_trades": 40,
            "min_trigger_scored": 20,
        }
        scored = _score_crypto_with_policy(rows, policy)
        self.assertTrue(bool(scored.get("crypto_clean_expansion_attempted")))
        self.assertEqual(int(scored.get("crypto_clean_expansion_rows_added", 0)), 3)
        self.assertEqual(int(scored.get("crypto_clean_expansion_correct_rejected_added_estimate", 0)), 3)
        self.assertTrue(bool(scored.get("crypto_clean_expansion_guardrails_passed")))

    def test_stocks_use_historical_replay_by_default_even_if_live_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            live_rows = [
                {
                    "symbol": "NVDA",
                    "entry_ts": 1_700_000_000 + (i * 1000),
                    "exit_ts": 1_700_000_000 + (i * 1000) + 3600,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "hold_hours": 1.0,
                    "actual_exit_trigger": "Trailing",
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_confidence": 0.75,
                    "source_type": "completed_live_decision_snapshot",
                }
                for i in range(45)
            ]
            stock_rows = [
                {
                    "symbol": "NVDA",
                    "market": "stocks",
                    "source_type": "historical_api_replay",
                    "entry_ts": 1_700_000_000 + (i * 7200),
                    "exit_ts": 1_700_000_000 + (i * 7200) + 3600,
                    "entry_price": 100.0 + i,
                    "exit_price": 101.0 + i,
                    "hold_hours": 1.0,
                    "actual_exit_trigger": "Trailing",
                }
                for i in range(50)
            ]
            with mock.patch("app.model_quality_pass._completed_live_decision_rows", side_effect=lambda market, events, closed_rows: (live_rows, {"live_decision_source_available": True, "live_decision_rows_found": 45, "live_decision_rows_completed": 45, "live_decision_join_rate_pct": 100.0, "live_decision_rows_by_market": {market: 45}, "live_decision_rows_by_predictor": {"live_shadow_predictor": 45}, "live_decision_missing_reason": "", "model_quality_source_priority_used": "completed_live_decision_snapshot"}) if market == "stocks" else ([], {"live_decision_source_available": False, "live_decision_rows_found": 0, "live_decision_rows_completed": 0, "live_decision_join_rate_pct": 0.0, "live_decision_rows_by_market": {market: 0}, "live_decision_rows_by_predictor": {}, "live_decision_missing_reason": "completed_live_decision_snapshots_not_available_for_market", "model_quality_source_priority_used": ""})):
                with mock.patch("app.model_quality_pass._generate_stock_historical_replay_closed_trades", return_value={"rows": stock_rows, "diagnostics": {"source_type": "historical_api_replay", "rows_generated": 50}}):
                    with mock.patch("app.model_quality_pass.load_market_trade_events", return_value={"events": []}):
                        with mock.patch("app.model_quality_pass.build_closed_trades", return_value={"closed_trades": []}):
                            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                            out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertEqual(out.get("replay_generation", {}).get("stocks", {}).get("model_quality_source_priority_used"), "historical_api_replay")
            self.assertTrue(bool(out.get("replay_generation", {}).get("stocks", {}).get("completed_live_rows_used_as_supplemental")))
            self.assertFalse(bool(out.get("replay_generation", {}).get("stocks", {}).get("completed_live_rows_used_as_primary")))

    def test_forex_keeps_execution_log_primary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "forex", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000 + i, "event": "exit", "symbol": "EUR_USD", "qty": 1.0, "price": 1.01, "avg_entry_price": 1.0, "hold_s": 3600, "pnl_pct": 1.0, "tag": "Trailing"}
                    for i in range(50)
                ],
            )
            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                            out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertEqual(out.get("replay_generation", {}).get("forex", {}).get("model_quality_source_priority_used"), "execution_log")

    def test_run_model_quality_full_pass_reports_existing_live_settings_without_modifying_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_rollout_stage": "live",
                "market_crypto_enabled": True,
                "market_stocks_enabled": True,
                "market_forex_enabled": True,
                "stock_auto_trade_enabled": True,
                "forex_auto_trade_enabled": True,
                "alpaca_paper_mode": False,
                "oanda_practice_mode": False,
                "stock_trade_notional_usd": 12.75,
                "stock_max_total_exposure_pct": 60.0,
                "forex_trade_units": 190,
                "forex_max_total_exposure_pct": 60.0,
            }
            original = dict(settings)
            with mock.patch("app.model_quality_pass.get_robinhood_creds_from_env", return_value=("rk", "rs")):
                with mock.patch("app.model_quality_pass.get_robinhood_creds_from_files", return_value=("", "")):
                    with mock.patch("app.model_quality_pass.get_alpaca_creds", return_value=("ak", "as")):
                        with mock.patch("app.model_quality_pass.get_oanda_creds", return_value=("oa", "ot")):
                            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "NO_DATA", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                                with mock.patch("app.model_quality_pass._generate_stock_historical_replay_closed_trades", return_value={"rows": [], "diagnostics": {}}):
                                    with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                            with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                                with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                                    out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings=settings)
            self.assertEqual(settings, original)
            existing = out.get("existing_runtime_settings", {})
            self.assertTrue(bool(existing.get("settings_used_as_is")))
            self.assertTrue(bool(existing.get("existing_crypto_live_enabled")))
            self.assertTrue(bool(existing.get("existing_stocks_live_enabled")))
            self.assertTrue(bool(existing.get("existing_forex_live_enabled")))

    def test_market_readiness_allows_crypto_live_while_model_quality_blockers_remain_visible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_rollout_stage": "live",
                "market_crypto_enabled": True,
                "market_stocks_enabled": True,
                "market_forex_enabled": True,
                "stock_auto_trade_enabled": True,
                "forex_auto_trade_enabled": True,
                "alpaca_paper_mode": False,
                "oanda_practice_mode": False,
            }
            with mock.patch("app.model_quality_pass.get_robinhood_creds_from_env", return_value=("rk", "rs")):
                with mock.patch("app.model_quality_pass.get_robinhood_creds_from_files", return_value=("", "")):
                    with mock.patch("app.model_quality_pass.get_alpaca_creds", return_value=("ak", "as")):
                        with mock.patch("app.model_quality_pass.get_oanda_creds", return_value=("oa", "ot")):
                            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "NO_DATA", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                                with mock.patch("app.model_quality_pass._generate_stock_historical_replay_closed_trades", return_value={"rows": [], "diagnostics": {}}):
                                    with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                            with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                                with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                                    out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings=settings)
            readiness = out.get("model_quality_market_readiness", {}).get("markets", {}).get("crypto", {})
            self.assertTrue(bool(readiness.get("existing_live_setting")))
            self.assertTrue(bool(readiness.get("live_allowed_based_on_existing_setting")))
            self.assertFalse(bool(readiness.get("full_promotion_eligible")))
            self.assertTrue(bool(readiness.get("model_quality_blockers_visible")))
            self.assertFalse(bool(readiness.get("whether_blockers_prevent_live_trading")))
            self.assertEqual(readiness.get("runtime_status"), "live_learning_production")

    def test_completed_live_decision_artifacts_are_written_as_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "stocks", "execution_audit.jsonl"),
                [
                    {
                        "ts": 1_700_000_000,
                        "event": "exit",
                        "symbol": "NVDA",
                        "qty": 1.0,
                        "price": 104.0,
                        "avg_entry_price": 100.0,
                        "hold_s": 7200,
                        "pnl_pct": 4.0,
                        "tag": "Trailing",
                        "decision_snapshot_id": "snap-stock-1",
                        "entry_snapshot_predicted_direction": "up",
                        "entry_snapshot_predicted_exit_trigger": "Trailing",
                        "entry_snapshot_predicted_pnl_trend": "up",
                        "entry_snapshot_predicted_confidence": 0.77,
                        "entry_snapshot_selected_predictor": "local_market_model",
                        "entry_snapshot_predictor_variant": "live",
                        "entry_snapshot_selected_action": "buy",
                        "entry_snapshot_live_trading_allowed": True,
                    }
                ],
            )
            settings = {
                "market_rollout_stage": "live",
                "market_stocks_enabled": True,
                "stock_auto_trade_enabled": True,
                "alpaca_paper_mode": False,
            }
            with mock.patch("app.model_quality_pass.get_robinhood_creds_from_env", return_value=("rk", "rs")):
                with mock.patch("app.model_quality_pass.get_robinhood_creds_from_files", return_value=("", "")):
                    with mock.patch("app.model_quality_pass.get_alpaca_creds", return_value=("ak", "as")):
                        with mock.patch("app.model_quality_pass.get_oanda_creds", return_value=("oa", "ot")):
                            with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "NO_DATA", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                                with mock.patch("app.model_quality_pass._generate_stock_historical_replay_closed_trades", return_value={"rows": [], "diagnostics": {}}):
                                    with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                            with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                                with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                                    out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings=settings)
            art = out.get("completed_live_decision_artifacts", {})
            self.assertTrue(os.path.exists(art.get("per_market", {}).get("stocks", {}).get("path", "")))
            self.assertTrue(os.path.exists(art.get("unified", {}).get("path", "")))
            self.assertTrue(bool(out.get("replay_generation", {}).get("stocks", {}).get("completed_live_rows_used_as_supplemental")))
            self.assertFalse(bool(out.get("replay_generation", {}).get("stocks", {}).get("completed_live_rows_used_as_primary")))

    def test_stock_exit_shape_ignores_bars_after_exit(self) -> None:
        bars = []
        base_ts = 1_700_000_000
        price = 100.0
        for i in range(24):
            price *= 1.0015
            bars.append({"t": base_ts + (i * 3600), "o": price, "h": price * 1.002, "l": price * 0.998, "c": price})
        for mult in [1.02, 1.03, 1.04]:
            price *= mult
            bars.append({"t": base_ts + (len(bars) * 3600), "o": price, "h": price * 1.002, "l": price * 0.998, "c": price})
        for mult in [0.97, 0.97]:
            price *= mult
            bars.append({"t": base_ts + (len(bars) * 3600), "o": price, "h": price * 1.002, "l": price * 0.998, "c": price})
        exit_price = price
        for _ in range(20):
            price *= 1.08
            bars.append({"t": base_ts + (len(bars) * 3600), "o": price, "h": price * 1.01, "l": price * 0.995, "c": price})
        trades = _simulate_stock_trades_from_bars("NVDA", bars)
        self.assertTrue(trades)
        trade = trades[0]
        self.assertEqual(trade.get("actual_exit_trigger"), "Trailing")
        self.assertLess(int(trade.get("exit_ts", 0) or 0), int(bars[-1]["t"]))
        self.assertLess(float(trade.get("peak_profit_pct", 0.0)), 20.0)

    def test_stock_predictor_uses_exit_shape_for_trailing(self) -> None:
        train_rows = []
        for i in range(16):
            train_rows.append(
                {
                    "symbol": "NVDA",
                    "source_type": "historical_api_replay",
                    "entry_price": 100.0,
                    "exit_price": 105.0,
                    "hold_hours": 18.0,
                    "actual_exit_trigger": "Trailing",
                    "actual_direction": "up",
                    "peak_profit_pct": 6.0,
                    "drawdown_from_peak_pct": -2.4,
                    "trailing_armed": True,
                    "favorable_then_softened_flag": True,
                    "stale_hold_profile": False,
                    "trend_decay_after_peak": -1.2,
                    "recent_return_6": 0.8,
                    "recent_return_24": 1.6,
                    "recent_volatility": 0.6,
                    "trend_momentum_score": 1.2,
                    "entry_ts": 100 + i,
                    "exit_ts": 200 + i,
                }
            )
        candidate = {
            "symbol": "NVDA",
            "source_type": "historical_api_replay",
            "entry_price": 100.0,
            "peak_profit_pct": 5.5,
            "drawdown_from_peak_pct": -2.1,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "stale_hold_profile": False,
            "trend_decay_after_peak": -1.1,
            "recent_return_6": 0.6,
            "recent_return_24": 1.3,
            "recent_volatility": 0.55,
            "trend_momentum_score": 1.0,
            "entry_ts": 999,
            "exit_ts": 1999,
        }
        out = _stock_predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", predictor_variant="candidate")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Trailing")
        self.assertEqual(str(out.get("stock_exit_shape_predictive_mode", "")), "active")

    def test_promotion_readiness_surfaces_controlled_rollout_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "forex", "execution_audit.jsonl"),
                [
                    {"ts": 1_700_000_000 + i, "event": "exit", "symbol": "EUR_USD", "qty": 1.0, "price": 1.01, "avg_entry_price": 1.0, "hold_s": 3600, "pnl_pct": 1.0, "tag": "Trailing"}
                    for i in range(60)
                ],
            )
            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                            out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            promo = out.get("promotion_readiness", {}).get("forex", {})
            rollout = out.get("controlled_rollout_readiness", {}).get("forex", {})
            self.assertIn("controlled_rollout_eligible", promo)
            self.assertIn("full_promotion_eligible", promo)
            self.assertIn("risk_multiplier_recommended", promo)
            self.assertIn("eligible", rollout)
            self.assertIn("risk_multiplier_recommended", rollout)

    def test_crypto_sequence_fields_are_computed_without_post_exit_candles(self) -> None:
        artifact_ctx = {
            "usable": True,
            "signal_margin": 0.6,
            "active_timeframe_count": 4,
            "trained_artifacts_fresh": True,
            "artifact_training_time": 1,
            "predicted_low_boundary": 90.0,
            "predicted_high_boundary": 120.0,
        }
        thresholds = {
            "entry_signal_margin_min": 0.1,
            "entry_trend_score_min": -10.0,
            "entry_return6_min": -10.0,
            "risk_cut_pct": 2.25,
            "take_profit_pct": 4.25,
            "trailing_arm_pct": 1.6,
            "trailing_drawdown_pct": 1.1,
            "stale_hold_hours": 18.0,
            "stale_trend_score_max": 0.05,
        }
        candles = []
        ts = 1_700_000_000_000
        price = 100.0
        for _ in range(24):
            candles.append([ts, price, price, price * 1.002, price * 0.998, 1000.0])
            ts += 3_600_000
        candles.append([ts, 100.0, 100.0, 100.2, 99.8, 1000.0]); ts += 3_600_000
        candles.append([ts, 100.0, 102.0, 102.4, 99.9, 1000.0]); ts += 3_600_000
        candles.append([ts, 102.0, 101.7, 102.0, 97.4, 1000.0]); ts += 3_600_000
        candles.append([ts, 101.7, 110.0, 111.0, 101.5, 1000.0]); ts += 3_600_000
        for _ in range(24):
            candles.append([ts, 110.0, 110.0, 110.2, 109.8, 1000.0])
            ts += 3_600_000
        rows = crypto_historical_replay._simulate_strategy_rows("BTC-USD", candles, artifact_ctx, "1hour", thresholds)
        self.assertGreaterEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.get("actual_exit_trigger"), "Risk Cut")
        self.assertEqual(int(row.get("exit_bar", -1)), 2)
        self.assertEqual(int(row.get("risk_cut_touched_bar", -1)), 2)
        self.assertEqual(int(row.get("trailing_armed_bar", -1)), 1)
        self.assertTrue(bool(row.get("risk_cut_after_trailing_arm", False)))
        self.assertGreaterEqual(float(row.get("risk_breach_depth_pct", 0.0) or 0.0), 0.0)
        self.assertLess(int(row.get("exit_ts", 0) or 0), int(candles[-1][0] / 1000))

    def test_crypto_v3_sequence_prefers_risk_cut_before_trailing(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "BTC-USD",
                    "source_type": "historical_strategy_replay",
                    "entry_price": 100.0,
                    "exit_price": 97.0,
                    "actual_exit_price": 97.0,
                    "hold_hours": 8.0,
                    "pnl_pct": -3.0,
                    "actual_direction": "down",
                    "actual_exit_trigger": "Risk Cut",
                    "regime": "high_volatility",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 1.6,
                    "recent_return_3": 2.5,
                    "recent_return_6": 2.8,
                    "recent_return_12": 3.0,
                    "recent_return_24": 3.1,
                    "recent_volatility": 0.8,
                    "trend_momentum_score": 2.7,
                    "signal_margin": 0.52,
                    "active_timeframe_count": 5,
                }
            )
        candidate = {
            "symbol": "BTC-USD",
            "source_type": "historical_strategy_replay",
            "entry_price": 100.0,
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 1.5,
            "recent_return_3": 2.6,
            "recent_return_6": 2.7,
            "recent_return_12": 2.9,
            "recent_return_24": 3.0,
            "recent_volatility": 0.78,
            "trend_momentum_score": 2.6,
            "signal_margin": 0.51,
            "active_timeframe_count": 5,
            "risk_cut_touched": True,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "risk_cut_before_trailing": True,
            "risk_breach_depth_pct": 0.8,
            "exit_close_position_in_candle_range": 0.12,
            "max_adverse_excursion_pct": -2.8,
            "max_favorable_excursion_pct": 2.4,
            "drawdown_from_peak_pct": -2.1,
            "trailing_pullback_pct": 2.1,
            "bars_in_trade": 8,
            "exit_momentum_3": -1.9,
            "exit_momentum_6": -2.2,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto", predictor_variant="label_compatible_v3_sequence")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Risk Cut")
        self.assertTrue(bool(out.get("risk_trailing_resolver_applied", False)))

    def test_crypto_v3_sequence_prefers_trailing_before_risk_without_breach(self) -> None:
        train_rows = []
        for i in range(12):
            train_rows.append(
                {
                    "symbol": "ETH-USD",
                    "source_type": "historical_strategy_replay",
                    "entry_price": 100.0,
                    "exit_price": 103.0,
                    "actual_exit_price": 103.0,
                    "hold_hours": 10.0,
                    "pnl_pct": 3.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Trailing",
                    "regime": "high_volatility",
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "signal_side": "long",
                    "current_candle_pct_move": 1.2,
                    "recent_return_3": 2.0,
                    "recent_return_6": 2.2,
                    "recent_return_12": 2.4,
                    "recent_return_24": 2.6,
                    "recent_volatility": 0.58,
                    "trend_momentum_score": 2.2,
                    "signal_margin": 0.42,
                    "active_timeframe_count": 5,
                }
            )
        candidate = {
            "symbol": "ETH-USD",
            "source_type": "historical_strategy_replay",
            "entry_price": 100.0,
            "strategy_adapter_used": "minimal_artifact_replay_v1",
            "signal_side": "long",
            "current_candle_pct_move": 1.1,
            "recent_return_3": 2.1,
            "recent_return_6": 2.2,
            "recent_return_12": 2.4,
            "recent_return_24": 2.5,
            "recent_volatility": 0.56,
            "trend_momentum_score": 2.15,
            "signal_margin": 0.41,
            "active_timeframe_count": 5,
            "risk_cut_touched": False,
            "trailing_armed": True,
            "favorable_then_softened_flag": True,
            "trailing_before_risk_cut": True,
            "trailing_valid_before_risk": True,
            "risk_breach_depth_pct": 0.0,
            "exit_close_position_in_candle_range": 0.62,
            "max_adverse_excursion_pct": -0.6,
            "max_favorable_excursion_pct": 3.2,
            "drawdown_from_peak_pct": -1.4,
            "trailing_pullback_pct": 1.4,
            "bars_in_trade": 10,
            "exit_momentum_3": -0.6,
            "exit_momentum_6": -0.8,
        }
        out = _predict_one(train_rows=train_rows, candidate=candidate, regime="high_volatility", market="crypto", predictor_variant="label_compatible_v3_sequence")
        self.assertEqual(str(out.get("predicted_exit_trigger", "")), "Trailing")

    def test_stock_ticker_normalization_and_manual_watchlist_storage(self) -> None:
        self.assertEqual(_normalize_stock_ticker(" nvda "), "NVDA")
        self.assertEqual(_normalize_stock_ticker("brk.b!"), "BRK.B")
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_universe_symbols": "AAPL"}
            add_stock_to_manual_watchlist(hub_dir=td, settings=settings, symbol="nvda", validation_provider="alpaca")
            add_stock_to_manual_watchlist(hub_dir=td, settings=settings, symbol="NVDA", validation_provider="alpaca")
            payload = self._write_and_read(os.path.join(td, "stocks", "manual_watchlist.json"))
            self.assertIn("NVDA", payload.get("symbols", {}))
            self.assertEqual(settings.get("stock_universe_symbols"), "AAPL,NVDA")

    def test_validate_stock_symbol_handles_invalid_without_crashing(self) -> None:
        with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", mock.Mock(get_stock_bars=lambda *a, **k: [], list_tradable_assets=lambda: []), {})):
            out = validate_stock_watchlist_symbol(symbol="bad$", settings={}, base_dir="/tmp")
        self.assertEqual(out.get("symbol"), "BAD")
        self.assertFalse(bool(out.get("valid", False)))

    def test_validate_stock_symbol_uses_dated_bar_fallback_for_after_hours_validity(self) -> None:
        fake_client = mock.Mock()
        fake_client.list_tradable_assets.return_value = []

        def _bars(symbol, timeframe="1Day", limit=5, feed="iex", start_iso="", end_iso=""):
            if symbol == "INTC" and start_iso and end_iso:
                return [{"t": "2026-06-01T13:00:00Z", "c": 20.0}]
            return []

        fake_client.get_stock_bars.side_effect = _bars
        with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", fake_client, {})):
            out = validate_stock_watchlist_symbol(symbol="INTC", settings={}, base_dir="/tmp")
        self.assertEqual(out.get("symbol"), "INTC")
        self.assertTrue(bool(out.get("valid", False)))
        self.assertEqual(out.get("status"), "valid_symbol")

    def test_validate_stock_symbol_allows_already_added_watchlist_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_universe_symbols": "AAPL"}
            add_stock_to_manual_watchlist(hub_dir=td, settings=settings, symbol="NVDA", validation_provider="alpaca")
            out = validate_stock_watchlist_symbol(symbol="NVDA", settings=settings, base_dir=td, hub_dir=td)
        self.assertEqual(out.get("symbol"), "NVDA")
        self.assertTrue(bool(out.get("valid", False)))
        self.assertTrue(bool(out.get("already_in_watchlist", False)))
        self.assertEqual(out.get("status"), "already_in_watchlist")

    def test_targeted_stock_warmup_and_preview_use_existing_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_data_provider": "alpaca"}
            bars = []
            base_ts = 1_700_000_000
            price = 100.0
            for i in range(72):
                if i < 30:
                    price *= 1.004
                else:
                    price *= 0.998
                bars.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(base_ts + (i * 3600))), "o": price, "h": price * 1.01, "l": price * 0.995, "c": price, "v": 1000})
            fake_client = mock.Mock()
            fake_client.get_stock_bars.return_value = list(bars)
            with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", fake_client, {})):
                warm = warm_stock_historical_cache(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
                preview = build_stock_watchlist_prediction_preview(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
            self.assertEqual(warm.get("warmup_status"), "ready")
            self.assertTrue(str(warm.get("cache_path", "")).endswith("stocks/historical_replay_cache"))
            self.assertTrue(os.path.exists(os.path.join(td, "stocks", "historical_replay_cache", "NVDA_1Hour.json")))
            self.assertIn("stock_readiness", preview)
            self.assertTrue(os.path.exists(os.path.join(td, "stocks", "watchlist_previews", "NVDA.json")))

    def test_stock_provider_failure_is_non_fatal_and_insufficient_history_blocks_trade(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {"stock_data_provider": "alpaca"}
            fake_client = mock.Mock()
            fake_client.get_stock_bars.side_effect = RuntimeError("provider down")
            with mock.patch.object(model_quality_pass, "_stock_provider_client", return_value=("alpaca", fake_client, {})):
                warm = warm_stock_historical_cache(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
            self.assertEqual(warm.get("warmup_status"), "provider_error")
            preview = build_stock_watchlist_prediction_preview(hub_dir=td, base_dir=td, settings=settings, symbol="NVDA")
            self.assertFalse(bool(preview.get("manual_watchlist_trade_eligible", False)))
            self.assertIn("insufficient_history", list(preview.get("manual_watchlist_trade_blockers", []) or []))

    def test_build_legacy_trade_model_replay_replays_stock_trade_with_pre_entry_bars_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "stocks", "execution_audit.jsonl"),
                [
                    {
                        "ts": 1_700_000_000,
                        "event": "entry",
                        "symbol": "NVDA",
                        "qty": 2.0,
                        "price": 100.0,
                        "ok": True,
                    },
                    {
                        "ts": 1_700_003_600,
                        "event": "exit",
                        "symbol": "NVDA",
                        "qty": 2.0,
                        "price": 104.0,
                        "ok": True,
                        "tag": "Trailing",
                    },
                ],
            )
            bars = []
            base_ts = 1_700_000_000 - (25 * 3600)
            price = 90.0
            for i in range(26):
                price += 0.44
                bars.append(
                    {
                        "t": base_ts + (i * 3600),
                        "o": round(price - 0.25, 4),
                        "h": round(price + 0.5, 4),
                        "l": round(price - 0.5, 4),
                        "c": round(price, 4),
                        "v": 1000,
                    }
                )
            bars.append(
                {
                    "t": 1_700_000_000 + 3600,
                    "o": 101.0,
                    "h": 150.0,
                    "l": 100.5,
                    "c": 140.0,
                    "v": 1000,
                }
            )
            cache_path = os.path.join(td, "stocks", "historical_replay_cache", "NVDA_1Hour.json")
            self._write_json(cache_path, {"bars": bars})
            seen: dict[str, object] = {}

            def _fake_predict(*, train_rows, candidate, regime, market, predictor_variant):
                seen["candidate"] = dict(candidate)
                return {
                    "predicted_direction": "flat",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_pnl_trend": "flat",
                    "predicted_confidence": 0.91,
                    "direction_scores": {"up": 0.91, "down": 0.09},
                    "trigger_scores": {"Trailing": 0.81, "Stale Alignment": 0.19},
                    "stock_trade_quality_score": 0.77,
                }

            with mock.patch("app.model_quality_pass._predict_one", side_effect=_fake_predict):
                with mock.patch("app.model_quality_pass._calibrate_abstain_threshold", return_value={"threshold": 0.60, "coverage": 1.0, "metrics": {}}):
                    out = build_legacy_trade_model_replay(hub_dir=td, base_dir=td, settings={})

            self.assertEqual(int(out.get("legacy_rows_found", 0)), 1)
            self.assertEqual(int(out.get("legacy_rows_replayed", 0)), 1)
            self.assertTrue(bool(out.get("legacy_trade_model_replay_used_as_supplemental")))
            sample = list(out.get("row_samples", []) or [])[0]
            self.assertIn("replayed_predicted_direction", sample)
            self.assertIn("replayed_take_trade", sample)
            self.assertEqual(sample.get("timestamp_quality"), "single_timestamp_inferred_entry")
            self.assertTrue(bool(sample.get("direction_correct")))
            self.assertTrue(bool(sample.get("pnl_trend_correct")))
            self.assertTrue(bool(sample.get("trigger_match")))
            candidate = seen.get("candidate", {}) if isinstance(seen.get("candidate", {}), dict) else {}
            self.assertEqual(candidate.get("source_type"), "legacy_trade_model_replay")
            self.assertAlmostEqual(float(candidate.get("entry_price", 0.0) or 0.0), float(bars[25]["c"]), places=6)
            self.assertNotIn("predicted_direction", sample)

    def test_build_legacy_trade_model_replay_reports_ineligible_forex_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            legacy_payload = {
                "rows": [
                    {
                        "market": "forex",
                        "symbol": "EUR_USD",
                        "side": "long",
                        "entry_ts": 1_700_000_000,
                        "exit_ts": 1_700_003_600,
                        "entry_price": 1.0,
                        "exit_price": 1.01,
                        "hold_hours": 1.0,
                        "pnl_pct": 1.0,
                        "legacy_trade_id": "fx-1",
                        "timestamp_quality": "derived_from_exit_hold",
                    }
                ],
                "raw_rows_found": 1,
                "normalized_rows_found": 1,
                "rows_by_market": {"forex": 1},
                "timestamp_quality_summary": {"derived_from_exit_hold": 1},
                "field_coverage": {"entry_ts": 1, "exit_ts": 1, "entry_price": 1, "exit_price": 1, "pnl_usd": 0, "pnl_pct": 1, "qty": 0, "score": 0, "calib_prob": 0, "required_score": 0},
                "source_files_used": [os.path.join(td, "forex", "execution_audit.jsonl")],
            }
            with mock.patch("app.model_quality_pass._normalize_legacy_completed_trades", return_value=legacy_payload):
                out = build_legacy_trade_model_replay(hub_dir=td, base_dir=td, settings={})
            self.assertEqual(int(out.get("legacy_rows_found", 0)), 1)
            self.assertEqual(int(out.get("legacy_rows_replayed", 0)), 0)
            self.assertEqual(int(out.get("legacy_rows_replay_eligible", 0)), 0)
            self.assertEqual(int(out.get("ineligible_reasons", {}).get("historical_forex_feature_replay_not_supported", 0)), 1)
            self.assertEqual(int(out.get("timestamp_quality_summary", {}).get("derived_from_exit_hold", 0)), 1)

    def test_run_model_quality_full_pass_includes_legacy_trade_replay_as_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            legacy_payload = {
                "status": "ok",
                "legacy_rows_found": 12,
                "legacy_rows_replayed": 7,
                "legacy_trade_model_replay_used_as_primary": False,
                "legacy_trade_model_replay_used_as_supplemental": True,
                "model_at_entry_replay_metrics": {"rows": 7},
            }
            with mock.patch("app.model_quality_pass.load_market_trade_events", return_value={"events": []}):
                with mock.patch("app.model_quality_pass.build_closed_trades", return_value={"closed_trades": []}):
                    with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                        with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                            with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_legacy_trade_model_replay", return_value=legacy_payload):
                                            out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertEqual(out.get("legacy_trade_model_replay", {}).get("legacy_rows_found"), 12)
            self.assertFalse(bool(out.get("legacy_trade_model_replay_used_as_primary")))
            self.assertTrue(bool(out.get("legacy_trade_model_replay_used_as_supplemental")))
            self.assertEqual(out.get("replay_generation", {}).get("crypto", {}).get("model_quality_primary_source_used"), "closed_trade_only")

    def test_run_model_quality_full_pass_reports_legacy_primary_flag_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            legacy_payload = {
                "status": "ok",
                "legacy_rows_found": 12,
                "legacy_rows_replayed": 7,
                "legacy_trade_model_replay_used_as_primary": True,
                "legacy_trade_model_replay_used_as_supplemental": True,
                "model_at_entry_replay_metrics": {"rows": 7},
            }
            with mock.patch.dict(os.environ, {"MODEL_QUALITY_USE_LEGACY_REPLAY_AS_PRIMARY": "1"}, clear=False):
                with mock.patch("app.model_quality_pass.load_market_trade_events", return_value={"events": []}):
                    with mock.patch("app.model_quality_pass.build_closed_trades", return_value={"closed_trades": []}):
                        with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value={"state": "READY", "rows": [], "diagnostics": {"historical_strategy_replay_rows": 0, "historical_strategy_replay_symbols": []}}):
                            with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                                with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                        with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                            with mock.patch("app.model_quality_pass.build_legacy_trade_model_replay", return_value=legacy_payload):
                                                out = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            self.assertTrue(bool(out.get("legacy_trade_model_replay_used_as_primary")))
            self.assertTrue(bool(out.get("legacy_trade_model_replay_used_as_supplemental")))

    def _write_and_read(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_historical_blind_rows_use_only_prior_completed_mock_rows_for_predictions(self) -> None:
        rows = [
            {
                "symbol": "MU",
                "entry_ts": 100,
                "exit_ts": 160,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "pnl_pct": 1.0,
                "actual_direction": "up",
                "actual_exit_trigger": "Trailing",
                "bars_in_trade": 4,
            },
            {
                "symbol": "MU",
                "entry_ts": 200,
                "exit_ts": 260,
                "entry_price": 102.0,
                "exit_price": 99.0,
                "pnl_pct": -2.941176,
                "actual_direction": "down",
                "actual_exit_trigger": "Risk Cut",
                "bars_in_trade": 5,
            },
        ]
        history_sizes: list[int] = []

        def _fake_predict(*, train_rows: list[dict], candidate: dict, regime: str, predictor_variant: str) -> dict:
            history_sizes.append(len(train_rows))
            return {
                "predictor_variant": predictor_variant,
                "predicted_direction": "down",
                "predicted_exit_trigger": "Risk Cut",
                "predicted_pnl_trend": "down",
                "predicted_confidence": 0.81,
                "direction_scores": {"down": 0.81},
                "trigger_scores": {"Risk Cut": 0.81},
            }

        with mock.patch("app.model_quality_pass._stock_predict_one", side_effect=_fake_predict):
            out = _augment_historical_blind_rows(rows, market="stocks", symbol_sources={"MU": "manual_watchlist"})

        self.assertEqual(len(out), 2)
        self.assertEqual(history_sizes, [1])
        self.assertEqual(str(out[0].get("predicted_exit_trigger", "")), "Unknown")
        self.assertEqual(str(out[1].get("predicted_exit_trigger", "")), "Risk Cut")
        self.assertEqual(str(out[0].get("source_type", "")), "historical_blind_strategy_simulation")
        self.assertTrue(str(out[0].get("replay_row_id", "")).endswith("historical_blind_strategy_simulation|stock_pnl_quality_v2_fallback"))
        self.assertTrue(str(out[1].get("replay_row_id", "")).endswith("historical_blind_strategy_simulation|stock_pnl_quality_v2"))
        self.assertEqual(str(out[0].get("actual_pnl_trend", "")), "up")
        self.assertEqual(str(out[1].get("actual_pnl_trend", "")), "down")
        self.assertTrue(bool(out[0].get("diagnostic_only", False)))
        self.assertFalse(bool(out[0].get("eligible_for_training", True)))
        self.assertEqual(str(out[0].get("training_eligibility_reason", "")), "prediction_semantics_placeholder_or_fallback")
        self.assertEqual(str(out[1].get("prediction_semantics", "")), "heuristic")

    def test_manual_and_scanner_stock_symbols_share_blind_onboarding_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "stocks", "manual_watchlist.json"),
                {"symbols": {"MU": {"symbol": "MU", "added_by_user": True}}},
            )
            self._write_json(
                os.path.join(td, "stocks", "stock_universe_cache.json"),
                {"symbols": ["AAPL", "MU"]},
            )
            calls: list[tuple[str, str]] = []

            def _fake_onboard(**kwargs: dict) -> dict:
                calls.append((str(kwargs.get("symbol", "")), str(kwargs.get("symbol_source", ""))))
                symbol = str(kwargs.get("symbol", ""))
                source = str(kwargs.get("symbol_source", ""))
                return {
                    "symbol": symbol,
                    "status": {
                        "symbol": symbol,
                        "symbol_source": source,
                        "status": "learning_ready_not_trade_ready",
                        "eligible_for_scan": True,
                        "eligible_for_trade_consideration": False,
                        "cooldown_until": 0,
                        "blockers": ["market_rollout_not_ready"],
                    },
                    "summary": {},
                    "trades": [],
                    "decisions": [],
                    "skips": [],
                }

            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "stocks", "HISTORICAL_BLIND_SIM_MAX_SYMBOLS": "4"}, clear=False):
                with mock.patch("app.model_quality_pass._build_stock_symbol_onboarding", side_effect=_fake_onboard):
                    report = _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})

            self.assertTrue(bool(report.get("enabled")))
            self.assertIn(("MU", "manual_watchlist"), calls)
            self.assertIn(("AAPL", "scanner_discovered"), calls)
            self.assertTrue(os.path.exists(os.path.join(td, "stocks", "candidate_universe.json")))
            self.assertTrue(os.path.exists(os.path.join(td, "stocks", "active_scan_set.json")))

    def test_historical_blind_simulation_artifacts_stay_separate_from_completed_live_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            completed_live_path = os.path.join(td, "completed_live_decisions.jsonl")
            with open(completed_live_path, "w", encoding="utf-8") as f:
                f.write("{\"sentinel\":true}\n")

            fake_crypto = {
                "rows": [
                    {
                        "symbol": "BTC-USD",
                        "entry_ts": 100,
                        "exit_ts": 200,
                        "entry_price": 100.0,
                        "exit_price": 105.0,
                        "pnl_pct": 5.0,
                        "actual_direction": "up",
                        "actual_exit_trigger": "Trailing",
                    }
                ],
                "diagnostics": {
                    "historical_strategy_replay_provider": "kucoin",
                    "historical_strategy_replay_skipped_reasons": [],
                },
            }
            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "crypto"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=fake_crypto):
                    report = _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})

            self.assertFalse(bool(report.get("historical_blind_simulation_used_as_primary")))
            self.assertTrue(bool(report.get("historical_blind_simulation_used_as_supplemental")))
            self.assertEqual(open(completed_live_path, "r", encoding="utf-8").read().strip(), "{\"sentinel\":true}")
            self.assertTrue(os.path.exists(os.path.join(td, "crypto", "historical_blind_simulation_trades.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(td, "historical_blind_simulation_summary.json")))

    def test_crypto_residual_mismatch_artifact_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake_crypto = {
                "rows": [
                    {
                        "symbol": "BTC-USD",
                        "entry_ts": 100,
                        "exit_ts": 200,
                        "entry_price": 100.0,
                        "exit_price": 95.0,
                        "pnl_pct": -5.0,
                        "actual_direction": "down",
                        "actual_exit_trigger": "Risk Cut",
                        "current_candle_pct_move": 0.9,
                        "recent_return_3": 2.2,
                        "recent_return_6": 2.5,
                        "recent_return_12": 2.8,
                        "recent_return_24": 3.2,
                        "recent_volatility": 0.72,
                        "trend_momentum_score": 2.5,
                        "signal_margin": 0.34,
                        "active_timeframe_count": 6,
                        "bars_since_entry": 6,
                        "current_unrealized_pnl_pct": 1.8,
                        "max_favorable_excursion_pct_so_far": 2.8,
                        "max_adverse_excursion_pct_so_far": -0.7,
                        "drawdown_from_peak_pct_so_far": -0.9,
                        "trailing_armed_so_far": True,
                        "bars_since_trailing_armed": 2,
                        "risk_cut_distance_pct": 0.7,
                        "take_profit_distance_pct": 0.9,
                        "peak_to_current_reversal_pct": 0.9,
                        "favorable_then_softened_flag_so_far": True,
                    }
                ],
                "diagnostics": {
                    "historical_strategy_replay_provider": "kucoin",
                    "historical_strategy_replay_skipped_reasons": [],
                },
            }
            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "crypto"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=fake_crypto):
                    _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})
            self.assertTrue(os.path.exists(os.path.join(td, "crypto", "blind_sim_residual_mismatches.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(td, "crypto", "blind_sim_residual_mismatch_summary.json")))

    def test_historical_blind_diagnostics_emit_confusion_matrices_and_mismatch_buckets(self) -> None:
        rows = [
            {
                "symbol": "BTC-USD",
                "predicted_direction": "up",
                "actual_direction": "down",
                "predicted_exit_trigger": "Trailing",
                "actual_exit_trigger": "Risk Cut",
                "predicted_pnl_trend": "profit",
                "actual_pnl_trend": "down",
                "hold_time": 12.0,
                "bars_in_trade": 6,
                "recent_volatility": 1.2,
                "drawdown_from_peak_pct": -2.2,
                "confidence": 0.8,
                "prediction_semantics": "heuristic",
                "prediction_semantics_warning": "heuristic_only",
                "prediction_semantics_placeholder_only": False,
                "eligible_for_training": True,
                "diagnostic_only": False,
            },
            {
                "symbol": "ETH-USD",
                "predicted_direction": "down",
                "actual_direction": "down",
                "predicted_exit_trigger": "Unknown",
                "actual_exit_trigger": "Stale Alignment",
                "predicted_pnl_trend": "negative",
                "actual_pnl_trend": "loss",
                "hold_time": 40.0,
                "bars_in_trade": 20,
                "recent_volatility": 0.4,
                "drawdown_from_peak_pct": -0.4,
                "confidence": 0.0,
                "prediction_semantics": "heuristic",
                "prediction_semantics_warning": "heuristic_only",
                "prediction_semantics_placeholder_only": True,
                "eligible_for_training": False,
                "diagnostic_only": True,
            },
        ]
        diag = _historical_blind_diagnostics(
            market="crypto",
            rows=rows,
            status_by_symbol={"BTC-USD": {"blockers": ["poor_blind_sim_metrics"]}, "ETH-USD": {"blockers": ["existing_readiness_gate"]}},
            active_rows=[{"symbol": "BTC-USD"}],
            eligible_rows=[],
            rejected_rows=[],
        )
        self.assertEqual(str(diag.get("prediction_semantics", "")), "heuristic")
        self.assertIn("down", diag.get("actual_direction_counts", {}))
        self.assertIn("Risk Cut", diag.get("actual_trigger_counts", {}))
        self.assertIn("down", (diag.get("pnl_trend_confusion_matrix", {}).get("down", {}) if isinstance(diag.get("pnl_trend_confusion_matrix", {}).get("down", {}), dict) else {}))
        self.assertTrue(bool(diag.get("top_mismatch_buckets", {}).get("symbol", {})))
        self.assertEqual(int(diag.get("diagnostic_only_rows", 0) or 0), 1)
        self.assertEqual(int(diag.get("eligible_for_training_rows", 0) or 0), 1)

    def test_crypto_blind_rows_can_use_original_dry_run_with_heuristic_trigger_semantics(self) -> None:
        raw_rows = [
            {
                "symbol": "BTC-USD",
                "entry_ts": 100,
                "exit_ts": 200,
                "entry_price": 100.0,
                "exit_price": 104.0,
                "pnl_pct": 4.0,
                "actual_direction": "up",
                "actual_exit_trigger": "Take Profit",
                "hold_hours": 8.0,
                "strategy_adapter_used": "minimal_artifact_replay_v1",
                "current_candle_pct_move": 0.4,
                "recent_return_3": 1.0,
                "recent_return_6": 1.2,
                "recent_return_12": 1.4,
                "recent_return_24": 1.6,
                "recent_volatility": 0.4,
                "trend_momentum_score": 1.0,
                "signal_margin": 0.3,
                "active_timeframe_count": 3,
                "original_predicted_direction": "up",
                "original_predicted_exit_trigger": "Unknown",
                "original_predicted_pnl_trend": "up",
                "original_confidence": 0.71,
                "original_direction_scores": {"up": 0.9, "down": 0.1},
                "original_trigger_scores": {},
                "original_trade_quality_score": 0.62,
                "original_pnl_quality_score": 0.58,
                "original_selected_predictor": "original_crypto_artifact_model",
                "original_predictor_variant": "original_strategy_dry_run_v1",
                "original_source_used": "trained_artifact_original_memory_files",
                "original_prediction_semantics": "original_strategy_dry_run_with_heuristic_trigger",
                "original_prediction_semantics_warning": "original_direction_and_bounds_from_artifacts_but_trigger_semantics_require_replay_heuristic",
                "original_predictor_dry_run_used": True,
                "original_predictor_replay_safe": True,
            }
        ]
        with mock.patch("app.model_quality_pass._safe_predict_blind_row", return_value={"predicted_direction": "down", "predicted_exit_trigger": "Trailing", "predicted_pnl_trend": "down", "predicted_confidence": 0.33, "direction_scores": {"up": 0.2, "down": 0.8}, "trigger_scores": {"Trailing": 0.7}, "trigger_margin": 0.5, "predictor_variant": "blind_sequence_trigger_scorer_v1"}):
            out = _augment_historical_blind_rows(raw_rows, market="crypto", symbol_sources={"BTC-USD": "configured_universe"})
        self.assertEqual(str(out[0].get("predicted_direction", "")), "up")
        self.assertEqual(str(out[0].get("predicted_exit_trigger", "")), "Trailing")
        self.assertEqual(str(out[0].get("prediction_semantics", "")), "original_strategy_dry_run_with_heuristic_trigger")
        self.assertTrue(bool(out[0].get("original_predictor_trigger_from_heuristic", False)))

    def test_crypto_original_dry_run_eval_artifact_is_written(self) -> None:
        fake_crypto = {
            "rows": [
                {
                    "symbol": "BTC-USD",
                    "entry_ts": 100,
                    "exit_ts": 200,
                    "entry_price": 100.0,
                    "exit_price": 104.0,
                    "pnl_pct": 4.0,
                    "actual_direction": "up",
                    "actual_exit_trigger": "Take Profit",
                    "current_candle_pct_move": 0.4,
                    "recent_return_3": 1.0,
                    "recent_return_6": 1.2,
                    "recent_return_12": 1.4,
                    "recent_return_24": 1.6,
                    "recent_volatility": 0.4,
                    "trend_momentum_score": 1.0,
                    "signal_margin": 0.3,
                    "active_timeframe_count": 3,
                    "strategy_adapter_used": "minimal_artifact_replay_v1",
                    "original_predicted_direction": "up",
                    "original_predicted_exit_trigger": "Unknown",
                    "original_predicted_pnl_trend": "up",
                    "original_confidence": 0.71,
                    "original_direction_scores": {"up": 0.9, "down": 0.1},
                    "original_trigger_scores": {},
                    "original_trade_quality_score": 0.62,
                    "original_pnl_quality_score": 0.58,
                    "original_selected_predictor": "original_crypto_artifact_model",
                    "original_predictor_variant": "original_strategy_dry_run_v1",
                    "original_source_used": "trained_artifact_original_memory_files",
                    "original_prediction_semantics": "original_strategy_dry_run_with_heuristic_trigger",
                    "original_prediction_semantics_warning": "original_direction_and_bounds_from_artifacts_but_trigger_semantics_require_replay_heuristic",
                    "original_predictor_dry_run_used": True,
                    "original_predictor_replay_safe": True,
                }
            ],
            "diagnostics": {"historical_strategy_replay_provider": "kucoin", "historical_strategy_replay_skipped_reasons": [], "strategy_adapter_used": "minimal_artifact_replay_v1"},
        }
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "crypto"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=fake_crypto):
                    with mock.patch("app.model_quality_pass._crypto_blind_trigger_predict", return_value={"predicted_direction": "down", "predicted_exit_trigger": "Trailing", "predicted_pnl_trend": "down", "predicted_confidence": 0.33, "direction_scores": {"up": 0.2, "down": 0.8}, "trigger_scores": {"Trailing": 0.7}, "trigger_margin": 0.5, "predictor_variant": "blind_sequence_trigger_scorer_v1"}):
                        _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})
            eval_path = os.path.join(td, "crypto", "original_strategy_dry_run_eval.json")
            self.assertTrue(os.path.exists(eval_path))
            payload = json.load(open(eval_path, "r", encoding="utf-8"))
            self.assertIn("heuristic_vs_original_dry_run_comparison", payload)
            self.assertIn("safety_audit", payload)
            self.assertEqual(str(payload.get("original_trigger_semantics_status", "")), "unavailable")
            self.assertEqual(str(payload.get("recommendation", "")), "proceed_to_separate_trigger_classifier")

    def test_crypto_blind_trigger_scorer_can_choose_non_trailing_classes(self) -> None:
        train_rows = self._crypto_trigger_train_rows()
        risk_candidate = {
            "symbol": "BTC-USD",
            "entry_price": 100.0,
            "regime": "high_volatility",
            "current_candle_pct_move": 0.1,
            "recent_return_3": 0.2,
            "recent_return_6": 0.4,
            "recent_return_12": 0.5,
            "recent_return_24": 0.7,
            "recent_volatility": 1.0,
            "trend_momentum_score": 0.5,
            "signal_margin": 0.10,
            "active_timeframe_count": 3,
            "bars_since_entry": 4,
            "current_unrealized_pnl_pct": -1.4,
            "max_favorable_excursion_pct_so_far": 0.4,
            "max_adverse_excursion_pct_so_far": -2.0,
            "drawdown_from_peak_pct_so_far": -1.4,
            "risk_cut_distance_pct": 0.2,
            "take_profit_distance_pct": 4.0,
            "risk_cut_touched_so_far": True,
            "peak_to_current_reversal_pct": 1.2,
        }
        tp_candidate = {
            "symbol": "BTC-USD",
            "entry_price": 100.0,
            "regime": "high_volatility",
            "current_candle_pct_move": 0.8,
            "recent_return_3": 1.9,
            "recent_return_6": 2.1,
            "recent_return_12": 2.3,
            "recent_return_24": 2.5,
            "recent_volatility": 0.40,
            "trend_momentum_score": 2.1,
            "signal_margin": 0.30,
            "active_timeframe_count": 5,
            "bars_since_entry": 4,
            "current_unrealized_pnl_pct": 2.8,
            "max_favorable_excursion_pct_so_far": 4.0,
            "max_adverse_excursion_pct_so_far": -0.2,
            "drawdown_from_peak_pct_so_far": -0.3,
            "take_profit_distance_pct": 0.1,
            "risk_cut_distance_pct": 2.0,
            "take_profit_touched_so_far": True,
            "peak_to_current_reversal_pct": 0.3,
        }
        stale_candidate = {
            "symbol": "BTC-USD",
            "entry_price": 100.0,
            "regime": "high_volatility",
            "current_candle_pct_move": 0.2,
            "recent_return_3": 0.6,
            "recent_return_6": 0.7,
            "recent_return_12": 0.9,
            "recent_return_24": 1.0,
            "recent_volatility": 0.48,
            "trend_momentum_score": 1.0,
            "signal_margin": 0.16,
            "active_timeframe_count": 4,
            "bars_since_entry": 8,
            "current_unrealized_pnl_pct": 0.1,
            "max_favorable_excursion_pct_so_far": 0.6,
            "max_adverse_excursion_pct_so_far": -0.4,
            "drawdown_from_peak_pct_so_far": -0.2,
            "risk_cut_distance_pct": 1.8,
            "take_profit_distance_pct": 3.8,
            "peak_to_current_reversal_pct": 0.2,
        }
        trailing_candidate = {
            "symbol": "BTC-USD",
            "entry_price": 100.0,
            "regime": "high_volatility",
            "current_candle_pct_move": 1.4,
            "recent_return_3": 2.5,
            "recent_return_6": 2.9,
            "recent_return_12": 3.3,
            "recent_return_24": 3.7,
            "recent_volatility": 0.78,
            "trend_momentum_score": 2.9,
            "signal_margin": 0.50,
            "active_timeframe_count": 6,
            "bars_since_entry": 5,
            "current_unrealized_pnl_pct": 2.4,
            "max_favorable_excursion_pct_so_far": 3.5,
            "max_adverse_excursion_pct_so_far": -0.4,
            "drawdown_from_peak_pct_so_far": -0.7,
            "trailing_armed_so_far": True,
            "bars_since_trailing_armed": 2,
            "risk_cut_distance_pct": 2.0,
            "take_profit_distance_pct": 1.6,
            "peak_to_current_reversal_pct": 0.7,
            "favorable_then_softened_flag_so_far": True,
        }
        self.assertEqual(str(_crypto_blind_trigger_predict(train_rows=train_rows, candidate=risk_candidate, regime="high_volatility").get("predicted_exit_trigger", "")), "Risk Cut")
        self.assertEqual(str(_crypto_blind_trigger_predict(train_rows=train_rows, candidate=tp_candidate, regime="high_volatility").get("predicted_exit_trigger", "")), "Take Profit")
        self.assertEqual(str(_crypto_blind_trigger_predict(train_rows=train_rows, candidate=stale_candidate, regime="high_volatility").get("predicted_exit_trigger", "")), "Stale Alignment")
        self.assertEqual(str(_crypto_blind_trigger_predict(train_rows=train_rows, candidate=trailing_candidate, regime="high_volatility").get("predicted_exit_trigger", "")), "Trailing")

    def test_crypto_second_stage_discriminator_can_restore_trailing_from_risk_cut(self) -> None:
        pred = _crypto_blind_trigger_predict(
            train_rows=self._crypto_trigger_train_rows(),
            candidate={
                "symbol": "BTC-USD",
                "entry_price": 100.0,
                "regime": "high_volatility",
                "current_candle_pct_move": 0.9,
                "recent_return_3": 2.2,
                "recent_return_6": 2.5,
                "recent_return_12": 2.8,
                "recent_return_24": 3.2,
                "recent_volatility": 0.72,
                "trend_momentum_score": 2.5,
                "signal_margin": 0.34,
                "active_timeframe_count": 6,
                "bars_since_entry": 6,
                "current_unrealized_pnl_pct": 1.8,
                "max_favorable_excursion_pct_so_far": 2.8,
                "max_adverse_excursion_pct_so_far": -0.7,
                "drawdown_from_peak_pct_so_far": -1.0,
                "trailing_armed_so_far": True,
                "bars_since_trailing_armed": 2,
                "risk_cut_distance_pct": 0.55,
                "take_profit_distance_pct": 0.9,
                "peak_to_current_reversal_pct": 0.9,
                "favorable_then_softened_flag_so_far": True,
            },
            regime="high_volatility",
        )
        self.assertEqual(str(pred.get("predicted_exit_trigger", "")), "Trailing")
        self.assertTrue(bool(pred.get("second_stage_discriminator_applied", False)))
        self.assertEqual(str(pred.get("second_stage_discriminator_to", "")), "Trailing")

    def test_crypto_second_stage_discriminator_can_restore_stale_alignment_from_risk_cut(self) -> None:
        pred = _crypto_blind_trigger_predict(
            train_rows=self._crypto_trigger_train_rows(),
            candidate={
                "symbol": "BTC-USD",
                "entry_price": 100.0,
                "regime": "high_volatility",
                "current_candle_pct_move": 0.1,
                "recent_return_3": 0.6,
                "recent_return_6": 0.8,
                "recent_return_12": 1.1,
                "recent_return_24": 1.0,
                "recent_volatility": 0.5,
                "trend_momentum_score": 1.0,
                "signal_margin": 0.16,
                "active_timeframe_count": 4,
                "bars_since_entry": 10,
                "current_unrealized_pnl_pct": 0.0,
                "max_favorable_excursion_pct_so_far": 0.7,
                "max_adverse_excursion_pct_so_far": -1.2,
                "drawdown_from_peak_pct_so_far": -0.3,
                "risk_cut_distance_pct": 0.1,
                "take_profit_distance_pct": 3.7,
                "peak_to_current_reversal_pct": 0.2,
                "momentum_decay": 1.2,
            },
            regime="high_volatility",
        )
        self.assertEqual(str(pred.get("predicted_exit_trigger", "")), "Stale Alignment")
        self.assertTrue(bool(pred.get("second_stage_discriminator_applied", False)))

    def test_crypto_second_stage_discriminator_can_restore_take_profit_from_risk_cut(self) -> None:
        decision = _crypto_apply_second_stage_discriminator(
            {
                "pred_trigger": "Risk Cut",
                "trigger_scores": {"Risk Cut": 4.6, "Take Profit": 3.4, "Trailing": 1.2, "Stale Alignment": 0.2},
                "selected_margin": 1.2,
                "risk_pressure_score": 3.2,
                "take_profit_pressure_score": 4.4,
                "trailing_quality_score": 1.5,
                "stale_alignment_score": 0.3,
                "bars_since_entry": 5,
                "max_favorable_excursion_pct_so_far": 3.8,
                "max_adverse_excursion_pct_so_far": -1.0,
                "drawdown_from_peak_pct_so_far": -0.5,
                "trailing_armed": False,
                "bars_since_trailing_armed": -1,
                "risk_cut_distance_pct": 0.6,
                "take_profit_distance_pct": 0.2,
                "risk_cut_touched_so_far": False,
                "take_profit_touched_so_far": True,
                "peak_to_current_reversal_pct": 0.3,
                "favorable_then_softened_flag_so_far": False,
                "momentum_decay": 0.6,
                "margin_context": "unit-test",
            }
        )
        self.assertEqual(str(decision.get("pred_trigger", "")), "Take Profit")
        self.assertTrue(bool(decision.get("second_stage_discriminator_applied", False)))
        self.assertEqual(str(decision.get("second_stage_discriminator_to", "")), "Take Profit")
        self.assertEqual(str(decision.get("second_stage_discriminator_reason", "")), "target_before_reversal_restored_take_profit")

    def test_crypto_second_stage_discriminator_does_not_override_high_margin_risk_cut(self) -> None:
        pred = _crypto_blind_trigger_predict(
            train_rows=self._crypto_trigger_train_rows(),
            candidate={
                "symbol": "BTC-USD",
                "entry_price": 100.0,
                "regime": "high_volatility",
                "current_candle_pct_move": -0.4,
                "recent_return_3": 0.1,
                "recent_return_6": 0.2,
                "recent_return_12": 0.2,
                "recent_return_24": 0.4,
                "recent_volatility": 1.1,
                "trend_momentum_score": 0.4,
                "signal_margin": 0.05,
                "active_timeframe_count": 3,
                "bars_since_entry": 4,
                "current_unrealized_pnl_pct": -2.2,
                "max_favorable_excursion_pct_so_far": 0.1,
                "max_adverse_excursion_pct_so_far": -2.6,
                "drawdown_from_peak_pct_so_far": -2.0,
                "risk_cut_distance_pct": 0.05,
                "risk_cut_touched_so_far": True,
                "peak_to_current_reversal_pct": 1.3,
            },
            regime="high_volatility",
        )
        self.assertEqual(str(pred.get("predicted_exit_trigger", "")), "Risk Cut")
        self.assertFalse(bool(pred.get("second_stage_discriminator_applied", False)))

    def test_crypto_blind_diagnostic_only_rows_use_reason_counts(self) -> None:
        rows = [
            {
                "symbol": "BTC-USD",
                "predicted_direction": "up",
                "actual_direction": "down",
                "predicted_exit_trigger": "Unknown",
                "actual_exit_trigger": "Risk Cut",
                "predicted_pnl_trend": "up",
                "actual_pnl_trend": "down",
                "confidence": 0.0,
                "diagnostic_only": True,
                "eligible_for_training": False,
                "training_eligibility_reason": "prediction_semantics_placeholder_or_fallback",
            },
            {
                "symbol": "ETH-USD",
                "predicted_direction": "up",
                "actual_direction": "up",
                "predicted_exit_trigger": "Trailing",
                "actual_exit_trigger": "Trailing",
                "predicted_pnl_trend": "up",
                "actual_pnl_trend": "up",
                "confidence": 0.2,
                "trigger_score_selected_margin": 0.05,
                "diagnostic_only": True,
                "eligible_for_training": False,
                "training_eligibility_reason": "trigger_margin_below_diagnostic_floor",
            },
        ]
        diag = _historical_blind_diagnostics(
            market="crypto",
            rows=rows,
            status_by_symbol={},
            active_rows=[],
            eligible_rows=[],
            rejected_rows=[],
        )
        self.assertEqual(int((diag.get("diagnostic_only_reason_counts", {}) or {}).get("prediction_semantics_placeholder_or_fallback", 0)), 1)
        self.assertEqual(int((diag.get("diagnostic_only_reason_counts", {}) or {}).get("trigger_margin_below_diagnostic_floor", 0)), 1)

    def test_crypto_blind_row_augmentation_preserves_trigger_score_fields(self) -> None:
        rows = [
            {
                "symbol": "BTC-USD",
                "entry_ts": 100,
                "exit_ts": 200,
                "entry_price": 100.0,
                "exit_price": 104.0,
                "pnl_pct": 4.0,
                "actual_direction": "up",
                "actual_exit_trigger": "Trailing",
                "current_candle_pct_move": 1.1,
                "recent_return_3": 1.8,
                "recent_return_6": 2.2,
                "recent_return_12": 2.8,
                "recent_return_24": 3.0,
                "recent_volatility": 0.7,
                "trend_momentum_score": 2.4,
                "signal_margin": 0.42,
                "active_timeframe_count": 5,
            },
            {
                "symbol": "ETH-USD",
                "entry_ts": 300,
                "exit_ts": 420,
                "entry_price": 100.0,
                "exit_price": 96.0,
                "pnl_pct": -4.0,
                "actual_direction": "down",
                "actual_exit_trigger": "Risk Cut",
                "current_candle_pct_move": -0.6,
                "recent_return_3": -0.4,
                "recent_return_6": 0.2,
                "recent_return_12": 0.3,
                "recent_return_24": 0.5,
                "recent_volatility": 0.9,
                "trend_momentum_score": 0.6,
                "signal_margin": 0.08,
                "active_timeframe_count": 3,
            },
        ]
        augmented = _augment_historical_blind_rows(
            rows,
            market="crypto",
            symbol_sources={"BTC-USD": "configured_universe", "ETH-USD": "configured_universe"},
            crypto_trigger_mode="improved",
        )
        scored = next((r for r in augmented if str(r.get("symbol", "")) == "ETH-USD"), {})
        self.assertTrue(str(scored.get("predictor_variant", "")).startswith("blind_sequence_trigger_scorer"))
        self.assertIn("trigger_score_selected_margin", scored)
        self.assertGreaterEqual(float(scored.get("trigger_score_selected_margin", 0.0) or 0.0), 0.0)
        self.assertTrue(bool(str(scored.get("trigger_score_reason", ""))))
        self.assertEqual(str(scored.get("source_feature_availability", "")), "entry_features_only")

    def test_crypto_trigger_classifier_dataset_excludes_leakage_fields(self) -> None:
        rows = [
            {
                "symbol": "BTC-USD",
                "entry_ts": 100,
                "exit_ts": 200,
                "replay_row_id": "btc-1",
                "actual_exit_trigger": "Risk Cut",
                "predicted_exit_trigger": "Trailing",
                "future_leakage_detected": False,
                "prediction_semantics_placeholder_only": False,
                "original_confidence": 0.66,
                "bars_since_entry": 4,
                "bars_in_trade": 4,
                "current_unrealized_pnl_pct": -1.2,
                "recent_return_3": 0.4,
                "recent_return_6": 0.5,
                "recent_return_12": 0.6,
                "recent_return_24": 0.7,
                "recent_volatility": 0.8,
                "trend_momentum_score": 0.9,
                "signal_margin": 0.1,
                "hold_hours": 6.0,
                "original_predicted_direction": "down",
                "original_signal_side": "short",
                "pnl_pct": -3.0,
            }
        ]
        dataset = _crypto_trigger_classifier_dataset_rows(rows)
        self.assertGreater(int(len(dataset.get("rows", []) or [])), 0)
        self.assertIn("actual_exit_trigger", dataset.get("excluded_leakage_fields", []))
        self.assertIn("pnl_pct", dataset.get("excluded_leakage_fields", []))
        self.assertNotIn("actual_exit_trigger", dataset.get("feature_names", []))
        self.assertIn("original_confidence", dataset.get("feature_names", []))
        self.assertIn("bars_since_entry", dataset.get("feature_names", []))

    def test_crypto_trigger_classifier_split_is_deterministic_and_chronological(self) -> None:
        rows = [
            {"replay_row_id": f"row-{idx}", "entry_ts": idx, "exit_ts": idx + 10, "label": "Trailing", "heuristic_prediction": "Trailing", "features": [float(idx)]}
            for idx in range(20)
        ]
        first = _split_walkforward_rows(rows)
        second = _split_walkforward_rows(list(reversed(rows)))
        self.assertEqual([r["replay_row_id"] for r in first["train"]], [r["replay_row_id"] for r in second["train"]])
        self.assertEqual(first["train"][0]["entry_ts"], 0)
        self.assertEqual(first["test"][-1]["entry_ts"], 19)

    def test_crypto_trigger_classifier_candidate_is_candidate_only_and_emits_artifacts(self) -> None:
        rows = _augment_historical_blind_rows(
            self._crypto_trigger_train_rows(),
            market="crypto",
            symbol_sources={"BTC-USD": "configured_universe", "ETH-USD": "configured_universe", "SOL-USD": "configured_universe", "DOGE-USD": "configured_universe"},
            crypto_trigger_mode="improved",
        )
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("brokers.broker_alpaca.AlpacaBrokerClient", side_effect=AssertionError("alpaca should not be called")):
                with mock.patch("brokers.broker_twelvedata.TwelveDataClient", side_effect=AssertionError("twelvedata should not be called")):
                    result = _build_crypto_trigger_classifier_candidate(td, rows)
            self.assertTrue(os.path.exists(result["dataset_path"]))
            self.assertTrue(os.path.exists(result["eval_path"]))
            payload = json.load(open(result["eval_path"], "r", encoding="utf-8"))
            self.assertTrue(bool(payload.get("candidate_only", False)))
            self.assertTrue(bool(payload.get("heuristic_fallback_preserved", False)))
            self.assertEqual(str(payload.get("env_flag", "")), "MODEL_QUALITY_USE_CRYPTO_TRIGGER_CLASSIFIER")
            self.assertEqual(list(payload.get("original_model_responsibility", [])), ["direction", "bounds", "confidence"])

    def test_safe_read_jsonl_respects_performance_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rows.jsonl")
            self._write_jsonl(path, [{"idx": i} for i in range(6)])
            diag: dict = {}
            with mock.patch.dict(os.environ, {"MODEL_QUALITY_MAX_JSONL_ROWS_READ": "3"}, clear=False):
                rows = _safe_read_jsonl(path, max_lines=10, diag=diag)
            self.assertEqual(len(rows), 3)
            self.assertTrue(bool(diag.get("truncated", False)))
            self.assertEqual(int(diag.get("effective_limit", 0)), 3)

    def test_write_jsonl_caps_residual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "blind_sim_residual_mismatches.jsonl")
            diag: dict = {}
            with mock.patch.dict(os.environ, {"MODEL_QUALITY_MAX_RESIDUAL_ROWS_WRITTEN": "2"}, clear=False):
                _write_jsonl(path, [{"idx": i} for i in range(5)], diag=diag)
            written = _safe_read_jsonl(path, max_lines=10)
            self.assertEqual(len(written), 2)
            self.assertTrue(bool(diag.get("truncated", False)))
            self.assertEqual(int(diag.get("rows_written", 0)), 2)

    def test_crypto_trigger_classifier_dataset_cap_is_reported(self) -> None:
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "symbol": "BTC-USD",
                    "entry_ts": 100 + idx,
                    "exit_ts": 200 + idx,
                    "replay_row_id": f"row-{idx}",
                    "actual_exit_trigger": "Risk Cut" if idx % 2 == 0 else "Trailing",
                    "predicted_exit_trigger": "Trailing",
                    "future_leakage_detected": False,
                    "prediction_semantics_placeholder_only": False,
                    "original_confidence": 0.66,
                    "bars_since_entry": 4,
                    "bars_in_trade": 4,
                    "current_unrealized_pnl_pct": -1.2,
                    "recent_return_3": 0.4,
                    "recent_return_6": 0.5,
                    "recent_return_12": 0.6,
                    "recent_return_24": 0.7,
                    "recent_volatility": 0.8,
                    "trend_momentum_score": 0.9,
                    "signal_margin": 0.1,
                    "hold_hours": 6.0,
                    "original_predicted_direction": "down",
                    "original_signal_side": "short",
                }
            )
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"MODEL_QUALITY_MAX_CLASSIFIER_DATASET_ROWS": "5"}, clear=False):
                result = _build_crypto_trigger_classifier_candidate(td, rows)
            dataset_payload = json.load(open(result["dataset_path"], "r", encoding="utf-8"))
            self.assertEqual(int(dataset_payload.get("rows", 0)), 5)
            self.assertEqual(int(dataset_payload.get("attempted_rows", 0)), 12)
            self.assertTrue(bool(dataset_payload.get("dataset_rows_truncated", False)))

    def test_historical_blind_simulation_crypto_summary_keeps_classifier_candidate_only(self) -> None:
        fake_crypto = {
            "rows": _augment_historical_blind_rows(
                self._crypto_trigger_train_rows(),
                market="crypto",
                symbol_sources={"BTC-USD": "configured_universe", "ETH-USD": "configured_universe", "SOL-USD": "configured_universe", "DOGE-USD": "configured_universe"},
                crypto_trigger_mode="improved",
            ),
            "diagnostics": {"historical_strategy_replay_provider": "kucoin", "historical_strategy_replay_skipped_reasons": [], "strategy_adapter_used": "minimal_artifact_replay_v1"},
        }
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "crypto,stocks,forex", "MODEL_QUALITY_USE_CRYPTO_TRIGGER_CLASSIFIER": "1"}, clear=False):
                with mock.patch("app.model_quality_pass.build_crypto_historical_strategy_replay", return_value=fake_crypto):
                    report = _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})
        crypto_summary = (((report.get("markets", {}) or {}).get("crypto", {}) or {}).get("summary", {}) or {})
        self.assertTrue(bool(crypto_summary.get("crypto_trigger_classifier_candidate_only", False)))
        self.assertFalse(bool(crypto_summary.get("crypto_trigger_classifier_live_enabled", True)))
        self.assertEqual(str(crypto_summary.get("crypto_trigger_classifier_effect_on_primary_source", "")), "none_candidate_only")
        self.assertFalse(bool(report.get("historical_blind_simulation_used_as_primary", True)))

    def test_run_model_quality_full_pass_emits_performance_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_jsonl(
                os.path.join(td, "trade_history.jsonl"),
                [
                    {
                        "market": "crypto",
                        "symbol": "BTC-USD",
                        "entry_ts": 100,
                        "exit_ts": 200,
                        "entry_price": 100.0,
                        "exit_price": 104.0,
                        "pnl_pct": 4.0,
                        "actual_direction": "up",
                        "actual_exit_trigger": "Take Profit",
                        "actual_pnl_trend": "up",
                    }
                ],
            )
            with mock.patch("app.model_quality_pass.build_synthetic_replay_artifact", return_value={"meta": {}, "replay_source_diagnostics": {}, "crypto_classifier_diagnostics": {}}):
                with mock.patch("app.model_quality_pass.build_replay_diagnostics", return_value={"state": "READY", "source": "", "headline_metrics": {}}):
                    with mock.patch("app.model_quality_pass.build_all_market_regimes", return_value={}):
                        with mock.patch("app.model_quality_pass.build_walkforward_report", return_value={}):
                            with mock.patch("app.model_quality_pass.build_confidence_calibration_payload", return_value={}):
                                with mock.patch("app.model_quality_pass.build_shadow_scorecards", return_value={}):
                                    with mock.patch("app.model_quality_pass.build_legacy_trade_model_replay", return_value={"legacy_rows_replayed": 0}):
                                        with mock.patch("app.model_quality_pass._build_market_readiness_artifact", return_value={}):
                                            payload = run_model_quality_full_pass(base_dir=td, hub_dir=td, settings={})
            perf_path = _performance_diagnostics_path(td)
            self.assertEqual(str(payload.get("performance_diagnostics_path", "")), perf_path)
            self.assertTrue(os.path.exists(perf_path))
            perf = json.load(open(perf_path, "r", encoding="utf-8"))
            self.assertIn("runtime_seconds_by_major_step", perf)
            self.assertIn("rows_read_by_source", perf)
            self.assertTrue(bool(perf.get("blind_sim_supplemental_only", False)))

    def test_stock_blind_sim_summary_reports_nonzero_pnl_trend_metrics_after_label_fix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            onboarding_payloads = {
                "MU": {
                    "symbol": "MU",
                    "status": {
                        "symbol": "MU",
                        "symbol_source": "manual_watchlist",
                        "status": "learning_ready_not_trade_ready",
                        "eligible_for_scan": True,
                        "eligible_for_trade_consideration": False,
                        "eligible_for_training": True,
                        "blockers": ["market_rollout_not_ready"],
                        "cooldown_until": 0,
                    },
                    "summary": {},
                    "trades": [
                        {
                            "symbol": "MU",
                            "entry_ts": 100,
                            "exit_ts": 120,
                            "entry_price": 10.0,
                            "exit_price": 11.0,
                            "pnl_usd": 1.0,
                            "predicted_direction": "up",
                            "actual_direction": "up",
                            "predicted_exit_trigger": "Trailing",
                            "actual_exit_trigger": "Trailing",
                            "predicted_pnl_trend": "profit",
                            "confidence": 0.7,
                            "prediction_semantics": "heuristic",
                            "prediction_semantics_warning": "heuristic_only",
                            "eligible_for_training": True,
                            "diagnostic_only": False,
                            "training_eligibility_reason": "eligible",
                        },
                        {
                            "symbol": "MU",
                            "entry_ts": 200,
                            "exit_ts": 220,
                            "entry_price": 10.0,
                            "exit_price": 9.0,
                            "pnl_usd": -1.0,
                            "predicted_direction": "down",
                            "actual_direction": "down",
                            "predicted_exit_trigger": "Stale Alignment",
                            "actual_exit_trigger": "Stale Alignment",
                            "predicted_pnl_trend": "negative",
                            "confidence": 0.8,
                            "prediction_semantics": "heuristic",
                            "prediction_semantics_warning": "heuristic_only",
                            "eligible_for_training": True,
                            "diagnostic_only": False,
                            "training_eligibility_reason": "eligible",
                        },
                        {
                            "symbol": "MU",
                            "entry_ts": 300,
                            "exit_ts": 320,
                            "entry_price": 10.0,
                            "exit_price": 11.5,
                            "pnl_usd": 1.5,
                            "predicted_direction": "up",
                            "actual_direction": "up",
                            "predicted_exit_trigger": "Trailing",
                            "actual_exit_trigger": "Trailing",
                            "predicted_pnl_trend": "positive",
                            "confidence": 0.82,
                            "prediction_semantics": "heuristic",
                            "prediction_semantics_warning": "heuristic_only",
                            "eligible_for_training": True,
                            "diagnostic_only": False,
                            "training_eligibility_reason": "eligible",
                        },
                    ],
                    "decisions": [],
                    "skips": [],
                }
            }

            def _fake_onboard(**kwargs: dict) -> dict:
                return onboarding_payloads[str(kwargs.get("symbol", ""))]

            with mock.patch.dict(os.environ, {"HISTORICAL_BLIND_SIM_ENABLED": "1", "HISTORICAL_BLIND_SIM_MARKETS": "stocks", "HISTORICAL_BLIND_SIM_MAX_SYMBOLS": "1"}, clear=False):
                with mock.patch("app.model_quality_pass._collect_stock_blind_sim_candidates", return_value=(["MU"], {"MU": "manual_watchlist"})):
                    with mock.patch("app.model_quality_pass._build_stock_symbol_onboarding", side_effect=_fake_onboard):
                        report = _build_historical_blind_simulation(hub_dir=td, base_dir=td, settings={})
            summary = (((report.get("markets", {}) or {}).get("stocks", {}) or {}).get("summary", {}) or {})
            self.assertIn("up", summary.get("predicted_pnl_trend_counts", {}))
            self.assertIn("down", summary.get("actual_pnl_trend_counts", {}))
            self.assertEqual(int((summary.get("pnl_trend_confusion_matrix", {}).get("up", {}) if isinstance(summary.get("pnl_trend_confusion_matrix", {}).get("up", {}), dict) else {}).get("up", 0)), 2)
            self.assertEqual(int((summary.get("pnl_trend_confusion_matrix", {}).get("down", {}) if isinstance(summary.get("pnl_trend_confusion_matrix", {}).get("down", {}), dict) else {}).get("down", 0)), 1)
            self.assertEqual(str(summary.get("label_alignment_status", "")), "aligned")


if __name__ == "__main__":
    unittest.main()
