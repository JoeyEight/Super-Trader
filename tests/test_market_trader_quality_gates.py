from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from engines import forex_trader, stock_trader


class _FakeAlpacaClient:
    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.order_calls = 0
        self.last_notional = 0.0

    def configured(self) -> bool:
        return True

    def list_positions(self) -> list[dict]:
        return []

    def get_mid_prices(self, symbols: list[str]) -> dict[str, float]:
        return {str(s).strip().upper(): 100.0 for s in symbols}

    def get_account_summary(self) -> dict:
        return {"equity": 10_000.0}

    def get_snapshot_details(self, symbols: list[str]) -> dict:
        return {str(s).strip().upper(): {"mid": 100.0, "spread_bps": 1.0} for s in symbols}

    def place_market_order(self, symbol: str, side: str, notional: float, client_order_id: str, max_retries: int = 2, max_retry_after_s: float = 300.0):
        self.order_calls += 1
        self.last_notional = float(notional)
        return True, "entry ok", {"id": "alpaca-order-1"}

    def close_position(self, symbol: str):
        return True, "ok", {}


class _FakeOandaClient:
    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        self.order_calls = 0
        self.last_units = 0

    def configured(self) -> bool:
        return True

    def fetch_snapshot(self) -> dict:
        return {"raw_positions": [], "nav": 10_000.0}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(p).strip().upper(): 1.2345 for p in instruments}

    def get_pricing_details(self, pairs: list[str]) -> dict:
        return {str(p).strip().upper(): {"mid": 1.2345, "spread_bps": 1.0} for p in pairs}

    def place_market_order(self, instrument: str, units: int, client_order_id: str, max_retries: int = 2, max_retry_after_s: float = 300.0):
        self.order_calls += 1
        self.last_units = int(units)
        return True, "entry ok", {"orderFillTransaction": {"id": "oanda-order-1"}}

    def close_position(self, instrument: str, side: str = "long"):
        return True, "ok", {}


class TestTraderQualityGates(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_stock_blocks_when_thinker_data_quality_is_bad(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": False},
                    "reject_summary": {"reject_rate_pct": 12.0},
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 95.0,
                "stock_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("data-quality gate", str(out.get("msg", "")).lower())
            self.assertIn("data-quality gate", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_blocks_on_reject_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 97.0},
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.62},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 90.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("reject-pressure gate", str(out.get("msg", "")).lower())
            self.assertIn("reject-pressure gate", str(out.get("entry_eval_top_reason", "")).lower())

    def test_stock_ignores_cooldown_dominated_reject_pressure_when_leaders_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {
                        "reject_rate_pct": 100.0,
                        "dominant_reason": "cooldown",
                        "dominant_ratio_pct": 91.0,
                    },
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 95.0,
                "stock_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertLess(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_pct", 100.0)), 95.0)
            self.assertAlmostEqual(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_raw_pct", 0.0)), 100.0, places=2)

    def test_stock_ignores_liquidity_dominated_reject_pressure_when_viable_leader_survives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {
                        "reject_rate_pct": 100.0,
                        "dominant_reason": "liquidity",
                        "dominant_ratio_pct": 81.67,
                    },
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 96.0,
                "stock_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertLess(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_pct", 100.0)), 96.0)
            self.assertAlmostEqual(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_raw_pct", 0.0)), 100.0, places=2)

    def test_forex_ignores_cooldown_dominated_reject_pressure_when_leaders_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {
                        "reject_rate_pct": 97.0,
                        "dominant_reason": "cooldown",
                        "dominant_ratio_pct": 88.0,
                    },
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.62},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 90.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertLess(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_pct", 100.0)), 90.0)
            self.assertAlmostEqual(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_raw_pct", 0.0)), 97.0, places=2)

    def test_forex_ignores_liquidity_dominated_reject_pressure_when_viable_leader_survives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {
                        "reject_rate_pct": 97.0,
                        "dominant_reason": "spread",
                        "dominant_ratio_pct": 80.0,
                    },
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.62},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 90.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertLess(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_pct", 100.0)), 90.0)
            self.assertAlmostEqual(float((out.get("entry_gate_flags", {}) or {}).get("reject_rate_raw_pct", 0.0)), 97.0, places=2)

    def test_stock_reduces_entry_notional_when_cached_fallback_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": True,
                    "fallback_age_s": 90,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 12.0},
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True, "calib_prob": 0.8, "samples": 20, "bars_count": 60}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True, "calib_prob": 0.8, "samples": 20, "bars_count": 60}],
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 95.0,
                "stock_block_entries_on_cached_scan": False,
                "stock_cached_scan_hard_block_age_s": 1200,
                "stock_cached_scan_entry_size_mult": 0.5,
                "market_rollout_stage": "execution_v2",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertEqual(float(out.get("trade_notional_entry_usd", 0.0) or 0.0), 50.0)
            self.assertEqual(float(out.get("entry_size_scale", 1.0) or 1.0), 0.5)

    def test_forex_reduces_units_when_cached_fallback_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": True,
                    "fallback_age_s": 80,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 10.0},
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.62},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True, "calib_prob": 0.8, "samples": 20, "bars_count": 60}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True, "calib_prob": 0.8, "samples": 20, "bars_count": 60}],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "forex_cached_scan_hard_block_age_s": 1200,
                "forex_cached_scan_entry_size_mult": 0.5,
                "market_rollout_stage": "execution_v2",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertEqual(int(out.get("trade_units_entry", 0) or 0), 500)
            self.assertEqual(float(out.get("entry_size_scale", 1.0) or 1.0), 0.5)

    def test_stock_shadow_only_status_reports_live_mode_and_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 12.0},
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "alpaca_paper_mode": False,
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 95.0,
                "stock_block_entries_on_cached_scan": False,
                "market_rollout_stage": "shadow_only",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=False),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("trader_state", "")), "Live shadow-run")
            self.assertEqual(str(out.get("broker_mode", "")), "live")
            self.assertFalse(bool(out.get("execution_enabled", True)))
            self.assertIn("market hours gate", str(out.get("msg", "")).lower())
            self.assertIn("real entries suppressed", str(out.get("msg", "")).lower())

    def test_stock_live_guarded_paper_mode_bypasses_calibration_sample_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 12.0},
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{
                        "symbol": "AAPL",
                        "side": "long",
                        "score": 0.9,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.8,
                        "samples": 0,
                        "bars_count": 60,
                    }],
                    "all_scores": [{
                        "symbol": "AAPL",
                        "side": "long",
                        "score": 0.9,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.8,
                        "samples": 0,
                        "bars_count": 60,
                    }],
                },
            )
            settings = {
                "alpaca_paper_mode": True,
                "stock_auto_trade_enabled": True,
                "stock_require_data_quality_ok_for_entries": True,
                "stock_require_reject_rate_max_pct": 95.0,
                "stock_block_entries_on_cached_scan": False,
                "market_rollout_stage": "live_guarded",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
                "stock_min_samples_live_guarded": 4,
                "stock_min_calib_prob_live_guarded": 0.5,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertNotIn("calibration sample gate", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_shadow_only_status_reports_live_mode_and_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 10.0},
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.62},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True}],
                },
            )
            settings = {
                "oanda_practice_mode": False,
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "shadow_only",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
                "forex_session_mode": "london",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
                patch("engines.forex_trader.datetime") as dt_mock,
            ):
                dt_mock.now.return_value = datetime(2026, 3, 10, 2, 0, tzinfo=timezone.utc)
                dt_mock.timezone = timezone
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("trader_state", "")), "Live shadow-run")
            self.assertEqual(str(out.get("broker_mode", "")), "live")
            self.assertFalse(bool(out.get("execution_enabled", True)))
            self.assertIn("session gate", str(out.get("msg", "")).lower())
            self.assertIn("real entries suppressed", str(out.get("msg", "")).lower())

    def test_forex_preserves_specific_entry_gate_reason_from_thinker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 10.0},
                    "top_pick": {
                        "pair": "EUR_USD",
                        "side": "watch",
                        "score": 0.62,
                        "eligible_for_entry": False,
                        "data_quality_ok": True,
                        "entry_gate_reason": "Calibration sample gate for EUR_USD (0 < 4)",
                    },
                    "leaders": [{
                        "pair": "EUR_USD",
                        "side": "watch",
                        "score": 0.62,
                        "eligible_for_entry": False,
                        "data_quality_ok": True,
                        "entry_gate_reason": "Calibration sample gate for EUR_USD (0 < 4)",
                        "calib_prob": 0.5,
                        "samples": 0,
                        "bars_count": 60,
                    }],
                    "all_scores": [{
                        "pair": "EUR_USD",
                        "side": "watch",
                        "score": 0.62,
                        "eligible_for_entry": False,
                        "data_quality_ok": True,
                        "entry_gate_reason": "Calibration sample gate for EUR_USD (0 < 4)",
                        "calib_prob": 0.5,
                        "samples": 0,
                        "bars_count": 60,
                    }],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "live_guarded",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("calibration sample gate", str(out.get("msg", "")).lower())
            self.assertIn("calibration sample gate", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_live_guarded_market_pooled_calibration_allows_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 10.0},
                    "top_pick": {
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    },
                    "leaders": [{
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    }],
                    "all_scores": [{
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    }],
                },
            )
            settings = {
                "oanda_practice_mode": False,
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "live_guarded",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 1,
                "forex_trade_units": 1000,
                "forex_session_mode": "all",
                "forex_min_samples_live_guarded": 4,
                "forex_min_calib_prob_live_guarded": 0.48,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertNotIn("calibration sample gate", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_risk_caps_downsize_entry_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            class _TinyNavOandaClient(_FakeOandaClient):
                def fetch_snapshot(self) -> dict:
                    return {"raw_positions": [], "nav": 100.0}

            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 10.0},
                    "top_pick": {
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    },
                    "leaders": [{
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    }],
                    "all_scores": [{
                        "pair": "EUR_USD",
                        "side": "long",
                        "score": 0.62,
                        "eligible_for_entry": True,
                        "data_quality_ok": True,
                        "calib_prob": 0.74,
                        "samples": 12,
                        "pair_samples": 0,
                        "market_calibration_samples": 12,
                        "calibration_effective_samples": 12,
                        "calibration_effective_prob": 0.74,
                        "calibration_scope": "market_pooled",
                        "bars_count": 60,
                    }],
                },
            )
            settings = {
                "oanda_practice_mode": False,
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "live_guarded",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 3,
                "forex_trade_units": 2000,
                "forex_session_mode": "all",
                "forex_min_samples_live_guarded": 4,
                "forex_min_calib_prob_live_guarded": 0.48,
                "forex_max_total_exposure_pct": 55.0,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _TinyNavOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertIn("risk-cap-size", str(out.get("msg", "")).lower())
            self.assertLess(float(out.get("risk_cap_size_scale", 1.0) or 1.0), 1.0)
            self.assertIn("units=891", " | ".join([str(x) for x in list(out.get("actions", []) or [])]))

    def test_forex_non_usd_quote_pair_uses_home_conversion_for_risk_caps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)

            class _CrossPairOandaClient(_FakeOandaClient):
                def fetch_snapshot(self) -> dict:
                    return {"raw_positions": [], "nav": 100.0}

                def get_pricing_details(self, pairs: list[str]) -> dict:
                    return {
                        str(p).strip().upper(): {
                            "mid": 5.55,
                            "spread_bps": 1.5,
                            "quote_to_home": 0.128,
                            "quote_to_home_positive": 0.128,
                            "quote_to_home_negative": 0.128,
                        }
                        for p in pairs
                    }

                def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
                    return {str(p).strip().upper(): 5.55 for p in instruments}

            thinker_row = {
                "pair": "AUD_HKD",
                "side": "long",
                "score": 0.62,
                "eligible_for_entry": True,
                "data_quality_ok": True,
                "calib_prob": 0.74,
                "samples": 12,
                "pair_samples": 0,
                "market_calibration_samples": 12,
                "calibration_effective_samples": 12,
                "calibration_effective_prob": 0.74,
                "calibration_scope": "market_pooled",
                "bars_count": 60,
            }
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 5.0},
                    "top_pick": dict(thinker_row),
                    "leaders": [dict(thinker_row)],
                    "all_scores": [dict(thinker_row)],
                },
            )
            settings = {
                "oanda_practice_mode": False,
                "forex_auto_trade_enabled": True,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "live_guarded",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 3,
                "forex_trade_units": 50,
                "forex_session_mode": "all",
                "forex_min_samples_live_guarded": 4,
                "forex_min_calib_prob_live_guarded": 0.48,
                "forex_max_total_exposure_pct": 55.0,
                "market_max_total_exposure_pct": 0.0,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _CrossPairOandaClient),
                patch("engines.forex_trader.time.time", return_value=1_700_000_100),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("entry placed", str(out.get("msg", "")).lower())
            self.assertNotIn("risk cap", str(out.get("entry_eval_top_reason", "")).lower())
            self.assertEqual(float(out.get("risk_cap_size_scale", 0.0) or 0.0), 1.0)
            self.assertIn("units=50", " | ".join([str(x) for x in list(out.get("actions", []) or [])]))


if __name__ == "__main__":
    unittest.main()
