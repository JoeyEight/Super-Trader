from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines import forex_trader, stock_trader


class _StockClient:
    place_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.place_calls = 0

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        del api_key_id, secret_key, base_url, data_url

    def configured(self) -> bool:
        return True

    def list_positions(self) -> list[dict]:
        return []

    def get_mid_prices(self, symbols: list[str]) -> dict[str, float]:
        return {str(sym).strip().upper(): 100.0 for sym in symbols}

    def get_account_summary(self) -> dict:
        return {
            "equity": 50_000.0,
            "buying_power": 50_000.0,
            "account_type": "margin",
            "multiplier": "2",
        }

    def get_snapshot_details(self, symbols: list[str]) -> dict:
        return {str(sym).strip().upper(): {"mid": 100.0, "spread_bps": 1.0} for sym in symbols}

    def place_market_order(
        self,
        symbol: str,
        side: str,
        notional: float,
        client_order_id: str,
        max_retries: int = 2,
        max_retry_after_s: float = 300.0,
    ):
        del symbol, side, notional, client_order_id, max_retries, max_retry_after_s
        type(self).place_calls += 1
        return True, "ok", {"id": "stock-order-1"}

    def close_position(self, symbol: str):
        del symbol
        return True, "ok", {"id": "stock-close-1"}


class _ForexClient:
    place_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.place_calls = 0

    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        del account_id, api_token, rest_url

    def configured(self) -> bool:
        return True

    def fetch_snapshot(self) -> dict:
        return {
            "raw_positions": [],
            "nav": 10_000.0,
            "margin_available": 9_000.0,
            "margin_rate": 0.05,
        }

    def get_pricing_details(self, instruments: list[str]) -> dict:
        return {str(inst).strip().upper(): {"mid": 1.1, "spread_bps": 1.0} for inst in instruments}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(inst).strip().upper(): 1.1 for inst in instruments}

    def place_market_order(
        self,
        instrument: str,
        units: int,
        client_order_id: str,
        max_retries: int = 2,
        max_retry_after_s: float = 300.0,
    ):
        del instrument, units, client_order_id, max_retries, max_retry_after_s
        type(self).place_calls += 1
        return True, "ok", {"id": "forex-order-1"}

    def close_position(self, instrument: str, side: str):
        del instrument, side
        return True, "ok", {"id": "forex-close-1"}


class TestTradeQualityLivePaths(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_stock_blocked_trade_quality_stops_live_order_and_persists_block_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _StockClient.reset()
            now_ts = 1_700_001_000
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                    "adaptive_threshold": 0.2,
                    "leaders": [
                        {
                            "symbol": "AAPL",
                            "side": "long",
                            "score": 0.80,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                            "calib_prob": 0.90,
                            "samples": 20,
                            "bars_count": 60,
                        }
                    ],
                },
            )
            self._write_json(os.path.join(stocks_dir, "stock_trader_state.json"), {})
            self._write_json(os.path.join(td, "runtime_state.json"), {"alerts": {"severity": "ok"}})
            settings = {
                "stock_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "stock_trade_notional_usd": 100.0,
                "stock_max_open_positions": 3,
                "stock_score_threshold": 0.2,
                "stock_require_reject_rate_max_pct": 99.0,
                "stock_min_bars_required": 24,
                "stock_min_samples_live_guarded": 0,
                "stock_min_calib_prob_live_guarded": 0.0,
                "stock_max_slippage_bps": 40.0,
                "stock_order_retry_count": 1,
                "stock_pdt_equity_threshold_usd": 25_000.0,
                "stock_pdt_max_day_trades_rolling_5d": 3,
            }
            quality_block = {
                "market": "stocks",
                "decision": "block",
                "confidence_score": 24.0,
                "size_multiplier": 0.0,
                "confidence_gate_pass": False,
                "layers": {
                    "signal_quality": True,
                    "execution_quality": False,
                    "compliance_permission": True,
                    "runtime_trust": True,
                },
                "block_reasons": ["Synthetic trade-quality block for test"],
                "components": {},
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _StockClient),
                patch.object(stock_trader, "evaluate_trade_quality", return_value=quality_block),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch.object(stock_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_StockClient.place_calls, 0)
            self.assertIn("trade-quality gate", str(out.get("entry_eval_top_reason", "")).lower())
            tq = out.get("trade_quality", {}) if isinstance(out.get("trade_quality", {}), dict) else {}
            reasons = tq.get("block_reasons", []) if isinstance(tq.get("block_reasons", []), list) else []
            self.assertTrue(any("synthetic trade-quality block" in str(r).lower() for r in reasons))
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertTrue(bool(flags.get("trade_quality_evaluated", False)))
            self.assertEqual(str(flags.get("trade_quality_decision", "")), "block")

    def test_forex_blocked_trade_quality_stops_live_order_and_persists_block_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexClient.reset()
            now_ts = 1_700_001_100
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                    "adaptive_threshold": 0.2,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "long",
                            "score": 0.75,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                            "calibration_effective_prob": 0.92,
                            "calibration_effective_samples": 24,
                            "calibration_scope": "pair",
                            "bars_count": 48,
                        }
                    ],
                },
            )
            self._write_json(os.path.join(fx_dir, "forex_trader_state.json"), {})
            self._write_json(os.path.join(td, "runtime_state.json"), {"alerts": {"severity": "ok"}})
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_trade_units": 1000,
                "forex_max_open_positions": 3,
                "forex_score_threshold": 0.2,
                "forex_require_reject_rate_max_pct": 99.0,
                "forex_min_bars_required": 24,
                "forex_min_samples_live_guarded": 0,
                "forex_min_calib_prob_live_guarded": 0.0,
                "forex_max_slippage_bps": 8.0,
                "forex_order_retry_count": 1,
                "forex_session_mode": "all",
            }
            quality_block = {
                "market": "forex",
                "decision": "block",
                "confidence_score": 21.0,
                "size_multiplier": 0.0,
                "confidence_gate_pass": False,
                "layers": {
                    "signal_quality": True,
                    "execution_quality": False,
                    "compliance_permission": True,
                    "runtime_trust": True,
                },
                "block_reasons": ["Synthetic trade-quality block for forex test"],
                "components": {},
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexClient),
                patch.object(forex_trader, "evaluate_trade_quality", return_value=quality_block),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexClient.place_calls, 0)
            self.assertIn("trade-quality gate", str(out.get("entry_eval_top_reason", "")).lower())
            tq = out.get("trade_quality", {}) if isinstance(out.get("trade_quality", {}), dict) else {}
            reasons = tq.get("block_reasons", []) if isinstance(tq.get("block_reasons", []), list) else []
            self.assertTrue(any("synthetic trade-quality block" in str(r).lower() for r in reasons))
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertTrue(bool(flags.get("trade_quality_evaluated", False)))
            self.assertEqual(str(flags.get("trade_quality_decision", "")), "block")

    def test_stock_portfolio_allocator_deprioritization_blocks_live_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _StockClient.reset()
            now_ts = 1_700_001_200
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                    "adaptive_threshold": 0.2,
                    "leaders": [
                        {
                            "symbol": "AAPL",
                            "side": "long",
                            "score": 0.90,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                            "calib_prob": 0.92,
                            "samples": 24,
                            "bars_count": 80,
                        }
                    ],
                },
            )
            self._write_json(os.path.join(stocks_dir, "stock_trader_state.json"), {})
            self._write_json(os.path.join(td, "runtime_state.json"), {"alerts": {"severity": "ok"}})
            quality_allow = {
                "market": "stocks",
                "decision": "allow",
                "confidence_score": 78.0,
                "size_multiplier": 1.0,
                "confidence_gate_pass": True,
                "layers": {
                    "signal_quality": True,
                    "execution_quality": True,
                    "compliance_permission": True,
                    "runtime_trust": True,
                },
                "block_reasons": [],
                "components": {},
            }
            allocator_block = {
                "decision": "deprioritize",
                "summary": "Stocks candidate deprioritized because crypto has higher opportunity quality and capital is constrained",
                "best_market": "crypto",
                "current_market_score": 54.0,
                "capital_constrained": True,
                "reasons": ["Crypto has stronger opportunity score while capital is constrained"],
            }
            settings = {
                "stock_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "stock_trade_notional_usd": 100.0,
                "stock_max_open_positions": 3,
                "stock_score_threshold": 0.2,
                "stock_require_reject_rate_max_pct": 99.0,
                "stock_min_bars_required": 24,
                "stock_min_samples_live_guarded": 0,
                "stock_min_calib_prob_live_guarded": 0.0,
                "stock_max_slippage_bps": 40.0,
                "stock_order_retry_count": 1,
                "stock_pdt_equity_threshold_usd": 25_000.0,
                "stock_pdt_max_day_trades_rolling_5d": 3,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _StockClient),
                patch.object(stock_trader, "evaluate_trade_quality", return_value=quality_allow),
                patch.object(stock_trader, "evaluate_cross_market_allocation", return_value=allocator_block),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch.object(stock_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_StockClient.place_calls, 0)
            self.assertIn("portfolio allocator", str(out.get("entry_eval_top_reason", "")).lower())
            alloc = out.get("opportunity_allocator", {}) if isinstance(out.get("opportunity_allocator", {}), dict) else {}
            self.assertEqual(str(alloc.get("decision", "")), "deprioritize")
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertEqual(str(flags.get("portfolio_allocator_decision", "")), "deprioritize")

    def test_forex_portfolio_allocator_deprioritization_blocks_live_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexClient.reset()
            now_ts = 1_700_001_300
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                    "adaptive_threshold": 0.2,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "long",
                            "score": 0.88,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                            "calibration_effective_prob": 0.90,
                            "calibration_effective_samples": 28,
                            "calibration_scope": "pair",
                            "bars_count": 60,
                        }
                    ],
                },
            )
            self._write_json(os.path.join(fx_dir, "forex_trader_state.json"), {})
            self._write_json(os.path.join(td, "runtime_state.json"), {"alerts": {"severity": "ok"}})
            quality_allow = {
                "market": "forex",
                "decision": "allow",
                "confidence_score": 75.0,
                "size_multiplier": 1.0,
                "confidence_gate_pass": True,
                "layers": {
                    "signal_quality": True,
                    "execution_quality": True,
                    "compliance_permission": True,
                    "runtime_trust": True,
                },
                "block_reasons": [],
                "components": {},
            }
            allocator_block = {
                "decision": "deprioritize",
                "summary": "Forex candidate deprioritized because crypto has higher opportunity quality and capital is constrained",
                "best_market": "crypto",
                "current_market_score": 53.0,
                "capital_constrained": True,
                "reasons": ["Crypto has stronger opportunity score while capital is constrained"],
            }
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_trade_units": 1000,
                "forex_max_open_positions": 3,
                "forex_score_threshold": 0.2,
                "forex_require_reject_rate_max_pct": 99.0,
                "forex_min_bars_required": 24,
                "forex_min_samples_live_guarded": 0,
                "forex_min_calib_prob_live_guarded": 0.0,
                "forex_max_slippage_bps": 8.0,
                "forex_order_retry_count": 1,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexClient),
                patch.object(forex_trader, "evaluate_trade_quality", return_value=quality_allow),
                patch.object(forex_trader, "evaluate_cross_market_allocation", return_value=allocator_block),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexClient.place_calls, 0)
            self.assertIn("portfolio allocator", str(out.get("entry_eval_top_reason", "")).lower())
            alloc = out.get("opportunity_allocator", {}) if isinstance(out.get("opportunity_allocator", {}), dict) else {}
            self.assertEqual(str(alloc.get("decision", "")), "deprioritize")
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertEqual(str(flags.get("portfolio_allocator_decision", "")), "deprioritize")


if __name__ == "__main__":
    unittest.main()
