from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines import forex_trader, stock_trader


class _FakeAlpacaClient:
    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.order_calls = 0

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

    def place_market_order(self, *args, **kwargs):  # pragma: no cover - should not be called in guard test
        self.order_calls += 1
        return False, "unexpected", {}

    def close_position(self, symbol: str):
        return True, "ok", {}


class _FakeOandaClient:
    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        self.order_calls = 0

    def configured(self) -> bool:
        return True

    def fetch_snapshot(self) -> dict:
        return {"raw_positions": [], "nav": 10_000.0}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(p).strip().upper(): 1.2345 for p in instruments}

    def get_pricing_details(self, pairs: list[str]) -> dict:
        return {str(p).strip().upper(): {"mid": 1.2345, "spread_bps": 1.1} for p in pairs}

    def place_market_order(self, *args, **kwargs):  # pragma: no cover - should not be called in guard test
        self.order_calls += 1
        return False, "unexpected", {}

    def close_position(self, instrument: str, side: str = "long"):
        return True, "ok", {}


class _FakeOandaEntryClient(_FakeOandaClient):
    place_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.place_calls = 0

    def place_market_order(self, instrument: str, units: int, client_order_id: str, max_retries: int = 2, max_retry_after_s: float = 300.0):
        del instrument, units, client_order_id, max_retries, max_retry_after_s
        type(self).place_calls += 1
        return True, "entry ok", {"orderFillTransaction": {"id": "oanda-order-1"}}


class TestCachedScanEntryGuard(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_stock_trader_blocks_new_entries_on_cached_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": True,
                    "fallback_age_s": 120,
                    "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.9},
                    "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True}],
                    "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.9, "eligible_for_entry": True}],
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "stock_block_entries_on_cached_scan": True,
                "market_rollout_stage": "execution_v2",
                "stock_max_signal_age_seconds": 600,
                "stock_max_open_positions": 1,
                "stock_trade_notional_usd": 100.0,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch("engines.stock_trader.time.time", return_value=1_700_000_100),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("cached fallback", str(out.get("msg", "")).lower())
            self.assertGreaterEqual(int(out.get("entry_eval_total", 0) or 0), 1)
            self.assertIn("cached fallback", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_trader_blocks_new_entries_on_cached_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": 1_700_000_000,
                    "fallback_cached": True,
                    "fallback_age_s": 95,
                    "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.42},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.42, "eligible_for_entry": True}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.42, "eligible_for_entry": True}],
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_block_entries_on_cached_scan": True,
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
            self.assertIn("cached fallback", str(out.get("msg", "")).lower())
            self.assertGreaterEqual(int(out.get("entry_eval_total", 0) or 0), 1)
            self.assertIn("cached fallback", str(out.get("entry_eval_top_reason", "")).lower())

    def test_forex_loss_streak_guard_auto_clears_when_flat_and_cooldown_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _FakeOandaEntryClient.reset()
            now_ts = 1_700_000_100
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "fallback_cached": False,
                    "health": {"data_ok": True},
                    "reject_summary": {"reject_rate_pct": 5.0},
                    "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True, "bars_count": 48}],
                    "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.62, "eligible_for_entry": True, "data_quality_ok": True, "bars_count": 48}],
                },
            )
            # Stuck guard state from a prior session: loss streak is high but we are currently flat.
            self._write_json(
                os.path.join(fx_dir, "forex_trader_state.json"),
                {
                    "loss_streak": 6,
                    "loss_streak_updated_at": now_ts - 7200,
                    "cooldown_until": {"EUR_USD": now_ts - 1200},
                    "open_meta": {},
                    "trail": {},
                    "pending": {},
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "forex_max_loss_streak": 5,
                "forex_loss_cooldown_seconds": 900,
                "forex_require_data_quality_ok_for_entries": True,
                "forex_require_reject_rate_max_pct": 95.0,
                "forex_block_entries_on_cached_scan": False,
                "market_rollout_stage": "execution_v2",
                "forex_max_signal_age_seconds": 600,
                "forex_max_open_positions": 2,
                "forex_trade_units": 1000,
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _FakeOandaEntryClient),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertNotIn("loss-streak guard active", str(out.get("msg", "")).lower())
            self.assertGreaterEqual(int(_FakeOandaEntryClient.place_calls), 1)

            with open(os.path.join(fx_dir, "forex_trader_state.json"), "r", encoding="utf-8") as f:
                persisted = json.load(f)
            self.assertEqual(int(persisted.get("loss_streak", 0) or 0), 0)
            gate_flags = persisted.get("entry_gate_flags", {}) if isinstance(persisted.get("entry_gate_flags", {}), dict) else {}
            self.assertEqual(int(gate_flags.get("loss_streak", 0) or 0), 0)


if __name__ == "__main__":
    unittest.main()
