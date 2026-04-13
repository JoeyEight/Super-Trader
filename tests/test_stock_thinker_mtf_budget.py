from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from engines import stock_thinker


def _mk_bar(idx: int, close_px: float) -> dict:
    base = datetime.now(timezone.utc) - timedelta(hours=30)
    ts = (base + timedelta(hours=int(idx))).strftime("%Y-%m-%dT%H:00:00Z")
    c = float(close_px)
    o = c * 0.998
    h = max(o, c) * 1.001
    low_px = min(o, c) * 0.999
    return {"t": ts, "o": o, "h": h, "l": low_px, "c": c, "v": 1000 + idx}


class _MtfBudgetClient:
    mtf_calls: list[str] = []

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def get_snapshot_details(self, universe: list[str], feed: str = "iex") -> dict[str, dict[str, float]]:
        return {str(sym).strip().upper(): {"mid": 100.0, "spread_bps": 2.0, "dollar_vol": 25_000_000.0} for sym in universe}

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 120,
        feed: str = "iex",
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        if str(timeframe) in {"4Hour", "1Day"} and int(limit or 0) == 36:
            type(self).mtf_calls.append(str(symbol).strip().upper())
        return [_mk_bar(i, 100.0 + (i * 0.5)) for i in range(max(48, int(limit or 48)))]


class TestStockThinkerMtfBudget(unittest.TestCase):
    def test_mtf_confirmation_uses_cached_bars_before_remote_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            _MtfBudgetClient.mtf_calls = []
            universe = ["AAPL", "MSFT", "NVDA", "TSLA"]
            bars = {sym: [_mk_bar(i, 100.0 + (i * 0.5)) for i in range(48)] for sym in universe}
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 4,
                "stock_mtf_confirm_max_symbols": 2,
                "stock_scan_use_daily_when_closed": False,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _MtfBudgetClient),
                patch.object(stock_thinker, "_select_universe", return_value=list(universe)),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_fetch_bars_for_symbols", return_value=bars),
            ):
                out = stock_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(_MtfBudgetClient.mtf_calls, [])
            scored = list(out.get("all_scores", []) or [])
            self.assertGreaterEqual(len(scored), 4)
            self.assertTrue(all((row.get("mtf_confirmed") is not None) for row in scored[:2]))
            self.assertTrue(all(row.get("mtf_confirmed") is None for row in scored[2:4]))


class _FallbackBudgetClient:
    fallback_symbols: list[str] = []

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def get_snapshot_details(self, universe: list[str], feed: str = "iex") -> dict[str, dict[str, float]]:
        return {str(sym).strip().upper(): {"mid": 50.0, "spread_bps": 2.0, "dollar_vol": 15_000_000.0} for sym in universe}

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 120,
        feed: str = "iex",
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        if str(timeframe).lower() in {"1hour", "1day"}:
            type(self).fallback_symbols.append(str(symbol).strip().upper())
            # Keep bars sparse so fallback stays active and easy to count.
            return [_mk_bar(i, 50.0 + (i * 0.05)) for i in range(16)]
        return [_mk_bar(i, 50.0 + (i * 0.05)) for i in range(max(24, int(limit or 24)))]


class TestStockThinkerFallbackBudget(unittest.TestCase):
    def test_symbol_fallback_budget_limits_expensive_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            _FallbackBudgetClient.fallback_symbols = []
            universe = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META"]
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 6,
                "stock_min_bars_required": 48,
                "stock_mtf_confirm_max_symbols": 0,
                "stock_scan_symbol_fallback_limit": 2,
                "stock_scan_use_daily_when_closed": False,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FallbackBudgetClient),
                patch.object(stock_thinker, "_select_universe", return_value=list(universe)),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_fetch_bars_for_symbols", return_value={}),
            ):
                out = stock_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            touched = {sym for sym in list(_FallbackBudgetClient.fallback_symbols)}
            self.assertLessEqual(len(touched), 2)
            self.assertEqual(int(out.get("symbol_fallback_limit", 0) or 0), 2)
            self.assertGreaterEqual(int(out.get("symbol_fallback_skipped", 0) or 0), 1)


if __name__ == "__main__":
    unittest.main()
