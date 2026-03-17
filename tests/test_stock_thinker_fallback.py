from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines import stock_thinker


def _mk_bar(idx: int, close_px: float) -> dict:
    hh = idx % 24
    day = 1 + (idx % 28)
    ts = f"2026-03-{day:02d}T{hh:02d}:00:00Z"
    c = float(close_px)
    o = c * 1.002
    h = max(o, c) * 1.001
    low_px = min(o, c) * 0.999
    return {"t": ts, "o": o, "h": h, "l": low_px, "c": c, "v": 1000 + idx}


class _FakeAlpacaClient:
    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def get_snapshot_details(self, universe: list[str], feed: str = "iex") -> dict[str, dict[str, float]]:
        return {str(sym).strip().upper(): {"mid": 100.0, "spread_bps": 2.0, "dollar_vol": 15_000_000.0} for sym in universe}

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 120,
        feed: str = "iex",
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        base = 200.0
        out: list[dict] = []
        # Descending closes -> negative score => side watch.
        for i in range(max(24, int(limit or 48))):
            out.append(_mk_bar(i, base - (i * 0.4)))
        return out


class _FakeRejectHeavyAlpacaClient:
    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def get_snapshot_details(self, universe: list[str], feed: str = "iex") -> dict[str, dict[str, float]]:
        # Valid price, but no symbol clears the liquidity floor.
        return {str(sym).strip().upper(): {"mid": 25.0, "spread_bps": 2.0, "dollar_vol": 0.0} for sym in universe}

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 120,
        feed: str = "iex",
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        return [_mk_bar(i, 25.0 + (i * 0.1)) for i in range(max(24, int(limit or 48)))]


class _FallbackProbeClient:
    calls: list[dict] = []

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def get_snapshot_details(self, universe: list[str], feed: str = "iex") -> dict[str, dict[str, float]]:
        return {str(sym).strip().upper(): {"mid": 100.0, "spread_bps": 2.0, "dollar_vol": 15_000_000.0} for sym in universe}

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 120,
        feed: str = "iex",
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict]:
        _FallbackProbeClient.calls.append(
            {
                "symbol": str(symbol).upper(),
                "timeframe": str(timeframe),
                "start_iso": str(start_iso or ""),
                "end_iso": str(end_iso or ""),
                "feed": str(feed),
            }
        )
        # Fallback path should provide a bounded intraday range while market is open.
        if str(timeframe).lower() == "1hour" and start_iso and end_iso:
            return [_mk_bar(i, 100.0 + (i * 0.05)) for i in range(48)]
        if str(timeframe).lower() == "4hour":
            return [_mk_bar(i, 100.0 + (i * 0.05)) for i in range(36)]
        return []


class TestStockThinkerFallback(unittest.TestCase):
    def test_uses_cached_scan_when_universe_selection_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            cached = {
                "state": "READY",
                "ai_state": "Scan ready",
                "msg": "cached baseline",
                "universe": ["AAPL"],
                "leaders": [{"symbol": "AAPL", "side": "watch", "score": -0.32, "reason": "trend"}],
                "all_scores": [{"symbol": "AAPL", "side": "watch", "score": -0.32, "reason": "trend"}],
                "top_pick": {"symbol": "AAPL", "side": "watch", "score": -0.32, "reason": "trend"},
                "top_chart": [{"t": "t1", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5}],
                "top_chart_map": {"AAPL": [{"t": "t1", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5}]},
                "updated_at": 1000,
                "reject_summary": {"reject_rate_pct": 8.0, "dominant_reason": "spread"},
            }
            with open(os.path.join(stocks_dir, "stock_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump(cached, f)

            settings = {"alpaca_api_key_id": "abc", "alpaca_secret_key": "xyz"}
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "_select_universe", side_effect=RuntimeError("boom")),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch("engines.stock_thinker.time.time", return_value=1300),
            ):
                out = stock_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertTrue(bool(out.get("fallback_cached", False)))
            self.assertIn("cached scan", str(out.get("msg", "")).lower())
            self.assertGreaterEqual(len(list(out.get("leaders", []) or [])), 1)

    def test_market_closed_uses_cached_status_without_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            cached = {
                "state": "READY",
                "ai_state": "Scan ready",
                "msg": "cached baseline",
                "universe": ["AAPL", "MSFT"],
                "leaders": [{"symbol": "AAPL", "side": "watch", "score": -0.22, "reason": "cached"}],
                "all_scores": [{"symbol": "AAPL", "side": "watch", "score": -0.22, "reason": "cached"}],
                "top_pick": {"symbol": "AAPL", "side": "watch", "score": -0.22, "reason": "cached"},
                "top_chart": [{"t": "t1", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5}],
                "top_chart_map": {"AAPL": [{"t": "t1", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5}]},
                "updated_at": 1000,
                "reject_summary": {"reject_rate_pct": 8.0, "dominant_reason": "spread"},
            }
            with open(os.path.join(stocks_dir, "stock_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump(cached, f)

            settings = {"alpaca_api_key_id": "abc", "alpaca_secret_key": "xyz"}
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(
                    stock_thinker,
                    "_market_clock_status",
                    return_value={
                        "market_open": False,
                        "source": "alpaca_clock",
                        "next_open": "2026-03-17T13:30:00Z",
                        "next_close": "2026-03-16T20:00:00Z",
                        "timestamp": "2026-03-16T20:10:00Z",
                    },
                ),
                patch.object(stock_thinker, "_select_universe", side_effect=AssertionError("universe scan should be paused")),
                patch("engines.stock_thinker.time.time", return_value=1300),
            ):
                out = stock_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertFalse(bool(out.get("market_open", True)))
            self.assertTrue(bool(out.get("fallback_cached", False)))
            self.assertIn("market closed", str(out.get("ai_state", "")).lower())
            self.assertIn("cached scan", str(out.get("msg", "")).lower())

    def test_publishes_watch_leaders_when_no_longs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_publish_watch_leaders": True,
                "stock_scan_watch_leaders_count": 4,
                "stock_scan_max_symbols": 20,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 200.0 - (i * 0.4)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(str(out.get("leader_mode", "")), "watch_fallback")
            self.assertGreaterEqual(len(list(out.get("leaders", []) or [])), 1)
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("symbol", "")), "AAPL")
            self.assertEqual(str(top.get("side", "")).lower(), "watch")
            self.assertTrue(bool(str(top.get("reason_logic", "") or "").strip()))
            self.assertTrue(bool(str(top.get("reason_data", "") or "").strip()))
            self.assertNotIn("6h", str(top.get("reason", "") or "").lower())

    def test_does_not_invent_fallback_candidates_when_all_symbols_fail_prefilters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            with open(os.path.join(stocks_dir, "stock_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": "READY",
                        "leaders": [{"symbol": "AAPL", "side": "long", "score": 0.8, "reason": "cached"}],
                        "all_scores": [{"symbol": "AAPL", "side": "long", "score": 0.8, "reason": "cached"}],
                        "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.8, "reason": "cached"},
                        "top_chart": [{"t": "t1", "o": 1, "h": 1, "l": 1, "c": 1}],
                        "top_chart_map": {"AAPL": [{"t": "t1", "o": 1, "h": 1, "l": 1, "c": 1}]},
                        "updated_at": 1_700_000_000,
                    },
                    f,
                )

            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
                "stock_min_dollar_volume": 2_500_000.0,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeRejectHeavyAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL", "MSFT", "QQQ"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
            ):
                out = stock_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertFalse(bool(out.get("fallback_cached", False)))
            self.assertEqual(list(out.get("leaders", []) or []), [])
            self.assertEqual(list(out.get("all_scores", []) or []), [])
            self.assertEqual(list(out.get("universe", []) or []), [])
            reject_summary = out.get("reject_summary", {}) if isinstance(out.get("reject_summary", {}), dict) else {}
            self.assertEqual(str(reject_summary.get("dominant_reason", "")), "liquidity")
            self.assertAlmostEqual(float(reject_summary.get("reject_rate_pct", 0.0) or 0.0), 100.0, places=2)

    def test_relaxes_missing_liquidity_gate_for_large_universe_feed_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            universe = [f"S{i:03d}" for i in range(40)]
            bars_map = {
                sym: [_mk_bar(i, 40.0 + (i * 0.05)) for i in range(64)]
                for sym in universe
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                sym = str(symbol).upper()
                idx = int(sym[1:]) if len(sym) > 1 and sym[1:].isdigit() else 0
                return {
                    "symbol": sym,
                    "score": round(1.25 - (idx * 0.001), 6),
                    "side": "long",
                    "last": 40.0,
                    "change_6h_pct": 0.8,
                    "change_24h_pct": 1.9,
                    "volatility_pct": 0.6,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 40,
                "stock_min_bars_required": 24,
                "stock_min_valid_bars_ratio": 0.70,
                "stock_min_dollar_volume": 2_500_000.0,
                "stock_max_spread_bps": 40.0,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeRejectHeavyAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=list(universe)),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(stock_thinker, "_fetch_bars_for_symbols", return_value=bars_map),
            ):
                out = stock_thinker.run_scan(settings, td)

            diag_path = os.path.join(stocks_dir, "scan_diagnostics.json")
            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertTrue(bool(diag.get("liquidity_missing_allowed", False)))
            self.assertGreater(float(diag.get("liquidity_missing_ratio_pct", 0.0) or 0.0), 90.0)
            self.assertGreater(int(len(list(out.get("leaders", []) or []))), 0)
            self.assertNotIn("No symbols passed stock marketability prefilters", str(out.get("msg", "")))

    def test_applies_leader_hysteresis_to_previous_top(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            with open(os.path.join(stocks_dir, "stock_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump({"top_pick": {"symbol": "MSFT", "side": "long", "score": 1.12}}, f)

            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
                "stock_leader_stability_margin_pct": 20.0,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                base = 1.10 if str(symbol).upper() == "MSFT" else 1.20
                return {
                    "symbol": str(symbol).upper(),
                    "score": float(base),
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.0,
                    "change_24h_pct": 2.0,
                    "volatility_pct": 0.5,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL", "MSFT"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={
                        "AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)],
                        "MSFT": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)],
                    },
                ),
            ):
                out = stock_thinker.run_scan(settings, td)
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("symbol", "")), "MSFT")
            self.assertTrue(bool(out.get("leader_stability_applied", False)))

    def test_live_guarded_demotes_undertrained_leader_to_watch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "alpaca_paper_mode": False,
                "market_rollout_stage": "live_guarded",
                "stock_min_samples_live_guarded": 4,
                "stock_min_calib_prob_live_guarded": 0.50,
                "stock_live_guarded_bootstrap_allow": False,
                "stock_scan_max_symbols": 20,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.9,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.0,
                    "change_24h_pct": 2.0,
                    "volatility_pct": 0.5,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "watch")
            self.assertFalse(bool(top.get("eligible_for_entry", True)))
            self.assertIn("Calibration sample gate", str(top.get("entry_gate_reason", "") or ""))

    def test_live_guarded_uses_market_pooled_calibration_for_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "alpaca_paper_mode": False,
                "market_rollout_stage": "live_guarded",
                "stock_min_samples_live_guarded": 4,
                "stock_min_calib_prob_live_guarded": 0.50,
                "stock_scan_max_symbols": 20,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.9,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.0,
                    "change_24h_pct": 2.0,
                    "volatility_pct": 0.5,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(stock_thinker, "_market_pooled_calibration_samples", return_value=12),
                patch.object(stock_thinker, "_calibrated_prob", return_value=0.74),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "long")
            self.assertTrue(bool(top.get("eligible_for_entry", False)))
            self.assertEqual(str(top.get("entry_gate_reason", "") or ""), "")
            self.assertEqual(int(top.get("calibration_effective_samples", 0) or 0), 12)
            self.assertEqual(str(top.get("calibration_scope", "") or ""), "market_pooled")

    def test_live_guarded_null_bootstrap_setting_defaults_to_allow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "alpaca_paper_mode": False,
                "market_rollout_stage": "live_guarded",
                "stock_min_samples_live_guarded": 4,
                "stock_min_calib_prob_live_guarded": 0.50,
                "stock_live_guarded_bootstrap_allow": None,
                "stock_scan_max_symbols": 20,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.9,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.0,
                    "change_24h_pct": 2.0,
                    "volatility_pct": 0.5,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(stock_thinker, "_market_pooled_calibration_samples", return_value=0),
                patch.object(stock_thinker, "_calibrated_prob", return_value=0.74),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "long")
            self.assertTrue(bool(top.get("eligible_for_entry", False)))
            self.assertEqual(str(top.get("entry_gate_reason", "") or ""), "")

    def test_live_guarded_paper_mode_keeps_undertrained_leader_tradeable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "alpaca_paper_mode": True,
                "market_rollout_stage": "live_guarded",
                "stock_min_samples_live_guarded": 4,
                "stock_min_calib_prob_live_guarded": 0.50,
                "stock_scan_max_symbols": 20,
                "stock_scan_use_daily_when_closed": True,
                "stock_scan_closed_pause_hours": 0,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.9,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.0,
                    "change_24h_pct": 2.0,
                    "volatility_pct": 0.5,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "long")
            self.assertTrue(bool(top.get("eligible_for_entry", False)))
            self.assertEqual(str(top.get("entry_gate_reason", "") or ""), "")

    def test_data_quality_cooldown_does_not_hard_block_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            now_ts = 1_700_000_000
            with open(os.path.join(stocks_dir, "symbol_cooldown.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": now_ts,
                        "symbols": {
                            "QQQ": {
                                "symbol": "QQQ",
                                "reason": "data_quality",
                                "hit_count": 0,
                                "until": now_ts + 900,
                                "updated_ts": now_ts,
                            }
                        },
                    },
                    f,
                )

            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["QQQ"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_fetch_bars_for_symbols", return_value={}),
                patch("engines.stock_thinker.time.time", return_value=float(now_ts)),
            ):
                out = stock_thinker.run_scan(settings, td)

            rejected = [row for row in list(out.get("rejected", []) or []) if isinstance(row, dict)]
            self.assertFalse(any(str(row.get("reason", "")).lower() == "cooldown" for row in rejected))
            symbols = {str((row or {}).get("symbol", "")).upper() for row in list(out.get("all_scores", []) or [])}
            self.assertIn("QQQ", symbols)

    def test_open_session_symbol_fallback_uses_time_bounded_intraday_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            _FallbackProbeClient.calls = []
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
            }
            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FallbackProbeClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_fetch_bars_for_symbols", return_value={}),
            ):
                out = stock_thinker.run_scan(settings, td)

            one_hour_calls = [c for c in _FallbackProbeClient.calls if str(c.get("timeframe", "")).lower() == "1hour"]
            self.assertGreaterEqual(len(one_hour_calls), 1)
            self.assertTrue(all(bool(c.get("start_iso")) and bool(c.get("end_iso")) for c in one_hour_calls))
            self.assertGreaterEqual(len(list(out.get("all_scores", []) or [])), 1)

    def test_persists_stock_opening_plan_from_scan_leaders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
                "stock_opening_plan_enabled": True,
                "stock_opening_plan_max_symbols": 5,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.88,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 1.2,
                    "change_24h_pct": 2.1,
                    "volatility_pct": 0.7,
                    "spread_bps": float(spread_bps),
                    "confidence": "HIGH",
                    "reason": "test",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            plan = out.get("opening_plan", {}) if isinstance(out.get("opening_plan", {}), dict) else {}
            rows = list(plan.get("rows", []) or []) if isinstance(plan, dict) else []
            self.assertTrue(bool(plan.get("enabled", False)))
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(str(rows[0].get("symbol", "")), "AAPL")
            persisted = stock_thinker._load_json_map(os.path.join(td, "stocks", "opening_plan.json"))
            self.assertEqual(str((persisted.get("rows", [{}])[0] or {}).get("symbol", "")), "AAPL")

    def test_news_event_weight_is_applied_to_stock_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            settings = {
                "alpaca_api_key_id": "abc",
                "alpaca_secret_key": "xyz",
                "stock_scan_max_symbols": 20,
                "stock_news_event_weight": 1.0,
                "news_event_enabled": True,
            }

            def _score(symbol: str, bars: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "symbol": str(symbol).upper(),
                    "score": 0.20,
                    "side": "long",
                    "last": 100.0,
                    "change_6h_pct": 0.8,
                    "change_24h_pct": 1.1,
                    "volatility_pct": 0.4,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "base momentum",
                    "reason_logic": "base momentum",
                    "reason_data": "6h/24h trend",
                }

            with (
                patch.object(stock_thinker, "get_alpaca_creds", return_value=("abc", "xyz")),
                patch.object(stock_thinker, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch.object(stock_thinker, "_select_universe", return_value=["AAPL"]),
                patch.object(stock_thinker, "_market_open_now", return_value=True),
                patch.object(stock_thinker, "_score_bars", side_effect=_score),
                patch.object(stock_thinker, "_apply_stock_mtf_confirmation", return_value=None),
                patch.object(
                    stock_thinker,
                    "build_unified_news_event_context",
                    return_value={
                        "enabled": True,
                        "market": "stocks",
                        "state": "live",
                        "state_code": "live_ok",
                        "symbols": {
                            "AAPL": {
                                "score": -1.0,
                                "confidence": 1.0,
                                "impact": 1.0,
                                "bias": "negative",
                                "headline_count": 3,
                                "event_risk": True,
                                "top_headline": "Major downgrade headline",
                            }
                        },
                        "errors": {},
                    },
                ),
                patch.object(
                    stock_thinker,
                    "_fetch_bars_for_symbols",
                    return_value={"AAPL": [_mk_bar(i, 100.0 + (i * 0.2)) for i in range(48)]},
                ),
            ):
                out = stock_thinker.run_scan(settings, td)

            rows = [row for row in list(out.get("all_scores", []) or []) if isinstance(row, dict)]
            self.assertGreaterEqual(len(rows), 1)
            row = rows[0]
            self.assertLess(float(row.get("score", 0.0) or 0.0), 0.0)
            self.assertEqual(str(row.get("side", "")).lower(), "watch")
            self.assertEqual(str(row.get("news_bias", "")), "negative")
            self.assertTrue(bool(out.get("news_event_context", {})))


if __name__ == "__main__":
    unittest.main()
