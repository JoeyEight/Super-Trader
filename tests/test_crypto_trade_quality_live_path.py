from __future__ import annotations

import base64
import importlib
import json
import os
import tempfile
import time
import types
import unittest


def _load_pt_trader_module():
    os.environ.setdefault("POWERTRADER_RH_API_KEY", "test-key")
    os.environ.setdefault("POWERTRADER_RH_PRIVATE_B64", base64.b64encode(b"0" * 32).decode("ascii"))
    return importlib.import_module("engines.pt_trader")


class CryptoTradeQualityLivePathTests(unittest.TestCase):
    def _build_bot(
        self,
        pt_trader,
        *,
        runtime_severity: str,
        allocator_override: dict | None = None,
        profile: str = "max_growth",
        long_count: int = 5,
        short_count: int = 0,
        dynamic_score: float = 0.9,
        trading_pairs: list[dict] | None = None,
        dynamic_status_overrides: dict | None = None,
    ):
        bot = object.__new__(pt_trader.CryptoAPITrading)
        bot.path_map = {}
        bot.dca_levels = [-2.5, -5.0, -10.0]
        bot.max_dca_buys_per_24h = 2
        bot.trailing_pm = {}
        bot.trailing_gap_pct = 0.5
        bot.pm_start_pct_no_dca = 5.0
        bot.pm_start_pct_with_dca = 2.5
        bot._last_trailing_settings_sig = (0.5, 5.0, 2.5)
        bot._loop_sleep_ok = 1.0
        bot._loop_sleep_error = 1.5
        bot._dca_buy_ts = {}
        bot._dca_last_sell_ts = {}
        bot._last_exit_ts = {}
        bot.entry_cooldown_seconds = 60.0
        bot._last_account_value_history_write_ts = 0.0
        bot._rate_limited_log_ts = {}
        bot._status_note = ""
        bot.cost_basis = {}
        bot.dca_levels_triggered = {}
        bot._pnl_ledger = {"open_positions": {}, "pending_orders": {}}
        bot._last_good_account_snapshot = {
            "total_account_value": None,
            "buying_power": None,
            "holdings_sell_value": None,
            "holdings_buy_value": None,
            "percent_in_trade": None,
        }
        bot._last_good_holdings_results = []
        bot._last_good_holdings_ts = time.time()
        bot._last_good_positions_snapshot = {}

        bot._reconcile_pending_orders = types.MethodType(lambda self, max_total_wait_s=0.5: None, bot)
        bot.get_account = types.MethodType(lambda self: {"buying_power": 1_000.0}, bot)
        bot.get_holdings = types.MethodType(lambda self: {"results": []}, bot)
        pair_rows = [{"symbol": "BTC-USD"}] if trading_pairs is None else list(trading_pairs)
        bot.get_trading_pairs = types.MethodType(lambda self: list(pair_rows), bot)
        bot._resolve_holdings_results = types.MethodType(lambda self, holdings, recent_trade=False: ([], False), bot)
        bot._refresh_missing_cost_basis = types.MethodType(lambda self, holdings_results: None, bot)
        bot.get_price = types.MethodType(lambda self, symbols: ({"BTC-USD": 101.0}, {"BTC-USD": 100.0}, ["BTC-USD"]), bot)
        bot._process_manual_sell_requests = types.MethodType(
            lambda self, holdings_results, current_sell_prices, valid_symbols: False, bot
        )
        bot._read_long_dca_signal = types.MethodType(lambda self, symbol: int(long_count), bot)
        bot._read_short_dca_signal = types.MethodType(lambda self, symbol: int(short_count), bot)
        bot._reset_dca_window_for_trade = types.MethodType(lambda self, base_symbol, sold=False, ts=None: None, bot)
        bot.calculate_cost_basis = types.MethodType(lambda self: {"BTC": 100.0}, bot)
        bot.initialize_dca_levels = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, path, obj: None, bot)

        buy_calls = []
        status_rows = []

        def _place_buy_order(self, *args, **kwargs):
            buy_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "ok"}

        bot.place_buy_order = types.MethodType(_place_buy_order, bot)
        bot._write_trader_status = types.MethodType(lambda self, payload: status_rows.append(dict(payload)), bot)

        with tempfile.TemporaryDirectory() as td:
            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write(str(int(long_count)))
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write(str(int(short_count)))
            with open(os.path.join(td, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump({"alerts": {"severity": runtime_severity}}, f)
            dynamic_status = {"ranked": [{"symbol": "BTC", "score": float(dynamic_score)}], "rejected": []}
            if isinstance(dynamic_status_overrides, dict):
                dynamic_status.update(dict(dynamic_status_overrides))
            with open(os.path.join(td, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
                json.dump(dynamic_status, f)

            orig_refresh = pt_trader._refresh_paths_and_symbols
            orig_base_paths = dict(pt_trader.base_paths)
            orig_symbols = list(pt_trader.crypto_symbols)
            orig_runtime = pt_trader.RUNTIME_STATE_PATH
            orig_dynamic = pt_trader.CRYPTO_DYNAMIC_STATUS_PATH
            orig_read_settings = pt_trader.read_settings_file
            orig_sleep = pt_trader.time.sleep
            orig_allocator = pt_trader.evaluate_cross_market_allocation
            self.addCleanup(setattr, pt_trader, "_refresh_paths_and_symbols", orig_refresh)
            self.addCleanup(setattr, pt_trader, "base_paths", orig_base_paths)
            self.addCleanup(setattr, pt_trader, "crypto_symbols", orig_symbols)
            self.addCleanup(setattr, pt_trader, "RUNTIME_STATE_PATH", orig_runtime)
            self.addCleanup(setattr, pt_trader, "CRYPTO_DYNAMIC_STATUS_PATH", orig_dynamic)
            self.addCleanup(setattr, pt_trader, "read_settings_file", orig_read_settings)
            self.addCleanup(setattr, pt_trader.time, "sleep", orig_sleep)
            self.addCleanup(setattr, pt_trader, "evaluate_cross_market_allocation", orig_allocator)

            pt_trader._refresh_paths_and_symbols = lambda: None
            pt_trader.base_paths = {"BTC": btc_dir}
            pt_trader.crypto_symbols = ["BTC"]
            pt_trader.RUNTIME_STATE_PATH = os.path.join(td, "runtime_state.json")
            pt_trader.CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(td, "crypto_dynamic_status.json")
            pt_trader.read_settings_file = lambda path, module_name=None: {
                "settings_profile": str(profile),
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 10,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_max_spread_bps": 150.0,
            }
            pt_trader.time.sleep = lambda *_args, **_kwargs: None
            if isinstance(allocator_override, dict):
                pt_trader.evaluate_cross_market_allocation = lambda **_kwargs: dict(allocator_override)

            pt_trader.CryptoAPITrading.manage_trades(bot)

        return buy_calls, status_rows

    def test_manage_trades_applies_crypto_trade_quality_and_places_entry_when_allowed(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(pt_trader, runtime_severity="ok")
        self.assertGreaterEqual(len(buy_calls), 1)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertEqual(str((latest.get("automation_policy", {}) if isinstance(latest.get("automation_policy", {}), dict) else {}).get("market", "")), "crypto")
        self.assertEqual(str((latest.get("trade_quality", {}) if isinstance(latest.get("trade_quality", {}), dict) else {}).get("decision", "")), "allow")
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertTrue(bool(flags.get("trade_quality_evaluated", False)))
        self.assertGreaterEqual(float(flags.get("max_spread_bps", 0.0) or 0.0), 150.0)

    def test_manage_trades_allows_max_growth_dynamic_fallback_when_long_signal_is_zero(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            profile="max_growth",
            long_count=0,
            short_count=0,
            dynamic_score=1.7,
        )
        self.assertGreaterEqual(len(buy_calls), 1)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("signal_gate_mode", "")), "dynamic_score_fallback")
        self.assertEqual(str(flags.get("policy_profile", "")), "max_growth")

    def test_manage_trades_blocks_weak_dynamic_fallback_with_entry_alignment_gate(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            profile="max_growth",
            long_count=0,
            short_count=0,
            dynamic_score=1.05,
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertIn("entry alignment gate blocked", str(latest.get("entry_eval_top_reason", "")).lower())
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("signal_gate_mode", "")), "dynamic_score_fallback")
        self.assertEqual(str(flags.get("entry_alignment_mode", "")), "dynamic_fallback_buffer")
        self.assertFalse(bool(flags.get("entry_alignment_pass", True)))

    def test_manage_trades_blocks_dynamic_fallback_when_short_pressure_indicates_stale_risk(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            profile="max_growth",
            long_count=1,
            short_count=0,
            dynamic_score=1.15,
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertIn("stale-risk guard blocked", str(latest.get("entry_eval_top_reason", "")).lower())
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("entry_alignment_mode", "")), "dynamic_fallback_buffer")
        self.assertIn("stale-risk", str(flags.get("entry_alignment_stale_entry_guard", "")).lower())

    def test_manage_trades_keeps_balanced_profile_strict_when_long_signal_is_zero(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            profile="balanced",
            long_count=0,
            short_count=0,
            dynamic_score=2.0,
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertIn("signal gate blocked", str(latest.get("entry_eval_top_reason", "")).lower())
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("signal_gate_mode", "")), "blocked")
        self.assertEqual(str(flags.get("policy_profile", "")), "balanced")

    def test_manage_trades_applies_dynamic_adaptive_threshold_floor(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            profile="max_growth",
            long_count=0,
            short_count=0,
            dynamic_score=1.0,
            dynamic_status_overrides={
                "adaptive_threshold": 1.4,
                "ranked": [{"symbol": "BTC", "score": 1.0, "samples": 0}],
            },
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertGreaterEqual(float(flags.get("signal_gate_min_dynamic_score", 0.0) or 0.0), 1.4)

    def test_manage_trades_blocks_entries_when_runtime_alerts_are_critical(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(pt_trader, runtime_severity="critical")
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertEqual(str((latest.get("trade_quality", {}) if isinstance(latest.get("trade_quality", {}), dict) else {}).get("decision", "")), "block")
        self.assertIn("runtime trust", str(latest.get("entry_eval_top_reason", "")).lower())

    def test_manage_trades_blocks_entries_when_portfolio_allocator_deprioritizes(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            allocator_override={
                "decision": "deprioritize",
                "summary": "Crypto candidate deprioritized because stocks has higher opportunity quality and capital is constrained",
                "best_market": "stocks",
                "current_market_score": 52.0,
                "capital_constrained": True,
                "reasons": ["Stocks has stronger opportunity score while capital is constrained"],
            },
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertIn("portfolio allocator", str(latest.get("entry_eval_top_reason", "")).lower())
        alloc = latest.get("opportunity_allocator", {}) if isinstance(latest.get("opportunity_allocator", {}), dict) else {}
        self.assertEqual(str(alloc.get("decision", "")), "deprioritize")
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("portfolio_allocator_decision", "")), "deprioritize")

    def test_manage_trades_handles_missing_trading_pairs_without_runtime_error(self) -> None:
        pt_trader = _load_pt_trader_module()
        buy_calls, status_rows = self._build_bot(
            pt_trader,
            runtime_severity="ok",
            trading_pairs=[],
        )
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertIn("trading pairs unavailable", str(latest.get("entry_eval_top_reason", "")).lower())
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertEqual(str(flags.get("signal_gate_symbol", "")), "")


if __name__ == "__main__":
    unittest.main()
