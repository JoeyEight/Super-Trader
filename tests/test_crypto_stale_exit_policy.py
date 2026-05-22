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


class CryptoStaleExitPolicyTests(unittest.TestCase):
    def test_manage_trades_exits_stale_position_before_new_entries(self) -> None:
        pt_trader = _load_pt_trader_module()
        bot = object.__new__(pt_trader.CryptoAPITrading)

        # Minimal runtime state required by manage_trades.
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
        bot.cost_basis = {"BTC": 100.0}
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
        bot._stale_alignment_streaks = {}

        bot._reconcile_pending_orders = types.MethodType(lambda self, max_total_wait_s=0.5: None, bot)
        bot.get_account = types.MethodType(lambda self: {"buying_power": 2_000.0}, bot)
        bot.get_holdings = types.MethodType(
            lambda self: {"results": [{"asset_code": "BTC", "total_quantity": 1.0}]},
            bot,
        )
        bot.get_trading_pairs = types.MethodType(lambda self: [{"symbol": "BTC-USD"}], bot)
        bot._resolve_holdings_results = types.MethodType(
            lambda self, holdings, recent_trade=False: (list(holdings.get("results", []) or []), False),
            bot,
        )
        bot._refresh_missing_cost_basis = types.MethodType(lambda self, holdings_results: None, bot)
        bot.get_price = types.MethodType(
            lambda self, symbols: (
                {"BTC-USD": 101.0},
                {"BTC-USD": 100.0},
                ["BTC-USD"],
            ),
            bot,
        )
        bot._process_manual_sell_requests = types.MethodType(
            lambda self, holdings_results, current_sell_prices, valid_symbols: False,
            bot,
        )
        bot._read_long_dca_signal = types.MethodType(lambda self, symbol: 1, bot)
        bot._read_short_dca_signal = types.MethodType(lambda self, symbol: 0, bot)
        bot._read_long_price_levels = types.MethodType(lambda self, symbol: [], bot)
        bot._write_current_price = types.MethodType(lambda self, symbol, price: None, bot)
        bot._dca_window_count = types.MethodType(lambda self, symbol, now_ts=None: 0, bot)
        bot._note_dca_buy = types.MethodType(lambda self, symbol, ts=None: None, bot)
        bot._reset_dca_window_for_trade = types.MethodType(lambda self, base_symbol, sold=False, ts=None: None, bot)
        bot.calculate_cost_basis = types.MethodType(lambda self: {"BTC": 100.0}, bot)
        bot.initialize_dca_levels = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, path, obj: None, bot)

        sell_calls = []
        buy_calls = []
        status_rows = []

        def _place_sell_order(self, *args, **kwargs):
            sell_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "sell-ok"}

        def _place_buy_order(self, *args, **kwargs):
            buy_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "buy-ok"}

        bot.place_sell_order = types.MethodType(_place_sell_order, bot)
        bot.place_buy_order = types.MethodType(_place_buy_order, bot)
        bot._write_trader_status = types.MethodType(lambda self, payload: status_rows.append(dict(payload)), bot)

        with tempfile.TemporaryDirectory() as td:
            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("0")
            with open(os.path.join(td, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump({"alerts": {"severity": "ok"}}, f)
            with open(os.path.join(td, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
                json.dump({"ranked": [{"symbol": "BTC", "score": 0.9}], "rejected": [], "current_coins": ["ETH"]}, f)

            orig_refresh = pt_trader._refresh_paths_and_symbols
            orig_base_paths = dict(pt_trader.base_paths)
            orig_symbols = list(pt_trader.crypto_symbols)
            orig_runtime = pt_trader.RUNTIME_STATE_PATH
            orig_dynamic = pt_trader.CRYPTO_DYNAMIC_STATUS_PATH
            orig_read_settings = pt_trader.read_settings_file
            orig_sleep = pt_trader.time.sleep
            self.addCleanup(setattr, pt_trader, "_refresh_paths_and_symbols", orig_refresh)
            self.addCleanup(setattr, pt_trader, "base_paths", orig_base_paths)
            self.addCleanup(setattr, pt_trader, "crypto_symbols", orig_symbols)
            self.addCleanup(setattr, pt_trader, "RUNTIME_STATE_PATH", orig_runtime)
            self.addCleanup(setattr, pt_trader, "CRYPTO_DYNAMIC_STATUS_PATH", orig_dynamic)
            self.addCleanup(setattr, pt_trader, "read_settings_file", orig_read_settings)
            self.addCleanup(setattr, pt_trader.time, "sleep", orig_sleep)

            pt_trader._refresh_paths_and_symbols = lambda: None
            pt_trader.base_paths = {"BTC": btc_dir}
            pt_trader.crypto_symbols = ["BTC"]
            pt_trader.RUNTIME_STATE_PATH = os.path.join(td, "runtime_state.json")
            pt_trader.CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(td, "crypto_dynamic_status.json")
            pt_trader.read_settings_file = lambda path, module_name=None: {
                "settings_profile": "max_growth",
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 10,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_max_spread_bps": 150.0,
                "crypto_stale_exit_enabled": True,
                "crypto_stale_alignment_grace_cycles": 1,
                "crypto_stale_max_exits_per_cycle": 2,
                "crypto_stale_min_notional_usd": 1.0,
            }
            pt_trader.time.sleep = lambda *_args, **_kwargs: None

            pt_trader.CryptoAPITrading.manage_trades(bot)

        self.assertEqual(len(sell_calls), 1)
        self.assertIn("POLICY_STALE_EXIT", str((sell_calls[0].get("kwargs", {}) or {}).get("tag", "")))
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertEqual(int(latest.get("stale_exit_count", 0) or 0), 1)
        flags = latest.get("entry_gate_flags", {}) if isinstance(latest.get("entry_gate_flags", {}), dict) else {}
        self.assertTrue(bool(flags.get("skip_new_entries_this_cycle", False)))

    def test_manage_trades_skips_dca_when_alignment_stale_during_grace_window(self) -> None:
        pt_trader = _load_pt_trader_module()
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
        bot.cost_basis = {"BTC": 100.0}
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
        bot._stale_alignment_streaks = {}

        bot._reconcile_pending_orders = types.MethodType(lambda self, max_total_wait_s=0.5: None, bot)
        bot.get_account = types.MethodType(lambda self: {"buying_power": 2_000.0}, bot)
        bot.get_holdings = types.MethodType(
            lambda self: {"results": [{"asset_code": "BTC", "total_quantity": 1.0}]},
            bot,
        )
        bot.get_trading_pairs = types.MethodType(lambda self: [{"symbol": "BTC-USD"}], bot)
        bot._resolve_holdings_results = types.MethodType(
            lambda self, holdings, recent_trade=False: (list(holdings.get("results", []) or []), False),
            bot,
        )
        bot._refresh_missing_cost_basis = types.MethodType(lambda self, holdings_results: None, bot)
        bot.get_price = types.MethodType(
            lambda self, symbols: (
                {"BTC-USD": 97.0},
                {"BTC-USD": 96.0},
                ["BTC-USD"],
            ),
            bot,
        )
        bot._process_manual_sell_requests = types.MethodType(
            lambda self, holdings_results, current_sell_prices, valid_symbols: False,
            bot,
        )
        bot._read_long_dca_signal = types.MethodType(lambda self, symbol: 1, bot)
        bot._read_short_dca_signal = types.MethodType(lambda self, symbol: 0, bot)
        bot._read_long_price_levels = types.MethodType(lambda self, symbol: [], bot)
        bot._write_current_price = types.MethodType(lambda self, symbol, price: None, bot)
        bot._dca_window_count = types.MethodType(lambda self, symbol, now_ts=None: 0, bot)
        bot._note_dca_buy = types.MethodType(lambda self, symbol, ts=None: None, bot)
        bot._reset_dca_window_for_trade = types.MethodType(lambda self, base_symbol, sold=False, ts=None: None, bot)
        bot.calculate_cost_basis = types.MethodType(lambda self: {"BTC": 100.0}, bot)
        bot.initialize_dca_levels = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, path, obj: None, bot)

        sell_calls = []
        buy_calls = []
        status_rows = []

        def _place_sell_order(self, *args, **kwargs):
            sell_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "sell-ok"}

        def _place_buy_order(self, *args, **kwargs):
            buy_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "buy-ok"}

        bot.place_sell_order = types.MethodType(_place_sell_order, bot)
        bot.place_buy_order = types.MethodType(_place_buy_order, bot)
        bot._write_trader_status = types.MethodType(lambda self, payload: status_rows.append(dict(payload)), bot)

        with tempfile.TemporaryDirectory() as td:
            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("0")
            with open(os.path.join(td, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump({"alerts": {"severity": "ok"}}, f)
            with open(os.path.join(td, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
                json.dump({"ranked": [{"symbol": "BTC", "score": 0.9}], "rejected": [], "current_coins": ["ETH"]}, f)

            orig_refresh = pt_trader._refresh_paths_and_symbols
            orig_base_paths = dict(pt_trader.base_paths)
            orig_symbols = list(pt_trader.crypto_symbols)
            orig_runtime = pt_trader.RUNTIME_STATE_PATH
            orig_dynamic = pt_trader.CRYPTO_DYNAMIC_STATUS_PATH
            orig_read_settings = pt_trader.read_settings_file
            orig_sleep = pt_trader.time.sleep
            self.addCleanup(setattr, pt_trader, "_refresh_paths_and_symbols", orig_refresh)
            self.addCleanup(setattr, pt_trader, "base_paths", orig_base_paths)
            self.addCleanup(setattr, pt_trader, "crypto_symbols", orig_symbols)
            self.addCleanup(setattr, pt_trader, "RUNTIME_STATE_PATH", orig_runtime)
            self.addCleanup(setattr, pt_trader, "CRYPTO_DYNAMIC_STATUS_PATH", orig_dynamic)
            self.addCleanup(setattr, pt_trader, "read_settings_file", orig_read_settings)
            self.addCleanup(setattr, pt_trader.time, "sleep", orig_sleep)

            pt_trader._refresh_paths_and_symbols = lambda: None
            pt_trader.base_paths = {"BTC": btc_dir}
            pt_trader.crypto_symbols = ["BTC"]
            pt_trader.RUNTIME_STATE_PATH = os.path.join(td, "runtime_state.json")
            pt_trader.CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(td, "crypto_dynamic_status.json")
            pt_trader.read_settings_file = lambda path, module_name=None: {
                "settings_profile": "max_growth",
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 10,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_max_spread_bps": 150.0,
                "crypto_stale_exit_enabled": True,
                "crypto_stale_alignment_grace_cycles": 2,
                "crypto_stale_max_exits_per_cycle": 2,
                "crypto_stale_min_notional_usd": 1.0,
            }
            pt_trader.time.sleep = lambda *_args, **_kwargs: None

            pt_trader.CryptoAPITrading.manage_trades(bot)

        self.assertEqual(len(sell_calls), 0)
        self.assertEqual(len(buy_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        self.assertEqual(int(latest.get("stale_exit_count", 0) or 0), 0)
        positions = latest.get("positions", {}) if isinstance(latest.get("positions", {}), dict) else {}
        btc_row = positions.get("BTC", {}) if isinstance(positions.get("BTC", {}), dict) else {}
        self.assertFalse(bool(btc_row.get("aligned_with_strategy", True)))

    def test_manage_trades_holds_fresh_mild_loss_position_instead_of_stale_exit(self) -> None:
        pt_trader = _load_pt_trader_module()
        bot = object.__new__(pt_trader.CryptoAPITrading)

        now_ts = time.time()
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
        bot._last_entry_ts = {"BTC": now_ts - 300.0}
        bot._last_exit_ts = {}
        bot.entry_cooldown_seconds = 60.0
        bot._last_account_value_history_write_ts = 0.0
        bot._rate_limited_log_ts = {}
        bot._status_note = ""
        bot.cost_basis = {"BTC": 100.0}
        bot.dca_levels_triggered = {}
        bot._pnl_ledger = {
            "open_positions": {"BTC": {"usd_cost": 100.0, "qty": 1.0, "opened_ts": now_ts - 300.0}},
            "pending_orders": {},
        }
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
        bot._stale_alignment_streaks = {}

        bot._reconcile_pending_orders = types.MethodType(lambda self, max_total_wait_s=0.5: None, bot)
        bot.get_account = types.MethodType(lambda self: {"buying_power": 2_000.0}, bot)
        bot.get_holdings = types.MethodType(
            lambda self: {"results": [{"asset_code": "BTC", "total_quantity": 1.0}]},
            bot,
        )
        bot.get_trading_pairs = types.MethodType(lambda self: [{"symbol": "BTC-USD"}], bot)
        bot._resolve_holdings_results = types.MethodType(
            lambda self, holdings, recent_trade=False: (list(holdings.get("results", []) or []), False),
            bot,
        )
        bot._refresh_missing_cost_basis = types.MethodType(lambda self, holdings_results: None, bot)
        bot.get_price = types.MethodType(
            lambda self, symbols: (
                {"BTC-USD": 99.5},
                {"BTC-USD": 99.0},
                ["BTC-USD"],
            ),
            bot,
        )
        bot._process_manual_sell_requests = types.MethodType(
            lambda self, holdings_results, current_sell_prices, valid_symbols: False,
            bot,
        )
        bot._read_long_dca_signal = types.MethodType(lambda self, symbol: 1, bot)
        bot._read_short_dca_signal = types.MethodType(lambda self, symbol: 1, bot)
        bot._read_long_price_levels = types.MethodType(lambda self, symbol: [], bot)
        bot._write_current_price = types.MethodType(lambda self, symbol, price: None, bot)
        bot._dca_window_count = types.MethodType(lambda self, symbol, now_ts=None: 0, bot)
        bot._note_dca_buy = types.MethodType(lambda self, symbol, ts=None: None, bot)
        bot._reset_dca_window_for_trade = types.MethodType(lambda self, base_symbol, sold=False, ts=None: None, bot)
        bot.calculate_cost_basis = types.MethodType(lambda self: {"BTC": 100.0}, bot)
        bot.initialize_dca_levels = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, path, obj: None, bot)

        sell_calls = []
        status_rows = []

        def _place_sell_order(self, *args, **kwargs):
            sell_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "sell-ok"}

        bot.place_sell_order = types.MethodType(_place_sell_order, bot)
        bot.place_buy_order = types.MethodType(lambda self, *args, **kwargs: {"id": "buy-ok"}, bot)
        bot._write_trader_status = types.MethodType(lambda self, payload: status_rows.append(dict(payload)), bot)

        with tempfile.TemporaryDirectory() as td:
            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("1")
            with open(os.path.join(td, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump({"alerts": {"severity": "ok"}}, f)
            with open(os.path.join(td, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
                json.dump({"ranked": [{"symbol": "BTC", "score": 0.9}], "rejected": [], "current_coins": ["ETH"]}, f)

            orig_refresh = pt_trader._refresh_paths_and_symbols
            orig_base_paths = dict(pt_trader.base_paths)
            orig_symbols = list(pt_trader.crypto_symbols)
            orig_runtime = pt_trader.RUNTIME_STATE_PATH
            orig_dynamic = pt_trader.CRYPTO_DYNAMIC_STATUS_PATH
            orig_read_settings = pt_trader.read_settings_file
            orig_sleep = pt_trader.time.sleep
            self.addCleanup(setattr, pt_trader, "_refresh_paths_and_symbols", orig_refresh)
            self.addCleanup(setattr, pt_trader, "base_paths", orig_base_paths)
            self.addCleanup(setattr, pt_trader, "crypto_symbols", orig_symbols)
            self.addCleanup(setattr, pt_trader, "RUNTIME_STATE_PATH", orig_runtime)
            self.addCleanup(setattr, pt_trader, "CRYPTO_DYNAMIC_STATUS_PATH", orig_dynamic)
            self.addCleanup(setattr, pt_trader, "read_settings_file", orig_read_settings)
            self.addCleanup(setattr, pt_trader.time, "sleep", orig_sleep)

            pt_trader._refresh_paths_and_symbols = lambda: None
            pt_trader.base_paths = {"BTC": btc_dir}
            pt_trader.crypto_symbols = ["BTC"]
            pt_trader.RUNTIME_STATE_PATH = os.path.join(td, "runtime_state.json")
            pt_trader.CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(td, "crypto_dynamic_status.json")
            pt_trader.read_settings_file = lambda path, module_name=None: {
                "settings_profile": "max_growth",
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 10,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_max_spread_bps": 150.0,
                "crypto_stale_exit_enabled": True,
                "crypto_stale_alignment_grace_cycles": 1,
                "crypto_stale_max_exits_per_cycle": 2,
                "crypto_stale_min_notional_usd": 1.0,
                "crypto_stale_min_hold_seconds": 3600,
                "crypto_stale_loss_cut_pct": -2.0,
            }
            pt_trader.time.sleep = lambda *_args, **_kwargs: None

            pt_trader.CryptoAPITrading.manage_trades(bot)

        self.assertEqual(len(sell_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        stale_events = latest.get("stale_exit_events", []) if isinstance(latest.get("stale_exit_events", []), list) else []
        reasons = [str((row or {}).get("reason", "") or "") for row in stale_events if isinstance(row, dict)]
        self.assertIn("stale_exit_hold_loss_guard", reasons)

    def test_manage_trades_holds_trailing_cross_when_alignment_is_strong_and_gain_is_near_flat(self) -> None:
        pt_trader = _load_pt_trader_module()
        bot = object.__new__(pt_trader.CryptoAPITrading)

        now_ts = time.time()
        bot.path_map = {}
        bot.dca_levels = [-2.5, -5.0, -10.0]
        bot.max_dca_buys_per_24h = 2
        bot.trailing_pm = {
            "BTC": {
                "active": True,
                "line": 103.0,
                "peak": 103.5,
                "was_above": True,
                "settings_sig": (0.5, 5.0, 2.5),
            }
        }
        bot.trailing_gap_pct = 0.5
        bot.pm_start_pct_no_dca = 5.0
        bot.pm_start_pct_with_dca = 2.5
        bot._last_trailing_settings_sig = (0.5, 5.0, 2.5)
        bot._loop_sleep_ok = 1.0
        bot._loop_sleep_error = 1.5
        bot._dca_buy_ts = {}
        bot._dca_last_sell_ts = {}
        bot._last_entry_ts = {"BTC": now_ts - 600.0}
        bot._last_exit_ts = {}
        bot.entry_cooldown_seconds = 60.0
        bot._last_account_value_history_write_ts = 0.0
        bot._rate_limited_log_ts = {}
        bot._status_note = ""
        bot.cost_basis = {"BTC": 100.0}
        bot.dca_levels_triggered = {}
        bot._pnl_ledger = {
            "open_positions": {"BTC": {"usd_cost": 100.0, "qty": 1.0, "opened_ts": now_ts - 600.0}},
            "pending_orders": {},
        }
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
        bot._stale_alignment_streaks = {}

        bot._reconcile_pending_orders = types.MethodType(lambda self, max_total_wait_s=0.5: None, bot)
        bot.get_account = types.MethodType(lambda self: {"buying_power": 2_000.0}, bot)
        bot.get_holdings = types.MethodType(
            lambda self: {"results": [{"asset_code": "BTC", "total_quantity": 1.0}]},
            bot,
        )
        bot.get_trading_pairs = types.MethodType(lambda self: [{"symbol": "BTC-USD"}], bot)
        bot._resolve_holdings_results = types.MethodType(
            lambda self, holdings, recent_trade=False: (list(holdings.get("results", []) or []), False),
            bot,
        )
        bot._refresh_missing_cost_basis = types.MethodType(lambda self, holdings_results: None, bot)
        bot.get_price = types.MethodType(
            lambda self, symbols: (
                {"BTC-USD": 102.7},
                {"BTC-USD": 102.6},
                ["BTC-USD"],
            ),
            bot,
        )
        bot._process_manual_sell_requests = types.MethodType(
            lambda self, holdings_results, current_sell_prices, valid_symbols: False,
            bot,
        )
        bot._read_long_dca_signal = types.MethodType(lambda self, symbol: 5, bot)
        bot._read_short_dca_signal = types.MethodType(lambda self, symbol: 0, bot)
        bot._read_long_price_levels = types.MethodType(lambda self, symbol: [], bot)
        bot._write_current_price = types.MethodType(lambda self, symbol, price: None, bot)
        bot._dca_window_count = types.MethodType(lambda self, symbol, now_ts=None: 0, bot)
        bot._note_dca_buy = types.MethodType(lambda self, symbol, ts=None: None, bot)
        bot._reset_dca_window_for_trade = types.MethodType(lambda self, base_symbol, sold=False, ts=None: None, bot)
        bot.calculate_cost_basis = types.MethodType(lambda self: {"BTC": 100.0}, bot)
        bot.initialize_dca_levels = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, path, obj: None, bot)

        sell_calls = []
        status_rows = []

        def _place_sell_order(self, *args, **kwargs):
            sell_calls.append({"args": args, "kwargs": kwargs})
            return {"id": "sell-ok"}

        bot.place_sell_order = types.MethodType(_place_sell_order, bot)
        bot.place_buy_order = types.MethodType(lambda self, *args, **kwargs: {"id": "buy-ok"}, bot)
        bot._write_trader_status = types.MethodType(lambda self, payload: status_rows.append(dict(payload)), bot)

        with tempfile.TemporaryDirectory() as td:
            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("5")
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("0")
            with open(os.path.join(td, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump({"alerts": {"severity": "ok"}}, f)
            with open(os.path.join(td, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ranked": [{"symbol": "BTC", "score": 1.8}],
                        "rejected": [],
                        "current_coins": ["BTC"],
                        "adaptive_threshold": 0.2,
                    },
                    f,
                )

            orig_refresh = pt_trader._refresh_paths_and_symbols
            orig_base_paths = dict(pt_trader.base_paths)
            orig_symbols = list(pt_trader.crypto_symbols)
            orig_runtime = pt_trader.RUNTIME_STATE_PATH
            orig_dynamic = pt_trader.CRYPTO_DYNAMIC_STATUS_PATH
            orig_read_settings = pt_trader.read_settings_file
            orig_sleep = pt_trader.time.sleep
            self.addCleanup(setattr, pt_trader, "_refresh_paths_and_symbols", orig_refresh)
            self.addCleanup(setattr, pt_trader, "base_paths", orig_base_paths)
            self.addCleanup(setattr, pt_trader, "crypto_symbols", orig_symbols)
            self.addCleanup(setattr, pt_trader, "RUNTIME_STATE_PATH", orig_runtime)
            self.addCleanup(setattr, pt_trader, "CRYPTO_DYNAMIC_STATUS_PATH", orig_dynamic)
            self.addCleanup(setattr, pt_trader, "read_settings_file", orig_read_settings)
            self.addCleanup(setattr, pt_trader.time, "sleep", orig_sleep)

            pt_trader._refresh_paths_and_symbols = lambda: None
            pt_trader.base_paths = {"BTC": btc_dir}
            pt_trader.crypto_symbols = ["BTC"]
            pt_trader.RUNTIME_STATE_PATH = os.path.join(td, "runtime_state.json")
            pt_trader.CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(td, "crypto_dynamic_status.json")
            pt_trader.read_settings_file = lambda path, module_name=None: {
                "settings_profile": "max_growth",
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 10,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_max_spread_bps": 150.0,
                "crypto_stale_exit_enabled": True,
                "crypto_stale_alignment_grace_cycles": 2,
                "crypto_stale_max_exits_per_cycle": 2,
                "crypto_stale_min_notional_usd": 1.0,
                "crypto_stale_min_hold_seconds": 3600,
            }
            pt_trader.time.sleep = lambda *_args, **_kwargs: None

            pt_trader.CryptoAPITrading.manage_trades(bot)

        self.assertEqual(len(sell_calls), 0)
        self.assertTrue(status_rows)
        latest = status_rows[-1]
        stale_events = latest.get("stale_exit_events", []) if isinstance(latest.get("stale_exit_events", []), list) else []
        reasons = [str((row or {}).get("reason", "") or "") for row in stale_events if isinstance(row, dict)]
        self.assertIn("trail_sell_hold_alignment_guard", reasons)
        self.assertTrue(bool(bot.trailing_pm.get("BTC", {}).get("was_above", False)))


if __name__ == "__main__":
    unittest.main()
