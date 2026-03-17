from __future__ import annotations

import unittest

from app.settings_utils import recommend_market_profile_overrides, sanitize_settings


class TestSettingsSanitize(unittest.TestCase):
    def test_clamps_and_normalizes(self) -> None:
        raw = {
            "coins": [" btc ", "ETH", "eth", "bad-coin", ""],
            "start_allocation_pct": "0.005",
            "market_bg_stocks_interval_s": "1",
            "market_bg_forex_interval_s": "9999",
            "runner_crash_lockout_s": "-2",
            "market_loop_jitter_pct": "2.0",
            "market_settings_reload_interval_s": "0.1",
            "runtime_incidents_max_lines": "99",
            "runtime_events_max_lines": "99999999",
            "stock_scan_open_cooldown_minutes": "-9",
            "stock_scan_close_cooldown_minutes": "999",
            "stock_scan_open_score_mult": "0.1",
            "stock_scan_close_score_mult": "2.0",
            "stock_loss_streak_size_step_pct": "-9",
            "stock_loss_streak_size_floor_pct": "0.01",
            "forex_session_weight_floor": "0.2",
            "forex_session_weight_ceiling": "9.0",
            "forex_loss_streak_size_step_pct": "9.0",
            "forex_loss_streak_size_floor_pct": "2.0",
            "market_chart_cache_symbols": "999",
            "market_chart_cache_bars": "3",
            "market_fallback_scan_max_age_s": "5",
            "market_fallback_snapshot_max_age_s": "999999",
            "kucoin_unsupported_cooldown_s": "2",
            "crypto_price_error_log_cooldown_s": "99999",
            "news_event_enabled": "0",
            "news_event_refresh_s": "10",
            "news_event_stale_max_s": "30",
            "news_event_timeout_s": "999",
            "news_event_max_symbols_per_market": "9999",
            "news_event_max_headlines_per_symbol": "0",
            "stock_news_event_weight": "9",
            "crypto_news_event_weight": "-1",
            "stock_scan_watch_leaders_count": "999",
            "stock_leader_stability_margin_pct": "999",
            "forex_leader_stability_margin_pct": "-4",
            "stock_block_entries_on_cached_scan": "0",
            "forex_block_entries_on_cached_scan": "false",
            "stock_cached_scan_hard_block_age_s": "3",
            "stock_cached_scan_entry_size_mult": "3.0",
            "stock_require_data_quality_ok_for_entries": "0",
            "stock_require_reject_rate_max_pct": "999",
            "stock_replay_adaptive_enabled": "0",
            "stock_replay_adaptive_weight": "2.5",
            "stock_replay_adaptive_step_cap_pct": "1",
            "forex_cached_scan_hard_block_age_s": "5",
            "forex_cached_scan_entry_size_mult": "0.01",
            "forex_require_data_quality_ok_for_entries": "no",
            "forex_require_reject_rate_max_pct": "-5",
            "forex_replay_adaptive_enabled": "no",
            "forex_replay_adaptive_weight": "-5",
            "forex_replay_adaptive_step_cap_pct": "200",
            "runtime_alert_cadence_warn_count": "0",
            "runtime_alert_cadence_crit_count": "0",
            "runtime_alert_cadence_late_warn_pct": "1",
            "runtime_alert_cadence_late_crit_pct": "5",
            "runtime_alert_cadence_min_samples": "1",
            "runtime_alert_cadence_cooldown_s": "1",
            "runtime_alert_market_loop_stale_s": "1",
            "broker_order_retry_after_cap_s": "99999",
            "scanner_quality_max_age_days": "9999",
            "stock_min_price": "10",
            "stock_max_price": "2",
            "stock_min_valid_bars_ratio": "1.4",
            "forex_min_valid_bars_ratio": "-1",
            "market_rollout_stage": "BAD_STAGE",
            "settings_control_mode": "unsupported",
            "settings_profile": "max_profit",
            "stock_universe_mode": "nope",
            "forex_session_mode": "none",
        }
        out = sanitize_settings(raw)
        self.assertEqual(out["coins"], ["BTC", "ETH"])
        self.assertAlmostEqual(float(out["start_allocation_pct"]), 0.5, places=6)
        self.assertEqual(float(out["market_bg_stocks_interval_s"]), 8.0)
        self.assertEqual(float(out["market_bg_forex_interval_s"]), 300.0)
        self.assertEqual(float(out["runner_crash_lockout_s"]), 30.0)
        self.assertEqual(float(out["market_loop_jitter_pct"]), 0.5)
        self.assertEqual(float(out["market_settings_reload_interval_s"]), 1.0)
        self.assertEqual(int(out["runtime_incidents_max_lines"]), 2000)
        self.assertEqual(int(out["runtime_events_max_lines"]), 1000000)
        self.assertEqual(int(out["stock_scan_open_cooldown_minutes"]), 0)
        self.assertEqual(int(out["stock_scan_close_cooldown_minutes"]), 120)
        self.assertEqual(float(out["stock_scan_open_score_mult"]), 0.5)
        self.assertEqual(float(out["stock_scan_close_score_mult"]), 1.0)
        self.assertEqual(float(out["stock_loss_streak_size_step_pct"]), 0.0)
        self.assertEqual(float(out["stock_loss_streak_size_floor_pct"]), 0.10)
        self.assertEqual(float(out["forex_session_weight_floor"]), 0.5)
        self.assertEqual(float(out["forex_session_weight_ceiling"]), 2.0)
        self.assertEqual(float(out["forex_loss_streak_size_step_pct"]), 0.9)
        self.assertEqual(float(out["forex_loss_streak_size_floor_pct"]), 1.0)
        self.assertEqual(int(out["market_chart_cache_symbols"]), 32)
        self.assertEqual(int(out["market_chart_cache_bars"]), 40)
        self.assertEqual(float(out["market_fallback_snapshot_max_age_s"]), 86400.0)
        self.assertEqual(float(out["market_fallback_scan_max_age_s"]), 86400.0)
        self.assertEqual(float(out["kucoin_unsupported_cooldown_s"]), 300.0)
        self.assertEqual(float(out["crypto_price_error_log_cooldown_s"]), 3600.0)
        self.assertFalse(bool(out["news_event_enabled"]))
        self.assertEqual(float(out["news_event_refresh_s"]), 60.0)
        self.assertEqual(float(out["news_event_stale_max_s"]), 60.0)
        self.assertEqual(float(out["news_event_timeout_s"]), 30.0)
        self.assertEqual(int(out["news_event_max_symbols_per_market"]), 200)
        self.assertEqual(int(out["news_event_max_headlines_per_symbol"]), 1)
        self.assertEqual(float(out["stock_news_event_weight"]), 1.0)
        self.assertEqual(float(out["crypto_news_event_weight"]), 0.0)
        self.assertEqual(int(out["stock_scan_watch_leaders_count"]), 20)
        self.assertEqual(float(out["stock_leader_stability_margin_pct"]), 100.0)
        self.assertEqual(float(out["forex_leader_stability_margin_pct"]), 0.0)
        self.assertFalse(bool(out["stock_block_entries_on_cached_scan"]))
        self.assertFalse(bool(out["forex_block_entries_on_cached_scan"]))
        self.assertEqual(int(out["stock_cached_scan_hard_block_age_s"]), 30)
        self.assertEqual(float(out["stock_cached_scan_entry_size_mult"]), 1.0)
        self.assertFalse(bool(out["stock_require_data_quality_ok_for_entries"]))
        self.assertEqual(float(out["stock_require_reject_rate_max_pct"]), 100.0)
        self.assertFalse(bool(out["stock_replay_adaptive_enabled"]))
        self.assertEqual(float(out["stock_replay_adaptive_weight"]), 1.0)
        self.assertEqual(float(out["stock_replay_adaptive_step_cap_pct"]), 5.0)
        self.assertEqual(int(out["forex_cached_scan_hard_block_age_s"]), 30)
        self.assertEqual(float(out["forex_cached_scan_entry_size_mult"]), 0.10)
        self.assertFalse(bool(out["forex_require_data_quality_ok_for_entries"]))
        self.assertEqual(float(out["forex_require_reject_rate_max_pct"]), 0.0)
        self.assertFalse(bool(out["forex_replay_adaptive_enabled"]))
        self.assertEqual(float(out["forex_replay_adaptive_weight"]), 0.0)
        self.assertEqual(float(out["forex_replay_adaptive_step_cap_pct"]), 90.0)
        self.assertEqual(int(out["runtime_alert_cadence_warn_count"]), 1)
        self.assertEqual(int(out["runtime_alert_cadence_crit_count"]), 1)
        self.assertEqual(float(out["runtime_alert_cadence_late_warn_pct"]), 10.0)
        self.assertEqual(float(out["runtime_alert_cadence_late_crit_pct"]), 20.0)
        self.assertEqual(int(out["runtime_alert_cadence_min_samples"]), 2)
        self.assertEqual(int(out["runtime_alert_cadence_cooldown_s"]), 30)
        self.assertEqual(float(out["runtime_alert_market_loop_stale_s"]), 10.0)
        self.assertEqual(float(out["broker_order_retry_after_cap_s"]), 3600.0)
        self.assertEqual(float(out["scanner_quality_max_age_days"]), 365.0)
        self.assertEqual(float(out["stock_max_price"]), 10.0)
        self.assertEqual(float(out["stock_min_valid_bars_ratio"]), 1.0)
        self.assertEqual(float(out["forex_min_valid_bars_ratio"]), 0.0)
        self.assertEqual(out["market_rollout_stage"], "live")
        self.assertEqual(out["settings_control_mode"], "self_managed")
        self.assertEqual(out["settings_profile"], "balanced")
        self.assertEqual(out["stock_universe_mode"], "all_tradable_filtered")
        self.assertEqual(out["forex_session_mode"], "all")

    def test_defaults_on_invalid_input(self) -> None:
        out = sanitize_settings(None)
        self.assertIn("coins", out)
        self.assertTrue(isinstance(out["coins"], list))
        self.assertGreaterEqual(int(out["candles_limit"]), 20)
        self.assertTrue(str(out["script_trader"]).endswith("pt_trader.py"))
        self.assertEqual(str(out.get("settings_control_mode", "")), "self_managed")
        self.assertEqual(str(out.get("settings_profile", "")), "balanced")
        self.assertEqual(int(out.get("stock_symbol_cooldown_minutes", 0) or 0), 15)
        self.assertEqual(int(out.get("stock_symbol_cooldown_min_hits", 0) or 0), 3)
        self.assertEqual(str(out.get("stock_symbol_cooldown_reject_reasons", "")), "data_quality,insufficient_bars")
        self.assertTrue(bool(out.get("news_event_enabled", False)))
        self.assertGreaterEqual(
            float(out.get("news_event_stale_max_s", 0.0) or 0.0),
            float(out.get("news_event_refresh_s", 0.0) or 0.0),
        )

    def test_market_profile_overrides_are_account_aware(self) -> None:
        overrides = recommend_market_profile_overrides(
            "performance",
            settings={"stock_scan_max_symbols": 240},
            crypto_status={
                "equity": "102.78",
                "buying_power": "84.90",
                "market_value": "17.88",
                "open_positions": "0",
            },
            crypto_trader={
                "account": {
                    "total_account_value": 102.78,
                    "buying_power": 84.90,
                    "holdings_sell_value": 17.88,
                },
                "positions": [],
            },
            stock_status={"buying_power": "199748.68", "equity": "99998.68", "open_positions": "2"},
            stock_trader={"account_value_usd": 99998.72, "open_positions": 2},
            forex_status={"buying_power": "97.3632 USD", "nav": 99.956, "open_positions": "1"},
            forex_trader={"account_value_usd": 99.956, "open_positions": 1},
        )
        self.assertEqual(int(overrides["trade_start_level"]), 2)
        self.assertGreater(float(overrides["start_allocation_pct"]), 1.0)
        self.assertGreaterEqual(float(overrides["max_total_exposure_pct"]), 50.0)
        self.assertGreaterEqual(int(overrides["crypto_dynamic_target_count"]), 9)
        self.assertLessEqual(float(overrides["crypto_dynamic_scan_interval_s"]), 120.0)
        self.assertLess(float(overrides["crypto_dynamic_min_projected_edge_pct"]), 0.20)
        self.assertEqual(int(overrides["stock_max_open_positions"]), 8)
        self.assertGreater(float(overrides["stock_trade_notional_usd"]), 200.0)
        self.assertEqual(int(overrides["forex_trade_units"]), 25)
        self.assertEqual(float(overrides["market_max_total_exposure_pct"]), 0.0)
        self.assertEqual(float(overrides["market_bg_stocks_interval_s"]), 20.0)
        self.assertEqual(int(overrides["stock_max_day_trades"]), 1)
        self.assertGreater(float(overrides["stock_profit_target_pct"]), 1.0)
        self.assertTrue(bool(overrides["stock_opening_plan_enabled"]))
        self.assertLessEqual(float(overrides["forex_score_threshold"]), 0.10)


if __name__ == "__main__":
    unittest.main()
