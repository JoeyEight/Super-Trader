from __future__ import annotations

import unittest

from app.automation_policy import build_market_automation_policy, runtime_trust_snapshot, stock_compliance_state, summarize_policy_snapshot


class TestAutomationPolicy(unittest.TestCase):
    def test_runtime_trust_snapshot_degrades_with_alerts_and_health(self) -> None:
        trust = runtime_trust_snapshot(
            {"severity": "warning"},
            {"data_ok": False, "broker_ok": True, "orders_ok": True, "drift_warning": True},
        )
        self.assertEqual(str(trust.get("mode", "")), "cautious")
        self.assertLess(float(trust.get("score", 100.0) or 100.0), 62.0)
        self.assertTrue(bool(trust.get("reasons", [])))

    def test_stock_compliance_under_25k_margin_blocks_when_day_trade_window_is_full(self) -> None:
        compliance = stock_compliance_state(
            {"multiplier": "2", "pattern_day_trader": False},
            equity_usd=12_000.0,
            pdt_equity_threshold_usd=25_000.0,
            day_trades_rolling_5d=3,
            pdt_max_day_trades_rolling_5d=3,
        )
        self.assertEqual(str(compliance.get("account_mode", "")), "margin")
        self.assertTrue(bool(compliance.get("pdt_restricted", False)))
        self.assertTrue(bool(compliance.get("entry_blocked", False)))
        self.assertIn("blocked to avoid pdt", str(compliance.get("entry_block_reason", "")).lower())

    def test_stock_compliance_cash_mode_uses_cash_protection_language(self) -> None:
        compliance = stock_compliance_state(
            {"account_type": "cash", "multiplier": "1"},
            equity_usd=5_000.0,
            pdt_equity_threshold_usd=25_000.0,
            day_trades_rolling_5d=4,
            pdt_max_day_trades_rolling_5d=3,
        )
        self.assertEqual(str(compliance.get("account_mode", "")), "cash")
        self.assertFalse(bool(compliance.get("pdt_restricted", False)))
        self.assertIn("cash-account compliance mode", str(compliance.get("status_text", "")).lower())

    def test_market_policy_scales_with_profile_and_runtime(self) -> None:
        policy = build_market_automation_policy(
            market="forex",
            settings={"market_bg_forex_interval_s": 8.0, "forex_max_total_exposure_pct": 50.0},
            profile_key="aggressive",
            broker_mode="live",
            account_value_usd=25_000.0,
            buying_power_usd=10_000.0,
            open_positions=1,
            runtime_alerts={"severity": "warning"},
            market_health={"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
            compliance_state={},
            reject_rate_pct=40.0,
            reject_rate_limit_pct=90.0,
            fallback_active=False,
            fallback_age_s=0,
            fallback_hard_block_age_s=1200,
            loss_streak=0,
            max_loss_streak=3,
        )
        self.assertEqual(str(policy.get("profile", "")), "aggressive")
        self.assertTrue(bool(policy.get("summary", "")))
        self.assertGreater(float(policy.get("size_multiplier", 0.0) or 0.0), 0.5)
        self.assertLessEqual(float(policy.get("size_multiplier", 2.0) or 2.0), 1.35)
        limits = policy.get("effective_limits", {}) if isinstance(policy.get("effective_limits", {}), dict) else {}
        self.assertEqual(float(limits.get("market_exposure_cap_pct", 0.0) or 0.0), 50.0)
        self.assertEqual(float(limits.get("global_exposure_cap_pct", 0.0) or 0.0), 0.0)
        self.assertEqual(float(limits.get("daily_loss_pct", 0.0) or 0.0), 0.0)

    def test_crypto_policy_exposes_effective_rotation_limits(self) -> None:
        policy = build_market_automation_policy(
            market="crypto",
            settings={
                "crypto_dynamic_scan_interval_s": 20.0,
                "crypto_dynamic_target_count": 12,
                "crypto_dynamic_max_new_per_scan": 3,
                "crypto_dynamic_rotation_cooldown_s": 240.0,
                "crypto_dynamic_min_projected_edge_pct": 0.14,
                "crypto_max_spread_bps": 150.0,
                "crypto_max_open_positions": 10,
            },
            profile_key="max_growth",
            broker_mode="live",
            account_value_usd=15_000.0,
            buying_power_usd=8_000.0,
            open_positions=2,
            runtime_alerts={"severity": "ok"},
            market_health={"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
            compliance_state={},
            reject_rate_pct=12.0,
            reject_rate_limit_pct=85.0,
            fallback_active=False,
            fallback_age_s=0,
            fallback_hard_block_age_s=1800,
            loss_streak=0,
            max_loss_streak=3,
        )
        self.assertEqual(str(policy.get("market", "")), "crypto")
        self.assertIn(str(policy.get("mode", "")), {"aggressive_rotation", "aggressive_guarded"})
        self.assertGreater(float(policy.get("size_multiplier", 0.0) or 0.0), 1.0)
        limits = policy.get("effective_limits", {}) if isinstance(policy.get("effective_limits", {}), dict) else {}
        self.assertGreater(int(limits.get("target_symbols", 0) or 0), 0)
        self.assertGreaterEqual(int(limits.get("max_new_entries_per_scan", 0) or 0), 1)
        self.assertGreaterEqual(float(limits.get("max_spread_bps", 0.0) or 0.0), 150.0)
        self.assertTrue(bool(policy.get("summary", "")))

    def test_policy_snapshot_includes_crypto_row(self) -> None:
        snap = summarize_policy_snapshot(
            {"automation_policy": {"summary": "Stocks summary"}},
            {"automation_policy": {"summary": "Forex summary"}},
            {"automation_policy": {"summary": "Crypto summary", "effective_limits": {"target_symbols": 12}}},
        )
        self.assertIn("crypto", snap)
        self.assertEqual(str((snap.get("crypto", {}) if isinstance(snap.get("crypto", {}), dict) else {}).get("summary", "")), "Crypto summary")


if __name__ == "__main__":
    unittest.main()
