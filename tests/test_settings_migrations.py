from __future__ import annotations

import unittest

from app.settings_migrations import CURRENT_SETTINGS_VERSION, migrate_settings
from app.settings_utils import sanitize_settings


class TestSettingsMigrations(unittest.TestCase):
    def test_migrate_legacy_script_paths(self) -> None:
        raw = {
            "settings_schema_version": 1,
            "script_neural_runner2": "pt_thinker.py",
            "script_trader": "pt_trader.py",
            "script_markets_runner": "pt_markets.py",
        }
        out, notes, from_v, to_v = migrate_settings(raw)
        self.assertEqual(from_v, 1)
        self.assertEqual(to_v, int(CURRENT_SETTINGS_VERSION))
        self.assertEqual(str(out.get("script_neural_runner2", "")), "engines/pt_thinker.py")
        self.assertEqual(str(out.get("script_trader", "")), "engines/pt_trader.py")
        self.assertEqual(str(out.get("script_markets_runner", "")), "runtime/pt_markets.py")
        self.assertTrue(isinstance(notes, list))

    def test_sanitize_applies_schema_and_upgrade_notes(self) -> None:
        raw = {
            "settings_schema_version": 1,
            "script_autopilot": "pt_autopilot.py",
            "script_autofix": "pt_autofix.py",
        }
        out = sanitize_settings(raw)
        self.assertEqual(int(out.get("settings_schema_version", 0) or 0), int(CURRENT_SETTINGS_VERSION))
        self.assertNotIn("script_autofix", out)
        notes = list(out.get("settings_upgrade_notes", []) or [])
        self.assertTrue(len(notes) >= 1)
        self.assertFalse(bool(out.get("openai_nightly_review_enabled", True)))
        self.assertEqual(int(out.get("openai_nightly_review_hour_local", -1) or -1), 2)
        self.assertEqual(int(out.get("openai_nightly_review_lookback_days", 0) or 0), 7)
        self.assertFalse(bool(out.get("openai_managed_auto_tuning_enabled", True)))
        self.assertFalse(bool(out.get("openai_nightly_review_apply_tuning_enabled", True)))
        self.assertFalse(bool(out.get("openai_position_review_enabled", True)))
        self.assertEqual(int(out.get("openai_position_review_max_positions", 0) or 0), 48)
        self.assertFalse(bool(out.get("openai_capital_planner_enabled", True)))
        self.assertEqual(int(out.get("openai_capital_planner_max_candidates_per_market", 0) or 0), 3)
        self.assertFalse(bool(out.get("openai_root_cause_enabled", True)))
        self.assertEqual(int(out.get("openai_root_cause_max_incidents", 0) or 0), 300)
        self.assertFalse(bool(out.get("openai_explanations_enabled", True)))
        self.assertEqual(int(out.get("openai_explanations_max_items", 0) or 0), 18)
        self.assertFalse(bool(out.get("openai_strategy_optimizer_enabled", True)))
        self.assertEqual(float(out.get("openai_strategy_optimizer_interval_s", 0.0) or 0.0), 3600.0)
        self.assertFalse(bool(out.get("openai_strategy_optimizer_auto_apply_enabled", True)))
        self.assertFalse(bool(out.get("openai_market_context_enabled", True)))
        self.assertEqual(float(out.get("openai_market_context_interval_s", 0.0) or 0.0), 300.0)
        self.assertEqual(int(out.get("openai_market_context_max_items", 0) or 0), 24)
        self.assertFalse(bool(out.get("openai_postmortem_enabled", True)))
        self.assertEqual(int(out.get("openai_postmortem_max_events", 0) or 0), 5000)
        self.assertTrue(bool(out.get("openai_postmortem_write_report_enabled", False)))
        self.assertFalse(bool(out.get("openai_postmortem_auto_apply_tuning_enabled", True)))

    def test_migrate_v4_adds_market_enable_flags(self) -> None:
        raw = {
            "settings_schema_version": 4,
        }
        out, notes, from_v, to_v = migrate_settings(raw)
        self.assertEqual(from_v, 4)
        self.assertEqual(to_v, int(CURRENT_SETTINGS_VERSION))
        self.assertTrue(bool(out.get("market_crypto_enabled", False)))
        self.assertTrue(bool(out.get("market_stocks_enabled", False)))
        self.assertTrue(bool(out.get("market_forex_enabled", False)))
        self.assertTrue(isinstance(notes, list))


if __name__ == "__main__":
    unittest.main()
