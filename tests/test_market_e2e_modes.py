from __future__ import annotations

import os
import tempfile
import unittest
import json
from unittest.mock import patch

import runtime.pt_markets as pt_markets


class TestMarketE2EModes(unittest.TestCase):
    def test_scanner_reject_rate_prefers_effective_quality_rate(self) -> None:
        rate = pt_markets._scanner_reject_rate_for_alerts(
            {
                "reject_summary": {"reject_rate_pct": 95.0},
                "universe_quality": {"reject_rate_pct": 58.25, "reject_rate_raw_pct": 95.0},
            }
        )
        self.assertAlmostEqual(rate, 58.25, places=3)

    def test_scanner_reject_rate_falls_back_to_raw_summary(self) -> None:
        rate = pt_markets._scanner_reject_rate_for_alerts({"reject_summary": {"reject_rate_pct": 67.5}})
        self.assertAlmostEqual(rate, 67.5, places=3)

    def test_scanner_reject_rate_ignores_cooldown_dominant_when_leaders_survive(self) -> None:
        rate = pt_markets._scanner_reject_rate_for_alerts(
            {
                "reject_summary": {
                    "reject_rate_pct": 88.8,
                    "dominant_reason": "cooldown",
                    "dominant_ratio_pct": 92.0,
                },
                "leaders_total": 5,
                "scores_total": 5,
            }
        )
        self.assertAlmostEqual(rate, 0.0, places=3)

    def test_update_scan_reject_drift_clears_active_alert_when_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            scan_drift_path = os.path.join(td, "scan_drift_alerts.json")
            with open(scan_drift_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 1,
                        "markets": {
                            "forex": {
                                "history": [0.0] * 12,
                                "updated_ts": 1,
                                "last_alert_ts": 0,
                            }
                        },
                        "active": [],
                    },
                    f,
                )
            settings = {
                "runtime_alert_reject_spike_min_samples": 6,
                "runtime_alert_reject_spike_min_rate_pct": 25.0,
                "runtime_alert_reject_spike_delta_pct": 25.0,
                "runtime_alert_reject_spike_ratio": 2.0,
            }
            with patch.object(pt_markets, "SCAN_DRIFT_PATH", scan_drift_path), patch.object(pt_markets, "_incident", return_value=None):
                pt_markets._update_scan_reject_drift("forex", 83.33, settings, "READY")
                pt_markets._update_scan_reject_drift("forex", 0.0, settings, "READY")

            with open(scan_drift_path, "r", encoding="utf-8") as f:
                out = json.load(f)
            active = out.get("active", [])
            self.assertIsInstance(active, list)
            self.assertFalse(any(str(row.get("market", "")).strip().lower() == "forex" for row in active if isinstance(row, dict)))

    def test_stocks_paper_mode_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_bg_stocks_interval_s": 15.0,
                "market_fallback_scan_max_age_s": 300.0,
                "alpaca_paper_mode": True,
            }
            with patch.object(pt_markets, "HUB_DATA_DIR", td), patch.object(pt_markets, "EXEC_GUARD_PATH", os.path.join(td, "broker_execution_guard.json")), patch.object(
                pt_markets, "SCAN_DRIFT_PATH", os.path.join(td, "scan_drift_alerts.json")
            ), patch.object(
                pt_markets, "CADENCE_DRIFT_PATH", os.path.join(td, "scanner_cadence_drift.json")
            ), patch.object(
                pt_markets, "_incident", return_value=None
            ), patch.object(
                pt_markets, "_update_sla_metrics", return_value=None
            ), patch.object(
                pt_markets, "_record_guard_result", return_value={"active": False}
            ), patch.object(
                pt_markets, "market_guard_status", return_value={"active": False}
            ), patch.object(
                pt_markets, "run_stock_scan", return_value={"state": "READY", "reject_summary": {"reject_rate_pct": 12.0}}
            ), patch.object(
                pt_markets, "run_stock_trader_step", return_value={"state": "READY", "msg": "ok"}
            ):
                out = pt_markets._run_stocks(settings)
            self.assertTrue(bool(out.get("scan_ok", False)))
            self.assertTrue(bool(out.get("step_ok", False)))
            self.assertEqual(str(out.get("scan_state", "")), "READY")

    def test_stocks_reject_drift_uses_effective_scanner_pressure_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_bg_stocks_interval_s": 15.0,
                "market_fallback_scan_max_age_s": 300.0,
                "alpaca_paper_mode": True,
            }
            with patch.object(pt_markets, "HUB_DATA_DIR", td), patch.object(pt_markets, "EXEC_GUARD_PATH", os.path.join(td, "broker_execution_guard.json")), patch.object(
                pt_markets, "SCAN_DRIFT_PATH", os.path.join(td, "scan_drift_alerts.json")
            ), patch.object(
                pt_markets, "CADENCE_DRIFT_PATH", os.path.join(td, "scanner_cadence_drift.json")
            ), patch.object(
                pt_markets, "_incident", return_value=None
            ), patch.object(
                pt_markets, "_update_sla_metrics", return_value=None
            ), patch.object(
                pt_markets, "_record_guard_result", return_value={"active": False}
            ), patch.object(
                pt_markets, "market_guard_status", return_value={"active": False}
            ), patch.object(
                pt_markets,
                "run_stock_scan",
                return_value={
                    "state": "READY",
                    "reject_summary": {"reject_rate_pct": 95.0},
                    "universe_quality": {"reject_rate_pct": 41.0, "reject_rate_raw_pct": 95.0},
                },
            ), patch.object(
                pt_markets, "run_stock_trader_step", return_value={"state": "READY", "msg": "ok"}
            ), patch.object(
                pt_markets, "_update_scan_reject_drift", return_value={}
            ) as reject_drift_mock:
                out = pt_markets._run_stocks(settings)
            self.assertTrue(bool(out.get("scan_ok", False)))
            self.assertTrue(bool(out.get("step_ok", False)))
            self.assertEqual(str(out.get("scan_state", "")), "READY")
            reject_drift_mock.assert_called()
            self.assertAlmostEqual(float(reject_drift_mock.call_args.args[1]), 41.0, places=3)

    def test_forex_practice_mode_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_bg_forex_interval_s": 10.0,
                "market_fallback_scan_max_age_s": 300.0,
                "oanda_practice_mode": True,
            }
            with patch.object(pt_markets, "HUB_DATA_DIR", td), patch.object(pt_markets, "EXEC_GUARD_PATH", os.path.join(td, "broker_execution_guard.json")), patch.object(
                pt_markets, "SCAN_DRIFT_PATH", os.path.join(td, "scan_drift_alerts.json")
            ), patch.object(
                pt_markets, "CADENCE_DRIFT_PATH", os.path.join(td, "scanner_cadence_drift.json")
            ), patch.object(
                pt_markets, "_incident", return_value=None
            ), patch.object(
                pt_markets, "_update_sla_metrics", return_value=None
            ), patch.object(
                pt_markets, "_record_guard_result", return_value={"active": False}
            ), patch.object(
                pt_markets, "market_guard_status", return_value={"active": False}
            ), patch.object(
                pt_markets, "run_forex_scan", return_value={"state": "READY", "reject_summary": {"reject_rate_pct": 14.0}}
            ), patch.object(
                pt_markets, "run_forex_trader_step", return_value={"state": "READY", "msg": "ok"}
            ):
                out = pt_markets._run_forex(settings)
            self.assertTrue(bool(out.get("scan_ok", False)))
            self.assertTrue(bool(out.get("step_ok", False)))
            self.assertEqual(str(out.get("scan_state", "")), "READY")


if __name__ == "__main__":
    unittest.main()
