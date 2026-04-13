from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import runtime.pt_markets as pt_markets


class TestMarketsLoopStatus(unittest.TestCase):
    def test_jittered_interval_bounds(self) -> None:
        base = 10.0
        for _ in range(100):
            v = pt_markets._jittered_interval(base, 0.25)
            self.assertGreaterEqual(v, 7.5)
            self.assertLessEqual(v, 12.5)

    def test_write_loop_status_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "market_loop_status.json")
            with patch.object(pt_markets, "MARKET_LOOP_STATUS_PATH", path):
                pt_markets._write_loop_status({"ts": 123, "ok": True})
            self.assertTrue(os.path.isfile(path))
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.assertEqual(int(obj.get("ts", 0) or 0), 123)
            self.assertTrue(bool(obj.get("ok", False)))

    def test_flush_loop_status_sets_heartbeat_and_updates_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "market_loop_status.json")
            loop_status = {"ts": 10, "phase": "idle"}
            with patch.object(pt_markets, "MARKET_LOOP_STATUS_PATH", path):
                pt_markets._flush_loop_status(
                    loop_status,
                    threading.Lock(),
                    now_ts=321,
                    phase="stocks_scan",
                    phase_started_ts=320,
                )
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.assertEqual(int(obj.get("ts", 0) or 0), 321)
            self.assertEqual(int(obj.get("heartbeat_ts", 0) or 0), 321)
            self.assertEqual(str(obj.get("phase", "")), "stocks_scan")
            self.assertEqual(int(obj.get("phase_started_ts", 0) or 0), 320)

    def test_update_scan_cadence_drift_creates_active_alert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scanner_cadence_drift.json")
            with patch.object(pt_markets, "CADENCE_DRIFT_PATH", path), patch.object(pt_markets, "_incident") as mock_incident:
                settings = {
                    "runtime_alert_cadence_min_samples": 2,
                    "runtime_alert_cadence_late_warn_pct": 50.0,
                    "runtime_alert_cadence_late_crit_pct": 100.0,
                    "runtime_alert_cadence_cooldown_s": 1,
                }
                pt_markets._update_scan_cadence_drift("stocks", 100, 10.0, settings, "READY")
                pt_markets._update_scan_cadence_drift("stocks", 130, 10.0, settings, "READY")
                out = pt_markets._update_scan_cadence_drift("stocks", 160, 10.0, settings, "READY")
                self.assertTrue(bool(out.get("late", False)))
                self.assertIn(str(out.get("level", "")), {"warning", "critical"})
                self.assertTrue(os.path.isfile(path))
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.assertGreaterEqual(len(list(payload.get("active", []) or [])), 1)
                self.assertGreaterEqual(mock_incident.call_count, 1)

    def test_update_scan_cadence_drift_requires_consecutive_late_samples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scanner_cadence_drift.json")
            with patch.object(pt_markets, "CADENCE_DRIFT_PATH", path), patch.object(pt_markets, "_incident") as mock_incident:
                settings = {
                    "runtime_alert_cadence_min_samples": 3,
                    "runtime_alert_cadence_late_warn_pct": 50.0,
                    "runtime_alert_cadence_late_crit_pct": 100.0,
                    "runtime_alert_cadence_cooldown_s": 1,
                }
                pt_markets._update_scan_cadence_drift("stocks", 100, 10.0, settings, "READY")
                pt_markets._update_scan_cadence_drift("stocks", 130, 10.0, settings, "READY")
                third = pt_markets._update_scan_cadence_drift("stocks", 160, 10.0, settings, "READY")
                self.assertEqual(int(third.get("late_streak", 0) or 0), 2)
                self.assertEqual(mock_incident.call_count, 0)
                fourth = pt_markets._update_scan_cadence_drift("stocks", 190, 10.0, settings, "READY")
                self.assertTrue(bool(fourth.get("triggered", False)))
                self.assertGreaterEqual(mock_incident.call_count, 1)

    def test_update_scan_cadence_drift_resets_streak_on_on_time_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "scanner_cadence_drift.json")
            with patch.object(pt_markets, "CADENCE_DRIFT_PATH", path), patch.object(pt_markets, "_incident") as mock_incident:
                settings = {
                    "runtime_alert_cadence_min_samples": 2,
                    "runtime_alert_cadence_late_warn_pct": 50.0,
                    "runtime_alert_cadence_late_crit_pct": 100.0,
                    "runtime_alert_cadence_cooldown_s": 1,
                }
                pt_markets._update_scan_cadence_drift("stocks", 100, 10.0, settings, "READY")
                pt_markets._update_scan_cadence_drift("stocks", 130, 10.0, settings, "READY")
                reset = pt_markets._update_scan_cadence_drift("stocks", 140, 10.0, settings, "READY")
                self.assertFalse(bool(reset.get("late", False)))
                self.assertEqual(int(reset.get("late_streak", -1)), 0)
                pt_markets._update_scan_cadence_drift("stocks", 170, 10.0, settings, "READY")
                before_trigger = pt_markets._update_scan_cadence_drift("stocks", 200, 10.0, settings, "READY")
                self.assertTrue(bool(before_trigger.get("triggered", False)))
                self.assertEqual(mock_incident.call_count, 1)

    def test_cached_status_fallback_age_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "status.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"ts": 150, "state": "READY"}, f)
            out = pt_markets._cached_status_fallback(path, max_age_s=60.0, now_ts=200)
            self.assertTrue(bool(out))
            self.assertEqual(int(out.get("fallback_age_s", -1)), 50)
            stale = pt_markets._cached_status_fallback(path, max_age_s=10.0, now_ts=200)
            self.assertEqual(stale, {})

    def test_effective_market_cycle_interval_expands_for_overrun(self) -> None:
        settings = {
            "market_scan_overrun_interval_mult": 1.10,
            "market_scan_overrun_min_pause_s": 1.0,
        }
        fast = pt_markets._effective_market_cycle_interval(12.0, 6.0, settings, market="stocks")
        slow = pt_markets._effective_market_cycle_interval(12.0, 121.0, settings, market="stocks")
        self.assertEqual(float(fast), 12.0)
        self.assertGreaterEqual(float(slow), 133.0)

    def test_sync_loop_phase_prefers_oldest_active_worker(self) -> None:
        loop_status = {
            "workers": {
                "stocks": {"active": True, "phase": "stocks_scan", "phase_started_ts": 100, "phase_detail": "stocks_cycle"},
                "forex": {"active": True, "phase": "forex_scan", "phase_started_ts": 120, "phase_detail": "forex_cycle"},
            }
        }
        pt_markets._sync_loop_phase(loop_status)
        self.assertEqual(str(loop_status.get("phase", "")), "stocks_scan")
        self.assertEqual(int(loop_status.get("phase_started_ts", -1) or -1), 100)
        self.assertEqual(str(loop_status.get("phase_detail", "")), "stocks_cycle")


if __name__ == "__main__":
    unittest.main()
