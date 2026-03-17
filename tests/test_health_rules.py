from __future__ import annotations

import unittest

from app.health_rules import evaluate_runtime_alerts


class TestHealthRules(unittest.TestCase):
    def test_warn_on_reject_pressure(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {
                "stocks": {"reject_rate_pct": 70.0},
                "forex": {"reject_rate_pct": 12.0},
            },
            "incidents_last_200": {"count": 1, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
        }
        settings = {"runtime_alert_scan_reject_warn_pct": 65.0, "runtime_alert_scan_reject_crit_pct": 90.0}
        out = evaluate_runtime_alerts(state, settings)
        self.assertEqual(out["severity"], "warn")
        self.assertIn("scan_reject_pressure", out["reasons"])

    def test_cooldown_dominated_rejects_do_not_trigger_pressure(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {
                "stocks": {
                    "reject_rate_pct": 100.0,
                    "leaders_total": 2,
                    "reject_dominant_reason": "cooldown",
                    "reject_dominant_ratio_pct": 92.0,
                },
                "forex": {"reject_rate_pct": 0.0, "leaders_total": 6},
            },
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0, "warning": 0, "info": 40}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_scan_reject_warn_pct": 65.0, "runtime_alert_scan_reject_crit_pct": 85.0})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("scan_reject_pressure", out["reasons"])
        self.assertEqual(float(out.get("metrics", {}).get("stocks_reject_rate_pct", -1.0)), 0.0)
        self.assertEqual(float(out.get("metrics", {}).get("stocks_reject_rate_raw_pct", -1.0)), 100.0)

    def test_info_only_incidents_do_not_raise_warn_or_critical(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 180, "by_severity": {"info": 180, "warning": 0, "error": 0}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_incident_warn_count": 8, "runtime_alert_incident_crit_count": 20})
        self.assertEqual(out["severity"], "ok")

    def test_critical_on_failed_checks(self) -> None:
        state = {
            "checks": {"ok": False, "warnings": ["x"]},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertEqual(out["severity"], "critical")
        self.assertIn("startup_checks_failed", out["reasons"])

    def test_missing_startup_checks_artifact_does_not_raise_false_critical(self) -> None:
        state = {
            "checks": {"ok": False, "warnings": [], "errors": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("startup_checks_failed", out["reasons"])
        self.assertTrue(bool(out.get("metrics", {}).get("startup_checks_indeterminate", False)))

    def test_warn_on_drift_spike(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 10.0}, "forex": {"reject_rate_pct": 10.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "scan_drift": {"active": [{"market": "stocks"}]},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_drift_spike_warn_count": 1})
        self.assertEqual(out["severity"], "warn")
        self.assertIn("scanner_reject_spike", out["reasons"])

    def test_warn_on_cadence_drift_pressure(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "scan_cadence": {
                "active": [{"market": "forex", "level": "warning"}],
            },
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_cadence_warn_count": 1, "runtime_alert_cadence_crit_count": 2})
        self.assertEqual(out["severity"], "warn")
        self.assertIn("cadence_drift_pressure", out["reasons"])

    def test_cadence_critical_is_not_promoted_without_other_instability(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0, "warning": 0}},
            "autopilot": {"api_unstable": False},
            "scan_cadence": {
                "active": [
                    {"market": "stocks", "level": "critical"},
                    {"market": "forex", "level": "critical"},
                ],
            },
            "market_loop": {"age_s": 5},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_cadence_warn_count": 1, "runtime_alert_cadence_crit_count": 2})
        self.assertEqual(out["severity"], "warn")
        self.assertEqual(int(out.get("metrics", {}).get("scan_cadence_critical_effective_count", -1)), 0)
        self.assertTrue(bool(out.get("metrics", {}).get("scan_cadence_critical_suppressed", False)))

    def test_warn_on_market_loop_stale(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "market_loop": {"age_s": 240},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_market_loop_stale_s": 90.0})
        self.assertEqual(out["severity"], "warn")
        self.assertIn("market_loop_stale", out["reasons"])

    def test_market_loop_stale_is_suppressed_during_startup_grace(self) -> None:
        state = {
            "ts": 1000,
            "checks": {"ok": True, "warnings": [], "ts": 950},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "market_loop": {"age_s": 240},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_market_loop_stale_s": 90.0, "runtime_alert_startup_grace_s": 120.0})
        self.assertNotIn("market_loop_stale", out["reasons"])

    def test_high_reject_without_dominant_reason_caps_pressure_when_leaders_exist(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {
                "stocks": {
                    "reject_rate_pct": 100.0,
                    "leaders_total": 2,
                    "scores_total": 3,
                    "reject_dominant_reason": "",
                },
                "forex": {"reject_rate_pct": 0.0},
            },
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0, "warning": 0, "info": 25}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_scan_reject_warn_pct": 65.0, "runtime_alert_reject_unknown_dom_cap_pct": 64.0})
        self.assertLess(float(out.get("metrics", {}).get("stocks_reject_rate_pct", 100.0)), 65.0)
        self.assertNotIn("scan_reject_pressure", out.get("reasons", []))

    def test_liquidity_dominated_rejects_with_live_leaders_do_not_raise_runtime_pressure(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {
                "stocks": {
                    "reject_rate_pct": 100.0,
                    "leaders_total": 1,
                    "scores_total": 2,
                    "reject_dominant_reason": "liquidity",
                    "reject_dominant_ratio_pct": 78.0,
                },
                "forex": {"reject_rate_pct": 0.0},
            },
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0, "warning": 0, "info": 10}},
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_scan_reject_warn_pct": 65.0})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("scan_reject_pressure", out.get("reasons", []))
        self.assertLess(float(out.get("metrics", {}).get("stocks_reject_rate_pct", 100.0)), 65.0)

    def test_warn_on_exposure_concentration(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "exposure_map": {"top_positions": [{"pct_of_total_exposure": 66.0, "pct_of_market_account": 12.0}]},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_exposure_concentration_warn_pct": 55.0})
        self.assertEqual(out["severity"], "warn")
        self.assertIn("exposure_concentration", out["reasons"])

    def test_exposure_concentration_is_suppressed_for_tiny_position_vs_account(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "exposure_map": {"top_positions": [{"pct_of_total_exposure": 79.2, "pct_of_market_account": 0.05}]},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_exposure_concentration_warn_pct": 55.0})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("exposure_concentration", out["reasons"])
        self.assertEqual(float(out.get("metrics", {}).get("top_exposure_pct_of_market_account", -1.0)), 0.05)

    def test_warn_on_execution_guard_active(self) -> None:
        state = {
            "ts": 1000,
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "execution_guard": {"markets": {"stocks": {"disabled_until": 1300}}},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertEqual(out["severity"], "warn")
        self.assertIn("execution_temporarily_disabled", out["reasons"])
        self.assertTrue(isinstance(out.get("quickfix_suggestions", []), list))
        self.assertGreaterEqual(len(out.get("quickfix_suggestions", [])), 1)

    def test_critical_on_drawdown_guard(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "drawdown_guard": {"triggered_recent": True},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertEqual(out["severity"], "critical")
        self.assertIn("drawdown_guard_triggered", out["reasons"])

    def test_critical_on_stop_flag_active(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "stop_flag": {"active": True},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertEqual(out["severity"], "critical")
        self.assertIn("stop_flag_active", out["reasons"])

    def test_shadow_scorecard_blocked_is_suppressed_in_live_guarded(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "shadow_scorecards": {
                "stocks": {"promotion_gate": "BLOCK"},
                "forex": {"promotion_gate": "BLOCK"},
            },
        }
        out = evaluate_runtime_alerts(state, {"market_rollout_stage": "live_guarded"})
        self.assertNotIn("shadow_scorecard_blocked", out["reasons"])
        self.assertEqual(out["severity"], "ok")

    def test_notification_center_critical_does_not_self_promote_alerts(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {"count": 0, "by_severity": {"error": 0}},
            "autopilot": {"api_unstable": False},
            "notification_center": {"by_severity": {"critical": 4}},
        }
        out = evaluate_runtime_alerts(state, {})
        self.assertNotIn("notification_center_critical", out["reasons"])
        self.assertEqual(out["severity"], "ok")

    def test_resolved_startup_incidents_do_not_count_as_runtime_warning(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": [], "errors": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {
                "count": 2,
                "by_severity_1h": {"warning": 2, "error": 0},
                "by_event_severity_1h": {"runner_startup_check": {"warning": 2}},
            },
            "autopilot": {"api_unstable": False},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_incident_warn_count": 1})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("startup_warnings", out["reasons"])

    def test_resolved_market_loop_incidents_do_not_count_as_runtime_warning(self) -> None:
        state = {
            "checks": {"ok": True, "warnings": [], "errors": []},
            "scan_health": {"stocks": {"reject_rate_pct": 0.0}, "forex": {"reject_rate_pct": 0.0}},
            "incidents_last_200": {
                "count": 2,
                "by_severity_1h": {"warning": 2, "error": 0},
                "by_event_severity_1h": {
                    "runner_market_loop_status_stale": {"warning": 1},
                    "runner_market_loop_restart": {"warning": 1},
                },
            },
            "autopilot": {"api_unstable": False},
            "market_loop": {"age_s": 10},
        }
        out = evaluate_runtime_alerts(state, {"runtime_alert_incident_warn_count": 1, "runtime_alert_market_loop_stale_s": 90.0})
        self.assertEqual(out["severity"], "ok")
        self.assertNotIn("market_loop_stale", out["reasons"])


if __name__ == "__main__":
    unittest.main()
