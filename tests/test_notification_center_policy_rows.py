from __future__ import annotations

import unittest

from app.notification_center import build_notification_center_payload


def _base_runtime(ts: int = 1_700_010_000) -> dict:
    return {
        "ts": ts,
        "alerts": {"severity": "ok", "reasons": [], "hints": []},
    }


class TestNotificationCenterPolicyRows(unittest.TestCase):
    def test_non_actionable_info_rows_are_suppressed(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["automation_policy"] = {
            "crypto": {
                "summary": "Max Growth preset; scan every 20s; rotation every 240s",
                "runtime_trust_score": 78.0,
                "compliance_status": "",
            }
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        self.assertFalse(rows)

    def test_critical_policy_row_is_kept(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["automation_policy"] = {
            "forex": {
                "summary": "Aggressive preset; scanner reject pressure elevated",
                "runtime_trust_score": 22.0,
                "compliance_status": "",
            }
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        self.assertTrue(rows)
        self.assertEqual(str(rows[0].get("severity", "")), "critical")
        self.assertEqual(str(rows[0].get("source", "")), "automation_policy")

    def test_openai_soft_fallback_info_is_suppressed(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_position_review"] = {
            "enabled": True,
            "running": False,
            "status": "timeout",
            "summary": "AI position review timed out; local logic remains active.",
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        self.assertFalse(rows)

    def test_openai_hard_error_is_warning(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_market_context"] = {
            "enabled": True,
            "running": False,
            "status": "schema_validation_failed",
            "summary": "AI market context schema validation failed.",
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        self.assertTrue(rows)
        row = rows[0]
        self.assertEqual(str(row.get("severity", "")), "warning")
        self.assertEqual(str(row.get("source", "")), "openai_market_context")

    def test_openai_nightly_applied_changes_are_emitted(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_nightly_review"] = {
            "enabled": True,
            "running": False,
            "status": "ok",
            "summary": "Nightly review complete.",
            "applied_tuning_count": 1,
            "persisted_verified_count": 1,
            "applied_tuning": [
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "old_value": 45.0,
                    "new_value": 35.0,
                    "confidence": 0.91,
                    "reason": "Reduce concentration for current account tier.",
                }
            ],
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=40)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        nightly_rows = [row for row in rows if str(row.get("source", "")) == "openai_nightly_review"]
        self.assertTrue(nightly_rows)
        self.assertTrue(any(bool(row.get("change_applied", False)) for row in nightly_rows))
        self.assertTrue(any("stock_max_total_exposure_pct" in str(row.get("title", "")) for row in nightly_rows))
        self.assertTrue(any("verified 1 persisted" in str(row.get("message", "")).lower() for row in nightly_rows))

    def test_openai_strategy_optimizer_applied_changes_are_emitted(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_strategy_optimizer"] = {
            "enabled": True,
            "running": False,
            "status": "ok",
            "summary": "Strategy optimizer updated.",
            "applied_tuning_count": 1,
            "applied_tuning": [
                {
                    "setting_key": "market_max_total_exposure_pct",
                    "old_value": 50.0,
                    "new_value": 40.0,
                    "confidence": 0.9,
                    "reason": "Reduce portfolio concentration pressure.",
                }
            ],
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=40)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        optimizer_rows = [row for row in rows if str(row.get("source", "")) == "openai_strategy_optimizer"]
        self.assertTrue(optimizer_rows)
        self.assertTrue(any(bool(row.get("change_applied", False)) for row in optimizer_rows))
        self.assertTrue(any("market_max_total_exposure_pct" in str(row.get("title", "")) for row in optimizer_rows))

    def test_openai_postmortem_applied_changes_are_emitted(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_postmortem_analysis"] = {
            "enabled": True,
            "running": False,
            "status": "ok",
            "summary": "Postmortem completed.",
            "applied_tuning_count": 1,
            "applied_tuning": [
                {
                    "setting_key": "forex_scan_interval_seconds",
                    "old_value": 8.0,
                    "new_value": 10.0,
                    "confidence": 0.86,
                    "reason": "Reduce reject-driven churn.",
                }
            ],
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=40)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        pm_rows = [row for row in rows if str(row.get("source", "")) == "openai_postmortem_analysis"]
        self.assertTrue(pm_rows)
        self.assertTrue(any(bool(row.get("change_applied", False)) for row in pm_rows))
        self.assertTrue(any("forex_scan_interval_seconds" in str(row.get("title", "")) for row in pm_rows))

    def test_openai_applied_allocator_decision_row_is_kept(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["cross_market_opportunity"] = {
            "active": True,
            "summary": "Crypto selected as best current opportunity.",
            "best_market": "crypto",
            "deprioritized_markets": [],
            "openai_decision": {
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "AI portfolio decision: Crypto ranked highest current opportunity.",
                "applied": True,
                "position_actions": [],
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=40)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        ai_rows = [
            row
            for row in rows
            if str(row.get("source", "")) == "opportunity_allocator"
            and "AI portfolio decision applied" in str(row.get("title", ""))
        ]
        self.assertTrue(ai_rows)
        self.assertTrue(all(bool(row.get("change_applied", False)) for row in ai_rows))

    def test_stale_openai_incident_is_suppressed_when_runtime_ok(self) -> None:
        runtime_state = _base_runtime()
        runtime_state["openai_capital_planner"] = {
            "enabled": True,
            "running": False,
            "status": "ok",
            "summary": "Planner healthy",
        }
        incidents = [
            {
                "ts": 1_700_009_995,
                "severity": "warning",
                "event": "openai_capital_planner_status",
                "msg": "OpenAI capital planner completed with status=http_error",
                "details": {"component": "runner"},
            }
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents, max_items=40)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        stale_rows = [row for row in rows if str(row.get("source", "")) == "incidents"]
        self.assertFalse(stale_rows)


if __name__ == "__main__":
    unittest.main()
