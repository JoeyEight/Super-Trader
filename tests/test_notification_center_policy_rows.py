from __future__ import annotations

import unittest

from app.notification_center import build_notification_center_payload


class TestNotificationCenterPolicyRows(unittest.TestCase):
    def test_policy_rows_include_summary_and_compliance_text(self) -> None:
        runtime_state = {
            "ts": 1_700_001_200,
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "automation_policy": {
                "stocks": {
                    "summary": "Max Growth preset; scan every 12s",
                    "allow_new_entries": False,
                    "runtime_trust_score": 60.0,
                    "compliance_status": "Stock automation is running in under-$25k compliance mode",
                },
                "forex": {},
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        stock_rows = [row for row in rows if str(row.get("source", "")) == "automation_policy" and str(row.get("market", "")) == "stocks"]
        self.assertTrue(stock_rows)
        stock_row = stock_rows[0]
        self.assertEqual(str(stock_row.get("severity", "")), "warning")
        self.assertIn("scan every 12s", str(stock_row.get("message", "")).lower())
        self.assertIn("compliance mode", str(stock_row.get("message", "")).lower())
        self.assertEqual(str(stock_row.get("message", "")).lower().count("compliance mode"), 1)

    def test_policy_rows_escalate_to_critical_when_runtime_trust_is_low(self) -> None:
        runtime_state = {
            "ts": 1_700_001_300,
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "automation_policy": {
                "crypto": {},
                "stocks": {},
                "forex": {
                    "summary": "Aggressive preset; scanner reject pressure elevated",
                    "allow_new_entries": True,
                    "runtime_trust_score": 22.0,
                    "compliance_status": "",
                },
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        forex_rows = [row for row in rows if str(row.get("source", "")) == "automation_policy" and str(row.get("market", "")) == "forex"]
        self.assertTrue(forex_rows)
        self.assertEqual(str(forex_rows[0].get("severity", "")), "critical")

    def test_policy_rows_include_crypto_policy_messages(self) -> None:
        runtime_state = {
            "ts": 1_700_001_400,
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "automation_policy": {
                "crypto": {
                    "summary": "Max Growth preset; scan every 20s; rotation every 240s",
                    "allow_new_entries": True,
                    "runtime_trust_score": 78.0,
                    "compliance_status": "",
                },
                "stocks": {},
                "forex": {},
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        crypto_rows = [row for row in rows if str(row.get("source", "")) == "automation_policy" and str(row.get("market", "")) == "crypto"]
        self.assertTrue(crypto_rows)
        self.assertIn("rotation every 240s", str(crypto_rows[0].get("message", "")).lower())

    def test_cross_market_opportunity_rows_are_included(self) -> None:
        runtime_state = {
            "ts": 1_700_001_500,
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "cross_market_opportunity": {
                "active": True,
                "summary": "Stocks candidate deprioritized because crypto has higher opportunity quality and capital is constrained",
                "best_market": "crypto",
                "deprioritized_markets": [
                    {
                        "market": "stocks",
                        "decision": "deprioritize",
                        "summary": "Stocks candidate deprioritized because crypto has higher opportunity quality and capital is constrained",
                    }
                ],
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        alloc_rows = [row for row in rows if str(row.get("source", "")) == "opportunity_allocator"]
        self.assertTrue(alloc_rows)
        self.assertTrue(any(str(row.get("market", "")) == "global" for row in alloc_rows))
        self.assertTrue(any(str(row.get("market", "")) == "stocks" for row in alloc_rows))

    def test_cross_market_opportunity_row_escalates_on_block(self) -> None:
        runtime_state = {
            "ts": 1_700_001_600,
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "cross_market_opportunity": {
                "active": True,
                "summary": "Blocked due to concentration risk",
                "best_market": "forex",
                "deprioritized_markets": [
                    {"market": "stocks", "decision": "block", "summary": "Blocked due to concentration risk"},
                ],
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[], max_items=32)
        rows = out.get("items", []) if isinstance(out.get("items", []), list) else []
        global_rows = [row for row in rows if str(row.get("source", "")) == "opportunity_allocator" and str(row.get("market", "")) == "global"]
        self.assertTrue(global_rows)
        self.assertEqual(str(global_rows[0].get("severity", "")), "warning")


if __name__ == "__main__":
    unittest.main()
