from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_capital_planner import request_openai_capital_planner, run_openai_capital_planner


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAICapitalPlannerTests(unittest.TestCase):
    def _valid_plan(self) -> dict:
        return {
            "summary": "Prefer crypto opportunities while preserving reserve capital.",
            "portfolio_plan": {
                "mode": "mixed",
                "preferred_market_order": ["crypto", "stocks", "forex"],
                "reserve_capital_pct": 20.0,
                "capital_constrained": True,
                "reason": "Crypto quality currently leads while exposure remains bounded.",
            },
            "market_actions": [
                {
                    "market": "crypto",
                    "action": "prefer",
                    "confidence": 0.81,
                    "suggested_capital_share_pct": 55.0,
                    "reason": "Highest confidence setup quality and healthy trust.",
                },
                {
                    "market": "stocks",
                    "action": "deprioritize",
                    "confidence": 0.74,
                    "suggested_capital_share_pct": 20.0,
                    "reason": "Current stock candidates are weaker than crypto.",
                },
            ],
            "global_risks": ["capital_constrained"],
        }

    def _seed_hub(self, root: str) -> str:
        hub = os.path.join(root, "hub_data")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(os.path.join(hub, "openai"), exist_ok=True)
        os.makedirs(os.path.join(hub, "stocks"), exist_ok=True)
        os.makedirs(os.path.join(hub, "forex"), exist_ok=True)
        os.makedirs(os.path.join(hub, "crypto"), exist_ok=True)
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 1000.0,
                    "buying_power_usd": 700.0,
                    "exposure_usd": 250.0,
                    "open_positions": 2,
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 84.0}},
                    "trade_quality": {"confidence_score": 78.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
            json.dump({"ranked": [{"symbol": "BTC", "score": 1.2, "spread_bps": 6.0}]}, f)
        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 1000.0,
                    "buying_power_usd": 700.0,
                    "exposure_usd": 90.0,
                    "open_positions": 1,
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 70.0}},
                    "trade_quality": {"confidence_score": 62.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "stocks", "stock_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": [{"symbol": "AAPL", "side": "long", "score": 0.35}]}, f)
        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 1000.0,
                    "margin_available_usd": 650.0,
                    "exposure_usd": 60.0,
                    "open_positions": 1,
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 68.0}},
                    "trade_quality": {"confidence_score": 58.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "forex", "forex_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.22}]}, f)
        with open(os.path.join(hub, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(hub, "stocks", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(hub, "forex", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        return hub

    def test_request_parses_valid_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_capital_planner.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_plan(), separators=(",", ":"))}),
        ):
            out = request_openai_capital_planner(
                settings={"openai_capital_planner_enabled": True},
                base_dir=".",
                planner_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        plan = out.get("plan", {}) if isinstance(out.get("plan", {}), dict) else {}
        portfolio_plan = plan.get("portfolio_plan", {}) if isinstance(plan.get("portfolio_plan", {}), dict) else {}
        self.assertEqual(str(portfolio_plan.get("mode", "")), "mixed")
        actions = plan.get("market_actions", []) if isinstance(plan.get("market_actions", []), list) else []
        self.assertTrue(actions)

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_capital_planner.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_capital_planner(
                settings={"openai_capital_planner_enabled": True},
                base_dir=".",
                planner_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_capital_planner.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_capital_planner(
                settings={"openai_capital_planner_enabled": True, "openai_capital_planner_timeout_s": 1.0},
                base_dir=".",
                planner_packet={"mode": "live"},
            )
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_run_writes_report_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            with open(os.path.join(td, "gui_settings.json"), "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3}, f)
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 3.0,
                "summary": "Planner complete.",
                "error": "",
                "latency_ms": 7,
                "plan": self._valid_plan(),
            }
            with patch("app.openai_capital_planner.request_openai_capital_planner", return_value=fake_result):
                out = run_openai_capital_planner(
                    settings={"openai_capital_planner_enabled": True},
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_200,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "capital_planner.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "capital_planner_status.json")))


if __name__ == "__main__":
    unittest.main()
