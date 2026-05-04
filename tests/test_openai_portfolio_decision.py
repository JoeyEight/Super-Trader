from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import requests

from app.openai_portfolio_decision import request_openai_portfolio_decision


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIPortfolioDecisionTests(unittest.TestCase):
    def _valid_payload(self) -> dict:
        return {
            "decision": "allow",
            "best_market": "crypto",
            "portfolio_action": "allow_top_candidate",
            "portfolio_confidence": 0.77,
            "capital_constrained": False,
            "top_recommendation": {
                "market": "crypto",
                "symbol": "BTC",
                "action": "allow",
                "confidence": 0.82,
                "size_multiplier": 0.9,
                "opportunity_score": 84.0,
                "reason": "Strong confidence with acceptable execution quality.",
            },
            "ranked_candidates": [
                {
                    "market": "crypto",
                    "symbol": "BTC",
                    "action": "allow",
                    "confidence": 0.82,
                    "size_multiplier": 0.9,
                    "opportunity_score": 84.0,
                    "risk_flags": ["capital_pressure_low"],
                    "reason": "Strong confidence with acceptable execution quality.",
                }
            ],
            "position_actions": [
                {
                    "market": "crypto",
                    "symbol": "BTC",
                    "action": "hold",
                    "confidence": 0.66,
                    "size_multiplier": 1.0,
                    "reduce_fraction": 0.0,
                    "reason": "Position remains aligned with current strategy.",
                    "risk_flags": [],
                }
            ],
            "global_risks": [],
            "explanation": "Crypto currently offers the strongest cross-market opportunity.",
        }

    def test_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_portfolio_decision.requests.post",
            return_value=_FakeResponse(
                200,
                {"id": "resp_ok", "output_text": json.dumps(self._valid_payload(), separators=(",", ":"))},
            ),
        ):
            out = request_openai_portfolio_decision(
                settings={"openai_decision_enabled": True, "openai_timeout_s": 3.0},
                base_dir=".",
                decision_packet={"timestamp": 1},
                broker_mode="live",
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        decision = out.get("decision", {}) if isinstance(out.get("decision", {}), dict) else {}
        self.assertEqual(str(decision.get("decision", "")), "allow")
        self.assertEqual(str((decision.get("top_recommendation", {}) or {}).get("symbol", "")), "BTC")
        position_actions = decision.get("position_actions", []) if isinstance(decision.get("position_actions", []), list) else []
        self.assertTrue(position_actions)
        self.assertEqual(str(position_actions[0].get("action", "")), "hold")

    def test_malformed_json_falls_back_safely(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_portfolio_decision.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_portfolio_decision(
                settings={"openai_decision_enabled": True},
                base_dir=".",
                decision_packet={"timestamp": 1},
                broker_mode="live",
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_timeout_falls_back_safely(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_portfolio_decision.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_portfolio_decision(
                settings={"openai_decision_enabled": True, "openai_timeout_s": 1.0},
                base_dir=".",
                decision_packet={"timestamp": 1},
                broker_mode="live",
            )
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_disabled_keeps_compatibility(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            out = request_openai_portfolio_decision(
                settings={"openai_decision_enabled": False},
                base_dir=".",
                decision_packet={"timestamp": 1},
                broker_mode="live",
            )
        self.assertFalse(bool(out.get("enabled", False)))
        self.assertEqual(str(out.get("status", "")), "disabled")


if __name__ == "__main__":
    unittest.main()
