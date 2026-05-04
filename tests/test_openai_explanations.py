from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_explanations import request_openai_explanations, run_openai_explanations


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIExplanationsTests(unittest.TestCase):
    def _valid_payload(self) -> dict:
        return {
            "items": [
                {
                    "kind": "trade_block",
                    "target": "stocks:AAPL",
                    "short_text": "Stock entry was blocked because rejection pressure stayed high.",
                    "reason_bullets": [
                        "Recent scanner rejects increased beyond the configured comfort range.",
                        "Runtime trust is currently below normal, so entry quality requirements tightened.",
                    ],
                },
                {
                    "kind": "portfolio_decision",
                    "target": "crypto:BTC",
                    "short_text": "Crypto was preferred because confidence was stronger for current capital usage.",
                    "reason_bullets": ["Cross-market opportunity ranking scored BTC highest this cycle."],
                },
            ]
        }

    def _seed_hub(self, root: str) -> str:
        hub = os.path.join(root, "hub_data")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(os.path.join(hub, "openai"), exist_ok=True)
        os.makedirs(os.path.join(hub, "stocks"), exist_ok=True)
        os.makedirs(os.path.join(hub, "forex"), exist_ok=True)

        with open(os.path.join(hub, "runtime_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alerts": {
                        "severity": "warning",
                        "reasons": ["stocks_reject_spike"],
                        "hints": ["Scanner rejects are elevated."],
                    },
                    "cross_market_opportunity": {
                        "summary": "Crypto candidate preferred due to stronger confidence.",
                        "decision": "allow",
                        "selected_candidate": {
                            "market": "crypto",
                            "symbol": "BTC",
                            "opportunity_score": 0.82,
                            "reason": "Highest quality under current trust and exposure limits.",
                        },
                    },
                    "pnl_decomposition": {
                        "realized_total_usd": -2.1,
                        "unrealized_total_usd": 0.7,
                        "trade_count": 12,
                        "summary": "Losses concentrated in stale exits.",
                    },
                    "openai_position_review": {
                        "position_actions": [
                            {
                                "market": "crypto",
                                "symbol": "ETH",
                                "action": "hold",
                                "effective_action": "hold",
                                "confidence": 0.74,
                                "reason": "Alignment remains stable.",
                            }
                        ]
                    },
                },
                f,
            )
        with open(os.path.join(hub, "incidents.jsonl"), "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": 1_710_010_000,
                        "severity": "warning",
                        "event": "scan_reject_spike",
                        "msg": "reject pressure rose in stocks",
                    }
                )
                + "\n"
            )
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump({"entry_eval_top_reason": "short block", "entry_eval_failed": 3, "entry_eval_total": 8}, f)
        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump({"entry_eval_top_reason": "reject pressure", "entry_eval_failed": 9, "entry_eval_total": 10, "stale_exit_count": 2}, f)
        with open(os.path.join(hub, "stocks", "stock_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"stale_alignment_streaks": {"AAPL": 2}}, f)
        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump({"entry_eval_top_reason": "", "entry_eval_failed": 0, "entry_eval_total": 0, "stale_exit_count": 0}, f)
        with open(os.path.join(hub, "forex", "forex_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"stale_alignment_streaks": {}}, f)
        return hub

    def test_request_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_explanations.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_payload(), separators=(",", ":"))}),
        ):
            out = request_openai_explanations(
                settings={"openai_explanations_enabled": True},
                base_dir=".",
                explanation_packet={
                    "mode": "live",
                    "explanation_requests": [{"kind": "trade_block", "target": "stocks:AAPL", "facts": {"reason": "reject"}}],
                },
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        explanations = out.get("explanations", {}) if isinstance(out.get("explanations", {}), dict) else {}
        items = explanations.get("items", []) if isinstance(explanations.get("items", []), list) else []
        self.assertTrue(items)
        self.assertEqual(str(items[0].get("kind", "")), "trade_block")

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_explanations.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_explanations(
                settings={"openai_explanations_enabled": True},
                base_dir=".",
                explanation_packet={
                    "mode": "live",
                    "explanation_requests": [{"kind": "trade_block", "target": "stocks:AAPL", "facts": {"reason": "reject"}}],
                },
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_explanations.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_explanations(
                settings={"openai_explanations_enabled": True, "openai_explanations_timeout_s": 1.0},
                base_dir=".",
                explanation_packet={
                    "mode": "live",
                    "explanation_requests": [{"kind": "trade_block", "target": "stocks:AAPL", "facts": {"reason": "reject"}}],
                },
            )
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_request_without_candidates_returns_safe_no_candidates(self) -> None:
        out = request_openai_explanations(
            settings={"openai_explanations_enabled": True},
            base_dir=".",
            explanation_packet={"mode": "live", "explanation_requests": []},
        )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "no_candidates")

    def test_run_writes_report_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            with open(os.path.join(td, "gui_settings.json"), "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3}, f)
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "Generated 2 AI explanation item(s).",
                "error": "",
                "latency_ms": 9,
                "explanations": self._valid_payload(),
            }
            with patch("app.openai_explanations.request_openai_explanations", return_value=fake_result):
                out = run_openai_explanations(
                    settings={"openai_explanations_enabled": True},
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_010_100,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertEqual(int(out.get("items_count", 0) or 0), 2)
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "explanations.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "explanations_status.json")))


if __name__ == "__main__":
    unittest.main()
