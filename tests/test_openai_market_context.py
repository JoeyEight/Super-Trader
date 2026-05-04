from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_market_context import request_openai_market_context, run_openai_market_context


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIMarketContextTests(unittest.TestCase):
    def _valid_context(self) -> dict:
        return {
            "summary": "Crypto context is supportive, stocks are neutral, and forex is adverse due to event pressure.",
            "market_context_scores": [
                {"market": "crypto", "context_state": "supportive", "confidence": 0.82, "reason": "Momentum and flow context remain supportive."},
                {"market": "stocks", "context_state": "neutral", "confidence": 0.56, "reason": "Mixed macro context."},
                {"market": "forex", "context_state": "adverse", "confidence": 0.78, "reason": "Event-risk context is elevated."},
            ],
            "symbol_context_scores": [
                {"market": "crypto", "symbol": "BTC", "context_state": "supportive", "confidence": 0.84, "reason": "Strong context alignment."},
                {"market": "forex", "symbol": "EUR_USD", "context_state": "adverse", "confidence": 0.73, "reason": "Adverse event context."},
            ],
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
                        "reasons": ["forex event risk elevated"],
                        "hints": ["review event-risk context"],
                    },
                    "market_trends": {
                        "stocks": {"why_not_traded": {"reason": "confidence gate"}},
                        "forex": {"why_not_traded": {"reason": "event-risk pressure"}},
                    },
                },
                f,
            )
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 82.0}},
                    "trade_quality": {"confidence_score": 76.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"ranked": [{"symbol": "BTC", "score": 1.12, "reason_logic": "strong momentum context"}]},
                f,
            )
        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 70.0}},
                    "trade_quality": {"confidence_score": 60.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "stocks", "stock_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": [{"symbol": "AAPL", "side": "long", "score": 0.33, "reason_logic": "stable trend"}]}, f)
        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 58.0}},
                    "trade_quality": {"confidence_score": 52.0, "decision": "deprioritize"},
                },
                f,
            )
        with open(os.path.join(hub, "forex", "forex_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.21, "reason_logic": "event pressure high"}]}, f)
        return hub

    def test_request_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_market_context.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_context(), separators=(",", ":"))}),
        ):
            out = request_openai_market_context(
                settings={"openai_market_context_enabled": True},
                base_dir=".",
                context_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        ctx = out.get("context", {}) if isinstance(out.get("context", {}), dict) else {}
        rows = ctx.get("market_context_scores", []) if isinstance(ctx.get("market_context_scores", []), list) else []
        self.assertEqual(len(rows), 3)

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_market_context.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_market_context(
                settings={"openai_market_context_enabled": True},
                base_dir=".",
                context_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_market_context.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_market_context(
                settings={"openai_market_context_enabled": True, "openai_market_context_timeout_s": 1.0},
                base_dir=".",
                context_packet={"mode": "live"},
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
                "summary": "Market context scoring complete.",
                "error": "",
                "latency_ms": 6,
                "context": self._valid_context(),
            }
            with patch("app.openai_market_context.request_openai_market_context", return_value=fake_result):
                out = run_openai_market_context(
                    settings={"openai_market_context_enabled": True},
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_020_000,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "market_context.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "market_context_status.json")))


if __name__ == "__main__":
    unittest.main()
