from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import requests

from app.openai_position_review import request_openai_position_review, run_openai_position_review


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIPositionReviewTests(unittest.TestCase):
    def _valid_review(self) -> dict:
        return {
            "summary": "Reduce stale adds and keep increasing only top aligned positions.",
            "position_actions": [
                {
                    "market": "crypto",
                    "symbol": "BTC",
                    "action": "increase",
                    "confidence": 0.82,
                    "size_multiplier": 1.1,
                    "reduce_fraction": 0.0,
                    "reason": "Strong alignment and healthy confidence.",
                    "risk_flags": ["capital_available"],
                },
                {
                    "market": "stocks",
                    "symbol": "AAPL",
                    "action": "increase",
                    "confidence": 0.78,
                    "size_multiplier": 1.05,
                    "reduce_fraction": 0.0,
                    "reason": "Momentum remains constructive.",
                    "risk_flags": [],
                },
            ],
            "portfolio_risks": ["churn_pressure"],
        }

    def _seed_hub(self, root: str) -> str:
        hub = os.path.join(root, "hub_data")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(os.path.join(hub, "openai"), exist_ok=True)
        os.makedirs(os.path.join(hub, "crypto"), exist_ok=True)
        os.makedirs(os.path.join(hub, "stocks"), exist_ok=True)
        os.makedirs(os.path.join(hub, "forex"), exist_ok=True)

        now_i = int(time.time())
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positions": {
                        "BTC": {
                            "quantity": 0.2,
                            "value_usd": 20.0,
                            "avg_cost_basis": 90.0,
                            "aligned_with_strategy": True,
                            "alignment_reasons": ["trend_aligned"],
                        }
                    },
                    "account_value_usd": 100.0,
                    "buying_power_usd": 70.0,
                    "exposure_usd": 20.0,
                    "automation_policy": {
                        "allow_new_entries": True,
                        "profile": "max_growth",
                        "mode": "managed",
                        "runtime_trust": {"score": 82.0},
                    },
                    "trade_quality": {"confidence_score": 75.0, "decision": "allow"},
                    "entry_gate_flags": {},
                    "account": {"total_account_value": 100.0, "buying_power": 70.0},
                },
                f,
            )
        with open(os.path.join(hub, "crypto_dynamic_status.json"), "w", encoding="utf-8") as f:
            json.dump({"ranked": [{"symbol": "BTC", "score": 1.2, "spread_bps": 5.0}]}, f)
        with open(os.path.join(hub, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_i - 2000, "event": "entry", "symbol": "BTC", "side": "long"}) + "\n")

        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "position_values_usd": {"AAPL": 12.0},
                    "account_value_usd": 100.0,
                    "buying_power_usd": 55.0,
                    "exposure_usd": 12.0,
                    "automation_policy": {
                        "allow_new_entries": True,
                        "profile": "max_growth",
                        "mode": "managed",
                        "runtime_trust": {"score": 78.0},
                    },
                    "trade_quality": {"confidence_score": 69.0, "decision": "allow"},
                    "entry_gate_flags": {"pdt_restricted": False},
                    "account": {"total_account_value": 100.0, "buying_power": 55.0},
                },
                f,
            )
        with open(os.path.join(hub, "stocks", "stock_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"open_meta": {"AAPL": {"entry_ts": now_i - 1800, "last_pnl_pct": 0.8, "qty": 1.0, "side": "long"}}, "stale_alignment_streaks": {"AAPL": 0}}, f)
        with open(os.path.join(hub, "stocks", "stock_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": [{"symbol": "AAPL", "score": 0.52, "spread_bps": 3.0}]}, f)
        with open(os.path.join(hub, "stocks", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_i - 1900, "event": "entry", "symbol": "AAPL", "side": "long"}) + "\n")

        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "position_values_usd": {},
                    "account_value_usd": 100.0,
                    "margin_available_usd": 90.0,
                    "exposure_usd": 0.0,
                    "automation_policy": {
                        "allow_new_entries": True,
                        "profile": "max_growth",
                        "mode": "managed",
                        "runtime_trust": {"score": 76.0},
                    },
                    "trade_quality": {"confidence_score": 64.0, "decision": "allow"},
                    "entry_gate_flags": {},
                },
                f,
            )
        with open(os.path.join(hub, "forex", "forex_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"open_meta": {}, "stale_alignment_streaks": {}}, f)
        with open(os.path.join(hub, "forex", "forex_thinker_status.json"), "w", encoding="utf-8") as f:
            json.dump({"leaders": []}, f)
        with open(os.path.join(hub, "forex", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        return hub

    def test_request_parses_valid_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_position_review.requests.post",
            return_value=_FakeResponse(
                200,
                {"id": "resp_ok", "output_text": json.dumps(self._valid_review(), separators=(",", ":"))},
            ),
        ):
            out = request_openai_position_review(
                settings={"openai_position_review_enabled": True},
                base_dir=".",
                review_packet={"mode": "live", "open_positions": []},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        review = out.get("review", {}) if isinstance(out.get("review", {}), dict) else {}
        actions = review.get("position_actions", []) if isinstance(review.get("position_actions", []), list) else []
        self.assertTrue(actions)
        self.assertEqual(str(actions[0].get("action", "")), "increase")

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_position_review.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_position_review(
                settings={"openai_position_review_enabled": True},
                base_dir=".",
                review_packet={"mode": "live", "open_positions": []},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_position_review.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_position_review(
                settings={"openai_position_review_enabled": True, "openai_position_review_timeout_s": 1.0},
                base_dir=".",
                review_packet={"mode": "live", "open_positions": []},
            )
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_run_writes_outputs_and_bounds_actions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            with open(os.path.join(td, "gui_settings.json"), "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3}, f)
            fake = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "Review complete.",
                "error": "",
                "latency_ms": 5,
                "review": self._valid_review(),
            }
            with patch("app.openai_position_review.request_openai_position_review", return_value=fake):
                out = run_openai_position_review(
                    settings={
                        "openai_position_review_enabled": True,
                        "openai_position_review_auto_act_enabled": False,
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_000,
                )

            self.assertEqual(str(out.get("status", "")), "ok")
            actions = out.get("position_actions", []) if isinstance(out.get("position_actions", []), list) else []
            self.assertTrue(actions)
            by_symbol = {str(row.get("symbol", "")): dict(row) for row in actions if isinstance(row, dict)}
            self.assertEqual(str((by_symbol.get("AAPL", {}) or {}).get("effective_action", "")), "block_add")
            self.assertFalse(bool((by_symbol.get("BTC", {}) or {}).get("auto_action_eligible", True)))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "position_review.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "position_review_status.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "crypto", "openai_position_review.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "stocks", "openai_position_review.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "forex", "openai_position_review.json")))

    def test_run_auto_act_flag_still_bounded_by_local_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            with open(os.path.join(td, "gui_settings.json"), "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3}, f)
            fake = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "Review complete.",
                "error": "",
                "latency_ms": 5,
                "review": self._valid_review(),
            }
            with patch("app.openai_position_review.request_openai_position_review", return_value=fake):
                out = run_openai_position_review(
                    settings={
                        "openai_position_review_enabled": True,
                        "openai_position_review_auto_act_enabled": True,
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_001,
                )
            actions = out.get("position_actions", []) if isinstance(out.get("position_actions", []), list) else []
            by_symbol = {str(row.get("symbol", "")): dict(row) for row in actions if isinstance(row, dict)}
            self.assertTrue(bool((by_symbol.get("BTC", {}) or {}).get("auto_action_eligible", False)))
            self.assertFalse(bool((by_symbol.get("AAPL", {}) or {}).get("auto_action_eligible", True)))


if __name__ == "__main__":
    unittest.main()
