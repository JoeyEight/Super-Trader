from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_postmortem_analysis import request_openai_postmortem_analysis, run_openai_postmortem_analysis


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIPostmortemAnalysisTests(unittest.TestCase):
    def _valid_analysis(self) -> dict:
        return {
            "summary": "Primary drag was stale exits and rapid turnover in forex and crypto.",
            "main_drags": ["stale exits", "rapid turnover"],
            "main_strengths": ["stock entries were selective"],
            "skip_recommendations": ["skip low-confidence adds during elevated churn"],
            "exit_improvement_recommendations": ["delay weak exits until alignment confirms"],
            "capital_reallocation_recommendations": ["allocate less to churn-prone setups"],
            "tuning_suggestions": [
                {
                    "setting_key": "crypto_dynamic_rotation_cooldown_s",
                    "suggested_value": 1200,
                    "confidence": 0.82,
                    "reason": "Slightly slower rotation should reduce churn.",
                }
            ],
        }

    def _seed_hub(self, root: str) -> str:
        hub = os.path.join(root, "hub_data")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(os.path.join(hub, "openai"), exist_ok=True)
        os.makedirs(os.path.join(hub, "crypto"), exist_ok=True)
        os.makedirs(os.path.join(hub, "stocks"), exist_ok=True)
        os.makedirs(os.path.join(hub, "forex"), exist_ok=True)

        with open(os.path.join(hub, "runtime_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cross_market_opportunity": {"summary": "Crypto currently leads."},
                    "openai_position_review": {"summary": "Reduce stale adds.", "status": "ok", "actions_count": 2},
                    "openai_capital_planner": {"summary": "Reserve some buying power.", "status": "ok"},
                    "openai_strategy_optimizer": {"summary": "Mildly too aggressive.", "status": "ok"},
                },
                f,
            )
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump({"open_positions": 2, "stale_exit_count": 1, "entry_eval_failed": 2, "entry_eval_total": 10}, f)
        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump({"open_positions": 1, "stale_exit_count": 1, "entry_eval_failed": 1, "entry_eval_total": 8}, f)
        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump({"open_positions": 1, "stale_exit_count": 2, "entry_eval_failed": 3, "entry_eval_total": 9}, f)

        with open(os.path.join(hub, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_100, "event": "entry", "symbol": "BTC", "msg": "entry"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_400, "event": "exit", "symbol": "BTC", "realized_pnl_usd": -0.4, "hold_s": 1200, "msg": "policy_stale_exit"}) + "\n")
        with open(os.path.join(hub, "stocks", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_200, "event": "entry", "symbol": "AAPL", "msg": "entry"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_700, "event": "exit", "symbol": "AAPL", "realized_pnl_usd": 1.2, "hold_s": 36000, "msg": "profit_target"}) + "\n")
        with open(os.path.join(hub, "forex", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_150, "event": "entry", "instrument": "EUR_USD", "msg": "entry"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_500, "event": "exit", "instrument": "EUR_USD", "realized_pnl_usd": -0.8, "hold_s": 1800, "msg": "policy_stale_exit"}) + "\n")
        return hub

    def test_request_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_postmortem_analysis.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_analysis(), separators=(",", ":"))}),
        ):
            out = request_openai_postmortem_analysis(
                settings={"openai_postmortem_enabled": True},
                base_dir=".",
                postmortem_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        analysis = out.get("analysis", {}) if isinstance(out.get("analysis", {}), dict) else {}
        self.assertTrue(bool(analysis.get("tuning_suggestions", [])))

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_postmortem_analysis.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_postmortem_analysis(
                settings={"openai_postmortem_enabled": True},
                base_dir=".",
                postmortem_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_postmortem_analysis.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_postmortem_analysis(
                settings={"openai_postmortem_enabled": True, "openai_postmortem_timeout_s": 1.0},
                base_dir=".",
                postmortem_packet={"mode": "live"},
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
                "summary": "Postmortem complete.",
                "error": "",
                "latency_ms": 8,
                "analysis": self._valid_analysis(),
            }
            with patch("app.openai_postmortem_analysis.request_openai_postmortem_analysis", return_value=fake_result):
                out = run_openai_postmortem_analysis(
                    settings={"openai_postmortem_enabled": True, "openai_postmortem_write_report_enabled": True},
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_030_000,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 0)
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "postmortem_analysis.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "postmortem_analysis_status.json")))

    def test_run_auto_apply_uses_safe_clamping_and_marks_profile_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "settings_schema_version": 3,
                        "settings_control_mode": "preset_managed",
                        "profile_manual_overrides": [],
                        "stock_max_total_exposure_pct": 45.0,
                    },
                    f,
                )

            fake_analysis = self._valid_analysis()
            fake_analysis["tuning_suggestions"] = [
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "suggested_value": 150.0,
                    "confidence": 0.91,
                    "reason": "Clamp to supported cap for safer concentration.",
                }
            ]
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "Postmortem complete.",
                "error": "",
                "latency_ms": 7,
                "analysis": fake_analysis,
            }
            with patch("app.openai_postmortem_analysis.request_openai_postmortem_analysis", return_value=fake_result):
                out = run_openai_postmortem_analysis(
                    settings={
                        "openai_postmortem_enabled": True,
                        "openai_postmortem_auto_apply_tuning_enabled": True,
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_030_001,
                )

            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertEqual(int(out.get("validated_suggestions_count", 0) or 0), 1)
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 1)
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(float(saved.get("stock_max_total_exposure_pct", 0.0) or 0.0), 90.0)
            overrides = saved.get("profile_manual_overrides", [])
            self.assertTrue(isinstance(overrides, list))
            self.assertIn("stock_max_total_exposure_pct", overrides)


if __name__ == "__main__":
    unittest.main()
