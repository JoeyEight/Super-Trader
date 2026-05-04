from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_strategy_optimizer import (
    request_openai_strategy_optimizer,
    run_openai_strategy_optimizer,
    validate_low_risk_strategy_suggestions,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIStrategyOptimizerTests(unittest.TestCase):
    def _valid_review(self) -> dict:
        return {
            "summary": "Current preset is slightly too aggressive for observed churn.",
            "preset_assessment": {
                "current_profile": "max_growth",
                "assessment": "too_aggressive",
                "reason": "Recent stale exits and churn are elevated.",
            },
            "strategy_suggestions": [
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "current_value": 45.0,
                    "suggested_value": 35.0,
                    "confidence": 0.87,
                    "reason": "Reduce concentration risk while preserving growth posture.",
                }
            ],
            "risk_flags": ["stale_exit_pressure"],
        }

    def _seed_hub(self, root: str) -> str:
        hub = os.path.join(root, "hub_data")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(os.path.join(hub, "openai"), exist_ok=True)
        os.makedirs(os.path.join(hub, "crypto"), exist_ok=True)
        os.makedirs(os.path.join(hub, "stocks"), exist_ok=True)
        os.makedirs(os.path.join(hub, "forex"), exist_ok=True)

        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 100.0,
                    "buying_power_usd": 75.0,
                    "exposure_usd": 20.0,
                    "open_positions": 2,
                    "positions": {
                        "BTC": {"quantity": 0.01, "aligned_with_strategy": True},
                        "ETH": {"quantity": 0.02, "aligned_with_strategy": False},
                    },
                    "entry_eval_total": 15,
                    "entry_eval_failed": 4,
                    "entry_eval_top_reason": "confidence_gate",
                    "entry_eval_reason_counts": {"confidence_gate": 4},
                    "stale_exit_count": 2,
                    "automation_policy": {
                        "profile": "max_growth",
                        "mode": "aggressive_rotation",
                        "allow_new_entries": True,
                        "runtime_trust": {"score": 82.0},
                        "summary": "Max Growth preset active.",
                    },
                    "trade_quality": {"decision": "allow", "confidence_score": 74.0},
                    "entry_gate_flags": {"loss_streak": 1},
                    "account": {"total_account_value": 100.0, "buying_power": 75.0},
                },
                f,
            )

        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 100.0,
                    "buying_power_usd": 60.0,
                    "exposure_usd": 35.0,
                    "open_positions": 1,
                    "entry_eval_total": 20,
                    "entry_eval_failed": 5,
                    "entry_eval_top_reason": "cached_fallback",
                    "entry_eval_reason_counts": {"cached_fallback": 5},
                    "stale_exit_count": 3,
                    "automation_policy": {
                        "profile": "max_growth",
                        "mode": "aggressive_guarded",
                        "allow_new_entries": True,
                        "runtime_trust": {"score": 68.0},
                        "summary": "Stock policy active.",
                    },
                    "trade_quality": {"decision": "allow", "confidence_score": 61.0},
                    "entry_gate_flags": {"loss_streak": 2},
                },
                f,
            )

        with open(os.path.join(hub, "stocks", "stock_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "open_meta": {"AAPL": {"entry_ts": 1_710_000_000}},
                    "stale_alignment_streaks": {"AAPL": 1},
                },
                f,
            )

        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "account_value_usd": 100.0,
                    "margin_available_usd": 55.0,
                    "exposure_usd": 15.0,
                    "open_positions": 1,
                    "entry_eval_total": 18,
                    "entry_eval_failed": 6,
                    "entry_eval_top_reason": "loss_streak_guard",
                    "entry_eval_reason_counts": {"loss_streak_guard": 6},
                    "stale_exit_count": 1,
                    "automation_policy": {
                        "profile": "max_growth",
                        "mode": "aggressive_guarded",
                        "allow_new_entries": False,
                        "runtime_trust": {"score": 52.0},
                        "summary": "Forex policy active.",
                    },
                    "trade_quality": {"decision": "deprioritize", "confidence_score": 49.0},
                    "entry_gate_flags": {"loss_streak": 3},
                },
                f,
            )

        with open(os.path.join(hub, "forex", "forex_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "open_meta": {"EUR_USD": {"entry_ts": 1_710_000_000}},
                    "stale_alignment_streaks": {"EUR_USD": 0},
                },
                f,
            )

        with open(os.path.join(hub, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_000, "event": "entry", "symbol": "BTC", "msg": "dca add"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_500, "event": "exit", "symbol": "ETH", "realized_pnl_usd": -1.2, "msg": "policy_stale_exit"}) + "\n")
        with open(os.path.join(hub, "stocks", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_000, "event": "entry", "symbol": "AAPL", "msg": "entry"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_500, "event": "shadow_live_divergence", "symbol": "AAPL", "msg": "cached fallback"}) + "\n")
        with open(os.path.join(hub, "forex", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_000_000, "event": "entry", "instrument": "EUR_USD", "msg": "entry"}) + "\n")
            f.write(json.dumps({"ts": 1_710_000_500, "event": "shadow_live_divergence", "instrument": "EUR_USD", "msg": "loss_streak guard"}) + "\n")

        with open(os.path.join(hub, "runtime_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cross_market_opportunity": {"summary": "Crypto currently preferred.", "best_market": "crypto"},
                    "openai_position_review": {"summary": "Reduce stale adds.", "status": "ok", "actions_count": 2},
                    "openai_capital_planner": {"summary": "Reserve some buying power.", "status": "ok", "portfolio_plan": {"mode": "mixed"}},
                    "openai_root_cause_analysis": {"summary": "Churn pressure elevated.", "status": "ok", "overall_assessment": "caution"},
                },
                f,
            )
        return hub

    def test_request_parses_valid_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_strategy_optimizer.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_review(), separators=(",", ":"))}),
        ):
            out = request_openai_strategy_optimizer(
                settings={"openai_strategy_optimizer_enabled": True},
                base_dir=".",
                optimizer_packet={"mode": "live", "settings_profile": "max_growth"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        review = out.get("review", {}) if isinstance(out.get("review", {}), dict) else {}
        assess = review.get("preset_assessment", {}) if isinstance(review.get("preset_assessment", {}), dict) else {}
        self.assertEqual(str(assess.get("assessment", "")), "too_aggressive")

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_strategy_optimizer.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_strategy_optimizer(
                settings={"openai_strategy_optimizer_enabled": True},
                base_dir=".",
                optimizer_packet={"mode": "live", "settings_profile": "max_growth"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_strategy_optimizer.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_strategy_optimizer(
                settings={"openai_strategy_optimizer_enabled": True, "openai_strategy_optimizer_timeout_s": 1.0},
                base_dir=".",
                optimizer_packet={"mode": "live", "settings_profile": "max_growth"},
            )
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_validate_suggestions_is_bounded(self) -> None:
        out = validate_low_risk_strategy_suggestions(
            settings={"stock_max_total_exposure_pct": 40.0},
            suggestions=[
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "suggested_value": 150.0,
                    "confidence": 0.95,
                    "reason": "Clamp it",
                },
                {
                    "setting_key": "unknown_key",
                    "suggested_value": 1,
                    "confidence": 0.99,
                    "reason": "not allowlisted",
                },
            ],
            min_confidence=0.82,
        )
        validated = out.get("validated", []) if isinstance(out.get("validated", []), list) else []
        skipped = out.get("skipped", []) if isinstance(out.get("skipped", []), list) else []
        self.assertEqual(len(validated), 1)
        self.assertEqual(float(validated[0].get("suggested_value", 0.0) or 0.0), 90.0)
        self.assertEqual(len(skipped), 1)

    def test_run_writes_report_and_keeps_advisory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3, "stock_max_total_exposure_pct": 45.0}, f)

            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 8.0,
                "summary": "Optimizer review complete.",
                "error": "",
                "latency_ms": 7,
                "review": self._valid_review(),
            }
            with patch("app.openai_strategy_optimizer.request_openai_strategy_optimizer", return_value=fake_result):
                out = run_openai_strategy_optimizer(
                    settings={
                        "openai_strategy_optimizer_enabled": True,
                        "openai_strategy_optimizer_auto_apply_enabled": False,
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_900,
                )

            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 0)
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "strategy_optimizer.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "strategy_optimizer_status.json")))
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(float(saved.get("stock_max_total_exposure_pct", 0.0) or 0.0), 45.0)

    def test_run_auto_apply_uses_safe_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = self._seed_hub(td)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3, "stock_max_total_exposure_pct": 45.0}, f)

            fake_review = self._valid_review()
            fake_review["strategy_suggestions"] = [
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "current_value": 45.0,
                    "suggested_value": 150.0,
                    "confidence": 0.91,
                    "reason": "Reduce concentration pressure.",
                }
            ]
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 8.0,
                "summary": "Optimizer review complete.",
                "error": "",
                "latency_ms": 9,
                "review": fake_review,
            }
            with patch("app.openai_strategy_optimizer.request_openai_strategy_optimizer", return_value=fake_result):
                out = run_openai_strategy_optimizer(
                    settings={
                        "openai_strategy_optimizer_enabled": True,
                        "openai_strategy_optimizer_auto_apply_enabled": True,
                        "stock_max_total_exposure_pct": 45.0,
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_901,
                )

            self.assertEqual(int(out.get("validated_suggestions_count", 0) or 0), 1)
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 1)
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(float(saved.get("stock_max_total_exposure_pct", 0.0) or 0.0), 90.0)

    def test_run_auto_apply_marks_profile_override_when_preset_managed(self) -> None:
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

            fake_review = self._valid_review()
            fake_review["strategy_suggestions"] = [
                {
                    "setting_key": "stock_max_total_exposure_pct",
                    "current_value": 45.0,
                    "suggested_value": 70.0,
                    "confidence": 0.93,
                    "reason": "Reduce concentration pressure.",
                }
            ]
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 8.0,
                "summary": "Optimizer review complete.",
                "error": "",
                "latency_ms": 9,
                "review": fake_review,
            }
            with patch("app.openai_strategy_optimizer.request_openai_strategy_optimizer", return_value=fake_result):
                out = run_openai_strategy_optimizer(
                    settings={
                        "openai_strategy_optimizer_enabled": True,
                        "openai_strategy_optimizer_auto_apply_enabled": True,
                        "settings_control_mode": "preset_managed",
                    },
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_000_902,
                )
            self.assertEqual(int(out.get("validated_suggestions_count", 0) or 0), 1)
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 1)
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(float(saved.get("stock_max_total_exposure_pct", 0.0) or 0.0), 70.0)
            overrides = saved.get("profile_manual_overrides", [])
            self.assertTrue(isinstance(overrides, list))
            self.assertIn("stock_max_total_exposure_pct", overrides)


if __name__ == "__main__":
    unittest.main()
