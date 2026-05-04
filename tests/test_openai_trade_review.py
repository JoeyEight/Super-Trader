from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_trade_review import (
    request_openai_nightly_trade_review,
    run_openai_nightly_trade_review,
    validate_low_risk_tuning_suggestions,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAINightlyTradeReviewTests(unittest.TestCase):
    def _valid_review(self) -> dict:
        return {
            "summary": "Recent behavior shows elevated churn from stale exits in crypto.",
            "overall_assessment": "caution",
            "market_reviews": [
                {
                    "market": "crypto",
                    "assessment": "caution",
                    "main_drags": ["stale exits elevated"],
                    "main_strengths": ["scanner freshness stable"],
                    "recommended_actions": ["raise crypto_dynamic_min_projected_edge_pct slightly"],
                },
                {
                    "market": "stocks",
                    "assessment": "healthy",
                    "main_drags": [],
                    "main_strengths": ["entry selectivity stable"],
                    "recommended_actions": ["hold settings"],
                },
                {
                    "market": "forex",
                    "assessment": "caution",
                    "main_drags": ["churn elevated"],
                    "main_strengths": ["execution latency stable"],
                    "recommended_actions": ["reduce scan breadth"],
                },
            ],
            "tuning_suggestions": [
                {
                    "setting_key": "stock_score_threshold",
                    "current_value": 0.3,
                    "suggested_value": 0.4,
                    "confidence": 0.91,
                    "reason": "Tighten low-quality entries after recent churn.",
                }
            ],
            "risk_flags": ["recent_churn_pressure"],
            "next_day_guidance": ["Prefer quality over trade count in crypto."],
        }

    def test_request_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_trade_review.requests.post",
            return_value=_FakeResponse(
                200,
                {"id": "resp_ok", "output_text": json.dumps(self._valid_review(), separators=(",", ":"))},
            ),
        ):
            out = request_openai_nightly_trade_review(
                settings={"openai_nightly_review_enabled": True, "openai_nightly_review_timeout_s": 3.0},
                base_dir=".",
                review_packet={"timestamp": 1},
            )

        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        review = out.get("review", {}) if isinstance(out.get("review", {}), dict) else {}
        self.assertEqual(str(review.get("overall_assessment", "")), "caution")
        self.assertEqual(str((review.get("market_reviews", [])[0] or {}).get("market", "")), "crypto")

    def test_request_malformed_json_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_trade_review.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_nightly_trade_review(
                settings={"openai_nightly_review_enabled": True},
                base_dir=".",
                review_packet={"timestamp": 1},
            )

        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_trade_review.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_nightly_trade_review(
                settings={"openai_nightly_review_enabled": True, "openai_nightly_review_timeout_s": 1.0},
                base_dir=".",
                review_packet={"timestamp": 1},
            )

        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "timeout")

    def test_run_writes_report_and_status_without_auto_apply_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = str(tmp)
            hub_dir = os.path.join(base_dir, "hub_data")
            os.makedirs(hub_dir, exist_ok=True)
            settings_path = os.path.join(base_dir, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3, "stock_score_threshold": 0.3}, f)

            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 3.0,
                "summary": "Nightly review complete.",
                "error": "",
                "latency_ms": 5,
                "review": self._valid_review(),
            }

            with patch("app.openai_trade_review.request_openai_nightly_trade_review", return_value=fake_result):
                out = run_openai_nightly_trade_review(
                    settings={
                        "openai_nightly_review_enabled": True,
                        "openai_nightly_review_write_report_enabled": True,
                        "openai_nightly_review_apply_tuning_enabled": False,
                    },
                    base_dir=base_dir,
                    hub_dir=hub_dir,
                    now_ts_value=1_710_000_000,
                )

            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 0)

            with open(settings_path, "r", encoding="utf-8") as f:
                post = json.load(f)
            self.assertAlmostEqual(float(post.get("stock_score_threshold", 0.0)), 0.3, places=6)

            report_path = os.path.join(hub_dir, "openai", "nightly_trade_review.json")
            status_path = os.path.join(hub_dir, "openai", "nightly_trade_review_status.json")
            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(os.path.isfile(status_path))

    def test_run_apply_enabled_uses_validated_clamped_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = str(tmp)
            hub_dir = os.path.join(base_dir, "hub_data")
            os.makedirs(hub_dir, exist_ok=True)
            settings_path = os.path.join(base_dir, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump({"settings_schema_version": 3, "stock_score_threshold": 0.3}, f)

            review = self._valid_review()
            review["tuning_suggestions"] = [
                {
                    "setting_key": "stock_score_threshold",
                    "current_value": 0.3,
                    "suggested_value": 9.0,
                    "confidence": 0.95,
                    "reason": "Clamp test",
                }
            ]
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 3.0,
                "summary": "Nightly review complete.",
                "error": "",
                "latency_ms": 5,
                "review": review,
            }

            with patch("app.openai_trade_review.request_openai_nightly_trade_review", return_value=fake_result):
                out = run_openai_nightly_trade_review(
                    settings={
                        "openai_nightly_review_enabled": True,
                        "openai_nightly_review_write_report_enabled": True,
                        "openai_nightly_review_apply_tuning_enabled": True,
                    },
                    base_dir=base_dir,
                    hub_dir=hub_dir,
                    now_ts_value=1_710_000_100,
                )

            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 1)

            with open(settings_path, "r", encoding="utf-8") as f:
                post = json.load(f)
            self.assertAlmostEqual(float(post.get("stock_score_threshold", 0.0)), 1.5, places=6)

    def test_run_apply_marks_profile_override_when_preset_managed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = str(tmp)
            hub_dir = os.path.join(base_dir, "hub_data")
            os.makedirs(hub_dir, exist_ok=True)
            settings_path = os.path.join(base_dir, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "settings_schema_version": 3,
                        "settings_control_mode": "preset_managed",
                        "profile_manual_overrides": [],
                        "stock_score_threshold": 0.3,
                    },
                    f,
                )

            review = self._valid_review()
            review["tuning_suggestions"] = [
                {
                    "setting_key": "stock_score_threshold",
                    "current_value": 0.3,
                    "suggested_value": 0.55,
                    "confidence": 0.91,
                    "reason": "Tighten low-quality entries.",
                }
            ]
            fake_result = {
                "enabled": True,
                "active": True,
                "status": "ok",
                "model": "gpt-5.4-mini",
                "timeout_s": 3.0,
                "summary": "Nightly review complete.",
                "error": "",
                "latency_ms": 5,
                "review": review,
            }
            with patch("app.openai_trade_review.request_openai_nightly_trade_review", return_value=fake_result):
                out = run_openai_nightly_trade_review(
                    settings={
                        "openai_nightly_review_enabled": True,
                        "openai_nightly_review_write_report_enabled": True,
                        "openai_nightly_review_apply_tuning_enabled": True,
                        "settings_control_mode": "preset_managed",
                    },
                    base_dir=base_dir,
                    hub_dir=hub_dir,
                    now_ts_value=1_710_000_101,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertEqual(int(out.get("applied_tuning_count", 0) or 0), 1)
            with open(settings_path, "r", encoding="utf-8") as f:
                post = json.load(f)
            self.assertAlmostEqual(float(post.get("stock_score_threshold", 0.0)), 0.55, places=6)
            overrides = post.get("profile_manual_overrides", [])
            self.assertTrue(isinstance(overrides, list))
            self.assertIn("stock_score_threshold", overrides)

    def test_validate_low_risk_suggestions_clamps_and_skips_safely(self) -> None:
        out = validate_low_risk_tuning_suggestions(
            settings={"max_dca_buys_per_24h": 2},
            suggestions=[
                {
                    "setting_key": "max_dca_buys_per_24h",
                    "suggested_value": 99,
                    "confidence": 0.9,
                    "reason": "Clamp to supported max.",
                },
                {
                    "setting_key": "not_allowlisted_key",
                    "suggested_value": 1,
                    "confidence": 0.95,
                    "reason": "Should skip.",
                },
                {
                    "setting_key": "stock_score_threshold",
                    "suggested_value": 0.6,
                    "confidence": 0.3,
                    "reason": "Low confidence should skip.",
                },
            ],
            min_confidence=0.8,
        )
        validated = out.get("validated", []) if isinstance(out.get("validated", []), list) else []
        skipped = out.get("skipped", []) if isinstance(out.get("skipped", []), list) else []
        self.assertEqual(len(validated), 1)
        self.assertEqual(str(validated[0].get("setting_key", "")), "max_dca_buys_per_24h")
        self.assertEqual(int(validated[0].get("suggested_value", 0) or 0), 8)
        skip_codes = {str(row.get("skip_code", "")) for row in skipped if isinstance(row, dict)}
        self.assertIn("not_allowlisted", skip_codes)
        self.assertIn("low_confidence", skip_codes)


if __name__ == "__main__":
    unittest.main()
