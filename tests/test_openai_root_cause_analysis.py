from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from app.openai_root_cause_analysis import request_openai_root_cause_analysis, run_openai_root_cause_analysis


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = int(status_code)
        self._payload = dict(payload)

    def json(self) -> dict:
        return dict(self._payload)


class OpenAIRootCauseAnalysisTests(unittest.TestCase):
    def _valid_analysis(self) -> dict:
        return {
            "summary": "Stocks and forex show reject pressure with latency drift; crypto remains stable.",
            "overall_assessment": "caution",
            "market_diagnoses": [
                {
                    "market": "crypto",
                    "assessment": "normal",
                    "likely_causes": ["No meaningful anomaly detected."],
                    "recommended_actions": ["Continue monitoring."],
                },
                {
                    "market": "stocks",
                    "assessment": "degraded",
                    "likely_causes": ["Reject spike from stale scan artifacts."],
                    "recommended_actions": ["Refresh scanner cache and tighten entry filter."],
                },
                {
                    "market": "forex",
                    "assessment": "caution",
                    "likely_causes": ["Latency drift during recent scans."],
                    "recommended_actions": ["Monitor scan latency and keep risk gates active."],
                },
            ],
            "global_risks": ["reject_spike", "latency_drift"],
            "throttle_recommendations": [
                {
                    "market": "crypto",
                    "recommendation": "none",
                    "confidence": 0.4,
                    "reason": "No sustained anomaly.",
                },
                {
                    "market": "stocks",
                    "recommendation": "throttle",
                    "confidence": 0.82,
                    "reason": "Reject spike persisted over multiple cycles.",
                },
                {
                    "market": "forex",
                    "recommendation": "monitor",
                    "confidence": 0.71,
                    "reason": "Latency elevated but still operational.",
                },
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
                    "alerts": {"severity": "warning", "reasons": ["scanner pressure"], "hints": ["check reject trends"]},
                    "scan_health": {
                        "stocks": {"reject_rate_pct": 91.0, "leaders_total": 10, "scores_total": 2},
                        "forex": {"reject_rate_pct": 75.0, "leaders_total": 12, "scores_total": 5},
                    },
                    "sla_metrics": {
                        "stocks_scan": {"last_ms": 8200.0, "p95_ms": 9300.0},
                        "forex_scan": {"last_ms": 4200.0, "p95_ms": 5100.0},
                    },
                    "incident_trend": {"count_1h": 4, "count_24h": 18},
                },
                f,
            )
        with open(os.path.join(hub, "incidents.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 1_710_001_000, "severity": "warning", "event": "scan_reject_spike", "msg": "reject rate high"}) + "\n")
            f.write(json.dumps({"ts": 1_710_001_200, "severity": "warning", "event": "scan_latency", "msg": "stocks scan latency elevated"}) + "\n")
        with open(os.path.join(hub, "trader_data.json"), "w", encoding="utf-8") as f:
            json.dump({"automation_policy": {"runtime_trust": {"score": 82.0}}, "trade_quality": {"confidence_score": 70.0, "decision": "allow"}}, f)
        with open(os.path.join(hub, "stocks", "stock_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entry_eval_top_reason": "reject pressure",
                    "entry_eval_reason_counts": {"reject pressure": 9},
                    "entry_gate_flags": {"loss_streak": 2, "reject_rate_pct": 90.0},
                    "automation_policy": {"runtime_trust": {"score": 55.0}},
                    "trade_quality": {"confidence_score": 42.0, "decision": "deprioritize"},
                    "stale_exit_count": 3,
                },
                f,
            )
        with open(os.path.join(hub, "stocks", "stock_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"open_meta": {"AAPL": {"entry_ts": 1_710_000_000}}, "stale_alignment_streaks": {"AAPL": 2}}, f)
        with open(os.path.join(hub, "stocks", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(hub, "forex", "forex_trader_status.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entry_eval_top_reason": "latency drift",
                    "entry_gate_flags": {"loss_streak": 1, "reject_rate_pct": 74.0},
                    "automation_policy": {"runtime_trust": {"score": 62.0}},
                    "trade_quality": {"confidence_score": 58.0, "decision": "allow"},
                },
                f,
            )
        with open(os.path.join(hub, "forex", "forex_trader_state.json"), "w", encoding="utf-8") as f:
            json.dump({"open_meta": {}, "stale_alignment_streaks": {}}, f)
        with open(os.path.join(hub, "forex", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        with open(os.path.join(hub, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
            f.write("")
        return hub

    def test_request_parses_valid_structured_response(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_root_cause_analysis.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_ok", "output_text": json.dumps(self._valid_analysis(), separators=(",", ":"))}),
        ):
            out = request_openai_root_cause_analysis(
                settings={"openai_root_cause_enabled": True},
                base_dir=".",
                analysis_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "ok")
        analysis = out.get("analysis", {}) if isinstance(out.get("analysis", {}), dict) else {}
        self.assertEqual(str(analysis.get("overall_assessment", "")), "caution")
        self.assertTrue(isinstance(analysis.get("market_diagnoses", []), list))

    def test_request_malformed_response_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_root_cause_analysis.requests.post",
            return_value=_FakeResponse(200, {"id": "resp_bad", "output_text": "not-json"}),
        ):
            out = request_openai_root_cause_analysis(
                settings={"openai_root_cause_enabled": True},
                base_dir=".",
                analysis_packet={"mode": "live"},
            )
        self.assertTrue(bool(out.get("enabled", False)))
        self.assertFalse(bool(out.get("active", False)))
        self.assertEqual(str(out.get("status", "")), "malformed_response")

    def test_request_timeout_falls_back(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), patch(
            "app.openai_root_cause_analysis.requests.post",
            side_effect=requests.Timeout(),
        ):
            out = request_openai_root_cause_analysis(
                settings={"openai_root_cause_enabled": True, "openai_root_cause_timeout_s": 1.0},
                base_dir=".",
                analysis_packet={"mode": "live"},
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
                "summary": "Root-cause analysis completed.",
                "error": "",
                "latency_ms": 8,
                "analysis": self._valid_analysis(),
            }
            with patch("app.openai_root_cause_analysis.request_openai_root_cause_analysis", return_value=fake_result):
                out = run_openai_root_cause_analysis(
                    settings={"openai_root_cause_enabled": True},
                    base_dir=td,
                    hub_dir=hub,
                    now_ts_value=1_710_010_000,
                )
            self.assertEqual(str(out.get("status", "")), "ok")
            self.assertTrue(bool(out.get("active", False)))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "root_cause_analysis.json")))
            self.assertTrue(os.path.isfile(os.path.join(hub, "openai", "root_cause_analysis_status.json")))


if __name__ == "__main__":
    unittest.main()
