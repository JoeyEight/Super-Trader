from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.confidence_calibration import build_confidence_calibration_payload
from app.notification_center import build_notification_center_from_hub, build_notification_center_payload
from app.regime_classifier import build_all_market_regimes, classify_regime_from_series
from app.shadow_scorecard import build_shadow_scorecards
from app.walkforward_report import build_walkforward_report


class TestMarketIntelligence(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _write_jsonl(self, path: str, rows: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_classify_regime_trend_up(self) -> None:
        series = [100.0 + (0.6 * i) for i in range(40)]
        out = classify_regime_from_series(series)
        self.assertEqual(str(out.get("regime", "")), "trend_up")
        self.assertGreater(float(out.get("change_pct", 0.0) or 0.0), 0.0)

    def test_build_all_market_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "stocks", "stock_thinker_status.json"),
                {
                    "top_pick": {"symbol": "SPY"},
                    "top_chart_map": {
                        "SPY": [{"c": 100.0 + i * 0.2} for i in range(60)],
                        "TSLA": [{"c": 200.0 - i * 0.1} for i in range(60)],
                    },
                },
            )
            self._write_json(
                os.path.join(td, "forex", "forex_thinker_status.json"),
                {
                    "top_pick": {"pair": "EUR_USD"},
                    "top_chart_map": {
                        "EUR_USD": [{"c": 1.09 + i * 0.0004} for i in range(70)],
                    },
                },
            )
            out = build_all_market_regimes(td)
            self.assertIn("stocks", out)
            self.assertIn("forex", out)
            self.assertTrue(bool((out.get("stocks", {}) if isinstance(out.get("stocks", {}), dict) else {}).get("focus_symbol", "")))

    def test_walkforward_report_has_windows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = []
            ts0 = 1_700_000_000
            for day in range(12):
                ts = ts0 + (day * 86400)
                rows.append({"ts": ts + 1, "event": "entry", "ok": True, "pnl_usd": 1.5})
                rows.append({"ts": ts + 2, "event": "entry_fail", "ok": False, "pnl_usd": -0.2})
            self._write_jsonl(os.path.join(td, "stocks", "execution_audit.jsonl"), rows)
            self._write_jsonl(os.path.join(td, "forex", "execution_audit.jsonl"), rows)
            out = build_walkforward_report(td)
            stocks = out.get("stocks", {}) if isinstance(out.get("stocks", {}), dict) else {}
            self.assertEqual(str(stocks.get("state", "")), "READY")
            self.assertGreaterEqual(int(stocks.get("events_considered", 0) or 0), 1)
            self.assertTrue(isinstance(stocks.get("windows", []), list))

    def test_confidence_calibration_curve(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = []
            ts0 = 1_700_000_000
            for i in range(60):
                score = 0.1 + (i * 0.03)
                rows.append({"ts": ts0 + i, "event": "entry", "ok": bool(i % 3), "score": score})
            self._write_jsonl(os.path.join(td, "stocks", "execution_audit.jsonl"), rows)
            self._write_jsonl(os.path.join(td, "forex", "execution_audit.jsonl"), rows)
            out = build_confidence_calibration_payload(
                td,
                {
                    "stock_score_threshold": 0.2,
                    "forex_score_threshold": 0.2,
                    "adaptive_confidence_min_samples": 6,
                    "adaptive_confidence_target_success_pct": 45.0,
                },
            )
            stocks = out.get("stocks", {}) if isinstance(out.get("stocks", {}), dict) else {}
            rec = stocks.get("recommendation", {}) if isinstance(stocks.get("recommendation", {}), dict) else {}
            self.assertTrue(isinstance(stocks.get("curve", []), list))
            self.assertIn("recommended_threshold", rec)
            self.assertIn("crypto", out)

    def test_confidence_calibration_crypto_reads_trade_history_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = [
                {"ts": 1_700_000_010, "side": "buy", "symbol": "BTC-USD", "score": 1.30},
                {"ts": 1_700_000_040, "side": "sell", "symbol": "BTC-USD", "score": 1.30, "pnl_pct": 1.4, "realized_profit_usd": 1.2},
                {"ts": 1_700_000_080, "side": "buy", "symbol": "ETH-USD", "score": 0.85},
                {"ts": 1_700_000_120, "side": "sell", "symbol": "ETH-USD", "score": 0.85, "pnl_pct": -0.9, "realized_profit_usd": -0.8},
            ]
            self._write_jsonl(os.path.join(td, "trade_history.jsonl"), rows)
            out = build_confidence_calibration_payload(
                td,
                {
                    "stock_score_threshold": 0.2,
                    "forex_score_threshold": 0.2,
                    "crypto_dynamic_min_projected_edge_pct": 0.2,
                    "adaptive_confidence_min_samples": 2,
                    "adaptive_confidence_target_success_pct": 45.0,
                },
            )
            crypto = out.get("crypto", {}) if isinstance(out.get("crypto", {}), dict) else {}
            self.assertEqual(str(crypto.get("market", "")), "crypto")
            self.assertTrue(isinstance(crypto.get("curve", []), list))

    def test_confidence_calibration_uses_realized_exit_outcomes_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = [
                # Entry should not count toward calibration outcomes.
                {"ts": 1_700_000_010, "event": "entry", "ok": True, "score": 1.20, "symbol": "BTC-USD"},
                # Losing exit marked ok=True must still count as a loss via realized pnl fields.
                {"ts": 1_700_000_040, "event": "exit", "ok": True, "score": 1.20, "symbol": "BTC-USD", "pnl_pct": -1.2, "realized_pnl_usd": -0.5},
                # Winning exit marked ok=False must still count as win via realized pnl fields.
                {"ts": 1_700_000_080, "event": "exit", "ok": False, "score": 1.20, "symbol": "ETH-USD", "pnl_pct": 1.8, "realized_pnl_usd": 0.7},
            ]
            self._write_jsonl(os.path.join(td, "crypto", "execution_audit.jsonl"), rows)
            out = build_confidence_calibration_payload(
                td,
                {
                    "stock_score_threshold": 0.2,
                    "forex_score_threshold": 0.2,
                    "crypto_dynamic_min_projected_edge_pct": 0.2,
                    "adaptive_confidence_min_samples": 2,
                    "adaptive_confidence_target_success_pct": 45.0,
                },
            )
            crypto = out.get("crypto", {}) if isinstance(out.get("crypto", {}), dict) else {}
            self.assertEqual(int(crypto.get("samples", 0) or 0), 2)
            self.assertEqual(int(crypto.get("wins", 0) or 0), 1)
            self.assertAlmostEqual(float(crypto.get("win_rate_pct", 0.0) or 0.0), 50.0, places=4)

    def test_shadow_scorecards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_json(
                os.path.join(td, "market_trends.json"),
                {
                    "stocks": {
                        "quality_aggregates": {"reject_rate_pct": 72.0},
                        "discrepancy_tracker": {"divergence_pressure_pct": 20.0},
                        "data_source_reliability": {"score": 88.0},
                        "cadence_aggregates": {"level": "ok"},
                    },
                    "forex": {
                        "quality_aggregates": {"reject_rate_pct": 74.0},
                        "discrepancy_tracker": {"divergence_pressure_pct": 24.0},
                        "data_source_reliability": {"score": 85.0},
                        "cadence_aggregates": {"level": "ok"},
                    },
                },
            )
            self._write_json(
                os.path.join(td, "walkforward_report.json"),
                {
                    "stocks": {"latest_window": {"test": {"win_rate_pct": 62.0}}},
                    "forex": {"latest_window": {"test": {"win_rate_pct": 59.0}}},
                },
            )
            self._write_json(
                os.path.join(td, "confidence_calibration.json"),
                {
                    "stocks": {"samples": 40},
                    "forex": {"samples": 35},
                },
            )
            self._write_json(
                os.path.join(td, "market_regimes.json"),
                {
                    "stocks": {"dominant_regime": "trend_up"},
                    "forex": {"dominant_regime": "range"},
                },
            )
            out = build_shadow_scorecards(td)
            self.assertIn(str((out.get("stocks", {}) if isinstance(out.get("stocks", {}), dict) else {}).get("promotion_gate", "")), {"PASS", "WARN", "BLOCK"})
            self.assertIn("all_markets_pass", out)

    def test_notification_center_payload(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "warning", "reasons": ["scan_reject_pressure"], "hints": ["Tune thresholds."]},
            "automation_policy": {
                "stocks": {
                    "summary": "Balanced preset; scan every 15s",
                    "allow_new_entries": True,
                    "runtime_trust_score": 78.0,
                },
                "forex": {
                    "summary": "Aggressive preset; runtime trust reduced size",
                    "allow_new_entries": False,
                    "runtime_trust_score": 30.0,
                    "compliance_status": "",
                },
            },
            "scan_cadence": {"active": []},
            "market_trends": {
                "stocks": {
                    "quality_aggregates": {"reject_rate_pct": 95.0},
                    "data_source_reliability": {"score": 62.0},
                    "why_not_traded": {"reason": "Top score below threshold"},
                },
                "forex": {
                    "quality_aggregates": {"reject_rate_pct": 30.0},
                    "data_source_reliability": {"score": 90.0},
                    "why_not_traded": {"reason": "Session risk gate"},
                },
            },
        }
        incidents = [
            {"ts": 1_699_999_900, "severity": "error", "event": "stocks_trader_error", "msg": "boom", "details": {"market": "stocks"}},
            {"ts": 1_699_999_901, "severity": "warning", "event": "forex_thinker_failed", "msg": "nope", "details": {"market": "forex"}},
            {"ts": 1_699_999_902, "severity": "critical", "event": "scanner_cadence_drift", "msg": "late", "details": {"market": "forex"}},
            {"ts": 1_699_999_903, "severity": "warning", "event": "runner_startup_check", "msg": "stale_pid_file_removed", "details": {"component": "runner"}},
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        self.assertGreaterEqual(int(out.get("total", 0) or 0), 2)
        self.assertTrue(isinstance(out.get("items", []), list))
        self.assertTrue(isinstance(out.get("by_market", {}), dict))
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("Stocks automation policy", titles)
        self.assertIn("Forex automation policy", titles)
        self.assertNotIn("scanner_cadence_drift", titles)
        self.assertNotIn("runner_startup_check", titles)

    def test_notification_center_filters_resolved_market_loop_incidents(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": [], "metrics": {"market_loop_stale": False}},
            "scan_cadence": {"active": []},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {"ts": 1_699_999_950, "severity": "warning", "event": "runner_market_loop_status_stale", "msg": "loop stale"},
            {"ts": 1_699_999_960, "severity": "warning", "event": "runner_market_loop_restart", "msg": "loop restart"},
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("runner_market_loop_status_stale", titles)
        self.assertNotIn("runner_market_loop_restart", titles)

    def test_notification_center_filters_housekeeping_startup_warnings(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": ["stale_pid_file_removed"], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {
                "ts": 1_699_999_990,
                "severity": "warning",
                "event": "runner_startup_check",
                "msg": "stale_pid_file_removed",
                "details": {"component": "runner"},
            }
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("runner_startup_check", titles)

    def test_notification_center_dedupes_transient_incidents_and_expires_old_ui_rows(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {"ts": 1_699_999_900, "severity": "warning", "event": "runner_watchdog_restart", "msg": "markets restarted"},
            {"ts": 1_699_999_990, "severity": "warning", "event": "runner_watchdog_restart", "msg": "markets restarted"},
            {"ts": 1_699_999_600, "severity": "warning", "event": "ui_market_panel_desync", "msg": "Forex leaders blank", "details": {"market": "forex"}},
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        items = list(out.get("items", []) or [])
        titles = [str((row or {}).get("title", "") or "") for row in items]
        self.assertEqual(titles.count("runner_watchdog_restart"), 1)
        self.assertNotIn("ui_market_panel_desync", titles)

    def test_notification_center_filters_resolved_autopilot_restarts_and_reject_spikes(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "scan_drift": {"active": []},
            "autopilot": {"ts": 1_700_000_000, "stable_cycles": 8, "api_unstable": False, "markets_healthy": True, "issue_open": False},
            "runner": {"state": "RUNNING", "children": {"autopilot": 12345}},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {
                "ts": 1_699_999_990,
                "severity": "warning",
                "event": "runner_watchdog_restart",
                "msg": "autopilot appears hung; restarting",
                "details": {"child": "autopilot"},
            },
            {
                "ts": 1_699_999_991,
                "severity": "warning",
                "event": "runner_child_exit",
                "msg": "autopilot exited code=0; restarting",
                "details": {"child": "autopilot", "code": 0},
            },
            {
                "ts": 1_699_999_992,
                "severity": "warning",
                "event": "scanner_reject_spike",
                "msg": "stocks reject spike",
                "details": {"market": "stocks"},
            },
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("runner_watchdog_restart", titles)
        self.assertNotIn("runner_child_exit", titles)
        self.assertNotIn("scanner_reject_spike", titles)

    def test_notification_center_keeps_active_autopilot_restart_issue(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "scan_drift": {"active": []},
            "autopilot": {"ts": 1_699_999_700, "stable_cycles": 0, "api_unstable": True, "markets_healthy": False, "issue_open": True},
            "runner": {"state": "RUNNING", "children": {"autopilot": 0}},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {
                "ts": 1_699_999_990,
                "severity": "warning",
                "event": "runner_watchdog_restart",
                "msg": "autopilot appears hung; restarting",
                "details": {"child": "autopilot"},
            }
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertIn("runner_watchdog_restart", titles)

    def test_notification_center_filters_resolved_trader_crash_loop(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "scan_drift": {"active": []},
            "runner": {"state": "RUNNING", "children": {"trader": 45678}},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {
                "ts": 1_699_999_995,
                "severity": "error",
                "event": "runner_child_crash_loop",
                "msg": "trader crash loop; lockout 180s",
                "details": {"child": "trader"},
            }
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("runner_child_crash_loop", titles)

    def test_notification_center_keeps_active_trader_crash_loop_when_child_down(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "scan_drift": {"active": []},
            "runner": {"state": "RUNNING", "children": {"trader": 0}},
            "market_trends": {"stocks": {}, "forex": {}},
        }
        incidents = [
            {
                "ts": 1_699_999_995,
                "severity": "error",
                "event": "runner_child_crash_loop",
                "msg": "trader crash loop; lockout 180s",
                "details": {"child": "trader"},
            }
        ]
        out = build_notification_center_payload(runtime_state, incidents_rows=incidents)
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertIn("runner_child_crash_loop", titles)

    def test_notification_center_from_hub_recomputes_runtime_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub_dir = os.path.join(td, "hub_data")
            self._write_json(
                os.path.join(hub_dir, "runtime_state.json"),
                {
                    "ts": 1_700_000_000,
                    "checks": {"ok": True, "warnings": [], "errors": []},
                    "alerts": {"severity": "critical", "reasons": ["scan_reject_pressure"], "hints": ["stale"]},
                    "scan_health": {
                        "stocks": {
                            "reject_rate_pct": 100.0,
                            "leaders_total": 1,
                            "scores_total": 2,
                            "reject_dominant_reason": "liquidity",
                            "reject_dominant_ratio_pct": 80.0,
                        },
                        "forex": {"reject_rate_pct": 0.0},
                    },
                    "market_trends": {"stocks": {}, "forex": {}},
                    "incidents_last_200": {"count": 0, "by_severity": {"error": 0, "warning": 0}},
                    "autopilot": {"api_unstable": False},
                },
            )
            self._write_json(os.path.join(td, "gui_settings.json"), {"runtime_alert_scan_reject_warn_pct": 65.0})
            self._write_jsonl(os.path.join(hub_dir, "incidents.jsonl"), [])
            out = build_notification_center_from_hub(hub_dir)
            titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
            self.assertNotIn("scan_reject_pressure", titles)

    def test_notification_center_suppresses_liquidity_dominated_reject_warning_when_effective_pressure_is_low(self) -> None:
        runtime_state = {
            "ts": 1_700_000_000,
            "checks": {"ok": True, "warnings": [], "errors": []},
            "alerts": {"severity": "ok", "reasons": [], "hints": []},
            "scan_cadence": {"active": []},
            "market_trends": {
                "stocks": {
                    "quality_aggregates": {
                        "reject_rate_pct": 60.0,
                        "reject_rate_raw_pct": 100.0,
                        "dominant_reason": "liquidity",
                        "reject_dominant_ratio_pct": 81.67,
                        "leaders_total": 1,
                        "scores_total": 2,
                    },
                    "data_source_reliability": {"score": 88.0},
                    "why_not_traded": {"reason": ""},
                },
                "forex": {},
            },
        }
        out = build_notification_center_payload(runtime_state, incidents_rows=[])
        titles = [str((row or {}).get("title", "") or "") for row in list(out.get("items", []) or [])]
        self.assertNotIn("High scanner rejection pressure", titles)


if __name__ == "__main__":
    unittest.main()
