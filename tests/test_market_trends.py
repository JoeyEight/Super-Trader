from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app.market_trends import build_market_trend_summary, build_trends_payload, parse_stale_signal_seconds


class TestMarketTrends(unittest.TestCase):
    def test_parse_stale_signal_seconds(self) -> None:
        self.assertEqual(parse_stale_signal_seconds("Signal stale (615s > 300s)"), 615)
        self.assertEqual(parse_stale_signal_seconds("other message"), 0)

    def test_build_market_trend_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            audit = os.path.join(stocks_dir, "execution_audit.jsonl")
            ranks = os.path.join(stocks_dir, "scanner_rankings.jsonl")
            status = os.path.join(stocks_dir, "stock_trader_status.json")
            thinker = os.path.join(stocks_dir, "stock_thinker_status.json")
            diag = os.path.join(stocks_dir, "scan_diagnostics.json")
            quality = os.path.join(stocks_dir, "universe_quality.json")
            cadence = os.path.join(td, "scanner_cadence_drift.json")

            rows = [
                {"ts": 1_700_000_000, "event": "shadow_live_divergence", "msg": "Signal stale (615s > 300s)"},
                {"ts": 1_700_000_010, "event": "shadow_live_divergence", "msg": "Max open positions reached (1/1)"},
                {"ts": 1_700_000_020, "event": "entry_fail", "spread_bps": 12.5, "msg": "bad fill"},
            ]
            with open(audit, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(ranks, "w", encoding="utf-8") as f:
                f.write(json.dumps({"top": [{"score": 0.5, "spread_bps": 3.0}, {"score": -1.2, "spread_bps": 6.0}]}) + "\n")
            with open(status, "w", encoding="utf-8") as f:
                json.dump({"state": "READY", "msg": "ok"}, f)
            with open(thinker, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fallback_cached": True,
                        "top_chart_map": {
                            "AAPL": [{"t": "t1", "o": 1, "h": 1, "l": 1, "c": 1}],
                            "MSFT": [{"t": "t1", "o": 2, "h": 2, "l": 2, "c": 2}],
                        },
                    },
                    f,
                )
            with open(diag, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": "READY",
                        "leaders_total": 2,
                        "scores_total": 4,
                        "candidate_churn_pct": 55.0,
                        "leader_churn_pct": 40.0,
                        "reject_summary": {"reject_rate_pct": 62.0, "dominant_reason": "spread"},
                    },
                    f,
                )
            with open(quality, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "reject_rate_pct": 61.0,
                        "candidate_churn_pct": 54.0,
                        "leader_churn_pct": 39.0,
                        "gate_pass_pct": 80.0,
                        "rejection_reasons": [{"reason": "spread", "count": 4, "pct": 66.0}],
                    },
                    f,
                )
            with open(cadence, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "markets": {
                            "stocks": {
                                "level": "warning",
                                "late_pct": 95.0,
                                "observed_s": 38.0,
                                "expected_s": 20.0,
                            }
                        },
                        "active": [{"market": "stocks", "level": "warning"}],
                    },
                    f,
                )

            with patch("app.market_trends.time.time", return_value=1_700_000_050):
                out = build_market_trend_summary(td, "stocks")
            self.assertEqual(out["market"], "stocks")
            self.assertIn("event_counts_total", out)
            self.assertIn("stale_signal", out)
            self.assertIn("spread_bps", out)
            self.assertIn("signal_score_abs", out)
            self.assertIn("quality_aggregates", out)
            self.assertIn("cadence_aggregates", out)
            self.assertIn("chart_coverage", out)
            quality_agg = out["quality_aggregates"]
            cadence_agg = out["cadence_aggregates"]
            chart_cov = out["chart_coverage"]
            self.assertAlmostEqual(float(quality_agg.get("reject_rate_pct", 0.0) or 0.0), 61.0, places=2)
            self.assertEqual(str(quality_agg.get("dominant_reason", "")), "spread")
            self.assertEqual(str(cadence_agg.get("level", "")), "warning")
            self.assertTrue(bool(cadence_agg.get("active", False)))
            self.assertEqual(int(chart_cov.get("symbols_cached", 0) or 0), 2)
            self.assertTrue(bool(chart_cov.get("fallback_cached", False)))

    def test_build_market_trend_summary_caps_liquidity_dominated_reject_pressure_when_leaders_survive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self_path = os.path.join(stocks_dir, "stock_thinker_status.json")
            trader_path = os.path.join(stocks_dir, "stock_trader_status.json")
            diag_path = os.path.join(stocks_dir, "scan_diagnostics.json")
            quality_path = os.path.join(stocks_dir, "universe_quality.json")

            with open(os.path.join(stocks_dir, "execution_audit.jsonl"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(stocks_dir, "scanner_rankings.jsonl"), "w", encoding="utf-8") as f:
                f.write("")
            with open(trader_path, "w", encoding="utf-8") as f:
                json.dump({"state": "READY", "msg": "ok"}, f)
            with open(self_path, "w", encoding="utf-8") as f:
                json.dump({"top_chart_map": {}, "top_pick": {"symbol": "AAPL"}}, f)
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": "READY",
                        "leaders_total": 1,
                        "scores_total": 2,
                        "reject_summary": {
                            "reject_rate_pct": 100.0,
                            "dominant_reason": "liquidity",
                            "dominant_ratio_pct": 81.67,
                        },
                    },
                    f,
                )
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "leaders_total": 1,
                        "scores_total": 2,
                        "reject_rate_pct": 60.0,
                        "reject_rate_raw_pct": 100.0,
                        "rejection_reasons": [{"reason": "liquidity", "count": 196, "pct": 81.67}],
                    },
                    f,
                )
            with open(os.path.join(td, "scanner_cadence_drift.json"), "w", encoding="utf-8") as f:
                json.dump({"markets": {}, "active": []}, f)

            out = build_market_trend_summary(td, "stocks")
            quality_agg = out.get("quality_aggregates", {}) if isinstance(out.get("quality_aggregates", {}), dict) else {}
            self.assertAlmostEqual(float(quality_agg.get("reject_rate_pct", 0.0) or 0.0), 60.0, places=2)
            self.assertAlmostEqual(float(quality_agg.get("reject_rate_raw_pct", 0.0) or 0.0), 100.0, places=2)

    def test_build_market_trend_summary_market_closed_zeros_reject_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            with open(os.path.join(stocks_dir, "execution_audit.jsonl"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(stocks_dir, "scanner_rankings.jsonl"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(stocks_dir, "stock_trader_status.json"), "w", encoding="utf-8") as f:
                json.dump({"state": "READY", "msg": "ok"}, f)
            with open(os.path.join(stocks_dir, "stock_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump({"top_chart_map": {}}, f)
            with open(os.path.join(stocks_dir, "scan_diagnostics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": "READY",
                        "market_open": False,
                        "leaders_total": 0,
                        "scores_total": 0,
                        "reject_summary": {
                            "reject_rate_pct": 0.0,
                            "dominant_reason": "market_closed",
                        },
                    },
                    f,
                )
            with open(os.path.join(stocks_dir, "universe_quality.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        # Simulate stale open-session quality stats still on disk.
                        "reject_rate_pct": 100.0,
                        "reject_rate_raw_pct": 100.0,
                        "leaders_total": 1,
                        "scores_total": 2,
                        "rejection_reasons": [{"reason": "data_quality", "count": 20, "pct": 100.0}],
                    },
                    f,
                )
            with open(os.path.join(td, "scanner_cadence_drift.json"), "w", encoding="utf-8") as f:
                json.dump({"markets": {}, "active": []}, f)

            out = build_market_trend_summary(td, "stocks")
            quality_agg = out.get("quality_aggregates", {}) if isinstance(out.get("quality_aggregates", {}), dict) else {}
            reliability = out.get("data_source_reliability", {}) if isinstance(out.get("data_source_reliability", {}), dict) else {}
            self.assertEqual(str(quality_agg.get("dominant_reason", "")), "market_closed")
            self.assertAlmostEqual(float(quality_agg.get("reject_rate_pct", -1.0)), 0.0, places=2)
            self.assertAlmostEqual(float(quality_agg.get("reject_rate_raw_pct", -1.0)), 0.0, places=2)
            self.assertEqual(str(reliability.get("level", "")), "high")
            self.assertAlmostEqual(float(reliability.get("score", -1.0)), 100.0, places=2)

    def test_build_trends_payload_contains_aggregate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for market, trader_file in (("stocks", "stock_trader_status.json"), ("forex", "forex_trader_status.json")):
                mdir = os.path.join(td, market)
                os.makedirs(mdir, exist_ok=True)
                with open(os.path.join(mdir, "execution_audit.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": 1_700_000_000, "event": "entry_fail"}) + "\n")
                with open(os.path.join(mdir, "scanner_rankings.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"top": []}) + "\n")
                with open(os.path.join(mdir, trader_file), "w", encoding="utf-8") as f:
                    json.dump({"state": "READY", "msg": "ok"}, f)
                with open(os.path.join(mdir, f"{market[:-1] if market.endswith('s') else market}_thinker_status.json"), "w", encoding="utf-8") as f:
                    json.dump({"top_chart_map": {}}, f)
                with open(os.path.join(mdir, "scan_diagnostics.json"), "w", encoding="utf-8") as f:
                    json.dump({"state": "READY", "leaders_total": 0, "scores_total": 0}, f)
                with open(os.path.join(mdir, "universe_quality.json"), "w", encoding="utf-8") as f:
                    json.dump({"reject_rate_pct": 0.0, "candidate_churn_pct": 0.0, "leader_churn_pct": 0.0}, f)
            with open(os.path.join(td, "scanner_cadence_drift.json"), "w", encoding="utf-8") as f:
                json.dump({"markets": {}, "active": []}, f)
            with patch("app.market_trends.time.time", return_value=1_700_000_100):
                payload = build_trends_payload(td)
            self.assertIn("stocks", payload)
            self.assertIn("forex", payload)
            self.assertIn("quality_aggregates", payload["stocks"])
            self.assertIn("cadence_aggregates", payload["stocks"])
            self.assertIn("chart_coverage", payload["stocks"])


if __name__ == "__main__":
    unittest.main()
