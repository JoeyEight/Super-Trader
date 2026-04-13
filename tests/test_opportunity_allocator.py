from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.opportunity_allocator import evaluate_cross_market_allocation, summarize_allocator_snapshot


class OpportunityAllocatorTests(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_deprioritizes_weaker_market_when_capital_is_constrained(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_010_000
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 82.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 85.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 200.0,
                    "account_value_usd": 2_500.0,
                    "buying_power_usd": 1_900.0,
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {
                    "ts": now_ts,
                    "ranked": [{"symbol": "BTC", "score": 1.25}],
                },
            )
            self._write_json(
                os.path.join(td, "forex", "forex_trader_status.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 35.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 60.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 120.0,
                    "account_value_usd": 2_500.0,
                    "buying_power_usd": 1_200.0,
                },
            )
            self._write_json(
                os.path.join(td, "forex", "forex_thinker_status.json"),
                {"updated_at": now_ts, "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.18}},
            )

            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "portfolio_allocator_priority_gap_min": 5.0,
                    "portfolio_allocator_stale_soft_s": 60,
                    "portfolio_allocator_stale_hard_s": 300,
                    "portfolio_allocator_concentration_warn_pct": 65.0,
                },
                market="stocks",
                candidate_id="AAPL",
                candidate_side="long",
                signal_score=0.40,
                required_score=0.20,
                trade_quality={"decision": "allow", "confidence_score": 48.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 62.0}},
                projected_trade_value_usd=150.0,
                market_exposure_usd=700.0,
                account_value_usd=1_000.0,
                buying_power_usd=140.0,
                spread_bps=4.0,
                max_slippage_bps=30.0,
                candidate_age_s=10,
                now_ts=now_ts,
            )
            self.assertEqual(str(out.get("decision", "")), "deprioritize")
            self.assertEqual(str(out.get("best_market", "")), "crypto")
            self.assertTrue(bool(out.get("capital_constrained", False)))
            self.assertIn("deprioritized", str(out.get("summary", "")).lower())

    def test_allows_when_current_market_is_best(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_020_000
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 30.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 60.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 50.0,
                    "account_value_usd": 5_000.0,
                    "buying_power_usd": 4_500.0,
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {"ts": now_ts, "ranked": [{"symbol": "BTC", "score": 0.12}]},
            )

            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"portfolio_allocator_priority_gap_min": 6.0},
                market="stocks",
                candidate_id="NVDA",
                candidate_side="long",
                signal_score=1.2,
                required_score=0.2,
                trade_quality={"decision": "allow", "confidence_score": 84.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 90.0}},
                projected_trade_value_usd=120.0,
                market_exposure_usd=200.0,
                account_value_usd=5_000.0,
                buying_power_usd=2_000.0,
                spread_bps=1.0,
                max_slippage_bps=40.0,
                candidate_age_s=8,
                now_ts=now_ts,
            )
            self.assertEqual(str(out.get("decision", "")), "allow")
            self.assertEqual(str(out.get("best_market", "")), "stocks")
            self.assertIn("selected", str(out.get("summary", "")).lower())

    def test_blocks_when_current_candidate_is_not_eligible(self) -> None:
        out = evaluate_cross_market_allocation(
            hub_dir=".",
            settings={},
            market="forex",
            candidate_id="EUR_USD",
            candidate_side="watch",
            signal_score=0.05,
            required_score=0.2,
            trade_quality={"decision": "block", "confidence_score": 20.0},
            automation_policy={"allow_new_entries": False, "runtime_trust": {"score": 25.0}},
            projected_trade_value_usd=100.0,
            market_exposure_usd=0.0,
            account_value_usd=1_000.0,
            buying_power_usd=1_000.0,
            candidate_age_s=5,
            now_ts=1_700_030_000,
        )
        self.assertEqual(str(out.get("decision", "")), "block")
        reasons = out.get("reasons", []) if isinstance(out.get("reasons", []), list) else []
        self.assertTrue(bool(reasons))

    def test_uses_portfolio_account_basis_and_nested_crypto_account_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_040_000
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "account": {
                        "total_account_value": 100.0,
                        "buying_power": 95.0,
                        "holdings_sell_value": 5.0,
                    },
                    "trade_quality": {"confidence_score": 65.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 75.0}},
                    "entry_eval_top_reason": "",
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {
                    "ts": now_ts,
                    "ranked": [{"symbol": "BTC", "score": 0.9}],
                },
            )
            self._write_json(
                os.path.join(td, "stocks", "stock_trader_status.json"),
                {
                    "updated_at": now_ts,
                    "exposure_usd": 55.0,
                    "account_value_usd": 100.0,
                    "trade_quality": {"confidence_score": 30.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 55.0}},
                },
            )
            self._write_json(
                os.path.join(td, "stocks", "stock_thinker_status.json"),
                {"updated_at": now_ts, "top_pick": {"symbol": "AAPL", "side": "long", "score": 0.2}},
            )

            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={},
                market="forex",
                candidate_id="EUR_USD",
                candidate_side="long",
                signal_score=0.8,
                required_score=0.2,
                trade_quality={"decision": "allow", "confidence_score": 70.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 80.0}},
                projected_trade_value_usd=10.0,
                market_exposure_usd=0.0,
                account_value_usd=100.0,
                buying_power_usd=95.0,
                spread_bps=1.0,
                max_slippage_bps=8.0,
                candidate_age_s=10,
                now_ts=now_ts,
            )
            # (stocks 55 + crypto 5 + projected 10) / (forex 100 + stocks 100 + crypto 100) = 23.333...%
            self.assertAlmostEqual(float(out.get("projected_total_exposure_pct", 0.0)), 23.3333, places=2)
            self.assertAlmostEqual(float(out.get("portfolio_account_value_usd", 0.0)), 300.0, places=4)

    def test_summarize_allocator_snapshot_uses_latest_market_payload(self) -> None:
        stocks = {
            "opportunity_allocator": {
                "ts": 100,
                "decision": "deprioritize",
                "best_market": "crypto",
                "scores": {"stocks": 54.0, "forex": 51.0, "crypto": 70.0},
                "summary": "Stocks candidate deprioritized because crypto leads.",
            }
        }
        forex = {"opportunity_allocator": {"ts": 90, "decision": "allow", "best_market": "crypto"}}
        crypto = {"opportunity_allocator": {"ts": 110, "decision": "allow", "best_market": "crypto", "summary": "Crypto selected as best opportunity."}}
        out = summarize_allocator_snapshot(stocks, forex, crypto)
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("best_market", "")), "crypto")
        self.assertIn("crypto", str(out.get("summary", "")).lower())
        decisions = out.get("decisions", {}) if isinstance(out.get("decisions", {}), dict) else {}
        self.assertEqual(str(decisions.get("stocks", "")), "deprioritize")


if __name__ == "__main__":
    unittest.main()
