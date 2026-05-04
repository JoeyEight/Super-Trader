from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import patch

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

    def test_independent_market_mode_bypasses_cross_market_deprioritization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_010_100
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 82.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 85.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 900.0,
                    "account_value_usd": 2_500.0,
                    "buying_power_usd": 1_900.0,
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {
                    "ts": now_ts,
                    "ranked": [{"symbol": "BTC", "score": 1.80}],
                },
            )
            self._write_json(
                os.path.join(td, "forex", "forex_trader_status.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 78.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 70.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 350.0,
                    "account_value_usd": 2_500.0,
                    "buying_power_usd": 1_400.0,
                },
            )
            self._write_json(
                os.path.join(td, "forex", "forex_thinker_status.json"),
                {"updated_at": now_ts, "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.50}},
            )

            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "market_independent_execution_enabled": True,
                    "portfolio_allocator_priority_gap_min": 5.0,
                },
                market="stocks",
                candidate_id="AAPL",
                candidate_side="long",
                signal_score=0.40,
                required_score=0.20,
                trade_quality={"decision": "allow", "confidence_score": 62.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 72.0}},
                projected_trade_value_usd=150.0,
                market_exposure_usd=700.0,
                account_value_usd=1_000.0,
                buying_power_usd=140.0,
                spread_bps=4.0,
                max_slippage_bps=30.0,
                candidate_age_s=10,
                now_ts=now_ts,
            )
            self.assertEqual(str(out.get("decision", "")), "allow")
            self.assertTrue(bool(out.get("independent_market_mode", False)))
            self.assertIn("independent-market mode", str(out.get("summary", "")).lower())

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

    def test_openai_layer_never_overrides_local_block(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch(
            "app.opportunity_allocator.request_openai_portfolio_decision"
        ) as mock_openai:
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "openai_decision_enabled": True,
                    "openai_decision_require_local_pass_first": True,
                },
                market="stocks",
                candidate_id="AAPL",
                candidate_side="watch",
                signal_score=0.05,
                required_score=0.20,
                trade_quality={"decision": "block", "confidence_score": 20.0},
                automation_policy={"allow_new_entries": False, "runtime_trust": {"score": 25.0}},
                projected_trade_value_usd=100.0,
                market_exposure_usd=0.0,
                account_value_usd=1_000.0,
                buying_power_usd=1_000.0,
                candidate_age_s=5,
                now_ts=1_700_030_000,
            )
        mock_openai.assert_not_called()
        self.assertEqual(str(out.get("decision", "")), "block")
        self.assertEqual(str(out.get("decision_source", "")), "local")
        openai_decision = out.get("openai_decision", {}) if isinstance(out.get("openai_decision", {}), dict) else {}
        self.assertEqual(str(openai_decision.get("status", "")), "skipped_local_gate")

    def test_openai_layer_can_deprioritize_allowed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch(
            "app.opportunity_allocator.request_openai_portfolio_decision",
            return_value={
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "Stocks should wait for stronger cross-market quality.",
                "decision": {
                    "decision": "deprioritize",
                    "best_market": "crypto",
                    "portfolio_action": "prefer_crypto",
                    "portfolio_confidence": 0.81,
                    "capital_constrained": True,
                    "top_recommendation": {
                        "market": "stocks",
                        "symbol": "AAPL",
                        "action": "deprioritize",
                        "confidence": 0.81,
                        "size_multiplier": 0.55,
                        "opportunity_score": 61.0,
                        "reason": "Crypto quality is stronger under current capital pressure.",
                    },
                    "ranked_candidates": [
                        {
                            "market": "stocks",
                            "symbol": "AAPL",
                            "action": "deprioritize",
                            "confidence": 0.81,
                            "size_multiplier": 0.55,
                            "opportunity_score": 61.0,
                            "risk_flags": ["capital_constrained"],
                            "reason": "Crypto quality is stronger under current capital pressure.",
                        }
                    ],
                    "global_risks": ["capital_constrained"],
                    "explanation": "Wait for better stock quality while crypto leads.",
                },
                "model": "gpt-5.4-mini",
                "latency_ms": 120,
            },
        ):
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "openai_decision_enabled": True,
                    "openai_decision_require_local_pass_first": True,
                },
                market="stocks",
                candidate_id="AAPL",
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
                now_ts=1_700_020_000,
            )
        self.assertEqual(str(out.get("decision", "")), "deprioritize")
        self.assertEqual(str(out.get("decision_source", "")), "local+openai")
        self.assertAlmostEqual(float(out.get("size_multiplier", 1.0)), 0.55, places=6)
        openai_decision = out.get("openai_decision", {}) if isinstance(out.get("openai_decision", {}), dict) else {}
        self.assertTrue(bool(openai_decision.get("active", False)))
        self.assertTrue(bool(openai_decision.get("applied", False)))

    def test_openai_failure_falls_back_to_local_allocator(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch(
            "app.opportunity_allocator.request_openai_portfolio_decision",
            return_value={
                "enabled": True,
                "active": False,
                "status": "timeout",
                "summary": "AI portfolio decision timed out; local allocator is active.",
                "decision": {},
            },
        ):
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "openai_decision_enabled": True,
                    "openai_decision_require_local_pass_first": True,
                },
                market="stocks",
                candidate_id="NVDA",
                candidate_side="long",
                signal_score=1.4,
                required_score=0.2,
                trade_quality={"decision": "allow", "confidence_score": 88.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 92.0}},
                projected_trade_value_usd=80.0,
                market_exposure_usd=150.0,
                account_value_usd=8_000.0,
                buying_power_usd=7_000.0,
                spread_bps=1.0,
                max_slippage_bps=35.0,
                candidate_age_s=6,
                now_ts=1_700_050_000,
            )
        self.assertEqual(str(out.get("decision", "")), "allow")
        self.assertEqual(str(out.get("decision_source", "")), "local")
        openai_decision = out.get("openai_decision", {}) if isinstance(out.get("openai_decision", {}), dict) else {}
        self.assertEqual(str(openai_decision.get("status", "")), "timeout")

    def test_openai_position_action_block_add_can_block_new_add(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch(
            "app.opportunity_allocator.request_openai_portfolio_decision",
            return_value={
                "enabled": True,
                "active": True,
                "status": "ok",
                "summary": "AI position review recommends blocking new adds for ETH.",
                "decision": {
                    "decision": "allow",
                    "best_market": "crypto",
                    "portfolio_action": "manage_existing",
                    "portfolio_confidence": 0.74,
                    "capital_constrained": False,
                    "top_recommendation": {
                        "market": "crypto",
                        "symbol": "ETH",
                        "action": "allow",
                        "confidence": 0.71,
                        "size_multiplier": 1.0,
                        "opportunity_score": 63.0,
                        "reason": "Manage existing ETH first.",
                    },
                    "ranked_candidates": [],
                    "position_actions": [
                        {
                            "market": "crypto",
                            "symbol": "ETH",
                            "action": "block_add",
                            "confidence": 0.83,
                            "size_multiplier": 1.0,
                            "reduce_fraction": 0.0,
                            "reason": "Block adding while alignment is stale.",
                            "risk_flags": ["alignment_stale"],
                        }
                    ],
                    "global_risks": [],
                    "explanation": "Manage existing ETH exposure before adding.",
                },
                "model": "gpt-5.4-mini",
                "latency_ms": 95,
            },
        ):
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={
                    "openai_decision_enabled": True,
                    "openai_decision_require_local_pass_first": True,
                },
                market="crypto",
                candidate_id="ETH",
                candidate_side="long",
                signal_score=1.2,
                required_score=0.2,
                trade_quality={"decision": "allow", "confidence_score": 84.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 90.0}},
                projected_trade_value_usd=20.0,
                market_exposure_usd=120.0,
                account_value_usd=2_000.0,
                buying_power_usd=1_000.0,
                spread_bps=4.0,
                max_slippage_bps=60.0,
                candidate_age_s=5,
                now_ts=1_700_060_000,
            )
        self.assertEqual(str(out.get("decision", "")), "block")
        self.assertEqual(str(out.get("decision_source", "")), "local+openai")
        openai_decision = out.get("openai_decision", {}) if isinstance(out.get("openai_decision", {}), dict) else {}
        matched = openai_decision.get("matched_position_action", {}) if isinstance(openai_decision.get("matched_position_action", {}), dict) else {}
        self.assertEqual(str(matched.get("action", "")), "block_add")

    def test_capital_planner_can_deprioritize_allowed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "openai"), exist_ok=True)
            self._write_json(
                os.path.join(td, "openai", "capital_planner_status.json"),
                {
                    "active": True,
                    "status": "ok",
                    "summary": "Reserve capital for stronger crypto opportunities.",
                    "portfolio_plan": {
                        "mode": "prefer_existing",
                        "reserve_capital_pct": 24.0,
                        "capital_constrained": True,
                        "reason": "Preserve dry powder for higher-ranked opportunities.",
                    },
                    "market_actions": [
                        {
                            "market": "stocks",
                            "action": "deprioritize",
                            "confidence": 0.82,
                            "suggested_capital_share_pct": 15.0,
                            "reason": "Stocks are weaker than crypto right now.",
                        }
                    ],
                    "market_actions_by_market": {
                        "stocks": {
                            "market": "stocks",
                            "action": "deprioritize",
                            "confidence": 0.82,
                            "suggested_capital_share_pct": 15.0,
                            "reason": "Stocks are weaker than crypto right now.",
                        }
                    },
                },
            )
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"openai_capital_planner_enabled": True},
                market="stocks",
                candidate_id="AAPL",
                candidate_side="long",
                signal_score=1.1,
                required_score=0.2,
                trade_quality={"decision": "allow", "confidence_score": 80.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 85.0}},
                projected_trade_value_usd=120.0,
                market_exposure_usd=100.0,
                account_value_usd=1000.0,
                buying_power_usd=90.0,
                spread_bps=2.0,
                max_slippage_bps=30.0,
                candidate_age_s=10,
                now_ts=1_700_080_000,
            )
        self.assertEqual(str(out.get("decision", "")), "deprioritize")
        self.assertEqual(str(out.get("decision_source", "")), "local+capital_planner")
        planner = out.get("openai_capital_planner", {}) if isinstance(out.get("openai_capital_planner", {}), dict) else {}
        self.assertTrue(bool(planner.get("applied", False)))
        self.assertEqual(str((planner.get("market_action", {}) or {}).get("action", "")), "deprioritize")

    def test_capital_planner_never_overrides_local_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "openai"), exist_ok=True)
            self._write_json(
                os.path.join(td, "openai", "capital_planner_status.json"),
                {
                    "active": True,
                    "status": "ok",
                    "summary": "Prefer crypto.",
                    "portfolio_plan": {"mode": "mixed"},
                    "market_actions_by_market": {
                        "stocks": {
                            "market": "stocks",
                            "action": "prefer",
                            "confidence": 0.9,
                            "suggested_capital_share_pct": 40.0,
                            "reason": "Planner preference only.",
                        }
                    },
                },
            )
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"openai_capital_planner_enabled": True},
                market="stocks",
                candidate_id="AAPL",
                candidate_side="watch",
                signal_score=0.01,
                required_score=0.2,
                trade_quality={"decision": "block", "confidence_score": 10.0},
                automation_policy={"allow_new_entries": False, "runtime_trust": {"score": 20.0}},
                projected_trade_value_usd=100.0,
                market_exposure_usd=0.0,
                account_value_usd=1000.0,
                buying_power_usd=1000.0,
                candidate_age_s=5,
                now_ts=1_700_080_001,
            )
        self.assertEqual(str(out.get("decision", "")), "block")
        self.assertEqual(str(out.get("decision_source", "")), "local")
        planner = out.get("openai_capital_planner", {}) if isinstance(out.get("openai_capital_planner", {}), dict) else {}
        self.assertFalse(bool(planner.get("applied", False)))

    def test_openai_packet_includes_open_positions_and_recent_performance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_070_000
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "positions": {
                        "BTC": {
                            "quantity": 0.1,
                            "avg_cost_basis": 60000.0,
                            "value_usd": 6100.0,
                            "gain_loss_pct_sell": 1.66,
                            "aligned_with_strategy": True,
                            "alignment_reasons": [],
                            "alignment_streak": 0,
                        }
                    },
                    "open_positions": 1,
                    "trade_quality": {"confidence_score": 70.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 80.0}},
                    "entry_gate_flags": {"runtime_trust_score": 80.0},
                    "exposure_usd": 6100.0,
                    "account_value_usd": 10000.0,
                    "buying_power_usd": 3000.0,
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {"ts": now_ts, "ranked": [{"symbol": "BTC", "score": 1.25}]},
            )
            os.makedirs(os.path.join(td, "crypto"), exist_ok=True)
            with open(os.path.join(td, "crypto", "execution_audit.jsonl"), "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": now_ts - 100,
                            "event": "entry",
                            "symbol": "BTC",
                            "side": "buy",
                            "qty": 0.1,
                            "price": 60000.0,
                            "ok": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            captured: Dict[str, Any] = {}

            def _fake_openai(**kwargs):
                captured.update(kwargs.get("decision_packet", {}) if isinstance(kwargs.get("decision_packet", {}), dict) else {})
                return {
                    "enabled": True,
                    "active": False,
                    "status": "timeout",
                    "summary": "AI portfolio decision timed out; local allocator is active.",
                    "decision": {},
                }

            with patch("app.opportunity_allocator.request_openai_portfolio_decision", side_effect=_fake_openai):
                _ = evaluate_cross_market_allocation(
                    hub_dir=td,
                    settings={
                        "openai_decision_enabled": True,
                        "openai_decision_require_local_pass_first": True,
                    },
                    market="crypto",
                    candidate_id="BTC",
                    candidate_side="long",
                    signal_score=1.25,
                    required_score=0.2,
                    trade_quality={"decision": "allow", "confidence_score": 74.0},
                    automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 84.0}},
                    projected_trade_value_usd=100.0,
                    market_exposure_usd=6100.0,
                    account_value_usd=10000.0,
                    buying_power_usd=3000.0,
                    spread_bps=8.0,
                    max_slippage_bps=120.0,
                    candidate_age_s=5,
                    now_ts=now_ts,
                )
            self.assertIn("open_positions", captured)
            self.assertIn("recent_performance_summary", captured)
            self.assertIsInstance(captured.get("open_positions", []), list)
            self.assertIsInstance(captured.get("recent_performance_summary", {}), dict)

    def test_market_context_advisory_is_loaded_and_applied_to_scores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_090_000
            os.makedirs(os.path.join(td, "openai"), exist_ok=True)
            self._write_json(
                os.path.join(td, "openai", "market_context_status.json"),
                {
                    "active": True,
                    "status": "ok",
                    "summary": "Stocks context is adverse while crypto is supportive.",
                    "market_context_scores": [
                        {"market": "stocks", "context_state": "adverse", "confidence": 0.92, "reason": "Macro drag."},
                        {"market": "crypto", "context_state": "supportive", "confidence": 0.85, "reason": "Momentum breadth."},
                        {"market": "forex", "context_state": "neutral", "confidence": 0.60, "reason": "Mixed session context."},
                    ],
                    "symbol_context_scores": [],
                    "by_market": {
                        "stocks": {"market": "stocks", "context_state": "adverse", "confidence": 0.92, "reason": "Macro drag."},
                        "crypto": {"market": "crypto", "context_state": "supportive", "confidence": 0.85, "reason": "Momentum breadth."},
                        "forex": {"market": "forex", "context_state": "neutral", "confidence": 0.60, "reason": "Mixed session context."},
                    },
                },
            )
            self._write_json(
                os.path.join(td, "trader_data.json"),
                {
                    "updated_at": now_ts,
                    "trade_quality": {"confidence_score": 72.0, "decision": "allow"},
                    "automation_policy": {"allow_new_entries": True, "runtime_trust": {"score": 80.0}},
                    "entry_eval_top_reason": "",
                    "exposure_usd": 120.0,
                    "account_value_usd": 2500.0,
                    "buying_power_usd": 1800.0,
                },
            )
            self._write_json(
                os.path.join(td, "crypto_dynamic_status.json"),
                {
                    "ts": now_ts,
                    "ranked": [{"symbol": "BTC", "score": 0.85}],
                },
            )

            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"openai_market_context_enabled": True},
                market="stocks",
                candidate_id="AAPL",
                candidate_side="long",
                signal_score=0.95,
                required_score=0.20,
                trade_quality={"decision": "allow", "confidence_score": 80.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 86.0}},
                projected_trade_value_usd=90.0,
                market_exposure_usd=200.0,
                account_value_usd=2500.0,
                buying_power_usd=1500.0,
                spread_bps=2.0,
                max_slippage_bps=30.0,
                candidate_age_s=10,
                now_ts=now_ts,
            )

        context_payload = out.get("openai_market_context", {}) if isinstance(out.get("openai_market_context", {}), dict) else {}
        self.assertTrue(bool(context_payload.get("enabled", False)))
        self.assertTrue(bool(context_payload.get("active", False)))
        self.assertEqual(str((out.get("context_state_by_market", {}) or {}).get("stocks", "")), "adverse")
        stocks_adj = float((out.get("context_adjustment_by_market", {}) or {}).get("stocks", 0.0) or 0.0)
        crypto_adj = float((out.get("context_adjustment_by_market", {}) or {}).get("crypto", 0.0) or 0.0)
        self.assertLess(stocks_adj, 0.0)
        self.assertGreater(crypto_adj, 0.0)

    def test_root_cause_throttle_can_deprioritize_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_091_000
            os.makedirs(os.path.join(td, "openai"), exist_ok=True)
            self._write_json(
                os.path.join(td, "openai", "root_cause_analysis_status.json"),
                {
                    "active": True,
                    "enabled": True,
                    "status": "ok",
                    "summary": "Forex entry quality degraded from churn.",
                    "overall_assessment": "degraded",
                    "throttle_recommendations": [
                        {"market": "forex", "recommendation": "throttle", "confidence": 0.92, "reason": "Reject/churn pressure elevated."},
                        {"market": "stocks", "recommendation": "none", "confidence": 0.20, "reason": ""},
                        {"market": "crypto", "recommendation": "none", "confidence": 0.20, "reason": ""},
                    ],
                },
            )
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"openai_root_cause_enabled": True},
                market="forex",
                candidate_id="EUR_USD",
                candidate_side="long",
                signal_score=1.10,
                required_score=0.20,
                trade_quality={"decision": "allow", "confidence_score": 81.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 80.0}},
                projected_trade_value_usd=80.0,
                market_exposure_usd=120.0,
                account_value_usd=2500.0,
                buying_power_usd=1600.0,
                spread_bps=1.0,
                max_slippage_bps=20.0,
                candidate_age_s=8,
                now_ts=now_ts,
            )
        self.assertEqual(str(out.get("decision", "")), "deprioritize")
        self.assertIn("root-cause", str(out.get("summary", "")).lower())
        self.assertIn("root_cause", str(out.get("decision_source", "")).lower())

    def test_root_cause_pause_can_block_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_091_200
            os.makedirs(os.path.join(td, "openai"), exist_ok=True)
            self._write_json(
                os.path.join(td, "openai", "root_cause_analysis_status.json"),
                {
                    "active": True,
                    "enabled": True,
                    "status": "ok",
                    "summary": "Stocks should pause while reliability is degraded.",
                    "overall_assessment": "critical",
                    "throttle_recommendations": [
                        {"market": "stocks", "recommendation": "pause", "confidence": 0.88, "reason": "Data reliability is unstable."},
                        {"market": "forex", "recommendation": "none", "confidence": 0.10, "reason": ""},
                        {"market": "crypto", "recommendation": "none", "confidence": 0.10, "reason": ""},
                    ],
                },
            )
            out = evaluate_cross_market_allocation(
                hub_dir=td,
                settings={"openai_root_cause_enabled": True},
                market="stocks",
                candidate_id="AAPL",
                candidate_side="long",
                signal_score=1.15,
                required_score=0.20,
                trade_quality={"decision": "allow", "confidence_score": 83.0},
                automation_policy={"allow_new_entries": True, "runtime_trust": {"score": 82.0}},
                projected_trade_value_usd=90.0,
                market_exposure_usd=180.0,
                account_value_usd=2500.0,
                buying_power_usd=1700.0,
                spread_bps=1.0,
                max_slippage_bps=20.0,
                candidate_age_s=8,
                now_ts=now_ts,
            )
        self.assertEqual(str(out.get("decision", "")), "block")
        self.assertIn("root-cause", str(out.get("summary", "")).lower())
        self.assertIn("root_cause", str(out.get("decision_source", "")).lower())


if __name__ == "__main__":
    unittest.main()
