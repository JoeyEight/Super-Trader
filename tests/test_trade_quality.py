from __future__ import annotations

import unittest

from app.trade_quality import evaluate_trade_quality


class TestTradeQuality(unittest.TestCase):
    def test_blocks_when_runtime_and_compliance_fail(self) -> None:
        out = evaluate_trade_quality(
            market="stocks",
            signal_score=0.4,
            required_score=0.2,
            data_quality_ok=True,
            broker_ok=True,
            runtime_trust_score=24.0,
            runtime_alert_severity="critical",
            compliance_allowed=False,
            compliance_reason="Blocked to avoid PDT violation risk",
            fallback_active=False,
            fallback_age_s=0,
            fallback_hard_block_age_s=1200,
            reject_rate_pct=10.0,
            reject_rate_limit_pct=90.0,
            spread_bps=2.0,
            max_slippage_bps=20.0,
            loss_streak=0,
            max_loss_streak=3,
            exposure_usage_pct=20.0,
        )
        self.assertEqual(str(out.get("decision", "")), "block")
        self.assertFalse(bool(((out.get("layers", {}) if isinstance(out.get("layers", {}), dict) else {}).get("runtime_trust", True))))
        self.assertFalse(bool(((out.get("layers", {}) if isinstance(out.get("layers", {}), dict) else {}).get("compliance_permission", True))))
        self.assertTrue(bool(out.get("block_reasons", [])))

    def test_allows_high_quality_trade(self) -> None:
        out = evaluate_trade_quality(
            market="forex",
            signal_score=0.8,
            required_score=0.2,
            data_quality_ok=True,
            broker_ok=True,
            runtime_trust_score=84.0,
            runtime_alert_severity="ok",
            compliance_allowed=True,
            fallback_active=False,
            fallback_age_s=0,
            fallback_hard_block_age_s=1200,
            reject_rate_pct=12.0,
            reject_rate_limit_pct=92.0,
            spread_bps=1.0,
            max_slippage_bps=8.0,
            loss_streak=0,
            max_loss_streak=4,
            exposure_usage_pct=15.0,
        )
        self.assertEqual(str(out.get("decision", "")), "allow")
        self.assertGreater(float(out.get("confidence_score", 0.0) or 0.0), 40.0)
        self.assertGreater(float(out.get("size_multiplier", 0.0) or 0.0), 0.0)

    def test_crypto_quality_can_scale_size_above_one_for_high_confidence(self) -> None:
        out = evaluate_trade_quality(
            market="crypto",
            signal_score=0.92,
            required_score=0.2,
            data_quality_ok=True,
            broker_ok=True,
            runtime_trust_score=91.0,
            runtime_alert_severity="ok",
            compliance_allowed=True,
            fallback_active=False,
            fallback_age_s=0,
            fallback_hard_block_age_s=1200,
            reject_rate_pct=4.0,
            reject_rate_limit_pct=85.0,
            spread_bps=2.0,
            max_slippage_bps=25.0,
            loss_streak=0,
            max_loss_streak=3,
            exposure_usage_pct=18.0,
            min_runtime_trust_score=29.0,
            min_confidence_score=33.0,
        )
        self.assertEqual(str(out.get("decision", "")), "allow")
        self.assertGreater(float(out.get("size_multiplier", 0.0) or 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
