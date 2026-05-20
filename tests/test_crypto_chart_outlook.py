from __future__ import annotations

import json
import os
import tempfile
import unittest

from ui.pt_hub import CandleChart


class CryptoChartOutlookTests(unittest.TestCase):
    def test_local_model_outlook_prefers_bullish_when_long_signal_is_stronger(self) -> None:
        chart = CandleChart.__new__(CandleChart)
        chart.coin = "DOT"
        chart.settings_getter = lambda: {"trade_start_level": 3}

        out = CandleChart._local_model_outlook_1h(
            chart,
            long_levels=[1.31, 1.29, 1.25],
            short_levels=[1.37, 1.39, 1.42],
            long_sig=4,
            short_sig=0,
            current_buy_price=1.35,
            current_sell_price=1.34,
            last_close_price=1.335,
        )

        self.assertEqual(str(out.get("bias", "")), "bullish")
        self.assertGreater(float(out.get("confidence", 0.0) or 0.0), 0.5)
        self.assertGreater(float(out.get("target_up", 0.0) or 0.0), 1.34)

    def test_openai_position_guidance_reads_portfolio_decision_position_action(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub_dir = os.path.join(td, "hub_data")
            openai_dir = os.path.join(hub_dir, "openai")
            os.makedirs(openai_dir, exist_ok=True)
            with open(os.path.join(hub_dir, "trade_history.jsonl"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(openai_dir, "position_review.json"), "w", encoding="utf-8") as f:
                json.dump({"ts": 1000, "position_actions": []}, f)
            with open(os.path.join(openai_dir, "portfolio_decision_last_result.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 2000,
                        "decision": {
                            "position_actions": [
                                {
                                    "market": "crypto",
                                    "symbol": "DOT",
                                    "action": "hold",
                                    "confidence": 0.72,
                                    "reason": "Alignment remains supportive.",
                                }
                            ]
                        },
                    },
                    f,
                )

            chart = CandleChart.__new__(CandleChart)
            chart.coin = "DOT"
            chart.trade_history_path = os.path.join(hub_dir, "trade_history.jsonl")

            out = CandleChart._openai_position_guidance(chart)
            self.assertEqual(str(out.get("action", "")), "hold")
            self.assertEqual(str(out.get("source", "")), "portfolio_decision")
            self.assertAlmostEqual(float(out.get("confidence", 0.0) or 0.0), 0.72, places=6)

    def test_build_next_hour_projection_returns_zone_with_future_tick(self) -> None:
        chart = CandleChart.__new__(CandleChart)
        out = CandleChart._build_next_hour_projection(
            chart,
            timeframe="1hour",
            candle_count=120,
            last_candle_ts=1_714_694_800,
            local_outlook={
                "bias": "bullish",
                "ref_price": 1.32,
                "target_up": 1.36,
                "target_down": 1.29,
                "up_pct": 3.03,
                "confidence": 0.81,
            },
            fallback_price=1.31,
        )
        self.assertTrue(bool(out))
        self.assertGreater(float(out.get("zone_end_x", 0.0) or 0.0), float(out.get("zone_start_x", 0.0) or 0.0))
        self.assertEqual(int(out.get("horizon_bars", 0) or 0), 1)
        self.assertIn("+1h", str(out.get("future_tick_label", "")))

    def test_build_next_hour_projection_caps_display_span_for_fast_timeframes(self) -> None:
        chart = CandleChart.__new__(CandleChart)
        out = CandleChart._build_next_hour_projection(
            chart,
            timeframe="1min",
            candle_count=80,
            last_candle_ts=1_714_694_800,
            local_outlook={
                "bias": "bearish",
                "ref_price": 100.0,
                "target_down": 98.0,
                "target_up": 102.0,
                "down_pct": -2.0,
                "confidence": 0.66,
            },
            fallback_price=100.0,
        )
        self.assertTrue(bool(out))
        self.assertEqual(int(out.get("horizon_bars", 0) or 0), 60)
        self.assertLessEqual(float(out.get("display_span", 0.0) or 0.0), 16.0)


if __name__ == "__main__":
    unittest.main()
