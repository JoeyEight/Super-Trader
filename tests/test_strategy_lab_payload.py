from __future__ import annotations

import os
import tempfile
import unittest

from app.strategy_lab import run_strategy_lab_suite


class StrategyLabPayloadTests(unittest.TestCase):
    def test_stocks_payload_includes_simulation_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "stocks"), exist_ok=True)
            payload = run_strategy_lab_suite(
                td,
                "stocks",
                settings={
                    "stock_score_threshold": 0.25,
                    "stock_trade_notional_usd": 20.0,
                    "stock_max_open_positions": 4,
                    "strategy_lab_fee_bps": 2.0,
                    "strategy_lab_monte_carlo_sims": 50,
                    "strategy_lab_seed": 7,
                },
            )
            self.assertEqual(str(payload.get("market", "")), "stocks")
            self.assertIn("simulation", payload)
            self.assertIn("recommendations", payload)
            sim = payload.get("simulation", {}) if isinstance(payload.get("simulation", {}), dict) else {}
            rec = payload.get("recommendations", {}) if isinstance(payload.get("recommendations", {}), dict) else {}
            settings = rec.get("settings", {}) if isinstance(rec.get("settings", {}), dict) else {}
            self.assertIn("starting_equity_usd", sim)
            self.assertIn("ending_equity_usd", sim)
            self.assertIn("stock_score_threshold", settings)
            self.assertIn("stock_trade_notional_usd", settings)
            self.assertIn("stock_max_open_positions", settings)

    def test_forex_payload_includes_simulation_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            payload = run_strategy_lab_suite(
                td,
                "forex",
                settings={
                    "forex_score_threshold": 0.2,
                    "forex_trade_units": 25,
                    "forex_max_open_positions": 5,
                    "strategy_lab_fee_bps": 2.0,
                    "strategy_lab_monte_carlo_sims": 50,
                    "strategy_lab_seed": 7,
                },
            )
            self.assertEqual(str(payload.get("market", "")), "forex")
            self.assertIn("simulation", payload)
            self.assertIn("recommendations", payload)
            sim = payload.get("simulation", {}) if isinstance(payload.get("simulation", {}), dict) else {}
            rec = payload.get("recommendations", {}) if isinstance(payload.get("recommendations", {}), dict) else {}
            settings = rec.get("settings", {}) if isinstance(rec.get("settings", {}), dict) else {}
            self.assertIn("starting_equity_usd", sim)
            self.assertIn("ending_equity_usd", sim)
            self.assertIn("forex_score_threshold", settings)
            self.assertIn("forex_trade_units", settings)
            self.assertIn("forex_max_open_positions", settings)


if __name__ == "__main__":
    unittest.main()

