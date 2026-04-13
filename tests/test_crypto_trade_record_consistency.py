from __future__ import annotations

import base64
import importlib
import os
import types
import unittest


def _load_pt_trader_module():
    os.environ.setdefault("POWERTRADER_RH_API_KEY", "test-key")
    os.environ.setdefault("POWERTRADER_RH_PRIVATE_B64", base64.b64encode(b"0" * 32).decode("ascii"))
    return importlib.import_module("engines.pt_trader")


class CryptoTradeRecordConsistencyTests(unittest.TestCase):
    def test_record_trade_sell_uses_realized_cost_for_pnl_pct_with_buying_power_tracking(self) -> None:
        pt_trader = _load_pt_trader_module()
        bot = object.__new__(pt_trader.CryptoAPITrading)
        bot._pnl_ledger = {
            "total_realized_profit_usd": 0.0,
            "open_positions": {"BTC": {"usd_cost": 100.0, "qty": 1.0}},
            "pending_orders": {},
        }
        captured = []
        bot._save_pnl_ledger = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, _path, payload: captured.append(dict(payload)), bot)

        pt_trader.CryptoAPITrading._record_trade(
            bot,
            side="sell",
            symbol="BTC-USD",
            qty=1.0,
            price=120.0,
            avg_cost_basis=80.0,
            pnl_pct=-55.0,
            tag="POLICY_STALE_EXIT",
            buying_power_delta=120.0,
        )

        self.assertEqual(len(captured), 1)
        row = captured[0]
        self.assertAlmostEqual(float(row.get("realized_profit_usd", 0.0) or 0.0), 20.0, places=6)
        self.assertAlmostEqual(float(row.get("position_cost_used_usd", 0.0) or 0.0), 100.0, places=6)
        self.assertAlmostEqual(float(row.get("pnl_pct", 0.0) or 0.0), 20.0, places=6)
        self.assertAlmostEqual(float(bot._pnl_ledger.get("total_realized_profit_usd", 0.0) or 0.0), 20.0, places=6)

    def test_record_trade_sell_fallback_realized_keeps_pnl_pct_in_sync(self) -> None:
        pt_trader = _load_pt_trader_module()
        bot = object.__new__(pt_trader.CryptoAPITrading)
        bot._pnl_ledger = {
            "total_realized_profit_usd": 0.0,
            "open_positions": {},
            "pending_orders": {},
        }
        captured = []
        bot._save_pnl_ledger = types.MethodType(lambda self: None, bot)
        bot._append_jsonl = types.MethodType(lambda self, _path, payload: captured.append(dict(payload)), bot)

        pt_trader.CryptoAPITrading._record_trade(
            bot,
            side="sell",
            symbol="ETH-USD",
            qty=2.0,
            price=10.0,
            avg_cost_basis=8.0,
            pnl_pct=-9.0,
            tag="TRAIL_SELL",
            fees_usd=0.0,
            buying_power_delta=None,
        )

        self.assertEqual(len(captured), 1)
        row = captured[0]
        self.assertAlmostEqual(float(row.get("realized_profit_usd", 0.0) or 0.0), 4.0, places=6)
        self.assertAlmostEqual(float(row.get("pnl_pct", 0.0) or 0.0), 25.0, places=6)
        self.assertAlmostEqual(float(bot._pnl_ledger.get("total_realized_profit_usd", 0.0) or 0.0), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
