from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines import stock_trader


class _FakeAlpacaClient:
    close_calls = 0
    equity = 10_000.0
    mid_price = 102.0
    positions = [
        {
            "symbol": "AAPL",
            "qty": "1",
            "avg_entry_price": "100.0",
            "market_value": "102.0",
        }
    ]

    @classmethod
    def reset(cls) -> None:
        cls.close_calls = 0
        cls.equity = 10_000.0
        cls.mid_price = 102.0
        cls.positions = [
            {
                "symbol": "AAPL",
                "qty": "1",
                "avg_entry_price": "100.0",
                "market_value": "102.0",
            }
        ]

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        pass

    def configured(self) -> bool:
        return True

    def list_positions(self) -> list[dict]:
        return [dict(row) for row in list(self.positions or [])]

    def get_mid_prices(self, symbols: list[str]) -> dict[str, float]:
        return {str(s).strip().upper(): float(self.mid_price) for s in symbols}

    def get_account_summary(self) -> dict:
        return {"equity": float(self.equity)}

    def get_snapshot_details(self, symbols: list[str]) -> dict:
        return {str(s).strip().upper(): {"mid": float(self.mid_price), "spread_bps": 1.0} for s in symbols}

    def place_market_order(self, symbol: str, side: str, notional: float, client_order_id: str, max_retries: int = 2, max_retry_after_s: float = 300.0):
        return False, "entries disabled in test", {}

    def close_position(self, symbol: str):
        type(self).close_calls += 1
        return True, "ok", {"id": "close-1"}


class TestStockTraderPDTHoldPolicy(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _read_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_jsonl(self, path: str) -> list[dict]:
        rows: list[dict] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                txt = str(ln or "").strip()
                if not txt:
                    continue
                rows.append(json.loads(txt))
        return rows

    def test_same_day_exit_blocked_by_default_hold_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _FakeAlpacaClient.reset()
            now_ts = 1_700_000_100
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(os.path.join(stocks_dir, "stock_thinker_status.json"), {"updated_at": now_ts})
            self._write_json(
                os.path.join(stocks_dir, "stock_trader_state.json"),
                {
                    "trail": {"AAPL": {"armed": True, "peak_pct": 3.40}},
                    "open_meta": {"AAPL": {"entry_ts": now_ts - 1800, "mfe_pct": 3.40, "mae_pct": 0.0}},
                },
            )
            settings = {
                "stock_auto_trade_enabled": False,
                "stock_profit_target_pct": 0.35,
                "stock_trailing_gap_pct": 0.20,
                "stock_min_hold_minutes": 1440,
                "stock_same_day_exit_exception_enabled": False,
                "market_rollout_stage": "execution_v2",
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_FakeAlpacaClient.close_calls, 0)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertIn("pdt hold gate", " ".join([str(x) for x in list(out.get("actions", []) or [])]).lower())
            state = self._read_json(os.path.join(stocks_dir, "stock_trader_state.json"))
            self.assertIn("AAPL", (state.get("trail", {}) or {}))

    def test_same_day_exit_exception_requires_boom_pullback_and_score_flip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _FakeAlpacaClient.reset()
            now_ts = 1_700_000_100
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [
                        {
                            "symbol": "AAPL",
                            "side": "watch",
                            "score": -0.30,
                            "eligible_for_entry": True,
                            "reason_logic": "Momentum rollover after sharp intraday spike",
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(stocks_dir, "stock_trader_state.json"),
                {
                    "trail": {"AAPL": {"armed": True, "peak_pct": 4.20}},
                    "open_meta": {"AAPL": {"entry_ts": now_ts - 3600, "mfe_pct": 4.20, "mae_pct": 0.0}},
                },
            )
            settings = {
                "stock_auto_trade_enabled": False,
                "stock_profit_target_pct": 0.35,
                "stock_trailing_gap_pct": 0.20,
                "stock_min_hold_minutes": 1440,
                "stock_same_day_exit_exception_enabled": True,
                "stock_same_day_exception_min_hold_minutes": 30,
                "stock_same_day_exception_min_pnl_pct": 1.5,
                "stock_same_day_exception_min_pullback_pct": 0.8,
                "stock_same_day_exception_require_score_flip": True,
                "stock_same_day_exception_score_floor_mult": 0.75,
                "stock_pdt_equity_threshold_usd": 25_000.0,
                "stock_pdt_max_day_trades_rolling_5d": 3,
                "market_rollout_stage": "execution_v2",
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_FakeAlpacaClient.close_calls, 1)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(int(out.get("day_trades_today", 0) or 0), 1)
            self.assertEqual(int(out.get("day_trades_rolling_5d", 0) or 0), 1)
            audit_rows = self._read_jsonl(os.path.join(stocks_dir, "execution_audit.jsonl"))
            exits = [row for row in audit_rows if str(row.get("event", "")) == "exit"]
            self.assertTrue(exits)
            self.assertTrue(bool(exits[-1].get("intraday_exception_used", False)))
            self.assertTrue(bool(exits[-1].get("same_day_roundtrip", False)))

    def test_same_day_exit_exception_blocked_when_pdt_window_cap_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _FakeAlpacaClient.reset()
            now_ts = 1_700_000_100
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            self._write_json(
                os.path.join(stocks_dir, "stock_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [{"symbol": "AAPL", "side": "watch", "score": -0.60}],
                },
            )
            self._write_json(
                os.path.join(stocks_dir, "stock_trader_state.json"),
                {
                    "trail": {"AAPL": {"armed": True, "peak_pct": 4.00}},
                    "open_meta": {"AAPL": {"entry_ts": now_ts - 3600, "mfe_pct": 4.00, "mae_pct": 0.0}},
                    "day_trade_events": [now_ts - 600, now_ts - 1800, now_ts - 7200],
                },
            )
            settings = {
                "stock_auto_trade_enabled": False,
                "stock_profit_target_pct": 0.35,
                "stock_trailing_gap_pct": 0.20,
                "stock_min_hold_minutes": 1440,
                "stock_same_day_exit_exception_enabled": True,
                "stock_same_day_exception_min_hold_minutes": 10,
                "stock_same_day_exception_min_pnl_pct": 1.0,
                "stock_same_day_exception_min_pullback_pct": 0.5,
                "stock_same_day_exception_require_score_flip": True,
                "stock_same_day_exception_score_floor_mult": 0.75,
                "stock_pdt_equity_threshold_usd": 25_000.0,
                "stock_pdt_max_day_trades_rolling_5d": 3,
                "market_rollout_stage": "execution_v2",
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _FakeAlpacaClient),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_FakeAlpacaClient.close_calls, 0)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(int(out.get("day_trades_rolling_5d", 0) or 0), 3)
            self.assertIn(
                "pdt guard",
                " ".join([str(x) for x in list(out.get("actions", []) or [])]).lower(),
            )


if __name__ == "__main__":
    unittest.main()
