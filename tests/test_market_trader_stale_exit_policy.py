from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines import forex_trader, stock_trader


class _StockStaleExitClient:
    close_calls = 0
    place_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.close_calls = 0
        cls.place_calls = 0

    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        del api_key_id, secret_key, base_url, data_url

    def configured(self) -> bool:
        return True

    def list_positions(self) -> list[dict]:
        return [
            {
                "symbol": "AAPL",
                "qty": "1",
                "avg_entry_price": "100.0",
                "market_value": "104.0",
            }
        ]

    def get_mid_prices(self, symbols: list[str]) -> dict[str, float]:
        return {str(sym).strip().upper(): 104.0 for sym in symbols}

    def get_account_summary(self) -> dict:
        return {
            "equity": 50_000.0,
            "buying_power": 50_000.0,
            "account_type": "margin",
            "multiplier": "2",
        }

    def get_snapshot_details(self, symbols: list[str]) -> dict:
        return {str(sym).strip().upper(): {"mid": 104.0, "spread_bps": 1.0} for sym in symbols}

    def place_market_order(
        self,
        symbol: str,
        side: str,
        notional: float,
        client_order_id: str,
        max_retries: int = 2,
        max_retry_after_s: float = 300.0,
    ):
        del symbol, side, notional, client_order_id, max_retries, max_retry_after_s
        type(self).place_calls += 1
        return True, "unexpected", {"id": "stock-entry"}

    def close_position(self, symbol: str):
        del symbol
        type(self).close_calls += 1
        return True, "ok", {"id": "stock-close-1"}


class _ForexStaleExitClient:
    close_calls = 0
    place_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.close_calls = 0
        cls.place_calls = 0

    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        del account_id, api_token, rest_url

    def configured(self) -> bool:
        return True

    def fetch_snapshot(self) -> dict:
        return {
            "raw_positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "averagePrice": "1.1000"},
                    "short": {"units": "0", "averagePrice": "0"},
                    "marginUsed": "45.0",
                }
            ],
            "nav": 10_000.0,
            "margin_available": 9_500.0,
            "margin_rate": 0.05,
        }

    def get_pricing_details(self, instruments: list[str]) -> dict:
        return {str(inst).strip().upper(): {"mid": 1.1200, "spread_bps": 1.1} for inst in instruments}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(inst).strip().upper(): 1.1200 for inst in instruments}

    def place_market_order(
        self,
        instrument: str,
        units: int,
        client_order_id: str,
        max_retries: int = 2,
        max_retry_after_s: float = 300.0,
    ):
        del instrument, units, client_order_id, max_retries, max_retry_after_s
        type(self).place_calls += 1
        return True, "unexpected", {"id": "forex-entry"}

    def close_position(self, instrument: str, side: str):
        del instrument, side
        type(self).close_calls += 1
        return True, "ok", {"id": "forex-close-1", "orderFillTransaction": {"realizedPL": "12.50"}}


class _ForexStaleHoldGuardClient(_ForexStaleExitClient):
    def fetch_snapshot(self) -> dict:
        return {
            "raw_positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "averagePrice": "1.1205"},
                    "short": {"units": "0", "averagePrice": "0"},
                    "marginUsed": "45.0",
                }
            ],
            "nav": 10_000.0,
            "margin_available": 9_500.0,
            "margin_rate": 0.05,
        }

    def get_pricing_details(self, instruments: list[str]) -> dict:
        return {str(inst).strip().upper(): {"mid": 1.1200, "spread_bps": 1.1} for inst in instruments}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(inst).strip().upper(): 1.1200 for inst in instruments}


class _ForexStaleNearFlatGainClient(_ForexStaleExitClient):
    def fetch_snapshot(self) -> dict:
        return {
            "raw_positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "averagePrice": "1.11995"},
                    "short": {"units": "0", "averagePrice": "0"},
                    "marginUsed": "45.0",
                }
            ],
            "nav": 10_000.0,
            "margin_available": 9_500.0,
            "margin_rate": 0.05,
        }

    def get_pricing_details(self, instruments: list[str]) -> dict:
        return {str(inst).strip().upper(): {"mid": 1.1200, "spread_bps": 1.1} for inst in instruments}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(inst).strip().upper(): 1.1200 for inst in instruments}


class _ForexStaleStrongGainClient(_ForexStaleExitClient):
    def fetch_snapshot(self) -> dict:
        return {
            "raw_positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {"units": "1000", "averagePrice": "1.1150"},
                    "short": {"units": "0", "averagePrice": "0"},
                    "marginUsed": "45.0",
                }
            ],
            "nav": 10_000.0,
            "margin_available": 9_500.0,
            "margin_rate": 0.05,
        }

    def get_pricing_details(self, instruments: list[str]) -> dict:
        return {str(inst).strip().upper(): {"mid": 1.1200, "spread_bps": 1.1} for inst in instruments}

    def get_mid_prices(self, instruments: list[str]) -> dict[str, float]:
        return {str(inst).strip().upper(): 1.1200 for inst in instruments}


class TestMarketTraderStaleExitPolicy(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _read_jsonl(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                txt = str(ln or "").strip()
                if not txt:
                    continue
                rows.append(json.loads(txt))
        return rows

    def test_stock_trader_purges_stale_position_when_alignment_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _StockStaleExitClient.reset()
            now_ts = 1_700_020_100
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
                            "score": -0.35,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(stocks_dir, "stock_trader_state.json"),
                {
                    "open_meta": {"AAPL": {"entry_ts": now_ts - 172800, "mfe_pct": 2.0, "mae_pct": 0.0}},
                    "trail": {},
                },
            )
            settings = {
                "stock_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "stock_score_threshold": 0.20,
                "stock_trade_notional_usd": 100.0,
                "stock_stale_exit_enabled": True,
                "stock_stale_alignment_grace_cycles": 1,
                "stock_stale_max_exits_per_cycle": 2,
                "stock_stale_min_notional_usd": 1.0,
                "stock_max_open_positions": 3,
                "stock_max_signal_age_seconds": 600,
            }
            with (
                patch.object(stock_trader, "get_alpaca_creds", return_value=("key", "secret")),
                patch.object(stock_trader, "AlpacaBrokerClient", _StockStaleExitClient),
                patch.object(stock_trader, "_market_open_now", return_value=True),
                patch.object(stock_trader, "_near_close_blocked", return_value=False),
                patch.object(stock_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.stock_trader.time.time", return_value=now_ts),
            ):
                out = stock_trader.run_step(settings, td)
            self.assertEqual(_StockStaleExitClient.close_calls, 1)
            self.assertEqual(_StockStaleExitClient.place_calls, 0)
            self.assertEqual(int(out.get("stale_exit_count", 0) or 0), 1)
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertTrue(bool(flags.get("skip_new_entries_this_cycle", False)))
            audit_rows = self._read_jsonl(os.path.join(stocks_dir, "execution_audit.jsonl"))
            exits = [row for row in audit_rows if str(row.get("event", "")) == "exit" and str(row.get("source", "")) == "policy_stale_exit"]
            self.assertTrue(exits)

    def test_forex_trader_purges_stale_position_when_alignment_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexStaleExitClient.reset()
            now_ts = 1_700_020_200
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "short",
                            "score": -0.55,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(fx_dir, "forex_trader_state.json"),
                {
                    "open_meta": {"EUR_USD": {"entry_ts": now_ts - 7200, "mfe_pct": 1.8, "mae_pct": 0.0}},
                    "trail": {},
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_score_threshold": 0.20,
                "forex_trade_units": 1000,
                "forex_stale_exit_enabled": True,
                "forex_stale_alignment_grace_cycles": 1,
                "forex_stale_max_exits_per_cycle": 2,
                "forex_stale_min_notional_usd": 1.0,
                "forex_max_signal_age_seconds": 600,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexStaleExitClient),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexStaleExitClient.close_calls, 1)
            self.assertEqual(_ForexStaleExitClient.place_calls, 0)
            self.assertEqual(int(out.get("stale_exit_count", 0) or 0), 1)
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertTrue(bool(flags.get("skip_new_entries_this_cycle", False)))
            audit_rows = self._read_jsonl(os.path.join(fx_dir, "execution_audit.jsonl"))
            exits = [row for row in audit_rows if str(row.get("event", "")) == "exit" and str(row.get("source", "")) == "policy_stale_exit"]
            self.assertTrue(exits)

    def test_forex_stale_exit_hold_guard_blocks_fresh_mild_loss_churn_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexStaleHoldGuardClient.reset()
            now_ts = 1_700_020_260
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "short",
                            "score": -0.22,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(fx_dir, "forex_trader_state.json"),
                {
                    "open_meta": {"EUR_USD": {"entry_ts": now_ts - 120, "mfe_pct": 0.2, "mae_pct": -0.1}},
                    "trail": {},
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_score_threshold": 0.20,
                "forex_trade_units": 1000,
                "forex_stale_exit_enabled": True,
                "forex_stale_alignment_grace_cycles": 1,
                "forex_stale_max_exits_per_cycle": 2,
                "forex_stale_min_notional_usd": 1.0,
                "forex_stale_min_hold_seconds": 3600,
                "forex_stale_loss_cut_pct": -0.35,
                "forex_max_signal_age_seconds": 600,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexStaleHoldGuardClient),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexStaleHoldGuardClient.close_calls, 0)
            self.assertEqual(int(out.get("stale_exit_count", 0) or 0), 0)
            stale_events = out.get("stale_exit_events", []) if isinstance(out.get("stale_exit_events", []), list) else []
            reasons = [str((row or {}).get("reason", "") or "") for row in stale_events if isinstance(row, dict)]
            self.assertIn("stale_exit_hold_loss_guard", reasons)

    def test_forex_stale_exit_hold_guard_blocks_fresh_near_flat_gain_churn_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexStaleNearFlatGainClient.reset()
            now_ts = 1_700_020_320
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "short",
                            "score": -0.22,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(fx_dir, "forex_trader_state.json"),
                {
                    "open_meta": {"EUR_USD": {"entry_ts": now_ts - 120, "mfe_pct": 0.2, "mae_pct": -0.1}},
                    "trail": {},
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_score_threshold": 0.20,
                "forex_trade_units": 1000,
                "forex_stale_exit_enabled": True,
                "forex_stale_alignment_grace_cycles": 1,
                "forex_stale_max_exits_per_cycle": 2,
                "forex_stale_min_notional_usd": 1.0,
                "forex_stale_min_hold_seconds": 3600,
                "forex_stale_loss_cut_pct": -0.35,
                "forex_stale_hold_near_flat_pct": 0.10,
                "forex_max_signal_age_seconds": 600,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexStaleNearFlatGainClient),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexStaleNearFlatGainClient.close_calls, 0)
            self.assertEqual(int(out.get("stale_exit_count", 0) or 0), 0)
            stale_events = out.get("stale_exit_events", []) if isinstance(out.get("stale_exit_events", []), list) else []
            reasons = [str((row or {}).get("reason", "") or "") for row in stale_events if isinstance(row, dict)]
            self.assertIn("stale_exit_hold_churn_guard", reasons)
            flags = out.get("entry_gate_flags", {}) if isinstance(out.get("entry_gate_flags", {}), dict) else {}
            self.assertAlmostEqual(float(flags.get("stale_exit_hold_near_flat_pct", 0.0) or 0.0), 0.10, places=4)

    def test_forex_stale_exit_does_not_block_fresh_strong_gain_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _ForexStaleStrongGainClient.reset()
            now_ts = 1_700_020_360
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            self._write_json(
                os.path.join(fx_dir, "forex_thinker_status.json"),
                {
                    "updated_at": now_ts,
                    "adaptive_threshold": 0.20,
                    "leaders": [
                        {
                            "pair": "EUR_USD",
                            "side": "short",
                            "score": -0.22,
                            "eligible_for_entry": True,
                            "data_quality_ok": True,
                        }
                    ],
                },
            )
            self._write_json(
                os.path.join(fx_dir, "forex_trader_state.json"),
                {
                    "open_meta": {"EUR_USD": {"entry_ts": now_ts - 120, "mfe_pct": 0.2, "mae_pct": -0.1}},
                    "trail": {},
                },
            )
            settings = {
                "forex_auto_trade_enabled": True,
                "market_rollout_stage": "execution_v2",
                "forex_score_threshold": 0.20,
                "forex_trade_units": 1000,
                "forex_stale_exit_enabled": True,
                "forex_stale_alignment_grace_cycles": 1,
                "forex_stale_max_exits_per_cycle": 2,
                "forex_stale_min_notional_usd": 1.0,
                "forex_stale_min_hold_seconds": 3600,
                "forex_stale_loss_cut_pct": -0.35,
                "forex_stale_hold_near_flat_pct": 0.10,
                "forex_max_signal_age_seconds": 600,
                "forex_session_mode": "all",
            }
            with (
                patch.object(forex_trader, "get_oanda_creds", return_value=("acct", "token")),
                patch.object(forex_trader, "OandaBrokerClient", _ForexStaleStrongGainClient),
                patch.object(forex_trader, "_session_blocked", return_value=False),
                patch.object(forex_trader, "_daily_loss_guard_triggered", return_value=False),
                patch("engines.forex_trader.time.time", return_value=now_ts),
            ):
                out = forex_trader.run_step(settings, td)
            self.assertEqual(_ForexStaleStrongGainClient.close_calls, 1)
            self.assertEqual(int(out.get("stale_exit_count", 0) or 0), 1)


if __name__ == "__main__":
    unittest.main()
