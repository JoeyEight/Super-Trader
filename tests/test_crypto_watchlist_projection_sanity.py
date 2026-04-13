from __future__ import annotations

import json
import os
import re
import tempfile
import types
import unittest
from unittest.mock import patch

from ui.pt_hub import PowerTraderHub


class _DummyLabel:
    def __init__(self) -> None:
        self.last_text = ""

    def config(self, **kwargs) -> None:
        if "text" in kwargs:
            self.last_text = str(kwargs.get("text", ""))


class _DummyCanvas:
    pass


def _parse_money(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    raw = raw.replace("$", "").replace(",", "")
    try:
        return float(raw)
    except Exception:
        return 0.0


def _parse_pct(text: str) -> float:
    raw = str(text or "").strip().replace("%", "").replace("+", "")
    try:
        return float(raw)
    except Exception:
        return 0.0


class TestCryptoWatchlistProjectionSanity(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_watchlist_rejects_mismatched_cached_price_for_projection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now_ts = 1_700_300_100.0
            hub = object.__new__(PowerTraderHub)
            hub.project_dir = td
            hub.hub_dir = os.path.join(td, "hub_data")
            os.makedirs(hub.hub_dir, exist_ok=True)
            hub.crypto_current_prices_dir = os.path.join(hub.hub_dir, "current_prices")
            os.makedirs(hub.crypto_current_prices_dir, exist_ok=True)
            hub.crypto_dynamic_status_path = os.path.join(hub.hub_dir, "crypto_dynamic_status.json")
            hub.trader_data_path = os.path.join(hub.hub_dir, "trader_data.json")
            hub.crypto_watchlist_canvas = _DummyCanvas()
            hub.lbl_crypto_watchlist_meta = _DummyLabel()
            hub._crypto_watchlist_last_refresh_ts = 0.0
            hub._crypto_watchlist_last_sig = None
            hub._crypto_watchlist_rows = []
            hub.settings = {
                "coins": ["ETH", "SOL"],  # intentionally excludes BTC
                "main_neural_dir": td,
                "trade_start_level": 2,
                "crypto_dynamic_min_projected_edge_pct": 0.14,
                "crypto_watchlist_rows_limit": 50,
            }
            hub._draw_crypto_watchlist_table = types.MethodType(lambda self: None, hub)

            self._write_json(
                hub.crypto_dynamic_status_path,
                {
                    "ranked": [{"symbol": "BTC", "score": 0.45, "trained": True}],
                    "current_coins": [],
                    "min_projected_edge_pct": 0.14,
                },
            )
            self._write_json(hub.trader_data_path, {"positions": {}})

            btc_dir = os.path.join(td, "BTC")
            os.makedirs(btc_dir, exist_ok=True)
            with open(os.path.join(btc_dir, "low_bound_prices.html"), "w", encoding="utf-8") as f:
                f.write("67865.83, 67495.60, 0.01, 0.01, 0.01")
            with open(os.path.join(btc_dir, "high_bound_prices.html"), "w", encoding="utf-8") as f:
                f.write("68824.83, 69139.68, 99999999999999999")
            with open(os.path.join(btc_dir, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("0")
            with open(os.path.join(btc_dir, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
                f.write("0")

            btc_price_path = os.path.join(hub.crypto_current_prices_dir, "BTC.txt")
            with open(btc_price_path, "w", encoding="utf-8") as f:
                f.write("101.0")
            os.utime(btc_price_path, (now_ts, now_ts))  # fresh but mismatched with model levels

            with patch("ui.pt_hub.time.time", return_value=now_ts):
                hub._refresh_crypto_watchlist_overview()

            rows = list(getattr(hub, "_crypto_watchlist_rows", []) or [])
            self.assertTrue(rows)
            btc_rows = [row for row in rows if str(row.get("coin", "")).strip().upper() == "BTC"]
            self.assertTrue(btc_rows)
            btc = btc_rows[0]
            entry_val = _parse_money(str(btc.get("entry", "")))
            exit_val = _parse_money(str(btc.get("exit", "")))
            gain_val = _parse_pct(str(btc.get("gain", "")))

            # If stale/mismatched cached price is used, entry would be near $101 and gain gigantic.
            self.assertGreater(entry_val, 10_000.0)
            self.assertGreater(exit_val, 10_000.0)
            self.assertLess(abs(gain_val), 50.0)


if __name__ == "__main__":
    unittest.main()
