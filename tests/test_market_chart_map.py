from __future__ import annotations

import unittest

from engines.forex_thinker import _build_top_chart_map as build_fx_chart_map
from engines.forex_thinker import _compact_chart_bars as compact_fx_chart_bars
from engines.stock_thinker import _build_top_chart_map as build_stock_chart_map
from engines.stock_thinker import _compact_chart_bars as compact_stock_chart_bars


class TestMarketChartMap(unittest.TestCase):
    def test_stock_compact_chart_bars_filters_invalid_rows(self) -> None:
        bars = [
            {"t": "2026-03-01T00:00:00Z", "o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 1000},
            {"t": "2026-03-01T01:00:00Z", "o": 10.5, "h": 10.8, "l": 10.2, "c": 0.0},
            {"t": "2026-03-01T02:00:00Z", "o": 10.5, "h": 11.2, "l": 10.4, "c": 11.0},
        ]
        out = compact_stock_chart_bars(bars, limit=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["t"], "2026-03-01T00:00:00Z")
        self.assertEqual(out[1]["t"], "2026-03-01T02:00:00Z")

    def test_stock_build_chart_map_keeps_top_symbols_only(self) -> None:
        leaders = [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "AAPL"}]
        lookup = {
            "AAPL": [
                {"t": "t1", "o": 1.0, "h": 1.2, "l": 0.9, "c": 1.1},
                {"t": "t2", "o": 1.1, "h": 1.3, "l": 1.0, "c": 1.2},
            ],
            "MSFT": [
                {"t": "t1", "o": 2.0, "h": 2.2, "l": 1.9, "c": 2.1},
                {"t": "t2", "o": 2.1, "h": 2.3, "l": 2.0, "c": 2.2},
            ],
        }
        out = build_stock_chart_map(leaders, lookup, max_symbols=1, limit=20)
        self.assertEqual(set(out.keys()), {"AAPL"})
        self.assertEqual(len(out["AAPL"]), 2)

    def test_stock_build_chart_map_prioritizes_include_symbols(self) -> None:
        leaders = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        lookup = {
            "AGRO": [
                {"t": "t1", "o": 3.0, "h": 3.2, "l": 2.9, "c": 3.1},
                {"t": "t2", "o": 3.1, "h": 3.3, "l": 3.0, "c": 3.2},
            ],
            "AAPL": [
                {"t": "t1", "o": 1.0, "h": 1.2, "l": 0.9, "c": 1.1},
                {"t": "t2", "o": 1.1, "h": 1.3, "l": 1.0, "c": 1.2},
            ],
            "MSFT": [
                {"t": "t1", "o": 2.0, "h": 2.2, "l": 1.9, "c": 2.1},
                {"t": "t2", "o": 2.1, "h": 2.3, "l": 2.0, "c": 2.2},
            ],
        }
        out = build_stock_chart_map(
            leaders,
            lookup,
            max_symbols=2,
            limit=20,
            include_symbols=["AGRO"],
        )
        self.assertEqual(list(out.keys())[0], "AGRO")
        self.assertEqual(len(out["AGRO"]), 2)

    def test_forex_compact_chart_bars_maps_mid_payload(self) -> None:
        candles = [
            {"time": "2026-03-01T00:00:00Z", "mid": {"o": "1.1000", "h": "1.1010", "l": "1.0990", "c": "1.1005"}, "volume": 100},
            {"time": "2026-03-01T01:00:00Z", "mid": {"o": "1.1005", "h": "1.1015", "l": "1.1000", "c": "1.1010"}, "volume": 120},
        ]
        out = compact_fx_chart_bars(candles, limit=10)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out[0]["c"]), 1.1005, places=6)
        self.assertEqual(int(out[1]["v"]), 120)

    def test_forex_build_chart_map_dedupes_pairs(self) -> None:
        leaders = [{"pair": "EUR_USD"}, {"pair": "EUR_USD"}, {"pair": "USD_JPY"}]
        lookup = {
            "EUR_USD": [
                {"time": "t1", "mid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.15"}, "volume": 100},
                {"time": "t2", "mid": {"o": "1.15", "h": "1.21", "l": "1.12", "c": "1.18"}, "volume": 100},
            ],
            "USD_JPY": [
                {"time": "t1", "mid": {"o": "150.0", "h": "150.4", "l": "149.8", "c": "150.2"}, "volume": 90},
                {"time": "t2", "mid": {"o": "150.2", "h": "150.5", "l": "150.0", "c": "150.3"}, "volume": 85},
            ],
        }
        out = build_fx_chart_map(leaders, lookup, max_pairs=2, limit=50)
        self.assertEqual(set(out.keys()), {"EUR_USD", "USD_JPY"})
        self.assertEqual(len(out["EUR_USD"]), 2)


if __name__ == "__main__":
    unittest.main()
