from __future__ import annotations

import unittest

from engines.forex_thinker import _score_candles
from engines.stock_thinker import _score_bars


def _stock_bars(closes: list[float]) -> list[dict]:
    out: list[dict] = []
    for idx, px in enumerate(list(closes or [])):
        out.append(
            {
                "t": f"2026-04-01T{idx:02d}:00:00Z",
                "c": float(px),
            }
        )
    return out


def _forex_candles(closes: list[float]) -> list[dict]:
    out: list[dict] = []
    for idx, px in enumerate(list(closes or [])):
        val = float(px)
        out.append(
            {
                "complete": True,
                "time": f"2026-04-01T{idx:02d}:00:00Z",
                "mid": {
                    "c": f"{val:.6f}",
                    "h": f"{(val * 1.0005):.6f}",
                    "l": f"{(val * 0.9995):.6f}",
                },
            }
        )
    return out


class TestMarketScannerOutlierGuardrails(unittest.TestCase):
    def test_stock_score_marks_outlier_jump(self) -> None:
        closes = ([0.90] * 9) + ([18.00] * 23)
        row = _score_bars("ADV", _stock_bars(closes), spread_bps=0.0)
        self.assertEqual(float(row.get("score", 0.0) or 0.0), -9999.0)
        self.assertTrue(bool(row.get("outlier", False)))
        self.assertIn("Outlier", str(row.get("reason_logic", "")))

    def test_stock_score_keeps_normal_series(self) -> None:
        closes = [100.0 + (0.12 * i) for i in range(32)]
        row = _score_bars("AAPL", _stock_bars(closes), spread_bps=2.0)
        self.assertGreater(float(row.get("score", -9999.0) or -9999.0), -9999.0)
        self.assertFalse(bool(row.get("outlier", False)))

    def test_forex_score_marks_outlier_jump(self) -> None:
        closes = ([1.0000] * 9) + ([1.3200] * 23)
        row = _score_candles("EUR_USD", _forex_candles(closes), spread_bps=1.0)
        self.assertEqual(float(row.get("score", 0.0) or 0.0), -9999.0)
        self.assertTrue(bool(row.get("outlier", False)))
        self.assertIn("Outlier", str(row.get("reason_logic", "")))

    def test_forex_score_keeps_normal_series(self) -> None:
        closes = [1.1000 + (0.0008 * i) for i in range(32)]
        row = _score_candles("EUR_USD", _forex_candles(closes), spread_bps=1.0)
        self.assertGreater(float(row.get("score", -9999.0) or -9999.0), -9999.0)
        self.assertFalse(bool(row.get("outlier", False)))


if __name__ == "__main__":
    unittest.main()
