from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from engines import stock_thinker
from engines import stock_trader


class TestStockMarketHours(unittest.TestCase):
    def test_good_friday_is_closed(self) -> None:
        self.assertTrue(stock_thinker._is_us_stock_holiday(date(2026, 4, 3)))
        self.assertFalse(
            stock_thinker._market_open_now(
                datetime(2026, 4, 3, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )

    def test_regular_session_open(self) -> None:
        self.assertFalse(stock_thinker._is_us_stock_holiday(date(2026, 4, 6)))
        self.assertTrue(
            stock_thinker._market_open_now(
                datetime(2026, 4, 6, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )

    def test_weekend_closed(self) -> None:
        self.assertFalse(
            stock_thinker._market_open_now(
                datetime(2026, 4, 4, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )

    def test_observed_new_year_close(self) -> None:
        self.assertTrue(stock_thinker._is_us_stock_holiday(date(2021, 12, 31)))
        self.assertFalse(
            stock_thinker._market_open_now(
                datetime(2021, 12, 31, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )

    def test_stock_trader_respects_good_friday_holiday(self) -> None:
        self.assertFalse(
            stock_trader._market_open_now(
                datetime(2026, 4, 3, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )

    def test_stock_trader_regular_session_open(self) -> None:
        self.assertTrue(
            stock_trader._market_open_now(
                datetime(2026, 4, 6, 11, 0, tzinfo=ZoneInfo("America/New_York"))
            )
        )


if __name__ == "__main__":
    unittest.main()
