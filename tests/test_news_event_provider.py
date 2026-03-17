from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from app import news_event_provider


class TestNewsEventProvider(unittest.TestCase):
    def test_build_context_live_then_cached_then_stale_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "news_event_enabled": True,
                "news_event_refresh_s": 120.0,
                "news_event_stale_max_s": 3600.0,
                "news_event_timeout_s": 5.0,
                "news_event_max_symbols_per_market": 5,
                "news_event_max_headlines_per_symbol": 4,
            }
            headline_rows = [
                {"title": "AAPL upgrade and strong growth", "pub_date": "Tue, 17 Mar 2026 12:00:00 GMT", "ts": 1_710_000_000},
                {"title": "AAPL SEC filing update", "pub_date": "Tue, 17 Mar 2026 12:15:00 GMT", "ts": 1_710_000_900},
            ]
            with patch.object(news_event_provider, "_fetch_symbol_headlines", return_value=headline_rows):
                out_live = news_event_provider.build_unified_news_event_context(
                    hub_dir=td,
                    settings=settings,
                    market="stocks",
                    symbols=["AAPL"],
                    now_ts=1_710_001_200,
                )
            self.assertEqual(str(out_live.get("state", "")), "live")
            self.assertIn("AAPL", dict(out_live.get("symbols", {})))

            with patch.object(news_event_provider, "_fetch_symbol_headlines", side_effect=RuntimeError("should not fetch on fresh cache")):
                out_cached = news_event_provider.build_unified_news_event_context(
                    hub_dir=td,
                    settings=settings,
                    market="stocks",
                    symbols=["AAPL"],
                    now_ts=1_710_001_240,
                )
            self.assertEqual(str(out_cached.get("state", "")), "cached")
            self.assertIn("AAPL", dict(out_cached.get("symbols", {})))

            with patch.object(news_event_provider, "_fetch_symbol_headlines", side_effect=RuntimeError("feed down")):
                out_stale = news_event_provider.build_unified_news_event_context(
                    hub_dir=td,
                    settings=settings,
                    market="stocks",
                    symbols=["AAPL"],
                    now_ts=1_710_001_900,
                )
            self.assertEqual(str(out_stale.get("state", "")), "cached_stale")
            self.assertIn("AAPL", dict(out_stale.get("symbols", {})))
            self.assertGreaterEqual(int(out_stale.get("retry_after_s", 0) or 0), 1)

    def test_blend_score_with_news_respects_directionality(self) -> None:
        improved = news_event_provider.blend_score_with_news(
            base_score=0.25,
            news_score=0.80,
            confidence=0.90,
            impact=0.80,
            weight=0.20,
        )
        reduced = news_event_provider.blend_score_with_news(
            base_score=0.25,
            news_score=-0.80,
            confidence=0.90,
            impact=0.80,
            weight=0.20,
        )
        self.assertGreater(float(improved), 0.25)
        self.assertLess(float(reduced), 0.25)


if __name__ == "__main__":
    unittest.main()
