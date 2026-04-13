from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from engines.forex_thinker import _append_reason_parts, _load_forexfactory_context, _pair_event_risk


class TestForexEventRisk(unittest.TestCase):
    def test_high_impact_event_blocks_entry_window(self) -> None:
        now_ts = 1_700_000_000
        calendar_ctx = {
            "enabled": True,
            "state": "live",
            "events": [
                {
                    "ts": now_ts + (20 * 60),
                    "currency": "USD",
                    "impact": "high",
                    "title": "NFP",
                }
            ],
        }
        settings = {
            "forex_event_max_lookahead_minutes": 180,
            "forex_event_post_event_minutes": 30,
            "forex_event_block_high_impact_minutes": 45,
            "forex_event_score_mult_high": 0.70,
            "forex_event_score_mult_medium": 0.85,
        }
        out = _pair_event_risk("USD_JPY", calendar_ctx, now_ts, settings)
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("severity", "")), "high")
        self.assertTrue(bool(out.get("block_entry", False)))
        self.assertAlmostEqual(float(out.get("score_mult", 0.0) or 0.0), 0.70, places=6)
        self.assertIn("macro risk", str(out.get("logic", "")).lower())
        self.assertIn("forexfactory", str(out.get("data", "")).lower())

    def test_medium_impact_event_dampens_without_block(self) -> None:
        now_ts = 1_700_000_000
        calendar_ctx = {
            "enabled": True,
            "state": "cached",
            "events": [
                {
                    "ts": now_ts + (70 * 60),
                    "currency": "EUR",
                    "impact": "medium",
                    "title": "CPI",
                }
            ],
        }
        settings = {
            "forex_event_max_lookahead_minutes": 180,
            "forex_event_post_event_minutes": 30,
            "forex_event_block_high_impact_minutes": 45,
            "forex_event_score_mult_high": 0.70,
            "forex_event_score_mult_medium": 0.85,
        }
        out = _pair_event_risk("EUR_USD", calendar_ctx, now_ts, settings)
        self.assertTrue(bool(out.get("active", False)))
        self.assertEqual(str(out.get("severity", "")), "medium")
        self.assertFalse(bool(out.get("block_entry", False)))
        self.assertAlmostEqual(float(out.get("score_mult", 0.0) or 0.0), 0.85, places=6)

    def test_missing_cache_returns_unavailable_and_schedules_async_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "forex_event_risk_enabled": True,
                "forex_event_cache_refresh_s": 600.0,
                "forex_event_cache_stale_max_s": 3600.0,
            }
            with (
                patch("engines.forex_thinker._spawn_forexfactory_refresh", return_value=True) as spawn_mock,
                patch("engines.forex_thinker._fetch_forexfactory_events", side_effect=AssertionError("blocking fetch should not run in scan path")) as fetch_mock,
            ):
                ctx = _load_forexfactory_context(td, settings, now_ts=1_000)
            self.assertEqual(str(ctx.get("state", "")), "unavailable")
            self.assertTrue(bool(ctx.get("refresh_scheduled", False)))
            self.assertEqual(str(ctx.get("refresh_state", "")), "scheduled")
            self.assertEqual(int(spawn_mock.call_count), 1)
            self.assertEqual(int(fetch_mock.call_count), 0)

    def test_stale_cache_is_used_and_refresh_is_scheduled_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            cache_path = os.path.join(fx_dir, "forexfactory_calendar_cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fetched_ts": 300,
                        "events": [
                            {
                                "ts": 1_200,
                                "currency": "USD",
                                "impact": "high",
                                "title": "NFP",
                            }
                        ],
                        "source": "forexfactory",
                        "last_error_ts": 0,
                        "last_error": "",
                    },
                    f,
                )
            settings = {
                "forex_event_risk_enabled": True,
                "forex_event_cache_refresh_s": 600.0,
                "forex_event_cache_stale_max_s": 3600.0,
            }
            with (
                patch("engines.forex_thinker._spawn_forexfactory_refresh", return_value=True) as spawn_mock,
                patch("engines.forex_thinker._fetch_forexfactory_events", side_effect=AssertionError("blocking fetch should not run in scan path")) as fetch_mock,
            ):
                ctx = _load_forexfactory_context(td, settings, now_ts=1_000)
            self.assertEqual(str(ctx.get("state", "")), "cached_stale")
            self.assertEqual(int(len(list(ctx.get("events", []) or []))), 1)
            self.assertTrue(bool(ctx.get("refresh_scheduled", False)))
            self.assertEqual(str(ctx.get("refresh_state", "")), "scheduled")
            self.assertEqual(int(spawn_mock.call_count), 1)
            self.assertEqual(int(fetch_mock.call_count), 0)

    def test_refresh_throttle_prevents_runaway_background_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            cache_path = os.path.join(fx_dir, "forexfactory_calendar_cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fetched_ts": 0,
                        "events": [],
                        "last_error_ts": 0,
                        "last_error": "",
                        "last_refresh_request_ts": 950,
                    },
                    f,
                )
            settings = {
                "forex_event_risk_enabled": True,
                "forex_event_cache_refresh_s": 600.0,
                "forex_event_cache_stale_max_s": 3600.0,
            }
            with patch("engines.forex_thinker._spawn_forexfactory_refresh", return_value=True) as spawn_mock:
                ctx = _load_forexfactory_context(td, settings, now_ts=1_000)
            self.assertEqual(str(ctx.get("state", "")), "unavailable")
            self.assertFalse(bool(ctx.get("refresh_scheduled", False)))
            self.assertEqual(str(ctx.get("refresh_state", "")), "throttled")
            self.assertEqual(int(spawn_mock.call_count), 0)

    def test_recent_error_stays_in_cooldown_without_refresh_spam(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            cache_path = os.path.join(fx_dir, "forexfactory_calendar_cache.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fetched_ts": 0,
                        "events": [],
                        "last_error_ts": 950,
                        "last_error": "URLError: dns",
                    },
                    f,
                )
            settings = {
                "forex_event_risk_enabled": True,
                "forex_event_cache_refresh_s": 600.0,
                "forex_event_cache_stale_max_s": 3600.0,
            }
            with patch("engines.forex_thinker._spawn_forexfactory_refresh", return_value=True) as spawn_mock:
                ctx = _load_forexfactory_context(td, settings, now_ts=1_000)
            self.assertEqual(str(ctx.get("state", "")), "cooldown")
            self.assertFalse(bool(ctx.get("refresh_scheduled", False)))
            self.assertEqual(str(ctx.get("refresh_state", "")), "not_needed")
            self.assertEqual(int(spawn_mock.call_count), 0)

    def test_append_reason_parts_separates_logic_and_data(self) -> None:
        row = {"reason": "Base logic"}
        _append_reason_parts(row, logic="Session dampened", data="session NY x0.85")
        self.assertIn("Base logic", str(row.get("reason_logic", "")))
        self.assertIn("Session dampened", str(row.get("reason_logic", "")))
        self.assertIn("session NY x0.85", str(row.get("reason_data", "")))
        self.assertEqual(str(row.get("reason", "")), str(row.get("reason_logic", "")))


if __name__ == "__main__":
    unittest.main()
