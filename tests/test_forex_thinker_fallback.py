from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from engines import forex_thinker


class _FakeOandaClient:
    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._rest_url = rest_url

    def list_tradeable_instruments(self) -> list[str]:
        return ["EUR_USD", "USD_JPY"]

    def get_pricing_details(self, universe: list[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for pair in universe:
            out[str(pair)] = {"spread_bps": 1.1}
        return out

    def get_candles(self, pair: str, granularity: str = "H4", count: int = 40) -> list[dict]:
        return []


def _mk_oanda_candles(count: int = 48, close: float = 1.1050) -> list[dict]:
    rows: list[dict] = []
    for i in range(max(1, int(count))):
        rows.append(
            {
                "complete": True,
                "time": f"2026-03-{1 + (i // 24):02d}T{(i % 24):02d}:00:00.000000000Z",
                "mid": {
                    "o": f"{close - 0.0010:.5f}",
                    "h": f"{close + 0.0010:.5f}",
                    "l": f"{close - 0.0020:.5f}",
                    "c": f"{close:.5f}",
                },
                "volume": 1000 + i,
            }
        )
    return rows


class _MtfBudgetClient:
    h4_calls: list[str] = []

    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._rest_url = rest_url

    def list_tradeable_instruments(self) -> list[str]:
        return [f"P{i:02d}_USD" for i in range(1, 13)]

    def get_pricing_details(self, universe: list[str]) -> dict[str, dict[str, float]]:
        return {str(pair): {"spread_bps": 1.1} for pair in list(universe or [])}

    def get_candles(self, pair: str, granularity: str = "H4", count: int = 40) -> list[dict]:
        if str(granularity or "").strip().upper() == "H4":
            type(self).h4_calls.append(str(pair).strip().upper())
            return _mk_oanda_candles(count=40, close=1.2050)
        return _mk_oanda_candles(count=max(10, int(count or 48)), close=1.1050)


class TestForexThinkerFallback(unittest.TestCase):
    def test_run_scan_never_calls_blocking_calendar_fetch_directly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 4,
                "forex_event_risk_enabled": True,
                "forex_event_cache_refresh_s": 600.0,
                "forex_event_cache_stale_max_s": 3600.0,
            }
            candles = [
                {
                    "complete": True,
                    "time": f"2026-03-01T{(i % 24):02d}:00:00.000000000Z",
                    "mid": {"o": "1.1000", "h": "1.1100", "l": "1.0900", "c": "1.1050"},
                    "volume": 1000,
                }
                for i in range(48)
            ]
            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(
                    forex_thinker,
                    "_fetch_forexfactory_events",
                    side_effect=AssertionError("blocking fetch should not run in scan loop"),
                ) as fetch_mock,
                patch.object(forex_thinker, "_spawn_forexfactory_refresh", return_value=False),
            ):
                out = forex_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertEqual(int(fetch_mock.call_count), 0)
            event_ctx = out.get("event_context", {}) if isinstance(out.get("event_context", {}), dict) else {}
            self.assertIn(str(event_ctx.get("state", "")), {"unavailable", "cooldown", "cached", "cached_stale"})

    def test_uses_cached_scan_when_network_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            forex_dir = os.path.join(td, "forex")
            os.makedirs(forex_dir, exist_ok=True)
            cached = {
                "state": "READY",
                "ai_state": "Scan ready",
                "msg": "cached baseline",
                "universe": ["EUR_USD", "USD_JPY"],
                "leaders": [{"pair": "EUR_USD", "side": "long", "score": 0.42, "reason": "trend"}],
                "all_scores": [{"pair": "EUR_USD", "side": "long", "score": 0.42, "reason": "trend"}],
                "top_pick": {"pair": "EUR_USD", "side": "long", "score": 0.42, "reason": "trend"},
                "top_chart": [{"t": "t1", "o": 1.1, "h": 1.2, "l": 1.0, "c": 1.15}],
                "top_chart_map": {"EUR_USD": [{"t": "t1", "o": 1.1, "h": 1.2, "l": 1.0, "c": 1.15}]},
                "updated_at": 1000,
                "reject_summary": {"reject_rate_pct": 5.0, "dominant_reason": "spread"},
            }
            with open(os.path.join(forex_dir, "forex_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump(cached, f)

            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 4,
            }

            with (
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", side_effect=urllib.error.URLError("dns down")),
                patch("engines.forex_thinker.time.time", return_value=1300),
            ):
                out = forex_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertTrue(bool(out.get("fallback_cached", False)))
            self.assertIn("cached scan", str(out.get("msg", "")).lower())
            self.assertGreaterEqual(len(list(out.get("leaders", []) or [])), 1)

    def test_no_cache_keeps_error_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 4,
            }
            with (
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", side_effect=urllib.error.URLError("dns down")),
                patch("engines.forex_thinker.time.time", return_value=1300),
            ):
                out = forex_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "ERROR")
            self.assertEqual(str(out.get("ai_state", "")), "Network error")

    def test_applies_leader_hysteresis_to_previous_top_pair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            with open(os.path.join(fx_dir, "forex_thinker_status.json"), "w", encoding="utf-8") as f:
                json.dump({"top_pick": {"pair": "USD_JPY", "side": "long", "score": 0.58}}, f)

            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 4,
                "forex_leader_stability_margin_pct": 10.0,
                "forex_max_stale_hours": 10000.0,
            }

            candles = [
                {"complete": True, "time": f"2026-03-01T{(i % 24):02d}:00:00.000000000Z", "mid": {"o": "1.1000", "h": "1.1100", "l": "1.0900", "c": "1.1050"}, "volume": 1000}
                for i in range(48)
            ]

            def _score(pair: str, rows: list[dict], spread_bps: float = 0.0) -> dict:
                base = 0.58 if str(pair).upper() == "USD_JPY" else 0.62
                return {
                    "pair": str(pair).upper(),
                    "score": float(base),
                    "side": "long",
                    "last": 1.2345,
                    "change_6h_pct": 0.2,
                    "change_24h_pct": 0.4,
                    "volatility_pct": 0.05,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(forex_thinker, "_score_candles", side_effect=_score),
            ):
                out = forex_thinker.run_scan(settings, td)
            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("pair", "")), "USD_JPY")
            self.assertTrue(bool(out.get("leader_stability_applied", False)))

    def test_live_guarded_demotes_undertrained_leader_to_watch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "market_rollout_stage": "live_guarded",
                "forex_min_samples_live_guarded": 4,
                "forex_min_calib_prob_live_guarded": 0.48,
                "forex_scan_max_pairs": 4,
                "forex_max_stale_hours": 10000.0,
            }
            candles = [
                {"complete": True, "time": f"2026-03-01T{(i % 24):02d}:00:00.000000000Z", "mid": {"o": "1.1000", "h": "1.1100", "l": "1.0900", "c": "1.1050"}, "volume": 1000}
                for i in range(48)
            ]

            def _score(pair: str, rows: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "pair": str(pair).upper(),
                    "score": 0.62,
                    "side": "long",
                    "last": 1.2345,
                    "change_6h_pct": 0.2,
                    "change_24h_pct": 0.4,
                    "volatility_pct": 0.05,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(forex_thinker, "_score_candles", side_effect=_score),
                patch.object(forex_thinker, "_load_forexfactory_context", return_value={"enabled": False, "events": [], "state": "disabled", "error": ""}),
            ):
                out = forex_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "watch")
            self.assertFalse(bool(top.get("eligible_for_entry", True)))
            self.assertIn("Calibration sample gate", str(top.get("entry_gate_reason", "") or ""))

    def test_live_guarded_uses_market_pooled_calibration_when_pair_history_is_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "market_rollout_stage": "live_guarded",
                "forex_min_samples_live_guarded": 4,
                "forex_min_calib_prob_live_guarded": 0.48,
                "forex_scan_max_pairs": 4,
                "forex_max_stale_hours": 10000.0,
            }
            candles = [
                {"complete": True, "time": f"2026-03-01T{(i % 24):02d}:00:00.000000000Z", "mid": {"o": "1.1000", "h": "1.1100", "l": "1.0900", "c": "1.1050"}, "volume": 1000}
                for i in range(48)
            ]

            def _score(pair: str, rows: list[dict], spread_bps: float = 0.0) -> dict:
                return {
                    "pair": str(pair).upper(),
                    "score": 0.62,
                    "side": "long",
                    "last": 1.2345,
                    "change_6h_pct": 0.2,
                    "change_24h_pct": 0.4,
                    "volatility_pct": 0.05,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            audit_path = os.path.join(fx_dir, "execution_audit.jsonl")
            with open(audit_path, "w", encoding="utf-8") as f:
                for idx in range(5):
                    f.write(json.dumps({"ts": 1_700_000_000 + idx, "event": "shadow_live_divergence", "instrument": f"PAIR_{idx}", "score": 0.62}) + "\n")

            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _FakeOandaClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(forex_thinker, "_score_candles", side_effect=_score),
                patch.object(forex_thinker, "_load_forexfactory_context", return_value={"enabled": False, "events": [], "state": "disabled", "error": ""}),
            ):
                out = forex_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            top = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
            self.assertEqual(str(top.get("side", "")).lower(), "long")
            self.assertTrue(bool(top.get("eligible_for_entry", False)))
            self.assertEqual(str(top.get("calibration_scope", "") or ""), "market_pooled")
            self.assertGreaterEqual(int(top.get("samples", 0) or 0), 5)
            self.assertEqual(str(top.get("entry_gate_reason", "") or ""), "")

    def test_h4_confirmation_is_budgeted_to_top_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            _MtfBudgetClient.h4_calls = []
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 12,
                "forex_mtf_confirm_max_pairs": 3,
                "forex_event_risk_enabled": False,
                "forex_session_weight_enabled": False,
                "forex_min_volatility_pct": 0.0,
                "forex_max_stale_hours": 10000.0,
            }
            candles = _mk_oanda_candles(count=48, close=1.1050)

            def _score(pair: str, rows: list[dict], spread_bps: float = 0.0) -> dict:
                p = str(pair).strip().upper()
                idx = int(p.split("_")[0].replace("P", "") or 0)
                if int(len(rows or [])) <= 40:
                    score = 0.55 if idx <= 3 else -0.55
                else:
                    score = 1.20 - (idx * 0.02)
                side = "long" if score > 0 else "short"
                return {
                    "pair": p,
                    "score": float(score),
                    "side": side,
                    "last": 1.2345,
                    "change_6h_pct": 0.15,
                    "change_24h_pct": 0.32,
                    "volatility_pct": 0.06,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _MtfBudgetClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(forex_thinker, "_score_candles", side_effect=_score),
                patch.object(forex_thinker, "_load_forexfactory_context", return_value={"enabled": False, "events": [], "state": "disabled", "error": ""}),
            ):
                out = forex_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            self.assertLessEqual(len(set(_MtfBudgetClient.h4_calls)), 3)
            mtf_payload = out.get("mtf_confirmation", {}) if isinstance(out.get("mtf_confirmation", {}), dict) else {}
            self.assertEqual(int(mtf_payload.get("limit", -1) or -1), 3)
            self.assertLessEqual(int(mtf_payload.get("selected_pairs", 0) or 0), 3)
            self.assertGreaterEqual(int(mtf_payload.get("deferred_pairs", 0) or 0), 1)

            scored = list(out.get("all_scores", []) or [])
            self.assertGreaterEqual(len(scored), 6)
            deferred_rows = [row for row in scored if str((row or {}).get("mtf_source", "") or "") == "deferred"]
            self.assertGreaterEqual(len(deferred_rows), 1)
            self.assertTrue(all((row.get("mtf_confirmed") is None) for row in deferred_rows))

            timings = out.get("scan_phase_timing_ms", {}) if isinstance(out.get("scan_phase_timing_ms", {}), dict) else {}
            self.assertIn("pricing_fetch", timings)
            self.assertIn("h1_scoring_loop", timings)
            self.assertIn("h4_confirmation_loop", timings)

            with open(os.path.join(td, "forex", "scan_diagnostics.json"), "r", encoding="utf-8") as f:
                diag = json.load(f)
            self.assertIn("scan_phase_timing_ms", diag)
            self.assertIn("mtf_confirmation", diag)

    def test_deferred_rows_skip_h4_penalty_while_selected_rows_apply_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "forex"), exist_ok=True)
            _MtfBudgetClient.h4_calls = []
            settings = {
                "oanda_account_id": "abc",
                "oanda_api_token": "xyz",
                "oanda_rest_url": "https://api-fxpractice.oanda.com",
                "forex_scan_max_pairs": 8,
                "forex_mtf_confirm_max_pairs": 2,
                "forex_event_risk_enabled": False,
                "forex_session_weight_enabled": False,
                "forex_min_volatility_pct": 0.0,
                "forex_max_stale_hours": 10000.0,
            }
            candles = _mk_oanda_candles(count=48, close=1.1050)

            def _score(pair: str, rows: list[dict], spread_bps: float = 0.0) -> dict:
                p = str(pair).strip().upper()
                if int(len(rows or [])) <= 40:
                    score = -0.8  # Force mismatch against H1 long on selected rows.
                else:
                    score = 1.0
                side = "long" if score > 0 else "short"
                return {
                    "pair": p,
                    "score": float(score),
                    "side": side,
                    "last": 1.2345,
                    "change_6h_pct": 0.2,
                    "change_24h_pct": 0.4,
                    "volatility_pct": 0.08,
                    "spread_bps": float(spread_bps),
                    "confidence": "MED",
                    "reason": "test",
                }

            with (
                patch.object(forex_thinker, "get_oanda_creds", return_value=("abc", "xyz")),
                patch.object(forex_thinker, "OandaBrokerClient", _MtfBudgetClient),
                patch.object(forex_thinker, "_request_json", return_value={"candles": candles}),
                patch.object(forex_thinker, "_score_candles", side_effect=_score),
                patch.object(forex_thinker, "_load_forexfactory_context", return_value={"enabled": False, "events": [], "state": "disabled", "error": ""}),
            ):
                out = forex_thinker.run_scan(settings, td)

            self.assertEqual(str(out.get("state", "")), "READY")
            scored = list(out.get("all_scores", []) or [])
            h4_rows = [row for row in scored if str((row or {}).get("mtf_source", "") or "") == "h4_remote"]
            deferred_rows = [row for row in scored if str((row or {}).get("mtf_source", "") or "") == "deferred"]
            self.assertGreaterEqual(len(h4_rows), 1)
            self.assertGreaterEqual(len(deferred_rows), 1)
            self.assertTrue(all(abs(float(row.get("score", 0.0) or 0.0) - 0.75) <= 1e-6 for row in h4_rows))
            self.assertTrue(all(abs(float(row.get("score", 0.0) or 0.0) - 1.0) <= 1e-6 for row in deferred_rows))


if __name__ == "__main__":
    unittest.main()
