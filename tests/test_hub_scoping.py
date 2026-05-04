from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest


def _install_matplotlib_stubs() -> None:
    if "matplotlib.figure" in sys.modules:
        return

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    figure_mod = types.ModuleType("matplotlib.figure")
    patches_mod = types.ModuleType("matplotlib.patches")
    ticker_mod = types.ModuleType("matplotlib.ticker")
    transforms_mod = types.ModuleType("matplotlib.transforms")
    backends_mod = types.ModuleType("matplotlib.backends")
    backends_mod.__path__ = []
    backend_tkagg_mod = types.ModuleType("matplotlib.backends.backend_tkagg")

    class Figure:  # pragma: no cover - import shim only
        pass

    class Rectangle:  # pragma: no cover - import shim only
        pass

    class FuncFormatter:  # pragma: no cover - import shim only
        def __init__(self, func=None) -> None:
            self.func = func

    class FigureCanvasTkAgg:  # pragma: no cover - import shim only
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    def blended_transform_factory(*args, **kwargs):
        return None

    figure_mod.Figure = Figure
    patches_mod.Rectangle = Rectangle
    ticker_mod.FuncFormatter = FuncFormatter
    transforms_mod.blended_transform_factory = blended_transform_factory
    backend_tkagg_mod.FigureCanvasTkAgg = FigureCanvasTkAgg

    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.figure"] = figure_mod
    sys.modules["matplotlib.patches"] = patches_mod
    sys.modules["matplotlib.ticker"] = ticker_mod
    sys.modules["matplotlib.transforms"] = transforms_mod
    sys.modules["matplotlib.backends"] = backends_mod
    sys.modules["matplotlib.backends.backend_tkagg"] = backend_tkagg_mod


_install_matplotlib_stubs()

from ui.pt_hub import PowerTraderHub, _resolve_rollout_stage_for_broker_modes


class HubScopingTests(unittest.TestCase):
    def _make_hub(self) -> PowerTraderHub:
        hub = PowerTraderHub.__new__(PowerTraderHub)
        hub.coins = []
        hub.trainers = {}
        hub.coin_folders = {}
        hub.settings = {}
        hub.project_dir = ""
        hub.crypto_dynamic_status_path = ""
        hub._active_market_key = lambda: "stocks"
        return hub

    def test_markets_for_global_alert_uses_top_exposure_market(self) -> None:
        hub = self._make_hub()
        markets = hub._markets_for_global_alert(
            "exposure_concentration",
            {
                "exposure_map": {
                    "top_positions": [
                        {"market": "stocks", "symbol": "AMZN", "pct_of_total_exposure": 86.0},
                    ]
                }
            },
        )
        self.assertEqual(markets, ["stocks"])

    def test_scoped_notification_items_filter_global_runtime_alerts_by_market(self) -> None:
        hub = self._make_hub()
        runtime_snapshot = {
            "scan_cadence": {
                "active": [
                    {"market": "stocks", "level": "critical"},
                ]
            },
            "shadow_scorecards": {
                "stocks": {"promotion_gate": "PASS"},
                "forex": {"promotion_gate": "BLOCK"},
            },
            "notification_center": {
                "items": [
                    {
                        "market": "global",
                        "source": "runtime_alerts",
                        "title": "cadence_drift_pressure",
                        "severity": "critical",
                    },
                    {
                        "market": "global",
                        "source": "runtime_alerts",
                        "title": "shadow_scorecard_blocked",
                        "severity": "critical",
                    },
                    {
                        "market": "stocks",
                        "source": "incidents",
                        "title": "scanner_cadence_drift",
                        "severity": "critical",
                    },
                    {
                        "market": "crypto",
                        "source": "incidents",
                        "title": "kucoin_throttle",
                        "severity": "warning",
                    },
                ]
            },
        }

        stocks_items = hub._scoped_notification_items(runtime_snapshot, "stocks")
        stocks_titles = [str(row.get("title", "") or "") for row in stocks_items]
        self.assertIn("cadence_drift_pressure", stocks_titles)
        self.assertIn("scanner_cadence_drift", stocks_titles)
        self.assertNotIn("shadow_scorecard_blocked", stocks_titles)
        self.assertNotIn("kucoin_throttle", stocks_titles)

    def test_filtered_notification_items_use_active_market_for_current_tab(self) -> None:
        hub = self._make_hub()
        runtime_snapshot = {
            "scan_cadence": {
                "active": [
                    {"market": "stocks", "level": "critical"},
                ]
            },
            "notification_center": {
                "items": [
                    {
                        "market": "global",
                        "source": "runtime_alerts",
                        "title": "cadence_drift_pressure",
                        "severity": "critical",
                    },
                    {
                        "market": "forex",
                        "source": "incidents",
                        "title": "scanner_cadence_drift",
                        "severity": "critical",
                    },
                    {
                        "market": "stocks",
                        "source": "incidents",
                        "title": "scanner_cadence_drift",
                        "severity": "critical",
                    },
                ]
            },
        }
        items = hub._filtered_notification_items(runtime_snapshot, runtime_snapshot.get("notification_center"), "Current Tab", "critical")
        titles = [str(row.get("title", "") or "") for row in items]
        markets = [str(row.get("market", "") or "") for row in items]
        self.assertEqual(titles, ["cadence_drift_pressure", "scanner_cadence_drift"])
        self.assertEqual(markets, ["global", "stocks"])

    def test_resolve_notification_market_filter_defaults_blank_to_all(self) -> None:
        hub = self._make_hub()
        self.assertEqual(hub._resolve_notification_market_filter(""), "all")
        self.assertEqual(hub._resolve_notification_market_filter(None), "all")
        self.assertEqual(hub._resolve_notification_market_filter("Current Tab"), "stocks")

    def test_notification_empty_state_text_includes_scoped_summary(self) -> None:
        hub = self._make_hub()
        runtime_snapshot = {
            "scan_cadence": {
                "active": [
                    {"market": "stocks", "level": "critical"},
                ]
            },
            "notification_center": {
                "items": [
                    {
                        "market": "global",
                        "source": "runtime_alerts",
                        "title": "cadence_drift_pressure",
                        "severity": "critical",
                        "message": "Scanner cadence drift active.",
                    }
                ]
            },
        }
        text = hub._notification_empty_state_text(runtime_snapshot, "stocks", "warning")
        self.assertIn("Stocks alerts:", text)
        self.assertIn("Severity filter: WARNING", text)
        self.assertIn("cadence_drift_pressure", text)

    def test_notification_payload_falls_back_to_runtime_state_without_runtime_state_attr(self) -> None:
        hub = self._make_hub()
        with tempfile.TemporaryDirectory() as tmp:
            hub.hub_dir = tmp
            runtime_state_path = os.path.join(tmp, "runtime_state.json")
            with open(runtime_state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 1_700_000_100,
                        "alerts": {
                            "severity": "critical",
                            "reasons": ["exposure_concentration"],
                            "hints": ["Exposure concentration is high."],
                        },
                    },
                    f,
                )
            payload = hub._notification_payload()

        self.assertTrue(isinstance(payload, dict))
        self.assertTrue(isinstance(payload.get("items", []), list))

    def test_notification_payload_prefers_rebuilt_live_snapshot_over_stale_file(self) -> None:
        hub = self._make_hub()
        with tempfile.TemporaryDirectory() as tmp:
            hub.hub_dir = tmp
            with open(os.path.join(tmp, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 1_700_000_000,
                        "checks": {"ok": True, "warnings": []},
                        "alerts": {
                            "severity": "critical",
                            "reasons": ["exposure_concentration"],
                            "hints": ["Exposure concentration is high."],
                        },
                        "exposure_map": {
                            "top_positions": [
                                {
                                    "market": "stocks",
                                    "pct_of_total_exposure": 86.0,
                                    "pct_of_market_account": 12.0,
                                }
                            ]
                        },
                        "market_trends": {},
                    },
                    f,
                )
            with open(os.path.join(tmp, "notification_center.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 1_699_999_000,
                        "total": 1,
                        "items": [{"title": "stale_alert", "severity": "critical", "market": "global"}],
                    },
                    f,
                )
            payload = hub._notification_payload()

        self.assertEqual(str((payload.get("items", [{}])[0] or {}).get("title", "")), "exposure_concentration")

    def test_scoped_notification_items_hide_shadow_scorecard_after_live_rollout(self) -> None:
        hub = self._make_hub()
        hub.settings = {"market_rollout_stage": "live_guarded"}
        runtime_snapshot = {
            "shadow_scorecards": {
                "stocks": {"promotion_gate": "BLOCK"},
                "forex": {"promotion_gate": "BLOCK"},
            },
            "notification_center": {
                "items": [
                    {
                        "market": "global",
                        "source": "runtime_alerts",
                        "title": "shadow_scorecard_blocked",
                        "severity": "critical",
                    }
                ]
            },
        }
        items = hub._scoped_notification_items(runtime_snapshot, "forex")
        titles = [str(row.get("title", "") or "") for row in items]
        self.assertNotIn("shadow_scorecard_blocked", titles)

    def test_notification_payload_recomputes_live_guarded_alerts_from_runtime_snapshot(self) -> None:
        hub = self._make_hub()
        hub.settings = {"market_rollout_stage": "live_guarded"}
        with tempfile.TemporaryDirectory() as tmp:
            hub.hub_dir = tmp
            with open(os.path.join(tmp, "runtime_state.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": 1_700_000_000,
                        "alerts": {
                            "severity": "critical",
                            "reasons": ["shadow_scorecard_blocked"],
                            "hints": ["stale"],
                        },
                        "scan_cadence": {"active": []},
                        "shadow_scorecards": {
                            "stocks": {"promotion_gate": "BLOCK"},
                            "forex": {"promotion_gate": "BLOCK"},
                        },
                        "market_trends": {
                            "forex": {
                                "why_not_traded": {"reason": "Calibration sample gate x14"},
                                "quality_aggregates": {"reject_rate_pct": 8.0},
                                "data_source_reliability": {"score": 96.0},
                            }
                        },
                        "notification_center": {
                            "items": [
                                {"market": "global", "source": "runtime_alerts", "title": "shadow_scorecard_blocked", "severity": "critical"},
                                {"market": "forex", "source": "incidents", "title": "scanner_cadence_drift", "severity": "critical"},
                            ]
                        },
                    },
                    f,
                )
            with open(os.path.join(tmp, "incidents.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": 1_699_999_999, "severity": "critical", "event": "scanner_cadence_drift", "msg": "late", "details": {"market": "forex"}}) + "\n")
            payload = hub._notification_payload()

        titles = [str((row or {}).get("title", "") or "") for row in list(payload.get("items", []) or [])]
        self.assertIn("Why top candidate was not traded", titles)
        self.assertNotIn("shadow_scorecard_blocked", titles)
        self.assertNotIn("scanner_cadence_drift", titles)

    def test_crypto_training_candidate_symbols_merges_dynamic_and_disk_symbols(self) -> None:
        hub = self._make_hub()
        with tempfile.TemporaryDirectory() as tmp:
            dynamic_path = os.path.join(tmp, "crypto_dynamic_status.json")
            with open(dynamic_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "current_coins": ["SOL"],
                        "ranked": [{"symbol": "LINK"}, {"symbol": "ETH"}],
                    },
                    f,
                )

            avax_dir = os.path.join(tmp, "AVAX")
            os.makedirs(avax_dir, exist_ok=True)
            with open(os.path.join(avax_dir, "trainer_status.json"), "w", encoding="utf-8") as f:
                json.dump({"state": "TRAINING"}, f)

            hub.coins = ["BTC", "ETH"]
            hub.trainers = {"DOT": object()}
            hub.settings = {"main_neural_dir": tmp}
            hub.project_dir = tmp
            hub.crypto_dynamic_status_path = dynamic_path

            symbols = hub._crypto_training_candidate_symbols()

        self.assertEqual(symbols[:5], ["BTC", "ETH", "SOL", "LINK", "DOT"])
        self.assertIn("AVAX", symbols)

    def test_live_forex_promotes_shadow_only_to_live(self) -> None:
        stage, note = _resolve_rollout_stage_for_broker_modes("shadow_only", True, False)
        self.assertEqual(stage, "live")
        self.assertIn("live", note)

    def test_live_stock_promotes_non_executable_stage(self) -> None:
        stage, note = _resolve_rollout_stage_for_broker_modes("risk_caps", False, True)
        self.assertEqual(stage, "live")
        self.assertIn("Live broker mode", note)

    def test_execution_stage_is_preserved_for_live_broker(self) -> None:
        stage, note = _resolve_rollout_stage_for_broker_modes("execution_v2", True, False)
        self.assertEqual(stage, "execution_v2")
        self.assertEqual(note, "")

    def test_paper_modes_do_not_change_stage(self) -> None:
        stage, note = _resolve_rollout_stage_for_broker_modes("shadow_only", True, True)
        self.assertEqual(stage, "shadow_only")
        self.assertEqual(note, "")


if __name__ == "__main__":
    unittest.main()
