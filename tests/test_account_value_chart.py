from __future__ import annotations

import importlib
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


def _load_pt_hub_module():
    _install_matplotlib_stubs()
    return importlib.import_module("ui.pt_hub")


class _Axis:
    class _YAxis:
        def __init__(self) -> None:
            self.formatter = None

        def set_major_formatter(self, formatter) -> None:
            self.formatter = formatter

    def __init__(self) -> None:
        self.lines = []
        self.patches = []
        self.collections = []
        self.texts = []
        self.annotations = []
        self.title = ""
        self.xticks = []
        self.xticklabels = []
        self.xlim = None
        self.yaxis = self._YAxis()

    def cla(self) -> None:
        self.lines.clear()
        self.patches.clear()
        self.collections.clear()
        self.texts.clear()
        self.annotations.clear()

    def plot(self, xs, ys, linewidth=1.5):
        self.lines.append((list(xs), list(ys), float(linewidth)))

    def scatter(self, xs, ys, **kwargs) -> None:
        self.collections.append((list(xs), list(ys), dict(kwargs)))

    def annotate(self, label, xy, **kwargs) -> None:
        self.annotations.append((str(label), tuple(xy), dict(kwargs)))

    def minorticks_off(self) -> None:
        pass

    def set_xticks(self, ticks) -> None:
        self.xticks = list(ticks)

    def set_xticklabels(self, labels) -> None:
        self.xticklabels = list(labels)

    def tick_params(self, **kwargs) -> None:
        pass

    def set_xlim(self, left, right) -> None:
        self.xlim = (float(left), float(right))

    def set_title(self, title, color=None) -> None:
        self.title = str(title)

    def text(self, *args, **kwargs) -> None:
        self.texts.append((args, kwargs))


class _Canvas:
    def __init__(self) -> None:
        self.draw_idle_calls = 0

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1


class _Label:
    def __init__(self) -> None:
        self.kwargs = {}

    def config(self, **kwargs) -> None:
        self.kwargs.update(kwargs)


class AccountValueChartTests(unittest.TestCase):
    def test_refresh_keeps_trade_annotations_for_all_coins_chart(self) -> None:
        pt_hub = _load_pt_hub_module()
        chart_cls = pt_hub.AccountValueChart

        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "account_value_history.jsonl")
            trade_history_path = os.path.join(td, "trade_history.jsonl")
            with open(history_path, "w", encoding="utf-8") as f:
                for ts, value in ((100, 1000.0), (200, 1005.0), (300, 1010.0)):
                    f.write(json.dumps({"ts": ts, "total_account_value": value}) + "\n")
            with open(trade_history_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": 200, "side": "buy", "symbol": "BTC-USD"}) + "\n")

            chart = types.SimpleNamespace(
                history_path=history_path,
                trade_history_path=trade_history_path,
                max_points=250,
                _last_mtime=None,
                ax=_Axis(),
                canvas=_Canvas(),
                last_update_label=_Label(),
                _apply_dark_chart_style=lambda: None,
            )

            chart_cls.refresh(chart)

            self.assertEqual(len(chart.ax.collections), 1)
            self.assertEqual(len(chart.ax.annotations), 1)
            self.assertEqual(chart.ax.annotations[0][0], "BTC BUY")
            self.assertGreater(chart.canvas.draw_idle_calls, 0)

    def test_market_history_trims_legacy_regime_after_large_value_jump(self) -> None:
        pt_hub = _load_pt_hub_module()
        hub_cls = pt_hub.PowerTraderHub

        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "stocks_account_value_history.jsonl")
            rows = [
                {"ts": 100, "total_account_value": 99996.74},
                {"ts": 200, "total_account_value": 100007.29},
                {"ts": 300, "total_account_value": 100.12},
                {"ts": 400, "total_account_value": 100.53},
            ]
            with open(history_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            hub = types.SimpleNamespace(
                settings={"alpaca_base_url": "https://api.alpaca.markets"},
                market_account_history_paths={"stocks": history_path},
                market_state_dirs={},
                hub_dir=td,
                project_dir=td,
            )
            hub._coerce_float_value = hub_cls._coerce_float_value.__get__(hub, hub_cls)
            hub._market_account_history_path = hub_cls._market_account_history_path.__get__(hub, hub_cls)
            hub._normalize_market_broker_mode = hub_cls._normalize_market_broker_mode.__get__(hub, hub_cls)
            hub._market_account_history_context = hub_cls._market_account_history_context.__get__(hub, hub_cls)
            hub._market_account_value_from_snapshot = hub_cls._market_account_value_from_snapshot.__get__(hub, hub_cls)

            points = hub_cls._read_market_account_history(
                hub,
                "stocks",
                status_data={"equity": 100.53, "ts": 450},
                trader_data={"account_value_usd": 100.53, "updated_at": 450, "broker_mode": "live"},
                max_points=250,
            )
            self.assertTrue(points)
            self.assertTrue(all(float(v) < 1000.0 for _, v in points))

    def test_market_history_filters_explicit_broker_mode_mismatch(self) -> None:
        pt_hub = _load_pt_hub_module()
        hub_cls = pt_hub.PowerTraderHub

        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "stocks_account_value_history.jsonl")
            rows = [
                {"ts": 100, "total_account_value": 99999.0, "broker_mode": "paper", "broker_endpoint": "https://paper-api.alpaca.markets"},
                {"ts": 200, "total_account_value": 100.10, "broker_mode": "live", "broker_endpoint": "https://api.alpaca.markets"},
                {"ts": 300, "total_account_value": 99990.0, "broker_mode": "paper", "broker_endpoint": "https://paper-api.alpaca.markets"},
                {"ts": 400, "total_account_value": 100.40, "broker_mode": "live", "broker_endpoint": "https://api.alpaca.markets"},
            ]
            with open(history_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            hub = types.SimpleNamespace(
                settings={"alpaca_base_url": "https://api.alpaca.markets"},
                market_account_history_paths={"stocks": history_path},
                market_state_dirs={},
                hub_dir=td,
                project_dir=td,
            )
            hub._coerce_float_value = hub_cls._coerce_float_value.__get__(hub, hub_cls)
            hub._market_account_history_path = hub_cls._market_account_history_path.__get__(hub, hub_cls)
            hub._normalize_market_broker_mode = hub_cls._normalize_market_broker_mode.__get__(hub, hub_cls)
            hub._market_account_history_context = hub_cls._market_account_history_context.__get__(hub, hub_cls)
            hub._market_account_value_from_snapshot = hub_cls._market_account_value_from_snapshot.__get__(hub, hub_cls)

            points = hub_cls._read_market_account_history(
                hub,
                "stocks",
                status_data={"equity": 100.50, "ts": 450},
                trader_data={"account_value_usd": 100.50, "updated_at": 450, "broker_mode": "live"},
                max_points=250,
            )
            self.assertTrue(points)
            self.assertTrue(all(float(v) < 1000.0 for _, v in points))


if __name__ == "__main__":
    unittest.main()
