from __future__ import annotations

import sys
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

    class Figure:  # pragma: no cover - shim only
        pass

    class Rectangle:  # pragma: no cover - shim only
        pass

    class FuncFormatter:  # pragma: no cover - shim only
        def __init__(self, func=None) -> None:
            self.func = func

    class FigureCanvasTkAgg:  # pragma: no cover - shim only
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

from ui.pt_hub import _effective_trade_pnl_pct


class TradeHistoryPnlDisplayConsistencyTests(unittest.TestCase):
    def test_sell_row_uses_realized_and_cost_for_effective_trade_pct(self) -> None:
        row = {
            "side": "sell",
            "pnl_pct": 68355.13,
            "realized_profit_usd": 5.0,
            "position_cost_used_usd": 100.0,
            "qty": 1.0,
            "avg_cost_basis": 20.0,
        }
        self.assertAlmostEqual(float(_effective_trade_pnl_pct(row) or 0.0), 5.0, places=6)

    def test_sell_row_falls_back_to_stored_pct_when_cost_unavailable(self) -> None:
        row = {
            "side": "sell",
            "pnl_pct": -2.75,
            "realized_profit_usd": -0.05,
            "position_cost_used_usd": None,
            "qty": 0.0,
            "avg_cost_basis": None,
        }
        self.assertAlmostEqual(float(_effective_trade_pnl_pct(row) or 0.0), -2.75, places=6)

    def test_buy_row_keeps_original_trade_pct(self) -> None:
        row = {
            "side": "buy",
            "tag": "DCA",
            "pnl_pct": -4.2,
            "realized_profit_usd": None,
        }
        self.assertAlmostEqual(float(_effective_trade_pnl_pct(row) or 0.0), -4.2, places=6)


if __name__ == "__main__":
    unittest.main()
