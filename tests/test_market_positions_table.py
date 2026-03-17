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

from ui.pt_hub import CandleChart, PowerTraderHub


class _Var:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _Tree:
    def __init__(self) -> None:
        self.rows = {}

    def get_children(self):
        return tuple(self.rows.keys())

    def delete(self, iid) -> None:
        self.rows.pop(iid, None)

    def insert(self, _parent, _where, values=(), tags=()):
        iid = f"row{len(self.rows) + 1}"
        self.rows[iid] = {"values": tuple(values), "tags": tuple(tags)}
        return iid


class _Canvas:
    def __init__(self, width: int = 320, height: int = 120) -> None:
        self.width = width
        self.height = height
        self.items = []
        self.config = {}
        self.root_x = 0
        self.root_y = 0
        self.pointer_x = 0
        self.pointer_y = 0

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height

    def winfo_rootx(self) -> int:
        return self.root_x

    def winfo_rooty(self) -> int:
        return self.root_y

    def winfo_pointerx(self) -> int:
        return self.pointer_x

    def winfo_pointery(self) -> int:
        return self.pointer_y

    def canvasx(self, value) -> float:
        return float(value)

    def canvasy(self, value) -> float:
        return float(value)

    def delete(self, *_args) -> None:
        self.items = []

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)

    def bind(self, *_args, **_kwargs) -> None:
        return None

    def xview(self, *_args, **_kwargs) -> None:
        return None

    def yview(self, *_args, **_kwargs) -> None:
        return None

    def create_rectangle(self, *args, **kwargs):
        self.items.append(("rectangle", args, kwargs))
        return len(self.items)

    def create_text(self, *args, **kwargs):
        self.items.append(("text", args, kwargs))
        return len(self.items)

    def create_line(self, *args, **kwargs):
        self.items.append(("line", args, kwargs))
        return len(self.items)

    def create_oval(self, *args, **kwargs):
        self.items.append(("oval", args, kwargs))
        return len(self.items)

    def create_arc(self, *args, **kwargs):
        self.items.append(("arc", args, kwargs))
        return len(self.items)


class _Font:
    def cget(self, key: str):
        if key == "family":
            return "TkDefaultFont"
        if key == "size":
            return 9
        return ""


class _FigureCanvas:
    def __init__(self, widget) -> None:
        self.widget = widget

    def get_tk_widget(self):
        return self.widget


class _Listbox:
    def __init__(self) -> None:
        self.rows = []
        self.item_styles = {}
        self.y_pos = None

    def delete(self, *_args) -> None:
        self.rows = []
        self.item_styles = {}

    def insert(self, _where, text) -> None:
        self.rows.append(str(text))

    def size(self) -> int:
        return len(self.rows)

    def itemconfig(self, idx, **kwargs) -> None:
        self.item_styles[int(idx)] = dict(kwargs)

    def yview_moveto(self, value) -> None:
        self.y_pos = float(value)


class _Button:
    def __init__(self) -> None:
        self.config = {}

    def configure(self, **kwargs) -> None:
        self.config.update(kwargs)


class MarketPositionsTableTests(unittest.TestCase):
    def _hub(self) -> PowerTraderHub:
        return PowerTraderHub.__new__(PowerTraderHub)

    def test_stock_market_position_rows_include_current_trade_details(self) -> None:
        hub = self._hub()
        hub._read_market_thinker_status = lambda market_key: {}
        payload = PowerTraderHub._market_position_rows(
            hub,
            "stocks",
            raw_positions=[
                {
                    "symbol": "AMZN",
                    "qty": "0.230557118",
                    "qty_available": "0.230557118",
                    "avg_entry_price": "216.866",
                    "side": "long",
                    "market_value": "48.388175",
                    "unrealized_pl": "-1.611825",
                    "unrealized_plpc": "-0.03224",
                    "current_price": "209.875",
                    "change_today": "-0.01305",
                }
            ],
        )

        row = payload["rows"][0]
        self.assertEqual(payload["schema"]["columns"][0], "symbol")
        self.assertEqual(row["symbol"], "AMZN")
        self.assertEqual(row["side"], "LONG")
        self.assertEqual(row["qty"], "0.230557")
        self.assertEqual(row["value"], "$48.39")
        self.assertEqual(row["unrealized_usd"], "-1.61")
        self.assertEqual(row["realized_usd"], "N/A")
        self.assertEqual(row["avg_cost"], "$216.866")
        self.assertEqual(row["ask_price"], "$209.875")
        self.assertEqual(row["day_pct"], "-1.31%")
        self.assertEqual(row["qty_available"], "0.230557")
        self.assertIn("Open trades: 1", payload["summary"])

    def test_forex_market_position_rows_include_current_trade_details(self) -> None:
        hub = self._hub()
        hub._read_market_thinker_status = lambda market_key: {"leaders": [{"pair": "AUD_HKD", "last": 5.54851}]}
        payload = PowerTraderHub._market_position_rows(
            hub,
            "forex",
            raw_positions=[
                {
                    "instrument": "AUD_HKD",
                    "marginUsed": "0.1417",
                    "financing": "0.0000",
                    "unrealizedPL": "0.0005",
                    "long": {
                        "units": "0",
                        "unrealizedPL": "0.0000",
                    },
                    "short": {
                        "units": "-2",
                        "averagePrice": "5.54943",
                        "financing": "0.0000",
                        "tradeIDs": ["7"],
                        "unrealizedPL": "0.0005",
                    },
                }
            ],
            status_data={"currency": "USD"},
            trader_data={"position_values_usd": {"AUD_HKD": 1.4199}},
        )

        row = payload["rows"][0]
        self.assertEqual(payload["schema"]["columns"][0], "pair")
        self.assertEqual(row["pair"], "AUD_HKD")
        self.assertEqual(row["side"], "SHORT")
        self.assertEqual(row["units"], "2")
        self.assertEqual(row["value"], "$0.14")
        self.assertEqual(row["notional_usd"], "$1.42")
        self.assertEqual(row["unrealized_usd"], "+0.0005 USD")
        self.assertEqual(row["realized_usd"], "+0.0000 USD")
        self.assertEqual(row["avg_cost"], "$5.5494")
        self.assertEqual(row["ask_price"], "$5.5485")
        self.assertEqual(row["margin"], "0.1417 USD")
        self.assertEqual(row["financing"], "+0.0000 USD")
        self.assertEqual(row["trades"], "1")
        self.assertIn("Value $0.14", payload["summary"])
        self.assertIn("Notional $1.42", payload["summary"])

    def test_forex_portfolio_snapshot_uses_margin_math(self) -> None:
        hub = self._hub()
        snapshot = PowerTraderHub._market_portfolio_snapshot(
            hub,
            "forex",
            status_data={
                "currency": "USD",
                "nav": 100.0093,
                "margin_available": 97.4183,
                "buying_power": "97.4183 USD",
                "open_positions": 4,
                "realized_pnl": "0.0000 USD",
                "raw_positions": [
                    {"instrument": "AUD_CAD", "marginUsed": "0.2973"},
                    {"instrument": "AUD_USD", "marginUsed": "0.0212"},
                    {"instrument": "GBP_USD", "marginUsed": "1.1029"},
                    {"instrument": "AUD_HKD", "marginUsed": "1.1740"},
                ],
            },
            trader_data={
                "account_value_usd": 100.0093,
                "exposure_usd": 54.7478,
                "open_positions": 4,
            },
        )

        self.assertEqual(snapshot["total_account_value"], "$100.01")
        self.assertEqual(snapshot["holdings_value"], "$2.60")
        self.assertEqual(snapshot["buying_power"], "$97.42")
        self.assertEqual(snapshot["percent_in_trade"], "2.60% (notional 54.74%)")
        self.assertEqual(snapshot["open_positions"], "4")
        self.assertEqual(snapshot["realized_pnl"], "+0.00")

    def test_stock_portfolio_snapshot_prefers_cash_over_margin_buying_power(self) -> None:
        hub = self._hub()
        snapshot = PowerTraderHub._market_portfolio_snapshot(
            hub,
            "stocks",
            status_data={
                "equity": 99997.33,
                "cash": 99750.00,
                "buying_power": 199747.33,
                "open_positions": 2,
                "raw_positions": [
                    {"symbol": "AMZN", "market_value": "48.277392"},
                    {"symbol": "SHOP", "market_value": "199.051117"},
                ],
                "realized_pnl": "N/A",
            },
            trader_data={
                "account_value_usd": 99997.33,
                "exposure_usd": 247.3285,
                "open_positions": 2,
            },
        )

        self.assertEqual(snapshot["total_account_value"], "$99,997.33")
        self.assertEqual(snapshot["holdings_value"], "$247.33")
        self.assertEqual(snapshot["buying_power"], "$99,750.00")
        self.assertEqual(snapshot["percent_in_trade"], "0.25%")
        self.assertEqual(snapshot["open_positions"], "2")
        self.assertEqual(snapshot["realized_pnl"], "N/A")

    def test_set_market_positions_populates_rich_stock_row_and_summary(self) -> None:
        hub = self._hub()
        hub._read_market_thinker_status = lambda market_key: {}
        tree = _Tree()
        summary_var = _Var()
        hub.market_panels = {
            "stocks": {
                "positions_tree": tree,
                "positions_summary_var": summary_var,
                "positions_columns": PowerTraderHub._market_position_schema(hub, "stocks")["columns"],
            }
        }

        PowerTraderHub._set_market_positions(
            hub,
            "stocks",
            [],
            raw_positions=[
                {
                    "symbol": "AMZN",
                    "qty": "0.230557118",
                    "qty_available": "0.230557118",
                    "avg_entry_price": "216.866",
                    "side": "long",
                    "market_value": "48.388175",
                    "unrealized_pl": "-1.611825",
                    "unrealized_plpc": "-0.03224",
                    "current_price": "209.875",
                    "change_today": "-0.01305",
                }
            ],
            status_data={},
        )

        self.assertEqual(len(tree.rows), 1)
        row = next(iter(tree.rows.values()))
        self.assertEqual(
            row["values"],
            (
                "AMZN",
                "LONG",
                "0.230557",
                "$48.39",
                "-1.61",
                "N/A",
                "$216.866",
                "$209.875",
                "-1.31%",
                "0.230557",
            ),
        )
        self.assertIn("Open trades: 1", summary_var.get())

    def test_market_position_cell_fg_matches_crypto_style_signal_cells(self) -> None:
        hub = self._hub()

        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "stocks", "symbol", "AMZN"), "#00E5FF")
        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "stocks", "side", "LONG"), "#00FF66")
        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "stocks", "day_pct", "-1.31%"), "#FF6B57")
        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "forex", "side", "SHORT"), "#FF6B57")
        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "forex", "unrealized_usd", "+0.0005 USD"), "#00FF66")
        self.assertEqual(PowerTraderHub._market_position_cell_fg(hub, "forex", "financing", "N/A"), "#8B949E")

    def test_set_market_positions_draws_scrollable_canvas_table(self) -> None:
        hub = self._hub()
        hub._read_market_thinker_status = lambda market_key: {}
        canvas = _Canvas(width=320, height=120)
        summary_var = _Var()
        hub.market_panels = {
            "stocks": {
                "positions_canvas": canvas,
                "positions_tree": _Tree(),
                "positions_summary_var": summary_var,
                "positions_columns": PowerTraderHub._market_position_schema(hub, "stocks")["columns"],
                "positions_headings": PowerTraderHub._market_position_schema(hub, "stocks")["headings"],
                "positions_widths": PowerTraderHub._market_position_schema(hub, "stocks")["widths"],
                "positions_numeric_cols": PowerTraderHub._market_position_schema(hub, "stocks")["numeric_cols"],
                "positions_center_cols": PowerTraderHub._market_position_schema(hub, "stocks")["center_cols"],
                "positions_rows": [],
            }
        }

        PowerTraderHub._set_market_positions(
            hub,
            "stocks",
            [],
            raw_positions=[
                {
                    "symbol": "AMZN",
                    "qty": "0.230557118",
                    "qty_available": "0.230557118",
                    "avg_entry_price": "216.866",
                    "side": "long",
                    "market_value": "48.388175",
                    "unrealized_pl": "-1.611825",
                    "unrealized_plpc": "-0.03224",
                    "current_price": "209.875",
                    "change_today": "-0.01305",
                }
            ],
            status_data={},
        )

        scrollregion = canvas.config.get("scrollregion")
        self.assertIsNotNone(scrollregion)
        self.assertGreater(scrollregion[2], canvas.winfo_width())
        texts = [entry[2].get("text") for entry in canvas.items if entry[0] == "text"]
        self.assertIn("AMZN", texts)
        self.assertIn("-1.61", texts)
        self.assertIn("Open trades: 1", summary_var.get())

    def test_forex_chart_overview_payload_exposes_avg_target_and_trail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            state_path = os.path.join(fx_dir, "forex_trader_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"trail": {"EUR_USD": {"armed": True, "peak_pct": 0.5}}}, f)

            thinker_data = {
                "top_pick": {
                    "pair": "EUR_USD",
                    "side": "long",
                    "confidence": "HIGH",
                    "eligible_for_entry": True,
                    "last": 1.1022,
                    "change_6h_pct": 0.31,
                    "change_24h_pct": 0.48,
                    "bars_count": 48,
                    "spread_bps": 1.2,
                    "calibration_effective_prob": 0.73,
                    "reason_logic": "Uptrend signal: both short-term and daily momentum are positive.",
                },
                "leaders": [{"pair": "EUR_USD", "last": 1.1022}],
            }

            hub = self._hub()
            hub.settings = {
                "forex_profit_target_pct": 0.2,
                "forex_trailing_gap_pct": 0.15,
                "forex_max_spread_bps": 8.0,
            }
            focus_var = _Var()
            focus_var.set("AUTO")
            hub.market_panels = {"forex": {"instrument_var": focus_var}}
            hub.market_state_dirs = {"forex": fx_dir}
            hub.market_trader_state_paths = {"forex": state_path}
            hub._read_market_thinker_status = lambda market_key: thinker_data

            payload = PowerTraderHub._market_chart_overview_payload(
                hub,
                "forex",
                thinker_data=thinker_data,
                status_data={
                    "currency": "USD",
                    "raw_positions": [
                        {
                            "instrument": "EUR_USD",
                            "long": {
                                "units": "25",
                                "averagePrice": "1.1000",
                                "unrealizedPL": "0.0550",
                            },
                            "short": {"units": "0"},
                        }
                    ],
                },
            )

            self.assertIn("EUR_USD | LONG | 25 units open", payload["body_lines"][0])
            self.assertIn("Target arm", payload["body_lines"][2])
            self.assertIn("Trail", payload["body_lines"][2])
            self.assertEqual([row["label"] for row in payload["overlays"]], ["Avg", "Target", "Trail"])

    def test_stock_chart_overview_payload_exposes_avg_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            state_path = os.path.join(stocks_dir, "stock_trader_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"trail": {"AAPL": {"armed": False, "peak_pct": 0.0}}}, f)

            thinker_data = {
                "top_pick": {
                    "symbol": "AAPL",
                    "side": "long",
                    "confidence": "HIGH",
                    "eligible_for_entry": True,
                    "last": 210.0,
                    "change_6h_pct": 0.52,
                    "change_24h_pct": 0.73,
                    "bars_count": 36,
                    "spread_bps": 0.9,
                    "calib_prob": 0.68,
                    "reason_logic": "Long-biased setup: near-term momentum has turned higher.",
                },
                "leaders": [{"symbol": "AAPL", "last": 210.0}],
            }

            hub = self._hub()
            hub.settings = {
                "stock_profit_target_pct": 0.35,
                "stock_trailing_gap_pct": 0.2,
                "stock_max_spread_bps": 8.0,
            }
            focus_var = _Var()
            focus_var.set("AUTO")
            hub.market_panels = {"stocks": {"instrument_var": focus_var}}
            hub.market_state_dirs = {"stocks": stocks_dir}
            hub.market_trader_state_paths = {"stocks": state_path}
            hub._read_market_thinker_status = lambda market_key: thinker_data

            payload = PowerTraderHub._market_chart_overview_payload(
                hub,
                "stocks",
                thinker_data=thinker_data,
                status_data={
                    "raw_positions": [
                        {
                            "symbol": "AAPL",
                            "side": "long",
                            "qty": "1",
                            "avg_entry_price": "209.00",
                            "current_price": "210.00",
                            "unrealized_pl": "1.00",
                        }
                    ],
                },
            )

            self.assertIn("AAPL | LONG | 1.000000 shares open", payload["body_lines"][0])
            self.assertIn("Trail waits for +0.35%", payload["body_lines"][2])
            self.assertEqual([row["label"] for row in payload["overlays"]], ["Avg", "Target"])

    def test_market_chart_focus_options_prioritize_account_positions_then_ranked_symbols(self) -> None:
        hub = self._hub()

        options = PowerTraderHub._market_chart_focus_options(
            hub,
            "stocks",
            thinker_data={
                "leaders": [{"symbol": "MSFT"}, {"symbol": "TSLA"}],
                "all_scores": [{"symbol": "NVDA"}, {"symbol": "MSFT"}],
            },
            status_data={
                "raw_positions": [
                    {"symbol": "AAPL"},
                    {"symbol": "TSLA"},
                ]
            },
        )

        self.assertEqual(options, ["ACCOUNT", "AAPL", "TSLA", "MSFT", "NVDA"])

    def test_market_chart_focus_options_keep_manual_selection_pinned(self) -> None:
        hub = self._hub()
        hub.market_panels = {"stocks": {"instrument_var": _Var()}}
        hub.market_panels["stocks"]["instrument_var"].set("AMZN")

        options = PowerTraderHub._market_chart_focus_options(
            hub,
            "stocks",
            thinker_data={"leaders": [{"symbol": "MSFT"}]},
            status_data={"raw_positions": [{"symbol": "AAPL"}]},
        )

        self.assertEqual(options, ["ACCOUNT", "AMZN", "AAPL", "MSFT"])

    def test_market_view_options_hide_market_tabs_for_stocks_and_forex(self) -> None:
        hub = self._hub()

        self.assertEqual(PowerTraderHub._market_view_options(hub, "stocks"), ("Overview",))
        self.assertEqual(PowerTraderHub._market_view_options(hub, "forex"), ("Overview",))

    def test_market_watchlist_rows_use_leaders(self) -> None:
        hub = self._hub()
        rows = PowerTraderHub._market_watchlist_rows(
            hub,
            "forex",
            thinker_data={
                "leaders": [
                    {
                        "pair": "AUD_USD",
                        "side": "short",
                        "score": -0.6607,
                        "confidence": "HIGH",
                        "reason": "Trend continuation",
                    }
                ]
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "AUD_USD")
        self.assertEqual(rows[0]["status"], "ENTRY WAIT")
        self.assertEqual(rows[0]["status"], "ENTRY WAIT")
        self.assertIn("-0.6607", rows[0]["score"])
        self.assertTrue(bool(str(rows[0].get("why", "") or "").strip()))
        self.assertIn("Needs SHORT setup", rows[0]["trigger"])

    def test_market_watchlist_rows_mark_ready_entries_with_trigger(self) -> None:
        hub = self._hub()

        rows = PowerTraderHub._market_watchlist_rows(
            hub,
            "stocks",
            thinker_data={
                "leaders": [
                    {
                        "symbol": "ACDC",
                        "side": "long",
                        "score": 6.388364,
                        "confidence": "HIGH",
                        "reason": "Uptrend pressure from positive 6h/24h momentum",
                        "eligible_for_entry": True,
                        "last": 6.96,
                        "calib_prob": 0.725,
                    }
                ]
            },
        )

        self.assertEqual(rows[0]["status"], "READY")
        self.assertIn("Trader step can open LONG", rows[0]["trigger"])
        self.assertIn("last $6.96", rows[0]["trigger"])

    def test_resolve_market_focus_chart_rows_keeps_selected_symbol_pinned_to_cache(self) -> None:
        hub = self._hub()
        hub.market_panels = {"stocks": {"instrument_var": _Var()}}
        hub.market_panels["stocks"]["instrument_var"].set("AMZN")
        hub._market_line_caches = {
            "stocks": {
                "AMZN": {
                    "rows": [
                        {"t": "2026-03-12T10:00:00", "o": 209.0, "h": 210.0, "l": 208.5, "c": 209.5},
                        {"t": "2026-03-12T11:00:00", "o": 209.5, "h": 210.5, "l": 209.1, "c": 210.25},
                    ],
                    "source": "focus",
                }
            }
        }

        payload = PowerTraderHub._resolve_market_focus_chart_rows(
            hub,
            "stocks",
            thinker_data={
                "top_pick": {"symbol": "MSFT"},
                "top_chart_map": {
                    "MSFT": [
                        {"t": "2026-03-12T10:00:00", "o": 390.0, "h": 392.0, "l": 389.5, "c": 391.5},
                        {"t": "2026-03-12T11:00:00", "o": 391.5, "h": 393.0, "l": 391.0, "c": 392.75},
                    ]
                },
            },
        )

        self.assertEqual(payload["focus_symbol"], "AMZN")
        self.assertTrue(payload["from_cache"])
        self.assertTrue(str(payload["source"]).endswith(":cache"))
        self.assertEqual([row["c"] for row in payload["rows"]], [209.5, 210.25])

    def test_resolve_market_focus_chart_rows_uses_live_focus_bars_when_available(self) -> None:
        hub = self._hub()
        hub.market_panels = {"forex": {"instrument_var": _Var()}}
        hub.market_panels["forex"]["instrument_var"].set("AUD_USD")
        hub._market_line_caches = {}

        payload = PowerTraderHub._resolve_market_focus_chart_rows(
            hub,
            "forex",
            thinker_data={
                "top_pick": {"pair": "AUD_HKD"},
                "top_chart_map": {
                    "AUD_USD": [
                        {"t": "2026-03-12T10:00:00", "o": 0.6510, "h": 0.6520, "l": 0.6505, "c": 0.6515},
                        {"t": "2026-03-12T11:00:00", "o": 0.6515, "h": 0.6530, "l": 0.6510, "c": 0.6528},
                    ],
                    "AUD_HKD": [
                        {"t": "2026-03-12T10:00:00", "o": 5.55, "h": 5.56, "l": 5.54, "c": 5.545},
                        {"t": "2026-03-12T11:00:00", "o": 5.545, "h": 5.55, "l": 5.53, "c": 5.535},
                    ],
                },
            },
        )

        self.assertEqual(payload["focus_symbol"], "AUD_USD")
        self.assertFalse(payload["from_cache"])
        self.assertEqual(payload["source"], "focus")
        self.assertEqual([row["c"] for row in payload["rows"]], [0.6515, 0.6528])

    def test_market_chart_benchmark_overlays_include_position_tooltips(self) -> None:
        hub = self._hub()

        overlays = PowerTraderHub._market_chart_benchmark_overlays(
            hub,
            "forex",
            position_ctx={"avg_price": 1.1000},
            parsed=[
                {"o": 1.0990, "h": 1.1010, "l": 1.0980, "c": 1.1005},
                {"o": 1.1005, "h": 1.1020, "l": 1.0995, "c": 1.1015},
            ],
            base_overlays=[
                {"label": "Avg", "price": 1.1000, "color": "#A3B1FF", "dash": (4, 2)},
                {"label": "Target", "price": 1.1022, "color": "#00E676", "dash": (6, 3)},
                {"label": "Trail", "price": 1.1011, "color": "#FFB347", "dash": (6, 3)},
            ],
        )

        self.assertEqual([row["label"] for row in overlays], ["Avg", "Target", "Trail"])
        self.assertIn("Meaning: Current blended entry price", overlays[0]["tooltip"])
        self.assertIn("Impact if hit:", overlays[1]["tooltip"])

    def test_market_chart_benchmark_overlays_add_candidate_range_levels(self) -> None:
        hub = self._hub()

        overlays = PowerTraderHub._market_chart_benchmark_overlays(
            hub,
            "stocks",
            focus_row={"side": "LONG"},
            parsed=[
                {"o": 209.0, "h": 210.0, "l": 208.5, "c": 209.5},
                {"o": 209.5, "h": 211.0, "l": 209.0, "c": 210.8},
                {"o": 210.8, "h": 212.4, "l": 210.1, "c": 211.9},
            ],
        )

        labels = [row["label"] for row in overlays]
        self.assertIn("Breakout", labels)
        self.assertIn("Range Low", labels)
        self.assertTrue(any("Impact if hit:" in str(row.get("tooltip", "")) for row in overlays))

    def test_market_history_display_rows_show_completed_stock_trades_only(self) -> None:
        hub = self._hub()
        hub._format_ui_timestamp = lambda ts, include_date=False: "2026-03-12 17:27:08"
        hub._market_fmt_num = lambda value, digits=6: f"{float(value):.{int(digits)}f}"
        hub._market_fmt_money = lambda value, digits=2: f"${float(value):,.{int(digits)}f}"

        rows = PowerTraderHub._market_history_display_rows(
            hub,
            "stocks",
            [
                {"event": "shadow_live_divergence", "symbol": "CF", "msg": "Market hours gate: market closed"},
                {"event": "entry", "symbol": "O", "side": "buy", "notional": 200.0, "price": 65.335, "ok": True, "ts": 1},
            ],
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("BUY/OPEN", rows[0]["text"])
        self.assertIn("O", rows[0]["text"])
        self.assertIn("qty=$200.00", rows[0]["text"])

    def test_market_history_display_rows_use_action_colors_and_show_realized_on_closes(self) -> None:
        hub = self._hub()
        hub._format_ui_timestamp = lambda ts, include_date=False: "2026-03-12 17:27:08"
        hub._market_fmt_num = lambda value, digits=6: f"{float(value):.{int(digits)}f}"
        hub._market_fmt_money = lambda value, digits=2: f"${float(value):,.{int(digits)}f}"

        stock_rows = PowerTraderHub._market_history_display_rows(
            hub,
            "stocks",
            [{"event": "exit", "symbol": "O", "side": "sell", "qty": 2.0, "price": 66.0, "pnl_usd": 1.23, "ok": True, "ts": 1}],
        )
        self.assertEqual(len(stock_rows), 1)
        self.assertIn("SELL/CLOSE", stock_rows[0]["text"])
        self.assertIn("realized=+1.23", stock_rows[0]["text"])
        self.assertEqual(stock_rows[0]["fg"], "#00FF66")

        forex_rows = PowerTraderHub._market_history_display_rows(
            hub,
            "forex",
            [{"event": "exit", "instrument": "AUD_USD", "side": "short", "units": 2, "price": 0.6500, "pnl_usd": -0.12, "ok": True, "ts": 2}],
        )
        self.assertEqual(len(forex_rows), 1)
        self.assertIn("SELL/CLOSE", forex_rows[0]["text"])
        self.assertIn("realized=-0.12", forex_rows[0]["text"])
        self.assertEqual(forex_rows[0]["fg"], "#00FF66")

    def test_resolved_market_history_rows_backfills_missing_open_stock_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stocks_dir = os.path.join(td, "stocks")
            os.makedirs(stocks_dir, exist_ok=True)
            audit_path = os.path.join(stocks_dir, "execution_audit.jsonl")
            with open(audit_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "entry", "symbol": "O", "side": "buy", "price": 65.335, "ts": 1, "ok": True}) + "\n")
                f.write(json.dumps({"event": "shadow_live_divergence", "symbol": "CF", "ts": 2}) + "\n")

            hub = self._hub()
            hub.project_dir = td
            hub.hub_dir = td
            hub.market_state_dirs = {"stocks": stocks_dir}

            rows = PowerTraderHub._resolved_market_history_rows(
                hub,
                "stocks",
                [{"event": "shadow_live_divergence", "symbol": "CF", "ts": 2}],
                status_data={
                    "raw_positions": [
                        {"symbol": "AMZN", "side": "long", "qty": "0.230557118", "avg_entry_price": "216.866"},
                        {"symbol": "O", "side": "long", "qty": "3.060912151", "avg_entry_price": "65.34"},
                    ],
                },
            )

            symbols = [str(row.get("symbol", "") or row.get("instrument", "") or "").strip().upper() for row in rows]
            self.assertIn("O", symbols)
            self.assertIn("AMZN", symbols)
            amzn_row = next(row for row in rows if str(row.get("symbol", "") or "").upper() == "AMZN")
            self.assertTrue(amzn_row.get("_synthetic"))

    def test_resolved_market_history_rows_reload_full_forex_audit_past_noise_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fx_dir = os.path.join(td, "forex")
            os.makedirs(fx_dir, exist_ok=True)
            audit_path = os.path.join(fx_dir, "execution_audit.jsonl")
            with open(audit_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"event": "entry", "instrument": "AUD_HKD", "side": "short", "units": -2, "price": 5.54943, "ts": 10, "ok": True}) + "\n")
                f.write(json.dumps({"event": "entry", "instrument": "GBP_USD", "side": "short", "units": -32, "price": 1.33455, "ts": 20, "ok": True}) + "\n")
                for idx in range(150):
                    f.write(json.dumps({"event": "shadow_live_divergence", "instrument": "AUD_USD", "ts": 100 + idx}) + "\n")

            hub = self._hub()
            hub.project_dir = td
            hub.hub_dir = td
            hub.market_state_dirs = {"forex": fx_dir}

            rows = PowerTraderHub._resolved_market_history_rows(
                hub,
                "forex",
                [{"event": "shadow_live_divergence", "instrument": "AUD_USD", "ts": 999}],
                status_data={
                    "raw_positions": [
                        {
                            "instrument": "AUD_HKD",
                            "long": {"units": "0"},
                            "short": {"units": "-2", "averagePrice": "5.54943"},
                        },
                        {
                            "instrument": "GBP_USD",
                            "long": {"units": "0"},
                            "short": {"units": "-32", "averagePrice": "1.33455"},
                        },
                    ],
                },
            )

            idents = [str(row.get("instrument", "") or row.get("symbol", "") or "").strip().upper() for row in rows]
            self.assertIn("AUD_HKD", idents)
            self.assertIn("GBP_USD", idents)
            self.assertEqual(sum(1 for ident in idents if ident == "AUD_HKD"), 1)

    def test_set_market_history_populates_listbox_like_crypto(self) -> None:
        hub = self._hub()
        listbox = _Listbox()
        auto_scroll_var = _Var()
        auto_scroll_var.set(True)
        hub.market_panels = {
            "forex": {
                "history_list": listbox,
                "history_autoscroll_var": auto_scroll_var,
            }
        }

        PowerTraderHub._set_market_history(
            hub,
            "forex",
            [
                {"text": "2026-03-12 16:58:24 | BUY/OPEN  | GBP_USD | qty=32 | px=1.33455", "fg": "#00E5FF"},
                {"text": "2026-03-12 17:19:51 | SELL/CLOSE | AUD_CAD | qty=14 | px=0.96554 | realized=+0.12", "fg": "#00FF66"},
            ],
        )

        self.assertEqual(len(listbox.rows), 2)
        self.assertIn("BUY/OPEN", listbox.rows[0])
        self.assertEqual(listbox.item_styles[0]["fg"], "#00E5FF")
        self.assertEqual(listbox.item_styles[1]["fg"], "#00FF66")
        self.assertEqual(listbox.y_pos, 0.0)

    def test_set_diagnostics_busy_ui_updates_all_buttons(self) -> None:
        hub = self._hub()
        hub.btn_quick_diag = _Button()
        hub.btn_toolbar_diag = _Button()

        PowerTraderHub._set_diagnostics_busy_ui(hub, True)
        self.assertEqual(hub.btn_quick_diag.config["state"], "disabled")
        self.assertEqual(hub.btn_quick_diag.config["text"], "Running Diagnostics")
        self.assertEqual(hub.btn_toolbar_diag.config["state"], "disabled")
        self.assertEqual(hub.btn_toolbar_diag.config["text"], "…")

        PowerTraderHub._set_diagnostics_busy_ui(hub, False)
        self.assertEqual(hub.btn_quick_diag.config["state"], "normal")
        self.assertEqual(hub.btn_quick_diag.config["text"], "Quick Diagnostics")
        self.assertEqual(hub.btn_toolbar_diag.config["state"], "normal")
        self.assertEqual(hub.btn_toolbar_diag.config["text"], "✚")

    def test_refresh_market_chart_hover_restores_tooltip_after_render(self) -> None:
        hub = self._hub()
        hub._live_log_font = _Font()
        canvas = _Canvas(width=420, height=220)
        canvas.root_x = 100
        canvas.root_y = 200
        canvas.pointer_x = 220
        canvas.pointer_y = 278
        hub.market_panels = {
            "forex": {
                "chart_canvas": canvas,
                "chart_hover_data": {
                    "mode": "candles",
                    "plot_left": 40.0,
                    "plot_right": 380.0,
                    "plot_top": 30.0,
                    "plot_bot": 180.0,
                    "x_points": [80.0, 120.0, 160.0],
                    "rows": [
                        {"t": "bar 1", "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05},
                        {"t": "bar 2", "o": 1.05, "h": 1.15, "l": 1.0, "c": 1.12},
                        {"t": "bar 3", "o": 1.12, "h": 1.18, "l": 1.1, "c": 1.14},
                    ],
                    "line_targets": [
                        {"y": 78.0, "color": "#00E676", "dash": (6, 3), "tooltip": "Target\nImpact if hit: trims position."}
                    ],
                },
                "chart_hover_idx": -1,
            }
        }

        PowerTraderHub._refresh_market_chart_hover(hub, "forex")

        self.assertEqual(hub.market_panels["forex"]["chart_hover_idx"], -2)
        self.assertTrue(any(item[0] == "text" for item in canvas.items))
        self.assertTrue(any("Impact if hit" in str(kwargs.get("text", "")) for _kind, _args, kwargs in canvas.items if isinstance(kwargs, dict)))

    def test_restore_legend_hover_replays_tooltip_for_stationary_pointer(self) -> None:
        widget = _Canvas(width=640, height=320)
        widget.root_x = 50
        widget.root_y = 80
        widget.pointer_x = 210
        widget.pointer_y = 240

        captured = []
        chart = CandleChart.__new__(CandleChart)
        chart.canvas = _FigureCanvas(widget)
        chart.ax = object()
        chart._legend_hover_motion_handler = lambda event: captured.append(event)
        chart._legend_hover_restore_after_id = None

        CandleChart._restore_legend_hover(chart)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].x, 160.0)
        self.assertEqual(captured[0].y, 160.0)
        self.assertEqual(captured[0].guiEvent.x_root, 210)
        self.assertEqual(captured[0].guiEvent.y_root, 240)

    def test_draw_watchlist_canvas_table_wraps_text_and_sets_scrollregion(self) -> None:
        hub = self._hub()
        hub._coerce_float_value = PowerTraderHub._coerce_float_value.__get__(hub, PowerTraderHub)
        canvas = _Canvas(width=560, height=140)

        regions = PowerTraderHub._draw_watchlist_canvas_table(
            hub,
            canvas,
            columns=("symbol", "status", "why", "logic", "trigger"),
            headings={
                "symbol": "Symbol",
                "status": "Status",
                "why": "Why Not Traded",
                "logic": "Logic",
                "trigger": "Trade Trigger",
            },
            rows=[
                {
                    "symbol": "AUD_HKD",
                    "status": "READY",
                    "why": "Waiting for the next qualified setup to remain valid through the next cycle.",
                    "logic": "Strong momentum plus volatility edge from the recent trend scan with additional confirmation still intact.",
                    "trigger": "Trader step can open SHORT on the next cycle if AUD_HKD keeps this setup and risk size still fits.",
                }
            ],
            base_widths={"symbol": 110, "status": 100, "why": 280, "logic": 320, "trigger": 360},
            kind="forex",
            selected_idx=0,
        )

        self.assertEqual(len(regions), 1)
        self.assertIn("scrollregion", canvas.config)
        self.assertGreater(canvas.config["scrollregion"][2], 0)
        self.assertTrue(any(item[0] == "rectangle" for item in canvas.items))
        self.assertTrue(any(item[0] == "text" for item in canvas.items))

    def test_activate_market_watchlist_selection_uses_canvas_selected_row(self) -> None:
        hub = self._hub()
        focus_var = _Var()
        focus_var.set("ACCOUNT")
        triggered = []
        hub.market_panels = {
            "forex": {
                "instrument_var": focus_var,
                "watch_selected_idx": 1,
                "watch_rows": [
                    {"symbol": "AUD_HKD"},
                    {"symbol": "GBP_USD"},
                ],
            }
        }
        hub._on_market_focus_changed = lambda market_key: triggered.append(market_key)

        PowerTraderHub._activate_market_watchlist_selection(hub, "forex")

        self.assertEqual(focus_var.get(), "GBP_USD")
        self.assertEqual(triggered, ["forex"])


if __name__ == "__main__":
    unittest.main()
