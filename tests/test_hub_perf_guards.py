from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch


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

from ui.pt_hub import PowerTraderHub


class _Var:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _Label:
    def __init__(self) -> None:
        self.calls = 0
        self.configs = []

    def configure(self, **kwargs) -> None:
        self.calls += 1
        self.configs.append(dict(kwargs))


class _Button:
    def __init__(self) -> None:
        self.calls = 0
        self.configs = []

    def configure(self, **kwargs) -> None:
        self.calls += 1
        self.configs.append(dict(kwargs))


class _TextWidget:
    def __init__(self) -> None:
        self.delete_calls = 0
        self.insert_calls = 0
        self.payload = ""

    def configure(self, **_kwargs) -> None:
        return None

    def delete(self, *_args) -> None:
        self.delete_calls += 1
        self.payload = ""

    def insert(self, *_args) -> None:
        self.insert_calls += 1
        if len(_args) >= 2:
            self.payload = str(_args[1] or "")

    def winfo_ismapped(self) -> bool:
        return True


class _Tree:
    def __init__(self) -> None:
        self.rows = {}
        self.delete_calls = 0
        self.insert_calls = 0

    def get_children(self):
        return tuple(self.rows.keys())

    def delete(self, iid) -> None:
        self.delete_calls += 1
        self.rows.pop(iid, None)

    def insert(self, _parent, _where, values=(), tags=()):
        self.insert_calls += 1
        iid = f"row{self.insert_calls}"
        self.rows[iid] = {"values": tuple(values), "tags": tuple(tags)}
        return iid


class HubPerfGuardTests(unittest.TestCase):
    def test_set_badge_style_skips_duplicate_widget_config(self) -> None:
        hub = object.__new__(PowerTraderHub)
        lbl = _Label()
        PowerTraderHub._set_badge_style(hub, lbl, "Data: OK", tone="good")
        PowerTraderHub._set_badge_style(hub, lbl, "Data: OK", tone="good")
        PowerTraderHub._set_badge_style(hub, lbl, "Data: WARN", tone="warn")
        self.assertEqual(lbl.calls, 2)

    def test_set_market_notes_skips_rewrite_when_payload_unchanged(self) -> None:
        hub = object.__new__(PowerTraderHub)
        txt = _TextWidget()
        hub.market_panels = {"stocks": {"notes_text": txt}}

        PowerTraderHub._set_market_notes(hub, "stocks", "line a")
        PowerTraderHub._set_market_notes(hub, "stocks", "line a")
        PowerTraderHub._set_market_notes(hub, "stocks", "line b")

        self.assertEqual(txt.insert_calls, 2)
        self.assertEqual(txt.delete_calls, 2)
        self.assertEqual(txt.payload, "line b\n")

    def test_set_market_positions_skips_full_table_redraw_on_same_rows(self) -> None:
        hub = object.__new__(PowerTraderHub)
        tree = _Tree()
        summary = _Var()
        draw_calls = []
        hub._draw_market_positions_table = lambda market_key: draw_calls.append(str(market_key))
        hub._market_position_rows = lambda *args, **kwargs: {
            "rows": [{"symbol": "AAPL", "qty": "1", "_upl_f": 1.0}],
            "schema": {"columns": ("symbol", "qty")},
            "summary": "Open positions: 1",
        }
        hub.market_panels = {
            "stocks": {
                "positions_tree": tree,
                "positions_canvas": object(),
                "positions_summary_var": summary,
                "positions_columns": ("symbol", "qty"),
            }
        }

        PowerTraderHub._set_market_positions(hub, "stocks", [])
        first_insert_calls = tree.insert_calls
        first_draw_calls = len(draw_calls)

        PowerTraderHub._set_market_positions(hub, "stocks", [])
        self.assertEqual(tree.insert_calls, first_insert_calls)
        self.assertEqual(len(draw_calls), first_draw_calls)

        hub._market_position_rows = lambda *args, **kwargs: {
            "rows": [{"symbol": "AAPL", "qty": "2", "_upl_f": -1.0}],
            "schema": {"columns": ("symbol", "qty")},
            "summary": "Open positions: 1",
        }
        PowerTraderHub._set_market_positions(hub, "stocks", [])
        self.assertGreater(tree.insert_calls, first_insert_calls)
        self.assertGreater(len(draw_calls), first_draw_calls)

    def test_training_status_map_uses_short_ttl_cache(self) -> None:
        hub = object.__new__(PowerTraderHub)
        hub.coins = ["BTC", "ETH"]
        hub._training_status_refresh_s = 2.5
        hub._training_status_cache = {}
        calls = {"running": 0, "trained": 0}

        def _running_trainers():
            calls["running"] += 1
            return ["BTC"]

        def _coin_is_trained(coin: str) -> bool:
            calls["trained"] += 1
            return str(coin or "").upper() == "ETH"

        hub._running_trainers = _running_trainers
        hub._coin_is_trained = _coin_is_trained

        with patch("ui.pt_hub.time.time", return_value=1000.0):
            out_a = PowerTraderHub._training_status_map(hub, ["BTC", "ETH"])
        with patch("ui.pt_hub.time.time", return_value=1001.0):
            out_b = PowerTraderHub._training_status_map(hub, ["BTC", "ETH"])
        with patch("ui.pt_hub.time.time", return_value=1004.0):
            out_c = PowerTraderHub._training_status_map(hub, ["BTC", "ETH"])

        self.assertEqual(out_a, {"BTC": "TRAINING", "ETH": "TRAINED"})
        self.assertEqual(out_b, out_a)
        self.assertEqual(out_c, out_a)
        self.assertEqual(calls["running"], 2)

    def test_refresh_log_file_to_text_skips_hidden_widgets(self) -> None:
        import tempfile

        hub = object.__new__(PowerTraderHub)
        txt = _TextWidget()
        txt.winfo_ismapped = lambda: False  # type: ignore[assignment]
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("line-a\nline-b\n")
            log_path = f.name
        try:
            PowerTraderHub._refresh_log_file_to_text(
                hub,
                path=log_path,
                txt=txt,  # type: ignore[arg-type]
                cache_key="_test_hidden_log_sig",
                max_lines=50,
            )
        finally:
            try:
                os.remove(log_path)
            except Exception:
                pass
        self.assertEqual(txt.delete_calls, 0)
        self.assertEqual(txt.insert_calls, 0)

    def test_refresh_crypto_scan_button_state_tracks_busy_flag(self) -> None:
        hub = object.__new__(PowerTraderHub)
        btn = _Button()
        hub.btn_crypto_run_scan = btn
        hub._crypto_scan_busy = False
        PowerTraderHub._refresh_crypto_scan_button_state(hub)
        self.assertTrue(btn.configs)
        self.assertEqual(btn.configs[-1].get("text"), "Run Scan")
        self.assertEqual(btn.configs[-1].get("state"), "normal")

        hub._crypto_scan_busy = True
        PowerTraderHub._refresh_crypto_scan_button_state(hub)
        self.assertEqual(btn.configs[-1].get("text"), "Scanning...")
        self.assertEqual(btn.configs[-1].get("state"), "disabled")


if __name__ == "__main__":
    unittest.main()
