from __future__ import annotations
import os
import sys
import json
import csv
import time
import math
import traceback
import textwrap
import queue
import threading
import subprocess
import shutil
import glob
import bisect
import signal
import zipfile
import re
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from app.file_watch import FileChangeWatch
from app.json_codec import dump as fast_json_dump
from app.json_codec import dumps as fast_json_dumps
from app.json_codec import load as fast_json_load
from app.json_codec import loads as fast_json_loads
from app.path_utils import resolve_runtime_paths, resolve_settings_path, log_once
from app.runtime_logging import append_jsonl, runtime_event
from app.rejection_replay import build_rejection_replay_report
from app.strategy_lab import run_strategy_lab_suite
from app.operator_notes import (
    append_operator_note_entry,
    ensure_operator_notes_files,
    read_operator_notes_markdown,
    read_recent_operator_note_entries,
    write_operator_notes_markdown,
)
from app.settings_utils import (
    PROFILE_MANUAL_OVERRIDE_ALLOWLIST,
    sanitize_settings,
    recommend_market_profile_overrides,
    normalize_settings_profile,
)
from app.market_awareness import build_awareness_payload
from app.health_rules import evaluate_runtime_alerts
from app.notification_center import build_notification_center_from_hub
from app.status_hydration import load_market_status_bundle, needs_market_snapshot_refresh, safe_read_jsonl_dicts
from app.api_endpoint_validation import (
    ALPACA_DATA_HOST,
    ALPACA_LIVE_HOST,
    ALPACA_PAPER_HOST,
    OANDA_LIVE_REST_HOST,
    OANDA_LIVE_STREAM_HOST,
    OANDA_PRACTICE_REST_HOST,
    OANDA_PRACTICE_STREAM_HOST,
    normalize_endpoint_url,
)
from brokers.broker_alpaca import AlpacaBrokerClient
from brokers.broker_oanda import OandaBrokerClient
from app.credential_utils import (
    alpaca_credential_paths,
    get_alpaca_creds,
    get_openai_api_key,
    get_oanda_creds,
    get_twelvedata_api_key,
    get_robinhood_creds_from_env,
    normalize_start_allocation_pct,
    oanda_credential_paths,
    openai_credential_path,
    robinhood_credential_paths,
    twelvedata_credential_path,
    get_robinhood_creds_from_files,
)
from engines.stock_thinker import run_scan as run_stock_scan
from engines.forex_thinker import run_scan as run_forex_scan
from engines.stock_trader import run_step as run_stock_trader_step
from engines.forex_trader import run_step as run_forex_trader_step

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# Maintain path resolution parity with legacy ui/pt_hub.py by pointing at ui/pt_hub.py
_PT_HUB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pt_hub.py")

DARK_BG = "#070B10"
DARK_BG2 = "#0B1220"
DARK_PANEL = "#0E1626"
DARK_PANEL2 = "#121C2F"
DARK_BORDER = "#243044"
DARK_FG = "#C7D1DB"
DARK_MUTED = "#8B949E"
DARK_ACCENT = "#00FF66"   
DARK_ACCENT2 = "#00E5FF"   
CYAN = DARK_ACCENT2
DARK_SELECT_BG = "#17324A"
DARK_SELECT_FG = "#00FF66"

# Chart-only palette tuned for a "terminal neon" look while keeping draw ops light.
CHART_BG = "#04070C"
CHART_PANEL = "#071019"
CHART_PANEL_ALT = "#0A1622"
CHART_BORDER = "#1D3A2C"
CHART_GRID = "#113323"
CHART_GRID_ALT = "#0C271B"
CHART_TITLE = "#3BFF9A"
CHART_TEXT = "#BFD2CC"
CHART_MUTED = "#7FA097"
CHART_UP = "#2BFF8B"
CHART_DOWN = "#FF667D"
CHART_LINE_MAIN = "#35E7FF"
CHART_LINE_GLOW = "#0E5C73"
CHART_LINE_ALT = "#45FFA8"
CHART_FILL_MAIN = "#0B1D2F"
CHART_EMA_FAST = "#35E7FF"
CHART_EMA_SLOW = "#FFC857"
CHART_LAST = "#35E7FF"
BADGE_STYLES: Dict[str, Tuple[str, str, str]] = {
    "good": ("#0F2B1D", "#6CFFB0", "#1E5A3C"),
    "warn": ("#2C2312", "#FFD27A", "#6A5324"),
    "bad": ("#2A1718", "#FF8D80", "#6A2C33"),
    "info": ("#12243A", "#8BD8FF", "#204A70"),
    "muted": ("#141B28", "#A7B4C4", "#2A3A52"),
}
BASE_DIR, SETTINGS_PATH, DEFAULT_HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(_PT_HUB_FILE, "pt_hub")

ROLLOUT_ORDER: Dict[str, int] = {
    "legacy": 0,
    "scan_expanded": 1,
    "risk_caps": 2,
    "execution_v2": 3,
    "shadow_only": 4,
    "live": 5,
    "live_guarded": 5,
}

ROLLOUT_STAGE_ALIASES: Dict[str, str] = {
    "live_guarded": "live",
}


def _normalize_rollout_stage(stage: str, default: str = "legacy") -> str:
    cur = str(stage or "").strip().lower()
    cur = str(ROLLOUT_STAGE_ALIASES.get(cur, cur))
    if cur not in ROLLOUT_ORDER:
        cur = str(default or "legacy").strip().lower()
    cur = str(ROLLOUT_STAGE_ALIASES.get(cur, cur))
    if cur not in ROLLOUT_ORDER:
        cur = "legacy"
    return cur


def _resolve_rollout_stage_for_broker_modes(stage: str, alpaca_paper_mode: bool, oanda_practice_mode: bool) -> Tuple[str, str]:
    cur = _normalize_rollout_stage(stage)
    original = cur
    live_markets: List[str] = []
    if not bool(alpaca_paper_mode):
        live_markets.append("Stocks/Alpaca")
    if not bool(oanda_practice_mode):
        live_markets.append("Forex/OANDA")
    if live_markets and int(ROLLOUT_ORDER.get(cur, 0)) < int(ROLLOUT_ORDER["execution_v2"]):
        cur = "live"
    elif live_markets and cur == "shadow_only":
        cur = "live"
    if live_markets and cur != original:
        return cur, "Live broker mode requires an executable rollout stage. Auto-promoted rollout to live."
    return cur, ""


@dataclass
class _WrapItem:
    w: tk.Widget
    padx: Tuple[int, int] = (0, 0)
    pady: Tuple[int, int] = (0, 0)


class WrapFrame(ttk.Frame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._items: List[_WrapItem] = []
        self._reflow_pending = False
        self._in_reflow = False
        self.bind("<Configure>", self._schedule_reflow)

    def add(self, widget: tk.Widget, padx=(0, 0), pady=(0, 0)) -> None:
        self._items.append(_WrapItem(widget, padx=padx, pady=pady))
        self._schedule_reflow()

    def clear(self, destroy_widgets: bool = True) -> None:

        for it in list(self._items):
            try:
                it.w.grid_forget()
            except Exception:
                pass
            if destroy_widgets:
                try:
                    it.w.destroy()
                except Exception:
                    pass
        self._items = []
        self._schedule_reflow()

    def _schedule_reflow(self, event=None) -> None:
        if self._reflow_pending:
            return
        self._reflow_pending = True
        self.after_idle(self._reflow)

    def _reflow(self) -> None:
        if self._in_reflow:
            self._reflow_pending = False
            return

        self._reflow_pending = False
        self._in_reflow = True
        try:
            width = self.winfo_width()
            if width <= 1:
                return
            usable_width = max(1, width - 6)

            for it in self._items:
                it.w.grid_forget()

            row = 0
            col = 0
            x = 0

            for it in self._items:
                reqw = max(it.w.winfo_reqwidth(), it.w.winfo_width())

                needed = 10 + reqw + it.padx[0] + it.padx[1]

                if col > 0 and (x + needed) > usable_width:
                    row += 1
                    col = 0
                    x = 0

                it.w.grid(row=row, column=col, sticky="w", padx=it.padx, pady=it.pady)
                x += needed
                col += 1
        finally:
            self._in_reflow = False


class NeuralSignalTile(ttk.Frame):

    def __init__(self, parent: tk.Widget, coin: str, bar_height: int = 52, levels: int = 8, trade_start_level: int = 3):
        super().__init__(parent)
        self.coin = coin

        self._hover_on = False
        self._normal_canvas_bg = DARK_PANEL2
        self._hover_canvas_bg = DARK_PANEL
        self._normal_border = DARK_BORDER
        self._hover_border = DARK_ACCENT2
        self._normal_fg = DARK_FG
        self._hover_fg = DARK_ACCENT2

        self._levels = max(2, int(levels))             
        self._display_levels = self._levels - 1        

        self._bar_h = int(bar_height)
        self._bar_w = 12
        self._gap = 16
        self._pad = 6

        self._base_fill = DARK_PANEL
        self._long_fill = "blue"
        self._short_fill = "orange"

        self.title_lbl = ttk.Label(self, text=coin)
        self.title_lbl.pack(anchor="center")

        w = (self._pad * 2) + (self._bar_w * 2) + self._gap
        h = (self._pad * 2) + self._bar_h

        self.canvas = tk.Canvas(
            self,
            width=w,
            height=h,
            bg=self._normal_canvas_bg,
            highlightthickness=1,
            highlightbackground=self._normal_border,
        )
        self.canvas.pack(padx=2, pady=(2, 0))

        x0 = self._pad
        x1 = x0 + self._bar_w
        x2 = x1 + self._gap
        x3 = x2 + self._bar_w
        yb = self._pad + self._bar_h

        # Build segmented bars: 7 segments for levels 1..7 (level 0 is "no highlight")
        self._long_segs: List[int] = []
        self._short_segs: List[int] = []

        for seg in range(self._display_levels):
            # seg=0 is bottom segment (level 1), seg=display_levels-1 is top segment (level 7)
            y_top = int(round(yb - ((seg + 1) * self._bar_h / self._display_levels)))
            y_bot = int(round(yb - (seg * self._bar_h / self._display_levels)))

            self._long_segs.append(
                self.canvas.create_rectangle(
                    x0, y_top, x1, y_bot,
                    fill=self._base_fill,
                    outline=DARK_BORDER,
                    width=1,
                )
            )
            self._short_segs.append(
                self.canvas.create_rectangle(
                    x2, y_top, x3, y_bot,
                    fill=self._base_fill,
                    outline=DARK_BORDER,
                    width=1,
                )
            )

        # Trade-start marker line (boundary before the trade-start level).
        # Example: trade_start_level=3 => line after 2nd block (between 2 and 3).
        self._trade_line_geom = (x0, x1, x2, x3, yb)
        self._trade_line_long = self.canvas.create_line(x0, yb, x1, yb, fill=DARK_FG, width=2)
        self._trade_line_short = self.canvas.create_line(x2, yb, x3, yb, fill=DARK_FG, width=2)
        self._trade_start_level = 3
        self.set_trade_start_level(trade_start_level)


        self.value_lbl = ttk.Label(self, text="L:0 S:0")
        self.value_lbl.pack(anchor="center", pady=(1, 0))

        self.set_values(0, 0)

    def set_hover(self, on: bool) -> None:
        """Visually highlight the tile on hover (like a button hover state)."""
        if bool(on) == bool(self._hover_on):
            return
        self._hover_on = bool(on)

        try:
            if self._hover_on:
                self.canvas.configure(
                    bg=self._hover_canvas_bg,
                    highlightbackground=self._hover_border,
                    highlightthickness=2,
                )
                self.title_lbl.configure(foreground=self._hover_fg)
                self.value_lbl.configure(foreground=self._hover_fg)
            else:
                self.canvas.configure(
                    bg=self._normal_canvas_bg,
                    highlightbackground=self._normal_border,
                    highlightthickness=1,
                )
                self.title_lbl.configure(foreground=self._normal_fg)
                self.value_lbl.configure(foreground=self._normal_fg)
        except Exception:
            pass

    def set_trade_start_level(self, level: Any) -> None:
        """Move the marker line to the boundary before the chosen start level."""
        self._trade_start_level = self._clamp_trade_start_level(level)
        self._update_trade_lines()

    def _clamp_trade_start_level(self, value: Any) -> int:
        try:
            v = int(float(value))
        except Exception:
            v = 3
        # Trade starts at levels 1..display_levels (usually 1..7)
        return max(1, min(v, self._display_levels))

    def _update_trade_lines(self) -> None:
        try:
            x0, x1, x2, x3, yb = self._trade_line_geom
        except Exception:
            return

        k = max(0, min(int(self._trade_start_level) - 1, self._display_levels))
        y = int(round(yb - (k * self._bar_h / self._display_levels)))

        try:
            self.canvas.coords(self._trade_line_long, x0, y, x1, y)
            self.canvas.coords(self._trade_line_short, x2, y, x3, y)
        except Exception:
            pass



    def _clamp_level(self, value: Any) -> int:
        try:
            v = int(float(value))
        except Exception:
            v = 0
        return max(0, min(v, self._levels - 1))  # logical clamp: 0..7

    def _set_level(self, seg_ids: List[int], level: int, active_fill: str) -> None:
        # Reset all segments to base
        for rid in seg_ids:
            self.canvas.itemconfigure(rid, fill=self._base_fill)

        # Level 0 -> show nothing (no highlight)
        if level <= 0:
            return

        # Level 1..7 -> fill from bottom up through the current level
        idx = level - 1  # level 1 maps to seg index 0
        if idx < 0:
            return
        if idx >= len(seg_ids):
            idx = len(seg_ids) - 1

        for i in range(idx + 1):
            self.canvas.itemconfigure(seg_ids[i], fill=active_fill)


    def set_values(self, long_sig: Any, short_sig: Any) -> None:
        ls = self._clamp_level(long_sig)
        ss = self._clamp_level(short_sig)

        self.value_lbl.config(text=f"L:{ls} S:{ss}")
        self._set_level(self._long_segs, ls, self._long_fill)
        self._set_level(self._short_segs, ss, self._short_fill)









# -----------------------------
# Settings / Paths
# -----------------------------

DEFAULT_SETTINGS = {
    "main_neural_dir": "market_data/coins",
    "coins": ["BTC", "ETH", "XRP", "BNB", "DOGE"],
    "trade_start_level": 3,  # trade starts when long signal >= this level (1..7)
    "start_allocation_pct": 0.5,  # % of total account value for initial entry (min $0.50 per coin)
    "dca_multiplier": 2.0,  # DCA buy size = current value * this (2.0 => total scales ~3x per DCA)
    "dca_levels": [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0],  # Hard DCA triggers (percent PnL)
    "max_dca_buys_per_24h": 2,  # max DCA buys per coin in rolling 24h window (0 disables DCA buys)

    # --- Trailing Profit Margin settings (used by pt_trader.py; shown in GUI settings) ---
    "pm_start_pct_no_dca": 5.0,
    "pm_start_pct_with_dca": 2.5,
    "trailing_gap_pct": 0.5,
    "max_position_usd_per_coin": 0.0,
    "max_total_exposure_pct": 0.0,

    "default_timeframe": "1hour",
    "timeframes": [
        "1min", "5min", "15min", "30min",
        "1hour", "2hour", "4hour", "8hour", "12hour",
        "1day", "1week"
    ],
    "candles_limit": 120,
    "ui_refresh_seconds": 1.0,
    "chart_refresh_seconds": 10.0,
    "auto_start_trading_when_all_trained": True,
    "hub_data_dir": "",  # if blank, defaults to <this_dir>/hub_data
    "script_neural_runner2": "engines/pt_thinker.py",
    "script_neural_trainer": "engines/pt_trainer.py",
    "script_trader": "engines/pt_trader.py",
    "crypto_trader_loop_sleep_s": 1.0,
    "crypto_trader_error_sleep_s": 1.5,
    "script_autopilot": "runtime/pt_autopilot.py",
    "kucoin_min_interval_sec": 0.40,
    "kucoin_cache_ttl_sec": 2.5,
    "kucoin_stale_max_sec": 120.0,
    "kucoin_unsupported_cooldown_s": 21600.0,
    "crypto_price_error_log_cooldown_s": 120.0,
    "crypto_dynamic_enabled": True,
    "crypto_dynamic_pool_symbols": "BTC,ETH,XRP,BNB,DOGE,SOL,ADA,PAXG,AVAX,LINK,LTC,UNI,AAVE,DOT,ATOM,MATIC",
    "crypto_dynamic_target_count": 8,
    "crypto_dynamic_scan_interval_s": 300,
    "crypto_dynamic_min_projected_edge_pct": 0.25,
    "crypto_dynamic_max_new_per_scan": 1,
    "crypto_dynamic_auto_train": True,
    "crypto_dynamic_max_trainers": 1,
    "crypto_dynamic_rotation_cooldown_s": 900,
    "crypto_replay_adaptive_enabled": True,
    "crypto_replay_adaptive_weight": 0.35,
    "crypto_replay_adaptive_step_cap_pct": 40.0,
    "crypto_min_calib_prob_live_guarded": 0.50,
    "crypto_min_samples_live_guarded": 6,
    "crypto_allocator_signal_floor": 0.15,
    "crypto_max_open_positions": 8,
    "news_event_enabled": True,
    "news_event_refresh_s": 900.0,
    "news_event_stale_max_s": 21600.0,
    "news_event_timeout_s": 8.0,
    "news_event_max_symbols_per_market": 20,
    "news_event_max_headlines_per_symbol": 6,
    "stock_news_event_weight": 0.12,
    "crypto_news_event_weight": 0.12,
    "market_chart_cache_symbols": 8,
    "market_chart_cache_bars": 120,
    "market_table_column_widths": {},
    "market_fallback_scan_max_age_s": 7200.0,
    "market_fallback_snapshot_max_age_s": 1800.0,
    "alpaca_api_key_id": "",
    "alpaca_secret_key": "",
    "alpaca_base_url": "https://api.alpaca.markets",
    "alpaca_data_url": "https://data.alpaca.markets",
    "stock_data_provider": "alpaca",
    "twelvedata_api_key": "",
    "twelvedata_base_url": "https://api.twelvedata.com",
    "twelvedata_api_credits_per_minute": 8,
    "twelvedata_daily_credits": 800,
    "twelvedata_scan_symbol_cap": 8,
    "alpaca_paper_mode": False,
    "market_rollout_stage": "live",  # internal rollout stage (locked to live)
    "settings_control_mode": "self_managed",  # preset_managed | self_managed
    "settings_profile": "balanced",  # safe | balanced | aggressive | max_growth
    "ui_role_mode": "basic",  # basic | advanced | admin
    "ui_timestamp_mode": "local_24h",  # local_24h | local_12h | utc_24h
    "ui_font_scale_preset": "normal",  # small | normal | large
    "ui_layout_preset": "auto",  # auto | compact | normal | wide
    "market_panel_compact_mode": False,
    "stock_universe_mode": "all_tradable_filtered",  # core | watchlist | all_tradable_filtered
    "stock_universe_symbols": "AAPL,MSFT,NVDA,AMZN,META,TSLA,SPY,QQQ",
    "stock_scan_max_symbols": 160,
    "stock_min_price": 5.0,
    "stock_max_price": 500.0,
    "stock_min_dollar_volume": 5000000.0,
    "stock_max_spread_bps": 40.0,
    "stock_gate_market_hours_scan": True,
    "stock_min_bars_required": 24,
    "stock_min_valid_bars_ratio": 0.7,
    "stock_max_stale_hours": 6.0,
    "stock_scan_open_cooldown_minutes": 15,
    "stock_scan_close_cooldown_minutes": 15,
    "stock_scan_open_score_mult": 0.85,
    "stock_scan_close_score_mult": 0.90,
    "stock_scan_publish_watch_leaders": True,
    "stock_scan_watch_leaders_count": 6,
    "stock_opening_plan_enabled": True,
    "stock_opening_plan_minutes": 45,
    "stock_opening_plan_max_symbols": 8,
    "stock_leader_stability_margin_pct": 10.0,
    "stock_show_rejected_rows": False,
    "stock_auto_trade_enabled": False,
    "stock_block_entries_on_cached_scan": True,
    "stock_cached_scan_hard_block_age_s": 1800,
    "stock_cached_scan_entry_size_mult": 0.60,
    "stock_require_data_quality_ok_for_entries": True,
    "stock_require_reject_rate_max_pct": 92.0,
    "stock_trade_notional_usd": 100.0,
    "stock_max_open_positions": 1,
    "stock_score_threshold": 0.2,
    "stock_replay_adaptive_enabled": True,
    "stock_replay_adaptive_weight": 0.35,
    "stock_replay_adaptive_step_cap_pct": 40.0,
    "stock_profit_target_pct": 0.35,
    "stock_trailing_gap_pct": 0.2,
    "stock_max_day_trades": 3,
    "stock_min_hold_minutes": 1440,
    "stock_same_day_exit_exception_enabled": True,
    "stock_same_day_exception_min_hold_minutes": 120,
    "stock_same_day_exception_min_pnl_pct": 2.5,
    "stock_same_day_exception_min_pullback_pct": 0.9,
    "stock_same_day_exception_require_score_flip": True,
    "stock_same_day_exception_score_floor_mult": 0.75,
    "stock_pdt_equity_threshold_usd": 25_000.0,
    "stock_pdt_max_day_trades_rolling_5d": 3,
    "stock_max_position_usd_per_symbol": 0.0,
    "stock_max_total_exposure_pct": 0.0,
    "stock_block_new_entries_near_close": True,
    "stock_no_new_entries_mins_to_close": 15,
    "stock_live_guarded_score_mult": 1.2,
    "stock_min_calib_prob_live_guarded": 0.58,
    "stock_max_slippage_bps": 35.0,
    "stock_order_retry_count": 2,
    "stock_max_loss_streak": 3,
    "stock_loss_streak_size_step_pct": 0.15,
    "stock_loss_streak_size_floor_pct": 0.40,
    "stock_loss_cooldown_seconds": 1800,
    "stock_max_daily_loss_usd": 0.0,
    "stock_max_daily_loss_pct": 0.0,
    "stock_min_samples_live_guarded": 5,
    "stock_max_signal_age_seconds": 300,
    "stock_reject_drift_warn_pct": 65.0,
    "stock_symbol_cooldown_minutes": 15,
    "stock_symbol_cooldown_min_hits": 3,
    "stock_symbol_cooldown_reject_reasons": "data_quality,insufficient_bars",
    "oanda_account_id": "",
    "oanda_api_token": "",
    "oanda_rest_url": "https://api-fxtrade.oanda.com",
    "oanda_stream_url": "https://stream-fxtrade.oanda.com",
    "oanda_practice_mode": False,
    "forex_auto_trade_enabled": False,
    "forex_universe_pairs": "",
    "forex_scan_max_pairs": 32,
    "forex_max_spread_bps": 8.0,
    "forex_min_volatility_pct": 0.01,
    "forex_min_bars_required": 24,
    "forex_min_valid_bars_ratio": 0.7,
    "forex_max_stale_hours": 8.0,
    "forex_session_weight_enabled": True,
    "forex_session_weight_floor": 0.85,
    "forex_session_weight_ceiling": 1.10,
    "forex_leader_stability_margin_pct": 12.0,
    "forex_show_rejected_rows": False,
    "forex_trade_units": 1000,
    "forex_block_entries_on_cached_scan": True,
    "forex_cached_scan_hard_block_age_s": 1200,
    "forex_cached_scan_entry_size_mult": 0.65,
    "forex_require_data_quality_ok_for_entries": True,
    "forex_require_reject_rate_max_pct": 92.0,
    "forex_max_open_positions": 1,
    "forex_max_position_usd_per_pair": 0.0,
    "forex_score_threshold": 0.2,
    "forex_replay_adaptive_enabled": True,
    "forex_replay_adaptive_weight": 0.35,
    "forex_replay_adaptive_step_cap_pct": 40.0,
    "forex_profit_target_pct": 0.25,
    "forex_trailing_gap_pct": 0.15,
    "forex_max_total_exposure_pct": 0.0,
    "forex_session_mode": "all",  # all | london_ny | london | ny | asia
    "forex_live_guarded_score_mult": 1.15,
    "forex_min_calib_prob_live_guarded": 0.56,
    "forex_max_slippage_bps": 6.0,
    "forex_order_retry_count": 2,
    "forex_max_loss_streak": 3,
    "forex_loss_streak_size_step_pct": 0.15,
    "forex_loss_streak_size_floor_pct": 0.40,
    "forex_loss_cooldown_seconds": 1800,
    "forex_max_daily_loss_usd": 0.0,
    "forex_max_daily_loss_pct": 0.0,
    "forex_min_samples_live_guarded": 5,
    "forex_max_signal_age_seconds": 300,
    "forex_reject_drift_warn_pct": 65.0,
    "market_max_total_exposure_pct": 0.0,
    "market_independent_execution_enabled": False,
    "market_bg_stocks_interval_s": 15.0,
    "market_bg_forex_interval_s": 10.0,
    "market_intelligence_interval_s": 180.0,
    "stock_trader_step_interval_s": 18.0,
    "forex_trader_step_interval_s": 12.0,
    "runner_crash_lockout_s": 180.0,
    "runtime_api_quota_warn_15m": 4,
    "runtime_api_quota_crit_15m": 10,
    "runtime_alert_cadence_warn_count": 1,
    "runtime_alert_cadence_crit_count": 2,
    "runtime_alert_cadence_late_warn_pct": 80.0,
    "runtime_alert_cadence_late_crit_pct": 180.0,
    "runtime_alert_cadence_min_samples": 3,
    "runtime_alert_cadence_cooldown_s": 300,
    "runtime_alert_market_loop_stale_s": 90.0,
    "runtime_incidents_max_lines": 25000,
    "runtime_events_max_lines": 50000,
    "broker_failure_disable_threshold": 4,
    "broker_failure_disable_cooldown_s": 900,
    "broker_order_retry_after_cap_s": 300.0,
    "adaptive_confidence_min_samples": 18,
    "adaptive_confidence_target_success_pct": 55.0,
    "replay_target_entries_crypto": 5,
    "replay_target_entries_stocks": 3,
    "replay_target_entries_forex": 4,
    "operator_notes_max_entries": 120,
    "market_loop_jitter_pct": 0.10,
    "market_settings_reload_interval_s": 8.0,
    "paper_only_unless_checklist_green": True,
    "key_rotation_warn_days": 0,
    "data_cache_max_age_days": 14.0,
    "scanner_quality_max_age_days": 14.0,
    "data_cache_max_total_mb": 300,
    "global_max_drawdown_pct": 0.0,
    "global_drawdown_lookback_hours": 24,
    "global_drawdown_auto_resume_enabled": True,
    "global_drawdown_resume_cooloff_s": 14400,
    "global_drawdown_resume_recovery_buffer_pct": 0.25,
    "global_drawdown_require_manual_ack": True,
    "equity_curve_anomaly_spike_pct": 3.0,
    "equity_curve_stale_after_s": 600,
    "auto_start_scripts": False,
}

_READ_INT_FILE_CACHE: Dict[str, Tuple[float, int]] = {}
_READ_JSON_CACHE: Dict[str, Tuple[Tuple[int, int], Any]] = {}
_TRADE_HISTORY_CACHE: Dict[str, Tuple[Tuple[int, int], List[dict]]] = {}











SETTINGS_FILE = "gui_settings.json"


def _safe_read_json(path: str) -> Optional[dict]:
    def _clone_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            return list(payload)
        return payload

    key = os.path.abspath(str(path or ""))
    try:
        st = os.stat(path)
        sig = (int(getattr(st, "st_mtime_ns", 0) or 0), int(getattr(st, "st_size", 0) or 0))
    except Exception:
        sig = None
    if sig is not None:
        cached = _READ_JSON_CACHE.get(key)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == sig:
            return _clone_payload(cached[1])
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = fast_json_load(f)
        if sig is not None:
            _READ_JSON_CACHE[key] = (sig, payload)
            if len(_READ_JSON_CACHE) > 384:
                try:
                    # Keep cache bounded; clear stale bulk safely.
                    for _drop_key in list(_READ_JSON_CACHE.keys())[: max(1, len(_READ_JSON_CACHE) - 320)]:
                        _READ_JSON_CACHE.pop(_drop_key, None)
                except Exception:
                    pass
        return _clone_payload(payload)
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError, ValueError) as exc:
        log_once(
            f"pt_hub:_safe_read_json:{path}:{type(exc).__name__}",
            f"[pt_hub._safe_read_json] path={path} {type(exc).__name__}: {exc}",
        )
        return None


def _safe_write_json(path: str, data: dict) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            fast_json_dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except (PermissionError, OSError, TypeError, ValueError) as exc:
        log_once(
            f"pt_hub:_safe_write_json:{path}:{type(exc).__name__}",
            f"[pt_hub._safe_write_json] path={path} {type(exc).__name__}: {exc}",
        )


def _read_trade_history_jsonl(path: str) -> List[dict]:
    """
    Reads hub_data/trade_history.jsonl written by pt_trader.py.
    Returns a list of dicts (only buy/sell rows).
    """
    out: List[dict] = []
    key = os.path.abspath(str(path or ""))
    try:
        st = os.stat(path)
        sig = (int(getattr(st, "st_mtime_ns", 0) or 0), int(getattr(st, "st_size", 0) or 0))
    except Exception:
        sig = None
    if sig is not None:
        cached = _TRADE_HISTORY_CACHE.get(key)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == sig:
            return list(cached[1])
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = fast_json_loads(ln, default=None)
                        side = str(obj.get("side", "")).lower().strip()
                        if side not in ("buy", "sell"):
                            continue
                        out.append(obj)
                    except Exception:
                        continue
    except Exception:
        pass
    if sig is not None:
        _TRADE_HISTORY_CACHE[key] = (sig, list(out))
        if len(_TRADE_HISTORY_CACHE) > 64:
            try:
                for _drop_key in list(_TRADE_HISTORY_CACHE.keys())[: max(1, len(_TRADE_HISTORY_CACHE) - 48)]:
                    _TRADE_HISTORY_CACHE.pop(_drop_key, None)
            except Exception:
                pass
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _fmt_money(x: float) -> str:
    """Format a USD *amount* (account value, position value, etc.) as dollars with 2 decimals."""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"


def _fmt_price(x: Any) -> str:
    """
    Format a USD *price/level* with dynamic decimals based on magnitude.
    Examples:
      50234.12   -> $50,234.12
      123.4567   -> $123.457
      1.234567   -> $1.2346
      0.06234567 -> $0.062346
      0.00012345 -> $0.00012345
    """
    try:
        if x is None:
            return "N/A"

        v = float(x)
        if not math.isfinite(v):
            return "N/A"

        sign = "-" if v < 0 else ""
        av = abs(v)

        # Choose decimals by magnitude (more detail for smaller prices).
        if av >= 1000:
            dec = 2
        elif av >= 100:
            dec = 3
        elif av >= 1:
            dec = 4
        elif av >= 0.1:
            dec = 5
        elif av >= 0.01:
            dec = 6
        elif av >= 0.001:
            dec = 7
        else:
            dec = 8

        s = f"{av:,.{dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")

        return f"{sign}${s}"
    except Exception:
        return "N/A"


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):+.2f}%"
    except Exception:
        return "N/A"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def _effective_trade_pnl_pct(row: Dict[str, Any]) -> Optional[float]:
    side = str(row.get("side", "") or "").strip().upper()
    raw_pct = _float_or_none(row.get("pnl_pct", None))
    if side != "SELL":
        return raw_pct

    realized = _float_or_none(row.get("realized_profit_usd", None))
    cost_used = _float_or_none(row.get("position_cost_used_usd", None))
    avg_cost = _float_or_none(row.get("avg_cost_basis", None))
    qty = _float_or_none(row.get("qty", None))
    if qty is not None:
        qty = abs(qty)

    cost_for_pct = cost_used if (cost_used is not None and cost_used > 0.0) else None
    if cost_for_pct is None and avg_cost is not None and qty is not None:
        est_cost = float(avg_cost) * float(qty)
        if est_cost > 0.0:
            cost_for_pct = est_cost

    if realized is not None and cost_for_pct is not None and cost_for_pct > 0.0:
        return (float(realized) / float(cost_for_pct)) * 100.0
    return raw_pct


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Neural folder detection
# -----------------------------

def build_coin_folders(main_dir: str, coins: List[str]) -> Dict[str, str]:
    """
    Coin folder layout:
      every coin (including BTC) uses <main_dir>/<COIN>

    Returns { "BTC": "...", "ETH": "...", ... }
    """
    out: Dict[str, str] = {}
    main_dir = main_dir or BASE_DIR

    for c in coins:
        c = c.upper().strip()
        if not c:
            continue
        p = os.path.join(main_dir, c)
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass
        out[c] = p

    if "BTC" not in out:
        btc_dir = os.path.join(main_dir, "BTC")
        try:
            os.makedirs(btc_dir, exist_ok=True)
        except Exception:
            pass
        out["BTC"] = btc_dir

    return out


def read_price_levels_from_html(path: str) -> List[float]:
    """
    pt_thinker writes a python-list-like string into low_bound_prices.html / high_bound_prices.html.

    Example (commas often remain):
        "43210.1, 43100.0, 42950.5"

    So we normalize separators before parsing.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return []

        # Normalize common separators that pt_thinker can leave behind
        raw = (
            raw.replace(",", " ")
               .replace("[", " ")
               .replace("]", " ")
               .replace("'", " ")
        )

        vals: List[float] = []
        for tok in raw.split():
            try:
                v = float(tok)

                # Filter obvious sentinel values used by pt_thinker for "inactive" slots
                if v <= 0:
                    continue
                if v >= 9e15:  # pt_thinker uses 99999999999999999
                    continue
                if abs(v - 0.01) <= 1e-9:  # pt_thinker low-side inactive placeholder
                    continue


                vals.append(v)
            except Exception:
                pass

        # De-dupe while preserving order (small rounding to avoid float-noise duplicates)
        out: List[float] = []
        seen = set()
        for v in vals:
            key = round(v, 12)
            if key in seen:
                continue
            seen.add(key)
            out.append(v)

        return out
    except Exception:
        return []



def read_int_from_file(path: str) -> int:
    try:
        mtime = os.path.getmtime(path)
    except (FileNotFoundError, PermissionError, OSError):
        return 0
    hit = _READ_INT_FILE_CACHE.get(path)
    if hit and hit[0] == mtime:
        return int(hit[1])
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        val = int(float(raw))
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        log_once(
            f"pt_hub:read_int_from_file:{path}:{type(exc).__name__}",
            f"[pt_hub.read_int_from_file] path={path} {type(exc).__name__}: {exc}",
        )
        val = 0
    _READ_INT_FILE_CACHE[path] = (mtime, val)
    return val


def read_short_signal(folder: str) -> int:
    txt = os.path.join(folder, "short_dca_signal.txt")
    if os.path.isfile(txt):
        return read_int_from_file(txt)
    else:
        return 0


# -----------------------------
# Candle fetching (KuCoin)
# -----------------------------

class CandleFetcher:
    """
    Uses kucoin-python if available; otherwise falls back to KuCoin REST via requests.
    """
    def __init__(self):
        self._mode = "kucoin_client"
        self._market = None
        try:
            from kucoin.client import Market  # type: ignore
            self._market = Market(url="https://api.kucoin.com")
        except Exception:
            self._mode = "rest"
            self._market = None

        if self._mode == "rest":
            import requests  # local import
            self._requests = requests

        # Small in-memory cache to keep timeframe switching snappy.
        # key: (pair, timeframe, limit) -> (saved_time_epoch, candles)
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[dict]]] = {}
        self._last_error: Dict[Tuple[str, str, int], Tuple[float, str]] = {}
        self._cache_ttl_seconds: float = 10.0
        self._lock = threading.Lock()
        self._pending: set[Tuple[str, str, int]] = set()
        self._result_q: "queue.Queue[Tuple[Tuple[str, str, int], float, List[dict], str]]" = queue.Queue()


    def _fetch_klines_sync(self, pair: str, timeframe: str, limit: int, now: float) -> Tuple[List[dict], str]:
        """
        Returns candles oldest->newest as:
          [{"ts": int, "open": float, "high": float, "low": float, "close": float}, ...]
        """
        limit = int(limit or 0)

        # rough window (timeframe-dependent) so we get enough candles
        tf_seconds = {
            "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
            "1hour": 3600, "2hour": 7200, "4hour": 14400, "8hour": 28800, "12hour": 43200,
            "1day": 86400, "1week": 604800
        }.get(timeframe, 3600)

        end_at = int(now)
        start_at = end_at - (tf_seconds * max(200, (limit + 50) if limit else 250))

        if self._mode == "kucoin_client" and self._market is not None:
            try:
                # IMPORTANT: limit the server response by passing startAt/endAt.
                # This avoids downloading a huge default kline set every switch.
                try:
                    raw = self._market.get_kline(pair, timeframe, startAt=start_at, endAt=end_at)  # type: ignore
                except Exception:
                    # fallback if that client version doesn't accept kwargs
                    raw = self._market.get_kline(pair, timeframe)  # returns newest->oldest

                candles: List[dict] = []
                for row in raw:
                    # KuCoin kline row format:
                    # [time, open, close, high, low, volume, turnover]
                    ts = int(float(row[0]))
                    o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                    candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
                candles.sort(key=lambda x: x["ts"])
                if limit and len(candles) > limit:
                    candles = candles[-limit:]
                return candles, ""
            except Exception as exc:
                return [], f"kucoin client: {type(exc).__name__}"

        # REST fallback
        last_err = "unknown error"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                url = "https://api.kucoin.com/api/v1/market/candles"
                params = {"symbol": pair, "type": timeframe, "startAt": start_at, "endAt": end_at}
                resp = self._requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                j = resp.json()
                if isinstance(j, dict):
                    code = str(j.get("code", "") or "").strip()
                    if code and code != "200000":
                        msg = str(j.get("msg", "") or "").strip()
                        raise RuntimeError(f"KuCoin error {code}: {msg}")
                data = j.get("data", []) if isinstance(j, dict) else []  # newest->oldest
                candles: List[dict] = []
                for row in data:
                    ts = int(float(row[0]))
                    o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                    candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
                candles.sort(key=lambda x: x["ts"])
                if limit and len(candles) > limit:
                    candles = candles[-limit:]
                if candles:
                    return candles, ""
                last_err = "empty candle payload"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(0.35 * attempt)
        return [], f"rest fetch failed: {last_err}"


    def _start_fetch(self, cache_key: Tuple[str, str, int]) -> None:
        with self._lock:
            if cache_key in self._pending:
                return
            self._pending.add(cache_key)

        def _worker() -> None:
            pair, timeframe, limit = cache_key
            now = time.time()
            candles, err = self._fetch_klines_sync(pair, timeframe, limit, now)
            try:
                self._result_q.put((cache_key, now, candles, err))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()


    def drain_results(self) -> bool:
        changed = False
        while True:
            try:
                cache_key, now, candles, err = self._result_q.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending.discard(cache_key)
                if candles:
                    self._cache[cache_key] = (now, candles)
                    self._last_error.pop(cache_key, None)
                    changed = True
                elif err:
                    self._last_error[cache_key] = (now, str(err)[:220])
        return changed


    def get_klines(self, symbol: str, timeframe: str, limit: int = 120) -> List[dict]:
        symbol = symbol.upper().strip()
        pair = f"{symbol}-USDT"
        limit = int(limit or 0)
        now = time.time()
        cache_key = (pair, timeframe, limit)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached and (now - float(cached[0])) <= float(self._cache_ttl_seconds):
            return cached[1]

        self._start_fetch(cache_key)
        if cached:
            return cached[1]
        return []

    def get_last_error(self, symbol: str, timeframe: str, limit: int = 120, max_age_s: float = 180.0) -> str:
        symbol = symbol.upper().strip()
        pair = f"{symbol}-USDT"
        cache_key = (pair, timeframe, int(limit or 0))
        now = time.time()
        with self._lock:
            row = self._last_error.get(cache_key)
        if not row:
            return ""
        ts, msg = row
        if (now - float(ts)) > float(max_age_s):
            return ""
        return str(msg or "")



# -----------------------------
# Chart widget
# -----------------------------

class CandleChart(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        fetcher: CandleFetcher,
        coin: str,
        settings_getter,
        trade_history_path: str,
    ):
        super().__init__(parent)
        self.fetcher = fetcher
        self.coin = coin
        self.settings_getter = settings_getter
        self.trade_history_path = trade_history_path

        self.timeframe_var = tk.StringVar(value=self.settings_getter()["default_timeframe"])


        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(4, 4))

        controls_row = ttk.Frame(top)
        controls_row.pack(fill="x")

        status_row = ttk.Frame(top)
        status_row.pack(fill="x", pady=(2, 0))

        ttk.Label(controls_row, text=f"{coin} chart").pack(side="left")

        display_controls = ttk.Frame(controls_row)
        display_controls.pack(side="left", padx=(10, 0))

        ttk.Label(display_controls, text="Timeframe:").pack(side="left", padx=(0, 4))
        self.tf_combo = ttk.Combobox(
            display_controls,
            textvariable=self.timeframe_var,
            values=self.settings_getter()["timeframes"],
            state="readonly",
            width=10,
        )
        self.tf_combo.pack(side="left")

        # Debounce rapid timeframe changes so redraws don't stack
        self._tf_after_id = None

        def _debounced_tf_change(*_):
            try:
                if self._tf_after_id:
                    self.after_cancel(self._tf_after_id)
            except Exception:
                pass

            def _do():
                # Ask the hub to refresh charts on the next tick (single refresh)
                try:
                    self.event_generate("<<TimeframeChanged>>", when="tail")
                except Exception:
                    pass

            self._tf_after_id = self.after(120, _do)

        self.tf_combo.bind("<<ComboboxSelected>>", _debounced_tf_change)

        self.detailed_overlays_var = tk.BooleanVar(value=False)
        self.detailed_overlays_chk = ttk.Checkbutton(
            display_controls,
            text="Detailed overlays",
            variable=self.detailed_overlays_var,
            command=lambda: self.event_generate("<<TimeframeChanged>>", when="tail"),
        )
        self.detailed_overlays_chk.pack(side="left", padx=(10, 0))


        self.neural_status_label = ttk.Label(status_row, text="Neural: N/A")
        self.neural_status_label.pack(side="left")

        self.chart_key_label = ttk.Label(status_row, text="Key: ★ Trail  ◆ DCA  ● Avg")
        self.chart_key_label.pack(side="left", padx=(12, 0))
        self.outlook_status_label = ttk.Label(status_row, text="1h Outlook: waiting for model data...")
        self.outlook_status_label.pack(side="left", padx=(12, 0))

        self.last_update_label = ttk.Label(status_row, text="Last: N/A")
        self.last_update_label.pack(side="right")

        # Structured legend payload consumed by the crypto left-panel legend renderer.
        self._legend_rows: List[Dict[str, Any]] = []
        self._legend_note: str = ""

        # Figure
        # IMPORTANT: keep a stable DPI and resize the figure to the widget's pixel size.
        # On Windows scaling, trying to "sync DPI" via winfo_fpixels("1i") can produce the
        # exact right-side blank/covered region you're seeing.
        self.fig = Figure(figsize=(6.5, 3.5), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)

        # Keep a small margin for the title and two-line x-axis labels, but otherwise
        # let the plot use as much of the canvas as possible.
        self.fig.subplots_adjust(left=0.05, bottom=0.12, right=0.982, top=0.89)

        self.ax = self.fig.add_subplot(111)
        self._apply_dark_chart_style()
        self.ax.set_title(f"{coin}", color=DARK_FG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas_w = self.canvas.get_tk_widget()
        canvas_w.configure(bg=DARK_BG)

        # Remove horizontal padding here so the chart widget truly fills the container.
        canvas_w.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        # Keep the matplotlib figure EXACTLY the same pixel size as the Tk widget.
        # FigureCanvasTkAgg already sizes its backing PhotoImage to e.width/e.height.
        # Multiplying by tk scaling here makes the renderer larger than the PhotoImage,
        # which produces the "blank/covered strip" on the right.
        self._last_canvas_px = (0, 0)
        self._resize_after_id = None

        def _on_canvas_configure(e):
            try:
                w = int(e.width)
                h = int(e.height)
                if w <= 1 or h <= 1:
                    return

                if (w, h) == self._last_canvas_px:
                    return
                self._last_canvas_px = (w, h)

                dpi = float(self.fig.get_dpi() or 100.0)
                self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

                # Debounce redraws during live resize
                if self._resize_after_id:
                    try:
                        self.after_cancel(self._resize_after_id)
                    except Exception:
                        pass
                self._resize_after_id = self.after_idle(self.canvas.draw_idle)
            except Exception:
                pass

        canvas_w.bind("<Configure>", _on_canvas_configure, add="+")







        self._last_refresh = 0.0
        self._legend_hover_motion_handler = None
        self._legend_hover_last_canvas_xy: Optional[Tuple[float, float]] = None
        self._legend_hover_restore_after_id = None
        self._legend_bbox_after_id = None

        def _hide_tip_event(_e=None):
            try:
                self.hide_legend_tooltip(clear_pointer=False)
            except Exception:
                pass

        try:
            canvas_w.bind("<Leave>", _hide_tip_event, add="+")
            canvas_w.bind("<Unmap>", _hide_tip_event, add="+")
            canvas_w.bind("<Destroy>", _hide_tip_event, add="+")
            top = canvas_w.winfo_toplevel()
            if top is not None:
                top.bind("<Unmap>", _hide_tip_event, add="+")
                top.bind("<FocusOut>", _hide_tip_event, add="+")
        except Exception:
            pass


    def _apply_dark_chart_style(self) -> None:
        """Apply dark styling (called on init and after every ax.clear())."""
        try:
            self.fig.patch.set_facecolor(CHART_BG)
            self.ax.set_facecolor(CHART_PANEL)
            self.ax.tick_params(colors=CHART_TEXT)
            for spine in self.ax.spines.values():
                spine.set_color(CHART_BORDER)
            self.ax.grid(True, color=CHART_GRID, linewidth=0.65, alpha=0.45, linestyle="--")
        except Exception:
            pass

    def _coin_symbol_aliases(self) -> Set[str]:
        coin = str(self.coin or "").strip().upper()
        if not coin:
            return set()
        aliases: Set[str] = {
            coin,
            coin.replace("-", "").replace("_", "").replace("/", ""),
            f"{coin}USD",
            f"{coin}-USD",
            f"{coin}_USD",
            f"{coin}/USD",
        }
        return {str(a).strip().upper() for a in aliases if str(a).strip()}

    def _openai_position_guidance(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            hub_dir = os.path.dirname(os.path.abspath(str(self.trade_history_path or "")))
            openai_dir = os.path.join(hub_dir, "openai")
            aliases = self._coin_symbol_aliases()
            if not aliases:
                return out

            def _match_symbol(raw_symbol: Any) -> bool:
                sym = str(raw_symbol or "").strip().upper()
                if not sym:
                    return False
                compact = sym.replace("-", "").replace("_", "").replace("/", "")
                return (sym in aliases) or (compact in aliases)

            sources: List[Tuple[str, List[Dict[str, Any]], int]] = []
            pos_review = _safe_read_json(os.path.join(openai_dir, "position_review.json")) or {}
            rows = pos_review.get("position_actions", []) if isinstance(pos_review.get("position_actions", []), list) else []
            ts_pos = int(float(pos_review.get("ts", 0) or 0))
            sources.append(("position_review", [r for r in rows if isinstance(r, dict)], ts_pos))

            port_last = _safe_read_json(os.path.join(openai_dir, "portfolio_decision_last_result.json")) or {}
            decision = port_last.get("decision", {}) if isinstance(port_last.get("decision", {}), dict) else {}
            rows2 = decision.get("position_actions", []) if isinstance(decision.get("position_actions", []), list) else []
            ts_port = int(float(port_last.get("ts", 0) or 0))
            sources.append(("portfolio_decision", [r for r in rows2 if isinstance(r, dict)], ts_port))

            sources.sort(key=lambda item: (int(item[2]), 1 if item[0] == "position_review" else 0), reverse=True)
            for source_name, source_rows, source_ts in sources:
                for row in source_rows:
                    market = str(row.get("market", "") or "").strip().lower()
                    if market and market != "crypto":
                        continue
                    if not _match_symbol(row.get("symbol", "")):
                        continue
                    action = str(row.get("action", "") or "").strip().lower()
                    if not action:
                        continue
                    conf = _float_or_none(row.get("confidence", None))
                    out = {
                        "source": source_name,
                        "ts": int(source_ts),
                        "action": action,
                        "confidence": float(conf) if conf is not None else None,
                        "reason": str(row.get("reason", "") or "").strip(),
                    }
                    return out
        except Exception:
            return {}
        return out

    def _local_model_outlook_1h(
        self,
        *,
        long_levels: List[float],
        short_levels: List[float],
        long_sig: int,
        short_sig: int,
        current_buy_price: Optional[float],
        current_sell_price: Optional[float],
        last_close_price: Optional[float],
    ) -> Dict[str, Any]:
        cfg = self.settings_getter() if callable(self.settings_getter) else {}
        try:
            start_level = max(1, min(7, int(float(cfg.get("trade_start_level", 3) or 3))))
        except Exception:
            start_level = 3

        ref = _float_or_none(current_sell_price)
        if ref is None or ref <= 0.0:
            ref = _float_or_none(current_buy_price)
        if (ref is None or ref <= 0.0) and (last_close_price is not None):
            ref = _float_or_none(last_close_price)
        if ref is None or ref <= 0.0:
            return {
                "bias": "unclear",
                "target_up": None,
                "target_down": None,
                "up_pct": None,
                "down_pct": None,
                "confidence": 0.0,
                "reason": "missing reference price",
            }

        def _clean_levels(values: List[float]) -> List[float]:
            out_vals: List[float] = []
            for raw in list(values or []):
                try:
                    vv = float(raw)
                except Exception:
                    continue
                if (not math.isfinite(vv)) or vv <= 0.0:
                    continue
                if vv < (ref * 0.40) or vv > (ref * 2.50):
                    continue
                out_vals.append(vv)
            return sorted(out_vals)

        lows = _clean_levels(long_levels)
        highs = _clean_levels(short_levels)

        target_up = None
        if highs:
            ups = [v for v in highs if v >= ref]
            target_up = min(ups) if ups else max(highs)
        target_down = None
        if lows:
            dns = [v for v in lows if v <= ref]
            target_down = max(dns) if dns else min(lows)

        up_pct = ((float(target_up) / ref) - 1.0) * 100.0 if target_up and ref > 0.0 else None
        down_pct = ((float(target_down) / ref) - 1.0) * 100.0 if target_down and ref > 0.0 else None

        if short_sig >= max(1, start_level - 1) and short_sig > long_sig:
            bias = "bearish"
        elif long_sig >= start_level and short_sig == 0:
            bias = "bullish"
        elif long_sig > short_sig:
            bias = "mild_bullish"
        elif short_sig > long_sig:
            bias = "mild_bearish"
        else:
            bias = "mixed"

        signal_edge = abs(int(long_sig or 0) - int(short_sig or 0))
        conf = 0.40 + (0.08 * float(min(4, signal_edge)))
        if bias in {"bullish", "bearish"}:
            conf += 0.15
        if up_pct is None and down_pct is None:
            conf -= 0.20
        conf = max(0.05, min(0.95, conf))

        return {
            "bias": bias,
            "target_up": float(target_up) if target_up is not None else None,
            "target_down": float(target_down) if target_down is not None else None,
            "up_pct": float(up_pct) if up_pct is not None and math.isfinite(up_pct) else None,
            "down_pct": float(down_pct) if down_pct is not None and math.isfinite(down_pct) else None,
            "confidence": float(conf),
            "reason": f"signals L{int(long_sig)} / S{int(short_sig)}",
            "ref_price": float(ref),
        }

    def _timeframe_seconds(self, timeframe: str) -> int:
        raw = str(timeframe or "").strip().lower()
        tf_map = {
            "1min": 60,
            "5min": 300,
            "15min": 900,
            "30min": 1800,
            "1hour": 3600,
            "2hour": 7200,
            "4hour": 14400,
            "8hour": 28800,
            "12hour": 43200,
            "1day": 86400,
            "1week": 604800,
        }
        if raw in tf_map:
            return int(tf_map[raw])
        m = re.match(r"^\s*(\d+)\s*(min|m|hour|h|day|d|week|w)\s*$", raw)
        if not m:
            return 3600
        qty = max(1, int(m.group(1)))
        unit = str(m.group(2)).lower()
        if unit in {"min", "m"}:
            return qty * 60
        if unit in {"hour", "h"}:
            return qty * 3600
        if unit in {"day", "d"}:
            return qty * 86400
        if unit in {"week", "w"}:
            return qty * 604800
        return 3600

    def _build_next_hour_projection(
        self,
        *,
        timeframe: str,
        candle_count: int,
        last_candle_ts: Optional[int],
        local_outlook: Dict[str, Any],
        fallback_price: Optional[float],
    ) -> Dict[str, Any]:
        try:
            n = max(0, int(candle_count or 0))
            if n <= 0:
                return {}
            tf_seconds = max(60, int(self._timeframe_seconds(timeframe)))
            horizon_seconds = 3600
            horizon_bars = max(1, int(math.ceil(float(horizon_seconds) / float(tf_seconds))))
            max_display_span = max(2.0, min(16.0, float(max(2, int(round(n * 0.22))))))
            display_span = float(max(1.0, min(float(horizon_bars), max_display_span)))

            bias = str(local_outlook.get("bias", "") or "").strip().lower()
            ref_price = _float_or_none(local_outlook.get("ref_price", None))
            if ref_price is None or ref_price <= 0.0:
                ref_price = _float_or_none(fallback_price)
            if ref_price is None or ref_price <= 0.0:
                return {}

            target_up = _float_or_none(local_outlook.get("target_up", None))
            target_down = _float_or_none(local_outlook.get("target_down", None))
            up_pct = _float_or_none(local_outlook.get("up_pct", None))
            down_pct = _float_or_none(local_outlook.get("down_pct", None))
            confidence = _float_or_none(local_outlook.get("confidence", None))
            conf = max(0.0, min(1.0, float(confidence if confidence is not None else 0.50)))

            if bias in {"bullish", "mild_bullish"}:
                target_price = target_up if target_up and target_up > 0.0 else ref_price
                low_est = target_down if target_down and target_down > 0.0 else (ref_price * (1.0 - max(0.0015, abs(float(up_pct or 0.0)) / 250.0)))
                high_est = max(float(target_price), float(ref_price))
            elif bias in {"bearish", "mild_bearish"}:
                target_price = target_down if target_down and target_down > 0.0 else ref_price
                high_est = target_up if target_up and target_up > 0.0 else (ref_price * (1.0 + max(0.0015, abs(float(down_pct or 0.0)) / 250.0)))
                low_est = min(float(target_price), float(ref_price))
            else:
                target_price = ref_price
                drift = max(0.001, (1.0 - conf) * 0.006)
                low_est = ref_price * (1.0 - drift)
                high_est = ref_price * (1.0 + drift)

            if not math.isfinite(float(target_price)) or float(target_price) <= 0.0:
                return {}

            zone_start_x = float(n) - 0.5
            zone_end_x = zone_start_x + float(display_span)
            if zone_end_x <= zone_start_x:
                zone_end_x = zone_start_x + 1.0

            future_label = ""
            try:
                if last_candle_ts is not None:
                    future_ts = int(last_candle_ts) + int(horizon_seconds)
                    future_label = time.strftime("+1h\n%m-%d %H:%M", time.localtime(future_ts))
            except Exception:
                future_label = "+1h"

            return {
                "bias": bias,
                "confidence": conf,
                "ref_price": float(ref_price),
                "target_price": float(target_price),
                "low_est": float(min(low_est, high_est)),
                "high_est": float(max(low_est, high_est)),
                "horizon_seconds": int(horizon_seconds),
                "horizon_bars": int(horizon_bars),
                "display_span": float(display_span),
                "zone_start_x": float(zone_start_x),
                "zone_end_x": float(zone_end_x),
                "future_tick_label": str(future_label or "+1h"),
            }
        except Exception:
            return {}

    def _update_outlook_status(
        self,
        *,
        long_levels: List[float],
        short_levels: List[float],
        long_sig: int,
        short_sig: int,
        current_buy_price: Optional[float],
        current_sell_price: Optional[float],
        last_close_price: Optional[float],
        avg_cost_basis: Optional[float],
        quantity: Optional[float],
    ) -> None:
        try:
            local = self._local_model_outlook_1h(
                long_levels=list(long_levels or []),
                short_levels=list(short_levels or []),
                long_sig=int(long_sig or 0),
                short_sig=int(short_sig or 0),
                current_buy_price=current_buy_price,
                current_sell_price=current_sell_price,
                last_close_price=last_close_price,
            )
            try:
                self._last_local_outlook = dict(local)
            except Exception:
                self._last_local_outlook = {}
            bias = str(local.get("bias", "unclear") or "unclear").strip().lower()
            up_pct = _float_or_none(local.get("up_pct", None))
            down_pct = _float_or_none(local.get("down_pct", None))
            target_up = _float_or_none(local.get("target_up", None))
            target_down = _float_or_none(local.get("target_down", None))
            model_conf = _float_or_none(local.get("confidence", None))

            if bias == "bullish":
                local_txt = "MODEL bullish"
                if up_pct is not None and target_up is not None and abs(up_pct) <= 250.0:
                    local_txt += f" {up_pct:+.2f}% to {_fmt_price(target_up)}"
            elif bias == "bearish":
                local_txt = "MODEL bearish"
                if down_pct is not None and target_down is not None and abs(down_pct) <= 250.0:
                    local_txt += f" {down_pct:+.2f}% to {_fmt_price(target_down)}"
            elif bias == "mild_bullish":
                local_txt = "MODEL mild bullish"
                if up_pct is not None and abs(up_pct) <= 250.0:
                    local_txt += f" ({up_pct:+.2f}%)"
            elif bias == "mild_bearish":
                local_txt = "MODEL mild bearish"
                if down_pct is not None and abs(down_pct) <= 250.0:
                    local_txt += f" ({down_pct:+.2f}%)"
            else:
                local_txt = "MODEL mixed/unclear"
            if model_conf is not None:
                local_txt += f" c={max(0.0, min(1.0, model_conf)):.2f}"

            ai = self._openai_position_guidance()
            try:
                self._last_ai_position_guidance = dict(ai)
            except Exception:
                self._last_ai_position_guidance = {}
            ai_action = str(ai.get("action", "") or "").strip().lower()
            ai_conf = _float_or_none(ai.get("confidence", None))
            if ai_action:
                ai_txt = f"AI {ai_action.upper()}"
                if ai_conf is not None:
                    ai_txt += f" {max(0.0, min(1.0, ai_conf)):.2f}"
            else:
                ai_txt = "AI n/a"

            upnl_txt = ""
            try:
                qty = float(quantity or 0.0)
                avg = float(avg_cost_basis or 0.0)
                px = float(current_sell_price or current_buy_price or 0.0)
                if qty > 0.0 and avg > 0.0 and px > 0.0:
                    upnl = (px - avg) * qty
                    upnl_txt = f" | uPnL {_fmt_money(upnl)}"
            except Exception:
                upnl_txt = ""

            text = f"1h: {local_txt} | {ai_txt}{upnl_txt}"
            color = CHART_TEXT
            if ai_action in {"exit", "reduce", "block_add"} or bias in {"bearish", "mild_bearish"}:
                color = CHART_DOWN
            elif ai_action in {"hold", "increase"} or bias in {"bullish", "mild_bullish"}:
                color = CHART_UP
            if len(text) > 160:
                text = text[:157].rstrip() + "..."
            self.outlook_status_label.config(text=text, foreground=color)
        except Exception:
            try:
                self.outlook_status_label.config(
                    text="1h Outlook: unavailable (model/read error)",
                    foreground=CHART_MUTED,
                )
            except Exception:
                pass

    def hide_legend_tooltip(self, *, clear_pointer: bool = True) -> None:
        try:
            aid = getattr(self, "_legend_hover_restore_after_id", None)
            if aid:
                self.after_cancel(aid)
        except Exception:
            pass
        self._legend_hover_restore_after_id = None
        try:
            aid = getattr(self, "_legend_bbox_after_id", None)
            if aid:
                self.after_cancel(aid)
        except Exception:
            pass
        self._legend_bbox_after_id = None
        try:
            for item in getattr(self, "_line_hover_targets", []) or []:
                if not isinstance(item, dict):
                    continue
                artist = item.get("artist")
                if artist is None:
                    continue
                artist.set_linewidth(float(item.get("line_width", 1.0)))
                artist.set_alpha(float(item.get("alpha", 0.9)))
        except Exception:
            pass
        self._active_hover_line = None
        try:
            tw = getattr(self, "_legend_tooltip_win", None)
            if tw is not None and bool(tw.winfo_exists()):
                tw.destroy()
        except Exception:
            pass
        self._legend_tooltip_win = None
        self._legend_tooltip_label = None
        if clear_pointer:
            self._legend_hover_last_canvas_xy = None

    def _can_show_legend_tooltip(self) -> bool:
        try:
            canvas_w = self.canvas.get_tk_widget()
        except Exception:
            return False
        try:
            if hasattr(canvas_w, "winfo_ismapped"):
                if not bool(canvas_w.winfo_ismapped()):
                    return False
        except Exception:
            pass
        try:
            if hasattr(canvas_w, "winfo_toplevel"):
                top = canvas_w.winfo_toplevel()
            else:
                top = None
            if top is not None:
                if hasattr(top, "state"):
                    state = str(top.state() or "").strip().lower()
                    if state in {"iconic", "withdrawn"}:
                        return False
                if hasattr(top, "winfo_viewable"):
                    if not bool(top.winfo_viewable()):
                        return False
        except Exception:
            pass
        return True

    def _schedule_restore_legend_hover(self) -> None:
        if not self._can_show_legend_tooltip():
            return
        try:
            if self._legend_hover_restore_after_id:
                self.after_cancel(self._legend_hover_restore_after_id)
        except Exception:
            pass

        try:
            self._legend_hover_restore_after_id = self.after_idle(self._restore_legend_hover)
        except Exception:
            self._legend_hover_restore_after_id = None

    def _restore_legend_hover(self) -> None:
        self._legend_hover_restore_after_id = None
        handler = getattr(self, "_legend_hover_motion_handler", None)
        if not callable(handler):
            return
        if not self._can_show_legend_tooltip():
            return
        try:
            canvas_w = self.canvas.get_tk_widget()
        except Exception:
            return
        try:
            width = int(canvas_w.winfo_width() or 0)
            height = int(canvas_w.winfo_height() or 0)
            root_x = int(canvas_w.winfo_rootx() or 0)
            root_y = int(canvas_w.winfo_rooty() or 0)
            pointer_x = int(canvas_w.winfo_pointerx() or 0)
            pointer_y = int(canvas_w.winfo_pointery() or 0)
            local_x = float(pointer_x - root_x)
            local_y = float(pointer_y - root_y)
            if not (0.0 <= local_x <= float(width) and 0.0 <= local_y <= float(height)):
                return
            self._legend_hover_last_canvas_xy = (local_x, local_y)
            handler(
                SimpleNamespace(
                    x=local_x,
                    y=max(0.0, float(height) - local_y),
                    inaxes=self.ax,
                    guiEvent=SimpleNamespace(x_root=pointer_x, y_root=pointer_y),
                )
            )
        except Exception:
            pass

    def refresh(
        self,
        coin_folders: Dict[str, str],
        current_buy_price: Optional[float] = None,
        current_sell_price: Optional[float] = None,
        trail_line: Optional[float] = None,
        dca_line_price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> None:



        cfg = self.settings_getter()

        tf = self.timeframe_var.get().strip()
        max_trade_labels = 1

        # Default to a cleaner chart and allow quick toggle without changing app settings.
        if not hasattr(self, "_chart_level_mode"):
            self._chart_level_mode = "clean"
        if not hasattr(self, "_chart_level_mode_bound"):
            try:
                canvas_w = self.canvas.get_tk_widget()

                def _toggle_chart_level_mode(_e=None):
                    try:
                        self._chart_level_mode = (
                            "detailed" if self._chart_level_mode == "clean" else "clean"
                        )
                        if hasattr(self, "detailed_overlays_var"):
                            self.detailed_overlays_var.set(self._chart_level_mode == "detailed")
                        self.event_generate("<<TimeframeChanged>>", when="tail")
                    except Exception:
                        self._chart_level_mode = "clean"
                        if hasattr(self, "detailed_overlays_var"):
                            self.detailed_overlays_var.set(False)

                # Double-click the chart to switch between Clean and Detailed overlays.
                canvas_w.bind("<Double-Button-1>", _toggle_chart_level_mode, add="+")
                self._chart_level_mode_bound = True
            except Exception:
                self._chart_level_mode_bound = False

        try:
            show_detailed_levels = bool(self.detailed_overlays_var.get())
        except Exception:
            show_detailed_levels = (getattr(self, "_chart_level_mode", "clean") == "detailed")
        self._chart_level_mode = "detailed" if show_detailed_levels else "clean"
        try:
            self.chart_key_label.config(
                text=("Key: ★ Trail  ◆ DCA  ● Avg  A Ask  B Bid" if show_detailed_levels else "Key: ★ Trail  ◆ DCA  ● Avg")
            )
        except Exception:
            pass

        if not hasattr(self, "_legend_hover_bound"):
            try:
                canvas_w = self.canvas.get_tk_widget()

                def _reset_hover_lines() -> None:
                    try:
                        for item in getattr(self, "_line_hover_targets", []):
                            artist = item.get("artist")
                            if artist is None:
                                continue
                            artist.set_linewidth(float(item.get("line_width", 1.0)))
                            artist.set_alpha(float(item.get("alpha", 0.9)))
                        self._active_hover_line = None
                    except Exception:
                        pass

                def _set_hover_line(active_item) -> None:
                    if getattr(self, "_active_hover_line", None) is active_item:
                        return
                    _reset_hover_lines()
                    if not active_item:
                        return
                    try:
                        artist = active_item.get("artist")
                        if artist is not None:
                            artist.set_linewidth(float(active_item.get("hover_line_width", active_item.get("line_width", 1.0))))
                            artist.set_alpha(float(active_item.get("hover_alpha", 1.0)))
                            self._active_hover_line = active_item
                    except Exception:
                        self._active_hover_line = None

                def _hide_legend_tooltip(_e=None, preserve_pointer: bool = False):
                    _reset_hover_lines()
                    self.hide_legend_tooltip(clear_pointer=(not preserve_pointer))

                def _show_legend_tooltip(x_root: int, y_root: int, text: str):
                    if not self._can_show_legend_tooltip():
                        _hide_legend_tooltip(preserve_pointer=True)
                        return
                    try:
                        tw = getattr(self, "_legend_tooltip_win", None)
                        lbl = getattr(self, "_legend_tooltip_label", None)
                        if tw is None or (not tw.winfo_exists()) or lbl is None or (not lbl.winfo_exists()):
                            tw = tk.Toplevel(canvas_w)
                            tw.withdraw()
                            tw.overrideredirect(True)
                            try:
                                tw.attributes("-topmost", True)
                            except Exception:
                                pass
                            lbl = tk.Label(
                                tw,
                                text=text,
                                justify="left",
                                anchor="w",
                                padx=8,
                                pady=6,
                                bg=DARK_BG2,
                                fg=DARK_FG,
                                bd=1,
                                relief="solid",
                            )
                            lbl.pack()
                            self._legend_tooltip_win = tw
                            self._legend_tooltip_label = lbl
                        else:
                            lbl.config(text=text)
                        tw.geometry(f"+{int(x_root) + 14}+{int(y_root) + 12}")
                        tw.deiconify()
                    except Exception:
                        pass

                def _line_tooltip_for_event(mpl_event):
                    x_disp = float(mpl_event.x)
                    y_disp = float(mpl_event.y)

                    for bbox_item in getattr(self, "_hover_regions_px", []):
                        try:
                            x0, y0, x1, y1 = bbox_item["bbox"]
                            if x0 <= x_disp <= x1 and y0 <= y_disp <= y1:
                                return str(bbox_item.get("text", "") or "").strip(), None
                        except Exception:
                            continue

                    if mpl_event.inaxes is not self.ax:
                        return "", None

                    try:
                        ax_bbox = self.ax.get_window_extent()
                        if not (ax_bbox.x0 <= x_disp <= ax_bbox.x1 and ax_bbox.y0 <= y_disp <= ax_bbox.y1):
                            return "", None
                    except Exception:
                        pass

                    nearest_text = ""
                    nearest_item = None
                    nearest_dist = 9.0
                    for line_item in getattr(self, "_line_hover_targets", []):
                        try:
                            dist = abs(y_disp - float(line_item["y_disp"]))
                            if dist <= nearest_dist:
                                nearest_text = str(line_item.get("text", "") or "").strip()
                                nearest_item = line_item
                                nearest_dist = dist
                        except Exception:
                            continue
                    return nearest_text, nearest_item

                def _on_legend_motion(mpl_event):
                    try:
                        if mpl_event.x is None or mpl_event.y is None:
                            _hide_legend_tooltip()
                            return
                        tip_txt, active_item = _line_tooltip_for_event(mpl_event)
                        if not tip_txt:
                            _hide_legend_tooltip()
                            return
                        _set_hover_line(active_item)
                        x_disp = float(mpl_event.x)
                        y_disp = float(mpl_event.y)
                        gui_evt = getattr(mpl_event, "guiEvent", None)
                        if gui_evt is not None and hasattr(gui_evt, "x_root") and hasattr(gui_evt, "y_root"):
                            x_root = int(gui_evt.x_root)
                            y_root = int(gui_evt.y_root)
                        else:
                            x_root = int(canvas_w.winfo_rootx() + x_disp)
                            y_root = int(canvas_w.winfo_rooty() + (canvas_w.winfo_height() - y_disp))
                        self._legend_hover_last_canvas_xy = (
                            float(x_disp),
                            max(0.0, float(canvas_w.winfo_height()) - float(y_disp)),
                        )
                        _show_legend_tooltip(x_root, y_root, tip_txt)
                    except Exception:
                        _hide_legend_tooltip()

                self._legend_hover_cid = self.canvas.mpl_connect("motion_notify_event", _on_legend_motion)
                self._legend_hover_leave_cid = self.canvas.mpl_connect("figure_leave_event", _hide_legend_tooltip)
                self._legend_hover_motion_handler = _on_legend_motion
                self._legend_hover_bound = True
            except Exception:
                self._legend_hover_bound = False

        def _nearest_levels(levels: List[float], anchor: Optional[float], keep: int = 2) -> List[float]:
            try:
                vals = [float(v) for v in levels if math.isfinite(float(v)) and float(v) > 0]
            except Exception:
                vals = []
            if show_detailed_levels:
                return vals
            try:
                aa = float(anchor)
                vals.sort(key=lambda v: abs(v - aa))
                return vals[:keep]
            except Exception:
                return vals[:keep]

        limit = int(cfg.get("candles_limit", 120))

        candles = self.fetcher.get_klines(self.coin, tf, limit=limit)
        try:
            series: List[Dict[str, Any]] = []
            for row in list(candles or [])[-220:]:
                if not isinstance(row, dict):
                    continue
                try:
                    ts_i = int(float(row.get("ts", 0) or 0))
                    close_f = float(row.get("close", 0.0) or 0.0)
                except Exception:
                    continue
                if ts_i <= 0 or close_f <= 0.0 or (not math.isfinite(close_f)):
                    continue
                series.append({"ts": ts_i, "close": close_f})
            self._last_candle_series = series
            self._last_candle_timeframe = str(tf or "")
            try:
                self._last_candle_step_s = int(self._timeframe_seconds(str(tf or "")))
            except Exception:
                self._last_candle_step_s = 3600
        except Exception:
            self._last_candle_series = []
            self._last_candle_timeframe = str(tf or "")
            self._last_candle_step_s = 3600

        folder = coin_folders.get(self.coin, "")
        low_path = os.path.join(folder, "low_bound_prices.html")
        high_path = os.path.join(folder, "high_bound_prices.html")

        # --- Cached neural reads (per path, by mtime) ---
        if not hasattr(self, "_neural_cache"):
            self._neural_cache = {}  # path -> (mtime, value)

        def _cached(path: str, loader, default):
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                return default
            hit = self._neural_cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
            v = loader(path)
            self._neural_cache[path] = (mtime, v)
            return v

        long_levels = _cached(low_path, read_price_levels_from_html, []) if folder else []
        short_levels = _cached(high_path, read_price_levels_from_html, []) if folder else []

        current_mid_price = None
        try:
            if (
                current_buy_price is not None
                and current_sell_price is not None
                and float(current_buy_price) > 0
                and float(current_sell_price) > 0
            ):
                current_mid_price = (float(current_buy_price) + float(current_sell_price)) / 2.0
        except Exception:
            current_mid_price = None
        anchor_price = current_mid_price if current_mid_price is not None else avg_cost_basis

        try:
            position_qty = float(quantity or 0.0)
        except Exception:
            position_qty = 0.0

        def _line_impact_text(line_name: str, line_price: Optional[float], meaning: str) -> str:
            try:
                price_txt = _fmt_price(float(line_price))
            except Exception:
                price_txt = "N/A"
            base = f"{line_name}: {price_txt}\nMeaning: {meaning}"
            try:
                lp = float(line_price)
                avg = float(avg_cost_basis or 0.0)
                qtyf = float(position_qty or 0.0)
                if lp > 0 and avg > 0 and qtyf > 0:
                    est_value = qtyf * lp
                    est_cost = qtyf * avg
                    est_pnl = est_value - est_cost
                    est_pct = ((lp - avg) / avg) * 100.0
                    base += (
                        f"\nImpact if hit: value {_fmt_money(est_value)}"
                        f" | est. PnL {est_pnl:+.2f} ({est_pct:+.2f}%)"
                    )
            except Exception:
                pass
            return base

        long_sig_path = os.path.join(folder, "long_dca_signal.txt")
        long_sig = _cached(long_sig_path, read_int_from_file, 0) if folder else 0
        short_sig = read_short_signal(folder) if folder else 0
        try:
            last_close_price = float(candles[-1].get("close", 0.0) or 0.0)
            if not math.isfinite(last_close_price) or last_close_price <= 0.0:
                last_close_price = None
        except Exception:
            last_close_price = None
        self._update_outlook_status(
            long_levels=long_levels,
            short_levels=short_levels,
            long_sig=long_sig,
            short_sig=short_sig,
            current_buy_price=current_buy_price,
            current_sell_price=current_sell_price,
            last_close_price=last_close_price,
            avg_cost_basis=avg_cost_basis,
            quantity=quantity,
        )

        # --- Avoid full ax.clear() (expensive). Just clear artists. ---
        try:
            self.ax.lines.clear()
            self.ax.patches.clear()
            self.ax.collections.clear()  # scatter dots live here
            self.ax.texts.clear()        # labels/annotations live here
        except Exception:
            # fallback if matplotlib version lacks .clear() on these lists
            self.ax.cla()
            self._apply_dark_chart_style()


        if not candles:
            try:
                self._last_candle_series = []
                self._last_candle_timeframe = str(tf or "")
            except Exception:
                pass
            try:
                self.outlook_status_label.config(
                    text="1h Outlook: waiting for candle/model data...",
                    foreground=CHART_MUTED,
                )
            except Exception:
                pass
            self._legend_panel_text = f"{self.coin}: waiting for candle data..."
            self._legend_rows = [
                {"label": "Green candle", "meaning": "Price closed above open for that bar.", "color": CHART_UP, "dash": (), "sample": "square"},
                {"label": "Red candle", "meaning": "Price closed below open for that bar.", "color": CHART_DOWN, "dash": (), "sample": "square"},
                {"label": "Long neural level", "meaning": "Blue support/reference level from neural model.", "color": CHART_LINE_MAIN, "dash": ()},
                {"label": "Short neural level", "meaning": "Orange resistance/reference level from neural model.", "color": CHART_EMA_SLOW, "dash": ()},
                {"label": "Trail line (★)", "meaning": "Trailing sell threshold once armed.", "color": CHART_UP, "dash": ()},
                {"label": "Next DCA (◆)", "meaning": "Next averaging-buy trigger level.", "color": CHART_DOWN, "dash": ()},
                {"label": "Average cost (●)", "meaning": "Current blended entry price.", "color": CHART_EMA_SLOW, "dash": ()},
            ]
            self._legend_note = "Chart is loading. Legend previews the indicator meanings."
            self._legend_tooltip_text = ""
            self._legend_bbox_px = None
            self._legend_hover_artist = None
            self._hover_regions_px = []
            self._hover_text_artists = []
            self._line_hover_targets = []
            err = ""
            try:
                err = str(self.fetcher.get_last_error(self.coin, tf, limit=limit) or "").strip()
            except Exception:
                err = ""
            spinner_char = ["|", "/", "-", "\\"][int(time.time() * 6.0) % 4]
            if err:
                self.ax.set_title(f"{self.coin} ({tf}) - feed retry {spinner_char}", color=CHART_TITLE)
                try:
                    self.neural_status_label.config(text=f"Neural: N/A | retrying feed")
                except Exception:
                    pass
            else:
                self.ax.set_title(f"{self.coin} ({tf}) - loading {spinner_char}", color=CHART_TITLE)
            try:
                self.ax.text(
                    0.5,
                    0.5,
                    f"{spinner_char}\nLoading candle data...",
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                    color=CHART_LINE_MAIN,
                    fontsize=12,
                    bbox={"facecolor": CHART_PANEL, "edgecolor": CHART_BORDER, "pad": 8},
                )
            except Exception:
                pass
            self.canvas.draw_idle()
            return


        # Candlestick drawing (green up / red down) - batch rectangles
        xs = getattr(self, "_xs", None)
        if not xs or len(xs) != len(candles):
            xs = list(range(len(candles)))
            self._xs = xs

        rects = []
        for i, c in enumerate(candles):
            o = float(c["open"])
            cl = float(c["close"])
            h = float(c["high"])
            l = float(c["low"])

            up = cl >= o
            candle_color = CHART_UP if up else CHART_DOWN

            # wick
            self.ax.plot([i, i], [l, h], linewidth=1, color=candle_color)

            # body
            bottom = min(o, cl)
            height = abs(cl - o)
            if height < 1e-12:
                height = 1e-12

            rects.append(
                Rectangle(
                    (i - 0.35, bottom),
                    0.7,
                    height,
                    facecolor=candle_color,
                    edgecolor=candle_color,
                    linewidth=1,
                    alpha=0.9,
                )
            )

        for r in rects:
            self.ax.add_patch(r)

        # Lock y-limits to candle range so overlay lines can go offscreen without expanding the chart.
        try:
            y_low = min(float(c["low"]) for c in candles)
            y_high = max(float(c["high"]) for c in candles)
            pad = (y_high - y_low) * 0.03
            if not math.isfinite(pad) or pad <= 0:
                pad = max(abs(y_low) * 0.001, 1e-6)
            self.ax.set_ylim(y_low - pad, y_high + pad)
        except Exception:
            pass

        # Reset the axes to its base geometry; chart legend now lives in the side panel.
        try:
            if not hasattr(self, "_base_ax_pos"):
                self._base_ax_pos = self.ax.get_position().frozen()
            self.ax.set_position(self._base_ax_pos)
        except Exception:
            pass



        # Overlay Neural levels (blue long, orange short)
        levels_to_draw_long = _nearest_levels(long_levels, anchor_price, keep=2)
        levels_to_draw_short = _nearest_levels(short_levels, anchor_price, keep=2)
        line_hover_targets = []
        for lv in levels_to_draw_long:
            try:
                yy = float(lv)
                artist = self.ax.axhline(
                    y=yy,
                    linewidth=1,
                    color=CHART_LINE_MAIN,
                    alpha=(0.8 if show_detailed_levels else 0.65),
                )
                line_hover_targets.append({
                    "y": yy,
                    "artist": artist,
                    "line_width": 1.0,
                    "hover_line_width": 1.8,
                    "alpha": (0.8 if show_detailed_levels else 0.65),
                    "hover_alpha": 1.0,
                    "text": _line_impact_text(
                        "Long level",
                        yy,
                        "Neural long support/reference level; price moving near it strengthens bullish context.",
                    ),
                })
            except Exception:
                pass

        for lv in levels_to_draw_short:
            try:
                yy = float(lv)
                artist = self.ax.axhline(
                    y=yy,
                    linewidth=1,
                    color=CHART_EMA_SLOW,
                    alpha=(0.8 if show_detailed_levels else 0.65),
                )
                line_hover_targets.append({
                    "y": yy,
                    "artist": artist,
                    "line_width": 1.0,
                    "hover_line_width": 1.8,
                    "alpha": (0.8 if show_detailed_levels else 0.65),
                    "hover_alpha": 1.0,
                    "text": _line_impact_text(
                        "Short level",
                        yy,
                        "Neural short resistance/reference level; price moving near it strengthens bearish context.",
                    ),
                })
            except Exception:
                pass


        # Overlay Trailing PM line (sell) and next DCA line
        try:
            if trail_line is not None and float(trail_line) > 0:
                yy = float(trail_line)
                artist = self.ax.axhline(y=yy, linewidth=1.5, color=CHART_UP, alpha=0.95)
                line_hover_targets.append({
                    "y": yy,
                    "artist": artist,
                    "line_width": 1.5,
                    "hover_line_width": 2.2,
                    "alpha": 0.95,
                    "hover_alpha": 1.0,
                    "text": _line_impact_text(
                        "Trail line",
                        yy,
                        "Active trailing sell threshold for the current position.",
                    ),
                })
        except Exception:
            pass

        try:
            if dca_line_price is not None and float(dca_line_price) > 0:
                yy = float(dca_line_price)
                artist = self.ax.axhline(y=yy, linewidth=1.5, color=CHART_DOWN, alpha=0.95)
                line_hover_targets.append({
                    "y": yy,
                    "artist": artist,
                    "line_width": 1.5,
                    "hover_line_width": 2.2,
                    "alpha": 0.95,
                    "hover_alpha": 1.0,
                    "text": _line_impact_text(
                        "Next DCA",
                        yy,
                        "Next configured DCA trigger price; touching it makes the next averaging buy eligible.",
                    ),
                })
        except Exception:
            pass

        # Overlay avg cost basis (yellow)
        try:
            if avg_cost_basis is not None and float(avg_cost_basis) > 0:
                yy = float(avg_cost_basis)
                artist = self.ax.axhline(y=yy, linewidth=1.5, color=CHART_EMA_SLOW, alpha=0.95)
                line_hover_targets.append({
                    "y": yy,
                    "artist": artist,
                    "line_width": 1.5,
                    "hover_line_width": 2.2,
                    "alpha": 0.95,
                    "hover_alpha": 1.0,
                    "text": _line_impact_text(
                        "Average cost",
                        yy,
                        "Current blended entry price; near break-even before fees/slippage.",
                    ),
                })
        except Exception:
            pass

        # Overlay current ask/bid prices
        try:
            if current_buy_price is not None and float(current_buy_price) > 0:
                yy = float(current_buy_price)
                artist = self.ax.axhline(y=yy, linewidth=1.5, color=CHART_LINE_MAIN, alpha=0.95)
                if show_detailed_levels:
                    line_hover_targets.append({
                        "y": yy,
                        "artist": artist,
                        "line_width": 1.5,
                        "hover_line_width": 2.2,
                        "alpha": 0.95,
                        "hover_alpha": 1.0,
                        "text": _line_impact_text(
                            "Ask",
                            yy,
                            "Current buy-side market reference price.",
                        ),
                    })
        except Exception:
            pass

        try:
            if current_sell_price is not None and float(current_sell_price) > 0:
                yy = float(current_sell_price)
                artist = self.ax.axhline(y=yy, linewidth=1.5, color=CHART_LINE_ALT, alpha=0.95)
                if show_detailed_levels:
                    line_hover_targets.append({
                        "y": yy,
                        "artist": artist,
                        "line_width": 1.5,
                        "hover_line_width": 2.2,
                        "alpha": 0.95,
                        "hover_alpha": 1.0,
                        "text": _line_impact_text(
                            "Bid",
                            yy,
                            "Current sell-side market reference price.",
                        ),
                    })
        except Exception:
            pass

        # One-hour outlook is surfaced in dedicated UI panels; keep the main candle chart clean.
        projected_line_price = None
        projected_line_bias = ""
        projected_line_pct = None
        projection_zone: Dict[str, Any] = {}
        try:
            local_outlook = getattr(self, "_last_local_outlook", {})
            if not isinstance(local_outlook, dict):
                local_outlook = {}
            projected_line_bias = str(local_outlook.get("bias", "") or "").strip().lower()
            if projected_line_bias in {"bullish", "mild_bullish"}:
                projected_line_price = _float_or_none(local_outlook.get("target_up", None))
                projected_line_pct = _float_or_none(local_outlook.get("up_pct", None))
            elif projected_line_bias in {"bearish", "mild_bearish"}:
                projected_line_price = _float_or_none(local_outlook.get("target_down", None))
                projected_line_pct = _float_or_none(local_outlook.get("down_pct", None))
        except Exception:
            projected_line_price = None
            projection_zone = {}

        # Right-side boxed price labels have been removed; line hover now carries the context instead.
        self._hover_text_artists = []

        # Build the chart legend text for the side panel.
        try:
            trade_start_level = int(cfg.get("trade_start_level", 3) or 3)
            dca_levels_cfg = list(cfg.get("dca_levels", []) or [])
            dca_mult = float(cfg.get("dca_multiplier", 2.0) or 2.0)
            max_dca_24h = int(cfg.get("max_dca_buys_per_24h", 2) or 2)
            pm_no_dca = float(cfg.get("pm_start_pct_no_dca", 5.0) or 5.0)
            pm_with_dca = float(cfg.get("pm_start_pct_with_dca", 2.5) or 2.5)
            trail_gap = float(cfg.get("trailing_gap_pct", 0.5) or 0.5)
            level_mode_label = "Detailed" if show_detailed_levels else "Clean"

            def _fmt_level_list(vals: List[float]) -> str:
                try:
                    if not vals:
                        return "N/A"
                    shown_vals = list(vals)
                    extra_count = 0
                    if show_detailed_levels and len(shown_vals) > 6:
                        extra_count = len(shown_vals) - 6
                        shown_vals = shown_vals[:6]
                    txt = ", ".join(_fmt_price(float(v)) for v in shown_vals)
                    if extra_count > 0:
                        txt += f" (+{extra_count} more)"
                    return txt
                except Exception:
                    return "N/A"

            def _fmt_optional_price(v: Optional[float]) -> str:
                try:
                    vv = float(v)
                    if vv > 0 and math.isfinite(vv):
                        return _fmt_price(vv)
                except Exception:
                    pass
                return "N/A"

            def _fmt_delta(anchor_val: Optional[float], target_val: Optional[float]) -> str:
                try:
                    aa = float(anchor_val)
                    tt = float(target_val)
                    if (not math.isfinite(aa)) or (not math.isfinite(tt)) or aa <= 0 or tt <= 0:
                        return "N/A"
                    delta_pct = ((tt - aa) / aa) * 100.0
                    return f"{delta_pct:+.2f}%"
                except Exception:
                    return "N/A"

            ask_text = _fmt_optional_price(current_buy_price)
            bid_text = _fmt_optional_price(current_sell_price)
            avg_text = _fmt_optional_price(avg_cost_basis)
            dca_text = _fmt_optional_price(dca_line_price)
            trail_text = _fmt_optional_price(trail_line)
            dca_delta_text = _fmt_delta(anchor_price, dca_line_price)
            trail_delta_text = _fmt_delta(anchor_price, trail_line)

            try:
                dca_levels_shown = dca_levels_cfg[:4]
                dca_extra = max(0, len(dca_levels_cfg) - len(dca_levels_shown))
                dca_levels_text = ", ".join(str(v) for v in dca_levels_shown) if dca_levels_shown else "N/A"
                if dca_extra > 0:
                    dca_levels_text += f" (+{dca_extra} more)"
            except Exception:
                dca_levels_text = "N/A"

            def _wrap_text_block(text: str, width: int = 60) -> str:
                lines = []
                for raw_line in str(text).splitlines():
                    line = raw_line.strip()
                    if len(line) <= width:
                        lines.append(line)
                        continue
                    current = ""
                    for word in line.split(" "):
                        test = word if not current else f"{current} {word}"
                        if len(test) <= width:
                            current = test
                        else:
                            if current:
                                lines.append(current)
                            current = word
                    if current:
                        lines.append(current)
                return "\n".join(lines)

            legend_lines = [
                f"Mode: {level_mode_label}",
                "Key: ★ Trail | ◆ DCA | ● Avg",
                f"Long: {_fmt_level_list(levels_to_draw_long)}",
                f"Short: {_fmt_level_list(levels_to_draw_short)}",
                f"Px: ● {avg_text} | ◆ {dca_text} | ★ {trail_text}",
                f"Δ: ◆ {dca_delta_text} | ★ {trail_delta_text}",
            ]
            if show_detailed_levels:
                legend_lines = [
                    "Mode: Detailed",
                    "Key: ★ Trail | ◆ DCA | ● Avg | A | B",
                    f"Long: {_fmt_level_list(levels_to_draw_long)}",
                    f"Short: {_fmt_level_list(levels_to_draw_short)}",
                    f"Px: A {ask_text} | B {bid_text} | ● {avg_text} | ★ {trail_text}",
                    f"Δ: ◆ {dca_delta_text} | ★ {trail_delta_text}",
                    "Params:",
                    f"Start L{trade_start_level}",
                    f"DCA%: [{dca_levels_text}]",
                    f"x{dca_mult:g} | Max {max_dca_24h}/coin/24h",
                    f"PM: +{pm_no_dca:g}% / +{pm_with_dca:g}% | Gap {trail_gap:g}%",
                ]

            legend_text = _wrap_text_block("\n".join(legend_lines), width=60)
            legend_rows: List[Dict[str, Any]] = [
                {
                    "label": "Green candle",
                    "meaning": "Price closed above open for that bar.",
                    "color": CHART_UP,
                    "dash": (),
                    "sample": "square",
                },
                {
                    "label": "Red candle",
                    "meaning": "Price closed below open for that bar.",
                    "color": CHART_DOWN,
                    "dash": (),
                    "sample": "square",
                },
                {
                    "label": "Long neural level",
                    "meaning": "Blue support/reference level from neural model.",
                    "color": CHART_LINE_MAIN,
                    "dash": (),
                },
                {
                    "label": "Short neural level",
                    "meaning": "Orange resistance/reference level from neural model.",
                    "color": CHART_EMA_SLOW,
                    "dash": (),
                },
                {
                    "label": "Trail line (★)",
                    "meaning": (
                        f"Trailing sell threshold ({trail_text}). "
                        "Crossing back through it can trigger a sell."
                    ),
                    "color": CHART_UP,
                    "dash": (),
                },
                {
                    "label": "Next DCA (◆)",
                    "meaning": (
                        f"Next averaging-buy trigger ({dca_text}). "
                        "Touching this line makes the next DCA buy eligible."
                    ),
                    "color": CHART_DOWN,
                    "dash": (),
                },
                {
                    "label": "Average cost (●)",
                    "meaning": f"Current blended entry price ({avg_text}).",
                    "color": CHART_EMA_SLOW,
                    "dash": (),
                },
            ]
            if show_detailed_levels:
                legend_rows.extend(
                    [
                        {
                            "label": "Ask line (A)",
                            "meaning": f"Current buy-side market reference ({ask_text}).",
                            "color": CHART_LINE_MAIN,
                            "dash": (),
                        },
                        {
                            "label": "Bid line (B)",
                            "meaning": f"Current sell-side market reference ({bid_text}).",
                            "color": CHART_LINE_ALT,
                            "dash": (),
                        },
                    ]
                )

            self._legend_panel_text = legend_text
            self._legend_rows = legend_rows
            self._legend_note = (
                f"Mode: {level_mode_label}. "
                "Use this key to map each chart line/symbol to entry/exit meaning."
            )
            self._legend_mode = level_mode_label
            self._legend_tooltip_text = ""
            self._legend_bbox_px = None
            self._legend_hover_artist = None
            self._legend_needs_scroll = bool(show_detailed_levels)
        except Exception:
            self._legend_panel_text = "Legend unavailable"
            self._legend_rows = []
            self._legend_note = "Legend unavailable."
            self._legend_mode = "N/A"
            self._legend_tooltip_text = ""
            self._legend_bbox_px = None
            self._legend_hover_artist = None
            self._legend_needs_scroll = False
            pass




        # --- Trade dots (BUY / DCA / SELL) for THIS coin only ---
        try:
            trades = _read_trade_history_jsonl(self.trade_history_path) if self.trade_history_path else []
            plotted_trade_points = []
            if trades:
                candle_ts = [int(c["ts"]) for c in candles]  # oldest->newest
                t_min = float(candle_ts[0])
                t_max = float(candle_ts[-1])

                plotted_trade_points = []
                for tr in trades:
                    sym = str(tr.get("symbol", "")).upper()
                    base = sym.split("-")[0].strip() if sym else ""
                    if base != self.coin.upper().strip():
                        continue

                    side = str(tr.get("side", "")).lower().strip()
                    tag = str(tr.get("tag") or "").upper().strip()

                    if side == "buy":
                        label = "DCA" if tag == "DCA" else "BUY"
                        color = "#8DE7FF" if tag == "DCA" else CHART_UP
                    elif side == "sell":
                        label = "SELL"
                        color = CHART_DOWN
                    else:
                        continue

                    tts = tr.get("ts", None)
                    if tts is None:
                        continue
                    try:
                        tts = float(tts)
                    except Exception:
                        continue
                    if tts < t_min or tts > t_max:
                        continue

                    i = bisect.bisect_left(candle_ts, tts)
                    if i <= 0:
                        idx = 0
                    elif i >= len(candle_ts):
                        idx = len(candle_ts) - 1
                    else:
                        idx = i if abs(candle_ts[i] - tts) < abs(tts - candle_ts[i - 1]) else (i - 1)

                    # y = trade price if present, else candle close
                    y = None
                    try:
                        p = tr.get("price", None)
                        if p is not None and float(p) > 0:
                            y = float(p)
                    except Exception:
                        y = None
                    if y is None:
                        try:
                            y = float(candles[idx].get("close", 0.0))
                        except Exception:
                            y = None
                    if y is None:
                        continue

                    x = idx
                    self.ax.scatter([x], [y], s=35, color=color, zorder=6)
                    plotted_trade_points.append((tts, label, x, y))
        except Exception:
            pass

        try:
            if plotted_trade_points:
                plotted_trade_points.sort(key=lambda item: item[0])
                for _, label, x, y in plotted_trade_points[-max_trade_labels:]:
                    self.ax.annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        color=CHART_TEXT,
                        zorder=7,
                    )
        except Exception:
            pass


        self.ax.set_xlim(-0.5, (len(candles) - 0.5) + 0.35)

        self.ax.set_title(f"{self.coin} ({tf})", color=CHART_TITLE)



        # x tick labels (date + time) - evenly spaced, never overlapping duplicates
        n = len(candles)
        want = 5  # keep it readable even when the window is narrow
        if n <= want:
            idxs = list(range(n))
        else:
            step = (n - 1) / float(want - 1)
            idxs = []
            last = -1
            for j in range(want):
                i = int(round(j * step))
                if i <= last:
                    i = last + 1
                if i >= n:
                    i = n - 1
                idxs.append(i)
                last = i

        tick_x = [xs[i] for i in idxs]
        tick_lbl = [
            time.strftime("%Y-%m-%d\n%H:%M", time.localtime(int(candles[i].get("ts", 0))))
            for i in idxs
        ]
        try:
            self.ax.minorticks_off()
            self.ax.set_xticks(tick_x)
            self.ax.set_xticklabels(tick_lbl)
            self.ax.tick_params(axis="x", labelsize=8)
        except Exception:
            pass


        self.canvas.draw_idle()
        try:
            try:
                self._line_hover_targets = [
                    {
                        "text": str(item.get("text", "") or "").strip(),
                        "y_disp": float(self.ax.transData.transform((0.0, float(item.get("y", 0.0))))[1]),
                        "artist": item.get("artist"),
                        "line_width": float(item.get("line_width", 1.0)),
                        "hover_line_width": float(item.get("hover_line_width", item.get("line_width", 1.0))),
                        "alpha": float(item.get("alpha", 0.9)),
                        "hover_alpha": float(item.get("hover_alpha", 1.0)),
                    }
                    for item in (line_hover_targets or [])
                    if str(item.get("text", "") or "").strip()
                ]
            except Exception:
                self._line_hover_targets = []

            if getattr(self, "_hover_text_artists", []):
                if getattr(self, "_legend_bbox_after_id", None):
                    self.after_cancel(self._legend_bbox_after_id)

                def _refresh_legend_bbox():
                    try:
                        renderer = self.canvas.get_renderer()
                        hover_regions = []
                        for artist, text in list(getattr(self, "_hover_text_artists", [])):
                            try:
                                bbox = artist.get_window_extent(renderer=renderer)
                                hover_regions.append({"bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1), "text": text})
                            except Exception:
                                continue
                        self._hover_regions_px = hover_regions
                    except Exception:
                        self._hover_regions_px = []
                    finally:
                        self._legend_bbox_after_id = None

                self._legend_bbox_after_id = self.after_idle(_refresh_legend_bbox)
            else:
                self._hover_regions_px = []
                self.hide_legend_tooltip(clear_pointer=False)
            self._schedule_restore_legend_hover()
        except Exception:
            pass


        self.neural_status_label.config(text=f"Neural: long={long_sig} short={short_sig} | levels L={len(long_levels)} S={len(short_levels)}")

        # show file update time if possible
        last_ts = None
        try:
            if os.path.isfile(low_path):
                last_ts = os.path.getmtime(low_path)
            elif os.path.isfile(high_path):
                last_ts = os.path.getmtime(high_path)
        except Exception:
            last_ts = None

        if last_ts:
            self.last_update_label.config(text=f"Last: {time.strftime('%H:%M:%S', time.localtime(last_ts))}")
        else:
            self.last_update_label.config(text="Last: N/A")

    def export_png(self, path: str) -> bool:
        try:
            self.fig.savefig(path, dpi=160, facecolor=self.fig.get_facecolor())
            return True
        except Exception:
            return False


# -----------------------------
# Account overview chart widget (canvas renderer)
# -----------------------------

class CanvasAccountOverviewChart(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        render_callback: Callable[[tk.Canvas, int, int], None],
        export_callback: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(parent)
        self._render_callback = render_callback
        self._export_callback = export_callback
        self._last_size: Tuple[int, int] = (0, 0)
        self._resize_after_id = None

        self.canvas = tk.Canvas(
            self,
            background=DARK_PANEL2,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True, padx=0, pady=(0, 6))
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")

    def _on_canvas_configure(self, event: tk.Event) -> None:
        try:
            w = int(getattr(event, "width", 0) or 0)
            h = int(getattr(event, "height", 0) or 0)
            if w <= 1 or h <= 1:
                return
            if (w, h) == self._last_size:
                return
            self._last_size = (w, h)
            if self._resize_after_id:
                try:
                    self.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.after_idle(self.refresh)
        except Exception:
            pass

    def refresh(self) -> None:
        try:
            if self._resize_after_id:
                self._resize_after_id = None
        except Exception:
            pass
        try:
            width = max(320, int(self.canvas.winfo_width() or 0))
            height = max(220, int(self.canvas.winfo_height() or 0))
        except Exception:
            width, height = 720, 320
        try:
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, width, height, fill=DARK_PANEL2, outline=DARK_BORDER)
            self._render_callback(self.canvas, width, height)
        except Exception:
            pass

    def export_png(self, path: str) -> bool:
        if callable(self._export_callback):
            try:
                return bool(self._export_callback(path))
            except Exception:
                return False
        return False


# -----------------------------
# Account Value chart widget (legacy matplotlib renderer)
# -----------------------------

class AccountValueChart(ttk.Frame):
    def __init__(self, parent: tk.Widget, history_path: str, trade_history_path: str, max_points: int = 250):
        super().__init__(parent)
        self.history_path = history_path
        self.trade_history_path = trade_history_path
        # Hard-cap to 250 points max (account value chart only)
        self.max_points = min(int(max_points or 0) or 250, 250)
        self._last_mtime: Optional[float] = None


        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Account value").pack(side="left")
        self.last_update_label = ttk.Label(top, text="Last: N/A")
        self.last_update_label.pack(side="right")

        self.fig = Figure(figsize=(6.5, 3.5), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)

        # Keep a modest buffer for labels/title while maximizing the visible chart area.
        self.fig.subplots_adjust(left=0.05, bottom=0.14, right=0.988, top=0.89)

        self.ax = self.fig.add_subplot(111)
        self._apply_dark_chart_style()
        self.ax.set_title("Account Value", color=DARK_FG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas_w = self.canvas.get_tk_widget()
        canvas_w.configure(bg=DARK_BG)

        # Remove horizontal padding here so the chart widget truly fills the container.
        canvas_w.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        # Keep the matplotlib figure EXACTLY the same pixel size as the Tk widget.
        # FigureCanvasTkAgg already sizes its backing PhotoImage to e.width/e.height.
        # Multiplying by tk scaling here makes the renderer larger than the PhotoImage,
        # which produces the "blank/covered strip" on the right.
        self._last_canvas_px = (0, 0)
        self._resize_after_id = None

        def _on_canvas_configure(e):
            try:
                w = int(e.width)
                h = int(e.height)
                if w <= 1 or h <= 1:
                    return

                if (w, h) == self._last_canvas_px:
                    return
                self._last_canvas_px = (w, h)

                dpi = float(self.fig.get_dpi() or 100.0)
                self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

                # Debounce redraws during live resize
                if self._resize_after_id:
                    try:
                        self.after_cancel(self._resize_after_id)
                    except Exception:
                        pass
                self._resize_after_id = self.after_idle(self.canvas.draw_idle)
            except Exception:
                pass

        canvas_w.bind("<Configure>", _on_canvas_configure, add="+")








    def _apply_dark_chart_style(self) -> None:
        try:
            self.fig.patch.set_facecolor(CHART_BG)
            self.ax.set_facecolor(CHART_PANEL)
            self.ax.tick_params(colors=CHART_TEXT)
            for spine in self.ax.spines.values():
                spine.set_color(CHART_BORDER)
            self.ax.grid(True, color=CHART_GRID, linewidth=0.65, alpha=0.45, linestyle="--")
        except Exception:
            pass

    def refresh(self) -> None:
        path = self.history_path

        # mtime cache so we don't redraw if nothing changed (account history OR trade history)
        try:
            m_hist = os.path.getmtime(path)
        except Exception:
            m_hist = None

        try:
            m_trades = os.path.getmtime(self.trade_history_path) if self.trade_history_path else None
        except Exception:
            m_trades = None

        candidates = [m for m in (m_hist, m_trades) if m is not None]
        mtime = max(candidates) if candidates else None

        if mtime is not None and self._last_mtime == mtime:
            return
        self._last_mtime = mtime


        points: List[Tuple[float, float]] = []

        try:
            if os.path.isfile(path):
                # Read the FULL history so the chart shows from the very beginning
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

                for ln in lines:
                    try:
                        obj = fast_json_loads(ln, default=None)
                        ts = obj.get("ts", None)
                        v = obj.get("total_account_value", None)
                        if ts is None or v is None:
                            continue

                        tsf = float(ts)
                        vf = float(v)

                        # Drop obviously invalid points early
                        if (not math.isfinite(tsf)) or (not math.isfinite(vf)) or (vf <= 0.0):
                            continue

                        points.append((tsf, vf))
                    except Exception:
                        continue
        except Exception:
            points = []

        # ---- Clean up history so single-tick bogus dips/spikes don't render ----
        if points:
            # Ensure chronological order
            points.sort(key=lambda x: x[0])

            # De-dupe identical timestamps (keep the latest occurrence)
            dedup: List[Tuple[float, float]] = []
            for tsf, vf in points:
                if dedup and tsf == dedup[-1][0]:
                    dedup[-1] = (tsf, vf)
                else:
                    dedup.append((tsf, vf))
            points = dedup


        # Downsample to <= 250 points by AVERAGING buckets instead of skipping points.
        # IMPORTANT: never average the VERY FIRST or VERY LAST point.
        # - First point should remain the true first historical value.
        # - Last point should remain the true current/final account value (so the title and chart end match account info).
        max_keep = min(max(2, int(self.max_points or 250)), 250)
        n = len(points)

        if n > max_keep:
            first_pt = points[0]
            last_pt = points[-1]

            mid_points = points[1:-1]
            mid_n = len(mid_points)
            keep_mid = max_keep - 2

            if keep_mid <= 0 or mid_n <= 0:
                points = [first_pt, last_pt]
            elif mid_n <= keep_mid:
                points = [first_pt] + mid_points + [last_pt]
            else:
                bucket_size = mid_n / float(keep_mid)
                new_mid: List[Tuple[float, float]] = []

                for i in range(keep_mid):
                    start = int(i * bucket_size)
                    end = int((i + 1) * bucket_size)
                    if end <= start:
                        end = start + 1
                    if start >= mid_n:
                        break
                    if end > mid_n:
                        end = mid_n

                    bucket = mid_points[start:end]
                    if not bucket:
                        continue

                    # Average timestamp and account value within the bucket (MID ONLY)
                    avg_ts = sum(p[0] for p in bucket) / len(bucket)
                    avg_val = sum(p[1] for p in bucket) / len(bucket)
                    new_mid.append((avg_ts, avg_val))

                points = [first_pt] + new_mid + [last_pt]



        # clear artists (fast) / fallback to cla()
        try:
            self.ax.lines.clear()
            self.ax.patches.clear()
            self.ax.collections.clear()  # scatter dots live here
            self.ax.texts.clear()        # labels/annotations live here
        except Exception:
            self.ax.cla()
            self._apply_dark_chart_style()


        if not points:
            spinner_char = ["|", "/", "-", "\\"][int(time.time() * 6.0) % 4]
            self.ax.set_title(f"Account Value - loading {spinner_char}", color=CHART_TITLE)
            self.last_update_label.config(text="Last: N/A")
            try:
                self.ax.text(
                    0.5,
                    0.5,
                    f"{spinner_char}\nLoading account history...",
                    transform=self.ax.transAxes,
                    ha="center",
                    va="center",
                    color=CHART_LINE_MAIN,
                    fontsize=12,
                    bbox={"facecolor": CHART_PANEL, "edgecolor": CHART_BORDER, "pad": 8},
                )
            except Exception:
                pass
            self.canvas.draw_idle()
            return

        xs = list(range(len(points)))
        # Only show cent-level changes (hide sub-cent noise)
        ys = [round(p[1], 2) for p in points]

        if len(xs) >= 2:
            try:
                self.ax.fill_between(xs, ys, [min(ys)] * len(ys), color=CHART_FILL_MAIN, alpha=0.52)
            except Exception:
                pass
            try:
                self.ax.plot(xs, ys, linewidth=3.0, color=CHART_LINE_GLOW)
            except TypeError:
                self.ax.plot(xs, ys, linewidth=3.0)
            except Exception:
                pass
        try:
            self.ax.plot(xs, ys, linewidth=1.6, color=CHART_LINE_MAIN)
        except TypeError:
            self.ax.plot(xs, ys, linewidth=1.6)

        # --- Trade dots (BUY / DCA / SELL) for ALL coins ---
        try:
            trades = _read_trade_history_jsonl(self.trade_history_path) if self.trade_history_path else []
            plotted_trade_points = []
            if trades:
                ts_list = [float(p[0]) for p in points]  # matches xs/ys indices
                t_min = ts_list[0]
                t_max = ts_list[-1]

                for tr in trades:
                    # Determine label/color
                    side = str(tr.get("side", "")).lower().strip()
                    tag = str(tr.get("tag", "")).upper().strip()

                    if side == "buy":
                        action_label = "DCA" if tag == "DCA" else "BUY"
                        color = "#8DE7FF" if tag == "DCA" else CHART_UP
                    elif side == "sell":
                        action_label = "SELL"
                        color = CHART_DOWN
                    else:
                        continue

                    # Prefix with coin (so the dot says which coin it is)
                    sym = str(tr.get("symbol", "")).upper().strip()
                    coin_tag = (sym.split("-")[0].split("/")[0].strip() if sym else "") or (sym or "?")
                    label = f"{coin_tag} {action_label}"

                    tts = tr.get("ts")
                    try:
                        tts = float(tts)
                    except Exception:
                        continue
                    if tts < t_min or tts > t_max:
                        continue

                    # nearest account-value point
                    i = bisect.bisect_left(ts_list, tts)
                    if i <= 0:
                        idx = 0
                    elif i >= len(ts_list):
                        idx = len(ts_list) - 1
                    else:
                        idx = i if abs(ts_list[i] - tts) < abs(tts - ts_list[i - 1]) else (i - 1)

                    x = idx
                    y = ys[idx]

                    self.ax.scatter([x], [y], s=30, color=color, zorder=6)
                    plotted_trade_points.append((tts, label, x, y))

                plotted_trade_points.sort(key=lambda item: item[0])
                for _, label, x, y in plotted_trade_points[-3:]:
                    self.ax.annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        color=CHART_TEXT,
                        zorder=7,
                    )

        except Exception:
            pass

        # Force 2 decimals on the y-axis labels (account value chart only)
        try:
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"${y:,.2f}"))
        except Exception:
            pass


        # x labels: show a few timestamps (date + time) - evenly spaced, never overlapping duplicates
        n = len(points)
        want = 5
        if n <= want:
            idxs = list(range(n))
        else:
            step = (n - 1) / float(want - 1)
            idxs = []
            last = -1
            for j in range(want):
                i = int(round(j * step))
                if i <= last:
                    i = last + 1
                if i >= n:
                    i = n - 1
                idxs.append(i)
                last = i

        tick_x = [xs[i] for i in idxs]
        tick_lbl = [time.strftime("%Y-%m-%d\n%H:%M:%S", time.localtime(points[i][0])) for i in idxs]
        try:
            self.ax.minorticks_off()
            self.ax.set_xticks(tick_x)
            self.ax.set_xticklabels(tick_lbl)
            self.ax.tick_params(axis="x", labelsize=8)
        except Exception:
            pass





        self.ax.set_xlim(-0.5, (len(points) - 0.5) + 0.6)

        try:
            self.ax.set_title(f"Account Value ({_fmt_money(ys[-1])})", color=DARK_FG)
        except Exception:
            self.ax.set_title("Account Value", color=DARK_FG)

        try:
            self.last_update_label.config(
                text=f"Last: {time.strftime('%H:%M:%S', time.localtime(points[-1][0]))}"
            )
        except Exception:
            self.last_update_label.config(text="Last: N/A")

        self.canvas.draw_idle()

    def export_png(self, path: str) -> bool:
        try:
            self.fig.savefig(path, dpi=160, facecolor=self.fig.get_facecolor())
            return True
        except Exception:
            return False



# -----------------------------
# Hub App
# -----------------------------

@dataclass
class ProcInfo:
    name: str
    path: str
    proc: Optional[subprocess.Popen] = None



@dataclass
class LogProc:
    """
    A running process with a live log queue for stdout/stderr lines.
    """
    info: ProcInfo
    log_q: "queue.Queue[str]"
    thread: Optional[threading.Thread] = None
    is_trainer: bool = False
    coin: Optional[str] = None
