import base64
import datetime
import json
import uuid
import time
import math
from typing import Any, Dict, List, Optional
import requests
from nacl.signing import SigningKey
import os
import glob
import colorama
from colorama import Fore, Style
import traceback
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from app.credential_utils import (
	get_robinhood_creds_from_env,
	get_robinhood_creds_from_files,
	normalize_start_allocation_pct,
)
from app.automation_policy import build_market_automation_policy
from app.http_utils import parse_retry_after_value
from app.opportunity_allocator import evaluate_cross_market_allocation
from app.path_utils import resolve_runtime_paths, resolve_settings_path, read_settings_file, log_once
from app.settings_utils import normalize_settings_profile, sanitize_settings
from app.trade_quality import evaluate_trade_quality

# -----------------------------
# GUI HUB OUTPUTS
# -----------------------------
BASE_DIR, _RESOLVED_SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "pt_trader")

TRADER_DETAIL_PATH = os.path.join(HUB_DATA_DIR, "trader_data.json")
TRADE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "trade_history.jsonl")
PNL_LEDGER_PATH = os.path.join(HUB_DATA_DIR, "pnl_ledger.json")
ACCOUNT_VALUE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "account_value_history.jsonl")
CURRENT_PRICE_DIR = os.path.join(HUB_DATA_DIR, "current_prices")
os.makedirs(CURRENT_PRICE_DIR, exist_ok=True)
MANUAL_CRYPTO_ORDERS_DIR = os.path.join(HUB_DATA_DIR, "crypto_manual_orders")
MANUAL_CRYPTO_ORDER_RESULTS_PATH = os.path.join(HUB_DATA_DIR, "crypto_manual_order_results.jsonl")
os.makedirs(MANUAL_CRYPTO_ORDERS_DIR, exist_ok=True)
RUNTIME_STATE_PATH = os.path.join(HUB_DATA_DIR, "runtime_state.json")
CRYPTO_DYNAMIC_STATUS_PATH = os.path.join(HUB_DATA_DIR, "crypto_dynamic_status.json")
CRYPTO_MARKET_DIR = os.path.join(HUB_DATA_DIR, "crypto")
CRYPTO_EXECUTION_AUDIT_PATH = os.path.join(CRYPTO_MARKET_DIR, "execution_audit.jsonl")
os.makedirs(CRYPTO_MARKET_DIR, exist_ok=True)



# Initialize colorama
colorama.init(autoreset=True)

# -----------------------------
# GUI SETTINGS (coins list + main_neural_dir)
# -----------------------------
_GUI_SETTINGS_PATH = _RESOLVED_SETTINGS_PATH

_gui_settings_cache = {
	"mtime": None,
	"path": None,
	"coins": ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE'],  # fallback defaults
	"main_neural_dir": None,
	"trade_start_level": 3,
	"start_allocation_pct": 0.5,
	"dca_multiplier": 2.0,
	"dca_levels": [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0],
	"max_dca_buys_per_24h": 2,
	"crypto_max_open_positions": 8,

	# Trailing PM settings (defaults match previous hardcoded behavior)
	"pm_start_pct_no_dca": 5.0,
	"pm_start_pct_with_dca": 2.5,
	"trailing_gap_pct": 0.5,
	"max_position_usd_per_coin": 0.0,
	"max_total_exposure_pct": 0.0,
	"crypto_trader_loop_sleep_s": 1.0,
	"crypto_trader_error_sleep_s": 1.5,
}

_SIGNAL_FILE_CACHE = {}  # path -> (mtime, int_value)
ENABLE_SIGNAL_CACHE_STATS = False
_SIGNAL_FILE_CACHE_STATS = {"hits": 0, "misses": 0}


def _read_int_file_cached(path: str, default: int = 0) -> int:
	try:
		mtime = os.path.getmtime(path)
	except Exception:
		return int(default)
	hit = _SIGNAL_FILE_CACHE.get(path)
	if hit and hit[0] == mtime:
		if ENABLE_SIGNAL_CACHE_STATS:
			_SIGNAL_FILE_CACHE_STATS["hits"] += 1
		return int(hit[1])
	try:
		with open(path, "r", encoding="utf-8") as f:
			raw = f.read().strip()
		val = int(float(raw))
	except Exception:
		val = int(default)
	if ENABLE_SIGNAL_CACHE_STATS:
		_SIGNAL_FILE_CACHE_STATS["misses"] += 1
	_SIGNAL_FILE_CACHE[path] = (mtime, val)
	return val







def _load_gui_settings() -> dict:
	"""
	Reads gui_settings.json and returns a dict with:
	- coins: uppercased list
	- main_neural_dir: string (may be None)
	Caches by mtime so it is cheap to call frequently.
	"""
	try:
		settings_path = resolve_settings_path(BASE_DIR) or _GUI_SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
		if not os.path.isfile(settings_path):
			return dict(_gui_settings_cache)

		mtime = os.path.getmtime(settings_path)
		if _gui_settings_cache["mtime"] == mtime and _gui_settings_cache.get("path") == settings_path:
			return dict(_gui_settings_cache)

		data = read_settings_file(settings_path, module_name="pt_trader") or {}

		coins = data.get("coins", None)
		if not isinstance(coins, list) or not coins:
			coins = list(_gui_settings_cache["coins"])
		coins = [str(c).strip().upper() for c in coins if str(c).strip()]
		if not coins:
			coins = list(_gui_settings_cache["coins"])

		main_neural_dir = data.get("main_neural_dir", None)
		if isinstance(main_neural_dir, str):
			main_neural_dir = main_neural_dir.strip() or None
		else:
			main_neural_dir = None

		trade_start_level = data.get("trade_start_level", _gui_settings_cache.get("trade_start_level", 3))
		try:
			trade_start_level = int(float(trade_start_level))
		except Exception:
			trade_start_level = int(_gui_settings_cache.get("trade_start_level", 3))
		trade_start_level = max(1, min(trade_start_level, 7))

		start_allocation_pct = normalize_start_allocation_pct(
			data.get("start_allocation_pct", _gui_settings_cache.get("start_allocation_pct", 0.5)),
			default_pct=float(_gui_settings_cache.get("start_allocation_pct", 0.5)),
		)

		dca_multiplier = data.get("dca_multiplier", _gui_settings_cache.get("dca_multiplier", 2.0))
		try:
			dca_multiplier = float(str(dca_multiplier).strip())
		except Exception:
			dca_multiplier = float(_gui_settings_cache.get("dca_multiplier", 2.0))
		if dca_multiplier < 0.0:
			dca_multiplier = 0.0

		dca_levels = data.get("dca_levels", _gui_settings_cache.get("dca_levels", [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]))
		if not isinstance(dca_levels, list) or not dca_levels:
			dca_levels = list(_gui_settings_cache.get("dca_levels", [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]))
		parsed = []
		for v in dca_levels:
			try:
				parsed.append(float(v))
			except Exception:
				pass
		if parsed:
			dca_levels = parsed
		else:
			dca_levels = list(_gui_settings_cache.get("dca_levels", [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]))

		max_dca_buys_per_24h = data.get("max_dca_buys_per_24h", _gui_settings_cache.get("max_dca_buys_per_24h", 2))
		try:
			max_dca_buys_per_24h = int(float(max_dca_buys_per_24h))
		except Exception:
			max_dca_buys_per_24h = int(_gui_settings_cache.get("max_dca_buys_per_24h", 2))
		if max_dca_buys_per_24h < 0:
			max_dca_buys_per_24h = 0

		crypto_max_open_positions = data.get("crypto_max_open_positions", _gui_settings_cache.get("crypto_max_open_positions", 8))
		try:
			crypto_max_open_positions = int(float(crypto_max_open_positions))
		except Exception:
			crypto_max_open_positions = int(_gui_settings_cache.get("crypto_max_open_positions", 8))
		if crypto_max_open_positions < 1:
			crypto_max_open_positions = 1


		# --- Trailing PM settings ---
		pm_start_pct_no_dca = data.get("pm_start_pct_no_dca", _gui_settings_cache.get("pm_start_pct_no_dca", 5.0))
		try:
			pm_start_pct_no_dca = float(str(pm_start_pct_no_dca).replace("%", "").strip())
		except Exception:
			pm_start_pct_no_dca = float(_gui_settings_cache.get("pm_start_pct_no_dca", 5.0))
		if pm_start_pct_no_dca < 0.0:
			pm_start_pct_no_dca = 0.0

		pm_start_pct_with_dca = data.get("pm_start_pct_with_dca", _gui_settings_cache.get("pm_start_pct_with_dca", 2.5))
		try:
			pm_start_pct_with_dca = float(str(pm_start_pct_with_dca).replace("%", "").strip())
		except Exception:
			pm_start_pct_with_dca = float(_gui_settings_cache.get("pm_start_pct_with_dca", 2.5))
		if pm_start_pct_with_dca < 0.0:
			pm_start_pct_with_dca = 0.0

		trailing_gap_pct = data.get("trailing_gap_pct", _gui_settings_cache.get("trailing_gap_pct", 0.5))
		try:
			trailing_gap_pct = float(str(trailing_gap_pct).replace("%", "").strip())
		except Exception:
			trailing_gap_pct = float(_gui_settings_cache.get("trailing_gap_pct", 0.5))
		if trailing_gap_pct < 0.0:
			trailing_gap_pct = 0.0

		max_position_usd_per_coin = data.get("max_position_usd_per_coin", _gui_settings_cache.get("max_position_usd_per_coin", 0.0))
		try:
			max_position_usd_per_coin = float(str(max_position_usd_per_coin).replace("%", "").strip())
		except Exception:
			max_position_usd_per_coin = float(_gui_settings_cache.get("max_position_usd_per_coin", 0.0))
		if max_position_usd_per_coin < 0.0:
			max_position_usd_per_coin = 0.0

		max_total_exposure_pct = data.get("max_total_exposure_pct", _gui_settings_cache.get("max_total_exposure_pct", 0.0))
		try:
			max_total_exposure_pct = float(str(max_total_exposure_pct).replace("%", "").strip())
		except Exception:
			max_total_exposure_pct = float(_gui_settings_cache.get("max_total_exposure_pct", 0.0))
		if max_total_exposure_pct < 0.0:
			max_total_exposure_pct = 0.0

		crypto_trader_loop_sleep_s = data.get("crypto_trader_loop_sleep_s", _gui_settings_cache.get("crypto_trader_loop_sleep_s", 1.0))
		try:
			crypto_trader_loop_sleep_s = float(str(crypto_trader_loop_sleep_s).strip())
		except Exception:
			crypto_trader_loop_sleep_s = float(_gui_settings_cache.get("crypto_trader_loop_sleep_s", 1.0))
		crypto_trader_loop_sleep_s = max(0.25, crypto_trader_loop_sleep_s)

		crypto_trader_error_sleep_s = data.get("crypto_trader_error_sleep_s", _gui_settings_cache.get("crypto_trader_error_sleep_s", 1.5))
		try:
			crypto_trader_error_sleep_s = float(str(crypto_trader_error_sleep_s).strip())
		except Exception:
			crypto_trader_error_sleep_s = float(_gui_settings_cache.get("crypto_trader_error_sleep_s", 1.5))
		crypto_trader_error_sleep_s = max(0.5, crypto_trader_error_sleep_s)


		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["path"] = settings_path
		_gui_settings_cache["coins"] = coins
		_gui_settings_cache["main_neural_dir"] = main_neural_dir
		_gui_settings_cache["trade_start_level"] = trade_start_level
		_gui_settings_cache["start_allocation_pct"] = start_allocation_pct
		_gui_settings_cache["dca_multiplier"] = dca_multiplier
		_gui_settings_cache["dca_levels"] = dca_levels
		_gui_settings_cache["max_dca_buys_per_24h"] = max_dca_buys_per_24h
		_gui_settings_cache["crypto_max_open_positions"] = crypto_max_open_positions

		_gui_settings_cache["pm_start_pct_no_dca"] = pm_start_pct_no_dca
		_gui_settings_cache["pm_start_pct_with_dca"] = pm_start_pct_with_dca
		_gui_settings_cache["trailing_gap_pct"] = trailing_gap_pct
		_gui_settings_cache["max_position_usd_per_coin"] = max_position_usd_per_coin
		_gui_settings_cache["max_total_exposure_pct"] = max_total_exposure_pct
		_gui_settings_cache["crypto_trader_loop_sleep_s"] = crypto_trader_loop_sleep_s
		_gui_settings_cache["crypto_trader_error_sleep_s"] = crypto_trader_error_sleep_s


		return {
			"mtime": mtime,
			"coins": list(coins),
			"main_neural_dir": main_neural_dir,
			"trade_start_level": trade_start_level,
			"start_allocation_pct": start_allocation_pct,
			"dca_multiplier": dca_multiplier,
			"dca_levels": list(dca_levels),
			"max_dca_buys_per_24h": max_dca_buys_per_24h,
			"crypto_max_open_positions": crypto_max_open_positions,

			"pm_start_pct_no_dca": pm_start_pct_no_dca,
			"pm_start_pct_with_dca": pm_start_pct_with_dca,
			"trailing_gap_pct": trailing_gap_pct,
			"max_position_usd_per_coin": max_position_usd_per_coin,
			"max_total_exposure_pct": max_total_exposure_pct,
			"crypto_trader_loop_sleep_s": crypto_trader_loop_sleep_s,
			"crypto_trader_error_sleep_s": crypto_trader_error_sleep_s,
		}




	except Exception:
		return dict(_gui_settings_cache)


def _build_base_paths(main_dir_in: str, coins_in: list) -> dict:
	"""
	Safety rule:
	- every coin (including BTC) uses <main_dir>/<SYM>
	- create missing coin folders so runtime outputs stay out of repo root
	"""
	out = {}
	try:
		for sym in coins_in:
			sym = str(sym).strip().upper()
			if not sym:
				continue
			sub = os.path.join(main_dir_in, sym)
			os.makedirs(sub, exist_ok=True)
			out[sym] = sub
	except Exception:
		pass
	if "BTC" not in out:
		try:
			btc_dir = os.path.join(main_dir_in, "BTC")
			os.makedirs(btc_dir, exist_ok=True)
			out["BTC"] = btc_dir
		except Exception:
			out["BTC"] = os.path.join(main_dir_in, "BTC")
	return out


# Live globals (will be refreshed inside manage_trades())
crypto_symbols = ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE']

# Default main_dir behavior if settings are missing
main_dir = BASE_DIR
base_paths = {"BTC": os.path.join(main_dir, "BTC")}
TRADE_START_LEVEL = 3
START_ALLOC_PCT = 0.5
DCA_MULTIPLIER = 2.0
DCA_LEVELS = [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]
MAX_DCA_BUYS_PER_24H = 2
MAX_OPEN_POSITIONS = 8
MAX_POSITION_USD_PER_COIN = 0.0
MAX_TOTAL_EXPOSURE_PCT = 0.0
CRYPTO_TRADER_LOOP_SLEEP_S = 1.0
CRYPTO_TRADER_ERROR_SLEEP_S = 1.5

# Trailing PM hot-reload globals (defaults match previous hardcoded behavior)
TRAILING_GAP_PCT = 0.5
PM_START_PCT_NO_DCA = 5.0
PM_START_PCT_WITH_DCA = 2.5



_last_settings_mtime = None




def _refresh_paths_and_symbols():
	"""
	Hot-reload GUI settings while trader is running.
	Updates globals: crypto_symbols, main_dir, base_paths,
	                TRADE_START_LEVEL, START_ALLOC_PCT, DCA_MULTIPLIER, DCA_LEVELS, MAX_DCA_BUYS_PER_24H,
	                TRAILING_GAP_PCT, PM_START_PCT_NO_DCA, PM_START_PCT_WITH_DCA
	"""
	global crypto_symbols, main_dir, base_paths
	global TRADE_START_LEVEL, START_ALLOC_PCT, DCA_MULTIPLIER, DCA_LEVELS, MAX_DCA_BUYS_PER_24H, MAX_OPEN_POSITIONS
	global MAX_POSITION_USD_PER_COIN, MAX_TOTAL_EXPOSURE_PCT
	global TRAILING_GAP_PCT, PM_START_PCT_NO_DCA, PM_START_PCT_WITH_DCA
	global CRYPTO_TRADER_LOOP_SLEEP_S, CRYPTO_TRADER_ERROR_SLEEP_S
	global _last_settings_mtime


	s = _load_gui_settings()
	mtime = s.get("mtime", None)

	# If settings file doesn't exist, keep current defaults
	if mtime is None:
		return

	if _last_settings_mtime == mtime:
		return

	_last_settings_mtime = mtime

	coins = s.get("coins") or list(crypto_symbols)
	mndir = s.get("main_neural_dir") or main_dir
	TRADE_START_LEVEL = max(1, min(int(s.get("trade_start_level", TRADE_START_LEVEL) or TRADE_START_LEVEL), 7))
	START_ALLOC_PCT = normalize_start_allocation_pct(
		s.get("start_allocation_pct", START_ALLOC_PCT),
		default_pct=float(START_ALLOC_PCT),
	)

	DCA_MULTIPLIER = float(s.get("dca_multiplier", DCA_MULTIPLIER) or DCA_MULTIPLIER)
	if DCA_MULTIPLIER < 0.0:
		DCA_MULTIPLIER = 0.0

	DCA_LEVELS = list(s.get("dca_levels", DCA_LEVELS) or DCA_LEVELS)

	try:
		MAX_DCA_BUYS_PER_24H = int(float(s.get("max_dca_buys_per_24h", MAX_DCA_BUYS_PER_24H) or MAX_DCA_BUYS_PER_24H))
	except Exception:
		MAX_DCA_BUYS_PER_24H = int(MAX_DCA_BUYS_PER_24H)
	if MAX_DCA_BUYS_PER_24H < 0:
		MAX_DCA_BUYS_PER_24H = 0

	try:
		MAX_OPEN_POSITIONS = int(float(s.get("crypto_max_open_positions", MAX_OPEN_POSITIONS) or MAX_OPEN_POSITIONS))
	except Exception:
		MAX_OPEN_POSITIONS = int(MAX_OPEN_POSITIONS)
	if MAX_OPEN_POSITIONS < 1:
		MAX_OPEN_POSITIONS = 1


	# Trailing PM hot-reload values
	TRAILING_GAP_PCT = float(s.get("trailing_gap_pct", TRAILING_GAP_PCT) or TRAILING_GAP_PCT)
	if TRAILING_GAP_PCT < 0.0:
		TRAILING_GAP_PCT = 0.0

	PM_START_PCT_NO_DCA = float(s.get("pm_start_pct_no_dca", PM_START_PCT_NO_DCA) or PM_START_PCT_NO_DCA)
	if PM_START_PCT_NO_DCA < 0.0:
		PM_START_PCT_NO_DCA = 0.0

	PM_START_PCT_WITH_DCA = float(s.get("pm_start_pct_with_dca", PM_START_PCT_WITH_DCA) or PM_START_PCT_WITH_DCA)
	if PM_START_PCT_WITH_DCA < 0.0:
		PM_START_PCT_WITH_DCA = 0.0

	MAX_POSITION_USD_PER_COIN = float(s.get("max_position_usd_per_coin", MAX_POSITION_USD_PER_COIN) or 0.0)
	if MAX_POSITION_USD_PER_COIN < 0.0:
		MAX_POSITION_USD_PER_COIN = 0.0

	MAX_TOTAL_EXPOSURE_PCT = float(s.get("max_total_exposure_pct", MAX_TOTAL_EXPOSURE_PCT) or 0.0)
	if MAX_TOTAL_EXPOSURE_PCT < 0.0:
		MAX_TOTAL_EXPOSURE_PCT = 0.0

	CRYPTO_TRADER_LOOP_SLEEP_S = float(s.get("crypto_trader_loop_sleep_s", CRYPTO_TRADER_LOOP_SLEEP_S) or CRYPTO_TRADER_LOOP_SLEEP_S)
	if CRYPTO_TRADER_LOOP_SLEEP_S < 0.25:
		CRYPTO_TRADER_LOOP_SLEEP_S = 0.25

	CRYPTO_TRADER_ERROR_SLEEP_S = float(s.get("crypto_trader_error_sleep_s", CRYPTO_TRADER_ERROR_SLEEP_S) or CRYPTO_TRADER_ERROR_SLEEP_S)
	if CRYPTO_TRADER_ERROR_SLEEP_S < 0.5:
		CRYPTO_TRADER_ERROR_SLEEP_S = 0.5


	# Keep it safe if folder isn't real on this machine
	if not os.path.isdir(mndir):
		mndir = BASE_DIR

	crypto_symbols = list(coins)
	main_dir = mndir
	base_paths = _build_base_paths(main_dir, crypto_symbols)






#API STUFF
API_KEY = ""
BASE64_PRIVATE_KEY = ""

try:
    API_KEY, BASE64_PRIVATE_KEY = get_robinhood_creds_from_env()
    if not API_KEY or not BASE64_PRIVATE_KEY:
        API_KEY, BASE64_PRIVATE_KEY = get_robinhood_creds_from_files(BASE_DIR)
except Exception:
    API_KEY = ""
    BASE64_PRIVATE_KEY = ""

if not API_KEY or not BASE64_PRIVATE_KEY:
    print(
        "\n[PowerTrader] Robinhood API credentials not found.\n"
        "Set env vars POWERTRADER_RH_API_KEY + POWERTRADER_RH_PRIVATE_B64, or use credential files in keys/.\n"
        "Open the GUI and go to Settings → Robinhood API → Setup / Update.\n"
        "That wizard will generate your keypair, tell you where to paste the public key on Robinhood,\n"
        "and will save keys/r_key.txt + keys/r_secret.txt so this trader can authenticate.\n"
    )
    raise SystemExit(1)

class CryptoAPITrading:
    def __init__(self):
        # keep a copy of the folder map (same idea as trader.py)
        self.path_map = dict(base_paths)

        self.api_key = API_KEY
        private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"

        self.dca_levels_triggered = {}  # Track DCA levels for each crypto
        self.dca_levels = list(DCA_LEVELS)  # Hard DCA triggers (percent PnL)


        # --- Trailing profit margin (per-coin state) ---
        # Each coin keeps its own trailing PM line, peak, and "was above line" flag.
        self.trailing_pm = {}  # { "BTC": {"active": bool, "line": float, "peak": float, "was_above": bool}, . }
        self.trailing_gap_pct = float(TRAILING_GAP_PCT)  # % trail gap behind peak
        self.pm_start_pct_no_dca = float(PM_START_PCT_NO_DCA)
        self.pm_start_pct_with_dca = float(PM_START_PCT_WITH_DCA)

        # Track trailing-related settings so we can reset trailing state if they change
        self._last_trailing_settings_sig = (
            float(self.trailing_gap_pct),
            float(self.pm_start_pct_no_dca),
            float(self.pm_start_pct_with_dca),
        )



        self.cost_basis = self.calculate_cost_basis()  # Initialize cost basis at startup
        self.initialize_dca_levels()  # Initialize DCA levels based on historical buy orders

        # GUI hub persistence
        self._pnl_ledger = self._load_pnl_ledger()
        self._reconcile_pending_orders()


        # Cache last known bid/ask per symbol so transient API misses don't zero out account value
        self._last_good_bid_ask = {}

        # Cache last *complete* account snapshot so transient holdings/price misses can't write a bogus low value
        self._last_good_account_snapshot = {
            "total_account_value": None,
            "buying_power": None,
            "holdings_sell_value": None,
            "holdings_buy_value": None,
            "percent_in_trade": None,
        }
        self._last_good_holdings_results: List[Dict[str, Any]] = []
        self._last_good_holdings_ts = 0.0
        self._last_good_positions_snapshot: Dict[str, Dict[str, Any]] = {}
        self._bootstrap_last_good_snapshot_from_disk()
        self._seed_cost_basis_from_fallbacks()
        if not self.dca_levels_triggered:
            for symbol, pos in self._last_good_positions_snapshot.items():
                try:
                    stages = int(float((pos or {}).get("dca_triggered_stages", 0) or 0.0))
                except Exception:
                    stages = 0
                if stages > 0:
                    self.dca_levels_triggered[str(symbol).upper().strip()] = list(range(stages))

        # --- DCA rate-limit (per trade, per coin, rolling 24h window) ---
        self.max_dca_buys_per_24h = int(MAX_DCA_BUYS_PER_24H)
        self.dca_window_seconds = 24 * 60 * 60

        self._dca_buy_ts = {}         # { "BTC": [ts, ts, ...] } (DCA buys only)
        self._dca_last_sell_ts = {}   # { "BTC": ts_of_last_sell }
        self._seed_dca_window_from_history()
        self._last_entry_ts = {}      # { "BTC": ts_of_last_successful_buy }
        self._last_exit_ts = {}       # { "BTC": ts_of_last_successful_sell }
        self.entry_cooldown_seconds = 30 * 60
        self._last_account_value_history_write_ts = 0.0
        self._rate_limited_log_ts = {}
        self._status_note = ""
        self._loop_sleep_ok = float(CRYPTO_TRADER_LOOP_SLEEP_S)
        self._loop_sleep_error = float(CRYPTO_TRADER_ERROR_SLEEP_S)
        self._stale_alignment_streaks: Dict[str, int] = {}

    def _seed_cost_basis_from_fallbacks(self) -> None:
        cost_basis = getattr(self, "cost_basis", {})
        if not isinstance(cost_basis, dict):
            self.cost_basis = {}
        else:
            self.cost_basis = cost_basis
        symbols = set(self._last_good_positions_snapshot.keys()) | self._ledger_open_position_bases()
        for symbol in symbols:
            coin = str(symbol or "").strip().upper()
            if not coin:
                continue
            try:
                existing = float(self.cost_basis.get(coin, 0.0) or 0.0)
            except Exception:
                existing = 0.0
            if existing > 0.0:
                continue
            fallback = self._fallback_avg_cost_basis(coin, quantity=0.0)
            if fallback > 0.0:
                self.cost_basis[coin] = float(fallback)








    def _rotate_jsonl_if_needed(
        self,
        path: str,
        max_bytes: int = 30 * 1024 * 1024,
        keep_files: int = 10,
        max_age_days: int = 30,
    ) -> None:
        if path not in (TRADE_HISTORY_PATH, ACCOUNT_VALUE_HISTORY_PATH):
            return
        try:
            if (not os.path.isfile(path)) or os.path.getsize(path) <= int(max_bytes):
                return
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            rotated = f"{path}.{stamp}.jsonl"
            os.replace(path, rotated)
            base_name = os.path.basename(path)
            folder = os.path.dirname(path)
            pattern = os.path.join(folder, f"{base_name}.*.jsonl")
            rotated_files = sorted(glob.glob(pattern), reverse=True)
            cutoff = time.time() - (float(max_age_days) * 86400.0)
            for old_path in rotated_files[keep_files:]:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
            for old_path in rotated_files[:keep_files]:
                try:
                    if os.path.getmtime(old_path) < cutoff:
                        os.remove(old_path)
                except OSError:
                    pass
        except OSError as exc:
            self._log_rate_limited(
                f"jsonl_rotate_{path}",
                f"[pt_trader._rotate_jsonl_if_needed] path={path} {type(exc).__name__}: {exc}",
                every_s=60.0,
            )

    def _atomic_write_json(self, path: str, data: dict) -> None:
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except (PermissionError, OSError, TypeError, ValueError) as exc:
            self._log_rate_limited(
                f"atomic_json_{path}",
                f"[pt_trader._atomic_write_json] path={path} {type(exc).__name__}: {exc}",
                every_s=60.0,
            )

    def _append_jsonl(self, path: str, obj: dict) -> None:
        try:
            self._rotate_jsonl_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except (PermissionError, OSError, TypeError, ValueError) as exc:
            self._log_rate_limited(
                f"append_jsonl_{path}",
                f"[pt_trader._append_jsonl] path={path} {type(exc).__name__}: {exc}",
                every_s=60.0,
            )

    def _log_rate_limited(self, key: str, message: str, every_s: float = 15.0) -> None:
        try:
            now = time.time()
            last = float(self._rate_limited_log_ts.get(key, 0.0) or 0.0)
            if (now - last) < float(every_s):
                return
            self._rate_limited_log_ts[key] = now
            print(message)
        except Exception:
            pass

    def _set_status_note(self, note: str) -> None:
        try:
            self._status_note = str(note or "").strip()
        except Exception:
            self._status_note = ""

    @staticmethod
    def _copy_holdings_results(holdings_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in list(holdings_results or []):
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    @staticmethod
    def _active_holding_rows(holdings_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []
        for row in list(holdings_results or []):
            if not isinstance(row, dict):
                continue
            try:
                qty = float(row.get("total_quantity", 0.0) or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0.0:
                continue
            coin = str(row.get("asset_code", "") or "").strip().upper()
            if not coin or coin == "USDC":
                continue
            active.append(dict(row))
        return active

    def _ledger_open_position_bases(self) -> set:
        out = set()
        try:
            open_positions = self._pnl_ledger.get("open_positions", {})
        except Exception:
            open_positions = {}
        if not isinstance(open_positions, dict):
            return out
        for base, row in open_positions.items():
            try:
                qty = float((row or {}).get("qty", 0.0) or 0.0)
                usd_cost = float((row or {}).get("usd_cost", 0.0) or 0.0)
            except Exception:
                qty = 0.0
                usd_cost = 0.0
            if qty > 0.0 or usd_cost > 1e-6:
                coin = str(base or "").strip().upper()
                if coin:
                    out.add(coin)
        return out

    def _ledger_avg_cost_basis(self, base_symbol: str, quantity: float = 0.0) -> float:
        coin = str(base_symbol or "").strip().upper()
        if not coin:
            return 0.0
        try:
            open_positions = self._pnl_ledger.get("open_positions", {})
        except Exception:
            open_positions = {}
        if not isinstance(open_positions, dict):
            return 0.0
        row = open_positions.get(coin, {})
        if not isinstance(row, dict):
            return 0.0
        try:
            usd_cost = float(row.get("usd_cost", 0.0) or 0.0)
        except Exception:
            usd_cost = 0.0
        try:
            qty = float(quantity or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0.0:
            try:
                qty = float(row.get("qty", 0.0) or 0.0)
            except Exception:
                qty = 0.0
        if usd_cost > 0.0 and qty > 0.0:
            return float(usd_cost / qty)
        return 0.0

    def _fallback_avg_cost_basis(self, base_symbol: str, quantity: float = 0.0) -> float:
        coin = str(base_symbol or "").strip().upper()
        if not coin:
            return 0.0
        try:
            direct = float((getattr(self, "cost_basis", {}) or {}).get(coin, 0.0) or 0.0)
        except Exception:
            direct = 0.0
        if direct > 0.0:
            return float(direct)
        try:
            snap = self._last_good_positions_snapshot.get(coin, {})
        except Exception:
            snap = {}
        if isinstance(snap, dict):
            try:
                snap_avg = float(snap.get("avg_cost_basis", 0.0) or 0.0)
            except Exception:
                snap_avg = 0.0
            if snap_avg > 0.0:
                return float(snap_avg)
        ledger_avg = self._ledger_avg_cost_basis(coin, quantity=quantity)
        if ledger_avg > 0.0:
            return float(ledger_avg)
        return 0.0

    def _bootstrap_last_good_snapshot_from_disk(self) -> None:
        try:
            with open(TRADER_DETAIL_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        account = payload.get("account", {}) if isinstance(payload.get("account", {}), dict) else {}
        snapshot = dict(self._last_good_account_snapshot)
        for key in ("total_account_value", "buying_power", "holdings_sell_value", "holdings_buy_value", "percent_in_trade"):
            try:
                raw_val = account.get(key, None)
                if raw_val is None or raw_val == "":
                    continue
                snapshot[key] = float(raw_val)
            except Exception:
                continue
        self._last_good_account_snapshot = snapshot

        positions = payload.get("positions", {}) if isinstance(payload.get("positions", {}), dict) else {}
        seeded_positions: Dict[str, Dict[str, Any]] = {}
        seeded_holdings: List[Dict[str, Any]] = []
        for symbol, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            coin = str(symbol or "").strip().upper()
            if not coin:
                continue
            try:
                qty = float(pos.get("quantity", 0.0) or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0.0:
                continue
            seeded_positions[coin] = dict(pos)
            seeded_holdings.append({"asset_code": coin, "total_quantity": qty})

        if not seeded_holdings:
            try:
                open_positions = self._pnl_ledger.get("open_positions", {})
            except Exception:
                open_positions = {}
            if isinstance(open_positions, dict):
                for symbol, row in open_positions.items():
                    try:
                        qty = float((row or {}).get("qty", 0.0) or 0.0)
                    except Exception:
                        qty = 0.0
                    coin = str(symbol or "").strip().upper()
                    if coin and qty > 0.0:
                        seeded_holdings.append({"asset_code": coin, "total_quantity": qty})

        if seeded_positions:
            self._last_good_positions_snapshot = seeded_positions
        if seeded_holdings:
            self._last_good_holdings_results = self._copy_holdings_results(seeded_holdings)
            try:
                self._last_good_holdings_ts = float(payload.get("timestamp", 0.0) or 0.0)
            except Exception:
                self._last_good_holdings_ts = time.time()
            if self._last_good_holdings_ts <= 0.0:
                self._last_good_holdings_ts = time.time()
        self._seed_cost_basis_from_fallbacks()

    def _remember_good_holdings(self, holdings_results: List[Dict[str, Any]]) -> None:
        active_rows = self._active_holding_rows(holdings_results)
        if not active_rows:
            return
        self._last_good_holdings_results = self._copy_holdings_results(active_rows)
        self._last_good_holdings_ts = time.time()

    def _should_reuse_cached_holdings(
        self,
        holdings_results: List[Dict[str, Any]],
        recent_trade: bool = False,
    ) -> bool:
        if recent_trade:
            return False
        if self._active_holding_rows(holdings_results):
            return False
        cached_rows = self._active_holding_rows(getattr(self, "_last_good_holdings_results", []))
        if not cached_rows:
            return False
        if self._ledger_open_position_bases():
            return True
        age_s = time.time() - float(getattr(self, "_last_good_holdings_ts", 0.0) or 0.0)
        return age_s <= 90.0

    def _resolve_holdings_results(self, holdings: Any, recent_trade: bool = False) -> tuple:
        holdings_results: List[Dict[str, Any]] = []
        if isinstance(holdings, dict):
            raw_results = holdings.get("results", [])
            if isinstance(raw_results, list):
                holdings_results = self._copy_holdings_results(raw_results)
        if self._active_holding_rows(holdings_results):
            self._remember_good_holdings(holdings_results)
            return holdings_results, False
        if self._should_reuse_cached_holdings(holdings_results, recent_trade=recent_trade):
            if not str(getattr(self, "_status_note", "") or "").strip():
                self._set_status_note("Broker holdings snapshot incomplete; using cached crypto positions.")
            return self._copy_holdings_results(self._last_good_holdings_results), True
        return holdings_results, False

    def _can_place_buy(
        self,
        base_symbol: str,
        buy_amount_usd: float,
        current_position_value_usd: float,
        total_account_value: float,
        holdings_sell_value: float,
    ) -> bool:
        base = str(base_symbol).upper().strip()
        buy_amt = float(buy_amount_usd or 0.0)
        cur_val = float(current_position_value_usd or 0.0)
        acct_val = float(total_account_value or 0.0)
        held_val = float(holdings_sell_value or 0.0)

        if buy_amt <= 0.0:
            return False

        if float(MAX_POSITION_USD_PER_COIN or 0.0) > 0.0:
            projected = cur_val + buy_amt
            if projected > float(MAX_POSITION_USD_PER_COIN):
                msg = (
                    f"Buy blocked for {base}: projected position ${projected:.2f} exceeds "
                    f"max_position_usd_per_coin ${float(MAX_POSITION_USD_PER_COIN):.2f}."
                )
                self._set_status_note(msg)
                self._log_rate_limited(f"cap_pos_{base}", msg)
                return False

        if float(MAX_TOTAL_EXPOSURE_PCT or 0.0) > 0.0 and acct_val > 0.0:
            projected_pct = ((held_val + buy_amt) / acct_val) * 100.0
            if projected_pct > float(MAX_TOTAL_EXPOSURE_PCT):
                msg = (
                    f"Buy blocked for {base}: projected exposure {projected_pct:.2f}% exceeds "
                    f"max_total_exposure_pct {float(MAX_TOTAL_EXPOSURE_PCT):.2f}%."
                )
                self._set_status_note(msg)
                self._log_rate_limited(f"cap_exp_{base}", msg)
                return False

        return True

    @staticmethod
    def _safe_read_json_file(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                row = json.load(f) or {}
            return row if isinstance(row, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _entry_fail_summary(reasons: List[str]) -> tuple[str, Dict[str, int]]:
        counts: Dict[str, int] = {}
        for reason in list(reasons or []):
            msg = str(reason or "").strip()
            if not msg:
                continue
            counts[msg] = int(counts.get(msg, 0) or 0) + 1
        if not counts:
            return "", {}
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]
        return str(top), counts

    def _append_execution_audit(self, row: Dict[str, Any]) -> None:
        try:
            payload = dict(row or {})
            payload.setdefault("ts", int(time.time()))
            payload.setdefault("date", time.strftime("%Y-%m-%d", time.localtime()))
            os.makedirs(os.path.dirname(CRYPTO_EXECUTION_AUDIT_PATH), exist_ok=True)
            with open(CRYPTO_EXECUTION_AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            pass

    @staticmethod
    def _profile_quality_thresholds(profile_key: Any) -> tuple[float, float]:
        pkey = normalize_settings_profile(profile_key, default="balanced")
        if pkey == "safe":
            return 38.0, 44.0
        if pkey == "aggressive":
            return 31.0, 35.0
        if pkey == "max_growth":
            return 29.0, 33.0
        return 34.0, 39.0

    @staticmethod
    def _crypto_signal_gate_rules(
        profile_key: Any,
        start_level: int,
        *,
        policy_mode: str = "",
        adaptive_dynamic_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        lvl = max(1, min(int(start_level or 3), 7))
        profile = normalize_settings_profile(profile_key, default="balanced")
        mode = str(policy_mode or "").strip().lower()
        min_long_count = lvl
        max_short_count = 0
        allow_dynamic_fallback = False
        min_dynamic_score = 0.0
        if profile == "max_growth":
            min_long_count = max(1, lvl - 1)
            if mode not in {"restricted", "safe"}:
                allow_dynamic_fallback = True
                min_dynamic_score = 1.05 if mode in {"aggressive_guarded", "balanced"} else 0.75
        elif profile == "aggressive":
            min_long_count = lvl
            if mode not in {"restricted", "safe"}:
                allow_dynamic_fallback = True
                min_dynamic_score = 1.10 if mode in {"aggressive", "aggressive_rotation"} else 1.35
        adaptive_floor = max(0.0, float(adaptive_dynamic_threshold or 0.0))
        if bool(allow_dynamic_fallback) and adaptive_floor > 0.0:
            min_dynamic_score = max(float(min_dynamic_score), float(adaptive_floor))
        requirement_text = f"need long>={int(min_long_count)} and short<={int(max_short_count)}"
        if bool(allow_dynamic_fallback):
            requirement_text = (
                f"need short<={int(max_short_count)} and "
                f"(long>={int(min_long_count)} or dynamic_score>={float(min_dynamic_score):.2f})"
            )
        return {
            "profile": str(profile),
            "mode": str(mode),
            "min_long_count": int(min_long_count),
            "max_short_count": int(max_short_count),
            "allow_dynamic_fallback": bool(allow_dynamic_fallback),
            "min_dynamic_score": float(min_dynamic_score),
            "adaptive_dynamic_threshold": float(adaptive_floor),
            "requirement_text": str(requirement_text),
        }

    @classmethod
    def _evaluate_crypto_signal_gate(
        cls,
        *,
        profile_key: Any,
        start_level: int,
        buy_count: int,
        sell_count: int,
        dynamic_score: float,
        policy_mode: str = "",
        adaptive_dynamic_threshold: float = 0.0,
        calibration_prob: float = 0.0,
        min_calibration_prob: float = 0.0,
    ) -> Dict[str, Any]:
        rules = cls._crypto_signal_gate_rules(
            profile_key,
            start_level,
            policy_mode=policy_mode,
            adaptive_dynamic_threshold=float(adaptive_dynamic_threshold or 0.0),
        )
        min_long_count = int(rules.get("min_long_count", start_level) or start_level)
        max_short_count = int(rules.get("max_short_count", 0) or 0)
        allow_dynamic_fallback = bool(rules.get("allow_dynamic_fallback", False))
        min_dynamic_score = float(rules.get("min_dynamic_score", 0.0) or 0.0)
        bcount = int(buy_count or 0)
        scount = int(sell_count or 0)
        dyn = float(dynamic_score or 0.0)
        short_ok = scount <= max_short_count
        long_ok = bcount >= min_long_count
        dynamic_ok = bool(allow_dynamic_fallback and short_ok and dyn >= min_dynamic_score)
        calib_prob = float(calibration_prob or 0.0)
        min_calib = max(0.0, float(min_calibration_prob or 0.0))
        calib_applied = bool(min_calib > 0.0 and calib_prob > 0.0)
        calib_ok = bool((not calib_applied) or (calib_prob >= min_calib))
        passed = bool(short_ok and (long_ok or dynamic_ok) and calib_ok)
        gate_mode = "long_signal" if bool(passed and long_ok) else ("dynamic_score_fallback" if bool(passed and dynamic_ok) else "blocked")
        failure_reason = ""
        if not passed:
            if not short_ok:
                failure_reason = f"short pressure active (N{scount} > N{max_short_count})"
            elif not calib_ok:
                failure_reason = f"calibration confidence too low ({calib_prob:.3f} < {min_calib:.3f})"
            elif allow_dynamic_fallback:
                failure_reason = (
                    f"insufficient momentum (long=N{bcount}, dynamic={dyn:.3f}, "
                    f"need long>=N{min_long_count} or dynamic>={min_dynamic_score:.2f})"
                )
            else:
                failure_reason = f"long signal below threshold (N{bcount} < N{min_long_count})"
        return {
            "passed": bool(passed),
            "gate_mode": str(gate_mode),
            "failure_reason": str(failure_reason),
            "requirement_text": str(rules.get("requirement_text", "") or ""),
            "min_long_count": int(min_long_count),
            "max_short_count": int(max_short_count),
            "allow_dynamic_fallback": bool(allow_dynamic_fallback),
            "min_dynamic_score": float(min_dynamic_score),
            "buy_count": int(bcount),
            "sell_count": int(scount),
            "dynamic_score": float(dyn),
            "adaptive_dynamic_threshold": float(rules.get("adaptive_dynamic_threshold", 0.0) or 0.0),
            "calibration_prob": float(calib_prob),
            "min_calibration_prob": float(min_calib),
            "calibration_gate_applied": bool(calib_applied),
        }

    @classmethod
    def _evaluate_crypto_entry_alignment_gate(
        cls,
        *,
        profile_key: Any,
        start_level: int,
        buy_count: int,
        sell_count: int,
        dynamic_score: float,
        policy_mode: str = "",
        adaptive_dynamic_threshold: float = 0.0,
        calibration_prob: float = 0.0,
        min_calibration_prob: float = 0.0,
    ) -> Dict[str, Any]:
        signal_eval = cls._evaluate_crypto_signal_gate(
            profile_key=profile_key,
            start_level=int(start_level),
            buy_count=int(buy_count),
            sell_count=int(sell_count),
            dynamic_score=float(dynamic_score),
            policy_mode=str(policy_mode or ""),
            adaptive_dynamic_threshold=float(adaptive_dynamic_threshold or 0.0),
            calibration_prob=float(calibration_prob or 0.0),
            min_calibration_prob=float(min_calibration_prob or 0.0),
        )
        bcount = int(signal_eval.get("buy_count", buy_count) or buy_count or 0)
        scount = int(signal_eval.get("sell_count", sell_count) or sell_count or 0)
        dyn = float(signal_eval.get("dynamic_score", dynamic_score) or dynamic_score or 0.0)
        min_long_count = int(signal_eval.get("min_long_count", start_level) or start_level)
        min_dynamic_score = float(signal_eval.get("min_dynamic_score", 0.0) or 0.0)
        allow_dynamic_fallback = bool(signal_eval.get("allow_dynamic_fallback", False))
        gate_mode = str(signal_eval.get("gate_mode", "blocked") or "blocked")
        profile = normalize_settings_profile(profile_key, default="balanced")
        mode = str(policy_mode or "").strip().lower()

        long_headroom = 0
        dynamic_margin = 0.0
        min_dynamic_long_count = 0
        if allow_dynamic_fallback:
            if profile == "max_growth":
                long_headroom = 1
                dynamic_margin = 0.35 if mode in {"aggressive", "aggressive_rotation"} else 0.30
            elif profile == "aggressive":
                long_headroom = 1
                dynamic_margin = 0.25 if mode in {"aggressive", "aggressive_rotation"} else 0.20

        entry_min_long_count = int(max(1, min_long_count + long_headroom))
        entry_min_dynamic_score = float(min_dynamic_score + dynamic_margin) if allow_dynamic_fallback else float(min_dynamic_score)
        passed = bool(signal_eval.get("passed", False))
        failure_reason = str(signal_eval.get("failure_reason", "") or "").strip()
        alignment_mode = "not_evaluated"
        requirement_text = str(signal_eval.get("requirement_text", "") or "").strip()

        if passed:
            if gate_mode == "long_signal":
                alignment_mode = "long_signal_buffer"
                requirement_text = f"entry requires long>=N{entry_min_long_count} with short pressure clear"
                passed = bcount >= entry_min_long_count
                if not passed:
                    failure_reason = (
                        f"long signal too thin for entry stability (long=N{bcount}; need >=N{entry_min_long_count})"
                    )
            elif gate_mode == "dynamic_score_fallback":
                alignment_mode = "dynamic_fallback_buffer"
                requirement_text = (
                    f"entry dynamic fallback requires dynamic>={entry_min_dynamic_score:.2f} "
                    f"and long>=N{int(min_dynamic_long_count)}"
                )
                dynamic_ok = dyn >= entry_min_dynamic_score
                long_floor_ok = bcount >= int(min_dynamic_long_count)
                passed = bool(dynamic_ok and long_floor_ok)
                if not passed:
                    failure_reason = (
                        f"dynamic fallback too thin for entry stability "
                        f"(long=N{bcount}, dynamic={dyn:.3f}; need dynamic>={entry_min_dynamic_score:.2f}, "
                        f"long>=N{int(min_dynamic_long_count)})"
                    )
            else:
                alignment_mode = "blocked"
                passed = False
                if not failure_reason:
                    failure_reason = "signal gate blocked"

        return {
            "passed": bool(passed),
            "failure_reason": str(failure_reason),
            "alignment_mode": str(alignment_mode),
            "requirement_text": str(requirement_text),
            "signal_gate_mode": str(gate_mode),
            "min_long_count": int(entry_min_long_count),
            "min_dynamic_score": float(entry_min_dynamic_score),
            "long_headroom": int(long_headroom),
            "dynamic_margin": float(dynamic_margin),
            "min_dynamic_long_count": int(min_dynamic_long_count),
            "buy_count": int(bcount),
            "sell_count": int(scount),
            "dynamic_score": float(dyn),
            "adaptive_dynamic_threshold": float(signal_eval.get("adaptive_dynamic_threshold", adaptive_dynamic_threshold) or 0.0),
            "calibration_prob": float(signal_eval.get("calibration_prob", calibration_prob) or 0.0),
            "min_calibration_prob": float(signal_eval.get("min_calibration_prob", min_calibration_prob) or 0.0),
            "calibration_gate_applied": bool(signal_eval.get("calibration_gate_applied", False)),
        }

    @staticmethod
    def _signal_score_for_candidate(
        dynamic_score: float,
        buy_count: int,
        start_level: int,
        adaptive_threshold: float = 0.0,
    ) -> tuple[float, float]:
        dyn = float(dynamic_score or 0.0)
        lvl = max(1, min(int(start_level or 3), 7))
        if dyn > 0.0:
            required = max(0.01, float(adaptive_threshold or 0.0))
            if required <= 0.01:
                required = max(0.01, dyn)
            return dyn, required
        proxy = float(max(0, int(buy_count or 0))) / 7.0
        required = max(float(lvl) / 7.0, float(adaptive_threshold or 0.0))
        return proxy, required

    def _crypto_alignment_snapshot(
        self,
        base_symbol: str,
        *,
        start_level: int,
        dynamic_current_set: set[str],
        profile_key: Any = "balanced",
        dynamic_score: float = 0.0,
        policy_mode: str = "",
        adaptive_dynamic_threshold: float = 0.0,
        calibration_prob: float = 0.0,
        min_calibration_prob: float = 0.0,
    ) -> Dict[str, Any]:
        coin = str(base_symbol or "").strip().upper()
        if not coin:
            return {"aligned": True, "reasons": [], "buy_count": 0, "sell_count": 0}
        buy_count = int(self._read_long_dca_signal(coin))
        sell_count = int(self._read_short_dca_signal(coin))
        reasons: List[str] = []
        if dynamic_current_set and coin not in dynamic_current_set:
            reasons.append("not in the active rotation set")
        gate_eval = self._evaluate_crypto_signal_gate(
            profile_key=profile_key,
            start_level=int(start_level),
            buy_count=int(buy_count),
            sell_count=int(sell_count),
            dynamic_score=float(dynamic_score or 0.0),
            policy_mode=str(policy_mode or ""),
            adaptive_dynamic_threshold=float(adaptive_dynamic_threshold or 0.0),
            calibration_prob=float(calibration_prob or 0.0),
            min_calibration_prob=float(min_calibration_prob or 0.0),
        )
        if not bool(gate_eval.get("passed", True)):
            failure_reason = str(gate_eval.get("failure_reason", "") or "").strip()
            if failure_reason:
                reasons.append(failure_reason)
            else:
                reasons.append(str(gate_eval.get("requirement_text", "signal gate blocked") or "signal gate blocked"))
        return {
            "aligned": bool(len(reasons) == 0),
            "reasons": reasons,
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
            "dynamic_score": float(dynamic_score or 0.0),
            "adaptive_dynamic_threshold": float(adaptive_dynamic_threshold or 0.0),
            "calibration_prob": float(calibration_prob or 0.0),
            "min_calibration_prob": float(min_calibration_prob or 0.0),
            "signal_gate_mode": str(gate_eval.get("gate_mode", "blocked") or "blocked"),
            "signal_requirement": str(gate_eval.get("requirement_text", "") or ""),
        }

    def _load_pnl_ledger(self) -> dict:
        try:
            if os.path.isfile(PNL_LEDGER_PATH):
                with open(PNL_LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if not isinstance(data, dict):
                    data = {}
                # Back-compat upgrades
                data.setdefault("total_realized_profit_usd", 0.0)
                data.setdefault("last_updated_ts", time.time())
                data.setdefault("open_positions", {})   # { "BTC": {"usd_cost": float, "qty": float} }
                data.setdefault("pending_orders", {})   # { "<order_id>": {...} }
                return data
        except Exception:
            pass
        return {
            "total_realized_profit_usd": 0.0,
            "last_updated_ts": time.time(),
            "open_positions": {},
            "pending_orders": {},
        }

    def _save_pnl_ledger(self) -> None:
        try:
            self._pnl_ledger["last_updated_ts"] = time.time()
            self._atomic_write_json(PNL_LEDGER_PATH, self._pnl_ledger)
        except Exception:
            pass

    def _trade_history_has_order_id(self, order_id: str) -> bool:
        try:
            if not order_id:
                return False
            if not os.path.isfile(TRADE_HISTORY_PATH):
                return False
            with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if str(obj.get("order_id", "")).strip() == str(order_id).strip():
                        return True
        except Exception:
            return False
        return False

    def _get_buying_power(self) -> float:
        try:
            acct = self.get_account()
            if isinstance(acct, dict):
                return float(acct.get("buying_power", 0.0) or 0.0)
        except Exception:
            pass
        return 0.0

    def _get_order_by_id(self, symbol: str, order_id: str) -> Optional[dict]:
        try:
            orders = self.get_orders(symbol)
            results = orders.get("results", []) if isinstance(orders, dict) else []
            for o in results:
                try:
                    if o.get("id") == order_id:
                        return o
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _extract_fill_from_order(self, order: dict) -> tuple:
        """Returns (filled_qty, avg_fill_price). avg_fill_price may be None."""
        try:
            execs = order.get("executions", []) or []
            total_qty = 0.0
            total_notional = 0.0
            for ex in execs:
                try:
                    q = float(ex.get("quantity", 0.0) or 0.0)
                    p = float(ex.get("effective_price", 0.0) or 0.0)
                    if q > 0.0 and p > 0.0:
                        total_qty += q
                        total_notional += (q * p)
                except Exception:
                    continue

            avg_price = (total_notional / total_qty) if (total_qty > 0.0 and total_notional > 0.0) else None

            # Fallbacks if executions are not populated yet
            if total_qty <= 0.0:
                for k in ("filled_asset_quantity", "filled_quantity", "asset_quantity", "quantity"):
                    if k in order:
                        try:
                            v = float(order.get(k) or 0.0)
                            if v > 0.0:
                                total_qty = v
                                break
                        except Exception:
                            continue

            if avg_price is None:
                for k in ("average_price", "avg_price", "price", "effective_price"):
                    if k in order:
                        try:
                            v = float(order.get(k) or 0.0)
                            if v > 0.0:
                                avg_price = v
                                break
                        except Exception:
                            continue

            return float(total_qty), (float(avg_price) if avg_price is not None else None)
        except Exception:
            return 0.0, None

    def _wait_for_order_terminal(
        self,
        symbol: str,
        order_id: str,
        timeout_s: float = 120.0,
        poll_s: float = 1.0,
    ) -> Optional[dict]:
        """Waits for order terminal state and returns the order dict, or None on timeout."""
        terminal = {"filled", "canceled", "cancelled", "rejected", "failed", "error"}
        deadline = time.time() + max(1.0, float(timeout_s or 120.0))
        poll = max(0.1, float(poll_s or 1.0))
        while True:
            if time.time() >= deadline:
                self._log_rate_limited(
                    f"order_wait_timeout_{symbol}_{order_id}",
                    f"[pt_trader] order wait timeout symbol={symbol} order_id={order_id}",
                    every_s=15.0,
                )
                return None
            o = self._get_order_by_id(symbol, order_id)
            if not o:
                time.sleep(poll)
                continue
            st = str(o.get("state", "")).lower().strip()
            if st in terminal:
                return o
            time.sleep(poll)

    def _reconcile_pending_orders(self, max_total_wait_s: float = 30.0) -> None:
        """
        If the hub/trader restarts mid-order, we keep the pre-order buying_power on disk and
        finish the accounting once the order shows as terminal in Robinhood.
        """
        try:
            pending = self._pnl_ledger.get("pending_orders", {})
            if not isinstance(pending, dict) or not pending:
                return

            # Loop until everything pending is resolved (matches your design: bot waits here).
            deadline = time.time() + max(2.0, float(max_total_wait_s or 30.0))
            while True:
                pending = self._pnl_ledger.get("pending_orders", {})
                if not isinstance(pending, dict) or not pending:
                    break
                if time.time() >= deadline:
                    self._log_rate_limited(
                        "pending_reconcile_timeout",
                        "[pt_trader] reconcile timeout; deferring unresolved pending orders",
                        every_s=15.0,
                    )
                    break

                progressed = False

                for order_id, info in list(pending.items()):
                    try:
                        if self._trade_history_has_order_id(order_id):
                            # Already recorded (e.g., crash after writing history) -> just clear pending.
                            self._pnl_ledger["pending_orders"].pop(order_id, None)
                            self._save_pnl_ledger()
                            progressed = True
                            continue

                        symbol = str(info.get("symbol", "")).strip()
                        side = str(info.get("side", "")).strip().lower()
                        bp_before = float(info.get("buying_power_before", 0.0) or 0.0)

                        if not symbol or not side or not order_id:
                            self._pnl_ledger["pending_orders"].pop(order_id, None)
                            self._save_pnl_ledger()
                            progressed = True
                            continue

                        order = self._wait_for_order_terminal(symbol, order_id, timeout_s=10.0, poll_s=0.5)
                        if not order:
                            continue

                        state = str(order.get("state", "")).lower().strip()
                        if state != "filled":
                            # Not filled -> no trade to record, clear pending.
                            self._pnl_ledger["pending_orders"].pop(order_id, None)
                            self._save_pnl_ledger()
                            progressed = True
                            continue

                        filled_qty, avg_price = self._extract_fill_from_order(order)
                        bp_after = self._get_buying_power()
                        bp_delta = float(bp_after) - float(bp_before)

                        self._record_trade(
                            side=side,
                            symbol=symbol,
                            qty=float(filled_qty),
                            price=float(avg_price) if avg_price is not None else None,
                            avg_cost_basis=info.get("avg_cost_basis", None),
                            pnl_pct=info.get("pnl_pct", None),
                            tag=info.get("tag", None),
                            order_id=order_id,
                            fees_usd=None,
                            buying_power_before=bp_before,
                            buying_power_after=bp_after,
                            buying_power_delta=bp_delta,
                        )

                        # Clear pending now that we recorded it
                        self._pnl_ledger["pending_orders"].pop(order_id, None)
                        self._save_pnl_ledger()
                        progressed = True

                    except Exception:
                        continue

                if not progressed:
                    time.sleep(0.5)

        except Exception:
            pass

    def _record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
        order_id: Optional[str] = None,
        fees_usd: Optional[float] = None,
        buying_power_before: Optional[float] = None,
        buying_power_after: Optional[float] = None,
        buying_power_delta: Optional[float] = None,
        score: Optional[float] = None,
        required_score: Optional[float] = None,
        calib_prob: Optional[float] = None,
    ) -> None:
        """
        Minimal local ledger for GUI:
        - append trade_history.jsonl
        - update pnl_ledger.json on sells (now using buying power delta when available)
        - persist per-coin open position cost (USD) so realized profit is exact
        """
        ts = time.time()

        side_l = str(side or "").lower().strip()
        base = str(symbol or "").upper().split("-")[0].strip()

        # Ensure ledger keys exist (back-compat)
        try:
            if not isinstance(self._pnl_ledger, dict):
                self._pnl_ledger = {}
            self._pnl_ledger.setdefault("total_realized_profit_usd", 0.0)
            self._pnl_ledger.setdefault("open_positions", {})
            self._pnl_ledger.setdefault("pending_orders", {})
        except Exception:
            pass

        realized = None
        position_cost_used = None
        position_cost_after = None

        # --- Exact USD-based accounting (your design) ---
        if base and (buying_power_delta is not None):
            try:
                bp_delta = float(buying_power_delta)
            except Exception:
                bp_delta = None

            if bp_delta is not None:
                try:
                    open_pos = self._pnl_ledger.get("open_positions", {})
                    if not isinstance(open_pos, dict):
                        open_pos = {}
                        self._pnl_ledger["open_positions"] = open_pos

                    pos = open_pos.get(base, None)
                    if not isinstance(pos, dict):
                        pos = {"usd_cost": 0.0, "qty": 0.0}
                        open_pos[base] = pos

                    pos_usd_cost = float(pos.get("usd_cost", 0.0) or 0.0)
                    pos_qty = float(pos.get("qty", 0.0) or 0.0)

                    q = float(qty or 0.0)

                    if side_l == "buy":
                        usd_used = -bp_delta  # buying power drops on buys
                        if usd_used < 0.0:
                            usd_used = 0.0

                        if float(pos_qty) <= 1e-12:
                            pos["opened_ts"] = float(ts)
                        elif ("opened_ts" not in pos) or (float(pos.get("opened_ts", 0.0) or 0.0) <= 0.0):
                            pos["opened_ts"] = float(ts)
                        pos["last_buy_ts"] = float(ts)
                        pos["usd_cost"] = float(pos_usd_cost) + float(usd_used)
                        pos["qty"] = float(pos_qty) + float(q if q > 0.0 else 0.0)

                        position_cost_after = float(pos["usd_cost"])
                        try:
                            self._last_entry_ts[base] = float(ts)
                        except Exception:
                            pass

                        # Save because open position changed (needs to persist across restarts)
                        self._save_pnl_ledger()

                    elif side_l == "sell":
                        usd_got = bp_delta  # buying power rises on sells
                        if usd_got < 0.0:
                            usd_got = 0.0

                        # If partial sell ever happens, allocate cost pro-rata by qty.
                        if pos_qty > 0.0 and q > 0.0:
                            frac = min(1.0, float(q) / float(pos_qty))
                        else:
                            frac = 1.0

                        cost_used = float(pos_usd_cost) * float(frac)
                        pos["usd_cost"] = float(pos_usd_cost) - float(cost_used)
                        pos["qty"] = float(pos_qty) - float(q if q > 0.0 else 0.0)

                        position_cost_used = float(cost_used)
                        position_cost_after = float(pos.get("usd_cost", 0.0) or 0.0)

                        realized = float(usd_got) - float(cost_used)
                        self._pnl_ledger["total_realized_profit_usd"] = float(self._pnl_ledger.get("total_realized_profit_usd", 0.0) or 0.0) + float(realized)

                        # Clean up tiny dust
                        if float(pos.get("qty", 0.0) or 0.0) <= 1e-12 or float(pos.get("usd_cost", 0.0) or 0.0) <= 1e-6:
                            open_pos.pop(base, None)
                            try:
                                self._last_entry_ts.pop(base, None)
                            except Exception:
                                pass

                        self._save_pnl_ledger()

                except Exception:
                    pass

        # --- Fallback (old behavior) if we couldn't compute from buying power ---
        if realized is None and side_l == "sell" and price is not None and avg_cost_basis is not None:
            try:
                fee_val = float(fees_usd) if fees_usd is not None else 0.0
                realized = (float(price) - float(avg_cost_basis)) * float(qty) - fee_val
                self._pnl_ledger["total_realized_profit_usd"] = float(self._pnl_ledger.get("total_realized_profit_usd", 0.0)) + float(realized)
                self._save_pnl_ledger()
            except Exception:
                realized = None

        effective_pnl_pct = pnl_pct
        if side_l == "sell":
            try:
                cost_for_pct = None
                if position_cost_used is not None and float(position_cost_used) > 0.0:
                    cost_for_pct = float(position_cost_used)
                elif avg_cost_basis is not None:
                    est_cost = float(avg_cost_basis) * float(qty or 0.0)
                    if est_cost > 0.0:
                        cost_for_pct = est_cost
                if realized is not None and cost_for_pct is not None and float(cost_for_pct) > 0.0:
                    effective_pnl_pct = (float(realized) / float(cost_for_pct)) * 100.0
            except Exception:
                pass

        entry = {
            "ts": ts,
            "side": side,
            "tag": tag,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "avg_cost_basis": avg_cost_basis,
            "pnl_pct": float(effective_pnl_pct) if effective_pnl_pct is not None else None,
            "fees_usd": fees_usd,
            "realized_profit_usd": realized,
            "order_id": order_id,
            "buying_power_before": float(buying_power_before) if buying_power_before is not None else None,
            "buying_power_after": float(buying_power_after) if buying_power_after is not None else None,
            "buying_power_delta": float(buying_power_delta) if buying_power_delta is not None else None,
            "position_cost_used_usd": float(position_cost_used) if position_cost_used is not None else None,
            "position_cost_after_usd": float(position_cost_after) if position_cost_after is not None else None,
            "score": float(score) if score is not None else None,
            "required_score": float(required_score) if required_score is not None else None,
            "calib_prob": float(calib_prob) if calib_prob is not None else None,
        }
        self._append_jsonl(TRADE_HISTORY_PATH, entry)
        audit_event = "entry" if side_l == "buy" else ("exit" if side_l == "sell" else "trade")
        self._append_execution_audit(
            {
                "ts": int(ts),
                "event": audit_event,
                "ok": True,
                "symbol": str(symbol or "").strip().upper(),
                "side": str(side_l),
                "qty": float(qty or 0.0),
                "price": float(price) if price is not None else None,
                "pnl_pct": float(effective_pnl_pct) if effective_pnl_pct is not None else None,
                "realized_pnl_usd": float(realized) if realized is not None else None,
                "score": float(score) if score is not None else None,
                "required_score": float(required_score) if required_score is not None else None,
                "calib_prob": float(calib_prob) if calib_prob is not None else None,
                "tag": tag,
                "order_id": order_id,
                "payload": dict(entry),
            }
        )




    def _write_trader_status(self, status: dict) -> None:
        self._atomic_write_json(TRADER_DETAIL_PATH, status)

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _fmt_price(price: float) -> str:
        """
        Dynamic decimal formatting by magnitude:
        - >= 1.0   -> 2 decimals (BTC/ETH/etc won't show 8 decimals)
        - <  1.0   -> enough decimals to show meaningful digits (based on first non-zero),
                     then trim trailing zeros.
        """
        try:
            p = float(price)
        except Exception:
            return "N/A"

        if p == 0:
            return "0"

        ap = abs(p)

        if ap >= 1.0:
            decimals = 2
        else:
            # Example:
            # 0.5      -> decimals ~ 4 (prints "0.5" after trimming zeros)
            # 0.05     -> 5
            # 0.005    -> 6
            # 0.000012 -> 8
            decimals = int(-math.floor(math.log10(ap))) + 3
            decimals = max(2, min(12, decimals))

        s = f"{p:.{decimals}f}"

        # Trim useless trailing zeros for cleaner output (0.5000 -> 0.5)
        if "." in s:
            s = s.rstrip("0").rstrip(".")

        return s


    def _read_long_dca_signal(self, symbol: str) -> int:
        """
        Reads long_dca_signal.txt from the per-coin folder (same folder rules as trader.py).

        Used for:
        - Start gate: start trades at level 3+
        - DCA assist: levels 4-7 map to trader DCA stages 0-3 (trade starts at level 3 => stage 0)
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, os.path.join(main_dir, sym))
        path = os.path.join(folder, "long_dca_signal.txt")
        try:
            return _read_int_file_cached(path, 0)
        except Exception:
            return 0


    def _read_short_dca_signal(self, symbol: str) -> int:
        """
        Reads short_dca_signal.txt from the per-coin folder (same folder rules as trader.py).

        Used for:
        - Start gate: start trades at level 3+
        - DCA assist: levels 4-7 map to trader DCA stages 0-3 (trade starts at level 3 => stage 0)
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, os.path.join(main_dir, sym))
        path = os.path.join(folder, "short_dca_signal.txt")
        try:
            return _read_int_file_cached(path, 0)
        except Exception:
            return 0

    @staticmethod
    def _read_long_price_levels(symbol: str) -> list:
        """
        Reads low_bound_prices.html from the per-coin folder and returns a list of LONG (blue) price levels.

        Returned ordering is highest->lowest so:
          N1 = 1st blue line (top)
          ...
          N7 = 7th blue line (bottom)
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, os.path.join(main_dir, sym))
        path = os.path.join(folder, "low_bound_prices.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            if not raw:
                return []

            # Normalize common formats: python-list, comma-separated, newline-separated
            raw = raw.strip().strip("[]()")
            raw = raw.replace(",", " ").replace(";", " ").replace("|", " ")
            raw = raw.replace("\n", " ").replace("\t", " ")
            parts = [p for p in raw.split() if p]

            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    continue

            # De-dupe, then sort high->low for stable N1..N7 mapping
            out = []
            seen = set()
            for v in vals:
                k = round(float(v), 12)
                if k in seen:
                    continue
                seen.add(k)
                out.append(float(v))
            out.sort(reverse=True)
            return out
        except Exception:
            return []



    def initialize_dca_levels(self):

        """
        Initializes the DCA levels_triggered dictionary based on the number of buy orders
        that have occurred after the first buy order following the most recent sell order
        for each cryptocurrency.
        """
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            print("No holdings found. Skipping DCA levels initialization.")
            return

        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]

            full_symbol = f"{symbol}-USD"
            orders = self.get_orders(full_symbol)
            
            if not orders or "results" not in orders:
                print(f"No orders found for {full_symbol}. Skipping.")
                continue

            # Filter for filled buy and sell orders
            filled_orders = [
                order for order in orders["results"]
                if order["state"] == "filled" and order["side"] in ["buy", "sell"]
            ]
            
            if not filled_orders:
                print(f"No filled buy or sell orders for {full_symbol}. Skipping.")
                continue

            # Sort orders by creation time in ascending order (oldest first)
            filled_orders.sort(key=lambda x: x["created_at"])

            # Find the timestamp of the most recent sell order
            most_recent_sell_time = None
            for order in reversed(filled_orders):
                if order["side"] == "sell":
                    most_recent_sell_time = order["created_at"]
                    break

            # Determine the cutoff time for buy orders
            if most_recent_sell_time:
                # Find all buy orders after the most recent sell
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy" and order["created_at"] > most_recent_sell_time
                ]
                if not relevant_buy_orders:
                    print(f"No buy orders after the most recent sell for {full_symbol}.")
                    self.dca_levels_triggered[symbol] = []
                    continue
                print(f"Most recent sell for {full_symbol} at {most_recent_sell_time}.")
            else:
                # If no sell orders, consider all buy orders
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy"
                ]
                if not relevant_buy_orders:
                    print(f"No buy orders for {full_symbol}. Skipping.")
                    self.dca_levels_triggered[symbol] = []
                    continue
                print(f"No sell orders found for {full_symbol}. Considering all buy orders.")

            # Ensure buy orders are sorted by creation time ascending
            relevant_buy_orders.sort(key=lambda x: x["created_at"])

            # Identify the first buy order in the relevant list
            first_buy_order = relevant_buy_orders[0]
            first_buy_time = first_buy_order["created_at"]

            # Count the number of buy orders after the first buy
            buy_orders_after_first = [
                order for order in relevant_buy_orders
                if order["created_at"] > first_buy_time
            ]

            triggered_levels_count = len(buy_orders_after_first)

            # Track DCA by stage index (0, 1, 2, ...) rather than % values.
            # This makes neural-vs-hardcoded clean, and allows repeating the -50% stage indefinitely.
            self.dca_levels_triggered[symbol] = list(range(triggered_levels_count))
            print(f"Initialized DCA stages for {symbol}: {triggered_levels_count}")


    def _seed_dca_window_from_history(self) -> None:
        """
        Seeds in-memory DCA buy timestamps from TRADE_HISTORY_PATH so the 24h limit
        works across restarts.

        Uses the local GUI trade history (tag == "DCA") and resets per trade at the most recent sell.
        """
        now_ts = time.time()
        cutoff = now_ts - float(getattr(self, "dca_window_seconds", 86400))

        self._dca_buy_ts = {}
        self._dca_last_sell_ts = {}

        if not os.path.isfile(TRADE_HISTORY_PATH):
            return

        try:
            with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    ts = obj.get("ts", None)
                    side = str(obj.get("side", "")).lower()
                    tag = obj.get("tag", None)
                    sym_full = str(obj.get("symbol", "")).upper().strip()
                    base = sym_full.split("-")[0].strip() if sym_full else ""
                    if not base:
                        continue

                    try:
                        ts_f = float(ts)
                    except Exception:
                        continue

                    if side == "sell":
                        prev = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
                        if ts_f > prev:
                            self._dca_last_sell_ts[base] = ts_f

                    elif side == "buy" and tag == "DCA":
                        self._dca_buy_ts.setdefault(base, []).append(ts_f)

        except Exception:
            return

        # Keep only DCA buys after the last sell (current trade) and within rolling 24h
        for base, ts_list in list(self._dca_buy_ts.items()):
            last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
            kept = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
            kept.sort()
            self._dca_buy_ts[base] = kept


    def _dca_window_count(self, base_symbol: str, now_ts: Optional[float] = None) -> int:
        """
        Count of DCA buys for this coin within rolling 24h in the *current trade*.
        Current trade boundary = most recent sell we observed for this coin.
        """
        base = str(base_symbol).upper().strip()
        if not base:
            return 0

        now = float(now_ts if now_ts is not None else time.time())
        cutoff = now - float(getattr(self, "dca_window_seconds", 86400))
        last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)

        ts_list = list(self._dca_buy_ts.get(base, []) or [])
        ts_list = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
        self._dca_buy_ts[base] = ts_list
        return len(ts_list)


    def _note_dca_buy(self, base_symbol: str, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        t = float(ts if ts is not None else time.time())
        self._dca_buy_ts.setdefault(base, []).append(t)
        self._dca_window_count(base, now_ts=t)  # prune in-place


    def _reset_dca_window_for_trade(self, base_symbol: str, sold: bool = False, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        if sold:
            self._dca_last_sell_ts[base] = float(ts if ts is not None else time.time())
        self._dca_buy_ts[base] = []


    def make_api_request(self, method: str, path: str, body: Optional[str] = "") -> Any:

        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        response = None
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                payload = str(body or "")
                response = requests.post(url, headers=headers, data=payload, timeout=10)

            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            try:
                # Parse and return the JSON error response
                error_response = response.json()
                retry_after_s = 0.0
                try:
                    retry_after_s = parse_retry_after_value((response.headers or {}).get("Retry-After", ""), max_wait_s=300.0)
                except Exception:
                    retry_after_s = 0.0
                if retry_after_s > 0.0 and isinstance(error_response, dict):
                    error_response.setdefault("retry_after_s", float(retry_after_s))
                return error_response  # Return the JSON error for further handling
            except Exception:
                return None
        except Exception:
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json",
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def get_holdings(self) -> Any:
        path = "/api/v1/crypto/trading/holdings/"
        return self.make_api_request("GET", path)

    def get_trading_pairs(self) -> Any:
        path = "/api/v1/crypto/trading/trading_pairs/"
        response = self.make_api_request("GET", path)

        if not response or "results" not in response:
            return []

        trading_pairs = response.get("results", [])
        if not trading_pairs:
            return []

        return trading_pairs

    def get_orders(self, symbol: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/?symbol={symbol}"
        return self.make_api_request("GET", path)

    def calculate_cost_basis(self):
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            return {}

        active_assets = {holding["asset_code"] for holding in holdings.get("results", [])}
        current_quantities = {
            holding["asset_code"]: float(holding["total_quantity"])
            for holding in holdings.get("results", [])
        }

        cost_basis = {}

        for asset_code in active_assets:
            orders = self.get_orders(f"{asset_code}-USD")
            if not orders or "results" not in orders:
                continue

            # Get all filled buy orders, sorted from most recent to oldest
            buy_orders = [
                order for order in orders["results"]
                if order["side"] == "buy" and order["state"] == "filled"
            ]
            buy_orders.sort(key=lambda x: x["created_at"], reverse=True)

            remaining_quantity = current_quantities[asset_code]
            total_cost = 0.0

            for order in buy_orders:
                for execution in order.get("executions", []):
                    quantity = float(execution["quantity"])
                    price = float(execution["effective_price"])

                    if remaining_quantity <= 0:
                        break

                    # Use only the portion of the quantity needed to match the current holdings
                    if quantity > remaining_quantity:
                        total_cost += remaining_quantity * price
                        remaining_quantity = 0
                    else:
                        total_cost += quantity * price
                        remaining_quantity -= quantity

                if remaining_quantity <= 0:
                    break

            if current_quantities[asset_code] > 0:
                cost_basis[asset_code] = total_cost / current_quantities[asset_code]
            else:
                cost_basis[asset_code] = 0.0

        return cost_basis

    def _calculate_symbol_cost_basis(self, base_symbol: str, current_quantity: float) -> float:
        coin = str(base_symbol or "").strip().upper()
        try:
            qty_needed = float(current_quantity or 0.0)
        except Exception:
            qty_needed = 0.0
        if (not coin) or qty_needed <= 0.0:
            return 0.0

        orders = self.get_orders(f"{coin}-USD")
        if not orders or "results" not in orders:
            return 0.0

        buy_orders = [
            order for order in orders.get("results", [])
            if isinstance(order, dict) and order.get("side") == "buy" and order.get("state") == "filled"
        ]
        if not buy_orders:
            return 0.0
        buy_orders.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        remaining_quantity = float(qty_needed)
        total_cost = 0.0

        for order in buy_orders:
            for execution in list(order.get("executions", []) or []):
                try:
                    quantity = float(execution.get("quantity", 0.0) or 0.0)
                    price = float(execution.get("effective_price", 0.0) or 0.0)
                except Exception:
                    continue

                if quantity <= 0.0 or price <= 0.0:
                    continue
                if remaining_quantity <= 0.0:
                    break

                used_quantity = min(quantity, remaining_quantity)
                total_cost += used_quantity * price
                remaining_quantity -= used_quantity
            if remaining_quantity <= 0.0:
                break

        if qty_needed > 0.0 and total_cost > 0.0:
            return float(total_cost / qty_needed)
        return 0.0

    def _refresh_missing_cost_basis(self, holdings_results: List[Dict[str, Any]]) -> None:
        if not isinstance(self.cost_basis, dict):
            self.cost_basis = {}
        for row in list(holdings_results or []):
            if not isinstance(row, dict):
                continue
            coin = str(row.get("asset_code", "") or "").strip().upper()
            if (not coin) or coin == "USDC":
                continue
            try:
                qty = float(row.get("total_quantity", 0.0) or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0.0:
                continue
            if self._fallback_avg_cost_basis(coin, quantity=qty) > 0.0:
                continue
            recomputed = self._calculate_symbol_cost_basis(coin, qty)
            if recomputed > 0.0:
                self.cost_basis[coin] = float(recomputed)

    def get_price(self, symbols: list) -> Dict[str, float]:
        buy_prices = {}
        sell_prices = {}
        valid_symbols = []

        for symbol in symbols:
            if symbol == "USDC-USD":
                continue

            path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
            response = self.make_api_request("GET", path)

            if response and "results" in response:
                rows = response.get("results", [])
                if not isinstance(rows, list) or not rows:
                    continue
                result = rows[0] if isinstance(rows[0], dict) else {}
                if not result:
                    continue
                try:
                    ask = float(result["ask_inclusive_of_buy_spread"])
                    bid = float(result["bid_inclusive_of_sell_spread"])
                except Exception:
                    continue

                buy_prices[symbol] = ask
                sell_prices[symbol] = bid
                valid_symbols.append(symbol)

                # Update cache for transient failures later
                try:
                    self._last_good_bid_ask[symbol] = {"ask": ask, "bid": bid, "ts": time.time()}
                except Exception:
                    pass
            else:
                # Fallback to cached bid/ask so account value never drops due to a transient miss
                cached = None
                try:
                    cached = self._last_good_bid_ask.get(symbol)
                except Exception:
                    cached = None

                if cached:
                    ask = float(cached.get("ask", 0.0) or 0.0)
                    bid = float(cached.get("bid", 0.0) or 0.0)
                    if ask > 0.0 and bid > 0.0:
                        buy_prices[symbol] = ask
                        sell_prices[symbol] = bid
                        valid_symbols.append(symbol)

        return buy_prices, sell_prices, valid_symbols


    def place_buy_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        amount_in_usd: float,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
        audit_meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # Fetch the current price of the asset (for sizing only)
        current_buy_prices, current_sell_prices, valid_symbols = self.get_price([symbol])
        current_price = current_buy_prices[symbol]
        asset_quantity = amount_in_usd / current_price

        max_retries = 5
        retries = 0

        while retries < max_retries:
            retries += 1
            response = None
            try:
                # Default precision to 8 decimals initially
                rounded_quantity = round(asset_quantity, 8)

                body = {
                    "client_order_id": client_order_id,
                    "side": side,
                    "type": order_type,
                    "symbol": symbol,
                    "market_order_config": {
                        "asset_quantity": f"{rounded_quantity:.8f}"  # Start with 8 decimal places
                    }
                }

                path = "/api/v1/crypto/trading/orders/"

                # --- exact profit tracking snapshot (BEFORE placing order) ---
                buying_power_before = self._get_buying_power()

                response = self.make_api_request("POST", path, json.dumps(body))
                if response and "errors" not in response:
                    order_id = response.get("id", None)

                    # Persist the pre-order buying power so restarts can reconcile precisely
                    try:
                        if order_id:
                            self._pnl_ledger.setdefault("pending_orders", {})
                            self._pnl_ledger["pending_orders"][order_id] = {
                                "symbol": symbol,
                                "side": "buy",
                                "buying_power_before": float(buying_power_before),
                                "avg_cost_basis": float(avg_cost_basis) if avg_cost_basis is not None else None,
                                "pnl_pct": float(pnl_pct) if pnl_pct is not None else None,
                                "tag": tag,
                                "created_ts": time.time(),
                            }
                            self._save_pnl_ledger()
                    except Exception:
                        pass

                    # Wait until the order is actually complete in the system, then use order history executions
                    if order_id:
                        order = self._wait_for_order_terminal(symbol, order_id)
                        state = str(order.get("state", "")).lower().strip() if isinstance(order, dict) else ""
                        if state != "filled":
                            # Not filled -> clear pending and do not record a trade
                            try:
                                self._pnl_ledger.get("pending_orders", {}).pop(order_id, None)
                                self._save_pnl_ledger()
                            except Exception:
                                pass
                            return None

                        filled_qty, avg_fill_price = self._extract_fill_from_order(order)

                        buying_power_after = self._get_buying_power()
                        buying_power_delta = float(buying_power_after) - float(buying_power_before)

                        # Record for GUI history (ACTUAL fill from order history)
                        self._record_trade(
                            side="buy",
                            symbol=symbol,
                            qty=float(filled_qty),
                            price=float(avg_fill_price) if avg_fill_price is not None else None,
                            avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                            pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                            tag=tag,
                            order_id=order_id,
                            buying_power_before=buying_power_before,
                            buying_power_after=buying_power_after,
                            buying_power_delta=buying_power_delta,
                            score=float((audit_meta or {}).get("score", 0.0) or 0.0) if isinstance(audit_meta, dict) and ((audit_meta or {}).get("score", None) is not None) else None,
                            required_score=float((audit_meta or {}).get("required_score", 0.0) or 0.0)
                            if isinstance(audit_meta, dict) and ((audit_meta or {}).get("required_score", None) is not None)
                            else None,
                            calib_prob=float((audit_meta or {}).get("calib_prob", 0.0) or 0.0)
                            if isinstance(audit_meta, dict) and ((audit_meta or {}).get("calib_prob", None) is not None)
                            else None,
                        )

                        # Clear pending now that it is recorded
                        try:
                            self._pnl_ledger.get("pending_orders", {}).pop(order_id, None)
                            self._save_pnl_ledger()
                        except Exception:
                            pass

                    return response  # Successfully placed (and fully filled) order

            except Exception:
                pass #print(traceback.format_exc())

            # Check for precision errors
            if response and "errors" in response:
                for error in response["errors"]:
                    if "has too much precision" in error.get("detail", ""):
                        # Extract required precision directly from the error message
                        detail = error["detail"]
                        nearest_value = detail.split("nearest ")[1].split(" ")[0]

                        decimal_places = len(nearest_value.split(".")[1].rstrip("0"))
                        asset_quantity = round(asset_quantity, decimal_places)
                        break
                    elif "must be greater than or equal to" in error.get("detail", ""):
                        return None

        return None



    def place_sell_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        asset_quantity: float,
        expected_price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Any:
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            "market_order_config": {
                "asset_quantity": f"{asset_quantity:.8f}"
            }
        }

        path = "/api/v1/crypto/trading/orders/"

        # --- exact profit tracking snapshot (BEFORE placing order) ---
        buying_power_before = self._get_buying_power()

        response = self.make_api_request("POST", path, json.dumps(body))

        if response and isinstance(response, dict) and "errors" not in response:
            order_id = response.get("id", None)

            # Persist the pre-order buying power so restarts can reconcile precisely
            try:
                if order_id:
                    self._pnl_ledger.setdefault("pending_orders", {})
                    self._pnl_ledger["pending_orders"][order_id] = {
                        "symbol": symbol,
                        "side": "sell",
                        "buying_power_before": float(buying_power_before),
                        "avg_cost_basis": float(avg_cost_basis) if avg_cost_basis is not None else None,
                        "pnl_pct": float(pnl_pct) if pnl_pct is not None else None,
                        "tag": tag,
                        "created_ts": time.time(),
                    }
                    self._save_pnl_ledger()
            except Exception:
                pass

            # Best-effort: pull actual avg fill price + fees from order executions
            actual_price = float(expected_price) if expected_price is not None else None
            actual_qty = float(asset_quantity)
            fees_usd = None

            def _fee_to_float(v: Any) -> float:
                try:
                    if v is None:
                        return 0.0
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, str):
                        return float(v)
                    if isinstance(v, dict):
                        # common shapes: {"amount": "0.12"}, {"value": 0.12}, etc.
                        for k in ("amount", "value", "usd_amount", "fee", "quantity"):
                            if k in v:
                                try:
                                    return float(v[k])
                                except Exception:
                                    continue
                    return 0.0
                except Exception:
                    return 0.0

            try:
                if order_id:
                    match = self._wait_for_order_terminal(symbol, order_id)
                    if not match:
                        return response

                    if str(match.get("state", "")).lower() != "filled":
                        # Not filled -> clear pending and do not record a trade
                        try:
                            self._pnl_ledger.get("pending_orders", {}).pop(order_id, None)
                            self._save_pnl_ledger()
                        except Exception:
                            pass
                        return response

                    execs = match.get("executions", []) or []
                    total_qty = 0.0
                    total_notional = 0.0
                    fee_total = 0.0

                    for ex in execs:
                        try:
                            q = float(ex.get("quantity", 0.0) or 0.0)
                            p = float(ex.get("effective_price", 0.0) or 0.0)
                            total_qty += q
                            total_notional += (q * p)

                            # Fees can show up under different keys; handle the common ones.
                            for fk in ("fee", "fees", "fee_amount", "fee_usd", "fee_in_usd"):
                                if fk in ex:
                                    fee_total += _fee_to_float(ex.get(fk))
                        except Exception:
                            continue

                    # Some payloads include order-level fee fields too
                    for fk in ("fee", "fees", "fee_amount", "fee_usd", "fee_in_usd"):
                        if fk in match:
                            fee_total += _fee_to_float(match.get(fk))

                    if total_qty > 0.0 and total_notional > 0.0:
                        actual_qty = total_qty
                        actual_price = total_notional / total_qty

                    fees_usd = float(fee_total) if fee_total else 0.0

            except Exception:
                pass #print(traceback.format_exc())

            # If we managed to get a better fill price, update the displayed PnL% too
            if avg_cost_basis is not None and actual_price is not None:
                try:
                    acb = float(avg_cost_basis)
                    if acb > 0:
                        pnl_pct = ((float(actual_price) - acb) / acb) * 100.0
                except Exception:
                    pass

            # --- exact profit tracking snapshot (AFTER the order is complete) ---
            buying_power_after = self._get_buying_power()
            buying_power_delta = float(buying_power_after) - float(buying_power_before)

            self._record_trade(
                side="sell",
                symbol=symbol,
                qty=float(actual_qty),
                price=float(actual_price) if actual_price is not None else None,
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag,
                order_id=order_id,
                fees_usd=float(fees_usd) if fees_usd is not None else None,
                buying_power_before=buying_power_before,
                buying_power_after=buying_power_after,
                buying_power_delta=buying_power_delta,
            )

            # Clear pending now that it is recorded
            try:
                if order_id:
                    self._pnl_ledger.get("pending_orders", {}).pop(order_id, None)
                    self._save_pnl_ledger()
            except Exception:
                pass

        return response

    def _append_manual_order_result(self, row: Dict[str, Any]) -> None:
        try:
            payload = dict(row or {})
            payload.setdefault("ts", time.time())
            self._append_jsonl(MANUAL_CRYPTO_ORDER_RESULTS_PATH, payload)
        except Exception:
            pass

    def _process_manual_sell_requests(
        self,
        holdings_results: List[Dict[str, Any]],
        current_sell_prices: Dict[str, float],
        valid_symbols: List[str],
    ) -> bool:
        trades_made = False
        try:
            req_files = sorted(glob.glob(os.path.join(MANUAL_CRYPTO_ORDERS_DIR, "*.json")))
        except Exception:
            req_files = []
        if not req_files:
            return False

        valid_set = {str(s or "").strip().upper() for s in (valid_symbols or [])}
        qty_by_coin: Dict[str, float] = {}
        for row in list(holdings_results or []):
            try:
                coin = str(row.get("asset_code", "") or "").strip().upper()
                qty = float(row.get("total_quantity", 0.0) or 0.0)
                if coin and qty > 0.0:
                    qty_by_coin[coin] = float(qty)
            except Exception:
                continue

        for req_path in req_files:
            req: Dict[str, Any] = {}
            req_id = os.path.basename(req_path)
            result: Dict[str, Any] = {"request_id": req_id, "ok": False, "path": req_path}
            try:
                try:
                    with open(req_path, "r", encoding="utf-8") as f:
                        req = json.load(f) or {}
                except Exception as exc:
                    result["error"] = f"invalid_request_json:{type(exc).__name__}"
                    self._append_manual_order_result(result)
                    continue

                req_id = str(req.get("id", req_id) or req_id).strip()
                action = str(req.get("action", "sell_usd") or "sell_usd").strip().lower()
                coin = str(req.get("coin", "") or "").strip().upper()
                try:
                    amount_usd = float(req.get("amount_usd", 0.0) or 0.0)
                except Exception:
                    amount_usd = 0.0

                result.update(
                    {
                        "request_id": req_id,
                        "action": action,
                        "coin": coin,
                        "requested_amount_usd": amount_usd,
                    }
                )

                if action not in {"sell_usd", "manual_sell_usd"}:
                    result["error"] = f"unsupported_action:{action}"
                    self._append_manual_order_result(result)
                    continue
                if (not coin) or amount_usd <= 0.0:
                    result["error"] = "invalid_request_fields"
                    self._append_manual_order_result(result)
                    continue

                qty_avail = float(qty_by_coin.get(coin, 0.0) or 0.0)
                full_symbol = f"{coin}-USD"
                try:
                    sell_px = float(current_sell_prices.get(full_symbol, 0.0) or 0.0)
                except Exception:
                    sell_px = 0.0
                if qty_avail <= 0.0:
                    result["error"] = "coin_not_held"
                    self._append_manual_order_result(result)
                    continue
                if full_symbol not in valid_set or sell_px <= 0.0:
                    result["error"] = "sell_price_unavailable"
                    self._append_manual_order_result(result)
                    continue

                max_notional = qty_avail * sell_px
                sell_notional = min(float(amount_usd), float(max_notional))
                sell_qty = sell_notional / sell_px if sell_px > 0.0 else 0.0
                sell_qty = float(max(0.0, round(sell_qty, 8)))
                if sell_qty <= 0.0:
                    result["error"] = "sell_quantity_zero"
                    self._append_manual_order_result(result)
                    continue

                # If a partial sell would leave sub-minimum dust, sell the full coin instead.
                # This avoids broker rejections for tiny residual quantities/notional.
                remainder_notional = max(0.0, float(max_notional - (sell_qty * sell_px)))
                if 0.0 < remainder_notional < 1.0:
                    sell_qty = float(max(0.0, round(qty_avail, 8)))

                def _extract_order_error_text(resp: Any) -> str:
                    if not (resp and isinstance(resp, dict)):
                        return ""
                    errs = resp.get("errors", [])
                    if not isinstance(errs, list):
                        return ""
                    chunks = []
                    for e in errs:
                        if isinstance(e, dict):
                            part = str(e.get("detail", "") or e.get("message", "") or "").strip()
                        else:
                            part = str(e or "").strip()
                        if part:
                            chunks.append(part)
                    return " | ".join(chunks)[:320]

                try:
                    avg_cost_basis = float(self.cost_basis.get(coin, 0.0) or 0.0)
                except Exception:
                    avg_cost_basis = 0.0
                pnl_pct = ((sell_px - avg_cost_basis) / avg_cost_basis) * 100.0 if avg_cost_basis > 0.0 else None

                # Retry with progressively coarser precision to satisfy broker increment rules.
                sell_qty_attempts: List[float] = []
                seen_qty = set()

                def _push_qty(q: float) -> None:
                    try:
                        qf = float(max(0.0, min(qty_avail, q)))
                    except Exception:
                        return
                    qf = float(round(qf, 8))
                    if qf <= 0.0:
                        return
                    key = round(qf, 8)
                    if key in seen_qty:
                        return
                    seen_qty.add(key)
                    sell_qty_attempts.append(qf)

                _push_qty(sell_qty)
                for dec in (7, 6, 5, 4, 3, 2, 1, 0):
                    _push_qty(round(sell_qty, dec))
                _push_qty(qty_avail)  # final fallback: full position

                response = None
                ok = False
                used_sell_qty = sell_qty
                broker_error_txt = ""
                for q_try in sell_qty_attempts:
                    used_sell_qty = q_try
                    response = self.place_sell_order(
                        str(uuid.uuid4()),
                        "sell",
                        "market",
                        full_symbol,
                        q_try,
                        expected_price=sell_px,
                        avg_cost_basis=avg_cost_basis if avg_cost_basis > 0.0 else None,
                        pnl_pct=pnl_pct,
                        tag="MANUAL_SELL_USD",
                    )
                    ok = bool(response and isinstance(response, dict) and ("errors" not in response))
                    if ok:
                        break
                    broker_error_txt = _extract_order_error_text(response)

                if ok:
                    trades_made = True
                    qty_by_coin[coin] = max(0.0, float(qty_avail - used_sell_qty))
                    self.trailing_pm.pop(coin, None)
                    self._reset_dca_window_for_trade(coin, sold=True)
                    self._last_exit_ts[coin] = time.time()
                    result["ok"] = True
                else:
                    result["error"] = "broker_sell_failed"
                    if broker_error_txt:
                        result["broker_error"] = broker_error_txt

                result.update(
                    {
                        "sell_qty": float(used_sell_qty),
                        "sell_price": float(sell_px),
                        "executed_notional_usd": round(float(used_sell_qty * sell_px), 6),
                        "max_notional_usd": round(float(max_notional), 6),
                    }
                )
                self._append_manual_order_result(result)
            finally:
                try:
                    os.remove(req_path)
                except Exception:
                    pass

        return bool(trades_made)





    def manage_trades(self):
        trades_made = False  # Flag to track if any trade was made in this iteration
        self._set_status_note("")
        policy: Dict[str, Any] = {}
        trade_quality_eval: Dict[str, Any] = {}
        opportunity_eval: Dict[str, Any] = {}
        entry_fail_reasons: List[str] = []
        entry_eval_top_reason = ""
        entry_eval_reason_counts: Dict[str, int] = {}
        entry_size_scale = 1.0
        allocator_size_scale = 1.0
        entry_gate_flags: Dict[str, Any] = {}
        stale_exit_events: List[Dict[str, Any]] = []
        stale_exit_count = 0
        stale_exit_enabled = True
        stale_exit_grace_cycles = 2
        stale_exit_max_per_cycle = 2
        stale_exit_min_notional_usd = 5.0
        skip_new_entries_this_cycle = False
        signal_gate_debug: Dict[str, Any] = {}
        entry_alignment_debug: Dict[str, Any] = {}
        try:
            self._reconcile_pending_orders(max_total_wait_s=0.5)
        except Exception:
            pass

        # Hot-reload coins list + paths + trade params from GUI settings while running
        try:
            _refresh_paths_and_symbols()
            self.path_map = dict(base_paths)
            self.dca_levels = list(DCA_LEVELS)
            self.max_dca_buys_per_24h = int(MAX_DCA_BUYS_PER_24H)

            # Trailing PM settings (hot-reload)
            old_sig = getattr(self, "_last_trailing_settings_sig", None)

            new_gap = float(TRAILING_GAP_PCT)
            new_pm0 = float(PM_START_PCT_NO_DCA)
            new_pm1 = float(PM_START_PCT_WITH_DCA)

            self.trailing_gap_pct = new_gap
            self.pm_start_pct_no_dca = new_pm0
            self.pm_start_pct_with_dca = new_pm1
            self._loop_sleep_ok = float(CRYPTO_TRADER_LOOP_SLEEP_S)
            self._loop_sleep_error = float(CRYPTO_TRADER_ERROR_SLEEP_S)

            new_sig = (float(new_gap), float(new_pm0), float(new_pm1))

            # If trailing settings changed, reset ALL trailing PM state so:
            # - the line updates immediately
            # - peak/armed/was_above are cleared
            if (old_sig is not None) and (new_sig != old_sig):
                self.trailing_pm = {}

            self._last_trailing_settings_sig = new_sig
        except Exception:
            pass




        # Fetch account details
        account = self.get_account()
        # Fetch holdings
        holdings = self.get_holdings()
        # Fetch trading pairs
        trading_pairs = self.get_trading_pairs()
        if not isinstance(account, dict):
            account = {}
        holdings_results, used_cached_holdings = self._resolve_holdings_results(holdings, recent_trade=bool(trades_made))
        holdings = {"results": holdings_results}
        if not isinstance(trading_pairs, list):
            trading_pairs = []
        self._refresh_missing_cost_basis(holdings_results)

        # Use the stored cost_basis instead of recalculating
        cost_basis = self.cost_basis
        # Fetch current prices
        symbols = [holding["asset_code"] + "-USD" for holding in holdings.get("results", [])]

        # ALSO fetch prices for tracked coins even if not currently held (so GUI can show bid/ask lines)
        for s in crypto_symbols:
            full = f"{s}-USD"
            if full not in symbols:
                symbols.append(full)

        current_buy_prices, current_sell_prices, valid_symbols = self.get_price(symbols)
        manual_trades_made = self._process_manual_sell_requests(holdings_results, current_sell_prices, valid_symbols)
        if manual_trades_made:
            trades_made = True
            time.sleep(2)
            account = self.get_account()
            holdings = self.get_holdings()
            trading_pairs = self.get_trading_pairs()
            if not isinstance(account, dict):
                account = {}
            holdings_results, used_cached_holdings = self._resolve_holdings_results(holdings, recent_trade=bool(trades_made))
            holdings = {"results": holdings_results}
            if not isinstance(trading_pairs, list):
                trading_pairs = []
            self._refresh_missing_cost_basis(holdings_results)
            symbols = [holding["asset_code"] + "-USD" for holding in holdings.get("results", [])]
            for s in crypto_symbols:
                full = f"{s}-USD"
                if full not in symbols:
                    symbols.append(full)
            current_buy_prices, current_sell_prices, valid_symbols = self.get_price(symbols)

        settings_path = resolve_settings_path(BASE_DIR) or _GUI_SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_trader") or {})
        profile_key = normalize_settings_profile(settings.get("settings_profile", "balanced"), default="balanced")
        start_level = max(1, min(int(TRADE_START_LEVEL or 3), 7))
        runtime_snapshot = self._safe_read_json_file(RUNTIME_STATE_PATH)
        runtime_alerts = runtime_snapshot.get("alerts", {}) if isinstance(runtime_snapshot.get("alerts", {}), dict) else {}
        dynamic_status = self._safe_read_json_file(CRYPTO_DYNAMIC_STATUS_PATH)
        dynamic_updated_ts = int(float(dynamic_status.get("ts", dynamic_status.get("updated_at", 0) or 0) or 0))
        dynamic_rank_rows = dynamic_status.get("ranked", []) if isinstance(dynamic_status.get("ranked", []), list) else []
        dynamic_rank_map: Dict[str, float] = {}
        dynamic_calib_prob_map: Dict[str, float] = {}
        dynamic_samples_map: Dict[str, int] = {}
        for row in list(dynamic_rank_rows):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "") or "").strip().upper()
            if not symbol:
                continue
            try:
                dynamic_rank_map[symbol] = float(row.get("score", 0.0) or 0.0)
            except Exception:
                dynamic_rank_map[symbol] = 0.0
            try:
                dynamic_calib_prob_map[symbol] = float(row.get("calib_prob", 0.0) or 0.0)
            except Exception:
                dynamic_calib_prob_map[symbol] = 0.0
            try:
                dynamic_samples_map[symbol] = int(float(row.get("samples", row.get("symbol_samples", 0)) or 0))
            except Exception:
                dynamic_samples_map[symbol] = 0
        dynamic_adaptive_threshold = max(
            0.0,
            float(
                dynamic_status.get(
                    "adaptive_threshold",
                    dynamic_status.get(
                        "min_projected_edge_pct",
                        settings.get("crypto_dynamic_min_projected_edge_pct", settings.get("crypto_allocator_signal_floor", 0.15)),
                    ),
                )
                or 0.0
            ),
        )
        dynamic_calibration_recommended_threshold = max(
            0.0,
            float(
                dynamic_status.get(
                    "calibration_recommended_threshold",
                    dynamic_adaptive_threshold,
                )
                or dynamic_adaptive_threshold
            ),
        )
        min_crypto_calib_prob = max(
            0.0,
            min(
                1.0,
                float(settings.get("crypto_min_calib_prob_live_guarded", 0.50) or 0.50),
            ),
        )
        min_crypto_calib_samples = max(
            0,
            int(float(settings.get("crypto_min_samples_live_guarded", settings.get("adaptive_confidence_min_samples", 6)) or 6)),
        )
        reject_rows = dynamic_status.get("rejected", []) if isinstance(dynamic_status.get("rejected", []), list) else []
        ranked_count = len([r for r in dynamic_rank_rows if isinstance(r, dict)])
        rejected_count = len([r for r in reject_rows if isinstance(r, dict)])
        reject_rate_pct = 0.0
        if (ranked_count + rejected_count) > 0:
            reject_rate_pct = (100.0 * float(rejected_count)) / float(ranked_count + rejected_count)
        current_coins = dynamic_status.get("current_coins", []) if isinstance(dynamic_status.get("current_coins", []), list) else []
        dynamic_current_set = {str(x or "").strip().upper() for x in current_coins if str(x or "").strip()}
        try:
            stale_exit_enabled = bool(settings.get("crypto_stale_exit_enabled", True))
        except Exception:
            stale_exit_enabled = True
        try:
            stale_exit_grace_cycles = max(1, int(float(settings.get("crypto_stale_alignment_grace_cycles", 2) or 2)))
        except Exception:
            stale_exit_grace_cycles = 2
        try:
            stale_exit_max_per_cycle = max(1, int(float(settings.get("crypto_stale_max_exits_per_cycle", 2) or 2)))
        except Exception:
            stale_exit_max_per_cycle = 2
        try:
            stale_exit_min_notional_usd = max(1.0, float(settings.get("crypto_stale_min_notional_usd", 5.0) or 5.0))
        except Exception:
            stale_exit_min_notional_usd = 5.0
        try:
            default_hold_s = 7200 if str(profile_key) == "max_growth" else 3600
            stale_exit_min_hold_s = max(0, int(float(settings.get("crypto_stale_min_hold_seconds", default_hold_s) or default_hold_s)))
        except Exception:
            stale_exit_min_hold_s = 7200 if str(profile_key) == "max_growth" else 3600
        try:
            stale_exit_loss_cut_pct = float(settings.get("crypto_stale_loss_cut_pct", -2.0) or -2.0)
        except Exception:
            stale_exit_loss_cut_pct = -2.0
        try:
            stale_exit_force_short_count = max(
                0,
                int(
                    float(
                        settings.get(
                            "crypto_stale_force_short_count",
                            max(4, int(start_level) + 1),
                        )
                        or max(4, int(start_level) + 1)
                    )
                ),
            )
        except Exception:
            stale_exit_force_short_count = max(4, int(start_level) + 1)

        # Calculate total account value (robust: never drop a held coin to $0 on transient API misses)
        snapshot_ok = True

        # buying power
        try:
            raw_buying_power = account.get("buying_power", None)
            if raw_buying_power is None or raw_buying_power == "":
                raise ValueError("missing buying_power")
            buying_power = float(raw_buying_power)
        except Exception:
            buying_power = 0.0
            snapshot_ok = False

        # holdings list (treat missing/invalid holdings payload as transient error)
        try:
            holdings_list = holdings.get("results", None) if isinstance(holdings, dict) else None
            if not isinstance(holdings_list, list):
                holdings_list = []
                snapshot_ok = False
        except Exception:
            holdings_list = []
            snapshot_ok = False

        last_good_buying_power = None
        try:
            raw_last_buying_power = (self._last_good_account_snapshot or {}).get("buying_power", None)
            if raw_last_buying_power is not None and raw_last_buying_power != "":
                last_good_buying_power = float(raw_last_buying_power)
        except Exception:
            last_good_buying_power = None

        if used_cached_holdings and last_good_buying_power is not None:
            if (last_good_buying_power - float(buying_power)) > 5.0:
                snapshot_ok = False

        holdings_buy_value = 0.0
        holdings_sell_value = 0.0

        for holding in holdings_list:
            try:
                asset = holding.get("asset_code")
                if asset == "USDC":
                    continue

                qty = float(holding.get("total_quantity", 0.0))
                if qty <= 0.0:
                    continue

                sym = f"{asset}-USD"
                bp = float(current_buy_prices.get(sym, 0.0) or 0.0)
                sp = float(current_sell_prices.get(sym, 0.0) or 0.0)

                # If any held asset is missing a usable price this tick, do NOT allow a new "low" snapshot
                if bp <= 0.0 or sp <= 0.0:
                    snapshot_ok = False
                    continue

                holdings_buy_value += qty * bp
                holdings_sell_value += qty * sp
            except Exception:
                snapshot_ok = False
                continue

        total_account_value = buying_power + holdings_sell_value
        in_use = (holdings_sell_value / total_account_value) * 100 if total_account_value > 0 else 0.0

        # If this tick is incomplete, fall back to last known-good snapshot so the GUI chart never gets a bogus dip.
        if (not snapshot_ok) or (total_account_value <= 0.0):
            last = getattr(self, "_last_good_account_snapshot", None) or {}
            if last.get("total_account_value") is not None:
                total_account_value = float(last["total_account_value"])
                buying_power = float(last.get("buying_power", buying_power or 0.0))
                holdings_sell_value = float(last.get("holdings_sell_value", holdings_sell_value or 0.0))
                holdings_buy_value = float(last.get("holdings_buy_value", holdings_buy_value or 0.0))
                in_use = float(last.get("percent_in_trade", in_use or 0.0))
        else:
            # Save last complete snapshot
            self._last_good_account_snapshot = {
                "total_account_value": float(total_account_value),
                "buying_power": float(buying_power),
                "holdings_sell_value": float(holdings_sell_value),
                "holdings_buy_value": float(holdings_buy_value),
                "percent_in_trade": float(in_use),
            }

        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n--- Account Summary ---")
        print(f"Total Account Value: ${total_account_value:.2f}")
        print(f"Holdings Value: ${holdings_sell_value:.2f}")
        print(f"Percent In Trade: {in_use:.2f}%")
        print(
            f"Trailing PM: start +{self.pm_start_pct_no_dca:.2f}% (no DCA) / +{self.pm_start_pct_with_dca:.2f}% (with DCA) "
            f"| gap {self.trailing_gap_pct:.2f}%"
        )
        print("\n--- Current Trades ---")

        positions = {}
        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]
            full_symbol = f"{symbol}-USD"

            if full_symbol not in valid_symbols or symbol == "USDC":
                continue

            quantity = float(holding["total_quantity"])
            current_buy_price = current_buy_prices.get(full_symbol, 0)
            current_sell_price = current_sell_prices.get(full_symbol, 0)
            avg_cost_basis = self._fallback_avg_cost_basis(symbol, quantity=quantity)
            if avg_cost_basis > 0.0:
                cost_basis[str(symbol).upper().strip()] = float(avg_cost_basis)

            if avg_cost_basis > 0:
                gain_loss_percentage_buy = ((current_buy_price - avg_cost_basis) / avg_cost_basis) * 100
                gain_loss_percentage_sell = ((current_sell_price - avg_cost_basis) / avg_cost_basis) * 100
            else:
                gain_loss_percentage_buy = 0
                gain_loss_percentage_sell = 0
                print(f"  Warning: Average Cost Basis is 0 for {symbol}, Gain/Loss calculation skipped.")

            value = quantity * current_sell_price
            triggered_levels_count = len(self.dca_levels_triggered.get(symbol, []))
            triggered_levels = triggered_levels_count  # Number of DCA levels triggered

            # Determine the next DCA trigger for this coin (hardcoded % and optional neural level)
            next_stage = triggered_levels_count  # stage 0 == first DCA after entry (trade starts at neural level 3)

            # Hardcoded % for this stage (repeat -50% after we reach it)
            hard_next = self.dca_levels[next_stage] if next_stage < len(self.dca_levels) else self.dca_levels[-1]

            # Neural DCA applies to the levels BELOW the trade-start level.
            # Example: trade_start_level=3 => stages 0..3 map to N4..N7 (4 total).
            neural_dca_max = max(0, 7 - start_level)

            if next_stage < neural_dca_max:
                neural_next = start_level + 1 + next_stage
                next_dca_display = f"{hard_next:.2f}% / N{neural_next}"
            else:
                next_dca_display = f"{hard_next:.2f}%"

            # --- DCA DISPLAY LINE (show whichever trigger will be hit first: higher of NEURAL line vs HARD line) ---
            # Hardcoded gives an actual price line: cost_basis * (1 + hard_next%).
            # Neural gives an actual price line from low_bound_prices.html (N1..N7).
            dca_line_source = "HARD"
            dca_line_price = 0.0
            dca_line_pct = 0.0

            if avg_cost_basis > 0:
                # Hardcoded trigger line price
                hard_line_price = avg_cost_basis * (1.0 + (hard_next / 100.0))

                # Default to hardcoded unless neural line is higher (hit first)
                dca_line_price = hard_line_price

                if next_stage < neural_dca_max:
                    neural_level_needed_disp = start_level + 1 + next_stage
                    neural_levels = self._read_long_price_levels(symbol)  # highest->lowest == N1..N7

                    neural_line_price = 0.0
                    if len(neural_levels) >= neural_level_needed_disp:
                        neural_line_price = float(neural_levels[neural_level_needed_disp - 1])

                    # Whichever is higher will be hit first as price drops
                    if neural_line_price > dca_line_price:
                        dca_line_price = neural_line_price
                        dca_line_source = f"NEURAL N{neural_level_needed_disp}"


                # PnL% shown alongside DCA is the normal buy-side PnL%
                # (same calculation as GUI "Buy Price PnL": current buy/ask vs avg cost basis)
                dca_line_pct = gain_loss_percentage_buy




            dca_line_price_disp = self._fmt_price(dca_line_price) if avg_cost_basis > 0 else "N/A"

            # Set color code:
            # - DCA is green if we're above the chosen DCA line, red if we're below it
            # - SELL stays based on profit vs cost basis (your original behavior)
            if dca_line_pct >= 0:
                color = Fore.GREEN
            else:
                color = Fore.RED

            if gain_loss_percentage_sell >= 0:
                color2 = Fore.GREEN
            else:
                color2 = Fore.RED

            # --- Trailing PM display (per-coin, isolated) ---
            # Display uses current state if present; otherwise shows the base PM start line.
            trail_status = "N/A"
            pm_start_pct_disp = 0.0
            base_pm_line_disp = 0.0
            trail_line_disp = 0.0
            trail_peak_disp = 0.0
            above_disp = False
            dist_to_trail_pct = 0.0

            if avg_cost_basis > 0:
                pm_start_pct_disp = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line_disp = avg_cost_basis * (1.0 + (pm_start_pct_disp / 100.0))

                state = self.trailing_pm.get(symbol)
                if state is None:
                    trail_line_disp = base_pm_line_disp
                    trail_peak_disp = 0.0
                    active_disp = False
                else:
                    trail_line_disp = float(state.get("line", base_pm_line_disp))
                    trail_peak_disp = float(state.get("peak", 0.0))
                    active_disp = bool(state.get("active", False))

                above_disp = current_sell_price >= trail_line_disp
                # If we're already above the line, trailing is effectively "on/armed" (even if active flips this tick)
                trail_status = "ON" if (active_disp or above_disp) else "OFF"

                if trail_line_disp > 0:
                    dist_to_trail_pct = ((current_sell_price - trail_line_disp) / trail_line_disp) * 100.0
            self._write_current_price(symbol, current_buy_price)
            positions[symbol] = {
                "quantity": quantity,
                "avg_cost_basis": avg_cost_basis,
                "current_buy_price": current_buy_price,
                "current_sell_price": current_sell_price,
                "gain_loss_pct_buy": gain_loss_percentage_buy,
                "gain_loss_pct_sell": gain_loss_percentage_sell,
                "value_usd": value,
                "dca_triggered_stages": int(triggered_levels_count),
                "next_dca_display": next_dca_display,
                "dca_line_price": float(dca_line_price) if dca_line_price else 0.0,
                "dca_line_source": dca_line_source,
                "dca_line_pct": float(dca_line_pct) if dca_line_pct else 0.0,
                "trail_active": True if (trail_status == "ON") else False,
                "trail_line": float(trail_line_disp) if trail_line_disp else 0.0,
                "trail_peak": float(trail_peak_disp) if trail_peak_disp else 0.0,
                "dist_to_trail_pct": float(dist_to_trail_pct) if dist_to_trail_pct else 0.0,
            }
            align_snapshot = self._crypto_alignment_snapshot(
                symbol,
                start_level=start_level,
                dynamic_current_set=dynamic_current_set,
                profile_key=profile_key,
                dynamic_score=float(dynamic_rank_map.get(str(symbol).upper().strip(), 0.0) or 0.0),
                policy_mode=str(policy.get("mode", "") or ""),
                adaptive_dynamic_threshold=float(dynamic_adaptive_threshold),
                calibration_prob=float(dynamic_calib_prob_map.get(str(symbol).upper().strip(), 0.0) or 0.0),
                min_calibration_prob=(
                    float(min_crypto_calib_prob)
                    if int(dynamic_samples_map.get(str(symbol).upper().strip(), 0) or 0) >= int(min_crypto_calib_samples)
                    else 0.0
                ),
            )
            align_reasons = list(align_snapshot.get("reasons", []) or [])
            if bool(align_snapshot.get("aligned", True)):
                self._stale_alignment_streaks.pop(symbol, None)
            else:
                self._stale_alignment_streaks[symbol] = int(self._stale_alignment_streaks.get(symbol, 0) or 0) + 1
            stale_streak = int(self._stale_alignment_streaks.get(symbol, 0) or 0)
            aligned_with_strategy = bool(align_snapshot.get("aligned", True))
            positions[symbol]["aligned_with_strategy"] = bool(aligned_with_strategy)
            positions[symbol]["alignment_reasons"] = [str(r) for r in align_reasons[:3]]
            positions[symbol]["alignment_streak"] = int(stale_streak)


            print(
                f"\nSymbol: {symbol}"
                f"  |  DCA: {color}{dca_line_pct:+.2f}%{Style.RESET_ALL} @ {self._fmt_price(current_buy_price)} (Line: {dca_line_price_disp} {dca_line_source} | Next: {next_dca_display})"
                f"  |  Gain/Loss SELL: {color2}{gain_loss_percentage_sell:.2f}%{Style.RESET_ALL} @ {self._fmt_price(current_sell_price)}"
                f"  |  DCA Levels Triggered: {triggered_levels}"
                f"  |  Trade Value: ${value:.2f}"
            )




            if avg_cost_basis > 0:
                print(
                    f"  Trailing Profit Margin"
                    f"  |  Line: {self._fmt_price(trail_line_disp)}"
                    f"  |  Above: {above_disp}"
                )
            else:
                print("  PM/Trail: N/A (avg_cost_basis is 0)")


            should_force_stale_exit = (
                bool(stale_exit_enabled)
                and (not bool(align_snapshot.get("aligned", True)))
                and stale_streak >= int(stale_exit_grace_cycles)
                and stale_exit_count < int(stale_exit_max_per_cycle)
            )
            if should_force_stale_exit:
                try:
                    open_pos_row = (
                        (self._pnl_ledger.get("open_positions", {}) if isinstance(self._pnl_ledger.get("open_positions", {}), dict) else {})
                        .get(str(symbol).upper(), {})
                    )
                except Exception:
                    open_pos_row = {}
                if not isinstance(open_pos_row, dict):
                    open_pos_row = {}
                last_entry_ts_map = getattr(self, "_last_entry_ts", {})
                if not isinstance(last_entry_ts_map, dict):
                    last_entry_ts_map = {}
                try:
                    opened_ts = float(
                        open_pos_row.get("opened_ts", last_entry_ts_map.get(str(symbol).upper(), 0.0))
                        or 0.0
                    )
                except Exception:
                    opened_ts = 0.0
                position_age_s = max(0, int(time.time() - opened_ts)) if opened_ts > 0.0 else -1
                sell_pressure = int(align_snapshot.get("sell_count", 0) or 0)
                mild_loss = (gain_loss_percentage_sell < 0.0) and (gain_loss_percentage_sell > float(stale_exit_loss_cut_pct))
                hold_guard_active = (
                    position_age_s >= 0
                    and position_age_s < int(stale_exit_min_hold_s)
                    and mild_loss
                    and sell_pressure < int(stale_exit_force_short_count)
                )
                if hold_guard_active:
                    stale_exit_events.append(
                        {
                            "symbol": str(symbol),
                            "ok": False,
                            "reason": "stale_exit_hold_loss_guard",
                            "detail": (
                                f"Alignment stale but holding {symbol} to avoid a churn exit at mild loss "
                                f"({gain_loss_percentage_sell:+.2f}% > {float(stale_exit_loss_cut_pct):+.2f}%) "
                                f"during the first {int(stale_exit_min_hold_s)}s."
                            ),
                            "streak": int(stale_streak),
                            "age_s": int(position_age_s),
                            "sell_pressure": int(sell_pressure),
                            "reasons": [str(r) for r in align_reasons[:3]],
                        }
                    )
                    continue
                if current_sell_price <= 0.0 or value < float(stale_exit_min_notional_usd):
                    stale_exit_events.append(
                        {
                            "symbol": str(symbol),
                            "ok": False,
                            "reason": "stale_exit_price_or_notional_guard",
                            "detail": f"Alignment stale but price/notional guard blocked exit ({value:.2f} USD).",
                            "streak": int(stale_streak),
                            "reasons": [str(r) for r in align_reasons[:3]],
                        }
                    )
                else:
                    reason_text = "; ".join([str(r) for r in align_reasons[:2]]) or "position no longer matches strategy"
                    print(
                        f"  Policy stale exit for {symbol}: {reason_text} "
                        f"(streak {stale_streak}/{int(stale_exit_grace_cycles)})."
                    )
                    response = self.place_sell_order(
                        str(uuid.uuid4()),
                        "sell",
                        "market",
                        full_symbol,
                        quantity,
                        expected_price=current_sell_price,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_percentage_sell,
                        tag="POLICY_STALE_EXIT",
                    )
                    if response and isinstance(response, dict) and "errors" not in response:
                        stale_exit_count += 1
                        trades_made = True
                        skip_new_entries_this_cycle = True
                        self.trailing_pm.pop(symbol, None)
                        self.dca_levels_triggered.pop(symbol, None)
                        self._reset_dca_window_for_trade(symbol, sold=True)
                        self._stale_alignment_streaks.pop(symbol, None)
                        positions.pop(symbol, None)
                        exit_msg = f"Exited {symbol}: no longer aligned with current strategy."
                        self._set_status_note(exit_msg)
                        stale_exit_events.append(
                            {
                                "symbol": str(symbol),
                                "ok": True,
                                "reason": "policy_stale_exit",
                                "detail": reason_text,
                                "streak": int(stale_streak),
                                "reasons": [str(r) for r in align_reasons[:3]],
                            }
                        )
                        time.sleep(2)
                        continue
                    stale_exit_events.append(
                        {
                            "symbol": str(symbol),
                            "ok": False,
                            "reason": "broker_sell_failed",
                            "detail": reason_text,
                            "streak": int(stale_streak),
                            "reasons": [str(r) for r in align_reasons[:3]],
                        }
                    )



            # --- Trailing profit margin (0.5% trail gap) ---
            # PM "start line" is the normal 5% / 2.5% line (depending on DCA levels hit).
            # Trailing activates once price is ABOVE the PM start line, then line follows peaks up
            # by 0.5%. Forced sell happens ONLY when price goes from ABOVE the trailing line to BELOW it.
            if avg_cost_basis > 0:
                pm_start_pct = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line = avg_cost_basis * (1.0 + (pm_start_pct / 100.0))
                trail_gap = self.trailing_gap_pct / 100.0  # 0.5% => 0.005

                # If trailing settings changed since this coin's state was created, reset it.
                settings_sig = (
                    float(self.trailing_gap_pct),
                    float(self.pm_start_pct_no_dca),
                    float(self.pm_start_pct_with_dca),
                )

                state = self.trailing_pm.get(symbol)
                if (state is None) or (state.get("settings_sig") != settings_sig):
                    state = {
                        "active": False,
                        "line": base_pm_line,
                        "peak": 0.0,
                        "was_above": False,
                        "settings_sig": settings_sig,
                    }
                    self.trailing_pm[symbol] = state
                else:
                    # Keep signature up to date
                    state["settings_sig"] = settings_sig

                    # IMPORTANT:
                    # If trailing hasn't activated yet, this is just the PM line.
                    # It MUST track the current avg_cost_basis (so it can move DOWN after each DCA).
                    if not state.get("active", False):
                        state["line"] = base_pm_line
                    else:
                        # Once trailing is active, the line should never be below the base PM start line.
                        if state.get("line", 0.0) < base_pm_line:
                            state["line"] = base_pm_line

                # Use SELL price because that's what you actually get when you market sell
                above_now = current_sell_price >= state["line"]

                # Activate trailing once we first get above the base PM line
                if (not state["active"]) and above_now:
                    state["active"] = True
                    state["peak"] = current_sell_price

                # If active, update peak and move trailing line up behind it
                if state["active"]:
                    if current_sell_price > state["peak"]:
                        state["peak"] = current_sell_price

                    new_line = state["peak"] * (1.0 - trail_gap)
                    if new_line < base_pm_line:
                        new_line = base_pm_line
                    if new_line > state["line"]:
                        state["line"] = new_line

                    # Forced sell on cross from ABOVE -> BELOW trailing line
                    if state["was_above"] and (current_sell_price < state["line"]):
                        print(
                            f"  Trailing PM hit for {symbol}. "
                            f"Sell price {current_sell_price:.8f} fell below trailing line {state['line']:.8f}."
                        )
                        response = self.place_sell_order(
                            str(uuid.uuid4()),
                            "sell",
                            "market",
                            full_symbol,
                            quantity,
                            expected_price=current_sell_price,
                            avg_cost_basis=avg_cost_basis,
                            pnl_pct=gain_loss_percentage_sell,
                            tag="TRAIL_SELL",
                        )

                        if response and isinstance(response, dict) and "errors" not in response:
                            trades_made = True
                            self.trailing_pm.pop(symbol, None)  # clear per-coin trailing state on exit

                            # Trade ended -> reset rolling 24h DCA window for this coin
                            self._reset_dca_window_for_trade(symbol, sold=True)

                            print(f"  Successfully sold {quantity} {symbol}.")
                            time.sleep(5)
                            holdings = self.get_holdings()
                            continue


                # Save this tick’s position relative to the line (needed for “above -> below” detection)
                state["was_above"] = above_now



            # DCA (NEURAL or hardcoded %, whichever hits first for the current stage)
            # Trade starts at neural level 3 => trader is at stage 0.
            # Neural-driven DCA stages (max 4):
            #   stage 0 => neural 4 OR -2.5%
            #   stage 1 => neural 5 OR -5.0%
            #   stage 2 => neural 6 OR -10.0%
            #   stage 3 => neural 7 OR -20.0%
            # After that: hardcoded only (-30, -40, -50, then repeat -50 forever).
            current_stage = len(self.dca_levels_triggered.get(symbol, []))

            # Hardcoded loss % for this stage (repeat last level after list ends)
            hard_level = self.dca_levels[current_stage] if current_stage < len(self.dca_levels) else self.dca_levels[-1]
            hard_hit = gain_loss_percentage_buy <= hard_level

            # Neural trigger only for first 4 DCA stages
            neural_level_needed = None
            neural_level_now = None
            neural_hit = False
            if current_stage < 4:
                neural_level_needed = current_stage + 4
                neural_level_now = self._read_long_dca_signal(symbol)

                # Keep it sane: don't DCA from neural if we're not even below cost basis.
                neural_hit = (gain_loss_percentage_buy < 0) and (neural_level_now >= neural_level_needed)

            if hard_hit or neural_hit:
                if not bool(aligned_with_strategy):
                    dca_skip_msg = (
                        f"  Skipping DCA for {symbol}. "
                        f"Position is no longer aligned with strategy ({'; '.join(align_reasons[:2]) or 'alignment gate'})."
                    )
                    self._log_rate_limited(f"dca_alignment_block_{symbol}", dca_skip_msg, every_s=30.0)
                    continue
                if neural_hit and hard_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed} OR HARD {hard_level:.2f}%"
                elif neural_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed}"
                else:
                    reason = f"HARD {hard_level:.2f}%"

                print(f"  DCAing {symbol} (stage {current_stage + 1}) via {reason}.")

                print(f"  Current Value: ${value:.2f}")
                dca_amount = value * float(DCA_MULTIPLIER or 0.0)
                print(f"  DCA Amount: ${dca_amount:.2f}")
                print(f"  Buying Power: ${buying_power:.2f}")


                recent_dca = self._dca_window_count(symbol)
                if recent_dca >= int(getattr(self, "max_dca_buys_per_24h", 2)):
                    print(
                        f"  Skipping DCA for {symbol}. "
                        f"Already placed {recent_dca} DCA buys in the last 24h (max {self.max_dca_buys_per_24h})."
                    )

                elif dca_amount <= buying_power:
                    if not self._can_place_buy(symbol, dca_amount, value, total_account_value, holdings_sell_value):
                        continue
                    response = self.place_buy_order(
                        str(uuid.uuid4()),
                        "buy",
                        "market",
                        full_symbol,
                        dca_amount,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_percentage_buy,
                        tag="DCA",
                    )

                    print(f"  Buy Response: {response}")
                    if response and "errors" not in response:
                        # record that we completed THIS stage (no matter what triggered it)
                        self.dca_levels_triggered.setdefault(symbol, []).append(current_stage)

                        # Only record a DCA buy timestamp on success (so skips never advance anything)
                        self._note_dca_buy(symbol)

                        # DCA changes avg_cost_basis, so the PM line must be rebuilt from the new basis
                        # (this will re-init to 5% if DCA=0, or 2.5% if DCA>=1)
                        self.trailing_pm.pop(symbol, None)

                        trades_made = True
                        print(f"  Successfully placed DCA buy order for {symbol}.")
                    else:
                        print(f"  Failed to place DCA buy order for {symbol}.")

                else:
                    print(f"  Skipping DCA for {symbol}. Not enough funds.")

            else:
                pass

        try:
            active_hold_symbols = {
                str(row.get("asset_code", "") or "").strip().upper()
                for row in list(holdings.get("results", []) or [])
                if isinstance(row, dict)
                and float(row.get("total_quantity", 0.0) or 0.0) > 0.0
                and str(row.get("asset_code", "") or "").strip().upper() not in {"", "USDC"}
            }
            for coin in list(self._stale_alignment_streaks.keys()):
                if str(coin or "").strip().upper() not in active_hold_symbols:
                    self._stale_alignment_streaks.pop(coin, None)
        except Exception:
            pass


        # --- ensure GUI gets bid/ask lines even for coins not currently held ---
        try:
            for sym in crypto_symbols:
                if sym in positions:
                    continue

                full_symbol = f"{sym}-USD"
                if full_symbol not in valid_symbols or sym == "USDC":
                    continue

                current_buy_price = current_buy_prices.get(full_symbol, 0.0)
                current_sell_price = current_sell_prices.get(full_symbol, 0.0)

                # keep the per-coin current price file behavior for consistency
                self._write_current_price(sym, current_buy_price)

                positions[sym] = {
                    "quantity": 0.0,
                    "avg_cost_basis": 0.0,
                    "current_buy_price": current_buy_price,
                    "current_sell_price": current_sell_price,
                    "gain_loss_pct_buy": 0.0,
                    "gain_loss_pct_sell": 0.0,
                    "value_usd": 0.0,
                    "dca_triggered_stages": int(len(self.dca_levels_triggered.get(sym, []))),
                    "next_dca_display": "",
                    "dca_line_price": 0.0,
                    "dca_line_source": "N/A",
                    "dca_line_pct": 0.0,
                    "trail_active": False,
                    "trail_line": 0.0,
                    "trail_peak": 0.0,
                    "dist_to_trail_pct": 0.0,
                }
        except Exception:
            pass

        try:
            live_positions = {}
            for symbol, pos in positions.items():
                if not isinstance(pos, dict):
                    continue
                try:
                    qty = float(pos.get("quantity", 0.0) or 0.0)
                except Exception:
                    qty = 0.0
                if qty > 0.0:
                    coin = str(symbol).upper().strip()
                    prev = dict(self._last_good_positions_snapshot.get(coin, {}) or {})
                    merged = dict(prev)
                    merged.update(dict(pos))
                    try:
                        merged_avg = float(merged.get("avg_cost_basis", 0.0) or 0.0)
                    except Exception:
                        merged_avg = 0.0
                    try:
                        prev_avg = float(prev.get("avg_cost_basis", 0.0) or 0.0)
                    except Exception:
                        prev_avg = 0.0
                    if merged_avg <= 0.0 and prev_avg > 0.0:
                        merged["avg_cost_basis"] = prev_avg
                    live_positions[coin] = merged
            if live_positions:
                self._last_good_positions_snapshot = live_positions
        except Exception:
            pass

        market_health = {
            "data_ok": bool(valid_symbols) and (not bool(used_cached_holdings)),
            "broker_ok": bool(isinstance(account, dict) and trading_pairs),
            "orders_ok": True,
            "drift_warning": bool(used_cached_holdings),
        }
        policy = build_market_automation_policy(
            market="crypto",
            settings=settings,
            profile_key=profile_key,
            broker_mode=("paper" if bool(settings.get("paper_only_unless_checklist_green", False)) else "live"),
            account_value_usd=float(total_account_value),
            buying_power_usd=float(buying_power),
            open_positions=int(len([row for row in holdings.get("results", []) if isinstance(row, dict)])),
            runtime_alerts=runtime_alerts,
            market_health=market_health,
            compliance_state={},
            reject_rate_pct=float(reject_rate_pct),
            reject_rate_limit_pct=float(settings.get("runtime_alert_scan_reject_crit_pct", 85.0) or 85.0),
            fallback_active=bool(used_cached_holdings),
            fallback_age_s=max(0, int(time.time() - float(getattr(self, "_last_good_holdings_ts", 0.0) or 0.0)))
            if bool(used_cached_holdings)
            else 0,
            fallback_hard_block_age_s=int(float(settings.get("market_fallback_snapshot_max_age_s", 1800.0) or 1800.0)),
            loss_streak=0,
            max_loss_streak=3,
        )
        policy_size_scale = max(0.20, float(policy.get("size_multiplier", 1.0) or 1.0))
        effective_limits = policy.get("effective_limits", {}) if isinstance(policy.get("effective_limits", {}), dict) else {}
        effective_crypto_max_spread_bps = max(
            1.0,
            float(
                effective_limits.get(
                    "max_spread_bps",
                    settings.get("crypto_max_spread_bps", 150.0),
                )
                or 150.0
            ),
        )
        try:
            self.entry_cooldown_seconds = max(
                60.0,
                float(
                    effective_limits.get(
                        "rotation_cooldown_s",
                        settings.get("crypto_dynamic_rotation_cooldown_s", self.entry_cooldown_seconds),
                    )
                    or self.entry_cooldown_seconds
                ),
            )
        except Exception:
            pass
        if not bool(policy.get("allow_new_entries", True)):
            policy_reason = str(
                ((policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}).get("reasons", [""])
                or [""])[0]
                or "Runtime trust is too low for new crypto entries"
            ).strip()
            if policy_reason:
                self._set_status_note(policy_reason)
                entry_fail_reasons.append(policy_reason)
        if stale_exit_count > 0:
            refresh_msg = f"Exited {stale_exit_count} stale position(s); waiting one cycle before new entries."
            entry_fail_reasons.append(refresh_msg)
            skip_new_entries_this_cycle = True
            cur_note = str(getattr(self, "_status_note", "") or "").strip()
            if cur_note:
                if refresh_msg.lower() not in cur_note.lower():
                    self._set_status_note(f"{cur_note} | {refresh_msg}")
            else:
                self._set_status_note(refresh_msg)

        if not trading_pairs:
            if not entry_fail_reasons:
                entry_fail_reasons.append("Trading pairs unavailable from broker API")
        else:
            alloc_pct = float(START_ALLOC_PCT or 0.5)
            allocation_base_usd = total_account_value * (alloc_pct / 100.0)
            if allocation_base_usd < 0.5:
                allocation_base_usd = 0.5
            allocation_policy_usd = max(0.5, float(allocation_base_usd) * float(policy_size_scale))

            holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]

            def _open_positions_count() -> int:
                count = 0
                for full in list(holding_full_symbols or []):
                    base = str(full).split("-", 1)[0].strip().upper()
                    if not base or base == "USDC":
                        continue
                    count += 1
                return count

            candidate_rows: List[Dict[str, Any]] = []
            for sym in list(crypto_symbols or []):
                base_symbol = str(sym or "").strip().upper()
                if not base_symbol:
                    continue
                full_symbol = f"{base_symbol}-USD"
                buy_count = self._read_long_dca_signal(base_symbol)
                sell_count = self._read_short_dca_signal(base_symbol)
                dyn_score = float(dynamic_rank_map.get(base_symbol, 0.0) or 0.0)
                candidate_rows.append(
                    {
                        "symbol": base_symbol,
                        "full_symbol": full_symbol,
                        "buy_count": int(buy_count),
                        "sell_count": int(sell_count),
                        "dynamic_score": float(dyn_score),
                    }
                )
            candidate_rows = sorted(
                candidate_rows,
                key=lambda row: (
                    float(row.get("dynamic_score", 0.0) or 0.0),
                    int(row.get("buy_count", 0) or 0),
                    -int(row.get("sell_count", 0) or 0),
                ),
                reverse=True,
            )

            min_runtime_trust_score, min_confidence_score = self._profile_quality_thresholds(profile_key)
            selected_symbol = ""
            if (not bool(policy.get("allow_new_entries", True))) or bool(skip_new_entries_this_cycle):
                candidate_rows = []
            for cand in candidate_rows:
                if int(MAX_OPEN_POSITIONS or 0) > 0:
                    open_count = _open_positions_count()
                    if open_count >= int(MAX_OPEN_POSITIONS):
                        msg = f"Max open positions reached ({open_count}/{int(MAX_OPEN_POSITIONS)})."
                        self._set_status_note(msg)
                        self._log_rate_limited("max_open_positions_reached", msg)
                        entry_fail_reasons.append(msg)
                        break
                base_symbol = str(cand.get("symbol", "") or "").strip().upper()
                full_symbol = str(cand.get("full_symbol", "") or "").strip().upper()
                if (not base_symbol) or (not full_symbol):
                    continue
                if full_symbol in holding_full_symbols:
                    continue
                if dynamic_current_set and base_symbol not in dynamic_current_set:
                    entry_fail_reasons.append(f"Rotation set blocked {base_symbol} (not in active coin rotation)")
                    continue
                last_exit_ts = float(self._last_exit_ts.get(base_symbol, 0.0) or 0.0)
                if last_exit_ts > 0.0 and (time.time() - last_exit_ts) < float(self.entry_cooldown_seconds):
                    cd_left = int(max(0.0, float(self.entry_cooldown_seconds) - (time.time() - last_exit_ts)))
                    entry_fail_reasons.append(f"Rotation cooldown active for {base_symbol} ({cd_left}s)")
                    continue
                px_buy = float(current_buy_prices.get(full_symbol, 0.0) or 0.0)
                px_sell = float(current_sell_prices.get(full_symbol, 0.0) or 0.0)
                if full_symbol not in valid_symbols or px_buy <= 0.0 or px_sell <= 0.0:
                    msg = f"Skipping {base_symbol}: missing bid/ask price for entry."
                    self._set_status_note(msg)
                    self._log_rate_limited(f"missing_price_{base_symbol}", msg)
                    entry_fail_reasons.append(msg)
                    continue
                folder = base_paths.get(base_symbol, os.path.join(main_dir, base_symbol))
                long_path = os.path.join(folder, "long_dca_signal.txt")
                short_path = os.path.join(folder, "short_dca_signal.txt")
                if (not os.path.isfile(long_path)) or (not os.path.isfile(short_path)):
                    msg = f"Skipping {base_symbol}: missing signal file(s)."
                    self._set_status_note(msg)
                    self._log_rate_limited(f"missing_signal_{base_symbol}", msg)
                    entry_fail_reasons.append(msg)
                    continue
                buy_count = int(cand.get("buy_count", 0) or 0)
                sell_count = int(cand.get("sell_count", 0) or 0)
                dynamic_score = float(cand.get("dynamic_score", 0.0) or 0.0)
                calib_prob = float(dynamic_calib_prob_map.get(base_symbol, 0.0) or 0.0)
                calib_samples = int(dynamic_samples_map.get(base_symbol, 0) or 0)
                calib_gate_min = float(min_crypto_calib_prob) if calib_samples >= int(min_crypto_calib_samples) else 0.0
                gate_eval = self._evaluate_crypto_signal_gate(
                    profile_key=profile_key,
                    start_level=int(start_level),
                    buy_count=int(buy_count),
                    sell_count=int(sell_count),
                    dynamic_score=float(dynamic_score),
                    policy_mode=str(policy.get("mode", "") or ""),
                    adaptive_dynamic_threshold=float(dynamic_adaptive_threshold),
                    calibration_prob=float(calib_prob),
                    min_calibration_prob=float(calib_gate_min),
                )
                signal_gate_debug = {
                    "symbol": str(base_symbol),
                    "gate_mode": str(gate_eval.get("gate_mode", "blocked") or "blocked"),
                    "requirement": str(gate_eval.get("requirement_text", "") or ""),
                    "min_long_count": int(gate_eval.get("min_long_count", start_level) or start_level),
                    "max_short_count": int(gate_eval.get("max_short_count", 0) or 0),
                    "allow_dynamic_fallback": bool(gate_eval.get("allow_dynamic_fallback", False)),
                    "min_dynamic_score": float(gate_eval.get("min_dynamic_score", 0.0) or 0.0),
                    "dynamic_score": round(float(gate_eval.get("dynamic_score", 0.0) or 0.0), 4),
                    "adaptive_dynamic_threshold": round(float(gate_eval.get("adaptive_dynamic_threshold", dynamic_adaptive_threshold) or dynamic_adaptive_threshold), 4),
                    "calibration_prob": round(float(calib_prob), 4),
                    "calibration_samples": int(calib_samples),
                    "min_calibration_prob": round(float(calib_gate_min), 4),
                    "calibration_gate_applied": bool(gate_eval.get("calibration_gate_applied", False)),
                }
                if not bool(gate_eval.get("passed", False)):
                    fail_reason = str(gate_eval.get("failure_reason", "") or "").strip()
                    req_text = str(gate_eval.get("requirement_text", "") or "").strip()
                    if fail_reason and req_text:
                        entry_fail_reasons.append(
                            f"Signal gate blocked {base_symbol} ({fail_reason}; {req_text})"
                        )
                    elif fail_reason:
                        entry_fail_reasons.append(f"Signal gate blocked {base_symbol} ({fail_reason})")
                    else:
                        entry_fail_reasons.append(f"Signal gate blocked {base_symbol} ({req_text or 'policy threshold'})")
                    continue
                entry_align_eval = self._evaluate_crypto_entry_alignment_gate(
                    profile_key=profile_key,
                    start_level=int(start_level),
                    buy_count=int(buy_count),
                    sell_count=int(sell_count),
                    dynamic_score=float(dynamic_score),
                    policy_mode=str(policy.get("mode", "") or ""),
                    adaptive_dynamic_threshold=float(dynamic_adaptive_threshold),
                    calibration_prob=float(calib_prob),
                    min_calibration_prob=float(calib_gate_min),
                )
                entry_alignment_debug = {
                    "symbol": str(base_symbol),
                    "alignment_mode": str(entry_align_eval.get("alignment_mode", "not_evaluated") or "not_evaluated"),
                    "requirement": str(entry_align_eval.get("requirement_text", "") or ""),
                    "min_long_count": int(entry_align_eval.get("min_long_count", start_level) or start_level),
                    "min_dynamic_score": float(entry_align_eval.get("min_dynamic_score", 0.0) or 0.0),
                    "long_headroom": int(entry_align_eval.get("long_headroom", 0) or 0),
                    "dynamic_margin": float(entry_align_eval.get("dynamic_margin", 0.0) or 0.0),
                    "min_dynamic_long_count": int(entry_align_eval.get("min_dynamic_long_count", 0) or 0),
                    "dynamic_score": round(float(entry_align_eval.get("dynamic_score", 0.0) or 0.0), 4),
                    "passed": bool(entry_align_eval.get("passed", False)),
                    "adaptive_dynamic_threshold": round(float(entry_align_eval.get("adaptive_dynamic_threshold", dynamic_adaptive_threshold) or dynamic_adaptive_threshold), 4),
                    "calibration_prob": round(float(entry_align_eval.get("calibration_prob", calib_prob) or calib_prob), 4),
                    "min_calibration_prob": round(float(entry_align_eval.get("min_calibration_prob", calib_gate_min) or calib_gate_min), 4),
                    "calibration_gate_applied": bool(entry_align_eval.get("calibration_gate_applied", False)),
                }
                if not bool(entry_align_eval.get("passed", False)):
                    align_reason = str(entry_align_eval.get("failure_reason", "") or "").strip()
                    req_text = str(entry_align_eval.get("requirement_text", "") or "").strip()
                    if align_reason and req_text:
                        entry_fail_reasons.append(
                            f"Entry alignment gate blocked {base_symbol} ({align_reason}; {req_text})"
                        )
                    elif align_reason:
                        entry_fail_reasons.append(f"Entry alignment gate blocked {base_symbol} ({align_reason})")
                    else:
                        entry_fail_reasons.append(
                            f"Entry alignment gate blocked {base_symbol} ({req_text or 'alignment threshold'})"
                        )
                    continue
                stale_entry_guard_reason = ""
                if str(entry_align_eval.get("alignment_mode", "") or "").strip().lower() == "dynamic_fallback_buffer":
                    try:
                        dynamic_margin_live = float(dynamic_score) - float(entry_align_eval.get("min_dynamic_score", 0.0) or 0.0)
                    except Exception:
                        dynamic_margin_live = 0.0
                    try:
                        dynamic_buffer_floor = float(
                            settings.get(
                                "crypto_entry_dynamic_buffer_min",
                                0.18 if str(profile_key) == "max_growth" else 0.12,
                            )
                            or (0.18 if str(profile_key) == "max_growth" else 0.12)
                        )
                    except Exception:
                        dynamic_buffer_floor = 0.18 if str(profile_key) == "max_growth" else 0.12
                    try:
                        dynamic_buy_floor = max(
                            1,
                            int(
                                float(
                                    settings.get(
                                        "crypto_entry_dynamic_min_long_count",
                                        max(1, int(start_level) - 1),
                                    )
                                    or max(1, int(start_level) - 1)
                                )
                            ),
                        )
                    except Exception:
                        dynamic_buy_floor = max(1, int(start_level) - 1)
                    if int(sell_count) > 0 and int(buy_count) < int(dynamic_buy_floor):
                        stale_entry_guard_reason = (
                            f"Entry stale-risk guard blocked {base_symbol} "
                            f"(short pressure S{int(sell_count)} with long N{int(buy_count)} < N{int(dynamic_buy_floor)})"
                        )
                    elif int(buy_count) < int(dynamic_buy_floor) and float(dynamic_margin_live) < float(dynamic_buffer_floor):
                        stale_entry_guard_reason = (
                            f"Entry stale-risk guard blocked {base_symbol} "
                            f"(dynamic margin {float(dynamic_margin_live):.3f} < {float(dynamic_buffer_floor):.3f} "
                            f"with long N{int(buy_count)} below N{int(dynamic_buy_floor)})"
                        )
                    entry_alignment_debug["stale_entry_dynamic_margin"] = float(round(float(dynamic_margin_live), 4))
                    entry_alignment_debug["stale_entry_dynamic_buffer_floor"] = float(round(float(dynamic_buffer_floor), 4))
                    entry_alignment_debug["stale_entry_dynamic_buy_floor"] = int(dynamic_buy_floor)
                if stale_entry_guard_reason:
                    entry_alignment_debug["stale_entry_guard"] = str(stale_entry_guard_reason)
                    entry_fail_reasons.append(str(stale_entry_guard_reason))
                    continue
                signal_score, required_score = self._signal_score_for_candidate(
                    float(dynamic_score),
                    buy_count,
                    start_level,
                    adaptive_threshold=float(dynamic_adaptive_threshold),
                )
                mid_px = (px_buy + px_sell) / 2.0 if (px_buy > 0.0 and px_sell > 0.0) else 0.0
                spread_bps = 0.0
                if mid_px > 0.0:
                    spread_bps = abs((px_buy - px_sell) / mid_px) * 10000.0
                projected_exposure_pct = (((holdings_sell_value + allocation_policy_usd) / total_account_value) * 100.0) if total_account_value > 0.0 else 0.0
                runtime_trust_score = float(
                    (
                        (policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}).get(
                            "score",
                            0.0,
                        )
                    )
                    or 0.0
                )
                runtime_reasons = (
                    (policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}).get(
                        "reasons",
                        [],
                    )
                )
                runtime_reasons = runtime_reasons if isinstance(runtime_reasons, list) else []
                compliance_reason = str(
                    (runtime_reasons[0] if runtime_reasons else "")
                    or "Crypto runtime trust gate is active"
                ).strip()
                fallback_age_s = (
                    max(0, int(time.time() - float(getattr(self, "_last_good_holdings_ts", 0.0) or 0.0)))
                    if bool(used_cached_holdings)
                    else 0
                )
                quality_eval = evaluate_trade_quality(
                    market="crypto",
                    signal_score=float(signal_score),
                    required_score=float(required_score),
                    data_quality_ok=bool(market_health.get("data_ok", True)),
                    broker_ok=bool(market_health.get("broker_ok", True)),
                    runtime_trust_score=float(runtime_trust_score),
                    runtime_alert_severity=str(runtime_alerts.get("severity", "ok") or "ok"),
                    compliance_allowed=bool(policy.get("allow_new_entries", True)),
                    compliance_reason=str(compliance_reason),
                    fallback_active=bool(used_cached_holdings),
                    fallback_age_s=int(fallback_age_s),
                    fallback_hard_block_age_s=int(float(settings.get("market_fallback_snapshot_max_age_s", 1800.0) or 1800.0)),
                    reject_rate_pct=float(reject_rate_pct),
                    reject_rate_limit_pct=float(settings.get("runtime_alert_scan_reject_crit_pct", 85.0) or 85.0),
                    spread_bps=float(spread_bps),
                    max_slippage_bps=float(effective_crypto_max_spread_bps),
                    loss_streak=0,
                    max_loss_streak=3,
                    exposure_usage_pct=float(projected_exposure_pct),
                    min_runtime_trust_score=float(min_runtime_trust_score),
                    min_confidence_score=float(min_confidence_score),
                )
                trade_quality_eval = dict(quality_eval)
                if str(quality_eval.get("decision", "block") or "block").strip().lower() != "allow":
                    q_reasons = quality_eval.get("block_reasons", []) if isinstance(quality_eval.get("block_reasons", []), list) else []
                    fail_reason = str((q_reasons[0] if q_reasons else "Trade-quality gate blocked crypto entry") or "").strip()
                    entry_fail_reasons.append(fail_reason)
                    continue

                quality_size_mult = max(0.20, min(1.25, float(quality_eval.get("size_multiplier", 1.0) or 1.0)))
                proposed_notional = max(0.5, float(allocation_policy_usd) * float(quality_size_mult))
                if not self._can_place_buy(base_symbol, proposed_notional, 0.0, total_account_value, holdings_sell_value):
                    entry_fail_reasons.append(f"Risk cap blocked {base_symbol} entry")
                    continue

                candidate_age_s = 0
                if int(dynamic_updated_ts) > 0:
                    candidate_age_s = max(0, int(time.time()) - int(dynamic_updated_ts))
                allocator_eval = evaluate_cross_market_allocation(
                    hub_dir=(os.path.dirname(CRYPTO_DYNAMIC_STATUS_PATH) or HUB_DATA_DIR),
                    settings=settings,
                    market="crypto",
                    candidate_id=base_symbol,
                    candidate_side="long",
                    signal_score=float(signal_score),
                    required_score=float(required_score),
                    trade_quality=quality_eval if isinstance(quality_eval, dict) else {},
                    automation_policy=policy if isinstance(policy, dict) else {},
                    projected_trade_value_usd=float(proposed_notional),
                    market_exposure_usd=float(holdings_sell_value),
                    account_value_usd=float(total_account_value),
                    buying_power_usd=float(buying_power),
                    spread_bps=float(spread_bps),
                    max_slippage_bps=float(effective_crypto_max_spread_bps),
                    candidate_age_s=int(candidate_age_s),
                    loss_streak=0,
                    max_loss_streak=3,
                )
                allocator_decision = str(allocator_eval.get("decision", "allow") or "allow").strip().lower()
                if allocator_decision != "allow":
                    opportunity_eval = dict(allocator_eval)
                    alloc_reason = str(
                        allocator_eval.get("summary", "")
                        or (
                            (allocator_eval.get("reasons", []) if isinstance(allocator_eval.get("reasons", []), list) else [""])
                            or [""]
                        )[0]
                        or "portfolio allocator deprioritized crypto entry"
                    ).strip()
                    entry_fail_reasons.append(f"Portfolio allocator: {alloc_reason}")
                    continue
                opportunity_eval = dict(allocator_eval)
                allocator_size_scale = max(
                    0.25,
                    min(1.0, float((allocator_eval.get("size_multiplier", 1.0) if isinstance(allocator_eval, dict) else 1.0) or 1.0)),
                )
                if abs(float(allocator_size_scale) - 1.0) >= 0.001:
                    proposed_notional = max(0.5, float(proposed_notional) * float(allocator_size_scale))

                response = self.place_buy_order(
                    str(uuid.uuid4()),
                    "buy",
                    "market",
                    full_symbol,
                    proposed_notional,
                    audit_meta={
                        "score": float(signal_score),
                        "required_score": float(required_score),
                        "calib_prob": float(calib_prob),
                    },
                )
                if response and "errors" not in response:
                    selected_symbol = base_symbol
                    entry_fail_reasons = []
                    entry_size_scale = quality_size_mult * float(allocator_size_scale)
                    trades_made = True
                    self._set_status_note(f"Entry placed for {base_symbol}")
                    self.dca_levels_triggered[base_symbol] = []
                    self._reset_dca_window_for_trade(base_symbol, sold=False)
                    self.trailing_pm.pop(base_symbol, None)
                    print(
                        f"Starting new trade for {full_symbol} "
                        f"(score={signal_score:.4f}, req={required_score:.4f}, quality={float(quality_eval.get('confidence_score', 0.0) or 0.0):.1f}). "
                        f"Allocating ${proposed_notional:.2f}."
                    )
                    time.sleep(5)
                    holdings = self.get_holdings()
                    holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]
                    break
                reject_msg = f"Order rejected for {base_symbol}"
                entry_fail_reasons.append(reject_msg)
                self._append_execution_audit(
                    {
                        "event": "entry_fail",
                        "ok": False,
                        "symbol": str(base_symbol),
                        "side": "buy",
                        "score": float(signal_score),
                        "required_score": float(required_score),
                        "calib_prob": float(calib_prob),
                        "notional": float(proposed_notional),
                        "msg": reject_msg,
                        "payload": response if isinstance(response, dict) else {},
                    }
                )

            if (not selected_symbol) and (not entry_fail_reasons):
                entry_fail_reasons.append("No crypto candidates passed policy and quality gates")

        # If any trades were made, recalculate the cost basis
        if trades_made:
            time.sleep(5)
            print("Trades were made in this iteration. Recalculating cost basis...")
            new_cost_basis = self.calculate_cost_basis()
            if new_cost_basis:
                self.cost_basis = new_cost_basis
                print("Cost basis recalculated successfully.")
            else:
                print("Failed to recalculcate cost basis.")
            self.initialize_dca_levels()

        entry_eval_top_reason, entry_eval_reason_counts = self._entry_fail_summary(entry_fail_reasons)
        if (not str(getattr(self, "_status_note", "") or "").strip()) and entry_eval_top_reason:
            self._set_status_note(entry_eval_top_reason)
        policy_summary = str(policy.get("summary", "") or "").strip()
        if policy_summary:
            cur_note = str(getattr(self, "_status_note", "") or "").strip()
            if cur_note and policy_summary.lower() not in cur_note.lower():
                self._set_status_note(f"{cur_note} | {policy_summary}")
            elif not cur_note:
                self._set_status_note(policy_summary)
        allocator_summary = str(opportunity_eval.get("summary", "") or "").strip() if isinstance(opportunity_eval, dict) else ""
        if allocator_summary:
            cur_note = str(getattr(self, "_status_note", "") or "").strip()
            if cur_note and allocator_summary.lower() not in cur_note.lower():
                self._set_status_note(f"{cur_note} | {allocator_summary}")
            elif not cur_note:
                self._set_status_note(allocator_summary)
        runtime_trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
        quality_layers = trade_quality_eval.get("layers", {}) if isinstance(trade_quality_eval.get("layers", {}), dict) else {}
        trade_quality_evaluated = bool(isinstance(trade_quality_eval, dict) and trade_quality_eval)
        allocator_evaluated = bool(isinstance(opportunity_eval, dict) and opportunity_eval)
        allocator_reasons = opportunity_eval.get("reasons", []) if isinstance(opportunity_eval.get("reasons", []), list) else []
        allocator_top_reason = str((allocator_reasons[0] if allocator_reasons else "") or "").strip()
        allocator_ai = opportunity_eval.get("openai_decision", {}) if isinstance(opportunity_eval.get("openai_decision", {}), dict) else {}
        trade_confidence_score = float(trade_quality_eval.get("confidence_score", 0.0) or 0.0) if trade_quality_evaluated else 0.0
        quality_size_scale = float(trade_quality_eval.get("size_multiplier", 1.0) or 1.0) if trade_quality_evaluated else 1.0
        try:
            allocator_size_from_eval = float(opportunity_eval.get("size_multiplier", 1.0) or 1.0)
        except Exception:
            allocator_size_from_eval = 1.0
        allocator_size_from_eval = max(0.25, min(1.0, float(allocator_size_from_eval)))
        try:
            openai_confidence = float(allocator_ai.get("portfolio_confidence", 0.0) or 0.0)
        except Exception:
            openai_confidence = 0.0
        openai_confidence = max(0.0, min(1.0, float(openai_confidence)))
        entry_gate_flags = {
            "data_quality_ok": bool(market_health.get("data_ok", True)),
            "broker_ok": bool(market_health.get("broker_ok", True)),
            "orders_ok": bool(market_health.get("orders_ok", True)),
            "drift_warning": bool(market_health.get("drift_warning", False)),
            "cached_fallback_active": bool(used_cached_holdings),
            "cached_fallback_age_s": max(0, int(time.time() - float(getattr(self, "_last_good_holdings_ts", 0.0) or 0.0)))
            if bool(used_cached_holdings)
            else 0,
            "reject_rate_pct": round(float(reject_rate_pct), 4),
            "reject_rate_max_pct": round(float(settings.get("runtime_alert_scan_reject_crit_pct", 85.0) or 85.0), 4),
            "adaptive_threshold_dynamic": round(float(dynamic_adaptive_threshold), 4),
            "adaptive_threshold_calibration_recommended": round(float(dynamic_calibration_recommended_threshold), 4),
            "min_calibration_prob": round(float(min_crypto_calib_prob), 4),
            "min_calibration_samples": int(min_crypto_calib_samples),
            "max_spread_bps": round(float(effective_crypto_max_spread_bps), 4),
            "runtime_trust_score": round(float(runtime_trust.get("score", 0.0) or 0.0), 4),
            "runtime_trust_mode": str(runtime_trust.get("mode", "") or ""),
            "policy_mode": str(policy.get("mode", "") or ""),
            "policy_profile": str(policy.get("profile", "") or ""),
            "policy_size_scale": round(float(policy_size_scale), 4),
            "trade_quality_evaluated": bool(trade_quality_evaluated),
            "trade_quality_decision": str(trade_quality_eval.get("decision", "not_evaluated") if trade_quality_evaluated else "not_evaluated"),
            "trade_confidence_score": round(float(trade_confidence_score), 4),
            "trade_quality_size_scale": round(float(quality_size_scale), 4),
            "portfolio_allocator_evaluated": bool(allocator_evaluated),
            "portfolio_allocator_decision": str(opportunity_eval.get("decision", "not_evaluated") if allocator_evaluated else "not_evaluated"),
            "portfolio_allocator_best_market": str(opportunity_eval.get("best_market", "") or ""),
            "portfolio_allocator_score": round(float(opportunity_eval.get("current_market_score", 0.0) or 0.0), 4),
            "portfolio_allocator_top_reason": allocator_top_reason,
            "portfolio_allocator_capital_constrained": bool(opportunity_eval.get("capital_constrained", False)) if allocator_evaluated else False,
            "portfolio_allocator_size_scale": round(float(allocator_size_from_eval), 4) if allocator_evaluated else 1.0,
            "portfolio_allocator_decision_source": str(opportunity_eval.get("decision_source", "") or "") if allocator_evaluated else "",
            "openai_decision_active": bool(allocator_ai.get("active", False)),
            "openai_decision_status": str(allocator_ai.get("status", "") or ""),
            "openai_decision": str(allocator_ai.get("decision", "") or ""),
            "openai_decision_best_market": str(allocator_ai.get("best_market", "") or ""),
            "openai_decision_confidence": round(float(openai_confidence), 4),
            "openai_decision_applied": bool(allocator_ai.get("applied", False)),
            "signal_quality_pass": bool(quality_layers.get("signal_quality", False)),
            "execution_quality_pass": bool(quality_layers.get("execution_quality", False)),
            "compliance_permission_pass": bool(quality_layers.get("compliance_permission", False)),
            "runtime_trust_pass": bool(quality_layers.get("runtime_trust", False)),
            "rotation_cooldown_s": int(max(0.0, float(self.entry_cooldown_seconds or 0.0))),
            "stale_exit_enabled": bool(stale_exit_enabled),
            "stale_exit_grace_cycles": int(stale_exit_grace_cycles),
            "stale_exit_count": int(stale_exit_count),
            "stale_exit_max_per_cycle": int(stale_exit_max_per_cycle),
            "stale_exit_min_notional_usd": round(float(stale_exit_min_notional_usd), 4),
            "stale_exit_min_hold_s": int(stale_exit_min_hold_s),
            "stale_exit_loss_cut_pct": round(float(stale_exit_loss_cut_pct), 4),
            "stale_exit_force_short_count": int(stale_exit_force_short_count),
            "skip_new_entries_this_cycle": bool(skip_new_entries_this_cycle),
            "signal_gate_symbol": str(signal_gate_debug.get("symbol", "") or ""),
            "signal_gate_mode": str(signal_gate_debug.get("gate_mode", "") or ""),
            "signal_gate_requirement": str(signal_gate_debug.get("requirement", "") or ""),
            "signal_gate_min_long_count": int(signal_gate_debug.get("min_long_count", start_level) or start_level),
            "signal_gate_max_short_count": int(signal_gate_debug.get("max_short_count", 0) or 0),
            "signal_gate_dynamic_fallback": bool(signal_gate_debug.get("allow_dynamic_fallback", False)),
            "signal_gate_min_dynamic_score": round(float(signal_gate_debug.get("min_dynamic_score", 0.0) or 0.0), 4),
            "signal_gate_dynamic_score": round(float(signal_gate_debug.get("dynamic_score", 0.0) or 0.0), 4),
            "signal_gate_adaptive_dynamic_threshold": round(float(signal_gate_debug.get("adaptive_dynamic_threshold", 0.0) or 0.0), 4),
            "signal_gate_calibration_prob": round(float(signal_gate_debug.get("calibration_prob", 0.0) or 0.0), 4),
            "signal_gate_calibration_samples": int(signal_gate_debug.get("calibration_samples", 0) or 0),
            "signal_gate_min_calibration_prob": round(float(signal_gate_debug.get("min_calibration_prob", 0.0) or 0.0), 4),
            "signal_gate_calibration_gate_applied": bool(signal_gate_debug.get("calibration_gate_applied", False)),
            "entry_alignment_symbol": str(entry_alignment_debug.get("symbol", "") or ""),
            "entry_alignment_mode": str(entry_alignment_debug.get("alignment_mode", "") or ""),
            "entry_alignment_requirement": str(entry_alignment_debug.get("requirement", "") or ""),
            "entry_alignment_min_long_count": int(entry_alignment_debug.get("min_long_count", start_level) or start_level),
            "entry_alignment_min_dynamic_score": round(float(entry_alignment_debug.get("min_dynamic_score", 0.0) or 0.0), 4),
            "entry_alignment_long_headroom": int(entry_alignment_debug.get("long_headroom", 0) or 0),
            "entry_alignment_dynamic_margin": round(float(entry_alignment_debug.get("dynamic_margin", 0.0) or 0.0), 4),
            "entry_alignment_min_dynamic_long_count": int(entry_alignment_debug.get("min_dynamic_long_count", 0) or 0),
            "entry_alignment_dynamic_score": round(float(entry_alignment_debug.get("dynamic_score", 0.0) or 0.0), 4),
            "entry_alignment_adaptive_dynamic_threshold": round(float(entry_alignment_debug.get("adaptive_dynamic_threshold", 0.0) or 0.0), 4),
            "entry_alignment_calibration_prob": round(float(entry_alignment_debug.get("calibration_prob", 0.0) or 0.0), 4),
            "entry_alignment_min_calibration_prob": round(float(entry_alignment_debug.get("min_calibration_prob", 0.0) or 0.0), 4),
            "entry_alignment_calibration_gate_applied": bool(entry_alignment_debug.get("calibration_gate_applied", False)),
            "entry_alignment_pass": bool(entry_alignment_debug.get("passed", False)),
            "entry_alignment_stale_entry_guard": str(entry_alignment_debug.get("stale_entry_guard", "") or ""),
            "entry_alignment_stale_dynamic_margin": round(float(entry_alignment_debug.get("stale_entry_dynamic_margin", 0.0) or 0.0), 4),
            "entry_alignment_stale_dynamic_buffer_floor": round(float(entry_alignment_debug.get("stale_entry_dynamic_buffer_floor", 0.0) or 0.0), 4),
            "entry_alignment_stale_dynamic_buy_floor": int(entry_alignment_debug.get("stale_entry_dynamic_buy_floor", 0) or 0),
        }

        # --- GUI HUB STATUS WRITE ---
        try:
            base_alloc_pct = float(START_ALLOC_PCT or 0.5)
            trade_notional_base = max(0.5, float(total_account_value) * (base_alloc_pct / 100.0))
            trade_notional_policy = max(0.5, float(trade_notional_base) * float(policy_size_scale))
            open_positions_count = 0
            for row in list(positions.values() if isinstance(positions, dict) else []):
                if not isinstance(row, dict):
                    continue
                try:
                    qty = float(row.get("quantity", 0.0) or 0.0)
                except Exception:
                    qty = 0.0
                if qty > 0.0:
                    open_positions_count += 1
            status = {
                "timestamp": time.time(),
                "status_note": getattr(self, "_status_note", ""),
                "state": "READY",
                "trader_state": "AUTO",
                "msg": str(getattr(self, "_status_note", "") or ""),
                "account": {
                    "total_account_value": total_account_value,
                    "buying_power": buying_power,
                    "holdings_sell_value": holdings_sell_value,
                    "holdings_buy_value": holdings_buy_value,
                    "percent_in_trade": in_use,
                    # trailing PM config (matches what's printed above current trades)
                    "pm_start_pct_no_dca": float(getattr(self, "pm_start_pct_no_dca", 0.0)),
                    "pm_start_pct_with_dca": float(getattr(self, "pm_start_pct_with_dca", 0.0)),
                    "trailing_gap_pct": float(getattr(self, "trailing_gap_pct", 0.0)),
                },
                "positions": positions,
                "open_positions": int(open_positions_count),
                "trade_notional_base_usd": round(float(trade_notional_base), 4),
                "trade_notional_policy_usd": round(float(trade_notional_policy), 4),
                "adaptive_threshold": round(float(dynamic_adaptive_threshold), 4),
                "calibration_recommended_threshold": round(float(dynamic_calibration_recommended_threshold), 4),
                "entry_size_scale": round(float(entry_size_scale), 4),
                "entry_eval_total": int(len(entry_fail_reasons)),
                "entry_eval_failed": int(len(entry_fail_reasons) > 0),
                "entry_eval_top_reason": str(entry_eval_top_reason),
                "entry_eval_reason_counts": dict(entry_eval_reason_counts),
                "automation_policy": policy if isinstance(policy, dict) else {},
                "trade_quality": trade_quality_eval if isinstance(trade_quality_eval, dict) else {},
                "opportunity_allocator": opportunity_eval if isinstance(opportunity_eval, dict) else {},
                "entry_gate_flags": dict(entry_gate_flags),
                "stale_exit_count": int(stale_exit_count),
                "stale_exit_events": list(stale_exit_events[:12]),
            }
            now_ts = float(status["timestamp"])
            if (now_ts - float(getattr(self, "_last_account_value_history_write_ts", 0.0) or 0.0)) >= 15.0:
                self._append_jsonl(
                    ACCOUNT_VALUE_HISTORY_PATH,
                    {"ts": status["timestamp"], "total_account_value": total_account_value},
                )
                self._last_account_value_history_write_ts = now_ts
            self._write_trader_status(status)
        except Exception:
            pass




    def run(self):
        while True:
            try:
                self.manage_trades()
                time.sleep(max(0.25, float(getattr(self, "_loop_sleep_ok", CRYPTO_TRADER_LOOP_SLEEP_S))))
            except Exception as e:
                print(traceback.format_exc())
                time.sleep(max(0.5, float(getattr(self, "_loop_sleep_error", CRYPTO_TRADER_ERROR_SLEEP_S))))

    @staticmethod
    def _write_current_price(base_symbol: str, price: float) -> None:
        sym = str(base_symbol or "").strip().upper()
        if not sym:
            return
        path = os.path.join(CURRENT_PRICE_DIR, f"{sym}.txt")
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(str(float(price)))
            os.replace(tmp, path)
        except Exception:
            pass

if __name__ == "__main__":
    trading_bot = CryptoAPITrading()
    trading_bot.run()
