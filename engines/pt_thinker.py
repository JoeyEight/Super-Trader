import os
import time
import random
import requests
import subprocess
from app.http_utils import retry_after_from_requests_exception
try:
	from kucoin.client import Market  # type: ignore
except Exception:
	Market = None  # type: ignore


class _KucoinMarketFallback:
	def __init__(self, url: str = "https://api.kucoin.com"):
		self.url = str(url or "https://api.kucoin.com").rstrip("/")
		self._session = requests.Session()

	def get_kline(self, symbol: str, kline_type: str):
		params = {"symbol": str(symbol or ""), "type": str(kline_type or "")}
		resp = self._session.get(f"{self.url}/api/v1/market/candles", params=params, timeout=12)
		resp.raise_for_status()
		payload = resp.json() if resp.content else {}
		if not isinstance(payload, dict):
			return []
		code = str(payload.get("code", "") or "").strip()
		if code and code != "200000":
			msg = str(payload.get("msg", "") or "").strip()
			raise RuntimeError(f"KuCoin error {code}: {msg}")
		data = payload.get("data", [])
		return data if isinstance(data, list) else []


if Market is not None:
	market = Market(url='https://api.kucoin.com')
else:
	market = _KucoinMarketFallback(url='https://api.kucoin.com')

# KuCoin request shaping: single-process throttle + short cache + backoff on failures.
_KUCOIN_MIN_INTERVAL_SEC = float(os.environ.get("PT_KUCOIN_MIN_INTERVAL_SEC", "0.40"))
_KUCOIN_CACHE_TTL_SEC = float(os.environ.get("PT_KUCOIN_CACHE_TTL_SEC", "2.5"))
_KUCOIN_STALE_MAX_SEC = float(os.environ.get("PT_KUCOIN_STALE_MAX_SEC", "120.0"))
_KUCOIN_UNSUPPORTED_COOLDOWN_S = float(os.environ.get("PT_KUCOIN_UNSUPPORTED_COOLDOWN_S", "21600.0"))
_CRYPTO_PRICE_ERR_LOG_COOLDOWN_S = float(os.environ.get("PT_CRYPTO_PRICE_ERR_LOG_COOLDOWN_S", "120.0"))
_RH_UNSUPPORTED_SYMBOL_COOLDOWN_S = float(os.environ.get("PT_RH_UNSUPPORTED_SYMBOL_COOLDOWN_S", "21600.0"))
_kucoin_last_call_mono = 0.0
_kucoin_error_streak = 0
_kucoin_cooldown_until_mono = 0.0
_kucoin_cache = {}
_kucoin_unsupported_until_mono = {}
_rh_unsupported_until_mono = {}
_kucoin_tune_cache = {
	"mtime": None,
	"vals": (
		_KUCOIN_MIN_INTERVAL_SEC,
		_KUCOIN_CACHE_TTL_SEC,
		_KUCOIN_STALE_MAX_SEC,
		_KUCOIN_UNSUPPORTED_COOLDOWN_S,
		_CRYPTO_PRICE_ERR_LOG_COOLDOWN_S,
	),
}


def _kucoin_tune_values() -> tuple[float, float, float, float, float]:
	global _kucoin_tune_cache
	try:
		settings_path = resolve_settings_path(BASE_DIR) or _GUI_SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
		if not settings_path or not os.path.isfile(settings_path):
			return _kucoin_tune_cache["vals"]
		mtime = os.path.getmtime(settings_path)
		if _kucoin_tune_cache.get("mtime") == mtime:
			return _kucoin_tune_cache["vals"]
		data = read_settings_file(settings_path, module_name="pt_thinker") or {}
		min_interval = float(data.get("kucoin_min_interval_sec", _KUCOIN_MIN_INTERVAL_SEC) or _KUCOIN_MIN_INTERVAL_SEC)
		cache_ttl = float(data.get("kucoin_cache_ttl_sec", _KUCOIN_CACHE_TTL_SEC) or _KUCOIN_CACHE_TTL_SEC)
		stale_max = float(data.get("kucoin_stale_max_sec", _KUCOIN_STALE_MAX_SEC) or _KUCOIN_STALE_MAX_SEC)
		unsupported_cd = float(data.get("kucoin_unsupported_cooldown_s", _KUCOIN_UNSUPPORTED_COOLDOWN_S) or _KUCOIN_UNSUPPORTED_COOLDOWN_S)
		price_log_cd = float(data.get("crypto_price_error_log_cooldown_s", _CRYPTO_PRICE_ERR_LOG_COOLDOWN_S) or _CRYPTO_PRICE_ERR_LOG_COOLDOWN_S)
		min_interval = max(0.20, min(3.0, min_interval))
		cache_ttl = max(0.5, min(30.0, cache_ttl))
		stale_max = max(10.0, min(3600.0, stale_max))
		unsupported_cd = max(300.0, min(172800.0, unsupported_cd))
		price_log_cd = max(5.0, min(3600.0, price_log_cd))
		_kucoin_tune_cache = {"mtime": mtime, "vals": (min_interval, cache_ttl, stale_max, unsupported_cd, price_log_cd)}
		return _kucoin_tune_cache["vals"]
	except Exception:
		return _kucoin_tune_cache["vals"]


def _extract_retry_after_seconds(exc: Exception) -> float:
	return retry_after_from_requests_exception(exc, max_wait_s=300.0)


def _is_unsupported_pair_error(err_msg: str) -> bool:
	msg = str(err_msg or "").strip().lower()
	if not msg:
		return False
	return ("unsupported trading pair" in msg) or ("\"code\":\"400100\"" in msg) or ("'code':'400100'" in msg)


def _is_retryable_price_exception(exc: Exception) -> bool:
	msg = str(exc or "").strip().lower()
	if not msg:
		return True
	hard_tokens = (
		"missing robinhood credentials",
		"api key is empty",
		"failed to decode robinhood private key",
		"invalid symbol",
		"validation_error",
		"unsupported trading pair",
		"400100",
		"best_bid_ask returned no results",
	)
	if any(tok in msg for tok in hard_tokens):
		return False
	retry_tokens = (
		"429",
		"rate limit",
		"too many requests",
		"timeout",
		"timed out",
		"temporarily",
		"connection",
		"network",
		"dns",
		"503",
		"502",
		"504",
		"connection reset",
	)
	return any(tok in msg for tok in retry_tokens)


def _is_robinhood_invalid_symbol_exception(exc: Exception) -> bool:
	msg = str(exc or "").strip().lower()
	if not msg:
		return False
	return ("invalid symbol" in msg) or ("validation_error" in msg)


def _robinhood_symbol_is_temporarily_unsupported(symbol: str) -> bool:
	now = time.monotonic()
	until = float(_rh_unsupported_until_mono.get(str(symbol or "").strip().upper(), 0.0) or 0.0)
	return now < until


def _mark_robinhood_symbol_unsupported(symbol: str) -> None:
	sym = str(symbol or "").strip().upper()
	if not sym:
		return
	_rh_unsupported_until_mono[sym] = time.monotonic() + float(max(300.0, _RH_UNSUPPORTED_SYMBOL_COOLDOWN_S))


def _get_kline_shaped(symbol: str, kline_type: str, retry_forever: bool = True, max_attempts: int = 4):
	global _kucoin_last_call_mono, _kucoin_error_streak, _kucoin_cooldown_until_mono, _kucoin_cache, _kucoin_unsupported_until_mono
	min_interval_sec, cache_ttl_sec, stale_max_sec, unsupported_cd_s, price_log_cd_s = _kucoin_tune_values()

	symbol = str(symbol or "").strip().upper()
	kline_type = str(kline_type or "").strip()
	key = (symbol, kline_type)

	now = time.monotonic()
	cached = _kucoin_cache.get(key)
	unsupported_until = float(_kucoin_unsupported_until_mono.get(key, 0.0) or 0.0)
	if now < unsupported_until:
		if isinstance(cached, dict):
			data = cached.get("data", [])
			if isinstance(data, list):
				return data
		return []
	if isinstance(cached, dict):
		ts = float(cached.get("ts", 0.0) or 0.0)
		data = cached.get("data", [])
		if (now - ts) <= cache_ttl_sec:
			return data if isinstance(data, list) else []

	attempts = 0
	while True:
		now = time.monotonic()

		if now < _kucoin_cooldown_until_mono:
			time.sleep(min(2.0, _kucoin_cooldown_until_mono - now))
			continue

		since_last = now - _kucoin_last_call_mono
		if since_last < min_interval_sec:
			time.sleep(min_interval_sec - since_last)

		try:
			data = market.get_kline(symbol, kline_type)
			_kucoin_last_call_mono = time.monotonic()
			_kucoin_error_streak = 0
			out = data if isinstance(data, list) else []
			_kucoin_cache[key] = {"ts": _kucoin_last_call_mono, "data": out}
			return out
		except Exception as e:
			_kucoin_last_call_mono = time.monotonic()
			attempts += 1
			msg = str(e)
			if _is_unsupported_pair_error(msg):
				_kucoin_unsupported_until_mono[key] = time.monotonic() + float(unsupported_cd_s)
				log_throttled(
					f"pt_thinker:kucoin_unsupported:{symbol}:{kline_type}",
					f"[pt_thinker] kucoin unsupported pair {symbol} ({kline_type}); suppressing retries for {int(unsupported_cd_s)}s",
					cooldown_s=max(60.0, min(float(unsupported_cd_s), 7200.0)),
				)
				if isinstance(cached, dict):
					data = cached.get("data", [])
					if isinstance(data, list):
						return data
				return []
			_kucoin_error_streak = min(8, _kucoin_error_streak + 1)
			retry_after_s = _extract_retry_after_seconds(e)
			backoff_exp = min(20.0, (0.8 * (2 ** (_kucoin_error_streak - 1))) + random.uniform(0.0, 0.7))
			backoff = max(backoff_exp, retry_after_s)
			_kucoin_cooldown_until_mono = time.monotonic() + backoff

			if (
				"Requests" not in msg
				and "429" not in msg
				and "Too many requests" not in msg
				and "Connection reset by peer" not in msg
				and "Failed to resolve 'api.kucoin.com'" not in msg
			):
				log_throttled(
					f"pt_thinker:kline_error:{symbol}:{kline_type}",
					f"[pt_thinker] kucoin kline error {symbol} {kline_type}: {msg}",
					cooldown_s=float(price_log_cd_s),
				)

			if (not retry_forever) and attempts >= max_attempts:
				if isinstance(cached, dict):
					ts = float(cached.get("ts", 0.0) or 0.0)
					data = cached.get("data", [])
					if (time.monotonic() - ts) <= stale_max_sec:
						return data if isinstance(data, list) else []
				raise
import sys
import datetime
import traceback
import linecache
import base64
import calendar
import hashlib
import hmac
from datetime import datetime
try:
	import psutil  # type: ignore
except Exception:
	psutil = None  # type: ignore
import logging
import json
import uuid
from app.credential_utils import get_robinhood_creds_from_env, get_robinhood_creds_from_files
from app.confidence_calibration import build_market_confidence_calibration
from app.news_event_provider import blend_score_with_news, build_unified_news_event_context
from app.path_utils import resolve_runtime_paths, resolve_settings_path, read_settings_file, log_once, log_throttled
from app.rejection_replay import recommend_threshold_from_scores, replay_target_entries_for_market

from nacl.signing import SigningKey

# -----------------------------
# Robinhood market-data (current ASK), same source as rhcb.py trader:
#   GET /api/v1/crypto/marketdata/best_bid_ask/?symbol=BTC-USD
#   use result["ask_inclusive_of_buy_spread"]
# -----------------------------
ROBINHOOD_BASE_URL = "https://trading.robinhood.com"

_RH_MD = None  # lazy-init so import doesn't explode if creds missing


class RobinhoodMarketData:
    def __init__(self, api_key: str, base64_private_key: str, base_url: str = ROBINHOOD_BASE_URL, timeout: int = 10):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("Robinhood API key is empty.")

        try:
            raw_private = base64.b64decode((base64_private_key or "").strip())
            self.private_key = SigningKey(raw_private)
        except Exception as e:
            raise RuntimeError(f"Failed to decode Robinhood private key: {e}")

        self.session = requests.Session()

    def _get_current_timestamp(self) -> int:
        return int(time.time())

    def _get_authorization_header(self, method: str, path: str, body: str, timestamp: int) -> dict:
        # matches the trader's signing format
        method = method.upper()
        body = body or ""
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))
        signature_b64 = base64.b64encode(signed.signature).decode("utf-8")

        return {
            "x-api-key": self.api_key,
            "x-timestamp": str(timestamp),
            "x-signature": signature_b64,
            "Content-Type": "application/json",
        }

    def make_api_request(self, method: str, path: str, body: str = "") -> dict:
        url = f"{self.base_url}{path}"
        ts = self._get_current_timestamp()
        headers = self._get_authorization_header(method, path, body, ts)

        resp = self.session.request(method=method.upper(), url=url, headers=headers, data=body or None, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Robinhood HTTP {resp.status_code}: {resp.text}")
        return resp.json()

    def get_current_ask(self, symbol: str) -> float:
        symbol = (symbol or "").strip().upper()
        path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
        data = self.make_api_request("GET", path)

        if not data or "results" not in data or not data["results"]:
            raise RuntimeError(f"Robinhood best_bid_ask returned no results for {symbol}: {data}")

        result = data["results"][0]
        # EXACTLY like rhcb.py's get_price(): ask_inclusive_of_buy_spread
        return float(result["ask_inclusive_of_buy_spread"])


def robinhood_current_ask(symbol: str) -> float:
    """
    Returns Robinhood current BUY price (ask_inclusive_of_buy_spread) for symbols like 'BTC-USD'.
    Reads creds from environment first, then credential files (keys/ preferred, legacy root fallback).
    """
    global _RH_MD
    if _RH_MD is None:
        api_key, priv_b64 = get_robinhood_creds_from_env()
        if not api_key or not priv_b64:
            project_base = globals().get("BASE_DIR") or os.path.dirname(os.path.abspath(__file__))
            api_key, priv_b64 = get_robinhood_creds_from_files(str(project_base))
            if not api_key or not priv_b64:
                raise RuntimeError(
                    "Missing Robinhood credentials. Set POWERTRADER_RH_API_KEY + POWERTRADER_RH_PRIVATE_B64, "
                    "or provide credential files in keys/ (legacy root files still supported)."
                )

        _RH_MD = RobinhoodMarketData(api_key=api_key, base64_private_key=priv_b64)

    return _RH_MD.get_current_ask(symbol)


def restart_program():
	"""Restarts the current program (no CLI args; uses hardcoded COIN_SYMBOLS)."""
	try:
		os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
	except Exception as e:
		print(f'Error during program restart: {e}')



def PrintException():
	exc_type, exc_obj, tb = sys.exc_info()

	# walk to the innermost frame (where the error actually happened)
	while tb and tb.tb_next:
		tb = tb.tb_next

	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename

	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	print('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj))

restarted = 'no'
short_started = 'no'
long_started = 'no'
minute = 0
last_minute = 0

# -----------------------------
# GUI SETTINGS (coins list)
# -----------------------------
BASE_DIR, _GUI_SETTINGS_PATH, HUB_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "pt_thinker")

_gui_settings_cache = {
	"mtime": None,
	"path": None,
	"coins": ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE'],  # fallback defaults
	"main_neural_dir": os.path.join(BASE_DIR, "market_data", "coins"),
}

DEFAULT_DYNAMIC_POOL = [
	"BTC", "ETH", "XRP", "BNB", "DOGE", "SOL", "ADA", "PAXG",
	"AVAX", "LINK", "LTC", "UNI", "AAVE", "DOT", "ATOM", "MATIC",
]


def _resolve_coin_workspace_dir(raw_main_dir) -> str:
	preferred_workspace = os.path.join(BASE_DIR, "market_data", "coins")
	try:
		mndir = str(raw_main_dir or "").strip()
	except Exception:
		mndir = ""
	if mndir and (not os.path.isabs(mndir)):
		mndir = os.path.abspath(os.path.join(BASE_DIR, mndir))
	if mndir and (os.path.abspath(mndir) != os.path.abspath(BASE_DIR)):
		return mndir
	return preferred_workspace


def _cached_current_price(sym: str, max_age_s: float = 900.0):
	symbol = str(sym or "").strip().upper()
	if not symbol:
		return None
	candidates = [
		os.path.join(HUB_DIR, "current_prices", f"{symbol}.txt"),
		os.path.join(BASE_DIR, "current_prices", f"{symbol}.txt"),
	]
	for path in candidates:
		try:
			if not os.path.isfile(path):
				continue
			age_s = max(0.0, time.time() - float(os.path.getmtime(path)))
			if age_s > float(max_age_s):
				continue
			with open(path, "r", encoding="utf-8") as f:
				val = float(str(f.read() or "").strip())
			if val > 0.0:
				return float(val)
		except Exception:
			continue
	return None


def _kucoin_live_current_price(sym: str):
	symbol = str(sym or "").strip().upper()
	if not symbol:
		return None
	pair = f"{symbol}-USDT"
	raw = _get_kline_shaped(pair, "1min", retry_forever=False, max_attempts=3)
	if not isinstance(raw, list) or not raw:
		return None
	for row in raw:
		try:
			close_price = float(row[2])  # close
		except Exception:
			continue
		if close_price > 0:
			return float(close_price)
	return None
	
def _load_gui_coins() -> list:
	"""
	Reads gui_settings.json and returns settings["coins"] as an uppercased list.
	Caches by mtime so it is cheap to call frequently.
	"""
	try:
		settings_path = resolve_settings_path(BASE_DIR) or _GUI_SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
		if not os.path.isfile(settings_path):
			return list(_gui_settings_cache["coins"])

		mtime = os.path.getmtime(settings_path)
		if _gui_settings_cache["mtime"] == mtime and _gui_settings_cache.get("path") == settings_path:
			return list(_gui_settings_cache["coins"])

		data = read_settings_file(settings_path, module_name="pt_thinker") or {}

		coins = data.get("coins", None)
		if not isinstance(coins, list) or not coins:
			coins = list(_gui_settings_cache["coins"])

		coins = [str(c).strip().upper() for c in coins if str(c).strip()]
		if not coins:
			coins = list(_gui_settings_cache["coins"])

		main_neural_dir = _resolve_coin_workspace_dir(data.get("main_neural_dir", _gui_settings_cache.get("main_neural_dir")))
		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["path"] = settings_path
		_gui_settings_cache["coins"] = coins
		_gui_settings_cache["main_neural_dir"] = main_neural_dir
		return list(coins)
	except Exception:
		return list(_gui_settings_cache["coins"])


def _load_gui_settings_with_path() -> tuple[dict, str]:
	try:
		settings_path = resolve_settings_path(BASE_DIR) or _GUI_SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
		data = read_settings_file(settings_path, module_name="pt_thinker") or {}
		return (data if isinstance(data, dict) else {}), str(settings_path or "")
	except Exception:
		return {}, ""


def _safe_read_json(path: str) -> dict:
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		return data if isinstance(data, dict) else {}
	except Exception:
		return {}


def _parse_symbol_list(raw) -> list[str]:
	if isinstance(raw, list):
		vals = raw
	else:
		vals = str(raw or "").replace("\n", ",").split(",")
	out = []
	seen = set()
	for tok in vals:
		s = str(tok or "").strip().upper()
		if not s or s in seen:
			continue
		seen.add(s)
		out.append(s)
	return out


def _is_coin_training(sym: str) -> bool:
	try:
		st = _safe_read_json(os.path.join(coin_folder(sym), "trainer_status.json"))
		return str((st or {}).get("state", "") or "").strip().upper() == "TRAINING"
	except Exception:
		return False


def _read_held_coins() -> set[str]:
	path = os.path.join(HUB_DIR, "trader_data.json")
	data = _safe_read_json(path)
	pos = data.get("positions", {}) if isinstance(data, dict) else {}
	out = set()
	if isinstance(pos, dict):
		for sym, row in pos.items():
			try:
				qty = float((row or {}).get("quantity", 0.0) or 0.0)
			except Exception:
				qty = 0.0
			if qty > 0:
				out.add(str(sym or "").strip().upper())
	return out


def _score_coin_projection(sym: str) -> float:
	pair = f"{sym}-USDT"
	raw = _get_kline_shaped(pair, "1hour", retry_forever=False, max_attempts=3)
	closes = []
	for row in (raw or []):
		try:
			closes.append(float(row[2]))  # close
		except Exception:
			continue
	if len(closes) < 30:
		raise RuntimeError("insufficient_1h_bars")
	closes = list(reversed(closes))  # newest->oldest to oldest->newest
	last = closes[-1]
	c6 = closes[max(0, len(closes) - 7)]
	c24 = closes[max(0, len(closes) - 25)]
	chg6 = ((last - c6) / c6) * 100.0 if c6 > 0 else 0.0
	chg24 = ((last - c24) / c24) * 100.0 if c24 > 0 else 0.0
	step = []
	for i in range(1, len(closes)):
		p0 = closes[i - 1]
		p1 = closes[i]
		if p0 > 0:
			step.append(abs(((p1 - p0) / p0) * 100.0))
	vol = (sum(step[-12:]) / max(1, len(step[-12:]))) if step else 0.0
	return float((chg6 * 0.60) + (chg24 * 0.30) + (vol * 0.10))


def _write_settings_coins(settings_path: str, coins: list[str]) -> bool:
	try:
		if settings_path and os.path.isfile(settings_path):
			data = _safe_read_json(settings_path)
		else:
			data = {}
		data["coins"] = list(coins)
		_atomic_write_json(settings_path or os.path.join(BASE_DIR, "gui_settings.json"), data)
		_gui_settings_cache["mtime"] = None
		return True
	except Exception:
		return False

# Initial coin list (will be kept live via _sync_coins_from_settings())
COIN_SYMBOLS = _load_gui_coins()
CURRENT_COINS = list(COIN_SYMBOLS)

def coin_folder(sym: str) -> str:
	sym = sym.upper()
	base_dir = _resolve_coin_workspace_dir(_gui_settings_cache.get("main_neural_dir"))
	try:
		os.makedirs(base_dir, exist_ok=True)
	except Exception:
		pass
	return os.path.join(base_dir, sym)


# --- training freshness gate (mirrors pt_hub.py) ---
_TRAINING_STALE_SECONDS = 14 * 24 * 60 * 60  # 14 days

def _coin_is_trained(sym: str) -> bool:
	"""
	Training freshness gate:

	pt_trainer.py writes `trainer_last_training_time.txt` in the coin folder
	when training starts. If that file is missing OR older than 14 days, we treat
	the coin as NOT TRAINED.

	This is intentionally the same logic as pt_hub.py so runner behavior matches
	what the GUI shows.
	"""

	try:
		folder = coin_folder(sym)
		stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
		if not os.path.isfile(stamp_path):
			return False
		with open(stamp_path, "r", encoding="utf-8") as f:
			raw = (f.read() or "").strip()
		ts = float(raw) if raw else 0.0
		if ts <= 0:
			return False
		return (time.time() - ts) <= _TRAINING_STALE_SECONDS
	except Exception:
		return False

# --- GUI HUB "runner ready" gate file (read by gui_hub.py Start All toggle) ---

RUNNER_READY_PATH = os.path.join(HUB_DIR, "runner_ready.json")

def _atomic_write_json(path: str, data: dict) -> None:
	try:
		tmp = path + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)
		os.replace(tmp, path)
	except (PermissionError, OSError, TypeError, ValueError) as exc:
		log_once(
			f"pt_thinker:_atomic_write_json:{path}:{type(exc).__name__}",
			f"[pt_thinker._atomic_write_json] path={path} {type(exc).__name__}: {exc}",
		)

def _write_runner_ready(ready: bool, stage: str, ready_coins=None, total_coins: int = 0) -> None:
	obj = {
		"timestamp": time.time(),
		"ready": bool(ready),
		"stage": stage,
		"ready_coins": ready_coins or [],
		"total_coins": int(total_coins or 0),
	}
	_atomic_write_json(RUNNER_READY_PATH, obj)


DYNAMIC_STATUS_PATH = os.path.join(HUB_DIR, "crypto_dynamic_status.json")
CRYPTO_SCAN_DIR = os.path.join(HUB_DIR, "crypto")
CRYPTO_SCANNER_RANKINGS_PATH = os.path.join(CRYPTO_SCAN_DIR, "scanner_rankings.jsonl")
_dynamic_last_scan_ts = 0.0
_dynamic_trainer_procs: dict[str, subprocess.Popen] = {}
_dynamic_last_rotation_ts = 0.0


def _append_jsonl(path: str, row: dict) -> None:
	try:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(row, separators=(",", ":")) + "\n")
	except Exception:
		pass


def _clamp(value: float, lo: float, hi: float) -> float:
	return float(max(lo, min(hi, float(value))))


def _cleanup_dynamic_trainers() -> None:
	for sym in list(_dynamic_trainer_procs.keys()):
		proc = _dynamic_trainer_procs.get(sym)
		try:
			if proc is None or proc.poll() is not None:
				_dynamic_trainer_procs.pop(sym, None)
		except Exception:
			_dynamic_trainer_procs.pop(sym, None)


def _start_dynamic_trainer(sym: str, settings: dict, settings_path: str) -> bool:
	_cleanup_dynamic_trainers()
	if sym in _dynamic_trainer_procs:
		return False
	if _is_coin_training(sym):
		return False
	try:
		max_trainers = max(1, int(float(settings.get("crypto_dynamic_max_trainers", 1) or 1)))
	except Exception:
		max_trainers = 1
	if len(_dynamic_trainer_procs) >= max_trainers:
		return False
	coin_dir = coin_folder(sym)
	os.makedirs(coin_dir, exist_ok=True)
	trainer_script = str(settings.get("script_neural_trainer", "engines/pt_trainer.py") or "engines/pt_trainer.py").strip()
	if not os.path.isabs(trainer_script):
		trainer_script = os.path.abspath(os.path.join(BASE_DIR, trainer_script))
	if not os.path.isfile(trainer_script):
		fallback = os.path.join(BASE_DIR, "engines", "pt_trainer.py")
		if os.path.isfile(fallback):
			trainer_script = fallback
		else:
			return False
	env = os.environ.copy()
	env["POWERTRADER_PROJECT_DIR"] = coin_dir
	env["POWERTRADER_HUB_DIR"] = HUB_DIR
	if settings_path:
		env["POWERTRADER_GUI_SETTINGS"] = settings_path
	prev_pp = str(env.get("PYTHONPATH", "") or "").strip()
	env["PYTHONPATH"] = BASE_DIR if not prev_pp else (BASE_DIR + os.pathsep + prev_pp)
	try:
		proc = subprocess.Popen(
			[sys.executable, "-u", trainer_script, sym],
			cwd=coin_dir,
			env=env,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)
		_dynamic_trainer_procs[sym] = proc
		return True
	except Exception:
		return False


def _dynamic_coin_manager() -> None:
	global _dynamic_last_scan_ts, _dynamic_last_rotation_ts
	settings, settings_path = _load_gui_settings_with_path()
	if not bool(settings.get("crypto_dynamic_enabled", True)):
		return
	now = time.time()
	try:
		interval_s = max(20.0, float(settings.get("crypto_dynamic_scan_interval_s", 300.0) or 300.0))
	except Exception:
		interval_s = 300.0
	if (now - _dynamic_last_scan_ts) < interval_s:
		return
	_dynamic_last_scan_ts = now
	_cleanup_dynamic_trainers()

	current_coins = _load_gui_coins()
	held = _read_held_coins()
	pool = _parse_symbol_list(settings.get("crypto_dynamic_pool_symbols", DEFAULT_DYNAMIC_POOL))
	if not pool:
		pool = list(DEFAULT_DYNAMIC_POOL)
	for c in current_coins:
		if c not in pool:
			pool.append(c)
	for c in held:
		if c not in pool:
			pool.append(c)
	try:
		target_count = int(float(settings.get("crypto_dynamic_target_count", len(current_coins) or 8) or 8))
	except Exception:
		target_count = len(current_coins) or 8
	target_count = max(1, min(24, target_count))
	try:
		min_edge = float(settings.get("crypto_dynamic_min_projected_edge_pct", 0.25) or 0.25)
	except Exception:
		min_edge = 0.25
	try:
		max_new_train = max(1, int(float(settings.get("crypto_dynamic_max_new_per_scan", 1) or 1)))
	except Exception:
		max_new_train = 1
	try:
		rotation_cooldown_s = max(30.0, float(settings.get("crypto_dynamic_rotation_cooldown_s", 900.0) or 900.0))
	except Exception:
		rotation_cooldown_s = 900.0
	try:
		news_weight = max(0.0, min(1.0, float(settings.get("crypto_news_event_weight", 0.12) or 0.12)))
	except Exception:
		news_weight = 0.12

	news_event_context = {
		"enabled": bool(settings.get("news_event_enabled", True)),
		"market": "crypto",
		"state": "disabled",
		"state_code": "disabled",
		"symbols": {},
		"errors": {},
	}
	try:
		news_event_context = build_unified_news_event_context(
			hub_dir=HUB_DIR,
			settings=settings,
			market="crypto",
			symbols=pool,
			now_ts=int(now),
		)
	except Exception as exc:
		news_event_context = {
			"enabled": bool(settings.get("news_event_enabled", True)),
			"market": "crypto",
			"state": "unavailable",
			"state_code": "provider_error",
			"symbols": {},
			"errors": {"provider": f"{type(exc).__name__}: {exc}"},
		}
	news_symbols = (
		dict(news_event_context.get("symbols", {}))
		if isinstance(news_event_context.get("symbols", {}), dict)
		else {}
	)

	ranked = []
	rejected = []
	for sym in pool:
		try:
			base_score = _score_coin_projection(sym)
			n_row = news_symbols.get(sym, {}) if isinstance(news_symbols.get(sym, {}), dict) else {}
			news_score = float(n_row.get("score", 0.0) or 0.0)
			news_conf = float(n_row.get("confidence", 0.0) or 0.0)
			news_impact = float(n_row.get("impact", 0.0) or 0.0)
			score = blend_score_with_news(
				base_score=base_score,
				news_score=news_score,
				confidence=news_conf,
				impact=news_impact,
				weight=news_weight,
			)
			ranked.append(
				{
					"symbol": sym,
					"score": round(score, 6),
					"score_base": round(base_score, 6),
					"trained": bool(_coin_is_trained(sym)),
					"news_score": round(news_score, 6),
					"news_confidence": round(news_conf, 4),
					"news_impact": round(news_impact, 4),
					"news_weight": round(news_weight, 4),
					"news_bias": str(n_row.get("bias", "neutral") or "neutral"),
					"news_headline_count": int(n_row.get("headline_count", 0) or 0),
					"news_event_risk": bool(n_row.get("event_risk", False)),
					"news_top_headline": str(n_row.get("top_headline", "") or ""),
				}
			)
		except Exception as exc:
			rejected.append({"symbol": sym, "reason": f"{type(exc).__name__}"})
	ranked.sort(key=lambda r: float(r.get("score", -9999.0) or -9999.0), reverse=True)
	abs_scores = [abs(float(r.get("score", 0.0) or 0.0)) for r in ranked if isinstance(r, dict)]
	abs_scores = [v for v in abs_scores if v > 0.0]
	vol_med = (sorted(abs_scores)[len(abs_scores) // 2] if abs_scores else 0.0)
	base_thr = max(
		0.01,
		float(
			settings.get(
				"crypto_dynamic_min_projected_edge_pct",
				settings.get("crypto_allocator_signal_floor", min_edge),
			)
			or min_edge
		),
	)
	volatility_threshold = float(round(base_thr * (1.15 if vol_med >= (base_thr * 1.6) else 1.0), 6))
	replay_enabled = bool(settings.get("crypto_replay_adaptive_enabled", True))
	replay_weight = _clamp(float(settings.get("crypto_replay_adaptive_weight", 0.35) or 0.35), 0.0, 1.0)
	replay_step_cap_pct = _clamp(float(settings.get("crypto_replay_adaptive_step_cap_pct", 40.0) or 40.0), 5.0, 90.0)
	replay_target_entries = replay_target_entries_for_market(settings, "crypto")
	replay_recommended = float(volatility_threshold)
	replay_clamped = float(volatility_threshold)
	replay_reason = ""
	if replay_enabled and ranked:
		replay_payload = recommend_threshold_from_scores(
			ranked,
			market="crypto",
			current_threshold=volatility_threshold,
			target_entries=replay_target_entries,
		)
		replay_rec = replay_payload.get("recommendation", {}) if isinstance(replay_payload.get("recommendation", {}), dict) else {}
		replay_recommended = max(0.01, float(replay_rec.get("recommended_threshold", volatility_threshold) or volatility_threshold))
		replay_reason = str(replay_rec.get("reason", "") or "")
		max_step = max(base_thr * 0.05, volatility_threshold * (replay_step_cap_pct / 100.0))
		replay_min = max(0.01, volatility_threshold - max_step)
		replay_max = volatility_threshold + max_step
		replay_clamped = min(replay_max, max(replay_min, replay_recommended))
	effective_weight = replay_weight if replay_enabled else 0.0
	adaptive_threshold = round(
		max(
			0.01,
			((1.0 - effective_weight) * volatility_threshold) + (effective_weight * replay_clamped),
		),
		4,
	)
	calibration_payload = build_market_confidence_calibration(
		HUB_DIR,
		"crypto",
		base_threshold=float(adaptive_threshold),
		min_samples=max(6, int(float(settings.get("adaptive_confidence_min_samples", 18) or 18))),
		target_success_pct=_clamp(float(settings.get("adaptive_confidence_target_success_pct", 55.0) or 55.0), 30.0, 90.0),
	)
	calibration_rec = calibration_payload.get("recommendation", {}) if isinstance(calibration_payload.get("recommendation", {}), dict) else {}
	calibration_recommended_threshold = float(calibration_rec.get("recommended_threshold", adaptive_threshold) or adaptive_threshold)
	calibration_curve = calibration_payload.get("curve", []) if isinstance(calibration_payload.get("curve", []), list) else []
	calibration_samples = int(float(calibration_payload.get("samples", 0) or 0))
	def _calib_prob_for_score(score_val: float) -> float:
		score_abs = abs(float(score_val or 0.0))
		for bin_row in calibration_curve:
			if not isinstance(bin_row, dict):
				continue
			lo = float(bin_row.get("min_score", 0.0) or 0.0)
			hi_raw = bin_row.get("max_score", None)
			hi = float(hi_raw) if hi_raw not in (None, "") else 999999.0
			if score_abs < lo:
				continue
			if score_abs >= hi:
				continue
			return _clamp(float(bin_row.get("success_rate_pct", 0.0) or 0.0) / 100.0, 0.0, 1.0)
		return 0.0
	for row in ranked:
		if not isinstance(row, dict):
			continue
		score_val = float(row.get("score", 0.0) or 0.0)
		row["side"] = "long" if score_val > 0.0 else "watch"
		row["eligible_for_entry"] = bool(score_val >= adaptive_threshold)
		row["adaptive_threshold"] = float(adaptive_threshold)
		row["required_score"] = float(adaptive_threshold)
		row["calib_prob"] = float(round(_calib_prob_for_score(score_val), 6))
		row["samples"] = int(calibration_samples)
		row["symbol_samples"] = int(calibration_samples)
		row["market_calibration_samples"] = int(calibration_samples)
		row["calibration_scope"] = "market_pooled"
		row["calibration_effective_samples"] = int(calibration_samples)
		row["calibration_effective_prob"] = float(row.get("calib_prob", 0.0) or 0.0)

	# Launch background training for promising untrained symbols.
	started_trainers = []
	if bool(settings.get("crypto_dynamic_auto_train", True)):
		for row in ranked:
			if len(started_trainers) >= max_new_train:
				break
			sym = str(row.get("symbol", "") or "").strip().upper()
			score = float(row.get("score", 0.0) or 0.0)
			if score < min_edge:
				continue
			if _coin_is_trained(sym) or _is_coin_training(sym):
				continue
			if sym in current_coins:
				continue
			if _start_dynamic_trainer(sym, settings, settings_path):
				started_trainers.append(sym)

	# Build target set: keep held symbols, then best trained by projection.
	target = []
	for c in current_coins:
		if c in held and c not in target:
			target.append(c)
	for row in ranked:
		sym = str(row.get("symbol", "") or "").strip().upper()
		if not sym or sym in target:
			continue
		if float(row.get("score", 0.0) or 0.0) < min_edge and sym not in current_coins:
			continue
		if not _coin_is_trained(sym):
			continue
		target.append(sym)
		if len(target) >= target_count:
			break
	for c in current_coins:
		if len(target) >= target_count:
			break
		if c not in target:
			target.append(c)
	target = target[:target_count]

	changed = False
	if target and target != current_coins and (now - _dynamic_last_rotation_ts) >= rotation_cooldown_s:
		# Never remove held symbols from active set.
		for h in held:
			if h not in target:
				target.insert(0, h)
		seen = set()
		dedup = []
		for c in target:
			cc = str(c or "").strip().upper()
			if cc and cc not in seen:
				seen.add(cc)
				dedup.append(cc)
		target = dedup[:target_count]
		changed = _write_settings_coins(settings_path, target)
		if changed:
			_dynamic_last_rotation_ts = now

	_append_jsonl(
		CRYPTO_SCANNER_RANKINGS_PATH,
		{
			"ts": int(now),
			"state": "READY",
			"universe_total": int(len(pool)),
			"candidates": int(len(ranked)),
			"rejected": rejected[:100],
			"top": ranked[:20],
			"adaptive_threshold": float(adaptive_threshold),
			"adaptive_threshold_base": float(base_thr),
			"adaptive_threshold_volatility": float(round(volatility_threshold, 6)),
			"adaptive_threshold_replay_recommended": float(round(replay_recommended, 6)),
			"adaptive_threshold_replay_clamped": float(round(replay_clamped, 6)),
			"adaptive_threshold_replay_weight": float(round(effective_weight, 4)),
			"adaptive_threshold_replay_target_entries": int(replay_target_entries),
			"adaptive_threshold_replay_reason": str(replay_reason),
			"adaptive_threshold_replay_enabled": bool(replay_enabled),
		},
	)

	_atomic_write_json(
		DYNAMIC_STATUS_PATH,
		{
			"ts": now,
			"enabled": True,
			"current_coins": current_coins,
			"target_coins": target,
				"changed": bool(changed),
				"held": sorted(list(held)),
				"ranked": ranked[:20],
				"rejected": rejected[:20],
				"adaptive_threshold": float(adaptive_threshold),
				"adaptive_threshold_base": float(base_thr),
				"adaptive_threshold_volatility": float(round(volatility_threshold, 6)),
				"adaptive_threshold_replay_recommended": float(round(replay_recommended, 6)),
				"adaptive_threshold_replay_clamped": float(round(replay_clamped, 6)),
				"adaptive_threshold_replay_weight": float(round(effective_weight, 4)),
				"adaptive_threshold_replay_target_entries": int(replay_target_entries),
				"adaptive_threshold_replay_reason": str(replay_reason),
				"adaptive_threshold_replay_enabled": bool(replay_enabled),
				"calibration": calibration_payload if isinstance(calibration_payload, dict) else {},
				"calibration_recommended_threshold": float(round(calibration_recommended_threshold, 6)),
				"started_trainers": started_trainers,
				"active_trainers": sorted(list(_dynamic_trainer_procs.keys())),
				"min_projected_edge_pct": min_edge,
				"news_event_context": news_event_context,
			},
		)


# Ensure folders exist for the current configured coins
for _sym in CURRENT_COINS:
	os.makedirs(coin_folder(_sym), exist_ok=True)


distance = 0.5
tf_choices = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']

def new_coin_state():
	return {
		'low_bound_prices': [.01] * len(tf_choices),
		'high_bound_prices': [99999999999999999] * len(tf_choices),

		'tf_times': [],
		'tf_choice_index': 0,

		'tf_update': ['yes'] * len(tf_choices),
		'messages': ['none'] * len(tf_choices),
		'last_messages': ['none'] * len(tf_choices),
		'margins': [0.25] * len(tf_choices),

		'high_tf_prices': [99999999999999999] * len(tf_choices),
		'low_tf_prices': [.01] * len(tf_choices),

		'tf_sides': ['none'] * len(tf_choices),
		'messaged': ['no'] * len(tf_choices),
		'updated': [0] * len(tf_choices),
		'perfects': ['active'] * len(tf_choices),
		'training_issues': [0] * len(tf_choices),

		# readiness gating (no placeholder-number checks; this is process-based)
		'bounds_version': 0,
		'last_display_bounds_version': -1,

	}

states = {}

display_cache = {sym: f"{sym}  (starting.)" for sym in CURRENT_COINS}

# Track which coins have produced REAL predicted levels (not placeholder 1 / 99999999999999999)
_ready_coins = set()

# We consider the runner "READY" only once it is ACTUALLY PRINTING real prediction messages
# (i.e. output lines start with WITHIN / LONG / SHORT). No numeric placeholder checks at all.
def _is_printing_real_predictions(messages) -> bool:
	try:
		for m in (messages or []):
			if not isinstance(m, str):
				continue
			# These are the only message types produced once predictions are being used in output.
			# (INACTIVE means it's still not printing real prediction output for that timeframe.)
			if m.startswith("WITHIN") or m.startswith("LONG") or m.startswith("SHORT"):
				return True
		return False
	except Exception:
		return False

def _sync_coins_from_settings():
	"""
	Hot-reload coins from gui_settings.json while runner is running.

	- Adds new coins: creates folder + init_coin() + starts stepping them
	- Removes coins: stops stepping them (leaves state on disk untouched)
	"""
	global CURRENT_COINS

	new_list = _load_gui_coins()
	if new_list == CURRENT_COINS:
		return

	old_list = list(CURRENT_COINS)
	added = [c for c in new_list if c not in old_list]
	removed = [c for c in old_list if c not in new_list]

	# Handle removed coins: stop stepping + clear UI cache entries
	for sym in removed:
		try:
			_ready_coins.discard(sym)
		except Exception:
			pass
		try:
			display_cache.pop(sym, None)
		except Exception:
			pass

	# Handle added coins: create folder + init state + show in UI output
	for sym in added:
		try:
			os.makedirs(coin_folder(sym), exist_ok=True)
		except Exception:
			pass
		try:
			display_cache[sym] = f"{sym}  (starting.)"
		except Exception:
			pass
		try:
			# init_coin switches CWD and does network calls, so do it carefully
			init_coin(sym)
			os.chdir(BASE_DIR)
		except Exception:
			try:
				os.chdir(BASE_DIR)
			except Exception:
				pass

	CURRENT_COINS = list(new_list)

_write_runner_ready(False, stage="starting", ready_coins=[], total_coins=len(CURRENT_COINS))





def init_coin(sym: str):
	# switch into the coin's folder so ALL existing relative file I/O stays working
	os.chdir(coin_folder(sym))

	# per-coin "version" + on/off files (no collisions between coins)
	with open('alerts_version.txt', 'w+') as f:
		f.write('5/3/2022/9am')

	with open('futures_long_onoff.txt', 'w+') as f:
		f.write('OFF')

	with open('futures_short_onoff.txt', 'w+') as f:
		f.write('OFF')

	st = new_coin_state()

	coin = sym + '-USDT'
	ind = 0
	tf_times_local = []
	while True:
		history_list = []
		while True:
			try:
				history = str(_get_kline_shaped(coin, tf_choices[ind])).replace(']]', '], ').replace('[[', '[')
				break
			except Exception as e:
				time.sleep(3.5)
				if 'Requests' in str(e):
					pass
				else:
					PrintException()
				continue

		history_list = history.split("], [")
		ind += 1
		try:
			working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
			the_time = working_minute[0].replace('[', '')
		except Exception:
			the_time = 0.0

		tf_times_local.append(the_time)
		if len(tf_times_local) >= len(tf_choices):
			break

	st['tf_times'] = tf_times_local
	states[sym] = st

# init all coins once (from GUI settings)
for _sym in CURRENT_COINS:
	init_coin(_sym)

# restore CWD to base after init
os.chdir(BASE_DIR)


wallet_addr_list = []
wallet_addr_users = []
total_long = 0
total_short = 0
last_hour = 565457457357

cc_index = 0
tf_choice = []
prices = []
starts = []
long_start_prices = []
short_start_prices = []
buy_coins = []
cc_update = 'yes'
wr_update = 'yes'

def find_purple_area(lines):
    """
    Given a list of (price, color) pairs (color is 'orange' or 'blue'),
    returns (purple_bottom, purple_top) if a purple area exists,
    else (None, None).
    """
    oranges = sorted([price for price, color in lines if color == 'orange'], reverse=True)
    blues   = sorted([price for price, color in lines if color == 'blue'])
    if not oranges or not blues:
        return (None, None)
    purple_bottom = None
    purple_top = None
    all_levels = sorted(set(oranges + blues + [float('-inf'), float('inf')]), reverse=True)
    for i in range(len(all_levels) - 1):
        top = all_levels[i]
        bottom = all_levels[i+1]
        oranges_below = [o for o in oranges if o < bottom]
        blues_above = [b for b in blues if b > top]
        has_orange_below = any(o < top for o in oranges)
        has_blue_above = any(b > bottom for b in blues)
        if has_orange_below and has_blue_above:
            if purple_bottom is None or bottom < purple_bottom:
                purple_bottom = bottom
            if purple_top is None or top > purple_top:
                purple_top = top
    if purple_bottom is not None and purple_top is not None and purple_top > purple_bottom:
        return (purple_bottom, purple_top)
    return (None, None)
def step_coin(sym: str):
	# run inside the coin folder so all existing file reads/writes stay relative + isolated
	os.chdir(coin_folder(sym))
	coin = sym + '-USDT'
	st = states[sym]

	# --- training freshness gate ---
	# If GUI would show NOT TRAINED (missing / stale trainer_last_training_time.txt),
	# skip this coin so no new trades can start until it is trained again.
	if not _coin_is_trained(sym):
		try:
			# Prevent new trades (and DCA) by forcing signals to 0 and keeping PM at baseline.
			with open('futures_long_profit_margin.txt', 'w+') as f:
				f.write('0.25')
			with open('futures_short_profit_margin.txt', 'w+') as f:
				f.write('0.25')
			with open('long_dca_signal.txt', 'w+') as f:
				f.write('0')
			with open('short_dca_signal.txt', 'w+') as f:
				f.write('0')
		except Exception:
			pass
		try:
			display_cache[sym] = sym + "  (NOT TRAINED / OUTDATED - run trainer)"
		except Exception:
			pass
		try:
			_ready_coins.discard(sym)
			all_ready = len(_ready_coins) >= len(CURRENT_COINS)
			_write_runner_ready(
				all_ready,
				stage=("real_predictions" if all_ready else "training_required"),
				ready_coins=sorted(list(_ready_coins)),
				total_coins=len(CURRENT_COINS),
			)

		except Exception:
			pass
		return


	# ensure new readiness-version keys exist even if restarting from an older state dict
	if 'bounds_version' not in st:
		st['bounds_version'] = 0
	if 'last_display_bounds_version' not in st:
		st['last_display_bounds_version'] = -1

	# pull state into local names (lists mutate in-place; ones that get reassigned we set back at end)
	low_bound_prices = st['low_bound_prices']
	high_bound_prices = st['high_bound_prices']
	tf_times = st['tf_times']
	tf_choice_index = st['tf_choice_index']

	tf_update = st['tf_update']
	messages = st['messages']
	last_messages = st['last_messages']
	margins = st['margins']

	high_tf_prices = st['high_tf_prices']
	low_tf_prices = st['low_tf_prices']
	tf_sides = st['tf_sides']
	messaged = st['messaged']
	updated = st['updated']
	perfects = st['perfects']
	training_issues = st.get('training_issues', [0] * len(tf_choices))
	# keep training_issues aligned to tf_choices
	if len(training_issues) < len(tf_choices):
		training_issues.extend([0] * (len(tf_choices) - len(training_issues)))
	elif len(training_issues) > len(tf_choices):
		del training_issues[len(tf_choices):]

	last_difference_between = 0.0


	# ====== ORIGINAL: fetch current candle for this timeframe index ======
	while True:
		history_list = []
		while True:
			try:
				history = str(_get_kline_shaped(coin, tf_choices[tf_choice_index])).replace(']]', '], ').replace('[[', '[')
				break
			except Exception as e:
				time.sleep(3.5)
				if 'Requests' in str(e):
					pass
				else:
					pass
				continue
		history_list = history.split("], [")
		# KuCoin can occasionally return an empty/short kline response.
		# Guard against history_list[1] raising IndexError.
		if len(history_list) < 2:
			time.sleep(0.2)
			continue
		working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
		try:
			openPrice = float(working_minute[1])
			closePrice = float(working_minute[2])
			break
		except Exception:
			continue


	current_candle = 100 * ((closePrice - openPrice) / openPrice)

	# ====== ORIGINAL: load threshold + memories/weights and compute moves ======
	file = open('neural_perfect_threshold_' + tf_choices[tf_choice_index] + '.txt', 'r')
	perfect_threshold = float(file.read())
	file.close()

	try:
		# If we can read/parse training files, this timeframe is NOT a training-file issue.
		training_issues[tf_choice_index] = 0

		file = open('memories_' + tf_choices[tf_choice_index] + '.txt', 'r')
		memory_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split('~')
		file.close()

		file = open('memory_weights_' + tf_choices[tf_choice_index] + '.txt', 'r')
		weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')
		file.close()

		file = open('memory_weights_high_' + tf_choices[tf_choice_index] + '.txt', 'r')
		high_weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')
		file.close()

		file = open('memory_weights_low_' + tf_choices[tf_choice_index] + '.txt', 'r')
		low_weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')
		file.close()

		mem_ind = 0
		candidates = []
		while mem_ind < len(memory_list):
			try:
				row = str(memory_list[mem_ind] or "")
				parts = row.split('{}')
				if len(parts) < 3:
					mem_ind += 1
					continue
				memory_pattern = parts[0].replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')
				memory_candle = float(memory_pattern[0])
				move_pct = float(memory_pattern[len(memory_pattern) - 1])
				high_diff = float(parts[1].replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').replace(' ', '')) / 100
				low_diff = float(parts[2].replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').replace(' ', '')) / 100
				w_move = abs(float(weight_list[mem_ind])) if mem_ind < len(weight_list) else 0.0
				w_high = abs(float(high_weight_list[mem_ind])) if mem_ind < len(high_weight_list) else 0.0
				w_low = abs(float(low_weight_list[mem_ind])) if mem_ind < len(low_weight_list) else 0.0
			except Exception:
				mem_ind += 1
				continue

			if current_candle == 0.0 and memory_candle == 0.0:
				diff_avg = 0.0
			else:
				try:
					diff_avg = abs((abs(current_candle - memory_candle) / ((current_candle + memory_candle) / 2)) * 100)
				except Exception:
					diff_avg = 0.0

			candidates.append({
				"diff": float(diff_avg),
				"move_pct": float(move_pct),
				"high_diff": float(high_diff),
				"low_diff": float(low_diff),
				"w_move": float(w_move),
				"w_high": float(w_high),
				"w_low": float(w_low),
			})
			mem_ind += 1

		final_moves = 0.0
		high_final_moves = 0.0
		low_final_moves = 0.0
		active_model = False
		if candidates:
			perfect = [c for c in candidates if c["diff"] <= float(perfect_threshold)]
			perfect.sort(key=lambda c: c["diff"])
			sorted_all = sorted(candidates, key=lambda c: c["diff"])

			# Use many nearest perfect matches when available; otherwise conservative nearest-neighbor fallback.
			min_perfect_count = 3
			top_k_perfect = 24
			top_k_fallback = 12
			shrink = 1.0
			selected = perfect[:top_k_perfect]
			if len(selected) >= min_perfect_count:
				active_model = True
			else:
				selected = sorted_all[:top_k_fallback]
				avg_diff = (sum(c["diff"] for c in selected) / len(selected)) if selected else 999.0
				# Keep fallback conservative in live trading: only activate if nearest-neighbor cluster is tight enough.
				active_model = (len(selected) >= 6) and (avg_diff <= max(1.0, float(perfect_threshold) * 1.5))
				shrink = 0.35

			def _weighted_mean(vals, ws):
				num = 0.0
				den = 0.0
				for v, w in zip(vals, ws):
					try:
						ww = float(w)
						if ww <= 0:
							continue
						num += float(v) * ww
						den += ww
					except Exception:
						continue
				return (num / den) if den > 0 else 0.0

			if selected:
				sim = [1.0 / (1.0 + max(0.0, float(c["diff"]))) for c in selected]
				move_ws = [max(0.001, float(c["w_move"])) * s for c, s in zip(selected, sim)]
				high_ws = [max(0.001, float(c["w_high"])) * s for c, s in zip(selected, sim)]
				low_ws = [max(0.001, float(c["w_low"])) * s for c, s in zip(selected, sim)]
				final_moves = _weighted_mean([c["move_pct"] for c in selected], move_ws) * shrink
				high_final_moves = _weighted_mean([c["high_diff"] for c in selected], high_ws) * shrink
				low_final_moves = _weighted_mean([c["low_diff"] for c in selected], low_ws) * shrink

		# Live safety clamp: avoid extreme single-tick forecast moves causing unstable trading.
		max_move_pct = 8.0
		max_move_frac = max_move_pct / 100.0
		final_moves = max(-max_move_pct, min(max_move_pct, float(final_moves)))
		high_final_moves = max(-max_move_frac, min(max_move_frac, float(high_final_moves)))
		low_final_moves = max(-max_move_frac, min(max_move_frac, float(low_final_moves)))

		del perfects[tf_choice_index]
		perfects.insert(tf_choice_index, ('active' if active_model else 'inactive'))

	except Exception:
		PrintException()
		training_issues[tf_choice_index] = 1
		final_moves = 0.0
		high_final_moves = 0.0
		low_final_moves = 0.0
		del perfects[tf_choice_index]
		perfects.insert(tf_choice_index, 'inactive')

	# keep threshold persisted (original behavior)
	file = open('neural_perfect_threshold_' + tf_choices[tf_choice_index] + '.txt', 'w+')
	file.write(str(perfect_threshold))
	file.close()

	# ====== ORIGINAL: compute new high/low predictions ======
	price_list2 = [openPrice, closePrice]
	current_pattern = [price_list2[0], price_list2[1]]

	try:
		c_diff = final_moves / 100
		high_diff = high_final_moves
		low_diff = low_final_moves

		start_price = current_pattern[len(current_pattern) - 1]
		high_new_price = start_price + (start_price * high_diff)
		low_new_price = start_price + (start_price * low_diff)
	except:
		start_price = current_pattern[len(current_pattern) - 1]
		high_new_price = start_price
		low_new_price = start_price

	if perfects[tf_choice_index] == 'inactive':
		del high_tf_prices[tf_choice_index]
		high_tf_prices.insert(tf_choice_index, start_price)
		del low_tf_prices[tf_choice_index]
		low_tf_prices.insert(tf_choice_index, start_price)
	else:
		del high_tf_prices[tf_choice_index]
		high_tf_prices.insert(tf_choice_index, high_new_price)
		del low_tf_prices[tf_choice_index]
		low_tf_prices.insert(tf_choice_index, low_new_price)

	# ====== advance tf index; if full sweep complete, compute signals ======
	tf_choice_index += 1

	if tf_choice_index >= len(tf_choices):
		tf_choice_index = 0

		# reset tf_update for this coin (but DO NOT block-wait; just detect updates and return)
		tf_update = ['no'] * len(tf_choices)

		# get current price ONCE per coin — prefer Robinhood for supported symbols,
		# but fall back directly to KuCoin after a confirmed invalid-symbol rejection.
		rh_symbol = f"{sym}-USD"
		current = None
		max_price_attempts = 25
		_, _, _, _, price_log_cd_s = _kucoin_tune_values()
		if not _robinhood_symbol_is_temporarily_unsupported(rh_symbol):
			for _attempt in range(1, max_price_attempts + 1):
				try:
					current = robinhood_current_ask(rh_symbol)
					break
				except Exception as e:
					if _is_robinhood_invalid_symbol_exception(e):
						_mark_robinhood_symbol_unsupported(rh_symbol)
						log_throttled(
							f"pt_thinker:price_feed_unsupported:{sym}",
							f"[pt_thinker] {sym} is not supported by Robinhood market-data; using KuCoin fallback",
							cooldown_s=float(max(60.0, min(_RH_UNSUPPORTED_SYMBOL_COOLDOWN_S, 7200.0))),
						)
						break
					log_throttled(
						f"pt_thinker:price_feed:{sym}:{type(e).__name__}",
						f"[pt_thinker] {sym} price feed error: {e}",
						cooldown_s=float(price_log_cd_s),
					)
					if not _is_retryable_price_exception(e):
						break
					time.sleep(min(5.0, 0.2 * _attempt))
		if current is None:
			try:
				kucoin_live = _kucoin_live_current_price(sym)
			except Exception as e:
				kucoin_live = None
				log_throttled(
					f"pt_thinker:price_feed_kucoin:{sym}:{type(e).__name__}",
					f"[pt_thinker] {sym} kucoin live-price fallback error: {e}",
					cooldown_s=float(price_log_cd_s),
				)
			if kucoin_live is not None:
				current = float(kucoin_live)
				log_throttled(
					f"pt_thinker:price_feed_kucoin_ok:{sym}",
					f"[pt_thinker] {sym} using KuCoin live close as current-price fallback",
					cooldown_s=float(price_log_cd_s),
				)
		if current is None:
			cached = _cached_current_price(sym, max_age_s=900.0)
			if cached is not None:
				current = float(cached)
				log_throttled(
					f"pt_thinker:price_feed_cached:{sym}",
					f"[pt_thinker] {sym} using cached current price while live quote feed is degraded",
					cooldown_s=float(price_log_cd_s),
				)
			else:
				try:
					display_cache[sym] = sym + "  (price feed unavailable)"
				except Exception:
					pass
				try:
					_ready_coins.discard(sym)
					all_ready = len(_ready_coins) >= len(CURRENT_COINS)
					_write_runner_ready(
						all_ready,
						stage=("real_predictions" if all_ready else "price_feed_error"),
						ready_coins=sorted(list(_ready_coins)),
						total_coins=len(CURRENT_COINS),
					)
				except Exception:
					pass
				return

		# IMPORTANT: messages printed below use the bounds currently in state.
		# We only allow "ready" once messages are generated using a non-startup bounds_version.
		bounds_version_used_for_messages = st.get('bounds_version', 0)

		# --- HARD GUARANTEE: all TF arrays stay length==len(tf_choices) (fallback placeholders) ---
		def _pad_to_len(lst, n, fill):
			if lst is None:
				lst = []
			if len(lst) < n:
				lst.extend([fill] * (n - len(lst)))
			elif len(lst) > n:
				del lst[n:]
			return lst

		n_tfs = len(tf_choices)

		# bounds: use your fake numbers when TF inactive / missing
		low_bound_prices = _pad_to_len(low_bound_prices, n_tfs, .01)
		high_bound_prices = _pad_to_len(high_bound_prices, n_tfs, 99999999999999999)

		# predicted prices: keep equal when missing so it never triggers LONG/SHORT
		high_tf_prices = _pad_to_len(high_tf_prices, n_tfs, current)
		low_tf_prices = _pad_to_len(low_tf_prices, n_tfs, current)

		# status arrays
		perfects = _pad_to_len(perfects, n_tfs, 'inactive')
		training_issues = _pad_to_len(training_issues, n_tfs, 0)
		messages = _pad_to_len(messages, n_tfs, 'none')

		tf_sides = _pad_to_len(tf_sides, n_tfs, 'none')
		messaged = _pad_to_len(messaged, n_tfs, 'no')
		margins = _pad_to_len(margins, n_tfs, 0.0)
		updated = _pad_to_len(updated, n_tfs, 0)

		# per-timeframe message logic (same decisions as before)
		inder = 0
		while inder < len(tf_choices):
			# update the_time snapshot (same as before)
			while True:

				try:
					history = str(_get_kline_shaped(coin, tf_choices[inder])).replace(']]', '], ').replace('[[', '[')
					break
				except Exception as e:
					time.sleep(3.5)
					if 'Requests' in str(e):
						pass
					else:
						PrintException()
					continue

			history_list = history.split("], [")
			try:
				working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
				the_time = working_minute[0].replace('[', '')
			except Exception:
				the_time = 0.0

			# (original comparisons)
			if current > high_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
				message = 'SHORT on ' + tf_choices[inder] + ' timeframe. ' + format(((high_bound_prices[inder] - current) / abs(current)) * 100, '.8f') + ' High Boundary: ' + str(high_bound_prices[inder])
				if messaged[inder] != 'yes':
					del messaged[inder]
					messaged.insert(inder, 'yes')
				del margins[inder]
				margins.insert(inder, ((high_tf_prices[inder] - current) / abs(current)) * 100)

				if 'SHORT' in messages[inder]:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 0)
				else:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 1)

				del tf_sides[inder]
				tf_sides.insert(inder, 'short')

			elif current < low_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
				message = 'LONG on ' + tf_choices[inder] + ' timeframe. ' + format(((low_bound_prices[inder] - current) / abs(current)) * 100, '.8f') + ' Low Boundary: ' + str(low_bound_prices[inder])
				if messaged[inder] != 'yes':
					del messaged[inder]
					messaged.insert(inder, 'yes')

				del margins[inder]
				margins.insert(inder, ((low_tf_prices[inder] - current) / abs(current)) * 100)

				del tf_sides[inder]
				tf_sides.insert(inder, 'long')

				if 'LONG' in messages[inder]:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 0)
				else:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 1)

			else:
				if perfects[inder] == 'inactive':
					if training_issues[inder] == 1:
						message = 'INACTIVE (training data issue) on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])
					else:
						message = 'INACTIVE on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])
				else:
					message = 'WITHIN on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])

				del margins[inder]
				margins.insert(inder, 0.0)

				if message == messages[inder]:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 0)
				else:
					del messages[inder]
					messages.insert(inder, message)
					del updated[inder]
					updated.insert(inder, 1)

				del tf_sides[inder]
				tf_sides.insert(inder, 'none')

				del messaged[inder]
				messaged.insert(inder, 'no')

			inder += 1


		# rebuild bounds (same math as before)
		prices_index = 0
		low_bound_prices = []
		high_bound_prices = []
		while True:
			new_low_price = low_tf_prices[prices_index] - (low_tf_prices[prices_index] * (distance / 100))
			new_high_price = high_tf_prices[prices_index] + (high_tf_prices[prices_index] * (distance / 100))
			if perfects[prices_index] != 'inactive':
				low_bound_prices.append(new_low_price)
				high_bound_prices.append(new_high_price)
			else:
				low_bound_prices.append(.01)
				high_bound_prices.append(99999999999999999)

			prices_index += 1
			if prices_index >= len(high_tf_prices):
				break

		new_low_bound_prices = sorted(low_bound_prices)
		new_low_bound_prices.reverse()
		new_high_bound_prices = sorted(high_bound_prices)

		og_index = 0
		og_low_index_list = []
		og_high_index_list = []
		while True:
			og_low_index_list.append(low_bound_prices.index(new_low_bound_prices[og_index]))
			og_high_index_list.append(high_bound_prices.index(new_high_bound_prices[og_index]))
			og_index += 1
			if og_index >= len(low_bound_prices):
				break

		og_index = 0
		gap_modifier = 0.0
		while True:
			if new_low_bound_prices[og_index] == .01 or new_low_bound_prices[og_index + 1] == .01 or new_high_bound_prices[og_index] == 99999999999999999 or new_high_bound_prices[og_index + 1] == 99999999999999999:
				pass
			else:
				try:
					low_perc_diff = (abs(new_low_bound_prices[og_index] - new_low_bound_prices[og_index + 1]) / ((new_low_bound_prices[og_index] + new_low_bound_prices[og_index + 1]) / 2)) * 100
				except:
					low_perc_diff = 0.0
				try:
					high_perc_diff = (abs(new_high_bound_prices[og_index] - new_high_bound_prices[og_index + 1]) / ((new_high_bound_prices[og_index] + new_high_bound_prices[og_index + 1]) / 2)) * 100
				except:
					high_perc_diff = 0.0

				if low_perc_diff < 0.25 + gap_modifier or new_low_bound_prices[og_index + 1] > new_low_bound_prices[og_index]:
					new_price = new_low_bound_prices[og_index + 1] - (new_low_bound_prices[og_index + 1] * 0.0005)
					del new_low_bound_prices[og_index + 1]
					new_low_bound_prices.insert(og_index + 1, new_price)
					continue

				if high_perc_diff < 0.25 + gap_modifier or new_high_bound_prices[og_index + 1] < new_high_bound_prices[og_index]:
					new_price = new_high_bound_prices[og_index + 1] + (new_high_bound_prices[og_index + 1] * 0.0005)
					del new_high_bound_prices[og_index + 1]
					new_high_bound_prices.insert(og_index + 1, new_price)
					continue

			og_index += 1
			gap_modifier += 0.25
			if og_index >= len(new_low_bound_prices) - 1:
				break

		og_index = 0
		low_bound_prices = []
		high_bound_prices = []
		while True:
			try:
				low_bound_prices.append(new_low_bound_prices[og_low_index_list.index(og_index)])
			except:
				pass
			try:
				high_bound_prices.append(new_high_bound_prices[og_high_index_list.index(og_index)])
			except:
				pass
			og_index += 1
			if og_index >= len(new_low_bound_prices):
				break

		# bump bounds_version now that we've computed a new set of prediction bounds
		st['bounds_version'] = bounds_version_used_for_messages + 1

		with open('low_bound_prices.html', 'w+') as file:
			file.write(str(new_low_bound_prices).replace("', '", " ").replace("[", "").replace("]", "").replace("'", ""))
		with open('high_bound_prices.html', 'w+') as file:
			file.write(str(new_high_bound_prices).replace("', '", " ").replace("[", "").replace("]", "").replace("'", ""))

		# cache display text for this coin (main loop prints everything on one screen)
		try:
			display_cache[sym] = (
				sym + '  ' + str(current) + '\n\n' +
				str(messages).replace("', '", "\n")
			)

			# The GUI-visible messages were generated using the bounds_version that was in state at the
			# start of this full-sweep (before we rebuilt bounds above).
			st['last_display_bounds_version'] = bounds_version_used_for_messages

			# Only consider this coin "ready" once we've already rebuilt bounds at least once
			# AND we're now printing messages generated from those rebuilt bounds.
			if (st['last_display_bounds_version'] >= 1) and _is_printing_real_predictions(messages):
				_ready_coins.add(sym)
			else:
				_ready_coins.discard(sym)



			all_ready = len(_ready_coins) >= len(COIN_SYMBOLS)
			_write_runner_ready(
				all_ready,
				stage=("real_predictions" if all_ready else "warming_up"),
				ready_coins=sorted(list(_ready_coins)),
				total_coins=len(COIN_SYMBOLS),
			)

		except:
			PrintException()




		# write PM + DCA signals (same as before)
		try:
			longs = tf_sides.count('long')
			shorts = tf_sides.count('short')

			# long pm
			current_pms = [m for m in margins if m != 0]
			try:
				pm = sum(current_pms) / len(current_pms)
				if pm < 0.25:
					pm = 0.25
			except:
				pm = 0.25

			with open('futures_long_profit_margin.txt', 'w+') as f:
				f.write(str(pm))
			with open('long_dca_signal.txt', 'w+') as f:
				f.write(str(longs))

			# short pm
			current_pms = [m for m in margins if m != 0]
			try:
				pm = sum(current_pms) / len(current_pms)
				if pm < 0.25:
					pm = 0.25
			except:
				pm = 0.25

			with open('futures_short_profit_margin.txt', 'w+') as f:
				f.write(str(abs(pm)))
			with open('short_dca_signal.txt', 'w+') as f:
				f.write(str(shorts))

		except:
			PrintException()

		# ====== NON-BLOCKING candle update check (single pass) ======
		this_index_now = 0
		while this_index_now < len(tf_update):
			while True:
				try:
					history = str(_get_kline_shaped(coin, tf_choices[this_index_now])).replace(']]', '], ').replace('[[', '[')
					break
				except Exception as e:
					time.sleep(3.5)
					if 'Requests' in str(e):
						pass
					else:
						PrintException()
					continue

			history_list = history.split("], [")
			try:
				working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
				the_time = working_minute[0].replace('[', '')
			except Exception:
				the_time = 0.0

			if the_time != tf_times[this_index_now]:
				del tf_update[this_index_now]
				tf_update.insert(this_index_now, 'yes')
				del tf_times[this_index_now]
				tf_times.insert(this_index_now, the_time)

			this_index_now += 1

	# ====== save state back ======
	st['low_bound_prices'] = low_bound_prices
	st['high_bound_prices'] = high_bound_prices
	st['tf_times'] = tf_times
	st['tf_choice_index'] = tf_choice_index

	# persist readiness gating fields
	st['bounds_version'] = st.get('bounds_version', 0)
	st['last_display_bounds_version'] = st.get('last_display_bounds_version', -1)

	st['tf_update'] = tf_update
	st['messages'] = messages
	st['last_messages'] = last_messages
	st['margins'] = margins

	st['high_tf_prices'] = high_tf_prices
	st['low_tf_prices'] = low_tf_prices
	st['tf_sides'] = tf_sides
	st['messaged'] = messaged
	st['updated'] = updated
	st['perfects'] = perfects
	st['training_issues'] = training_issues

	states[sym] = st




try:
	while True:
		# Optional: dynamic coin rotation + background trainer orchestration.
		_dynamic_coin_manager()

		# Hot-reload coins from GUI settings while running
		_sync_coins_from_settings()

		for _sym in CURRENT_COINS:
			step_coin(_sym)

		# clear + re-print one combined screen (so you don't see old output above new)
		os.system('cls' if os.name == 'nt' else 'clear')

		for _sym in CURRENT_COINS:
			print(display_cache.get(_sym, _sym + "  (no data yet)"))
			print("\n" + ("-" * 60) + "\n")

		# small sleep so you don't peg CPU when running many coins
		time.sleep(0.40)

except Exception:
	PrintException()


