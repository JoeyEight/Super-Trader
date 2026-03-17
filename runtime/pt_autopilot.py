from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.path_utils import read_settings_file, resolve_runtime_paths, resolve_settings_path
from app.runtime_logging import atomic_write_json, runtime_event
from app.settings_utils import sanitize_settings

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "pt_autopilot")
STOP_FLAG_PATH = os.path.join(HUB_DATA_DIR, "stop_trading.flag")
LOG_DIR = os.path.join(HUB_DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
AUTOPILOT_LOG_PATH = os.path.join(LOG_DIR, "autopilot.log")

AUTOPILOT_STATUS_PATH = os.path.join(HUB_DATA_DIR, "autopilot_status.json")
AUTOPILOT_STATE_PATH = os.path.join(HUB_DATA_DIR, "autopilot_state.json")
ISSUES_PATH = os.path.join(HUB_DATA_DIR, "user_action_required.json")
RUNTIME_EVENTS_PATH = os.path.join(HUB_DATA_DIR, "runtime_events.jsonl")

THINKER_LOG_PATH = os.path.join(LOG_DIR, "thinker.log")


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    try:
        with open(AUTOPILOT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    runtime_event(
        RUNTIME_EVENTS_PATH,
        component="autopilot",
        event="log",
        level="info",
        msg=str(msg or ""),
    )


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _load_settings_with_path() -> Tuple[Dict[str, Any], str]:
    settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
    data = read_settings_file(settings_path, module_name="pt_autopilot") or {}
    return sanitize_settings(data if isinstance(data, dict) else {}), str(settings_path)


def _save_settings(path: str, data: Dict[str, Any]) -> bool:
    try:
        _atomic_write_json(path, sanitize_settings(data if isinstance(data, dict) else {}))
        return True
    except Exception:
        return False


def _tail_error_counts(path: str, offsets: Dict[str, int]) -> Tuple[int, int]:
    if path not in offsets:
        try:
            offsets[path] = int(os.path.getsize(path))
        except Exception:
            offsets[path] = 0
        return 0, 0
    off = int(offsets.get(path, 0) or 0)
    kucoin_err = 0
    rate_err = 0
    try:
        size = os.path.getsize(path)
        if off < 0 or off > size:
            off = max(0, size - 250000)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(off)
            chunk = f.read()
            offsets[path] = f.tell()
        if chunk:
            kucoin_err += chunk.count("Connection reset by peer")
            kucoin_err += chunk.count("Failed to resolve 'api.kucoin.com'")
            kucoin_err += chunk.count("Max retries exceeded")
            rate_err += chunk.count("Too many requests")
            rate_err += chunk.count("429000")
    except Exception:
        pass
    return kucoin_err, rate_err


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _set_if_changed(settings: Dict[str, Any], key: str, value: Any, changes: Dict[str, Any]) -> None:
    cur = settings.get(key)
    if cur == value:
        return
    settings[key] = value
    changes[key] = value


def run_once(dry_run: bool = False) -> Dict[str, Any]:
    now = int(time.time())
    state = _safe_read_json(AUTOPILOT_STATE_PATH)
    offsets = state.get("offsets", {}) if isinstance(state.get("offsets", {}), dict) else {}
    stable_cycles = int(state.get("stable_cycles", 0) or 0)

    settings, settings_path = _load_settings_with_path()

    # health inputs
    stock_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "stock_trader_status.json"))
    forex_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "forex", "forex_trader_status.json"))
    stock_health = (stock_trader.get("health", {}) if isinstance(stock_trader, dict) else {}) or {}
    forex_health = (forex_trader.get("health", {}) if isinstance(forex_trader, dict) else {}) or {}

    kucoin_err, rate_err = _tail_error_counts(THINKER_LOG_PATH, offsets)
    api_unstable = (kucoin_err + rate_err) >= 4
    markets_healthy = bool(stock_health.get("data_ok", True)) and bool(forex_health.get("data_ok", True))
    markets_healthy = markets_healthy and (not bool(stock_health.get("drift_warning", False))) and (not bool(forex_health.get("drift_warning", False)))

    changes: Dict[str, Any] = {}
    notes = []

    # Crypto API pressure auto-tuning.
    min_interval = _to_float(settings.get("kucoin_min_interval_sec", 0.40), 0.40)
    cache_ttl = _to_float(settings.get("kucoin_cache_ttl_sec", 2.5), 2.5)
    trader_loop = _to_float(settings.get("crypto_trader_loop_sleep_s", 1.0), 1.0)
    trader_err_sleep = _to_float(settings.get("crypto_trader_error_sleep_s", 1.5), 1.5)

    if api_unstable:
        min_interval = _clamp(min_interval + 0.10, 0.35, 2.00)
        cache_ttl = _clamp(cache_ttl + 0.5, 1.5, 8.0)
        trader_loop = _clamp(trader_loop + 0.10, 0.5, 3.0)
        trader_err_sleep = _clamp(trader_err_sleep + 0.25, 1.0, 6.0)
        stable_cycles = 0
        notes.append("Detected API instability; reduced request aggressiveness.")
    else:
        stable_cycles += 1
        if stable_cycles >= 10:
            min_interval = _clamp(min_interval - 0.05, 0.35, 2.00)
            cache_ttl = _clamp(cache_ttl - 0.25, 1.5, 8.0)
            trader_loop = _clamp(trader_loop - 0.05, 0.5, 3.0)
            trader_err_sleep = _clamp(trader_err_sleep - 0.10, 1.0, 6.0)
            stable_cycles = 7
            notes.append("Stable APIs; cautiously improved responsiveness.")

    _set_if_changed(settings, "kucoin_min_interval_sec", round(min_interval, 3), changes)
    _set_if_changed(settings, "kucoin_cache_ttl_sec", round(cache_ttl, 3), changes)
    _set_if_changed(settings, "crypto_trader_loop_sleep_s", round(trader_loop, 3), changes)
    _set_if_changed(settings, "crypto_trader_error_sleep_s", round(trader_err_sleep, 3), changes)

    # Stocks/forex autonomous cadence tuning.
    s_every = _to_float(settings.get("market_bg_stocks_interval_s", 18.0), 18.0)
    f_every = _to_float(settings.get("market_bg_forex_interval_s", 12.0), 12.0)
    stock_size_step = _to_float(settings.get("stock_loss_streak_size_step_pct", 0.15), 0.15)
    stock_size_floor = _to_float(settings.get("stock_loss_streak_size_floor_pct", 0.40), 0.40)
    forex_size_step = _to_float(settings.get("forex_loss_streak_size_step_pct", 0.15), 0.15)
    forex_size_floor = _to_float(settings.get("forex_loss_streak_size_floor_pct", 0.40), 0.40)
    if markets_healthy and (not api_unstable):
        s_every = _clamp(s_every - 1.0, 12.0, 60.0)
        f_every = _clamp(f_every - 1.0, 8.0, 60.0)
        stock_size_step = _clamp(stock_size_step + 0.01, 0.05, 0.35)
        stock_size_floor = _clamp(stock_size_floor + 0.02, 0.35, 0.80)
        forex_size_step = _clamp(forex_size_step + 0.01, 0.05, 0.35)
        forex_size_floor = _clamp(forex_size_floor + 0.02, 0.35, 0.80)
    else:
        s_every = _clamp(s_every + 1.0, 12.0, 60.0)
        f_every = _clamp(f_every + 1.0, 8.0, 60.0)
        # Under degraded market/broker health, reduce new-entry aggression automatically.
        stock_size_step = _clamp(stock_size_step - 0.01, 0.05, 0.35)
        stock_size_floor = _clamp(stock_size_floor - 0.02, 0.20, 0.80)
        forex_size_step = _clamp(forex_size_step - 0.01, 0.05, 0.35)
        forex_size_floor = _clamp(forex_size_floor - 0.02, 0.20, 0.80)
    _set_if_changed(settings, "market_bg_stocks_interval_s", round(s_every, 2), changes)
    _set_if_changed(settings, "market_bg_forex_interval_s", round(f_every, 2), changes)
    _set_if_changed(settings, "stock_loss_streak_size_step_pct", round(stock_size_step, 3), changes)
    _set_if_changed(settings, "stock_loss_streak_size_floor_pct", round(stock_size_floor, 3), changes)
    _set_if_changed(settings, "forex_loss_streak_size_step_pct", round(forex_size_step, 3), changes)
    _set_if_changed(settings, "forex_loss_streak_size_floor_pct", round(forex_size_floor, 3), changes)

    # Safely promote execution stage in paper/practice after sustained health.
    stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    if (
        stage == "shadow_only"
        and bool(settings.get("alpaca_paper_mode", True))
        and bool(settings.get("oanda_practice_mode", True))
        and stable_cycles >= 12
        and markets_healthy
    ):
        _set_if_changed(settings, "market_rollout_stage", "execution_v2", changes)
        notes.append("Promoted rollout stage to execution_v2 (paper/practice healthy).")

    # Issue file: only if user action is required.
    issue_payload: Dict[str, Any] = {}
    if (kucoin_err + rate_err) >= 20:
        issue_payload = {
            "ts": now,
            "severity": "high",
            "title": "Persistent exchange/network instability",
            "detail": "Autopilot reduced aggressiveness; verify internet/exchange status and credentials.",
            "metrics": {"kucoin_errors": kucoin_err, "rate_errors": rate_err},
        }

    if not dry_run and changes:
        if _save_settings(settings_path, settings):
            _log(f"settings updated: {changes}")
        else:
            _log("settings update failed")
    elif changes:
        _log(f"dry-run settings delta: {changes}")

    if not dry_run:
        if issue_payload:
            _atomic_write_json(ISSUES_PATH, issue_payload)
        else:
            try:
                if os.path.isfile(ISSUES_PATH):
                    os.remove(ISSUES_PATH)
            except Exception:
                pass

    state_out = {
        "ts": now,
        "stable_cycles": stable_cycles,
        "offsets": offsets,
    }
    status_out = {
        "ts": now,
        "autonomous": True,
        "api_unstable": api_unstable,
        "markets_healthy": markets_healthy,
        "stable_cycles": stable_cycles,
        "kucoin_errors_window": kucoin_err,
        "rate_errors_window": rate_err,
        "changes": changes,
        "notes": notes[:6],
        "issue_open": bool(issue_payload),
        "current": {
            "kucoin_min_interval_sec": settings.get("kucoin_min_interval_sec", 0.40),
            "kucoin_cache_ttl_sec": settings.get("kucoin_cache_ttl_sec", 2.5),
            "crypto_trader_loop_sleep_s": settings.get("crypto_trader_loop_sleep_s", 1.0),
            "market_bg_stocks_interval_s": settings.get("market_bg_stocks_interval_s", 18.0),
            "market_bg_forex_interval_s": settings.get("market_bg_forex_interval_s", 12.0),
            "market_rollout_stage": settings.get("market_rollout_stage", "legacy"),
            "stock_loss_streak_size_step_pct": settings.get("stock_loss_streak_size_step_pct", 0.15),
            "stock_loss_streak_size_floor_pct": settings.get("stock_loss_streak_size_floor_pct", 0.40),
            "forex_loss_streak_size_step_pct": settings.get("forex_loss_streak_size_step_pct", 0.15),
            "forex_loss_streak_size_floor_pct": settings.get("forex_loss_streak_size_floor_pct", 0.40),
        },
    }
    if not dry_run:
        _atomic_write_json(AUTOPILOT_STATE_PATH, state_out)
        _atomic_write_json(AUTOPILOT_STATUS_PATH, status_out)
    return status_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Super Trader autonomous optimizer.")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    running = {"ok": True}

    def _stop(_signum: int, _frame: Any) -> None:
        running["ok"] = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if args.once:
        out = run_once(dry_run=bool(args.dry_run))
        _log(f"once complete stable={out.get('stable_cycles')} changes={out.get('changes')}")
        return 0

    interval_s = 30.0
    while running["ok"]:
        if os.path.exists(STOP_FLAG_PATH):
            break
        try:
            out = run_once(dry_run=bool(args.dry_run))
            _log(
                "tick "
                f"stable={out.get('stable_cycles')} "
                f"api_unstable={out.get('api_unstable')} "
                f"changes={len((out.get('changes') or {}))}"
            )
        except Exception as exc:
            _log(f"tick error {type(exc).__name__}: {exc}")
        time.sleep(interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
