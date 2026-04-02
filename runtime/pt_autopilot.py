from __future__ import annotations

import argparse
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
from app.json_codec import load as json_load
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
            data = json_load(f, default={})
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


def _stock_account_value_usd(stock_status: Dict[str, Any], stock_trader: Dict[str, Any]) -> float:
    try:
        v = _to_float(stock_trader.get("account_value_usd", 0.0), 0.0)
        if v > 0.0:
            return float(v)
    except Exception:
        pass
    for key in ("equity", "account_equity", "account_value"):
        try:
            v = _to_float(stock_status.get(key, 0.0), 0.0)
            if v > 0.0:
                return float(v)
        except Exception:
            continue
    return 0.0


def _local_load_ratio() -> float:
    try:
        load1, _load5, _load15 = os.getloadavg()
        cpu_n = max(1, int(os.cpu_count() or 1))
        return max(0.0, float(load1) / float(cpu_n))
    except Exception:
        return 0.0


def run_once(dry_run: bool = False) -> Dict[str, Any]:
    now = int(time.time())
    state = _safe_read_json(AUTOPILOT_STATE_PATH)
    offsets = state.get("offsets", {}) if isinstance(state.get("offsets", {}), dict) else {}
    stable_cycles = int(state.get("stable_cycles", 0) or 0)

    settings, settings_path = _load_settings_with_path()

    # health inputs
    stock_status = _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "stock_thinker_status.json"))
    stock_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "stock_trader_status.json"))
    forex_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "forex", "forex_trader_status.json"))
    stock_health = (stock_trader.get("health", {}) if isinstance(stock_trader, dict) else {}) or {}
    forex_health = (forex_trader.get("health", {}) if isinstance(forex_trader, dict) else {}) or {}
    stock_account_value = _stock_account_value_usd(
        stock_status if isinstance(stock_status, dict) else {},
        stock_trader if isinstance(stock_trader, dict) else {},
    )

    kucoin_err, rate_err = _tail_error_counts(THINKER_LOG_PATH, offsets)
    api_unstable = (kucoin_err + rate_err) >= 4
    markets_healthy = bool(stock_health.get("data_ok", True)) and bool(forex_health.get("data_ok", True))
    markets_healthy = markets_healthy and (not bool(stock_health.get("drift_warning", False))) and (not bool(forex_health.get("drift_warning", False)))
    load_ratio = _local_load_ratio()
    try:
        load_high_threshold = _clamp(_to_float(settings.get("autopilot_load_high_threshold", 0.60), 0.60), 0.30, 1.5)
    except Exception:
        load_high_threshold = 0.60
    local_load_high = bool(load_ratio >= load_high_threshold)

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
        if local_load_high:
            trader_loop = _clamp(trader_loop + 0.10, 0.8, 4.0)
            trader_err_sleep = _clamp(trader_err_sleep + 0.20, 1.2, 8.0)
            min_interval = _clamp(min_interval + 0.05, 0.45, 2.50)
            cache_ttl = _clamp(cache_ttl + 0.25, 1.8, 10.0)
            notes.append(f"Local system load high ({load_ratio:.2f}); easing crypto loop cadence.")
        elif stable_cycles >= 10:
            min_interval = _clamp(min_interval - 0.05, 0.35, 2.00)
            cache_ttl = _clamp(cache_ttl - 0.25, 1.5, 8.0)
            trader_loop = _clamp(trader_loop - 0.05, 0.5, 3.0)
            trader_err_sleep = _clamp(trader_err_sleep - 0.10, 1.0, 6.0)
            stable_cycles = 7
            notes.append("Stable APIs; cautiously improved responsiveness.")

    # Hard floor for sustained machine responsiveness (prevents runaway aggressiveness).
    min_crypto_loop_floor = 0.85 if local_load_high else 0.70
    trader_loop = max(float(min_crypto_loop_floor), float(trader_loop))
    trader_err_sleep = max(1.0, float(trader_err_sleep))

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
    try:
        min_stocks_interval = _clamp(_to_float(settings.get("autopilot_min_stocks_scan_interval_s", 16.0), 16.0), 8.0, 120.0)
    except Exception:
        min_stocks_interval = 16.0
    try:
        min_forex_interval = _clamp(_to_float(settings.get("autopilot_min_forex_scan_interval_s", 10.0), 10.0), 6.0, 120.0)
    except Exception:
        min_forex_interval = 10.0
    def _step_toward(cur: float, target: float, step: float, lo: float, hi: float) -> float:
        if cur < target:
            return _clamp(cur + abs(step), lo, hi)
        if cur > target:
            return _clamp(cur - abs(step), lo, hi)
        return _clamp(cur, lo, hi)

    if local_load_high:
        target_s_every = max(float(min_stocks_interval), 24.0)
        target_f_every = max(float(min_forex_interval), 18.0)
        s_every = _step_toward(s_every, target_s_every, 1.5, max(16.0, min_stocks_interval), 120.0)
        f_every = _step_toward(f_every, target_f_every, 1.5, max(10.0, min_forex_interval), 120.0)
        stock_size_step = _clamp(stock_size_step - 0.01, 0.05, 0.35)
        stock_size_floor = _clamp(stock_size_floor - 0.02, 0.20, 0.80)
        forex_size_step = _clamp(forex_size_step - 0.01, 0.05, 0.35)
        forex_size_floor = _clamp(forex_size_floor - 0.02, 0.20, 0.80)
        notes.append(f"Local system load high ({load_ratio:.2f}); increasing stock/forex scan intervals.")
    elif markets_healthy and (not api_unstable):
        s_every = _step_toward(s_every, float(min_stocks_interval), 0.75, float(min_stocks_interval), 120.0)
        f_every = _step_toward(f_every, float(min_forex_interval), 0.75, float(min_forex_interval), 120.0)
        stock_size_step = _clamp(stock_size_step + 0.01, 0.05, 0.35)
        stock_size_floor = _clamp(stock_size_floor + 0.02, 0.35, 0.80)
        forex_size_step = _clamp(forex_size_step + 0.01, 0.05, 0.35)
        forex_size_floor = _clamp(forex_size_floor + 0.02, 0.35, 0.80)
    else:
        target_s_every = max(float(min_stocks_interval), 20.0)
        target_f_every = max(float(min_forex_interval), 14.0)
        s_every = _step_toward(s_every, target_s_every, 1.0, float(min_stocks_interval), 120.0)
        f_every = _step_toward(f_every, target_f_every, 1.0, float(min_forex_interval), 120.0)
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

    # Stock scanner load shaping (capital-aware + local load-aware).
    scan_max = max(8, int(_to_float(settings.get("stock_scan_max_symbols", 160), 160)))
    mtf_max = max(0, int(_to_float(settings.get("stock_mtf_confirm_max_symbols", 12), 12)))
    fallback_limit = max(0, int(_to_float(settings.get("stock_scan_symbol_fallback_limit", 48), 48)))
    if stock_account_value >= 100_000.0:
        scan_target = 180
    elif stock_account_value >= 25_000.0:
        scan_target = 140
    elif stock_account_value >= 5_000.0:
        scan_target = 100
    elif stock_account_value >= 1_000.0:
        scan_target = 72
    elif stock_account_value >= 250.0:
        scan_target = 54
    else:
        scan_target = 40
    if local_load_high:
        scan_target = int(max(40, round(float(scan_target) * 0.85)))
    elif api_unstable or (not markets_healthy):
        scan_target = int(max(32, round(float(scan_target) * 0.80)))
    else:
        scan_target = int(max(32, scan_target))
    if scan_max > scan_target:
        scan_max = max(scan_target, scan_max - 12)
    elif (scan_max < scan_target) and (stable_cycles >= 10) and (not local_load_high):
        scan_max = min(scan_target, scan_max + 6)
    scan_max = int(max(16, min(220, scan_max)))
    mtf_target = int(max(0, min(16, round(float(scan_max) * 0.12))))
    fallback_target = int(max(0, min(120, round(float(scan_max) * 0.35))))
    if local_load_high:
        mtf_target = min(mtf_target, 8)
        fallback_target = int(max(8, round(float(fallback_target) * 0.75)))
    else:
        mtf_target = max(4, mtf_target)
        fallback_target = max(12, fallback_target)
    if mtf_max > mtf_target:
        mtf_max = max(mtf_target, mtf_max - 2)
    elif (mtf_max < mtf_target) and (stable_cycles >= 10) and (not local_load_high):
        mtf_max = min(mtf_target, mtf_max + 1)
    if fallback_limit > fallback_target:
        fallback_limit = max(fallback_target, fallback_limit - 6)
    elif (fallback_limit < fallback_target) and (stable_cycles >= 10) and (not local_load_high):
        fallback_limit = min(fallback_target, fallback_limit + 3)
    _set_if_changed(settings, "stock_scan_max_symbols", int(scan_max), changes)
    _set_if_changed(settings, "stock_mtf_confirm_max_symbols", int(max(0, mtf_max)), changes)
    _set_if_changed(settings, "stock_scan_symbol_fallback_limit", int(max(0, fallback_limit)), changes)
    if local_load_high:
        notes.append(
            f"Stock scanner load cap active: {int(scan_max)} symbols, {int(mtf_max)} MTF checks."
        )

    # UI cadence tuning for responsiveness on constrained systems.
    ui_refresh_s = _to_float(settings.get("ui_refresh_seconds", 1.0), 1.0)
    chart_refresh_s = _to_float(settings.get("chart_refresh_seconds", 10.0), 10.0)
    if local_load_high:
        ui_refresh_s = _clamp(ui_refresh_s + 0.15, 1.0, 3.0)
        chart_refresh_s = _clamp(chart_refresh_s + 0.75, 8.0, 25.0)
        notes.append("Raised UI/chart refresh intervals to reduce render pressure.")
    elif markets_healthy and (not api_unstable):
        ui_refresh_s = _clamp(ui_refresh_s - 0.05, 0.9, 3.0)
        chart_refresh_s = _clamp(chart_refresh_s - 0.25, 6.0, 25.0)
    else:
        ui_refresh_s = _clamp(ui_refresh_s + 0.10, 0.9, 3.0)
        chart_refresh_s = _clamp(chart_refresh_s + 0.50, 6.0, 25.0)
    _set_if_changed(settings, "ui_refresh_seconds", round(ui_refresh_s, 2), changes)
    _set_if_changed(settings, "chart_refresh_seconds", round(chart_refresh_s, 2), changes)

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
        "local_load_ratio": round(float(load_ratio), 3),
        "local_load_high": bool(local_load_high),
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
            "stock_scan_max_symbols": settings.get("stock_scan_max_symbols", 160),
            "stock_mtf_confirm_max_symbols": settings.get("stock_mtf_confirm_max_symbols", 12),
            "stock_scan_symbol_fallback_limit": settings.get("stock_scan_symbol_fallback_limit", 48),
            "stock_account_value_usd": round(float(stock_account_value), 2),
            "ui_refresh_seconds": settings.get("ui_refresh_seconds", 1.0),
            "chart_refresh_seconds": settings.get("chart_refresh_seconds", 10.0),
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
