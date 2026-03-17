from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from app.credential_utils import get_alpaca_creds, get_twelvedata_api_key
from app.confidence_calibration import build_market_confidence_calibration
from app.http_utils import retry_after_from_urllib_http_error
from app.news_event_provider import blend_score_with_news, build_unified_news_event_context
from app.path_utils import resolve_runtime_paths
from app.rejection_replay import recommend_threshold_from_scores, replay_target_entries_for_market
from app.scan_diagnostics_schema import with_scan_schema
from app.scanner_quality import build_universe_quality_report, effective_reject_pressure, quality_hints, turnover_pct
from brokers.broker_alpaca import AlpacaBrokerClient
from brokers.broker_twelvedata import TwelveDataClient

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "stock_thinker")

DEFAULT_STOCK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "SPY", "QQQ"]
_UNIVERSE_PRIORITY_SYMBOLS = [
    "QQQ",
    "SPY",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "IWM",
    "DIA",
    "AMD",
    "GOOGL",
    "NFLX",
    "PLTR",
    "AVGO",
    "SMCI",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "UNH",
    "COST",
]
_UNIVERSE_CACHE_SCHEMA = 2
_SCANNABLE_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}
ROLLOUT_ORDER = {
    "legacy": 0,
    "scan_expanded": 1,
    "risk_caps": 2,
    "execution_v2": 3,
    "shadow_only": 4,
    "live": 5,
    "live_guarded": 5,
}


def _normalize_rollout_stage(stage: Any) -> str:
    cur = str(stage or "legacy").strip().lower()
    if cur == "live_guarded":
        return "live"
    return cur


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _setting_bool(settings: Dict[str, Any], key: str, default: bool = False) -> bool:
    raw = settings.get(key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _stock_data_provider(settings: Dict[str, Any]) -> str:
    provider = str(settings.get("stock_data_provider", "alpaca") or "alpaca").strip().lower()
    return provider if provider in {"alpaca", "twelvedata"} else "alpaca"


def _twelvedata_scan_limits(settings: Dict[str, Any], symbol_count: int) -> Dict[str, float]:
    credits_per_min = max(1.0, float(settings.get("twelvedata_api_credits_per_minute", 8) or 8))
    daily_credits = max(1.0, float(settings.get("twelvedata_daily_credits", 800) or 800))
    credits_per_scan = max(1.0, float(symbol_count or 1))
    min_interval_min = (credits_per_scan / credits_per_min) * 60.0
    min_interval_day = (credits_per_scan / daily_credits) * 86400.0
    min_interval = max(60.0, min_interval_min, min_interval_day)
    return {
        "credits_per_min": credits_per_min,
        "daily_credits": daily_credits,
        "credits_per_scan": credits_per_scan,
        "min_interval_s": float(min_interval),
    }


def _twelvedata_snap_from_bars(bars_by_symbol: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    snap: Dict[str, Dict[str, float]] = {}
    for sym, bars in (bars_by_symbol or {}).items():
        rows = list(bars or [])
        if not rows:
            continue
        last_row = rows[-1]
        last_px = _float(last_row.get("c", last_row.get("close", 0.0)), 0.0)
        vol_sum = 0.0
        dollar_vol = 0.0
        for row in rows[-24:]:
            close_px = _float(row.get("c", row.get("close", 0.0)), 0.0)
            vol = _float(row.get("v", row.get("volume", 0.0)), 0.0)
            if vol > 0.0:
                vol_sum += vol
                if close_px > 0.0:
                    dollar_vol += close_px * vol
        snap[str(sym).strip().upper()] = {
            "bid": 0.0,
            "ask": 0.0,
            "mid": last_px,
            "last": last_px,
            "spread_bps": 0.0,
            "volume": vol_sum,
            "dollar_vol": dollar_vol,
        }
    return snap


def _request_json(url: str, headers: Dict[str, str], timeout: float = 10.0) -> Any:
    max_attempts = 4
    base_backoff_s = 0.35
    retry_http_codes = {408, 425, 429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            code = int(getattr(exc, "code", 0) or 0)
            if code not in retry_http_codes or attempt >= max_attempts:
                raise
            retry_after = retry_after_from_urllib_http_error(exc, max_wait_s=120.0)
            wait_s = max(
                retry_after,
                min(6.0, (base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)),
            )
            time.sleep(wait_s)
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            wait_s = min(6.0, (base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35))
            time.sleep(wait_s)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            wait_s = min(4.0, (0.2 * attempt) + random.uniform(0.0, 0.2))
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("stock_thinker request failed")


def _rollout_at_least(settings: Dict[str, Any], stage: str) -> bool:
    cur = _normalize_rollout_stage(settings.get("market_rollout_stage", "legacy"))
    target = _normalize_rollout_stage(stage)
    return int(ROLLOUT_ORDER.get(cur, 0)) >= int(ROLLOUT_ORDER.get(target, 0))


def _now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))


def _market_open_now() -> bool:
    now = _now_et()
    if now.weekday() >= 5:
        return False
    mins = (now.hour * 60) + now.minute
    return (9 * 60 + 30) <= mins < (16 * 60)


def _market_clock_status(settings: Dict[str, Any], api_key: str, secret: str) -> Dict[str, Any]:
    out = {
        "market_open": bool(_market_open_now()),
        "source": "local_schedule",
        "next_open": "",
        "next_close": "",
        "timestamp": "",
    }
    try:
        client = AlpacaBrokerClient(
            api_key_id=str(api_key or "").strip(),
            secret_key=str(secret or "").strip(),
            base_url=str(settings.get("alpaca_base_url", "https://paper-api.alpaca.markets") or ""),
            data_url=str(settings.get("alpaca_data_url", "https://data.alpaca.markets") or ""),
        )
        clock = client.get_market_clock()
        if isinstance(clock, dict) and clock:
            out["market_open"] = bool(clock.get("is_open", out["market_open"]))
            out["source"] = "alpaca_clock"
            out["next_open"] = str(clock.get("next_open", "") or "").strip()
            out["next_close"] = str(clock.get("next_close", "") or "").strip()
            out["timestamp"] = str(clock.get("timestamp", "") or "").strip()
    except Exception:
        pass
    return out


def _cache_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "stock_universe_cache.json")


def _rankings_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "scanner_rankings.jsonl")


def _execution_audit_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")


def _warmup_queue_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "warmup_queue.json")


def _scan_diag_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "scan_diagnostics.json")


def _scan_pause_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "scan_pause.json")


def _scan_rotation_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "scan_rotation.json")


def _feed_health_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "feed_health.json")


def _symbol_cooldown_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "symbol_cooldown.json")


def _quality_report_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "universe_quality.json")


def _opening_plan_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "opening_plan.json")


def _norm_id_list(value: Any) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    items = value if isinstance(value, list) else []
    for row in items:
        s = str(row or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _compact_chart_bars(rows: List[Dict[str, Any]], limit: int = 120) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    take_n = max(2, int(limit))
    for bar in list(rows or [])[-take_n:]:
        if not isinstance(bar, dict):
            continue
        c = _float(bar.get("c", 0.0), 0.0)
        if c <= 0.0:
            continue
        o = _float(bar.get("o", c), c)
        h = _float(bar.get("h", max(o, c)), max(o, c))
        low_px = _float(bar.get("l", min(o, c)), min(o, c))
        out.append(
            {
                "t": str(bar.get("t", "") or ""),
                "o": float(o),
                "h": float(max(h, o, c)),
                "l": float(min(low_px, o, c)),
                "c": float(c),
                "v": _float(bar.get("v", 0.0), 0.0),
            }
        )
    return out


def _build_top_chart_map(
    leaders: List[Dict[str, Any]],
    bars_lookup: Dict[str, List[Dict[str, Any]]],
    max_symbols: int = 6,
    limit: int = 120,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in list(leaders or []):
        if len(out) >= max(1, int(max_symbols)):
            break
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        bars = _compact_chart_bars(list(bars_lookup.get(symbol, []) or []), limit=limit)
        if len(bars) >= 2:
            out[symbol] = bars
    return out


def _leader_rank_score(row: Dict[str, Any]) -> float:
    score = _float(row.get("score", 0.0), 0.0)
    quality = _float(row.get("quality_score", 0.0), 0.0)
    calib_prob = _float(row.get("calib_prob", 0.0), 0.0)
    spread_bps = _float(row.get("spread_bps", 0.0), 0.0)
    valid_ratio = _float(row.get("valid_ratio", 0.0), 0.0)
    eligible_bonus = 2.0 if bool(row.get("eligible_for_entry", True)) else -2.0
    data_bonus = 1.0 if bool(row.get("data_quality_ok", True)) else -1.0
    return (
        (score * 12.0)
        + (quality * 0.08)
        + (calib_prob * 8.0)
        + (valid_ratio * 4.0)
        + eligible_bonus
        + data_bonus
        - (spread_bps * 0.05)
    )


def _apply_leader_hysteresis(
    leaders: List[Dict[str, Any]],
    prev_symbol: str,
    margin_pct: float,
) -> tuple[List[Dict[str, Any]], bool]:
    rows = [dict(r) for r in list(leaders or []) if isinstance(r, dict)]
    target = str(prev_symbol or "").strip().upper()
    margin = max(0.0, float(margin_pct or 0.0))
    if (not rows) or (not target) or margin <= 0.0:
        return rows, False
    top = rows[0]
    top_score = abs(_float(top.get("leader_rank_score", top.get("score", 0.0)), 0.0))
    top_side = str(top.get("side", "watch") or "watch").strip().lower()
    if top_score <= 0.0:
        return rows, False
    for idx, row in enumerate(rows[1:], start=1):
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if symbol != target:
            continue
        row_side = str(row.get("side", "watch") or "watch").strip().lower()
        if row_side != top_side:
            return rows, False
        row_score = abs(_float(row.get("leader_rank_score", row.get("score", 0.0)), 0.0))
        if row_score <= 0.0:
            return rows, False
        delta_pct = ((top_score - row_score) / max(1e-9, top_score)) * 100.0
        if delta_pct <= margin:
            return [row] + rows[:idx] + rows[idx + 1 :], True
        return rows, False
    return rows, False


def _stock_scan_window_policy(settings: Dict[str, Any], now_et: datetime | None = None) -> Dict[str, Any]:
    now = now_et or _now_et()
    if now.weekday() >= 5:
        return {"active": False, "window": "OFF", "score_mult": 1.0, "minutes": 0, "since_open_min": -1, "to_close_min": -1}
    mins = (now.hour * 60) + now.minute
    open_m = (9 * 60) + 30
    close_m = 16 * 60
    since_open = int(mins - open_m)
    to_close = int(close_m - mins)
    open_window = max(0, int(float(settings.get("stock_scan_open_cooldown_minutes", 15) or 15)))
    close_window = max(0, int(float(settings.get("stock_scan_close_cooldown_minutes", 15) or 15)))
    open_mult = max(0.5, min(1.0, float(settings.get("stock_scan_open_score_mult", 0.85) or 0.85)))
    close_mult = max(0.5, min(1.0, float(settings.get("stock_scan_close_score_mult", 0.90) or 0.90)))
    if since_open >= 0 and since_open < open_window:
        return {
            "active": True,
            "window": "OPENING",
            "score_mult": float(open_mult),
            "minutes": int(open_window),
            "since_open_min": int(since_open),
            "to_close_min": int(to_close),
        }
    if to_close > 0 and to_close <= close_window:
        return {
            "active": True,
            "window": "CLOSING",
            "score_mult": float(close_mult),
            "minutes": int(close_window),
            "since_open_min": int(since_open),
            "to_close_min": int(to_close),
        }
    return {
        "active": False,
        "window": "NONE",
        "score_mult": 1.0,
        "minutes": 0,
        "since_open_min": int(since_open),
        "to_close_min": int(to_close),
    }


def _rotate_jsonl(path: str, max_bytes: int = 25 * 1024 * 1024, keep: int = 10) -> None:
    try:
        if not os.path.isfile(path):
            return
        if os.path.getsize(path) <= int(max_bytes):
            return
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        rotated = f"{path}.{ts}"
        os.replace(path, rotated)
        prefix = os.path.basename(path) + "."
        base_dir = os.path.dirname(path)
        olds = sorted([os.path.join(base_dir, n) for n in os.listdir(base_dir) if n.startswith(prefix)])
        if len(olds) > int(keep):
            for old in olds[:-keep]:
                try:
                    os.remove(old)
                except Exception:
                    pass
    except Exception:
        pass


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _rotate_jsonl(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _load_universe_cache(hub_dir: str, ttl_s: int = 1800) -> List[str]:
    path = _cache_path(hub_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return []
        schema = int(payload.get("schema", 1) or 1)
        if schema != int(_UNIVERSE_CACHE_SCHEMA):
            return []
        ts = int(payload.get("ts", 0) or 0)
        if (int(time.time()) - ts) > int(ttl_s):
            return []
        symbols = payload.get("symbols", []) or []
        if not isinstance(symbols, list):
            return []
        out = [str(x).strip().upper() for x in symbols if str(x).strip()]
        return out
    except Exception:
        return []


def _save_universe_cache(hub_dir: str, symbols: List[str]) -> None:
    path = _cache_path(hub_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"schema": int(_UNIVERSE_CACHE_SCHEMA), "ts": int(time.time()), "symbols": symbols}, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _parse_watchlist(settings: Dict[str, Any]) -> List[str]:
    raw = str(settings.get("stock_universe_symbols", "") or "")
    out: List[str] = []
    for tok in raw.replace("\n", ",").split(","):
        s = tok.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _load_warmup_queue(hub_dir: str, ttl_s: int = 7200) -> Dict[str, Dict[str, Any]]:
    path = _warmup_queue_path(hub_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        if not isinstance(rows, dict):
            return {}
        now_ts = int(time.time())
        out: Dict[str, Dict[str, Any]] = {}
        for sym, row in rows.items():
            s = str(sym or "").strip().upper()
            if not s or not isinstance(row, dict):
                continue
            last_seen = int(row.get("last_seen", 0) or 0)
            if (now_ts - last_seen) > int(ttl_s):
                continue
            out[s] = row
        return out
    except Exception:
        return {}


def _save_warmup_queue(hub_dir: str, queue_map: Dict[str, Dict[str, Any]]) -> None:
    path = _warmup_queue_path(hub_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "symbols": queue_map}, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _save_scan_diagnostics(hub_dir: str, payload: Dict[str, Any]) -> None:
    path = _scan_diag_path(hub_dir)
    try:
        row = with_scan_schema(payload, market="stocks")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_json_map(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json_map(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _thinker_status_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")


def _confidence_calibration_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "confidence_calibration.json")


def _build_opening_plan(
    settings: Dict[str, Any],
    leaders: List[Dict[str, Any]],
    all_scores: List[Dict[str, Any]],
    adaptive_threshold: float,
    ts_now: int,
    market_clock: Dict[str, Any] | None,
    market_open: bool,
    msg: str,
    fallback_cached: bool = False,
) -> Dict[str, Any]:
    enabled = _setting_bool(settings, "stock_opening_plan_enabled", True)
    rows_limit = max(1, min(20, int(float(settings.get("stock_opening_plan_max_symbols", 8) or 8))))
    window_minutes = max(5, min(240, int(float(settings.get("stock_opening_plan_minutes", 45) or 45))))
    next_open = str((market_clock or {}).get("next_open", "") or "").strip()
    threshold = max(0.01, float(adaptive_threshold or settings.get("stock_score_threshold", 0.2) or 0.2))

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in list(leaders or []) + list(all_scores or []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if (not symbol) or (symbol in seen):
            continue
        seen.add(symbol)
        merged.append(dict(row))
    merged.sort(
        key=lambda r: (
            1 if bool(r.get("eligible_for_entry", False)) else 0,
            float(r.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for row in merged[:rows_limit]:
        symbol = str(row.get("symbol", "") or "").strip().upper()
        side = str(row.get("side", "watch") or "watch").strip().lower()
        score = float(row.get("score", 0.0) or 0.0)
        eligible = bool(row.get("eligible_for_entry", False))
        data_ok = bool(row.get("data_quality_ok", True))
        confidence = str(row.get("confidence", "N/A") or "N/A").strip().upper() or "N/A"
        calib_prob = float(row.get("calibration_effective_prob", row.get("calib_prob", 0.0)) or 0.0)
        quality_score = float(row.get("quality_score", 0.0) or 0.0)
        gate_reason = str(row.get("entry_gate_reason", "") or "").strip()
        logic = str(row.get("reason_logic", row.get("reason", "")) or "").strip()

        ready = bool(side == "long" and eligible and data_ok and (score >= threshold))
        if ready:
            status = "READY"
            trigger = f"Open-entry candidate if score >= {threshold:.3f} and spread/slippage checks pass."
        elif side == "long":
            status = "ENTRY WAIT"
            trigger = gate_reason or "Needs scanner promotion or quality gate clearance before entry."
        else:
            status = "WATCH"
            trigger = "Watch for scanner promotion from WATCH to LONG."

        rows.append(
            {
                "symbol": symbol,
                "side": side.upper(),
                "score": round(score, 6),
                "confidence": confidence,
                "calib_prob": round(calib_prob, 4),
                "quality_score": round(quality_score, 3),
                "status": status,
                "logic": logic,
                "trigger": trigger,
            }
        )

    ready_count = sum(1 for row in rows if str(row.get("status", "")).upper() == "READY")
    return {
        "enabled": bool(enabled),
        "generated_ts": int(ts_now),
        "market_open": bool(market_open),
        "next_open": next_open,
        "plan_window_minutes": int(window_minutes),
        "adaptive_threshold": round(float(threshold), 6),
        "source_msg": str(msg or ""),
        "fallback_cached": bool(fallback_cached),
        "rows": rows if enabled else [],
        "summary": f"{ready_count} ready | {max(0, len(rows) - ready_count)} watch/wait",
    }


def _persist_opening_plan(
    settings: Dict[str, Any],
    hub_dir: str,
    *,
    leaders: List[Dict[str, Any]],
    all_scores: List[Dict[str, Any]],
    adaptive_threshold: float,
    ts_now: int,
    market_clock: Dict[str, Any] | None,
    market_open: bool,
    msg: str,
    fallback_cached: bool = False,
) -> Dict[str, Any]:
    plan = _build_opening_plan(
        settings=settings,
        leaders=leaders,
        all_scores=all_scores,
        adaptive_threshold=adaptive_threshold,
        ts_now=ts_now,
        market_clock=market_clock,
        market_open=market_open,
        msg=msg,
        fallback_cached=fallback_cached,
    )
    _save_json_map(_opening_plan_path(hub_dir), plan)
    return plan


def _cached_scan_fallback(
    hub_dir: str,
    ts_now: int,
    msg: str,
    universe: List[str],
    market_open: bool,
) -> Dict[str, Any] | None:
    prev = _load_json_map(_thinker_status_path(hub_dir))
    if not prev:
        return None
    leaders = list(prev.get("leaders", []) or []) if isinstance(prev.get("leaders", []), list) else []
    all_scores = list(prev.get("all_scores", []) or []) if isinstance(prev.get("all_scores", []), list) else []
    top_pick_raw = prev.get("top_pick", {})
    top_pick = dict(top_pick_raw) if isinstance(top_pick_raw, dict) else {}
    top_chart = list(prev.get("top_chart", []) or []) if isinstance(prev.get("top_chart", []), list) else []
    top_chart_map_raw = prev.get("top_chart_map", {})
    top_chart_map = dict(top_chart_map_raw) if isinstance(top_chart_map_raw, dict) else {}
    if not top_chart:
        top_symbol = str((top_pick or {}).get("symbol", "") or "").strip().upper()
        if top_symbol and isinstance(top_chart_map.get(top_symbol, None), list):
            top_chart = list(top_chart_map.get(top_symbol, []) or [])
    if (not leaders) and (not all_scores) and (not top_chart):
        return None

    prev_updated = int(float(prev.get("updated_at", prev.get("ts", 0)) or 0))
    age_s = max(0, int(ts_now - prev_updated)) if prev_updated > 0 else 0
    fallback_msg = f"{str(msg or '').strip()} | using cached scan ({age_s}s old)"
    prev_hints = list(prev.get("hints", []) or []) if isinstance(prev.get("hints", []), list) else []
    hints = [f"Network/data degraded; serving cached leaders ({age_s}s old)."]
    for h in prev_hints[:4]:
        sh = str(h or "").strip()
        if sh and sh not in hints:
            hints.append(sh)

    reject_summary = (dict(prev.get("reject_summary", {})) if isinstance(prev.get("reject_summary", {}), dict) else {})
    prev_adaptive = float(prev.get("adaptive_threshold", 0.2) or 0.2)
    return {
        "state": "READY",
        "ai_state": "Scan degraded (cached)",
        "msg": fallback_msg,
        "universe": list(universe or prev.get("universe", []) or []),
        "leaders": leaders[:10],
        "all_scores": all_scores[:40],
        "top_pick": (top_pick if top_pick else (leaders[0] if leaders and isinstance(leaders[0], dict) else None)),
        "top_chart": top_chart[-120:],
        "top_chart_map": top_chart_map,
        "top_chart_source": str(prev.get("top_chart_source", "") or ""),
        "adaptive_threshold": float(prev_adaptive),
        "adaptive_threshold_base": float(prev.get("adaptive_threshold_base", prev_adaptive) or prev_adaptive),
        "adaptive_threshold_volatility": float(prev.get("adaptive_threshold_volatility", prev_adaptive) or prev_adaptive),
        "adaptive_threshold_replay_recommended": float(prev.get("adaptive_threshold_replay_recommended", prev_adaptive) or prev_adaptive),
        "adaptive_threshold_replay_clamped": float(prev.get("adaptive_threshold_replay_clamped", prev_adaptive) or prev_adaptive),
        "adaptive_threshold_replay_weight": float(prev.get("adaptive_threshold_replay_weight", 0.0) or 0.0),
        "adaptive_threshold_replay_target_entries": int(prev.get("adaptive_threshold_replay_target_entries", 0) or 0),
        "adaptive_threshold_replay_reason": str(prev.get("adaptive_threshold_replay_reason", "") or ""),
        "adaptive_threshold_replay_enabled": bool(prev.get("adaptive_threshold_replay_enabled", False)),
        "updated_at": int(ts_now),
        "market_open": bool(market_open),
        "rejected": list(prev.get("rejected", []) or [])[:30],
        "reject_summary": reject_summary,
        "feed_order": list(prev.get("feed_order", []) or [])[:4],
        "hints": hints[:5],
        "candidate_churn_pct": float(prev.get("candidate_churn_pct", 0.0) or 0.0),
        "leader_churn_pct": float(prev.get("leader_churn_pct", 0.0) or 0.0),
        "window_policy": (dict(prev.get("window_policy", {})) if isinstance(prev.get("window_policy", {}), dict) else {}),
        "window_policy_hits": int(prev.get("window_policy_hits", 0) or 0),
        "universe_quality": (dict(prev.get("universe_quality", {})) if isinstance(prev.get("universe_quality", {}), dict) else {}),
        "leader_mode": str(prev.get("leader_mode", "cached") or "cached"),
        "leader_stability_applied": bool(prev.get("leader_stability_applied", False)),
        "leader_stability_prev_symbol": str(prev.get("leader_stability_prev_symbol", "") or ""),
        "fallback_cached": True,
        "health": {"data_ok": False, "broker_ok": True, "orders_ok": True, "drift_warning": True},
        "pdt_note": "Paper mode can still simulate PDT protections; live day-trading may be limited under $25k.",
    }


def _adaptive_feed_order(base_order: List[str], feed_health: Dict[str, Any]) -> List[str]:
    rows = (feed_health or {}).get("feeds", {}) if isinstance(feed_health, dict) else {}
    if not isinstance(rows, dict):
        rows = {}

    def _score(feed: str) -> float:
        row = rows.get(feed, {}) if isinstance(rows.get(feed, {}), dict) else {}
        ok_count = float(row.get("ok_count", 0.0) or 0.0)
        err_count = float(row.get("err_count", 0.0) or 0.0)
        bars_avg = float(row.get("avg_bars", 0.0) or 0.0)
        fresh_bonus = 0.0
        try:
            age = max(0.0, time.time() - float(row.get("updated_ts", 0.0) or 0.0))
            fresh_bonus = max(0.0, 3.0 - min(3.0, age / 1800.0))
        except Exception:
            fresh_bonus = 0.0
        return (ok_count * 2.0) - (err_count * 1.2) + (bars_avg * 0.02) + fresh_bonus

    unique = []
    for f in list(base_order or []):
        ff = str(f or "").strip().lower()
        if ff and ff not in unique:
            unique.append(ff)
    if not unique:
        unique = ["sip", "iex"]
    return sorted(unique, key=lambda x: _score(x), reverse=True)


def _update_feed_health(feed_health: Dict[str, Any], feed: str, ok: bool, bars_total: int = 0) -> Dict[str, Any]:
    data = dict(feed_health if isinstance(feed_health, dict) else {})
    rows = data.get("feeds", {})
    if not isinstance(rows, dict):
        rows = {}
    key = str(feed or "").strip().lower()
    if not key:
        return data
    row = rows.get(key, {})
    if not isinstance(row, dict):
        row = {}
    row["ok_count"] = int(row.get("ok_count", 0) or 0) + (1 if ok else 0)
    row["err_count"] = int(row.get("err_count", 0) or 0) + (0 if ok else 1)
    prev_samples = int(row.get("samples", 0) or 0)
    new_samples = prev_samples + 1
    prev_avg = float(row.get("avg_bars", 0.0) or 0.0)
    row["avg_bars"] = ((prev_avg * prev_samples) + float(max(0, bars_total))) / float(max(1, new_samples))
    row["samples"] = int(new_samples)
    row["updated_ts"] = int(time.time())
    rows[key] = row
    data["feeds"] = rows
    data["ts"] = int(time.time())
    return data


def _cooldown_reasons(settings: Dict[str, Any], key: str, default_csv: str) -> set[str]:
    raw = str(settings.get(key, default_csv) or default_csv)
    out = set()
    for tok in raw.replace(";", ",").split(","):
        r = str(tok or "").strip().lower()
        if r:
            out.add(r)
    # Legacy defaults cooled down spread/liquidity rejects too aggressively and
    # could leave most of the universe muted. Treat that exact legacy set as the
    # newer hard-data-only default unless the operator explicitly customizes it.
    if out == {"data_quality", "insufficient_bars", "spread", "liquidity"}:
        return {"data_quality", "insufficient_bars"}
    return out


def _apply_symbol_cooldown(
    cooldown_map: Dict[str, Any],
    symbol: str,
    reason: str,
    settings: Dict[str, Any],
    now_ts: int,
) -> None:
    sym = str(symbol or "").strip().upper()
    rsn = str(reason or "").strip().lower()
    if not sym or not rsn:
        return
    reasons = _cooldown_reasons(
        settings,
        "stock_symbol_cooldown_reject_reasons",
        "data_quality,insufficient_bars,spread,liquidity",
    )
    if rsn not in reasons:
        return
    mins = max(1, int(float(settings.get("stock_symbol_cooldown_minutes", 30) or 30)))
    min_hits = max(1, int(float(settings.get("stock_symbol_cooldown_min_hits", 2) or 2)))
    row = cooldown_map.get(sym, {})
    if not isinstance(row, dict):
        row = {}
    hit_count = int(row.get("hit_count", 0) or 0) + 1
    until = int(row.get("until", 0) or 0)
    if hit_count >= min_hits:
        until = int(now_ts + (mins * 60))
        hit_count = 0
    cooldown_map[sym] = {
        "symbol": sym,
        "reason": rsn,
        "hit_count": int(hit_count),
        "until": int(until),
        "updated_ts": int(now_ts),
    }


def _prune_cooldown_map(cooldown_map: Dict[str, Any], now_ts: int, settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    allowed_reasons = _cooldown_reasons(
        settings if isinstance(settings, dict) else {},
        "stock_symbol_cooldown_reject_reasons",
        "data_quality,insufficient_bars",
    )
    for sym, row in (cooldown_map or {}).items():
        if not isinstance(row, dict):
            continue
        s = str(sym or "").strip().upper()
        if not s:
            continue
        reason = str(row.get("reason", "") or "").strip().lower()
        if reason and reason not in allowed_reasons:
            continue
        until = int(row.get("until", 0) or 0)
        updated = int(row.get("updated_ts", 0) or 0)
        if until > now_ts:
            out[s] = row
            continue
        # Keep recent hit counters for a short period even if no active lockout.
        if (now_ts - updated) <= 7200:
            out[s] = row
    return out


def _fetch_bars_for_symbols(
    base_url: str,
    headers: Dict[str, str],
    symbols: List[str],
    start_iso: str,
    end_iso: str,
    feed: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    chunk = 50
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        params = {
            "symbols": ",".join(part),
            "timeframe": "1Hour",
            "limit": "96",
            "adjustment": "raw",
            "feed": feed,
            "start": start_iso,
            "end": end_iso,
            # Prefer latest bars; API sorts by symbol then timestamp.
            "sort": "desc",
        }
        url = f"{base_url}/v2/stocks/bars?{urllib.parse.urlencode(params)}"
        payload = _request_json(url, headers=headers, timeout=15.0)
        bars = payload.get("bars", {}) or {}
        if isinstance(bars, dict):
            for sym, rows in bars.items():
                key = str(sym).strip().upper()
                if key:
                    norm_rows = [row for row in list(rows or []) if isinstance(row, dict)]
                    norm_rows.sort(key=lambda row: str(row.get("t", "") or ""))
                    out[key] = norm_rows
    return out


def _score_bars(symbol: str, bars: List[Dict[str, Any]], spread_bps: float = 0.0) -> Dict[str, Any]:
    closes = []
    for row in bars or []:
        if not isinstance(row, dict):
            continue
        close_val = _float(row.get("c", row.get("close", 0.0)), 0.0)
        if close_val > 0:
            closes.append(close_val)
    if len(closes) < 8:
        reason_logic = "Insufficient market history for a reliable trend call"
        reason_data = "bars<8 on 1h sample"
        return {
            "symbol": symbol,
            "score": -9999.0,
            "side": "watch",
            "last": closes[-1] if closes else 0.0,
            "change_6h_pct": 0.0,
            "change_24h_pct": 0.0,
            "volatility_pct": 0.0,
            "spread_bps": round(float(spread_bps), 4),
            "confidence": "LOW",
            "reason_logic": reason_logic,
            "reason_data": reason_data,
            "reason": reason_logic,
        }

    last_px = closes[-1]
    px_6 = closes[max(0, len(closes) - 7)]
    px_24 = closes[max(0, len(closes) - min(24, len(closes)))]
    change_6 = ((last_px - px_6) / px_6) * 100.0 if px_6 > 0 else 0.0
    change_24 = ((last_px - px_24) / px_24) * 100.0 if px_24 > 0 else 0.0
    step_moves = []
    for idx in range(1, len(closes)):
        prev_px = closes[idx - 1]
        cur_px = closes[idx]
        if prev_px > 0:
            step_moves.append(abs(((cur_px - prev_px) / prev_px) * 100.0))
    volatility = (sum(step_moves[-12:]) / max(1, len(step_moves[-12:]))) if step_moves else 0.0

    spread_penalty = max(0.0, float(spread_bps) / 8.0)
    score = (change_6 * 0.60) + (change_24 * 0.25) + (volatility * 0.20) - spread_penalty
    side = "long" if score > 0 else "watch"
    abs_score = abs(score)
    if abs_score >= 4.0:
        confidence = "HIGH"
    elif abs_score >= 1.75:
        confidence = "MED"
    else:
        confidence = "LOW"
    if side == "long":
        if change_6 >= 0.0 and change_24 >= 0.0:
            reason_logic = "Uptrend pressure from positive 6h/24h momentum"
        else:
            reason_logic = "Long bias from recent upside, but momentum is mixed"
    else:
        if change_6 <= 0.0 and change_24 <= 0.0:
            reason_logic = "Downtrend pressure; watchlist only until long trigger appears"
        elif change_6 <= 0.0:
            reason_logic = "Near-term weakness; kept on watchlist for reversal confirmation"
        else:
            reason_logic = "Watchlist candidate with mixed momentum signals"
    if abs(float(score)) < 0.10 and volatility < 0.20:
        reason_logic = "Range/low-volatility behavior; no clear directional edge"
    reason_data = f"6h {change_6:+.2f}% | 24h {change_24:+.2f}% | vol {volatility:.2f}% | spr {float(spread_bps):.2f}bps"
    return {
        "symbol": symbol,
        "score": round(score, 6),
        "side": side,
        "last": round(last_px, 6),
        "change_6h_pct": round(change_6, 6),
        "change_24h_pct": round(change_24, 6),
        "volatility_pct": round(volatility, 6),
        "spread_bps": round(float(spread_bps), 4),
        "confidence": confidence,
        "reason_logic": reason_logic,
        "reason_data": reason_data,
        "reason": reason_logic,
    }


def _append_reason_parts(row: Dict[str, Any], logic: str = "", data: str = "") -> None:
    cur_logic = str(row.get("reason_logic", row.get("reason", "")) or "").strip()
    cur_data = str(row.get("reason_data", "") or "").strip()
    add_logic = str(logic or "").strip()
    add_data = str(data or "").strip()
    if add_logic:
        cur_logic = f"{cur_logic} | {add_logic}" if cur_logic else add_logic
    if add_data:
        cur_data = f"{cur_data} | {add_data}" if cur_data else add_data
    row["reason_logic"] = cur_logic
    row["reason_data"] = cur_data
    row["reason"] = cur_logic if cur_logic else cur_data


def _live_guarded_entry_gate_reason(settings: Dict[str, Any], row: Dict[str, Any]) -> str:
    stage = _normalize_rollout_stage(settings.get("market_rollout_stage", "legacy"))
    if stage != "live":
        return ""
    # Paper mode is the calibration warmup path; keep live-only calibration gates off
    # so the stock paper trader can accumulate the samples required for live.
    if _setting_bool(settings, "alpaca_paper_mode", True):
        return ""
    symbol = str(row.get("symbol", "") or "").strip().upper()
    if not symbol:
        return ""
    sample_count = int(float(row.get("calibration_effective_samples", row.get("samples", 0)) or 0))
    symbol_samples = int(float(row.get("symbol_samples", row.get("samples", 0)) or 0))
    market_samples = int(float(row.get("market_calibration_samples", 0) or 0))
    min_samples_guarded = max(0, int(float(settings.get("stock_min_samples_live_guarded", 5) or 5)))
    bootstrap_allow = _setting_bool(settings, "stock_live_guarded_bootstrap_allow", True)
    if sample_count < min_samples_guarded:
        if bootstrap_allow and symbol_samples <= 0 and market_samples <= 0:
            return ""
        if market_samples > symbol_samples:
            return f"Calibration sample gate for {symbol} ({symbol_samples} symbol / {market_samples} pooled < {min_samples_guarded})"
        return f"Calibration sample gate for {symbol} ({symbol_samples} < {min_samples_guarded})"
    calib_prob = float(row.get("calibration_effective_prob", row.get("calib_prob", 0.0)) or 0.0)
    if calib_prob <= 0.0:
        calib_prob = 0.5
    min_calib_prob = max(0.0, min(1.0, float(settings.get("stock_min_calib_prob_live_guarded", 0.58) or 0.58)))
    if calib_prob < min_calib_prob:
        return f"Calibrated confidence gate for {symbol} ({calib_prob:.2f} < {min_calib_prob:.2f})"
    return ""


def _market_pooled_calibration_samples(hub_dir: str, settings: Dict[str, Any]) -> int:
    payload = _load_json_map(_confidence_calibration_path(hub_dir))
    row: Dict[str, Any] = {}
    if isinstance(payload.get("stocks", {}), dict):
        row = dict(payload.get("stocks", {}) or {})
    elif str(payload.get("market", "") or "").strip().lower() == "stocks":
        row = payload
    try:
        samples = int(float(row.get("samples", 0) or 0))
    except Exception:
        samples = 0
    if samples > 0:
        return samples
    try:
        base_threshold = float(settings.get("stock_score_threshold", 0.2) or 0.2)
    except Exception:
        base_threshold = 0.2
    try:
        adaptive_min_samples = max(6, int(float(settings.get("adaptive_confidence_min_samples", 18) or 18)))
    except Exception:
        adaptive_min_samples = 18
    try:
        target_success = max(30.0, min(90.0, float(settings.get("adaptive_confidence_target_success_pct", 55.0) or 55.0)))
    except Exception:
        target_success = 55.0
    built = build_market_confidence_calibration(
        hub_dir,
        "stocks",
        base_threshold=base_threshold,
        min_samples=adaptive_min_samples,
        target_success_pct=target_success,
    )
    try:
        return int(float(built.get("samples", 0) or 0))
    except Exception:
        return 0


def _apply_stock_mtf_confirmation(
    scored: List[Dict[str, Any]],
    client: AlpacaBrokerClient,
    feed: str,
    settings: Dict[str, Any],
) -> None:
    if not scored:
        return
    try:
        max_symbols = max(0, int(float(settings.get("stock_mtf_confirm_max_symbols", 24) or 24)))
    except Exception:
        max_symbols = 24
    for row in scored:
        row["mtf_side"] = "skipped"
        row["mtf_confirmed"] = None
    if max_symbols <= 0:
        return
    ranked = [
        row
        for row in scored
        if isinstance(row, dict) and str(row.get("side", "watch")).strip().lower() == "long"
    ]
    ranked.sort(key=lambda row: abs(_float(row.get("score", 0.0), 0.0)), reverse=True)
    for row in ranked[:max_symbols]:
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        spread_bps = _float(row.get("spread_bps", 0.0), 0.0)
        mtf_side = "watch"
        try:
            bars_4h = client.get_stock_bars(symbol, timeframe="4Hour", limit=36, feed=feed)
            if len(bars_4h) < 8:
                bars_4h = client.get_stock_bars(symbol, timeframe="1Day", limit=36, feed=feed)
            mtf = _score_bars(symbol, bars_4h, spread_bps=spread_bps)
            mtf_score = float(mtf.get("score", 0.0) or 0.0)
            mtf_side = "long" if mtf_score > 0 else "watch"
        except Exception:
            mtf_side = "watch"
        row["mtf_side"] = mtf_side
        row["mtf_confirmed"] = bool(mtf_side == "long")
        if not bool(row["mtf_confirmed"]):
            row["score"] = round(float(row.get("score", 0.0) or 0.0) * 0.70, 6)
            _append_reason_parts(
                row,
                logic="Multi-timeframe mismatch lowered conviction",
                data="mtf mismatch",
            )


def _bar_quality(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    if not bars:
        return {"valid_ratio": 0.0, "stale_hours": 9999.0}
    valid = 0
    latest_ts = 0.0
    for row in bars:
        if not isinstance(row, dict):
            continue
        c = _float(row.get("c", 0.0), 0.0)
        if c > 0:
            valid += 1
        t = str(row.get("t", "") or "").strip()
        if t:
            try:
                ts = _parse_iso_ts(t)
                latest_ts = max(latest_ts, ts)
            except Exception:
                pass
    ratio = float(valid) / float(max(1, len(bars)))
    stale_h = 9999.0
    if latest_ts > 0:
        stale_h = max(0.0, (time.time() - latest_ts) / 3600.0)
    return {"valid_ratio": ratio, "stale_hours": stale_h}


def _parse_iso_ts(raw_ts: str) -> float:
    s = str(raw_ts or "").strip()
    if not s:
        return 0.0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "." in s:
        try:
            head, tail = s.split(".", 1)
            tz_idx = max(tail.rfind("+"), tail.rfind("-"))
            if tz_idx > 0:
                frac = tail[:tz_idx]
                tz = tail[tz_idx:]
            else:
                frac = tail
                tz = ""
            frac = (frac + "000000")[:6]
            s = f"{head}.{frac}{tz}"
        except Exception:
            pass
    return datetime.fromisoformat(s).timestamp()


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compute_outcome_map(hub_dir: str, limit: int = 500) -> Dict[str, Dict[str, float]]:
    path = _execution_audit_path(hub_dir)
    recent: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if str(obj.get("event", "")).lower() in {"exit", "shadow_exit"}:
                    recent.append(obj)
    except Exception:
        return {}
    if len(recent) > int(limit):
        recent = recent[-int(limit):]
    per: Dict[str, List[float]] = {}
    for row in recent:
        sym = str(row.get("symbol", "") or "").strip().upper()
        if not sym:
            continue
        pnl = _float(row.get("pnl_pct", 0.0), 0.0)
        per.setdefault(sym, []).append(pnl)
    out: Dict[str, Dict[str, float]] = {}
    for sym, pnls in per.items():
        wins = sum(1 for p in pnls if p > 0.0)
        out[sym] = {
            "hit_rate_pct": round((100.0 * wins / max(1, len(pnls))), 2),
            "avg_pnl_pct": round((sum(pnls) / max(1, len(pnls))), 4),
            "samples": float(len(pnls)),
        }
    return out


def _calibrated_prob(score: float, hit_rate_pct: float, avg_pnl_pct: float) -> float:
    # Lightweight calibration: blend score magnitude + realized hit rate + expectancy.
    score_term = max(0.0, min(1.0, abs(score) / 3.0))
    hit_term = max(0.0, min(1.0, float(hit_rate_pct) / 100.0))
    pnl_term = max(0.0, min(1.0, (float(avg_pnl_pct) + 2.0) / 4.0))
    return round((0.45 * score_term) + (0.40 * hit_term) + (0.15 * pnl_term), 4)


def _market_hints_from_rejects(reject_summary: Dict[str, Any]) -> List[str]:
    counts = (reject_summary or {}).get("counts", {}) or {}
    if not isinstance(counts, dict):
        counts = {}
    dominant = str((reject_summary or {}).get("dominant_reason", "") or "").strip().lower()
    rate = float((reject_summary or {}).get("reject_rate_pct", 0.0) or 0.0)
    hints: List[str] = []
    if rate >= 70.0:
        hints.append("High reject rate: widen universe or relax one gate at a time.")
    if dominant == "data_quality":
        hints.append("Data quality dominates: lower min valid bars ratio or increase max stale hours.")
    elif dominant == "insufficient_bars":
        hints.append("Insufficient bars dominates: reduce min bars required or let warmup run longer.")
    elif dominant == "spread":
        hints.append("Spread dominates: raise max spread bps slightly or focus on large-cap symbols.")
    elif dominant == "liquidity":
        hints.append("Liquidity dominates: lower min dollar volume or scan fewer, more liquid names.")
    elif dominant == "price_band":
        hints.append("Price band dominates: expand min/max price range in settings.")
    elif dominant == "warmup_pending":
        hints.append("Warmup pending: keep scanner running to hydrate sparse symbols.")
    if not hints and counts:
        hints.append("Scanner healthy; tune score threshold for more/less selectivity.")
    return hints[:3]


_REJECT_REASON_PRIORITY = {
    "data_quality": 100,
    "insufficient_bars": 90,
    "warmup_pending": 80,
    "spread": 70,
    "liquidity": 60,
    "price_band": 50,
    "unknown": 10,
}


def _summarize_rejections(rejected: List[Dict[str, Any]], universe_size: int) -> Dict[str, Any]:
    # Summarize by symbol (not raw events) so reject rate remains bounded to 0..100%.
    best_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in list(rejected or []):
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "") or "").strip().upper()
        reason = str(row.get("reason", "unknown") or "unknown").strip().lower() or "unknown"
        if not sym:
            continue
        cur = best_by_symbol.get(sym)
        cur_reason = str((cur or {}).get("reason", "unknown") or "unknown").strip().lower() if isinstance(cur, dict) else "unknown"
        cur_pri = int(_REJECT_REASON_PRIORITY.get(cur_reason, 0))
        new_pri = int(_REJECT_REASON_PRIORITY.get(reason, 0))
        if (cur is None) or (new_pri > cur_pri):
            best_by_symbol[sym] = {"symbol": sym, "reason": reason}

    reason_counts: Dict[str, int] = {}
    for row in best_by_symbol.values():
        reason = str(row.get("reason", "unknown") or "unknown")
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    total_unique = int(len(best_by_symbol))
    dominant_reason = max(reason_counts.items(), key=lambda item: item[1])[0] if reason_counts else ""
    dominant_ratio = (float(reason_counts.get(dominant_reason, 0)) / float(max(1, total_unique))) if total_unique > 0 else 0.0
    reject_rate_pct = (100.0 * float(total_unique) / float(max(1, int(universe_size or 0)))) if universe_size else 0.0
    return {
        "total_rejected": total_unique,
        "total_rejected_events": int(len(rejected or [])),
        "reject_rate_pct": round(reject_rate_pct, 2),
        "dominant_reason": dominant_reason,
        "dominant_ratio_pct": round(dominant_ratio * 100.0, 2),
        "counts": reason_counts,
    }


def _symbol_is_scannable(symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    if not re.fullmatch(r"[A-Z]{1,5}", sym):
        return False
    return True


def _recent_priority_symbols(hub_dir: str) -> List[str]:
    out: List[str] = []

    def _add(value: Any) -> None:
        sym = str(value or "").strip().upper()
        if _symbol_is_scannable(sym) and sym not in out:
            out.append(sym)

    try:
        diag = _load_json_map(_scan_diag_path(hub_dir))
        for sym in list(diag.get("leader_symbols", []) or []):
            _add(sym)
        for sym in list(diag.get("candidate_symbols", []) or []):
            _add(sym)
        _add(diag.get("top_symbol"))
    except Exception:
        pass
    try:
        thinker = _load_json_map(_thinker_status_path(hub_dir))
        top_pick = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        _add(top_pick.get("symbol"))
        for row in list(thinker.get("leaders", []) or []):
            if isinstance(row, dict):
                _add(row.get("symbol"))
    except Exception:
        pass
    try:
        status = _load_json_map(os.path.join(hub_dir, "stocks", "alpaca_status.json"))
        for row in list(status.get("raw_positions", []) or []):
            if isinstance(row, dict):
                _add(row.get("symbol"))
    except Exception:
        pass
    return out


def _prioritize_universe_symbols(symbols: List[str], watch: List[str], hub_dir: str) -> List[str]:
    ordered_pool: List[str] = []
    seen: set[str] = set()
    for sym in list(symbols or []):
        norm = str(sym or "").strip().upper()
        if not _symbol_is_scannable(norm) or norm in seen:
            continue
        seen.add(norm)
        ordered_pool.append(norm)

    out: List[str] = []
    used: set[str] = set()

    def _push(sym: Any) -> None:
        norm = str(sym or "").strip().upper()
        if (not _symbol_is_scannable(norm)) or (norm in used):
            return
        if norm not in seen and norm not in {str(s or "").strip().upper() for s in list(watch or [])}:
            return
        used.add(norm)
        out.append(norm)

    for sym in list(watch or []):
        _push(sym)
    for sym in _recent_priority_symbols(hub_dir):
        _push(sym)
    for sym in _UNIVERSE_PRIORITY_SYMBOLS:
        _push(sym)
    for sym in DEFAULT_STOCK_UNIVERSE:
        _push(sym)
    for sym in sorted(ordered_pool):
        _push(sym)
    return out


def _load_open_position_symbols(hub_dir: str) -> List[str]:
    out: List[str] = []
    try:
        status = _load_json_map(os.path.join(hub_dir, "stocks", "alpaca_status.json"))
        for row in list(status.get("raw_positions", []) or []):
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol", "") or "").strip().upper()
            if _symbol_is_scannable(sym) and (sym not in out):
                out.append(sym)
    except Exception:
        pass
    return out


def _select_twelvedata_scan_slice(
    settings: Dict[str, Any],
    hub_dir: str,
    symbols: List[str],
    max_scan: int,
    prev_candidates: List[str],
    prev_leaders: List[str],
    prev_top_symbol: str,
) -> tuple[List[str], Dict[str, Any]]:
    ordered: List[str] = []
    seen: set[str] = set()
    for raw in list(symbols or []):
        sym = str(raw or "").strip().upper()
        if (not _symbol_is_scannable(sym)) or (sym in seen):
            continue
        seen.add(sym)
        ordered.append(sym)
    cap = max(1, int(max_scan or 1))
    if len(ordered) <= cap:
        return ordered, {
            "active": False,
            "core_size": len(ordered),
            "tail_size": 0,
            "offset": 0,
            "next_offset": 0,
            "min_rotate_slots": 0,
        }

    seed: List[str] = []
    seed.extend(_load_open_position_symbols(hub_dir))
    seed.extend(_parse_watchlist(settings))
    if str(prev_top_symbol or "").strip():
        seed.append(str(prev_top_symbol).strip().upper())
    seed.extend([str(s or "").strip().upper() for s in list(prev_leaders or [])[:12]])
    seed.extend([str(s or "").strip().upper() for s in list(prev_candidates or [])[:6]])

    pool = set(ordered)
    core: List[str] = []
    core_seen: set[str] = set()
    for raw in seed:
        sym = str(raw or "").strip().upper()
        if (not sym) or (sym in core_seen) or (sym not in pool):
            continue
        core_seen.add(sym)
        core.append(sym)

    if (not core) and ordered:
        core = [ordered[0]]
        core_seen = {ordered[0]}

    min_rotate_slots = 0
    if cap >= 6:
        min_rotate_slots = 2
    elif cap >= 3:
        min_rotate_slots = 1
    core_limit = max(0, cap - min_rotate_slots)
    if len(core) > core_limit:
        core = core[:core_limit]
        core_seen = set(core)

    tail = [sym for sym in ordered if sym not in core_seen]
    rotate_slots = max(0, cap - len(core))
    if (not tail) or rotate_slots <= 0:
        selected = (core if core else ordered)[:cap]
        return selected, {
            "active": False,
            "core_size": len(selected),
            "tail_size": 0,
            "offset": 0,
            "next_offset": 0,
            "min_rotate_slots": int(min_rotate_slots),
        }

    rotation_state = _load_json_map(_scan_rotation_path(hub_dir))
    try:
        offset = int(rotation_state.get("offset", 0) or 0)
    except Exception:
        offset = 0
    offset = offset % len(tail)
    tail_len = len(tail)

    selected_tail: List[str] = []
    idx = int(offset)
    for _ in range(min(rotate_slots, tail_len)):
        selected_tail.append(tail[idx])
        idx = (idx + 1) % tail_len
    next_offset = int(idx)
    selected = (core + selected_tail)[:cap]

    _save_json_map(
        _scan_rotation_path(hub_dir),
        {
            "ts": int(time.time()),
            "provider": "twelvedata",
            "offset": int(next_offset),
            "last_offset": int(offset),
            "core_size": int(len(core)),
            "tail_size": int(len(tail)),
            "scan_size": int(len(selected)),
            "min_rotate_slots": int(min_rotate_slots),
            "core_symbols": list(core),
            "selected_symbols": list(selected),
        },
    )
    return selected, {
        "active": True,
        "core_size": int(len(core)),
        "tail_size": int(len(tail)),
        "offset": int(offset),
        "next_offset": int(next_offset),
        "min_rotate_slots": int(min_rotate_slots),
    }


def _select_universe(settings: Dict[str, Any], hub_dir: str, api_key: str, secret: str) -> List[str]:
    mode = str(settings.get("stock_universe_mode", "all_tradable_filtered") or "all_tradable_filtered").strip().lower()
    watch = _parse_watchlist(settings)
    if mode == "watchlist":
        return watch if watch else list(DEFAULT_STOCK_UNIVERSE)
    if mode == "core":
        return watch if watch else list(DEFAULT_STOCK_UNIVERSE)
    if mode != "all_tradable_filtered":
        return watch if watch else list(DEFAULT_STOCK_UNIVERSE)
    if not _rollout_at_least(settings, "scan_expanded"):
        return watch if watch else list(DEFAULT_STOCK_UNIVERSE)

    cached = _load_universe_cache(hub_dir, ttl_s=1800)
    if cached:
        return _prioritize_universe_symbols(cached, watch, hub_dir)

    client = AlpacaBrokerClient(
        api_key_id=api_key,
        secret_key=secret,
        base_url=str(settings.get("alpaca_base_url", "https://paper-api.alpaca.markets") or ""),
        data_url=str(settings.get("alpaca_data_url", "https://data.alpaca.markets") or ""),
    )
    assets = client.list_tradable_assets()
    symbols: List[str] = []
    for row in assets:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("tradable", False)):
            continue
        if str(row.get("status", "")).lower() != "active":
            continue
        if str(row.get("class", "")).lower() not in ("us_equity", "us_equities", ""):
            continue
        exch = str(row.get("exchange", "") or "").strip().upper()
        if exch and (exch not in _SCANNABLE_EXCHANGES):
            continue
        marginable = row.get("marginable", None)
        if marginable is not None and (not bool(marginable)):
            continue
        fractionable = row.get("fractionable", None)
        if fractionable is not None and (not bool(fractionable)):
            continue
        sym = str(row.get("symbol", "") or "").strip().upper()
        if not _symbol_is_scannable(sym):
            continue
        if sym and sym not in symbols:
            symbols.append(sym)
    symbols = _prioritize_universe_symbols(symbols, watch, hub_dir)
    if not symbols:
        return watch if watch else list(DEFAULT_STOCK_UNIVERSE)
    _save_universe_cache(hub_dir, symbols)
    return symbols


def _parse_feed_order(settings: Dict[str, Any]) -> List[str]:
    raw = str(settings.get("stock_data_feeds", "iex,sip") or "iex,sip")
    out: List[str] = []
    for tok in raw.replace(";", ",").split(","):
        feed = str(tok or "").strip().lower()
        if feed in {"sip", "iex"} and feed not in out:
            out.append(feed)
    if not out:
        out = ["iex", "sip"]
    return out


def run_scan(settings: Dict[str, Any], hub_dir: str) -> Dict[str, Any]:
    prev_diag = _load_json_map(_scan_diag_path(hub_dir))
    prev_candidates = _norm_id_list(prev_diag.get("candidate_symbols", []))
    prev_leaders = _norm_id_list(prev_diag.get("leader_symbols", []))
    prev_top_symbol = str(prev_diag.get("top_symbol", "") or "").strip().upper()
    if not prev_top_symbol:
        prev_status = _load_json_map(_thinker_status_path(hub_dir))
        prev_top = prev_status.get("top_pick", {}) if isinstance(prev_status.get("top_pick", {}), dict) else {}
        prev_top_symbol = str(prev_top.get("symbol", "") or "").strip().upper()
    provider = _stock_data_provider(settings)
    api_key, secret = get_alpaca_creds(settings, base_dir=BASE_DIR)
    base_url = str(settings.get("alpaca_data_url", settings.get("alpaca_base_url", "https://data.alpaca.markets")) or "").strip().rstrip("/")
    ts_now = int(time.time())
    if not api_key or not secret:
        return {
            "state": "NOT CONFIGURED",
            "ai_state": "Credentials missing",
            "msg": "Add Alpaca keys in Settings",
            "universe": list(DEFAULT_STOCK_UNIVERSE),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
            "market_open": _market_open_now(),
        }
    td_api_key = ""
    if provider == "twelvedata":
        td_api_key = get_twelvedata_api_key(settings, base_dir=BASE_DIR)
        if not td_api_key:
            return {
                "state": "NOT CONFIGURED",
                "ai_state": "Credentials missing",
                "msg": "Add Twelve Data API key in Settings",
                "universe": list(DEFAULT_STOCK_UNIVERSE),
                "leaders": [],
                "all_scores": [],
                "top_chart": [],
                "top_chart_map": {},
                "updated_at": ts_now,
                "market_open": _market_open_now(),
            }

    clock_status = _market_clock_status(settings, api_key, secret)
    market_open = bool(clock_status.get("market_open", _market_open_now()))
    pause_state = _load_json_map(_scan_pause_path(hub_dir))
    if int(pause_state.get("pause_until_ts", 0) or 0) > 0:
        _save_json_map(
            _scan_pause_path(hub_dir),
            {
                "ts": ts_now,
                "pause_until_ts": 0,
                "reason": "scan_pause_disabled",
                "market_open": bool(market_open),
            },
        )
    rotation_info: Dict[str, Any] = {
        "active": False,
        "core_size": 0,
        "tail_size": 0,
        "offset": 0,
        "next_offset": 0,
        "min_rotate_slots": 0,
    }
    if not market_open:
        next_open = str(clock_status.get("next_open", "") or "").strip()
        msg = "Market closed; stocks scanner paused"
        if next_open:
            msg = f"{msg} until {next_open}"
        closed_universe = list(prev_candidates or prev_diag.get("candidate_symbols", []) or DEFAULT_STOCK_UNIVERSE)
        fallback = _cached_scan_fallback(
            hub_dir,
            ts_now,
            msg,
            universe=list(closed_universe),
            market_open=False,
        )
        if fallback:
            out = dict(fallback)
            out.update(
                {
                    "state": "READY",
                    "ai_state": "Market closed (cached)",
                    "msg": str(out.get("msg", msg) or msg),
                    "updated_at": ts_now,
                    "market_open": False,
                    "market_clock": dict(clock_status),
                    "hints": [
                        f"Stocks market is closed; showing cached scan until next open{f' ({next_open})' if next_open else ''}.",
                        "Scanning/data pulls are paused outside market hours.",
                    ],
                    "scan_rotation": dict(rotation_info),
                    "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                }
            )
        else:
            out = {
                "state": "READY",
                "ai_state": "Market closed",
                "msg": msg,
                "universe": list(closed_universe),
                "leaders": [],
                "all_scores": [],
                "top_pick": {},
                "top_chart": [],
                "top_chart_map": {},
                "top_chart_source": "",
                "updated_at": ts_now,
                "market_open": False,
                "market_clock": dict(clock_status),
                "rejected": [],
                "reject_summary": {"total_rejected_events": 0, "reject_rate_pct": 0.0, "dominant_reason": "market_closed"},
                "feed_order": [],
                "hints": [
                    f"Stocks market is closed; waiting for next open{f' ({next_open})' if next_open else ''}.",
                    "No stock scans/data pulls are performed while closed.",
                ],
                "candidate_churn_pct": 0.0,
                "leader_churn_pct": 0.0,
                "leader_mode": "none",
                "leader_stability_applied": False,
                "leader_stability_prev_symbol": str(prev_top_symbol),
                "scan_rotation": dict(rotation_info),
                "window_policy": {"active": False, "window": "OFF", "score_mult": 1.0, "minutes": 0},
                "window_policy_hits": 0,
                "universe_quality": {},
                "fallback_cached": False,
                "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": False},
                "pdt_note": "Paper mode can still simulate PDT protections; live day-trading may be limited under $25k.",
            }
        leaders_out = [row for row in list(out.get("leaders", []) or []) if isinstance(row, dict)]
        top_pick = out.get("top_pick", {}) if isinstance(out.get("top_pick", {}), dict) else {}
        opening_plan = _persist_opening_plan(
            settings,
            hub_dir,
            leaders=list(leaders_out),
            all_scores=[row for row in list(out.get("all_scores", []) or []) if isinstance(row, dict)],
            adaptive_threshold=float(
                out.get(
                    "adaptive_threshold",
                    settings.get("stock_score_threshold", 0.2),
                )
                or settings.get("stock_score_threshold", 0.2)
                or 0.2
            ),
            ts_now=ts_now,
            market_clock=clock_status,
            market_open=False,
            msg=str(out.get("msg", msg) or msg),
            fallback_cached=bool(out.get("fallback_cached", False)),
        )
        out["opening_plan"] = dict(opening_plan)
        _save_scan_diagnostics(
            hub_dir,
            {
                "ts": ts_now,
                "state": "READY",
                "mode": "closed_cache",
                "market_open": False,
                "universe_total": int(len(list(out.get("universe", []) or []))),
                "candidates_total": int(len(list(out.get("universe", []) or []))),
                "scores_total": int(len(list(out.get("all_scores", []) or []))),
                "leaders_total": int(len(leaders_out)),
                "top_symbol": str((top_pick.get("symbol", "") if isinstance(top_pick, dict) else "") or ""),
                "top_score": float((top_pick.get("score", 0.0) if isinstance(top_pick, dict) else 0.0) or 0.0),
                "msg": str(out.get("msg", msg) or msg),
                "reject_summary": dict(out.get("reject_summary", {}) if isinstance(out.get("reject_summary", {}), dict) else {}),
                "feed_order": list(out.get("feed_order", []) or []),
                "candidate_symbols": list(out.get("universe", []) or []),
                "leader_symbols": [str((row or {}).get("symbol", "") or "").strip().upper() for row in leaders_out],
                "leader_mode": str(out.get("leader_mode", "none") or "none"),
                "leader_stability_applied": bool(out.get("leader_stability_applied", False)),
                "leader_stability_prev_symbol": str(out.get("leader_stability_prev_symbol", prev_top_symbol) or prev_top_symbol),
                "scan_rotation": dict(rotation_info),
                "candidate_churn_pct": float(out.get("candidate_churn_pct", 0.0) or 0.0),
                "leader_churn_pct": float(out.get("leader_churn_pct", 0.0) or 0.0),
                "quality_summary": "Market closed; scanner paused and cached stock scan is displayed.",
            },
        )
        return out
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=10)
    start_iso = _iso_utc(start_utc)
    end_iso = _iso_utc(now_utc)
    daily_start_iso = _iso_utc(now_utc - timedelta(days=220))
    daily_end_iso = end_iso
    try:
        universe = _select_universe(settings, hub_dir, api_key, secret)
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        fallback = _cached_scan_fallback(
            hub_dir,
            ts_now,
            err_msg,
            universe=list(prev_candidates or DEFAULT_STOCK_UNIVERSE),
            market_open=market_open,
        )
        if fallback:
            return fallback
        return {
            "state": "ERROR",
            "ai_state": "Scan failed",
            "msg": err_msg,
            "universe": list(prev_candidates or DEFAULT_STOCK_UNIVERSE),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
            "market_open": market_open,
            "market_clock": dict(clock_status),
        }
    max_scan = max(8, int(float(settings.get("stock_scan_max_symbols", 120) or 120)))
    if provider == "twelvedata":
        td_cap = max(1, int(float(settings.get("twelvedata_scan_symbol_cap", 8) or 8)))
        max_scan = min(max_scan, td_cap)
        universe, rotation_info = _select_twelvedata_scan_slice(
            settings=settings,
            hub_dir=hub_dir,
            symbols=list(universe or []),
            max_scan=max_scan,
            prev_candidates=list(prev_candidates or []),
            prev_leaders=list(prev_leaders or []),
            prev_top_symbol=str(prev_top_symbol or ""),
        )
    else:
        universe = universe[:max_scan]

    if provider == "twelvedata":
        prev_updated = int(float(prev_diag.get("updated_at", prev_diag.get("ts", 0)) or 0))
        limits = _twelvedata_scan_limits(settings, len(universe))
        if prev_updated > 0:
            age_s = max(0, int(ts_now - prev_updated))
            if age_s < int(limits.get("min_interval_s", 60.0)):
                wait_s = max(1, int(float(limits.get("min_interval_s", 60.0)) - age_s))
                msg = f"Twelve Data rate guard: next scan in {wait_s}s"
                fallback = _cached_scan_fallback(
                    hub_dir,
                    ts_now,
                    msg,
                    universe=list(universe),
                    market_open=bool(market_open),
                )
                if fallback:
                    return fallback
                return {
                    "state": "READY",
                    "ai_state": "Scan throttled",
                    "msg": msg,
                    "universe": list(universe),
                    "leaders": [],
                    "all_scores": [],
                    "top_chart": [],
                    "top_chart_map": {},
                    "updated_at": ts_now,
                    "market_open": bool(market_open),
                    "rejected": [],
                    "reject_summary": {"total_rejected": 0, "reject_rate_pct": 0.0, "dominant_reason": "rate_guard"},
                    "hints": ["Scanner throttled to respect Twelve Data rate limits."],
                    "candidate_churn_pct": 0.0,
                    "leader_churn_pct": 0.0,
                    "leader_mode": "none",
                    "leader_stability_applied": False,
                    "leader_stability_prev_symbol": str(prev_top_symbol),
                    "scan_rotation": dict(rotation_info),
                    "universe_quality": {},
                    "health": {"data_ok": False, "broker_ok": True, "orders_ok": True, "drift_warning": True},
                    "pdt_note": "Paper mode can still simulate PDT protections; live day-trading may be limited under $25k.",
                }

    data_url = str(settings.get("alpaca_data_url", "https://data.alpaca.markets") or "").strip().rstrip("/")
    if (not data_url) or ("paper-api.alpaca.markets" in data_url):
        data_url = "https://data.alpaca.markets"
    base_url = data_url

    feed_health = _load_json_map(_feed_health_path(hub_dir))
    feed_errors: Dict[str, str] = {}
    snap: Dict[str, Dict[str, float]] = {}
    snap_last_exc: Exception | None = None
    feed_order: List[str] = []
    td_bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    client: AlpacaBrokerClient | None = None

    if provider == "twelvedata":
        feed_order = ["twelvedata"]
        td_client = TwelveDataClient(
            api_key=td_api_key,
            base_url=str(settings.get("twelvedata_base_url", "https://api.twelvedata.com") or "https://api.twelvedata.com"),
        )
        td_outputsize = max(96, int(float(settings.get("stock_min_bars_required", 24) or 24) * 2))
        try:
            td_bars_by_symbol = td_client.get_time_series_batch(
                universe,
                interval="1h",
                outputsize=td_outputsize,
            )
            snap = _twelvedata_snap_from_bars(td_bars_by_symbol)
            feed_health = _update_feed_health(feed_health, "twelvedata", ok=bool(snap), bars_total=len(td_bars_by_symbol))
            if not snap:
                feed_errors["twelvedata"] = "empty response"
        except Exception as exc:
            snap_last_exc = exc
            feed_errors["twelvedata"] = f"{type(exc).__name__}: {exc}"
            feed_health = _update_feed_health(feed_health, "twelvedata", ok=False, bars_total=0)
    else:
        client = AlpacaBrokerClient(
            api_key_id=api_key,
            secret_key=secret,
            base_url=str(settings.get("alpaca_base_url", "https://paper-api.alpaca.markets") or ""),
            data_url=data_url,
        )
        feed_order_cfg = _parse_feed_order(settings)
        feed_order = _adaptive_feed_order(feed_order_cfg, feed_health)
        for feed in feed_order:
            try:
                snap = client.get_snapshot_details(universe, feed=feed)
                feed_health = _update_feed_health(feed_health, feed, ok=bool(snap), bars_total=len(snap))
                if snap:
                    break
            except Exception as exc:
                snap_last_exc = exc
                try:
                    feed_errors[str(feed or "").strip().lower()] = f"{type(exc).__name__}: {exc}"
                except Exception:
                    pass
                feed_health = _update_feed_health(feed_health, feed, ok=False, bars_total=0)
                continue
    if (not snap) and (snap_last_exc is not None):
        _append_jsonl(
            _rankings_path(hub_dir),
            {
                "ts": int(ts_now),
                "state": "WARN",
                "reason": f"snapshot degraded: {type(snap_last_exc).__name__}: {snap_last_exc}",
                "universe_total": len(universe),
            },
        )
    allow_missing_liquidity = False
    liquidity_missing_ratio_pct = 0.0
    liquidity_relief_min_universe = 25
    liquidity_relief_missing_ratio_pct = 65.0
    try:
        if universe:
            missing = sum(
                1
                for sym in universe
                if _float((snap.get(sym, {}) or {}).get("dollar_vol", 0.0), 0.0) <= 0.0
            )
            liquidity_missing_ratio_pct = (float(missing) / float(len(universe))) * 100.0
            # Some feeds intermittently omit dollar-volume for most symbols while
            # price/spread fields are still valid. In that case, keep scan flow alive
            # for broad universes and let downstream bars/quality gates decide.
            allow_missing_liquidity = bool(
                len(universe) >= int(liquidity_relief_min_universe)
                and liquidity_missing_ratio_pct >= float(liquidity_relief_missing_ratio_pct)
            )
    except Exception:
        allow_missing_liquidity = False
    min_price = max(0.0, float(settings.get("stock_min_price", 2.0) or 2.0))
    max_price = max(min_price, float(settings.get("stock_max_price", 500.0) or 500.0))
    min_dollar_vol = max(0.0, float(settings.get("stock_min_dollar_volume", 2_000_000.0) or 2_000_000.0))
    max_spread_bps = max(0.0, float(settings.get("stock_max_spread_bps", 40.0) or 40.0))
    use_daily_when_closed = _setting_bool(settings, "stock_scan_use_daily_when_closed", True)
    closed_max_stale_hours = max(1.0, float(settings.get("stock_closed_max_stale_hours", 96.0) or 96.0))
    min_bars_required = max(8, int(float(settings.get("stock_min_bars_required", 24) or 24)))

    candidates: List[str] = []
    rejected: List[Dict[str, Any]] = []
    rejected_seen: set[tuple[str, str, str]] = set()

    def add_rejected(row: Dict[str, Any]) -> None:
        sym = str(row.get("symbol", "") or "").strip().upper()
        reason = str(row.get("reason", "") or "").strip().lower()
        source = str(row.get("source", "") or "").strip().lower()
        key = (sym, reason, source)
        if key in rejected_seen:
            return
        rejected_seen.add(key)
        rejected.append(row)

    warm_queue = _load_warmup_queue(hub_dir)
    cooldown_state = _load_json_map(_symbol_cooldown_path(hub_dir))
    cooldown_map = cooldown_state.get("symbols", {}) if isinstance(cooldown_state.get("symbols", {}), dict) else {}
    market_open = bool(market_open)
    now_ts = int(time.time())
    window_policy = _stock_scan_window_policy(settings) if market_open else {"active": False, "window": "OFF", "score_mult": 1.0, "minutes": 0}
    window_policy_hits = 0
    cooldown_map = _prune_cooldown_map(cooldown_map, now_ts, settings=settings)

    # Warmup prefetch queue: pull extra daily history for recently short symbols.
    if warm_queue and provider != "twelvedata" and client is not None:
        for sym in list(warm_queue.keys())[:50]:
            best_count = 0
            best_source = str((warm_queue.get(sym, {}) or {}).get("source", "") or "")
            for feed in (feed_order or ["sip", "iex"]):
                try:
                    wb = client.get_stock_bars(
                        sym,
                        timeframe="1Day",
                        limit=120,
                        feed=feed,
                        start_iso=daily_start_iso,
                        end_iso=daily_end_iso,
                    )
                    n = int(len(wb or []))
                    if n > best_count:
                        best_count = n
                        best_source = f"symbol_1d:{feed}"
                    if n >= min_bars_required:
                        break
                except Exception:
                    continue
            try:
                warm_queue[sym]["bars_count"] = int(best_count)
                warm_queue[sym]["source"] = str(best_source or warm_queue[sym].get("source", ""))
                warm_queue[sym]["last_seen"] = now_ts
                if int(warm_queue[sym]["bars_count"]) >= min_bars_required:
                    warm_queue.pop(sym, None)
                else:
                    prev_retry = int(warm_queue[sym].get("retry_s", 60) or 60)
                    warm_queue[sym]["retry_s"] = min(900, max(60, prev_retry * 2))
                    warm_queue[sym]["retry_after"] = now_ts + int(warm_queue[sym]["retry_s"])
            except Exception:
                pass

    cooldown_enforced = bool(market_open)
    for sym in universe:
        c_row = cooldown_map.get(sym, {}) if isinstance(cooldown_map.get(sym, {}), dict) else {}
        cooldown_reason = str(c_row.get("reason", "") or "").strip().lower()
        if cooldown_enforced and int(c_row.get("until", 0) or 0) > now_ts:
            # Data-quality failures are often transient (feed timing/shape noise).
            # Keep probing each cycle instead of hard-blocking symbols for minutes.
            if cooldown_reason == "data_quality":
                pass
            else:
                add_rejected(
                    {
                        "symbol": sym,
                        "reason": "cooldown",
                        "cooldown_until": int(c_row.get("until", 0) or 0),
                        "source": "cooldown",
                    }
                )
                continue
        d = snap.get(sym, {}) if isinstance(snap, dict) else {}
        px = _float(d.get("mid", 0.0), 0.0)
        spread_bps = _float(d.get("spread_bps", 0.0), 0.0)
        dollar_vol = _float(d.get("dollar_vol", 0.0), 0.0)
        warm = warm_queue.get(sym, {}) or {}
        retry_after = int(warm.get("retry_after", 0) or 0)
        if (not market_open) and (retry_after > now_ts) and int(warm.get("bars_count", 0) or 0) < min_bars_required:
            add_rejected(
                {
                    "symbol": sym,
                    "reason": "warmup_pending",
                    "bars_count": int(warm.get("bars_count", 0) or 0),
                    "source": str(warm.get("source", "") or ""),
                }
            )
            continue
        # Keep scanner active off-hours. Market-hours gating is enforced in trader preflight.
        # When closed, we prefer daily bars for better historical coverage.
        if px > 0.0 and (px < min_price or px > max_price):
            add_rejected({"symbol": sym, "reason": "price_band", "price": px})
            if cooldown_enforced:
                _apply_symbol_cooldown(cooldown_map, sym, "price_band", settings, now_ts)
            continue
        if market_open and max_spread_bps > 0.0 and spread_bps > 0.0 and spread_bps > max_spread_bps:
            add_rejected({"symbol": sym, "reason": "spread", "spread_bps": spread_bps})
            if cooldown_enforced:
                _apply_symbol_cooldown(cooldown_map, sym, "spread", settings, now_ts)
            continue
        if market_open and dollar_vol <= 0.0 and (not allow_missing_liquidity):
            add_rejected({"symbol": sym, "reason": "liquidity", "dollar_vol": dollar_vol})
            if cooldown_enforced:
                _apply_symbol_cooldown(cooldown_map, sym, "liquidity", settings, now_ts)
            continue
        if dollar_vol > 0.0 and dollar_vol < min_dollar_vol:
            add_rejected({"symbol": sym, "reason": "liquidity", "dollar_vol": dollar_vol})
            if cooldown_enforced:
                _apply_symbol_cooldown(cooldown_map, sym, "liquidity", settings, now_ts)
            continue
        candidates.append(sym)
    scored: List[Dict[str, Any]] = []
    last_exc: Exception | None = None
    bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    best_bars_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    min_valid_ratio = max(0.0, min(1.0, float(settings.get("stock_min_valid_bars_ratio", 0.70) or 0.70)))
    max_stale_hours = max(0.5, float(settings.get("stock_max_stale_hours", 6.0) or 6.0))
    max_stale_hours_effective = max_stale_hours
    if (not market_open) and use_daily_when_closed:
        max_stale_hours_effective = max(max_stale_hours, closed_max_stale_hours)
    for feed in feed_order:
        try:
            if provider == "twelvedata":
                bars_by_symbol = dict(td_bars_by_symbol or {})
            elif market_open or (not use_daily_when_closed):
                bars_by_symbol = _fetch_bars_for_symbols(base_url, headers, candidates, start_iso, end_iso, feed)
            else:
                bars_by_symbol = {}
            scored = []
            for symbol in candidates:
                symbol_bars: List[Dict[str, Any]] = []
                data_source = ""
                open_daily_fallback = False
                if market_open or (not use_daily_when_closed) or provider == "twelvedata":
                    symbol_bars = list(bars_by_symbol.get(symbol, []) or [])
                    data_source = "batch_1h" if provider != "twelvedata" else "twelvedata_1h"
                else:
                    if client is not None:
                        try:
                            symbol_bars = client.get_stock_bars(
                                symbol,
                                timeframe="1Day",
                                limit=120,
                                feed=feed,
                                start_iso=daily_start_iso,
                                end_iso=daily_end_iso,
                            )
                            data_source = "symbol_1d"
                        except Exception:
                            symbol_bars = []
                            data_source = "symbol_1d"
                # Fallback path: if batch bars are sparse/missing, try symbol endpoint directly.
                if len(symbol_bars) < min_bars_required:
                    if provider != "twelvedata" and client is not None:
                        try:
                            if market_open:
                                symbol_bars = client.get_stock_bars(
                                    symbol,
                                    timeframe="1Hour",
                                    limit=160,
                                    feed=feed,
                                    start_iso=start_iso,
                                    end_iso=end_iso,
                                )
                                data_source = "symbol_1h"
                            else:
                                # Closed-session fallback: prefer richer intraday window if daily bars are sparse.
                                symbol_bars = client.get_stock_bars(
                                    symbol,
                                    timeframe="1Hour",
                                    limit=240,
                                    feed=feed,
                                    start_iso=start_iso,
                                    end_iso=end_iso,
                                )
                                data_source = "symbol_1h"
                        except Exception:
                            symbol_bars = list(symbol_bars or [])
                if len(symbol_bars) < min_bars_required:
                    # Last resort for thin symbols / feed limitations: daily bars.
                    if provider != "twelvedata" and client is not None:
                        try:
                            if market_open:
                                symbol_bars = client.get_stock_bars(
                                    symbol,
                                    timeframe="1Day",
                                    limit=120,
                                    feed=feed,
                                    start_iso=daily_start_iso,
                                    end_iso=daily_end_iso,
                                )
                                data_source = "symbol_1d_open"
                                open_daily_fallback = True
                            elif data_source != "symbol_1d":
                                symbol_bars = client.get_stock_bars(
                                    symbol,
                                    timeframe="1Day",
                                    limit=120,
                                    feed=feed,
                                    start_iso=daily_start_iso,
                                    end_iso=daily_end_iso,
                                )
                                data_source = "symbol_1d"
                        except Exception:
                            symbol_bars = list(symbol_bars or [])
                bars_count = int(len(symbol_bars or []))
                if bars_count < min_bars_required:
                    retry_s = min(900, max(60, int((warm_queue.get(symbol, {}) or {}).get("retry_s", 60) or 60) * 2))
                    warm_queue[symbol] = {
                        "last_seen": now_ts,
                        "bars_count": bars_count,
                        "reason": "insufficient_bars",
                        "source": data_source,
                        "retry_s": retry_s,
                        "retry_after": now_ts + retry_s,
                    }
                    add_rejected(
                        {
                            "symbol": symbol,
                            "reason": "insufficient_bars",
                            "bars_count": bars_count,
                            "source": f"{data_source}:{feed}",
                            "min_bars_required": min_bars_required,
                            "requested_start": daily_start_iso if data_source == "symbol_1d" else start_iso,
                            "requested_end": daily_end_iso if data_source == "symbol_1d" else end_iso,
                        }
                    )
                    if cooldown_enforced:
                        _apply_symbol_cooldown(cooldown_map, symbol, "insufficient_bars", settings, now_ts)
                    continue
                spread_bps = _float((snap.get(symbol, {}) or {}).get("spread_bps", 0.0), 0.0)
                row = _score_bars(symbol, symbol_bars, spread_bps=spread_bps)
                row["spread_bps"] = round(spread_bps, 4)
                row["dollar_vol"] = round(_float((snap.get(symbol, {}) or {}).get("dollar_vol", 0.0), 0.0), 2)
                row["bars_count"] = bars_count
                row["data_source"] = f"{data_source}:{feed}"
                if open_daily_fallback:
                    row["open_daily_fallback"] = True
                q = _bar_quality(symbol_bars)
                row["valid_ratio"] = round(float(q.get("valid_ratio", 0.0)), 4)
                row["stale_hours"] = round(float(q.get("stale_hours", 9999.0)), 3)
                row["data_quality_ok"] = bool((row["valid_ratio"] >= min_valid_ratio) and (row["stale_hours"] <= max_stale_hours_effective))
                if bool(window_policy.get("active", False)) and str(row.get("side", "watch")).lower() == "long":
                    raw_score = float(row.get("score", 0.0) or 0.0)
                    mult = float(window_policy.get("score_mult", 1.0) or 1.0)
                    row["score_raw"] = round(raw_score, 6)
                    row["score"] = round(raw_score * mult, 6)
                    row["window_policy"] = str(window_policy.get("window", "NONE") or "NONE")
                    row["window_score_mult"] = round(mult, 4)
                    row["window_minutes"] = int(window_policy.get("minutes", 0) or 0)
                    _append_reason_parts(
                        row,
                        logic=f"{str(row['window_policy']).title()} session dampener reduced entry conviction",
                        data=f"{str(row['window_policy']).lower()} window x{mult:.2f}",
                    )
                    window_policy_hits += 1
                if not bool(row.get("data_quality_ok", True)):
                    add_rejected(
                        {
                            "symbol": symbol,
                            "reason": "data_quality",
                            "valid_ratio": row.get("valid_ratio"),
                            "stale_hours": row.get("stale_hours"),
                            "bars_count": bars_count,
                            "source": f"{data_source}:{feed}",
                        }
                    )
                    if cooldown_enforced:
                        _apply_symbol_cooldown(cooldown_map, symbol, "data_quality", settings, now_ts)
                    row["side"] = "watch"
                    continue
                if float(row.get("score", -9999.0) or -9999.0) <= -9999.0:
                    add_rejected(
                        {
                            "symbol": symbol,
                            "reason": "insufficient_bars",
                            "bars_count": bars_count,
                            "source": f"{data_source}:{feed}",
                        }
                    )
                    if cooldown_enforced:
                        _apply_symbol_cooldown(cooldown_map, symbol, "insufficient_bars", settings, now_ts)
                    continue
                best_bars_by_symbol[symbol] = list(symbol_bars or [])
                scored.append(row)
            _apply_stock_mtf_confirmation(scored, client, feed, settings)
            bars_total = 0
            try:
                bars_total = int(sum(len(list(v or [])) for v in best_bars_by_symbol.values()))
            except Exception:
                bars_total = 0
            feed_health = _update_feed_health(feed_health, feed, ok=bool(scored), bars_total=bars_total)
            if any(float(row.get("score", -9999.0)) > -9999.0 for row in scored):
                break
        except Exception as exc:
            last_exc = exc
            try:
                if str(feed or "").strip().lower() not in feed_errors:
                    feed_errors[str(feed or "").strip().lower()] = f"{type(exc).__name__}: {exc}"
            except Exception:
                pass
            feed_health = _update_feed_health(feed_health, feed, ok=False, bars_total=0)
            scored = []

    if not scored:
        # Keep thinker responsive even when bars are sparse/off-hours; execution still gates entries.
        feed_issue_parts: List[str] = []
        for feed_name, raw_err in list(feed_errors.items())[:3]:
            err_txt = str(raw_err or "").strip()
            if (not err_txt) and (last_exc is not None):
                err_txt = f"{type(last_exc).__name__}: {last_exc}"
            low = err_txt.lower()
            if "error code: 1010" in low:
                err_txt = "403 Forbidden (Cloudflare 1010: request blocked)"
            elif ("http error 403" in low) or ("forbidden" in low):
                if str(feed_name or "").strip().lower() == "twelvedata":
                    err_txt = "403 Forbidden (provider access denied)"
                else:
                    err_txt = "403 Forbidden (feed entitlement)"
            elif len(err_txt) > 92:
                err_txt = err_txt[:89] + "..."
            feed_issue_parts.append(f"{feed_name}:{err_txt}")
        if (not candidates) and rejected:
            msg = "No symbols passed stock marketability prefilters"
        elif rejected:
            msg = "No viable symbols after data-quality gates"
        else:
            msg = (f"{type(last_exc).__name__}: {last_exc}" if last_exc else "No viable symbols after data-quality gates")
        if feed_issue_parts:
            msg = f"{msg} | feed issues: {', '.join(feed_issue_parts)}"
        _append_jsonl(
            _rankings_path(hub_dir),
            {
                "ts": ts_now,
                "state": "READY",
                "market_open": market_open,
                "mode": ("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
                "reason": msg,
                "universe_total": len(universe),
                "candidates": len(candidates),
                "rejected": rejected[:100],
                "top": [],
            },
        )
        _save_warmup_queue(hub_dir, warm_queue)
        _save_json_map(_feed_health_path(hub_dir), feed_health)
        _save_json_map(_symbol_cooldown_path(hub_dir), {"ts": int(time.time()), "symbols": cooldown_map})
        reject_summary = _summarize_rejections(rejected, len(universe))
        candidate_churn_pct = turnover_pct(prev_candidates, candidates)
        leader_churn_pct = turnover_pct(prev_leaders, [])
        quality_report = build_universe_quality_report(
            market="stocks",
            ts=int(ts_now),
            mode=("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
            universe_total=int(len(universe)),
            candidates_total=int(len(candidates)),
            scores_total=0,
            leaders_total=0,
            reject_summary=dict(reject_summary),
            rejected_rows=list(rejected),
            scored_rows=[],
            candidate_churn_pct=float(candidate_churn_pct),
            leader_churn_pct=float(leader_churn_pct),
        )
        _save_json_map(_quality_report_path(hub_dir), quality_report)
        reason_counts = dict(reject_summary.get("counts", {}) or {})
        dominant_reason = str(reject_summary.get("dominant_reason", "") or "")
        dominant_ratio = float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0) / 100.0
        reject_rate_pct = effective_reject_pressure(
            reject_summary.get("reject_rate_pct", 0.0),
            dominant_reason=dominant_reason,
            dominant_ratio_pct=float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0),
            leaders_total=0,
            scores_total=0,
        )
        reject_warn_pct = max(10.0, float(settings.get("stock_reject_drift_warn_pct", 65.0) or 65.0))
        drift_warning = bool((reject_rate_pct >= reject_warn_pct) and (dominant_ratio >= 0.60))
        _save_scan_diagnostics(
            hub_dir,
            {
                "ts": ts_now,
                "state": "READY",
                "mode": ("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
                "market_open": bool(market_open),
                "universe_total": int(len(universe)),
                "candidates_total": int(len(candidates)),
                "scores_total": 0,
                "leaders_total": 0,
                "top_symbol": "",
                "top_score": 0.0,
                "msg": str(msg),
                "reject_summary": dict(reject_summary),
                "feed_order": list(feed_order),
                "feed_health": dict((feed_health.get("feeds", {}) if isinstance(feed_health.get("feeds", {}), dict) else {})),
                "cooldown_active": int(sum(1 for v in (cooldown_map or {}).values() if int((v or {}).get("until", 0) or 0) > now_ts)),
                "window_policy": dict(window_policy),
                "window_policy_hits": int(window_policy_hits),
                "candidate_symbols": list(candidates),
                "leader_symbols": [],
                "leader_mode": "none",
                "leader_stability_applied": False,
                "leader_stability_prev_symbol": str(prev_top_symbol),
                "scan_rotation": dict(rotation_info),
                "candidate_churn_pct": float(candidate_churn_pct),
                "leader_churn_pct": float(leader_churn_pct),
                "quality_summary": str(quality_report.get("summary", "") or ""),
                "liquidity_missing_ratio_pct": round(float(liquidity_missing_ratio_pct), 3),
                "liquidity_missing_allowed": bool(allow_missing_liquidity),
            },
        )
        hints = _market_hints_from_rejects(
            {
                "counts": reason_counts,
                "reject_rate_pct": round(reject_rate_pct, 2),
                "dominant_reason": dominant_reason,
            }
        )
        if any("403 forbidden" in str(x).lower() for x in feed_issue_parts):
            hints.insert(0, "Feed entitlement warning: SIP can return 403 in paper mode; IEX fallback remains active.")
        for h in quality_hints(quality_report):
            if h not in hints:
                hints.append(h)
        if not market_open:
            hints.append("Off-hours scan mode: spread/cooldown gates relaxed until market open.")
        if bool(rotation_info.get("active", False)):
            hints.append(
                "Stock universe rotation active: "
                f"sticky core {int(rotation_info.get('core_size', 0))}, "
                f"rotating tail {int(rotation_info.get('tail_size', 0))}."
            )
        opening_plan = _persist_opening_plan(
            settings,
            hub_dir,
            leaders=[],
            all_scores=[],
            adaptive_threshold=float(settings.get("stock_score_threshold", 0.2) or 0.2),
            ts_now=ts_now,
            market_clock=clock_status,
            market_open=bool(market_open),
            msg=str(msg),
            fallback_cached=False,
        )
        fallback_allowed = bool(feed_issue_parts or (last_exc is not None))
        fallback = None
        if fallback_allowed:
            fallback = _cached_scan_fallback(
                hub_dir,
                ts_now,
                msg,
                universe=list(candidates),
                market_open=bool(market_open),
            )
        if fallback:
            fallback["opening_plan"] = dict(opening_plan)
            _save_scan_diagnostics(
                hub_dir,
                {
                    "ts": ts_now,
                    "state": "READY",
                    "mode": ("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
                    "market_open": bool(market_open),
                    "universe_total": int(len(list(fallback.get("universe", []) or []))),
                    "candidates_total": int(len(list(fallback.get("universe", []) or []))),
                    "scores_total": int(len(list(fallback.get("all_scores", []) or []))),
                    "leaders_total": int(len(list(fallback.get("leaders", []) or []))),
                    "top_symbol": str(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("symbol", "")) or ""),
                    "top_score": float(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("score", 0.0)) or 0.0),
                    "msg": str(fallback.get("msg", "") or msg),
                    "reject_summary": (dict(fallback.get("reject_summary", {})) if isinstance(fallback.get("reject_summary", {}), dict) else {}),
                    "feed_order": list(feed_order),
                    "feed_health": dict((feed_health.get("feeds", {}) if isinstance(feed_health.get("feeds", {}), dict) else {})),
                    "cooldown_active": int(sum(1 for v in (cooldown_map or {}).values() if int((v or {}).get("until", 0) or 0) > now_ts)),
                    "window_policy": dict(window_policy),
                    "window_policy_hits": int(window_policy_hits),
                    "candidate_symbols": list(candidates),
                    "leader_symbols": [str((row or {}).get("symbol", "") or "").strip().upper() for row in list(fallback.get("leaders", []) or []) if isinstance(row, dict)],
                    "leader_mode": str(fallback.get("leader_mode", "cached") or "cached"),
                    "leader_stability_applied": bool(fallback.get("leader_stability_applied", False)),
                    "leader_stability_prev_symbol": str(fallback.get("leader_stability_prev_symbol", prev_top_symbol) or prev_top_symbol),
                    "scan_rotation": dict(rotation_info),
                    "candidate_churn_pct": float(fallback.get("candidate_churn_pct", candidate_churn_pct) or candidate_churn_pct),
                    "leader_churn_pct": float(fallback.get("leader_churn_pct", leader_churn_pct) or leader_churn_pct),
                    "quality_summary": str((dict(fallback.get("universe_quality", {})).get("summary", "") if isinstance(fallback.get("universe_quality", {}), dict) else "") or ""),
                    "liquidity_missing_ratio_pct": round(float(liquidity_missing_ratio_pct), 3),
                    "liquidity_missing_allowed": bool(allow_missing_liquidity),
                    "fallback_cached": True,
                },
            )
            return fallback
        return {
            "state": "READY",
            "ai_state": "Scan ready",
            "msg": msg,
            "universe": candidates,
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
            "market_open": market_open,
            "rejected": rejected[:30],
            "reject_summary": reject_summary,
            "hints": hints[:5],
            "candidate_churn_pct": float(candidate_churn_pct),
            "leader_churn_pct": float(leader_churn_pct),
            "leader_mode": "none",
            "leader_stability_applied": False,
            "leader_stability_prev_symbol": str(prev_top_symbol),
            "scan_rotation": dict(rotation_info),
            "universe_quality": quality_report,
            "opening_plan": dict(opening_plan),
            "health": {"data_ok": False, "broker_ok": True, "orders_ok": True, "drift_warning": drift_warning},
            "pdt_note": "Paper mode can still simulate PDT protections; live day-trading may be limited under $25k.",
        }

    news_weight = max(0.0, min(1.0, float(settings.get("stock_news_event_weight", 0.12) or 0.12)))
    news_event_context: Dict[str, Any] = {
        "enabled": bool(settings.get("news_event_enabled", True)),
        "market": "stocks",
        "state": "disabled",
        "state_code": "disabled",
        "symbols": {},
        "errors": {},
    }
    if scored:
        try:
            news_event_context = build_unified_news_event_context(
                hub_dir=hub_dir,
                settings=settings,
                market="stocks",
                symbols=[str((row or {}).get("symbol", "") or "").strip().upper() for row in scored],
                now_ts=ts_now,
            )
        except Exception as exc:
            news_event_context = {
                "enabled": bool(settings.get("news_event_enabled", True)),
                "market": "stocks",
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
        if news_weight > 0.0 and news_symbols:
            for row in scored:
                symbol = str(row.get("symbol", "") or "").strip().upper()
                if not symbol:
                    continue
                n_row = news_symbols.get(symbol, {}) if isinstance(news_symbols.get(symbol, {}), dict) else {}
                if not n_row:
                    continue
                base_score = float(row.get("score", 0.0) or 0.0)
                n_score = _float(n_row.get("score", 0.0), 0.0)
                n_conf = _float(n_row.get("confidence", 0.0), 0.0)
                n_impact = _float(n_row.get("impact", 0.0), 0.0)
                merged_score = blend_score_with_news(
                    base_score=base_score,
                    news_score=n_score,
                    confidence=n_conf,
                    impact=n_impact,
                    weight=news_weight,
                )
                row["score_base"] = round(base_score, 6)
                row["score_news"] = round(float(n_score), 6)
                row["news_confidence"] = round(float(n_conf), 4)
                row["news_impact"] = round(float(n_impact), 4)
                row["news_weight"] = round(float(news_weight), 4)
                row["news_bias"] = str(n_row.get("bias", "neutral") or "neutral")
                row["news_headline_count"] = int(n_row.get("headline_count", 0) or 0)
                row["news_event_risk"] = bool(n_row.get("event_risk", False))
                row["news_top_headline"] = str(n_row.get("top_headline", "") or "")
                row["score"] = round(float(merged_score), 6)
                row["side"] = "long" if float(row["score"]) > 0.0 else "watch"
                if abs(float(row["score"]) - base_score) >= 0.0001:
                    _append_reason_parts(
                        row,
                        logic=f"News/event {row['news_bias']} bias adjusted conviction",
                        data=(
                            f"news {n_score:+.3f} | conf {n_conf:.2f} | "
                            f"impact {n_impact:.2f} | w {news_weight:.2f}"
                        ),
                    )

    scored.sort(key=lambda row: float(row.get("score", -9999.0)), reverse=True)
    outcome_map = _compute_outcome_map(hub_dir)
    market_calibration_samples = _market_pooled_calibration_samples(hub_dir, settings)
    for row in scored:
        sym = str(row.get("symbol", "") or "").strip().upper()
        m = outcome_map.get(sym, {})
        hr = float(m.get("hit_rate_pct", 50.0) or 50.0)
        ap = float(m.get("avg_pnl_pct", 0.0) or 0.0)
        smp = float(m.get("samples", 0.0) or 0.0)
        row["hit_rate_pct"] = round(hr, 2)
        row["avg_pnl_pct"] = round(ap, 4)
        row["calib_prob"] = _calibrated_prob(float(row.get("score", 0.0) or 0.0), hr, ap)
        row["samples"] = int(smp)
        row["symbol_samples"] = int(smp)
        row["market_calibration_samples"] = int(market_calibration_samples)
        row["calibration_scope"] = "symbol"
        row["calibration_effective_samples"] = int(smp)
        row["calibration_effective_prob"] = float(row.get("calib_prob", 0.0) or 0.0)
        quality_score = (
            (100.0 * float(row.get("valid_ratio", 0.0)))
            - (2.0 * float(row.get("spread_bps", 0.0)))
            + (0.8 * hr)
        )
        row["quality_score"] = round(quality_score, 3)
    # Universe health ranking and execution bucket.
    ranked_health = sorted(scored, key=lambda r: float(r.get("quality_score", -9999.0)), reverse=True)
    exec_n = max(6, int(len(ranked_health) * 0.35))
    exec_bucket = {str(r.get("symbol", "")).strip().upper() for r in ranked_health[:exec_n]}
    for row in scored:
        row["entry_gate_reason"] = ""
        row["eligible_for_entry"] = str(row.get("symbol", "")).strip().upper() in exec_bucket
        if market_open and bool(row.get("open_daily_fallback", False)):
            row["eligible_for_entry"] = False
            row["entry_gate_reason"] = "Daily bars fallback during market open"
            if str(row.get("side", "watch")).lower() == "long":
                row["side"] = "watch"
            _append_reason_parts(
                row,
                logic="Daily bars fallback during market open; hold as watch",
                data="open session daily fallback",
            )
        if (not row["eligible_for_entry"]) and (str(row.get("side", "watch")).lower() == "long"):
            _append_reason_parts(
                row,
                logic="Demoted to watchlist by universe quality ranking",
                data="universe health bucket",
            )
            row["side"] = "watch"
        elif str(row.get("side", "watch") or "watch").strip().lower() == "long":
            min_samples_guarded = max(0, int(float(settings.get("stock_min_samples_live_guarded", 5) or 5)))
            symbol_samples = int(float(row.get("symbol_samples", row.get("samples", 0)) or 0))
            pooled_samples = int(float(row.get("market_calibration_samples", 0) or 0))
            if symbol_samples < min_samples_guarded and pooled_samples >= min_samples_guarded:
                row["samples"] = int(pooled_samples)
                row["calibration_scope"] = "market_pooled"
                row["calibration_effective_samples"] = int(pooled_samples)
                row["calibration_effective_prob"] = float(row.get("calib_prob", 0.0) or 0.0)
            entry_gate_reason = _live_guarded_entry_gate_reason(settings, row)
            if entry_gate_reason:
                row["eligible_for_entry"] = False
                row["entry_gate_reason"] = entry_gate_reason
                row["side"] = "watch"
                _append_reason_parts(
                    row,
                    logic="Calibration history insufficient for live entry; hold as watch",
                    data=entry_gate_reason,
                )
        row["leader_rank_score"] = round(_leader_rank_score(row), 6)

    # Adaptive threshold blends volatility regime + replay recommendation (bounded step).
    vols = [float(r.get("volatility_pct", 0.0) or 0.0) for r in scored if float(r.get("volatility_pct", 0.0) or 0.0) > 0]
    vol_med = (sorted(vols)[len(vols) // 2] if vols else 0.0)
    base_thr = max(0.05, float(settings.get("stock_score_threshold", 0.2) or 0.2))
    volatility_threshold = float(round(base_thr * (1.25 if vol_med >= 0.65 else 1.0), 6))
    replay_enabled = _setting_bool(settings, "stock_replay_adaptive_enabled", True)
    replay_weight = max(0.0, min(1.0, float(settings.get("stock_replay_adaptive_weight", 0.35) or 0.35)))
    replay_step_cap_pct = max(5.0, min(90.0, float(settings.get("stock_replay_adaptive_step_cap_pct", 40.0) or 40.0)))
    replay_target_entries = replay_target_entries_for_market(settings, "stocks")
    replay_recommended = float(volatility_threshold)
    replay_clamped = float(volatility_threshold)
    replay_reason = ""
    if replay_enabled and scored:
        replay_payload = recommend_threshold_from_scores(
            scored,
            market="stocks",
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

    leaders_long = sorted(
        [row for row in scored if str(row.get("side", "")).lower() == "long"],
        key=lambda r: (
            1 if bool(r.get("eligible_for_entry", False)) else 0,
            float(r.get("leader_rank_score", r.get("score", -9999.0)) or -9999.0),
        ),
        reverse=True,
    )[:10]
    leader_mode = "long"
    leaders = list(leaders_long)
    publish_watch = _setting_bool(settings, "stock_scan_publish_watch_leaders", True)
    if (not leaders) and publish_watch and scored:
        watch_n = max(1, min(10, int(float(settings.get("stock_scan_watch_leaders_count", 6) or 6))))
        leaders = sorted(
            list(scored),
            key=lambda r: (
                1 if bool(r.get("eligible_for_entry", False)) else 0,
                float(r.get("leader_rank_score", r.get("score", -9999.0)) or -9999.0),
            ),
            reverse=True,
        )[:watch_n]
        leader_mode = "watch_fallback"
    try:
        stability_margin = max(0.0, min(100.0, float(settings.get("stock_leader_stability_margin_pct", 10.0) or 10.0)))
    except Exception:
        stability_margin = 10.0
    leaders, stability_applied = _apply_leader_hysteresis(leaders, prev_top_symbol, stability_margin)
    top_pick = leaders[0] if leaders else (scored[0] if scored else None)
    msg = "No viable long candidates"
    if leader_mode == "watch_fallback":
        msg = "No long setups yet; showing strongest watchlist candidates"
    if top_pick:
        msg = f"Top pick {top_pick['symbol']} | {top_pick['reason']}"
    top_symbol = str((top_pick or {}).get("symbol", "") or "").strip().upper()
    top_chart: List[Dict[str, Any]] = []
    chart_seed: List[Dict[str, Any]] = []
    if isinstance(top_pick, dict):
        chart_seed.append(dict(top_pick))
    chart_seed.extend([dict(r) for r in leaders[:10] if isinstance(r, dict)])
    chart_map_symbols = max(2, int(float(settings.get("market_chart_cache_symbols", 8) or 8)))
    chart_map_bars = max(40, int(float(settings.get("market_chart_cache_bars", 120) or 120)))
    top_chart_map = _build_top_chart_map(
        chart_seed,
        best_bars_by_symbol,
        max_symbols=chart_map_symbols,
        limit=chart_map_bars,
    )
    top_source = str((top_pick or {}).get("data_source", "") or "")
    if top_symbol:
        top_chart = list(top_chart_map.get(top_symbol, []) or [])
    if (not top_chart) and top_symbol:
        source_bars = list(best_bars_by_symbol.get(top_symbol, []) or bars_by_symbol.get(top_symbol, []) or [])
        top_chart = _compact_chart_bars(source_bars, limit=chart_map_bars)

    _append_jsonl(
        _rankings_path(hub_dir),
        {
            "ts": ts_now,
            "state": "READY",
            "market_open": market_open,
            "mode": ("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
            "universe_total": len(universe),
            "candidates": len(candidates),
            "rejected": rejected[:100],
            "top": leaders[:20],
        },
    )
    # Persist warmup queue (newly queued or cleared).
    _save_warmup_queue(hub_dir, warm_queue)
    _save_json_map(_feed_health_path(hub_dir), feed_health)
    _save_json_map(_symbol_cooldown_path(hub_dir), {"ts": int(time.time()), "symbols": cooldown_map})

    reject_summary = _summarize_rejections(rejected, len(universe))
    leader_symbols = [str((row or {}).get("symbol", "") or "").strip().upper() for row in leaders if isinstance(row, dict)]
    candidate_churn_pct = turnover_pct(prev_candidates, candidates)
    leader_churn_pct = turnover_pct(prev_leaders, leader_symbols)
    quality_report = build_universe_quality_report(
        market="stocks",
        ts=int(ts_now),
        mode=("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
        universe_total=int(len(universe)),
        candidates_total=int(len(candidates)),
        scores_total=int(len(scored)),
        leaders_total=int(len(leaders)),
        reject_summary=dict(reject_summary),
        rejected_rows=list(rejected),
        scored_rows=list(scored),
        candidate_churn_pct=float(candidate_churn_pct),
        leader_churn_pct=float(leader_churn_pct),
    )
    _save_json_map(_quality_report_path(hub_dir), quality_report)
    reason_counts = dict(reject_summary.get("counts", {}) or {})
    dominant_reason = str(reject_summary.get("dominant_reason", "") or "")
    dominant_ratio = float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0) / 100.0
    reject_rate_pct = effective_reject_pressure(
        reject_summary.get("reject_rate_pct", 0.0),
        dominant_reason=dominant_reason,
        dominant_ratio_pct=float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0),
        leaders_total=int(len(leaders)),
        scores_total=int(len(scored)),
    )
    reject_warn_pct = max(10.0, float(settings.get("stock_reject_drift_warn_pct", 65.0) or 65.0))
    drift_warning = bool((reject_rate_pct >= reject_warn_pct) and (dominant_ratio >= 0.60))
    _save_scan_diagnostics(
        hub_dir,
        {
            "ts": ts_now,
            "state": "READY",
            "mode": ("closed_daily" if ((not market_open) and use_daily_when_closed) else "intraday"),
            "market_open": bool(market_open),
            "universe_total": int(len(universe)),
            "candidates_total": int(len(candidates)),
            "scores_total": int(len(scored)),
            "leaders_total": int(len(leaders)),
            "top_symbol": str((top_pick or {}).get("symbol", "") or ""),
            "top_score": float((top_pick or {}).get("score", 0.0) or 0.0),
            "msg": str(msg),
            "reject_summary": dict(reject_summary),
            "feed_order": list(feed_order),
            "feed_health": dict((feed_health.get("feeds", {}) if isinstance(feed_health.get("feeds", {}), dict) else {})),
            "cooldown_active": int(sum(1 for v in (cooldown_map or {}).values() if int((v or {}).get("until", 0) or 0) > now_ts)),
            "window_policy": dict(window_policy),
            "window_policy_hits": int(window_policy_hits),
            "candidate_symbols": list(candidates),
            "leader_symbols": list(leader_symbols),
            "leader_mode": str(leader_mode),
            "leader_stability_applied": bool(stability_applied),
            "leader_stability_prev_symbol": str(prev_top_symbol),
            "scan_rotation": dict(rotation_info),
            "candidate_churn_pct": float(candidate_churn_pct),
            "leader_churn_pct": float(leader_churn_pct),
            "quality_summary": str(quality_report.get("summary", "") or ""),
            "liquidity_missing_ratio_pct": round(float(liquidity_missing_ratio_pct), 3),
            "liquidity_missing_allowed": bool(allow_missing_liquidity),
            "adaptive_threshold_base": float(base_thr),
            "adaptive_threshold_volatility": float(round(volatility_threshold, 6)),
            "adaptive_threshold_replay_recommended": float(round(replay_recommended, 6)),
            "adaptive_threshold_replay_clamped": float(round(replay_clamped, 6)),
            "adaptive_threshold_replay_weight": float(round(effective_weight, 4)),
            "adaptive_threshold_replay_target_entries": int(replay_target_entries),
            "adaptive_threshold_replay_reason": str(replay_reason),
            "adaptive_threshold_replay_enabled": bool(replay_enabled),
            "news_event_state": str(news_event_context.get("state", "") or ""),
            "news_event_state_code": str(news_event_context.get("state_code", "") or ""),
            "news_event_symbols": int(
                len(
                    dict(news_event_context.get("symbols", {}))
                    if isinstance(news_event_context.get("symbols", {}), dict)
                    else {}
                )
            ),
            "news_event_errors": int(
                len(
                    dict(news_event_context.get("errors", {}))
                    if isinstance(news_event_context.get("errors", {}), dict)
                    else {}
                )
            ),
        },
    )
    hints = _market_hints_from_rejects(
        {
            "counts": reason_counts,
            "reject_rate_pct": round(reject_rate_pct, 2),
            "dominant_reason": dominant_reason,
        }
    )
    for h in quality_hints(quality_report):
        if h not in hints:
            hints.append(h)
    if not market_open:
        hints.append("Off-hours scan mode: spread/cooldown gates relaxed until market open.")
    if bool(rotation_info.get("active", False)):
        hints.append(
            "Stock universe rotation active: "
            f"sticky core {int(rotation_info.get('core_size', 0))}, "
            f"rotating tail {int(rotation_info.get('tail_size', 0))}."
        )
    if replay_enabled:
        hints.append(
            f"Adaptive threshold {volatility_threshold:.3f} -> {adaptive_threshold:.3f} "
            f"(replay target {int(replay_target_entries)})."
        )
    opening_plan = _persist_opening_plan(
        settings,
        hub_dir,
        leaders=[dict(row) for row in leaders[:10] if isinstance(row, dict)],
        all_scores=[dict(row) for row in scored[:40] if isinstance(row, dict)],
        adaptive_threshold=float(adaptive_threshold),
        ts_now=ts_now,
        market_clock=clock_status,
        market_open=bool(market_open),
        msg=str(msg),
        fallback_cached=False,
    )

    return {
        "state": "READY",
        "ai_state": "Scan ready",
        "msg": msg,
        "universe": candidates,
        "leaders": leaders[:10],
        "all_scores": scored[:40],
        "top_pick": top_pick,
        "top_chart": top_chart,
        "top_chart_map": top_chart_map,
        "top_chart_source": top_source,
        "adaptive_threshold": adaptive_threshold,
        "adaptive_threshold_base": float(base_thr),
        "adaptive_threshold_volatility": float(round(volatility_threshold, 6)),
        "adaptive_threshold_replay_recommended": float(round(replay_recommended, 6)),
        "adaptive_threshold_replay_clamped": float(round(replay_clamped, 6)),
        "adaptive_threshold_replay_weight": float(round(effective_weight, 4)),
        "adaptive_threshold_replay_target_entries": int(replay_target_entries),
        "adaptive_threshold_replay_reason": str(replay_reason),
        "adaptive_threshold_replay_enabled": bool(replay_enabled),
        "updated_at": ts_now,
        "market_open": market_open,
        "market_clock": dict(clock_status),
        "rejected": rejected[:30],
        "reject_summary": reject_summary,
        "feed_order": list(feed_order),
        "hints": hints[:5],
        "candidate_churn_pct": float(candidate_churn_pct),
        "leader_churn_pct": float(leader_churn_pct),
        "leader_mode": str(leader_mode),
        "leader_stability_applied": bool(stability_applied),
        "leader_stability_prev_symbol": str(prev_top_symbol),
        "scan_rotation": dict(rotation_info),
        "window_policy": dict(window_policy),
        "window_policy_hits": int(window_policy_hits),
        "universe_quality": quality_report,
        "news_event_context": news_event_context,
        "opening_plan": dict(opening_plan),
        "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": drift_warning},
        "pdt_note": "Paper mode can still simulate PDT protections; live day-trading may be limited under $25k.",
    }


def main() -> int:
    print("stock_thinker.py is designed to be imported by the hub/runner first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
