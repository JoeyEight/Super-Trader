from __future__ import annotations

import csv
import io
import json
import os
import random
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.confidence_calibration import build_market_confidence_calibration
from app.credential_utils import get_oanda_creds
from app.http_utils import retry_after_from_urllib_http_error
from app.market_awareness import forex_session_bias
from app.path_utils import resolve_runtime_paths
from app.rejection_replay import recommend_threshold_from_scores, replay_target_entries_for_market
from app.scan_diagnostics_schema import with_scan_schema
from app.scanner_quality import build_universe_quality_report, effective_reject_pressure, quality_hints, turnover_pct
from brokers.broker_oanda import OandaBrokerClient

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "forex_thinker")

DEFAULT_FX_UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_JPY"]
ROLLOUT_ORDER = {
    "legacy": 0,
    "scan_expanded": 1,
    "risk_caps": 2,
    "execution_v2": 3,
    "shadow_only": 4,
    "live": 5,
    "live_guarded": 5,
}
FOREX_FACTORY_EXPORT_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.csv"
FOREX_FACTORY_USER_AGENT = "Mozilla/5.0 (SuperTrader/1.0)"
_FF_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
_FOREX_OUTLIER_MAX_ABS_CHANGE_6H_PCT = 8.0
_FOREX_OUTLIER_MAX_ABS_CHANGE_24H_PCT = 12.0
_FOREX_OUTLIER_MAX_STEP_MOVE_PCT = 3.0


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


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
    raise RuntimeError("forex_thinker request failed")


def _rollout_at_least(settings: Dict[str, Any], stage: str) -> bool:
    cur = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    target = str(stage or "").strip().lower()
    if cur == "live_guarded":
        cur = "live"
    if target == "live_guarded":
        target = "live"
    return int(ROLLOUT_ORDER.get(cur, 0)) >= int(ROLLOUT_ORDER.get(target, 0))


def _rankings_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "scanner_rankings.jsonl")


def _execution_audit_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _scan_diag_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "scan_diagnostics.json")


def _pair_cooldown_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "pair_cooldown.json")


def _quality_report_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "universe_quality.json")


def _calendar_cache_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "forexfactory_calendar_cache.json")


def _calendar_refresh_marker_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "forex", "forexfactory_calendar_refresh.lock")


def _confidence_calibration_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "confidence_calibration.json")


def _norm_impact(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if "high" in text:
        return "high"
    if "med" in text:
        return "medium"
    if "low" in text:
        return "low"
    return ""


def _parse_ff_datetime(date_raw: Any, time_raw: Any, now_ts: int) -> float:
    date_txt = " ".join(str(date_raw or "").replace(",", " ").split())
    time_txt = " ".join(str(time_raw or "").replace(".", "").split())
    if (not date_txt) or (not time_txt):
        return 0.0
    if time_txt.strip().lower() in {"all day", "day", "tentative", "n/a"}:
        return 0.0
    now_dt = datetime.fromtimestamp(float(now_ts), tz=timezone.utc)
    candidates = [f"{date_txt} {time_txt} {now_dt.year}", f"{date_txt} {time_txt}"]
    fmts = (
        "%a %b %d %I:%M%p %Y",
        "%a %b %d %H:%M %Y",
        "%b %d %I:%M%p %Y",
        "%b %d %H:%M %Y",
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M%p",
    )
    for cand in candidates:
        for fmt in fmts:
            try:
                dt = datetime.strptime(cand, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return float(dt.timestamp())
            except Exception:
                continue
        try:
            iso = cand
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            continue
    return 0.0


def _fetch_forexfactory_events(now_ts: int, timeout_s: float = 8.0) -> List[Dict[str, Any]]:
    req = urllib.request.Request(
        FOREX_FACTORY_EXPORT_URL,
        headers={"User-Agent": FOREX_FACTORY_USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read().decode("utf-8", "replace")
    reader = csv.DictReader(io.StringIO(raw))
    events: List[Dict[str, Any]] = []
    for row in reader:
        norm = {str(k or "").strip().lower(): row.get(k) for k in row.keys()}
        currency = str(norm.get("currency", "") or "").strip().upper()
        impact = _norm_impact(norm.get("impact", ""))
        title = str(norm.get("event", "") or norm.get("title", "") or "").strip()
        ts = _parse_ff_datetime(norm.get("date", ""), norm.get("time", ""), now_ts=now_ts)
        if (not currency) or (not impact) or (not title) or ts <= 0.0:
            continue
        events.append(
            {
                "ts": int(ts),
                "currency": currency,
                "impact": impact,
                "title": title,
                "forecast": str(norm.get("forecast", "") or "").strip(),
                "previous": str(norm.get("previous", "") or "").strip(),
                "actual": str(norm.get("actual", "") or "").strip(),
                "source": "forexfactory",
            }
        )
    events.sort(key=lambda e: int(e.get("ts", 0) or 0))
    return events


def _try_acquire_calendar_refresh_marker(marker_path: str, now_ts: int, stale_after_s: float) -> bool:
    stale_window_s = max(30.0, float(stale_after_s or 0.0))
    if os.path.isfile(marker_path):
        marker = _load_json_map(marker_path)
        started_ts = _float(marker.get("started_ts", 0.0), 0.0)
        if started_ts <= 0.0:
            try:
                started_ts = float(os.path.getmtime(marker_path))
            except Exception:
                started_ts = float(now_ts)
        if (float(now_ts) - started_ts) <= stale_window_s:
            return False
        try:
            os.remove(marker_path)
        except Exception:
            return False
    try:
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except Exception:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"started_ts": int(now_ts), "pid": int(os.getpid())}, f, indent=2)
    except Exception:
        return False
    return True


def _release_calendar_refresh_marker(marker_path: str) -> None:
    try:
        if os.path.isfile(marker_path):
            os.remove(marker_path)
    except Exception:
        pass


def _refresh_forexfactory_cache(hub_dir: str, settings: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    path = _calendar_cache_path(hub_dir)
    cache = _load_json_map(path)
    cached_events = list(cache.get("events", []) or []) if isinstance(cache.get("events", []), list) else []
    try:
        fetched_ts = float(cache.get("fetched_ts", 0.0) or 0.0)
    except Exception:
        fetched_ts = 0.0
    timeout_s = 8.0
    try:
        timeout_s = max(1.0, min(20.0, float(settings.get("news_event_timeout_s", 8.0) or 8.0)))
    except Exception:
        timeout_s = 8.0
    try:
        events = _fetch_forexfactory_events(now_ts=now_ts, timeout_s=timeout_s)
        payload = dict(cache)
        payload.update(
            {
                "fetched_ts": int(now_ts),
                "events": events,
                "source": "forexfactory",
                "last_error_ts": 0,
                "last_error": "",
                "last_refresh_attempt_ts": int(now_ts),
                "last_refresh_ok_ts": int(now_ts),
                "last_refresh_state": "ok",
            }
        )
        _save_json_map(path, payload)
        return {"ok": True, "events_total": int(len(events))}
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        payload = dict(cache)
        payload.update(
            {
                "fetched_ts": int(fetched_ts),
                "events": cached_events,
                "source": "forexfactory",
                "last_error_ts": int(now_ts),
                "last_error": err,
                "last_refresh_attempt_ts": int(now_ts),
                "last_refresh_state": "error",
            }
        )
        _save_json_map(path, payload)
        return {"ok": False, "events_total": int(len(cached_events)), "error": err}


def _spawn_forexfactory_refresh(hub_dir: str, settings: Dict[str, Any], marker_path: str) -> bool:
    def _runner() -> None:
        try:
            _refresh_forexfactory_cache(hub_dir, dict(settings or {}), int(time.time()))
        finally:
            _release_calendar_refresh_marker(marker_path)

    try:
        thread = threading.Thread(
            target=_runner,
            name="forex_factory_refresh",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        _release_calendar_refresh_marker(marker_path)
        return False


def _load_forexfactory_context_cached(hub_dir: str, settings: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    enabled = bool(settings.get("forex_event_risk_enabled", True))
    if not enabled:
        return {"enabled": False, "state": "disabled", "events": [], "error": ""}

    refresh_s = max(60.0, float(settings.get("forex_event_cache_refresh_s", 1800.0) or 1800.0))
    stale_max_s = max(refresh_s, float(settings.get("forex_event_cache_stale_max_s", 86400.0) or 86400.0))
    path = _calendar_cache_path(hub_dir)
    cache = _load_json_map(path)
    cached_events = list(cache.get("events", []) or []) if isinstance(cache.get("events", []), list) else []
    try:
        fetched_ts = float(cache.get("fetched_ts", 0.0) or 0.0)
    except Exception:
        fetched_ts = 0.0
    try:
        last_error_ts = float(cache.get("last_error_ts", 0.0) or 0.0)
    except Exception:
        last_error_ts = 0.0
    last_error = str(cache.get("last_error", "") or "")
    cache_age = (float(now_ts) - fetched_ts) if fetched_ts > 0 else 1e12
    cache_age_s = int(max(0.0, cache_age)) if fetched_ts > 0 else -1

    if cached_events and cache_age <= refresh_s:
        return {
            "enabled": True,
            "state": "cached",
            "events": cached_events,
            "error": "",
            "cache_age_s": int(max(0.0, cache_age)),
            "refresh_s": float(refresh_s),
            "stale_max_s": float(stale_max_s),
            "refresh_eligible": False,
        }
    if cached_events and cache_age <= stale_max_s:
        return {
            "enabled": True,
            "state": "cached_stale",
            "events": cached_events,
            "error": str(last_error),
            "cache_age_s": int(max(0.0, cache_age)),
            "refresh_s": float(refresh_s),
            "stale_max_s": float(stale_max_s),
            "refresh_eligible": True,
        }
    if (not cached_events) and last_error_ts > 0 and (float(now_ts) - last_error_ts) <= refresh_s:
        return {
            "enabled": True,
            "state": "cooldown",
            "events": [],
            "error": str(last_error),
            "cache_age_s": int(cache_age_s),
            "refresh_s": float(refresh_s),
            "stale_max_s": float(stale_max_s),
            "refresh_eligible": False,
        }
    return {
        "enabled": True,
        "state": "unavailable",
        "events": [],
        "error": str(last_error),
        "cache_age_s": int(cache_age_s),
        "refresh_s": float(refresh_s),
        "stale_max_s": float(stale_max_s),
        "refresh_eligible": True,
    }


def _maybe_schedule_forexfactory_refresh(hub_dir: str, settings: Dict[str, Any], calendar_ctx: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    if not bool((calendar_ctx or {}).get("enabled", False)):
        return {"scheduled": False, "reason": "disabled"}
    if not bool((calendar_ctx or {}).get("refresh_eligible", False)):
        return {"scheduled": False, "reason": "not_needed"}

    refresh_s = max(60.0, float(settings.get("forex_event_cache_refresh_s", 1800.0) or 1800.0))
    stale_max_s = max(refresh_s, float(settings.get("forex_event_cache_stale_max_s", 86400.0) or 86400.0))
    request_cooldown_s = max(30.0, min(300.0, refresh_s * 0.20))
    path = _calendar_cache_path(hub_dir)
    cache = _load_json_map(path)
    last_request_ts = _float(cache.get("last_refresh_request_ts", 0.0), 0.0)
    if last_request_ts > 0.0 and (float(now_ts) - last_request_ts) < request_cooldown_s:
        return {"scheduled": False, "reason": "throttled"}

    marker_path = _calendar_refresh_marker_path(hub_dir)
    marker_ttl_s = max(120.0, min(1800.0, stale_max_s))
    if not _try_acquire_calendar_refresh_marker(marker_path, now_ts, stale_after_s=marker_ttl_s):
        return {"scheduled": False, "reason": "in_flight"}

    payload = dict(cache)
    payload.update(
        {
            "last_refresh_request_ts": int(now_ts),
            "last_refresh_state": "scheduled",
        }
    )
    _save_json_map(path, payload)
    if not _spawn_forexfactory_refresh(hub_dir, settings, marker_path):
        return {"scheduled": False, "reason": "spawn_failed"}
    return {"scheduled": True, "reason": "scheduled"}


def _load_forexfactory_context(hub_dir: str, settings: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    ctx = _load_forexfactory_context_cached(hub_dir, settings, now_ts)
    refresh_meta = _maybe_schedule_forexfactory_refresh(hub_dir, settings, ctx, now_ts)
    out = dict(ctx or {})
    out["refresh_scheduled"] = bool(refresh_meta.get("scheduled", False))
    out["refresh_state"] = str(refresh_meta.get("reason", "not_needed") or "not_needed")
    return out


def _pair_ccys(pair: str) -> tuple[str, str]:
    raw = str(pair or "").strip().upper()
    if "_" in raw:
        a, b = raw.split("_", 1)
        return str(a or "").strip().upper(), str(b or "").strip().upper()
    if len(raw) >= 6:
        return raw[:3], raw[3:6]
    return raw, ""


def _pair_event_risk(
    pair: str,
    calendar_ctx: Dict[str, Any],
    now_ts: int,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "active": False,
        "severity": "none",
        "minutes_to_event": 99999,
        "score_mult": 1.0,
        "block_entry": False,
        "logic": "",
        "data": "",
        "state": str(calendar_ctx.get("state", "disabled") or "disabled"),
    }
    events = list(calendar_ctx.get("events", []) or []) if isinstance(calendar_ctx.get("events", []), list) else []
    if not events:
        return out
    base, quote = _pair_ccys(pair)
    watch_ccys = {x for x in (base, quote) if x}
    if not watch_ccys:
        return out

    lookahead_min = max(5, int(float(settings.get("forex_event_max_lookahead_minutes", 180) or 180)))
    post_min = max(0, int(float(settings.get("forex_event_post_event_minutes", 30) or 30)))
    block_high_min = max(0, int(float(settings.get("forex_event_block_high_impact_minutes", 45) or 45)))
    score_mult_high = max(0.10, min(1.0, float(settings.get("forex_event_score_mult_high", 0.70) or 0.70)))
    score_mult_medium = max(0.10, min(1.0, float(settings.get("forex_event_score_mult_medium", 0.85) or 0.85)))

    pick: Dict[str, Any] | None = None
    pick_rank = -1
    pick_abs_min = 1e9
    for evt in events:
        if not isinstance(evt, dict):
            continue
        ccy = str(evt.get("currency", "") or "").strip().upper()
        if ccy not in watch_ccys:
            continue
        impact = _norm_impact(evt.get("impact", ""))
        if impact not in {"high", "medium"}:
            continue
        try:
            evt_ts = int(float(evt.get("ts", 0) or 0))
        except Exception:
            evt_ts = 0
        if evt_ts <= 0:
            continue
        delta_min = int(round((float(evt_ts) - float(now_ts)) / 60.0))
        if delta_min > lookahead_min or delta_min < (-1 * post_min):
            continue
        rank = int(_FF_IMPACT_RANK.get(impact, 0))
        abs_min = abs(delta_min)
        if (pick is None) or (rank > pick_rank) or ((rank == pick_rank) and (abs_min < pick_abs_min)):
            pick = evt
            pick_rank = rank
            pick_abs_min = abs_min
            pick["delta_min"] = delta_min

    if not pick:
        return out

    impact = _norm_impact(pick.get("impact", ""))
    delta_min = int(pick.get("delta_min", 0) or 0)
    ccy = str(pick.get("currency", "") or "").strip().upper()
    title = str(pick.get("title", "") or "").strip()
    state = str(calendar_ctx.get("state", "disabled") or "disabled")
    when_txt = ("now" if delta_min == 0 else (f"in {delta_min}m" if delta_min > 0 else f"{abs(delta_min)}m ago"))
    score_mult = score_mult_high if impact == "high" else score_mult_medium
    block_entry = bool((impact == "high") and (delta_min <= block_high_min) and (delta_min >= -5))
    logic = (
        f"High-impact macro risk near {ccy} ({when_txt})"
        if impact == "high"
        else f"Medium-impact macro risk near {ccy} ({when_txt})"
    )
    data = f"{ccy} | {title} | impact {impact.upper()} | T{delta_min:+}m | source ForexFactory/{state}"
    out.update(
        {
            "active": True,
            "severity": impact,
            "minutes_to_event": delta_min,
            "score_mult": float(score_mult),
            "block_entry": bool(block_entry),
            "logic": logic,
            "data": data,
        }
    )
    return out


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
    for c in list(rows or [])[-take_n:]:
        if not isinstance(c, dict):
            continue
        mid = c.get("mid", {}) if isinstance(c.get("mid", {}), dict) else {}
        close_px = _float(mid.get("c", 0.0), 0.0)
        if close_px <= 0.0:
            continue
        open_px = _float(mid.get("o", close_px), close_px)
        high_px = _float(mid.get("h", max(open_px, close_px)), max(open_px, close_px))
        low_px = _float(mid.get("l", min(open_px, close_px)), min(open_px, close_px))
        out.append(
            {
                "t": str(c.get("time", "") or ""),
                "o": float(open_px),
                "h": float(max(high_px, open_px, close_px)),
                "l": float(min(low_px, open_px, close_px)),
                "c": float(close_px),
                "v": int(_float(c.get("volume", 0.0), 0.0)),
            }
        )
    return out


def _build_top_chart_map(
    leaders: List[Dict[str, Any]],
    candles_lookup: Dict[str, List[Dict[str, Any]]],
    max_pairs: int = 6,
    limit: int = 120,
    include_pairs: List[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    seen: set[str] = set()
    max_take = max(1, int(max_pairs))

    def _add_pair(raw_pair: Any) -> None:
        if len(out) >= max_take:
            return
        pair = str(raw_pair or "").strip().upper()
        if (not pair) or (pair in seen):
            return
        seen.add(pair)
        bars = _compact_chart_bars(list(candles_lookup.get(pair, []) or []), limit=limit)
        if len(bars) >= 2:
            out[pair] = bars

    for pair in list(include_pairs or []):
        _add_pair(pair)
    for row in list(leaders or []):
        if len(out) >= max_take:
            break
        if not isinstance(row, dict):
            continue
        _add_pair(row.get("pair"))
    return out


def _load_open_position_pairs(hub_dir: str) -> List[str]:
    out: List[str] = []
    try:
        status = _load_json_map(os.path.join(hub_dir, "forex", "oanda_status.json"))
        for row in list(status.get("raw_positions", []) or []):
            if not isinstance(row, dict):
                continue
            pair = str(row.get("instrument", "") or row.get("pair", "") or "").strip().upper()
            if pair and (pair not in out):
                out.append(pair)
    except Exception:
        pass
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
        (abs(score) * 10.0)
        + (quality * 0.08)
        + (calib_prob * 8.0)
        + (valid_ratio * 4.0)
        + eligible_bonus
        + data_bonus
        - (spread_bps * 0.12)
    )


def _apply_leader_hysteresis(
    leaders: List[Dict[str, Any]],
    prev_pair: str,
    margin_pct: float,
) -> tuple[List[Dict[str, Any]], bool]:
    rows = [dict(r) for r in list(leaders or []) if isinstance(r, dict)]
    target = str(prev_pair or "").strip().upper()
    margin = max(0.0, float(margin_pct or 0.0))
    if (not rows) or (not target) or margin <= 0.0:
        return rows, False
    top = rows[0]
    top_score = abs(_float(top.get("leader_rank_score", top.get("score", 0.0)), 0.0))
    top_side = str(top.get("side", "watch") or "watch").strip().lower()
    if top_score <= 0.0:
        return rows, False
    for idx, row in enumerate(rows[1:], start=1):
        pair = str(row.get("pair", "") or "").strip().upper()
        if pair != target:
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


def _session_weight_multiplier(settings: Dict[str, Any], side: str, session_ctx: Dict[str, Any]) -> tuple[float, str]:
    enabled = bool(settings.get("forex_session_weight_enabled", True))
    if not enabled:
        return 1.0, "disabled"
    s = str(side or "watch").strip().lower()
    if s not in {"long", "short"}:
        return 1.0, "watch"
    floor = max(0.5, min(1.0, float(settings.get("forex_session_weight_floor", 0.85) or 0.85)))
    ceiling = max(1.0, min(2.0, float(settings.get("forex_session_weight_ceiling", 1.10) or 1.10)))
    if ceiling < floor:
        ceiling = floor
    bias = str((session_ctx or {}).get("bias", "FLAT") or "FLAT").strip().upper()
    if bias == "TREND":
        return float(ceiling), "trend_boost"
    if bias in {"RANGE", "MEAN-REV"}:
        return float(floor), "range_dampen"
    return 1.0, "flat"


def _logic_reason_from_score(
    side: str,
    score: float,
    change_6: float,
    change_24: float,
    volatility: float,
) -> str:
    s = str(side or "watch").strip().lower()
    if s == "long":
        if change_6 >= 0.0 and change_24 >= 0.0:
            return "Uptrend pressure from positive 6h/24h momentum"
        return "Long bias from recent upside, but momentum is mixed"
    if s == "short":
        if change_6 <= 0.0 and change_24 <= 0.0:
            return "Trending to downside from negative 6h/24h momentum"
        return "Short bias from recent weakness, but momentum is mixed"
    if abs(float(score)) < 0.10 and volatility < 0.02:
        return "Range/low-volatility behavior; no clear directional edge"
    return "Watch bias; directional edge is weak"


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
    stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    if stage not in {"live_guarded", "live"}:
        return ""
    pair = str(row.get("pair", "") or "").strip().upper()
    if not pair:
        return ""
    sample_count = int(float(row.get("calibration_effective_samples", row.get("samples", 0)) or 0))
    pair_samples = int(float(row.get("pair_samples", row.get("samples", 0)) or 0))
    market_samples = int(float(row.get("market_calibration_samples", 0) or 0))
    min_samples_guarded = max(0, int(float(settings.get("forex_min_samples_live_guarded", 5) or 5)))
    if sample_count < min_samples_guarded:
        if market_samples > pair_samples:
            return f"Calibration sample gate for {pair} ({pair_samples} pair / {market_samples} pooled < {min_samples_guarded})"
        return f"Calibration sample gate for {pair} ({pair_samples} < {min_samples_guarded})"
    calib_prob = float(row.get("calibration_effective_prob", row.get("calib_prob", 0.0)) or 0.0)
    if calib_prob <= 0.0:
        calib_prob = 0.5
    min_calib_prob = max(0.0, min(1.0, float(settings.get("forex_min_calib_prob_live_guarded", 0.56) or 0.56)))
    if calib_prob < min_calib_prob:
        return f"Calibrated confidence gate for {pair} ({calib_prob:.2f} < {min_calib_prob:.2f})"
    return ""


def _market_pooled_calibration_samples(hub_dir: str, settings: Dict[str, Any]) -> int:
    payload = _load_json_map(_confidence_calibration_path(hub_dir))
    row: Dict[str, Any] = {}
    if isinstance(payload.get("forex", {}), dict):
        row = dict(payload.get("forex", {}) or {})
    elif str(payload.get("market", "") or "").strip().lower() == "forex":
        row = payload
    try:
        samples = int(float(row.get("samples", 0) or 0))
    except Exception:
        samples = 0
    if samples > 0:
        return samples
    try:
        base_threshold = float(settings.get("forex_score_threshold", 0.2) or 0.2)
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
        "forex",
        base_threshold=base_threshold,
        min_samples=adaptive_min_samples,
        target_success_pct=target_success,
    )
    try:
        return int(float(built.get("samples", 0) or 0))
    except Exception:
        return 0


def _rotate_jsonl(path: str, max_bytes: int = 64 * 1024 * 1024, keep: int = 8) -> None:
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


def _save_scan_diagnostics(hub_dir: str, payload: Dict[str, Any]) -> None:
    path = _scan_diag_path(hub_dir)
    try:
        row = with_scan_schema(payload, market="forex")
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
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _cached_scan_fallback(
    hub_dir: str,
    ts_now: int,
    msg: str,
    universe: List[str],
    session_ctx: Dict[str, Any],
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
        top_pair = str((top_pick or {}).get("pair", "") or "").strip().upper()
        if top_pair and isinstance(top_chart_map.get(top_pair, None), list):
            top_chart = list(top_chart_map.get(top_pair, []) or [])
    if (not leaders) and (not all_scores) and (not top_chart):
        return None

    prev_updated = int(float(prev.get("updated_at", prev.get("ts", 0)) or 0))
    age_s = max(0, int(ts_now - prev_updated)) if prev_updated > 0 else 0
    fallback_msg = f"{str(msg or '').strip()} | using cached scan ({age_s}s old)"

    prev_hints = list(prev.get("hints", []) or []) if isinstance(prev.get("hints", []), list) else []
    hints = [f"Network degraded; serving cached leaders ({age_s}s old)."]
    for h in prev_hints[:4]:
        sh = str(h or "").strip()
        if sh and sh not in hints:
            hints.append(sh)

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
        "rejected": list(prev.get("rejected", []) or [])[:30],
        "reject_summary": (dict(prev.get("reject_summary", {})) if isinstance(prev.get("reject_summary", {}), dict) else {}),
        "cooldown_active": int(prev.get("cooldown_active", 0) or 0),
        "hints": hints[:5],
        "candidate_churn_pct": float(prev.get("candidate_churn_pct", 0.0) or 0.0),
        "leader_churn_pct": float(prev.get("leader_churn_pct", 0.0) or 0.0),
        "session_context": dict(session_ctx or {}),
        "session_weighted_candidates": int(prev.get("session_weighted_candidates", 0) or 0),
        "universe_quality": (dict(prev.get("universe_quality", {})) if isinstance(prev.get("universe_quality", {}), dict) else {}),
        "event_context": (dict(prev.get("event_context", {})) if isinstance(prev.get("event_context", {}), dict) else {}),
        "leader_stability_applied": bool(prev.get("leader_stability_applied", False)),
        "leader_stability_prev_pair": str(prev.get("leader_stability_prev_pair", "") or ""),
        "fallback_cached": True,
        "health": {"data_ok": False, "broker_ok": True, "orders_ok": True, "drift_warning": True},
    }


def _cooldown_reasons(settings: Dict[str, Any], key: str, default_csv: str) -> set[str]:
    raw = str(settings.get(key, default_csv) or default_csv)
    out = set()
    for tok in raw.replace(";", ",").split(","):
        r = str(tok or "").strip().lower()
        if r:
            out.add(r)
    return out


def _apply_pair_cooldown(
    cooldown_map: Dict[str, Any],
    pair: str,
    reason: str,
    settings: Dict[str, Any],
    now_ts: int,
) -> None:
    p = str(pair or "").strip().upper()
    rsn = str(reason or "").strip().lower()
    if not p or not rsn:
        return
    reasons = _cooldown_reasons(
        settings,
        "forex_pair_cooldown_reject_reasons",
        "data_quality,insufficient_bars,spread,low_volatility",
    )
    if rsn not in reasons:
        return
    mins = max(1, int(float(settings.get("forex_pair_cooldown_minutes", 20) or 20)))
    min_hits = max(1, int(float(settings.get("forex_pair_cooldown_min_hits", 2) or 2)))
    row = cooldown_map.get(p, {})
    if not isinstance(row, dict):
        row = {}
    hit_count = int(row.get("hit_count", 0) or 0) + 1
    until = int(row.get("until", 0) or 0)
    if hit_count >= min_hits:
        until = int(now_ts + (mins * 60))
        hit_count = 0
    cooldown_map[p] = {
        "pair": p,
        "reason": rsn,
        "hit_count": int(hit_count),
        "until": int(until),
        "updated_ts": int(now_ts),
    }


def _prune_cooldown_map(cooldown_map: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair, row in (cooldown_map or {}).items():
        if not isinstance(row, dict):
            continue
        p = str(pair or "").strip().upper()
        if not p:
            continue
        until = int(row.get("until", 0) or 0)
        updated = int(row.get("updated_ts", 0) or 0)
        if until > now_ts:
            out[p] = row
            continue
        if (now_ts - updated) <= 7200:
            out[p] = row
    return out


def _parse_pairs(settings: Dict[str, Any]) -> List[str]:
    raw = str(settings.get("forex_universe_pairs", "") or "")
    out: List[str] = []
    for tok in raw.replace("\n", ",").split(","):
        p = tok.strip().upper()
        if p and p not in out:
            out.append(p)
    return out


def _score_candles(pair: str, candles: List[Dict[str, Any]], spread_bps: float = 0.0) -> Dict[str, Any]:
    closes = []
    highs = []
    lows = []
    for row in candles or []:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("complete", True)):
            continue
        mid = row.get("mid", {}) or {}
        c = _float(mid.get("c", 0.0), 0.0)
        h = _float(mid.get("h", 0.0), 0.0)
        low_px = _float(mid.get("l", 0.0), 0.0)
        if c > 0:
            closes.append(c)
        if h > 0:
            highs.append(h)
        if low_px > 0:
            lows.append(low_px)
    if len(closes) < 8:
        reason_logic = "Insufficient market history for a reliable trend call"
        reason_data = "bars<8 on H1 sample"
        return {
            "pair": pair,
            "score": -9999.0,
            "outlier": False,
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
    max_step_move = max(step_moves[-24:]) if step_moves else 0.0
    volatility = (sum(step_moves[-12:]) / max(1, len(step_moves[-12:]))) if step_moves else 0.0

    # Protect against malformed candles / feed spikes that produce unrealistic scores.
    if (
        abs(float(change_6)) > float(_FOREX_OUTLIER_MAX_ABS_CHANGE_6H_PCT)
        or abs(float(change_24)) > float(_FOREX_OUTLIER_MAX_ABS_CHANGE_24H_PCT)
        or float(max_step_move) > float(_FOREX_OUTLIER_MAX_STEP_MOVE_PCT)
    ):
        reason_logic = "Outlier price jump detected; waiting for cleaner candles"
        reason_data = (
            f"6h {change_6:+.3f}% | 24h {change_24:+.3f}% | "
            f"max step {max_step_move:.3f}% | spr {float(spread_bps):.2f}bps"
        )
        return {
            "pair": pair,
            "score": -9999.0,
            "outlier": True,
            "side": "watch",
            "last": round(last_px, 6),
            "change_6h_pct": round(change_6, 6),
            "change_24h_pct": round(change_24, 6),
            "volatility_pct": round(volatility, 6),
            "spread_bps": round(float(spread_bps), 4),
            "confidence": "LOW",
            "reason_logic": reason_logic,
            "reason_data": reason_data,
            "reason": reason_logic,
        }
    spread_penalty = max(0.0, float(spread_bps) / 10.0)
    score = (change_6 * 0.60) + (change_24 * 0.25) + (volatility * 0.20) - spread_penalty
    side = "long" if score > 0 else "short"
    abs_score = abs(score)
    if abs_score >= 0.45:
        confidence = "HIGH"
    elif abs_score >= 0.20:
        confidence = "MED"
    else:
        confidence = "LOW"
    reason_logic = _logic_reason_from_score(side, score, change_6, change_24, volatility)
    reason_data = f"6h {change_6:+.3f}% | 24h {change_24:+.3f}% | vol {volatility:.3f}% | spr {float(spread_bps):.2f}bps"
    return {
        "pair": pair,
        "score": round(score, 6),
        "outlier": False,
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


def _bar_quality(candles: List[Dict[str, Any]]) -> Dict[str, float]:
    if not candles:
        return {"valid_ratio": 0.0, "stale_hours": 9999.0}
    valid = 0
    latest_ts = 0.0
    for row in candles:
        if not isinstance(row, dict):
            continue
        c = _float(((row.get("mid") or {}).get("c", 0.0)), 0.0)
        if c > 0:
            valid += 1
        t = str(row.get("time", "") or "").strip()
        if t:
            try:
                ts = _parse_iso_ts(t)
                latest_ts = max(latest_ts, ts)
            except Exception:
                pass
    ratio = float(valid) / float(max(1, len(candles)))
    stale_h = 9999.0
    if latest_ts > 0:
        stale_h = max(0.0, (time.time() - latest_ts) / 3600.0)
    return {"valid_ratio": ratio, "stale_hours": stale_h}


def _parse_iso_ts(raw_ts: str) -> float:
    s = str(raw_ts or "").strip()
    if not s:
        return 0.0
    # OANDA may emit nanosecond precision; datetime.fromisoformat supports up to microseconds.
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
        pair = str(row.get("instrument", "") or row.get("pair", "") or "").strip().upper()
        if not pair:
            continue
        pnl = _float(row.get("pnl_pct", 0.0), 0.0)
        per.setdefault(pair, []).append(pnl)
    out: Dict[str, Dict[str, float]] = {}
    for pair, pnls in per.items():
        wins = sum(1 for p in pnls if p > 0.0)
        out[pair] = {
            "hit_rate_pct": round((100.0 * wins / max(1, len(pnls))), 2),
            "avg_pnl_pct": round((sum(pnls) / max(1, len(pnls))), 4),
            "samples": float(len(pnls)),
        }
    return out


def _calibrated_prob(score: float, hit_rate_pct: float, avg_pnl_pct: float) -> float:
    score_term = max(0.0, min(1.0, abs(score) / 0.8))
    hit_term = max(0.0, min(1.0, float(hit_rate_pct) / 100.0))
    pnl_term = max(0.0, min(1.0, (float(avg_pnl_pct) + 1.0) / 2.0))
    return round((0.45 * score_term) + (0.40 * hit_term) + (0.15 * pnl_term), 4)


def _market_hints_from_rejects(reject_summary: Dict[str, Any]) -> List[str]:
    counts = (reject_summary or {}).get("counts", {}) or {}
    if not isinstance(counts, dict):
        counts = {}
    dominant = str((reject_summary or {}).get("dominant_reason", "") or "").strip().lower()
    rate = float((reject_summary or {}).get("reject_rate_pct", 0.0) or 0.0)
    hints: List[str] = []
    if rate >= 70.0:
        hints.append("High reject rate: widen pairs or relax one gate at a time.")
    if dominant == "data_quality":
        hints.append("Data quality dominates: lower min valid bars ratio or increase max stale hours.")
    elif dominant == "insufficient_bars":
        hints.append("Insufficient bars dominates: reduce min bars required.")
    elif dominant == "spread":
        hints.append("Spread dominates: raise max spread bps slightly or focus on major pairs.")
    elif dominant == "low_volatility":
        hints.append("Low volatility dominates: lower minimum volatility threshold.")
    if not hints and counts:
        hints.append("Scanner healthy; tune score threshold for more/less selectivity.")
    return hints[:3]


_REJECT_REASON_PRIORITY = {
    "data_quality": 100,
    "insufficient_bars": 90,
    "spread": 80,
    "low_volatility": 70,
    "unknown": 10,
}


def _select_forex_mtf_confirmation_rows(
    scored: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], int, float]:
    rows = [row for row in list(scored or []) if isinstance(row, dict)]
    try:
        limit = max(0, int(float(settings.get("forex_mtf_confirm_max_pairs", 10) or 10)))
    except Exception:
        limit = 10
    if (limit <= 0) or (not rows):
        return [], int(limit), 0.0
    ranked = sorted(rows, key=lambda row: abs(_float(row.get("score", 0.0), 0.0)), reverse=True)
    try:
        base_threshold = max(0.02, float(settings.get("forex_score_threshold", 0.2) or 0.2))
    except Exception:
        base_threshold = 0.2
    near_ready_threshold = max(base_threshold * 1.50, 0.35)
    selected: List[Dict[str, Any]] = list(ranked[:limit])
    return selected, int(limit), float(near_ready_threshold)


def _apply_forex_mtf_confirmation(
    scored: List[Dict[str, Any]],
    client: OandaBrokerClient,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    rows = [row for row in list(scored or []) if isinstance(row, dict)]
    for row in rows:
        row["mtf_side"] = "unverified"
        row["mtf_confirmed"] = None
        row["mtf_source"] = "deferred"

    selected, limit, near_ready_threshold = _select_forex_mtf_confirmation_rows(rows, settings)
    if not selected:
        return {
            "limit": int(limit),
            "selected_pairs": 0,
            "h4_calls": 0,
            "deferred_pairs": int(len(rows)),
            "near_ready_threshold": float(round(near_ready_threshold, 6)),
        }

    h4_calls = 0
    selected_pairs: set[str] = set()
    for row in selected:
        pair = str(row.get("pair", "") or "").strip().upper()
        if (not pair) or (pair in selected_pairs):
            continue
        selected_pairs.add(pair)
        spread_bps = _float(row.get("spread_bps", 0.0), 0.0)
        mtf_side = "watch"
        mtf_source = "h4_remote"
        try:
            c4 = client.get_candles(pair, granularity="H4", count=40)
            h4_calls += 1
            m4 = _score_candles(pair, c4, spread_bps=spread_bps)
            if bool(m4.get("outlier", False)) or float(m4.get("score", -9999.0) or -9999.0) <= -9999.0:
                mtf_side = "watch"
            else:
                mtf_side = "long" if float(m4.get("score", 0.0) or 0.0) > 0 else "short"
        except Exception:
            mtf_side = "watch"
            mtf_source = "h4_error"
        row["mtf_side"] = mtf_side
        row["mtf_source"] = mtf_source
        row["mtf_confirmed"] = bool(str(row.get("side", "watch")).lower() == mtf_side)
        if not bool(row["mtf_confirmed"]):
            row["score"] = round(float(row.get("score", 0.0) or 0.0) * 0.75, 6)
            _append_reason_parts(
                row,
                logic="Multi-timeframe trend mismatch; reducing conviction",
                data=f"H1 side {str(row.get('side', 'watch')).upper()} vs H4 side {str(mtf_side).upper()}",
            )

    return {
        "limit": int(limit),
        "selected_pairs": int(len(selected_pairs)),
        "h4_calls": int(h4_calls),
        "deferred_pairs": int(max(0, len(rows) - len(selected_pairs))),
        "near_ready_threshold": float(round(near_ready_threshold, 6)),
    }


def _summarize_rejections(rejected: List[Dict[str, Any]], universe_size: int) -> Dict[str, Any]:
    best_by_pair: Dict[str, Dict[str, Any]] = {}
    for row in list(rejected or []):
        if not isinstance(row, dict):
            continue
        pair = str(row.get("pair", "") or "").strip().upper()
        reason = str(row.get("reason", "unknown") or "unknown").strip().lower() or "unknown"
        if not pair:
            continue
        cur = best_by_pair.get(pair)
        cur_reason = str((cur or {}).get("reason", "unknown") or "unknown").strip().lower() if isinstance(cur, dict) else "unknown"
        cur_pri = int(_REJECT_REASON_PRIORITY.get(cur_reason, 0))
        new_pri = int(_REJECT_REASON_PRIORITY.get(reason, 0))
        if (cur is None) or (new_pri > cur_pri):
            best_by_pair[pair] = {"pair": pair, "reason": reason}

    reason_counts: Dict[str, int] = {}
    for row in best_by_pair.values():
        reason = str(row.get("reason", "unknown") or "unknown")
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    total_unique = int(len(best_by_pair))
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


def run_scan(settings: Dict[str, Any], hub_dir: str) -> Dict[str, Any]:
    scan_start = time.perf_counter()

    def _scan_elapsed_ms() -> int:
        try:
            elapsed = (time.perf_counter() - float(scan_start)) * 1000.0
        except Exception:
            elapsed = 0.0
        return int(max(0.0, elapsed))

    prev_diag = _load_json_map(_scan_diag_path(hub_dir))
    prev_status = _load_json_map(_thinker_status_path(hub_dir))
    prev_candidates = _norm_id_list(prev_diag.get("candidate_pairs", []))
    prev_leaders = _norm_id_list(prev_diag.get("leader_pairs", []))
    prev_top_pair = str(prev_diag.get("top_pair", "") or "").strip().upper()
    if not prev_top_pair:
        prev_top = prev_status.get("top_pick", {}) if isinstance(prev_status.get("top_pick", {}), dict) else {}
        prev_top_pair = str(prev_top.get("pair", "") or "").strip().upper()
    session_ctx = forex_session_bias()
    account_id, token = get_oanda_creds(settings, base_dir=BASE_DIR)
    rest_url = str(settings.get("oanda_rest_url", "https://api-fxpractice.oanda.com") or "").strip().rstrip("/")
    ts_now = int(time.time())
    calendar_ctx = _load_forexfactory_context(hub_dir, settings, ts_now)
    calendar_events = list(calendar_ctx.get("events", []) or []) if isinstance(calendar_ctx.get("events", []), list) else []
    lookahead_min = max(5, int(float(settings.get("forex_event_max_lookahead_minutes", 180) or 180)))
    upcoming_high = 0
    upcoming_medium = 0
    for evt in calendar_events:
        if not isinstance(evt, dict):
            continue
        impact = _norm_impact(evt.get("impact", ""))
        if impact not in {"high", "medium"}:
            continue
        try:
            evt_ts = int(float(evt.get("ts", 0) or 0))
        except Exception:
            evt_ts = 0
        if evt_ts <= 0:
            continue
        delta_min = int(round((float(evt_ts) - float(ts_now)) / 60.0))
        if delta_min < 0 or delta_min > lookahead_min:
            continue
        if impact == "high":
            upcoming_high += 1
        elif impact == "medium":
            upcoming_medium += 1

    def _write_diag(
        state: str,
        msg: str,
        universe_total: int = 0,
        candidates_total: int = 0,
        scores_total: int = 0,
        leaders_total: int = 0,
        top_pair: str = "",
        top_score: float = 0.0,
        cooldown_active: int = 0,
        reject_summary: Dict[str, Any] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "ts": int(ts_now),
            "scan_elapsed_ms": int(_scan_elapsed_ms()),
            "state": str(state or ""),
            "market_open": True,  # FX runs 24/5; scanner has no equity-hours gate.
            "universe_total": int(universe_total),
            "candidates_total": int(candidates_total),
            "scores_total": int(scores_total),
            "leaders_total": int(leaders_total),
            "top_pair": str(top_pair or ""),
            "top_score": float(top_score or 0.0),
            "cooldown_active": int(cooldown_active),
            "msg": str(msg or ""),
            "reject_summary": dict(reject_summary or {}),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        _save_scan_diagnostics(hub_dir, payload)

    if not account_id or not token or not rest_url:
        _write_diag("NOT CONFIGURED", "Add OANDA account/token in Settings")
        return {
            "state": "NOT CONFIGURED",
            "ai_state": "Credentials missing",
            "msg": "Add OANDA account/token in Settings",
            "universe": list(DEFAULT_FX_UNIVERSE),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
        }

    client = OandaBrokerClient(account_id=account_id, api_token=token, rest_url=rest_url)
    parsed = _parse_pairs(settings)
    if parsed:
        universe = parsed
    else:
        universe = client.list_tradeable_instruments() or list(DEFAULT_FX_UNIVERSE)
    max_scan = max(4, int(float(settings.get("forex_scan_max_pairs", 24) or 24)))
    universe = universe[:max_scan]
    open_position_pairs = _load_open_position_pairs(hub_dir)

    max_spread_bps = max(0.0, float(settings.get("forex_max_spread_bps", 8.0) or 8.0))
    min_volatility_pct = max(0.0, float(settings.get("forex_min_volatility_pct", 0.01) or 0.01))
    min_bars_required = max(8, int(float(settings.get("forex_min_bars_required", 24) or 24)))
    pricing_fetch_started = time.perf_counter()
    price_rows = client.get_pricing_details(universe)
    pricing_fetch_ms = int(max(0.0, (time.perf_counter() - float(pricing_fetch_started)) * 1000.0))
    cooldown_state = _load_json_map(_pair_cooldown_path(hub_dir))
    cooldown_map = cooldown_state.get("pairs", {}) if isinstance(cooldown_state.get("pairs", {}), dict) else {}
    cooldown_map = _prune_cooldown_map(cooldown_map, ts_now)
    candidates: List[str] = []
    rejected: List[Dict[str, Any]] = []
    for pair in universe:
        c_row = cooldown_map.get(pair, {}) if isinstance(cooldown_map.get(pair, {}), dict) else {}
        if int(c_row.get("until", 0) or 0) > ts_now:
            rejected.append({"pair": pair, "reason": "cooldown", "cooldown_until": int(c_row.get("until", 0) or 0)})
            continue
        p = price_rows.get(pair, {})
        spr = _float(p.get("spread_bps", 0.0), 0.0)
        if max_spread_bps > 0.0 and spr > max_spread_bps:
            rejected.append({"pair": pair, "reason": "spread", "spread_bps": spr})
            _apply_pair_cooldown(cooldown_map, pair, "spread", settings, ts_now)
            continue
        candidates.append(pair)
    scored = []
    candles_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    min_valid_ratio = max(0.0, min(1.0, float(settings.get("forex_min_valid_bars_ratio", 0.70) or 0.70)))
    max_stale_hours = max(0.5, float(settings.get("forex_max_stale_hours", 8.0) or 8.0))
    session_weighted_candidates = 0
    h1_scoring_loop_ms = 0
    h4_confirmation_loop_ms = 0
    chart_hydration_loop_ms = 0
    mtf_confirmation = {
        "limit": 0,
        "selected_pairs": 0,
        "h4_calls": 0,
        "deferred_pairs": 0,
        "near_ready_threshold": 0.0,
    }
    try:
        h1_loop_started = time.perf_counter()
        for pair in candidates:
            params = urllib.parse.urlencode({"price": "M", "granularity": "H1", "count": "48"})
            url = f"{rest_url}/v3/instruments/{pair}/candles?{params}"
            payload = _request_json(url, headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
            candles = payload.get("candles", []) or []
            if not isinstance(candles, list):
                candles = []
            bars_count = int(len(candles))
            if bars_count < min_bars_required:
                rejected.append(
                    {
                        "pair": pair,
                        "reason": "insufficient_bars",
                        "bars_count": bars_count,
                        "source": "oanda_h1",
                        "min_bars_required": min_bars_required,
                    }
                )
                _apply_pair_cooldown(cooldown_map, pair, "insufficient_bars", settings, ts_now)
                continue
            spread_bps = _float((price_rows.get(pair, {}) or {}).get("spread_bps", 0.0), 0.0)
            row = _score_candles(pair, candles, spread_bps=spread_bps)
            row["spread_bps"] = round(spread_bps, 4)
            row["bars_count"] = bars_count
            row["data_source"] = "oanda_h1"
            if bool(row.get("outlier", False)):
                rejected.append(
                    {
                        "pair": pair,
                        "reason": "outlier_jump",
                        "change_6h_pct": row.get("change_6h_pct"),
                        "change_24h_pct": row.get("change_24h_pct"),
                        "volatility_pct": row.get("volatility_pct"),
                        "source": row.get("data_source"),
                    }
                )
                _apply_pair_cooldown(cooldown_map, pair, "data_quality", settings, ts_now)
                continue
            q = _bar_quality(candles)
            row["valid_ratio"] = round(float(q.get("valid_ratio", 0.0)), 4)
            row["stale_hours"] = round(float(q.get("stale_hours", 9999.0)), 3)
            row["data_quality_ok"] = bool((row["valid_ratio"] >= min_valid_ratio) and (row["stale_hours"] <= max_stale_hours))
            if _float(row.get("volatility_pct", 0.0), 0.0) < min_volatility_pct:
                rejected.append({"pair": pair, "reason": "low_volatility", "volatility_pct": row.get("volatility_pct", 0.0)})
                _apply_pair_cooldown(cooldown_map, pair, "low_volatility", settings, ts_now)
                continue
            if not row["data_quality_ok"]:
                rejected.append(
                    {
                        "pair": pair,
                        "reason": "data_quality",
                        "valid_ratio": row.get("valid_ratio"),
                        "stale_hours": row.get("stale_hours"),
                        "bars_count": row.get("bars_count"),
                        "source": row.get("data_source"),
                    }
                )
                _apply_pair_cooldown(cooldown_map, pair, "data_quality", settings, ts_now)
                continue
            candles_by_pair[pair] = list(candles or [])
            scored.append(row)
        h1_scoring_loop_ms = int(max(0.0, (time.perf_counter() - float(h1_loop_started)) * 1000.0))

        h4_loop_started = time.perf_counter()
        mtf_confirmation = _apply_forex_mtf_confirmation(scored, client, settings)
        h4_confirmation_loop_ms = int(max(0.0, (time.perf_counter() - float(h4_loop_started)) * 1000.0))

        for row in scored:
            pair = str(row.get("pair", "") or "").strip().upper()
            mult, session_mode = _session_weight_multiplier(settings, str(row.get("side", "watch")), session_ctx)
            row["session_name"] = str((session_ctx or {}).get("session", "N/A") or "N/A")
            row["session_bias"] = str((session_ctx or {}).get("bias", "FLAT") or "FLAT")
            row["session_weight_mode"] = str(session_mode)
            row["session_weight_mult"] = round(float(mult), 4)
            row["score_raw"] = round(float(row.get("score", 0.0) or 0.0), 6)
            if abs(float(mult) - 1.0) > 1e-9:
                row["score"] = round(float(row.get("score", 0.0) or 0.0) * float(mult), 6)
                if str(session_mode) == "trend_boost":
                    session_logic = "Session regime supports trend continuation"
                elif str(session_mode) == "range_dampen":
                    session_logic = "Session regime is range-prone; trend conviction reduced"
                else:
                    session_logic = "Session weighting adjusted confidence"
                _append_reason_parts(
                    row,
                    logic=session_logic,
                    data=f"session {row['session_name']} | bias {row['session_bias']} | mult x{float(mult):.2f}",
                )
                session_weighted_candidates += 1
            event_risk = _pair_event_risk(pair, calendar_ctx, ts_now, settings)
            row["event_risk"] = dict(event_risk)
            row["event_risk_active"] = bool(event_risk.get("active", False))
            row["event_risk_severity"] = str(event_risk.get("severity", "none") or "none")
            row["event_risk_block_entry"] = bool(event_risk.get("block_entry", False))
            row["event_risk_score_mult"] = round(float(event_risk.get("score_mult", 1.0) or 1.0), 4)
            row["event_risk_minutes_to_event"] = int(event_risk.get("minutes_to_event", 99999) or 99999)
            if row["event_risk_active"]:
                ev_mult = max(0.10, min(1.0, float(event_risk.get("score_mult", 1.0) or 1.0)))
                row["score"] = round(float(row.get("score", 0.0) or 0.0) * ev_mult, 6)
                _append_reason_parts(
                    row,
                    logic=str(event_risk.get("logic", "") or ""),
                    data=str(event_risk.get("data", "") or ""),
                )

        # Keep chart hydration independent from eligibility gates: open positions should
        # always have chart bars available, even when currently filtered out of leaders.
        chart_hydration_started = time.perf_counter()
        for pair in list(open_position_pairs or []):
            p = str(pair or "").strip().upper()
            if not p:
                continue
            if len(list(candles_by_pair.get(p, []) or [])) >= 2:
                continue
            try:
                params = urllib.parse.urlencode({"price": "M", "granularity": "H1", "count": "48"})
                url = f"{rest_url}/v3/instruments/{p}/candles?{params}"
                payload = _request_json(url, headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
                candles = payload.get("candles", []) or []
                if isinstance(candles, list) and len(candles) >= 2:
                    candles_by_pair[p] = list(candles)
            except Exception:
                continue
        chart_hydration_loop_ms = int(max(0.0, (time.perf_counter() - float(chart_hydration_started)) * 1000.0))

        scored.sort(key=lambda row: abs(float(row.get("score", 0.0))), reverse=True)
        outcome_map = _compute_outcome_map(hub_dir)
        market_calibration_samples = _market_pooled_calibration_samples(hub_dir, settings)
        for row in scored:
            pair = str(row.get("pair", "") or "").strip().upper()
            m = outcome_map.get(pair, {})
            hr = float(m.get("hit_rate_pct", 50.0) or 50.0)
            ap = float(m.get("avg_pnl_pct", 0.0) or 0.0)
            smp = int(float(m.get("samples", 0.0) or 0.0))
            row["hit_rate_pct"] = round(hr, 2)
            row["avg_pnl_pct"] = round(ap, 4)
            row["samples"] = smp
            row["calib_prob"] = _calibrated_prob(float(row.get("score", 0.0) or 0.0), hr, ap)
            row["pair_samples"] = int(smp)
            row["market_calibration_samples"] = int(market_calibration_samples)
            row["calibration_scope"] = "pair"
            row["calibration_effective_samples"] = int(smp)
            row["calibration_effective_prob"] = float(row.get("calib_prob", 0.0) or 0.0)
            row["quality_score"] = round(
                (100.0 * float(row.get("valid_ratio", 0.0)))
                - (3.0 * float(row.get("spread_bps", 0.0)))
                + (0.7 * hr),
                3,
            )
        ranked_health = sorted(scored, key=lambda r: float(r.get("quality_score", -9999.0)), reverse=True)
        exec_n = max(6, int(len(ranked_health) * 0.40))
        exec_bucket = {str(r.get("pair", "")).strip().upper() for r in ranked_health[:exec_n]}
        for row in scored:
            row["entry_gate_reason"] = ""
            row["eligible_for_entry"] = str(row.get("pair", "")).strip().upper() in exec_bucket
            if not row["eligible_for_entry"]:
                _append_reason_parts(
                    row,
                    logic="Filtered by universe quality gate; hold as watch",
                    data="ranked outside execution-quality bucket",
                )
                row["side"] = "watch"
            if bool(row.get("event_risk_block_entry", False)):
                row["eligible_for_entry"] = False
                row["side"] = "watch"
                _append_reason_parts(
                    row,
                    logic="Entry paused around nearby high-impact macro event",
                    data=f"event window block | severity {str(row.get('event_risk_severity', 'none')).upper()}",
                )
            elif str(row.get("side", "watch") or "watch").strip().lower() in {"long", "short"}:
                min_samples_guarded = max(0, int(float(settings.get("forex_min_samples_live_guarded", 5) or 5)))
                pair_samples = int(float(row.get("pair_samples", row.get("samples", 0)) or 0))
                pooled_samples = int(float(row.get("market_calibration_samples", 0) or 0))
                if pair_samples < min_samples_guarded and pooled_samples >= min_samples_guarded:
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
        leaders = sorted(
            scored,
            key=lambda r: (
                1 if bool(r.get("eligible_for_entry", False)) else 0,
                float(r.get("leader_rank_score", abs(float(r.get("score", 0.0) or 0.0))) or 0.0),
            ),
            reverse=True,
        )[:10]
        event_risk_active_count = int(sum(1 for row in scored if bool((row or {}).get("event_risk_active", False))))
        event_risk_block_count = int(sum(1 for row in scored if bool((row or {}).get("event_risk_block_entry", False))))
        event_context = {
            "enabled": bool(calendar_ctx.get("enabled", False)),
            "state": str(calendar_ctx.get("state", "disabled") or "disabled"),
            "error": str(calendar_ctx.get("error", "") or ""),
            "cache_age_s": int(calendar_ctx.get("cache_age_s", -1) or -1),
            "refresh_scheduled": bool(calendar_ctx.get("refresh_scheduled", False)),
            "refresh_state": str(calendar_ctx.get("refresh_state", "not_needed") or "not_needed"),
            "events_total": int(len(calendar_events)),
            "upcoming_high": int(upcoming_high),
            "upcoming_medium": int(upcoming_medium),
            "active_candidates": int(event_risk_active_count),
            "blocked_candidates": int(event_risk_block_count),
        }
        try:
            stability_margin = max(0.0, min(100.0, float(settings.get("forex_leader_stability_margin_pct", 12.0) or 12.0)))
        except Exception:
            stability_margin = 12.0
        leaders, stability_applied = _apply_leader_hysteresis(leaders, prev_top_pair, stability_margin)
        top_pick = leaders[0] if leaders else None
        msg = "No FX leaders"
        if top_pick:
            msg = f"Top pair {top_pick['pair']} | {top_pick['side']} | {top_pick['reason']}"
        vols = [float(r.get("volatility_pct", 0.0) or 0.0) for r in scored if float(r.get("volatility_pct", 0.0) or 0.0) > 0]
        vol_med = (sorted(vols)[len(vols) // 2] if vols else 0.0)
        base_thr = max(0.02, float(settings.get("forex_score_threshold", 0.2) or 0.2))
        volatility_threshold = float(round(base_thr * (1.20 if vol_med >= 0.12 else 1.0), 6))
        replay_enabled = bool(settings.get("forex_replay_adaptive_enabled", True))
        replay_weight = max(0.0, min(1.0, float(settings.get("forex_replay_adaptive_weight", 0.35) or 0.35)))
        replay_step_cap_pct = max(5.0, min(90.0, float(settings.get("forex_replay_adaptive_step_cap_pct", 40.0) or 40.0)))
        replay_target_entries = replay_target_entries_for_market(settings, "forex")
        replay_recommended = float(volatility_threshold)
        replay_clamped = float(volatility_threshold)
        replay_reason = ""
        if replay_enabled and scored:
            replay_payload = recommend_threshold_from_scores(
                scored,
                market="forex",
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
        top_pair = str((top_pick or {}).get("pair", "") or "").strip().upper()
        chart_seed: List[Dict[str, Any]] = []
        if isinstance(top_pick, dict):
            chart_seed.append(dict(top_pick))
        for pair in list(open_position_pairs or []):
            p = str(pair or "").strip().upper()
            if p:
                chart_seed.append({"pair": p})
        chart_seed.extend([dict(r) for r in leaders[:10] if isinstance(r, dict)])
        chart_map_pairs = max(
            2,
            int(float(settings.get("market_chart_cache_symbols", 8) or 8)),
            int(len([p for p in list(open_position_pairs or []) if str(p or "").strip()])),
        )
        chart_map_bars = max(40, int(float(settings.get("market_chart_cache_bars", 120) or 120)))
        prev_chart_map_raw = prev_status.get("top_chart_map", {}) if isinstance(prev_status.get("top_chart_map", {}), dict) else {}
        prev_chart_map = dict(prev_chart_map_raw) if isinstance(prev_chart_map_raw, dict) else {}
        top_chart_map = _build_top_chart_map(
            chart_seed,
            candles_by_pair,
            max_pairs=chart_map_pairs,
            limit=chart_map_bars,
            include_pairs=list(open_position_pairs or []),
        )
        if prev_chart_map and open_position_pairs:
            for pair in list(open_position_pairs or []):
                p = str(pair or "").strip().upper()
                if (not p) or (p in top_chart_map):
                    continue
                cached_rows = _compact_chart_bars(list(prev_chart_map.get(p, []) or []), limit=chart_map_bars)
                if len(cached_rows) >= 2:
                    top_chart_map[p] = cached_rows
                if len(top_chart_map) >= chart_map_pairs:
                    break
        top_chart = list(top_chart_map.get(top_pair, []) or [])
        _append_jsonl(
            _rankings_path(hub_dir),
            {
                "ts": ts_now,
                "state": "READY",
                "universe_total": len(universe),
                "candidates": len(candidates),
                "rejected": rejected[:100],
                "top": leaders[:20],
            },
        )
        reject_summary = _summarize_rejections(rejected, len(universe))
        leader_pairs = [str((row or {}).get("pair", "") or "").strip().upper() for row in leaders if isinstance(row, dict)]
        candidate_churn_pct = turnover_pct(prev_candidates, candidates)
        leader_churn_pct = turnover_pct(prev_leaders, leader_pairs)
        quality_report = build_universe_quality_report(
            market="forex",
            ts=int(ts_now),
            mode="intraday",
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
        reject_warn_pct = max(10.0, float(settings.get("forex_reject_drift_warn_pct", 65.0) or 65.0))
        drift_warning = bool((reject_rate_pct >= reject_warn_pct) and (dominant_ratio >= 0.60))
        scan_phase_timing = {
            "pricing_fetch": int(pricing_fetch_ms),
            "h1_scoring_loop": int(h1_scoring_loop_ms),
            "h4_confirmation_loop": int(h4_confirmation_loop_ms),
            "chart_hydration_loop": int(chart_hydration_loop_ms),
        }
        _write_diag(
            "READY",
            msg,
            universe_total=len(universe),
            candidates_total=len(candidates),
            scores_total=len(scored),
            leaders_total=len(leaders),
            top_pair=str((top_pick or {}).get("pair", "") or ""),
            top_score=float((top_pick or {}).get("score", 0.0) or 0.0),
            cooldown_active=int(sum(1 for v in (cooldown_map or {}).values() if int((v or {}).get("until", 0) or 0) > ts_now)),
            reject_summary=reject_summary,
            extra={
                "candidate_pairs": list(candidates),
                "leader_pairs": list(leader_pairs),
                "candidate_churn_pct": float(candidate_churn_pct),
                "leader_churn_pct": float(leader_churn_pct),
                "leader_stability_applied": bool(stability_applied),
                "leader_stability_prev_pair": str(prev_top_pair),
                "session_context": dict(session_ctx),
                "session_weighted_candidates": int(session_weighted_candidates),
                "quality_summary": str(quality_report.get("summary", "") or ""),
                "event_context": dict(event_context),
                "adaptive_threshold_base": float(base_thr),
                "adaptive_threshold_volatility": float(round(volatility_threshold, 6)),
                "adaptive_threshold_replay_recommended": float(round(replay_recommended, 6)),
                "adaptive_threshold_replay_clamped": float(round(replay_clamped, 6)),
                "adaptive_threshold_replay_weight": float(round(effective_weight, 4)),
                "adaptive_threshold_replay_target_entries": int(replay_target_entries),
                "adaptive_threshold_replay_reason": str(replay_reason),
                "adaptive_threshold_replay_enabled": bool(replay_enabled),
                "scan_phase_timing_ms": dict(scan_phase_timing),
                "mtf_confirmation": dict(mtf_confirmation),
            },
        )
        _save_json_map(_pair_cooldown_path(hub_dir), {"ts": int(time.time()), "pairs": cooldown_map})
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
        if replay_enabled:
            hints.append(
                f"Adaptive threshold {volatility_threshold:.3f} -> {adaptive_threshold:.3f} "
                f"(replay target {int(replay_target_entries)})."
            )
        if event_risk_active_count > 0:
            hints.append(f"Macro-event risk filter active on {event_risk_active_count} pair(s).")
        if str(event_context.get("state", "") or "") in {"unavailable", "cooldown"}:
            hints.append("ForexFactory event feed unavailable; running scanner without event dampener.")
        if int(mtf_confirmation.get("deferred_pairs", 0) or 0) > 0:
            hints.append(
                f"H4 confirmation budgeted: {int(mtf_confirmation.get('selected_pairs', 0) or 0)} checked, "
                f"{int(mtf_confirmation.get('deferred_pairs', 0) or 0)} deferred."
            )
        return {
            "state": "READY",
            "ai_state": "Scan ready",
            "msg": msg,
            "universe": list(candidates),
            "leaders": leaders[:10],
            "all_scores": scored[:40],
            "top_pick": top_pick,
            "top_chart": top_chart,
            "top_chart_map": top_chart_map,
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
            "rejected": rejected[:30],
            "reject_summary": reject_summary,
            "cooldown_active": int(sum(1 for v in (cooldown_map or {}).values() if int((v or {}).get("until", 0) or 0) > ts_now)),
            "hints": hints[:5],
            "candidate_churn_pct": float(candidate_churn_pct),
            "leader_churn_pct": float(leader_churn_pct),
            "leader_stability_applied": bool(stability_applied),
            "leader_stability_prev_pair": str(prev_top_pair),
            "session_context": dict(session_ctx),
            "session_weighted_candidates": int(session_weighted_candidates),
            "universe_quality": quality_report,
            "event_context": dict(event_context),
            "scan_phase_timing_ms": dict(scan_phase_timing),
            "mtf_confirmation": dict(mtf_confirmation),
            "health": {"data_ok": True, "broker_ok": True, "orders_ok": True, "drift_warning": drift_warning},
        }
    except urllib.error.HTTPError as exc:
        err_msg = f"HTTP {exc.code}: {exc.reason}"
        _append_jsonl(_rankings_path(hub_dir), {"ts": ts_now, "state": "ERROR", "reason": err_msg})
        fallback = _cached_scan_fallback(hub_dir, ts_now, err_msg, universe=list(universe), session_ctx=session_ctx)
        if fallback:
            _write_diag(
                "READY",
                str(fallback.get("msg", "") or err_msg),
                universe_total=len(list(fallback.get("universe", []) or [])),
                candidates_total=int(len(list(fallback.get("universe", []) or []))),
                scores_total=int(len(list(fallback.get("all_scores", []) or []))),
                leaders_total=int(len(list(fallback.get("leaders", []) or []))),
                top_pair=str(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("pair", "")) or ""),
                top_score=float(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("score", 0.0)) or 0.0),
                reject_summary=(dict(fallback.get("reject_summary", {})) if isinstance(fallback.get("reject_summary", {}), dict) else {}),
                extra={
                    "fallback_cached": True,
                    "leader_stability_applied": bool(fallback.get("leader_stability_applied", False)),
                    "leader_stability_prev_pair": str(fallback.get("leader_stability_prev_pair", prev_top_pair) or prev_top_pair),
                    "event_context": (dict(fallback.get("event_context", {})) if isinstance(fallback.get("event_context", {}), dict) else {}),
                },
            )
            return fallback
        _write_diag("ERROR", err_msg, universe_total=len(universe))
        _save_json_map(_pair_cooldown_path(hub_dir), {"ts": int(time.time()), "pairs": cooldown_map})
        return {
            "state": "ERROR",
            "ai_state": "HTTP error",
            "msg": err_msg,
            "universe": list(universe),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
        }
    except urllib.error.URLError as exc:
        err_msg = f"Network error: {exc.reason}"
        _append_jsonl(_rankings_path(hub_dir), {"ts": ts_now, "state": "ERROR", "reason": err_msg})
        fallback = _cached_scan_fallback(hub_dir, ts_now, err_msg, universe=list(universe), session_ctx=session_ctx)
        if fallback:
            _write_diag(
                "READY",
                str(fallback.get("msg", "") or err_msg),
                universe_total=len(list(fallback.get("universe", []) or [])),
                candidates_total=int(len(list(fallback.get("universe", []) or []))),
                scores_total=int(len(list(fallback.get("all_scores", []) or []))),
                leaders_total=int(len(list(fallback.get("leaders", []) or []))),
                top_pair=str(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("pair", "")) or ""),
                top_score=float(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("score", 0.0)) or 0.0),
                reject_summary=(dict(fallback.get("reject_summary", {})) if isinstance(fallback.get("reject_summary", {}), dict) else {}),
                extra={
                    "fallback_cached": True,
                    "leader_stability_applied": bool(fallback.get("leader_stability_applied", False)),
                    "leader_stability_prev_pair": str(fallback.get("leader_stability_prev_pair", prev_top_pair) or prev_top_pair),
                    "event_context": (dict(fallback.get("event_context", {})) if isinstance(fallback.get("event_context", {}), dict) else {}),
                },
            )
            return fallback
        _write_diag("ERROR", err_msg, universe_total=len(universe))
        _save_json_map(_pair_cooldown_path(hub_dir), {"ts": int(time.time()), "pairs": cooldown_map})
        return {
            "state": "ERROR",
            "ai_state": "Network error",
            "msg": err_msg,
            "universe": list(universe),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
        }
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        _append_jsonl(_rankings_path(hub_dir), {"ts": ts_now, "state": "ERROR", "reason": err_msg})
        fallback = _cached_scan_fallback(hub_dir, ts_now, err_msg, universe=list(universe), session_ctx=session_ctx)
        if fallback:
            _write_diag(
                "READY",
                str(fallback.get("msg", "") or err_msg),
                universe_total=len(list(fallback.get("universe", []) or [])),
                candidates_total=int(len(list(fallback.get("universe", []) or []))),
                scores_total=int(len(list(fallback.get("all_scores", []) or []))),
                leaders_total=int(len(list(fallback.get("leaders", []) or []))),
                top_pair=str(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("pair", "")) or ""),
                top_score=float(((fallback.get("top_pick", {}) if isinstance(fallback.get("top_pick", {}), dict) else {}).get("score", 0.0)) or 0.0),
                reject_summary=(dict(fallback.get("reject_summary", {})) if isinstance(fallback.get("reject_summary", {}), dict) else {}),
                extra={
                    "fallback_cached": True,
                    "leader_stability_applied": bool(fallback.get("leader_stability_applied", False)),
                    "leader_stability_prev_pair": str(fallback.get("leader_stability_prev_pair", prev_top_pair) or prev_top_pair),
                    "event_context": (dict(fallback.get("event_context", {})) if isinstance(fallback.get("event_context", {}), dict) else {}),
                },
            )
            return fallback
        _write_diag("ERROR", err_msg, universe_total=len(universe))
        _save_json_map(_pair_cooldown_path(hub_dir), {"ts": int(time.time()), "pairs": cooldown_map})
        return {
            "state": "ERROR",
            "ai_state": "Scan failed",
            "msg": err_msg,
            "universe": list(universe),
            "leaders": [],
            "all_scores": [],
            "top_chart": [],
            "top_chart_map": {},
            "updated_at": ts_now,
        }


def main() -> int:
    print("forex_thinker.py is designed to be imported by the hub/runner first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
