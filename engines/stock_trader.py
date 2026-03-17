from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from app.credential_utils import get_alpaca_creds
from app.http_utils import parse_retry_after_value
from app.path_utils import resolve_runtime_paths
from app.runtime_logging import runtime_event
from app.scanner_quality import effective_reject_pressure
from brokers.broker_alpaca import AlpacaBrokerClient

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "stock_trader")
ROLLOUT_ORDER = {
    "legacy": 0,
    "scan_expanded": 1,
    "risk_caps": 2,
    "execution_v2": 3,
    "shadow_only": 4,
    "live": 5,
    "live_guarded": 5,
}


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _rollout_at_least(settings: Dict[str, Any], stage: str) -> bool:
    cur = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    target = str(stage or "").strip().lower()
    if cur == "live_guarded":
        cur = "live"
    if target == "live_guarded":
        target = "live"
    return int(ROLLOUT_ORDER.get(cur, 0)) >= int(ROLLOUT_ORDER.get(target, 0))


def _broker_mode_label(settings: Dict[str, Any]) -> str:
    return "Paper" if bool(settings.get("alpaca_paper_mode", True)) else "Live"


def _trader_state_label(settings: Dict[str, Any], auto_enabled: bool, shadow_only: bool) -> str:
    mode = _broker_mode_label(settings)
    if auto_enabled:
        return f"{mode} shadow-run" if shadow_only else f"{mode} auto-run"
    return f"{mode} manual-ready"


def _parse_positions(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        try:
            qty = float(row.get("qty", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        try:
            avg = float(row.get("avg_entry_price", 0.0) or 0.0)
        except Exception:
            avg = 0.0
        try:
            mv = float(row.get("market_value", 0.0) or 0.0)
        except Exception:
            mv = 0.0
        out[symbol] = {"symbol": symbol, "qty": qty, "avg_entry_price": avg, "market_value": mv}
    return out


def _now_et() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))


def _et_date_from_ts(ts: float | int) -> date:
    try:
        val = float(ts or 0.0)
    except Exception:
        val = 0.0
    if val <= 0.0:
        return _now_et().date()
    return datetime.fromtimestamp(val, timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


def _same_et_day(ts_a: float | int, ts_b: float | int) -> bool:
    return _et_date_from_ts(ts_a) == _et_date_from_ts(ts_b)


def _normalize_event_timestamps(raw: Any, now_ts: int, keep_days: int = 45) -> List[int]:
    seq = raw if isinstance(raw, list) else []
    out: List[int] = []
    cutoff_ts = max(0, int(now_ts - (max(1, int(keep_days)) * 86400)))
    for item in seq:
        try:
            ts = int(float(item or 0))
        except Exception:
            continue
        if ts <= 0:
            continue
        if ts < cutoff_ts:
            continue
        if ts > int(now_ts + 300):
            continue
        out.append(ts)
    out = sorted(set(out))
    return out[-500:]


def _count_events_on_et_day(events: List[int], ref_ts: int) -> int:
    if not events:
        return 0
    ref_day = _et_date_from_ts(ref_ts)
    count = 0
    for ts in events:
        if _et_date_from_ts(ts) == ref_day:
            count += 1
    return int(count)


def _pdt_window_start_ts(now_ts: int, trading_days: int = 5) -> int:
    local = datetime.fromtimestamp(max(0, int(now_ts)), timezone.utc).astimezone(ZoneInfo("America/New_York"))
    target_days = max(1, int(trading_days))
    cursor = local.date()
    counted = 1 if cursor.weekday() < 5 else 0
    while counted < target_days:
        cursor = cursor - timedelta(days=1)
        if cursor.weekday() < 5:
            counted += 1
    start_local = datetime(cursor.year, cursor.month, cursor.day, 0, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    return int(start_local.astimezone(timezone.utc).timestamp())


def _count_events_in_window(events: List[int], start_ts: int, end_ts: int) -> int:
    count = 0
    lo = int(start_ts)
    hi = int(end_ts)
    for ts in events:
        if lo <= int(ts) <= hi:
            count += 1
    return int(count)


def _market_open_now() -> bool:
    now = _now_et()
    if now.weekday() >= 5:
        return False
    mins = (now.hour * 60) + now.minute
    return (9 * 60 + 30) <= mins < (16 * 60)


def _near_close_blocked(settings: Dict[str, Any]) -> bool:
    if not bool(settings.get("stock_block_new_entries_near_close", True)):
        return False
    mins_to_close = max(0, int(float(settings.get("stock_no_new_entries_mins_to_close", 15) or 15)))
    now = _now_et()
    if now.weekday() >= 5:
        return True
    cur = now.hour * 60 + now.minute
    close = 16 * 60
    return (close - cur) <= mins_to_close


def _daily_loss_guard_triggered(
    audit_path: str,
    max_daily_loss_usd: float,
    max_daily_loss_pct: float,
    equity: float,
) -> bool:
    if max_daily_loss_usd <= 0.0 and max_daily_loss_pct <= 0.0:
        return False
    today = time.strftime("%Y-%m-%d", time.localtime())
    loss_usd = 0.0
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if str(row.get("date", "") or "") != today:
                    continue
                if str(row.get("event", "")).lower() not in {"exit", "shadow_exit"}:
                    continue
                pnl_usd = float(row.get("pnl_usd", 0.0) or 0.0)
                if pnl_usd < 0:
                    loss_usd += abs(pnl_usd)
    except Exception:
        return False
    if max_daily_loss_usd > 0.0 and loss_usd >= max_daily_loss_usd:
        return True
    if max_daily_loss_pct > 0.0 and equity > 0.0 and ((loss_usd / equity) * 100.0) >= max_daily_loss_pct:
        return True
    return False


def _parse_order_id(msg: str, payload: Dict[str, Any]) -> str:
    oid = ""
    try:
        oid = str((payload or {}).get("id", "") or "").strip()
    except Exception:
        oid = ""
    if oid:
        return oid
    txt = str(msg or "")
    if "order_id=" in txt:
        return txt.split("order_id=", 1)[1].strip().split(" ", 1)[0]
    return ""


def _safe_float_from_dict(d: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        try:
            if k in d:
                return float(d.get(k, 0.0) or 0.0)
        except Exception:
            continue
    return 0.0


def _crypto_holdings_usd(hub_dir: str) -> float:
    path = os.path.join(hub_dir, "trader_status.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict):
        return 0.0
    return _safe_float_from_dict(
        data,
        [
            "total_holdings_value",
            "holdings_value",
            "holdingsValue",
            "holdings_usd",
        ],
    )


def _market_status_exposure_usd(hub_dir: str, market_key: str) -> float:
    path = os.path.join(hub_dir, market_key, f"{market_key[:-1] if market_key.endswith('s') else market_key}_trader_status.json")
    # Explicit fallback names for current file layout.
    if market_key == "forex":
        path = os.path.join(hub_dir, "forex", "forex_trader_status.json")
    elif market_key == "stocks":
        path = os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict):
        return 0.0
    return _safe_float_from_dict(data, ["exposure_usd", "total_positions_value_usd", "positions_value_usd"])


def _stock_candidates_from_thinker(thinker: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("leaders", "all_scores"):
        payload = thinker.get(key, []) if isinstance(thinker, dict) else []
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "") or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            out = dict(row)
            out["symbol"] = symbol
            seen.add(symbol)
            rows.append(out)
    top = thinker.get("top_pick", {}) if isinstance(thinker, dict) else {}
    if isinstance(top, dict):
        symbol = str(top.get("symbol", "") or "").strip().upper()
        if symbol and symbol not in seen:
            out = dict(top)
            out["symbol"] = symbol
            rows.append(out)
    rows.sort(key=lambda r: float(r.get("score", -9999.0) or -9999.0), reverse=True)
    return rows


def _stock_entry_priority(row: Dict[str, Any]) -> float:
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return float(default)

    side = str(row.get("side", "watch") or "watch").strip().lower()
    side_bonus = 45.0 if side == "long" else -45.0
    eligible_bonus = 22.0 if bool(row.get("eligible_for_entry", True)) else -22.0
    score = _f("score", 0.0)
    calib = _f("calib_prob", 0.5)
    quality = _f("quality_score", 0.0)
    spread = _f("spread_bps", 0.0)
    bars = max(0.0, _f("bars_count", 0.0))
    return side_bonus + eligible_bonus + (score * 18.0) + (calib * 12.0) + (quality * 0.06) + (min(240.0, bars) * 0.02) - (spread * 0.08)


def _fail_reason_summary(reasons: List[str], max_items: int = 4) -> tuple[str, Dict[str, int]]:
    buckets: Dict[str, int] = {}
    for raw in list(reasons or []):
        txt = str(raw or "").strip()
        if not txt:
            continue
        head = txt.split(" for ", 1)[0].strip()
        head = head.split("(", 1)[0].strip()
        if not head:
            head = txt[:48]
        buckets[head] = int(buckets.get(head, 0)) + 1
    if not buckets:
        return "", {}
    ranked = sorted(buckets.items(), key=lambda item: (-int(item[1]), str(item[0])))
    top = ranked[0]
    clipped = {str(k): int(v) for k, v in ranked[: max(1, int(max_items))]}
    return f"{str(top[0])} x{int(top[1])}", clipped


def _effective_reject_pressure(settings: Dict[str, Any], thinker: Dict[str, Any]) -> tuple[float, float]:
    reject_summary = thinker.get("reject_summary", {}) if isinstance(thinker.get("reject_summary", {}), dict) else {}
    try:
        raw_rate = max(0.0, float(reject_summary.get("reject_rate_pct", 0.0) or 0.0))
    except Exception:
        raw_rate = 0.0
    dominant_reason = str(reject_summary.get("dominant_reason", "") or "").strip().lower()
    try:
        dominant_ratio_pct = max(0.0, float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0))
    except Exception:
        dominant_ratio_pct = 0.0
    leaders_total = int(len(list(thinker.get("leaders", []) or []))) if isinstance(thinker, dict) else 0
    scores_total = int(len(list(thinker.get("all_scores", []) or []))) if isinstance(thinker, dict) else 0
    effective = effective_reject_pressure(
        raw_rate,
        dominant_reason=dominant_reason,
        dominant_ratio_pct=dominant_ratio_pct,
        leaders_total=leaders_total,
        scores_total=scores_total,
        unknown_dom_cap_pct=100.0,
    )
    return effective, raw_rate


def run_step(settings: Dict[str, Any], hub_dir: str) -> Dict[str, Any]:
    stocks_dir = os.path.join(hub_dir, "stocks")
    os.makedirs(stocks_dir, exist_ok=True)
    thinker_path = os.path.join(stocks_dir, "stock_thinker_status.json")
    state_path = os.path.join(stocks_dir, "stock_trader_state.json")
    audit_path = os.path.join(stocks_dir, "execution_audit.jsonl")
    health_path = os.path.join(stocks_dir, "health_status.json")
    runtime_events_path = os.path.join(hub_dir, "runtime_events.jsonl")

    auto_enabled = bool(settings.get("stock_auto_trade_enabled", False))
    trade_notional = max(1.0, float(settings.get("stock_trade_notional_usd", 100.0) or 100.0))
    loss_size_step_pct = max(0.0, min(0.9, float(settings.get("stock_loss_streak_size_step_pct", 0.15) or 0.15)))
    loss_size_floor_pct = max(0.10, min(1.0, float(settings.get("stock_loss_streak_size_floor_pct", 0.40) or 0.40)))
    max_open_positions = max(1, int(float(settings.get("stock_max_open_positions", 1) or 1)))
    score_threshold = max(0.0, float(settings.get("stock_score_threshold", 0.2) or 0.2))
    guarded_score_mult = max(1.0, float(settings.get("stock_live_guarded_score_mult", 1.2) or 1.2))
    profit_target_pct = max(0.0, float(settings.get("stock_profit_target_pct", 0.35) or 0.35))
    trailing_gap_pct = max(0.0, float(settings.get("stock_trailing_gap_pct", 0.2) or 0.2))
    max_day_trades = max(0, int(float(settings.get("stock_max_day_trades", 3) or 3)))
    min_hold_minutes = max(0, int(float(settings.get("stock_min_hold_minutes", 1440) or 1440)))
    same_day_exception_enabled = bool(settings.get("stock_same_day_exit_exception_enabled", True))
    same_day_exception_min_hold_minutes = max(
        0,
        int(float(settings.get("stock_same_day_exception_min_hold_minutes", 120) or 120)),
    )
    same_day_exception_min_pnl_pct = max(
        0.0,
        float(settings.get("stock_same_day_exception_min_pnl_pct", 2.5) or 2.5),
    )
    same_day_exception_min_pullback_pct = max(
        0.0,
        float(settings.get("stock_same_day_exception_min_pullback_pct", 0.9) or 0.9),
    )
    same_day_exception_require_score_flip = bool(
        settings.get("stock_same_day_exception_require_score_flip", True),
    )
    same_day_exception_score_floor_mult = max(
        0.0,
        float(settings.get("stock_same_day_exception_score_floor_mult", 0.75) or 0.75),
    )
    pdt_equity_threshold_usd = max(
        0.0,
        float(settings.get("stock_pdt_equity_threshold_usd", 25_000.0) or 25_000.0),
    )
    pdt_max_day_trades_rolling_5d = max(
        0,
        int(float(settings.get("stock_pdt_max_day_trades_rolling_5d", 3) or 3)),
    )
    max_pos_usd = max(0.0, float(settings.get("stock_max_position_usd_per_symbol", 0.0) or 0.0))
    max_total_exposure_pct = max(0.0, float(settings.get("stock_max_total_exposure_pct", 0.0) or 0.0))
    max_daily_loss_usd = max(0.0, float(settings.get("stock_max_daily_loss_usd", 0.0) or 0.0))
    max_daily_loss_pct = max(0.0, float(settings.get("stock_max_daily_loss_pct", 0.0) or 0.0))
    block_cached_scan = bool(settings.get("stock_block_entries_on_cached_scan", True))
    require_data_quality_ok = bool(settings.get("stock_require_data_quality_ok_for_entries", True))
    try:
        reject_rate_gate_pct = max(0.0, min(100.0, float(settings.get("stock_require_reject_rate_max_pct", 92.0) or 92.0)))
    except Exception:
        reject_rate_gate_pct = 92.0
    try:
        cached_scan_hard_block_age_s = max(30, int(float(settings.get("stock_cached_scan_hard_block_age_s", 1800) or 1800)))
    except Exception:
        cached_scan_hard_block_age_s = 1800
    try:
        cached_scan_entry_size_mult = max(0.10, min(1.0, float(settings.get("stock_cached_scan_entry_size_mult", 0.60) or 0.60)))
    except Exception:
        cached_scan_entry_size_mult = 0.60
    stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    if stage == "live_guarded":
        stage = "live"
    enable_exec_v2 = _rollout_at_least(settings, "execution_v2")
    enable_risk_caps = _rollout_at_least(settings, "risk_caps")
    shadow_only = stage == "shadow_only"
    live_guarded = stage == "live"
    live_guarded_calibration = live_guarded and (not bool(settings.get("alpaca_paper_mode", True)))

    alpaca_key, alpaca_secret = get_alpaca_creds(settings, base_dir=BASE_DIR)
    client = AlpacaBrokerClient(
        api_key_id=alpaca_key,
        secret_key=alpaca_secret,
        base_url=str(settings.get("alpaca_base_url", "https://paper-api.alpaca.markets") or ""),
        data_url=str(settings.get("alpaca_data_url", "https://data.alpaca.markets") or ""),
    )
    now_ts = int(time.time())
    if not client.configured():
        return {
            "state": "IDLE",
            "trader_state": "Credentials missing",
            "msg": "Alpaca credentials not configured",
            "auto_enabled": auto_enabled,
            "updated_at": now_ts,
        }

    thinker = _safe_read_json(thinker_path)
    candidate_rows = _stock_candidates_from_thinker(thinker)
    candidate_rows = sorted(candidate_rows, key=_stock_entry_priority, reverse=True)
    candidate_lookup: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rows:
        symbol = str((row or {}).get("symbol", "") or "").strip().upper()
        if symbol and symbol not in candidate_lookup:
            candidate_lookup[symbol] = row
    top_pick = candidate_rows[0] if candidate_rows else {}
    adaptive_thr_hint = float(thinker.get("adaptive_threshold", score_threshold) or score_threshold)
    same_day_score_floor = max(0.0, float(adaptive_thr_hint) * float(same_day_exception_score_floor_mult))

    today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
    state = _safe_read_json(state_path)
    trail_state = state.get("trail", {}) or {}
    opened_today = state.get("opened_today", {}) or {}
    day_trades = state.get("day_trades", {}) or {}
    day_trade_events = _normalize_event_timestamps(state.get("day_trade_events", []), now_ts=now_ts, keep_days=45)
    cooldown_until = state.get("cooldown_until", {}) or {}
    loss_streak = int(float(state.get("loss_streak", 0) or 0))
    last_divergence_ts = int(float(state.get("last_divergence_ts", 0) or 0))
    last_divergence_msg = str(state.get("last_divergence_msg", "") or "")
    open_meta = state.get("open_meta", {}) or {}
    pending = state.get("pending", {}) or {}
    if not isinstance(trail_state, dict):
        trail_state = {}
    if not isinstance(opened_today, dict):
        opened_today = {}
    if not isinstance(day_trades, dict):
        day_trades = {}
    if not isinstance(cooldown_until, dict):
        cooldown_until = {}
    if not isinstance(open_meta, dict):
        open_meta = {}
    if not isinstance(pending, dict):
        pending = {}
    legacy_day_trades_today = int(float(day_trades.get(today, 0) or 0))
    day_trades_today = max(legacy_day_trades_today, _count_events_on_et_day(day_trade_events, now_ts))
    day_trades = {today: day_trades_today}
    normalized_opened_today: Dict[str, int] = {}
    for k, v in opened_today.items():
        symbol = str(k).strip().upper()
        if not symbol:
            continue
        try:
            ts = int(float(v) or 0)
        except Exception:
            ts = 0
        if ts <= 0:
            continue
        if not _same_et_day(ts, now_ts):
            continue
        normalized_opened_today[symbol] = ts
    opened_today = normalized_opened_today
    cooldown_until = {str(k).upper(): float(v) for k, v in cooldown_until.items() if str(k).strip()}
    open_meta = {str(k).upper(): (v if isinstance(v, dict) else {}) for k, v in open_meta.items() if str(k).strip()}
    loss_size_scale = max(loss_size_floor_pct, 1.0 - (loss_size_step_pct * float(max(0, loss_streak))))
    trade_notional_effective = max(1.0, float(trade_notional) * float(loss_size_scale))
    fallback_active = bool(thinker.get("fallback_cached", False)) if isinstance(thinker, dict) else False
    try:
        fallback_age_s = int(float(thinker.get("fallback_age_s", 0) or 0)) if fallback_active else 0
    except Exception:
        fallback_age_s = 0
    thinker_reject_rate_pct, thinker_reject_rate_raw_pct = _effective_reject_pressure(settings, thinker if isinstance(thinker, dict) else {})
    entry_size_scale = 1.0
    if fallback_active and (not block_cached_scan):
        entry_size_scale = float(cached_scan_entry_size_mult)
    trade_notional_entry = max(1.0, float(trade_notional_effective) * float(entry_size_scale))

    raw_positions = client.list_positions()
    positions = _parse_positions(raw_positions)
    all_symbols = set(positions.keys())
    top_symbol = str(top_pick.get("symbol", "") or "").strip().upper()
    for row in candidate_rows[:12]:
        symbol = str((row or {}).get("symbol", "") or "").strip().upper()
        if symbol:
            all_symbols.add(symbol)
    prices = client.get_mid_prices(sorted(all_symbols))
    acct = client.get_account_summary()
    equity = float(acct.get("equity", 0.0) or 0.0) if isinstance(acct, dict) else 0.0
    total_positions_value = sum(max(0.0, float(pos.get("market_value", 0.0) or 0.0)) for pos in positions.values())
    pdt_window_start_ts = _pdt_window_start_ts(now_ts, trading_days=5)
    day_trades_rolling_5d = _count_events_in_window(day_trade_events, pdt_window_start_ts, now_ts)
    pdt_restricted = bool(pdt_equity_threshold_usd > 0.0 and equity > 0.0 and equity < pdt_equity_threshold_usd)

    actions: List[str] = []
    thinker_health = thinker.get("health", {}) if isinstance(thinker, dict) else {}
    thinker_data_ok = bool((thinker_health or {}).get("data_ok", True))
    drift_warning = False
    # Reconciliation for prior pending submissions.
    if pending:
        p_sym = str(pending.get("symbol", "") or "").strip().upper()
        p_ts = float(pending.get("ts", 0.0) or 0.0)
        if p_sym and p_sym in positions:
            actions.append(f"RECONCILE OK {p_sym} reflected")
            pending = {}
        elif p_sym and (now_ts - p_ts) > 90:
            drift_warning = True
            actions.append(f"RECONCILE WARN {p_sym} not reflected")
            _append_jsonl(
                audit_path,
                {"ts": now_ts, "date": today, "event": "reconcile_warning", "symbol": p_sym, "msg": "pending timed out"},
            )
            pending = {}
    for symbol, pos in positions.items():
        qty = float(pos.get("qty", 0.0) or 0.0)
        avg = float(pos.get("avg_entry_price", 0.0) or 0.0)
        mid = float(prices.get(symbol, 0.0) or 0.0)
        if qty <= 0 or avg <= 0 or mid <= 0:
            continue
        pnl = ((mid - avg) / avg) * 100.0
        pnl_usd = (mid - avg) * qty
        meta = open_meta.get(symbol, {}) or {}
        mfe = max(float(meta.get("mfe_pct", pnl) or pnl), pnl)
        mae = min(float(meta.get("mae_pct", pnl) or pnl), pnl)
        entry_ts = float(meta.get("entry_ts", now_ts) or now_ts)
        open_meta[symbol] = {"entry_ts": entry_ts, "mfe_pct": mfe, "mae_pct": mae, "last_pnl_pct": pnl}
        st = trail_state.get(symbol, {}) or {}
        armed = bool(st.get("armed", False))
        peak = float(st.get("peak_pct", pnl) or pnl)
        if pnl >= profit_target_pct:
            armed = True
            peak = max(peak, pnl)
        if armed:
            peak = max(peak, pnl)
            if pnl <= (peak - trailing_gap_pct):
                hold_s = max(0, int(now_ts - entry_ts))
                same_day_roundtrip = _same_et_day(entry_ts, now_ts)
                hold_gate_block = ""
                intraday_exception_used = False
                if same_day_roundtrip and hold_s < (max(0, min_hold_minutes) * 60):
                    min_exc_hold_s = max(0, same_day_exception_min_hold_minutes) * 60
                    if not same_day_exception_enabled:
                        hold_gate_block = f"PDT hold gate: same-day exits disabled for {symbol}"
                    elif hold_s < min_exc_hold_s:
                        hold_gate_block = (
                            f"PDT hold gate: {symbol} held {int(hold_s // 60)}m; "
                            f"need >= {int(same_day_exception_min_hold_minutes)}m before exception"
                        )
                    elif pdt_restricted and pdt_max_day_trades_rolling_5d > 0 and day_trades_rolling_5d >= pdt_max_day_trades_rolling_5d:
                        hold_gate_block = (
                            "PDT guard: rolling 5-day day-trade cap reached "
                            f"({day_trades_rolling_5d}/{pdt_max_day_trades_rolling_5d})"
                        )
                    elif pnl < same_day_exception_min_pnl_pct:
                        hold_gate_block = (
                            f"PDT hold gate: boom exception needs pnl >= {same_day_exception_min_pnl_pct:.2f}% for {symbol}"
                        )
                    elif (mfe - pnl) < same_day_exception_min_pullback_pct:
                        hold_gate_block = (
                            f"PDT hold gate: boom exception needs pullback >= {same_day_exception_min_pullback_pct:.2f}% for {symbol}"
                        )
                    else:
                        cand = candidate_lookup.get(symbol, {}) if isinstance(candidate_lookup.get(symbol, {}), dict) else {}
                        cand_side = str(cand.get("side", "watch") or "watch").strip().lower()
                        cand_score = float(cand.get("score", 0.0) or 0.0)
                        score_flip_confirmed = bool((cand_side != "long") or (cand_score < same_day_score_floor))
                        if same_day_exception_require_score_flip and (not score_flip_confirmed):
                            hold_gate_block = (
                                "PDT hold gate: momentum still long; no dip-risk confirmation "
                                f"for {symbol} (score {cand_score:.4f} >= floor {same_day_score_floor:.4f})"
                            )
                        else:
                            intraday_exception_used = True
                if hold_gate_block:
                    actions.append(f"HOLD {symbol} | {hold_gate_block}")
                    trail_state[symbol] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                    continue
                if intraday_exception_used:
                    actions.append(
                        f"INTRADAY EXIT EXCEPTION {symbol} | pnl {pnl:.2f}% from peak {mfe:.2f}% "
                        f"(pullback {max(0.0, mfe - pnl):.2f}%)"
                    )
                ok, msg, payload = client.close_position(symbol)
                actions.append(f"CLOSE {symbol} | {'OK' if ok else 'FAIL'} | {msg}")
                _append_jsonl(
                    audit_path,
                    {
                        "ts": now_ts,
                        "date": today,
                        "event": "exit" if ok else "exit_fail",
                        "symbol": symbol,
                        "side": "sell",
                        "qty": qty,
                        "avg_entry_price": avg,
                        "price": mid,
                        "pnl_pct": pnl,
                        "pnl_usd": pnl_usd,
                        "mfe_pct": round(mfe, 4),
                        "mae_pct": round(mae, 4),
                        "hold_s": hold_s,
                        "same_day_roundtrip": bool(same_day_roundtrip),
                        "intraday_exception_used": bool(intraday_exception_used),
                        "ok": ok,
                        "msg": msg,
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
                if ok:
                    trail_state.pop(symbol, None)
                    open_meta.pop(symbol, None)
                    if same_day_roundtrip:
                        day_trades_today += 1
                        day_trades[today] = day_trades_today
                        day_trade_events.append(int(now_ts))
                        day_trade_events = _normalize_event_timestamps(day_trade_events, now_ts=now_ts, keep_days=45)
                        day_trades_rolling_5d = _count_events_in_window(day_trade_events, pdt_window_start_ts, now_ts)
                    opened_today.pop(symbol, None)
                    if pnl_usd < 0:
                        loss_streak += 1
                        cooldown_until[symbol] = float(now_ts + max(60, int(float(settings.get("stock_loss_cooldown_seconds", 1800) or 1800))))
                    else:
                        loss_streak = 0
                else:
                    trail_state[symbol] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                continue
        trail_state[symbol] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}

    signal_symbol = top_symbol
    signal_side = str(top_pick.get("side", "watch") or "watch").strip().lower()
    signal_score = float(top_pick.get("score", 0.0) or 0.0)
    entry_fail_reasons: List[str] = []
    entry_msg = "Auto-trade disabled (paper-safe)"
    if auto_enabled:
        signal_age_s = max(0, int(now_ts - int(float(thinker.get("updated_at", now_ts) or now_ts))))
        max_signal_age_s = max(30, int(float(settings.get("stock_max_signal_age_seconds", 300) or 300)))
        min_bars_required = max(8, int(float(settings.get("stock_min_bars_required", 24) or 24)))
        min_samples_guarded = max(0, int(float(settings.get("stock_min_samples_live_guarded", 5) or 5)))
        adaptive_thr = float(thinker.get("adaptive_threshold", score_threshold) or score_threshold)
        required_score = ((adaptive_thr if adaptive_thr > 0 else score_threshold) * (guarded_score_mult if live_guarded else 1.0))
        max_slippage_bps = max(0.0, float(settings.get("stock_max_slippage_bps", 35.0) or 35.0))
        max_loss_streak = max(0, int(float(settings.get("stock_max_loss_streak", 3) or 3)))
        global_cap_pct = max(0.0, float(settings.get("market_max_total_exposure_pct", 0.0) or 0.0))
        crypto_exposure_usd = _crypto_holdings_usd(hub_dir)
        forex_exposure_usd = _market_status_exposure_usd(hub_dir, "forex")
        if signal_age_s > max_signal_age_s:
            entry_msg = f"Signal stale ({signal_age_s}s > {max_signal_age_s}s)"
        elif require_data_quality_ok and (not thinker_data_ok):
            entry_msg = "Data-quality gate: thinker health not OK"
        elif reject_rate_gate_pct > 0.0 and thinker_reject_rate_pct >= reject_rate_gate_pct:
            entry_msg = f"Reject-pressure gate active ({thinker_reject_rate_pct:.1f}% >= {reject_rate_gate_pct:.1f}%)"
        elif fallback_active and fallback_age_s > cached_scan_hard_block_age_s:
            entry_msg = f"Thinker cached fallback too old ({fallback_age_s}s > {cached_scan_hard_block_age_s}s)"
        elif block_cached_scan and fallback_active:
            entry_msg = f"Thinker cached fallback active ({fallback_age_s}s); blocking new entries"
        elif max_loss_streak > 0 and loss_streak >= max_loss_streak:
            entry_msg = f"Loss-streak guard active ({loss_streak}/{max_loss_streak})"
        elif max_day_trades > 0 and day_trades_today >= max_day_trades:
            entry_msg = f"PDT guard: day-trade cap reached ({day_trades_today}/{max_day_trades})"
        elif not _market_open_now():
            entry_msg = "Market hours gate: market closed"
        elif _near_close_blocked(settings):
            entry_msg = "Near-close gate: no new entries"
        elif _daily_loss_guard_triggered(audit_path, max_daily_loss_usd, max_daily_loss_pct, equity):
            entry_msg = "Daily loss guard active: blocking new entries"
        elif enable_risk_caps and max_total_exposure_pct > 0.0 and equity > 0.0 and (((total_positions_value + trade_notional_entry) / equity) * 100.0) > max_total_exposure_pct:
            entry_msg = f"Risk cap: projected exposure exceeds {max_total_exposure_pct:.2f}%"
        elif global_cap_pct > 0.0 and equity > 0.0 and (((total_positions_value + trade_notional_entry + crypto_exposure_usd + forex_exposure_usd) / equity) * 100.0) > global_cap_pct:
            entry_msg = f"Global cap: projected cross-market exposure exceeds {global_cap_pct:.2f}%"
        elif not enable_exec_v2:
            entry_msg = "Execution gated by rollout stage"
        else:
            quote_symbols = [str((row or {}).get("symbol", "") or "").strip().upper() for row in candidate_rows]
            quote_symbols = [s for s in quote_symbols if s][:12]
            spreads = client.get_snapshot_details(quote_symbols) if quote_symbols else {}
            fail_reasons: List[str] = []
            selected_symbol = ""
            selected_mid = 0.0
            selected_spread_bps = 0.0
            selected_score = 0.0
            selected_calib_prob = 0.0
            selected_samples = 0
            selected_bars = 0
            for cand in candidate_rows:
                symbol = str((cand or {}).get("symbol", "") or "").strip().upper()
                if not symbol:
                    continue
                score = float(cand.get("score", 0.0) or 0.0)
                side = str(cand.get("side", "watch") or "watch").strip().lower()
                calib_prob = float(cand.get("calib_prob", 0.0) or 0.0)
                sample_count = int(float(cand.get("samples", 0) or 0))
                bars_count = int(float(cand.get("bars_count", 0) or 0))
                mid = float((spreads.get(symbol, {}) or {}).get("mid", 0.0) or 0.0)
                spread_bps = float((spreads.get(symbol, {}) or {}).get("spread_bps", 0.0) or 0.0)
                if live_guarded_calibration and (calib_prob <= 0.0):
                    calib_prob = 0.5
                fail = ""
                if bars_count > 0 and bars_count < min_bars_required:
                    fail = f"Bars preflight failed for {symbol} ({bars_count} < {min_bars_required})"
                elif str(cand.get("entry_gate_reason", "") or "").strip():
                    fail = str(cand.get("entry_gate_reason", "") or "").strip()
                elif side != "long":
                    fail = f"Top pick {symbol} is {side.upper()}"
                elif not bool(cand.get("eligible_for_entry", True)):
                    fail = f"Universe health gate blocked {symbol}"
                elif not bool(cand.get("data_quality_ok", True)):
                    fail = f"Data quality gate blocked {symbol}"
                elif mid <= 0.0:
                    fail = f"Quote preflight failed for {symbol}"
                elif live_guarded_calibration and sample_count < min_samples_guarded:
                    fail = f"Calibration sample gate for {symbol} ({sample_count} < {min_samples_guarded})"
                elif live_guarded_calibration and calib_prob < float(settings.get("stock_min_calib_prob_live_guarded", 0.58) or 0.58):
                    fail = f"Calibrated confidence gate for {symbol} ({calib_prob:.2f})"
                elif score < required_score:
                    fail = f"Top score below threshold for {symbol} ({score:.4f} < {required_score:.4f})"
                elif float(cooldown_until.get(symbol, 0.0) or 0.0) > float(now_ts):
                    cd_left = int(float(cooldown_until.get(symbol, 0.0) - now_ts))
                    fail = f"Cooldown active for {symbol} ({cd_left}s)"
                elif len(positions) >= max_open_positions and symbol not in positions:
                    fail = f"Max open positions reached ({len(positions)}/{max_open_positions})"
                elif symbol in positions:
                    fail = f"Already in position: {symbol}"
                elif max_slippage_bps > 0.0 and spread_bps > max_slippage_bps:
                    fail = f"Slippage guard for {symbol}: spread {spread_bps:.2f}bps > {max_slippage_bps:.2f}bps"
                elif enable_risk_caps and max_pos_usd > 0.0 and (float(positions.get(symbol, {}).get("market_value", 0.0) or 0.0) + trade_notional_entry) > max_pos_usd:
                    fail = f"Risk cap: {symbol} projected position exceeds ${max_pos_usd:.2f}"
                if fail:
                    fail_reasons.append(fail)
                    continue
                selected_symbol = symbol
                selected_mid = mid
                selected_spread_bps = spread_bps
                selected_score = score
                selected_calib_prob = calib_prob
                selected_samples = sample_count
                selected_bars = bars_count
                break
            if not selected_symbol:
                entry_msg = fail_reasons[0] if fail_reasons else "No symbols available from thinker"
                if fail_reasons:
                    entry_fail_reasons.extend([str(x) for x in fail_reasons[:20] if str(x).strip()])
            elif shadow_only:
                signal_symbol = selected_symbol
                signal_side = "long"
                signal_score = selected_score
                entry_msg = f"SHADOW entry simulated for {selected_symbol}"
                actions.append(f"SHADOW ENTRY {selected_symbol} BUY ${trade_notional_entry:.2f}")
                _append_jsonl(
                    audit_path,
                    {
                        "ts": now_ts,
                        "date": today,
                        "event": "shadow_entry",
                        "symbol": selected_symbol,
                        "side": "buy",
                        "notional": trade_notional_entry,
                        "configured_notional": trade_notional,
                        "entry_size_scale": float(round(entry_size_scale, 4)),
                        "score": selected_score,
                        "ok": True,
                        "msg": "shadow_only stage",
                    },
                )
            else:
                signal_symbol = selected_symbol
                signal_side = "long"
                signal_score = selected_score
                client_id = f"pt-{selected_symbol}-{now_ts}"
                ok, msg, payload = client.place_market_order(
                    selected_symbol,
                    side="buy",
                    notional=trade_notional_entry,
                    client_order_id=client_id,
                    max_retries=max(1, int(float(settings.get("stock_order_retry_count", 2) or 2))),
                    max_retry_after_s=max(1.0, float(settings.get("broker_order_retry_after_cap_s", 300.0) or 300.0)),
                )
                actions.append(f"ENTRY {selected_symbol} BUY ${trade_notional_entry:.2f} | {'OK' if ok else 'FAIL'} | {msg}")
                oid = _parse_order_id(msg, payload if isinstance(payload, dict) else {})
                retry_after_wait_s = parse_retry_after_value(str(msg or ""), max_wait_s=3600.0)
                if retry_after_wait_s > 0.0:
                    runtime_event(
                        runtime_events_path,
                        component="stocks_trader",
                        event="broker_retry_after_wait",
                        level="warning",
                        msg=f"Stocks broker retry-after wait {retry_after_wait_s:.2f}s",
                        details={
                            "market": "stocks",
                            "symbol": selected_symbol,
                            "wait_s": float(round(retry_after_wait_s, 3)),
                            "ok": bool(ok),
                        },
                    )
                _append_jsonl(
                    audit_path,
                    {
                        "ts": now_ts,
                        "date": today,
                        "event": "entry" if ok else "entry_fail",
                        "symbol": selected_symbol,
                        "side": "buy",
                        "notional": trade_notional_entry,
                        "configured_notional": trade_notional,
                        "entry_size_scale": float(round(entry_size_scale, 4)),
                        "score": selected_score,
                        "calib_prob": selected_calib_prob,
                        "samples": selected_samples,
                        "bars_count": selected_bars,
                        "price": selected_mid,
                        "spread_bps": selected_spread_bps,
                        "client_order_id": client_id,
                        "order_id": oid,
                        "retry_after_wait_s": float(round(retry_after_wait_s, 3)),
                        "ok": ok,
                        "msg": msg,
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
                entry_msg = f"Entry {'placed' if ok else 'failed'} for {selected_symbol}"
                if ok:
                    opened_today[selected_symbol] = now_ts
                    open_meta[selected_symbol] = {"entry_ts": now_ts, "mfe_pct": 0.0, "mae_pct": 0.0, "last_pnl_pct": 0.0}
                    pending = {"symbol": selected_symbol, "side": "buy", "ts": now_ts, "order_id": oid, "client_order_id": client_id}
                    recon_positions = _parse_positions(client.list_positions())
                    if selected_symbol not in recon_positions:
                        drift_warning = True
                        _append_jsonl(
                            audit_path,
                            {
                                "ts": now_ts,
                                "date": today,
                                "event": "reconcile_warning",
                                "symbol": selected_symbol,
                                "msg": "submitted but not reflected in positions",
                            },
                        )
    else:
        entry_msg = "Auto-trade disabled (paper-safe)"
    if auto_enabled and ("Entry placed" not in str(entry_msg)):
        entry_fail_reasons.append(str(entry_msg))
    entry_eval_top_reason, entry_eval_reason_counts = _fail_reason_summary(entry_fail_reasons)

    out_state = {
        "trail": trail_state,
        "opened_today": opened_today,
        "day_trades": day_trades,
        "day_trade_events": day_trade_events,
        "day_trades_rolling_5d": int(day_trades_rolling_5d),
        "cooldown_until": cooldown_until,
        "loss_streak": int(loss_streak),
        "open_meta": open_meta,
        "pending": pending,
        "last_divergence_ts": int(last_divergence_ts),
        "last_divergence_msg": str(last_divergence_msg),
        "entry_eval_total": int(len(entry_fail_reasons)),
        "entry_eval_top_reason": str(entry_eval_top_reason),
        "entry_eval_reason_counts": dict(entry_eval_reason_counts),
        "entry_gate_flags": {
            "data_quality_required": bool(require_data_quality_ok),
            "data_quality_ok": bool(thinker_data_ok),
            "reject_rate_pct": float(round(thinker_reject_rate_pct, 4)),
            "reject_rate_raw_pct": float(round(thinker_reject_rate_raw_pct, 4)),
            "reject_rate_max_pct": float(round(reject_rate_gate_pct, 4)),
            "cached_fallback_active": bool(fallback_active),
            "cached_fallback_age_s": int(fallback_age_s),
            "cached_fallback_hard_block_age_s": int(cached_scan_hard_block_age_s),
            "pdt_restricted": bool(pdt_restricted),
            "pdt_equity_threshold_usd": float(round(pdt_equity_threshold_usd, 4)),
            "day_trades_rolling_5d": int(day_trades_rolling_5d),
            "pdt_max_day_trades_rolling_5d": int(pdt_max_day_trades_rolling_5d),
            "same_day_exit_min_hold_minutes": int(min_hold_minutes),
        },
        "trade_notional_entry_usd": round(float(trade_notional_entry), 4),
        "entry_size_scale": round(float(entry_size_scale), 4),
        "last_actions": actions[-80:],
        "updated_at": now_ts,
    }
    _safe_write_json(state_path, out_state)
    _safe_write_json(
        health_path,
        {
            "ts": now_ts,
            "data_ok": thinker_data_ok,
            "broker_ok": True,
            "orders_ok": True,
            "drift_warning": drift_warning,
        },
    )

    if auto_enabled and (not shadow_only):
        try:
            if signal_side == "long" and signal_symbol and ("Entry placed" not in entry_msg) and ("Already in position" not in entry_msg):
                should_log = ((now_ts - int(last_divergence_ts)) >= 300) or (str(entry_msg) != str(last_divergence_msg))
                if should_log:
                    _append_jsonl(
                        audit_path,
                        {
                            "ts": now_ts,
                            "date": today,
                            "event": "shadow_live_divergence",
                            "symbol": signal_symbol,
                            "score": signal_score,
                            "msg": entry_msg,
                        },
                    )
                    last_divergence_ts = int(now_ts)
                    last_divergence_msg = str(entry_msg)
        except Exception:
            pass
    out_state["last_divergence_ts"] = int(last_divergence_ts)
    out_state["last_divergence_msg"] = str(last_divergence_msg)
    _safe_write_json(state_path, out_state)

    msg_parts = [entry_msg]
    if shadow_only:
        msg_parts.append("rollout shadow_only: real entries suppressed")
    elif not enable_exec_v2:
        msg_parts.append(f"rollout {stage}: execution disabled")
    if trade_notional_effective < trade_notional:
        msg_parts.append(f"size x{loss_size_scale:.2f}")
    if trade_notional_entry < trade_notional_effective:
        msg_parts.append(f"scan-size x{entry_size_scale:.2f}")
    if actions:
        msg_parts.append(actions[-1])
    return {
        "state": "READY",
        "trader_state": _trader_state_label(settings, auto_enabled, shadow_only),
        "msg": " | ".join(x for x in msg_parts if x),
        "actions": actions[-12:],
        "open_positions": len(positions),
        "auto_enabled": auto_enabled,
        "broker_mode": str(_broker_mode_label(settings)).lower(),
        "day_trades_today": day_trades_today,
        "day_trades_rolling_5d": int(day_trades_rolling_5d),
        "rollout_stage": str(settings.get("market_rollout_stage", "legacy") or "legacy"),
        "execution_enabled": enable_exec_v2 and (not shadow_only),
        "trade_notional_usd": round(float(trade_notional), 4),
        "trade_notional_effective_usd": round(float(trade_notional_effective), 4),
        "trade_notional_entry_usd": round(float(trade_notional_entry), 4),
        "loss_size_scale": round(float(loss_size_scale), 4),
        "entry_size_scale": round(float(entry_size_scale), 4),
        "exposure_usd": round(total_positions_value, 4),
        "crypto_exposure_usd": round(crypto_exposure_usd, 4) if auto_enabled else round(_crypto_holdings_usd(hub_dir), 4),
        "other_market_exposure_usd": round(forex_exposure_usd, 4) if auto_enabled else round(_market_status_exposure_usd(hub_dir, "forex"), 4),
        "account_value_usd": round(equity, 4),
        "entry_eval_total": int(len(entry_fail_reasons)),
        "entry_eval_failed": int(len(entry_fail_reasons) > 0),
        "entry_eval_top_reason": str(entry_eval_top_reason),
        "entry_eval_reason_counts": dict(entry_eval_reason_counts),
        "entry_gate_flags": {
            "data_quality_required": bool(require_data_quality_ok),
            "data_quality_ok": bool(thinker_data_ok),
            "reject_rate_pct": float(round(thinker_reject_rate_pct, 4)),
            "reject_rate_raw_pct": float(round(thinker_reject_rate_raw_pct, 4)),
            "reject_rate_max_pct": float(round(reject_rate_gate_pct, 4)),
            "cached_fallback_active": bool(fallback_active),
            "cached_fallback_age_s": int(fallback_age_s),
            "cached_fallback_hard_block_age_s": int(cached_scan_hard_block_age_s),
            "pdt_restricted": bool(pdt_restricted),
            "pdt_equity_threshold_usd": float(round(pdt_equity_threshold_usd, 4)),
            "day_trades_rolling_5d": int(day_trades_rolling_5d),
            "pdt_max_day_trades_rolling_5d": int(pdt_max_day_trades_rolling_5d),
            "same_day_exit_min_hold_minutes": int(min_hold_minutes),
        },
        "updated_at": now_ts,
        "health": {"data_ok": thinker_data_ok, "broker_ok": True, "orders_ok": True, "drift_warning": drift_warning},
    }


def main() -> int:
    print("stock_trader.py is designed to be imported by the hub/runner first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
