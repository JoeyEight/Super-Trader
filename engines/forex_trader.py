from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from app.automation_policy import build_market_automation_policy
from app.credential_utils import get_oanda_creds
from app.http_utils import parse_retry_after_value
from app.opportunity_allocator import evaluate_cross_market_allocation
from app.path_utils import resolve_runtime_paths
from app.runtime_logging import runtime_event
from app.scanner_quality import effective_reject_pressure
from app.settings_utils import normalize_settings_profile
from app.stale_exit_policy import evaluate_stale_profit_hold
from app.trade_quality import evaluate_trade_quality
from brokers.broker_oanda import OandaBrokerClient

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "forex_trader")
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
    return "Practice" if bool(settings.get("oanda_practice_mode", True)) else "Live"


def _forex_uses_exposure_budget_slots(settings: Dict[str, Any]) -> bool:
    profile = normalize_settings_profile(settings.get("settings_profile", "balanced"), default="balanced")
    return str(profile) == "max_growth"


def _forex_effective_open_position_hard_cap(
    settings: Dict[str, Any],
    configured_cap: int,
    current_open_positions: int,
) -> int:
    base_cap = max(1, int(configured_cap or 1))
    if not _forex_uses_exposure_budget_slots(settings):
        return max(base_cap, int(current_open_positions or 0))
    # Max Growth treats raw position count as a soft guide and lets exposure/margin
    # controls determine how many small forex tickets can be held safely.
    scaled = max(base_cap * 3, 3)
    hard_cap = min(12, max(base_cap, scaled))
    return max(int(current_open_positions or 0), int(hard_cap))


def _trader_state_label(settings: Dict[str, Any], auto_enabled: bool, shadow_only: bool) -> str:
    mode = _broker_mode_label(settings)
    if auto_enabled:
        return f"{mode} shadow-run" if shadow_only else f"{mode} auto-run"
    return f"{mode} manual-ready"


def _parse_positions(raw_positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw_positions or []:
        if not isinstance(row, dict):
            continue
        inst = str(row.get("instrument", "") or "").strip().upper()
        if not inst:
            continue
        long_leg = row.get("long", {}) or {}
        short_leg = row.get("short", {}) or {}
        try:
            long_units = float(long_leg.get("units", 0.0) or 0.0)
        except Exception:
            long_units = 0.0
        try:
            short_units = float(short_leg.get("units", 0.0) or 0.0)
        except Exception:
            short_units = 0.0
        try:
            long_avg = float(long_leg.get("averagePrice", 0.0) or 0.0)
        except Exception:
            long_avg = 0.0
        try:
            short_avg = float(short_leg.get("averagePrice", 0.0) or 0.0)
        except Exception:
            short_avg = 0.0
        out[inst] = {
            "instrument": inst,
            "long_units": long_units,
            "short_units": short_units,
            "long_avg": long_avg,
            "short_avg": short_avg,
        }
    return out


def _pnl_pct(position: Dict[str, Any], mid_px: float) -> Tuple[str, float]:
    lu = float(position.get("long_units", 0.0) or 0.0)
    su = float(position.get("short_units", 0.0) or 0.0)
    if lu > 0:
        avg = float(position.get("long_avg", 0.0) or 0.0)
        if avg > 0:
            return "long", ((mid_px - avg) / avg) * 100.0
        return "long", 0.0
    if su < 0:
        avg = float(position.get("short_avg", 0.0) or 0.0)
        if avg > 0:
            return "short", ((avg - mid_px) / avg) * 100.0
        return "short", 0.0
    return "flat", 0.0


def _session_blocked(settings: Dict[str, Any]) -> bool:
    mode = str(settings.get("forex_session_mode", "all") or "all").strip().lower()
    if mode in {"all", ""}:
        return False
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if mode == "london_ny":
        return not (12 <= hour <= 16)
    if mode == "london":
        return not (7 <= hour <= 15)
    if mode == "ny":
        return not (12 <= hour <= 20)
    if mode == "asia":
        return not (0 <= hour <= 8)
    return False


def _daily_loss_guard_triggered(audit_path: str, max_loss_usd: float, max_loss_pct: float, nav: float) -> bool:
    if max_loss_usd <= 0.0 and max_loss_pct <= 0.0:
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
                if str(row.get("date", "")) != today:
                    continue
                if str(row.get("event", "")).lower() not in {"exit", "shadow_exit"}:
                    continue
                pnl_usd = _audit_realized_pnl_usd(row)
                if pnl_usd < 0:
                    loss_usd += abs(pnl_usd)
    except Exception:
        return False
    if max_loss_usd > 0.0 and loss_usd >= max_loss_usd:
        return True
    if max_loss_pct > 0.0 and nav > 0.0 and ((loss_usd / nav) * 100.0) >= max_loss_pct:
        return True
    return False


def _parse_order_id(msg: str, payload: Dict[str, Any]) -> str:
    oid = ""
    try:
        oid = str((payload or {}).get("orderFillTransaction", {}).get("id", "") or "").strip()
    except Exception:
        oid = ""
    if oid:
        return oid
    txt = str(msg or "")
    if "order_id=" in txt:
        return txt.split("order_id=", 1)[1].strip().split(" ", 1)[0]
    return ""


def _close_fill_transaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("orderFillTransaction", "shortOrderFillTransaction", "longOrderFillTransaction"):
        row = payload.get(key, {})
        if isinstance(row, dict) and row:
            return row
    return {}


def _realized_pnl_from_close_payload(payload: Dict[str, Any]) -> float | None:
    fill_txn = _close_fill_transaction(payload if isinstance(payload, dict) else {})
    if not fill_txn:
        return None
    trades_closed = fill_txn.get("tradesClosed", [])
    if isinstance(trades_closed, list) and trades_closed:
        total = 0.0
        hits = 0
        for row in trades_closed:
            if not isinstance(row, dict):
                continue
            try:
                total += float(row.get("realizedPL", 0.0) or 0.0)
                hits += 1
            except Exception:
                continue
        if hits > 0:
            return float(total)
    for key in ("realizedPL", "pl"):
        try:
            raw = fill_txn.get(key, None)
            if raw in (None, ""):
                continue
            return float(raw)
        except Exception:
            continue
    return None


def _audit_realized_pnl_usd(row: Dict[str, Any]) -> float:
    if not isinstance(row, dict):
        return 0.0
    for key in ("realized_pnl", "realized_pl", "realized_profit_usd", "realized"):
        try:
            raw = row.get(key, None)
            if raw in (None, ""):
                continue
            return float(raw)
        except Exception:
            continue
    payload = row.get("payload", {})
    realized_from_payload = _realized_pnl_from_close_payload(payload if isinstance(payload, dict) else {})
    if realized_from_payload is not None:
        return float(realized_from_payload)
    try:
        return float(row.get("pnl_usd", 0.0) or 0.0)
    except Exception:
        return 0.0


def _align_pnl_pct_sign_with_realized(
    *,
    pnl_pct_est: float,
    pnl_usd_est: float,
    realized_pnl_usd: float | None,
) -> float:
    est = float(pnl_pct_est)
    realized = None if realized_pnl_usd is None else float(realized_pnl_usd)
    if realized is None:
        return est
    if abs(realized) <= 1e-12:
        return 0.0
    est_sign = 0 if abs(est) <= 1e-12 else (1 if est > 0.0 else -1)
    real_sign = 1 if realized > 0.0 else -1
    if est_sign == 0:
        # Estimate is effectively flat; carry only directionality from realized PnL
        # using a tiny placeholder magnitude in pct units.
        return float(real_sign) * 0.000001
    if est_sign == real_sign:
        return est
    return float(real_sign) * abs(est)


def _safe_float_from_dict(d: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        try:
            if k in d:
                return float(d.get(k, 0.0) or 0.0)
        except Exception:
            continue
    return 0.0


def _trader_data_account(hub_dir: str) -> Dict[str, Any]:
    data = _safe_read_json(os.path.join(hub_dir, "trader_data.json"))
    if not isinstance(data, dict):
        return {}
    account = data.get("account", {})
    return account if isinstance(account, dict) else {}


def _crypto_holdings_usd(hub_dir: str) -> float:
    account = _trader_data_account(hub_dir)
    if isinstance(account, dict) and account:
        return _safe_float_from_dict(
            account,
            [
                "holdings_sell_value",
                "holdings_buy_value",
                "holdings_value",
                "holdings_usd",
                "total_holdings_value",
            ],
        )
    data = _safe_read_json(os.path.join(hub_dir, "trader_data.json"))
    if not isinstance(data, dict):
        return 0.0
    return _safe_float_from_dict(data, ["exposure_usd", "holdings_value", "total_holdings_value", "holdings_usd"])


def _crypto_account_value_usd(hub_dir: str) -> float:
    account = _trader_data_account(hub_dir)
    if isinstance(account, dict) and account:
        return _safe_float_from_dict(account, ["total_account_value", "account_value_usd", "equity", "nav"])
    data = _safe_read_json(os.path.join(hub_dir, "trader_data.json"))
    if not isinstance(data, dict):
        return 0.0
    return _safe_float_from_dict(data, ["account_value_usd", "equity", "nav"])


def _market_status_exposure_usd(hub_dir: str, market_key: str) -> float:
    if market_key == "stocks":
        path = os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    elif market_key == "forex":
        path = os.path.join(hub_dir, "forex", "forex_trader_status.json")
    else:
        path = os.path.join(hub_dir, market_key, f"{market_key}_trader_status.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict):
        return 0.0
    return _safe_float_from_dict(data, ["exposure_usd", "total_positions_value_usd", "positions_value_usd"])


def _market_status_account_value_usd(hub_dir: str, market_key: str) -> float:
    if market_key == "crypto":
        return _crypto_account_value_usd(hub_dir)
    if market_key == "stocks":
        path = os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    elif market_key == "forex":
        path = os.path.join(hub_dir, "forex", "forex_trader_status.json")
    else:
        path = os.path.join(hub_dir, market_key, f"{market_key}_trader_status.json")
    data = _safe_read_json(path)
    if not isinstance(data, dict):
        return 0.0
    account = data.get("account", {})
    account_dict = account if isinstance(account, dict) else {}
    return max(
        0.0,
        _safe_float_from_dict(
            data,
            [
                "account_value_usd",
                "equity",
                "nav",
            ],
        ),
        _safe_float_from_dict(account_dict, ["total_account_value", "equity", "nav", "account_value_usd"]),
    )


def _portfolio_account_value_usd(
    hub_dir: str,
    *,
    current_market: str,
    current_account_value_usd: float,
) -> float:
    current_key = str(current_market or "").strip().lower()
    total = max(0.0, float(current_account_value_usd or 0.0))
    for mk in ("crypto", "stocks", "forex"):
        if mk == current_key:
            continue
        total += max(0.0, _market_status_account_value_usd(hub_dir, mk))
    return float(total)


def _forex_unit_notional_usd(instrument: str, mid: float, pricing_row: Dict[str, Any] | None = None) -> float:
    mid_px = abs(float(mid or 0.0))
    if mid_px <= 0.0:
        return 0.0
    inst = str(instrument or "").strip().upper()
    base_ccy = ""
    quote_ccy = ""
    if "_" in inst:
        try:
            base_ccy, quote_ccy = [str(part or "").strip().upper() for part in inst.split("_", 1)]
        except Exception:
            base_ccy = ""
            quote_ccy = ""
    # When the base is USD, each unit is already 1 USD of notional.
    if base_ccy == "USD":
        return 1.0
    # USD-quoted pairs already express one base unit directly in USD.
    if quote_ccy == "USD":
        return mid_px
    row = pricing_row if isinstance(pricing_row, dict) else {}
    try:
        q_home = max(
            abs(float(row.get("quote_to_home", 0.0) or 0.0)),
            abs(float(row.get("quote_to_home_positive", 0.0) or 0.0)),
            abs(float(row.get("quote_to_home_negative", 0.0) or 0.0)),
        )
    except Exception:
        q_home = 0.0
    if q_home > 0.0:
        return mid_px * q_home
    return mid_px


def _risk_capped_units(
    desired_units: int,
    unit_notional_usd: float,
    unit_margin_usd: float,
    nav: float,
    total_exposure_usd: float,
    total_margin_used_usd: float,
    margin_available_usd: float,
    crypto_exposure_usd: float,
    stocks_exposure_usd: float,
    max_total_exposure_pct: float,
    max_pos_usd: float,
    global_cap_pct: float,
    global_cap_account_value_usd: float,
) -> Tuple[int, float]:
    units_abs = abs(int(desired_units or 0))
    if units_abs <= 0:
        return 0, 0.0
    unit_notional = abs(float(unit_notional_usd or 0.0))
    if unit_notional <= 0.0:
        return int(desired_units or 0), 1.0
    unit_margin = abs(float(unit_margin_usd or 0.0))
    if unit_margin <= 0.0:
        unit_margin = unit_notional * 0.05
    max_units = int(units_abs)
    if max_pos_usd > 0.0:
        max_units = min(max_units, max(0, int(max(0.0, float(max_pos_usd)) / unit_notional)))
    margin_allowance = max(0.0, float(margin_available_usd or 0.0))
    if max_total_exposure_pct > 0.0 and nav > 0.0:
        margin_util_allowance = max(
            0.0,
            (float(nav) * float(max_total_exposure_pct) / 100.0) - float(total_margin_used_usd),
        )
        margin_allowance = min(margin_allowance, margin_util_allowance)
    if unit_margin > 0.0:
        max_units = min(max_units, max(0, int(margin_allowance / unit_margin)))
    global_cap_base = max(0.0, float(global_cap_account_value_usd or 0.0), float(nav or 0.0))
    if global_cap_pct > 0.0 and global_cap_base > 0.0:
        allowed_global_notional = max(
            0.0,
            (float(global_cap_base) * float(global_cap_pct) / 100.0)
            - (float(total_exposure_usd) + float(crypto_exposure_usd) + float(stocks_exposure_usd)),
        )
        max_units = min(max_units, max(0, int(allowed_global_notional / unit_notional)))
    if max_units >= units_abs:
        return int(desired_units or 0), 1.0
    capped_units = max(0, int(max_units))
    if capped_units <= 0:
        return 0, 0.0
    capped_units = min(units_abs, capped_units)
    scale = float(capped_units) / float(max(1, units_abs))
    signed_units = capped_units if int(desired_units or 0) >= 0 else (-1 * capped_units)
    return int(signed_units), float(scale)


def _forex_candidates_from_thinker(thinker: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("leaders", "all_scores"):
        payload = thinker.get(key, []) if isinstance(thinker, dict) else []
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            pair = str(row.get("pair", "") or "").strip().upper()
            if not pair or pair in seen:
                continue
            out = dict(row)
            out["pair"] = pair
            seen.add(pair)
            rows.append(out)
    top = thinker.get("top_pick", {}) if isinstance(thinker, dict) else {}
    if isinstance(top, dict):
        pair = str(top.get("pair", "") or "").strip().upper()
        if pair and pair not in seen:
            out = dict(top)
            out["pair"] = pair
            rows.append(out)
    rows.sort(key=lambda r: abs(float(r.get("score", 0.0) or 0.0)), reverse=True)
    return rows


def _forex_entry_priority(row: Dict[str, Any]) -> float:
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return float(default)

    side = str(row.get("side", "watch") or "watch").strip().lower()
    side_bonus = 35.0 if side in {"long", "short"} else -35.0
    eligible_bonus = 20.0 if bool(row.get("eligible_for_entry", True)) else -20.0
    score_abs = abs(_f("score", 0.0))
    calib = _f("calib_prob", 0.5)
    quality = _f("quality_score", 0.0)
    spread = _f("spread_bps", 0.0)
    bars = max(0.0, _f("bars_count", 0.0))
    return side_bonus + eligible_bonus + (score_abs * 22.0) + (calib * 10.0) + (quality * 0.06) + (min(120.0, bars) * 0.03) - (spread * 0.35)


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


def _forex_alignment_snapshot(
    instrument: str,
    *,
    position_side: str,
    candidate_lookup: Dict[str, Dict[str, Any]],
    required_score: float,
) -> Dict[str, Any]:
    pair = str(instrument or "").strip().upper()
    cand = candidate_lookup.get(pair, {}) if isinstance(candidate_lookup, dict) else {}
    reasons: List[str] = []
    side = "watch"
    score = 0.0
    eligible = False
    data_quality_ok = True
    entry_gate_reason = ""
    if not isinstance(cand, dict) or (not cand):
        reasons.append("pair is no longer in the active forex scanner set")
        return {
            "aligned": False,
            "reasons": reasons,
            "side": side,
            "score": score,
            "eligible_for_entry": eligible,
            "data_quality_ok": data_quality_ok,
            "entry_gate_reason": entry_gate_reason,
        }
    side = str(cand.get("side", "watch") or "watch").strip().lower()
    try:
        score = float(cand.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    eligible = bool(cand.get("eligible_for_entry", True))
    data_quality_ok = bool(cand.get("data_quality_ok", True))
    entry_gate_reason = str(cand.get("entry_gate_reason", "") or "").strip()
    pos_side = str(position_side or "flat").strip().lower()
    if side not in {"long", "short"}:
        reasons.append("scanner side is WATCH")
    elif side != pos_side:
        reasons.append(f"scanner side flipped to {side.upper()} while position is {pos_side.upper()}")
    if abs(float(score)) < float(required_score):
        reasons.append(
            f"absolute score {abs(float(score)):.4f} is below active threshold {float(required_score):.4f}"
        )
    if not eligible:
        reasons.append("scanner marked pair as not eligible for entry")
    if not data_quality_ok:
        reasons.append("scanner marked pair data quality as degraded")
    if entry_gate_reason:
        reasons.append(f"entry gate active: {entry_gate_reason}")
    return {
        "aligned": bool(len(reasons) == 0),
        "reasons": reasons,
        "side": side,
        "score": float(score),
        "eligible_for_entry": bool(eligible),
        "data_quality_ok": bool(data_quality_ok),
        "entry_gate_reason": entry_gate_reason,
    }


def run_step(settings: Dict[str, Any], hub_dir: str) -> Dict[str, Any]:
    forex_dir = os.path.join(hub_dir, "forex")
    os.makedirs(forex_dir, exist_ok=True)
    thinker_path = os.path.join(forex_dir, "forex_thinker_status.json")
    state_path = os.path.join(forex_dir, "forex_trader_state.json")
    audit_path = os.path.join(forex_dir, "execution_audit.jsonl")
    health_path = os.path.join(forex_dir, "health_status.json")
    runtime_events_path = os.path.join(hub_dir, "runtime_events.jsonl")

    auto_enabled = bool(settings.get("forex_auto_trade_enabled", False))
    independent_market_mode = bool(settings.get("market_independent_execution_enabled", False))
    trade_units = int(float(settings.get("forex_trade_units", 1000) or 1000))
    loss_size_step_pct = max(0.0, min(0.9, float(settings.get("forex_loss_streak_size_step_pct", 0.15) or 0.15)))
    loss_size_floor_pct = max(0.10, min(1.0, float(settings.get("forex_loss_streak_size_floor_pct", 0.40) or 0.40)))
    max_open_positions = max(1, int(float(settings.get("forex_max_open_positions", 1) or 1)))
    forex_exposure_slot_mode = _forex_uses_exposure_budget_slots(settings)
    score_threshold = float(settings.get("forex_score_threshold", 0.2) or 0.2)
    guarded_score_mult = max(1.0, float(settings.get("forex_live_guarded_score_mult", 1.15) or 1.15))
    profit_target_pct = float(settings.get("forex_profit_target_pct", 0.25) or 0.25)
    trailing_gap_pct = float(settings.get("forex_trailing_gap_pct", 0.15) or 0.15)
    max_total_exposure_pct = max(0.0, float(settings.get("forex_max_total_exposure_pct", 0.0) or 0.0))
    max_pos_usd = max(0.0, float(settings.get("forex_max_position_usd_per_pair", 0.0) or 0.0))
    max_daily_loss_usd = max(0.0, float(settings.get("forex_max_daily_loss_usd", 0.0) or 0.0))
    max_daily_loss_pct = max(0.0, float(settings.get("forex_max_daily_loss_pct", 0.0) or 0.0))
    max_loss_streak_setting = max(0, int(float(settings.get("forex_max_loss_streak", 3) or 3)))
    block_cached_scan = bool(settings.get("forex_block_entries_on_cached_scan", True))
    require_data_quality_ok = bool(settings.get("forex_require_data_quality_ok_for_entries", True))
    try:
        reject_rate_gate_pct = max(0.0, min(100.0, float(settings.get("forex_require_reject_rate_max_pct", 92.0) or 92.0)))
    except Exception:
        reject_rate_gate_pct = 92.0
    try:
        cached_scan_hard_block_age_s = max(30, int(float(settings.get("forex_cached_scan_hard_block_age_s", 1200) or 1200)))
    except Exception:
        cached_scan_hard_block_age_s = 1200
    try:
        cached_scan_entry_size_mult = max(0.10, min(1.0, float(settings.get("forex_cached_scan_entry_size_mult", 0.65) or 0.65)))
    except Exception:
        cached_scan_entry_size_mult = 0.65
    stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    if stage == "live_guarded":
        stage = "live"
    enable_exec_v2 = _rollout_at_least(settings, "execution_v2")
    enable_risk_caps = _rollout_at_least(settings, "risk_caps")
    shadow_only = stage == "shadow_only"
    live_guarded = stage == "live"

    oanda_account, oanda_token = get_oanda_creds(settings, base_dir=BASE_DIR)
    client = OandaBrokerClient(
        account_id=oanda_account,
        api_token=oanda_token,
        rest_url=str(settings.get("oanda_rest_url", "https://api-fxpractice.oanda.com") or ""),
    )
    now_ts = int(time.time())
    if not client.configured():
        return {
            "state": "IDLE",
            "trader_state": "Credentials missing",
            "msg": "OANDA credentials not configured",
            "auto_enabled": auto_enabled,
            "updated_at": now_ts,
        }

    broker_snap = client.fetch_snapshot()
    raw_positions = list(broker_snap.get("raw_positions", []) or [])
    positions = _parse_positions(raw_positions)
    thinker = _safe_read_json(thinker_path)
    candidate_rows = _forex_candidates_from_thinker(thinker)
    candidate_rows = sorted(candidate_rows, key=_forex_entry_priority, reverse=True)
    candidate_lookup: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rows:
        pair = str((row or {}).get("pair", "") or "").strip().upper()
        if pair and pair not in candidate_lookup:
            candidate_lookup[pair] = row
    top_pick = candidate_rows[0] if candidate_rows else {}
    adaptive_thr_hint = float(thinker.get("adaptive_threshold", score_threshold) or score_threshold)
    alignment_required_score = (
        (adaptive_thr_hint if adaptive_thr_hint > 0 else score_threshold)
        * (guarded_score_mult if live_guarded else 1.0)
    )

    state = _safe_read_json(state_path)
    trail_state = state.get("trail", {}) or {}
    cooldown_until = state.get("cooldown_until", {}) or {}
    loss_streak = int(float(state.get("loss_streak", 0) or 0))
    try:
        loss_streak_updated_at = int(float(state.get("loss_streak_updated_at", 0) or 0))
    except Exception:
        loss_streak_updated_at = 0
    last_divergence_ts = int(float(state.get("last_divergence_ts", 0) or 0))
    last_divergence_msg = str(state.get("last_divergence_msg", "") or "")
    open_meta = state.get("open_meta", {}) or {}
    pending = state.get("pending", {}) or {}
    stale_alignment_streaks_raw = state.get("stale_alignment_streaks", {}) or {}
    if not isinstance(trail_state, dict):
        trail_state = {}
    if not isinstance(cooldown_until, dict):
        cooldown_until = {}
    if not isinstance(open_meta, dict):
        open_meta = {}
    if not isinstance(pending, dict):
        pending = {}
    if not isinstance(stale_alignment_streaks_raw, dict):
        stale_alignment_streaks_raw = {}
    cooldown_until = {str(k).upper(): float(v) for k, v in cooldown_until.items() if str(k).strip()}
    open_meta = {str(k).upper(): (v if isinstance(v, dict) else {}) for k, v in open_meta.items() if str(k).strip()}
    stale_alignment_streaks: Dict[str, int] = {}
    for k, v in stale_alignment_streaks_raw.items():
        inst = str(k or "").strip().upper()
        if not inst:
            continue
        try:
            streak = max(0, int(float(v) or 0))
        except Exception:
            streak = 0
        if streak > 0:
            stale_alignment_streaks[inst] = int(streak)
    for inst, until_ts in list(cooldown_until.items()):
        if float(until_ts) <= float(now_ts):
            cooldown_until.pop(str(inst).strip().upper(), None)
    loss_streak_auto_clear_msg = ""
    loss_cooldown_s = max(60, int(float(settings.get("forex_loss_cooldown_seconds", 1800) or 1800)))
    if loss_streak > 0:
        active_loss_cooldowns = bool(any(float(until_ts) > float(now_ts) for until_ts in cooldown_until.values()))
        has_pending_orders = bool(any(bool(v) for v in dict(pending).values()))
        # Prevent deadlock: if we are flat and all cooldowns are over, clear the streak guard automatically.
        if (not positions) and (not has_pending_orders) and (not active_loss_cooldowns):
            streak_age_s = (
                max(0, int(now_ts - int(loss_streak_updated_at)))
                if int(loss_streak_updated_at) > 0
                else int(loss_cooldown_s)
            )
            loss_streak = 0
            loss_streak_updated_at = int(now_ts)
            loss_streak_auto_clear_msg = (
                f"Loss-streak guard auto-cleared after {int(streak_age_s)}s flat with no active cooldowns."
            )
    loss_size_scale = max(loss_size_floor_pct, 1.0 - (loss_size_step_pct * float(max(0, loss_streak))))
    trade_units_effective = max(1, int(round(abs(float(trade_units)) * float(loss_size_scale))))
    fallback_active = bool(thinker.get("fallback_cached", False)) if isinstance(thinker, dict) else False
    try:
        fallback_age_s = int(float(thinker.get("fallback_age_s", 0) or 0)) if fallback_active else 0
    except Exception:
        fallback_age_s = 0
    thinker_reject_rate_pct, thinker_reject_rate_raw_pct = _effective_reject_pressure(settings, thinker if isinstance(thinker, dict) else {})
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    runtime_alerts = runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}
    entry_size_scale = 1.0
    allocator_size_scale = 1.0
    if fallback_active and (not block_cached_scan):
        entry_size_scale = float(cached_scan_entry_size_mult)
    trade_units_entry = max(1, int(round(float(trade_units_effective) * float(entry_size_scale))))
    risk_cap_size_scale = 1.0
    actions: List[str] = []
    thinker_health = thinker.get("health", {}) if isinstance(thinker, dict) else {}
    thinker_data_ok = bool((thinker_health or {}).get("data_ok", True))
    signal_age_policy_s = max(0, int(now_ts - int(float(thinker.get("updated_at", now_ts) or now_ts))))
    max_signal_age_policy_s = max(30, int(float(settings.get("forex_max_signal_age_seconds", 300) or 300)))
    stale_alignment_data_ok = bool(thinker_data_ok) and (signal_age_policy_s <= max_signal_age_policy_s)
    if fallback_active and fallback_age_s > cached_scan_hard_block_age_s:
        stale_alignment_data_ok = False
    drift_warning = False
    all_instruments = set(positions.keys())
    top_inst = str(top_pick.get("pair", "") or "").strip().upper()
    for row in candidate_rows[:16]:
        inst = str((row or {}).get("pair", "") or "").strip().upper()
        if inst:
            all_instruments.add(inst)
    pricing_details = client.get_pricing_details(sorted(all_instruments)) if all_instruments else {}
    prices: Dict[str, float] = {}
    for inst, row in list(pricing_details.items()):
        if not isinstance(row, dict):
            continue
        try:
            mid_px = float(row.get("mid", 0.0) or 0.0)
        except Exception:
            mid_px = 0.0
        if mid_px > 0.0:
            prices[str(inst).strip().upper()] = mid_px
    missing_prices = [inst for inst in sorted(all_instruments) if float(prices.get(inst, 0.0) or 0.0) <= 0.0]
    if missing_prices:
        for inst, mid_px in (client.get_mid_prices(missing_prices) or {}).items():
            try:
                px = float(mid_px or 0.0)
            except Exception:
                px = 0.0
            if px > 0.0:
                prices[str(inst).strip().upper()] = px

    nav = _safe_float_from_dict(broker_snap if isinstance(broker_snap, dict) else {}, ["nav", "NAV", "account_value_usd"])
    if nav <= 0.0:
        # Backward-compatible fallback for older snapshot schema.
        nav_text = str((broker_snap or {}).get("msg", "") or "")
        try:
            if "NAV" in nav_text:
                nav = float(nav_text.split("NAV", 1)[1].strip().split(" ", 1)[0])
        except Exception:
            nav = 0.0

    total_exposure_usd = 0.0
    total_margin_used_usd = 0.0
    position_values_usd: Dict[str, float] = {}
    for raw_row in raw_positions:
        if not isinstance(raw_row, dict):
            continue
        try:
            total_margin_used_usd += abs(float(raw_row.get("marginUsed", 0.0) or 0.0))
        except Exception:
            continue
    for inst, pos in positions.items():
        mid_px = float(prices.get(inst, 0.0) or 0.0)
        if mid_px <= 0:
            continue
        pricing_row = pricing_details.get(inst, {}) if isinstance(pricing_details.get(inst, {}), dict) else {}
        lu = abs(float(pos.get("long_units", 0.0) or 0.0))
        su = abs(float(pos.get("short_units", 0.0) or 0.0))
        unit_notional_usd = _forex_unit_notional_usd(inst, mid_px, pricing_row)
        inst_notional_usd = (lu + su) * unit_notional_usd
        total_exposure_usd += inst_notional_usd
        if inst_notional_usd > 0.0:
            position_values_usd[str(inst).strip().upper()] = round(float(inst_notional_usd), 6)
    effective_open_positions_hard_cap = _forex_effective_open_position_hard_cap(
        settings,
        configured_cap=max_open_positions,
        current_open_positions=len(positions),
    )
    margin_available = _safe_float_from_dict(
        broker_snap if isinstance(broker_snap, dict) else {},
        ["margin_available", "marginAvailable"],
    )
    if margin_available <= 0.0 and nav > 0.0:
        margin_available = max(0.0, float(nav) - float(total_margin_used_usd))
    margin_rate_est = _safe_float_from_dict(
        broker_snap if isinstance(broker_snap, dict) else {},
        ["margin_rate", "marginRate"],
    )
    if margin_rate_est <= 0.0 and total_exposure_usd > 0.0 and total_margin_used_usd > 0.0:
        margin_rate_est = float(total_margin_used_usd) / float(total_exposure_usd)
    if margin_rate_est <= 0.0:
        margin_rate_est = 0.05
    margin_rate_est = max(0.001, min(1.0, float(margin_rate_est)))
    policy = build_market_automation_policy(
        market="forex",
        settings=settings,
        profile_key=settings.get("settings_profile", "balanced"),
        broker_mode=_broker_mode_label(settings),
        account_value_usd=nav,
        buying_power_usd=margin_available,
        open_positions=len(positions),
        runtime_alerts=runtime_alerts,
        market_health=thinker_health,
        compliance_state={},
        reject_rate_pct=thinker_reject_rate_pct,
        reject_rate_limit_pct=reject_rate_gate_pct,
        fallback_active=fallback_active,
        fallback_age_s=fallback_age_s,
        fallback_hard_block_age_s=cached_scan_hard_block_age_s,
        loss_streak=loss_streak,
        max_loss_streak=max_loss_streak_setting,
    )
    policy_size_scale = max(0.2, float(policy.get("size_multiplier", 1.0) or 1.0))
    trade_units_entry = max(1, int(round(float(trade_units_entry) * float(policy_size_scale))))
    policy_block_reason = ""
    if not bool(policy.get("allow_new_entries", True)):
        policy_block_reason = "Runtime trust is degraded; pausing new forex entries"
    try:
        stale_exit_enabled = bool(settings.get("forex_stale_exit_enabled", True))
    except Exception:
        stale_exit_enabled = True
    try:
        stale_exit_grace_cycles = max(1, int(float(settings.get("forex_stale_alignment_grace_cycles", 2) or 2)))
    except Exception:
        stale_exit_grace_cycles = 2
    try:
        stale_exit_max_per_cycle = max(1, int(float(settings.get("forex_stale_max_exits_per_cycle", 2) or 2)))
    except Exception:
        stale_exit_max_per_cycle = 2
    try:
        stale_exit_min_notional_usd = max(1.0, float(settings.get("forex_stale_min_notional_usd", 5.0) or 5.0))
    except Exception:
        stale_exit_min_notional_usd = 5.0
    try:
        stale_exit_min_hold_s = max(0, int(float(settings.get("forex_stale_min_hold_seconds", 1800) or 1800)))
    except Exception:
        stale_exit_min_hold_s = 1800
    try:
        stale_exit_loss_cut_pct = float(settings.get("forex_stale_loss_cut_pct", -0.35) or -0.35)
    except Exception:
        stale_exit_loss_cut_pct = -0.35
    try:
        stale_exit_hold_near_flat_pct = max(
            0.0,
            float(settings.get("forex_stale_hold_near_flat_pct", 0.10) or 0.10),
        )
    except Exception:
        stale_exit_hold_near_flat_pct = 0.10
    try:
        stale_exit_reverse_score_mult = max(1.0, float(settings.get("forex_stale_reverse_score_mult", 1.25) or 1.25))
    except Exception:
        stale_exit_reverse_score_mult = 1.25
    try:
        stale_profit_hold_min_pct = max(0.0, float(settings.get("forex_stale_profit_hold_min_pct", 0.15) or 0.15))
    except Exception:
        stale_profit_hold_min_pct = 0.15
    try:
        stale_profit_hold_extra_cycles = max(0, int(float(settings.get("forex_stale_profit_hold_extra_cycles", 3) or 3)))
    except Exception:
        stale_profit_hold_extra_cycles = 3
    try:
        stale_profit_hold_max_s = max(0, int(float(settings.get("forex_stale_profit_hold_max_seconds", 21600) or 21600)))
    except Exception:
        stale_profit_hold_max_s = 21600
    try:
        stale_profit_hold_max_pullback_pct = max(
            0.0,
            float(settings.get("forex_stale_profit_hold_max_pullback_pct", 0.20) or 0.20),
        )
    except Exception:
        stale_profit_hold_max_pullback_pct = 0.20
    stale_exit_events: List[Dict[str, Any]] = []
    stale_exit_count = 0
    skip_new_entries_this_cycle = False
    stale_exit_refresh_msg = ""

    today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
    if pending:
        p_inst = str(pending.get("instrument", "") or "").strip().upper()
        p_ts = float(pending.get("ts", 0.0) or 0.0)
        if p_inst and p_inst in positions:
            actions.append(f"RECONCILE OK {p_inst} reflected")
            pending = {}
        elif p_inst and (now_ts - p_ts) > 90:
            drift_warning = True
            actions.append(f"RECONCILE WARN {p_inst} not reflected")
            _append_jsonl(
                audit_path,
                {"ts": now_ts, "date": today, "event": "reconcile_warning", "instrument": p_inst, "msg": "pending timed out"},
            )
            pending = {}
    for inst, pos in positions.items():
        mid_px = float(prices.get(inst, 0.0) or 0.0)
        if mid_px <= 0:
            continue
        side, pnl = _pnl_pct(pos, mid_px)
        meta = open_meta.get(inst, {}) or {}
        mfe = max(float(meta.get("mfe_pct", pnl) or pnl), pnl)
        mae = min(float(meta.get("mae_pct", pnl) or pnl), pnl)
        entry_ts = float(meta.get("entry_ts", now_ts) or now_ts)
        open_meta[inst] = {"entry_ts": entry_ts, "mfe_pct": mfe, "mae_pct": mae, "last_pnl_pct": pnl}
        units = abs(float(pos.get("long_units", 0.0) or 0.0)) + abs(float(pos.get("short_units", 0.0) or 0.0))
        avg_px = float(pos.get("long_avg", 0.0) or 0.0) if side == "long" else float(pos.get("short_avg", 0.0) or 0.0)
        pnl_usd = ((mid_px - avg_px) * units) if (side == "long" and avg_px > 0) else ((avg_px - mid_px) * units if avg_px > 0 else 0.0)
        st = trail_state.get(inst, {}) or {}
        armed = bool(st.get("armed", False))
        peak = float(st.get("peak_pct", pnl) or pnl)
        if pnl >= profit_target_pct:
            armed = True
            peak = max(peak, pnl)
        align_snapshot = _forex_alignment_snapshot(
            inst,
            position_side=side,
            candidate_lookup=candidate_lookup,
            required_score=float(alignment_required_score),
        )
        align_reasons = [str(x) for x in list(align_snapshot.get("reasons", []) or []) if str(x).strip()]
        if bool(stale_alignment_data_ok):
            if bool(align_snapshot.get("aligned", True)):
                stale_alignment_streaks.pop(inst, None)
            else:
                stale_alignment_streaks[inst] = int(stale_alignment_streaks.get(inst, 0) or 0) + 1
        stale_streak = int(stale_alignment_streaks.get(inst, 0) or 0)
        should_force_stale_exit = (
            bool(stale_alignment_data_ok)
            and bool(stale_exit_enabled)
            and (not bool(align_snapshot.get("aligned", True)))
            and stale_streak >= int(stale_exit_grace_cycles)
            and stale_exit_count < int(stale_exit_max_per_cycle)
        )
        if should_force_stale_exit:
            close_side = "long" if side == "long" else "short"
            pricing_row = pricing_details.get(inst, {}) if isinstance(pricing_details.get(inst, {}), dict) else {}
            unit_notional_usd = _forex_unit_notional_usd(inst, mid_px, pricing_row)
            est_notional_usd = abs(float(units)) * max(0.0, float(unit_notional_usd))
            reason_text = "; ".join([str(r) for r in align_reasons[:2]]) or "position no longer matches current forex strategy"
            entry_age_s = max(0, int(now_ts - entry_ts))
            align_side = str(align_snapshot.get("side", "watch") or "watch").strip().lower()
            try:
                align_score_abs = abs(float(align_snapshot.get("score", 0.0) or 0.0))
            except Exception:
                align_score_abs = 0.0
            hard_reverse = (
                align_side in {"long", "short"}
                and align_side != str(side or "").strip().lower()
                and align_score_abs >= (float(alignment_required_score) * float(stale_exit_reverse_score_mult))
            )
            mild_loss = (float(pnl) < 0.0) and (float(pnl) > float(stale_exit_loss_cut_pct))
            near_flat_gain = (float(pnl) >= 0.0) and (float(pnl) <= float(stale_exit_hold_near_flat_pct))
            hold_guard_reason = ""
            hold_guard_detail = ""
            hold_guard_active = False
            if entry_age_s < int(stale_exit_min_hold_s) and (not hard_reverse):
                if mild_loss:
                    hold_guard_active = True
                    hold_guard_reason = "stale_exit_hold_loss_guard"
                    hold_guard_detail = (
                        f"Alignment stale but holding {inst} to avoid a churn exit at mild loss "
                        f"({float(pnl):+.3f}% > {float(stale_exit_loss_cut_pct):+.3f}%) "
                        f"during the first {int(stale_exit_min_hold_s)}s."
                    )
                elif near_flat_gain:
                    hold_guard_active = True
                    hold_guard_reason = "stale_exit_hold_churn_guard"
                    hold_guard_detail = (
                        f"Alignment stale but holding {inst} to avoid a churn exit at a near-flat gain "
                        f"({float(pnl):+.3f}% <= +{float(stale_exit_hold_near_flat_pct):.3f}%) "
                        f"during the first {int(stale_exit_min_hold_s)}s."
                    )
            if hold_guard_active:
                stale_exit_events.append(
                    {
                        "instrument": str(inst),
                        "ok": False,
                        "reason": hold_guard_reason,
                        "detail": hold_guard_detail,
                        "streak": int(stale_streak),
                        "age_s": int(entry_age_s),
                        "reasons": [str(r) for r in align_reasons[:3]],
                    }
                )
                trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                continue
            trend_support = (
                align_side in {"long", "short"}
                and align_side == str(side or "").strip().lower()
                and align_score_abs >= float(alignment_required_score)
                and bool(align_snapshot.get("eligible_for_entry", True))
            )
            pullback_pct = max(0.0, float(mfe) - float(pnl))
            stale_profit_guard = evaluate_stale_profit_hold(
                pnl_pct=float(pnl),
                stale_streak=int(stale_streak),
                grace_cycles=int(stale_exit_grace_cycles),
                position_age_s=int(entry_age_s),
                hard_reverse=bool(hard_reverse),
                trend_support=bool(trend_support),
                profit_hold_min_pct=float(stale_profit_hold_min_pct),
                profit_hold_extra_cycles=int(stale_profit_hold_extra_cycles),
                profit_hold_max_s=int(stale_profit_hold_max_s),
                pullback_pct=float(pullback_pct),
                profit_hold_max_pullback_pct=float(stale_profit_hold_max_pullback_pct),
            )
            if bool(stale_profit_guard.get("hold", False)):
                guard_detail = str(stale_profit_guard.get("detail", "") or "").strip()
                stale_exit_events.append(
                    {
                        "instrument": str(inst),
                        "ok": False,
                        "reason": "stale_exit_profit_guard",
                        "detail": guard_detail or "stale-profit hold guard active",
                        "streak": int(stale_streak),
                        "age_s": int(entry_age_s),
                        "reasons": [str(r) for r in align_reasons[:3]],
                    }
                )
                trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                continue
            if est_notional_usd < float(stale_exit_min_notional_usd):
                stale_exit_events.append(
                    {
                        "instrument": str(inst),
                        "ok": False,
                        "reason": "stale_exit_notional_guard",
                        "detail": (
                            f"Alignment stale but notional guard blocked exit "
                            f"({est_notional_usd:.2f} USD < {float(stale_exit_min_notional_usd):.2f} USD)."
                        ),
                        "streak": int(stale_streak),
                        "reasons": [str(r) for r in align_reasons[:3]],
                    }
                )
                trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                continue
            ok, msg, payload = client.close_position(inst, side=close_side)
            realized_close_pnl = _realized_pnl_from_close_payload(payload if isinstance(payload, dict) else {})
            pnl_for_audit = float(realized_close_pnl) if realized_close_pnl is not None else float(pnl_usd)
            pnl_pct_for_audit = _align_pnl_pct_sign_with_realized(
                pnl_pct_est=float(pnl),
                pnl_usd_est=float(pnl_usd),
                realized_pnl_usd=realized_close_pnl,
            )
            actions.append(f"POLICY STALE EXIT {inst} {close_side} | {'OK' if ok else 'FAIL'} | {msg}")
            _append_jsonl(
                audit_path,
                {
                    "ts": now_ts,
                    "date": today,
                    "event": "exit" if ok else "exit_fail",
                    "source": "policy_stale_exit",
                    "instrument": inst,
                    "side": close_side,
                    "units": units,
                    "price": mid_px,
                    "pnl_pct": pnl_pct_for_audit,
                    "pnl_pct_est": float(pnl),
                    "pnl_usd": pnl_for_audit,
                    "pnl_usd_est": float(pnl_usd),
                    "realized_pnl": pnl_for_audit if ok else None,
                    "mfe_pct": round(mfe, 4),
                    "mae_pct": round(mae, 4),
                    "hold_s": max(0, int(now_ts - entry_ts)),
                    "stale_alignment_streak": int(stale_streak),
                    "stale_alignment_reasons": [str(r) for r in align_reasons[:3]],
                    "ok": ok,
                    "msg": msg,
                    "payload": payload if isinstance(payload, dict) else {},
                },
            )
            if ok:
                stale_exit_count += 1
                skip_new_entries_this_cycle = True
                trail_state.pop(inst, None)
                open_meta.pop(inst, None)
                stale_alignment_streaks.pop(inst, None)
                if pnl_for_audit < 0:
                    loss_streak += 1
                    loss_streak_updated_at = int(now_ts)
                    cooldown_until[inst] = float(now_ts + int(loss_cooldown_s))
                else:
                    loss_streak = 0
                    loss_streak_updated_at = int(now_ts)
            else:
                trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
            stale_exit_events.append(
                {
                    "instrument": str(inst),
                    "ok": bool(ok),
                    "reason": ("policy_stale_exit" if ok else "broker_close_failed"),
                    "detail": reason_text,
                    "streak": int(stale_streak),
                    "reasons": [str(r) for r in align_reasons[:3]],
                }
            )
            continue
        if armed:
            peak = max(peak, pnl)
            if pnl <= (peak - trailing_gap_pct):
                close_side = "long" if side == "long" else "short"
                ok, msg, payload = client.close_position(inst, side=close_side)
                realized_close_pnl = _realized_pnl_from_close_payload(payload if isinstance(payload, dict) else {})
                pnl_for_audit = float(realized_close_pnl) if realized_close_pnl is not None else float(pnl_usd)
                pnl_pct_for_audit = _align_pnl_pct_sign_with_realized(
                    pnl_pct_est=float(pnl),
                    pnl_usd_est=float(pnl_usd),
                    realized_pnl_usd=realized_close_pnl,
                )
                actions.append(f"CLOSE {inst} {close_side} | {'OK' if ok else 'FAIL'} | {msg}")
                _append_jsonl(
                    audit_path,
                    {
                        "ts": now_ts,
                        "date": today,
                        "event": "exit" if ok else "exit_fail",
                        "instrument": inst,
                        "side": close_side,
                        "units": units,
                        "price": mid_px,
                        "pnl_pct": pnl_pct_for_audit,
                        "pnl_pct_est": float(pnl),
                        "pnl_usd": pnl_for_audit,
                        "pnl_usd_est": float(pnl_usd),
                        "realized_pnl": pnl_for_audit if ok else None,
                        "mfe_pct": round(mfe, 4),
                        "mae_pct": round(mae, 4),
                        "hold_s": max(0, int(now_ts - entry_ts)),
                        "ok": ok,
                        "msg": msg,
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
                if ok:
                    trail_state.pop(inst, None)
                    open_meta.pop(inst, None)
                    if pnl_for_audit < 0:
                        loss_streak += 1
                        loss_streak_updated_at = int(now_ts)
                        cooldown_until[inst] = float(now_ts + int(loss_cooldown_s))
                    else:
                        loss_streak = 0
                        loss_streak_updated_at = int(now_ts)
                else:
                    trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
                continue
        trail_state[inst] = {"armed": armed, "peak_pct": peak, "last_pnl_pct": pnl, "updated_at": now_ts}
    for tracked_inst in list(stale_alignment_streaks.keys()):
        if str(tracked_inst or "").strip().upper() not in positions:
            stale_alignment_streaks.pop(str(tracked_inst or "").strip().upper(), None)
    for tracked_inst in list(open_meta.keys()):
        if str(tracked_inst or "").strip().upper() not in positions:
            open_meta.pop(str(tracked_inst or "").strip().upper(), None)
    for tracked_inst in list(trail_state.keys()):
        if str(tracked_inst or "").strip().upper() not in positions:
            trail_state.pop(str(tracked_inst or "").strip().upper(), None)
    if stale_exit_count > 0:
        stale_exit_refresh_msg = (
            f"Exited {stale_exit_count} stale forex position(s); waiting one cycle before new entries."
        )

    signal_inst = top_inst
    signal_side = str(top_pick.get("side", "watch") or "watch").strip().lower()
    signal_score = float(top_pick.get("score", 0.0) or 0.0)
    entry_fail_reasons: List[str] = []
    trade_quality_eval: Dict[str, Any] = {}
    opportunity_eval: Dict[str, Any] = {}
    entry_msg = "Auto-trade disabled"
    if auto_enabled:
        signal_age_s = max(0, int(now_ts - int(float(thinker.get("updated_at", now_ts) or now_ts))))
        max_signal_age_s = max(30, int(float(settings.get("forex_max_signal_age_seconds", 300) or 300)))
        min_bars_required = max(8, int(float(settings.get("forex_min_bars_required", 24) or 24)))
        min_samples_guarded = max(0, int(float(settings.get("forex_min_samples_live_guarded", 5) or 5)))
        required_score = float(alignment_required_score)
        max_slippage_bps = max(0.0, float(settings.get("forex_max_slippage_bps", 6.0) or 6.0))
        global_cap_pct = max(0.0, float(settings.get("market_max_total_exposure_pct", 0.0) or 0.0))
        if independent_market_mode:
            global_cap_pct = 0.0
        crypto_exposure_usd = _crypto_holdings_usd(hub_dir)
        stocks_exposure_usd = _market_status_exposure_usd(hub_dir, "stocks")
        if independent_market_mode:
            cross_market_exposure_usd = float(total_exposure_usd)
            cross_market_cap_basis_usd = float(max(0.0, nav))
        else:
            cross_market_exposure_usd = (
                float(total_exposure_usd)
                + float(max(0.0, crypto_exposure_usd))
                + float(max(0.0, stocks_exposure_usd))
            )
            cross_market_cap_basis_usd = _portfolio_account_value_usd(
                hub_dir,
                current_market="forex",
                current_account_value_usd=float(nav),
            )
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
        elif max_loss_streak_setting > 0 and loss_streak >= max_loss_streak_setting:
            entry_msg = f"Loss-streak guard active ({loss_streak}/{max_loss_streak_setting})"
        elif policy_block_reason:
            entry_msg = policy_block_reason
        elif skip_new_entries_this_cycle:
            entry_msg = stale_exit_refresh_msg or "Stale forex exit cooldown: waiting one cycle before new entries"
        elif nav <= 0.0:
            entry_msg = "NAV unavailable; blocking new entries for safety"
        elif _session_blocked(settings):
            entry_msg = "Session gate: blocked for current UTC hour"
        elif _daily_loss_guard_triggered(audit_path, max_daily_loss_usd, max_daily_loss_pct, nav):
            entry_msg = "Daily loss guard active: blocking new entries"
        elif not enable_exec_v2:
            entry_msg = "Execution gated by rollout stage"
        else:
            fail_reasons: List[str] = []
            selected_pair = ""
            selected_side = "watch"
            selected_score = 0.0
            selected_calib_prob = 0.0
            selected_samples = 0
            selected_calibration_scope = "pair"
            selected_bars = 0
            selected_mid = 0.0
            selected_spread_bps = 0.0
            selected_units = 0
            selected_risk_cap_size_scale = 1.0
            selected_quality_eval: Dict[str, Any] = {}
            selected_opportunity_eval: Dict[str, Any] = {}
            for cand in candidate_rows:
                pair = str((cand or {}).get("pair", "") or "").strip().upper()
                if not pair:
                    continue
                score = float(cand.get("score", 0.0) or 0.0)
                side = str(cand.get("side", "watch") or "watch").strip().lower()
                calib_prob = float(cand.get("calibration_effective_prob", cand.get("calib_prob", 0.0)) or 0.0)
                sample_count = int(float(cand.get("calibration_effective_samples", cand.get("samples", 0)) or 0))
                calibration_scope = str(cand.get("calibration_scope", "pair") or "pair").strip().lower() or "pair"
                bars_count = int(float(cand.get("bars_count", 0) or 0))
                pricing_row = pricing_details.get(pair, {}) if isinstance(pricing_details.get(pair, {}), dict) else {}
                mid = float(pricing_row.get("mid", 0.0) or 0.0)
                spread_bps = float(pricing_row.get("spread_bps", 0.0) or 0.0)
                raw_units = int(trade_units_entry)
                if side == "short":
                    raw_units = -raw_units
                if live_guarded and (calib_prob <= 0.0):
                    calib_prob = 0.5
                unit_notional_usd = _forex_unit_notional_usd(
                    pair,
                    (mid if mid > 0.0 else float(prices.get(pair, 0.0) or 0.0)),
                    pricing_row,
                )
                fail = ""
                if bars_count > 0 and bars_count < min_bars_required:
                    fail = f"Bars preflight failed for {pair} ({bars_count} < {min_bars_required})"
                elif str(cand.get("entry_gate_reason", "") or "").strip():
                    fail = str(cand.get("entry_gate_reason", "") or "").strip()
                elif side not in ("long", "short"):
                    fail = f"Top pair {pair} is WATCH"
                elif not bool(cand.get("eligible_for_entry", True)):
                    fail = f"Universe health gate blocked {pair}"
                elif not bool(cand.get("data_quality_ok", True)):
                    fail = f"Data quality gate blocked {pair}"
                elif mid <= 0.0:
                    fail = f"Quote preflight failed for {pair}"
                elif pair in positions:
                    fail = f"Already in position: {pair}"
                elif len(positions) >= max_open_positions:
                    if (not forex_exposure_slot_mode) or (len(positions) >= int(effective_open_positions_hard_cap)):
                        cap_txt = int(effective_open_positions_hard_cap) if forex_exposure_slot_mode else int(max_open_positions)
                        mode_txt = "Exposure-budget cap" if forex_exposure_slot_mode else "Max open positions reached"
                        fail = f"{mode_txt} ({len(positions)}/{cap_txt})"
                elif live_guarded and sample_count < min_samples_guarded:
                    fail = f"Calibration sample gate for {pair} ({sample_count} < {min_samples_guarded})"
                elif live_guarded and calib_prob < float(settings.get("forex_min_calib_prob_live_guarded", 0.56) or 0.56):
                    fail = f"Calibrated confidence gate for {pair} ({calib_prob:.2f})"
                elif abs(score) < required_score:
                    fail = f"Top score below threshold for {pair} ({score:.4f} < {required_score:.4f})"
                elif float(cooldown_until.get(pair, 0.0) or 0.0) > float(now_ts):
                    cd_left = int(float(cooldown_until.get(pair, 0.0) - now_ts))
                    fail = f"Cooldown active for {pair} ({cd_left}s)"
                units = int(raw_units)
                pair_risk_cap_size_scale = 1.0
                est_entry_notional = abs(float(units)) * unit_notional_usd
                est_entry_margin = abs(float(units)) * (unit_notional_usd * margin_rate_est)
                if not fail:
                    units, pair_risk_cap_size_scale = _risk_capped_units(
                        units,
                        unit_notional_usd=unit_notional_usd,
                        unit_margin_usd=(unit_notional_usd * margin_rate_est),
                        nav=nav,
                        total_exposure_usd=total_exposure_usd,
                        total_margin_used_usd=total_margin_used_usd,
                        margin_available_usd=margin_available,
                        crypto_exposure_usd=crypto_exposure_usd,
                        stocks_exposure_usd=stocks_exposure_usd,
                        max_total_exposure_pct=(max_total_exposure_pct if enable_risk_caps else 0.0),
                        max_pos_usd=(max_pos_usd if enable_risk_caps else 0.0),
                        global_cap_pct=global_cap_pct,
                        global_cap_account_value_usd=cross_market_cap_basis_usd,
                    )
                    est_entry_notional = abs(float(units)) * unit_notional_usd
                    est_entry_margin = abs(float(units)) * (unit_notional_usd * margin_rate_est)
                    if max_slippage_bps > 0.0 and spread_bps > max_slippage_bps:
                        fail = f"Slippage guard for {pair}: spread {spread_bps:.2f}bps > {max_slippage_bps:.2f}bps"
                    elif abs(int(units)) <= 0:
                        if global_cap_pct > 0.0 and cross_market_cap_basis_usd > 0.0:
                            allowed_global_notional = (float(cross_market_cap_basis_usd) * float(global_cap_pct) / 100.0)
                            consumed_global_notional = float(cross_market_exposure_usd)
                            if consumed_global_notional >= allowed_global_notional:
                                fail = (
                                    f"Global cap: cross-market exposure already uses "
                                    f"${consumed_global_notional:.2f}/${allowed_global_notional:.2f}"
                                )
                        if not fail:
                            fail = f"Risk cap: no tradable size fits current margin/exposure for {pair}"
                    elif enable_risk_caps and max_pos_usd > 0.0 and est_entry_notional > max_pos_usd:
                        fail = f"Risk cap: projected pair notional exceeds ${max_pos_usd:.2f}"
                    elif enable_risk_caps and max_total_exposure_pct > 0.0 and nav > 0.0:
                        projected_margin_pct = ((total_margin_used_usd + max(0.0, est_entry_margin)) / nav) * 100.0
                        if projected_margin_pct > max_total_exposure_pct:
                            fail = f"Risk cap: projected margin utilization exceeds {max_total_exposure_pct:.2f}%"
                    elif global_cap_pct > 0.0 and cross_market_cap_basis_usd > 0.0:
                        projected_global_exposure = cross_market_exposure_usd + max(0.0, est_entry_notional)
                        if ((projected_global_exposure / cross_market_cap_basis_usd) * 100.0) > global_cap_pct:
                            fail = f"Global cap: projected cross-market exposure exceeds {global_cap_pct:.2f}%"
                quality_eval: Dict[str, Any] = {}
                if not fail:
                    projected_exposure_pct = ((total_exposure_usd + max(0.0, est_entry_notional)) / nav) * 100.0 if nav > 0.0 else 0.0
                    quality_eval = evaluate_trade_quality(
                        market="forex",
                        signal_score=score,
                        required_score=required_score,
                        data_quality_ok=bool(cand.get("data_quality_ok", True)) and bool(thinker_data_ok),
                        broker_ok=True,
                        runtime_trust_score=float(
                            (
                                (policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}).get(
                                    "score",
                                    0.0,
                                )
                            )
                            or 0.0
                        ),
                        runtime_alert_severity=str(runtime_alerts.get("severity", "ok") or "ok"),
                        compliance_allowed=bool(policy.get("allow_new_entries", True)),
                        compliance_reason=str(policy_block_reason or "Forex runtime trust protection active"),
                        fallback_active=bool(fallback_active),
                        fallback_age_s=int(fallback_age_s),
                        fallback_hard_block_age_s=int(cached_scan_hard_block_age_s),
                        reject_rate_pct=float(thinker_reject_rate_pct),
                        reject_rate_limit_pct=float(reject_rate_gate_pct),
                        spread_bps=float(spread_bps),
                        max_slippage_bps=float(max_slippage_bps),
                        loss_streak=int(loss_streak),
                        max_loss_streak=int(max_loss_streak_setting),
                        exposure_usage_pct=float(projected_exposure_pct),
                        min_runtime_trust_score=34.0,
                        min_confidence_score=38.0,
                    )
                    if str(quality_eval.get("decision", "block") or "block").strip().lower() != "allow":
                        q_reasons = quality_eval.get("block_reasons", []) if isinstance(quality_eval.get("block_reasons", []), list) else []
                        q_reason = str((q_reasons[0] if q_reasons else "trade quality gate blocked entry") or "").strip()
                        trade_quality_eval = dict(quality_eval)
                        fail = f"Trade-quality gate: {q_reason}"
                    else:
                        quality_units_mult = max(0.20, min(1.0, float(quality_eval.get("size_multiplier", 1.0) or 1.0)))
                        units_abs = int(round(abs(float(units)) * quality_units_mult))
                        if units_abs <= 0:
                            fail = f"Trade-quality gate: no tradable size remains for {pair}"
                        else:
                            units = units_abs if int(units) >= 0 else (-1 * units_abs)
                            selected_quality_eval = dict(quality_eval)
                            trade_quality_eval = dict(quality_eval)
                if not fail:
                    est_notional_for_allocator = abs(float(units)) * unit_notional_usd
                    allocator_eval = evaluate_cross_market_allocation(
                        hub_dir=hub_dir,
                        settings=settings,
                        market="forex",
                        candidate_id=pair,
                        candidate_side=side,
                        signal_score=float(score),
                        required_score=float(required_score),
                        trade_quality=quality_eval if isinstance(quality_eval, dict) else {},
                        automation_policy=policy,
                        projected_trade_value_usd=float(max(0.0, est_notional_for_allocator)),
                        market_exposure_usd=float(total_exposure_usd),
                        account_value_usd=float(nav),
                        buying_power_usd=float(margin_available),
                        spread_bps=float(spread_bps),
                        max_slippage_bps=float(max_slippage_bps),
                        candidate_age_s=int(signal_age_s),
                        loss_streak=int(loss_streak),
                        max_loss_streak=int(max_loss_streak_setting),
                        now_ts=int(now_ts),
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
                            or "portfolio allocator deprioritized this entry"
                        ).strip()
                        fail = f"Portfolio allocator: {alloc_reason}"
                        trade_quality_eval = dict(quality_eval) if isinstance(quality_eval, dict) else {}
                    else:
                        allocator_size_mult = max(
                            0.25,
                            min(
                                1.0,
                                float(
                                    (
                                        allocator_eval.get("size_multiplier", 1.0)
                                        if isinstance(allocator_eval, dict)
                                        else 1.0
                                    )
                                    or 1.0
                                ),
                            ),
                        )
                        if abs(float(allocator_size_mult) - 1.0) >= 0.001:
                            units_abs = max(1, int(round(abs(float(units)) * float(allocator_size_mult))))
                            units = units_abs if int(units) >= 0 else (-1 * units_abs)
                        selected_opportunity_eval = dict(allocator_eval)
                        allocator_size_scale = float(allocator_size_mult)
                        opportunity_eval = dict(allocator_eval)
                if fail:
                    fail_reasons.append(fail)
                    continue
                selected_pair = pair
                selected_side = side
                selected_score = score
                selected_calib_prob = calib_prob
                selected_samples = sample_count
                selected_calibration_scope = calibration_scope
                selected_bars = bars_count
                selected_mid = mid
                selected_spread_bps = spread_bps
                selected_units = units
                selected_risk_cap_size_scale = pair_risk_cap_size_scale
                break
            if not selected_pair:
                entry_msg = fail_reasons[0] if fail_reasons else "No pairs available from thinker"
                if fail_reasons:
                    entry_fail_reasons.extend([str(x) for x in fail_reasons[:24] if str(x).strip()])
            elif shadow_only:
                signal_inst = selected_pair
                signal_side = selected_side
                signal_score = selected_score
                risk_cap_size_scale = float(selected_risk_cap_size_scale)
                entry_msg = f"SHADOW entry simulated for {selected_pair}"
                actions.append(f"SHADOW ENTRY {selected_pair} {selected_side.upper()} units={selected_units}")
                _append_jsonl(
                    audit_path,
                    {
                        "ts": now_ts,
                        "date": today,
                        "event": "shadow_entry",
                        "instrument": selected_pair,
                        "side": selected_side,
                        "units": int(selected_units),
                        "configured_units": int(abs(trade_units)),
                        "entry_size_scale": float(round(entry_size_scale, 4)),
                        "policy_size_scale": float(round(policy_size_scale, 4)),
                        "score": selected_score,
                        "calib_prob": selected_calib_prob,
                        "samples": selected_samples,
                        "calibration_scope": selected_calibration_scope,
                        "bars_count": selected_bars,
                        "spread_bps": selected_spread_bps,
                        "risk_cap_size_scale": float(round(risk_cap_size_scale, 4)),
                        "trade_quality": selected_quality_eval if isinstance(selected_quality_eval, dict) else {},
                        "opportunity_allocator": selected_opportunity_eval if isinstance(selected_opportunity_eval, dict) else {},
                        "ok": True,
                        "msg": "shadow_only stage",
                    },
                )
            else:
                signal_inst = selected_pair
                signal_side = selected_side
                signal_score = selected_score
                risk_cap_size_scale = float(selected_risk_cap_size_scale)
                client_id = f"ptfx-{selected_pair}-{now_ts}"
                ok, msg, payload = client.place_market_order(
                    selected_pair,
                    selected_units,
                    client_order_id=client_id,
                    max_retries=max(1, int(float(settings.get("forex_order_retry_count", 2) or 2))),
                    max_retry_after_s=max(1.0, float(settings.get("broker_order_retry_after_cap_s", 300.0) or 300.0)),
                )
                actions.append(f"ENTRY {selected_pair} {selected_side.upper()} units={selected_units} | {'OK' if ok else 'FAIL'} | {msg}")
                oid = _parse_order_id(msg, payload if isinstance(payload, dict) else {})
                retry_after_wait_s = parse_retry_after_value(str(msg or ""), max_wait_s=3600.0)
                if retry_after_wait_s > 0.0:
                    runtime_event(
                        runtime_events_path,
                        component="forex_trader",
                        event="broker_retry_after_wait",
                        level="warning",
                        msg=f"Forex broker retry-after wait {retry_after_wait_s:.2f}s",
                        details={
                            "market": "forex",
                            "pair": selected_pair,
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
                        "instrument": selected_pair,
                        "side": selected_side,
                        "units": selected_units,
                        "entry_size_scale": float(round(entry_size_scale, 4)),
                        "policy_size_scale": float(round(policy_size_scale, 4)),
                        "score": selected_score,
                        "calib_prob": selected_calib_prob,
                        "samples": selected_samples,
                        "calibration_scope": selected_calibration_scope,
                        "bars_count": selected_bars,
                        "price": selected_mid,
                        "spread_bps": selected_spread_bps,
                        "risk_cap_size_scale": float(round(risk_cap_size_scale, 4)),
                        "trade_quality": selected_quality_eval if isinstance(selected_quality_eval, dict) else {},
                        "opportunity_allocator": selected_opportunity_eval if isinstance(selected_opportunity_eval, dict) else {},
                        "client_order_id": client_id,
                        "order_id": oid,
                        "retry_after_wait_s": float(round(retry_after_wait_s, 3)),
                        "ok": ok,
                        "msg": msg,
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
                entry_msg = f"Entry {'placed' if ok else 'failed'} for {selected_pair}"
                if ok:
                    open_meta[selected_pair] = {"entry_ts": now_ts, "mfe_pct": 0.0, "mae_pct": 0.0, "last_pnl_pct": 0.0}
                    pending = {"instrument": selected_pair, "side": selected_side, "units": selected_units, "ts": now_ts, "order_id": oid, "client_order_id": client_id}
                    recon_positions = _parse_positions(list((client.fetch_snapshot() or {}).get("raw_positions", []) or []))
                    if selected_pair not in recon_positions:
                        drift_warning = True
                        _append_jsonl(
                            audit_path,
                            {
                                "ts": now_ts,
                                "date": today,
                                "event": "reconcile_warning",
                                "instrument": selected_pair,
                                "msg": "submitted but not reflected in open positions",
                            },
                        )
    else:
        entry_msg = "Auto-trade disabled (practice-safe)"
    if auto_enabled and ("Entry placed" not in str(entry_msg)):
        entry_fail_reasons.append(str(entry_msg))
    entry_eval_top_reason, entry_eval_reason_counts = _fail_reason_summary(entry_fail_reasons)
    runtime_trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
    quality_layers = trade_quality_eval.get("layers", {}) if isinstance(trade_quality_eval.get("layers", {}), dict) else {}
    trade_quality_evaluated = bool(isinstance(trade_quality_eval, dict) and trade_quality_eval)
    allocator_evaluated = bool(isinstance(opportunity_eval, dict) and opportunity_eval)
    allocator_reasons = opportunity_eval.get("reasons", []) if isinstance(opportunity_eval.get("reasons", []), list) else []
    allocator_top_reason = str((allocator_reasons[0] if allocator_reasons else "") or "").strip()
    allocator_ai = opportunity_eval.get("openai_decision", {}) if isinstance(opportunity_eval.get("openai_decision", {}), dict) else {}
    trade_confidence_score = float(trade_quality_eval.get("confidence_score", 0.0) or 0.0) if isinstance(trade_quality_eval, dict) else 0.0
    quality_size_scale = float(trade_quality_eval.get("size_multiplier", 1.0) or 1.0) if isinstance(trade_quality_eval, dict) else 1.0
    try:
        allocator_size_from_eval = float(opportunity_eval.get("size_multiplier", 1.0) or 1.0)
    except Exception:
        allocator_size_from_eval = 1.0
    allocator_size_from_eval = max(0.25, min(1.0, float(allocator_size_from_eval)))
    allocator_size_scale = float(allocator_size_from_eval)
    try:
        openai_confidence = float(allocator_ai.get("portfolio_confidence", 0.0) or 0.0)
    except Exception:
        openai_confidence = 0.0
    openai_confidence = max(0.0, min(1.0, float(openai_confidence)))
    if auto_enabled:
        cross_market_exposure_effective_usd = float(cross_market_exposure_usd)
        cross_market_cap_basis_effective_usd = float(cross_market_cap_basis_usd)
    elif independent_market_mode:
        cross_market_exposure_effective_usd = float(total_exposure_usd)
        cross_market_cap_basis_effective_usd = float(max(0.0, nav))
    else:
        cross_market_exposure_effective_usd = (
            float(total_exposure_usd)
            + float(_crypto_holdings_usd(hub_dir))
            + float(_market_status_exposure_usd(hub_dir, "stocks"))
        )
        cross_market_cap_basis_effective_usd = float(
            _portfolio_account_value_usd(
                hub_dir,
                current_market="forex",
                current_account_value_usd=float(nav),
            )
        )
    cross_market_exposure_effective_pct = (
        (cross_market_exposure_effective_usd / max(1e-6, cross_market_cap_basis_effective_usd)) * 100.0
    )
    entry_gate_flags = {
        "data_quality_required": bool(require_data_quality_ok),
        "data_quality_ok": bool(thinker_data_ok),
        "reject_rate_pct": float(round(thinker_reject_rate_pct, 4)),
        "reject_rate_raw_pct": float(round(thinker_reject_rate_raw_pct, 4)),
        "reject_rate_max_pct": float(round(reject_rate_gate_pct, 4)),
        "cached_fallback_active": bool(fallback_active),
        "cached_fallback_age_s": int(fallback_age_s),
        "cached_fallback_hard_block_age_s": int(cached_scan_hard_block_age_s),
        "stale_alignment_data_ok": bool(stale_alignment_data_ok),
        "stale_alignment_signal_age_s": int(signal_age_policy_s),
        "stale_alignment_max_signal_age_s": int(max_signal_age_policy_s),
        "runtime_trust_score": float(round(float(runtime_trust.get("score", 0.0) or 0.0), 4)),
        "runtime_trust_mode": str(runtime_trust.get("mode", "") or ""),
        "policy_mode": str(policy.get("mode", "") or ""),
        "policy_profile": str(policy.get("profile", "") or ""),
        "independent_market_mode": bool(independent_market_mode),
        "policy_size_scale": float(round(policy_size_scale, 4)),
        "loss_streak": int(loss_streak),
        "max_loss_streak": int(max_loss_streak_setting),
        "trade_quality_evaluated": bool(trade_quality_evaluated),
        "trade_quality_decision": str(
            trade_quality_eval.get("decision", "not_evaluated") if trade_quality_evaluated else "not_evaluated"
        ),
        "trade_confidence_score": float(round(trade_confidence_score, 4)),
        "trade_quality_size_scale": float(round(quality_size_scale, 4)),
        "portfolio_allocator_evaluated": bool(allocator_evaluated),
        "portfolio_allocator_decision": str(
            opportunity_eval.get("decision", "not_evaluated") if allocator_evaluated else "not_evaluated"
        ),
        "portfolio_allocator_best_market": str(opportunity_eval.get("best_market", "") or ""),
        "portfolio_allocator_score": float(round(float(opportunity_eval.get("current_market_score", 0.0) or 0.0), 4)),
        "portfolio_allocator_top_reason": allocator_top_reason,
        "portfolio_allocator_capital_constrained": bool(opportunity_eval.get("capital_constrained", False)) if allocator_evaluated else False,
        "portfolio_allocator_size_scale": float(round(float(allocator_size_from_eval), 4)) if allocator_evaluated else 1.0,
        "portfolio_allocator_decision_source": str(opportunity_eval.get("decision_source", "") or "") if allocator_evaluated else "",
        "openai_decision_active": bool(allocator_ai.get("active", False)),
        "openai_decision_status": str(allocator_ai.get("status", "") or ""),
        "openai_decision": str(allocator_ai.get("decision", "") or ""),
        "openai_decision_best_market": str(allocator_ai.get("best_market", "") or ""),
        "openai_decision_confidence": float(round(openai_confidence, 4)),
        "openai_decision_applied": bool(allocator_ai.get("applied", False)),
        "signal_quality_pass": bool(quality_layers.get("signal_quality", False)),
        "execution_quality_pass": bool(quality_layers.get("execution_quality", False)),
        "compliance_permission_pass": bool(quality_layers.get("compliance_permission", False)),
        "runtime_trust_pass": bool(quality_layers.get("runtime_trust", False)),
        "alignment_required_score": float(round(float(alignment_required_score), 6)),
        "stale_exit_enabled": bool(stale_exit_enabled),
        "stale_exit_grace_cycles": int(stale_exit_grace_cycles),
        "stale_exit_count": int(stale_exit_count),
        "stale_exit_max_per_cycle": int(stale_exit_max_per_cycle),
        "stale_exit_min_notional_usd": float(round(float(stale_exit_min_notional_usd), 4)),
        "stale_exit_min_hold_s": int(stale_exit_min_hold_s),
        "stale_exit_loss_cut_pct": float(round(float(stale_exit_loss_cut_pct), 4)),
        "stale_exit_hold_near_flat_pct": float(round(float(stale_exit_hold_near_flat_pct), 4)),
        "stale_exit_reverse_score_mult": float(round(float(stale_exit_reverse_score_mult), 4)),
        "skip_new_entries_this_cycle": bool(skip_new_entries_this_cycle),
        "position_cap_mode": ("exposure_budget" if forex_exposure_slot_mode else "count"),
        "max_open_positions_setting": int(max_open_positions),
        "max_open_positions_effective_hard": int(effective_open_positions_hard_cap),
        "cross_market_exposure_usd": float(round(cross_market_exposure_effective_usd, 4)),
        "cross_market_cap_basis_usd": float(round(cross_market_cap_basis_effective_usd, 4)),
        "cross_market_exposure_pct": float(round(cross_market_exposure_effective_pct, 4)),
    }

    out_state = {
        "trail": trail_state,
        "cooldown_until": cooldown_until,
        "loss_streak": int(loss_streak),
        "loss_streak_updated_at": int(loss_streak_updated_at),
        "open_meta": open_meta,
        "pending": pending,
        "stale_alignment_streaks": dict(stale_alignment_streaks),
        "last_divergence_ts": int(last_divergence_ts),
        "last_divergence_msg": str(last_divergence_msg),
        "entry_eval_total": int(len(entry_fail_reasons)),
        "entry_eval_top_reason": str(entry_eval_top_reason),
        "entry_eval_reason_counts": dict(entry_eval_reason_counts),
        "automation_policy": policy if isinstance(policy, dict) else {},
        "trade_quality": trade_quality_eval if isinstance(trade_quality_eval, dict) else {},
        "opportunity_allocator": opportunity_eval if isinstance(opportunity_eval, dict) else {},
        "entry_gate_flags": dict(entry_gate_flags),
        "trade_units_entry": int(trade_units_entry),
        "entry_size_scale": round(float(entry_size_scale), 4),
        "allocator_size_scale": round(float(allocator_size_scale), 4),
        "risk_cap_size_scale": round(float(risk_cap_size_scale), 4),
        "stale_exit_count": int(stale_exit_count),
        "stale_exit_events": list(stale_exit_events[:24]),
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

    msg_parts = [entry_msg]
    if stale_exit_refresh_msg and str(stale_exit_refresh_msg).strip() and (str(stale_exit_refresh_msg) not in msg_parts):
        msg_parts.append(str(stale_exit_refresh_msg))
    if loss_streak_auto_clear_msg and str(loss_streak_auto_clear_msg).strip() and (str(loss_streak_auto_clear_msg) not in msg_parts):
        msg_parts.append(str(loss_streak_auto_clear_msg))
    if shadow_only:
        msg_parts.append("rollout shadow_only: real entries suppressed")
    elif not enable_exec_v2:
        msg_parts.append(f"rollout {stage}: execution disabled")
    if str(policy.get("summary", "") or "").strip():
        msg_parts.append(str(policy.get("summary", "")))
    if trade_units_effective < abs(trade_units):
        msg_parts.append(f"size x{loss_size_scale:.2f}")
    if trade_units_entry < trade_units_effective:
        msg_parts.append(f"scan-size x{entry_size_scale:.2f}")
    if abs(float(policy_size_scale) - 1.0) >= 0.01:
        msg_parts.append(f"policy-size x{policy_size_scale:.2f}")
    if trade_quality_eval and abs(float(quality_size_scale) - 1.0) >= 0.01:
        msg_parts.append(f"quality-size x{quality_size_scale:.2f}")
    if allocator_evaluated and abs(float(allocator_size_scale) - 1.0) >= 0.01:
        msg_parts.append(f"allocator-size x{allocator_size_scale:.2f}")
    if allocator_evaluated:
        allocator_summary = str(opportunity_eval.get("summary", "") or "").strip()
        if allocator_summary:
            msg_parts.append(allocator_summary)
    if risk_cap_size_scale < 0.999:
        msg_parts.append(f"risk-cap-size x{risk_cap_size_scale:.2f}")
    if actions:
        msg_parts.append(actions[-1])
    if auto_enabled and (not shadow_only):
        try:
            if signal_side in {"long", "short"} and signal_inst and ("Entry placed" not in entry_msg) and ("Already in position" not in entry_msg):
                should_log = ((now_ts - int(last_divergence_ts)) >= 300) or (str(entry_msg) != str(last_divergence_msg))
                if should_log:
                    _append_jsonl(
                        audit_path,
                        {
                            "ts": now_ts,
                            "date": today,
                            "event": "shadow_live_divergence",
                            "instrument": signal_inst,
                            "side": signal_side,
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
    return {
        "state": "READY",
        "trader_state": _trader_state_label(settings, auto_enabled, shadow_only),
        "msg": " | ".join(x for x in msg_parts if x),
        "actions": actions[-12:],
        "open_positions": len(positions),
        "auto_enabled": auto_enabled,
        "broker_mode": str(_broker_mode_label(settings)).lower(),
        "rollout_stage": str(settings.get("market_rollout_stage", "legacy") or "legacy"),
        "execution_enabled": enable_exec_v2 and (not shadow_only),
        "trade_units": int(abs(trade_units)),
        "trade_units_effective": int(trade_units_effective),
        "trade_units_entry": int(trade_units_entry),
        "loss_size_scale": round(float(loss_size_scale), 4),
        "entry_size_scale": round(float(entry_size_scale), 4),
        "allocator_size_scale": round(float(allocator_size_scale), 4),
        "risk_cap_size_scale": round(float(risk_cap_size_scale), 4),
        "exposure_usd": round(total_exposure_usd, 4),
        "margin_used_usd": round(total_margin_used_usd, 4),
        "margin_available_usd": round(margin_available, 4),
        "margin_rate_est": round(margin_rate_est, 6),
        "position_values_usd": dict(position_values_usd),
        "crypto_exposure_usd": round(crypto_exposure_usd, 4) if auto_enabled else round(_crypto_holdings_usd(hub_dir), 4),
        "other_market_exposure_usd": round(stocks_exposure_usd, 4) if auto_enabled else round(_market_status_exposure_usd(hub_dir, "stocks"), 4),
        "cross_market_exposure_usd": round(cross_market_exposure_effective_usd, 4),
        "cross_market_cap_basis_usd": round(cross_market_cap_basis_effective_usd, 4),
        "account_value_usd": round(nav, 4),
        "entry_eval_total": int(len(entry_fail_reasons)),
        "entry_eval_failed": int(len(entry_fail_reasons) > 0),
        "entry_eval_top_reason": str(entry_eval_top_reason),
        "entry_eval_reason_counts": dict(entry_eval_reason_counts),
        "automation_policy": policy if isinstance(policy, dict) else {},
        "trade_quality": trade_quality_eval if isinstance(trade_quality_eval, dict) else {},
        "opportunity_allocator": opportunity_eval if isinstance(opportunity_eval, dict) else {},
        "entry_gate_flags": dict(entry_gate_flags),
        "stale_exit_count": int(stale_exit_count),
        "stale_exit_events": list(stale_exit_events[:24]),
        "updated_at": now_ts,
        "health": {"data_ok": thinker_data_ok, "broker_ok": True, "orders_ok": True, "drift_warning": drift_warning},
    }


def main() -> int:
    print("forex_trader.py is designed to be imported by the hub/runner first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
