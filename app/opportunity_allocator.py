from __future__ import annotations

from collections import Counter, deque
import json
import os
import time
from typing import Any, Dict, List

from app.openai_portfolio_decision import request_openai_portfolio_decision


_MARKETS = ("crypto", "stocks", "forex")


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl_tail(path: str, limit: int = 4000) -> List[Dict[str, Any]]:
    lim = max(1, int(limit or 1))
    out: List[Dict[str, Any]] = []
    try:
        buf: deque[str] = deque(maxlen=lim)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = str(line or "").strip()
                if txt:
                    buf.append(txt)
        for txt in list(buf):
            try:
                row = json.loads(txt)
                if isinstance(row, dict):
                    out.append(row)
            except Exception:
                continue
    except Exception:
        return []
    return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _market_key(market: Any) -> str:
    mk = str(market or "").strip().lower()
    if mk == "stock":
        return "stocks"
    return mk if mk in _MARKETS else "stocks"


def _required_score(settings: Dict[str, Any], market: str) -> float:
    mk = _market_key(market)
    if mk == "stocks":
        return max(0.01, _f(settings.get("stock_score_threshold", 0.2), 0.2))
    if mk == "forex":
        return max(0.01, _f(settings.get("forex_score_threshold", 0.2), 0.2))
    # Crypto scores are dynamic rank values and are not normalized to one fixed threshold.
    return max(0.01, _f(settings.get("crypto_allocator_signal_floor", 0.15), 0.15))


def _trading_mode(settings: Dict[str, Any]) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    if bool(cfg.get("alpaca_paper_mode", False)) or bool(cfg.get("oanda_practice_mode", False)):
        return "paper"
    return "live"


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_thinker_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto_dynamic_status.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _market_state_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_state.json")
    return os.path.join(hub_dir, "forex", "forex_trader_state.json")


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _audit_symbol(market: str, row: Dict[str, Any]) -> str:
    mk = _market_key(market)
    if mk == "forex":
        return str(row.get("instrument", row.get("pair", row.get("symbol", ""))) or "").strip().upper()
    return str(row.get("symbol", row.get("pair", row.get("instrument", ""))) or "").strip().upper()


def _audit_event(row: Dict[str, Any]) -> str:
    return str(row.get("event", "") or "").strip().lower()


def _audit_realized_pnl_usd(row: Dict[str, Any]) -> float:
    for key in ("realized_pnl_usd", "realized_pnl", "realized_pl", "pnl_usd", "realized"):
        val = _f(row.get(key, 0.0), 0.0)
        if abs(val) > 0.0 or key in row:
            return float(val)
    return 0.0


def _collect_market_audit_maps(market: str, audit_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    mk = _market_key(market)
    latest_entry_by_symbol: Dict[str, Dict[str, Any]] = {}
    latest_event_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in list(audit_rows or []):
        if not isinstance(row, dict):
            continue
        symbol = _audit_symbol(mk, row)
        if not symbol:
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        ev = _audit_event(row)
        last = latest_event_by_symbol.get(symbol, {})
        if (not last) or ts >= int(_f(last.get("ts", 0), 0.0)):
            latest_event_by_symbol[symbol] = {
                "ts": int(ts),
                "event": ev,
                "side": str(row.get("side", "") or "").strip().lower(),
                "price": float(_f(row.get("price", 0.0), 0.0)),
                "units": int(_f(row.get("units", row.get("qty", 0)), 0.0)),
                "qty": float(_f(row.get("qty", row.get("units", 0.0)), 0.0)),
                "notional": float(_f(row.get("notional", 0.0), 0.0)),
            }
        if ev == "entry":
            last_entry = latest_entry_by_symbol.get(symbol, {})
            if (not last_entry) or ts >= int(_f(last_entry.get("ts", 0), 0.0)):
                latest_entry_by_symbol[symbol] = {
                    "ts": int(ts),
                    "event": ev,
                    "side": str(row.get("side", "") or "").strip().lower(),
                    "price": float(_f(row.get("price", 0.0), 0.0)),
                    "units": int(_f(row.get("units", row.get("qty", 0)), 0.0)),
                    "qty": float(_f(row.get("qty", row.get("units", 0.0)), 0.0)),
                    "notional": float(_f(row.get("notional", row.get("configured_notional", 0.0)), 0.0)),
                }
    return {
        "latest_entry_by_symbol": latest_entry_by_symbol,
        "latest_event_by_symbol": latest_event_by_symbol,
    }


def _recent_performance_from_audit(
    market: str,
    audit_rows: List[Dict[str, Any]],
    *,
    now_ts: int,
) -> Dict[str, Any]:
    mk = _market_key(market)
    cutoff = int(max(0, int(now_ts) - (7 * 24 * 3600)))
    realized = 0.0
    exits = 0
    entries = 0
    wins = 0
    losses = 0
    stale_exit_count = 0
    rapid_turnover_count = 0
    drag_reasons: Counter[str] = Counter()

    for row in list(audit_rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts < cutoff:
            continue
        ev = _audit_event(row)
        msg = str(row.get("msg", row.get("tag", "")) or "").strip().lower()
        if ev == "entry":
            entries += 1
        if ev == "exit":
            exits += 1
            pnl = float(_audit_realized_pnl_usd(row))
            realized += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
            hold_s = int(max(0.0, _f(row.get("hold_s", 0), 0.0)))
            if hold_s > 0 and hold_s <= 6 * 3600:
                rapid_turnover_count += 1
        stale_markers = ("policy_stale_exit", "stale", "misalign")
        if any(tok in msg for tok in stale_markers):
            stale_exit_count += 1
            drag_reasons["stale_or_misaligned_exits"] += 1
        if "risk cap" in msg:
            drag_reasons["risk_cap_pressure"] += 1
        if "confidence" in msg:
            drag_reasons["confidence_gate_pressure"] += 1
        if "cooldown" in msg:
            drag_reasons["cooldown_pressure"] += 1

    if mk == "crypto" and abs(realized) <= 1e-9:
        # Crypto exits may sometimes carry realized value only in pnl_ledger rollups.
        pass

    return {
        "realized_pnl_7d_usd": round(float(realized), 6),
        "entries_7d": int(entries),
        "exits_7d": int(exits),
        "wins_7d": int(wins),
        "losses_7d": int(losses),
        "stale_exit_count_7d": int(stale_exit_count),
        "rapid_turnover_count_7d": int(rapid_turnover_count),
        "drag_reasons": [str(k) for k, _ in drag_reasons.most_common(5)],
    }


def _position_rows_for_market(
    *,
    market: str,
    now_ts: int,
    trader_status: Dict[str, Any],
    market_state: Dict[str, Any],
    market_candidates: List[Dict[str, Any]],
    audit_maps: Dict[str, Any],
    policy_summary_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mk = _market_key(market)
    trader = trader_status if isinstance(trader_status, dict) else {}
    state = market_state if isinstance(market_state, dict) else {}
    policy = trader.get("automation_policy", {}) if isinstance(trader.get("automation_policy", {}), dict) else {}
    gate = trader.get("entry_gate_flags", {}) if isinstance(trader.get("entry_gate_flags", {}), dict) else {}
    latest_entry = (
        audit_maps.get("latest_entry_by_symbol", {})
        if isinstance(audit_maps.get("latest_entry_by_symbol", {}), dict)
        else {}
    )
    latest_event = (
        audit_maps.get("latest_event_by_symbol", {})
        if isinstance(audit_maps.get("latest_event_by_symbol", {}), dict)
        else {}
    )
    candidates_by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in list(market_candidates or []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if symbol:
            candidates_by_symbol[symbol] = row

    cooldown_until = int(max(0.0, _f(state.get("cooldown_until", 0), 0.0)))
    cooldown_active = bool(cooldown_until > int(now_ts))
    allow_new_entries = bool(policy.get("allow_new_entries", True))
    skip_new_entries = bool(gate.get("skip_new_entries_this_cycle", False))
    base_add_allowed = bool(allow_new_entries and (not skip_new_entries) and (not cooldown_active))

    out: List[Dict[str, Any]] = []
    if mk == "crypto":
        positions = trader.get("positions", {}) if isinstance(trader.get("positions", {}), dict) else {}
        for symbol, row in positions.items():
            if not isinstance(row, dict):
                continue
            sym = str(symbol or "").strip().upper()
            if not sym:
                continue
            qty = float(max(0.0, _f(row.get("quantity", 0.0), 0.0)))
            if qty <= 0.0:
                continue
            avg_cost = float(max(0.0, _f(row.get("avg_cost_basis", 0.0), 0.0)))
            market_value = float(max(0.0, _f(row.get("value_usd", 0.0), 0.0)))
            pnl_pct = float(_f(row.get("gain_loss_pct_sell", row.get("gain_loss_pct_buy", 0.0)), 0.0))
            pnl_usd = (market_value * (pnl_pct / 100.0)) if market_value > 0.0 else 0.0
            le = latest_entry.get(sym, {}) if isinstance(latest_entry.get(sym, {}), dict) else {}
            entry_ts = int(max(0.0, _f(le.get("ts", 0), 0.0)))
            entry_age_minutes = int(max(0, (int(now_ts) - entry_ts) // 60)) if entry_ts > 0 else 0
            cand = candidates_by_symbol.get(sym, {}) if isinstance(candidates_by_symbol.get(sym, {}), dict) else {}
            aligned = bool(row.get("aligned_with_strategy", True))
            alignment_reasons = row.get("alignment_reasons", [])
            if not isinstance(alignment_reasons, list):
                alignment_reasons = []
            out.append(
                {
                    "market": mk,
                    "symbol": sym,
                    "side": "long",
                    "quantity": round(float(qty), 10),
                    "units": 0,
                    "notional_usd": round(float(max(0.0, avg_cost * qty)), 6),
                    "market_value_usd": round(float(market_value), 6),
                    "unrealized_pnl_usd": round(float(pnl_usd), 6),
                    "unrealized_pnl_pct": round(float(pnl_pct), 6),
                    "realized_pnl_usd_recent": 0.0,
                    "entry_ts": int(entry_ts),
                    "entry_age_minutes": int(entry_age_minutes),
                    "last_add_ts": int(entry_ts),
                    "last_add_age_minutes": int(entry_age_minutes),
                    "alignment_with_strategy": bool(aligned),
                    "alignment_reasons": [str(x or "")[:120] for x in alignment_reasons[:4] if str(x or "").strip()],
                    "alignment_streak": int(max(0.0, _f(row.get("alignment_streak", 0), 0.0))),
                    "trade_quality_confidence_score": round(float(_f(policy_summary_row.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
                    "runtime_trust_score": round(float(_f(policy_summary_row.get("runtime_trust_score", 0.0), 0.0)), 4),
                    "policy_mode": str(policy.get("mode", "") or ""),
                    "policy_profile": str(policy.get("profile", "") or ""),
                    "cooldown_active": bool(cooldown_active),
                    "cooldown_remaining_s": int(max(0, cooldown_until - int(now_ts))) if cooldown_active else 0,
                    "add_allowed_local": bool(base_add_allowed),
                    "exit_allowed_local": True,
                    "current_signal_score": round(float(_f(cand.get("signal_score", 0.0), 0.0)), 6),
                    "current_local_rank": int(max(0.0, _f(cand.get("local_rank", 0), 0.0))),
                    "spread_bps": round(float(_f(cand.get("spread_bps", 0.0), 0.0)), 6),
                    "max_slippage_bps": round(float(_f(cand.get("max_slippage_bps", 0.0), 0.0)), 6),
                    "local_reason_summary": str(cand.get("entry_gate_reason", trader.get("entry_eval_top_reason", "")) or "")[:180],
                }
            )
        return out

    open_meta = state.get("open_meta", {}) if isinstance(state.get("open_meta", {}), dict) else {}
    stale_streaks = (
        state.get("stale_alignment_streaks", {})
        if isinstance(state.get("stale_alignment_streaks", {}), dict)
        else {}
    )
    position_values = (
        trader.get("position_values_usd", {})
        if isinstance(trader.get("position_values_usd", {}), dict)
        else {}
    )
    min_hold_minutes = int(max(0.0, _f(gate.get("same_day_exit_min_hold_minutes", 0), 0.0)))
    pdt_restricted = bool(gate.get("pdt_restricted", False))
    compliance_entry_blocked = bool(gate.get("compliance_entry_blocked", False))

    for symbol, meta in open_meta.items():
        if not isinstance(meta, dict):
            continue
        sym = str(symbol or "").strip().upper()
        if not sym:
            continue
        entry_ts = int(max(0.0, _f(meta.get("entry_ts", 0), 0.0)))
        entry_age_minutes = int(max(0, (int(now_ts) - entry_ts) // 60)) if entry_ts > 0 else 0
        last_pnl_pct = float(_f(meta.get("last_pnl_pct", 0.0), 0.0))
        market_value = float(max(0.0, _f(position_values.get(sym, 0.0), 0.0)))
        le = latest_entry.get(sym, {}) if isinstance(latest_entry.get(sym, {}), dict) else {}
        lv = latest_event.get(sym, {}) if isinstance(latest_event.get(sym, {}), dict) else {}
        side = str(le.get("side", lv.get("side", "long")) or "long").strip().lower()
        if side not in {"long", "short"}:
            if int(_f(le.get("units", lv.get("units", 0)), 0.0)) < 0:
                side = "short"
            else:
                side = "long"
        units = int(_f(le.get("units", lv.get("units", 0)), 0.0))
        qty = float(_f(le.get("qty", lv.get("qty", 0.0)), 0.0))
        est_notional = float(max(0.0, _f(le.get("notional", 0.0), 0.0)))
        if est_notional <= 0.0 and market_value > 0.0:
            est_notional = market_value
        pnl_usd = (market_value * (last_pnl_pct / 100.0)) if market_value > 0.0 else 0.0
        cand = candidates_by_symbol.get(sym, {}) if isinstance(candidates_by_symbol.get(sym, {}), dict) else {}
        stale_streak = int(max(0.0, _f(stale_streaks.get(sym, 0), 0.0)))
        aligned = bool(stale_streak <= 0)
        alignment_reasons: List[str] = []
        if stale_streak > 0:
            alignment_reasons.append("alignment_streak_active")
        add_allowed = bool(base_add_allowed and (not compliance_entry_blocked))
        if mk == "stocks":
            add_allowed = False
        exit_allowed = True
        if mk == "stocks" and pdt_restricted and min_hold_minutes > 0 and entry_age_minutes < min_hold_minutes:
            exit_allowed = False
            alignment_reasons.append("pdt_hold_gate_active")
        out.append(
            {
                "market": mk,
                "symbol": sym,
                "side": side,
                "quantity": round(float(max(0.0, qty)), 10),
                "units": int(units),
                "notional_usd": round(float(max(0.0, est_notional)), 6),
                "market_value_usd": round(float(max(0.0, market_value)), 6),
                "unrealized_pnl_usd": round(float(pnl_usd), 6),
                "unrealized_pnl_pct": round(float(last_pnl_pct), 6),
                "realized_pnl_usd_recent": 0.0,
                "entry_ts": int(entry_ts),
                "entry_age_minutes": int(entry_age_minutes),
                "last_add_ts": int(max(entry_ts, int(_f(le.get("ts", 0), 0.0)))),
                "last_add_age_minutes": int(max(0, (int(now_ts) - int(max(entry_ts, int(_f(le.get("ts", 0), 0.0))))) // 60))
                if (entry_ts > 0 or int(_f(le.get("ts", 0), 0.0)) > 0)
                else 0,
                "alignment_with_strategy": bool(aligned),
                "alignment_reasons": [str(x or "")[:120] for x in alignment_reasons[:4] if str(x or "").strip()],
                "alignment_streak": int(stale_streak),
                "trade_quality_confidence_score": round(float(_f(policy_summary_row.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
                "runtime_trust_score": round(float(_f(policy_summary_row.get("runtime_trust_score", 0.0), 0.0)), 4),
                "policy_mode": str(policy.get("mode", "") or ""),
                "policy_profile": str(policy.get("profile", "") or ""),
                "cooldown_active": bool(cooldown_active),
                "cooldown_remaining_s": int(max(0, cooldown_until - int(now_ts))) if cooldown_active else 0,
                "add_allowed_local": bool(add_allowed),
                "exit_allowed_local": bool(exit_allowed),
                "current_signal_score": round(float(_f(cand.get("signal_score", 0.0), 0.0)), 6),
                "current_local_rank": int(max(0.0, _f(cand.get("local_rank", 0), 0.0))),
                "spread_bps": round(float(_f(cand.get("spread_bps", 0.0), 0.0)), 6),
                "max_slippage_bps": round(float(_f(cand.get("max_slippage_bps", 0.0), 0.0)), 6),
                "local_reason_summary": str(cand.get("entry_gate_reason", trader.get("entry_eval_top_reason", "")) or "")[:180],
            }
        )
    return out


def _market_symbol_key(market: str) -> str:
    mk = _market_key(market)
    if mk == "forex":
        return "pair"
    return "symbol"


def _side_actionable(market: str, side: str) -> bool:
    mk = _market_key(market)
    sd = str(side or "").strip().lower()
    if mk == "stocks":
        return sd == "long"
    if mk == "forex":
        return sd in {"long", "short"}
    return sd == "long"


def _soft_confidence(signal_score: float, required_score: float) -> float:
    req = max(0.01, float(required_score or 0.01))
    ratio = abs(float(signal_score or 0.0)) / req
    return _clamp(42.0 + (ratio * 18.0), 20.0, 72.0)


def _top_candidate_from_market_files(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    market: str,
    now_ts: int,
) -> Dict[str, Any]:
    mk = _market_key(market)
    thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
    trader = _safe_read_json(_market_status_path(hub_dir, mk))
    policy = trader.get("automation_policy", {}) if isinstance(trader.get("automation_policy", {}), dict) else {}
    quality = trader.get("trade_quality", {}) if isinstance(trader.get("trade_quality", {}), dict) else {}
    gate_flags = trader.get("entry_gate_flags", {}) if isinstance(trader.get("entry_gate_flags", {}), dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}

    candidate_id = ""
    side = "watch"
    signal_score = 0.0
    spread_bps = 0.0
    updated_at = int(_f(thinker.get("updated_at", thinker.get("ts", 0) or 0), 0.0))
    required = _required_score(settings, mk)
    if mk == "crypto":
        ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
        top = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
        candidate_id = str(top.get("symbol", "") or "").strip().upper()
        signal_score = _f(top.get("score", 0.0), 0.0)
        side = "long" if signal_score > 0.0 else "watch"
        required = max(required, _f(thinker.get("adaptive_threshold", 0.0), 0.0))
    elif mk == "stocks":
        top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        candidate_id = str(top.get("symbol", "") or "").strip().upper()
        side = str(top.get("side", "watch") or "watch").strip().lower()
        signal_score = _f(top.get("score", 0.0), 0.0)
        spread_bps = _f(top.get("spread_bps", 0.0), 0.0)
    else:
        top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        candidate_id = str(top.get("pair", "") or "").strip().upper()
        side = str(top.get("side", "watch") or "watch").strip().lower()
        signal_score = _f(top.get("score", 0.0), 0.0)
        spread_bps = _f(top.get("spread_bps", 0.0), 0.0)

    if updated_at <= 0:
        updated_at = int(_f(trader.get("updated_at", trader.get("timestamp", 0) or 0), 0.0))
    candidate_age_s = max(0, int(now_ts - updated_at)) if updated_at > 0 else 10_000
    confidence = _f(quality.get("confidence_score", 0.0), 0.0)
    if confidence <= 0.0:
        confidence = _soft_confidence(signal_score, required)

    account = trader.get("account", {}) if isinstance(trader.get("account", {}), dict) else {}
    market_exposure_usd = float(max(0.0, _f(trader.get("exposure_usd", 0.0), 0.0)))
    account_value_usd = float(max(0.0, _f(trader.get("account_value_usd", 0.0), 0.0)))
    buying_power_usd = float(
        max(
            0.0,
            _f(trader.get("buying_power_usd", 0.0), 0.0),
            _f(account.get("buying_power", 0.0), 0.0),
        )
    )
    if mk == "crypto":
        market_exposure_usd = float(
            max(
                market_exposure_usd,
                _f(account.get("holdings_sell_value", 0.0), 0.0),
                _f(account.get("holdings_value", 0.0), 0.0),
                _f(account.get("holdings_usd", 0.0), 0.0),
            )
        )
        account_value_usd = float(
            max(
                account_value_usd,
                _f(account.get("total_account_value", 0.0), 0.0),
                _f(account.get("account_value_usd", 0.0), 0.0),
                _f(account.get("equity", 0.0), 0.0),
                _f(account.get("nav", 0.0), 0.0),
            )
        )

    return {
        "market": mk,
        "candidate_id": candidate_id,
        "side": side,
        "signal_score": float(signal_score),
        "required_score": float(required),
        "spread_bps": float(max(0.0, spread_bps)),
        "max_slippage_bps": float(max(0.0, _f(settings.get(f"{mk[:-1] if mk.endswith('s') else mk}_max_slippage_bps", 0.0), 0.0))),
        "candidate_age_s": int(candidate_age_s),
        "trade_quality_confidence_score": float(confidence),
        "trade_quality_decision": str(quality.get("decision", "") or ""),
        "runtime_trust_score": float(_f(trust.get("score", 0.0), _f(gate_flags.get("runtime_trust_score", 0.0), 0.0))),
        "allow_new_entries": bool(policy.get("allow_new_entries", True)),
        "market_exposure_usd": float(market_exposure_usd),
        "account_value_usd": float(account_value_usd),
        "buying_power_usd": float(buying_power_usd),
        "reason_hint": str(trader.get("entry_eval_top_reason", "") or "").strip(),
    }


def _score_candidate(snapshot: Dict[str, Any], *, stale_soft_s: int, stale_hard_s: int) -> Dict[str, Any]:
    mk = _market_key(snapshot.get("market", ""))
    side = str(snapshot.get("side", "watch") or "watch").strip().lower()
    signal = float(snapshot.get("signal_score", 0.0) or 0.0)
    required = max(0.01, float(snapshot.get("required_score", 0.01) or 0.01))
    confidence = _clamp(float(snapshot.get("trade_quality_confidence_score", 0.0) or 0.0), 0.0, 100.0)
    trust = _clamp(float(snapshot.get("runtime_trust_score", 0.0) or 0.0), 0.0, 100.0)
    spread_bps = max(0.0, float(snapshot.get("spread_bps", 0.0) or 0.0))
    max_slippage_bps = max(0.0, float(snapshot.get("max_slippage_bps", 0.0) or 0.0))
    age_s = max(0, int(snapshot.get("candidate_age_s", 0) or 0))
    allow_entries = bool(snapshot.get("allow_new_entries", True))
    quality_decision = str(snapshot.get("trade_quality_decision", "") or "").strip().lower()
    actionable = _side_actionable(mk, side)

    signal_ratio = abs(signal) / required
    signal_points = _clamp(signal_ratio / 1.6, 0.0, 1.0) * 34.0
    confidence_points = (confidence / 100.0) * 32.0
    trust_points = (trust / 100.0) * 16.0

    freshness_points = 10.0
    if age_s > stale_soft_s:
        freshness_points = max(0.0, 10.0 - ((float(age_s - stale_soft_s) / max(1.0, float(stale_hard_s - stale_soft_s))) * 10.0))

    spread_penalty = 0.0
    if max_slippage_bps > 0.0:
        spread_penalty = _clamp((spread_bps / max(1e-6, max_slippage_bps)) * 9.0, 0.0, 12.0)

    projected_total_exposure_pct = _clamp(float(snapshot.get("projected_total_exposure_pct", 0.0) or 0.0), 0.0, 300.0)
    projected_market_share_pct = _clamp(float(snapshot.get("projected_market_share_pct", 0.0) or 0.0), 0.0, 100.0)
    exposure_penalty = _clamp((projected_total_exposure_pct * 0.06) + max(0.0, projected_market_share_pct - 50.0) * 0.12, 0.0, 14.0)

    loss_streak = max(0, int(snapshot.get("loss_streak", 0) or 0))
    max_loss_streak = max(1, int(snapshot.get("max_loss_streak", 1) or 1))
    loss_ratio = _clamp(float(loss_streak) / float(max_loss_streak), 0.0, 1.5)
    discipline_penalty = 8.0 * min(1.0, loss_ratio)

    context_state = str(snapshot.get("context_state", "unclear") or "unclear").strip().lower()
    if context_state not in {"supportive", "neutral", "adverse", "unclear"}:
        context_state = "unclear"
    context_confidence = _clamp(float(snapshot.get("context_confidence", 0.0) or 0.0), 0.0, 1.0)
    context_reason = str(snapshot.get("context_reason", "") or "").strip()
    context_adjustment = 0.0
    if context_state == "supportive":
        context_adjustment = 4.0 * context_confidence
    elif context_state == "adverse":
        context_adjustment = -8.0 * context_confidence
    elif context_state == "unclear":
        context_adjustment = -1.5 * context_confidence

    gate_penalty = 0.0
    reasons: List[str] = []
    if not actionable:
        gate_penalty += 22.0
        reasons.append("top candidate side is not actionable")
    if not allow_entries:
        gate_penalty += 20.0
        reasons.append("market policy currently blocks new entries")
    if quality_decision == "block":
        gate_penalty += 16.0
        reasons.append("trade-quality gate is blocking entry")
    if age_s > stale_hard_s:
        gate_penalty += 16.0
        reasons.append(f"candidate is stale ({age_s}s old)")
    if context_state == "adverse":
        if context_reason:
            reasons.append(f"market context adverse: {context_reason[:96]}")
        else:
            reasons.append("market context is adverse")

    score = signal_points + confidence_points + trust_points + freshness_points + context_adjustment - spread_penalty - exposure_penalty - discipline_penalty - gate_penalty
    score = _clamp(score, 0.0, 100.0)
    eligible = bool(actionable and allow_entries and quality_decision != "block" and age_s <= stale_hard_s and score >= 8.0)

    return {
        "score": float(round(score, 4)),
        "eligible": bool(eligible),
        "reasons": reasons[:4],
        "context_state": context_state,
        "context_adjustment": float(round(context_adjustment, 4)),
    }


def _candidate_rows_for_market(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    market: str,
    now_ts: int,
    snapshot: Dict[str, Any],
    limit: int,
    context_by_market: Dict[str, Dict[str, Any]] | None = None,
    context_by_symbol: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    mk = _market_key(market)
    lim = max(1, int(limit or 1))
    ctx_market_map = context_by_market if isinstance(context_by_market, dict) else {}
    ctx_symbol_map = context_by_symbol if isinstance(context_by_symbol, dict) else {}
    thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
    required = float(max(0.01, _required_score(settings, mk)))
    updated_at = int(
        _f(
            thinker.get("updated_at", thinker.get("ts", snapshot.get("candidate_updated_ts", 0) or 0) or 0),
            0.0,
        )
    )
    if updated_at <= 0:
        updated_at = int(now_ts)

    ranked_rows: List[Dict[str, Any]] = []
    if mk == "crypto":
        ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
        ranked_rows = [row for row in ranked if isinstance(row, dict)]
    else:
        leaders = thinker.get("leaders", []) if isinstance(thinker.get("leaders", []), list) else []
        ranked_rows = [row for row in leaders if isinstance(row, dict)]
        if not ranked_rows:
            top_pick = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
            if top_pick:
                ranked_rows = [top_pick]

    out: List[Dict[str, Any]] = []
    symbol_key = _market_symbol_key(mk)
    for idx, row in enumerate(ranked_rows[:lim]):
        ident = str(row.get(symbol_key, "") or "").strip().upper()
        if not ident:
            continue
        ctx_symbol = ctx_symbol_map.get(f"{mk}:{ident}", {}) if isinstance(ctx_symbol_map.get(f"{mk}:{ident}", {}), dict) else {}
        ctx_market = ctx_market_map.get(mk, {}) if isinstance(ctx_market_map.get(mk, {}), dict) else {}
        ctx = ctx_symbol if ctx_symbol else ctx_market
        raw_score = float(_f(row.get("score", 0.0), 0.0))
        row_required = max(required, float(_f(row.get("required_score", row.get("adaptive_threshold", 0.0)), 0.0)))
        side = str(row.get("side", "watch") or "watch").strip().lower()
        if not side:
            side = "long" if raw_score > 0.0 else "watch"
        stale_hours = float(max(0.0, _f(row.get("stale_hours", 0.0), 0.0)))
        age_s = max(0, int(stale_hours * 3600.0))
        if age_s <= 0 and updated_at > 0:
            age_s = max(0, int(now_ts - updated_at))
        out.append(
            {
                "market": mk,
                "symbol": ident,
                "side": side,
                "local_rank": int(idx + 1),
                "signal_score": round(raw_score, 6),
                "required_score": round(float(row_required), 6),
                "signal_ratio": round((abs(raw_score) / max(0.01, float(row_required))), 6),
                "trade_quality_confidence_score": round(float(_f(snapshot.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
                "trade_quality_decision": str(snapshot.get("trade_quality_decision", "") or "").strip().lower(),
                "runtime_trust_score": round(float(_f(snapshot.get("runtime_trust_score", 0.0), 0.0)), 4),
                "spread_bps": round(float(max(0.0, _f(row.get("spread_bps", snapshot.get("spread_bps", 0.0)), 0.0))), 6),
                "max_slippage_bps": round(float(max(0.0, _f(snapshot.get("max_slippage_bps", 0.0), 0.0))), 6),
                "candidate_age_s": int(age_s),
                "freshness_s": int(age_s),
                "eligible_for_entry": bool(row.get("eligible_for_entry", True)),
                "entry_gate_reason": str(row.get("entry_gate_reason", "") or "").strip()[:180],
                "data_quality_ok": bool(row.get("data_quality_ok", True)),
                "mtf_confirmed": row.get("mtf_confirmed", None),
                "calib_prob": round(float(max(0.0, _f(row.get("calib_prob", 0.0), 0.0))), 6),
                "samples": int(max(0, _f(row.get("samples", 0), 0.0))),
                "quality_score": round(float(max(0.0, _f(row.get("quality_score", 0.0), 0.0))), 6),
                "reason": str(row.get("reason_logic", row.get("reason", "")) or "").strip()[:200],
                "context_state": str(ctx.get("context_state", "unclear") or "unclear").strip().lower(),
                "context_confidence": round(float(_clamp(_f(ctx.get("confidence", 0.0), 0.0), 0.0, 1.0)), 6),
                "context_reason": str(ctx.get("reason", "") or "").strip()[:160],
            }
        )

    if out:
        return out

    fallback_id = str(snapshot.get("candidate_id", "") or "").strip().upper()
    if not fallback_id:
        return []
    return [
        {
            "market": mk,
            "symbol": fallback_id,
            "side": str(snapshot.get("side", "watch") or "watch").strip().lower(),
            "local_rank": 1,
            "signal_score": round(float(_f(snapshot.get("signal_score", 0.0), 0.0)), 6),
            "required_score": round(float(max(0.01, _f(snapshot.get("required_score", 0.01), 0.01))), 6),
            "signal_ratio": round(
                abs(float(_f(snapshot.get("signal_score", 0.0), 0.0)))
                / max(0.01, float(max(0.01, _f(snapshot.get("required_score", 0.01), 0.01)))),
                6,
            ),
            "trade_quality_confidence_score": round(float(_f(snapshot.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
            "trade_quality_decision": str(snapshot.get("trade_quality_decision", "") or "").strip().lower(),
            "runtime_trust_score": round(float(_f(snapshot.get("runtime_trust_score", 0.0), 0.0)), 4),
            "spread_bps": round(float(max(0.0, _f(snapshot.get("spread_bps", 0.0), 0.0))), 6),
            "max_slippage_bps": round(float(max(0.0, _f(snapshot.get("max_slippage_bps", 0.0), 0.0))), 6),
            "candidate_age_s": int(max(0, _f(snapshot.get("candidate_age_s", 0), 0.0))),
            "freshness_s": int(max(0, _f(snapshot.get("candidate_age_s", 0), 0.0))),
            "eligible_for_entry": bool(_side_actionable(mk, snapshot.get("side", "watch"))),
            "entry_gate_reason": str(snapshot.get("reason_hint", "") or "").strip()[:180],
            "data_quality_ok": True,
            "mtf_confirmed": None,
            "calib_prob": 0.0,
            "samples": 0,
            "quality_score": 0.0,
            "reason": str(snapshot.get("reason_hint", "") or "").strip()[:200],
            "context_state": str(snapshot.get("context_state", "unclear") or "unclear").strip().lower(),
            "context_confidence": round(float(_clamp(_f(snapshot.get("context_confidence", 0.0), 0.0), 0.0, 1.0)), 6),
            "context_reason": str(snapshot.get("context_reason", "") or "").strip()[:160],
        }
    ]


def _build_openai_decision_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    now_ts: int,
    market_key: str,
    current_snapshot: Dict[str, Any],
    snapshots: Dict[str, Dict[str, Any]],
    scores: Dict[str, float],
    eligibility: Dict[str, bool],
    market_reasons: Dict[str, List[str]],
    local_decision: str,
    local_summary: str,
    capital_constrained: bool,
    projected_total_exposure_pct: float,
    projected_market_share_pct: float,
    projected_trade_value_usd: float,
    global_cap_pct: float,
    market_context_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    max_candidates = max(1, int(_f(settings.get("openai_decision_max_candidates_per_market", 3), 3)))
    market_candidates: Dict[str, List[Dict[str, Any]]] = {}
    open_positions: Dict[str, int] = {}
    open_position_rows: List[Dict[str, Any]] = []
    recent_performance: Dict[str, Dict[str, Any]] = {}
    policy_summary: Dict[str, Dict[str, Any]] = {}
    market_status_rows: Dict[str, Dict[str, Any]] = {}
    market_state_rows: Dict[str, Dict[str, Any]] = {}
    market_audit_rows: Dict[str, List[Dict[str, Any]]] = {}
    market_audit_maps: Dict[str, Dict[str, Any]] = {}

    context_payload = market_context_payload if isinstance(market_context_payload, dict) else {}
    context_by_market = context_payload.get("by_market", {}) if isinstance(context_payload.get("by_market", {}), dict) else {}
    context_by_symbol = context_payload.get("by_symbol", {}) if isinstance(context_payload.get("by_symbol", {}), dict) else {}

    for mk in _MARKETS:
        snap = snapshots.get(mk, {}) if isinstance(snapshots.get(mk, {}), dict) else {}
        market_candidates[mk] = _candidate_rows_for_market(
            hub_dir=hub_dir,
            settings=settings,
            market=mk,
            now_ts=now_ts,
            snapshot=snap,
            limit=max_candidates,
            context_by_market=context_by_market,
            context_by_symbol=context_by_symbol,
        )
        trader = _safe_read_json(_market_status_path(hub_dir, mk))
        state_row = _safe_read_json(_market_state_path(hub_dir, mk))
        audit_rows = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=3000)
        audit_maps = _collect_market_audit_maps(mk, audit_rows)
        market_status_rows[mk] = dict(trader)
        market_state_rows[mk] = dict(state_row)
        market_audit_rows[mk] = list(audit_rows)
        market_audit_maps[mk] = dict(audit_maps)
        gate = trader.get("entry_gate_flags", {}) if isinstance(trader.get("entry_gate_flags", {}), dict) else {}
        positions = trader.get("positions", {})
        position_count = int(max(0, _f(trader.get("open_positions", 0), 0.0)))
        if position_count <= 0:
            if isinstance(positions, dict):
                position_count = len([key for key, row in positions.items() if isinstance(row, dict)])
            elif isinstance(positions, list):
                position_count = len([row for row in positions if isinstance(row, dict)])
        open_positions[mk] = int(max(0, position_count))
        recent_performance[mk] = {
            "entry_eval_failed": int(max(0, _f(trader.get("entry_eval_failed", 0), 0.0))),
            "entry_eval_total": int(max(0, _f(trader.get("entry_eval_total", 0), 0.0))),
            "entry_eval_top_reason": str(trader.get("entry_eval_top_reason", "") or "").strip()[:180],
            "loss_streak": int(max(0, _f(gate.get("loss_streak", trader.get("loss_streak", 0)), 0.0))),
            "stale_exit_count": int(max(0, _f(trader.get("stale_exit_count", 0), 0.0))),
        }
        policy_summary[mk] = {
            "allow_new_entries": bool(snap.get("allow_new_entries", True)),
            "runtime_trust_score": round(float(_f(snap.get("runtime_trust_score", 0.0), 0.0)), 4),
            "trade_quality_confidence_score": round(float(_f(snap.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
            "trade_quality_decision": str(snap.get("trade_quality_decision", "") or "").strip().lower(),
        }

    for mk in _MARKETS:
        position_rows = _position_rows_for_market(
            market=mk,
            now_ts=now_ts,
            trader_status=market_status_rows.get(mk, {}) if isinstance(market_status_rows.get(mk, {}), dict) else {},
            market_state=market_state_rows.get(mk, {}) if isinstance(market_state_rows.get(mk, {}), dict) else {},
            market_candidates=market_candidates.get(mk, []) if isinstance(market_candidates.get(mk, []), list) else [],
            audit_maps=market_audit_maps.get(mk, {}) if isinstance(market_audit_maps.get(mk, {}), dict) else {},
            policy_summary_row=policy_summary.get(mk, {}) if isinstance(policy_summary.get(mk, {}), dict) else {},
        )
        open_position_rows.extend(position_rows)
        if int(open_positions.get(mk, 0) or 0) <= 0:
            open_positions[mk] = int(len(position_rows))

    for mk in _MARKETS:
        perf = _recent_performance_from_audit(
            mk,
            market_audit_rows.get(mk, []) if isinstance(market_audit_rows.get(mk, []), list) else [],
            now_ts=now_ts,
        )
        base = recent_performance.get(mk, {}) if isinstance(recent_performance.get(mk, {}), dict) else {}
        merged_perf = dict(base)
        merged_perf.update(perf)
        recent_performance[mk] = merged_perf

    market_exposure_map: Dict[str, float] = {}
    account_value_map: Dict[str, float] = {}
    buying_power_map: Dict[str, float] = {}
    for mk in _MARKETS:
        snap = snapshots.get(mk, {}) if isinstance(snapshots.get(mk, {}), dict) else {}
        market_exposure_map[mk] = round(float(max(0.0, _f(snap.get("market_exposure_usd", 0.0), 0.0))), 4)
        account_value_map[mk] = round(float(max(0.0, _f(snap.get("account_value_usd", 0.0), 0.0))), 4)
        buying_power_map[mk] = round(float(max(0.0, _f(snap.get("buying_power_usd", 0.0), 0.0))), 4)

    total_exposure_usd = round(float(sum(market_exposure_map.values())), 4)
    total_buying_power_usd = round(float(max(buying_power_map.values()) if buying_power_map else 0.0), 4)
    portfolio_value_usd = round(float(max(account_value_map.values()) if account_value_map else 0.0), 4)
    if portfolio_value_usd <= 0.0:
        portfolio_value_usd = round(float(max(0.0, _f(current_snapshot.get("account_value_usd", 0.0), 0.0))), 4)

    top_candidates = []
    for mk in _MARKETS:
        rows = market_candidates.get(mk, [])
        if rows:
            top_candidates.append(dict(rows[0]))

    profile_key = str(settings.get("settings_profile", "balanced") or "balanced").strip().lower()
    control_mode = str(settings.get("settings_control_mode", "self_managed") or "self_managed").strip().lower()
    per_market_exposure_caps_pct = {
        "crypto": round(float(max(0.0, _f(settings.get("max_total_exposure_pct", 0.0), 0.0))), 4),
        "stocks": round(float(max(0.0, _f(settings.get("stock_max_total_exposure_pct", 0.0), 0.0))), 4),
        "forex": round(float(max(0.0, _f(settings.get("forex_max_total_exposure_pct", 0.0), 0.0))), 4),
    }
    open_position_rows_sorted = sorted(
        [row for row in open_position_rows if isinstance(row, dict)],
        key=lambda item: (
            _market_key(item.get("market", "")),
            str(item.get("symbol", "") or "").upper(),
        ),
    )

    return {
        "timestamp": int(now_ts),
        "mode": _trading_mode(settings),
        "portfolio": {
            "portfolio_value_usd": float(portfolio_value_usd),
            "settings_profile": profile_key,
            "settings_control_mode": control_mode,
            "total_buying_power_usd": float(total_buying_power_usd),
            "total_exposure_usd": float(total_exposure_usd),
            "total_exposure_pct": round(
                float(
                    ((float(total_exposure_usd) / max(1e-6, float(portfolio_value_usd))) * 100.0)
                    if float(portfolio_value_usd) > 0.0
                    else 0.0
                ),
                4,
            ),
            "projected_trade_value_usd": round(float(max(0.0, projected_trade_value_usd)), 4),
            "projected_total_exposure_pct": round(float(max(0.0, projected_total_exposure_pct)), 4),
            "projected_market_share_pct": round(float(max(0.0, projected_market_share_pct)), 4),
            "capital_constrained": bool(capital_constrained),
            "global_exposure_cap_pct": round(float(max(0.0, global_cap_pct)), 4),
            "per_market_exposure_caps_pct": per_market_exposure_caps_pct,
            "market_exposure_usd": market_exposure_map,
            "account_value_usd_by_market": account_value_map,
            "buying_power_usd_by_market": buying_power_map,
            "open_positions_by_market": open_positions,
        },
        "local_allocator": {
            "decision": str(local_decision),
            "summary": str(local_summary)[:220],
            "best_market": str(max(scores, key=scores.get) if scores else market_key),
            "scores": {mk: round(float(_f(scores.get(mk, 0.0), 0.0)), 4) for mk in _MARKETS},
            "eligibility": {mk: bool(eligibility.get(mk, False)) for mk in _MARKETS},
            "market_reasons": {mk: list(market_reasons.get(mk, []))[:4] for mk in _MARKETS},
        },
        "policy_trust_summary": policy_summary,
        "recent_performance_summary": recent_performance,
        "open_positions": open_position_rows_sorted[:96],
        "current_candidate": {
            "market": market_key,
            "symbol": str(current_snapshot.get("candidate_id", "") or "").strip().upper(),
            "side": str(current_snapshot.get("side", "watch") or "watch").strip().lower(),
            "signal_score": round(float(_f(current_snapshot.get("signal_score", 0.0), 0.0)), 6),
            "required_score": round(float(max(0.01, _f(current_snapshot.get("required_score", 0.01), 0.01))), 6),
            "trade_quality_confidence_score": round(float(_f(current_snapshot.get("trade_quality_confidence_score", 0.0), 0.0)), 4),
            "trade_quality_decision": str(current_snapshot.get("trade_quality_decision", "") or "").strip().lower(),
            "runtime_trust_score": round(float(_f(current_snapshot.get("runtime_trust_score", 0.0), 0.0)), 4),
            "candidate_age_s": int(max(0, _f(current_snapshot.get("candidate_age_s", 0), 0.0))),
            "spread_bps": round(float(max(0.0, _f(current_snapshot.get("spread_bps", 0.0), 0.0))), 6),
            "max_slippage_bps": round(float(max(0.0, _f(current_snapshot.get("max_slippage_bps", 0.0), 0.0))), 6),
            "reason_hint": str(current_snapshot.get("reason_hint", "") or "").strip()[:180],
        },
        "top_candidates": top_candidates[:9],
        "candidates_by_market": {mk: list(market_candidates.get(mk, []))[:max_candidates] for mk in _MARKETS},
        "market_context": {
            "summary": str(context_payload.get("summary", "") or "").strip()[:220],
            "status": str(context_payload.get("status", "") or "").strip().lower(),
            "by_market": dict(context_by_market),
            "symbol_context_scores": [
                dict(row)
                for row in list(context_payload.get("symbol_context_scores", []) or [])[:36]
                if isinstance(row, dict)
            ],
        },
    }


def _matching_ai_candidate(ai_decision: Dict[str, Any], *, market_key: str, candidate_id: str) -> Dict[str, Any]:
    mk = _market_key(market_key)
    cid = str(candidate_id or "").strip().upper()
    ranked = ai_decision.get("ranked_candidates", []) if isinstance(ai_decision.get("ranked_candidates", []), list) else []
    for row in ranked:
        if not isinstance(row, dict):
            continue
        row_market = _market_key(row.get("market", ""))
        row_symbol = str(row.get("symbol", "") or "").strip().upper()
        if row_market == mk and row_symbol and row_symbol == cid:
            return dict(row)
    top = ai_decision.get("top_recommendation", {}) if isinstance(ai_decision.get("top_recommendation", {}), dict) else {}
    top_market = _market_key(top.get("market", ""))
    top_symbol = str(top.get("symbol", "") or "").strip().upper()
    if top and top_market == mk and (not cid or top_symbol == cid):
        return dict(top)
    return {}


def _matching_position_action(ai_decision: Dict[str, Any], *, market_key: str, candidate_id: str) -> Dict[str, Any]:
    mk = _market_key(market_key)
    cid = str(candidate_id or "").strip().upper()
    actions = ai_decision.get("position_actions", []) if isinstance(ai_decision.get("position_actions", []), list) else []
    if not cid:
        return {}
    for row in actions:
        if not isinstance(row, dict):
            continue
        row_market = _market_key(row.get("market", ""))
        row_symbol = str(row.get("symbol", "") or "").strip().upper()
        if row_market == mk and row_symbol == cid:
            return dict(row)
    return {}


def _apply_openai_advisory(
    *,
    local_decision: str,
    local_summary: str,
    reasons: List[str],
    market_key: str,
    candidate_id: str,
    ai_result: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "decision": str(local_decision),
        "summary": str(local_summary),
        "reasons": list(reasons),
        "size_multiplier": 1.0,
        "applied": False,
        "ai_status": str(ai_result.get("status", "") or ""),
        "ai_summary": str(ai_result.get("summary", "") or "").strip(),
    }
    if not bool(ai_result.get("active", False)):
        return out
    ai_decision = ai_result.get("decision", {}) if isinstance(ai_result.get("decision", {}), dict) else {}
    if not ai_decision:
        return out

    matched = _matching_ai_candidate(ai_decision, market_key=market_key, candidate_id=candidate_id)
    matched_position_action = _matching_position_action(ai_decision, market_key=market_key, candidate_id=candidate_id)
    matched_action = str(matched.get("action", "") or "").strip().lower()
    global_action = str(ai_decision.get("decision", "") or "").strip().lower()
    action = matched_action if matched_action else global_action
    if action not in {"allow", "deprioritize", "block", "no_trade"}:
        action = "allow"

    ai_reason = str(matched.get("reason", "") or ai_decision.get("explanation", "") or "").strip()
    ai_size = float(_clamp(_f(matched.get("size_multiplier", 1.0), 1.0), 0.25, 1.0))
    position_action = str(matched_position_action.get("action", "") or "").strip().lower()
    position_reason = str(matched_position_action.get("reason", "") or "").strip()
    portfolio_action = str(ai_decision.get("portfolio_action", "") or "").strip().lower()
    decision = str(local_decision)
    if str(local_decision) == "allow":
        if action == "block":
            decision = "block"
        elif action in {"deprioritize", "no_trade"}:
            decision = "deprioritize"
        if position_action == "block_add":
            decision = "block"
            if position_reason:
                ai_reason = position_reason
        elif position_action in {"reduce", "exit", "monitor"}:
            decision = "deprioritize"
            if position_reason:
                ai_reason = position_reason
        elif (portfolio_action == "manage_existing") and (action not in {"block"}):
            decision = "deprioritize"
            if not ai_reason:
                ai_reason = "Managing existing positions is preferred over opening a new trade right now."

    if decision != str(local_decision):
        out["applied"] = True
        if ai_reason:
            out["reasons"] = [ai_reason] + [row for row in out["reasons"] if str(row or "").strip()]
        out["summary"] = f"AI portfolio decision: {ai_reason or 'cross-market opportunity quality is weak; wait for stronger setup'}"[:220]
    elif ai_size < 0.999:
        out["applied"] = True
        out["summary"] = f"{str(local_summary)[:140]} | AI reduced size for risk-adjusted portfolio allocation."[:220]
    out["decision"] = decision
    out["size_multiplier"] = float(ai_size)
    if isinstance(matched_position_action, dict) and matched_position_action:
        out["matched_position_action"] = dict(matched_position_action)
    return out


def _load_capital_planner_advisory(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    enabled = bool(cfg.get("openai_capital_planner_enabled", False))
    payload = {
        "enabled": bool(enabled),
        "active": False,
        "status": "disabled" if not enabled else "not_requested",
        "summary": "",
        "error": "",
        "model": str(cfg.get("openai_capital_planner_model", cfg.get("openai_model", "gpt-5.4-mini")) or "gpt-5.4-mini"),
        "latency_ms": 0,
        "portfolio_plan": {},
        "market_actions": [],
        "market_actions_by_market": {},
        "global_risks": [],
    }
    if not enabled:
        return payload
    status_path = os.path.join(hub_dir, "openai", "capital_planner_status.json")
    report_path = os.path.join(hub_dir, "openai", "capital_planner.json")
    row = _safe_read_json(status_path)
    if not row:
        row = _safe_read_json(report_path)
    if not isinstance(row, dict):
        return payload

    plan = row.get("portfolio_plan", {}) if isinstance(row.get("portfolio_plan", {}), dict) else {}
    actions = row.get("market_actions", []) if isinstance(row.get("market_actions", []), list) else []
    by_market = row.get("market_actions_by_market", {}) if isinstance(row.get("market_actions_by_market", {}), dict) else {}
    if not by_market:
        by_market = {
            _market_key(item.get("market", "")): dict(item)
            for item in actions
            if isinstance(item, dict) and str(item.get("market", "") or "").strip()
        }
    for mk in _MARKETS:
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "action": "neutral",
                "confidence": 0.0,
                "suggested_capital_share_pct": 0.0,
                "reason": "No explicit planner recommendation for this market.",
            }
    payload.update(
        {
            "active": bool(row.get("active", False)),
            "status": str(row.get("status", payload["status"]) or payload["status"]).strip().lower(),
            "summary": str(row.get("summary", "") or "").strip()[:220],
            "error": str(row.get("error", "") or "").strip()[:180],
            "model": str(row.get("model", payload["model"]) or payload["model"]),
            "latency_ms": int(max(0, _f(row.get("latency_ms", 0), 0.0))),
            "portfolio_plan": dict(plan),
            "market_actions": [dict(item) for item in actions[:12] if isinstance(item, dict)],
            "market_actions_by_market": dict(by_market),
            "global_risks": [str(x or "")[:120] for x in list(row.get("global_risks", []) or [])[:12] if str(x or "").strip()],
        }
    )
    return payload


def _load_market_context_advisory(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    enabled = bool(cfg.get("openai_market_context_enabled", False))
    payload = {
        "enabled": bool(enabled),
        "active": False,
        "status": "disabled" if not enabled else "not_requested",
        "summary": "",
        "error": "",
        "model": str(cfg.get("openai_market_context_model", cfg.get("openai_model", "gpt-5.4-mini")) or "gpt-5.4-mini"),
        "latency_ms": 0,
        "market_context_scores": [],
        "symbol_context_scores": [],
        "by_market": {},
        "by_symbol": {},
    }
    if not enabled:
        return payload
    status_path = os.path.join(hub_dir, "openai", "market_context_status.json")
    report_path = os.path.join(hub_dir, "openai", "market_context.json")
    row = _safe_read_json(status_path)
    if not row:
        row = _safe_read_json(report_path)
    if not isinstance(row, dict):
        return payload

    market_scores = row.get("market_context_scores", []) if isinstance(row.get("market_context_scores", []), list) else []
    symbol_scores = row.get("symbol_context_scores", []) if isinstance(row.get("symbol_context_scores", []), list) else []
    by_market = row.get("by_market", {}) if isinstance(row.get("by_market", {}), dict) else {}
    by_symbol = row.get("by_symbol", {}) if isinstance(row.get("by_symbol", {}), dict) else {}
    if not by_market:
        by_market = {
            _market_key(item.get("market", "")): dict(item)
            for item in market_scores
            if isinstance(item, dict) and str(item.get("market", "") or "").strip()
        }
    for mk in _MARKETS:
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "context_state": "unclear",
                "confidence": 0.0,
                "reason": "No explicit context score returned.",
            }
    if not by_symbol:
        by_symbol = {}
        for item in symbol_scores:
            if not isinstance(item, dict):
                continue
            mk = _market_key(item.get("market", ""))
            symbol = str(item.get("symbol", "") or "").strip().upper()
            if not symbol:
                continue
            key = f"{mk}:{symbol}"
            if key in by_symbol:
                continue
            by_symbol[key] = dict(item)

    payload.update(
        {
            "active": bool(row.get("active", False)),
            "status": str(row.get("status", payload["status"]) or payload["status"]).strip().lower(),
            "summary": str(row.get("summary", "") or "").strip()[:220],
            "error": str(row.get("error", "") or "").strip()[:180],
            "model": str(row.get("model", payload["model"]) or payload["model"]),
            "latency_ms": int(max(0, _f(row.get("latency_ms", 0), 0.0))),
            "market_context_scores": [dict(item) for item in market_scores[:12] if isinstance(item, dict)],
            "symbol_context_scores": [dict(item) for item in symbol_scores[:80] if isinstance(item, dict)],
            "by_market": dict(by_market),
            "by_symbol": dict(by_symbol),
        }
    )
    return payload


def _load_root_cause_advisory(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    enabled = bool(cfg.get("openai_root_cause_enabled", False))
    payload = {
        "enabled": bool(enabled),
        "active": False,
        "status": "disabled" if not enabled else "not_requested",
        "summary": "",
        "error": "",
        "model": str(cfg.get("openai_root_cause_model", cfg.get("openai_model", "gpt-5.4-mini")) or "gpt-5.4-mini"),
        "latency_ms": 0,
        "overall_assessment": "",
        "global_risks": [],
        "throttle_recommendations": [],
        "throttle_by_market": {},
    }
    if not enabled:
        return payload
    status_path = os.path.join(hub_dir, "openai", "root_cause_analysis_status.json")
    report_path = os.path.join(hub_dir, "openai", "root_cause_analysis.json")
    row = _safe_read_json(status_path)
    if not row:
        row = _safe_read_json(report_path)
    if not isinstance(row, dict):
        return payload

    throttle_rows = row.get("throttle_recommendations", []) if isinstance(row.get("throttle_recommendations", []), list) else []
    throttle_by_market = {
        _market_key(item.get("market", "")): dict(item)
        for item in throttle_rows
        if isinstance(item, dict) and str(item.get("market", "") or "").strip()
    }
    for mk in _MARKETS:
        if mk not in throttle_by_market:
            throttle_by_market[mk] = {
                "market": mk,
                "recommendation": "none",
                "confidence": 0.0,
                "reason": "No explicit throttle recommendation for this market.",
            }

    payload.update(
        {
            "active": bool(row.get("active", False)),
            "status": str(row.get("status", payload["status"]) or payload["status"]).strip().lower(),
            "summary": str(row.get("summary", "") or "").strip()[:220],
            "error": str(row.get("error", "") or "").strip()[:180],
            "model": str(row.get("model", payload["model"]) or payload["model"]),
            "latency_ms": int(max(0, _f(row.get("latency_ms", 0), 0.0))),
            "overall_assessment": str(row.get("overall_assessment", "") or "").strip().lower()[:24],
            "global_risks": [str(x or "")[:120] for x in list(row.get("global_risks", []) or [])[:12] if str(x or "").strip()],
            "throttle_recommendations": [dict(item) for item in throttle_rows[:12] if isinstance(item, dict)],
            "throttle_by_market": dict(throttle_by_market),
        }
    )
    return payload


def _apply_capital_planner_advisory(
    *,
    local_decision: str,
    local_summary: str,
    reasons: List[str],
    market_key: str,
    capital_constrained: bool,
    buying_power_usd: float,
    projected_trade_value_usd: float,
    planner_result: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "decision": str(local_decision),
        "summary": str(local_summary),
        "reasons": list(reasons),
        "applied": False,
        "market_action": {},
    }
    if str(local_decision) != "allow":
        return out
    if not bool(planner_result.get("enabled", False)):
        return out
    if not bool(planner_result.get("active", False)):
        return out
    plan = planner_result.get("portfolio_plan", {}) if isinstance(planner_result.get("portfolio_plan", {}), dict) else {}
    by_market = planner_result.get("market_actions_by_market", {}) if isinstance(planner_result.get("market_actions_by_market", {}), dict) else {}
    market_action = by_market.get(market_key, {}) if isinstance(by_market.get(market_key, {}), dict) else {}
    action = str(market_action.get("action", "neutral") or "neutral").strip().lower()
    confidence = float(_clamp(_f(market_action.get("confidence", 0.0), 0.0), 0.0, 1.0))
    reason = str(market_action.get("reason", plan.get("reason", "")) or "").strip()
    reserve_capital_pct = float(max(0.0, _f(plan.get("reserve_capital_pct", 0.0), 0.0)))
    plan_mode = str(plan.get("mode", "") or "").strip().lower()
    decision = str(local_decision)

    if action == "block":
        decision = "block"
    elif action == "deprioritize":
        decision = "deprioritize"
    elif action == "neutral":
        reserve_hit = reserve_capital_pct >= 20.0
        need_buffer = bool(projected_trade_value_usd > 0.0 and buying_power_usd > 0.0 and buying_power_usd <= (projected_trade_value_usd * 1.05))
        if (plan_mode in {"wait", "prefer_existing"}) and (capital_constrained or reserve_hit or need_buffer) and confidence >= 0.60:
            decision = "deprioritize"
            if not reason:
                reason = "AI capital planner recommends holding capital for stronger opportunities."
    elif action == "prefer":
        if reason:
            out["summary"] = f"{str(local_summary)[:150]} | AI capital planner prefers {market_key.title()}: {reason[:56]}".strip()[:220]

    if decision != str(local_decision):
        out["applied"] = True
        out["decision"] = decision
        if reason:
            out["reasons"] = [reason] + [row for row in out["reasons"] if str(row or "").strip()]
        if decision == "block":
            out["summary"] = f"AI capital planner blocked {market_key.title()} allocation: {reason or 'capital should be reserved for stronger opportunities.'}"[:220]
        else:
            out["summary"] = f"AI capital planner deprioritized {market_key.title()}: {reason or 'capital should be reserved for stronger opportunities.'}"[:220]

    out["market_action"] = {
        "market": str(market_key),
        "action": str(action),
        "confidence": round(float(confidence), 6),
        "reason": reason[:180],
        "plan_mode": plan_mode[:24],
        "reserve_capital_pct": round(float(reserve_capital_pct), 6),
    }
    return out


def _apply_root_cause_advisory(
    *,
    local_decision: str,
    local_summary: str,
    reasons: List[str],
    market_key: str,
    root_cause_result: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "decision": str(local_decision),
        "summary": str(local_summary),
        "reasons": list(reasons),
        "applied": False,
        "market_action": {},
    }
    if str(local_decision) != "allow":
        return out
    if not bool(root_cause_result.get("enabled", False)):
        return out
    if not bool(root_cause_result.get("active", False)):
        return out
    by_market = root_cause_result.get("throttle_by_market", {}) if isinstance(root_cause_result.get("throttle_by_market", {}), dict) else {}
    market_action = by_market.get(market_key, {}) if isinstance(by_market.get(market_key, {}), dict) else {}
    recommendation = str(market_action.get("recommendation", "none") or "none").strip().lower()
    confidence = float(_clamp(_f(market_action.get("confidence", 0.0), 0.0), 0.0, 1.0))
    reason = str(market_action.get("reason", root_cause_result.get("summary", "")) or "").strip()
    decision = str(local_decision)

    if recommendation == "pause" and confidence >= 0.55:
        decision = "block"
    elif recommendation == "throttle" and confidence >= 0.60:
        decision = "deprioritize"

    if decision != str(local_decision):
        out["applied"] = True
        out["decision"] = decision
        if reason:
            out["reasons"] = [reason] + [row for row in out["reasons"] if str(row or "").strip()]
        if decision == "block":
            out["summary"] = f"AI root-cause paused {market_key.title()} entries: {reason or 'runtime diagnostics indicate elevated risk.'}"[:220]
        else:
            out["summary"] = f"AI root-cause throttled {market_key.title()} entries: {reason or 'runtime diagnostics indicate elevated risk.'}"[:220]

    out["market_action"] = {
        "market": str(market_key),
        "recommendation": str(recommendation),
        "confidence": round(float(confidence), 6),
        "reason": reason[:180],
        "overall_assessment": str(root_cause_result.get("overall_assessment", "") or "").strip().lower()[:24],
    }
    return out


def evaluate_cross_market_allocation(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    market: str,
    candidate_id: str,
    candidate_side: str,
    signal_score: float,
    required_score: float,
    trade_quality: Dict[str, Any] | None,
    automation_policy: Dict[str, Any] | None,
    projected_trade_value_usd: float,
    market_exposure_usd: float,
    account_value_usd: float,
    buying_power_usd: float,
    spread_bps: float = 0.0,
    max_slippage_bps: float = 0.0,
    candidate_age_s: int = 0,
    loss_streak: int = 0,
    max_loss_streak: int = 0,
    now_ts: int | None = None,
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    market_key = _market_key(market)
    independent_mode = bool(cfg.get("market_independent_execution_enabled", False))
    now = int(now_ts if isinstance(now_ts, int) and now_ts > 0 else int(time.time()))
    policy = automation_policy if isinstance(automation_policy, dict) else {}
    quality = trade_quality if isinstance(trade_quality, dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}

    stale_soft_s = max(20, int(_f(cfg.get("portfolio_allocator_stale_soft_s", 120), 120)))
    stale_hard_s = max(stale_soft_s + 30, int(_f(cfg.get("portfolio_allocator_stale_hard_s", 420), 420)))

    current_snapshot: Dict[str, Any] = {
        "market": market_key,
        "candidate_id": str(candidate_id or "").strip().upper(),
        "side": str(candidate_side or "watch").strip().lower(),
        "signal_score": float(signal_score or 0.0),
        "required_score": float(max(0.01, float(required_score or 0.01))),
        "spread_bps": float(max(0.0, spread_bps)),
        "max_slippage_bps": float(max(0.0, max_slippage_bps)),
        "candidate_age_s": max(0, int(candidate_age_s or 0)),
        "trade_quality_confidence_score": float(max(0.0, _f(quality.get("confidence_score", 0.0), 0.0))),
        "trade_quality_decision": str(quality.get("decision", "allow") or "allow"),
        "runtime_trust_score": float(max(0.0, _f(trust.get("score", 0.0), 0.0))),
        "allow_new_entries": bool(policy.get("allow_new_entries", True)),
        "market_exposure_usd": float(max(0.0, market_exposure_usd)),
        "account_value_usd": float(max(0.0, account_value_usd)),
        "buying_power_usd": float(max(0.0, buying_power_usd)),
        "projected_trade_value_usd": float(max(0.0, projected_trade_value_usd)),
        "loss_streak": int(max(0, loss_streak)),
        "max_loss_streak": int(max(1, max_loss_streak)),
        "reason_hint": "",
    }
    if current_snapshot["trade_quality_confidence_score"] <= 0.0:
        current_snapshot["trade_quality_confidence_score"] = _soft_confidence(
            current_snapshot["signal_score"],
            current_snapshot["required_score"],
        )

    snapshots: Dict[str, Dict[str, Any]] = {market_key: current_snapshot}
    for mk in _MARKETS:
        if mk == market_key:
            continue
        snapshots[mk] = _top_candidate_from_market_files(hub_dir=hub_dir, settings=cfg, market=mk, now_ts=now)
        snapshots[mk]["projected_trade_value_usd"] = 0.0
        snapshots[mk]["loss_streak"] = int(max(0, _f((snapshots[mk].get("entry_gate_flags", {}) or {}).get("loss_streak", 0), 0.0)))
        snapshots[mk]["max_loss_streak"] = 3

    market_context_result = _load_market_context_advisory(hub_dir=hub_dir, settings=cfg)
    context_by_market = (
        market_context_result.get("by_market", {})
        if isinstance(market_context_result.get("by_market", {}), dict)
        else {}
    )
    context_by_symbol = (
        market_context_result.get("by_symbol", {})
        if isinstance(market_context_result.get("by_symbol", {}), dict)
        else {}
    )
    for mk in _MARKETS:
        snap = snapshots.get(mk, {})
        if not isinstance(snap, dict):
            continue
        symbol = str(snap.get("candidate_id", "") or "").strip().upper()
        ctx_symbol = (
            context_by_symbol.get(f"{mk}:{symbol}", {})
            if isinstance(context_by_symbol.get(f"{mk}:{symbol}", {}), dict)
            else {}
        )
        ctx_market = context_by_market.get(mk, {}) if isinstance(context_by_market.get(mk, {}), dict) else {}
        ctx = ctx_symbol if ctx_symbol else ctx_market
        state = str(ctx.get("context_state", "unclear") or "unclear").strip().lower()
        if state not in {"supportive", "neutral", "adverse", "unclear"}:
            state = "unclear"
        snap["context_state"] = state
        snap["context_confidence"] = float(_clamp(_f(ctx.get("confidence", 0.0), 0.0), 0.0, 1.0))
        snap["context_reason"] = str(ctx.get("reason", "") or "").strip()[:180]

    projected_total_exposure_usd = 0.0
    for mk in _MARKETS:
        row = snapshots.get(mk, {})
        if not isinstance(row, dict):
            continue
        projected_total_exposure_usd += max(0.0, _f(row.get("market_exposure_usd", 0.0), 0.0))
    projected_total_exposure_usd += max(0.0, float(projected_trade_value_usd or 0.0))
    portfolio_account_value_usd = 0.0
    for mk in _MARKETS:
        row = snapshots.get(mk, {})
        if not isinstance(row, dict):
            continue
        portfolio_account_value_usd += max(0.0, _f(row.get("account_value_usd", 0.0), 0.0))
    if portfolio_account_value_usd <= 0.0:
        portfolio_account_value_usd = max(0.0, float(account_value_usd or 0.0))

    projected_market_exposure_usd = max(0.0, float(market_exposure_usd or 0.0)) + max(0.0, float(projected_trade_value_usd or 0.0))
    projected_total_exposure_pct = (
        ((projected_total_exposure_usd / max(1e-6, float(portfolio_account_value_usd or 0.0))) * 100.0)
        if float(portfolio_account_value_usd or 0.0) > 0.0
        else 0.0
    )
    projected_market_share_pct = (
        ((projected_market_exposure_usd / max(1e-6, projected_total_exposure_usd)) * 100.0)
        if projected_total_exposure_usd > 0.0
        else 0.0
    )
    current_snapshot["projected_total_exposure_pct"] = float(projected_total_exposure_pct)
    current_snapshot["projected_market_share_pct"] = float(projected_market_share_pct)

    scores: Dict[str, float] = {}
    eligibility: Dict[str, bool] = {}
    market_reasons: Dict[str, List[str]] = {}
    context_adjustments: Dict[str, float] = {}
    context_states: Dict[str, str] = {}
    for mk in _MARKETS:
        snap = snapshots.get(mk, {})
        if not isinstance(snap, dict):
            continue
        scored = _score_candidate(snap, stale_soft_s=stale_soft_s, stale_hard_s=stale_hard_s)
        scores[mk] = float(scored.get("score", 0.0) or 0.0)
        eligibility[mk] = bool(scored.get("eligible", False))
        market_reasons[mk] = [str(x or "").strip() for x in list(scored.get("reasons", []) or []) if str(x or "").strip()]
        context_adjustments[mk] = float(_f(scored.get("context_adjustment", 0.0), 0.0))
        context_states[mk] = str(scored.get("context_state", snapshots.get(mk, {}).get("context_state", "unclear")) or "unclear").strip().lower()

    ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    best_market = ranked[0][0] if ranked else market_key
    best_score = float(ranked[0][1]) if ranked else 0.0
    current_score = float(scores.get(market_key, 0.0))
    score_gap = float(best_score - current_score)

    global_cap_pct = max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))
    if independent_mode:
        global_cap_pct = 0.0
    concentration_warn_pct = _clamp(_f(cfg.get("portfolio_allocator_concentration_warn_pct", 68.0), 68.0), 40.0, 95.0)
    concentration_block_pct = _clamp(_f(cfg.get("portfolio_allocator_concentration_block_pct", 82.0), 82.0), concentration_warn_pct + 4.0, 99.0)
    priority_gap_min = _clamp(_f(cfg.get("portfolio_allocator_priority_gap_min", 8.0), 8.0), 2.0, 30.0)
    buying_power_buffer_mult = _clamp(_f(cfg.get("portfolio_allocator_buying_power_buffer_mult", 1.1), 1.1), 1.0, 2.0)

    buying_power_tight = bool(float(buying_power_usd or 0.0) > 0.0 and float(projected_trade_value_usd or 0.0) > 0.0 and float(buying_power_usd or 0.0) <= (float(projected_trade_value_usd or 0.0) * buying_power_buffer_mult))
    near_global_cap = bool(global_cap_pct > 0.0 and projected_total_exposure_pct >= (global_cap_pct * 0.9))
    concentration_warn = bool(projected_market_share_pct >= concentration_warn_pct)
    concentration_block = bool(projected_market_share_pct >= concentration_block_pct)
    capital_constrained = bool(buying_power_tight or near_global_cap or concentration_warn)
    if independent_mode:
        capital_constrained = bool(buying_power_tight)

    decision = "allow"
    reasons: List[str] = []

    if not bool(eligibility.get(market_key, False)):
        decision = "block"
        reasons.extend(market_reasons.get(market_key, []))
        if not reasons:
            reasons.append(f"{market_key.title()} candidate is not currently eligible")
    else:
        if independent_mode:
            if buying_power_tight:
                reasons.append("Remaining buying power is tight for this entry")
        else:
            best_eligible = bool(eligibility.get(best_market, False))
            if best_market != market_key and best_eligible and score_gap >= priority_gap_min and capital_constrained:
                decision = "deprioritize"
                reasons.append(
                    f"{best_market.title()} has higher opportunity quality ({best_score:.1f} vs {current_score:.1f}) while capital is constrained"
                )
            elif best_market != market_key and best_eligible and concentration_block and score_gap >= 3.0:
                decision = "block"
                reasons.append(
                    f"Blocked due to concentration risk ({projected_market_share_pct:.1f}% in {market_key}) while {best_market} is comparable"
                )
            elif concentration_warn and best_market == market_key:
                reasons.append(f"Portfolio concentration in {market_key} is elevated ({projected_market_share_pct:.1f}%)")

    if (not independent_mode) and near_global_cap:
        reasons.append(f"Projected global exposure is near cap ({projected_total_exposure_pct:.1f}% of {global_cap_pct:.1f}%)")
    if buying_power_tight and (not independent_mode):
        reasons.append("Remaining buying power is tight for this entry")

    if decision == "allow":
        if independent_mode:
            summary = f"{market_key.title()} allowed in independent-market mode"
        else:
            summary = (
                f"{market_key.title()} selected as best current opportunity"
                if best_market == market_key
                else f"{market_key.title()} allowed despite {best_market.title()} leading score"
            )
    elif decision == "deprioritize":
        summary = (
            f"{market_key.title()} candidate deprioritized because {best_market.title()} has higher opportunity quality and capital is constrained"
        )
    else:
        summary = reasons[0] if reasons else f"{market_key.title()} candidate blocked by portfolio allocator"

    local_decision = str(decision)
    local_summary = str(summary)[:220]
    final_reasons = [str(x or "").strip() for x in reasons[:6] if str(x or "").strip()]
    final_size_multiplier = 1.0
    decision_source = "local"
    capital_planner_result = _load_capital_planner_advisory(hub_dir=hub_dir, settings=cfg)
    planner_market_action: Dict[str, Any] = {}
    planner_applied = False
    if independent_mode:
        planner_market_action = {
            "market": str(market_key),
            "action": "neutral",
            "confidence": 1.0,
            "reason": "Independent-market mode bypassed cross-market capital planner influence.",
        }
    else:
        planner_merge = _apply_capital_planner_advisory(
            local_decision=local_decision,
            local_summary=local_summary,
            reasons=final_reasons,
            market_key=market_key,
            capital_constrained=capital_constrained,
            buying_power_usd=float(max(0.0, buying_power_usd)),
            projected_trade_value_usd=float(max(0.0, projected_trade_value_usd)),
            planner_result=capital_planner_result,
        )
        local_decision = str(planner_merge.get("decision", local_decision) or local_decision)
        local_summary = str(planner_merge.get("summary", local_summary) or local_summary)[:220]
        final_reasons = [
            str(x or "").strip()
            for x in list(planner_merge.get("reasons", final_reasons) or final_reasons)[:6]
            if str(x or "").strip()
        ]
        planner_market_action = planner_merge.get("market_action", {}) if isinstance(planner_merge.get("market_action", {}), dict) else {}
        planner_applied = bool(planner_merge.get("applied", False))
    decision = str(local_decision)
    summary = str(local_summary)
    if planner_applied:
        decision_source = "local+capital_planner"

    root_cause_result = _load_root_cause_advisory(hub_dir=hub_dir, settings=cfg)
    root_cause_merge = _apply_root_cause_advisory(
        local_decision=local_decision,
        local_summary=local_summary,
        reasons=final_reasons,
        market_key=market_key,
        root_cause_result=root_cause_result,
    )
    local_decision = str(root_cause_merge.get("decision", local_decision) or local_decision)
    local_summary = str(root_cause_merge.get("summary", local_summary) or local_summary)[:220]
    final_reasons = [
        str(x or "").strip()
        for x in list(root_cause_merge.get("reasons", final_reasons) or final_reasons)[:6]
        if str(x or "").strip()
    ]
    root_cause_market_action = root_cause_merge.get("market_action", {}) if isinstance(root_cause_merge.get("market_action", {}), dict) else {}
    root_cause_applied = bool(root_cause_merge.get("applied", False))
    decision = str(local_decision)
    summary = str(local_summary)
    if root_cause_applied:
        if planner_applied:
            decision_source = "local+capital_planner+root_cause"
        else:
            decision_source = "local+root_cause"

    matched_position_action: Dict[str, Any] = {}

    openai_enabled = bool(cfg.get("openai_decision_enabled", False))
    require_local_pass_first = bool(cfg.get("openai_decision_require_local_pass_first", True))
    openai_result: Dict[str, Any] = {
        "enabled": bool(openai_enabled),
        "active": False,
        "status": "not_requested",
        "model": str(cfg.get("openai_model", "gpt-5.4-mini") or "gpt-5.4-mini"),
        "timeout_s": float(_f(cfg.get("openai_timeout_s", 6.0), 6.0)),
        "summary": "",
        "error": "",
        "decision": {},
        "latency_ms": 0,
    }
    if openai_enabled:
        if independent_mode:
            openai_result["status"] = "skipped_independent_mode"
            openai_result["summary"] = "AI portfolio decision skipped: independent-market mode keeps market entries decoupled."
        if require_local_pass_first and local_decision != "allow":
            openai_result["status"] = "skipped_local_gate"
            openai_result["summary"] = "AI portfolio decision skipped because local allocator did not allow this candidate."
        elif not independent_mode:
            decision_packet = _build_openai_decision_packet(
                hub_dir=hub_dir,
                settings=cfg,
                now_ts=now,
                market_key=market_key,
                current_snapshot=current_snapshot,
                snapshots=snapshots,
                scores=scores,
                eligibility=eligibility,
                market_reasons=market_reasons,
                local_decision=local_decision,
                local_summary=local_summary,
                capital_constrained=capital_constrained,
                projected_total_exposure_pct=projected_total_exposure_pct,
                projected_market_share_pct=projected_market_share_pct,
                projected_trade_value_usd=projected_trade_value_usd,
                global_cap_pct=global_cap_pct,
                market_context_payload=market_context_result,
            )
            openai_result = request_openai_portfolio_decision(
                settings=cfg,
                base_dir=os.path.dirname(hub_dir) if str(hub_dir or "").strip() else os.getcwd(),
                decision_packet=decision_packet,
                broker_mode=str(_trading_mode(cfg)),
            )
            merged = _apply_openai_advisory(
                local_decision=local_decision,
                local_summary=local_summary,
                reasons=final_reasons,
                market_key=market_key,
                candidate_id=str(candidate_id or ""),
                ai_result=openai_result,
            )
            decision = str(merged.get("decision", local_decision) or local_decision)
            summary = str(merged.get("summary", local_summary) or local_summary)[:220]
            final_reasons = [
                str(x or "").strip()
                for x in list(merged.get("reasons", final_reasons) or final_reasons)[:6]
                if str(x or "").strip()
            ]
            final_size_multiplier = float(_clamp(_f(merged.get("size_multiplier", 1.0), 1.0), 0.25, 1.0))
            matched_position_action = (
                dict(merged.get("matched_position_action", {}) or {})
                if isinstance(merged.get("matched_position_action", {}), dict)
                else {}
            )
            if bool(merged.get("applied", False)):
                if planner_applied and root_cause_applied:
                    decision_source = "local+capital_planner+root_cause+openai"
                elif planner_applied:
                    decision_source = "local+capital_planner+openai"
                elif root_cause_applied:
                    decision_source = "local+root_cause+openai"
                else:
                    decision_source = "local+openai"

    ai_decision = openai_result.get("decision", {}) if isinstance(openai_result.get("decision", {}), dict) else {}
    openai_summary = str(openai_result.get("summary", "") or "").strip()
    if openai_enabled and (not openai_summary):
        if bool(openai_result.get("active", False)) and isinstance(ai_decision, dict):
            openai_summary = str(ai_decision.get("explanation", "") or "").strip()
        elif str(openai_result.get("status", "") or "").strip() not in {"", "not_requested"}:
            openai_summary = "AI portfolio decision unavailable; local allocator is active."
    openai_decision_payload = {
        "enabled": bool(openai_result.get("enabled", openai_enabled)),
        "active": bool(openai_result.get("active", False)),
        "status": str(openai_result.get("status", "") or "").strip(),
        "summary": openai_summary[:220],
        "error": str(openai_result.get("error", "") or "").strip()[:180],
        "model": str(openai_result.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(openai_result.get("latency_ms", 0), 0.0))),
        "decision": str(ai_decision.get("decision", "") or "").strip().lower(),
        "best_market": str(ai_decision.get("best_market", "") or "").strip().lower(),
        "portfolio_action": str(ai_decision.get("portfolio_action", "") or "").strip().lower(),
        "portfolio_confidence": round(float(_clamp(_f(ai_decision.get("portfolio_confidence", 0.0), 0.0), 0.0, 1.0)), 6),
        "capital_constrained": bool(ai_decision.get("capital_constrained", False)),
        "top_recommendation": dict(ai_decision.get("top_recommendation", {}) or {})
        if isinstance(ai_decision.get("top_recommendation", {}), dict)
        else {},
        "ranked_candidates": [
            dict(row)
            for row in list(ai_decision.get("ranked_candidates", []) or [])[:24]
            if isinstance(row, dict)
        ],
        "position_actions": [
            dict(row)
            for row in list(ai_decision.get("position_actions", []) or [])[:48]
            if isinstance(row, dict)
        ],
        "position_actions_summary": [
            {
                "market": str(row.get("market", "") or "").strip().lower(),
                "symbol": str(row.get("symbol", "") or "").strip().upper(),
                "action": str(row.get("action", "") or "").strip().lower(),
                "confidence": round(float(_clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0)), 6),
                "reason": str(row.get("reason", "") or "").strip()[:160],
            }
            for row in list(ai_decision.get("position_actions", []) or [])[:18]
            if isinstance(row, dict)
        ],
        "matched_position_action": dict(matched_position_action) if isinstance(matched_position_action, dict) else {},
        "applied": bool(str(decision_source).endswith("+openai")),
        "applied_size_multiplier": round(float(final_size_multiplier), 6),
    }

    planner_plan = capital_planner_result.get("portfolio_plan", {}) if isinstance(capital_planner_result.get("portfolio_plan", {}), dict) else {}
    planner_actions = capital_planner_result.get("market_actions", []) if isinstance(capital_planner_result.get("market_actions", []), list) else []
    planner_summary = str(capital_planner_result.get("summary", "") or "").strip()
    if bool(capital_planner_result.get("enabled", False)) and (not planner_summary):
        if bool(capital_planner_result.get("active", False)):
            planner_summary = str(planner_plan.get("reason", "") or "").strip()
        elif str(capital_planner_result.get("status", "") or "").strip() not in {"", "not_requested"}:
            planner_summary = "AI capital planner unavailable; local allocator remains active."
    openai_capital_planner_payload = {
        "enabled": bool(capital_planner_result.get("enabled", False)),
        "active": bool(capital_planner_result.get("active", False)),
        "status": str(capital_planner_result.get("status", "") or "").strip(),
        "summary": planner_summary[:220],
        "error": str(capital_planner_result.get("error", "") or "").strip()[:180],
        "model": str(capital_planner_result.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(capital_planner_result.get("latency_ms", 0), 0.0))),
        "portfolio_plan": dict(planner_plan),
        "market_actions": [dict(item) for item in planner_actions[:12] if isinstance(item, dict)],
        "market_action": dict(planner_market_action),
        "global_risks": [str(x or "")[:120] for x in list(capital_planner_result.get("global_risks", []) or [])[:12] if str(x or "").strip()],
        "applied": bool(planner_applied),
    }

    market_context_summary = str(market_context_result.get("summary", "") or "").strip()
    if bool(market_context_result.get("enabled", False)) and (not market_context_summary):
        if bool(market_context_result.get("active", False)):
            state = str(context_states.get(market_key, snapshots.get(market_key, {}).get("context_state", "unclear")) or "unclear").strip().lower()
            market_context_summary = f"Context for {market_key.title()} is {state}."
        elif str(market_context_result.get("status", "") or "").strip() not in {"", "not_requested"}:
            market_context_summary = "AI market context unavailable; local ranking remains active."
    openai_market_context_payload = {
        "enabled": bool(market_context_result.get("enabled", False)),
        "active": bool(market_context_result.get("active", False)),
        "status": str(market_context_result.get("status", "") or "").strip(),
        "summary": market_context_summary[:220],
        "error": str(market_context_result.get("error", "") or "").strip()[:180],
        "model": str(market_context_result.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(market_context_result.get("latency_ms", 0), 0.0))),
        "market_context_scores": [
            dict(item)
            for item in list(market_context_result.get("market_context_scores", []) or [])[:12]
            if isinstance(item, dict)
        ],
        "symbol_context_scores": [
            dict(item)
            for item in list(market_context_result.get("symbol_context_scores", []) or [])[:36]
            if isinstance(item, dict)
        ],
        "by_market": dict(context_by_market),
        "context_state_by_market": {mk: str(context_states.get(mk, "unclear")) for mk in _MARKETS},
        "context_adjustment_by_market": {mk: round(float(_f(context_adjustments.get(mk, 0.0), 0.0)), 4) for mk in _MARKETS},
        "applied": bool(any(abs(float(_f(context_adjustments.get(mk, 0.0), 0.0))) > 0.01 for mk in _MARKETS)),
    }

    root_cause_summary = str(root_cause_result.get("summary", "") or "").strip()
    if bool(root_cause_result.get("enabled", False)) and (not root_cause_summary):
        if bool(root_cause_result.get("active", False)):
            root_cause_summary = str(root_cause_market_action.get("reason", "") or "").strip()
        elif str(root_cause_result.get("status", "") or "").strip() not in {"", "not_requested"}:
            root_cause_summary = "AI root-cause analysis unavailable; local allocator remains active."
    openai_root_cause_payload = {
        "enabled": bool(root_cause_result.get("enabled", False)),
        "active": bool(root_cause_result.get("active", False)),
        "status": str(root_cause_result.get("status", "") or "").strip(),
        "summary": root_cause_summary[:220],
        "error": str(root_cause_result.get("error", "") or "").strip()[:180],
        "model": str(root_cause_result.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(root_cause_result.get("latency_ms", 0), 0.0))),
        "overall_assessment": str(root_cause_result.get("overall_assessment", "") or "").strip().lower()[:24],
        "throttle_recommendations": [
            dict(item)
            for item in list(root_cause_result.get("throttle_recommendations", []) or [])[:12]
            if isinstance(item, dict)
        ],
        "market_action": dict(root_cause_market_action),
        "global_risks": [str(x or "")[:120] for x in list(root_cause_result.get("global_risks", []) or [])[:12] if str(x or "").strip()],
        "applied": bool(root_cause_applied),
    }

    return {
        "ts": int(now),
        "market": market_key,
        "independent_market_mode": bool(independent_mode),
        "candidate_id": str(candidate_id or "").strip().upper(),
        "decision": str(decision),
        "decision_source": str(decision_source),
        "size_multiplier": round(float(final_size_multiplier), 6),
        "summary": str(summary)[:220],
        "reasons": final_reasons,
        "best_market": str(best_market),
        "best_market_score": round(float(best_score), 4),
        "current_market_score": round(float(current_score), 4),
        "score_gap": round(float(score_gap), 4),
        "capital_constrained": bool(capital_constrained),
        "portfolio_account_value_usd": round(float(max(0.0, portfolio_account_value_usd)), 4),
        "projected_trade_value_usd": round(float(max(0.0, projected_trade_value_usd)), 4),
        "projected_total_exposure_pct": round(float(max(0.0, projected_total_exposure_pct)), 4),
        "projected_market_share_pct": round(float(max(0.0, projected_market_share_pct)), 4),
        "scores": {mk: round(float(scores.get(mk, 0.0)), 4) for mk in _MARKETS},
        "eligibility": {mk: bool(eligibility.get(mk, False)) for mk in _MARKETS},
        "context_state_by_market": {mk: str(context_states.get(mk, "unclear")) for mk in _MARKETS},
        "context_adjustment_by_market": {mk: round(float(_f(context_adjustments.get(mk, 0.0), 0.0)), 4) for mk in _MARKETS},
        "market_reasons": {mk: list(market_reasons.get(mk, [])) for mk in _MARKETS},
        "openai_decision": openai_decision_payload,
        "openai_capital_planner": openai_capital_planner_payload,
        "openai_root_cause_analysis": openai_root_cause_payload,
        "openai_market_context": openai_market_context_payload,
    }


def summarize_allocator_snapshot(
    stocks: Dict[str, Any] | None,
    forex: Dict[str, Any] | None,
    crypto: Dict[str, Any] | None,
) -> Dict[str, Any]:
    market_rows = {
        "stocks": stocks if isinstance(stocks, dict) else {},
        "forex": forex if isinstance(forex, dict) else {},
        "crypto": crypto if isinstance(crypto, dict) else {},
    }

    allocator_rows: Dict[str, Dict[str, Any]] = {}
    latest: Dict[str, Any] = {}
    latest_ts = 0
    for mk, row in market_rows.items():
        alloc = row.get("opportunity_allocator", {}) if isinstance(row.get("opportunity_allocator", {}), dict) else {}
        if alloc:
            allocator_rows[mk] = dict(alloc)
            ts = int(_f(alloc.get("ts", 0), 0.0))
            if ts >= latest_ts:
                latest = dict(alloc)
                latest_ts = ts

    if not allocator_rows:
        return {"active": False, "ts": int(time.time()), "summary": "", "scores": {}, "decisions": {}, "best_market": ""}

    scores = latest.get("scores", {}) if isinstance(latest.get("scores", {}), dict) else {}
    if not scores:
        scores = {}
        for mk in _MARKETS:
            row = allocator_rows.get(mk, {})
            if not isinstance(row, dict):
                continue
            scores[mk] = _f(row.get("current_market_score", 0.0), 0.0)
    best_market = str(latest.get("best_market", "") or "")
    if not best_market and scores:
        ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
        best_market = str(ranked[0][0])
    best_score = _f(((scores or {}).get(best_market, 0.0) if best_market else 0.0), 0.0)

    decisions: Dict[str, str] = {}
    deprioritized: List[Dict[str, Any]] = []
    for mk in _MARKETS:
        row = allocator_rows.get(mk, {})
        decision = str(row.get("decision", "") or "").strip().lower()
        if not decision:
            decision = "unknown"
        decisions[mk] = decision
        if decision in {"deprioritize", "block"}:
            deprioritized.append(
                {
                    "market": mk,
                    "decision": decision,
                    "summary": str(row.get("summary", "") or "")[:220],
                }
            )

    summary = str(latest.get("summary", "") or "").strip()
    if not summary and best_market:
        summary = f"Best current opportunity: {best_market.title()} ({best_score:.1f})"
    openai_latest = latest.get("openai_decision", {}) if isinstance(latest.get("openai_decision", {}), dict) else {}
    openai_summary = str(openai_latest.get("summary", "") or "").strip()
    openai_position_actions = (
        openai_latest.get("position_actions_summary", [])
        if isinstance(openai_latest.get("position_actions_summary", []), list)
        else []
    )
    position_action_counts: Dict[str, int] = {}
    for row in openai_position_actions:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action", "") or "").strip().lower()
        if not action:
            continue
        position_action_counts[action] = int(position_action_counts.get(action, 0) or 0) + 1
    openai_payload = {
        "enabled": bool(openai_latest.get("enabled", False)),
        "active": bool(openai_latest.get("active", False)),
        "status": str(openai_latest.get("status", "") or "").strip(),
        "summary": openai_summary[:220],
        "decision": str(openai_latest.get("decision", "") or "").strip().lower(),
        "best_market": str(openai_latest.get("best_market", "") or "").strip().lower(),
        "portfolio_action": str(openai_latest.get("portfolio_action", "") or "").strip().lower(),
        "portfolio_confidence": round(float(_clamp(_f(openai_latest.get("portfolio_confidence", 0.0), 0.0), 0.0, 1.0)), 6),
        "latency_ms": int(max(0, _f(openai_latest.get("latency_ms", 0), 0.0))),
        "applied": bool(openai_latest.get("applied", False)),
        "position_actions_count": int(len([row for row in openai_position_actions if isinstance(row, dict)])),
        "position_action_counts": {str(k): int(v) for k, v in position_action_counts.items()},
        "position_actions": [dict(row) for row in openai_position_actions[:12] if isinstance(row, dict)],
    }
    planner_latest = latest.get("openai_capital_planner", {}) if isinstance(latest.get("openai_capital_planner", {}), dict) else {}
    planner_payload = {
        "enabled": bool(planner_latest.get("enabled", False)),
        "active": bool(planner_latest.get("active", False)),
        "status": str(planner_latest.get("status", "") or "").strip(),
        "summary": str(planner_latest.get("summary", "") or "").strip()[:220],
        "model": str(planner_latest.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(planner_latest.get("latency_ms", 0), 0.0))),
        "applied": bool(planner_latest.get("applied", False)),
        "market_action": dict(planner_latest.get("market_action", {}) or {}) if isinstance(planner_latest.get("market_action", {}), dict) else {},
        "portfolio_plan": dict(planner_latest.get("portfolio_plan", {}) or {}) if isinstance(planner_latest.get("portfolio_plan", {}), dict) else {},
        "market_actions": [
            dict(row)
            for row in list(planner_latest.get("market_actions", []) or [])[:12]
            if isinstance(row, dict)
        ],
        "global_risks": [str(x or "")[:120] for x in list(planner_latest.get("global_risks", []) or [])[:12] if str(x or "").strip()],
    }
    root_latest = latest.get("openai_root_cause_analysis", {}) if isinstance(latest.get("openai_root_cause_analysis", {}), dict) else {}
    root_payload = {
        "enabled": bool(root_latest.get("enabled", False)),
        "active": bool(root_latest.get("active", False)),
        "status": str(root_latest.get("status", "") or "").strip(),
        "summary": str(root_latest.get("summary", "") or "").strip()[:220],
        "model": str(root_latest.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(root_latest.get("latency_ms", 0), 0.0))),
        "applied": bool(root_latest.get("applied", False)),
        "overall_assessment": str(root_latest.get("overall_assessment", "") or "").strip().lower()[:24],
        "market_action": dict(root_latest.get("market_action", {}) or {}) if isinstance(root_latest.get("market_action", {}), dict) else {},
        "throttle_recommendations": [
            dict(row)
            for row in list(root_latest.get("throttle_recommendations", []) or [])[:12]
            if isinstance(row, dict)
        ],
        "global_risks": [str(x or "")[:120] for x in list(root_latest.get("global_risks", []) or [])[:12] if str(x or "").strip()],
    }
    context_latest = latest.get("openai_market_context", {}) if isinstance(latest.get("openai_market_context", {}), dict) else {}
    context_payload = {
        "enabled": bool(context_latest.get("enabled", False)),
        "active": bool(context_latest.get("active", False)),
        "status": str(context_latest.get("status", "") or "").strip(),
        "summary": str(context_latest.get("summary", "") or "").strip()[:220],
        "model": str(context_latest.get("model", "") or "").strip(),
        "latency_ms": int(max(0, _f(context_latest.get("latency_ms", 0), 0.0))),
        "applied": bool(context_latest.get("applied", False)),
        "by_market": dict(context_latest.get("by_market", {}) or {}) if isinstance(context_latest.get("by_market", {}), dict) else {},
        "context_state_by_market": (
            dict(context_latest.get("context_state_by_market", {}) or {})
            if isinstance(context_latest.get("context_state_by_market", {}), dict)
            else {}
        ),
        "context_adjustment_by_market": (
            dict(context_latest.get("context_adjustment_by_market", {}) or {})
            if isinstance(context_latest.get("context_adjustment_by_market", {}), dict)
            else {}
        ),
        "market_context_scores": [
            dict(row)
            for row in list(context_latest.get("market_context_scores", []) or [])[:12]
            if isinstance(row, dict)
        ],
    }
    if bool(openai_payload.get("active", False)) and openai_summary:
        summary = f"{summary} | AI: {openai_summary}"[:220]
    elif bool(openai_payload.get("active", False)) and int(openai_payload.get("position_actions_count", 0) or 0) > 0:
        summary = f"{summary} | AI reviewed {int(openai_payload.get('position_actions_count', 0) or 0)} open positions"[:220]
    planner_summary = str(planner_payload.get("summary", "") or "").strip()
    if bool(planner_payload.get("active", False)) and planner_summary:
        summary = f"{summary} | Planner: {planner_summary}"[:220]
    root_summary = str(root_payload.get("summary", "") or "").strip()
    if bool(root_payload.get("active", False)) and root_summary:
        summary = f"{summary} | Root cause: {root_summary}"[:220]
    context_summary = str(context_payload.get("summary", "") or "").strip()
    if bool(context_payload.get("active", False)) and context_summary:
        summary = f"{summary} | Context: {context_summary}"[:220]

    return {
        "active": True,
        "ts": int(max(latest_ts, int(time.time()))),
        "best_market": str(best_market),
        "best_score": round(float(best_score), 4),
        "scores": {mk: round(float(_f(scores.get(mk, 0.0), 0.0)), 4) for mk in _MARKETS},
        "decisions": decisions,
        "deprioritized_markets": deprioritized[:6],
        "openai_decision": openai_payload,
        "openai_capital_planner": planner_payload,
        "openai_root_cause_analysis": root_payload,
        "openai_market_context": context_payload,
        "summary": summary[:220],
    }
