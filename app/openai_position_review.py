from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_ALLOWED_MARKET = {"crypto", "stocks", "forex"}
_ALLOWED_ACTION = {"hold", "reduce", "increase", "exit", "block_add", "monitor"}

OPENAI_POSITION_REVIEW_SYSTEM_PROMPT = """
You are the scheduled open-position review advisor for an automated multi-market trading app.

Your job is to evaluate existing open positions across crypto, stocks, and forex and return strict JSON guidance.

You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must be conservative when important data is missing, stale, or weak.

Primary objective:
Improve portfolio-level management of existing positions by recommending bounded, explainable actions that reduce avoidable churn and risk while preserving growth opportunities.

Review principles:
1. Prefer practical, explainable actions over aggressive churn.
2. Penalize stale or misaligned positions, weak confidence context, and adverse execution conditions.
3. Prefer hold/monitor over forced action when evidence is weak.
4. Respect local add/exit permission flags as authoritative context.
5. Recommend increases only for clearly aligned and well-supported positions.
6. Recommend reduce/exit conservatively when risk pressure is elevated.

Return only JSON matching the schema.
""".strip()


OPENAI_POSITION_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "position_actions", "portfolio_risks"],
    "properties": {
        "summary": {"type": "string"},
        "position_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "market",
                    "symbol",
                    "action",
                    "confidence",
                    "size_multiplier",
                    "reduce_fraction",
                    "reason",
                    "risk_flags",
                ],
                "properties": {
                    "market": {"type": "string", "enum": ["crypto", "stocks", "forex"]},
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(_ALLOWED_ACTION)},
                    "confidence": {"type": "number"},
                    "size_multiplier": {"type": "number"},
                    "reduce_fraction": {"type": "number"},
                    "reason": {"type": "string"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "portfolio_risks": {"type": "array", "items": {"type": "string"}},
    },
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if txt in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _s(value: Any) -> str:
    return str(value or "").strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _market(value: Any) -> str:
    mk = _s(value).lower()
    if mk == "stock":
        return "stocks"
    return mk if mk in _ALLOWED_MARKET else "crypto"


def _trading_mode(settings: Dict[str, Any]) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    if bool(cfg.get("alpaca_paper_mode", False)) or bool(cfg.get("oanda_practice_mode", False)):
        return "paper"
    return "live"


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl_tail(path: str, limit: int = 3000) -> List[Dict[str, Any]]:
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


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_state_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_state.json")
    if mk == "forex":
        return os.path.join(hub_dir, "forex", "forex_trader_state.json")
    return ""


def _market_thinker_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto_dynamic_status.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _audit_symbol(market: str, row: Dict[str, Any]) -> str:
    mk = _market(market)
    if mk == "forex":
        return _s(row.get("instrument", row.get("pair", row.get("symbol", "")))).upper()
    return _s(row.get("symbol", row.get("pair", row.get("instrument", "")))).upper()


def _audit_event(row: Dict[str, Any]) -> str:
    return _s(row.get("event")).lower()


def _audit_realized_pnl_usd(row: Dict[str, Any]) -> float:
    for key in ("realized_pnl_usd", "realized_pnl", "realized_pl", "pnl_usd", "realized"):
        if key in row:
            return float(_f(row.get(key, 0.0), 0.0))
    return 0.0


def _collect_audit_context(
    *,
    market: str,
    audit_rows: List[Dict[str, Any]],
    cutoff_ts: int,
) -> Dict[str, Any]:
    latest_entry_ts: Dict[str, int] = {}
    realized_by_symbol: Dict[str, float] = {}
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    stale_exit_count = 0
    churn_count = 0
    top_exit_reasons: Counter[str] = Counter()

    for row in list(audit_rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts <= 0:
            continue
        symbol = _audit_symbol(market, row)
        event = _audit_event(row)
        msg = _s(row.get("msg", row.get("tag", row.get("reason", "")))).lower()
        if event == "entry":
            entries += 1
            if symbol:
                if ts >= int(latest_entry_ts.get(symbol, 0) or 0):
                    latest_entry_ts[symbol] = int(ts)
            continue
        if event == "exit":
            exits += 1
            pnl = float(_audit_realized_pnl_usd(row))
            if ts >= int(cutoff_ts):
                realized_by_symbol[symbol] = float(realized_by_symbol.get(symbol, 0.0) + pnl)
                if pnl > 0.0:
                    wins += 1
                elif pnl < 0.0:
                    losses += 1
                hold_s = int(max(0.0, _f(row.get("hold_s", 0), 0.0)))
                if hold_s > 0 and hold_s <= 6 * 3600:
                    churn_count += 1
            reason_token = _s(row.get("tag", row.get("reason", row.get("source", "unknown")))).lower()
            if reason_token:
                top_exit_reasons[reason_token[:80]] += 1
            if ("stale" in msg) or ("policy_stale_exit" in msg) or ("misalign" in msg):
                stale_exit_count += 1
            continue

    return {
        "latest_entry_ts": latest_entry_ts,
        "realized_by_symbol": realized_by_symbol,
        "entries_lookback": int(entries),
        "exits_lookback": int(exits),
        "wins_lookback": int(wins),
        "losses_lookback": int(losses),
        "stale_exit_count_lookback": int(stale_exit_count),
        "churn_count_lookback": int(churn_count),
        "top_exit_reasons": [
            {"reason": str(key), "count": int(val)}
            for key, val in top_exit_reasons.most_common(5)
        ],
    }


def _candidate_map(market: str, thinker: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mk = _market(market)
    out: Dict[str, Dict[str, Any]] = {}
    if mk == "crypto":
        ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
        for idx, row in enumerate(ranked):
            if not isinstance(row, dict):
                continue
            symbol = _s(row.get("symbol")).upper()
            if not symbol:
                continue
            out[symbol] = {
                "signal_score": round(_f(row.get("score", 0.0), 0.0), 6),
                "local_rank": int(idx + 1),
                "spread_bps": round(max(0.0, _f(row.get("spread_bps", 0.0), 0.0)), 6),
                "reason": _s(row.get("reason_logic", row.get("reason", "")))[:180],
            }
    else:
        rows = thinker.get("leaders", []) if isinstance(thinker.get("leaders", []), list) else []
        if not rows:
            top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
            if top:
                rows = [top]
        symbol_key = "pair" if mk == "forex" else "symbol"
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            symbol = _s(row.get(symbol_key)).upper()
            if not symbol:
                continue
            out[symbol] = {
                "signal_score": round(_f(row.get("score", 0.0), 0.0), 6),
                "local_rank": int(idx + 1),
                "spread_bps": round(max(0.0, _f(row.get("spread_bps", 0.0), 0.0)), 6),
                "reason": _s(row.get("reason_logic", row.get("reason", "")))[:180],
            }
    return out


def _collect_open_positions(
    *,
    market: str,
    now_ts: int,
    trader_status: Dict[str, Any],
    market_state: Dict[str, Any],
    candidate_by_symbol: Dict[str, Dict[str, Any]],
    audit_ctx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mk = _market(market)
    trader = trader_status if isinstance(trader_status, dict) else {}
    state = market_state if isinstance(market_state, dict) else {}
    policy = trader.get("automation_policy", {}) if isinstance(trader.get("automation_policy", {}), dict) else {}
    trade_quality = trader.get("trade_quality", {}) if isinstance(trader.get("trade_quality", {}), dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
    gate = trader.get("entry_gate_flags", {}) if isinstance(trader.get("entry_gate_flags", {}), dict) else {}

    latest_entry = audit_ctx.get("latest_entry_ts", {}) if isinstance(audit_ctx.get("latest_entry_ts", {}), dict) else {}
    realized_by_symbol = (
        audit_ctx.get("realized_by_symbol", {})
        if isinstance(audit_ctx.get("realized_by_symbol", {}), dict)
        else {}
    )

    cooldown_until = int(max(0.0, _f(state.get("cooldown_until", 0), 0.0)))
    cooldown_active = bool(cooldown_until > int(now_ts))
    skip_new_entries = bool(gate.get("skip_new_entries_this_cycle", False))
    base_add_allowed = bool(policy.get("allow_new_entries", True) and (not skip_new_entries) and (not cooldown_active))
    trade_quality_conf = round(max(0.0, _f(trade_quality.get("confidence_score", 0.0), 0.0)), 6)
    runtime_trust_score = round(max(0.0, _f(trust.get("score", 0.0), 0.0)), 6)

    out: List[Dict[str, Any]] = []
    if mk == "crypto":
        positions = trader.get("positions", {}) if isinstance(trader.get("positions", {}), dict) else {}
        for symbol, row in positions.items():
            if not isinstance(row, dict):
                continue
            sym = _s(symbol).upper()
            if not sym:
                continue
            qty = max(0.0, _f(row.get("quantity", 0.0), 0.0))
            if qty <= 0.0:
                continue
            market_value = max(0.0, _f(row.get("value_usd", 0.0), 0.0))
            avg_cost = max(0.0, _f(row.get("avg_cost_basis", 0.0), 0.0))
            notional = max(0.0, avg_cost * qty)
            pnl_pct = _f(row.get("gain_loss_pct_sell", row.get("gain_loss_pct_buy", 0.0)), 0.0)
            pnl_usd = _f(row.get("unrealized_pnl_usd", market_value * (pnl_pct / 100.0)), 0.0)
            entry_ts = int(max(0.0, _f(row.get("entry_ts", latest_entry.get(sym, 0)), 0.0)))
            if entry_ts <= 0:
                entry_ts = int(max(0.0, _f(latest_entry.get(sym, 0), 0.0)))
            age_m = int(max(0, (int(now_ts) - entry_ts) // 60)) if entry_ts > 0 else 0
            last_add_ts = int(max(0.0, _f(row.get("last_add_ts", entry_ts), 0.0)))
            last_add_age_m = int(max(0, (int(now_ts) - last_add_ts) // 60)) if last_add_ts > 0 else age_m
            cand = candidate_by_symbol.get(sym, {}) if isinstance(candidate_by_symbol.get(sym, {}), dict) else {}
            alignment_reasons = row.get("alignment_reasons", [])
            if not isinstance(alignment_reasons, list):
                alignment_reasons = []
            out.append(
                {
                    "market": mk,
                    "symbol": sym,
                    "side": "long",
                    "quantity": round(float(qty), 10),
                    "units": int(max(0.0, _f(row.get("units", 0), 0.0))),
                    "notional_usd": round(float(notional), 6),
                    "market_value_usd": round(float(market_value), 6),
                    "unrealized_pnl_usd": round(float(pnl_usd), 6),
                    "unrealized_pnl_pct": round(float(pnl_pct), 6),
                    "realized_pnl_usd_recent": round(float(_f(realized_by_symbol.get(sym, 0.0), 0.0)), 6),
                    "entry_ts": int(entry_ts),
                    "entry_age_minutes": int(age_m),
                    "last_add_ts": int(last_add_ts),
                    "last_add_age_minutes": int(last_add_age_m),
                    "alignment_with_strategy": bool(row.get("aligned_with_strategy", True)),
                    "alignment_reasons": [str(x or "")[:120] for x in alignment_reasons[:5] if str(x or "").strip()],
                    "alignment_streak": int(max(0.0, _f(row.get("alignment_streak", 0), 0.0))),
                    "trade_quality_confidence_score": float(trade_quality_conf),
                    "runtime_trust_score": float(runtime_trust_score),
                    "policy_profile": _s(policy.get("profile"))[:32],
                    "policy_mode": _s(policy.get("mode"))[:32],
                    "cooldown_active": bool(cooldown_active),
                    "cooldown_remaining_s": int(max(0, cooldown_until - int(now_ts))) if cooldown_active else 0,
                    "add_allowed_local": bool(base_add_allowed),
                    "exit_allowed_local": True,
                    "current_signal_score": round(float(_f(cand.get("signal_score", 0.0), 0.0)), 6),
                    "current_local_rank": int(max(0.0, _f(cand.get("local_rank", 0), 0.0))),
                    "spread_bps": round(float(max(0.0, _f(cand.get("spread_bps", 0.0), 0.0))), 6),
                    "max_slippage_bps": round(float(max(0.0, _f(trader.get("max_slippage_bps", 0.0), 0.0))), 6),
                    "local_reason_summary": _s(cand.get("reason", trader.get("entry_eval_top_reason", "")))[:180],
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
    compliance_entry_blocked = bool(gate.get("compliance_entry_blocked", False))
    pdt_restricted = bool(gate.get("pdt_restricted", False))
    min_hold_minutes = int(max(0.0, _f(gate.get("same_day_exit_min_hold_minutes", 0), 0.0)))

    for symbol, meta in open_meta.items():
        if not isinstance(meta, dict):
            continue
        sym = _s(symbol).upper()
        if not sym:
            continue
        entry_ts = int(max(0.0, _f(meta.get("entry_ts", latest_entry.get(sym, 0)), 0.0)))
        if entry_ts <= 0:
            entry_ts = int(max(0.0, _f(latest_entry.get(sym, 0), 0.0)))
        age_m = int(max(0, (int(now_ts) - entry_ts) // 60)) if entry_ts > 0 else 0
        last_add_ts = int(max(entry_ts, int(max(0.0, _f(meta.get("last_add_ts", entry_ts), 0.0)))))
        last_add_age_m = int(max(0, (int(now_ts) - last_add_ts) // 60)) if last_add_ts > 0 else age_m
        market_value = float(max(0.0, _f(position_values.get(sym, 0.0), 0.0)))
        notional = float(max(0.0, _f(meta.get("notional", market_value), market_value)))
        pnl_pct = float(_f(meta.get("last_pnl_pct", 0.0), 0.0))
        pnl_usd = float(_f(meta.get("last_pnl_usd", market_value * (pnl_pct / 100.0)), 0.0))
        side = _s(meta.get("side")).lower()
        if side not in {"long", "short"}:
            units = int(_f(meta.get("units", 0), 0.0))
            side = "short" if units < 0 else "long"
        stale_streak = int(max(0.0, _f(stale_streaks.get(sym, 0), 0.0)))
        alignment_reasons: List[str] = []
        if stale_streak > 0:
            alignment_reasons.append("alignment_streak_active")
        add_allowed = bool(base_add_allowed and (not compliance_entry_blocked))
        if mk == "stocks":
            add_allowed = False
        exit_allowed = True
        if mk == "stocks" and pdt_restricted and min_hold_minutes > 0 and age_m < min_hold_minutes:
            exit_allowed = False
            alignment_reasons.append("pdt_hold_gate_active")
        cand = candidate_by_symbol.get(sym, {}) if isinstance(candidate_by_symbol.get(sym, {}), dict) else {}
        out.append(
            {
                "market": mk,
                "symbol": sym,
                "side": side,
                "quantity": round(float(max(0.0, _f(meta.get("qty", meta.get("quantity", 0.0)), 0.0))), 10),
                "units": int(_f(meta.get("units", 0), 0.0)),
                "notional_usd": round(float(notional), 6),
                "market_value_usd": round(float(market_value), 6),
                "unrealized_pnl_usd": round(float(pnl_usd), 6),
                "unrealized_pnl_pct": round(float(pnl_pct), 6),
                "realized_pnl_usd_recent": round(float(_f(realized_by_symbol.get(sym, 0.0), 0.0)), 6),
                "entry_ts": int(entry_ts),
                "entry_age_minutes": int(age_m),
                "last_add_ts": int(last_add_ts),
                "last_add_age_minutes": int(last_add_age_m),
                "alignment_with_strategy": bool(stale_streak <= 0),
                "alignment_reasons": [str(x or "")[:120] for x in alignment_reasons[:5] if str(x or "").strip()],
                "alignment_streak": int(stale_streak),
                "trade_quality_confidence_score": float(trade_quality_conf),
                "runtime_trust_score": float(runtime_trust_score),
                "policy_profile": _s(policy.get("profile"))[:32],
                "policy_mode": _s(policy.get("mode"))[:32],
                "cooldown_active": bool(cooldown_active),
                "cooldown_remaining_s": int(max(0, cooldown_until - int(now_ts))) if cooldown_active else 0,
                "add_allowed_local": bool(add_allowed),
                "exit_allowed_local": bool(exit_allowed),
                "current_signal_score": round(float(_f(cand.get("signal_score", 0.0), 0.0)), 6),
                "current_local_rank": int(max(0.0, _f(cand.get("local_rank", 0), 0.0))),
                "spread_bps": round(float(max(0.0, _f(cand.get("spread_bps", 0.0), 0.0))), 6),
                "max_slippage_bps": round(float(max(0.0, _f(meta.get("max_slippage_bps", trader.get("max_slippage_bps", 0.0)), 0.0))), 6),
                "local_reason_summary": _s(cand.get("reason", trader.get("entry_eval_top_reason", "")))[:180],
            }
        )
    return out


def build_openai_position_review_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    lookback_days = max(1, min(30, int(_f(cfg.get("openai_nightly_review_lookback_days", 7), 7))))
    cutoff_ts = int(now_i - (lookback_days * 86400))
    max_positions = max(8, min(200, int(_f(cfg.get("openai_position_review_max_positions", 48), 48))))

    open_rows: List[Dict[str, Any]] = []
    market_summary: Dict[str, Dict[str, Any]] = {}
    total_exposure_usd = 0.0
    portfolio_value_usd = 0.0
    buying_power_est = 0.0
    margin_available_est = 0.0
    mode = _trading_mode(cfg)

    for mk in ("crypto", "stocks", "forex"):
        status = _safe_read_json(_market_status_path(hub_dir, mk))
        state_path = _market_state_path(hub_dir, mk)
        state = _safe_read_json(state_path) if state_path else {}
        thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
        audit_rows = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=5000)
        audit_ctx = _collect_audit_context(market=mk, audit_rows=audit_rows, cutoff_ts=cutoff_ts)
        candidate_by_symbol = _candidate_map(mk, thinker)
        positions = _collect_open_positions(
            market=mk,
            now_ts=now_i,
            trader_status=status,
            market_state=state,
            candidate_by_symbol=candidate_by_symbol,
            audit_ctx=audit_ctx,
        )
        open_rows.extend(positions)

        exposure_usd = float(max(0.0, _f(status.get("exposure_usd", 0.0), 0.0)))
        account_value_usd = float(
            max(
                0.0,
                _f(status.get("account_value_usd", 0.0), 0.0),
                _f((status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}).get("total_account_value", 0.0), 0.0),
            )
        )
        buying_power = float(
            max(
                0.0,
                _f(status.get("buying_power_usd", 0.0), 0.0),
                _f((status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}).get("buying_power", 0.0), 0.0),
                _f(status.get("margin_available_usd", 0.0), 0.0),
            )
        )
        margin_available = float(max(0.0, _f(status.get("margin_available_usd", buying_power), buying_power)))
        total_exposure_usd += exposure_usd
        portfolio_value_usd = max(portfolio_value_usd, account_value_usd)
        buying_power_est = max(buying_power_est, buying_power)
        margin_available_est = max(margin_available_est, margin_available)

        gate = status.get("entry_gate_flags", {}) if isinstance(status.get("entry_gate_flags", {}), dict) else {}
        policy = status.get("automation_policy", {}) if isinstance(status.get("automation_policy", {}), dict) else {}
        trade_quality = status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}
        trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
        market_summary[mk] = {
            "open_positions": int(len(positions)),
            "realized_pnl_lookback_usd": round(float(sum(_f(row.get("realized_pnl_usd_recent", 0.0), 0.0) for row in positions)), 6),
            "entries_lookback": int(audit_ctx.get("entries_lookback", 0) or 0),
            "exits_lookback": int(audit_ctx.get("exits_lookback", 0) or 0),
            "wins_lookback": int(audit_ctx.get("wins_lookback", 0) or 0),
            "losses_lookback": int(audit_ctx.get("losses_lookback", 0) or 0),
            "stale_exit_count_lookback": int(audit_ctx.get("stale_exit_count_lookback", 0) or 0),
            "churn_count_lookback": int(audit_ctx.get("churn_count_lookback", 0) or 0),
            "top_exit_reasons": list(audit_ctx.get("top_exit_reasons", []) or [])[:5],
            "runtime_trust_score": round(float(max(0.0, _f(trust.get("score", 0.0), 0.0))), 6),
            "trade_quality_confidence_score": round(float(max(0.0, _f(trade_quality.get("confidence_score", 0.0), 0.0))), 6),
            "trade_quality_decision": _s(trade_quality.get("decision"))[:24].lower(),
            "allow_new_entries": bool(policy.get("allow_new_entries", True)),
            "loss_streak": int(max(0.0, _f(gate.get("loss_streak", status.get("loss_streak", 0)), 0.0))),
        }

    rows_sorted = sorted(
        [row for row in open_rows if isinstance(row, dict)],
        key=lambda item: (_market(item.get("market", "")), _s(item.get("symbol", "")).upper()),
    )
    if len(rows_sorted) > max_positions:
        rows_sorted = rows_sorted[:max_positions]

    total_exposure_pct = (
        ((float(total_exposure_usd) / max(1e-6, float(portfolio_value_usd))) * 100.0)
        if float(portfolio_value_usd) > 0.0
        else 0.0
    )
    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": str(mode),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "portfolio_context": {
            "account_value_usd_est": round(float(max(0.0, portfolio_value_usd)), 6),
            "buying_power_usd_est": round(float(max(0.0, buying_power_est)), 6),
            "margin_available_usd_est": round(float(max(0.0, margin_available_est)), 6),
            "total_exposure_usd_est": round(float(max(0.0, total_exposure_usd)), 6),
            "total_exposure_pct_est": round(float(max(0.0, total_exposure_pct)), 6),
            "market_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))), 6),
            "crypto_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("stock_max_total_exposure_pct", 0.0), 0.0))), 6),
            "forex_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("forex_max_total_exposure_pct", 0.0), 0.0))), 6),
        },
        "market_summary": market_summary,
        "open_positions": rows_sorted,
    }


def _extract_json_text(response_json: Dict[str, Any]) -> str:
    if not isinstance(response_json, dict):
        return ""
    out_text = response_json.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()

    chunks: List[str] = []
    output = response_json.get("output", [])
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "output_text", "value"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return "\n".join(chunks).strip()


def _trim_json_object(text: str) -> str:
    src = _s(text)
    if not src:
        return ""
    if src.startswith("{") and src.endswith("}"):
        return src
    start = src.find("{")
    end = src.rfind("}")
    if start >= 0 and end > start:
        return src[start : end + 1]
    return src


def _normalize_position_action(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    action = _s(src.get("action")).lower()
    if action not in _ALLOWED_ACTION:
        action = "monitor"
    risk_flags = src.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    return {
        "market": market,
        "symbol": _s(src.get("symbol"))[:48].upper(),
        "action": action,
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "size_multiplier": round(_clamp(_f(src.get("size_multiplier", 1.0), 1.0), 0.25, 1.5), 6),
        "reduce_fraction": round(_clamp(_f(src.get("reduce_fraction", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason"))[:220],
        "risk_flags": [_s(x)[:96] for x in risk_flags[:8] if _s(x)],
    }


def _normalize_review_payload(raw: Dict[str, Any], *, max_positions: int) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Position review response was not a JSON object"

    actions_src = raw.get("position_actions", [])
    if not isinstance(actions_src, list):
        actions_src = []
    actions = [
        _normalize_position_action(row)
        for row in actions_src[: max(8, int(max_positions))]
        if isinstance(row, dict)
    ]
    risks_src = raw.get("portfolio_risks", [])
    if not isinstance(risks_src, list):
        risks_src = []
    return {
        "summary": _s(raw.get("summary"))[:260],
        "position_actions": actions,
        "portfolio_risks": [_s(x)[:120] for x in risks_src[:16] if _s(x)],
    }, ""


def _position_review_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_position_review_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_position_review_live_enabled"), True)):
        return False, "live_disabled"
    if cur_mode == "paper" and (not _b(cfg.get("openai_position_review_paper_enabled"), True)):
        return False, "paper_disabled"
    return True, "enabled"


def request_openai_position_review(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    review_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, reason = _position_review_enabled(cfg, mode=str(review_packet.get("mode", "")))
    model = _s(cfg.get("openai_position_review_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_position_review_timeout_s", 8.0), 8.0), 1.0, 30.0)
    max_positions = max(8, min(200, int(_f(cfg.get("openai_position_review_max_positions", 48), 48))))
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "",
            "error": "",
            "review": {},
            "latency_ms": 0,
        }

    api_key = get_openai_api_key(cfg, base_dir=base_dir)
    if not api_key:
        return {
            "enabled": True,
            "active": False,
            "status": "missing_api_key",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "review": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_POSITION_REVIEW_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "position_review", "position_review_input": review_packet},
                            separators=(",", ":"),
                            ensure_ascii=True,
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "position_review",
                "strict": True,
                "schema": OPENAI_POSITION_REVIEW_SCHEMA,
            }
        },
    }

    started = time.time()
    try:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            timeout=(2.0, float(timeout_s)),
        )
    except requests.Timeout:
        return {
            "enabled": True,
            "active": False,
            "status": "timeout",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review timed out; local logic remains active.",
            "error": "OpenAI request timed out",
            "review": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review request failed; local logic remains active.",
            "error": _s(exc)[:180],
            "review": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }

    latency_ms = int(round((time.time() - started) * 1000.0))
    if int(resp.status_code) >= 400:
        return {
            "enabled": True,
            "active": False,
            "status": "http_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review unavailable from OpenAI API; local logic remains active.",
            "error": f"HTTP {int(resp.status_code)}",
            "review": {},
            "latency_ms": latency_ms,
        }

    try:
        response_json = resp.json()
    except Exception:
        return {
            "enabled": True,
            "active": False,
            "status": "invalid_json",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review returned invalid JSON; local logic remains active.",
            "error": "OpenAI response could not be parsed as JSON",
            "review": {},
            "latency_ms": latency_ms,
        }

    text = _trim_json_object(_extract_json_text(response_json))
    if not text:
        return {
            "enabled": True,
            "active": False,
            "status": "empty_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review returned no structured output; local logic remains active.",
            "error": "No output text found in OpenAI response",
            "review": {},
            "latency_ms": latency_ms,
        }

    try:
        raw = json.loads(text)
    except Exception:
        return {
            "enabled": True,
            "active": False,
            "status": "malformed_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review output was malformed; local logic remains active.",
            "error": "Could not decode AI position review JSON payload",
            "review": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_review_payload(raw, max_positions=max_positions)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI position review schema validation failed; local logic remains active.",
            "error": _s(err)[:180],
            "review": {},
            "latency_ms": latency_ms,
        }

    return {
        "enabled": True,
        "active": True,
        "status": "ok",
        "model": model,
        "timeout_s": float(timeout_s),
        "summary": _s(normalized.get("summary", ""))[:220],
        "error": "",
        "review": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def _position_key(market: Any, symbol: Any) -> str:
    return f"{_market(market)}::{_s(symbol).upper()}"


def _apply_local_bounds(
    *,
    review: Dict[str, Any],
    packet: Dict[str, Any],
    auto_act_enabled: bool,
) -> Dict[str, Any]:
    rows = review.get("position_actions", []) if isinstance(review.get("position_actions", []), list) else []
    open_rows = packet.get("open_positions", []) if isinstance(packet.get("open_positions", []), list) else []
    local_map: Dict[str, Dict[str, Any]] = {}
    for row in open_rows:
        if not isinstance(row, dict):
            continue
        key = _position_key(row.get("market", ""), row.get("symbol", ""))
        local_map[key] = dict(row)

    bounded: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    by_market: Dict[str, List[Dict[str, Any]]] = {"crypto": [], "stocks": [], "forex": []}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        key = _position_key(item.get("market", ""), item.get("symbol", ""))
        local = local_map.get(key, {})
        action = _s(item.get("action")).lower()
        eff = str(action)
        blocked_reason = ""
        add_allowed = bool(local.get("add_allowed_local", False))
        exit_allowed = bool(local.get("exit_allowed_local", True))
        if action == "increase" and (not add_allowed):
            eff = "block_add"
            blocked_reason = "increase_not_allowed_by_local_guard"
        elif action in {"reduce", "exit"} and (not exit_allowed):
            eff = "monitor"
            blocked_reason = "exit_not_allowed_by_local_guard"

        item["effective_action"] = eff
        item["auto_action_eligible"] = bool(
            auto_act_enabled and eff in {"increase", "reduce", "exit", "block_add"} and (not blocked_reason)
        )
        item["local_add_allowed"] = bool(add_allowed)
        item["local_exit_allowed"] = bool(exit_allowed)
        if blocked_reason:
            item["bounded_reason"] = blocked_reason
            blocked.append(
                {
                    "market": _market(item.get("market", "")),
                    "symbol": _s(item.get("symbol", "")).upper(),
                    "action": action,
                    "effective_action": eff,
                    "bounded_reason": blocked_reason,
                }
            )
        bounded.append(item)
        mk = _market(item.get("market", ""))
        if mk in by_market:
            by_market[mk].append(item)

    by_market_payload: Dict[str, Dict[str, Any]] = {}
    for mk in ("crypto", "stocks", "forex"):
        rows_mk = by_market.get(mk, [])
        top = rows_mk[:12]
        by_market_payload[mk] = {
            "market": mk,
            "reviewed_positions": int(len(rows_mk)),
            "action_counts": dict(Counter([_s(r.get("effective_action", r.get("action", ""))).lower() for r in rows_mk])),
            "actions": [
                {
                    "symbol": _s(r.get("symbol", "")).upper(),
                    "action": _s(r.get("action", "")).lower(),
                    "effective_action": _s(r.get("effective_action", r.get("action", ""))).lower(),
                    "confidence": round(_clamp(_f(r.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
                    "reason": _s(r.get("reason", ""))[:180],
                }
                for r in top
            ],
        }

    return {
        "bounded_actions": bounded,
        "bounded_count": int(len(bounded)),
        "blocked_count": int(len(blocked)),
        "blocked_actions": blocked[:24],
        "by_market": by_market_payload,
    }


def _write_market_sidecars(
    *,
    hub_dir: str,
    status: Dict[str, Any],
) -> None:
    by_market = status.get("by_market", {}) if isinstance(status.get("by_market", {}), dict) else {}
    ts = int(status.get("ts", 0) or 0)
    global_status = _s(status.get("status", ""))
    summary = _s(status.get("summary", ""))
    for mk in ("crypto", "stocks", "forex"):
        folder = os.path.join(hub_dir, mk)
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            continue
        payload = {
            "ts": int(ts),
            "status": global_status,
            "summary": summary[:220],
            "market": mk,
            "review": dict(by_market.get(mk, {}) or {}) if isinstance(by_market.get(mk, {}), dict) else {},
        }
        try:
            atomic_write_json(os.path.join(folder, "openai_position_review.json"), payload)
        except Exception:
            continue


def run_openai_position_review(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    hub_dir: str,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    date_local = time.strftime("%Y-%m-%d", time.localtime(now_i))
    openai_dir = os.path.join(hub_dir, "openai")
    os.makedirs(openai_dir, exist_ok=True)
    report_path = os.path.join(openai_dir, "position_review.json")
    status_path = os.path.join(openai_dir, "position_review_status.json")

    enabled, enabled_reason = _position_review_enabled(cfg, mode=_trading_mode(cfg))
    auto_act_enabled = bool(cfg.get("openai_position_review_auto_act_enabled", False))
    max_positions = max(8, min(200, int(_f(cfg.get("openai_position_review_max_positions", 48), 48))))
    timeout_s = _clamp(_f(cfg.get("openai_position_review_timeout_s", 8.0), 8.0), 1.0, 30.0)
    model = _s(cfg.get("openai_position_review_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI position review in progress." if enabled else "AI position review disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "max_positions": int(max_positions),
        "auto_act_enabled": bool(auto_act_enabled),
        "actions_count": 0,
        "blocked_count": 0,
        "error": "",
        "report_path": str(report_path),
        "report_written": False,
        "by_market": {"crypto": {}, "stocks": {}, "forex": {}},
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_position_review_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_position_review(settings=cfg, base_dir=base_dir, review_packet=packet)
    review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}
    bounded = {"bounded_actions": [], "bounded_count": 0, "blocked_count": 0, "blocked_actions": [], "by_market": {"crypto": {}, "stocks": {}, "forex": {}}}
    if bool(result.get("active", False)) and review:
        bounded = _apply_local_bounds(
            review=review,
            packet=packet,
            auto_act_enabled=auto_act_enabled,
        )

    report_payload = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(review.get("summary", result.get("summary", "")))[:260],
        "portfolio_risks": [_s(x)[:120] for x in list(review.get("portfolio_risks", []) or [])[:16] if _s(x)],
        "position_actions": [
            dict(row)
            for row in list(bounded.get("bounded_actions", []) or [])[:max_positions]
            if isinstance(row, dict)
        ],
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "mode": str(packet.get("mode", "")),
            "latency_ms": int(max(0, _f(result.get("latency_ms", 0), 0.0))),
            "auto_act_enabled": bool(auto_act_enabled),
            "actions_count": int(bounded.get("bounded_count", 0) or 0),
            "blocked_count": int(bounded.get("blocked_count", 0) or 0),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
        },
        "by_market": dict(bounded.get("by_market", {}) or {}) if isinstance(bounded.get("by_market", {}), dict) else {"crypto": {}, "stocks": {}, "forex": {}},
        "input_packet": {
            "portfolio_context": dict(packet.get("portfolio_context", {}) or {}) if isinstance(packet.get("portfolio_context", {}), dict) else {},
            "market_summary": dict(packet.get("market_summary", {}) or {}) if isinstance(packet.get("market_summary", {}), dict) else {},
            "open_positions_count": int(len([r for r in list(packet.get("open_positions", []) or []) if isinstance(r, dict)])),
        },
    }
    atomic_write_json(report_path, report_payload)

    final_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": False,
        "status": _s(result.get("status", enabled_reason or "disabled")).lower(),
        "summary": _s(review.get("summary", result.get("summary", "")))[:260],
        "portfolio_risks": [_s(x)[:120] for x in list(review.get("portfolio_risks", []) or [])[:8] if _s(x)] if review else [],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "max_positions": int(max_positions),
        "auto_act_enabled": bool(auto_act_enabled),
        "actions_count": int(bounded.get("bounded_count", 0) or 0),
        "blocked_count": int(bounded.get("blocked_count", 0) or 0),
        "error": _s(result.get("error", ""))[:180],
        "latency_ms": int(max(0, _f(result.get("latency_ms", 0), 0.0))),
        "report_path": str(report_path),
        "report_written": True,
        "position_actions": [
            {
                "market": _market(row.get("market", "")),
                "symbol": _s(row.get("symbol", "")).upper(),
                "action": _s(row.get("action", "")).lower(),
                "effective_action": _s(row.get("effective_action", row.get("action", ""))).lower(),
                "confidence": round(_clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
                "reason": _s(row.get("reason", ""))[:180],
                "auto_action_eligible": bool(row.get("auto_action_eligible", False)),
            }
            for row in list(bounded.get("bounded_actions", []) or [])[:24]
            if isinstance(row, dict)
        ],
        "by_market": dict(bounded.get("by_market", {}) or {}) if isinstance(bounded.get("by_market", {}), dict) else {"crypto": {}, "stocks": {}, "forex": {}},
    }
    atomic_write_json(status_path, final_status)
    _write_market_sidecars(hub_dir=hub_dir, status=final_status)
    return final_status


def load_latest_openai_position_review_status(hub_dir: str) -> Dict[str, Any]:
    path = os.path.join(hub_dir, "openai", "position_review_status.json")
    return _safe_read_json(path)


def load_latest_openai_position_review_report(hub_dir: str) -> Dict[str, Any]:
    path = os.path.join(hub_dir, "openai", "position_review.json")
    return _safe_read_json(path)


def load_current_settings_for_position_review(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_position_review") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
