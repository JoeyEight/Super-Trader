from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.openai_trade_review import apply_validated_tuning_suggestions
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_ALLOWED_PRESET_ASSESSMENT = {"well_matched", "too_aggressive", "too_restrictive", "mixed"}
_MARKETS = ("crypto", "stocks", "forex")

OPENAI_STRATEGY_OPTIMIZER_SYSTEM_PROMPT = """
You are the preset and strategy optimizer advisor for an automated multi-market trading app.

Your job is to evaluate whether the current preset/profile and account-level risk configuration are still well matched to observed trading behavior.

You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must be conservative when evidence is weak, stale, or incomplete.

Primary objective:
Recommend bounded preset/strategy tuning changes that can improve expected outcomes while reducing avoidable churn, stale exits, and loss-streak pressure.

Optimizer principles:
1. Prefer advisory clarity over aggressive change volume.
2. Recommend no change when confidence is weak.
3. Prioritize risk fit for current account size and exposure profile.
4. Penalize churn and stale exits; avoid suggesting looser controls without evidence.
5. Keep recommendations explainable and tied to provided data.
6. Treat host application hard guards as authoritative.

Return only JSON matching the schema.
""".strip()


OPENAI_STRATEGY_OPTIMIZER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "preset_assessment", "strategy_suggestions", "risk_flags"],
    "properties": {
        "summary": {"type": "string"},
        "preset_assessment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["current_profile", "assessment", "reason"],
            "properties": {
                "current_profile": {"type": "string"},
                "assessment": {"type": "string", "enum": sorted(_ALLOWED_PRESET_ASSESSMENT)},
                "reason": {"type": "string"},
            },
        },
        "strategy_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["setting_key", "current_value", "suggested_value", "confidence", "reason"],
                "properties": {
                    "setting_key": {"type": "string"},
                    "current_value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "integer"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                    "suggested_value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "integer"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
}


# Optional low-risk allowlist for auto-apply (disabled by default).
_LOW_RISK_OPTIMIZER_RULES: Dict[str, Dict[str, Any]] = {
    "max_dca_buys_per_24h": {"type": "int", "min": 0, "max": 8},
    "dca_multiplier": {"type": "float", "min": 1.0, "max": 4.0, "precision": 3},
    "crypto_dynamic_target_count": {"type": "int", "min": 4, "max": 24},
    "crypto_dynamic_scan_interval_s": {"type": "float", "min": 10.0, "max": 900.0, "precision": 2},
    "crypto_dynamic_min_projected_edge_pct": {"type": "float", "min": 0.05, "max": 2.5, "precision": 4},
    "crypto_dynamic_rotation_cooldown_s": {"type": "float", "min": 60.0, "max": 10800.0, "precision": 2},
    "crypto_max_open_positions": {"type": "int", "min": 1, "max": 16},
    "stock_max_open_positions": {"type": "int", "min": 1, "max": 12},
    "forex_max_open_positions": {"type": "int", "min": 1, "max": 12},
    "stock_score_threshold": {"type": "float", "min": 0.05, "max": 1.5, "precision": 4},
    "forex_score_threshold": {"type": "float", "min": 0.05, "max": 1.5, "precision": 4},
    "max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "stock_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "forex_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "market_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "stock_max_daily_loss_pct": {"type": "float", "min": 0.5, "max": 8.0, "precision": 3},
    "forex_max_daily_loss_pct": {"type": "float", "min": 0.5, "max": 8.0, "precision": 3},
    "stock_stale_alignment_grace_cycles": {"type": "int", "min": 1, "max": 8},
    "forex_stale_alignment_grace_cycles": {"type": "int", "min": 1, "max": 8},
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
    return mk if mk in _MARKETS else "crypto"


def _trading_mode(settings: Dict[str, Any]) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    if bool(cfg.get("alpaca_paper_mode", False)) or bool(cfg.get("oanda_practice_mode", False)):
        return "paper"
    return "live"


def _capital_bucket(account_value_usd: float) -> str:
    val = max(0.0, float(account_value_usd or 0.0))
    if val >= 50_000.0:
        return "large"
    if val >= 10_000.0:
        return "mid"
    if val >= 2_500.0:
        return "small"
    return "micro"


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


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _open_position_count(market: str, status: Dict[str, Any], state: Dict[str, Any]) -> int:
    mk = _market(market)
    count = int(max(0.0, _f(status.get("open_positions", 0), 0.0)))
    if mk == "crypto":
        positions = status.get("positions", {}) if isinstance(status.get("positions", {}), dict) else {}
        if positions:
            active = 0
            for row in positions.values():
                if not isinstance(row, dict):
                    continue
                qty = _f(row.get("quantity", row.get("qty", 0.0)), 0.0)
                if abs(qty) > 0.0:
                    active += 1
            count = max(count, int(active))
        return int(max(0, count))

    open_meta = state.get("open_meta", {}) if isinstance(state.get("open_meta", {}), dict) else {}
    if open_meta:
        count = max(count, len([k for k, v in open_meta.items() if _s(k) and isinstance(v, dict)]))
    position_values = status.get("position_values_usd", {}) if isinstance(status.get("position_values_usd", {}), dict) else {}
    if position_values:
        count = max(count, len([k for k, v in position_values.items() if _s(k) and abs(_f(v, 0.0)) > 0.0]))
    return int(max(0, count))


def _stale_alignment_count(market: str, status: Dict[str, Any], state: Dict[str, Any]) -> int:
    mk = _market(market)
    if mk == "crypto":
        positions = status.get("positions", {}) if isinstance(status.get("positions", {}), dict) else {}
        stale = 0
        for row in positions.values():
            if not isinstance(row, dict):
                continue
            qty = _f(row.get("quantity", row.get("qty", 0.0)), 0.0)
            if abs(qty) <= 0.0:
                continue
            aligned = bool(row.get("aligned_with_strategy", True))
            streak = int(max(0.0, _f(row.get("stale_alignment_streak", 0), 0.0)))
            if (not aligned) or streak > 0:
                stale += 1
        return int(max(0, stale))

    streaks = state.get("stale_alignment_streaks", {}) if isinstance(state.get("stale_alignment_streaks", {}), dict) else {}
    stale = 0
    for val in streaks.values():
        if int(max(0.0, _f(val, 0.0))) > 0:
            stale += 1
    return int(max(0, stale))


def _audit_symbol(market: str, row: Dict[str, Any]) -> str:
    mk = _market(market)
    if mk == "forex":
        return _s(row.get("instrument", row.get("pair", row.get("symbol", "")))).upper()
    return _s(row.get("symbol", row.get("pair", row.get("instrument", "")))).upper()


def _audit_event(row: Dict[str, Any]) -> str:
    return _s(row.get("event", "")).lower()


def _audit_reason(row: Dict[str, Any]) -> str:
    for key in ("msg", "tag", "reason", "source"):
        val = _s(row.get(key, ""))
        if val:
            return val
    return ""


def _audit_realized_pnl_usd(row: Dict[str, Any]) -> float:
    for key in ("realized_pnl_usd", "realized_pnl", "realized_pl", "realized", "pnl_usd", "pnl"):
        if key in row:
            return float(_f(row.get(key, 0.0), 0.0))
    return 0.0


def _audit_hold_s(row: Dict[str, Any]) -> int:
    return int(max(0.0, _f(row.get("hold_s", 0.0), 0.0)))


def _is_stale_reason(reason: str) -> bool:
    txt = _s(reason).lower()
    if not txt:
        return False
    return any(tok in txt for tok in ("stale", "policy_stale_exit", "misalign", "alignment"))


def _is_dca_or_add_reason(reason: str) -> bool:
    txt = _s(reason).lower()
    if not txt:
        return False
    return any(tok in txt for tok in ("dca", "add", "averag"))


def _tokenize_reason(reason: str) -> str:
    txt = _s(reason).lower()
    if not txt:
        return "unknown"
    out = "".join(ch if ch.isalnum() else "_" for ch in txt)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out[:64] if out else "unknown"


def _recent_market_metrics(
    *,
    market: str,
    status: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
    cutoff_ts: int,
) -> Dict[str, Any]:
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    realized_pnl = 0.0
    stale_exits = 0
    churn_count = 0
    dca_add_count = 0
    exit_reasons: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    symbol_losses: Dict[str, float] = {}

    for row in list(audit_rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts <= 0 or ts < int(cutoff_ts):
            continue
        event = _audit_event(row)
        reason = _audit_reason(row)
        token = _tokenize_reason(reason)
        if event in {"entry", "order_entry", "buy"}:
            entries += 1
            if _is_dca_or_add_reason(reason):
                dca_add_count += 1
            continue
        if event in {"exit", "sell", "close"}:
            exits += 1
            pnl = float(_audit_realized_pnl_usd(row))
            realized_pnl += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
                sym = _audit_symbol(market, row)
                if sym:
                    symbol_losses[sym] = float(symbol_losses.get(sym, 0.0) + pnl)
            hold_s = _audit_hold_s(row)
            if hold_s > 0 and hold_s <= 6 * 3600:
                churn_count += 1
            if _is_stale_reason(reason):
                stale_exits += 1
            exit_reasons[token] += 1
            if _is_dca_or_add_reason(reason):
                dca_add_count += 1
            continue
        if event in {"reject", "entry_reject", "shadow_live_divergence"}:
            reject_reasons[token] += 1

    entry_eval_reasons = status.get("entry_eval_reason_counts", {}) if isinstance(status.get("entry_eval_reason_counts", {}), dict) else {}
    for key, value in entry_eval_reasons.items():
        k = _tokenize_reason(_s(key))
        v = int(max(0.0, _f(value, 0.0)))
        if k and v > 0:
            reject_reasons[k] += v

    underperformers = sorted(symbol_losses.items(), key=lambda item: float(item[1]))[:6]
    top_underperformers = [{"symbol": str(sym), "realized_pnl_usd": round(float(val), 6)} for sym, val in underperformers]

    return {
        "entries": int(entries),
        "exits": int(exits),
        "wins": int(wins),
        "losses": int(losses),
        "realized_pnl_usd": round(float(realized_pnl), 6),
        "stale_exit_count": int(stale_exits),
        "churn_count": int(churn_count),
        "dca_add_count": int(dca_add_count),
        "top_exit_reasons": [{"reason": str(k), "count": int(v)} for k, v in exit_reasons.most_common(8)],
        "top_reject_reasons": [{"reason": str(k), "count": int(v)} for k, v in reject_reasons.most_common(8)],
        "top_underperformers": top_underperformers,
    }


def _market_metrics_packet(
    *,
    market: str,
    status: Dict[str, Any],
    state: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
    cutoff_ts: int,
) -> Dict[str, Any]:
    mk = _market(market)
    policy = status.get("automation_policy", {}) if isinstance(status.get("automation_policy", {}), dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
    trade_quality = status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}
    gate = status.get("entry_gate_flags", {}) if isinstance(status.get("entry_gate_flags", {}), dict) else {}
    account = status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}

    account_value_usd = max(
        _f(status.get("account_value_usd", 0.0), 0.0),
        _f(account.get("total_account_value", 0.0), 0.0),
    )
    buying_power_usd = max(
        _f(status.get("buying_power_usd", 0.0), 0.0),
        _f(account.get("buying_power", 0.0), 0.0),
        _f(status.get("margin_available_usd", 0.0), 0.0),
    )
    exposure_usd = max(0.0, _f(status.get("exposure_usd", 0.0), 0.0))
    open_positions = _open_position_count(mk, status, state)
    stale_alignment = _stale_alignment_count(mk, status, state)
    recent = _recent_market_metrics(market=mk, status=status, audit_rows=audit_rows, cutoff_ts=cutoff_ts)

    return {
        "market": mk,
        "account_value_usd": round(float(max(0.0, account_value_usd)), 6),
        "buying_power_usd": round(float(max(0.0, buying_power_usd)), 6),
        "exposure_usd": round(float(max(0.0, exposure_usd)), 6),
        "open_positions": int(max(0, open_positions)),
        "stale_alignment_positions": int(max(0, stale_alignment)),
        "loss_streak": int(max(0.0, _f(gate.get("loss_streak", status.get("loss_streak", 0)), 0.0))),
        "allow_new_entries": bool(policy.get("allow_new_entries", True)),
        "policy_profile": _s(policy.get("profile"))[:32],
        "policy_mode": _s(policy.get("mode"))[:32],
        "policy_summary": _s(policy.get("summary"))[:220],
        "runtime_trust_score": round(float(max(0.0, _f(trust.get("score", 0.0), 0.0))), 6),
        "trade_quality_decision": _s(trade_quality.get("decision"))[:32].lower(),
        "trade_quality_confidence": round(float(max(0.0, _f(trade_quality.get("confidence_score", 0.0), 0.0))), 6),
        "entry_eval_total": int(max(0.0, _f(status.get("entry_eval_total", 0), 0.0))),
        "entry_eval_failed": int(max(0.0, _f(status.get("entry_eval_failed", 0), 0.0))),
        "entry_eval_top_reason": _s(status.get("entry_eval_top_reason"))[:120],
        "stale_exit_count_runtime": int(max(0.0, _f(status.get("stale_exit_count", 0), 0.0))),
        "recent": recent,
    }


def build_openai_strategy_optimizer_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    lookback_days = max(1, min(30, int(_f(cfg.get("openai_nightly_review_lookback_days", 7), 7))))
    cutoff_ts = int(now_i - (lookback_days * 86400))

    per_market: Dict[str, Dict[str, Any]] = {}
    account_value = 0.0
    buying_power = 0.0
    total_exposure = 0.0
    total_open_positions = 0
    total_stale_exit_count = 0
    total_churn_count = 0
    total_dca_add_count = 0
    total_realized_pnl = 0.0

    for mk in _MARKETS:
        status = _safe_read_json(_market_status_path(hub_dir, mk))
        state_path = _market_state_path(hub_dir, mk)
        state = _safe_read_json(state_path) if state_path else {}
        audit_rows = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=6000)
        row = _market_metrics_packet(
            market=mk,
            status=status,
            state=state,
            audit_rows=audit_rows,
            cutoff_ts=cutoff_ts,
        )
        per_market[mk] = row
        account_value = max(account_value, _f(row.get("account_value_usd", 0.0), 0.0))
        buying_power = max(buying_power, _f(row.get("buying_power_usd", 0.0), 0.0))
        total_exposure += max(0.0, _f(row.get("exposure_usd", 0.0), 0.0))
        total_open_positions += int(max(0.0, _f(row.get("open_positions", 0), 0.0)))
        recent = row.get("recent", {}) if isinstance(row.get("recent", {}), dict) else {}
        total_stale_exit_count += int(max(0.0, _f(recent.get("stale_exit_count", 0), 0.0)))
        total_churn_count += int(max(0.0, _f(recent.get("churn_count", 0), 0.0)))
        total_dca_add_count += int(max(0.0, _f(recent.get("dca_add_count", 0), 0.0)))
        total_realized_pnl += float(_f(recent.get("realized_pnl_usd", 0.0), 0.0))

    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    allocator_ctx = runtime_state.get("cross_market_opportunity", {}) if isinstance(runtime_state.get("cross_market_opportunity", {}), dict) else {}
    position_review_ctx = runtime_state.get("openai_position_review", {}) if isinstance(runtime_state.get("openai_position_review", {}), dict) else {}
    planner_ctx = runtime_state.get("openai_capital_planner", {}) if isinstance(runtime_state.get("openai_capital_planner", {}), dict) else {}
    root_cause_ctx = runtime_state.get("openai_root_cause_analysis", {}) if isinstance(runtime_state.get("openai_root_cause_analysis", {}), dict) else {}

    total_exposure_pct = ((total_exposure / max(1e-6, account_value)) * 100.0) if account_value > 0.0 else 0.0

    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "account_tier": _capital_bucket(account_value),
        "lookback_days": int(lookback_days),
        "portfolio_context": {
            "account_value_usd": round(float(max(0.0, account_value)), 6),
            "buying_power_usd": round(float(max(0.0, buying_power)), 6),
            "total_exposure_usd": round(float(max(0.0, total_exposure)), 6),
            "total_exposure_pct": round(float(max(0.0, total_exposure_pct)), 6),
            "total_open_positions": int(max(0, total_open_positions)),
            "max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("stock_max_total_exposure_pct", 0.0), 0.0))), 6),
            "forex_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("forex_max_total_exposure_pct", 0.0), 0.0))), 6),
            "market_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("stock_max_daily_loss_pct", 0.0), 0.0))), 6),
            "forex_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("forex_max_daily_loss_pct", 0.0), 0.0))), 6),
            "stock_max_open_positions": int(max(0.0, _f(cfg.get("stock_max_open_positions", 0), 0.0))),
            "forex_max_open_positions": int(max(0.0, _f(cfg.get("forex_max_open_positions", 0), 0.0))),
            "crypto_max_open_positions": int(max(0.0, _f(cfg.get("crypto_max_open_positions", 0), 0.0))),
        },
        "recent_performance_summary": {
            "realized_pnl_lookback_usd": round(float(total_realized_pnl), 6),
            "stale_exit_count_lookback": int(max(0, total_stale_exit_count)),
            "churn_count_lookback": int(max(0, total_churn_count)),
            "dca_add_count_lookback": int(max(0, total_dca_add_count)),
        },
        "per_market": per_market,
        "recent_allocator_and_ai": {
            "allocator_summary": _s(allocator_ctx.get("summary", ""))[:220],
            "allocator_best_market": _s(allocator_ctx.get("best_market", "")).lower(),
            "position_review_summary": _s(position_review_ctx.get("summary", ""))[:220],
            "position_review_status": _s(position_review_ctx.get("status", "")).lower(),
            "position_review_actions_count": int(max(0.0, _f(position_review_ctx.get("actions_count", 0), 0.0))),
            "capital_planner_summary": _s(planner_ctx.get("summary", ""))[:220],
            "capital_planner_status": _s(planner_ctx.get("status", "")).lower(),
            "capital_planner_mode": _s((planner_ctx.get("portfolio_plan", {}) if isinstance(planner_ctx.get("portfolio_plan", {}), dict) else {}).get("mode", "")).lower(),
            "root_cause_summary": _s(root_cause_ctx.get("summary", ""))[:220],
            "root_cause_status": _s(root_cause_ctx.get("status", "")).lower(),
            "root_cause_assessment": _s(root_cause_ctx.get("overall_assessment", "")).lower(),
        },
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
                val = part.get(key)
                if isinstance(val, str) and val.strip():
                    chunks.append(val.strip())
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


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return _s(value)[:120]


def _normalize_preset_assessment(src: Dict[str, Any] | None, fallback_profile: str) -> Dict[str, Any]:
    row = src if isinstance(src, dict) else {}
    profile = _s(row.get("current_profile", fallback_profile)).lower()
    if profile not in {"safe", "balanced", "aggressive", "max_growth"}:
        profile = str(fallback_profile or "balanced").strip().lower() or "balanced"
    assessment = _s(row.get("assessment")).lower()
    if assessment not in _ALLOWED_PRESET_ASSESSMENT:
        assessment = "mixed"
    return {
        "current_profile": profile,
        "assessment": assessment,
        "reason": _s(row.get("reason"))[:220],
    }


def _normalize_strategy_suggestion(src: Dict[str, Any] | None, settings: Dict[str, Any]) -> Dict[str, Any]:
    row = src if isinstance(src, dict) else {}
    key = _s(row.get("setting_key"))[:96]
    current_value = row.get("current_value")
    if current_value is None and key:
        current_value = settings.get(key)
    return {
        "setting_key": key,
        "current_value": _normalize_scalar(current_value),
        "suggested_value": _normalize_scalar(row.get("suggested_value")),
        "confidence": round(_clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(row.get("reason"))[:220],
    }


def _normalize_optimizer_payload(
    raw: Dict[str, Any],
    *,
    settings: Dict[str, Any],
    packet_profile: str,
    max_suggestions: int = 24,
) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Strategy optimizer response was not a JSON object"

    assessment = _normalize_preset_assessment(raw.get("preset_assessment", {}), packet_profile)
    suggestions_src = raw.get("strategy_suggestions", [])
    if not isinstance(suggestions_src, list):
        suggestions_src = []
    suggestions = [
        _normalize_strategy_suggestion(row, settings)
        for row in suggestions_src[: max(1, int(max_suggestions))]
        if isinstance(row, dict)
    ]

    risk_flags_src = raw.get("risk_flags", [])
    if not isinstance(risk_flags_src, list):
        risk_flags_src = []

    return {
        "summary": _s(raw.get("summary"))[:260],
        "preset_assessment": assessment,
        "strategy_suggestions": suggestions,
        "risk_flags": [_s(x)[:120] for x in risk_flags_src[:16] if _s(x)],
    }, ""


def _optimizer_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_strategy_optimizer_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_strategy_optimizer_live_enabled"), True)):
        return False, "live_disabled"
    return True, "enabled"


def request_openai_strategy_optimizer(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    optimizer_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _optimizer_enabled(cfg, mode=_s(optimizer_packet.get("mode", "")))
    model = _s(cfg.get("openai_strategy_optimizer_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_strategy_optimizer_timeout_s", 8.0), 8.0), 1.0, 30.0)
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
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
            "summary": "AI strategy optimizer unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "review": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": OPENAI_STRATEGY_OPTIMIZER_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "strategy_optimizer", "strategy_optimizer_input": optimizer_packet},
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
                "name": "strategy_optimizer",
                "strict": True,
                "schema": OPENAI_STRATEGY_OPTIMIZER_SCHEMA,
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
            "summary": "AI strategy optimizer timed out; local strategy remains unchanged.",
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
            "summary": "AI strategy optimizer request failed; local strategy remains unchanged.",
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
            "summary": "AI strategy optimizer unavailable from OpenAI API; local strategy remains unchanged.",
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
            "summary": "AI strategy optimizer returned invalid JSON; local strategy remains unchanged.",
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
            "summary": "AI strategy optimizer returned no structured output; local strategy remains unchanged.",
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
            "summary": "AI strategy optimizer output was malformed; local strategy remains unchanged.",
            "error": "Could not decode AI strategy optimizer JSON payload",
            "review": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_optimizer_payload(
        raw,
        settings=cfg,
        packet_profile=_s(optimizer_packet.get("settings_profile", "balanced")).lower() or "balanced",
        max_suggestions=24,
    )
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI strategy optimizer schema validation failed; local strategy remains unchanged.",
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


def _coerce_tuning_value(rule: Dict[str, Any], value: Any) -> Tuple[Any, bool]:
    kind = _s(rule.get("type")).lower()
    lo = _f(rule.get("min", 0.0), 0.0)
    hi = _f(rule.get("max", 0.0), 0.0)
    if kind == "int":
        try:
            val = int(round(_f(value, lo)))
        except Exception:
            return None, False
        val = int(max(int(lo), min(int(hi), int(val))))
        return int(val), True
    if kind == "float":
        try:
            val_f = _f(value, lo)
        except Exception:
            return None, False
        val_f = float(max(lo, min(hi, float(val_f))))
        precision = int(max(0, _f(rule.get("precision", 4), 4.0)))
        return round(float(val_f), precision), True
    if kind == "bool":
        return bool(_b(value, False)), True
    return None, False


def validate_low_risk_strategy_suggestions(
    *,
    settings: Dict[str, Any],
    suggestions: List[Dict[str, Any]] | None,
    min_confidence: float = 0.82,
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    rows = suggestions if isinstance(suggestions, list) else []
    validated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _s(row.get("setting_key"))
        conf = _clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0)
        reason = _s(row.get("reason"))[:220]
        if key not in _LOW_RISK_OPTIMIZER_RULES:
            skipped.append(
                {
                    "setting_key": key,
                    "reason": reason or "setting not in low-risk allowlist",
                    "skip_code": "not_allowlisted",
                    "confidence": round(conf, 6),
                }
            )
            continue
        if conf < float(min_confidence):
            skipped.append(
                {
                    "setting_key": key,
                    "reason": reason or "confidence below apply threshold",
                    "skip_code": "low_confidence",
                    "confidence": round(conf, 6),
                }
            )
            continue
        rule = _LOW_RISK_OPTIMIZER_RULES.get(key, {})
        suggested_value, ok = _coerce_tuning_value(rule, row.get("suggested_value"))
        if not ok:
            skipped.append(
                {
                    "setting_key": key,
                    "reason": reason or "suggested value could not be coerced",
                    "skip_code": "bad_value",
                    "confidence": round(conf, 6),
                }
            )
            continue
        current_value = cfg.get(key)
        validated.append(
            {
                "setting_key": key,
                "current_value": _normalize_scalar(current_value),
                "suggested_value": _normalize_scalar(suggested_value),
                "confidence": round(conf, 6),
                "reason": reason,
            }
        )

    return {
        "validated": validated,
        "skipped": skipped,
        "allowlist_size": int(len(_LOW_RISK_OPTIMIZER_RULES)),
    }


def _apply_validated_strategy_tuning(
    *,
    base_dir: str,
    settings: Dict[str, Any],
    validated_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return apply_validated_tuning_suggestions(
        base_dir=base_dir,
        settings=settings,
        validated_rows=validated_rows,
    )


def run_openai_strategy_optimizer(
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

    report_path = os.path.join(openai_dir, "strategy_optimizer.json")
    status_path = os.path.join(openai_dir, "strategy_optimizer_status.json")

    enabled, enabled_reason = _optimizer_enabled(cfg, mode=_trading_mode(cfg))
    interval_s = _clamp(_f(cfg.get("openai_strategy_optimizer_interval_s", 3600.0), 3600.0), 30.0, 86400.0)
    timeout_s = _clamp(_f(cfg.get("openai_strategy_optimizer_timeout_s", 8.0), 8.0), 1.0, 30.0)
    model = _s(cfg.get("openai_strategy_optimizer_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    auto_apply_enabled = bool(cfg.get("openai_strategy_optimizer_auto_apply_enabled", False))

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI strategy optimizer in progress." if enabled else "AI strategy optimizer disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "preset_assessment": {},
        "strategy_suggestions": [],
        "risk_flags": [],
        "strategy_suggestions_count": 0,
        "validated_suggestions_count": 0,
        "applied_tuning_count": 0,
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_strategy_optimizer_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_strategy_optimizer(settings=cfg, base_dir=base_dir, optimizer_packet=packet)
    review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}
    suggestions = review.get("strategy_suggestions", []) if isinstance(review.get("strategy_suggestions", []), list) else []

    validated_details: Dict[str, Any] = {"validated": [], "skipped": [], "allowlist_size": int(len(_LOW_RISK_OPTIMIZER_RULES))}
    apply_details: Dict[str, Any] = {
        "applied": [],
        "applied_count": 0,
        "persisted_verified_count": 0,
        "settings_path": "",
        "changed": False,
    }
    if bool(result.get("active", False)) and bool(auto_apply_enabled) and suggestions:
        validated_details = validate_low_risk_strategy_suggestions(
            settings=cfg,
            suggestions=suggestions,
            min_confidence=0.82,
        )
        validated = validated_details.get("validated", []) if isinstance(validated_details.get("validated", []), list) else []
        if validated:
            apply_details = _apply_validated_strategy_tuning(
                base_dir=base_dir,
                settings=cfg,
                validated_rows=validated,
            )

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(review.get("summary", result.get("summary", "")))[:260],
        "preset_assessment": dict(review.get("preset_assessment", {}) or {}) if isinstance(review.get("preset_assessment", {}), dict) else {},
        "strategy_suggestions": [dict(row) for row in list(suggestions or [])[:24] if isinstance(row, dict)],
        "risk_flags": [_s(x)[:120] for x in list(review.get("risk_flags", []) or [])[:16] if _s(x)],
        "validated_suggestions": [
            dict(row)
            for row in list(validated_details.get("validated", []) or [])[:24]
            if isinstance(row, dict)
        ],
        "skipped_suggestions": [
            dict(row)
            for row in list(validated_details.get("skipped", []) or [])[:24]
            if isinstance(row, dict)
        ],
        "applied_tuning": [dict(row) for row in list(apply_details.get("applied", []) or [])[:24] if isinstance(row, dict)],
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "interval_s": float(interval_s),
            "auto_apply_enabled": bool(auto_apply_enabled),
            "advisory_only": True,
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "validated_suggestions_count": int(
                len([r for r in list(validated_details.get("validated", []) or []) if isinstance(r, dict)])
            ),
            "applied_tuning_count": int(max(0.0, _f(apply_details.get("applied_count", 0), 0.0))),
            "persisted_verified_count": int(max(0.0, _f(apply_details.get("persisted_verified_count", 0), 0.0))),
            "settings_path": str(apply_details.get("settings_path", "") or ""),
        },
        "input_packet": {
            "portfolio_context": dict(packet.get("portfolio_context", {}) or {}) if isinstance(packet.get("portfolio_context", {}), dict) else {},
            "recent_performance_summary": dict(packet.get("recent_performance_summary", {}) or {}) if isinstance(packet.get("recent_performance_summary", {}), dict) else {},
            "recent_allocator_and_ai": dict(packet.get("recent_allocator_and_ai", {}) or {}) if isinstance(packet.get("recent_allocator_and_ai", {}), dict) else {},
            "per_market": dict(packet.get("per_market", {}) or {}) if isinstance(packet.get("per_market", {}), dict) else {},
        },
    }
    atomic_write_json(report_path, report)

    status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": False,
        "active": bool(result.get("active", False)),
        "status": _s(result.get("status", enabled_reason if not enabled else "ok")).lower(),
        "summary": _s(review.get("summary", result.get("summary", "")))[:260],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "advisory_only": True,
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": True,
        "preset_assessment": dict(review.get("preset_assessment", {}) or {}) if isinstance(review.get("preset_assessment", {}), dict) else {},
        "strategy_suggestions": [dict(row) for row in list(suggestions or [])[:24] if isinstance(row, dict)],
        "strategy_suggestions_count": int(len([r for r in list(suggestions or []) if isinstance(r, dict)])),
        "validated_suggestions_count": int(len([r for r in list(validated_details.get("validated", []) or []) if isinstance(r, dict)])),
        "applied_tuning_count": int(max(0.0, _f(apply_details.get("applied_count", 0), 0.0))),
        "persisted_verified_count": int(max(0.0, _f(apply_details.get("persisted_verified_count", 0), 0.0))),
        "risk_flags": [_s(x)[:120] for x in list(review.get("risk_flags", []) or [])[:16] if _s(x)],
        "applied_tuning": [dict(row) for row in list(apply_details.get("applied", []) or [])[:24] if isinstance(row, dict)],
        "skipped_suggestions": [
            dict(row)
            for row in list(validated_details.get("skipped", []) or [])[:24]
            if isinstance(row, dict)
        ],
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_strategy_optimizer_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "strategy_optimizer_status.json"))


def load_latest_openai_strategy_optimizer_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "strategy_optimizer.json"))


def load_current_settings_for_strategy_optimizer(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_strategy_optimizer") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
