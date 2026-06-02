from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict

from app.runtime_logging import append_jsonl


def _s(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _norm_trigger(tag: str, msg: str = "") -> str:
    txt = f"{_s(tag)} {_s(msg)}".upper()
    if not txt:
        return "Unknown"
    if "MANUAL" in txt:
        return "Manual"
    if "TRAIL" in txt:
        return "Trailing"
    if "STALE" in txt or "MISALIGN" in txt or "ALIGNMENT" in txt:
        return "Stale Alignment"
    if "BLOCK" in txt:
        return "Blocked"
    if ("AI" in txt) and ("EXIT" in txt or "CLOSE" in txt):
        return "AI Exit"
    if ("RISK" in txt) or ("STOP" in txt):
        return "Risk Cut"
    if ("TAKE" in txt) or ("PROFIT" in txt):
        return "Take Profit"
    return "Unknown"


def _decision_type(row: Dict[str, Any]) -> str:
    event = _s(row.get("event", "")).lower()
    trigger = _norm_trigger(_s(row.get("tag", "")), _s(row.get("msg", "")))
    if event == "entry":
        return "entry"
    if event in {"entry_fail", "reject"}:
        return "reject"
    if event == "exit":
        if trigger == "Manual":
            return "manual_close"
        if trigger == "Trailing":
            return "trailing_exit"
        if trigger == "Stale Alignment":
            return "stale_exit"
        if trigger == "Risk Cut":
            return "risk_cut"
        if trigger == "Take Profit":
            return "take_profit"
        if trigger == "AI Exit":
            return "ai_exit"
        return "exit"
    return event or "decision"


def _snapshot_id(ts: int, market: str, symbol: str, event: str, trigger: str) -> str:
    raw = f"{int(ts)}|{_s(market).lower()}|{_s(symbol).upper()}|{_s(event).lower()}|{_s(trigger)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _snapshot_path(hub_dir: str, market: str) -> str:
    return os.path.join(hub_dir, _s(market).lower(), "decision_snapshots.jsonl")


def attach_crypto_decision_snapshot(
    row: Dict[str, Any],
    *,
    hub_dir: str,
    source_module: str,
    source_function: str,
) -> Dict[str, Any]:
    payload = dict(row or {})
    market = "crypto"
    ts = int(_f(payload.get("ts", time.time()), time.time()))
    symbol = _s(payload.get("symbol", "")).upper()
    if not symbol:
        return payload
    trigger = _norm_trigger(_s(payload.get("tag", "")), _s(payload.get("msg", "")))
    event = _s(payload.get("event", "")).lower() or "decision"
    snapshot_id = _snapshot_id(ts, market, symbol, event, trigger)
    decision_type = _decision_type(payload)
    entry_features = payload.get("entry_features", {}) if isinstance(payload.get("entry_features", {}), dict) else {}
    snapshot = {
        "schema_version": 1,
        "decision_snapshot_id": snapshot_id,
        "timestamp": int(ts),
        "market": market,
        "symbol": symbol,
        "decision_type": decision_type,
        "selected_action": "buy" if event == "entry" else "sell" if event == "exit" else event,
        "raw_rule_reason": _s(payload.get("msg", "")) or _s(payload.get("tag", "")),
        "normalized_trigger": trigger,
        "stale_score": payload.get("stale_score"),
        "trailing_score": payload.get("trailing_score"),
        "manual_score": payload.get("manual_score"),
        "manual_flag": bool(trigger == "Manual"),
        "ai_confidence": payload.get("calib_prob", entry_features.get("calib_prob")),
        "strategy_score": payload.get("score", entry_features.get("signal_score")),
        "required_score": payload.get("required_score", entry_features.get("required_score")),
        "trigger_reliability": entry_features.get("trigger_reliability"),
        "trend_signal_count": entry_features.get("buy_count"),
        "counter_signal_count": entry_features.get("sell_count"),
        "signal_gate_mode": entry_features.get("signal_gate_mode"),
        "entry_alignment_mode": entry_features.get("entry_alignment_mode"),
        "policy_mode": entry_features.get("policy_mode"),
        "profile": entry_features.get("profile"),
        "candle_timeframe": "1h",
        "last_price": payload.get("price"),
        "entry_price": payload.get("avg_cost_basis"),
        "unrealized_pnl": payload.get("unrealized_pnl_usd"),
        "hold_time_seconds": payload.get("hold_s"),
        "rejection_reason": _s(payload.get("msg", "")) if event in {"entry_fail", "reject"} else "",
        "source_module": _s(source_module),
        "source_function": _s(source_function),
    }
    append_jsonl(_snapshot_path(hub_dir, market), snapshot, async_mode=True)
    payload["decision_snapshot_id"] = snapshot_id
    payload["raw_rule_reason"] = snapshot["raw_rule_reason"]
    payload["normalized_trigger"] = trigger
    return payload
