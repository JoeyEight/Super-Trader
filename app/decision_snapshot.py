from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Optional

from app.runtime_logging import append_jsonl
from app.trigger_normalization import normalize_exit_trigger


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


def _decision_type(row: Dict[str, Any]) -> str:
    event = _s(row.get("event", "")).lower()
    trigger = normalize_exit_trigger(_s(row.get("tag", "")), _s(row.get("msg", "")))
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




def _market_runtime_mode(settings: Dict[str, Any] | None, market: str) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    mk = _s(market).lower()
    stage = _s(cfg.get("market_rollout_stage", "legacy")).lower() or "legacy"
    if stage == "shadow_only":
        return "shadow_only"
    if mk == "stocks":
        return "paper" if bool(cfg.get("alpaca_paper_mode", True)) else "live"
    if mk == "forex":
        return "practice" if bool(cfg.get("oanda_practice_mode", True)) else "live"
    return "live" if stage in {"live", "live_guarded", "execution_v2", "risk_caps", "scan_expanded"} else stage


def _market_live_allowed_from_existing_settings(settings: Dict[str, Any] | None, market: str) -> bool:
    cfg = settings if isinstance(settings, dict) else {}
    mk = _s(market).lower()
    stage = _s(cfg.get("market_rollout_stage", "legacy")).lower() or "legacy"
    live_stage = stage in {"live", "live_guarded", "execution_v2", "risk_caps", "scan_expanded"}
    if mk == "stocks":
        return bool(cfg.get("market_stocks_enabled", True)) and bool(cfg.get("stock_auto_trade_enabled", False)) and (not bool(cfg.get("alpaca_paper_mode", True))) and live_stage
    if mk == "forex":
        return bool(cfg.get("market_forex_enabled", True)) and bool(cfg.get("forex_auto_trade_enabled", False)) and (not bool(cfg.get("oanda_practice_mode", True))) and live_stage
    if mk == "crypto":
        return bool(cfg.get("market_crypto_enabled", True)) and live_stage
    return False


def _market_risk_fields(settings: Dict[str, Any] | None, market: str) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    mk = _s(market).lower()
    if mk == "stocks":
        return {
            "configured_notional": cfg.get("stock_trade_notional_usd"),
            "max_open_positions": cfg.get("stock_max_open_positions"),
            "max_position_usd": cfg.get("stock_max_position_usd_per_symbol"),
            "max_total_exposure_pct": cfg.get("stock_max_total_exposure_pct"),
            "max_daily_loss_usd": cfg.get("stock_max_daily_loss_usd"),
            "max_daily_loss_pct": cfg.get("stock_max_daily_loss_pct"),
        }
    if mk == "forex":
        return {
            "configured_units": cfg.get("forex_trade_units"),
            "max_open_positions": cfg.get("forex_max_open_positions"),
            "max_position_usd": cfg.get("forex_max_position_usd_per_pair"),
            "max_total_exposure_pct": cfg.get("forex_max_total_exposure_pct"),
            "max_daily_loss_usd": cfg.get("forex_max_daily_loss_usd"),
            "max_daily_loss_pct": cfg.get("forex_max_daily_loss_pct"),
        }
    return {
        "max_open_positions": cfg.get("crypto_max_open_positions"),
        "min_calib_prob_live_guarded": cfg.get("crypto_min_calib_prob_live_guarded"),
        "min_samples_live_guarded": cfg.get("crypto_min_samples_live_guarded"),
        "allocator_signal_floor": cfg.get("crypto_allocator_signal_floor"),
        "max_spread_bps": cfg.get("crypto_max_spread_bps"),
    }


def attach_market_decision_snapshot(
    row: Dict[str, Any],
    *,
    market: str,
    hub_dir: str,
    settings: Optional[Dict[str, Any]] = None,
    source_module: str,
    source_function: str,
) -> Dict[str, Any]:
    payload = dict(row or {})
    mk = _s(market).lower()
    ts = int(_f(payload.get("ts", time.time()), time.time()))
    symbol = _s(payload.get("symbol", payload.get("instrument", ""))).upper()
    if not symbol or mk not in {"stocks", "forex"}:
        return payload
    event = _s(payload.get("event", "")).lower() or "decision"
    if event not in {"entry", "entry_fail", "exit", "exit_fail", "reject"}:
        return payload
    trigger = normalize_exit_trigger(_s(payload.get("tag", "")), _s(payload.get("msg", "")))
    snapshot_id = _snapshot_id(ts, mk, symbol, event, trigger)
    decision_type = _decision_type(payload)
    selected_action = _s(payload.get("side", "")).lower()
    if not selected_action:
        selected_action = "buy" if event.startswith("entry") else "sell" if event.startswith("exit") else event
    pred_dir = _s(payload.get("predicted_direction", "")).lower() or ("up" if selected_action in {"buy", "long"} else "down" if selected_action in {"sell", "short"} else "")
    pred_trigger = _s(payload.get("predicted_exit_trigger", "")) or None
    pred_pnl = _s(payload.get("predicted_pnl_trend", "")).lower() or (pred_dir if pred_dir in {"up", "down", "flat"} else None)
    confidence = payload.get("predicted_confidence", payload.get("calib_prob"))
    trade_quality = payload.get("trade_quality", {}) if isinstance(payload.get("trade_quality", {}), dict) else {}
    opportunity = payload.get("opportunity_allocator", {}) if isinstance(payload.get("opportunity_allocator", {}), dict) else {}
    entry_cal = payload.get("entry_calibration_gate", {}) if isinstance(payload.get("entry_calibration_gate", {}), dict) else {}
    runtime_mode = _market_runtime_mode(settings, mk)
    live_allowed = _market_live_allowed_from_existing_settings(settings, mk)
    risk_fields = _market_risk_fields(settings, mk)
    snapshot = {
        "schema_version": 1,
        "decision_snapshot_id": snapshot_id,
        "timestamp": int(ts),
        "market": mk,
        "symbol": symbol,
        "decision_type": decision_type,
        "selected_action": selected_action,
        "runtime_mode": runtime_mode,
        "live_trading_allowed_from_existing_settings": bool(live_allowed),
        "selected_predictor": _s(payload.get("selected_predictor_name", payload.get("predictor_name", ""))) or "local_market_model",
        "predictor_variant": _s(payload.get("predictor_variant", "")) or "live",
        "source_used": _s(payload.get("source_used", "")) or "execution_log",
        "predicted_direction": pred_dir or None,
        "predicted_exit_trigger": pred_trigger,
        "predicted_pnl_trend": pred_pnl,
        "confidence": _f(confidence, 0.0) if confidence is not None else None,
        "direction_scores": payload.get("direction_scores") if isinstance(payload.get("direction_scores"), dict) else None,
        "trigger_scores": payload.get("trigger_scores") if isinstance(payload.get("trigger_scores"), dict) else None,
        "pnl_quality_score": payload.get("pnl_quality_score", trade_quality.get("confidence_score")),
        "trade_quality_score": payload.get("trade_quality_score", trade_quality.get("confidence_score")),
        "entry_reason": _s(payload.get("entry_reason", payload.get("msg", ""))) or None,
        "signal_side": _s(payload.get("signal_side", payload.get("side", ""))).lower() or None,
        "signal_margin": payload.get("signal_margin"),
        "model_quality_blockers_visible": payload.get("model_quality_blockers_visible"),
        "full_promotion_eligible": payload.get("full_promotion_eligible"),
        "raw_rule_reason": _s(payload.get("msg", "")) or _s(payload.get("tag", "")),
        "normalized_trigger": trigger,
        "strategy_score": payload.get("score"),
        "required_score": payload.get("required_score"),
        "trigger_reliability": payload.get("trigger_reliability", entry_cal.get("trigger_reliability")),
        "source_module": _s(source_module),
        "source_function": _s(source_function),
        "risk_settings": risk_fields,
        "missing_fields": [],
    }
    missing = []
    for key in ("predicted_exit_trigger", "direction_scores", "trigger_scores", "model_quality_blockers_visible", "full_promotion_eligible"):
        if snapshot.get(key) is None:
            missing.append(key)
    snapshot["missing_fields"] = missing
    append_jsonl(_snapshot_path(hub_dir, mk), snapshot, async_mode=True)
    payload.setdefault("decision_snapshot_id", snapshot_id)
    payload["event_decision_snapshot_id"] = snapshot_id
    payload.setdefault("raw_rule_reason", snapshot["raw_rule_reason"])
    payload.setdefault("normalized_trigger", trigger)
    return payload
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
    trigger = normalize_exit_trigger(_s(payload.get("tag", "")), _s(payload.get("msg", "")))
    event = _s(payload.get("event", "")).lower() or "decision"
    snapshot_id = _snapshot_id(ts, market, symbol, event, trigger)
    decision_type = _decision_type(payload)
    entry_features = payload.get("entry_features", {}) if isinstance(payload.get("entry_features", {}), dict) else {}
    predicted_direction = _s(payload.get("predicted_direction", "")).lower()
    if not predicted_direction:
        selected_action = _s(payload.get("side", "")).lower()
        predicted_direction = "up" if selected_action in {"buy", "long"} else ("down" if selected_action in {"sell", "short"} else "")
    predicted_exit_trigger = _s(payload.get("predicted_exit_trigger", "")) or None
    predicted_pnl_trend = _s(payload.get("predicted_pnl_trend", "")).lower() or (predicted_direction if predicted_direction in {"up", "down", "flat"} else None)
    confidence = payload.get("predicted_confidence", payload.get("calib_prob", entry_features.get("calib_prob")))
    selected_predictor = _s(payload.get("selected_predictor_name", payload.get("predictor_name", ""))) or "crypto_live_model"
    predictor_variant = _s(payload.get("predictor_variant", "")) or "live"
    source_used = _s(payload.get("source_used", "")) or "execution_log"
    snapshot = {
        "schema_version": 1,
        "decision_snapshot_id": snapshot_id,
        "timestamp": int(ts),
        "market": market,
        "symbol": symbol,
        "decision_type": decision_type,
        "selected_action": "buy" if event == "entry" else "sell" if event == "exit" else event,
        "selected_predictor": selected_predictor,
        "predictor_variant": predictor_variant,
        "source_used": source_used,
        "predicted_direction": predicted_direction or None,
        "predicted_exit_trigger": predicted_exit_trigger,
        "predicted_pnl_trend": predicted_pnl_trend,
        "confidence": _f(confidence, 0.0) if confidence is not None else None,
        "direction_scores": payload.get("direction_scores") if isinstance(payload.get("direction_scores"), dict) else None,
        "trigger_scores": payload.get("trigger_scores") if isinstance(payload.get("trigger_scores"), dict) else None,
        "trade_quality_score": payload.get("trade_quality_score"),
        "pnl_quality_score": payload.get("pnl_quality_score"),
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
