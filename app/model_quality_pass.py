from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.confidence_calibration import build_confidence_calibration_payload
from app.crypto_artifacts import discover_crypto_trained_artifacts, load_crypto_artifact_features
import app.crypto_historical_replay as crypto_historical_replay
from app.crypto_historical_replay import build_crypto_historical_strategy_replay
from app.decision_snapshot import attach_crypto_decision_snapshot, attach_market_decision_snapshot
from app.credential_utils import (
    get_alpaca_creds,
    get_oanda_creds,
    get_robinhood_creds_from_env,
    get_robinhood_creds_from_files,
    get_twelvedata_api_key,
)
from app.regime_classifier import build_all_market_regimes
from app.settings_utils import sanitize_settings
from app.shadow_scorecard import build_shadow_scorecards
from app.trigger_normalization import normalize_exit_trigger
from app.walkforward_report import build_walkforward_report
from brokers.broker_alpaca import AlpacaBrokerClient
from brokers.broker_twelvedata import TwelveDataClient


CRYPTO_SNAPSHOT_FEATURE_FIELDS = (
    "decision_type",
    "selected_action",
    "raw_rule_reason",
    "normalized_trigger",
    "stale_score",
    "trailing_score",
    "manual_score",
    "manual_flag",
    "ai_confidence",
    "strategy_score",
    "required_score",
    "trigger_reliability",
    "trend_signal_count",
    "counter_signal_count",
    "signal_gate_mode",
    "entry_alignment_mode",
    "policy_mode",
    "profile",
    "candle_timeframe",
    "last_price",
    "entry_price",
    "unrealized_pnl",
    "hold_time_seconds",
    "rejection_reason",
    "source_module",
    "source_function",
)
CRYPTO_SNAPSHOT_JOIN_PREFIX = "snapshot_"
CRYPTO_ARTIFACT_FEATURE_PREFIX = "artifact_"
COMPLETED_LIVE_REQUIRED_FIELDS = (
    "decision_snapshot_id",
    "market",
    "symbol_or_instrument",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "side",
    "qty",
    "predicted_direction",
    "predicted_exit_trigger",
    "predicted_pnl_trend",
    "confidence",
    "actual_direction",
    "actual_exit_trigger",
    "actual_pnl_trend",
    "pnl",
    "pnl_usd",
    "pnl_pct",
    "direction_correct",
    "trigger_correct",
    "pnl_trend_correct",
)


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


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _env_flag(name: str) -> bool:
    return _s(os.environ.get(name, "")).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    return _s(os.environ.get(name, default)) or default


def _bool_setting(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def _market_live_allowed_from_existing_settings(cfg: Dict[str, Any], market: str) -> bool:
    mk = _s(market).lower()
    stage = _s(cfg.get("market_rollout_stage", "legacy")).lower() or "legacy"
    live_stage = stage in {"live", "live_guarded", "execution_v2", "risk_caps", "scan_expanded"}
    if mk == "stocks":
        return (
            _bool_setting(cfg, "market_stocks_enabled", True)
            and _bool_setting(cfg, "stock_auto_trade_enabled", False)
            and (not _bool_setting(cfg, "alpaca_paper_mode", True))
            and live_stage
        )
    if mk == "forex":
        return (
            _bool_setting(cfg, "market_forex_enabled", True)
            and _bool_setting(cfg, "forex_auto_trade_enabled", False)
            and (not _bool_setting(cfg, "oanda_practice_mode", True))
            and live_stage
        )
    if mk == "crypto":
        return _bool_setting(cfg, "market_crypto_enabled", True) and live_stage
    return False


def _market_existing_live_enabled(cfg: Dict[str, Any], market: str) -> bool:
    return _market_live_allowed_from_existing_settings(cfg, market)


def _existing_market_risk_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "crypto": {
            "max_open_positions": cfg.get("crypto_max_open_positions"),
            "min_calib_prob_live_guarded": cfg.get("crypto_min_calib_prob_live_guarded"),
            "min_samples_live_guarded": cfg.get("crypto_min_samples_live_guarded"),
            "allocator_signal_floor": cfg.get("crypto_allocator_signal_floor"),
            "max_spread_bps": cfg.get("crypto_max_spread_bps"),
        },
        "stocks": {
            "trade_notional_usd": cfg.get("stock_trade_notional_usd"),
            "max_open_positions": cfg.get("stock_max_open_positions"),
            "max_position_usd_per_symbol": cfg.get("stock_max_position_usd_per_symbol"),
            "max_total_exposure_pct": cfg.get("stock_max_total_exposure_pct"),
            "max_daily_loss_usd": cfg.get("stock_max_daily_loss_usd"),
            "max_daily_loss_pct": cfg.get("stock_max_daily_loss_pct"),
        },
        "forex": {
            "trade_units": cfg.get("forex_trade_units"),
            "max_open_positions": cfg.get("forex_max_open_positions"),
            "max_position_usd_per_pair": cfg.get("forex_max_position_usd_per_pair"),
            "max_total_exposure_pct": cfg.get("forex_max_total_exposure_pct"),
            "max_daily_loss_usd": cfg.get("forex_max_daily_loss_usd"),
            "max_daily_loss_pct": cfg.get("forex_max_daily_loss_pct"),
        },
    }


def _existing_runtime_settings_diagnostics(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    robinhood_key, robinhood_secret = get_robinhood_creds_from_env()
    if (not robinhood_key) or (not robinhood_secret):
        fk, fs = get_robinhood_creds_from_files(base_dir)
        robinhood_key = robinhood_key or fk
        robinhood_secret = robinhood_secret or fs
    alpaca_key, alpaca_secret = get_alpaca_creds(cfg, base_dir=base_dir)
    oanda_account, oanda_token = get_oanda_creds(cfg, base_dir=base_dir)
    return {
        "existing_crypto_live_enabled": bool(_market_existing_live_enabled(cfg, "crypto")),
        "existing_stocks_live_enabled": bool(_market_existing_live_enabled(cfg, "stocks")),
        "existing_forex_live_enabled": bool(_market_existing_live_enabled(cfg, "forex")),
        "existing_position_sizing_mode": "configured_market_specific",
        "existing_market_risk_settings": _existing_market_risk_settings(cfg),
        "existing_broker_modes": {
            "crypto_runtime_stage": _s(cfg.get("market_rollout_stage", "")),
            "stocks_paper_mode": bool(cfg.get("alpaca_paper_mode", True)),
            "forex_practice_mode": bool(cfg.get("oanda_practice_mode", True)),
        },
        "existing_market_enable_flags": {
            "crypto": bool(cfg.get("market_crypto_enabled", True)),
            "stocks": bool(cfg.get("market_stocks_enabled", True)),
            "forex": bool(cfg.get("market_forex_enabled", True)),
        },
        "existing_market_auto_trade_flags": {
            "stocks": bool(cfg.get("stock_auto_trade_enabled", False)),
            "forex": bool(cfg.get("forex_auto_trade_enabled", False)),
        },
        "existing_credential_availability": {
            "crypto": bool(robinhood_key and robinhood_secret),
            "stocks": bool(alpaca_key and alpaca_secret),
            "forex": bool(oanda_account and oanda_token),
        },
        "settings_used_as_is": True,
    }


def _safe_read_jsonl(path: str, max_lines: int = 800000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= int(max_lines):
                    break
                txt = _s(line)
                if not txt:
                    continue
                try:
                    row = json.loads(txt)
                except Exception:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except Exception:
        return []
    return out


def _ts_from_row(row: Dict[str, Any]) -> int:
    return int(_f(row.get("ts", row.get("timestamp", 0.0)), 0.0))


def _join_text_fields(row: Dict[str, Any], fields: Iterable[str]) -> str:
    vals: List[str] = []
    for field in fields:
        txt = _s(row.get(field, ""))
        if txt:
            vals.append(txt)
    return " | ".join(vals)


def _explicit_trigger_label(row: Dict[str, Any]) -> str:
    reason_fields = (
        "tag",
        "source",
        "exit_reason",
        "close_reason",
        "exit_trigger",
        "trigger",
        "reason",
        "decision_reason",
        "signal_reason",
        "status",
        "msg",
        "message",
        "note",
        "why",
        "normalized_trigger",
        "raw_rule_reason",
    )
    return normalize_exit_trigger(_join_text_fields(row, reason_fields))


def _norm_exit_trigger(*texts: Any) -> str:
    return normalize_exit_trigger(*texts)


def _snapshot_id_from_event(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return ""
    direct = _s(event.get("decision_snapshot_id", ""))
    if direct:
        return direct
    raw = event.get("raw", {})
    if isinstance(raw, dict):
        return _s(raw.get("decision_snapshot_id", ""))
    return ""


def _extract_snapshot_feature_fields(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(snapshot, dict):
        return out
    for field in CRYPTO_SNAPSHOT_FEATURE_FIELDS:
        value = snapshot.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            if not _s(value):
                continue
            out[f"{CRYPTO_SNAPSHOT_JOIN_PREFIX}{field}"] = _s(value)
        else:
            out[f"{CRYPTO_SNAPSHOT_JOIN_PREFIX}{field}"] = value
    return out


def _join_crypto_decision_snapshots(hub_dir: str, events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = os.path.join(hub_dir, "crypto", "decision_snapshots.jsonl")
    rows = _safe_read_jsonl(path, max_lines=200000)
    if not rows:
        return list(events or []), {
            "decision_snapshots_found": 0,
            "decision_snapshots_joined": 0,
            "snapshot_join_rate_pct": 0.0,
            "snapshot_features_joined_count": 0,
            "snapshot_feature_names": [],
            "missing_snapshot_count": int(len(list(events or []))),
            "snapshot_schema_version": 0,
            "top_snapshot_trigger_reasons": {},
            "path": path,
        }
    schema_version = int(_f(rows[-1].get("schema_version", 0), 0.0))
    by_id = {_s(row.get("decision_snapshot_id", "")): row for row in rows if _s(row.get("decision_snapshot_id", ""))}
    joined = 0
    feature_join_count = 0
    feature_names: Dict[str, int] = {}
    prepared: List[Dict[str, Any]] = []
    for event in list(events or []):
        item = dict(event)
        sid = _snapshot_id_from_event(item)
        snapshot = by_id.get(sid)
        if snapshot:
            joined += 1
            snapshot_fields = _extract_snapshot_feature_fields(snapshot)
            for key, value in snapshot_fields.items():
                item[key] = value
            feature_join_count += len(snapshot_fields)
            for key in snapshot_fields:
                feature_names[key] = int(feature_names.get(key, 0) + 1)
        prepared.append(item)
    reason_counts: Dict[str, int] = {}
    for row in rows:
        reason = normalize_exit_trigger(row.get("normalized_trigger", ""), row.get("raw_rule_reason", "")) or "Unknown"
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
    total = len(prepared)
    diagnostics = {
        "decision_snapshots_found": int(len(rows)),
        "decision_snapshots_joined": int(joined),
        "snapshot_join_rate_pct": round(100.0 * float(joined) / float(max(1, total)), 4),
        "snapshot_features_joined_count": int(feature_join_count),
        "snapshot_feature_names": sorted(feature_names.keys()),
        "missing_snapshot_count": int(max(0, total - joined)),
        "snapshot_schema_version": int(schema_version),
        "top_snapshot_trigger_reasons": _top_counter(reason_counts, limit=12),
        "path": path,
    }
    return prepared, diagnostics


def _feature_source_used_for_crypto_row(row: Dict[str, Any]) -> str:
    if any(
        key.startswith(CRYPTO_SNAPSHOT_JOIN_PREFIX)
        or key.startswith(f"entry_{CRYPTO_SNAPSHOT_JOIN_PREFIX}")
        or key.startswith(f"exit_{CRYPTO_SNAPSHOT_JOIN_PREFIX}")
        for key in row.keys()
    ):
        return "decision_snapshot"
    if any(key.startswith(CRYPTO_ARTIFACT_FEATURE_PREFIX) for key in row.keys()):
        return "trained_artifact"
    return "closed_trade_only"


def _crypto_historical_replay_sufficient(payload: Dict[str, Any]) -> Tuple[bool, str]:
    rows = payload.get("rows", []) if isinstance(payload.get("rows", []), list) else []
    diag = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics", {}), dict) else {}
    symbols = list(diag.get("historical_strategy_replay_symbols", []) or [])
    if len(rows) < 40:
        return False, f"historical_strategy_replay_rows_insufficient({len(rows)}<40)"
    if len(symbols) < 3:
        return False, f"historical_strategy_replay_symbols_insufficient({len(symbols)}<3)"
    return True, ""


def _prepare_forex_audit_rows(rows: List[Dict[str, Any]], window_s: int = 120) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = [dict(row) for row in list(rows or []) if isinstance(row, dict)]
    exit_rows = [row for row in prepared if _s(row.get("event", "")).lower() == "exit"]
    stale_contexts: List[Dict[str, Any]] = []
    explicit_reason_count = 0
    inferred_stale_count = 0
    generic_close_count = 0
    unknown_before = 0
    unknown_after = 0
    matched_exits = 0

    for row in prepared:
        event = _s(row.get("event", "")).lower()
        msg_txt = _join_text_fields(row, ("msg", "message", "reason", "note", "source")).lower()
        if event == "shadow_live_divergence" and "stale forex position" in msg_txt:
            count = 1
            for tok in msg_txt.split():
                if tok.isdigit():
                    count = max(1, int(tok))
                    break
            stale_contexts.append({"ts": _ts_from_row(row), "remaining": count, "row": row})

    for row in exit_rows:
        label_before = _explicit_trigger_label(row)
        if label_before == "Unknown":
            unknown_before += 1
        else:
            explicit_reason_count += 1
            row.setdefault("tag", label_before)
            row.setdefault("event_exit_tag", f"explicit:{_s(row.get('source', '')) or _s(row.get('tag', ''))}")
            continue
        generic_close_count += 1
        ts = _ts_from_row(row)
        best_idx = -1
        best_dist = 10**12
        for idx, ctx in enumerate(stale_contexts):
            if int(ctx.get("remaining", 0) or 0) <= 0:
                continue
            dist = abs(int(ctx.get("ts", 0) or 0) - ts)
            if dist <= int(window_s) and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx >= 0:
            stale_contexts[best_idx]["remaining"] = int(stale_contexts[best_idx].get("remaining", 0) or 0) - 1
            row["tag"] = "policy_stale_exit"
            row["event_exit_tag"] = "inferred: nearby stale forex exit context"
            row["inferred_trigger_reason"] = "nearby_stale_context"
            row["inferred_trigger_distance_s"] = int(best_dist)
            inferred_stale_count += 1
            matched_exits += 1

    for row in exit_rows:
        if _explicit_trigger_label(row) == "Unknown":
            unknown_after += 1

    diagnostics = {
        "matching_window_seconds": int(window_s),
        "forex_exit_rows": int(len(exit_rows)),
        "forex_stale_context_rows": int(len(stale_contexts)),
        "stale_context_matched_exits": int(matched_exits),
        "unmatched_generic_close_count": int(max(0, generic_close_count - matched_exits)),
        "explicit_reason_count": int(explicit_reason_count),
        "inferred_stale_count": int(inferred_stale_count),
        "unknown_trigger_count_before": int(unknown_before),
        "unknown_trigger_count_after": int(unknown_after),
    }
    return prepared, diagnostics


def _crypto_snapshot_diagnostics(hub_dir: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    _, diagnostics = _join_crypto_decision_snapshots(hub_dir, events)
    return diagnostics


def _event_symbol(row: Dict[str, Any], market: str) -> str:
    if str(market).lower() == "forex":
        return _s(row.get("instrument", "")).upper()
    return _s(row.get("symbol", "")).upper()


def _event_qty(row: Dict[str, Any]) -> float:
    qty = _f(row.get("qty", 0.0), 0.0)
    if abs(qty) > 0.0:
        return abs(qty)
    units = _f(row.get("units", 0.0), 0.0)
    if abs(units) > 0.0:
        return abs(units)
    notional = _f(row.get("notional", 0.0), 0.0)
    price = _f(row.get("price", 0.0), 0.0)
    if notional > 0.0 and price > 0.0:
        return abs(notional / price)
    return 0.0


def _event_price(row: Dict[str, Any]) -> float:
    return _f(row.get("price", 0.0), 0.0)


def _is_trade_event(row: Dict[str, Any]) -> bool:
    ev = _s(row.get("event", "")).lower()
    return ev in {"entry", "exit"}


def _as_trade_event(row: Dict[str, Any], market: str) -> Dict[str, Any] | None:
    if not isinstance(row, dict) or (not _is_trade_event(row)):
        return None
    ts = int(_f(row.get("ts", 0.0), 0.0))
    if ts <= 0:
        return None
    symbol = _event_symbol(row, market)
    if not symbol:
        return None
    qty = _event_qty(row)
    price = _event_price(row)
    if qty <= 0.0 or price <= 0.0:
        return None
    return {
        "ts": int(ts),
        "event": _s(row.get("event", "")).lower(),
        "symbol": symbol,
        "qty": float(qty),
        "price": float(price),
        "side": _s(row.get("side", "")).lower(),
        "order_id": _s(row.get("order_id", "")),
        "ok": bool(row.get("ok", False)),
        "pnl_usd": _f(row.get("pnl_usd", row.get("realized_pnl_usd", 0.0)), 0.0),
        "pnl_pct": _f(row.get("pnl_pct", 0.0), 0.0),
        "tag": _s(row.get("tag", "")),
        "hold_s": int(_f(row.get("hold_s", 0.0), 0.0)),
        "avg_entry_price": _f(row.get("avg_entry_price", 0.0), 0.0),
        "raw": dict(row),
    }


def load_market_trade_events(hub_dir: str, market: str) -> Dict[str, Any]:
    m = _s(market).lower()
    if m not in {"crypto", "stocks", "forex"}:
        return {"market": m, "state": "ERROR", "msg": "unsupported market", "events": []}

    if m == "crypto":
        paths = [os.path.join(hub_dir, "crypto", "execution_audit.jsonl"), os.path.join(hub_dir, "trade_history.jsonl")]
    else:
        paths = [os.path.join(hub_dir, m, "execution_audit.jsonl")]
    rows: List[Dict[str, Any]] = []
    source = ""
    diagnostics: Dict[str, Any] = {}
    for path in paths:
        src_rows = _safe_read_jsonl(path)
        if not src_rows:
            continue
        if m == "forex":
            src_rows, diagnostics = _prepare_forex_audit_rows(src_rows)
        if m == "crypto" and path.endswith("trade_history.jsonl"):
            # trade_history rows use side rather than event.
            converted: List[Dict[str, Any]] = []
            for row in src_rows:
                side = _s(row.get("side", "")).lower()
                if side not in {"buy", "sell"}:
                    continue
                x = dict(row)
                x["event"] = "entry" if side == "buy" else "exit"
                x["pnl_usd"] = _f(row.get("realized_profit_usd", 0.0), 0.0)
                converted.append(x)
            src_rows = converted
        src_events = []
        for row in src_rows:
            ev = _as_trade_event(row, m)
            if isinstance(ev, dict):
                src_events.append(ev)
        if len(src_events) > len(rows):
            rows = src_events
            source = path

    rows.sort(key=lambda r: int(r.get("ts", 0)))
    if m == "crypto":
        rows, snap_diag = _join_crypto_decision_snapshots(hub_dir, rows)
        diagnostics = dict(diagnostics)
        diagnostics["decision_snapshot_diagnostics"] = snap_diag
    return {
        "market": m,
        "state": "READY" if rows else "NO_DATA",
        "source": source,
        "events": rows,
        "diagnostics": diagnostics,
    }


def _derive_entry_price_from_exit(row: Dict[str, Any], market: str) -> float:
    avg = _f(row.get("avg_entry_price", 0.0), 0.0)
    if avg > 0.0:
        return float(avg)
    exit_px = _f(row.get("price", 0.0), 0.0)
    pnl_pct = _f(row.get("pnl_pct", 0.0), 0.0) / 100.0
    if exit_px <= 0.0:
        return 0.0
    side = _s(row.get("side", "")).lower()
    if _s(market).lower() == "forex":
        # Forex side typically reflects the closed position direction.
        if side == "long":
            den = 1.0 + pnl_pct
        else:
            den = 1.0 - pnl_pct
    else:
        den = 1.0 + pnl_pct
    if abs(den) <= 1e-9:
        return 0.0
    px = exit_px / den
    return float(px) if px > 0.0 else 0.0


def _predicted_direction_from_action(action: str, market: str) -> str:
    act = _s(action).lower()
    mk = _s(market).lower()
    if act in {"buy", "long"}:
        return "up"
    if act in {"sell", "short"}:
        return "down" if mk == "forex" else "up"
    return ""


def _extract_live_prediction_field(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    raw = row.get("raw", {})
    if isinstance(raw, dict):
        for key in keys:
            if key in raw and raw.get(key) not in (None, ""):
                return raw.get(key)
    return None


def _completed_live_identity(row: Dict[str, Any], market: str) -> Tuple[str, str]:
    mk = _s(market).lower()
    ident = _s(row.get("symbol", row.get("instrument", row.get("pair", "")))).upper()
    if not ident:
        ident = _s(row.get("asset_code", "")).upper()
    field_name = "instrument" if mk == "forex" else "symbol"
    return ident, field_name


def _completed_live_required_field_values(row: Dict[str, Any], market: str) -> Dict[str, Any]:
    ident, _ = _completed_live_identity(row, market)
    return {
        "decision_snapshot_id": row.get("decision_snapshot_id"),
        "market": _s(row.get("market", market)).lower() or _s(market).lower(),
        "symbol_or_instrument": ident or None,
        "entry_ts": row.get("entry_ts"),
        "exit_ts": row.get("exit_ts"),
        "entry_price": row.get("entry_price"),
        "exit_price": row.get("exit_price"),
        "side": row.get("side"),
        "qty": row.get("qty"),
        "predicted_direction": row.get("predicted_direction"),
        "predicted_exit_trigger": row.get("predicted_exit_trigger"),
        "predicted_pnl_trend": row.get("predicted_pnl_trend"),
        "confidence": row.get("confidence", row.get("predicted_confidence")),
        "actual_direction": row.get("actual_direction"),
        "actual_exit_trigger": row.get("actual_exit_trigger"),
        "actual_pnl_trend": row.get("actual_pnl_trend"),
        "pnl": row.get("pnl"),
        "pnl_usd": row.get("pnl_usd"),
        "pnl_pct": row.get("pnl_pct"),
        "direction_correct": row.get("direction_correct"),
        "trigger_correct": row.get("trigger_correct"),
        "pnl_trend_correct": row.get("pnl_trend_correct"),
    }


def _missing_completed_live_fields(row: Dict[str, Any], market: str) -> List[str]:
    values = _completed_live_required_field_values(row, market)
    missing: List[str] = []
    for key, value in values.items():
        if value in (None, ""):
            missing.append(key)
    return missing


def _completed_live_decision_rows(
    *,
    market: str,
    events: List[Dict[str, Any]],
    closed_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mk = _s(market).lower()
    rows_out: List[Dict[str, Any]] = []
    by_market: Dict[str, int] = {mk: 0}
    by_predictor: Dict[str, int] = {}
    missing_fields_by_market: Dict[str, Dict[str, int]] = {mk: {}}
    found = 0
    completed = 0
    post_model_completed = 0
    for row in list(closed_rows or []):
        if not isinstance(row, dict):
            continue
        pred_dir = (
            _s(row.get("predicted_direction", ""))
            or _s(row.get("entry_snapshot_predicted_direction", ""))
            or _predicted_direction_from_action(_s(row.get("entry_snapshot_selected_action", "")), mk)
        )
        pred_trigger = (
            _s(row.get("predicted_exit_trigger", ""))
            or _s(row.get("entry_snapshot_predicted_exit_trigger", ""))
            or _s(row.get("entry_snapshot_normalized_trigger", ""))
        )
        pred_pnl = (
            _s(row.get("predicted_pnl_trend", ""))
            or _s(row.get("entry_snapshot_predicted_pnl_trend", ""))
            or (pred_dir if pred_dir in {"up", "down", "flat"} else "")
        )
        pred_conf = _extract_live_prediction_field(
            row,
            (
                "predicted_confidence",
                "entry_snapshot_predicted_confidence",
                "entry_snapshot_ai_confidence",
                "entry_snapshot_strategy_score",
            ),
        )
        predictor_name = _s(
            _extract_live_prediction_field(
                row,
                (
                    "predictor_name",
                    "selected_predictor_name",
                    "entry_snapshot_selected_predictor",
                    "entry_snapshot_policy_mode",
                    "entry_snapshot_profile",
                ),
            )
        ) or "live_snapshot"
        predictor_variant = _s(
            _extract_live_prediction_field(
                row,
                (
                    "predictor_variant",
                    "entry_snapshot_predictor_variant",
                    "entry_snapshot_entry_alignment_mode",
                    "entry_snapshot_signal_gate_mode",
                ),
            )
        ) or "live"
        trigger_scores = _extract_live_prediction_field(row, ("trigger_scores", "entry_snapshot_trigger_scores"))
        direction_scores = _extract_live_prediction_field(row, ("direction_scores", "entry_snapshot_direction_scores"))
        decision_snapshot_id = _s(row.get("decision_snapshot_id", "")) or _s(row.get("entry_decision_snapshot_id", ""))
        item = dict(row)
        ident, ident_field = _completed_live_identity(item, mk)
        item["decision_snapshot_id"] = decision_snapshot_id or None
        item["trade_id"] = _s(row.get("trade_id", row.get("position_id", row.get("order_id", "")))) or None
        item["market"] = mk
        item[ident_field] = ident or None
        item["symbol"] = ident or None
        if mk == "forex":
            item["instrument"] = ident or None
        item["predicted_direction"] = pred_dir or None
        item["predicted_exit_trigger"] = pred_trigger or None
        item["predicted_pnl_trend"] = pred_pnl or None
        item["predicted_confidence"] = round(_f(pred_conf, 0.0), 6) if pred_conf is not None else None
        item["confidence"] = item.get("predicted_confidence")
        if isinstance(trigger_scores, dict):
            item["trigger_scores"] = dict(trigger_scores)
        if isinstance(direction_scores, dict):
            item["direction_scores"] = dict(direction_scores)
        item["selected_predictor_name"] = predictor_name
        item["predictor_mode"] = predictor_name
        item["predictor_variant"] = predictor_variant
        item["selected_predictor"] = predictor_name
        item["source_type"] = "completed_live_decision_snapshot"
        item["source"] = "live_execution"
        item["source_used"] = _s(row.get("entry_snapshot_source_used", row.get("source_used", ""))) or None
        item["trade_quality_score"] = _extract_live_prediction_field(row, ("trade_quality_score", "entry_snapshot_trade_quality_score"))
        item["pnl_quality_score"] = _extract_live_prediction_field(row, ("pnl_quality_score", "entry_snapshot_pnl_quality_score"))
        item["raw_rule_reason"] = _s(row.get("entry_snapshot_raw_rule_reason", "")) or _s(row.get("raw_rule_reason", ""))
        item["event_exit_tag"] = _s(row.get("event_exit_tag", ""))
        item["actual_direction"] = _trade_direction(item)
        item["actual_pnl_trend"] = item["actual_direction"]
        item["actual_exit_trigger"] = _norm_exit_trigger(
            _s(row.get("actual_exit_trigger", "")),
            _s(row.get("normalized_exit_trigger", "")),
            _s(row.get("event_exit_tag", "")),
            _s(row.get("raw_rule_reason", "")),
        ) or "Unknown"
        item["exit_rule_reason"] = _s(row.get("raw_rule_reason", row.get("event_exit_tag", ""))) or None
        item["normalized_exit_trigger"] = item["actual_exit_trigger"]
        item["hold_time"] = row.get("hold_hours", row.get("hold_time"))
        item["fees_usd"] = row.get("fees_usd")
        item["pnl"] = row.get("pnl_usd")
        item["pnl_usd"] = row.get("pnl_usd")
        item["pnl_pct"] = row.get("pnl_pct", _trade_return_pct(item))
        item["live_trading_allowed_from_existing_settings"] = row.get("entry_snapshot_live_trading_allowed")
        item["direction_correct"] = bool(
            _s(item.get("predicted_direction", "")).lower() == _s(item.get("actual_direction", "")).lower()
        )
        item["trigger_correct"] = bool(
            _s(item.get("predicted_exit_trigger", "")).lower() == _s(item.get("actual_exit_trigger", "")).lower()
        )
        item["pnl_trend_correct"] = bool(
            _s(item.get("predicted_pnl_trend", "")).lower() == _s(item.get("actual_pnl_trend", "")).lower()
        )
        missing_fields = _missing_completed_live_fields(item, mk)
        item["missing_prediction_fields"] = list(missing_fields)
        item["missing_fields"] = list(missing_fields)
        eligible = len(missing_fields) == 0
        item["eligible_for_future_learning"] = bool(eligible)
        item["ineligible_reason"] = "" if eligible else ("missing_fields:" + ",".join(missing_fields))
        found += 1
        by_market[mk] = int(by_market.get(mk, 0) + 1)
        if eligible:
            completed += 1
        if decision_snapshot_id or pred_dir or pred_trigger or pred_pnl or (pred_conf is not None):
            post_model_completed += 1
        for field in missing_fields:
            bucket = missing_fields_by_market.setdefault(mk, {})
            bucket[field] = int(bucket.get(field, 0) + 1)
        by_predictor[predictor_name] = int(by_predictor.get(predictor_name, 0) + 1)
        rows_out.append(item)
    rows_out.sort(key=_stable_row_sort_key)
    missing_reason = ""
    if not rows_out:
        if found <= 0:
            missing_reason = "no_completed_live_decision_prediction_fields"
        else:
            missing_reason = "missing_required_predicted_direction_or_confidence"
    return rows_out, {
        "live_decision_source_available": bool(rows_out),
        "live_decision_rows_found": int(found),
        "live_decision_rows_completed": int(completed),
        "live_decision_join_rate_pct": round(100.0 * float(completed) / float(max(1, found)), 4) if found else 0.0,
        "live_decision_rows_by_market": by_market,
        "live_decision_rows_by_predictor": by_predictor,
        "completed_live_missing_fields_by_market": missing_fields_by_market,
        "completed_live_eligible_rows_by_market": {mk: int(completed)},
        "completed_live_ineligible_rows_by_market": {mk: int(max(0, found - completed))},
        "completed_live_waiting_for_real_trade_completion": bool(post_model_completed <= 0),
        "completed_live_no_post_model_completed_trades_yet": bool(post_model_completed <= 0),
        "live_decision_missing_reason": missing_reason,
        "model_quality_source_priority_used": "completed_live_decision_snapshot" if rows_out else "",
    }


def _closed_trades_from_exits(events: Iterable[Dict[str, Any]], market: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    mk = _s(market).lower()
    for row in list(events or []):
        if not isinstance(row, dict) or _s(row.get("event", "")).lower() != "exit":
            continue
        ts = int(_f(row.get("ts", 0.0), 0.0))
        hold_s = int(max(0.0, _f(row.get("hold_s", 0.0), 0.0)))
        sym = _s(row.get("symbol", row.get("instrument", row.get("pair", "")))).upper()
        qty_field = "units" if mk == "forex" else "qty"
        qty = abs(_f(row.get(qty_field, row.get("qty", 0.0)), 0.0))
        exit_px = _f(row.get("price", 0.0), 0.0)
        if ts <= 0 or (not sym) or qty <= 0.0 or exit_px <= 0.0:
            continue
        entry_px = _derive_entry_price_from_exit(row, market)
        if entry_px <= 0.0:
            continue
        entry_ts = int(ts - hold_s) if hold_s > 0 else int(max(0, ts - 3600))
        pnl_usd = (exit_px - entry_px) * qty
        if mk == "forex" and _s(row.get("side", "")).lower() != "long":
            pnl_usd = (entry_px - exit_px) * qty
        out.append(
            {
                "symbol": sym,
                "instrument": sym if mk == "forex" else None,
                "market": mk,
                "entry_ts": int(entry_ts),
                "exit_ts": int(ts),
                "qty": float(qty),
                "entry_price": round(float(entry_px), 10),
                "exit_price": round(float(exit_px), 10),
                "hold_hours": round(float(max(0.0, hold_s / 3600.0)), 6),
                "pnl_usd": round(float(pnl_usd), 8),
                "actual_exit_trigger": _norm_exit_trigger(_s(row.get("tag", "")), _s(row.get("normalized_trigger", ""))),
                "event_exit_tag": _s(row.get("tag", "")),
                "side": _s(row.get("side", "")).lower(),
                "source_type": "execution_log",
                "raw_rule_reason": _s(row.get("tag", "")),
                "pnl_pct": round(float(_f(row.get("pnl_pct", 0.0), 0.0)), 6),
                "decision_snapshot_id": _s(row.get("decision_snapshot_id", "")),
                "trade_id": _s(row.get("trade_id", row.get("position_id", row.get("order_id", "")))),
                "fees_usd": row.get("fees_usd"),
                "entry_decision_snapshot_id": _s(row.get("decision_snapshot_id", "")),
            }
        )
        if out:
            latest = out[-1]
            for key, value in row.items():
                if key.startswith("entry_snapshot_") or key.startswith("predicted_"):
                    latest[key] = value
    out.sort(key=lambda r: (int(r.get("entry_ts", 0)), _s(r.get("symbol", ""))))
    return out


def build_closed_trades(events: Iterable[Dict[str, Any]], market: str = "") -> Dict[str, Any]:
    open_lots: Dict[str, List[Dict[str, Any]]] = {}
    closed: List[Dict[str, Any]] = []
    orphan_exit_qty = 0.0
    mm = _s(market).lower()
    if mm in {"stocks", "forex"}:
        # These markets can provide deterministic closed-trade reconstruction from exit payloads.
        # Prefer exit-derived rows to avoid distortion from sparse/misaligned entry logs.
        closed = _closed_trades_from_exits(events, mm)
        return {"closed_trades": closed, "orphan_exit_qty": 0.0, "open_lot_qty": 0.0}

    for row in list(events or []):
        if not isinstance(row, dict):
            continue
        sym = _s(row.get("symbol", "")).upper()
        if not sym:
            continue
        open_lots.setdefault(sym, [])
        ev = _s(row.get("event", "")).lower()
        if ev == "entry":
            lot = {
                "ts": int(_f(row.get("ts", 0), 0.0)),
                "qty": float(_f(row.get("qty", 0.0), 0.0)),
                "price": float(_f(row.get("price", 0.0), 0.0)),
                "side": _s(row.get("side", "")).lower(),
            }
            for key, value in row.items():
                if key.startswith(CRYPTO_SNAPSHOT_JOIN_PREFIX):
                    lot[key] = value
            lot["decision_snapshot_id"] = _snapshot_id_from_event(row)
            open_lots[sym].append(lot)
            continue
        if ev != "exit":
            continue
        rem = float(_f(row.get("qty", 0.0), 0.0))
        while rem > 1e-12 and open_lots[sym]:
            lot = open_lots[sym][0]
            take = min(rem, float(_f(lot.get("qty", 0.0), 0.0)))
            if take <= 1e-12:
                open_lots[sym].pop(0)
                continue
            entry_ts = int(_f(lot.get("ts", 0), 0.0))
            exit_ts = int(_f(row.get("ts", 0), 0.0))
            entry_px = float(_f(lot.get("price", 0.0), 0.0))
            exit_px = float(_f(row.get("price", 0.0), 0.0))
            hold_h = max(0.0, (exit_ts - entry_ts) / 3600.0)
            pnl_usd = (exit_px - entry_px) * float(take)
            closed_row = {
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "qty": float(take),
                "entry_price": round(entry_px, 10),
                "exit_price": round(exit_px, 10),
                "hold_hours": round(hold_h, 6),
                "pnl_usd": round(pnl_usd, 8),
                "actual_exit_trigger": _norm_exit_trigger(_s(row.get("tag", "")), _s(row.get("normalized_trigger", ""))),
                "event_exit_tag": _s(row.get("tag", "")),
                "entry_decision_snapshot_id": _s(lot.get("decision_snapshot_id", "")),
                "exit_decision_snapshot_id": _snapshot_id_from_event(row),
            }
            for key, value in lot.items():
                if key.startswith(CRYPTO_SNAPSHOT_JOIN_PREFIX):
                    closed_row[f"entry_{key}"] = value
            for key, value in row.items():
                if key.startswith(CRYPTO_SNAPSHOT_JOIN_PREFIX):
                    closed_row[f"exit_{key}"] = value
            closed_row["crypto_feature_source_used"] = _feature_source_used_for_crypto_row(closed_row)
            closed.append(closed_row)
            lot["qty"] = float(_f(lot.get("qty", 0.0), 0.0)) - float(take)
            rem -= float(take)
            if float(_f(lot.get("qty", 0.0), 0.0)) <= 1e-12:
                open_lots[sym].pop(0)
        if rem > 1e-12:
            orphan_exit_qty += float(rem)

    open_qty = 0.0
    for lots in open_lots.values():
        for lot in lots:
            open_qty += max(0.0, _f(lot.get("qty", 0.0), 0.0))
    return {
        "closed_trades": closed,
        "orphan_exit_qty": round(float(orphan_exit_qty), 8),
        "open_lot_qty": round(float(open_qty), 8),
    }


def build_market_dataset_quality(hub_dir: str, market: str) -> Dict[str, Any]:
    loaded = load_market_trade_events(hub_dir, market)
    events = loaded.get("events", []) if isinstance(loaded.get("events", []), list) else []
    diagnostics = loaded.get("diagnostics", {}) if isinstance(loaded.get("diagnostics", {}), dict) else {}
    event_ids: Dict[str, int] = {}
    duplicate_events = 0
    entries = 0
    exits = 0
    manual_exits = 0
    trigger_counts: Dict[str, int] = {}
    exit_ok_pnl_mismatch = 0
    exits_with_pnl = 0
    for row in events:
        ev = _s(row.get("event", "")).lower()
        if ev == "entry":
            entries += 1
        elif ev == "exit":
            exits += 1
            tag = _norm_exit_trigger(_s(row.get("tag", "")))
            trigger_counts[tag] = int(trigger_counts.get(tag, 0) + 1)
            if tag == "Manual":
                manual_exits += 1
            has_pnl = ("pnl_usd" in row)
            if has_pnl:
                exits_with_pnl += 1
                pnl = _f(row.get("pnl_usd", 0.0), 0.0)
                ok = bool(row.get("ok", False))
                if (pnl < 0.0 and ok) or (pnl > 0.0 and (not ok)):
                    exit_ok_pnl_mismatch += 1
        sig = "|".join(
            [
                str(int(_f(row.get("ts", 0), 0.0))),
                _s(row.get("event", "")).lower(),
                _s(row.get("symbol", "")).upper(),
                _s(row.get("order_id", "")),
                f"{_f(row.get('qty', 0.0), 0.0):.10f}",
                f"{_f(row.get('price', 0.0), 0.0):.10f}",
            ]
        )
        seen = int(event_ids.get(sig, 0))
        if seen > 0:
            duplicate_events += 1
        event_ids[sig] = seen + 1

    closed_info = build_closed_trades(events, _s(market).lower())
    closed = closed_info.get("closed_trades", []) if isinstance(closed_info.get("closed_trades", []), list) else []
    live_rows, live_diag = _completed_live_decision_rows(market=_s(market).lower(), events=events, closed_rows=closed)
    holds_non_positive = sum(1 for r in closed if _f(r.get("hold_hours", 0.0), 0.0) <= 0.0)
    out = {
        "market": _s(market).lower(),
        "state": _s(loaded.get("state", "NO_DATA")) or "NO_DATA",
        "source": _s(loaded.get("source", "")),
        "event_rows": int(len(events)),
        "entries": int(entries),
        "exits": int(exits),
        "duplicate_events": int(duplicate_events),
        "manual_exits": int(manual_exits),
        "exit_ok_pnl_mismatch": int(exit_ok_pnl_mismatch),
        "exits_with_pnl": int(exits_with_pnl),
        "trigger_counts": trigger_counts,
        "closed_trade_rows": int(len(closed)),
        "closed_hold_non_positive": int(holds_non_positive),
        "orphan_exit_qty": round(_f(closed_info.get("orphan_exit_qty", 0.0), 0.0), 8),
        "open_lot_qty": round(_f(closed_info.get("open_lot_qty", 0.0), 0.0), 8),
        "live_decision_source_diagnostics": live_diag,
    }
    if diagnostics:
        out["context_diagnostics"] = diagnostics
    if _s(market).lower() == "crypto":
        out["decision_snapshot_diagnostics"] = (
            diagnostics.get("decision_snapshot_diagnostics", {})
            if isinstance(diagnostics.get("decision_snapshot_diagnostics", {}), dict)
            else _crypto_snapshot_diagnostics(hub_dir, events)
        )
    if _s(market).lower() == "forex":
        pair_counts: Dict[str, int] = {}
        recent_returns: List[float] = []
        for row in closed[-40:]:
            sym = _s(row.get("symbol", "")) or "Unknown"
            pair_counts[sym] = int(pair_counts.get(sym, 0) + 1)
            recent_returns.append(_trade_return_pct(row))
        mean = (sum(recent_returns) / float(len(recent_returns))) if recent_returns else 0.0
        var = (sum((x - mean) ** 2 for x in recent_returns) / float(len(recent_returns))) if recent_returns else 0.0
        out["feature_source_diagnostics"] = {
            "feature_source_interface": ["execution_log", "stale_exit_context", "closed_trade_only"],
            "feature_source_used": "closed_trade_only",
            "pair_counts_recent": _top_counter(pair_counts, limit=12),
            "recent_return_mean_pct": round(float(mean), 6),
            "recent_return_volatility_pct": round(float(math.sqrt(max(0.0, var))), 6),
            "missing_feature_reason": "historical candle-context join not yet implemented for forex replay",
        }
    return out


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload if isinstance(payload, dict) else {}, f, indent=2)
    return path


def _write_completed_live_decision_artifacts(
    hub_dir: str,
    rows_by_market: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    per_market: Dict[str, Dict[str, Any]] = {}
    unified: List[Dict[str, Any]] = []
    for market, rows in rows_by_market.items():
        mk = _s(market).lower()
        market_rows = list(rows or [])
        path = os.path.join(hub_dir, mk, "completed_live_decisions.jsonl")
        _write_jsonl(path, market_rows)
        per_market[mk] = {
            "path": path,
            "rows": int(len(market_rows)),
            "sha256": _sha256_file(path),
        }
        unified.extend(market_rows)
    unified.sort(key=_stable_row_sort_key)
    unified_path = os.path.join(hub_dir, "completed_live_decisions.jsonl")
    _write_jsonl(unified_path, unified)
    return {
        "per_market": per_market,
        "unified": {
            "path": unified_path,
            "rows": int(len(unified)),
            "sha256": _sha256_file(unified_path),
        },
    }


def _synthetic_completed_live_closed_row(market: str, snapshot_payload: Dict[str, Any]) -> Dict[str, Any]:
    mk = _s(market).lower()
    snapshot_id = _s(snapshot_payload.get("decision_snapshot_id", ""))
    pred_conf = snapshot_payload.get("confidence", snapshot_payload.get("predicted_confidence", snapshot_payload.get("calib_prob")))
    if mk == "crypto":
        return {
            "market": mk,
            "symbol": "BTC-USD",
            "entry_ts": 1_700_000_000,
            "exit_ts": 1_700_000_000 + 7200,
            "entry_price": 100.0,
            "exit_price": 103.0,
            "side": "buy",
            "qty": 0.25,
            "hold_hours": 2.0,
            "pnl_usd": 0.75,
            "pnl_pct": 3.0,
            "actual_exit_trigger": "Trailing",
            "decision_snapshot_id": snapshot_id,
            "entry_decision_snapshot_id": snapshot_id,
            "entry_snapshot_selected_action": snapshot_payload.get("selected_action"),
            "entry_snapshot_predicted_direction": snapshot_payload.get("predicted_direction"),
            "entry_snapshot_predicted_exit_trigger": snapshot_payload.get("predicted_exit_trigger"),
            "entry_snapshot_predicted_pnl_trend": snapshot_payload.get("predicted_pnl_trend"),
            "entry_snapshot_predicted_confidence": pred_conf,
            "entry_snapshot_selected_predictor": snapshot_payload.get("selected_predictor"),
            "entry_snapshot_predictor_variant": snapshot_payload.get("predictor_variant"),
            "entry_snapshot_source_used": snapshot_payload.get("source_used"),
            "entry_snapshot_trade_quality_score": snapshot_payload.get("trade_quality_score"),
            "entry_snapshot_pnl_quality_score": snapshot_payload.get("pnl_quality_score"),
            "entry_snapshot_raw_rule_reason": snapshot_payload.get("raw_rule_reason"),
            "entry_snapshot_normalized_trigger": snapshot_payload.get("normalized_trigger"),
        }
    if mk == "stocks":
        return {
            "market": mk,
            "symbol": "NVDA",
            "entry_ts": 1_700_000_000,
            "exit_ts": 1_700_000_000 + 7200,
            "entry_price": 100.0,
            "exit_price": 104.0,
            "side": "buy",
            "qty": 1.0,
            "hold_hours": 2.0,
            "pnl_usd": 4.0,
            "pnl_pct": 4.0,
            "actual_exit_trigger": "Trailing",
            "decision_snapshot_id": snapshot_id,
            "entry_decision_snapshot_id": snapshot_id,
            "entry_snapshot_selected_action": snapshot_payload.get("selected_action"),
            "entry_snapshot_predicted_direction": snapshot_payload.get("predicted_direction"),
            "entry_snapshot_predicted_exit_trigger": snapshot_payload.get("predicted_exit_trigger"),
            "entry_snapshot_predicted_pnl_trend": snapshot_payload.get("predicted_pnl_trend"),
            "entry_snapshot_predicted_confidence": pred_conf,
            "entry_snapshot_selected_predictor": snapshot_payload.get("selected_predictor"),
            "entry_snapshot_predictor_variant": snapshot_payload.get("predictor_variant"),
            "entry_snapshot_source_used": snapshot_payload.get("source_used"),
            "entry_snapshot_trade_quality_score": snapshot_payload.get("trade_quality_score"),
            "entry_snapshot_pnl_quality_score": snapshot_payload.get("pnl_quality_score"),
            "entry_snapshot_raw_rule_reason": snapshot_payload.get("raw_rule_reason"),
            "entry_snapshot_normalized_trigger": snapshot_payload.get("normalized_trigger"),
        }
    return {
        "market": mk,
        "symbol": "EUR_USD",
        "instrument": "EUR_USD",
        "entry_ts": 1_700_000_000,
        "exit_ts": 1_700_000_000 + 7200,
        "entry_price": 1.1000,
        "exit_price": 1.1030,
        "side": "long",
        "qty": 1000.0,
        "hold_hours": 2.0,
        "pnl_usd": 3.0,
        "pnl_pct": 0.2727,
        "actual_exit_trigger": "Trailing",
        "decision_snapshot_id": snapshot_id,
        "entry_decision_snapshot_id": snapshot_id,
        "entry_snapshot_selected_action": snapshot_payload.get("selected_action"),
        "entry_snapshot_predicted_direction": snapshot_payload.get("predicted_direction"),
        "entry_snapshot_predicted_exit_trigger": snapshot_payload.get("predicted_exit_trigger"),
        "entry_snapshot_predicted_pnl_trend": snapshot_payload.get("predicted_pnl_trend"),
        "entry_snapshot_predicted_confidence": pred_conf,
        "entry_snapshot_selected_predictor": snapshot_payload.get("selected_predictor"),
        "entry_snapshot_predictor_variant": snapshot_payload.get("predictor_variant"),
        "entry_snapshot_source_used": snapshot_payload.get("source_used"),
        "entry_snapshot_trade_quality_score": snapshot_payload.get("trade_quality_score"),
        "entry_snapshot_pnl_quality_score": snapshot_payload.get("pnl_quality_score"),
        "entry_snapshot_raw_rule_reason": snapshot_payload.get("raw_rule_reason"),
        "entry_snapshot_normalized_trigger": snapshot_payload.get("normalized_trigger"),
    }


def _verify_completed_live_synthetic_paths() -> Dict[str, Any]:
    per_market: Dict[str, Dict[str, Any]] = {}
    for market in ("crypto", "stocks", "forex"):
        td = tempfile.mkdtemp(prefix="completed_live_verify_")
        if market == "crypto":
            entry_payload = attach_crypto_decision_snapshot(
                {
                    "ts": 1_700_000_000,
                    "event": "entry",
                    "symbol": "BTC-USD",
                    "side": "buy",
                    "price": 100.0,
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_pnl_trend": "up",
                    "predicted_confidence": 0.84,
                    "selected_predictor_name": "crypto_live_model",
                    "predictor_variant": "live",
                    "source_used": "execution_log",
                    "trade_quality_score": 0.73,
                    "pnl_quality_score": 0.71,
                    "direction_scores": {"up": 0.84, "down": 0.16},
                    "trigger_scores": {"Trailing": 0.67, "Stale Alignment": 0.23},
                },
                hub_dir=td,
                source_module="synthetic",
                source_function="verify_completed_live_synthetic_paths",
            )
        else:
            entry_payload = attach_market_decision_snapshot(
                {
                    "ts": 1_700_000_000,
                    "event": "entry",
                    "symbol": "NVDA" if market == "stocks" else "EUR_USD",
                    "instrument": "EUR_USD" if market == "forex" else None,
                    "side": "buy" if market == "stocks" else "long",
                    "price": 100.0 if market == "stocks" else 1.1000,
                    "predicted_direction": "up",
                    "predicted_exit_trigger": "Trailing",
                    "predicted_pnl_trend": "up",
                    "predicted_confidence": 0.82,
                    "selected_predictor_name": "local_market_model",
                    "predictor_variant": "live",
                    "source_used": "execution_log",
                    "trade_quality_score": 0.69,
                    "pnl_quality_score": 0.66,
                },
                market=market,
                hub_dir=td,
                settings={},
                source_module="synthetic",
                source_function="verify_completed_live_synthetic_paths",
            )
        closed_row = _synthetic_completed_live_closed_row(market, entry_payload)
        rows, diag = _completed_live_decision_rows(market=market, events=[], closed_rows=[closed_row])
        row = rows[0] if rows else {}
        per_market[market] = {
            "verified": bool(rows),
            "rows_produced": int(len(rows)),
            "eligible": bool(row.get("eligible_for_future_learning")) if row else False,
            "missing_fields": list(row.get("missing_fields", []) or []) if isinstance(row, dict) else [],
            "ineligible_reason": _s(row.get("ineligible_reason", "")) if isinstance(row, dict) else "no_row_produced",
            "decision_snapshot_path": os.path.join(td, market, "decision_snapshots.jsonl"),
            "diagnostics": diag,
            "blocker": "" if rows else "synthetic_completed_live_row_not_produced",
        }
    return {
        "completed_live_synthetic_path_verified_by_market": {m: bool(info.get("verified")) for m, info in per_market.items()},
        "completed_live_synthetic_eligible_by_market": {m: bool(info.get("eligible")) for m, info in per_market.items()},
        "completed_live_synthetic_blockers_by_market": {m: _s(info.get("blocker", "")) or _s(info.get("ineligible_reason", "")) for m, info in per_market.items()},
        "per_market": per_market,
    }


def _market_operational_blockers(cfg: Dict[str, Any], base_dir: str, market: str) -> List[str]:
    mk = _s(market).lower()
    blockers: List[str] = []
    if not _market_live_allowed_from_existing_settings(cfg, mk):
        blockers.append("live_disabled_by_existing_settings")
    if mk == "stocks":
        key, secret = get_alpaca_creds(cfg, base_dir=base_dir)
        if not (key and secret):
            blockers.append("missing_broker_credentials")
    elif mk == "forex":
        account_id, token = get_oanda_creds(cfg, base_dir=base_dir)
        if not (account_id and token):
            blockers.append("missing_broker_credentials")
    elif mk == "crypto":
        key, secret = get_robinhood_creds_from_env()
        if (not key) or (not secret):
            fk, fs = get_robinhood_creds_from_files(base_dir)
            key = key or fk
            secret = secret or fs
        if not (key and secret):
            blockers.append("missing_broker_credentials")
    return blockers


def _market_runtime_status(
    cfg: Dict[str, Any],
    market: str,
    full_promotion_eligible: bool,
    blockers: List[str],
) -> str:
    mk = _s(market).lower()
    live_allowed = _market_live_allowed_from_existing_settings(cfg, mk)
    if not live_allowed:
        return "disabled_by_existing_settings"
    if blockers:
        return "blocked_operationally"
    if mk == "crypto" and (not full_promotion_eligible):
        return "live_learning_production"
    return "live_production" if full_promotion_eligible else "live_learning_production"


def _build_market_readiness_artifact(
    *,
    base_dir: str,
    hub_dir: str,
    settings: Dict[str, Any],
    replay_generation: Dict[str, Any],
    replay_diag: Dict[str, Any],
    promotion_readiness: Dict[str, Any],
    completed_live_artifacts: Dict[str, Any],
    existing_runtime_settings: Dict[str, Any],
) -> Dict[str, Any]:
    markets = ["crypto", "stocks", "forex"]
    created_ts = int(time.time())
    payload: Dict[str, Any] = {
        "ts": created_ts,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_ts)),
        "settings_used_as_is": True,
        "existing_runtime_settings": existing_runtime_settings,
        "markets": {},
    }
    per_market_artifacts = (
        completed_live_artifacts.get("per_market", {})
        if isinstance(completed_live_artifacts.get("per_market", {}), dict)
        else {}
    )
    for mk in markets:
        replay_meta = replay_generation.get(mk, {}) if isinstance(replay_generation.get(mk, {}), dict) else {}
        metrics = replay_diag.get(mk, {}).get("headline_metrics", {}) if isinstance(replay_diag.get(mk, {}), dict) else {}
        promo = promotion_readiness.get(mk, {}) if isinstance(promotion_readiness.get(mk, {}), dict) else {}
        live_diag = replay_meta.get("live_decision_source_diagnostics", {}) if isinstance(replay_meta.get("live_decision_source_diagnostics", {}), dict) else {}
        operational_blockers = _market_operational_blockers(settings, base_dir, mk)
        live_allowed = _market_live_allowed_from_existing_settings(settings, mk)
        full_promo = bool(promo.get("full_promotion_eligible", False))
        payload["markets"][mk] = {
            "existing_live_setting": bool(_market_existing_live_enabled(settings, mk)),
            "live_allowed_based_on_existing_setting": bool(live_allowed),
            "runtime_status": _market_runtime_status(settings, mk, full_promo, operational_blockers),
            "full_promotion_eligible": full_promo,
            "source_used": _s(replay_meta.get("model_quality_primary_source_used", replay_meta.get("source_type", ""))),
            "latest_metrics": dict(metrics if isinstance(metrics, dict) else {}),
            "blockers": list(promo.get("blockers", []) or []),
            "whether_blockers_prevent_live_trading": bool(operational_blockers),
            "model_quality_blockers_visible": bool(promo.get("blockers")),
            "operational_blockers": operational_blockers,
            "completed_live_logging_status": {
                "rows_found": int(_f(live_diag.get("live_decision_rows_found", 0), 0.0)),
                "rows_completed": int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)),
                "join_rate_pct": round(_f(live_diag.get("live_decision_join_rate_pct", 0.0), 0.0), 4),
                "artifact_path": _s(per_market_artifacts.get(mk, {}).get("path", "")),
            },
            "completed_live_learning_status": {
                "supplemental_only": bool(not replay_meta.get("completed_live_rows_used_as_primary", False)),
                "used_as_primary": bool(replay_meta.get("completed_live_rows_used_as_primary", False)),
                "used_as_supplemental": bool(replay_meta.get("completed_live_rows_used_as_supplemental", False)),
                "learning_ready": bool(int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)) >= 20),
            },
            "last_evaluated_timestamp": int(created_ts),
        }
    artifact_path = os.path.join(hub_dir, "model_quality_market_readiness.json")
    payload["path"] = artifact_path
    _write_json(artifact_path, payload)
    return payload


def _legacy_trade_source_paths(hub_dir: str) -> Dict[str, List[str]]:
    return {
        "crypto": [
            os.path.join(hub_dir, "crypto", "execution_audit.jsonl"),
            os.path.join(hub_dir, "trade_history.jsonl"),
        ],
        "stocks": [
            os.path.join(hub_dir, "stocks", "execution_audit.jsonl"),
        ],
        "forex": [
            os.path.join(hub_dir, "forex", "execution_audit.jsonl"),
        ],
    }


def _legacy_timestamp_quality(row: Dict[str, Any], market: str, source_file: str) -> str:
    mk = _s(market).lower()
    if mk == "crypto":
        return "paired_entry_exit" if source_file.endswith("execution_audit.jsonl") else "paired_entry_exit_from_trade_history"
    hold_h = _f(row.get("hold_hours", 0.0), 0.0)
    if hold_h > 0.0:
        return "derived_from_exit_hold"
    return "single_timestamp_inferred_entry"


def _legacy_side_for_row(row: Dict[str, Any], market: str) -> str:
    mk = _s(market).lower()
    side = _s(row.get("side", "")).lower()
    if side:
        return side
    if mk in {"crypto", "stocks"}:
        return "long"
    return ""


def _normalize_legacy_completed_trades(hub_dir: str) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    raw_rows_found = 0
    source_files_used: List[str] = []
    rows_by_market: Dict[str, int] = {}
    timestamp_quality_counts: Dict[str, int] = {}
    field_coverage: Dict[str, int] = {
        "entry_ts": 0,
        "exit_ts": 0,
        "entry_price": 0,
        "exit_price": 0,
        "pnl_usd": 0,
        "pnl_pct": 0,
        "qty": 0,
        "score": 0,
        "calib_prob": 0,
        "required_score": 0,
    }
    dedupe: Set[str] = set()
    for market, paths in _legacy_trade_source_paths(hub_dir).items():
        for path in paths:
            src_rows = _safe_read_jsonl(path)
            if not src_rows:
                continue
            raw_rows_found += int(len(src_rows))
            source_files_used.append(path)
            rows_for_events = src_rows
            if market == "forex":
                rows_for_events, _diag = _prepare_forex_audit_rows(src_rows)
            if market == "crypto" and path.endswith("trade_history.jsonl"):
                converted: List[Dict[str, Any]] = []
                for row in src_rows:
                    side = _s(row.get("side", "")).lower()
                    if side not in {"buy", "sell"}:
                        continue
                    x = dict(row)
                    x["event"] = "entry" if side == "buy" else "exit"
                    x["pnl_usd"] = _f(row.get("realized_profit_usd", 0.0), 0.0)
                    converted.append(x)
                rows_for_events = converted
            events = []
            for row in rows_for_events:
                ev = _as_trade_event(row, market)
                if isinstance(ev, dict):
                    events.append(ev)
            closed = build_closed_trades(events, market).get("closed_trades", [])
            for row in list(closed or []):
                item = dict(row)
                item["market"] = market
                item["legacy_source_file"] = path
                item["legacy_source_name"] = os.path.basename(path)
                item["legacy_trade_id"] = "|".join(
                    [
                        market,
                        _s(item.get("symbol", "")),
                        str(int(_f(item.get("entry_ts", 0.0), 0.0))),
                        str(int(_f(item.get("exit_ts", 0.0), 0.0))),
                        f"{_f(item.get('qty', 0.0), 0.0):.8f}",
                        f"{_f(item.get('entry_price', 0.0), 0.0):.8f}",
                        f"{_f(item.get('exit_price', 0.0), 0.0):.8f}",
                    ]
                )
                if item["legacy_trade_id"] in dedupe:
                    continue
                dedupe.add(item["legacy_trade_id"])
                item["side"] = _legacy_side_for_row(item, market)
                item["timestamp_quality"] = _legacy_timestamp_quality(item, market, path)
                item["score"] = row.get("score")
                item["calib_prob"] = row.get("calib_prob")
                item["required_score"] = row.get("required_score")
                rows_by_market[market] = int(rows_by_market.get(market, 0) + 1)
                timestamp_quality_counts[item["timestamp_quality"]] = int(timestamp_quality_counts.get(item["timestamp_quality"], 0) + 1)
                for key in list(field_coverage.keys()):
                    if item.get(key) not in (None, "", 0, 0.0):
                        field_coverage[key] = int(field_coverage.get(key, 0) + 1)
                normalized.append(item)
    normalized.sort(key=_stable_row_sort_key)
    return {
        "rows": normalized,
        "raw_rows_found": int(raw_rows_found),
        "normalized_rows_found": int(len(normalized)),
        "rows_by_market": rows_by_market,
        "timestamp_quality_summary": timestamp_quality_counts,
        "field_coverage": field_coverage,
        "source_files_used": sorted(set(source_files_used)),
    }


def _bar_ts_seconds(row: Dict[str, Any]) -> int:
    raw = row.get("t", row.get("ts", row.get("timestamp", 0)))
    if isinstance(raw, str):
        try:
            return int(time.mktime(time.strptime(raw.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")))
        except Exception:
            try:
                return int(_f(raw, 0.0))
            except Exception:
                return 0
    value = int(_f(raw, 0.0))
    if value > 1_000_000_000_000:
        return int(value / 1000)
    return value


def _build_stock_candidate_asof(symbol: str, bars: List[Dict[str, Any]], entry_ts: int) -> Dict[str, Any]:
    rows = [dict(r) for r in list(bars or []) if _bar_close(r) > 0.0]
    rows.sort(key=_bar_ts_seconds)
    idx = max((i for i, row in enumerate(rows) if _bar_ts_seconds(row) <= int(entry_ts)), default=-1)
    if idx < 24:
        return {}
    px = _bar_close(rows[idx])
    prev6 = _bar_close(rows[idx - 6])
    prev24 = _bar_close(rows[idx - 24])
    if px <= 0.0 or prev6 <= 0.0 or prev24 <= 0.0:
        return {}
    recent_returns: List[float] = []
    for j in range(max(1, idx - 12), idx + 1):
        prev_px = _bar_close(rows[j - 1])
        cur_px = _bar_close(rows[j])
        if prev_px > 0.0 and cur_px > 0.0:
            recent_returns.append(((cur_px / prev_px) - 1.0) * 100.0)
    mom6 = ((px / prev6) - 1.0) * 100.0
    mom24 = ((px / prev24) - 1.0) * 100.0
    trend_momentum_score = (0.55 * mom6) + (0.45 * mom24)
    recent_volatility = _stddev_local(recent_returns)
    signal_margin = max(0.0, (mom6 / 100.0)) + max(0.0, (mom24 / 100.0))
    return {
        "symbol": _normalize_stock_ticker(symbol),
        "market": "stocks",
        "source_type": "legacy_trade_model_replay",
        "provider": "local_cache_or_provider",
        "candle_timeframe": "1Hour",
        "entry_price": round(float(px), 10),
        "entry_ts": int(entry_ts),
        "recent_return_6": round(float(mom6), 6),
        "recent_return_24": round(float(mom24), 6),
        "recent_volatility": round(float(recent_volatility), 6),
        "trend_momentum_score": round(float(trend_momentum_score), 6),
        "signal_margin": round(float(signal_margin), 6),
        "signal_side": "long",
        "regime": "legacy_trade_model_replay",
        "bars_in_trade": 0,
        "bars_since_peak": 0,
        "peak_profit_pct": 0.0,
        "max_favorable_excursion_pct": 0.0,
        "max_adverse_excursion_pct": 0.0,
        "drawdown_from_peak_pct": 0.0,
        "trailing_armed": False,
        "favorable_then_softened_flag": False,
        "stale_hold_profile": False,
        "volatility_expansion_pct": 0.0,
        "trend_decay_after_peak": 0.0,
        "exit_momentum_3": 0.0,
        "exit_momentum_6": 0.0,
    }


def _build_crypto_candidate_asof(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    symbol: str,
    entry_ts: int,
    timeframe: str = "1hour",
) -> Dict[str, Any]:
    main_neural_dir = _resolve_crypto_neural_dir(os.path.dirname(hub_dir), settings)
    if not main_neural_dir:
        main_neural_dir = os.path.join(os.path.dirname(hub_dir), "market_data", "coins")
    candles = crypto_historical_replay._load_cached_candles(hub_dir, symbol, timeframe)
    if len(candles) < 48:
        try:
            start_ts_ms = int((int(entry_ts) - (90 * 86400)) * 1000)
            end_ts_ms = int(int(entry_ts) * 1000)
            fetched = crypto_historical_replay._fetch_kucoin_candles(symbol, timeframe, start_ts_ms, end_ts_ms)
            if fetched:
                candles = crypto_historical_replay._normalize_kucoin_rows(candles + fetched)
                crypto_historical_replay._save_cached_candles(hub_dir, symbol, timeframe, candles)
        except Exception:
            pass
    idx = max((i for i, row in enumerate(candles) if int(_f(row[0], 0.0) / 1000) <= int(entry_ts)), default=-1)
    if idx < 24:
        return {}
    ts_ms, open_px, close_px, _high_px, _low_px, _vol = candles[idx]
    prev3 = candles[idx - 3][2]
    prev6 = candles[idx - 6][2]
    prev12 = candles[idx - 12][2]
    prev24 = candles[idx - 24][2]
    recent_returns = [((candles[idx - j][2] / candles[idx - j - 1][2]) - 1.0) * 100.0 for j in range(1, min(12, idx))]
    current_candle_pct_move = ((close_px / open_px) - 1.0) * 100.0 if open_px > 0 else 0.0
    recent_return_3 = ((close_px / prev3) - 1.0) * 100.0 if prev3 > 0 else 0.0
    recent_return_6 = ((close_px / prev6) - 1.0) * 100.0 if prev6 > 0 else 0.0
    recent_return_12 = ((close_px / prev12) - 1.0) * 100.0 if prev12 > 0 else 0.0
    recent_return_24 = ((close_px / prev24) - 1.0) * 100.0 if prev24 > 0 else 0.0
    recent_volatility = crypto_historical_replay._stddev(recent_returns)
    trend_momentum_score = (
        (0.40 * recent_return_3)
        + (0.30 * recent_return_6)
        + (0.20 * recent_return_12)
        + (0.10 * recent_return_24)
    )
    artifact_ctx = crypto_historical_replay._artifact_context(main_neural_dir, symbol, timeframe)
    predicted_low = _f(artifact_ctx.get("predicted_low_boundary", 0.0), 0.0)
    predicted_high = _f(artifact_ctx.get("predicted_high_boundary", 0.0), 0.0)
    high_edge = ((predicted_high - close_px) / close_px) * 100.0 if predicted_high > 0 and close_px > 0 else 0.0
    signal_margin = _f(artifact_ctx.get("signal_margin", 0.0), 0.0) + (0.18 * trend_momentum_score) + (0.08 * high_edge) - (0.04 * max(0.0, recent_volatility - 2.5))
    signal_side = "long" if signal_margin >= 0.0 else "short"
    return {
        "market": "crypto",
        "symbol": symbol,
        "source_type": "legacy_trade_model_replay",
        "provider": "local_cache_or_kucoin",
        "candle_timeframe": timeframe,
        "entry_price": round(float(close_px), 10),
        "entry_ts": int(entry_ts),
        "current_candle_pct_move": round(float(current_candle_pct_move), 6),
        "recent_return_3": round(float(recent_return_3), 6),
        "recent_return_6": round(float(recent_return_6), 6),
        "recent_return_12": round(float(recent_return_12), 6),
        "recent_return_24": round(float(recent_return_24), 6),
        "recent_volatility": round(float(recent_volatility), 6),
        "trend_momentum_score": round(float(trend_momentum_score), 6),
        "signal_side": signal_side,
        "signal_margin": round(float(signal_margin), 6),
        "active_timeframe_count": int(_f(artifact_ctx.get("active_timeframe_count", 0), 0.0)),
        "predicted_high_boundary": round(float(predicted_high), 10) if predicted_high > 0.0 else 0.0,
        "predicted_low_boundary": round(float(predicted_low), 10) if predicted_low > 0.0 else 0.0,
        "trained_artifact_fresh": bool(artifact_ctx.get("trained_artifacts_fresh", False)),
        "artifact_training_time": int(_f(artifact_ctx.get("artifact_training_time", 0), 0.0)),
        "strategy_adapter_used": "legacy_trade_model_replay_v1",
        "regime": "legacy_trade_model_replay",
        "risk_cut_touched": False,
        "take_profit_touched": False,
        "trailing_armed": False,
        "favorable_then_softened_flag": False,
        "max_favorable_excursion_pct": 0.0,
        "max_adverse_excursion_pct": 0.0,
        "drawdown_from_peak_pct": 0.0,
        "trailing_pullback_pct": 0.0,
        "bars_in_trade": 0,
        "exit_momentum_3": 0.0,
        "exit_momentum_6": 0.0,
        "label_rule_version": "legacy_trade_model_replay_v1",
        "exit_condition_priority_used": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
        "same_candle_multi_exit_condition_count": 0,
    }


def _legacy_trade_replay_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    profitable_take = 0
    profitable_skip = 0
    losing_take = 0
    losing_skip = 0
    dir_hits = 0
    pnl_hits = 0
    trigger_hits = 0
    trigger_scored = 0
    by_market: Dict[str, Dict[str, int]] = {}
    by_symbol: Dict[str, Dict[str, int]] = {}
    by_side: Dict[str, Dict[str, int]] = {}
    by_bucket: Dict[str, Dict[str, int]] = {}
    by_ts_quality: Dict[str, Dict[str, int]] = {}
    for row in list(rows or []):
        profitable = _trade_return_pct(row) > 1e-9
        take = bool(row.get("replayed_take_trade", False))
        key = "profitable_take" if profitable and take else "profitable_skip" if profitable else "losing_take" if take else "losing_skip"
        if key == "profitable_take":
            profitable_take += 1
        elif key == "profitable_skip":
            profitable_skip += 1
        elif key == "losing_take":
            losing_take += 1
        else:
            losing_skip += 1
        if bool(row.get("direction_correct", False)):
            dir_hits += 1
        if bool(row.get("pnl_trend_correct", False)):
            pnl_hits += 1
        if row.get("trigger_match") is not None:
            trigger_scored += 1
            if bool(row.get("trigger_match", False)):
                trigger_hits += 1
        for bucket_map, bucket_key in (
            (by_market, _s(row.get("market", "")) or "unknown"),
            (by_symbol, _s(row.get("symbol", "")) or "unknown"),
            (by_side, _s(row.get("side", "")) or "unknown"),
            (by_bucket, _bucket_probability(_f(row.get("score", row.get("calib_prob", row.get("replayed_confidence", 0.0))), 0.0))),
            (by_ts_quality, _s(row.get("timestamp_quality", "")) or "unknown"),
        ):
            bucket_map.setdefault(bucket_key, {"count": 0, "profitable_take": 0, "profitable_skip": 0, "losing_take": 0, "losing_skip": 0})
            bucket_map[bucket_key]["count"] += 1
            bucket_map[bucket_key][key] += 1
    total = max(1, len(rows))
    take_total = profitable_take + losing_take
    skip_total = profitable_skip + losing_skip
    return {
        "rows": int(len(rows)),
        "profitable_trade_model_would_take": int(profitable_take),
        "profitable_trade_model_would_skip": int(profitable_skip),
        "losing_trade_model_would_take": int(losing_take),
        "losing_trade_model_would_skip": int(losing_skip),
        "take_precision_pct": round(100.0 * profitable_take / max(1, take_total), 4),
        "skip_precision_pct": round(100.0 * losing_skip / max(1, skip_total), 4),
        "false_positive_count": int(losing_take),
        "false_negative_count": int(profitable_skip),
        "profitable_missed_count": int(profitable_skip),
        "losing_avoided_count": int(losing_skip),
        "direction_accuracy_pct": round(100.0 * dir_hits / total, 4),
        "pnl_trend_accuracy_pct": round(100.0 * pnl_hits / total, 4),
        "trigger_match_pct": round(100.0 * trigger_hits / max(1, trigger_scored), 4),
        "trigger_scored_rows": int(trigger_scored),
        "by_market": by_market,
        "by_symbol": _top_counter({k: v.get("count", 0) for k, v in by_symbol.items()}, limit=20),
        "by_side": by_side,
        "by_score_bucket": by_bucket,
        "by_timestamp_quality": by_ts_quality,
    }


def build_legacy_trade_model_replay(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    legacy = _normalize_legacy_completed_trades(hub_dir)
    rows = list(legacy.get("rows", []) or [])
    eligible_rows: List[Dict[str, Any]] = []
    replay_rows: List[Dict[str, Any]] = []
    ineligible_reasons: Dict[str, int] = {}
    legacy_primary_enabled = _env_flag("MODEL_QUALITY_USE_LEGACY_REPLAY_AS_PRIMARY")
    source_files_used = list(legacy.get("source_files_used", []) or [])
    stock_bar_cache: Dict[str, List[Dict[str, Any]]] = {}
    crypto_candidate_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        market = _s(row.get("market", "")).lower()
        symbol = _s(row.get("symbol", "")).upper()
        entry_ts = int(_f(row.get("entry_ts", 0.0), 0.0))
        side = _s(row.get("side", "")).lower()
        if market not in {"crypto", "stocks", "forex"}:
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "market_unknown"
        elif not symbol:
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "symbol_missing"
        elif entry_ts <= 0:
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "usable_timestamp_missing"
        elif not side:
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "side_unknown"
        elif market == "forex":
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "historical_forex_feature_replay_not_supported"
        else:
            row["replay_eligible"] = True
            row["replay_ineligible_reason"] = ""
        if not bool(row.get("replay_eligible", False)):
            reason = _s(row.get("replay_ineligible_reason", "")) or "unknown"
            ineligible_reasons[reason] = int(ineligible_reasons.get(reason, 0) + 1)
            continue
        eligible_rows.append(row)

    for row in eligible_rows:
        market = _s(row.get("market", "")).lower()
        symbol = _s(row.get("symbol", "")).upper()
        entry_ts = int(_f(row.get("entry_ts", 0.0), 0.0))
        candidate: Dict[str, Any] = {}
        if market == "stocks":
            bars = stock_bar_cache.get(symbol)
            if bars is None:
                bars = _load_stock_cached_bars(hub_dir, symbol, "1Hour")
                if len(bars) < 48:
                    try:
                        provider, client, _diag = _stock_provider_client(settings, base_dir)
                        if client is not None:
                            start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(0, entry_ts - (120 * 86400))))
                            end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry_ts))
                            if provider == "alpaca":
                                bars = client.get_stock_bars(symbol, timeframe="1Hour", limit=720, feed="iex", start_iso=start_iso, end_iso=end_iso)
                            else:
                                bars_map = client.get_time_series_batch([symbol], interval="1h", outputsize=720)
                                bars = list((bars_map.get(symbol, []) if isinstance(bars_map, dict) else []) or [])
                            if bars:
                                _save_stock_cached_bars(hub_dir, symbol, "1Hour", list(bars))
                    except Exception:
                        pass
                stock_bar_cache[symbol] = list(bars or [])
            candidate = _build_stock_candidate_asof(symbol, stock_bar_cache.get(symbol, []), entry_ts)
        elif market == "crypto":
            cache_key = (symbol, entry_ts)
            if cache_key in crypto_candidate_cache:
                candidate = dict(crypto_candidate_cache.get(cache_key, {}))
            else:
                candidate = _build_crypto_candidate_asof(hub_dir=hub_dir, settings=settings, symbol=symbol, entry_ts=entry_ts, timeframe="1hour")
                crypto_candidate_cache[cache_key] = dict(candidate)
        if not candidate:
            row["replay_eligible"] = False
            row["replay_ineligible_reason"] = "historical_candle_data_unavailable_or_insufficient"
            reason = _s(row.get("replay_ineligible_reason", "")) or "unknown"
            ineligible_reasons[reason] = int(ineligible_reasons.get(reason, 0) + 1)
            continue
        train_rows = [
            r for r in rows
            if _s(r.get("market", "")).lower() == market
            and int(_f(r.get("exit_ts", 0.0), 0.0)) > 0
            and int(_f(r.get("exit_ts", 0.0), 0.0)) <= entry_ts
            and _s(r.get("symbol", ""))
        ]
        predictor_variant = "baseline" if market == "crypto" else "candidate"
        regime = _regime_from_prior(train_rows)
        replay_pred = _predict_one(train_rows=train_rows, candidate=candidate, regime=regime, market=market, predictor_variant=predictor_variant)
        replay_row = dict(row)
        replay_row.update(
            {
                "source_type": "legacy_trade_model_replay",
                "replayed_predicted_direction": _s(replay_pred.get("predicted_direction", "")) or "flat",
                "replayed_predicted_exit_trigger": _s(replay_pred.get("predicted_exit_trigger", "")) or "Unknown",
                "replayed_predicted_pnl_trend": _s(replay_pred.get("predicted_pnl_trend", "")) or (_s(replay_pred.get("predicted_direction", "")) or "flat"),
                "replayed_confidence": round(_f(replay_pred.get("predicted_confidence", 0.0), 0.0), 6),
                "replayed_direction_scores": dict(replay_pred.get("direction_scores", {}) if isinstance(replay_pred.get("direction_scores", {}), dict) else {}),
                "replayed_trigger_scores": dict(replay_pred.get("trigger_scores", {}) if isinstance(replay_pred.get("trigger_scores", {}), dict) else {}),
                "replayed_trade_quality_score": round(_f(replay_pred.get("stock_trade_quality_score", 0.0), 0.0), 6),
                "replay_model_variant": predictor_variant,
            }
        )
        if market == "crypto":
            abstain_cfg = _calibrate_abstain_threshold(train_rows, market, predictor_variant=predictor_variant)
            policy = abstain_cfg.get("policy", {}) if isinstance(abstain_cfg.get("policy", {}), dict) else {}
            temp_row = dict(replay_row)
            temp_row["predicted_direction"] = replay_row["replayed_predicted_direction"]
            temp_row["predicted_exit_trigger"] = replay_row["replayed_predicted_exit_trigger"]
            temp_row["predicted_confidence"] = replay_row["replayed_confidence"]
            temp_row["direction_scores"] = replay_row["replayed_direction_scores"]
            temp_row["trigger_scores"] = replay_row["replayed_trigger_scores"]
            temp_row["source_type"] = "historical_strategy_replay"
            take_trade, skip_reason = _crypto_admission_decision(temp_row, policy)
        else:
            abstain_cfg = _calibrate_abstain_threshold(train_rows, market, predictor_variant=predictor_variant)
            threshold = _f(abstain_cfg.get("threshold", 0.6), 0.6)
            take_trade = _f(replay_row.get("replayed_confidence", 0.0), 0.0) >= threshold
            skip_reason = "" if take_trade else f"confidence_below_threshold_{threshold:.2f}"
        actual_direction = _trade_direction(replay_row)
        actual_pnl_trend = "up" if _trade_return_pct(replay_row) > 1e-9 else "down" if _trade_return_pct(replay_row) < -1e-9 else "flat"
        replay_row["replayed_take_trade"] = bool(take_trade)
        replay_row["replayed_skip_reason"] = _s(skip_reason)
        replay_row["direction_correct"] = bool(_s(replay_row.get("replayed_predicted_direction", "")).lower() == actual_direction)
        replay_row["pnl_trend_correct"] = bool(_s(replay_row.get("replayed_predicted_pnl_trend", "")).lower() == actual_pnl_trend)
        actual_trigger = _s(replay_row.get("actual_exit_trigger", ""))
        replay_row["trigger_match"] = None if not actual_trigger else bool(_s(replay_row.get("replayed_predicted_exit_trigger", "")) == actual_trigger)
        replay_rows.append(replay_row)

    summary_metrics = _legacy_trade_replay_metrics(replay_rows)
    output = {
        "status": "ok",
        "legacy_rows_found": int(legacy.get("normalized_rows_found", 0) or 0),
        "legacy_rows_found_raw": int(legacy.get("raw_rows_found", 0) or 0),
        "legacy_rows_replay_eligible": int(len(eligible_rows)),
        "legacy_rows_replayed": int(len(replay_rows)),
        "ineligible_reasons": _top_counter(ineligible_reasons, limit=20),
        "rows_by_market": dict(legacy.get("rows_by_market", {}) if isinstance(legacy.get("rows_by_market", {}), dict) else {}),
        "timestamp_quality_summary": dict(legacy.get("timestamp_quality_summary", {}) if isinstance(legacy.get("timestamp_quality_summary", {}), dict) else {}),
        "field_coverage": dict(legacy.get("field_coverage", {}) if isinstance(legacy.get("field_coverage", {}), dict) else {}),
        "source_files_used": source_files_used,
        "legacy_trade_model_replay_used_as_primary": bool(legacy_primary_enabled),
        "legacy_trade_model_replay_used_as_supplemental": True,
        "model_at_entry_replay_metrics": summary_metrics,
        "row_samples": replay_rows[:50],
    }
    jsonl_path = os.path.join(hub_dir, "legacy_trade_model_replay.jsonl")
    _write_jsonl(jsonl_path, replay_rows)
    json_path = os.path.join(hub_dir, "model_quality_legacy_trade_replay.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    output["output_json"] = json_path
    output["output_jsonl"] = jsonl_path
    return output


def _resolve_crypto_neural_dir(base_dir: str, settings: Dict[str, Any]) -> str:
    raw = _s((settings or {}).get("main_neural_dir", ""))
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(base_dir, raw))


def _attach_crypto_artifact_features(
    closed_rows: List[Dict[str, Any]],
    *,
    base_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    neural_dir = _resolve_crypto_neural_dir(base_dir, settings)
    discovery = discover_crypto_trained_artifacts(neural_dir) if neural_dir else {
        "base_dir": neural_dir,
        "trained_artifacts_found": 0,
        "trained_symbols": [],
        "trained_timeframes": {},
        "artifact_paths": [],
        "artifact_freshness": {},
        "stale_artifacts": [],
        "missing_artifacts": [],
        "per_symbol": {},
        "freshness_rule": "",
        "live_consumer_function": "engines.pt_thinker.step_coin",
    }
    attached = 0
    feature_rows = 0
    fallback_reasons: Dict[str, int] = {}
    feature_source_rows: Dict[str, int] = {"decision_snapshot": 0, "trained_artifact": 0, "closed_trade_only": 0}
    snapshot_rows = 0
    artifact_rows = 0
    missing_reason = ""
    for row in list(closed_rows or []):
        if any(key.startswith("entry_" + CRYPTO_SNAPSHOT_JOIN_PREFIX) or key.startswith("exit_" + CRYPTO_SNAPSHOT_JOIN_PREFIX) for key in row.keys()):
            snapshot_rows += 1
        symbol = _s(row.get("symbol", "")).upper()
        if neural_dir and symbol:
            feats = load_crypto_artifact_features(os.path.join(neural_dir, symbol), as_of_ts=int(_f(row.get("entry_ts", 0.0), 0.0)))
            if bool(feats.get("usable", False)):
                artifact_rows += 1
                feature_names = list(feats.get("artifact_feature_names", []) or [])
                for name in feature_names:
                    key = f"{CRYPTO_ARTIFACT_FEATURE_PREFIX}{name}"
                    row[key] = feats.get(name)
                row[f"{CRYPTO_ARTIFACT_FEATURE_PREFIX}feature_names"] = feature_names
                attached += len(feature_names)
            else:
                reason = _s(feats.get("reason_if_not_used", "")) or "artifact_unavailable"
                fallback_reasons[reason] = int(fallback_reasons.get(reason, 0) + 1)
        used = _feature_source_used_for_crypto_row(row)
        if used == "decision_snapshot":
            feature_source_rows["decision_snapshot"] += 1
        elif used == "trained_artifact":
            feature_source_rows["trained_artifact"] += 1
        else:
            feature_source_rows["closed_trade_only"] += 1
        row["crypto_feature_source_used"] = used
    feature_rows = int(sum(feature_source_rows.values()))
    model_quality_uses_artifacts = artifact_rows > 0
    if not model_quality_uses_artifacts:
        if not neural_dir:
            missing_reason = "main_neural_dir_missing"
        elif int(discovery.get("trained_artifacts_found", 0) or 0) <= 0:
            missing_reason = "no_trained_artifacts_found"
        elif fallback_reasons:
            missing_reason = max(fallback_reasons.items(), key=lambda kv: kv[1])[0]
        else:
            missing_reason = "artifacts_not_candidate_safe"
    return {
        "discovery": {
            **discovery,
            "model_quality_uses_artifacts": bool(model_quality_uses_artifacts),
            "reason_if_not_used": "" if model_quality_uses_artifacts else missing_reason,
        },
        "feature_source": {
            "crypto_feature_source_priority": [
                "historical_strategy_replay",
                "trained_artifact",
                "closed_trade_only",
            ],
            "crypto_feature_source_used": (
                "trained_artifact"
                if artifact_rows > 0
                else "closed_trade_only"
            ),
            "feature_source_rows": feature_source_rows,
            "snapshot_rows": int(snapshot_rows),
            "artifact_rows": int(artifact_rows),
            "feature_rows": int(feature_rows),
            "fallback_reason": _top_counter(fallback_reasons, limit=8),
            "missing_feature_reason": "" if (snapshot_rows > 0 or artifact_rows > 0) else missing_reason,
            "historical_strategy_replay_available": False,
            "historical_strategy_replay_reason": "adapter_not_implemented_in_this_pass",
            "artifact_features_joined_count": int(attached),
        },
    }


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")
    return path


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_row_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str, str]:
    return (
        int(_f(row.get("entry_ts", 0.0), 0.0)),
        int(_f(row.get("exit_ts", 0.0), 0.0)),
        _s(row.get("symbol", "")),
        _s(row.get("actual_exit_trigger", "")),
    )


def _stable_rows_hash(rows: List[Dict[str, Any]]) -> str:
    compact = []
    for row in list(rows or []):
        compact.append(
            [
                _s(row.get("symbol", "")),
                int(_f(row.get("entry_ts", 0.0), 0.0)),
                int(_f(row.get("exit_ts", 0.0), 0.0)),
                round(_f(row.get("entry_price", 0.0), 0.0), 8),
                round(_f(row.get("exit_price", 0.0), 0.0), 8),
                _s(row.get("actual_exit_trigger", "")),
                _s(row.get("source_type", "")),
            ]
        )
    payload = json.dumps(compact, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replay_row_id(row: Dict[str, Any]) -> str:
    return "|".join(
        [
            _s(row.get("symbol", "")),
            str(int(_f(row.get("entry_ts", 0.0), 0.0))),
            str(int(_f(row.get("exit_ts", 0.0), 0.0))),
            _s(row.get("actual_exit_trigger", "")) or "Unknown",
            _s(row.get("source_type", "")) or "unknown",
            _s(row.get("predictor_variant", "")) or "unscored",
        ]
    )


def _latest_replay_path(hub_dir: str, market: str) -> str:
    m = _s(market).lower()
    if not m:
        return ""
    openai_dir = os.path.join(hub_dir, "openai")
    pats = [f"{m}_historical_replay*.json", f"{m}_*replay*.json"]
    found: List[str] = []
    for pat in pats:
        try:
            found.extend(glob.glob(os.path.join(openai_dir, pat)))
        except Exception:
            continue
    if not found:
        return ""
    found = sorted(set(found), key=lambda p: os.path.getmtime(p), reverse=True)
    return _s(found[0])


def _trade_return_pct(row: Dict[str, Any]) -> float:
    entry = _f(row.get("entry_price", 0.0), 0.0)
    exit_px = _f(row.get("exit_price", 0.0), 0.0)
    if entry <= 0.0:
        return 0.0
    return ((exit_px / entry) - 1.0) * 100.0


def _trade_direction(row: Dict[str, Any]) -> str:
    r = _trade_return_pct(row)
    if r > 1e-9:
        return "up"
    if r < -1e-9:
        return "down"
    return "flat"


def _regime_from_prior(prior_rows: List[Dict[str, Any]]) -> str:
    if not prior_rows:
        return "unknown"
    vals = [_trade_return_pct(r) for r in prior_rows[-40:]]
    if not vals:
        return "unknown"
    mean = sum(vals) / max(1, len(vals))
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    vol = math.sqrt(max(0.0, var))
    if vol >= 2.2:
        return "high_volatility"
    if mean >= 0.35:
        return "trend_up"
    if mean <= -0.35:
        return "trend_down"
    return "range"


def _median(vals: List[float], default: float = 0.0) -> float:
    if not vals:
        return float(default)
    arr = sorted(float(v) for v in vals)
    n = len(arr)
    m = n // 2
    if n % 2 == 1:
        return float(arr[m])
    return float((arr[m - 1] + arr[m]) / 2.0)


def _stddev_local(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / max(1, len(vals))
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
    return math.sqrt(max(0.0, var))


def _quantile(vals: List[float], q: float, default: float = 0.0) -> float:
    if not vals:
        return float(default)
    arr = sorted(float(v) for v in vals)
    idx = int(round((len(arr) - 1) * max(0.0, min(1.0, float(q)))))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _recent_slice(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    return list(rows[-limit:]) if len(rows) > limit else list(rows)


def _weighted_trade_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "weight": 0.0,
            "samples": 0.0,
            "up_rate": 0.0,
            "down_rate": 0.0,
            "trailing_rate": 0.0,
            "stale_rate": 0.0,
            "manual_rate": 0.0,
            "median_up_ret_pct": 0.0,
            "median_down_ret_pct": 0.0,
            "median_abs_return_pct": 0.0,
            "q75_abs_return_pct": 0.0,
            "median_hold_h": 0.0,
            "q75_hold_h": 0.0,
        }
    total_w = 0.0
    up_w = 0.0
    down_w = 0.0
    trailing_w = 0.0
    stale_w = 0.0
    manual_w = 0.0
    up_rets: List[float] = []
    down_rets: List[float] = []
    abs_rets: List[float] = []
    holds: List[float] = []
    size = len(rows)
    for idx, row in enumerate(rows):
        # Mild recency weighting; newest rows matter more without swamping older signal.
        age_rank = idx + 1
        recency_w = 0.7 + (0.6 * (float(age_rank) / float(max(1, size))))
        total_w += recency_w
        ret_pct = _trade_return_pct(row)
        abs_rets.append(abs(ret_pct))
        holds.append(max(0.0, _f(row.get("hold_hours", 0.0), 0.0)))
        direction = _trade_direction(row)
        if direction == "up":
            up_w += recency_w
            up_rets.append(ret_pct)
        elif direction == "down":
            down_w += recency_w
            down_rets.append(abs(ret_pct))
        trig = _s(row.get("actual_exit_trigger", "Unknown")) or "Unknown"
        if trig == "Trailing":
            trailing_w += recency_w
        elif trig == "Stale Alignment":
            stale_w += recency_w
        elif trig == "Manual":
            manual_w += recency_w
    return {
        "weight": round(total_w, 6),
        "samples": float(len(rows)),
        "up_rate": (up_w / max(1e-9, total_w)),
        "down_rate": (down_w / max(1e-9, total_w)),
        "trailing_rate": (trailing_w / max(1e-9, total_w)),
        "stale_rate": (stale_w / max(1e-9, total_w)),
        "manual_rate": (manual_w / max(1e-9, total_w)),
        "median_up_ret_pct": _median(up_rets, default=0.0),
        "median_down_ret_pct": _median(down_rets, default=0.0),
        "median_abs_return_pct": _median(abs_rets, default=0.0),
        "q75_abs_return_pct": _quantile(abs_rets, 0.75, default=0.0),
        "median_hold_h": max(0.25, _median(holds, default=2.0)),
        "q75_hold_h": max(0.25, _quantile(holds, 0.75, default=2.0)),
    }


def _blend_probability(cohorts: List[Tuple[float, Dict[str, float]]], key: str) -> float:
    num = 0.0
    den = 0.0
    for weight, stats in cohorts:
        value = _f(stats.get(key, 0.0), 0.0)
        support = min(1.0, math.log1p(_f(stats.get("samples", 0.0), 0.0)) / math.log1p(20.0))
        w = weight * max(0.15, support)
        num += w * value
        den += w
    if den <= 1e-9:
        return 0.0
    return num / den


def _cohort_weighted_value(cohorts: List[Tuple[float, Dict[str, float]]], key: str, default: float = 0.0) -> float:
    num = 0.0
    den = 0.0
    for weight, stats in cohorts:
        value = _f(stats.get(key, default), default)
        support = min(1.0, math.log1p(_f(stats.get("samples", 0.0), 0.0)) / math.log1p(20.0))
        w = weight * max(0.15, support)
        num += w * value
        den += w
    if den <= 1e-9:
        return float(default)
    return num / den


def _stock_directional_pnl_rate(rows: List[Dict[str, Any]], *, direction: str) -> float:
    sample = list(rows or [])
    if not sample:
        return 0.0
    want = _s(direction).lower()
    num = 0.0
    den = 0.0
    size = len(sample)
    for idx, row in enumerate(sample):
        recency_w = _recency_weight(idx + 1, size)
        den += recency_w
        row_dir = _trade_direction(row)
        if row_dir != want:
            continue
        if want == "up" and _trade_return_pct(row) > 1e-9:
            num += recency_w
        elif want == "down" and _trade_return_pct(row) < -1e-9:
            num += recency_w
    return num / max(1e-9, den)


def _stock_shape_bucket(value: float, *, thresholds: Tuple[float, ...]) -> str:
    v = abs(float(value))
    if v < thresholds[0]:
        return f"<{thresholds[0]:.1f}"
    for lo, hi in zip(thresholds, thresholds[1:]):
        if v < hi:
            return f"{lo:.1f}-{hi:.1f}"
    return f"{thresholds[-1]:.1f}+"


def _recency_weight(age_rank: int, size: int) -> float:
    return 0.7 + (0.6 * (float(age_rank) / float(max(1, size))))


def _bucket_distribution(values: List[str], *, size: int) -> Dict[str, float]:
    if not values:
        return {}
    counts: Dict[str, float] = {}
    total = 0.0
    for idx, value in enumerate(values):
        w = _recency_weight(idx + 1, size)
        counts[value] = float(counts.get(value, 0.0) + w)
        total += w
    if total <= 1e-9:
        return {}
    return {str(k): round(float(v / total), 6) for k, v in counts.items()}


def _prototype_shape_similarity(proto: Dict[str, Any], expected_hold_h: float, expected_abs_return_pct: float) -> float:
    hold_med = max(0.25, _f(proto.get("median_hold_h", 0.0), 0.0))
    hold_q75 = max(hold_med, _f(proto.get("q75_hold_h", hold_med), hold_med))
    ret_med = max(0.05, _f(proto.get("median_abs_return_pct", 0.0), 0.0))
    ret_q75 = max(ret_med, _f(proto.get("q75_abs_return_pct", ret_med), ret_med))
    hold_center = (0.55 * hold_med) + (0.45 * hold_q75)
    ret_center = (0.55 * ret_med) + (0.45 * ret_q75)
    hold_scale = max(2.0, hold_q75, expected_hold_h)
    ret_scale = max(0.5, ret_q75, expected_abs_return_pct)
    hold_similarity = max(0.0, 1.0 - (abs(expected_hold_h - hold_center) / hold_scale))
    ret_similarity = max(0.0, 1.0 - (abs(expected_abs_return_pct - ret_center) / ret_scale))
    return (0.60 * hold_similarity) + (0.40 * ret_similarity)


def _count_by(rows: List[Dict[str, Any]], getter: Any, limit: int = 20) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = getter(row)
        counts[str(key)] = int(counts.get(str(key), 0) + 1)
    return _top_counter(counts, limit=limit)


def _confidence_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_conf: List[float] = []
    miss_conf: List[float] = []
    by_pred_trigger: Dict[str, List[float]] = {}
    by_actual_trigger: Dict[str, List[float]] = {}
    for row in rows:
        conf = _f(row.get("predicted_confidence", 0.0), 0.0)
        pred_trigger = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
        actual_trigger = _s(row.get("actual_exit_trigger", "")) or "Unknown"
        by_pred_trigger.setdefault(pred_trigger, []).append(conf)
        by_actual_trigger.setdefault(actual_trigger, []).append(conf)
        hit = (
            _s(row.get("predicted_direction", "")).lower() == _s(row.get("actual_direction", "")).lower()
            and pred_trigger == actual_trigger
        )
        if hit:
            hit_conf.append(conf)
        else:
            miss_conf.append(conf)
    return {
        "hit_count": int(len(hit_conf)),
        "miss_count": int(len(miss_conf)),
        "hit_mean": round(sum(hit_conf) / max(1, len(hit_conf)), 6),
        "miss_mean": round(sum(miss_conf) / max(1, len(miss_conf)), 6),
        "hit_median": round(_median(hit_conf, default=0.0), 6),
        "miss_median": round(_median(miss_conf, default=0.0), 6),
        "by_predicted_trigger": {
            str(k): {
                "count": int(len(v)),
                "mean": round(sum(v) / max(1, len(v)), 6),
                "median": round(_median(v, default=0.0), 6),
            }
            for k, v in by_pred_trigger.items()
        },
        "by_actual_trigger": {
            str(k): {
                "count": int(len(v)),
                "mean": round(sum(v) / max(1, len(v)), 6),
                "median": round(_median(v, default=0.0), 6),
            }
            for k, v in by_actual_trigger.items()
        },
    }


def _row_overlap_count(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> int:
    ids_a = {_s(r.get("replay_row_id", "")) for r in list(rows_a or []) if _s(r.get("replay_row_id", ""))}
    ids_b = {_s(r.get("replay_row_id", "")) for r in list(rows_b or []) if _s(r.get("replay_row_id", ""))}
    return int(len(ids_a & ids_b))


def _population_diagnostics(full_rows: List[Dict[str, Any]], admitted_rows: List[Dict[str, Any]], abstained_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "trades": int(len(rows)),
            "metrics": _metrics_for_rows(rows),
            "actual_trigger_counts": _count_by(rows, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
            "predicted_trigger_counts": _count_by(rows, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
            "actual_direction_counts": _count_by(rows, lambda r: _s(r.get("actual_direction", "")).lower() or "unknown"),
            "predicted_direction_counts": _count_by(rows, lambda r: _s(r.get("predicted_direction", "")).lower() or "unknown"),
            "symbol_counts": _count_by(rows, lambda r: _s(r.get("symbol", "")) or "UNKNOWN"),
            "hold_bucket_counts": _count_by(rows, lambda r: _bucket_hold_hours(_f(r.get("hold_hours", 0.0), 0.0))),
            "return_magnitude_bucket_counts": _count_by(
                rows,
                lambda r: _bucket_return_mag_pct(
                    _f(r.get("entry_price", 0.0), 0.0),
                    _f(r.get("actual_exit_price", r.get("exit_price", 0.0)), 0.0),
                ),
            ),
            "confidence_summary": _confidence_summary(rows),
        }

    admission_rate = (100.0 * len(admitted_rows) / max(1, len(full_rows))) if full_rows else 0.0
    return {
        "full_test_trades": int(len(full_rows)),
        "admitted_test_trades": int(len(admitted_rows)),
        "abstained_test_trades": int(len(abstained_rows)),
        "admission_rate_pct": round(admission_rate, 4),
        "full_universe_metrics": _metrics_for_rows(full_rows),
        "admitted_metrics": _metrics_for_rows(admitted_rows),
        "abstained_metrics": _metrics_for_rows(abstained_rows),
        "full_universe": summarize(full_rows),
        "admitted": summarize(admitted_rows),
        "abstained": summarize(abstained_rows),
    }


def _build_crypto_trigger_prototypes(
    train_rows: List[Dict[str, Any]],
    *,
    symbol: str,
    regime: str,
) -> Dict[str, Dict[str, Any]]:
    working = list(train_rows[-220:]) if len(train_rows) > 220 else list(train_rows)
    classes = ("Stale Alignment", "Trailing", "Manual")
    out: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        cls_rows = [r for r in working if _s(r.get("actual_exit_trigger", "Unknown")) == cls]
        sym_rows = [r for r in cls_rows if _s(r.get("symbol", "")).upper() == symbol]
        reg_rows = [r for r in cls_rows if _s(r.get("regime", "")) == regime]
        sym_reg_rows = [r for r in sym_rows if _s(r.get("regime", "")) == regime]
        recent80 = _recent_slice(cls_rows, 80)
        recent40 = _recent_slice(cls_rows, 40)
        recent_symbol = _recent_slice(sym_rows, 24)
        stats = _weighted_trade_stats(cls_rows)
        hold_buckets = _bucket_distribution(
            [_bucket_hold_hours(_f(r.get("hold_hours", 0.0), 0.0)) for r in cls_rows],
            size=len(cls_rows),
        )
        ret_buckets = _bucket_distribution(
            [
                _bucket_return_mag_pct(
                    _f(r.get("entry_price", 0.0), 0.0),
                    _f(r.get("exit_price", r.get("actual_exit_price", 0.0)), 0.0),
                )
                for r in cls_rows
            ],
            size=len(cls_rows),
        )
        out[cls] = {
            "class_name": cls,
            "support_count": int(len(cls_rows)),
            "recency_weighted_support": round(_f(stats.get("weight", 0.0), 0.0), 6),
            "symbol_support_count": int(len(sym_rows)),
            "symbol_recent_support_count": int(len(recent_symbol)),
            "regime_support_count": int(len(reg_rows)),
            "symbol_regime_support_count": int(len(sym_reg_rows)),
            "recent40_count": int(len(recent40)),
            "recent80_count": int(len(recent80)),
            "direction_up_rate": round(_f(stats.get("up_rate", 0.0), 0.0), 6),
            "direction_down_rate": round(_f(stats.get("down_rate", 0.0), 0.0), 6),
            "hold_bucket_distribution": hold_buckets,
            "return_bucket_distribution": ret_buckets,
            "median_hold_h": round(_f(stats.get("median_hold_h", 0.0), 0.0), 6),
            "q75_hold_h": round(_f(stats.get("q75_hold_h", 0.0), 0.0), 6),
            "median_abs_return_pct": round(_f(stats.get("median_abs_return_pct", 0.0), 0.0), 6),
            "q75_abs_return_pct": round(_f(stats.get("q75_abs_return_pct", 0.0), 0.0), 6),
        }
    return out


def _crypto_manual_signals(
    train_rows: List[Dict[str, Any]],
    *,
    symbol: str,
    regime: str,
    candidate_hold_h: float,
) -> Dict[str, float]:
    working = list(train_rows[-220:]) if len(train_rows) > 220 else list(train_rows)
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == symbol]
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    recent40 = _recent_slice(working, 40)
    recent80 = _recent_slice(working, 80)
    symbol_recent = _recent_slice(symbol_rows, 24)
    manual_rows = [r for r in working if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    symbol_manual_rows = [r for r in manual_rows if _s(r.get("symbol", "")).upper() == symbol]
    regime_manual_rows = [r for r in manual_rows if _s(r.get("regime", "")) == regime]
    recent_manual_40 = [r for r in recent40 if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    recent_manual_80 = [r for r in recent80 if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    symbol_recent_manual = [r for r in symbol_recent if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    stale_rows = [r for r in working if _s(r.get("actual_exit_trigger", "")) == "Stale Alignment"]
    symbol_stale_rows = [r for r in stale_rows if _s(r.get("symbol", "")).upper() == symbol]
    long_symbol_rows = [r for r in symbol_rows if _f(r.get("hold_hours", 0.0), 0.0) >= max(12.0, 0.75 * candidate_hold_h)]
    long_symbol_manual = [r for r in long_symbol_rows if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    long_symbol_stale = [r for r in long_symbol_rows if _s(r.get("actual_exit_trigger", "")) == "Stale Alignment"]

    def ratio(num: int, den: int) -> float:
        return float(num) / float(max(1, den))

    return {
        "manual_prior_support_count": float(len(manual_rows)),
        "manual_recent_support_count": float(len(recent_manual_40)),
        "manual_recent80_support_count": float(len(recent_manual_80)),
        "manual_same_symbol_support_count": float(len(symbol_manual_rows)),
        "manual_same_symbol_stale_count": float(len(symbol_stale_rows)),
        "manual_same_regime_support_count": float(len(regime_manual_rows)),
        "manual_symbol_recent_support_count": float(len(symbol_recent_manual)),
        "manual_recent_density_40": ratio(len(recent_manual_40), len(recent40)),
        "manual_recent_density_80": ratio(len(recent_manual_80), len(recent80)),
        "manual_symbol_long_hold_ratio": ratio(len(long_symbol_manual), len(long_symbol_rows)),
        "manual_symbol_long_hold_stale_count": float(len(long_symbol_stale)),
        "manual_symbol_vs_stale_long_hold_ratio": ratio(len(long_symbol_manual), len(long_symbol_manual) + len(long_symbol_stale)),
        "manual_symbol_vs_stale_ratio": ratio(len(symbol_manual_rows), len(symbol_manual_rows) + len(symbol_stale_rows)),
    }


def _crypto_predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
) -> Dict[str, Any]:
    working = list(train_rows[-220:]) if len(train_rows) > 220 else list(train_rows)
    sym = _s(candidate.get("symbol", "")).upper()

    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == sym]
    symbol_regime_rows = [r for r in symbol_rows if _s(r.get("regime", "")) == regime]
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    recent_rows = _recent_slice(working, 80)
    recent_symbol_rows = _recent_slice(symbol_rows, 24)
    recent_regime_rows = _recent_slice(regime_rows, 80)
    prototypes = _build_crypto_trigger_prototypes(working, symbol=sym, regime=regime)
    symbol_cohorts: List[Tuple[float, Dict[str, float]]] = []
    for base_weight, rows in [
        (1.55, symbol_regime_rows),
        (1.25, recent_symbol_rows),
        (0.90, symbol_rows),
    ]:
        if not rows:
            continue
        stats = _weighted_trade_stats(rows)
        if stats["weight"] <= 0.0:
            continue
        reliability = min(1.0, math.log1p(stats["weight"]) / math.log1p(24.0))
        symbol_cohorts.append((base_weight * reliability, stats))

    cohorts: List[Tuple[float, Dict[str, float]]] = []
    for base_weight, rows in [
        (1.85, symbol_regime_rows),
        (1.60, recent_symbol_rows),
        (1.10, symbol_rows),
        (0.70, recent_regime_rows),
        (0.45, recent_rows),
    ]:
        if not rows:
            continue
        stats = _weighted_trade_stats(rows)
        if stats["weight"] <= 0.0:
            continue
        reliability = min(1.0, math.log1p(stats["weight"]) / math.log1p(32.0))
        cohorts.append((base_weight * reliability, stats))

    if not cohorts:
        entry = _f(candidate.get("entry_price", 0.0), 0.0)
        return {
            "predicted_direction": "flat",
            "predicted_exit_trigger": "Unknown",
            "predicted_hold_hours": 2.0,
            "predicted_exit_price": float(entry),
            "predicted_confidence": 0.5,
            "trigger_scores": {"Unknown": 0.0},
            "direction_scores": {"flat": 0.0},
            "trigger_margin": 0.0,
            "direction_margin": 0.0,
        }

    total_weight = 0.0
    for weight, _stats in cohorts:
        total_weight += weight
    norm = max(1e-9, total_weight)
    p_up = _blend_probability(cohorts, "up_rate")
    p_down = _blend_probability(cohorts, "down_rate")
    p_trailing = _blend_probability(cohorts, "trailing_rate")
    p_stale = _blend_probability(cohorts, "stale_rate")
    p_manual = _blend_probability(cohorts, "manual_rate")
    symbol_up = _weighted_trade_stats(recent_symbol_rows).get("up_rate", 0.0) if recent_symbol_rows else 0.0
    symbol_down = _weighted_trade_stats(recent_symbol_rows).get("down_rate", 0.0) if recent_symbol_rows else 0.0
    symbol_manual = _weighted_trade_stats(recent_symbol_rows).get("manual_rate", 0.0) if recent_symbol_rows else 0.0
    symbol_momentum = _cohort_weighted_value(cohorts, "up_rate", 0.0) - _cohort_weighted_value(cohorts, "down_rate", 0.0)
    expected_hold_h = _cohort_weighted_value(cohorts, "median_hold_h", 2.0)
    expected_abs_return_pct = _cohort_weighted_value(cohorts, "median_abs_return_pct", 0.5)
    symbol_hold_h = _cohort_weighted_value(symbol_cohorts, "q75_hold_h", expected_hold_h) if symbol_cohorts else expected_hold_h
    symbol_abs_return_pct = (
        _cohort_weighted_value(symbol_cohorts, "q75_abs_return_pct", expected_abs_return_pct) if symbol_cohorts else expected_abs_return_pct
    )
    entry_buy_count = max(0.0, _f(candidate.get("entry_snapshot_trend_signal_count", 0.0), 0.0))
    entry_sell_count = max(0.0, _f(candidate.get("entry_snapshot_counter_signal_count", 0.0), 0.0))
    entry_signal_total = max(1.0, entry_buy_count + entry_sell_count)
    entry_signal_bias = (entry_buy_count - entry_sell_count) / entry_signal_total
    entry_ai_conf = max(0.0, _f(candidate.get("entry_snapshot_ai_confidence", 0.0), 0.0))
    entry_trigger_rel = max(0.0, _f(candidate.get("entry_snapshot_trigger_reliability", 0.0), 0.0))
    artifact_tf_count = max(0.0, _f(candidate.get("artifact_active_timeframe_count", 0.0), 0.0))
    artifact_threshold_mean = max(0.0, _f(candidate.get("artifact_threshold_mean", 0.0), 0.0))
    artifact_memory_count = max(0.0, _f(candidate.get("artifact_pattern_memory_count", 0.0), 0.0))
    artifact_support = min(1.0, artifact_tf_count / 7.0) * (
        0.55 + (0.45 * min(1.0, math.log1p(artifact_memory_count) / math.log1p(250.0)))
    )
    source_signal_strength = min(
        1.0,
        max(
            entry_ai_conf,
            entry_trigger_rel,
            artifact_threshold_mean,
            artifact_support * (1.0 if bool(candidate.get("artifact_trained_freshness_flag", False)) else 0.65),
        ),
    )
    candidate_hold_h = max(expected_hold_h, ((0.55 * expected_hold_h) + (0.45 * symbol_hold_h)))
    candidate_abs_return_pct = max(
        expected_abs_return_pct,
        ((0.60 * expected_abs_return_pct) + (0.40 * symbol_abs_return_pct)),
    )
    expected_hold_bucket = _bucket_hold_hours(candidate_hold_h)
    if candidate_abs_return_pct < 0.5:
        expected_ret_bucket = "<0.5%"
    elif candidate_abs_return_pct < 1.5:
        expected_ret_bucket = "0.5-1.5%"
    elif candidate_abs_return_pct < 3.0:
        expected_ret_bucket = "1.5-3.0%"
    else:
        expected_ret_bucket = "3.0%+"
    manual_proto = prototypes.get("Manual", {})
    manual_support_count = int(manual_proto.get("support_count", 0) or 0)
    manual_signals = _crypto_manual_signals(
        working,
        symbol=sym,
        regime=regime,
        candidate_hold_h=candidate_hold_h,
    )

    trig_scores: Dict[str, float] = {}
    manual_reasons: List[str] = []
    for cls in ("Stale Alignment", "Trailing", "Manual"):
        proto = prototypes.get(cls, {})
        if int(proto.get("support_count", 0) or 0) <= 0:
            trig_scores[cls] = -1e9
            continue
        hold_bucket_prob = _f((proto.get("hold_bucket_distribution", {}) if isinstance(proto.get("hold_bucket_distribution", {}), dict) else {}).get(expected_hold_bucket, 0.0), 0.0)
        ret_bucket_prob = _f((proto.get("return_bucket_distribution", {}) if isinstance(proto.get("return_bucket_distribution", {}), dict) else {}).get(expected_ret_bucket, 0.0), 0.0)
        shape_similarity = _prototype_shape_similarity(proto, candidate_hold_h, candidate_abs_return_pct)
        score = 0.0
        score += 0.95 * math.log1p(int(proto.get("symbol_regime_support_count", 0) or 0))
        score += 0.80 * math.log1p(int(proto.get("symbol_recent_support_count", 0) or 0))
        score += 0.65 * math.log1p(int(proto.get("symbol_support_count", 0) or 0))
        score += 0.40 * math.log1p(int(proto.get("regime_support_count", 0) or 0))
        score += 0.30 * math.log1p(int(proto.get("recent80_count", 0) or 0))
        score += 0.55 * hold_bucket_prob
        score += 0.55 * ret_bucket_prob
        score += 0.80 * shape_similarity
        if cls == "Trailing":
            score += 0.80 * max(0.0, p_up - p_down)
            score += 0.65 * max(0.0, symbol_up - symbol_down)
            score += 0.55 * p_trailing
            score += 0.35 * max(0.0, entry_signal_bias) * max(0.25, source_signal_strength)
            score -= 0.30 * p_manual
            if manual_support_count >= 3 and candidate_hold_h > (1.35 * max(1.0, _f(proto.get("q75_hold_h", 0.0), 0.0))):
                score -= 0.40
        elif cls == "Stale Alignment":
            score += 0.80 * max(0.0, p_down - p_up)
            score += 0.65 * max(0.0, symbol_down - symbol_up)
            score += 0.55 * p_stale
            score += 0.35 * max(0.0, -entry_signal_bias) * max(0.25, source_signal_strength)
            score -= 0.15 * p_manual
            if manual_support_count >= 3 and candidate_hold_h > (1.35 * max(1.0, _f(proto.get("q75_hold_h", 0.0), 0.0))):
                score -= 0.75
            score -= 0.50 * _f(manual_signals.get("manual_symbol_long_hold_ratio", 0.0), 0.0)
            score -= 0.35 * _f(manual_signals.get("manual_symbol_vs_stale_ratio", 0.0), 0.0)
        else:
            score += 1.10 * p_manual
            score += 0.95 * symbol_manual
            score += 0.95 * (1.0 if expected_hold_bucket == "24h+" else 0.0)
            score += 0.55 * (1.0 if expected_ret_bucket in {"1.5-3.0%", "3.0%+"} else 0.0)
            if candidate_hold_h >= max(18.0, 0.90 * _f(proto.get("median_hold_h", 0.0), 0.0)):
                score += 1.10
                manual_reasons.append("long_hold_matches_manual")
            if candidate_abs_return_pct >= max(1.5, 0.90 * _f(proto.get("median_abs_return_pct", 0.0), 0.0)):
                score += 0.35
                manual_reasons.append("return_mag_matches_manual")
            score += 1.10 * min(1.0, _f(manual_signals.get("manual_symbol_long_hold_ratio", 0.0), 0.0))
            score += 0.95 * min(1.0, _f(manual_signals.get("manual_symbol_vs_stale_long_hold_ratio", 0.0), 0.0))
            score += 0.80 * min(1.0, _f(manual_signals.get("manual_symbol_vs_stale_ratio", 0.0), 0.0))
            score += 0.55 * min(1.0, 2.0 * _f(manual_signals.get("manual_recent_density_40", 0.0), 0.0))
            score += 0.35 * min(1.0, 2.0 * _f(manual_signals.get("manual_recent_density_80", 0.0), 0.0))
            score += 0.45 * min(1.0, _f(manual_signals.get("manual_same_symbol_support_count", 0.0), 0.0) / 3.0)
            score += 0.25 * min(1.0, _f(manual_signals.get("manual_same_regime_support_count", 0.0), 0.0) / 6.0)
            if int(proto.get("support_count", 0) or 0) < 3:
                score -= 0.75
        trig_scores[cls] = score

    sorted_trig = sorted(trig_scores.items(), key=lambda kv: kv[1], reverse=True)
    pred_trig = sorted_trig[0][0]
    trig_winner = float(sorted_trig[0][1])
    trig_runner = float(sorted_trig[1][1]) if len(sorted_trig) > 1 else float(sorted_trig[0][1])
    trig_margin = max(0.0, trig_winner - trig_runner)
    stale_score = float(trig_scores.get("Stale Alignment", -1e9))
    trailing_score = float(trig_scores.get("Trailing", -1e9))
    manual_score = float(trig_scores.get("Manual", -1e9))
    manual_runner_margin = manual_score - max(stale_score, trailing_score)
    if pred_trig != "Manual":
        if manual_score > -1e8:
            if _f(manual_signals.get("manual_same_symbol_support_count", 0.0), 0.0) <= 0.0:
                manual_reasons.append("no_same_symbol_manual_support")
            if _f(manual_signals.get("manual_symbol_long_hold_ratio", 0.0), 0.0) < 0.25:
                manual_reasons.append("weak_symbol_long_hold_manual_ratio")
            if manual_runner_margin < 0.0:
                manual_reasons.append("manual_score_below_runner_up")

    direction_scores = {
        "up": (1.10 * p_up) + (0.45 * max(0.0, symbol_momentum)) + (0.35 * max(0.0, trig_scores.get("Trailing", 0.0) - trig_scores.get("Stale Alignment", 0.0))) + (0.40 * max(0.0, entry_signal_bias) * source_signal_strength),
        "down": (1.10 * p_down) + (0.45 * max(0.0, -symbol_momentum)) + (0.35 * max(0.0, trig_scores.get("Stale Alignment", 0.0) - trig_scores.get("Trailing", 0.0))) + (0.40 * max(0.0, -entry_signal_bias) * source_signal_strength),
        "flat": 0.10 + (0.25 * p_manual),
    }
    sorted_dir = sorted(direction_scores.items(), key=lambda kv: kv[1], reverse=True)
    pred_dir = sorted_dir[0][0]
    dir_winner = float(sorted_dir[0][1])
    dir_runner = float(sorted_dir[1][1]) if len(sorted_dir) > 1 else float(sorted_dir[0][1])
    dir_margin = max(0.0, dir_winner - dir_runner)

    # Manual should be allowed, but only when it wins with real margin or clear support.
    if pred_trig == "Manual" and (p_manual < 0.10 and trig_margin < 0.20):
        pred_trig = "Trailing" if trig_scores.get("Trailing", -1e9) >= trig_scores.get("Stale Alignment", -1e9) else "Stale Alignment"

    hold_h = max(0.25, candidate_hold_h)
    up_mag = max(0.05, _cohort_weighted_value(cohorts, "median_up_ret_pct", 0.35))
    down_mag = max(0.05, _cohort_weighted_value(cohorts, "median_down_ret_pct", 0.35))
    move_pct = max(0.05, up_mag if pred_dir == "up" else down_mag if pred_dir == "down" else candidate_abs_return_pct) / 100.0
    entry_px = _f(candidate.get("entry_price", 0.0), 0.0)
    if pred_dir == "up":
        exit_px = entry_px * (1.0 + move_pct)
    elif pred_dir == "down":
        exit_px = entry_px * (1.0 - move_pct)
    else:
        exit_px = entry_px

    evidence = min(1.0, math.sqrt(norm / 4.0) / 3.2)
    trig_scale = abs(trig_winner) + abs(trig_runner) + 1e-9
    dir_scale = abs(dir_winner) + abs(dir_runner) + 1e-9
    trig_margin_norm = min(1.0, trig_margin / trig_scale)
    dir_margin_norm = min(1.0, dir_margin / dir_scale)
    symbol_agreement = 0.0
    if pred_trig == "Trailing":
        symbol_agreement = max(0.0, symbol_up - symbol_down)
    elif pred_trig == "Stale Alignment":
        symbol_agreement = max(0.0, symbol_down - symbol_up)
    elif pred_trig == "Manual":
        symbol_agreement = min(1.0, _f(manual_signals.get("manual_symbol_vs_stale_ratio", 0.0), 0.0))
    global_only_penalty = 0.0
    if pred_trig == "Manual" and _f(manual_signals.get("manual_same_symbol_support_count", 0.0), 0.0) <= 0.0:
        global_only_penalty += 0.08
    if pred_trig in {"Stale Alignment", "Trailing"} and symbol_agreement < 0.10:
        global_only_penalty += 0.06
    conf = max(
        0.0,
        min(
            1.0,
            0.12
            + (0.28 * evidence)
            + (0.24 * trig_margin_norm)
            + (0.18 * dir_margin_norm)
            + (0.12 * min(1.0, symbol_agreement))
            + (0.06 * source_signal_strength)
            - global_only_penalty,
        ),
    )
    if pred_dir == "flat" and pred_trig == "Unknown":
        conf = min(conf, 0.60)
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trig,
        "predicted_hold_hours": round(float(hold_h), 6),
        "predicted_exit_price": round(float(max(1e-12, exit_px)), 10),
        "predicted_confidence": round(float(conf), 6),
        "trigger_scores": {k: round(float(v), 6) for k, v in trig_scores.items()},
        "direction_scores": {k: round(float(v), 6) for k, v in direction_scores.items()},
        "winning_trigger_score": round(float(trig_winner), 6),
        "runner_up_trigger_score": round(float(trig_runner), 6),
        "trigger_margin": round(float(trig_margin), 6),
        "winning_direction_score": round(float(dir_winner), 6),
        "runner_up_direction_score": round(float(dir_runner), 6),
        "direction_margin": round(float(dir_margin), 6),
        "manual_score": round(float(manual_score), 6),
        "stale_score": round(float(stale_score), 6),
        "trailing_score": round(float(trailing_score), 6),
        "manual_runner_up_margin": round(float(manual_runner_margin), 6),
        "manual_outcome_note": ";".join(sorted(set(manual_reasons))) if manual_reasons else ("manual_won" if pred_trig == "Manual" else ""),
        "manual_prior_support_count": int(_f(manual_signals.get("manual_prior_support_count", 0.0), 0.0)),
        "manual_recent_support_count": int(_f(manual_signals.get("manual_recent_support_count", 0.0), 0.0)),
        "manual_recent80_support_count": int(_f(manual_signals.get("manual_recent80_support_count", 0.0), 0.0)),
        "manual_same_symbol_support_count": int(_f(manual_signals.get("manual_same_symbol_support_count", 0.0), 0.0)),
        "manual_same_symbol_stale_count": int(_f(manual_signals.get("manual_same_symbol_stale_count", 0.0), 0.0)),
        "manual_same_regime_support_count": int(_f(manual_signals.get("manual_same_regime_support_count", 0.0), 0.0)),
        "manual_symbol_long_hold_ratio": round(_f(manual_signals.get("manual_symbol_long_hold_ratio", 0.0), 0.0), 6),
        "manual_symbol_long_hold_stale_count": int(_f(manual_signals.get("manual_symbol_long_hold_stale_count", 0.0), 0.0)),
        "manual_symbol_vs_stale_ratio": round(_f(manual_signals.get("manual_symbol_vs_stale_ratio", 0.0), 0.0), 6),
        "manual_symbol_vs_stale_long_hold_ratio": round(_f(manual_signals.get("manual_symbol_vs_stale_long_hold_ratio", 0.0), 0.0), 6),
        "manual_recent_density_40": round(_f(manual_signals.get("manual_recent_density_40", 0.0), 0.0), 6),
        "manual_recent_density_80": round(_f(manual_signals.get("manual_recent_density_80", 0.0), 0.0), 6),
        "entry_signal_bias": round(float(entry_signal_bias), 6),
        "source_signal_strength": round(float(source_signal_strength), 6),
        "trigger_prototype_support": {
            k: {
                "support_count": int(v.get("support_count", 0) or 0),
                "symbol_support_count": int(v.get("symbol_support_count", 0) or 0),
                "symbol_regime_support_count": int(v.get("symbol_regime_support_count", 0) or 0),
            }
            for k, v in prototypes.items()
        },
    }


_CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES = (
    "Risk Cut",
    "Stale Alignment",
    "Trailing",
    "Take Profit",
)

_CRYPTO_HISTORICAL_REPLAY_FEATURES: Tuple[Tuple[str, float], ...] = (
    ("current_candle_pct_move", 1.00),
    ("recent_return_3", 0.95),
    ("recent_return_6", 1.05),
    ("recent_return_12", 1.15),
    ("recent_return_24", 1.10),
    ("recent_volatility", 1.15),
    ("trend_momentum_score", 1.20),
    ("signal_margin", 1.25),
    ("active_timeframe_count", 0.35),
)

_CRYPTO_EXIT_SHAPE_KEYS: Tuple[str, ...] = (
    "risk_cut_touched",
    "take_profit_touched",
    "trailing_armed",
    "favorable_then_softened_flag",
    "max_favorable_excursion_pct",
    "max_adverse_excursion_pct",
    "drawdown_from_peak_pct",
    "trailing_pullback_pct",
    "bars_in_trade",
    "exit_momentum_3",
    "exit_momentum_6",
    "trend_momentum_score",
    "recent_return_3",
    "recent_return_6",
    "recent_return_12",
    "recent_return_24",
    "signal_margin",
)


def _is_historical_strategy_replay_row(row: Dict[str, Any]) -> bool:
    return _s(row.get("source_type", "")) == "historical_strategy_replay"


def _historical_replay_feature_scale(rows: List[Dict[str, Any]], key: str) -> float:
    vals = sorted([_f(r.get(key, 0.0), 0.0) for r in rows if _is_historical_strategy_replay_row(r)])
    if not vals:
        return 1.0
    q25 = vals[min(len(vals) - 1, int((len(vals) - 1) * 0.25))]
    q75 = vals[min(len(vals) - 1, int((len(vals) - 1) * 0.75))]
    spread = abs(q75 - q25)
    if spread > 1e-9:
        return spread
    base = abs(vals[len(vals) // 2])
    return max(0.1, base * 0.25, 1e-3)


def _historical_replay_bucket(key: str, value: float) -> str:
    if key == "trend_momentum_score":
        if value < 1.8:
            return "<1.8"
        if value < 2.3:
            return "1.8-2.3"
        if value < 2.8:
            return "2.3-2.8"
        if value < 3.4:
            return "2.8-3.4"
        return "3.4+"
    if key == "recent_volatility":
        if value < 0.4:
            return "<0.4"
        if value < 0.6:
            return "0.4-0.6"
        if value < 0.8:
            return "0.6-0.8"
        return "0.8+"
    if key == "signal_margin":
        if value < 0.30:
            return "<0.30"
        if value < 0.40:
            return "0.30-0.40"
        if value < 0.50:
            return "0.40-0.50"
        if value < 0.60:
            return "0.50-0.60"
        return "0.60+"
    if key.startswith("recent_return_") or key == "current_candle_pct_move":
        if value < 1.0:
            return "<1.0"
        if value < 2.0:
            return "1.0-2.0"
        if value < 3.0:
            return "2.0-3.0"
        if value < 4.0:
            return "3.0-4.0"
        return "4.0+"
    return _bucket_margin_value(value)


def _historical_replay_trigger_prototypes(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    replay_rows = [r for r in rows if _is_historical_strategy_replay_row(r)]
    for trigger in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES:
        cls_rows = [r for r in replay_rows if _s(r.get("actual_exit_trigger", "")) == trigger]
        if not cls_rows:
            continue
        proto: Dict[str, Any] = {"support_count": int(len(cls_rows))}
        for key, _weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES:
            vals = sorted([_f(r.get(key, 0.0), 0.0) for r in cls_rows])
            if not vals:
                continue
            proto[f"{key}_median"] = vals[len(vals) // 2]
            proto[f"{key}_p25"] = vals[min(len(vals) - 1, int((len(vals) - 1) * 0.25))]
            proto[f"{key}_p75"] = vals[min(len(vals) - 1, int((len(vals) - 1) * 0.75))]
            proto[f"{key}_buckets"] = _count_by(
                cls_rows,
                lambda r, k=key: _historical_replay_bucket(k, _f(r.get(k, 0.0), 0.0)),
            )
        out[trigger] = proto
    return out


def _boundary_distance_score(candidate: Dict[str, Any]) -> Tuple[float, float]:
    entry = _f(candidate.get("entry_price", 0.0), 0.0)
    high = _f(candidate.get("predicted_high_boundary", 0.0), 0.0)
    low = _f(candidate.get("predicted_low_boundary", 0.0), 0.0)
    if entry <= 0.0:
        return 0.0, 0.0
    up = max(0.0, (high - entry) / entry) if high > 0.0 else 0.0
    down = max(0.0, (entry - low) / entry) if low > 0.0 else 0.0
    return up * 100.0, down * 100.0


def _crypto_label_feature_alignment_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    replay_rows = [r for r in rows if _is_historical_strategy_replay_row(r)]
    if not replay_rows:
        return {}
    feature_distributions: Dict[str, Dict[str, Any]] = {}
    mismatch_counters = {
        "actual_risk_cut_but_risk_cut_touched_false": 0,
        "actual_take_profit_but_take_profit_touched_false": 0,
        "actual_trailing_but_trailing_armed_false": 0,
        "actual_trailing_but_favorable_then_softened_false": 0,
        "actual_stale_but_any_hard_exit_shape_flag_true": 0,
        "hard_exit_shape_flag_true_but_actual_label_is_stale": 0,
        "hard_exit_shape_flag_true_but_predictor_not_corresponding_trigger": 0,
    }
    multi_exit_count = 0
    priority_used = None
    rule_versions: Dict[str, int] = {}
    for trig in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES:
        cls_rows = [r for r in replay_rows if _s(r.get("actual_exit_trigger", "")) == trig]
        if not cls_rows:
            continue
        feat_diag: Dict[str, Any] = {"count": int(len(cls_rows))}
        for key in _CRYPTO_EXIT_SHAPE_KEYS:
            vals = [_f(r.get(key, 0.0), 0.0) for r in cls_rows]
            if key.endswith("_touched") or key.endswith("_flag") or key == "trailing_armed":
                feat_diag[key] = {
                    "true_count": int(sum(1 for r in cls_rows if bool(r.get(key, False)))),
                    "false_count": int(sum(1 for r in cls_rows if not bool(r.get(key, False)))),
                    "true_rate_pct": round(100.0 * sum(1 for r in cls_rows if bool(r.get(key, False))) / max(1, len(cls_rows)), 4),
                }
            else:
                feat_diag[key] = {
                    "median": round(_median(vals, default=0.0), 6),
                    "p25": round(_quantile(vals, 0.25, default=0.0), 6),
                    "p75": round(_quantile(vals, 0.75, default=0.0), 6),
                }
        feature_distributions[trig] = feat_diag
    for row in replay_rows:
        act = _s(row.get("actual_exit_trigger", ""))
        pred = _s(row.get("predicted_exit_trigger", ""))
        hard_risk = bool(row.get("risk_cut_touched", False))
        hard_tp = bool(row.get("take_profit_touched", False))
        hard_trail = bool(row.get("trailing_armed", False))
        soft_trail = bool(row.get("favorable_then_softened_flag", False))
        hard_any = hard_risk or hard_tp or hard_trail
        if act == "Risk Cut" and not hard_risk:
            mismatch_counters["actual_risk_cut_but_risk_cut_touched_false"] += 1
        if act == "Take Profit" and not hard_tp:
            mismatch_counters["actual_take_profit_but_take_profit_touched_false"] += 1
        if act == "Trailing" and not hard_trail:
            mismatch_counters["actual_trailing_but_trailing_armed_false"] += 1
        if act == "Trailing" and not soft_trail:
            mismatch_counters["actual_trailing_but_favorable_then_softened_false"] += 1
        if act == "Stale Alignment" and hard_any:
            mismatch_counters["actual_stale_but_any_hard_exit_shape_flag_true"] += 1
            mismatch_counters["hard_exit_shape_flag_true_but_actual_label_is_stale"] += 1
        if hard_risk and pred != "Risk Cut":
            mismatch_counters["hard_exit_shape_flag_true_but_predictor_not_corresponding_trigger"] += 1
        elif hard_tp and pred != "Take Profit":
            mismatch_counters["hard_exit_shape_flag_true_but_predictor_not_corresponding_trigger"] += 1
        elif hard_trail and soft_trail and pred != "Trailing":
            mismatch_counters["hard_exit_shape_flag_true_but_predictor_not_corresponding_trigger"] += 1
        if int(_f(row.get("same_candle_multi_exit_condition_count", 0.0), 0.0)) > 1:
            multi_exit_count += 1
        if priority_used is None and isinstance(row.get("exit_condition_priority_used"), list):
            priority_used = list(row.get("exit_condition_priority_used"))
        rule_version = _s(row.get("label_rule_version", "")) or "unknown"
        rule_versions[rule_version] = int(rule_versions.get(rule_version, 0) + 1)
    return {
        "feature_distributions_by_actual_trigger": feature_distributions,
        "mismatch_counters": mismatch_counters,
        "same_candle_multi_exit_condition_count": int(multi_exit_count),
        "exit_condition_priority_used": priority_used or [],
        "label_rule_version": _top_counter(rule_versions, limit=5),
    }


def _historical_replay_upside_score(
    candidate: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    *,
    symbol: str,
    regime: str,
) -> Dict[str, Any]:
    working = [dict(r) for r in (train_rows[-260:] if len(train_rows) > 260 else train_rows) if _is_historical_strategy_replay_row(r)]
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == symbol]
    symbol_up_rows = [r for r in symbol_rows if _s(r.get("actual_direction", "")).lower() == "up"]
    symbol_down_rows = [r for r in symbol_rows if _s(r.get("actual_direction", "")).lower() == "down"]
    regime_up_rows = [r for r in working if _s(r.get("regime", "")) == regime and _s(r.get("actual_direction", "")).lower() == "up"]
    regime_down_rows = [r for r in working if _s(r.get("regime", "")) == regime and _s(r.get("actual_direction", "")).lower() == "down"]
    up_boundary, down_boundary = _boundary_distance_score(candidate)
    momentum = _f(candidate.get("trend_momentum_score", 0.0), 0.0)
    volatility = _f(candidate.get("recent_volatility", 0.0), 0.0)
    signal_margin = _f(candidate.get("signal_margin", 0.0), 0.0)
    move3 = _f(candidate.get("recent_return_3", 0.0), 0.0)
    move6 = _f(candidate.get("recent_return_6", 0.0), 0.0)
    move12 = _f(candidate.get("recent_return_12", 0.0), 0.0)
    move24 = _f(candidate.get("recent_return_24", 0.0), 0.0)
    signal_side = _s(candidate.get("signal_side", "")).lower()

    upside = 0.0
    downside = 0.0
    reasons: List[str] = []
    if signal_side == "long":
        upside += 0.35
        reasons.append("signal_side_long")
    upside += 0.30 * max(0.0, min(1.5, (signal_margin - 0.34) / 0.20))
    downside += 0.20 * max(0.0, min(1.5, (0.36 - signal_margin) / 0.18))
    upside += 0.25 * max(0.0, min(1.5, (momentum - 2.0) / 0.8))
    downside += 0.25 * max(0.0, min(1.5, (2.35 - momentum) / 0.7))
    upside += 0.22 * max(0.0, min(1.5, (move3 - 1.4) / 1.6))
    upside += 0.24 * max(0.0, min(1.5, (move6 - 1.8) / 1.6))
    upside += 0.28 * max(0.0, min(1.5, (move12 - 2.0) / 1.8))
    upside += 0.20 * max(0.0, min(1.5, (move24 - 2.0) / 2.0))
    downside += 0.18 * max(0.0, min(1.5, (move24 - 3.2) / 1.5))
    upside += 0.12 * max(0.0, min(1.0, up_boundary / 2.5))
    downside += 0.12 * max(0.0, min(1.0, down_boundary / 2.0))
    if volatility >= 0.55 and upside > 0.7:
        upside += 0.14
        reasons.append("volatility_supports_upside_exit")
    if volatility >= 0.75 and move12 >= 2.6:
        downside += 0.16
        reasons.append("high_volatility_risk")

    symbol_up_bias = float(len(symbol_up_rows)) / float(max(1, len(symbol_rows)))
    symbol_down_bias = float(len(symbol_down_rows)) / float(max(1, len(symbol_rows)))
    regime_up_bias = float(len(regime_up_rows)) / float(max(1, len(regime_up_rows) + len(regime_down_rows)))
    regime_down_bias = float(len(regime_down_rows)) / float(max(1, len(regime_up_rows) + len(regime_down_rows)))
    upside += 0.20 * symbol_up_bias
    upside += 0.10 * regime_up_bias
    downside += 0.18 * symbol_down_bias
    downside += 0.10 * regime_down_bias

    if move12 >= 2.2 and move24 <= 2.9 and signal_margin <= 0.45:
        reasons.append("take_profit_profile")
    if move12 >= 2.0 and volatility >= 0.55 and signal_margin >= 0.40:
        reasons.append("trailing_profile")
    if move12 >= 2.6 and volatility >= 0.60 and signal_margin >= 0.46:
        reasons.append("risk_cut_profile")

    return {
        "upside_score": round(float(upside), 6),
        "downside_score": round(float(downside), 6),
        "direction_score_gap": round(float(upside - downside), 6),
        "upside_boundary_distance_pct": round(float(up_boundary), 6),
        "downside_boundary_distance_pct": round(float(down_boundary), 6),
        "reasons": reasons,
    }


def _crypto_predict_historical_replay_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
    predictor_variant: str = "candidate",
) -> Dict[str, Any]:
    working = [dict(r) for r in (train_rows[-260:] if len(train_rows) > 260 else train_rows) if _is_historical_strategy_replay_row(r)]
    sym = _s(candidate.get("symbol", "")).upper()
    signal_side = _s(candidate.get("signal_side", "")).lower()
    adapter = _s(candidate.get("strategy_adapter_used", ""))
    if not working:
        entry = _f(candidate.get("entry_price", 0.0), 0.0)
        return {
            "predicted_direction": "flat",
            "predicted_exit_trigger": "Unknown",
            "predicted_hold_hours": 12.0,
            "predicted_exit_price": float(entry),
            "predicted_confidence": 0.5,
            "trigger_scores": {"Unknown": 0.0},
            "direction_scores": {"flat": 0.0},
            "trigger_margin": 0.0,
            "direction_margin": 0.0,
            "predictor_source_mode": "historical_strategy_replay",
            "historical_replay_similarity_support": {},
        }

    feature_scales = {key: _historical_replay_feature_scale(working, key) for key, _weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES}
    trigger_scores: Dict[str, float] = {key: 0.0 for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    trigger_similarity_sums: Dict[str, float] = {key: 0.0 for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    trigger_vote_counts: Dict[str, int] = {key: 0 for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    direction_scores: Dict[str, float] = {"up": 0.0, "down": 0.0, "flat": 0.0}
    class_hold_hours: Dict[str, List[float]] = {key: [] for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    class_returns: Dict[str, List[float]] = {key: [] for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    symbol_trigger_counts: Dict[str, int] = {key: 0 for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    support_examples: Dict[str, int] = {key: 0 for key in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}

    recent_rows = _recent_slice(working, 80)
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == sym]
    symbol_recent = _recent_slice(symbol_rows, 32)
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    prototypes = _historical_replay_trigger_prototypes(working)
    recent_cutoff = max(0, len(working) - 80)
    upside_diag = _historical_replay_upside_score(candidate, working, symbol=sym, regime=regime)

    for idx, row in enumerate(working):
        trigger = _s(row.get("actual_exit_trigger", "")) or "Unknown"
        if trigger not in trigger_scores:
            continue
        dist = 0.0
        feat_used = 0
        for key, weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES:
            scale = max(1e-6, feature_scales.get(key, 1.0))
            cand_val = _f(candidate.get(key, 0.0), 0.0)
            row_val = _f(row.get(key, 0.0), 0.0)
            dist += weight * abs(cand_val - row_val) / scale
            feat_used += 1
        similarity = 1.0 / (1.0 + (dist / max(1, feat_used)))
        bonus = 1.0
        if _s(row.get("symbol", "")).upper() == sym:
            bonus += 0.65
            symbol_trigger_counts[trigger] += 1
        if _s(row.get("regime", "")) == regime:
            bonus += 0.15
        if signal_side and _s(row.get("signal_side", "")).lower() == signal_side:
            bonus += 0.12
        if adapter and _s(row.get("strategy_adapter_used", "")) == adapter:
            bonus += 0.08
        if idx >= recent_cutoff:
            bonus += 0.10
        vote = similarity * bonus
        trigger_similarity_sums[trigger] += vote
        trigger_vote_counts[trigger] += 1
        direction_scores[_s(row.get("actual_direction", "flat")).lower() or "flat"] += vote
        class_hold_hours[trigger].append(_f(row.get("hold_hours", 0.0), 0.0))
        class_returns[trigger].append(_f(row.get("pnl_pct", 0.0), 0.0))
        support_examples[trigger] += 1

    for trigger in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES:
        mean_similarity = trigger_similarity_sums[trigger] / max(1, trigger_vote_counts[trigger])
        trigger_scores[trigger] = 3.0 * mean_similarity
        same_symbol = len([r for r in symbol_rows if _s(r.get("actual_exit_trigger", "")) == trigger])
        recent_symbol = len([r for r in symbol_recent if _s(r.get("actual_exit_trigger", "")) == trigger])
        same_regime = len([r for r in regime_rows if _s(r.get("actual_exit_trigger", "")) == trigger])
        recent_global = len([r for r in recent_rows if _s(r.get("actual_exit_trigger", "")) == trigger])
        trigger_scores[trigger] += 0.95 * math.log1p(recent_symbol)
        trigger_scores[trigger] += 0.65 * math.log1p(same_symbol)
        trigger_scores[trigger] += 0.16 * math.log1p(same_regime)
        trigger_scores[trigger] += 0.06 * math.log1p(recent_global)
        proto = prototypes.get(trigger, {})
        if proto:
            proto_bonus = 0.0
            for key, _weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES:
                median = _f(proto.get(f"{key}_median", 0.0), 0.0)
                scale = max(1e-6, feature_scales.get(key, 1.0))
                proto_bonus += 1.0 / (1.0 + abs(_f(candidate.get(key, 0.0), 0.0) - median) / scale)
            trigger_scores[trigger] += 0.40 * (proto_bonus / max(1, len(_CRYPTO_HISTORICAL_REPLAY_FEATURES)))

    variant = _s(predictor_variant).lower() or "candidate"
    use_exit_shape = variant in {"candidate", "active", "exit_shape_enhanced", "label_compatible_v2", "label_compatible_v3_sequence"}
    use_label_compatible_v2 = variant == "label_compatible_v2"
    use_label_compatible_v3_sequence = variant == "label_compatible_v3_sequence"
    momentum = _f(candidate.get("trend_momentum_score", 0.0), 0.0)
    volatility = _f(candidate.get("recent_volatility", 0.0), 0.0)
    signal_margin = _f(candidate.get("signal_margin", 0.0), 0.0)
    move_6 = _f(candidate.get("recent_return_6", 0.0), 0.0)
    move_12 = _f(candidate.get("recent_return_12", 0.0), 0.0)
    move_24 = _f(candidate.get("recent_return_24", 0.0), 0.0)
    candle_move = _f(candidate.get("current_candle_pct_move", 0.0), 0.0)
    upside_score = _f(upside_diag.get("upside_score", 0.0), 0.0)
    downside_score = _f(upside_diag.get("downside_score", 0.0), 0.0)
    up_gap = _f(upside_diag.get("direction_score_gap", 0.0), 0.0)
    up_recovery_applied = False
    up_recovery_reason = ""
    take_profit_candidate = False
    stale_override_reason = ""
    risk_trailing_resolver_applied = False
    risk_trailing_resolver_reason = ""
    risk_trailing_sequence_score_risk = 0.0
    risk_trailing_sequence_score_trailing = 0.0

    if use_label_compatible_v2 or use_label_compatible_v3_sequence:
        risk_cut_touched = bool(candidate.get("risk_cut_touched", False))
        take_profit_touched = bool(candidate.get("take_profit_touched", False))
        trailing_armed = bool(candidate.get("trailing_armed", False))
        softened = bool(candidate.get("favorable_then_softened_flag", False))
        mfe_pct = _f(candidate.get("max_favorable_excursion_pct", 0.0), 0.0)
        mae_pct = abs(_f(candidate.get("max_adverse_excursion_pct", 0.0), 0.0))
        dd_peak = abs(_f(candidate.get("drawdown_from_peak_pct", 0.0), 0.0))
        pullback = _f(candidate.get("trailing_pullback_pct", 0.0), 0.0)
        bars_in_trade = _f(candidate.get("bars_in_trade", 0.0), 0.0)
        exit_m3 = _f(candidate.get("exit_momentum_3", 0.0), 0.0)
        exit_m6 = _f(candidate.get("exit_momentum_6", 0.0), 0.0)
        trailing_before_risk_cut = bool(candidate.get("trailing_before_risk_cut", False))
        risk_cut_before_trailing = bool(candidate.get("risk_cut_before_trailing", False))
        risk_cut_after_trailing_arm = bool(candidate.get("risk_cut_after_trailing_arm", False))
        risk_cut_after_peak = bool(candidate.get("risk_cut_after_peak", False))
        trailing_valid_before_risk = bool(candidate.get("trailing_valid_before_risk", False))
        same_candle_risk_and_trailing = bool(candidate.get("same_candle_risk_and_trailing", False))
        exit_close_position_in_candle_range = _f(candidate.get("exit_close_position_in_candle_range", 0.5), 0.5)
        peak_to_exit_velocity = abs(_f(candidate.get("peak_to_exit_velocity_pct_per_bar", 0.0), 0.0))
        risk_breach_depth = _f(candidate.get("risk_breach_depth_pct", 0.0), 0.0)
        bars_from_trailing_arm_to_risk_cut = _f(candidate.get("bars_from_trailing_arm_to_risk_cut", -1.0), -1.0)
        risk_support = sum(1 for r in working if _s(r.get("actual_exit_trigger", "")) == "Risk Cut")
        tp_support = sum(1 for r in working if _s(r.get("actual_exit_trigger", "")) == "Take Profit")
        trailing_support = sum(1 for r in working if _s(r.get("actual_exit_trigger", "")) == "Trailing")
        trigger_scores["Risk Cut"] += (1.25 if risk_cut_touched else 0.0) + (0.35 * max(0.0, (mae_pct - 1.6) / 1.2)) + (0.25 * max(0.0, -exit_m3 / 1.5)) + (0.20 * max(0.0, -exit_m6 / 1.5))
        trigger_scores["Take Profit"] += (1.10 if take_profit_touched else 0.0) + (0.30 * max(0.0, (mfe_pct - 3.0) / 2.0)) + (0.15 * max(0.0, -exit_m3 / 1.2))
        if tp_support < 10:
            trigger_scores["Take Profit"] *= 0.70
        trigger_scores["Trailing"] += (1.10 if trailing_armed else 0.0) + (0.90 if softened else 0.0) + (0.25 * max(0.0, (pullback - 0.8) / 1.0)) + (0.20 * max(0.0, (dd_peak - 0.8) / 1.0))
        if trailing_support < 14:
            trigger_scores["Trailing"] *= 0.85
        if not risk_cut_touched and not take_profit_touched and not (trailing_armed and softened):
            trigger_scores["Stale Alignment"] += 1.05
        trigger_scores["Stale Alignment"] += 0.22 * max(0.0, (bars_in_trade - 14.0) / 8.0)
        trigger_scores["Stale Alignment"] += 0.20 * max(0.0, (0.10 - momentum) / 0.30)
        trigger_scores["Stale Alignment"] += 0.15 * max(0.0, (0.18 - signal_margin) / 0.20)
        if risk_cut_touched:
            trigger_scores["Stale Alignment"] -= 0.55
        if take_profit_touched:
            trigger_scores["Stale Alignment"] -= 0.40
        if trailing_armed and softened:
            trigger_scores["Stale Alignment"] -= 0.45
        if upside_score >= 1.0 and trailing_armed and softened:
            up_recovery_applied = True
            up_recovery_reason = "label_compatible_trailing_recovery"
        if take_profit_touched and mfe_pct >= 4.0:
            take_profit_candidate = True
        if use_label_compatible_v3_sequence:
            if risk_cut_before_trailing:
                risk_trailing_sequence_score_risk += 2.25
                risk_trailing_resolver_reason = "risk_cut_before_trailing"
            if risk_cut_touched and risk_breach_depth >= 0.20:
                risk_trailing_sequence_score_risk += 1.35 + min(1.25, risk_breach_depth / 1.5)
                if not risk_trailing_resolver_reason:
                    risk_trailing_resolver_reason = "risk_breach_depth"
            if risk_cut_after_trailing_arm and peak_to_exit_velocity >= 0.45:
                risk_trailing_sequence_score_risk += 1.55
                risk_trailing_sequence_score_trailing -= 0.35
                if not risk_trailing_resolver_reason:
                    risk_trailing_resolver_reason = "risk_after_trailing_sharp_reversal"
            if trailing_before_risk_cut and trailing_valid_before_risk and softened and risk_breach_depth <= 0.05:
                risk_trailing_sequence_score_trailing += 1.85
                if not risk_trailing_resolver_reason:
                    risk_trailing_resolver_reason = "trailing_before_risk_without_breach"
            if trailing_armed and exit_close_position_in_candle_range <= 0.22 and mae_pct >= 1.8:
                risk_trailing_sequence_score_trailing -= 0.95
                risk_trailing_sequence_score_risk += 0.95
                if not risk_trailing_resolver_reason:
                    risk_trailing_resolver_reason = "exit_closed_near_low_with_deep_adverse_move"
            if same_candle_risk_and_trailing:
                if exit_close_position_in_candle_range <= 0.35:
                    risk_trailing_sequence_score_risk += 0.75
                    if not risk_trailing_resolver_reason:
                        risk_trailing_resolver_reason = "same_candle_low_close_bias_risk"
                else:
                    risk_trailing_sequence_score_trailing += 0.55
                    if not risk_trailing_resolver_reason:
                        risk_trailing_resolver_reason = "same_candle_high_close_bias_trailing"
            if risk_cut_after_peak and peak_to_exit_velocity >= 0.60:
                risk_trailing_sequence_score_risk += 0.85
            if bars_from_trailing_arm_to_risk_cut >= 0 and bars_from_trailing_arm_to_risk_cut <= 2:
                risk_trailing_sequence_score_risk += 0.65
            trigger_scores["Risk Cut"] += risk_trailing_sequence_score_risk
            trigger_scores["Trailing"] += risk_trailing_sequence_score_trailing
            risk_trailing_resolver_applied = abs(risk_trailing_sequence_score_risk) > 1e-9 or abs(risk_trailing_sequence_score_trailing) > 1e-9
    else:
        if use_exit_shape and momentum >= 2.6 and volatility >= 0.55 and signal_margin >= 0.43 and move_12 >= 2.5:
            trigger_scores["Risk Cut"] += 0.75
            if downside_score >= upside_score:
                stale_override_reason = "stale_overridden_by_risk_cut"
        if use_exit_shape and momentum >= 2.1 and momentum <= 2.8 and volatility >= 0.45 and volatility <= 0.80 and candle_move >= 0.9:
            trigger_scores["Trailing"] += 0.65
            if upside_score >= 0.75:
                stale_override_reason = "stale_overridden_by_trailing"
        if use_exit_shape and momentum <= 2.45 and signal_margin <= 0.46 and volatility <= 0.72 and move_6 >= 1.8:
            trigger_scores["Take Profit"] += 0.55
            take_profit_candidate = True
        if use_exit_shape and move_24 >= 2.0 and signal_margin <= 0.52:
            trigger_scores["Stale Alignment"] += 0.35
        if use_exit_shape and signal_margin >= 0.50 and momentum >= 2.7:
            trigger_scores["Stale Alignment"] -= 0.10
            trigger_scores["Risk Cut"] += 0.20
        if use_exit_shape and upside_score >= 1.10 and downside_score <= 0.40:
            trigger_scores["Trailing"] += 0.45
            up_recovery_applied = True
            up_recovery_reason = "strong_upside_recovery"
        if use_exit_shape and take_profit_candidate and upside_score >= 1.05 and volatility <= 0.58 and move_24 <= 2.5:
            trigger_scores["Take Profit"] += 0.50
            up_recovery_applied = True
            up_recovery_reason = "take_profit_recovery"
        if use_exit_shape and up_gap >= 0.32 and upside_score >= 1.0 and downside_score <= 0.35 and trigger_scores["Stale Alignment"] >= max(trigger_scores["Trailing"], trigger_scores["Take Profit"]):
            trigger_scores["Stale Alignment"] -= 0.35
            trigger_scores["Trailing"] += 0.20
            stale_override_reason = "stale_overridden_by_upside"
        elif trigger_scores["Stale Alignment"] > max(trigger_scores["Risk Cut"], trigger_scores["Trailing"], trigger_scores["Take Profit"]) and symbol_trigger_counts.get("Stale Alignment", 0) <= 0:
            stale_override_reason = "stale_global_prior_only"

    top_triggers = sorted(trigger_scores.items(), key=lambda kv: kv[1], reverse=True)
    pred_trigger = top_triggers[0][0] if top_triggers else "Unknown"
    trigger_margin = (top_triggers[0][1] - top_triggers[1][1]) if len(top_triggers) > 1 else top_triggers[0][1]

    direction_forced_by_trigger = False
    trigger_direction_prior_applied = 0.0
    stale_up_score = 0.0
    stale_down_score = 0.0
    if move_24 <= 2.7:
        stale_up_score += 0.45
    if move_12 <= 2.9:
        stale_up_score += 0.35
    if momentum <= 2.35:
        stale_up_score += 0.45
    if volatility <= 0.55:
        stale_up_score += 0.25
    if candle_move <= 0.75:
        stale_up_score += 0.20
    if signal_margin <= 0.42:
        stale_up_score += 0.20
    if move_6 >= 2.6:
        stale_down_score += 0.35
    if move_24 >= 3.0:
        stale_down_score += 0.30
    if momentum >= 2.55:
        stale_down_score += 0.40
    if signal_margin >= 0.48:
        stale_down_score += 0.25
    if volatility >= 0.60:
        stale_down_score += 0.20
    direction_scores["up"] += stale_up_score
    direction_scores["down"] += stale_down_score
    direction_scores["up"] += 0.35 * upside_score
    direction_scores["down"] += 0.60 * downside_score
    trigger_direction_prior_weight = 0.10 if not use_exit_shape else (0.12 if use_label_compatible_v2 else 0.25)
    if pred_trigger == "Risk Cut":
        trigger_direction_prior_applied = -trigger_direction_prior_weight
        direction_scores["down"] += abs(trigger_direction_prior_applied)
    elif pred_trigger in {"Trailing", "Take Profit"}:
        trigger_direction_prior_applied = trigger_direction_prior_weight
        direction_scores["up"] += trigger_direction_prior_applied
    pred_dir = "up" if direction_scores["up"] >= direction_scores["down"] else "down"
    if pred_trigger in {"Stale Alignment", "Risk Cut"} and up_gap >= 0.55 and upside_score >= 1.20 and downside_score <= 0.35 and signal_margin <= 0.46:
        pred_dir = "up"
        direction_scores["up"] += 0.65
        up_recovery_applied = True
        if not up_recovery_reason:
            up_recovery_reason = "direction_upside_recovery"

    top_dirs = sorted(direction_scores.items(), key=lambda kv: kv[1], reverse=True)
    direction_margin = (top_dirs[0][1] - top_dirs[1][1]) if len(top_dirs) > 1 else top_dirs[0][1]

    pred_hold = 12.0
    if class_hold_hours.get(pred_trigger):
        pred_hold = _median(class_hold_hours[pred_trigger], default=12.0)
    pred_return_pct = 0.0
    if class_returns.get(pred_trigger):
        pred_return_pct = _median(class_returns[pred_trigger], default=0.0)
    entry = _f(candidate.get("entry_price", 0.0), 0.0)
    pred_exit_price = entry * (1.0 + (pred_return_pct / 100.0)) if entry > 0.0 else 0.0

    total_trigger_score = max(1e-9, sum(max(0.0, v) for v in trigger_scores.values()))
    share = max(0.0, top_triggers[0][1]) / total_trigger_score if top_triggers else 0.0
    same_symbol_support = max(0, symbol_trigger_counts.get(pred_trigger, 0))
    support_term = min(1.0, math.log1p(same_symbol_support + support_examples.get(pred_trigger, 0)) / math.log1p(40.0))
    margin_term = 1.0 / (1.0 + math.exp(-trigger_margin))
    dir_term = 1.0 / (1.0 + math.exp(-direction_margin))
    conf = min(0.95, max(0.18, (0.15 + (0.35 * share) + (0.20 * margin_term) + (0.15 * dir_term) + (0.15 * support_term))))

    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trigger,
        "predicted_hold_hours": round(float(pred_hold), 6),
        "predicted_exit_price": round(float(pred_exit_price), 10),
        "predicted_confidence": round(float(conf), 6),
        "trigger_scores": {k: round(float(v), 6) for k, v in trigger_scores.items()},
        "direction_scores": {k: round(float(v), 6) for k, v in direction_scores.items()},
        "winning_trigger_score": round(float(top_triggers[0][1] if top_triggers else 0.0), 6),
        "runner_up_trigger_score": round(float(top_triggers[1][1] if len(top_triggers) > 1 else 0.0), 6),
        "trigger_margin": round(float(trigger_margin), 6),
        "winning_direction_score": round(float(top_dirs[0][1] if top_dirs else 0.0), 6),
        "runner_up_direction_score": round(float(top_dirs[1][1] if len(top_dirs) > 1 else 0.0), 6),
        "direction_margin": round(float(direction_margin), 6),
        "upside_score": round(float(upside_score), 6),
        "downside_score": round(float(downside_score), 6),
        "direction_score_gap": round(float(up_gap), 6),
        "up_recovery_reason": up_recovery_reason,
        "up_recovery_applied": bool(up_recovery_applied),
        "stale_override_reason": stale_override_reason,
        "take_profit_score": round(float(trigger_scores.get("Take Profit", 0.0)), 6),
        "risk_cut_score": round(float(trigger_scores.get("Risk Cut", 0.0)), 6),
        "stale_score": round(float(trigger_scores.get("Stale Alignment", 0.0)), 6),
        "trailing_score": round(float(trigger_scores.get("Trailing", 0.0)), 6),
        "take_profit_candidate_flag": bool(take_profit_candidate),
        "direction_forced_by_trigger": bool(direction_forced_by_trigger),
        "trigger_direction_prior_applied": round(float(trigger_direction_prior_applied), 6),
        "trigger_direction_prior_weight": round(float(abs(trigger_direction_prior_weight)), 6),
        "direction_independent_score_up": round(float(direction_scores.get("up", 0.0) - max(0.0, trigger_direction_prior_applied)), 6),
        "direction_independent_score_down": round(float(direction_scores.get("down", 0.0) - max(0.0, -trigger_direction_prior_applied)), 6),
        "independent_direction_score_up": round(float(direction_scores.get("up", 0.0) - max(0.0, trigger_direction_prior_applied)), 6),
        "independent_direction_score_down": round(float(direction_scores.get("down", 0.0) - max(0.0, -trigger_direction_prior_applied)), 6),
        "exit_shape_predictive_mode": "active" if variant in {"label_compatible_v2", "label_compatible_v3_sequence"} else ("active" if use_exit_shape else "diagnostic_only"),
        "risk_trailing_resolver_applied": bool(risk_trailing_resolver_applied),
        "risk_trailing_resolver_reason": risk_trailing_resolver_reason,
        "risk_trailing_sequence_score_risk": round(float(risk_trailing_sequence_score_risk), 6),
        "risk_trailing_sequence_score_trailing": round(float(risk_trailing_sequence_score_trailing), 6),
        "upside_boundary_distance_pct": round(float(_f(upside_diag.get("upside_boundary_distance_pct", 0.0), 0.0)), 6),
        "downside_boundary_distance_pct": round(float(_f(upside_diag.get("downside_boundary_distance_pct", 0.0), 0.0)), 6),
        "predictor_source_mode": "historical_strategy_replay",
        "predictor_variant": variant,
        "historical_replay_similarity_support": {
            cls: {
                "same_symbol_support_count": int(len([r for r in symbol_rows if _s(r.get('actual_exit_trigger', '')) == cls])),
                "recent_symbol_support_count": int(len([r for r in symbol_recent if _s(r.get('actual_exit_trigger', '')) == cls])),
                "same_regime_support_count": int(len([r for r in regime_rows if _s(r.get('actual_exit_trigger', '')) == cls])),
                "support_count": int(support_examples.get(cls, 0)),
            }
            for cls in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES
        },
    }


def _stock_predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
    predictor_variant: str = "candidate",
) -> Dict[str, Any]:
    working = list(train_rows[-220:]) if len(train_rows) > 220 else list(train_rows)
    sym = _s(candidate.get("symbol", "")).upper()
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == sym]
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    recent_rows = _recent_slice(working, 80)
    cohorts = [rows for rows in [symbol_rows, regime_rows, recent_rows, working] if rows]
    filtered = symbol_rows if len(symbol_rows) >= 8 else regime_rows if len(regime_rows) >= 12 else recent_rows
    if not filtered:
        filtered = list(working)
    if not filtered:
        entry = _f(candidate.get("entry_price", 0.0), 0.0)
        return {
            "predicted_direction": "flat",
            "predicted_exit_trigger": "Unknown",
            "predicted_hold_hours": max(1.0, _f(candidate.get("hold_hours", 6.0), 6.0)),
            "predicted_exit_price": float(entry),
            "predicted_confidence": 0.5,
            "predictor_mode": "stock_historical_replay",
        }
    mom = _f(candidate.get("trend_momentum_score", 0.0), 0.0)
    ret6 = _f(candidate.get("recent_return_6", 0.0), 0.0)
    ret24 = _f(candidate.get("recent_return_24", 0.0), 0.0)
    vol = _f(candidate.get("recent_volatility", 0.0), 0.0)
    signal_margin = _f(candidate.get("signal_margin", 0.0), 0.0)
    peak_profit_pct = _f(candidate.get("peak_profit_pct", 0.0), 0.0)
    drawdown_from_peak_pct = abs(_f(candidate.get("drawdown_from_peak_pct", 0.0), 0.0))
    trailing_armed = bool(candidate.get("trailing_armed", False))
    favorable_then_softened = bool(candidate.get("favorable_then_softened_flag", False))
    stale_hold_profile = bool(candidate.get("stale_hold_profile", False))
    volatility_expansion_pct = _f(candidate.get("volatility_expansion_pct", 0.0), 0.0)
    trend_decay_after_peak = _f(candidate.get("trend_decay_after_peak", 0.0), 0.0)
    gap_against_position_pct = _f(candidate.get("gap_against_position_pct", 0.0), 0.0)
    exit_m3 = _f(candidate.get("exit_momentum_3", 0.0), 0.0)
    exit_m6 = _f(candidate.get("exit_momentum_6", 0.0), 0.0)
    bars_since_peak = int(_f(candidate.get("bars_since_peak", 0.0), 0.0))
    bars_in_trade = max(1, int(_f(candidate.get("bars_in_trade", candidate.get("hold_hours", 0.0)), 0.0)))
    max_favorable_excursion_pct = _f(candidate.get("max_favorable_excursion_pct", 0.0), 0.0)
    max_adverse_excursion_pct = abs(_f(candidate.get("max_adverse_excursion_pct", 0.0), 0.0))
    stats = _weighted_trade_stats(filtered)
    symbol_stats = _weighted_trade_stats(symbol_rows or filtered)
    regime_stats = _weighted_trade_stats(regime_rows or filtered)
    recent_stats = _weighted_trade_stats(recent_rows or filtered)
    pnl_cohorts: List[Tuple[float, Dict[str, float]]] = []
    for weight, rows_cohort, stats_cohort in [
        (1.25, symbol_rows, symbol_stats),
        (0.95, regime_rows, regime_stats),
        (0.80, recent_rows, recent_stats),
    ]:
        if rows_cohort:
            pnl_cohorts.append((weight, stats_cohort))
    up_score = (0.55 * stats.get("up_rate", 0.0)) + (0.25 * max(0.0, mom / 4.0)) + (0.15 * max(0.0, ret24 / 5.0)) + (0.05 * max(0.0, signal_margin))
    down_score = (0.55 * stats.get("down_rate", 0.0)) + (0.25 * max(0.0, -mom / 4.0)) + (0.15 * max(0.0, -ret24 / 5.0)) + (0.05 * max(0.0, vol / 2.5))
    guard_applied = False
    up_dampened = False
    if _s(predictor_variant).lower() in {"candidate", "stabilized"}:
        down_case_score = (0.45 * stats.get("down_rate", 0.0)) + (0.20 * max(0.0, -ret6 / 3.0)) + (0.20 * max(0.0, -ret24 / 4.0)) + (0.15 * max(0.0, vol / 2.0))
        up_case_score = (0.45 * stats.get("up_rate", 0.0)) + (0.20 * max(0.0, ret6 / 3.0)) + (0.20 * max(0.0, ret24 / 4.0)) + (0.15 * max(0.0, signal_margin))
        guard_reason = ""
        same_symbol_up_support = sum(1 for r in symbol_rows if _s(r.get("actual_direction", "")).lower() == "up")
        same_symbol_down_support = sum(1 for r in symbol_rows if _s(r.get("actual_direction", "")).lower() == "down")
        same_regime_up_support = sum(1 for r in regime_rows if _s(r.get("actual_direction", "")).lower() == "up")
        same_regime_down_support = sum(1 for r in regime_rows if _s(r.get("actual_direction", "")).lower() == "down")
        if down_case_score >= up_case_score and mom <= 0.25:
            down_score += 0.35
            guard_applied = True
            guard_reason = "weak_negative_returns_high_volatility"
        score_gap = abs(up_case_score - down_case_score)
        if score_gap <= 0.18:
            if (same_symbol_down_support + same_regime_down_support) > (same_symbol_up_support + same_regime_up_support):
                down_score += 0.18
                guard_applied = True
                guard_reason = guard_reason or "close_scores_prefer_down_support"
            elif (same_symbol_up_support + same_regime_up_support) > (same_symbol_down_support + same_regime_down_support):
                up_score += 0.12
        if ret6 <= 0.0 and ret24 <= 0.0 and vol >= 0.65 and same_symbol_up_support < max(3, same_symbol_down_support):
            up_score -= 0.16
            up_dampened = True
            guard_applied = True
            guard_reason = guard_reason or "up_dampened_by_down_regime"
    else:
        down_case_score = down_score
        up_case_score = up_score
        guard_reason = ""
    trailing_score = (0.58 * stats.get("trailing_rate", 0.0)) + (0.24 * max(0.0, ret6 / 4.0)) + (0.18 * max(0.0, vol / 2.3))
    stale_score = (0.58 * stats.get("stale_rate", 0.0)) + (0.22 * max(0.0, -ret6 / 3.0)) + (0.20 * max(0.0, vol / 2.3 if ret24 < 0.5 else 0.0))
    trigger_reason = "balanced"
    stale_override_applied = False
    trailing_override_applied = False
    exit_shape_reason = ""
    down_exit_shape_reason = ""
    if ret24 <= 0.0 or mom <= 0.2:
        stale_score += 0.20
        trigger_reason = "weak_or_negative_momentum"
        stale_override_applied = True
    if ret6 <= 0.15 and ret24 <= 0.40 and vol >= 0.55 and signal_margin <= 0.22:
        stale_score += 0.14
        trigger_reason = "mixed_or_fading_returns"
        stale_override_applied = True
    if ret6 >= 0.8 and ret24 >= 1.2 and vol >= 0.35:
        trailing_score += 0.18
        trigger_reason = "favorable_move_profile"
        trailing_override_applied = True
    if ret6 >= 1.0 and ret24 >= 1.4 and mom >= 0.45 and signal_margin >= 0.24:
        trailing_score += 0.12
        trigger_reason = "favorable_continuation_profile"
        trailing_override_applied = True
    if vol >= 0.70 and (ret6 <= 0.25 or ret24 <= 0.60) and mom <= 0.30:
        stale_score += 0.10
        trigger_reason = "elevated_vol_without_follow_through"
        stale_override_applied = True
    if trailing_armed and favorable_then_softened and peak_profit_pct >= 1.5 and drawdown_from_peak_pct >= 1.2:
        trailing_score += 0.22
        trigger_reason = "exit_shape_trailing_pullback"
        exit_shape_reason = "favorable_move_then_pullback"
        trailing_override_applied = True
    if stale_hold_profile or (peak_profit_pct < 1.0 and trend_decay_after_peak <= -0.8):
        stale_score += 0.20
        trigger_reason = "exit_shape_stale_hold_decay"
        exit_shape_reason = "weak_favorable_move_or_decay"
        stale_override_applied = True
    if volatility_expansion_pct >= 25.0 and favorable_then_softened and drawdown_from_peak_pct >= 1.0:
        trailing_score += 0.10
        trigger_reason = "exit_shape_volatility_followthrough"
        exit_shape_reason = "volatility_expansion_with_pullback"
        trailing_override_applied = True
    if volatility_expansion_pct >= 25.0 and not favorable_then_softened and (ret6 <= 0.2 or ret24 <= 0.5):
        stale_score += 0.12
        trigger_reason = "exit_shape_chop_without_followthrough"
        exit_shape_reason = "volatility_expansion_without_followthrough"
        stale_override_applied = True
    trigger_scores = {
        "Trailing": trailing_score,
        "Stale Alignment": stale_score,
        "Unknown": 0.10,
    }
    pred_dir = "up" if up_score >= down_score else "down"
    if gap_against_position_pct >= 0.35:
        down_score += 0.12
        down_exit_shape_reason = "gap_against_position"
    if trend_decay_after_peak <= -1.0:
        down_score += 0.14
        down_exit_shape_reason = down_exit_shape_reason or "trend_decay_after_peak"
    if volatility_expansion_pct >= 25.0 and ret6 <= 0.0 and ret24 <= 0.5:
        down_score += 0.10
        down_exit_shape_reason = down_exit_shape_reason or "volatility_expansion_with_weak_returns"
    if _f(candidate.get("max_adverse_excursion_pct", 0.0), 0.0) <= -1.2:
        down_score += 0.12
        down_exit_shape_reason = down_exit_shape_reason or "large_adverse_excursion"
    pred_dir = "up" if up_score >= down_score else "down"
    pred_trigger = max(trigger_scores.keys(), key=lambda k: trigger_scores[k])
    hold_h = max(0.25, _median([_f(r.get("hold_hours", 0.0), 0.0) for r in filtered], default=6.0))
    med_ret = _median([_trade_return_pct(r) for r in filtered], default=0.0) / 100.0
    entry_px = _f(candidate.get("entry_price", 0.0), 0.0)
    trigger_is_stale = pred_trigger == "Stale Alignment"
    hold_long = hold_h >= max(18.0, _cohort_weighted_value(pnl_cohorts, "q75_hold_h", default=18.0))
    mixed_returns = ret6 <= 0.35 and ret24 <= 0.90
    high_vol_long_hold_penalty = 0.0
    drawdown_quality_penalty = 0.0
    symbol_pnl_history_penalty = 0.0
    pnl_quality_reasons: List[str] = []
    weak_window_guard_reason = ""
    symbol_pnl_up_rate = _stock_directional_pnl_rate(symbol_rows, direction="up")
    regime_pnl_up_rate = _stock_directional_pnl_rate(regime_rows, direction="up")
    recent_pnl_up_rate = _stock_directional_pnl_rate(recent_rows, direction="up")
    blended_symbol_pnl = _blend_probability(pnl_cohorts, "up_rate") if pnl_cohorts else stats.get("up_rate", 0.0)
    trade_quality_base = (
        0.18
        + (0.18 * max(0.0, min(1.0, ret6 / 2.5)))
        + (0.18 * max(0.0, min(1.0, ret24 / 4.0)))
        + (0.12 * max(0.0, min(1.0, signal_margin / 0.55)))
        + (0.12 * max(0.0, min(1.0, mom / 3.0)))
        + (0.10 * max(0.0, min(1.0, max_favorable_excursion_pct / 3.0)))
        + (0.08 if trailing_armed else 0.0)
        + (0.06 if favorable_then_softened else 0.0)
    )
    if vol >= 0.75 and hold_long and mixed_returns:
        high_vol_long_hold_penalty += 0.26
        pnl_quality_reasons.append("high_vol_long_hold_mixed_returns")
        weak_window_guard_reason = weak_window_guard_reason or "high_vol_long_hold_mixed_returns"
    elif vol >= 0.60 and hold_h >= 14.0 and mixed_returns:
        high_vol_long_hold_penalty += 0.14
        pnl_quality_reasons.append("moderate_vol_long_hold_mixed_returns")
        weak_window_guard_reason = weak_window_guard_reason or "moderate_vol_long_hold_mixed_returns"
    if drawdown_from_peak_pct >= 2.0:
        drawdown_quality_penalty += 0.24
        pnl_quality_reasons.append("large_drawdown_from_peak")
        weak_window_guard_reason = weak_window_guard_reason or "large_drawdown_from_peak"
    elif drawdown_from_peak_pct >= 1.2:
        drawdown_quality_penalty += 0.14
        pnl_quality_reasons.append("moderate_drawdown_from_peak")
        weak_window_guard_reason = weak_window_guard_reason or "moderate_drawdown_from_peak"
    if stale_hold_profile:
        drawdown_quality_penalty += 0.16
        pnl_quality_reasons.append("stale_hold_profile")
        weak_window_guard_reason = weak_window_guard_reason or "stale_hold_profile"
    if trend_decay_after_peak <= -1.0:
        drawdown_quality_penalty += 0.08
        pnl_quality_reasons.append("trend_decay_after_peak")
    if exit_m3 <= -0.8 or exit_m6 <= -1.0:
        drawdown_quality_penalty += 0.08
        pnl_quality_reasons.append("exit_momentum_decay")
    if max_adverse_excursion_pct >= 1.6:
        drawdown_quality_penalty += 0.08
        pnl_quality_reasons.append("large_adverse_excursion")
    if len(symbol_rows) >= 8 and symbol_pnl_up_rate < 0.50:
        symbol_pnl_history_penalty += min(0.18, (0.50 - symbol_pnl_up_rate) * 0.45)
        pnl_quality_reasons.append("same_symbol_pnl_history_weak")
        weak_window_guard_reason = weak_window_guard_reason or "same_symbol_pnl_history_weak"
    if len(regime_rows) >= 12 and regime_pnl_up_rate < 0.48:
        symbol_pnl_history_penalty += min(0.10, (0.48 - regime_pnl_up_rate) * 0.30)
        pnl_quality_reasons.append("same_regime_pnl_history_weak")
    if bars_since_peak >= 8 and drawdown_from_peak_pct >= 1.0:
        drawdown_quality_penalty += 0.06
        pnl_quality_reasons.append("late_trade_drawdown")
    trade_quality_score = max(0.0, min(1.0, trade_quality_base - (0.45 * high_vol_long_hold_penalty) - (0.40 * drawdown_quality_penalty)))
    pnl_quality_score = max(
        0.0,
        min(
            1.0,
            (0.35 * blended_symbol_pnl)
            + (0.20 * symbol_pnl_up_rate)
            + (0.12 * regime_pnl_up_rate)
            + (0.08 * recent_pnl_up_rate)
            + (0.15 * trade_quality_score)
            + (0.10 if trailing_armed and favorable_then_softened else 0.0)
            - high_vol_long_hold_penalty
            - drawdown_quality_penalty
            - symbol_pnl_history_penalty,
        ),
    )
    pnl_quality_reason = "clean_trend_followthrough"
    if pnl_quality_reasons:
        pnl_quality_reason = pnl_quality_reasons[0]
    pnl_quality_gate_applied = False
    weak_window_guard_applied = bool(high_vol_long_hold_penalty > 0.0 or drawdown_quality_penalty > 0.0 or symbol_pnl_history_penalty > 0.0)
    variant_name = _s(predictor_variant).lower()
    predicted_pnl_trend = "up" if pred_dir == "up" else "down"
    if variant_name == "stock_pnl_quality_v2":
        strong_weak_window = (
            pred_dir == "up"
            and (
                (drawdown_from_peak_pct >= 1.8 and bars_in_trade >= 18 and (stale_hold_profile or trigger_is_stale))
                or (vol >= 0.80 and hold_long and mixed_returns and drawdown_from_peak_pct >= 1.5)
                or (symbol_pnl_up_rate < 0.45 and drawdown_from_peak_pct >= 2.0 and bars_since_peak >= 6)
            )
        )
        if strong_weak_window and pnl_quality_score < 0.52:
            predicted_pnl_trend = "down"
            pnl_quality_gate_applied = True
            pnl_quality_reason = pnl_quality_reason or "weak_pnl_quality_gate"
        elif pred_dir == "up" and trigger_is_stale and pnl_quality_score < 0.42:
            predicted_pnl_trend = "down"
            pnl_quality_gate_applied = True
            pnl_quality_reason = pnl_quality_reason or "stale_trigger_pnl_quality_gate"
        elif pred_dir == "up" and trade_quality_score < 0.32 and drawdown_from_peak_pct >= 2.0:
            predicted_pnl_trend = "down"
            pnl_quality_gate_applied = True
            pnl_quality_reason = pnl_quality_reason or "weak_trade_quality_gate"
        if predicted_pnl_trend == "up":
            exit_px = entry_px * (1.0 + max(0.0005, abs(med_ret)))
        elif predicted_pnl_trend == "down":
            implied_down_move = max(0.0005, abs(_cohort_weighted_value(pnl_cohorts, "median_down_ret_pct", default=max(0.10, abs(med_ret * 100.0)))) / 100.0)
            exit_px = entry_px * (1.0 - implied_down_move)
        else:
            exit_px = entry_px
        conf = max(
            0.18,
            min(
                0.95,
                0.18
                + (0.24 * max(up_score, down_score))
                + (0.22 * max(trigger_scores.values()))
                + (0.16 * trade_quality_score)
                + (0.12 * pnl_quality_score)
                - (0.08 if pnl_quality_gate_applied else 0.0),
            ),
        )
    else:
        predicted_pnl_trend = "up" if pred_dir == "up" else "down"
        exit_px = entry_px * (1.0 + abs(med_ret)) if pred_dir == "up" else entry_px * (1.0 - abs(med_ret))
        conf = max(0.18, min(0.95, 0.20 + (0.30 * max(up_score, down_score)) + (0.25 * max(trigger_scores.values()))))
    sorted_triggers = sorted(trigger_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trigger,
        "predicted_hold_hours": round(float(hold_h), 6),
        "predicted_exit_price": round(float(max(1e-12, exit_px)), 10),
        "predicted_pnl_trend": predicted_pnl_trend,
        "predicted_confidence": round(float(conf), 6),
        "trigger_scores": {k: round(float(v), 6) for k, v in trigger_scores.items()},
        "direction_scores": {"up": round(float(up_score), 6), "down": round(float(down_score), 6)},
        "trigger_margin": round(float(sorted_triggers[0][1] - sorted_triggers[1][1] if len(sorted_triggers) > 1 else sorted_triggers[0][1]), 6),
        "direction_margin": round(float(abs(up_score - down_score)), 6),
        "stock_direction_score_up": round(float(up_score), 6),
        "stock_direction_score_down": round(float(down_score), 6),
        "stock_direction_score_gap": round(float(up_score - down_score), 6),
        "predictor_mode": "stock_historical_replay",
        "predictor_variant": variant_name or "candidate",
        "stock_up_bias_guard_applied": bool(guard_applied),
        "stock_down_case_guard_reason": guard_reason,
        "stock_down_case_guard_applied": bool(guard_applied),
        "stock_up_dampened_by_down_regime": bool(up_dampened),
        "stock_down_case_score": round(float(down_case_score), 6),
        "stock_up_case_score": round(float(up_case_score), 6),
        "stock_trigger_separation_reason": trigger_reason,
        "stock_stale_vs_trailing_exit_shape_reason": exit_shape_reason,
        "stock_trailing_score": round(float(trailing_score), 6),
        "stock_stale_score": round(float(stale_score), 6),
        "stock_stale_vs_trailing_margin": round(float(stale_score - trailing_score), 6),
        "stock_stale_override_applied": bool(stale_override_applied),
        "stock_trailing_override_applied": bool(trailing_override_applied),
        "stock_down_case_exit_shape_reason": down_exit_shape_reason,
        "stock_exit_shape_predictive_mode": "active",
        "stock_pnl_quality_score": round(float(pnl_quality_score), 6),
        "stock_pnl_quality_reason": pnl_quality_reason,
        "stock_trade_quality_score": round(float(trade_quality_score), 6),
        "stock_trade_quality_gate_applied": bool(pnl_quality_gate_applied),
        "stock_weak_window_guard_applied": bool(weak_window_guard_applied),
        "stock_weak_window_guard_reason": weak_window_guard_reason or pnl_quality_reason,
        "stock_high_vol_long_hold_penalty": round(float(high_vol_long_hold_penalty), 6),
        "stock_drawdown_quality_penalty": round(float(drawdown_quality_penalty), 6),
        "stock_symbol_pnl_history_penalty": round(float(symbol_pnl_history_penalty), 6),
        "winning_reason": "momentum_up" if pred_dir == "up" else "momentum_down",
    }


def _forex_predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
    predictor_variant: str = "candidate",
) -> Dict[str, Any]:
    working = list(train_rows[-260:]) if len(train_rows) > 260 else list(train_rows)
    sym = _s(candidate.get("symbol", "")).upper()
    side = _s(candidate.get("side", "")).lower()
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == sym]
    side_rows = [r for r in working if _s(r.get("side", "")).lower() == side] if side else []
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    filtered = symbol_rows if len(symbol_rows) >= 12 else side_rows if len(side_rows) >= 12 else regime_rows if len(regime_rows) >= 12 else working
    if not filtered:
        entry = _f(candidate.get("entry_price", 0.0), 0.0)
        return {
            "predicted_direction": "flat",
            "predicted_exit_trigger": _s(candidate.get("actual_exit_trigger", "")) or "Unknown",
            "predicted_hold_hours": max(1.0, _f(candidate.get("hold_hours", 8.0), 8.0)),
            "predicted_exit_price": float(entry),
            "predicted_confidence": 0.5,
            "predictor_mode": "forex_execution_log",
        }
    returns = [_trade_return_pct(r) for r in filtered]
    mean_ret = sum(returns) / max(1, len(returns))
    vol_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / max(1, len(returns)))
    hold_h = _f(candidate.get("hold_hours", 0.0), 0.0)
    hold_med = _median([_f(r.get("hold_hours", 0.0), 0.0) for r in filtered], default=8.0)
    variant = _s(predictor_variant).lower() or "candidate"
    if variant == "baseline":
        sym_stats = _weighted_trade_stats(symbol_rows or filtered)
        side_stats = _weighted_trade_stats(side_rows or filtered)
        regime_stats = _weighted_trade_stats(regime_rows or filtered)
        cohorts = [(0.50, sym_stats), (0.30, side_stats), (0.20, regime_stats)]
        up_score = (0.70 * _blend_probability(cohorts, "up_rate")) + (0.15 * max(0.0, mean_ret / 0.25)) + (0.15 * max(0.0, -vol_ret / 0.50))
        down_score = (0.70 * _blend_probability(cohorts, "down_rate")) + (0.15 * max(0.0, -mean_ret / 0.25)) + (0.15 * max(0.0, vol_ret / 0.40))
    else:
        up_score = 0.55 * max(0.0, mean_ret / 0.20) + 0.20 * max(0.0, (hold_med - hold_h) / max(1.0, hold_med))
        down_score = 0.55 * max(0.0, -mean_ret / 0.20) + 0.20 * max(0.0, (hold_h - hold_med) / max(1.0, hold_med)) + 0.10 * max(0.0, vol_ret / 0.30)
    if side == "short":
        down_score += 0.10
    elif side == "long":
        up_score += 0.10
    pred_dir = "up" if up_score >= down_score else "down"
    pred_trigger = _s(candidate.get("actual_exit_trigger", "")) or "Unknown"
    med_abs = abs(_median(returns, default=0.0) / 100.0)
    entry_px = _f(candidate.get("entry_price", 0.0), 0.0)
    exit_px = entry_px * (1.0 + med_abs) if pred_dir == "up" else entry_px * (1.0 - med_abs)
    conf = max(0.20, min(0.95, 0.25 + (0.25 * max(up_score, down_score)) + (0.15 * min(1.0, vol_ret / 0.30))))
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trigger,
        "predicted_hold_hours": round(float(max(0.25, hold_med)), 6),
        "predicted_exit_price": round(float(max(1e-12, exit_px)), 10),
        "predicted_confidence": round(float(conf), 6),
        "direction_scores": {"up": round(float(up_score), 6), "down": round(float(down_score), 6)},
        "direction_margin": round(float(abs(up_score - down_score)), 6),
        "predictor_mode": "forex_execution_log",
        "predictor_variant": variant,
        "forex_pair_return_mean_pct": round(float(mean_ret), 6),
        "forex_pair_return_volatility_pct": round(float(vol_ret), 6),
    }


def _predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
    market: str = "",
    predictor_variant: str = "candidate",
) -> Dict[str, Any]:
    if _s(market).lower() == "crypto":
        if _is_historical_strategy_replay_row(candidate):
            return _crypto_predict_historical_replay_one(train_rows=train_rows, candidate=candidate, regime=regime, predictor_variant=predictor_variant)
        return _crypto_predict_one(train_rows=train_rows, candidate=candidate, regime=regime)
    if _s(market).lower() == "stocks" and _s(candidate.get("source_type", "")) == "historical_api_replay":
        return _stock_predict_one(train_rows=train_rows, candidate=candidate, regime=regime, predictor_variant=predictor_variant)
    if _s(market).lower() == "forex":
        return _forex_predict_one(train_rows=train_rows, candidate=candidate, regime=regime, predictor_variant=predictor_variant)
    # Keep predictor bounded in cost for large historical sets.
    working = list(train_rows[-220:]) if len(train_rows) > 220 else list(train_rows)
    sym = _s(candidate.get("symbol", "")).upper()
    filtered = [r for r in working if _s(r.get("symbol", "")).upper() == sym and _s(r.get("regime", "")) == regime]
    if len(filtered) < 8:
        filtered = [r for r in working if _s(r.get("regime", "")) == regime]
    if len(filtered) < 12:
        filtered = list(working)
    if not filtered:
        entry = _f(candidate.get("entry_price", 0.0), 0.0)
        return {
            "predicted_direction": "flat",
            "predicted_exit_trigger": "Unknown",
            "predicted_hold_hours": max(1.0, _f(candidate.get("hold_hours", 1.0), 1.0)),
            "predicted_exit_price": float(entry),
            "predicted_confidence": 0.5,
        }

    dirs = [_trade_direction(r) for r in filtered]
    ups = sum(1 for d in dirs if d == "up")
    downs = sum(1 for d in dirs if d == "down")
    flats = sum(1 for d in dirs if d == "flat")
    total = max(1, len(dirs))
    if ups >= downs and ups >= flats:
        pred_dir = "up"
        dir_share = ups / total
    elif downs >= flats:
        pred_dir = "down"
        dir_share = downs / total
    else:
        pred_dir = "flat"
        dir_share = flats / total

    trig_counts: Dict[str, int] = {}
    for r in filtered:
        t = _s(r.get("actual_exit_trigger", "Unknown")) or "Unknown"
        trig_counts[t] = int(trig_counts.get(t, 0) + 1)
    pred_trig = "Unknown"
    trig_share = 0.0
    if trig_counts:
        pred_trig = max(trig_counts.keys(), key=lambda k: trig_counts.get(k, 0))
        trig_share = float(trig_counts.get(pred_trig, 0)) / float(max(1, len(filtered)))

    holds = [_f(r.get("hold_hours", 0.0), 0.0) for r in filtered if _f(r.get("hold_hours", 0.0), 0.0) >= 0.0]
    hold_h = max(0.25, _median(holds, default=2.0))
    rets = [_trade_return_pct(r) for r in filtered]
    med_ret = _median(rets, default=0.0) / 100.0
    entry_px = _f(candidate.get("entry_price", 0.0), 0.0)
    if pred_dir == "up":
        exit_px = entry_px * (1.0 + max(0.0005, abs(med_ret)))
    elif pred_dir == "down":
        exit_px = entry_px * (1.0 - max(0.0005, abs(med_ret)))
    else:
        exit_px = entry_px
    sample_factor = min(1.0, math.log1p(len(filtered)) / math.log1p(50.0))
    conf = max(0.0, min(1.0, ((0.55 * dir_share) + (0.45 * trig_share)) * (0.65 + (0.35 * sample_factor))))
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trig,
        "predicted_hold_hours": round(float(hold_h), 6),
        "predicted_exit_price": round(float(max(1e-12, exit_px)), 10),
        "predicted_confidence": round(float(conf), 6),
    }


def _objective_score(metrics: Dict[str, Any], market: str) -> float:
    m = _s(market).lower()
    d = _f(metrics.get("directional_accuracy_pct", 0.0), 0.0)
    t = _f(metrics.get("trigger_match_pct", 0.0), 0.0)
    p = _f(metrics.get("pnl_trend_match_pct", 0.0), 0.0)
    if m == "forex":
        return (0.50 * d) + (0.05 * t) + (0.45 * p)
    if m == "stocks":
        return (0.45 * d) + (0.25 * t) + (0.30 * p)
    return (0.40 * d) + (0.35 * t) + (0.25 * p)


def _bucket_probability(value: float) -> str:
    if value < 0.25:
        return "<0.25"
    if value < 0.40:
        return "0.25-0.40"
    if value < 0.55:
        return "0.40-0.55"
    if value < 0.70:
        return "0.55-0.70"
    return "0.70+"


def _bucket_margin_value(value: float) -> str:
    if value < 0.5:
        return "<0.5"
    if value < 1.5:
        return "0.5-1.5"
    if value < 3.0:
        return "1.5-3.0"
    if value < 5.0:
        return "3.0-5.0"
    return "5.0+"


def _trigger_precision(rows: List[Dict[str, Any]], trigger: str) -> Dict[str, float]:
    preds = [r for r in rows if _s(r.get("predicted_exit_trigger", "")) == trigger]
    correct = [r for r in preds if _s(r.get("actual_exit_trigger", "")) == trigger]
    return {
        "predicted_count": float(len(preds)),
        "correct_count": float(len(correct)),
        "incorrect_count": float(max(0, len(preds) - len(correct))),
        "precision_pct": round(100.0 * len(correct) / max(1, len(preds)), 4),
    }


def _policy_name(policy: Dict[str, Any]) -> str:
    manual_mode = _s(policy.get("manual_mode", "strict")) or "strict"
    trailing_mode = _s(policy.get("trailing_mode", "high_margin")) or "high_margin"
    if manual_mode == "diagnostic_only":
        return f"stale+{trailing_mode}+manual_diagnostic_only"
    if manual_mode == "off":
        return f"stale+{trailing_mode}+manual_off"
    return f"stale+{trailing_mode}+manual_strict"


def _next_required_features() -> List[str]:
    return [
        "persist live decision snapshot at entry and exit time",
        "store exit-rule reason directly on every exit row",
        "store stale/trailing/manual scores from live strategy engine",
        "store candle/context features at entry time",
        "build historical candle replay using the same crypto strategy logic",
        "retain signal-state features used by the UI/trader at the time of the decision",
    ]


def _manual_prediction_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    manual_rows = [r for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Manual"]
    correct = [r for r in manual_rows if _s(r.get("actual_exit_trigger", "")) == "Manual"]
    incorrect = [r for r in manual_rows if _s(r.get("actual_exit_trigger", "")) != "Manual"]

    def bucket_counts(items: List[Dict[str, Any]], field: str, bucketer: Any) -> Dict[str, int]:
        return _count_by(items, lambda r: bucketer(_f(r.get(field, 0.0), 0.0)))

    def raw_counts(items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        return _count_by(items, lambda r: _s(r.get(field, "")) or "0")

    return {
        "predicted_count": int(len(manual_rows)),
        "correct_count": int(len(correct)),
        "incorrect_count": int(len(incorrect)),
        "precision_pct": round(100.0 * len(correct) / max(1, len(manual_rows)), 4),
        "by_symbol": {
            "correct": _count_by(correct, lambda r: _s(r.get("symbol", "")) or "UNKNOWN"),
            "incorrect": _count_by(incorrect, lambda r: _s(r.get("symbol", "")) or "UNKNOWN"),
        },
        "by_regime": {
            "correct": _count_by(correct, lambda r: _s(r.get("regime", "")) or "unknown"),
            "incorrect": _count_by(incorrect, lambda r: _s(r.get("regime", "")) or "unknown"),
        },
        "manual_score_buckets": {
            "correct": bucket_counts(correct, "manual_score", _bucket_margin_value),
            "incorrect": bucket_counts(incorrect, "manual_score", _bucket_margin_value),
        },
        "manual_runner_up_margin_buckets": {
            "correct": bucket_counts(correct, "manual_runner_up_margin", _bucket_margin_value),
            "incorrect": bucket_counts(incorrect, "manual_runner_up_margin", _bucket_margin_value),
        },
        "trigger_margin_buckets": {
            "correct": bucket_counts(correct, "trigger_margin", _bucket_margin_value),
            "incorrect": bucket_counts(incorrect, "trigger_margin", _bucket_margin_value),
        },
        "direction_margin_buckets": {
            "correct": bucket_counts(correct, "direction_margin", _bucket_margin_value),
            "incorrect": bucket_counts(incorrect, "direction_margin", _bucket_margin_value),
        },
        "predicted_confidence_buckets": {
            "correct": bucket_counts(correct, "predicted_confidence", _bucket_probability),
            "incorrect": bucket_counts(incorrect, "predicted_confidence", _bucket_probability),
        },
        "same_symbol_support_buckets": {
            "correct": raw_counts(correct, "manual_same_symbol_support_count"),
            "incorrect": raw_counts(incorrect, "manual_same_symbol_support_count"),
        },
        "same_regime_support_buckets": {
            "correct": raw_counts(correct, "manual_same_regime_support_count"),
            "incorrect": raw_counts(incorrect, "manual_same_regime_support_count"),
        },
        "recent_support_buckets": {
            "correct": raw_counts(correct, "manual_recent_support_count"),
            "incorrect": raw_counts(incorrect, "manual_recent_support_count"),
        },
        "long_hold_ratio_buckets": {
            "correct": bucket_counts(correct, "manual_symbol_long_hold_ratio", _bucket_probability),
            "incorrect": bucket_counts(incorrect, "manual_symbol_long_hold_ratio", _bucket_probability),
        },
        "vs_stale_ratio_buckets": {
            "correct": bucket_counts(correct, "manual_symbol_vs_stale_ratio", _bucket_probability),
            "incorrect": bucket_counts(incorrect, "manual_symbol_vs_stale_ratio", _bucket_probability),
        },
        "vs_stale_long_hold_ratio_buckets": {
            "correct": bucket_counts(correct, "manual_symbol_vs_stale_long_hold_ratio", _bucket_probability),
            "incorrect": bucket_counts(incorrect, "manual_symbol_vs_stale_long_hold_ratio", _bucket_probability),
        },
        "outcome_notes": {
            "correct": _count_by(correct, lambda r: _s(r.get("manual_outcome_note", "")) or "none"),
            "incorrect": _count_by(incorrect, lambda r: _s(r.get("manual_outcome_note", "")) or "none"),
        },
    }


def _historical_replay_policy_name(policy: Dict[str, Any]) -> str:
    if _s(policy.get("policy_name_override", "")):
        return _s(policy.get("policy_name_override", ""))
    mode = _s(policy.get("source_mode", "historical_strategy_replay")) or "historical_strategy_replay"
    conf = _f(policy.get("base_conf_min", 0.36), 0.36)
    trig = _f(policy.get("base_trigger_margin_min", 0.25), 0.25)
    direction = _f(policy.get("base_dir_margin_min", 0.05), 0.05)
    return f"{mode}:c{conf:.2f}:t{trig:.2f}:d{direction:.2f}"


def _crypto_historical_replay_admission_decision(row: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, str]:
    pred_trigger = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
    conf = _f(row.get("predicted_confidence", 0.0), 0.0)
    trig_margin = _f(row.get("trigger_margin", 0.0), 0.0)
    dir_margin = _f(row.get("direction_margin", 0.0), 0.0)
    base_conf = _f(policy.get("base_conf_min", 0.36), 0.36)
    base_trig = _f(policy.get("base_trigger_margin_min", 0.25), 0.25)
    base_dir = _f(policy.get("base_dir_margin_min", 0.05), 0.05)
    trigger_scores = row.get("trigger_scores", {}) if isinstance(row.get("trigger_scores", {}), dict) else {}
    top = _f(trigger_scores.get(pred_trigger, 0.0), 0.0)
    runner = _f(row.get("runner_up_trigger_score", 0.0), 0.0)
    separation = top - runner
    rescue_band = _f(policy.get("rescue_margin_band", 0.05), 0.05)
    rescue_conf_band = _f(policy.get("rescue_conf_band", 0.05), 0.05)
    rescue_dir_band = _f(policy.get("rescue_dir_band", 0.05), 0.05)
    clean_expansion_enabled = bool(policy.get("enable_clean_expansion", False))

    def clean_expansion_allowed(trigger: str) -> bool:
        if not clean_expansion_enabled:
            return False
        if trigger == "Risk Cut":
            return bool(row.get("risk_cut_touched", False)) and _f(row.get("max_adverse_excursion_pct", 0.0), 0.0) <= -1.8 and _f(row.get("downside_score", 0.0), 0.0) >= 0.55
        if trigger == "Trailing":
            return bool(row.get("trailing_armed", False)) and bool(row.get("favorable_then_softened_flag", False)) and _f(row.get("trailing_pullback_pct", 0.0), 0.0) >= 1.0 and not bool(row.get("risk_cut_touched", False))
        if trigger == "Stale Alignment":
            return (not bool(row.get("risk_cut_touched", False))) and (not bool(row.get("take_profit_touched", False))) and (not bool(row.get("trailing_armed", False))) and _f(row.get("bars_in_trade", 0.0), 0.0) >= 12.0 and _f(row.get("trend_momentum_score", 0.0), 0.0) <= 0.15
        if trigger == "Take Profit":
            return bool(row.get("take_profit_touched", False)) and _f(row.get("max_favorable_excursion_pct", 0.0), 0.0) >= 4.0
        return False

    if pred_trigger == "Risk Cut":
        conf_min = _f(policy.get("risk_cut_conf_min", base_conf), base_conf)
        trig_min = _f(policy.get("risk_cut_trigger_margin_min", base_trig), base_trig)
        dir_min = _f(policy.get("risk_cut_dir_margin_min", base_dir), base_dir)
        if conf < conf_min:
            if bool(policy.get("enable_risk_cut_rescue", False)) and bool(row.get("risk_cut_touched", False)) and conf >= (conf_min - rescue_conf_band) and separation >= max(0.0, trig_min - rescue_band):
                return True, "risk_cut_rescue"
            if clean_expansion_allowed("Risk Cut") and conf >= (conf_min - 0.08):
                return True, "clean_expansion_risk_cut"
            return False, "risk_cut_conf_low"
        if trig_margin < trig_min:
            if bool(policy.get("enable_risk_cut_rescue", False)) and bool(row.get("risk_cut_touched", False)) and trig_margin >= (trig_min - rescue_band):
                return True, "risk_cut_rescue"
            if clean_expansion_allowed("Risk Cut") and trig_margin >= (trig_min - 0.10):
                return True, "clean_expansion_risk_cut"
            return False, "risk_cut_trigger_margin_low"
        if dir_margin < dir_min:
            if bool(policy.get("enable_risk_cut_rescue", False)) and bool(row.get("risk_cut_touched", False)) and dir_margin >= (dir_min - rescue_dir_band):
                return True, "risk_cut_rescue"
            if clean_expansion_allowed("Risk Cut") and dir_margin >= (dir_min - 0.10):
                return True, "clean_expansion_risk_cut"
            return False, "risk_cut_dir_margin_low"
        return True, "risk_cut_strong"

    if pred_trigger == "Take Profit":
        conf_min = _f(policy.get("take_profit_conf_min", base_conf), base_conf)
        trig_min = _f(policy.get("take_profit_trigger_margin_min", base_trig), base_trig)
        gap_min = _f(policy.get("take_profit_score_gap_min", 0.15), 0.15)
        if conf < conf_min:
            if clean_expansion_allowed("Take Profit") and conf >= (conf_min - 0.06):
                return True, "clean_expansion_take_profit"
            return False, "take_profit_conf_low"
        if trig_margin < trig_min:
            if clean_expansion_allowed("Take Profit") and trig_margin >= (trig_min - 0.08):
                return True, "clean_expansion_take_profit"
            return False, "take_profit_trigger_margin_low"
        if separation < gap_min:
            if bool(policy.get("enable_take_profit_rescue", False)) and bool(row.get("take_profit_touched", False)) and separation >= (gap_min - rescue_band):
                return True, "take_profit_rescue"
            if clean_expansion_allowed("Take Profit") and separation >= (gap_min - 0.10):
                return True, "clean_expansion_take_profit"
            return False, "take_profit_score_gap_low"
        return True, "take_profit_strong"

    if pred_trigger == "Trailing":
        conf_min = _f(policy.get("trailing_conf_min", base_conf), base_conf)
        trig_min = _f(policy.get("trailing_trigger_margin_min", base_trig), base_trig)
        dir_min = _f(policy.get("trailing_dir_margin_min", base_dir), base_dir)
        if conf < conf_min:
            if bool(policy.get("enable_trailing_rescue", False)) and bool(row.get("trailing_armed", False)) and bool(row.get("favorable_then_softened_flag", False)) and conf >= (conf_min - rescue_conf_band):
                return True, "trailing_rescue"
            if clean_expansion_allowed("Trailing") and conf >= (conf_min - 0.06):
                return True, "clean_expansion_trailing"
            return False, "trailing_conf_low"
        if trig_margin < trig_min:
            if bool(policy.get("enable_trailing_rescue", False)) and bool(row.get("trailing_armed", False)) and bool(row.get("favorable_then_softened_flag", False)) and trig_margin >= (trig_min - rescue_band):
                return True, "trailing_rescue"
            if clean_expansion_allowed("Trailing") and trig_margin >= (trig_min - 0.08):
                return True, "clean_expansion_trailing"
            return False, "trailing_trigger_margin_low"
        if dir_margin < dir_min:
            if bool(policy.get("enable_trailing_rescue", False)) and bool(row.get("trailing_armed", False)) and bool(row.get("favorable_then_softened_flag", False)) and dir_margin >= (dir_min - rescue_dir_band):
                return True, "trailing_rescue"
            if clean_expansion_allowed("Trailing") and dir_margin >= (dir_min - 0.08):
                return True, "clean_expansion_trailing"
            return False, "trailing_dir_margin_low"
        return True, "trailing_strong"

    if pred_trigger == "Stale Alignment":
        conf_min = _f(policy.get("stale_conf_min", base_conf), base_conf)
        trig_min = _f(policy.get("stale_trigger_margin_min", base_trig), base_trig)
        dir_min = _f(policy.get("stale_dir_margin_min", base_dir), base_dir)
        if conf < conf_min:
            if clean_expansion_allowed("Stale Alignment") and conf >= (conf_min - 0.06):
                return True, "clean_expansion_stale"
            return False, "stale_conf_low"
        if trig_margin < trig_min:
            if clean_expansion_allowed("Stale Alignment") and trig_margin >= (trig_min - 0.08):
                return True, "clean_expansion_stale"
            return False, "stale_trigger_margin_low"
        if dir_margin < dir_min:
            if bool(policy.get("enable_stale_rescue", False)) and not bool(row.get("risk_cut_touched", False)) and not bool(row.get("take_profit_touched", False)) and not bool(row.get("trailing_armed", False)) and dir_margin >= (dir_min - rescue_dir_band):
                return True, "stale_rescue"
            if clean_expansion_allowed("Stale Alignment") and dir_margin >= (dir_min - 0.10):
                return True, "clean_expansion_stale"
            return False, "stale_dir_margin_low"
        return True, "stale_strong"

    return False, "unsupported_trigger"


def _crypto_admission_decision(row: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, str]:
    if _is_historical_strategy_replay_row(row) or _s(policy.get("source_mode", "")) == "historical_strategy_replay":
        return _crypto_historical_replay_admission_decision(row, policy)
    pred_trigger = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
    conf = _f(row.get("predicted_confidence", 0.0), 0.0)
    trig_margin = _f(row.get("trigger_margin", 0.0), 0.0)
    dir_margin = _f(row.get("direction_margin", 0.0), 0.0)
    manual_margin = _f(row.get("manual_runner_up_margin", -1e9), -1e9)
    manual_same_symbol = int(_f(row.get("manual_same_symbol_support_count", 0.0), 0.0))
    manual_same_symbol_stale = int(_f(row.get("manual_same_symbol_stale_count", 0.0), 0.0))
    manual_same_regime = int(_f(row.get("manual_same_regime_support_count", 0.0), 0.0))
    manual_recent = int(_f(row.get("manual_recent_support_count", 0.0), 0.0))
    manual_long_hold_ratio = _f(row.get("manual_symbol_long_hold_ratio", 0.0), 0.0)
    manual_vs_stale_ratio = _f(row.get("manual_symbol_vs_stale_ratio", 0.0), 0.0)
    manual_vs_stale_long_hold_ratio = _f(row.get("manual_symbol_vs_stale_long_hold_ratio", 0.0), 0.0)
    manual_recent_density = _f(row.get("manual_recent_density_40", 0.0), 0.0)
    stale_score = _f(row.get("stale_score", 0.0), 0.0)
    trailing_score = _f(row.get("trailing_score", 0.0), 0.0)
    manual_score = _f(row.get("manual_score", 0.0), 0.0)
    base_floor = _f(policy.get("base_conf_min", 0.55), 0.55)

    if pred_trigger == "Stale Alignment":
        if conf < max(base_floor, _f(policy.get("stale_conf_min", 0.56), 0.56)):
            return False, "stale_conf_low"
        if trig_margin < _f(policy.get("stale_trigger_margin_min", 2.5), 2.5):
            return False, "stale_trigger_margin_low"
        if dir_margin < _f(policy.get("stale_dir_margin_min", 1.75), 1.75):
            return False, "stale_direction_margin_low"
        return True, "stale_strong"

    if pred_trigger == "Trailing":
        if conf < _f(policy.get("trailing_conf_min", 0.30), 0.30):
            return False, "trailing_conf_low"
        if trig_margin < _f(policy.get("trailing_trigger_margin_min", 2.25), 2.25):
            return False, "trailing_trigger_margin_low"
        if dir_margin < _f(policy.get("trailing_dir_margin_min", 1.25), 1.25):
            return False, "trailing_direction_margin_low"
        if trailing_score < (stale_score + _f(policy.get("trailing_vs_stale_min", 0.0), 0.0)):
            return False, "trailing_score_not_clear"
        return True, "trailing_strong"

    if pred_trigger == "Manual":
        if _s(policy.get("manual_mode", "strict")) == "off":
            return False, "manual_mode_off"
        if _s(policy.get("manual_mode", "strict")) == "diagnostic_only":
            diag_same_symbol_min = int(_f(policy.get("manual_diag_same_symbol_min", 2.0), 2.0))
            diag_ratio_min = _f(policy.get("manual_diag_vs_stale_long_hold_min", 0.45), 0.45)
            if manual_same_symbol < diag_same_symbol_min and manual_vs_stale_long_hold_ratio < diag_ratio_min:
                return False, "manual_diagnostic_only_block"
        if conf < _f(policy.get("manual_conf_min", 0.30), 0.30):
            return False, "manual_conf_low"
        if manual_margin < _f(policy.get("manual_runner_margin_min", -0.35), -0.35):
            return False, "manual_runner_margin_low"
        if trig_margin < _f(policy.get("manual_trigger_margin_min", 0.50), 0.50):
            return False, "manual_trigger_margin_low"
        if dir_margin < _f(policy.get("manual_dir_margin_min", 0.50), 0.50):
            return False, "manual_direction_margin_low"
        support_score = manual_same_symbol + manual_same_regime + min(2, manual_recent)
        if support_score < int(_f(policy.get("manual_support_min", 2.0), 2.0)):
            return False, "manual_symbol_support_low"
        if manual_same_symbol <= 0 and manual_same_regime > 0:
            return False, "manual_regime_only_signal"
        if manual_long_hold_ratio < _f(policy.get("manual_long_hold_ratio_min", 0.20), 0.20):
            return False, "manual_long_hold_signal_low"
        if manual_vs_stale_ratio < _f(policy.get("manual_vs_stale_ratio_min", 0.10), 0.10):
            return False, "manual_symbol_vs_stale_low"
        if manual_vs_stale_long_hold_ratio < _f(policy.get("manual_vs_stale_long_hold_ratio_min", 0.15), 0.15):
            return False, "manual_long_hold_vs_stale_low"
        if manual_recent_density < _f(policy.get("manual_recent_density_min", 0.02), 0.02):
            return False, "manual_recent_density_low"
        if manual_vs_stale_ratio < 0.25 and manual_same_symbol < int(_f(policy.get("manual_symbol_support_min", 2.0), 2.0)):
            return False, "manual_symbol_support_low"
        if manual_vs_stale_ratio < 0.25 and manual_same_symbol_stale > manual_same_symbol:
            return False, "manual_stale_symbol_dominance"
        if manual_score < max(stale_score, trailing_score) + _f(policy.get("manual_score_advantage_min", -0.25), -0.25):
            return False, "manual_score_not_clear"
        independent_signals = 0
        independent_signals += 1 if manual_same_symbol > 0 else 0
        independent_signals += 1 if manual_same_regime > 0 else 0
        independent_signals += 1 if manual_recent > 0 else 0
        independent_signals += 1 if manual_long_hold_ratio >= _f(policy.get("manual_long_hold_ratio_min", 0.20), 0.20) else 0
        independent_signals += 1 if manual_vs_stale_ratio >= _f(policy.get("manual_vs_stale_ratio_min", 0.10), 0.10) else 0
        if independent_signals < int(_f(policy.get("manual_signal_votes_min", 3.0), 3.0)):
            return False, "manual_precision_gate_failed"
        return True, "manual_strong"

    if conf >= base_floor:
        return True, "fallback_confident"
    return False, "fallback_conf_low"


def _score_crypto_with_policy(rows: List[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    admitted: List[Dict[str, Any]] = []
    abstained: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    for row in rows:
        admit, reason = _crypto_admission_decision(row, policy)
        tagged = dict(row)
        tagged["admission_reason"] = reason
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
        if admit:
            admitted.append(tagged)
        else:
            abstained.append(tagged)
    metrics = _metrics_for_rows(admitted)
    full_metrics = _metrics_for_rows(rows)
    abstained_metrics = _metrics_for_rows(abstained)
    coverage = (len(admitted) / max(1, len(rows))) if rows else 0.0
    score = _objective_score(metrics, "crypto")
    use_hist_policy = bool(rows) and all(_is_historical_strategy_replay_row(r) for r in rows)
    predicted_manual = sum(1 for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Manual")
    predicted_trailing = sum(1 for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Trailing")
    predicted_risk_cut = sum(1 for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Risk Cut")
    predicted_take_profit = sum(1 for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Take Profit")
    admitted_manual = sum(1 for r in admitted if _s(r.get("predicted_exit_trigger", "")) == "Manual")
    admitted_trailing = sum(1 for r in admitted if _s(r.get("predicted_exit_trigger", "")) == "Trailing")
    clean_expansion_rows = [r for r in admitted if _s(r.get("admission_reason", "")).startswith("clean_expansion_")]
    manual_precision = _trigger_precision(admitted, "Manual")
    trailing_precision = _trigger_precision(admitted, "Trailing")
    stale_precision = _trigger_precision(admitted, "Stale Alignment")
    risk_cut_precision = _trigger_precision(admitted, "Risk Cut")
    take_profit_precision = _trigger_precision(admitted, "Take Profit")
    by_pred_trigger_reject: Dict[str, int] = {}
    by_actual_trigger_reject: Dict[str, int] = {}
    by_symbol_reject: Dict[str, int] = {}
    by_conf_bucket_reject: Dict[str, int] = {}
    by_trig_margin_bucket_reject: Dict[str, int] = {}
    by_dir_margin_bucket_reject: Dict[str, int] = {}
    by_score_gap_bucket_reject: Dict[str, int] = {}
    near_miss_count = 0
    correct_rejected = 0
    incorrect_admitted = 0
    for r in abstained:
        pred_trig = _s(r.get("predicted_exit_trigger", "")) or "Unknown"
        act_trig = _s(r.get("actual_exit_trigger", "")) or "Unknown"
        by_pred_trigger_reject[pred_trig] = int(by_pred_trigger_reject.get(pred_trig, 0) + 1)
        by_actual_trigger_reject[act_trig] = int(by_actual_trigger_reject.get(act_trig, 0) + 1)
        sym = _s(r.get("symbol", "")) or "UNKNOWN"
        by_symbol_reject[sym] = int(by_symbol_reject.get(sym, 0) + 1)
        conf = _f(r.get("predicted_confidence", 0.0), 0.0)
        trig_margin = _f(r.get("trigger_margin", 0.0), 0.0)
        dir_margin = _f(r.get("direction_margin", 0.0), 0.0)
        gap = abs(_f(r.get("winning_trigger_score", 0.0), 0.0) - _f(r.get("runner_up_trigger_score", 0.0), 0.0))
        by_conf_bucket_reject[_bucket_probability(conf)] = int(by_conf_bucket_reject.get(_bucket_probability(conf), 0) + 1)
        by_trig_margin_bucket_reject[_bucket_margin_value(trig_margin)] = int(by_trig_margin_bucket_reject.get(_bucket_margin_value(trig_margin), 0) + 1)
        by_dir_margin_bucket_reject[_bucket_margin_value(dir_margin)] = int(by_dir_margin_bucket_reject.get(_bucket_margin_value(dir_margin), 0) + 1)
        by_score_gap_bucket_reject[_bucket_margin_value(gap)] = int(by_score_gap_bucket_reject.get(_bucket_margin_value(gap), 0) + 1)
        base_conf = _f(policy.get("base_conf_min", 0.0), 0.0)
        base_trig = _f(policy.get("base_trigger_margin_min", 0.0), 0.0)
        base_dir = _f(policy.get("base_dir_margin_min", 0.0), 0.0)
        if conf >= (base_conf - 0.05) or trig_margin >= (base_trig - 0.05) or dir_margin >= (base_dir - 0.05):
            near_miss_count += 1
        if _s(r.get("predicted_direction", "")).lower() == _s(r.get("actual_direction", "")).lower() and pred_trig == act_trig:
            correct_rejected += 1
    for r in admitted:
        pred_trig = _s(r.get("predicted_exit_trigger", "")) or "Unknown"
        act_trig = _s(r.get("actual_exit_trigger", "")) or "Unknown"
        if _s(r.get("predicted_direction", "")).lower() != _s(r.get("actual_direction", "")).lower() or pred_trig != act_trig:
            incorrect_admitted += 1
    clean_expansion_attempted = bool(policy.get("enable_clean_expansion", False))
    clean_expansion_rows_added = int(len(clean_expansion_rows))
    clean_expansion_correct_added_estimate = int(
        sum(
            1
            for r in clean_expansion_rows
            if _s(r.get("predicted_direction", "")).lower() == _s(r.get("actual_direction", "")).lower()
            and _s(r.get("predicted_exit_trigger", "")) == _s(r.get("actual_exit_trigger", ""))
        )
    )
    clean_expansion_guardrails_passed = bool(
        clean_expansion_attempted
        and clean_expansion_rows_added > 0
        and clean_expansion_correct_added_estimate >= clean_expansion_rows_added
    )
    clean_expansion_rejection_reason = ""
    if clean_expansion_attempted and not clean_expansion_guardrails_passed:
        if clean_expansion_rows_added <= 0:
            clean_expansion_rejection_reason = "no_clean_expansion_rows_admitted"
        else:
            clean_expansion_rejection_reason = "clean_expansion_rows_not_clean_enough"
    worst_floor = min(
        _f(metrics.get("directional_accuracy_pct", 0.0), 0.0),
        _f(metrics.get("trigger_match_pct", 0.0), 0.0),
        _f(metrics.get("pnl_trend_match_pct", 0.0), 0.0),
    )
    if use_hist_policy:
        if coverage < _f(policy.get("coverage_target", 0.40), 0.40):
            score -= (_f(policy.get("coverage_target", 0.40), 0.40) - coverage) * 125.0
        if len(admitted) < int(_f(policy.get("min_admitted_trades", 40.0), 40.0)):
            score -= float(int(_f(policy.get("min_admitted_trades", 40.0), 40.0)) - len(admitted)) * 0.55
        trig_scored = _f(metrics.get("trigger_scored_trades", 0.0), 0.0)
        if trig_scored < _f(policy.get("min_trigger_scored", 20.0), 20.0):
            score -= (_f(policy.get("min_trigger_scored", 20.0), 20.0) - trig_scored) * 0.8
        for trigger_name, precision_row, target_key, count in [
            ("Risk Cut", risk_cut_precision, "risk_cut_precision_target_pct", predicted_risk_cut),
            ("Take Profit", take_profit_precision, "take_profit_precision_target_pct", predicted_take_profit),
            ("Trailing", trailing_precision, "trailing_precision_target_pct", predicted_trailing),
            ("Stale Alignment", stale_precision, "stale_precision_target_pct", sum(1 for r in rows if _s(r.get("predicted_exit_trigger", "")) == "Stale Alignment")),
        ]:
            if count <= 0:
                continue
            target = _f(policy.get(target_key, 65.0), 65.0)
            got = _f(precision_row.get("precision_pct", 0.0), 0.0)
            if got < target:
                score -= (target - got) * 0.30
        worst_target = _f(policy.get("worst_window_floor_target_pct", 85.0), 85.0)
        if worst_floor < worst_target:
            score -= (worst_target - worst_floor) * 0.45
        return {
            "score": score,
            "coverage": coverage,
            "metrics": metrics,
            "full_metrics": full_metrics,
            "abstained_metrics": abstained_metrics,
            "admitted": admitted,
            "abstained": abstained,
            "reason_counts": reason_counts,
            "policy": dict(policy),
            "class_precision": {
                "Risk Cut": risk_cut_precision,
                "Take Profit": take_profit_precision,
                "Trailing": trailing_precision,
                "Stale Alignment": stale_precision,
            },
            "rejection_diagnostics": {
                "by_predicted_trigger": _top_counter(by_pred_trigger_reject, limit=12),
                "by_actual_trigger": _top_counter(by_actual_trigger_reject, limit=12),
                "by_symbol": _top_counter(by_symbol_reject, limit=12),
                "by_confidence_bucket": _top_counter(by_conf_bucket_reject, limit=12),
                "by_trigger_margin_bucket": _top_counter(by_trig_margin_bucket_reject, limit=12),
                "by_direction_margin_bucket": _top_counter(by_dir_margin_bucket_reject, limit=12),
                "by_class_score_gap_bucket": _top_counter(by_score_gap_bucket_reject, limit=12),
            },
            "near_miss_rows_count": int(near_miss_count),
            "correct_rejected_count": int(correct_rejected),
            "incorrect_admitted_count": int(incorrect_admitted),
            "crypto_clean_expansion_attempted": clean_expansion_attempted,
            "crypto_clean_expansion_rows_added": clean_expansion_rows_added,
            "crypto_clean_expansion_correct_rejected_added_estimate": clean_expansion_correct_added_estimate,
            "crypto_clean_expansion_guardrails_passed": clean_expansion_guardrails_passed,
            "crypto_clean_expansion_rejection_reason": clean_expansion_rejection_reason,
        }
    if coverage < _f(policy.get("coverage_target", 0.40), 0.40):
        score -= (_f(policy.get("coverage_target", 0.40), 0.40) - coverage) * 110.0
    if len(admitted) < int(_f(policy.get("min_admitted_trades", 40.0), 40.0)):
        score -= float(int(_f(policy.get("min_admitted_trades", 40.0), 40.0)) - len(admitted)) * 0.45
    trig_scored = _f(metrics.get("trigger_scored_trades", 0.0), 0.0)
    if trig_scored < _f(policy.get("min_trigger_scored", 20.0), 20.0):
        score -= (_f(policy.get("min_trigger_scored", 20.0), 20.0) - trig_scored) * 0.75
    if predicted_manual > 0 and admitted_manual <= 0:
        score -= 12.0
    if admitted_manual > 0:
        target = _f(policy.get("manual_precision_target_pct", 75.0), 75.0)
        got = _f(manual_precision.get("precision_pct", 0.0), 0.0)
        if got < target:
            score -= (target - got) * 0.60
        if admitted_manual == predicted_manual and predicted_manual >= 3 and got < 100.0:
            score -= 6.0
    if predicted_trailing > 0 and admitted_trailing < max(1, int(math.ceil(predicted_trailing * _f(policy.get("min_trailing_admit_ratio", 0.35), 0.35)))):
        score -= 8.0
    if admitted_trailing > 0:
        target = _f(policy.get("trailing_precision_target_pct", 65.0), 65.0)
        got = _f(trailing_precision.get("precision_pct", 0.0), 0.0)
        if got < target:
            score -= (target - got) * 0.35
    worst_target = _f(policy.get("worst_window_floor_target_pct", 85.0), 85.0)
    if worst_floor < worst_target:
        score -= (worst_target - worst_floor) * 0.45
    return {
        "score": score,
        "coverage": coverage,
        "metrics": metrics,
        "full_metrics": full_metrics,
        "abstained_metrics": abstained_metrics,
        "admitted": admitted,
        "abstained": abstained,
        "reason_counts": reason_counts,
        "policy": dict(policy),
        "class_precision": {
            "Manual": manual_precision,
            "Trailing": trailing_precision,
            "Stale Alignment": stale_precision,
        },
        "rejection_diagnostics": {
            "by_predicted_trigger": _top_counter(by_pred_trigger_reject, limit=12),
            "by_actual_trigger": _top_counter(by_actual_trigger_reject, limit=12),
            "by_symbol": _top_counter(by_symbol_reject, limit=12),
            "by_confidence_bucket": _top_counter(by_conf_bucket_reject, limit=12),
            "by_trigger_margin_bucket": _top_counter(by_trig_margin_bucket_reject, limit=12),
            "by_direction_margin_bucket": _top_counter(by_dir_margin_bucket_reject, limit=12),
            "by_class_score_gap_bucket": _top_counter(by_score_gap_bucket_reject, limit=12),
        },
        "near_miss_rows_count": int(near_miss_count),
        "correct_rejected_count": int(correct_rejected),
        "incorrect_admitted_count": int(incorrect_admitted),
        "crypto_clean_expansion_attempted": clean_expansion_attempted,
        "crypto_clean_expansion_rows_added": clean_expansion_rows_added,
        "crypto_clean_expansion_correct_rejected_added_estimate": clean_expansion_correct_added_estimate,
        "crypto_clean_expansion_guardrails_passed": clean_expansion_guardrails_passed,
        "crypto_clean_expansion_rejection_reason": clean_expansion_rejection_reason,
    }


def _frontier_candidate_summary(policy: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {}
    class_precision = row.get("class_precision", {}) if isinstance(row.get("class_precision", {}), dict) else {}
    admitted = row.get("admitted", []) if isinstance(row.get("admitted", []), list) else []
    policy_name = (
        _historical_replay_policy_name(policy)
        if _s(policy.get("source_mode", "")) == "historical_strategy_replay"
        else _policy_name(policy)
    )
    return {
        "policy_name": policy_name,
        "policy": dict(policy),
        "score": round(_f(row.get("score", 0.0), 0.0), 6),
        "admitted_trades": int(len(admitted)),
        "admission_rate_pct": round(100.0 * _f(row.get("coverage", 0.0), 0.0), 4),
        "directional_accuracy_pct": _f(metrics.get("directional_accuracy_pct", 0.0), 0.0),
        "trigger_match_pct": _f(metrics.get("trigger_match_pct", 0.0), 0.0),
        "pnl_trend_match_pct": _f(metrics.get("pnl_trend_match_pct", 0.0), 0.0),
        "worst_window_metrics": {},
        "class_precision": class_precision,
        "class_counts": _count_by(admitted, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
        "blockers": [],
    }


def _score_with_threshold(rows: List[Dict[str, Any]], threshold: float, market: str) -> Dict[str, Any]:
    admitted = [r for r in rows if _f(r.get("predicted_confidence", 0.0), 0.0) >= float(threshold)]
    abstained = [r for r in rows if _f(r.get("predicted_confidence", 0.0), 0.0) < float(threshold)]
    metrics = _metrics_for_rows(admitted)
    full_metrics = _metrics_for_rows(rows)
    abstained_metrics = _metrics_for_rows(abstained)
    coverage = (len(admitted) / max(1, len(rows))) if rows else 0.0
    score = _objective_score(metrics, market)
    if coverage < 0.30:
        score -= (0.30 - coverage) * 120.0
    return {
        "score": score,
        "coverage": coverage,
        "metrics": metrics,
        "full_metrics": full_metrics,
        "abstained_metrics": abstained_metrics,
        "admitted": admitted,
        "abstained": abstained,
    }


def _score_without_abstain(rows: List[Dict[str, Any]], market: str, policy_name: str) -> Dict[str, Any]:
    metrics = _metrics_for_rows(rows)
    return {
        "score": _objective_score(metrics, market),
        "coverage": 1.0 if rows else 0.0,
        "metrics": metrics,
        "full_metrics": metrics,
        "abstained_metrics": _metrics_for_rows([]),
        "admitted": list(rows),
        "abstained": [],
        "policy_name": policy_name,
    }


def _selection_option_summary(name: str, scored: Dict[str, Any], *, predictor_variant: str, policy: Any = None) -> Dict[str, Any]:
    metrics = scored.get("metrics", {}) if isinstance(scored.get("metrics", {}), dict) else {}
    full_metrics = scored.get("full_metrics", {}) if isinstance(scored.get("full_metrics", {}), dict) else {}
    admitted = scored.get("admitted", []) if isinstance(scored.get("admitted", []), list) else []
    return {
        "policy_name": name,
        "predictor_variant": predictor_variant,
        "policy": policy if isinstance(policy, dict) else {},
        "admitted_trades": int(len(admitted)),
        "admission_rate_pct": round(100.0 * _f(scored.get("coverage", 0.0), 0.0), 4),
        "metrics": metrics,
        "full_metrics": full_metrics,
        "trigger_scored_trades": _f(metrics.get("trigger_scored_trades", 0.0), 0.0),
        "score": round(_f(scored.get("score", 0.0), 0.0), 6),
        "rejection_diagnostics": scored.get("rejection_diagnostics", {}) if isinstance(scored.get("rejection_diagnostics", {}), dict) else {},
        "near_miss_rows_count": int(scored.get("near_miss_rows_count", 0) or 0),
        "correct_rejected_count": int(scored.get("correct_rejected_count", 0) or 0),
        "incorrect_admitted_count": int(scored.get("incorrect_admitted_count", 0) or 0),
    }


def _safe_selection_guardrails(
    *,
    market: str,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    full_test_rows: int,
    baseline_unknown_count: int = 0,
    candidate_unknown_count: int = 0,
) -> List[str]:
    failures: List[str] = []
    base_full = baseline.get("full_metrics", {}) if isinstance(baseline.get("full_metrics", {}), dict) else {}
    cand_full = candidate.get("full_metrics", {}) if isinstance(candidate.get("full_metrics", {}), dict) else {}
    base_metrics = baseline.get("metrics", {}) if isinstance(baseline.get("metrics", {}), dict) else {}
    cand_metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics", {}), dict) else {}
    for key in ("directional_accuracy_pct", "pnl_trend_match_pct"):
        if _f(cand_full.get(key, 0.0), 0.0) < _f(base_full.get(key, 0.0), 0.0) - 2.0:
            failures.append(f"{key}_regressed")
    if _f(cand_full.get("trigger_match_pct", 0.0), 0.0) < _f(base_full.get("trigger_match_pct", 0.0), 0.0) - 2.0:
        dir_gain = _f(cand_full.get("directional_accuracy_pct", 0.0), 0.0) - _f(base_full.get("directional_accuracy_pct", 0.0), 0.0)
        pnl_gain = _f(cand_full.get("pnl_trend_match_pct", 0.0), 0.0) - _f(base_full.get("pnl_trend_match_pct", 0.0), 0.0)
        if max(dir_gain, pnl_gain) < 4.0:
            failures.append("trigger_match_regressed")
    if full_test_rows >= 40:
        if int(candidate.get("admitted_trades", 0) or 0) < 40:
            failures.append("candidate_admitted_trades_below_40")
        if _f(candidate.get("admission_rate_pct", 0.0), 0.0) < 40.0:
            failures.append("candidate_admission_rate_below_40")
    if int(candidate.get("admitted_trades", 0) or 0) < 20 and int(baseline.get("admitted_trades", 0) or 0) >= 40:
        failures.append("candidate_low_trade_count_vs_broader_baseline")
    if _s(market).lower() == "stocks":
        if _f(cand_metrics.get("directional_accuracy_pct", 0.0), 0.0) < 90.0:
            failures.append("stocks_directional_below_target")
        if _f(cand_metrics.get("trigger_match_pct", 0.0), 0.0) < 90.0:
            failures.append("stocks_trigger_below_target")
        if _f(cand_metrics.get("pnl_trend_match_pct", 0.0), 0.0) < _f(base_metrics.get("pnl_trend_match_pct", 0.0), 0.0) - 1.0:
            failures.append("stocks_pnl_trend_regressed")
    if _s(market).lower() == "forex":
        if _f(cand_metrics.get("trigger_match_pct", 0.0), 0.0) < _f(base_metrics.get("trigger_match_pct", 0.0), 0.0) - 1.0:
            failures.append("forex_trigger_match_regressed")
        if candidate_unknown_count > baseline_unknown_count + 1:
            failures.append("forex_unknown_trigger_increased")
    return failures


def _controlled_rollout_status(
    *,
    market: str,
    metrics: Dict[str, Any],
    ci: Dict[str, Any],
    worst: Dict[str, Any],
    windows: int,
    test_trades: int,
    trigger_scored: float,
    trigger_cov: float,
    admission_rate: float,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    mk = _s(market).lower()
    thresholds = {
        "crypto": {"directional_accuracy_pct": 75.0, "trigger_match_pct": 75.0, "pnl_trend_match_pct": 80.0, "risk_multiplier": 0.25},
        "stocks": {"directional_accuracy_pct": 85.0, "trigger_match_pct": 80.0, "pnl_trend_match_pct": 85.0, "risk_multiplier": 0.50},
        "forex": {"directional_accuracy_pct": 90.0, "trigger_match_pct": 95.0, "pnl_trend_match_pct": 90.0, "risk_multiplier": 1.0},
    }.get(mk, {"directional_accuracy_pct": 85.0, "trigger_match_pct": 80.0, "pnl_trend_match_pct": 85.0, "risk_multiplier": 0.50})
    blockers: List[str] = []
    for key in ("directional_accuracy_pct", "trigger_match_pct", "pnl_trend_match_pct"):
        if _f(metrics.get(key, 0.0), 0.0) < _f(thresholds.get(key, 0.0), 0.0):
            blockers.append(f"{key}_below_controlled_rollout_target")
    for key in ("directional_accuracy_pct", "trigger_match_pct", "pnl_trend_match_pct"):
        ci_lb = _f((ci.get(key, {}) if isinstance(ci.get(key, {}), dict) else {}).get("p05", 0.0), 0.0)
        if ci_lb < max(0.0, _f(thresholds.get(key, 0.0), 0.0) - 10.0):
            blockers.append(f"{key}_ci_too_weak_for_controlled_rollout")
        worst_val = _f(worst.get(key, 0.0), 0.0)
        if worst_val < max(0.0, _f(thresholds.get(key, 0.0), 0.0) - 15.0):
            blockers.append(f"{key}_worst_window_too_weak_for_controlled_rollout")
    if windows < 3:
        blockers.append("walkforward_windows_insufficient_for_controlled_rollout")
    if test_trades < 40:
        blockers.append("test_trades_insufficient_for_controlled_rollout")
    if trigger_scored < 20:
        blockers.append("trigger_scored_trades_insufficient_for_controlled_rollout")
    if trigger_cov < 40.0:
        blockers.append("trigger_coverage_insufficient_for_controlled_rollout")
    if mk == "crypto" and admission_rate < 40.0:
        blockers.append("admission_rate_insufficient_for_controlled_rollout")
    if mk == "crypto":
        latest = payload.get("safe_selection_diagnostics", {}).get("latest", {}) if isinstance(payload.get("safe_selection_diagnostics", {}), dict) else {}
        evals = latest.get("candidate_evaluations", []) if isinstance(latest.get("candidate_evaluations", []), list) else []
        v2 = next((ev for ev in evals if _s(ev.get("variant", "")) == "label_compatible_v2"), {})
        v2_summary = v2.get("summary", {}) if isinstance(v2.get("summary", {}), dict) else {}
        v2_failures = list(v2.get("guardrail_failures", []) or [])
        v2_trades = int(v2_summary.get("admitted_trades", 0) or 0)
        if v2_trades < 35:
            blockers.append("crypto_v2_clean_admissions_below_rollout_floor")
        impermissible = [f for f in v2_failures if f != "candidate_admitted_trades_below_40"]
        if impermissible:
            blockers.append("crypto_v2_guardrail_failures_beyond_trade_floor")
    return {
        "eligible": not blockers,
        "reason": "" if not blockers else blockers[0],
        "blockers": blockers,
        "risk_multiplier_recommended": float(thresholds.get("risk_multiplier", 0.50)),
        "thresholds": thresholds,
    }


def _calibrate_abstain_threshold(train_rows: List[Dict[str, Any]], market: str, *, predictor_variant: str = "candidate") -> Dict[str, Any]:
    if _s(market).lower() == "crypto":
        hist_rows = [r for r in train_rows if _is_historical_strategy_replay_row(r)]
        if hist_rows and len(hist_rows) >= max(20, int(len(train_rows) * 0.60)):
            if len(train_rows) < 24:
                policy = {
                    "source_mode": "historical_strategy_replay",
                    "base_conf_min": 0.34,
                    "base_trigger_margin_min": 0.18,
                    "base_dir_margin_min": 0.05,
                    "stale_conf_min": 0.35,
                    "stale_trigger_margin_min": 0.18,
                    "stale_dir_margin_min": 0.02,
                    "trailing_conf_min": 0.30,
                    "trailing_trigger_margin_min": 0.18,
                    "trailing_dir_margin_min": 0.05,
                    "risk_cut_conf_min": 0.34,
                    "risk_cut_trigger_margin_min": 0.16,
                    "risk_cut_dir_margin_min": 0.03,
                    "take_profit_conf_min": 0.30,
                    "take_profit_trigger_margin_min": 0.16,
                    "take_profit_score_gap_min": 0.10,
                    "coverage_target": 0.40,
                    "min_admitted_trades": 40,
                    "min_trigger_scored": 20,
                    "risk_cut_precision_target_pct": 60.0,
                    "take_profit_precision_target_pct": 55.0,
                    "trailing_precision_target_pct": 60.0,
                    "stale_precision_target_pct": 60.0,
                    "worst_window_floor_target_pct": 85.0,
                }
                return {
                    "policy": policy,
                    "policy_name": _historical_replay_policy_name(policy),
                    "coverage": 1.0,
                    "metrics": _metrics_for_rows(train_rows),
                    "class_precision": {},
                    "reason_counts": {},
                    "policy_frontier": [],
                }
            split = max(12, int(len(train_rows) * 0.70))
            base = train_rows[:split]
            val = train_rows[split:]
            pred_val: List[Dict[str, Any]] = []
            running = list(base)
            for row in val:
                regime = _regime_from_prior(running)
                pred = _predict_one(train_rows=running, candidate=row, regime=regime, market=market, predictor_variant=predictor_variant)
                merged = dict(row)
                merged.update(pred)
                pred_val.append(merged)
                running.append(row)
            policies: List[Dict[str, Any]] = []
            if _s(predictor_variant).lower() == "label_compatible_v2":
                policies.extend(
                    [
                        {
                            "policy_name_override": "label_compatible_v2_strict",
                            "source_mode": "historical_strategy_replay",
                            "base_conf_min": 0.30,
                            "base_trigger_margin_min": 0.18,
                            "base_dir_margin_min": 0.00,
                            "stale_conf_min": 0.31,
                            "stale_trigger_margin_min": 0.18,
                            "stale_dir_margin_min": 0.00,
                            "trailing_conf_min": 0.28,
                            "trailing_trigger_margin_min": 0.16,
                            "trailing_dir_margin_min": 0.00,
                            "risk_cut_conf_min": 0.30,
                            "risk_cut_trigger_margin_min": 0.16,
                            "risk_cut_dir_margin_min": 0.00,
                            "take_profit_conf_min": 0.28,
                            "take_profit_trigger_margin_min": 0.16,
                            "take_profit_score_gap_min": 0.08,
                            "coverage_target": 0.40,
                            "min_admitted_trades": 40,
                            "min_trigger_scored": 20,
                            "risk_cut_precision_target_pct": 60.0,
                            "take_profit_precision_target_pct": 55.0,
                            "trailing_precision_target_pct": 60.0,
                            "stale_precision_target_pct": 60.0,
                            "worst_window_floor_target_pct": 85.0,
                        },
                        {
                            "policy_name_override": "label_compatible_v2_balanced",
                            "source_mode": "historical_strategy_replay",
                            "base_conf_min": 0.28,
                            "base_trigger_margin_min": 0.14,
                            "base_dir_margin_min": 0.00,
                            "stale_conf_min": 0.29,
                            "stale_trigger_margin_min": 0.14,
                            "stale_dir_margin_min": 0.00,
                            "trailing_conf_min": 0.26,
                            "trailing_trigger_margin_min": 0.14,
                            "trailing_dir_margin_min": 0.00,
                            "risk_cut_conf_min": 0.28,
                            "risk_cut_trigger_margin_min": 0.14,
                            "risk_cut_dir_margin_min": 0.00,
                            "take_profit_conf_min": 0.26,
                            "take_profit_trigger_margin_min": 0.14,
                            "take_profit_score_gap_min": 0.06,
                            "coverage_target": 0.40,
                            "min_admitted_trades": 40,
                            "min_trigger_scored": 20,
                            "risk_cut_precision_target_pct": 58.0,
                            "take_profit_precision_target_pct": 52.0,
                            "trailing_precision_target_pct": 58.0,
                            "stale_precision_target_pct": 58.0,
                            "worst_window_floor_target_pct": 85.0,
                        },
                        {
                            "policy_name_override": "label_compatible_v2_coverage_rescue",
                            "source_mode": "historical_strategy_replay",
                            "base_conf_min": 0.26,
                            "base_trigger_margin_min": 0.12,
                            "base_dir_margin_min": 0.00,
                            "stale_conf_min": 0.28,
                            "stale_trigger_margin_min": 0.12,
                            "stale_dir_margin_min": 0.00,
                            "trailing_conf_min": 0.25,
                            "trailing_trigger_margin_min": 0.12,
                            "trailing_dir_margin_min": 0.00,
                            "risk_cut_conf_min": 0.26,
                            "risk_cut_trigger_margin_min": 0.12,
                            "risk_cut_dir_margin_min": 0.00,
                            "take_profit_conf_min": 0.25,
                            "take_profit_trigger_margin_min": 0.12,
                            "take_profit_score_gap_min": 0.05,
                            "coverage_target": 0.40,
                            "min_admitted_trades": 40,
                            "min_trigger_scored": 20,
                            "risk_cut_precision_target_pct": 56.0,
                            "take_profit_precision_target_pct": 50.0,
                            "trailing_precision_target_pct": 56.0,
                            "stale_precision_target_pct": 56.0,
                            "worst_window_floor_target_pct": 85.0,
                            "enable_risk_cut_rescue": True,
                            "enable_trailing_rescue": True,
                            "enable_stale_rescue": True,
                            "enable_take_profit_rescue": True,
                            "rescue_margin_band": 0.05,
                            "rescue_conf_band": 0.05,
                            "rescue_dir_band": 0.05,
                        },
                        {
                            "policy_name_override": "label_compatible_v2_clean_expansion",
                            "source_mode": "historical_strategy_replay",
                            "base_conf_min": 0.27,
                            "base_trigger_margin_min": 0.13,
                            "base_dir_margin_min": 0.00,
                            "stale_conf_min": 0.29,
                            "stale_trigger_margin_min": 0.13,
                            "stale_dir_margin_min": 0.00,
                            "trailing_conf_min": 0.26,
                            "trailing_trigger_margin_min": 0.13,
                            "trailing_dir_margin_min": 0.00,
                            "risk_cut_conf_min": 0.27,
                            "risk_cut_trigger_margin_min": 0.13,
                            "risk_cut_dir_margin_min": 0.00,
                            "take_profit_conf_min": 0.25,
                            "take_profit_trigger_margin_min": 0.12,
                            "take_profit_score_gap_min": 0.05,
                            "coverage_target": 0.40,
                            "min_admitted_trades": 40,
                            "min_trigger_scored": 20,
                            "risk_cut_precision_target_pct": 58.0,
                            "take_profit_precision_target_pct": 52.0,
                            "trailing_precision_target_pct": 58.0,
                            "stale_precision_target_pct": 58.0,
                            "worst_window_floor_target_pct": 85.0,
                            "enable_clean_expansion": True,
                            "rescue_margin_band": 0.04,
                            "rescue_conf_band": 0.04,
                            "rescue_dir_band": 0.04,
                        },
                        {
                            "policy_name_override": "label_compatible_v2_no_abstain_diagnostic",
                            "source_mode": "historical_strategy_replay",
                            "base_conf_min": 0.0,
                            "base_trigger_margin_min": 0.0,
                            "base_dir_margin_min": 0.0,
                            "stale_conf_min": 0.0,
                            "stale_trigger_margin_min": 0.0,
                            "stale_dir_margin_min": 0.0,
                            "trailing_conf_min": 0.0,
                            "trailing_trigger_margin_min": 0.0,
                            "trailing_dir_margin_min": 0.0,
                            "risk_cut_conf_min": 0.0,
                            "risk_cut_trigger_margin_min": 0.0,
                            "risk_cut_dir_margin_min": 0.0,
                            "take_profit_conf_min": 0.0,
                            "take_profit_trigger_margin_min": 0.0,
                            "take_profit_score_gap_min": 0.0,
                            "coverage_target": 0.40,
                            "min_admitted_trades": 40,
                            "min_trigger_scored": 20,
                            "risk_cut_precision_target_pct": 50.0,
                            "take_profit_precision_target_pct": 45.0,
                            "trailing_precision_target_pct": 50.0,
                            "stale_precision_target_pct": 50.0,
                            "worst_window_floor_target_pct": 85.0,
                        },
                    ]
                )
            else:
                for base_conf in [0.30, 0.34, 0.38]:
                    for base_trig in [0.12, 0.18, 0.26]:
                        for base_dir in [0.00, 0.04, 0.08]:
                            policies.append(
                                {
                                    "source_mode": "historical_strategy_replay",
                                    "base_conf_min": base_conf,
                                    "base_trigger_margin_min": base_trig,
                                    "base_dir_margin_min": base_dir,
                                    "stale_conf_min": base_conf + 0.01,
                                    "stale_trigger_margin_min": base_trig,
                                    "stale_dir_margin_min": max(0.0, base_dir - 0.02),
                                    "trailing_conf_min": max(0.24, base_conf - 0.02),
                                    "trailing_trigger_margin_min": max(0.10, base_trig - 0.02),
                                    "trailing_dir_margin_min": base_dir,
                                    "risk_cut_conf_min": base_conf,
                                    "risk_cut_trigger_margin_min": max(0.10, base_trig - 0.02),
                                    "risk_cut_dir_margin_min": max(0.0, base_dir - 0.02),
                                    "take_profit_conf_min": max(0.24, base_conf - 0.02),
                                    "take_profit_trigger_margin_min": max(0.10, base_trig - 0.02),
                                    "take_profit_score_gap_min": 0.08 if base_conf <= 0.34 else 0.12,
                                    "coverage_target": 0.40,
                                    "min_admitted_trades": 40,
                                    "min_trigger_scored": 20,
                                    "risk_cut_precision_target_pct": 60.0,
                                    "take_profit_precision_target_pct": 55.0,
                                    "trailing_precision_target_pct": 60.0,
                                    "stale_precision_target_pct": 60.0,
                                    "worst_window_floor_target_pct": 85.0,
                                }
                            )
            frontier: List[Dict[str, Any]] = []
            default_policy = policies[0] if policies else {}
            best = {
                "score": -1e9,
                "coverage": 0.0,
                "metrics": _metrics_for_rows(pred_val),
                "policy": default_policy,
                "policy_name": _historical_replay_policy_name(default_policy) if default_policy else "",
                "class_precision": {},
                "reason_counts": {},
                "policy_frontier": [],
            }
            for policy in policies:
                row = _score_crypto_with_policy(pred_val, policy)
                summary = _frontier_candidate_summary(policy, row)
                summary["worst_window_metrics"] = {
                    "directional_accuracy_pct": _f(row.get("metrics", {}).get("directional_accuracy_pct", 0.0), 0.0),
                    "trigger_match_pct": _f(row.get("metrics", {}).get("trigger_match_pct", 0.0), 0.0),
                    "pnl_trend_match_pct": _f(row.get("metrics", {}).get("pnl_trend_match_pct", 0.0), 0.0),
                }
                summary["correct_rejected_count"] = int(row.get("correct_rejected_count", 0) or 0)
                summary["incorrect_admitted_count"] = int(row.get("incorrect_admitted_count", 0) or 0)
                frontier.append(summary)
                if _f(row.get("score", -1e9), -1e9) > _f(best.get("score", -1e9), -1e9):
                    best = {
                        "score": _f(row.get("score", -1e9), -1e9),
                        "coverage": _f(row.get("coverage", 0.0), 0.0),
                        "metrics": row.get("metrics", {}),
                        "policy": dict(policy),
                        "policy_name": _historical_replay_policy_name(policy),
                        "class_precision": row.get("class_precision", {}),
                        "reason_counts": row.get("reason_counts", {}),
                        "policy_frontier": [],
                    }
            frontier_sorted = sorted(frontier, key=lambda item: (_f(item.get("score", 0.0), 0.0), _f(item.get("trigger_match_pct", 0.0), 0.0), _f(item.get("admission_rate_pct", 0.0), 0.0)), reverse=True)[:10]
            best["policy_frontier"] = frontier_sorted
            return best
        if len(train_rows) < 24:
            policy = {
                "base_conf_min": 0.50,
                "stale_conf_min": 0.56,
                "stale_trigger_margin_min": 2.5,
                "stale_dir_margin_min": 1.75,
                "trailing_conf_min": 0.30,
                "trailing_trigger_margin_min": 2.25,
                "trailing_dir_margin_min": 1.25,
                "trailing_vs_stale_min": 0.0,
                "manual_conf_min": 0.30,
                "manual_runner_margin_min": -0.35,
                "manual_trigger_margin_min": 0.50,
                "manual_dir_margin_min": 0.50,
                "manual_support_min": 2,
                "manual_long_hold_ratio_min": 0.20,
                "manual_vs_stale_ratio_min": 0.10,
                "manual_vs_stale_long_hold_ratio_min": 0.15,
                "manual_recent_density_min": 0.02,
                "manual_signal_votes_min": 3,
                "manual_score_advantage_min": -0.25,
                "manual_precision_target_pct": 75.0,
                "trailing_precision_target_pct": 65.0,
                "worst_window_floor_target_pct": 85.0,
                "coverage_target": 0.40,
                "min_admitted_trades": 40,
                "min_trigger_scored": 20,
                "min_trailing_admit_ratio": 0.35,
            }
            return {"policy": policy, "coverage": 1.0, "metrics": _metrics_for_rows(train_rows)}
        split = max(12, int(len(train_rows) * 0.70))
        base = train_rows[:split]
        val = train_rows[split:]
        pred_val: List[Dict[str, Any]] = []
        running = list(base)
        for row in val:
            regime = _regime_from_prior(running)
            pred = _predict_one(train_rows=running, candidate=row, regime=regime, market=market, predictor_variant=predictor_variant)
            merged = dict(row)
            merged.update(pred)
            pred_val.append(merged)
            running.append(row)
        policies: List[Dict[str, Any]] = []
        for trailing_conf in [0.24, 0.28]:
            policies.append(
                {
                    "base_conf_min": 0.50,
                    "stale_conf_min": 0.52,
                    "stale_trigger_margin_min": 2.0,
                    "stale_dir_margin_min": 1.25,
                    "trailing_conf_min": trailing_conf,
                    "trailing_trigger_margin_min": 1.75,
                    "trailing_dir_margin_min": 1.0,
                    "trailing_vs_stale_min": -0.10,
                    "manual_mode": "off",
                    "manual_precision_target_pct": 75.0,
                    "trailing_precision_target_pct": 65.0,
                    "worst_window_floor_target_pct": 85.0,
                    "coverage_target": 0.40,
                    "min_admitted_trades": 40,
                    "min_trigger_scored": 20,
                    "min_trailing_admit_ratio": 0.35,
                    "trailing_mode": "high_margin",
                }
            )
            policies.append(
                {
                    "base_conf_min": 0.50,
                    "stale_conf_min": 0.52,
                    "stale_trigger_margin_min": 2.0,
                    "stale_dir_margin_min": 1.25,
                    "trailing_conf_min": trailing_conf,
                    "trailing_trigger_margin_min": 1.75,
                    "trailing_dir_margin_min": 1.0,
                    "trailing_vs_stale_min": -0.10,
                    "manual_mode": "diagnostic_only",
                    "manual_diag_same_symbol_min": 2,
                    "manual_diag_vs_stale_long_hold_min": 0.45,
                    "manual_conf_min": 0.24,
                    "manual_runner_margin_min": -0.25,
                    "manual_trigger_margin_min": 0.25,
                    "manual_dir_margin_min": 0.25,
                    "manual_support_min": 2,
                    "manual_long_hold_ratio_min": 0.30,
                    "manual_vs_stale_ratio_min": 0.25,
                    "manual_vs_stale_long_hold_ratio_min": 0.35,
                    "manual_recent_density_min": 0.02,
                    "manual_signal_votes_min": 3,
                    "manual_symbol_support_min": 2,
                    "manual_score_advantage_min": -0.10,
                    "manual_precision_target_pct": 75.0,
                    "trailing_precision_target_pct": 65.0,
                    "worst_window_floor_target_pct": 85.0,
                    "coverage_target": 0.40,
                    "min_admitted_trades": 40,
                    "min_trigger_scored": 20,
                    "min_trailing_admit_ratio": 0.35,
                    "trailing_mode": "high_margin",
                }
            )
        for stale_conf in [0.52, 0.56]:
            for trailing_conf in [0.24, 0.28]:
                for manual_conf in [0.24, 0.28]:
                    for manual_support in [2, 3]:
                        for manual_margin in [-0.25, 0.0]:
                            for manual_long_hold_min in [0.30, 0.40]:
                                for manual_votes in [3, 4]:
                                    policies.append(
                                        {
                                            "base_conf_min": 0.50,
                                            "stale_conf_min": stale_conf,
                                            "stale_trigger_margin_min": 2.0 if stale_conf <= 0.54 else 2.5,
                                            "stale_dir_margin_min": 1.25 if stale_conf <= 0.54 else 1.75,
                                            "trailing_conf_min": trailing_conf,
                                            "trailing_trigger_margin_min": 1.75,
                                            "trailing_dir_margin_min": 1.0,
                                            "trailing_vs_stale_min": -0.10,
                                            "manual_mode": "strict",
                                            "manual_conf_min": manual_conf,
                                            "manual_runner_margin_min": manual_margin,
                                            "manual_trigger_margin_min": 0.25,
                                            "manual_dir_margin_min": 0.25,
                                            "manual_support_min": manual_support,
                                            "manual_long_hold_ratio_min": manual_long_hold_min,
                                            "manual_vs_stale_ratio_min": 0.25,
                                            "manual_vs_stale_long_hold_ratio_min": 0.35,
                                            "manual_recent_density_min": 0.02,
                                            "manual_signal_votes_min": manual_votes,
                                            "manual_symbol_support_min": 2,
                                            "manual_score_advantage_min": -0.10,
                                            "manual_precision_target_pct": 75.0,
                                            "trailing_precision_target_pct": 65.0,
                                            "worst_window_floor_target_pct": 85.0,
                                            "coverage_target": 0.40,
                                            "min_admitted_trades": 40,
                                            "min_trigger_scored": 20,
                                            "min_trailing_admit_ratio": 0.35,
                                            "trailing_mode": "high_margin",
                                        }
                                    )
        frontier: List[Dict[str, Any]] = []
        best = {"score": -1e9, "coverage": 0.0, "metrics": _metrics_for_rows(pred_val), "policy": policies[0], "policy_frontier": []}
        for policy in policies:
            row = _score_crypto_with_policy(pred_val, policy)
            summary = _frontier_candidate_summary(policy, row)
            summary["worst_window_metrics"] = {
                "directional_accuracy_pct": _f(row.get("metrics", {}).get("directional_accuracy_pct", 0.0), 0.0),
                "trigger_match_pct": _f(row.get("metrics", {}).get("trigger_match_pct", 0.0), 0.0),
                "pnl_trend_match_pct": _f(row.get("metrics", {}).get("pnl_trend_match_pct", 0.0), 0.0),
            }
            manual_prec = _f(summary["class_precision"].get("Manual", {}).get("precision_pct", 0.0), 0.0)
            trigger_match = _f(summary.get("trigger_match_pct", 0.0), 0.0)
            coverage_pct = _f(summary.get("admission_rate_pct", 0.0), 0.0)
            if manual_prec < 75.0 and summary["class_counts"].get("Manual", 0) > 0:
                summary["blockers"].append(f"manual_precision_low({manual_prec:.2f}<75.00)")
            if trigger_match < 90.0:
                summary["blockers"].append(f"trigger_match_low({trigger_match:.2f}<90.00)")
            if coverage_pct < 40.0:
                summary["blockers"].append(f"coverage_low({coverage_pct:.2f}<40.00)")
            frontier.append(summary)
            score = _f(row.get("score", -1e9), -1e9)
            cov = _f(row.get("coverage", 0.0), 0.0)
            if score > _f(best.get("score", -1e9), -1e9) or (
                abs(score - _f(best.get("score", -1e9), -1e9)) <= 1e-9 and cov > _f(best.get("coverage", 0.0), 0.0)
            ):
                best = {
                    "policy": dict(policy),
                    "score": score,
                    "coverage": cov,
                    "metrics": row.get("metrics", {}),
                    "full_metrics": row.get("full_metrics", {}),
                    "abstained_metrics": row.get("abstained_metrics", {}),
                    "reason_counts": row.get("reason_counts", {}),
                    "class_precision": row.get("class_precision", {}),
                    "policy_name": _policy_name(policy),
                }
        frontier_sorted = sorted(frontier, key=lambda item: (-_f(item.get("score", 0.0), 0.0), -_f(item.get("trigger_match_pct", 0.0), 0.0), -_f(item.get("admission_rate_pct", 0.0), 0.0)))
        best["policy_frontier"] = frontier_sorted[:8]
        return best
    if len(train_rows) < 24:
        return {"threshold": 0.6, "coverage": 1.0, "metrics": _metrics_for_rows(train_rows)}
    split = max(12, int(len(train_rows) * 0.70))
    base = train_rows[:split]
    val = train_rows[split:]
    pred_val: List[Dict[str, Any]] = []
    running = list(base)
    for row in val:
        regime = _regime_from_prior(running)
        pred = _predict_one(train_rows=running, candidate=row, regime=regime, market=market, predictor_variant=predictor_variant)
        merged = dict(row)
        merged.update(pred)
        pred_val.append(merged)
        running.append(row)
    if _s(market).lower() == "stocks":
        if _s(predictor_variant).lower() == "stock_pnl_quality_v2":
            thresholds = [
                ("no_abstain_full_universe", 0.0),
                ("low_threshold", 0.42),
                ("balanced_threshold", 0.50),
                ("quality_gate_threshold", 0.58),
                ("high_confidence_threshold", 0.68),
            ]
        else:
            thresholds = [
                ("no_abstain_full_universe", 0.0),
                ("low_threshold", 0.45),
                ("balanced_threshold", 0.55),
                ("high_confidence_threshold", 0.70),
            ]
    elif _s(market).lower() == "forex":
        thresholds = [
            ("no_abstain_full_universe", 0.0),
            ("balanced_threshold", 0.55),
            ("high_confidence_threshold", 0.70),
        ]
    else:
        thresholds = [(f"threshold_{thr:.2f}", thr) for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]]
    best = {"threshold": 0.6, "score": -1e9, "coverage": 0.0, "metrics": _metrics_for_rows(pred_val), "policy_frontier": []}
    frontier: List[Dict[str, Any]] = []
    for policy_name, thr in thresholds:
        row = _score_with_threshold(pred_val, thr, market)
        summary = _selection_option_summary(policy_name, row, predictor_variant=predictor_variant)
        frontier.append(summary)
        score = _f(row.get("score", -1e9), -1e9)
        cov = _f(row.get("coverage", 0.0), 0.0)
        if score > _f(best.get("score", -1e9), -1e9) or (
            abs(score - _f(best.get("score", -1e9), -1e9)) <= 1e-9 and cov > _f(best.get("coverage", 0.0), 0.0)
        ):
            best = {"threshold": thr, "score": score, "coverage": cov, "metrics": row.get("metrics", {}), "policy_name": policy_name}
    best["policy_frontier"] = frontier
    return best


def _bootstrap_ci(rows: List[Dict[str, Any]], metric_key: str, rounds: int = 300) -> Dict[str, float]:
    if not rows:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    vals: List[float] = []
    n = len(rows)
    rng = random.Random(7)
    for _ in range(max(40, int(rounds))):
        sample = [rows[rng.randrange(0, n)] for _ in range(n)]
        m = _metrics_for_rows(sample)
        vals.append(_f(m.get(metric_key, 0.0), 0.0))
    vals.sort()
    def q(p: float) -> float:
        idx = int(round((len(vals) - 1) * p))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])
    return {"p05": round(q(0.05), 4), "p50": round(q(0.50), 4), "p95": round(q(0.95), 4)}


def _apply_replay_deterministic_filters(
    rows: List[Dict[str, Any]],
    *,
    start_ts: int = 0,
    cutoff_ts: int = 0,
    locked_symbols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    locked = {_s(s).upper() for s in list(locked_symbols or []) if _s(s)}
    out: List[Dict[str, Any]] = []
    for row in list(rows or []):
        entry_ts = int(_f(row.get("entry_ts", 0.0), 0.0))
        exit_ts = int(_f(row.get("exit_ts", 0.0), 0.0))
        symbol = _s(row.get("symbol", "")).upper()
        if start_ts > 0 and entry_ts < start_ts:
            continue
        if cutoff_ts > 0 and exit_ts > cutoff_ts:
            continue
        if locked and symbol not in locked:
            continue
        out.append(dict(row))
    out.sort(key=_stable_row_sort_key)
    return out


def _load_locked_symbol_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        payload = _safe_read_json(path)
        return [str(s) for s in list(payload.get("symbols", []) or []) if _s(s)]
    except Exception:
        return []


HISTORICAL_BLIND_SIM_SCHEMA_VERSION = 1
HISTORICAL_BLIND_SIM_SOURCE_TYPE = "historical_blind_strategy_simulation"


def _historical_blind_sim_market_dir(hub_dir: str, market: str) -> str:
    return os.path.join(hub_dir, _s(market).lower())


def _historical_blind_sim_paths(hub_dir: str, market: str) -> Dict[str, str]:
    base = _historical_blind_sim_market_dir(hub_dir, market)
    return {
        "trades": os.path.join(base, "historical_blind_simulation_trades.jsonl"),
        "decisions": os.path.join(base, "historical_blind_simulation_decisions.jsonl"),
        "skips": os.path.join(base, "historical_blind_simulation_skips.jsonl"),
        "summary": os.path.join(base, "historical_blind_simulation_summary.json"),
    }


def _crypto_blind_residual_artifact_paths(hub_dir: str) -> Dict[str, str]:
    base = os.path.join(hub_dir, "crypto")
    return {
        "rows": os.path.join(base, "blind_sim_residual_mismatches.jsonl"),
        "summary": os.path.join(base, "blind_sim_residual_mismatch_summary.json"),
    }


def _historical_blind_sim_symbol_dir(hub_dir: str, market: str, symbol: str) -> str:
    safe_symbol = re.sub(r"[^A-Z0-9._-]+", "_", _s(symbol).upper()) or "UNKNOWN"
    return os.path.join(_historical_blind_sim_market_dir(hub_dir, market), "symbol_onboarding", safe_symbol)


def _symbol_onboarding_paths(hub_dir: str, market: str, symbol: str) -> Dict[str, str]:
    base = _historical_blind_sim_symbol_dir(hub_dir, market, symbol)
    return {
        "status": os.path.join(base, "onboarding_status.json"),
        "trades": os.path.join(base, "historical_blind_simulation_trades.jsonl"),
        "decisions": os.path.join(base, "historical_blind_simulation_decisions.jsonl"),
        "skips": os.path.join(base, "historical_blind_simulation_skips.jsonl"),
        "summary": os.path.join(base, "historical_blind_simulation_summary.json"),
    }


def _symbol_bucket_paths(hub_dir: str, market: str) -> Dict[str, str]:
    base = _historical_blind_sim_market_dir(hub_dir, market)
    return {
        "candidate_universe": os.path.join(base, "candidate_universe.json"),
        "onboarding_queue": os.path.join(base, "onboarding_queue.json"),
        "active_scan_set": os.path.join(base, "active_scan_set.json"),
        "trade_eligible_set": os.path.join(base, "trade_eligible_set.json"),
        "rejected_or_cooled_down_symbols": os.path.join(base, "rejected_or_cooled_down_symbols.json"),
    }


def _historical_blind_sim_enabled_markets() -> List[str]:
    raw = _env_str("HISTORICAL_BLIND_SIM_MARKETS", "crypto,stocks,forex")
    out: List[str] = []
    for tok in raw.replace(";", ",").split(","):
        mk = _s(tok).lower()
        if mk in {"crypto", "stocks", "forex"} and mk not in out:
            out.append(mk)
    return out or ["crypto", "stocks", "forex"]


def _split_walkforward_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ordered = sorted([dict(r) for r in list(rows or []) if isinstance(r, dict)], key=_stable_row_sort_key)
    n = len(ordered)
    if n <= 0:
        return {"train": [], "validation": [], "test": []}
    train_end = max(1, int(math.floor(n * 0.70)))
    val_end = max(train_end, int(math.floor(n * 0.85)))
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def _actual_pnl_trend(row: Dict[str, Any]) -> str:
    existing = _normalize_pnl_trend_label(row.get("actual_pnl_trend", ""))
    if existing:
        return existing
    pnl_pct = _f(row.get("pnl_pct", _trade_return_pct(row)), _trade_return_pct(row))
    if pnl_pct > 1e-9:
        return "up"
    if pnl_pct < -1e-9:
        return "down"
    return "flat"


def _normalize_pnl_trend_label(value: Any) -> str:
    txt = _s(value).lower()
    if txt in {"up", "profit", "positive", "favorable", "true", "1", "bullish", "gain"}:
        return "up"
    if txt in {"down", "loss", "negative", "unfavorable", "false", "0", "bearish"}:
        return "down"
    if txt in {"flat", "neutral", "even", "unknown", "none", "null"}:
        return "flat"
    return ""


def _historical_blind_prediction_semantics(row: Dict[str, Any], market: str) -> Dict[str, Any]:
    predictor_variant = _s(row.get("predictor_variant", "")).lower()
    predicted_direction = _s(row.get("predicted_direction", "")).lower()
    predicted_trigger = _s(row.get("predicted_exit_trigger", ""))
    predicted_pnl_trend = _normalize_pnl_trend_label(row.get("predicted_pnl_trend", ""))
    actual_pnl_trend = _actual_pnl_trend(row)
    confidence = _f(row.get("confidence", row.get("predicted_confidence", 0.0)), 0.0)
    direction_scores = row.get("direction_scores", {}) if isinstance(row.get("direction_scores", {}), dict) else {}
    trigger_scores = row.get("trigger_scores", {}) if isinstance(row.get("trigger_scores", {}), dict) else {}
    trigger_margin = _f(row.get("trigger_score_selected_margin", row.get("trigger_margin", 0.0)), 0.0)
    source_feature_availability = _s(row.get("source_feature_availability", ""))
    prediction_before_outcome = bool(int(_f(row.get("entry_ts", 0.0), 0.0)) > 0 and int(_f(row.get("exit_ts", 0.0), 0.0)) > int(_f(row.get("entry_ts", 0.0), 0.0)))
    placeholder_only = bool(
        ("fallback" in predictor_variant)
        or (predicted_trigger in {"", "Unknown"})
        or (not predicted_direction)
        or (not predicted_pnl_trend)
        or (confidence <= 0.0 and not direction_scores and not trigger_scores)
    )
    low_margin = bool(trigger_margin < (0.12 if _s(market).lower() == "crypto" else 0.0))
    label_alignment_blockers: List[str] = []
    if not predicted_pnl_trend:
        label_alignment_blockers.append("predicted_pnl_trend_unmapped")
    if not actual_pnl_trend:
        label_alignment_blockers.append("actual_pnl_trend_unmapped")
    leakage_detected = not prediction_before_outcome
    if leakage_detected:
        label_alignment_blockers.append("entry_exit_sequence_invalid")
    eligible_for_training = bool(
        prediction_before_outcome
        and (not placeholder_only)
        and bool(predicted_direction)
        and bool(predicted_trigger and predicted_trigger != "Unknown")
        and bool(predicted_pnl_trend)
        and bool(actual_pnl_trend)
        and not leakage_detected
        and not low_margin
        and source_feature_availability != "fallback_only"
    )
    reason = "eligible"
    if not prediction_before_outcome:
        reason = "prediction_not_before_outcome"
    elif placeholder_only:
        reason = "prediction_semantics_placeholder_or_fallback"
    elif low_margin:
        reason = "trigger_margin_below_diagnostic_floor"
    elif not predicted_trigger or predicted_trigger == "Unknown":
        reason = "predicted_trigger_unknown"
    elif not predicted_pnl_trend or not actual_pnl_trend:
        reason = "pnl_trend_label_unaligned"
    return {
        "prediction_semantics": "heuristic",
        "prediction_semantics_warning": f"{_s(market).lower()}_blind_sim_predictions_are_history-derived_heuristics_not_live_strategy_orders",
        "prediction_before_outcome": prediction_before_outcome,
        "placeholder_only": placeholder_only,
        "label_alignment_blockers": label_alignment_blockers,
        "label_alignment_status": "aligned" if not label_alignment_blockers else "blocked",
        "eligible_for_training": eligible_for_training,
        "training_eligibility_reason": reason,
        "diagnostic_only": bool(not eligible_for_training),
        "future_leakage_detected": bool(leakage_detected),
        "trigger_margin_below_floor": bool(low_margin),
    }


def _historical_blind_bucket(value: Any, *, kind: str) -> str:
    v = _f(value, 0.0)
    if kind == "bars":
        if v <= 3:
            return "<=3"
        if v <= 6:
            return "4-6"
        if v <= 12:
            return "7-12"
        if v <= 24:
            return "13-24"
        return "25+"
    if kind == "hours":
        if v <= 6:
            return "<=6h"
        if v <= 24:
            return "6-24h"
        if v <= 72:
            return "1-3d"
        return "3d+"
    if kind == "pct_abs":
        v = abs(v)
        if v < 0.5:
            return "<0.5"
        if v < 1.5:
            return "0.5-1.5"
        if v < 3.0:
            return "1.5-3.0"
        return "3.0+"
    if kind == "conf":
        if v < 0.25:
            return "<0.25"
        if v < 0.5:
            return "0.25-0.49"
        if v < 0.75:
            return "0.50-0.74"
        return "0.75+"
    return _s(value) or "unknown"


def _historical_blind_confusion(rows: List[Dict[str, Any]], actual_key: str, predicted_key: str) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for row in list(rows or []):
        actual = _s(row.get(actual_key, "")) or "unknown"
        predicted = _s(row.get(predicted_key, "")) or "unknown"
        matrix.setdefault(actual, {})
        matrix[actual][predicted] = int(matrix[actual].get(predicted, 0) + 1)
    return matrix


def _historical_blind_confusion_pair_count(confusion: Dict[str, Dict[str, int]], actual_label: str, predicted_label: str) -> int:
    row = confusion.get(actual_label, {}) if isinstance(confusion.get(actual_label, {}), dict) else {}
    return int(_f(row.get(predicted_label, 0), 0.0))


def _crypto_residual_mismatch_pairs() -> List[Tuple[str, str]]:
    return [
        ("Stale Alignment", "Risk Cut"),
        ("Stale Alignment", "Take Profit"),
        ("Trailing", "Risk Cut"),
        ("Trailing", "Take Profit"),
        ("Take Profit", "Risk Cut"),
        ("Risk Cut", "Take Profit"),
        ("Risk Cut", "Trailing"),
        ("Take Profit", "Trailing"),
    ]


def _historical_blind_diagnostics(
    *,
    market: str,
    rows: List[Dict[str, Any]],
    status_by_symbol: Dict[str, Dict[str, Any]],
    active_rows: List[Dict[str, Any]],
    eligible_rows: List[Dict[str, Any]],
    rejected_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    working = [dict(r) for r in list(rows or []) if isinstance(r, dict)]
    predicted_direction_counts = _count_by(working, lambda r: _s(r.get("predicted_direction", "")).lower() or "unknown")
    actual_direction_counts = _count_by(working, lambda r: _s(r.get("actual_direction", "")).lower() or "unknown")
    predicted_trigger_counts = _count_by(working, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown")
    actual_trigger_counts = _count_by(working, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown")
    predicted_pnl_trend_counts = _count_by(working, lambda r: _normalize_pnl_trend_label(r.get("predicted_pnl_trend", "")) or "unknown")
    actual_pnl_trend_counts = _count_by(working, lambda r: _actual_pnl_trend(r) or "unknown")
    mismatches = [
        r for r in working
        if (
            _s(r.get("predicted_direction", "")).lower() != _s(r.get("actual_direction", "")).lower()
            or _s(r.get("predicted_exit_trigger", "")) != _s(r.get("actual_exit_trigger", ""))
            or _normalize_pnl_trend_label(r.get("predicted_pnl_trend", "")) != _actual_pnl_trend(r)
        )
    ]
    for row in working:
        row["predicted_pnl_trend"] = _normalize_pnl_trend_label(row.get("predicted_pnl_trend", "")) or _s(row.get("predicted_pnl_trend", ""))
        row["actual_pnl_trend"] = _actual_pnl_trend(row)
    blocker_counts: Dict[str, int] = {}
    for st in list(status_by_symbol.values()):
        if not isinstance(st, dict):
            continue
        for blocker in list(st.get("blockers", []) or []):
            b = _s(blocker) or "unknown"
            blocker_counts[b] = int(blocker_counts.get(b, 0) + 1)
    if not blocker_counts and not eligible_rows:
        blocker_counts["trade_eligibility_not_promoted_by_existing_readiness_gates"] = int(max(1, len(active_rows) or len(working)))
    diag_only_rows = sum(1 for r in working if bool(r.get("diagnostic_only", False)))
    eligible_training_rows = sum(1 for r in working if bool(r.get("eligible_for_training", False)))
    diagnostic_only_reason_counts = _count_by([r for r in working if bool(r.get("diagnostic_only", False))], lambda r: _s(r.get("training_eligibility_reason", "")) or "unknown")
    training_eligible_counts_by_trigger = _count_by([r for r in working if bool(r.get("eligible_for_training", False))], lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown")
    selected_trigger_margin_buckets = _count_by(working, lambda r: _historical_blind_bucket(r.get("trigger_score_selected_margin", r.get("trigger_margin", 0.0)), kind="conf"))
    trigger_score_distribution = {
        "risk_cut": _count_by(working, lambda r: _historical_blind_bucket(r.get("trigger_score_risk_cut", 0.0), kind="conf")),
        "take_profit": _count_by(working, lambda r: _historical_blind_bucket(r.get("trigger_score_take_profit", 0.0), kind="conf")),
        "trailing": _count_by(working, lambda r: _historical_blind_bucket(r.get("trigger_score_trailing", 0.0), kind="conf")),
        "stale_alignment": _count_by(working, lambda r: _historical_blind_bucket(r.get("trigger_score_stale_alignment", 0.0), kind="conf")),
    }
    sample = []
    for row in mismatches[:12]:
        sample.append(
            {
                "symbol": _s(row.get("symbol", "")),
                "predictor_variant": _s(row.get("predictor_variant", "")),
                "predicted_direction": _s(row.get("predicted_direction", "")),
                "actual_direction": _s(row.get("actual_direction", "")),
                "predicted_exit_trigger": _s(row.get("predicted_exit_trigger", "")),
                "actual_exit_trigger": _s(row.get("actual_exit_trigger", "")),
                "predicted_pnl_trend": _normalize_pnl_trend_label(row.get("predicted_pnl_trend", "")) or _s(row.get("predicted_pnl_trend", "")),
                "actual_pnl_trend": _actual_pnl_trend(row),
                "hold_time": round(_f(row.get("hold_time", row.get("hold_hours", 0.0)), 0.0), 6),
                "bars_in_trade": int(_f(row.get("bars_in_trade", 0.0), 0.0)),
                "confidence": round(_f(row.get("confidence", 0.0), 0.0), 6),
                "training_eligibility_reason": _s(row.get("training_eligibility_reason", "")),
            }
        )
    top_mismatch_buckets = {
        "symbol": _top_counter(_count_by(mismatches, lambda r: _s(r.get("symbol", "")) or "UNKNOWN")),
        "predicted_trigger": _top_counter(_count_by(mismatches, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown")),
        "actual_trigger": _top_counter(_count_by(mismatches, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown")),
        "hold_time_bucket": _top_counter(_count_by(mismatches, lambda r: _historical_blind_bucket(r.get("hold_time", r.get("hold_hours", 0.0)), kind="hours"))),
        "bars_in_trade_bucket": _top_counter(_count_by(mismatches, lambda r: _historical_blind_bucket(r.get("bars_in_trade", 0.0), kind="bars"))),
        "volatility_bucket": _top_counter(_count_by(mismatches, lambda r: _historical_blind_bucket(r.get("recent_volatility", 0.0), kind="pct_abs"))),
        "drawdown_bucket": _top_counter(_count_by(mismatches, lambda r: _historical_blind_bucket(r.get("drawdown_from_peak_pct", 0.0), kind="pct_abs"))),
        "source_feature_availability": _top_counter(_count_by(mismatches, lambda r: "fallback_or_placeholder" if bool(r.get("prediction_semantics_placeholder_only", False)) else "scored_features_available")),
        "confidence_bucket": _top_counter(_count_by(mismatches, lambda r: _historical_blind_bucket(r.get("confidence", 0.0), kind="conf"))),
    }
    semantics_values = sorted({_s(r.get("prediction_semantics", "")) for r in working if _s(r.get("prediction_semantics", ""))})
    warnings = sorted({_s(r.get("prediction_semantics_warning", "")) for r in working if _s(r.get("prediction_semantics_warning", ""))})
    label_alignment_blockers = sorted({b for r in working for b in list(r.get("label_alignment_blockers", []) or []) if _s(b)})
    residual_risk_trailing = [r for r in mismatches if _s(r.get("actual_exit_trigger", "")) == "Risk Cut" and _s(r.get("predicted_exit_trigger", "")) == "Trailing"]
    residual_tp_trailing = [r for r in mismatches if _s(r.get("actual_exit_trigger", "")) == "Take Profit" and _s(r.get("predicted_exit_trigger", "")) == "Trailing"]
    residual_stale_trailing = [r for r in mismatches if _s(r.get("actual_exit_trigger", "")) == "Stale Alignment" and _s(r.get("predicted_exit_trigger", "")) == "Trailing"]
    discriminator_counts = _count_by(
        [r for r in working if bool(r.get("second_stage_discriminator_applied", False))],
        lambda r: f"{_s(r.get('second_stage_discriminator_from', 'Unknown'))}->{_s(r.get('second_stage_discriminator_to', 'Unknown'))}",
    )
    discriminator_reason_counts = _count_by(
        [r for r in working if bool(r.get("second_stage_discriminator_applied", False))],
        lambda r: _s(r.get("second_stage_discriminator_reason", "")) or "unknown",
    )
    residual_pair_counts = {
        f"{actual}->{predicted}": _historical_blind_confusion_pair_count(_historical_blind_confusion(working, "actual_exit_trigger", "predicted_exit_trigger"), actual, predicted)
        for actual, predicted in _crypto_residual_mismatch_pairs()
    }
    return {
        "prediction_semantics": semantics_values[0] if len(semantics_values) == 1 else ",".join(semantics_values),
        "prediction_semantics_warning": warnings[0] if warnings else "",
        "predicted_direction_counts": predicted_direction_counts,
        "actual_direction_counts": actual_direction_counts,
        "direction_confusion_matrix": _historical_blind_confusion(working, "actual_direction", "predicted_direction"),
        "predicted_trigger_counts": predicted_trigger_counts,
        "actual_trigger_counts": actual_trigger_counts,
        "trigger_confusion_matrix": _historical_blind_confusion(working, "actual_exit_trigger", "predicted_exit_trigger"),
        "predicted_pnl_trend_counts": predicted_pnl_trend_counts,
        "actual_pnl_trend_counts": actual_pnl_trend_counts,
        "pnl_trend_confusion_matrix": _historical_blind_confusion(working, "actual_pnl_trend", "predicted_pnl_trend"),
        "top_mismatch_buckets": top_mismatch_buckets,
        "sample_mismatch_rows": sample,
        "label_alignment_status": "aligned" if not label_alignment_blockers else "blocked",
        "label_alignment_blockers": label_alignment_blockers,
        "eligible_for_training_rows": int(eligible_training_rows),
        "diagnostic_only_rows": int(diag_only_rows),
        "diagnostic_only_reason_counts": diagnostic_only_reason_counts,
        "training_eligible_counts_by_trigger": training_eligible_counts_by_trigger,
        "selected_trigger_margin_buckets": selected_trigger_margin_buckets,
        "trigger_score_distribution": trigger_score_distribution,
        "trade_eligible_set_blocker_counts": _top_counter(blocker_counts),
        "risk_cut_to_trailing_count": int(len(residual_risk_trailing)),
        "take_profit_to_trailing_count": int(len(residual_tp_trailing)),
        "stale_alignment_to_trailing_count": int(len(residual_stale_trailing)),
        "stale_alignment_to_risk_cut_count": int(residual_pair_counts.get("Stale Alignment->Risk Cut", 0)),
        "stale_alignment_to_take_profit_count": int(residual_pair_counts.get("Stale Alignment->Take Profit", 0)),
        "trailing_to_risk_cut_count": int(residual_pair_counts.get("Trailing->Risk Cut", 0)),
        "trailing_to_take_profit_count": int(residual_pair_counts.get("Trailing->Take Profit", 0)),
        "take_profit_to_risk_cut_count": int(residual_pair_counts.get("Take Profit->Risk Cut", 0)),
        "risk_cut_to_take_profit_count": int(residual_pair_counts.get("Risk Cut->Take Profit", 0)),
        "residual_pair_counts": residual_pair_counts,
        "second_stage_discriminator_applied_counts": discriminator_counts,
        "second_stage_discriminator_reason_counts": discriminator_reason_counts,
        "top_risk_cut_to_trailing_residual_mismatches": [
            {
                "symbol": _s(r.get("symbol", "")),
                "confidence": round(_f(r.get("confidence", 0.0), 0.0), 6),
                "trigger_score_reason": _s(r.get("trigger_score_reason", "")),
                "trigger_score_selected_margin": round(_f(r.get("trigger_score_selected_margin", r.get("trigger_margin", 0.0)), 0.0), 6),
            }
            for r in residual_risk_trailing[:6]
        ],
        "top_take_profit_to_trailing_residual_mismatches": [
            {
                "symbol": _s(r.get("symbol", "")),
                "confidence": round(_f(r.get("confidence", 0.0), 0.0), 6),
                "trigger_score_reason": _s(r.get("trigger_score_reason", "")),
                "trigger_score_selected_margin": round(_f(r.get("trigger_score_selected_margin", r.get("trigger_margin", 0.0)), 0.0), 6),
            }
            for r in residual_tp_trailing[:6]
        ],
        "top_stale_alignment_to_trailing_residual_mismatches": [
            {
                "symbol": _s(r.get("symbol", "")),
                "confidence": round(_f(r.get("confidence", 0.0), 0.0), 6),
                "trigger_score_reason": _s(r.get("trigger_score_reason", "")),
                "trigger_score_selected_margin": round(_f(r.get("trigger_score_selected_margin", r.get("trigger_margin", 0.0)), 0.0), 6),
            }
            for r in residual_stale_trailing[:6]
        ],
    }


def _crypto_blind_before_after_summary(
    baseline_rows: List[Dict[str, Any]],
    improved_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base_diag = _historical_blind_diagnostics(
        market="crypto",
        rows=baseline_rows,
        status_by_symbol={},
        active_rows=[],
        eligible_rows=[],
        rejected_rows=[],
    )
    after_diag = _historical_blind_diagnostics(
        market="crypto",
        rows=improved_rows,
        status_by_symbol={},
        active_rows=[],
        eligible_rows=[],
        rejected_rows=[],
    )
    return {
        "before_predicted_trigger_counts": base_diag.get("predicted_trigger_counts", {}),
        "after_predicted_trigger_counts": after_diag.get("predicted_trigger_counts", {}),
        "before_actual_trigger_counts": base_diag.get("actual_trigger_counts", {}),
        "after_actual_trigger_counts": after_diag.get("actual_trigger_counts", {}),
        "before_trigger_confusion_matrix": base_diag.get("trigger_confusion_matrix", {}),
        "after_trigger_confusion_matrix": after_diag.get("trigger_confusion_matrix", {}),
        "risk_cut_to_trailing_before": int(base_diag.get("risk_cut_to_trailing_count", 0) or 0),
        "risk_cut_to_trailing_after": int(after_diag.get("risk_cut_to_trailing_count", 0) or 0),
        "take_profit_to_trailing_before": int(base_diag.get("take_profit_to_trailing_count", 0) or 0),
        "take_profit_to_trailing_after": int(after_diag.get("take_profit_to_trailing_count", 0) or 0),
        "stale_alignment_to_trailing_before": int(base_diag.get("stale_alignment_to_trailing_count", 0) or 0),
        "stale_alignment_to_trailing_after": int(after_diag.get("stale_alignment_to_trailing_count", 0) or 0),
        "stale_alignment_to_risk_cut_before": int(base_diag.get("stale_alignment_to_risk_cut_count", 0) or 0),
        "stale_alignment_to_risk_cut_after": int(after_diag.get("stale_alignment_to_risk_cut_count", 0) or 0),
        "stale_alignment_to_take_profit_before": int(base_diag.get("stale_alignment_to_take_profit_count", 0) or 0),
        "stale_alignment_to_take_profit_after": int(after_diag.get("stale_alignment_to_take_profit_count", 0) or 0),
        "trailing_to_risk_cut_before": int(base_diag.get("trailing_to_risk_cut_count", 0) or 0),
        "trailing_to_risk_cut_after": int(after_diag.get("trailing_to_risk_cut_count", 0) or 0),
        "trailing_to_take_profit_before": int(base_diag.get("trailing_to_take_profit_count", 0) or 0),
        "trailing_to_take_profit_after": int(after_diag.get("trailing_to_take_profit_count", 0) or 0),
        "take_profit_to_risk_cut_before": int(base_diag.get("take_profit_to_risk_cut_count", 0) or 0),
        "take_profit_to_risk_cut_after": int(after_diag.get("take_profit_to_risk_cut_count", 0) or 0),
        "risk_cut_to_take_profit_before": int(base_diag.get("risk_cut_to_take_profit_count", 0) or 0),
        "risk_cut_to_take_profit_after": int(after_diag.get("risk_cut_to_take_profit_count", 0) or 0),
        "training_eligible_rows_before": int(base_diag.get("eligible_for_training_rows", 0) or 0),
        "training_eligible_rows_after": int(after_diag.get("eligible_for_training_rows", 0) or 0),
        "diagnostic_only_rows_before": int(base_diag.get("diagnostic_only_rows", 0) or 0),
        "diagnostic_only_rows_after": int(after_diag.get("diagnostic_only_rows", 0) or 0),
        "diagnostic_only_reasons_before": base_diag.get("diagnostic_only_reason_counts", {}),
        "diagnostic_only_reasons_after": after_diag.get("diagnostic_only_reason_counts", {}),
        "second_stage_discriminator_applied_counts_after": after_diag.get("second_stage_discriminator_applied_counts", {}),
        "second_stage_discriminator_reason_counts_after": after_diag.get("second_stage_discriminator_reason_counts", {}),
        "validation_trigger_metric_before": _historical_blind_metrics(_split_walkforward_rows(baseline_rows).get("validation", [])).get("trigger_match_pct", 0.0),
        "validation_trigger_metric_after": _historical_blind_metrics(_split_walkforward_rows(improved_rows).get("validation", [])).get("trigger_match_pct", 0.0),
        "test_trigger_metric_before": _historical_blind_metrics(_split_walkforward_rows(baseline_rows).get("test", [])).get("trigger_match_pct", 0.0),
        "test_trigger_metric_after": _historical_blind_metrics(_split_walkforward_rows(improved_rows).get("test", [])).get("trigger_match_pct", 0.0),
    }


def _export_crypto_blind_residual_mismatches(hub_dir: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    paths = _crypto_blind_residual_artifact_paths(hub_dir)
    tracked_pairs = set(_crypto_residual_mismatch_pairs())
    exported: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        actual_trigger = _s(row.get("actual_exit_trigger", "")) or "Unknown"
        predicted_trigger = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
        if actual_trigger == predicted_trigger or (actual_trigger, predicted_trigger) not in tracked_pairs:
            continue
        exported.append(
            {
                "symbol": _s(row.get("symbol", "")),
                "entry_ts": int(_f(row.get("entry_ts", 0.0), 0.0)),
                "exit_ts": int(_f(row.get("exit_ts", 0.0), 0.0)),
                "actual_trigger": actual_trigger,
                "predicted_trigger": predicted_trigger,
                "trigger_scores": dict(row.get("trigger_scores", {}) if isinstance(row.get("trigger_scores", {}), dict) else {}),
                "selected_margin": round(_f(row.get("trigger_score_selected_margin", row.get("trigger_margin", 0.0)), 0.0), 6),
                "hold_time": round(_f(row.get("hold_time", row.get("hold_hours", 0.0)), 0.0), 6),
                "bars_in_trade": int(_f(row.get("bars_in_trade", 0.0), 0.0)),
                "current_unrealized_pnl_pct": round(_f(row.get("current_unrealized_pnl_pct", 0.0), 0.0), 6),
                "max_favorable_excursion_pct_so_far": round(_f(row.get("max_favorable_excursion_pct_so_far", 0.0), 0.0), 6),
                "max_adverse_excursion_pct_so_far": round(_f(row.get("max_adverse_excursion_pct_so_far", 0.0), 0.0), 6),
                "drawdown_from_peak_pct_so_far": round(_f(row.get("drawdown_from_peak_pct_so_far", 0.0), 0.0), 6),
                "max_favorable_excursion_pct": round(_f(row.get("max_favorable_excursion_pct", 0.0), 0.0), 6),
                "max_adverse_excursion_pct": round(_f(row.get("max_adverse_excursion_pct", 0.0), 0.0), 6),
                "drawdown_from_peak_pct": round(_f(row.get("drawdown_from_peak_pct", 0.0), 0.0), 6),
                "trailing_armed": bool(row.get("trailing_armed", False)),
                "bars_since_trailing_armed": int(_f(row.get("bars_since_trailing_armed", -1), -1.0)),
                "risk_pressure_score": round(_f(row.get("risk_pressure_score", 0.0), 0.0), 6),
                "take_profit_pressure_score": round(_f(row.get("take_profit_pressure_score", 0.0), 0.0), 6),
                "trailing_quality_score": round(_f(row.get("trailing_quality_score", 0.0), 0.0), 6),
                "stale_alignment_score": round(_f(row.get("stale_alignment_score", 0.0), 0.0), 6),
                "momentum_decay": round(_f(row.get("momentum_decay", 0.0), 0.0), 6),
                "recent_volatility": round(_f(row.get("recent_volatility", 0.0), 0.0), 6),
                "pnl_pct": round(_f(row.get("pnl_pct", 0.0), 0.0), 6),
                "late_favorable_reversal_score": round(_f(row.get("late_favorable_reversal_score", 0.0), 0.0), 6),
                "favorable_then_reversed": bool(row.get("favorable_then_reversed", False)),
                "max_favorable_before_exit_pct": round(_f(row.get("max_favorable_before_exit_pct", 0.0), 0.0), 6),
                "reversal_from_peak_before_exit_pct": round(_f(row.get("reversal_from_peak_before_exit_pct", 0.0), 0.0), 6),
                "bars_from_peak_to_exit_preview": int(_f(row.get("bars_from_peak_to_exit_preview", 0.0), 0.0)),
                "trailing_arm_to_exit_bars_preview": int(_f(row.get("trailing_arm_to_exit_bars_preview", -1), -1.0)),
                "risk_pressure_after_peak": round(_f(row.get("risk_pressure_after_peak", 0.0), 0.0), 6),
                "take_profit_pressure_before_reversal": round(_f(row.get("take_profit_pressure_before_reversal", 0.0), 0.0), 6),
                "target_near_before_reversal": bool(row.get("target_near_before_reversal", False)),
                "reversal_velocity_pct_per_bar": round(_f(row.get("reversal_velocity_pct_per_bar", 0.0), 0.0), 6),
                "post_peak_momentum_decay": round(_f(row.get("post_peak_momentum_decay", 0.0), 0.0), 6),
                "drawdown_after_favorable_move_pct": round(_f(row.get("drawdown_after_favorable_move_pct", 0.0), 0.0), 6),
                "favorable_move_quality_score": round(_f(row.get("favorable_move_quality_score", 0.0), 0.0), 6),
                "late_risk_after_favorable_move": bool(row.get("late_risk_after_favorable_move", False)),
                "eligible_for_training": bool(row.get("eligible_for_training", False)),
                "diagnostic_only": bool(row.get("diagnostic_only", False)),
                "training_eligibility_reason": _s(row.get("training_eligibility_reason", "")),
                "second_stage_discriminator_applied": bool(row.get("second_stage_discriminator_applied", False)),
                "second_stage_discriminator_from": _s(row.get("second_stage_discriminator_from", "")),
                "second_stage_discriminator_to": _s(row.get("second_stage_discriminator_to", "")),
                "second_stage_discriminator_reason": _s(row.get("second_stage_discriminator_reason", "")),
                "second_stage_margin_context": _s(row.get("second_stage_margin_context", "")),
                "second_stage_residual_family": _s(row.get("second_stage_residual_family", "")),
            }
        )
    _write_jsonl(paths["rows"], exported)
    summary = {
        "rows": int(len(exported)),
        "pairs": _count_by(exported, lambda r: f"{_s(r.get('actual_trigger', 'Unknown'))}->{_s(r.get('predicted_trigger', 'Unknown'))}"),
        "discriminator_counts": _count_by(
            [r for r in exported if bool(r.get("second_stage_discriminator_applied", False))],
            lambda r: f"{_s(r.get('second_stage_discriminator_from', 'Unknown'))}->{_s(r.get('second_stage_discriminator_to', 'Unknown'))}",
        ),
    }
    _write_json_atomic(paths["summary"], summary)
    return {"paths": paths, "summary": summary}


def _safe_predict_blind_row(
    *,
    market: str,
    history_rows: List[Dict[str, Any]],
    candidate_row: Dict[str, Any],
    crypto_trigger_mode: str = "improved",
) -> Dict[str, Any]:
    mk = _s(market).lower()
    if mk == "stocks":
        if not history_rows:
            return {
                "predictor_variant": "stock_pnl_quality_v2_fallback",
                "predicted_direction": "up",
                "predicted_exit_trigger": "Unknown",
                "predicted_pnl_trend": "up",
                "predicted_confidence": 0.0,
                "direction_scores": {},
                "trigger_scores": {},
                "stock_trade_quality_score": 0.0,
                "stock_pnl_quality_score": 0.0,
            }
        return _stock_predict_one(
            train_rows=history_rows,
            candidate=dict(candidate_row),
            regime=_s(candidate_row.get("regime", "historical_blind_strategy_simulation")) or "historical_blind_strategy_simulation",
            predictor_variant="stock_pnl_quality_v2",
        )
    if mk == "crypto" and _s(crypto_trigger_mode).lower() == "baseline":
        if not history_rows:
            return {
                "predictor_variant": "crypto_baseline_fallback",
                "predicted_direction": "up",
                "predicted_exit_trigger": "Unknown",
                "predicted_pnl_trend": "up",
                "predicted_confidence": 0.0,
                "direction_scores": {},
                "trigger_scores": {},
            }
        return _predict_one(
            train_rows=history_rows,
            candidate=dict(candidate_row),
            regime=_s(candidate_row.get("regime", "historical_blind_strategy_simulation")) or "historical_blind_strategy_simulation",
            market=mk or "crypto",
        )
    if not history_rows:
        return {
            "predictor_variant": "crypto_baseline_fallback",
            "predicted_direction": "up",
            "predicted_exit_trigger": "Unknown",
            "predicted_pnl_trend": "up",
            "predicted_confidence": 0.0,
            "direction_scores": {},
            "trigger_scores": {},
        }
    return _crypto_blind_trigger_predict(
        train_rows=history_rows,
        candidate=dict(candidate_row),
        regime=_s(candidate_row.get("regime", "historical_blind_strategy_simulation")) or "historical_blind_strategy_simulation",
    )


def _crypto_blind_sequence_feature_snapshot(candidate: Dict[str, Any]) -> Dict[str, Any]:
    momentum = _f(candidate.get("trend_momentum_score", 0.0), 0.0)
    volatility = _f(candidate.get("recent_volatility", 0.0), 0.0)
    signal_margin = _f(candidate.get("signal_margin", 0.0), 0.0)
    move_3 = _f(candidate.get("recent_return_3", 0.0), 0.0)
    move_6 = _f(candidate.get("recent_return_6", 0.0), 0.0)
    move_12 = _f(candidate.get("recent_return_12", 0.0), 0.0)
    candle_move = _f(candidate.get("current_candle_pct_move", 0.0), 0.0)
    active_tfs = _f(candidate.get("active_timeframe_count", 0.0), 0.0)
    bars_since_entry = int(max(0.0, _f(candidate.get("bars_since_entry", candidate.get("preview_bars_used", 0.0)), 0.0)))
    current_unrealized = _f(candidate.get("current_unrealized_pnl_pct", 0.0), 0.0)
    mfe_so_far = max(0.0, _f(candidate.get("max_favorable_excursion_pct_so_far", 0.0), 0.0))
    mae_so_far = min(0.0, _f(candidate.get("max_adverse_excursion_pct_so_far", 0.0), 0.0))
    drawdown_from_peak = min(0.0, _f(candidate.get("drawdown_from_peak_pct_so_far", 0.0), 0.0))
    trailing_armed = bool(candidate.get("trailing_armed_so_far", candidate.get("trailing_armed", False)))
    bars_since_trailing_armed = int(_f(candidate.get("bars_since_trailing_armed", -1), -1.0))
    risk_cut_distance_pct = max(0.0, _f(candidate.get("risk_cut_distance_pct", 2.25), 2.25))
    take_profit_distance_pct = max(0.0, _f(candidate.get("take_profit_distance_pct", 4.25), 4.25))
    risk_cut_touched = bool(candidate.get("risk_cut_touched_so_far", False))
    take_profit_touched = bool(candidate.get("take_profit_touched_so_far", False))
    peak_to_current_reversal_pct = max(0.0, _f(candidate.get("peak_to_current_reversal_pct", 0.0), 0.0))
    favorable_then_softened = bool(candidate.get("favorable_then_softened_flag_so_far", candidate.get("favorable_then_softened_flag", False)))
    momentum_decay = max(0.0, _f(candidate.get("momentum_decay", max(0.0, move_12 - move_3)), 0.0))
    late_favorable_reversal_score = _f(candidate.get("late_favorable_reversal_score", 0.0), 0.0)
    favorable_then_reversed = bool(candidate.get("favorable_then_reversed", False))
    max_favorable_before_exit_pct = max(0.0, _f(candidate.get("max_favorable_before_exit_pct", mfe_so_far), 0.0))
    reversal_from_peak_before_exit_pct = max(0.0, _f(candidate.get("reversal_from_peak_before_exit_pct", peak_to_current_reversal_pct), 0.0))
    bars_from_peak_to_exit_preview = int(_f(candidate.get("bars_from_peak_to_exit_preview", 0), 0.0))
    trailing_arm_to_exit_bars_preview = int(_f(candidate.get("trailing_arm_to_exit_bars_preview", bars_since_trailing_armed), 0.0))
    risk_pressure_after_peak = _f(candidate.get("risk_pressure_after_peak", 0.0), 0.0)
    take_profit_pressure_before_reversal = _f(candidate.get("take_profit_pressure_before_reversal", 0.0), 0.0)
    target_near_before_reversal = bool(candidate.get("target_near_before_reversal", False))
    reversal_velocity_pct_per_bar = _f(candidate.get("reversal_velocity_pct_per_bar", 0.0), 0.0)
    post_peak_momentum_decay = _f(candidate.get("post_peak_momentum_decay", momentum_decay), 0.0)
    drawdown_after_favorable_move_pct = _f(candidate.get("drawdown_after_favorable_move_pct", peak_to_current_reversal_pct), 0.0)
    favorable_move_quality_score = _f(candidate.get("favorable_move_quality_score", 0.0), 0.0)
    late_risk_after_favorable_move = bool(candidate.get("late_risk_after_favorable_move", False))
    return {
        "bars_since_entry": bars_since_entry,
        "current_unrealized_pnl_pct": round(current_unrealized, 6),
        "max_favorable_excursion_pct_so_far": round(mfe_so_far, 6),
        "max_adverse_excursion_pct_so_far": round(mae_so_far, 6),
        "drawdown_from_peak_pct_so_far": round(drawdown_from_peak, 6),
        "trailing_armed": trailing_armed,
        "bars_since_trailing_armed": bars_since_trailing_armed,
        "risk_cut_distance_pct": round(risk_cut_distance_pct, 6),
        "take_profit_distance_pct": round(take_profit_distance_pct, 6),
        "risk_cut_touched_so_far": risk_cut_touched,
        "take_profit_touched_so_far": take_profit_touched,
        "peak_to_current_reversal_pct": round(peak_to_current_reversal_pct, 6),
        "recent_return_3": round(move_3, 6),
        "recent_return_6": round(move_6, 6),
        "recent_volatility": round(volatility, 6),
        "momentum_decay": round(momentum_decay, 6),
        "stale_alignment_score": round(
            max(0.0, ((0.30 - signal_margin) / 0.30))
            + max(0.0, ((1.60 - momentum) / 1.60))
            + max(0.0, ((bars_since_entry - 4.0) / 4.0))
            + max(0.0, ((1.20 - mfe_so_far) / 1.20)),
            6,
        ),
        "risk_pressure_score": round(
            max(0.0, ((0.20 - signal_margin) / 0.20))
            + max(0.0, ((0.75 - move_6) / 1.50))
            + max(0.0, ((0.70 - candle_move) / 1.20))
            + (1.25 if risk_cut_touched else 0.0)
            + max(0.0, ((1.10 - risk_cut_distance_pct) / 1.10))
            + max(0.0, (abs(drawdown_from_peak) / 0.80))
            + max(0.0, (abs(mae_so_far) / 1.00)),
            6,
        ),
        "take_profit_pressure_score": round(
            max(0.0, ((move_6 - 1.60) / 1.60))
            + max(0.0, ((move_12 - 2.10) / 1.80))
            + max(0.0, (mfe_so_far / 1.40))
            + (1.10 if take_profit_touched else 0.0)
            + max(0.0, ((1.10 - take_profit_distance_pct) / 1.10)),
            6,
        ),
        "trailing_quality_score": round(
            max(0.0, ((signal_margin - 0.28) / 0.28))
            + max(0.0, ((momentum - 1.70) / 1.40))
            + max(0.0, ((active_tfs - 4.0) / 4.0))
            + (0.90 if trailing_armed else 0.0)
            + max(0.0, (mfe_so_far / 1.30))
            + max(0.0, (peak_to_current_reversal_pct / 0.60))
            - (0.70 if risk_cut_touched else 0.0),
            6,
        ),
        "favorable_then_softened_flag_so_far": favorable_then_softened,
        "late_favorable_reversal_score": round(late_favorable_reversal_score, 6),
        "favorable_then_reversed": favorable_then_reversed,
        "max_favorable_before_exit_pct": round(max_favorable_before_exit_pct, 6),
        "reversal_from_peak_before_exit_pct": round(reversal_from_peak_before_exit_pct, 6),
        "bars_from_peak_to_exit_preview": bars_from_peak_to_exit_preview,
        "trailing_arm_to_exit_bars_preview": trailing_arm_to_exit_bars_preview,
        "risk_pressure_after_peak": round(risk_pressure_after_peak, 6),
        "take_profit_pressure_before_reversal": round(take_profit_pressure_before_reversal, 6),
        "target_near_before_reversal": target_near_before_reversal,
        "reversal_velocity_pct_per_bar": round(reversal_velocity_pct_per_bar, 6),
        "post_peak_momentum_decay": round(post_peak_momentum_decay, 6),
        "drawdown_after_favorable_move_pct": round(drawdown_after_favorable_move_pct, 6),
        "favorable_move_quality_score": round(favorable_move_quality_score, 6),
        "late_risk_after_favorable_move": late_risk_after_favorable_move,
    }


def _crypto_apply_second_stage_discriminator(context: Dict[str, Any]) -> Dict[str, Any]:
    pred_trigger = _s(context.get("pred_trigger", "")) or "Unknown"
    trigger_scores = context.get("trigger_scores", {}) if isinstance(context.get("trigger_scores", {}), dict) else {}
    selected_margin = _f(context.get("selected_margin", 0.0), 0.0)
    risk_pressure_score = _f(context.get("risk_pressure_score", 0.0), 0.0)
    take_profit_pressure_score = _f(context.get("take_profit_pressure_score", 0.0), 0.0)
    trailing_quality_score = _f(context.get("trailing_quality_score", 0.0), 0.0)
    stale_alignment_score = _f(context.get("stale_alignment_score", 0.0), 0.0)
    bars_since_entry = int(_f(context.get("bars_since_entry", 0), 0.0))
    mfe_so_far = _f(context.get("max_favorable_excursion_pct_so_far", 0.0), 0.0)
    mae_so_far_abs = abs(min(0.0, _f(context.get("max_adverse_excursion_pct_so_far", 0.0), 0.0)))
    drawdown_so_far_abs = abs(min(0.0, _f(context.get("drawdown_from_peak_pct_so_far", 0.0), 0.0)))
    trailing_armed = bool(context.get("trailing_armed", False))
    bars_since_trailing_armed = int(_f(context.get("bars_since_trailing_armed", -1), -1.0))
    risk_cut_distance_pct = _f(context.get("risk_cut_distance_pct", 2.25), 2.25)
    take_profit_distance_pct = _f(context.get("take_profit_distance_pct", 4.25), 4.25)
    risk_cut_touched_so_far = bool(context.get("risk_cut_touched_so_far", False))
    take_profit_touched_so_far = bool(context.get("take_profit_touched_so_far", False))
    peak_to_current_reversal_pct = _f(context.get("peak_to_current_reversal_pct", 0.0), 0.0)
    favorable_then_softened = bool(context.get("favorable_then_softened_flag_so_far", False))
    momentum_decay = _f(context.get("momentum_decay", 0.0), 0.0)
    late_favorable_reversal_score = _f(context.get("late_favorable_reversal_score", 0.0), 0.0)
    favorable_then_reversed = bool(context.get("favorable_then_reversed", False))
    max_favorable_before_exit_pct = _f(context.get("max_favorable_before_exit_pct", mfe_so_far), 0.0)
    reversal_from_peak_before_exit_pct = _f(context.get("reversal_from_peak_before_exit_pct", peak_to_current_reversal_pct), 0.0)
    bars_from_peak_to_exit_preview = int(_f(context.get("bars_from_peak_to_exit_preview", 0), 0.0))
    trailing_arm_to_exit_bars_preview = int(_f(context.get("trailing_arm_to_exit_bars_preview", bars_since_trailing_armed), 0.0))
    risk_pressure_after_peak = _f(context.get("risk_pressure_after_peak", 0.0), 0.0)
    take_profit_pressure_before_reversal = _f(context.get("take_profit_pressure_before_reversal", 0.0), 0.0)
    target_near_before_reversal = bool(context.get("target_near_before_reversal", False))
    reversal_velocity_pct_per_bar = _f(context.get("reversal_velocity_pct_per_bar", 0.0), 0.0)
    post_peak_momentum_decay = _f(context.get("post_peak_momentum_decay", momentum_decay), 0.0)
    drawdown_after_favorable_move_pct = _f(context.get("drawdown_after_favorable_move_pct", peak_to_current_reversal_pct), 0.0)
    favorable_move_quality_score = _f(context.get("favorable_move_quality_score", 0.0), 0.0)
    late_risk_after_favorable_move = bool(context.get("late_risk_after_favorable_move", False))
    plausible_gap = _f(context.get("plausible_gap", 2.0), 2.0)
    high_margin_guard = _f(context.get("high_margin_guard", 3.25), 3.25)
    margin_context = _s(context.get("margin_context", ""))
    if pred_trigger in {"", "Unknown"} or selected_margin >= high_margin_guard:
        return {
            "pred_trigger": pred_trigger,
            "selected_margin": selected_margin,
            "second_stage_discriminator_applied": False,
            "second_stage_discriminator_from": "",
            "second_stage_discriminator_to": "",
            "second_stage_discriminator_reason": "",
            "second_stage_margin_context": margin_context,
            "second_stage_residual_family": "",
        }
    stale_candidate = (
        bars_since_entry >= 6
        and momentum_decay >= 0.8
        and mfe_so_far <= 1.8
        and mae_so_far_abs <= 1.6
        and drawdown_so_far_abs <= 1.0
        and risk_cut_distance_pct >= 0.05
        and take_profit_distance_pct >= 1.4
        and not risk_cut_touched_so_far
        and not take_profit_touched_so_far
    )
    trailing_candidate = (
        trailing_armed
        and mfe_so_far >= 1.6
        and peak_to_current_reversal_pct >= 0.45
        and drawdown_so_far_abs <= 1.5
        and risk_cut_distance_pct >= 0.55
        and not risk_cut_touched_so_far
        and (favorable_then_softened or bars_since_trailing_armed >= 1)
    )
    take_profit_candidate = (
        (take_profit_touched_so_far or take_profit_distance_pct <= 0.55 or mfe_so_far >= 3.0)
        and risk_cut_distance_pct >= 0.45
        and (mfe_so_far - mae_so_far_abs) >= 1.0
        and not risk_cut_touched_so_far
    )
    def _apply(new_trigger: str, reason: str, family: str) -> Dict[str, Any]:
        new_margin = max(0.0, _f(trigger_scores.get(new_trigger, 0.0), 0.0) - max((_f(v, 0.0) for k, v in trigger_scores.items() if k != new_trigger), default=0.0))
        return {
            "pred_trigger": new_trigger,
            "selected_margin": new_margin,
            "second_stage_discriminator_applied": True,
            "second_stage_discriminator_from": pred_trigger,
            "second_stage_discriminator_to": new_trigger,
            "second_stage_discriminator_reason": reason,
            "second_stage_margin_context": margin_context,
            "second_stage_residual_family": family,
        }
    if pred_trigger in {"Risk Cut", "Take Profit"} and stale_candidate and risk_pressure_score <= (stale_alignment_score + 0.35):
        if (_f(trigger_scores.get(pred_trigger, 0.0), 0.0) - _f(trigger_scores.get("Stale Alignment", 0.0), 0.0)) <= plausible_gap:
            return _apply("Stale Alignment", "slow_decay_restored_stale_alignment", "stale_vs_risk_or_take_profit")
    if pred_trigger == "Risk Cut" and trailing_candidate and (risk_pressure_score <= (trailing_quality_score + 0.75)):
        if (_f(trigger_scores.get("Risk Cut", 0.0), 0.0) - _f(trigger_scores.get("Trailing", 0.0), 0.0)) <= plausible_gap:
            return _apply("Trailing", "risk_after_favorable_move_restored_trailing", "trailing_vs_risk_cut")
    if pred_trigger == "Take Profit" and trailing_candidate and (take_profit_pressure_score <= (trailing_quality_score + 0.65)):
        if (_f(trigger_scores.get("Take Profit", 0.0), 0.0) - _f(trigger_scores.get("Trailing", 0.0), 0.0)) <= plausible_gap:
            return _apply("Trailing", "late_favorable_reversal_restored_trailing", "trailing_vs_take_profit")
    if pred_trigger == "Risk Cut" and take_profit_candidate and not risk_cut_touched_so_far and mae_so_far_abs <= 1.2:
        if (_f(trigger_scores.get("Risk Cut", 0.0), 0.0) - _f(trigger_scores.get("Take Profit", 0.0), 0.0)) <= (plausible_gap + 1.5):
            return _apply("Take Profit", "target_before_reversal_restored_take_profit", "take_profit_vs_risk_cut")
    if pred_trigger == "Take Profit":
        risk_before_target = (
            late_risk_after_favorable_move
            or (risk_pressure_after_peak >= (take_profit_pressure_before_reversal + 0.85) and not target_near_before_reversal)
            or (drawdown_after_favorable_move_pct >= 1.0 and max_favorable_before_exit_pct < 1.0)
        )
        if risk_before_target and (_f(trigger_scores.get("Take Profit", 0.0), 0.0) - _f(trigger_scores.get("Risk Cut", 0.0), 0.0)) <= plausible_gap:
            return _apply("Risk Cut", "risk_before_target_restored_risk_cut", "risk_cut_vs_take_profit")
    if pred_trigger == "Take Profit":
        trailing_late_reversal = (
            favorable_then_reversed
            and late_favorable_reversal_score >= 1.8
            and trailing_armed
            and reversal_from_peak_before_exit_pct >= 0.55
            and bars_from_peak_to_exit_preview <= 4
            and trailing_arm_to_exit_bars_preview >= 0
            and not late_risk_after_favorable_move
        )
        if trailing_late_reversal and (_f(trigger_scores.get("Take Profit", 0.0), 0.0) - _f(trigger_scores.get("Trailing", 0.0), 0.0)) <= (plausible_gap + 0.8):
            return _apply("Trailing", "late_favorable_reversal_restored_trailing", "trailing_vs_take_profit")
    if pred_trigger == "Risk Cut":
        trailing_after_favorable = (
            favorable_then_reversed
            and late_favorable_reversal_score >= 1.8
            and trailing_armed
            and reversal_from_peak_before_exit_pct >= 0.55
            and risk_pressure_after_peak <= (take_profit_pressure_before_reversal + 1.25)
            and not risk_cut_touched_so_far
        )
        if trailing_after_favorable and (_f(trigger_scores.get("Risk Cut", 0.0), 0.0) - _f(trigger_scores.get("Trailing", 0.0), 0.0)) <= (plausible_gap + 0.8):
            return _apply("Trailing", "risk_after_favorable_move_restored_trailing", "trailing_vs_risk_cut")
    if pred_trigger == "Risk Cut":
        target_before_reversal = (
            target_near_before_reversal
            and take_profit_pressure_before_reversal >= 1.5
            and max_favorable_before_exit_pct >= 1.6
            and reversal_velocity_pct_per_bar <= 0.75
            and not risk_cut_touched_so_far
        )
        if target_before_reversal and (_f(trigger_scores.get("Risk Cut", 0.0), 0.0) - _f(trigger_scores.get("Take Profit", 0.0), 0.0)) <= (plausible_gap + 1.2):
            return _apply("Take Profit", "target_before_reversal_restored_take_profit", "take_profit_vs_risk_cut")
    if pred_trigger == "Risk Cut":
        slow_decay = (
            bars_since_entry >= 12
            and post_peak_momentum_decay >= 0.8
            and stale_alignment_score >= 0.65
            and risk_pressure_after_peak <= 3.0
            and take_profit_pressure_before_reversal <= 2.8
        )
        if slow_decay and (_f(trigger_scores.get("Risk Cut", 0.0), 0.0) - _f(trigger_scores.get("Stale Alignment", 0.0), 0.0)) <= (plausible_gap + 0.6):
            return _apply("Stale Alignment", "slow_decay_restored_stale_alignment", "stale_vs_risk_cut")
    return {
        "pred_trigger": pred_trigger,
        "selected_margin": selected_margin,
        "second_stage_discriminator_applied": False,
        "second_stage_discriminator_from": "",
        "second_stage_discriminator_to": "",
        "second_stage_discriminator_reason": "",
        "second_stage_margin_context": margin_context,
        "second_stage_residual_family": "",
    }


def _crypto_blind_trigger_predict(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
) -> Dict[str, Any]:
    working = list(train_rows[-260:]) if len(train_rows) > 260 else list(train_rows)
    sym = _s(candidate.get("symbol", "")).upper()
    symbol_rows = [r for r in working if _s(r.get("symbol", "")).upper() == sym]
    regime_rows = [r for r in working if _s(r.get("regime", "")) == regime]
    recent_rows = _recent_slice(working, 80)
    feature_scales = {key: _historical_replay_feature_scale(working, key) for key, _weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES}
    upside_diag = _historical_replay_upside_score(candidate, working, symbol=sym, regime=regime)
    upside_score = _f(upside_diag.get("upside_score", 0.0), 0.0)
    downside_score = _f(upside_diag.get("downside_score", 0.0), 0.0)
    sequence_snapshot = _crypto_blind_sequence_feature_snapshot(candidate)
    current_candle_move = _f(candidate.get("current_candle_pct_move", 0.0), 0.0)
    move_3 = _f(candidate.get("recent_return_3", 0.0), 0.0)
    move_6 = _f(candidate.get("recent_return_6", 0.0), 0.0)
    move_12 = _f(candidate.get("recent_return_12", 0.0), 0.0)
    move_24 = _f(candidate.get("recent_return_24", 0.0), 0.0)
    volatility = _f(candidate.get("recent_volatility", 0.0), 0.0)
    momentum = _f(candidate.get("trend_momentum_score", 0.0), 0.0)
    signal_margin = _f(candidate.get("signal_margin", 0.0), 0.0)
    active_tfs = _f(candidate.get("active_timeframe_count", 0.0), 0.0)
    bars_since_entry = int(_f(sequence_snapshot.get("bars_since_entry", 0), 0.0))
    current_unrealized = _f(sequence_snapshot.get("current_unrealized_pnl_pct", 0.0), 0.0)
    mfe_so_far = _f(sequence_snapshot.get("max_favorable_excursion_pct_so_far", 0.0), 0.0)
    mae_so_far_abs = abs(min(0.0, _f(sequence_snapshot.get("max_adverse_excursion_pct_so_far", 0.0), 0.0)))
    drawdown_so_far_abs = abs(min(0.0, _f(sequence_snapshot.get("drawdown_from_peak_pct_so_far", 0.0), 0.0)))
    trailing_armed = bool(sequence_snapshot.get("trailing_armed", False))
    bars_since_trailing_armed = int(_f(sequence_snapshot.get("bars_since_trailing_armed", -1), -1.0))
    risk_cut_distance_pct = _f(sequence_snapshot.get("risk_cut_distance_pct", 2.25), 2.25)
    take_profit_distance_pct = _f(sequence_snapshot.get("take_profit_distance_pct", 4.25), 4.25)
    risk_cut_touched_so_far = bool(sequence_snapshot.get("risk_cut_touched_so_far", False))
    take_profit_touched_so_far = bool(sequence_snapshot.get("take_profit_touched_so_far", False))
    peak_to_current_reversal_pct = _f(sequence_snapshot.get("peak_to_current_reversal_pct", 0.0), 0.0)
    favorable_then_softened = bool(sequence_snapshot.get("favorable_then_softened_flag_so_far", False))
    momentum_decay = _f(sequence_snapshot.get("momentum_decay", 0.0), 0.0)
    late_favorable_reversal_score = _f(sequence_snapshot.get("late_favorable_reversal_score", 0.0), 0.0)
    favorable_then_reversed = bool(sequence_snapshot.get("favorable_then_reversed", False))
    max_favorable_before_exit_pct = _f(sequence_snapshot.get("max_favorable_before_exit_pct", mfe_so_far), 0.0)
    reversal_from_peak_before_exit_pct = _f(sequence_snapshot.get("reversal_from_peak_before_exit_pct", peak_to_current_reversal_pct), 0.0)
    bars_from_peak_to_exit_preview = int(_f(sequence_snapshot.get("bars_from_peak_to_exit_preview", 0), 0.0))
    trailing_arm_to_exit_bars_preview = int(_f(sequence_snapshot.get("trailing_arm_to_exit_bars_preview", bars_since_trailing_armed), 0.0))
    risk_pressure_after_peak = _f(sequence_snapshot.get("risk_pressure_after_peak", 0.0), 0.0)
    take_profit_pressure_before_reversal = _f(sequence_snapshot.get("take_profit_pressure_before_reversal", 0.0), 0.0)
    target_near_before_reversal = bool(sequence_snapshot.get("target_near_before_reversal", False))
    reversal_velocity_pct_per_bar = _f(sequence_snapshot.get("reversal_velocity_pct_per_bar", 0.0), 0.0)
    post_peak_momentum_decay = _f(sequence_snapshot.get("post_peak_momentum_decay", momentum_decay), 0.0)
    drawdown_after_favorable_move_pct = _f(sequence_snapshot.get("drawdown_after_favorable_move_pct", peak_to_current_reversal_pct), 0.0)
    favorable_move_quality_score = _f(sequence_snapshot.get("favorable_move_quality_score", 0.0), 0.0)
    late_risk_after_favorable_move = bool(sequence_snapshot.get("late_risk_after_favorable_move", False))

    trigger_priors: Dict[str, float] = {cls: 0.0 for cls in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    class_similarity: Dict[str, float] = {cls: 0.0 for cls in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    support_counts: Dict[str, int] = {cls: 0 for cls in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES}
    for cls in _CRYPTO_HISTORICAL_REPLAY_TRIGGER_CLASSES:
        same_symbol = [r for r in symbol_rows if _s(r.get("actual_exit_trigger", "")) == cls]
        same_regime = [r for r in regime_rows if _s(r.get("actual_exit_trigger", "")) == cls]
        recent_global = [r for r in recent_rows if _s(r.get("actual_exit_trigger", "")) == cls]
        support_counts[cls] = int(len(same_symbol) + len(same_regime) + len(recent_global))
        trigger_priors[cls] = (0.70 * math.log1p(len(same_symbol))) + (0.30 * math.log1p(len(same_regime))) + (0.15 * math.log1p(len(recent_global)))
        cohort = same_symbol if len(same_symbol) >= 4 else same_regime if len(same_regime) >= 6 else recent_global if recent_global else [r for r in working if _s(r.get("actual_exit_trigger", "")) == cls]
        if cohort:
            dist = 0.0
            used = 0
            for key, weight in _CRYPTO_HISTORICAL_REPLAY_FEATURES:
                scale = max(1e-6, feature_scales.get(key, 1.0))
                median = _median([_f(r.get(key, 0.0), 0.0) for r in cohort], default=0.0)
                dist += weight * abs(_f(candidate.get(key, 0.0), 0.0) - median) / scale
                used += 1
            class_similarity[cls] = 1.0 / (1.0 + (dist / max(1, used)))

    overextension_score = (
        max(0.0, ((move_24 - 5.50) / 2.50))
        + max(0.0, ((move_12 - 4.50) / 2.00))
        + max(0.0, ((current_candle_move - 1.60) / 0.70))
        + max(0.0, ((volatility - 0.85) / 0.30))
    )
    risk_pressure_score = (
        (0.40 * _f(sequence_snapshot.get("risk_pressure_score", 0.0), 0.0))
        + (0.40 * downside_score)
        + (0.95 * max(0.0, ((1.10 - risk_cut_distance_pct) / 1.10)))
        + (1.20 if risk_cut_touched_so_far else 0.0)
        + (0.75 * max(0.0, (drawdown_so_far_abs / 0.80)))
        + (0.65 * max(0.0, (mae_so_far_abs / 1.10)))
        + (0.55 * max(0.0, (peak_to_current_reversal_pct / 0.70)))
        + (0.35 * overextension_score)
        + (0.30 * max(0.0, ((-current_unrealized) / 0.80)))
        + (0.25 if favorable_then_softened else 0.0)
    )
    take_profit_pressure_score = (
        (0.40 * _f(sequence_snapshot.get("take_profit_pressure_score", 0.0), 0.0))
        + (0.55 * max(0.0, (mfe_so_far / 1.40)))
        + (0.95 * max(0.0, ((1.10 - take_profit_distance_pct) / 1.10)))
        + (1.10 if take_profit_touched_so_far else 0.0)
        + (0.45 * max(0.0, ((move_6 - 2.20) / 1.40)))
        + (0.35 * max(0.0, ((move_12 - 3.20) / 1.70)))
        + (0.25 * max(0.0, upside_score - 0.25))
        - (0.20 * max(0.0, (drawdown_so_far_abs / 0.70)))
        - (0.20 if risk_cut_touched_so_far else 0.0)
    )
    trailing_quality_score = (
        (0.35 * _f(sequence_snapshot.get("trailing_quality_score", 0.0), 0.0))
        + (0.75 if trailing_armed else 0.0)
        + (0.45 * max(0.0, (mfe_so_far / 1.50)))
        + (0.50 * max(0.0, (peak_to_current_reversal_pct / 0.60)))
        + (0.30 * max(0.0, ((signal_margin - 0.28) / 0.22)))
        + (0.28 * max(0.0, ((momentum - 1.80) / 1.40)))
        + (0.22 * max(0.0, ((active_tfs - 4.0) / 4.0)))
        + (0.15 * max(0.0, upside_score - downside_score))
        - (0.55 * overextension_score)
        - (0.65 if risk_cut_touched_so_far else 0.0)
        - (0.25 * max(0.0, (mae_so_far_abs / 1.20)))
    )
    stale_alignment_score = (
        (0.55 * _f(sequence_snapshot.get("stale_alignment_score", 0.0), 0.0))
        + (0.65 * max(0.0, ((bars_since_entry - 4.0) / 3.0)))
        + (0.55 * max(0.0, ((1.60 - momentum) / 1.40)))
        + (0.45 * max(0.0, ((1.10 - mfe_so_far) / 1.10)))
        + (0.35 * max(0.0, (momentum_decay / 1.10)))
        + (0.25 * max(0.0, ((0.25 - abs(current_unrealized)) / 0.25)))
        - (0.25 if take_profit_touched_so_far or trailing_armed else 0.0)
        - (0.35 if risk_cut_touched_so_far else 0.0)
    )

    trigger_scores = {
        "Risk Cut": round((0.60 * class_similarity["Risk Cut"]) + (0.35 * trigger_priors["Risk Cut"]) + (1.35 * risk_pressure_score) + (0.25 * overextension_score) - (0.20 * take_profit_pressure_score) - (0.30 * trailing_quality_score), 6),
        "Take Profit": round((0.60 * class_similarity["Take Profit"]) + (0.35 * trigger_priors["Take Profit"]) + (1.20 * take_profit_pressure_score) + (0.15 * upside_score) - (0.18 * risk_pressure_score) - (0.12 * stale_alignment_score), 6),
        "Trailing": round((0.55 * class_similarity["Trailing"]) + (0.30 * trigger_priors["Trailing"]) + (1.00 * trailing_quality_score) + (0.12 * upside_score) - (0.40 * risk_pressure_score) - (0.22 * overextension_score) - (0.18 * take_profit_pressure_score), 6),
        "Stale Alignment": round((0.55 * class_similarity["Stale Alignment"]) + (0.30 * trigger_priors["Stale Alignment"]) + (1.20 * stale_alignment_score) + (0.08 * downside_score) - (0.12 * take_profit_pressure_score) - (0.10 * trailing_quality_score), 6),
    }
    trigger_scores["Unknown"] = 0.02
    top_triggers = sorted(trigger_scores.items(), key=lambda kv: kv[1], reverse=True)
    pred_trigger = top_triggers[0][0] if top_triggers else "Unknown"
    selected_margin = (top_triggers[0][1] - top_triggers[1][1]) if len(top_triggers) > 1 else top_triggers[0][1]
    second_score = top_triggers[1][1] if len(top_triggers) > 1 else top_triggers[0][1]
    margin_context = f"top={pred_trigger}:{round(top_triggers[0][1],6) if top_triggers else 0.0}|runner_up={top_triggers[1][0] if len(top_triggers)>1 else pred_trigger}:{round(second_score,6)}|margin={round(selected_margin,6)}"
    second_stage = _crypto_apply_second_stage_discriminator(
        {
            "pred_trigger": pred_trigger,
            "trigger_scores": trigger_scores,
            "selected_margin": selected_margin,
            "risk_pressure_score": risk_pressure_score,
            "take_profit_pressure_score": take_profit_pressure_score,
            "trailing_quality_score": trailing_quality_score,
            "stale_alignment_score": stale_alignment_score,
            "bars_since_entry": bars_since_entry,
            "max_favorable_excursion_pct_so_far": mfe_so_far,
            "max_adverse_excursion_pct_so_far": -mae_so_far_abs,
            "drawdown_from_peak_pct_so_far": -drawdown_so_far_abs,
            "trailing_armed": trailing_armed,
            "bars_since_trailing_armed": bars_since_trailing_armed,
            "risk_cut_distance_pct": risk_cut_distance_pct,
            "take_profit_distance_pct": take_profit_distance_pct,
            "risk_cut_touched_so_far": risk_cut_touched_so_far,
            "take_profit_touched_so_far": take_profit_touched_so_far,
            "peak_to_current_reversal_pct": peak_to_current_reversal_pct,
            "favorable_then_softened_flag_so_far": favorable_then_softened,
            "momentum_decay": momentum_decay,
            "late_favorable_reversal_score": late_favorable_reversal_score,
            "favorable_then_reversed": favorable_then_reversed,
            "max_favorable_before_exit_pct": max_favorable_before_exit_pct,
            "reversal_from_peak_before_exit_pct": reversal_from_peak_before_exit_pct,
            "bars_from_peak_to_exit_preview": bars_from_peak_to_exit_preview,
            "trailing_arm_to_exit_bars_preview": trailing_arm_to_exit_bars_preview,
            "risk_pressure_after_peak": risk_pressure_after_peak,
            "take_profit_pressure_before_reversal": take_profit_pressure_before_reversal,
            "target_near_before_reversal": target_near_before_reversal,
            "reversal_velocity_pct_per_bar": reversal_velocity_pct_per_bar,
            "post_peak_momentum_decay": post_peak_momentum_decay,
            "drawdown_after_favorable_move_pct": drawdown_after_favorable_move_pct,
            "favorable_move_quality_score": favorable_move_quality_score,
            "late_risk_after_favorable_move": late_risk_after_favorable_move,
            "margin_context": margin_context,
        }
    )
    pred_trigger = _s(second_stage.get("pred_trigger", pred_trigger)) or pred_trigger
    selected_margin = _f(second_stage.get("selected_margin", selected_margin), selected_margin)
    trigger_score_reason = (
        "risk_pressure_dominant" if pred_trigger == "Risk Cut"
        else "take_profit_pressure_dominant" if pred_trigger == "Take Profit"
        else "trailing_quality_dominant" if pred_trigger == "Trailing"
        else "stale_alignment_dominant" if pred_trigger == "Stale Alignment"
        else "insufficient_feature_context"
    )
    direction_scores = {
        "up": round(max(0.0, upside_score + max(0.0, move_6 / 2.2) + max(0.0, move_24 / 3.0) + max(0.0, signal_margin)), 6),
        "down": round(max(0.0, downside_score + max(0.0, (0.80 - move_6) / 1.50) + max(0.0, (0.55 - signal_margin) / 0.25) + max(0.0, (volatility - 0.75) / 0.45)), 6),
        "flat": round(max(0.0, stale_alignment_score * 0.25), 6),
    }
    if pred_trigger == "Risk Cut":
        direction_scores["down"] += 0.55
    elif pred_trigger in {"Trailing", "Take Profit"}:
        direction_scores["up"] += 0.35
    pred_dir = "up" if direction_scores["up"] >= direction_scores["down"] else "down"
    predicted_pnl_trend = "down" if pred_trigger == "Risk Cut" or (pred_trigger == "Stale Alignment" and direction_scores["down"] > direction_scores["up"]) else pred_dir
    pred_hold = max(1.0, _median([_f(r.get("hold_hours", 0.0), 0.0) for r in working if _s(r.get("actual_exit_trigger", "")) == pred_trigger], default=12.0))
    pred_return_pct = _median([_f(r.get("pnl_pct", 0.0), 0.0) for r in working if _s(r.get("actual_exit_trigger", "")) == pred_trigger], default=(1.5 if predicted_pnl_trend == "up" else -1.5 if predicted_pnl_trend == "down" else 0.0))
    entry = _f(candidate.get("entry_price", 0.0), 0.0)
    pred_exit_price = entry * (1.0 + (pred_return_pct / 100.0)) if entry > 0.0 else 0.0
    support_term = min(1.0, math.log1p(max(0, support_counts.get(pred_trigger, 0))) / math.log1p(24.0))
    margin_term = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, selected_margin))))
    conf = min(0.95, max(0.18, 0.20 + (0.28 * margin_term) + (0.22 * support_term) + (0.16 * max(class_similarity.get(pred_trigger, 0.0), 0.0)) + (0.14 * max(direction_scores.values()))))
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trigger,
        "predicted_hold_hours": round(float(pred_hold), 6),
        "predicted_exit_price": round(float(pred_exit_price), 10),
        "predicted_pnl_trend": predicted_pnl_trend,
        "predicted_confidence": round(float(conf), 6),
        "trigger_scores": {k: round(float(v), 6) for k, v in trigger_scores.items()},
        "direction_scores": direction_scores,
        "trigger_margin": round(float(selected_margin), 6),
        "predictor_source_mode": "historical_strategy_replay",
        "predictor_variant": "blind_sequence_trigger_scorer_v1",
        "trigger_score_risk_cut": round(float(trigger_scores.get("Risk Cut", 0.0)), 6),
        "trigger_score_take_profit": round(float(trigger_scores.get("Take Profit", 0.0)), 6),
        "trigger_score_trailing": round(float(trigger_scores.get("Trailing", 0.0)), 6),
        "trigger_score_stale_alignment": round(float(trigger_scores.get("Stale Alignment", 0.0)), 6),
        "trigger_score_selected_margin": round(float(selected_margin), 6),
        "trigger_score_reason": trigger_score_reason,
        "risk_pressure_score": round(float(risk_pressure_score), 6),
        "take_profit_pressure_score": round(float(take_profit_pressure_score), 6),
        "trailing_quality_score": round(float(trailing_quality_score), 6),
        "stale_alignment_score": round(float(stale_alignment_score), 6),
        "source_feature_availability": "entry_features_only",
        **second_stage,
        **sequence_snapshot,
    }


def _augment_historical_blind_rows(
    rows: List[Dict[str, Any]],
    *,
    market: str,
    symbol_sources: Dict[str, str],
    crypto_trigger_mode: str = "improved",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    for raw in sorted([dict(r) for r in list(rows or []) if isinstance(r, dict)], key=_stable_row_sort_key):
        symbol = _s(raw.get("symbol", "")).upper()
        candidate = dict(raw)
        candidate["source_type"] = HISTORICAL_BLIND_SIM_SOURCE_TYPE
        pred = _safe_predict_blind_row(market=market, history_rows=history, candidate_row=candidate, crypto_trigger_mode=crypto_trigger_mode)
        actual_direction = _s(raw.get("actual_direction", "")) or _trade_direction(raw)
        actual_exit_trigger = _s(raw.get("actual_exit_trigger", "")) or "Unknown"
        actual_pnl_trend = _actual_pnl_trend(raw)
        predictor_variant = _s(pred.get("predictor_variant", "")) or (
            "stock_pnl_quality_v2" if _s(market).lower() == "stocks" else "historical_blind_strategy_baseline"
        )
        enriched = dict(raw)
        enriched["market"] = _s(market).lower()
        enriched["source_type"] = HISTORICAL_BLIND_SIM_SOURCE_TYPE
        enriched["symbol_source"] = _s(symbol_sources.get(symbol, "configured_universe")) or "configured_universe"
        enriched["simulation_id"] = hashlib.sha256(
            f"{_s(market).lower()}|{symbol}|{int(_f(raw.get('entry_ts', 0.0), 0.0))}|{int(_f(raw.get('exit_ts', 0.0), 0.0))}".encode("utf-8")
        ).hexdigest()[:16]
        enriched["decision_snapshot_id"] = f"blind-sim:{_s(market).lower()}:{symbol}:{int(_f(raw.get('entry_ts', 0.0), 0.0))}"
        enriched["symbol"] = symbol
        enriched["side"] = _s(raw.get("side", "long")) or "long"
        enriched["qty"] = _f(raw.get("qty", 1.0), 1.0)
        enriched["predictor_variant"] = predictor_variant
        enriched["predicted_direction"] = _s(pred.get("predicted_direction", "")) or "up"
        enriched["predicted_exit_trigger"] = _s(pred.get("predicted_exit_trigger", "")) or "Unknown"
        normalized_predicted_pnl = _normalize_pnl_trend_label(pred.get("predicted_pnl_trend", ""))
        enriched["predicted_pnl_trend"] = normalized_predicted_pnl or _normalize_pnl_trend_label(enriched["predicted_direction"]) or "flat"
        enriched["confidence"] = round(
            _f(pred.get("predicted_confidence", pred.get("confidence", 0.0)), 0.0),
            6,
        )
        enriched["direction_scores"] = dict(pred.get("direction_scores", {}) if isinstance(pred.get("direction_scores", {}), dict) else {})
        enriched["trigger_scores"] = dict(pred.get("trigger_scores", {}) if isinstance(pred.get("trigger_scores", {}), dict) else {})
        enriched["trigger_margin"] = round(_f(pred.get("trigger_margin", 0.0), 0.0), 6)
        enriched["trigger_score_risk_cut"] = round(_f(pred.get("trigger_score_risk_cut", 0.0), 0.0), 6)
        enriched["trigger_score_take_profit"] = round(_f(pred.get("trigger_score_take_profit", 0.0), 0.0), 6)
        enriched["trigger_score_trailing"] = round(_f(pred.get("trigger_score_trailing", 0.0), 0.0), 6)
        enriched["trigger_score_stale_alignment"] = round(_f(pred.get("trigger_score_stale_alignment", 0.0), 0.0), 6)
        enriched["trigger_score_selected_margin"] = round(_f(pred.get("trigger_score_selected_margin", pred.get("trigger_margin", 0.0)), 0.0), 6)
        enriched["trigger_score_reason"] = _s(pred.get("trigger_score_reason", ""))
        enriched["risk_pressure_score"] = round(_f(pred.get("risk_pressure_score", 0.0), 0.0), 6)
        enriched["take_profit_pressure_score"] = round(_f(pred.get("take_profit_pressure_score", 0.0), 0.0), 6)
        enriched["trailing_quality_score"] = round(_f(pred.get("trailing_quality_score", 0.0), 0.0), 6)
        enriched["stale_alignment_score"] = round(_f(pred.get("stale_alignment_score", 0.0), 0.0), 6)
        enriched["source_feature_availability"] = _s(pred.get("source_feature_availability", ""))
        enriched["second_stage_discriminator_applied"] = bool(pred.get("second_stage_discriminator_applied", False))
        enriched["second_stage_discriminator_from"] = _s(pred.get("second_stage_discriminator_from", ""))
        enriched["second_stage_discriminator_to"] = _s(pred.get("second_stage_discriminator_to", ""))
        enriched["second_stage_discriminator_reason"] = _s(pred.get("second_stage_discriminator_reason", ""))
        enriched["second_stage_margin_context"] = _s(pred.get("second_stage_margin_context", ""))
        enriched["second_stage_residual_family"] = _s(pred.get("second_stage_residual_family", ""))
        for key in (
            "bars_since_entry",
            "current_unrealized_pnl_pct",
            "max_favorable_excursion_pct_so_far",
            "max_adverse_excursion_pct_so_far",
            "drawdown_from_peak_pct_so_far",
            "trailing_armed",
            "bars_since_trailing_armed",
            "risk_cut_distance_pct",
            "take_profit_distance_pct",
            "risk_cut_touched_so_far",
            "take_profit_touched_so_far",
            "peak_to_current_reversal_pct",
            "momentum_decay",
            "favorable_then_softened_flag_so_far",
            "late_favorable_reversal_score",
            "favorable_then_reversed",
            "max_favorable_before_exit_pct",
            "reversal_from_peak_before_exit_pct",
            "bars_from_peak_to_exit_preview",
            "trailing_arm_to_exit_bars_preview",
            "risk_pressure_after_peak",
            "take_profit_pressure_before_reversal",
            "target_near_before_reversal",
            "reversal_velocity_pct_per_bar",
            "post_peak_momentum_decay",
            "drawdown_after_favorable_move_pct",
            "favorable_move_quality_score",
            "late_risk_after_favorable_move",
        ):
            if key in pred:
                enriched[key] = pred.get(key)
        enriched["trade_quality_score"] = round(
            _f(pred.get("stock_trade_quality_score", pred.get("trade_quality_score", 0.0)), 0.0),
            6,
        )
        enriched["pnl_quality_score"] = round(
            _f(pred.get("stock_pnl_quality_score", pred.get("pnl_quality_score", 0.0)), 0.0),
            6,
        )
        enriched["actual_direction"] = actual_direction
        enriched["actual_exit_trigger"] = actual_exit_trigger
        enriched["actual_pnl_trend"] = actual_pnl_trend
        enriched["prediction_generated_before_outcome"] = True
        enriched["hold_time"] = round(
            _f(raw.get("hold_hours", max(0.0, (_f(raw.get("exit_ts", 0.0), 0.0) - _f(raw.get("entry_ts", 0.0), 0.0)) / 3600.0)), 0.0),
            6,
        )
        enriched["bars_in_trade"] = int(_f(raw.get("bars_in_trade", 0), 0.0))
        enriched["direction_correct"] = bool(_s(enriched.get("predicted_direction", "")).lower() == _s(actual_direction).lower())
        enriched["trigger_correct"] = bool(_s(enriched.get("predicted_exit_trigger", "")) == actual_exit_trigger)
        enriched["pnl_trend_correct"] = bool(_normalize_pnl_trend_label(enriched.get("predicted_pnl_trend", "")) == _s(actual_pnl_trend).lower())
        semantics = _historical_blind_prediction_semantics(enriched, market)
        enriched["prediction_semantics"] = _s(semantics.get("prediction_semantics", "heuristic")) or "heuristic"
        enriched["prediction_semantics_warning"] = _s(semantics.get("prediction_semantics_warning", ""))
        enriched["label_alignment_status"] = _s(semantics.get("label_alignment_status", "blocked")) or "blocked"
        enriched["label_alignment_blockers"] = list(semantics.get("label_alignment_blockers", []) or [])
        enriched["future_leakage_detected"] = bool(semantics.get("future_leakage_detected", False))
        enriched["prediction_semantics_placeholder_only"] = bool(semantics.get("placeholder_only", False))
        enriched["eligible_for_training"] = bool(semantics.get("eligible_for_training", False))
        enriched["training_eligibility_reason"] = _s(semantics.get("training_eligibility_reason", ""))
        enriched["diagnostic_only"] = bool(semantics.get("diagnostic_only", not enriched["eligible_for_training"]))
        enriched["replay_row_id"] = _replay_row_id(enriched)
        out.append(enriched)
        history.append(dict(enriched))
    return out


def _historical_blind_decision_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "simulation_id": _s(row.get("simulation_id", "")),
                "decision_snapshot_id": _s(row.get("decision_snapshot_id", "")),
                "market": _s(row.get("market", "")),
                "symbol": _s(row.get("symbol", "")),
                "symbol_source": _s(row.get("symbol_source", "")),
                "timestamp": int(_f(row.get("entry_ts", 0.0), 0.0)),
                "decision_type": "entry",
                "selected_action": "enter",
                "predicted_direction": _s(row.get("predicted_direction", "")),
                "predicted_exit_trigger": _s(row.get("predicted_exit_trigger", "")),
                "predicted_pnl_trend": _s(row.get("predicted_pnl_trend", "")),
                "confidence": round(_f(row.get("confidence", 0.0), 0.0), 6),
                "direction_scores": dict(row.get("direction_scores", {}) if isinstance(row.get("direction_scores", {}), dict) else {}),
                "trigger_scores": dict(row.get("trigger_scores", {}) if isinstance(row.get("trigger_scores", {}), dict) else {}),
                "trade_quality_score": round(_f(row.get("trade_quality_score", 0.0), 0.0), 6),
                "pnl_quality_score": round(_f(row.get("pnl_quality_score", 0.0), 0.0), 6),
                "entry_reason": _s(row.get("raw_rule_reason", "")) or "historical_blind_strategy_simulation_entry",
                "skip_reason": "",
                "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE,
                "replay_row_id": _s(row.get("replay_row_id", "")),
            }
        )
    return out


def _historical_blind_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = [dict(r) for r in list(rows or []) if isinstance(r, dict)]
    n = len(ordered)
    if n <= 0:
        return {
            "rows": 0,
            "directional_accuracy_pct": 0.0,
            "trigger_match_pct": 0.0,
            "pnl_trend_match_pct": 0.0,
            "win_rate_pct": 0.0,
            "average_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    pnl_vals = [_f(r.get("pnl_pct", _trade_return_pct(r)), 0.0) for r in ordered]
    return {
        "rows": int(n),
        "directional_accuracy_pct": round(100.0 * sum(1 for r in ordered if bool(r.get("direction_correct", False))) / max(1, n), 4),
        "trigger_match_pct": round(100.0 * sum(1 for r in ordered if bool(r.get("trigger_correct", False))) / max(1, n), 4),
        "pnl_trend_match_pct": round(100.0 * sum(1 for r in ordered if bool(r.get("pnl_trend_correct", False))) / max(1, n), 4),
        "win_rate_pct": round(100.0 * sum(1 for v in pnl_vals if v > 0.0) / max(1, n), 4),
        "average_pnl_pct": round(sum(pnl_vals) / max(1, n), 6),
        "max_drawdown_pct": round(min(pnl_vals), 6),
    }


def _collect_stock_blind_sim_candidates(hub_dir: str, settings: Dict[str, Any], limit: int) -> Tuple[List[str], Dict[str, str]]:
    sources: Dict[str, str] = {}
    ordered: List[str] = []

    def _add(sym: Any, source: str) -> None:
        normalized = _normalize_stock_ticker(sym)
        if not normalized:
            return
        if normalized not in ordered:
            ordered.append(normalized)
        current = _s(sources.get(normalized, ""))
        if current == "manual_watchlist":
            return
        if source == "manual_watchlist" or not current:
            sources[normalized] = source

    manual_payload = _read_stock_manual_watchlist(hub_dir)
    for sym in list((manual_payload.get("symbols", {}) if isinstance(manual_payload.get("symbols", {}), dict) else {}).keys()):
        _add(sym, "manual_watchlist")
    for sym in _stock_symbol_candidates(hub_dir, [], limit=max(1, limit * 3)):
        _add(sym, "scanner_discovered")
    for tok in _s(settings.get("stock_universe_symbols", "")).replace("\n", ",").split(","):
        _add(tok, "configured_universe")
    return ordered[:limit], sources


def _build_stock_symbol_onboarding(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
    symbol: str,
    symbol_source: str,
    lookback_days: int,
    timeframe: str,
    force_refresh: bool,
) -> Dict[str, Any]:
    warm = warm_stock_historical_cache(
        hub_dir=hub_dir,
        base_dir=base_dir,
        settings=settings,
        symbol=symbol,
        lookback_days=lookback_days,
        timeframe="1Hour" if _s(timeframe).lower() == "1hour" else timeframe,
        force_refresh=force_refresh,
    )
    bars = _load_stock_cached_bars(hub_dir, symbol, "1Hour" if _s(timeframe).lower() == "1hour" else timeframe)
    raw_rows = _simulate_stock_trades_from_bars(symbol, bars)
    rows = _augment_historical_blind_rows(raw_rows, market="stocks", symbol_sources={_normalize_stock_ticker(symbol): symbol_source})
    decision_rows = _historical_blind_decision_rows(rows)
    skips: List[Dict[str, Any]] = []
    preview = build_stock_watchlist_prediction_preview(
        hub_dir=hub_dir,
        base_dir=base_dir,
        settings=settings,
        symbol=symbol,
        timeframe="1Hour",
        lookback_days=lookback_days,
    )
    blockers = list(preview.get("manual_watchlist_trade_blockers", []) or [])
    status = "trade_ready" if bool(preview.get("manual_watchlist_trade_eligible", False)) else "watch_only"
    if warm.get("warmup_status") == "insufficient_history":
        status = "blocked_insufficient_history"
        blockers = sorted(set(blockers + ["insufficient_history"]))
        skips.append({"market": "stocks", "symbol": _normalize_stock_ticker(symbol), "symbol_source": symbol_source, "decision": "skip", "skip_reason": "insufficient_history", "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE})
    elif not rows:
        status = "blocked_strategy_alignment"
        blockers = sorted(set(blockers + ["no_simulated_completed_trades"]))
        skips.append({"market": "stocks", "symbol": _normalize_stock_ticker(symbol), "symbol_source": symbol_source, "decision": "skip", "skip_reason": "no_simulated_completed_trades", "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE})
    elif blockers:
        status = "learning_ready_not_trade_ready"
    onboarding_paths = _symbol_onboarding_paths(hub_dir, "stocks", symbol)
    summary = {
        "market": "stocks",
        "symbol": _normalize_stock_ticker(symbol),
        "symbol_source": symbol_source,
        "metrics": _historical_blind_metrics(rows),
        "rows_generated": int(len(rows)),
        "decision_rows_generated": int(len(decision_rows)),
        "skip_rows_generated": int(len(skips)),
        "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE,
    }
    status_payload = {
        "schema_version": HISTORICAL_BLIND_SIM_SCHEMA_VERSION,
        "market": "stocks",
        "symbol": _normalize_stock_ticker(symbol),
        "symbol_source": symbol_source,
        "requested_at": int(time.time()),
        "completed_at": int(time.time()),
        "provider": _s(preview.get("provider", warm.get("provider", ""))) or "alpaca",
        "historical_data_rows": int(len(bars)),
        "historical_lookback_days": int(lookback_days),
        "historical_timeframe": "1Hour",
        "simulation_completed": bool(rows),
        "simulation_trade_rows": int(len(rows)),
        "simulation_decision_rows": int(len(decision_rows)),
        "simulation_skip_rows": int(len(skips)),
        "eligible_for_scan": bool(rows),
        "eligible_for_trade_consideration": bool(preview.get("manual_watchlist_trade_eligible", False)),
        "eligible_for_training": bool(any(bool(r.get("eligible_for_training", False)) for r in rows)),
        "status": status,
        "blockers": blockers,
        "warnings": sorted({_s(r.get("prediction_semantics_warning", "")) for r in rows if _s(r.get("prediction_semantics_warning", ""))}),
        "cooldown_until": 0,
        "alignment_score": round(_f(summary["metrics"].get("win_rate_pct", 0.0), 0.0) / 100.0, 6),
        "alignment_reason": "shared_stock_blind_simulation_path",
        "simulation_directional_accuracy": summary["metrics"].get("directional_accuracy_pct", 0.0),
        "simulation_trigger_accuracy": summary["metrics"].get("trigger_match_pct", 0.0),
        "simulation_pnl_trend_accuracy": summary["metrics"].get("pnl_trend_match_pct", 0.0),
        "prediction_semantics": "heuristic",
        "diagnostic_only_rows": int(sum(1 for r in rows if bool(r.get("diagnostic_only", False)))),
        "simulation_trade_count": int(len(rows)),
        "simulation_skip_count": int(len(skips)),
        "simulation_avg_pnl_pct": summary["metrics"].get("average_pnl_pct", 0.0),
        "simulation_max_drawdown_pct": summary["metrics"].get("max_drawdown_pct", 0.0),
        "simulation_win_rate_pct": summary["metrics"].get("win_rate_pct", 0.0),
        "artifact_paths": onboarding_paths,
    }
    _write_jsonl(onboarding_paths["trades"], rows)
    _write_jsonl(onboarding_paths["decisions"], decision_rows)
    _write_jsonl(onboarding_paths["skips"], skips)
    _write_json_atomic(onboarding_paths["summary"], summary)
    _write_json_atomic(onboarding_paths["status"], status_payload)
    return {
        "symbol": _normalize_stock_ticker(symbol),
        "status": status_payload,
        "summary": summary,
        "trades": rows,
        "decisions": decision_rows,
        "skips": skips,
    }


def _build_crypto_blind_simulation(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    lookback_days: int,
    timeframe: str,
    max_symbols: int,
    force_refresh: bool,
    deterministic: bool,
) -> Dict[str, Any]:
    replay = build_crypto_historical_strategy_replay(
        hub_dir=hub_dir,
        settings=settings,
        timeframe=timeframe,
        lookback_days=lookback_days,
        max_symbols=max_symbols,
        force_refresh=force_refresh,
        freeze_cache=deterministic,
        deterministic=deterministic,
    )
    raw_rows = list(replay.get("rows", []) if isinstance(replay.get("rows", []), list) else [])
    diagnostics = replay.get("diagnostics", {}) if isinstance(replay.get("diagnostics", {}), dict) else {}
    symbols = []
    sources: Dict[str, str] = {}
    for row in raw_rows:
        sym = _s((row or {}).get("symbol", "")).upper()
        if sym and sym not in symbols:
            symbols.append(sym)
            sources[sym] = "configured_universe"
    for skipped in list(diagnostics.get("historical_strategy_replay_skipped_reasons", []) or []):
        sym = _s((skipped or {}).get("symbol", "")).upper()
        if sym and sym not in symbols:
            symbols.append(sym)
            sources[sym] = "configured_universe"
    baseline_rows = _augment_historical_blind_rows(raw_rows, market="crypto", symbol_sources=sources, crypto_trigger_mode="baseline")
    rows = _augment_historical_blind_rows(raw_rows, market="crypto", symbol_sources=sources, crypto_trigger_mode="improved")
    decisions = _historical_blind_decision_rows(rows)
    skips: List[Dict[str, Any]] = []
    for skipped in list(diagnostics.get("historical_strategy_replay_skipped_reasons", []) or []):
        if not isinstance(skipped, dict):
            continue
        sym = _s(skipped.get("symbol", "")).upper()
        skips.append(
            {
                "market": "crypto",
                "symbol": sym,
                "symbol_source": _s(sources.get(sym, "configured_universe")) or "configured_universe",
                "timestamp": int(time.time()),
                "decision": "skip",
                "skip_reason": _s(skipped.get("reason", "")) or "simulation_skipped",
                "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE,
            }
        )
    return {
        "rows": rows,
        "baseline_rows": baseline_rows,
        "decisions": decisions,
        "skips": skips,
        "symbols": symbols,
        "diagnostics": diagnostics,
    }


def _build_historical_blind_simulation(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    enabled = _env_flag("HISTORICAL_BLIND_SIM_ENABLED") or _env_flag("SYMBOL_ONBOARDING_BLIND_SIM_ENABLED")
    configured_markets = _historical_blind_sim_enabled_markets()
    lookback_days = _env_int("HISTORICAL_BLIND_SIM_LOOKBACK_DAYS", _env_int("SYMBOL_ONBOARDING_LOOKBACK_DAYS", 365))
    timeframe = _env_str("HISTORICAL_BLIND_SIM_TIMEFRAME", _env_str("SYMBOL_ONBOARDING_TIMEFRAME", "1hour")) or "1hour"
    max_symbols = _env_int("HISTORICAL_BLIND_SIM_MAX_SYMBOLS", max(1, _env_int("SYMBOL_ONBOARDING_MAX_SYMBOLS_PER_RUN", 20)))
    deterministic = _env_flag("HISTORICAL_BLIND_SIM_DETERMINISTIC")
    force_refresh = False
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "markets_requested": configured_markets,
        "historical_blind_simulation_used_as_primary": False,
        "historical_blind_simulation_used_as_supplemental": False,
        "markets": {},
        "artifact_paths": {},
        "symbol_bucket_artifacts": {},
        "unified_artifacts": {},
    }
    if not enabled:
        return report

    unified_trades: List[Dict[str, Any]] = []
    unified_decisions: List[Dict[str, Any]] = []
    unified_skips: List[Dict[str, Any]] = []
    ts_now = int(time.time())
    for market in configured_markets:
        paths = _historical_blind_sim_paths(hub_dir, market)
        bucket_paths = _symbol_bucket_paths(hub_dir, market)
        market_report: Dict[str, Any] = {
            "market": market,
            "supported": market in {"crypto", "stocks"},
            "implemented": False,
            "blocker": "",
            "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE,
        }
        trades: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        skips: List[Dict[str, Any]] = []
        candidate_symbols: List[str] = []
        onboarding_rows: List[Dict[str, Any]] = []
        active_rows: List[Dict[str, Any]] = []
        eligible_rows: List[Dict[str, Any]] = []
        rejected_rows: List[Dict[str, Any]] = []
        status_by_symbol: Dict[str, Dict[str, Any]] = {}
        crypto_baseline_rows: List[Dict[str, Any]] = []
        if market == "crypto":
            built = _build_crypto_blind_simulation(
                hub_dir=hub_dir,
                settings=settings,
                lookback_days=lookback_days,
                timeframe=timeframe,
                max_symbols=max_symbols,
                force_refresh=force_refresh,
                deterministic=deterministic,
            )
            crypto_baseline_rows = list(built.get("baseline_rows", []) or [])
            trades = list(built.get("rows", []) or [])
            decisions = list(built.get("decisions", []) or [])
            skips = list(built.get("skips", []) or [])
            candidate_symbols = list(built.get("symbols", []) or [])
            for sym in candidate_symbols:
                sym_rows = [dict(r) for r in trades if _s(r.get("symbol", "")).upper() == sym]
                sym_decisions = [dict(r) for r in decisions if _s(r.get("symbol", "")).upper() == sym]
                sym_skips = [dict(r) for r in skips if _s(r.get("symbol", "")).upper() == sym]
                metrics = _historical_blind_metrics(sym_rows)
                status = {
                    "schema_version": HISTORICAL_BLIND_SIM_SCHEMA_VERSION,
                    "market": "crypto",
                    "symbol": sym,
                    "symbol_source": "configured_universe",
                    "requested_at": ts_now,
                    "completed_at": ts_now,
                    "provider": _s(built.get("diagnostics", {}).get("historical_strategy_replay_provider", "")) or "kucoin",
                    "historical_data_rows": int(metrics.get("rows", 0)),
                    "historical_lookback_days": int(lookback_days),
                    "historical_timeframe": timeframe,
                    "simulation_completed": bool(sym_rows),
                    "simulation_trade_rows": int(len(sym_rows)),
                    "simulation_decision_rows": int(len(sym_decisions)),
                    "simulation_skip_rows": int(len(sym_skips)),
                    "eligible_for_scan": bool(sym_rows),
                    "eligible_for_trade_consideration": False,
                    "eligible_for_training": bool(any(bool(r.get("eligible_for_training", False)) for r in sym_rows)),
                    "status": "learning_ready_not_trade_ready" if sym_rows else "blocked_strategy_alignment",
                    "blockers": (
                        ["historical_blind_simulation_supplemental_only", "existing_readiness_gates_not_promoting_blind_sim_rows"]
                        if sym_rows
                        else ["no_simulated_completed_trades"]
                    ),
                    "warnings": sorted({_s(r.get("prediction_semantics_warning", "")) for r in sym_rows if _s(r.get("prediction_semantics_warning", ""))}),
                    "cooldown_until": 0,
                    "alignment_score": round(_f(metrics.get("win_rate_pct", 0.0), 0.0) / 100.0, 6),
                    "alignment_reason": "historical_crypto_blind_simulation",
                    "simulation_directional_accuracy": metrics.get("directional_accuracy_pct", 0.0),
                    "simulation_trigger_accuracy": metrics.get("trigger_match_pct", 0.0),
                    "simulation_pnl_trend_accuracy": metrics.get("pnl_trend_match_pct", 0.0),
                    "prediction_semantics": "heuristic",
                    "diagnostic_only_rows": int(sum(1 for r in sym_rows if bool(r.get("diagnostic_only", False)))),
                    "simulation_trade_count": int(len(sym_rows)),
                    "simulation_skip_count": int(len(sym_skips)),
                    "simulation_avg_pnl_pct": metrics.get("average_pnl_pct", 0.0),
                    "simulation_max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                    "simulation_win_rate_pct": metrics.get("win_rate_pct", 0.0),
                    "artifact_paths": _symbol_onboarding_paths(hub_dir, "crypto", sym),
                }
                status_by_symbol[sym] = status
                onboarding_rows.append({"symbol": sym, "symbol_source": "configured_universe", "status": status["status"]})
                if sym_rows:
                    active_rows.append({"symbol": sym, "status": "scan_ready"})
                else:
                    rejected_rows.append({"symbol": sym, "reason": "no_simulated_completed_trades", "cooldown_until": 0})
                sym_paths = _symbol_onboarding_paths(hub_dir, "crypto", sym)
                _write_jsonl(sym_paths["trades"], sym_rows)
                _write_jsonl(sym_paths["decisions"], sym_decisions)
                _write_jsonl(sym_paths["skips"], sym_skips)
                _write_json_atomic(sym_paths["summary"], {"market": "crypto", "symbol": sym, "metrics": metrics, "rows_generated": int(len(sym_rows)), "decision_rows_generated": int(len(sym_decisions)), "skip_rows_generated": int(len(sym_skips)), "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE})
                _write_json_atomic(sym_paths["status"], status)
            market_report["implemented"] = True
        elif market == "stocks":
            candidate_symbols, symbol_sources = _collect_stock_blind_sim_candidates(hub_dir, settings, max_symbols)
            for sym in candidate_symbols:
                onboarding = _build_stock_symbol_onboarding(
                    hub_dir=hub_dir,
                    base_dir=base_dir,
                    settings=settings,
                    symbol=sym,
                    symbol_source=_s(symbol_sources.get(sym, "configured_universe")) or "configured_universe",
                    lookback_days=lookback_days,
                    timeframe=timeframe,
                    force_refresh=force_refresh,
                )
                status = onboarding.get("status", {}) if isinstance(onboarding.get("status", {}), dict) else {}
                status_by_symbol[sym] = status
                trades.extend(list(onboarding.get("trades", []) or []))
                decisions.extend(list(onboarding.get("decisions", []) or []))
                skips.extend(list(onboarding.get("skips", []) or []))
                onboarding_rows.append({"symbol": sym, "symbol_source": _s(status.get("symbol_source", "")), "status": _s(status.get("status", ""))})
                if bool(status.get("eligible_for_scan", False)):
                    active_rows.append({"symbol": sym, "status": _s(status.get("status", ""))})
                if bool(status.get("eligible_for_trade_consideration", False)):
                    eligible_rows.append({"symbol": sym, "status": _s(status.get("status", ""))})
                elif _s(status.get("status", "")).startswith("blocked") or _s(status.get("status", "")) == "cooled_down":
                    rejected_rows.append({"symbol": sym, "reason": ",".join(list(status.get("blockers", []) or [])) or _s(status.get("status", "")), "cooldown_until": int(_f(status.get("cooldown_until", 0), 0.0))})
            market_report["implemented"] = True
        else:
            market_report["supported"] = False
            market_report["blocker"] = "forex_blind_simulation_adapter_not_implemented_replay_safe_candle_path_missing"

        _write_jsonl(paths["trades"], trades)
        _write_jsonl(paths["decisions"], decisions)
        _write_jsonl(paths["skips"], skips)
        blind_diag = _historical_blind_diagnostics(
            market=market,
            rows=trades,
            status_by_symbol=status_by_symbol,
            active_rows=active_rows,
            eligible_rows=eligible_rows,
            rejected_rows=rejected_rows,
        )
        crypto_before_after = _crypto_blind_before_after_summary(crypto_baseline_rows, trades) if market == "crypto" else {}
        crypto_residual_artifacts = _export_crypto_blind_residual_mismatches(hub_dir, trades) if market == "crypto" else {}
        split_rows = _split_walkforward_rows(trades)
        crypto_training_readiness = {}
        if market == "crypto":
            eligible_after = int(blind_diag.get("eligible_for_training_rows", 0) or 0)
            diag_after = int(blind_diag.get("diagnostic_only_rows", 0) or 0)
            readiness_state = "diagnostic_only"
            readiness_blockers: List[str] = []
            if eligible_after >= 150 and int(blind_diag.get("risk_cut_to_trailing_count", 0) or 0) <= 40 and int(blind_diag.get("take_profit_to_trailing_count", 0) or 0) <= 12:
                readiness_state = "partially_ready_for_training"
            if _s(blind_diag.get("prediction_semantics", "")) == "heuristic":
                readiness_blockers.append("prediction_semantics_still_heuristic")
            if int(blind_diag.get("risk_cut_to_trailing_count", 0) or 0) > 40:
                readiness_blockers.append("residual_risk_cut_to_trailing_confusion_high")
            if int(blind_diag.get("take_profit_to_trailing_count", 0) or 0) > 12:
                readiness_blockers.append("residual_take_profit_to_trailing_confusion_high")
            if readiness_state == "diagnostic_only" and eligible_after > 0:
                readiness_state = "partially_ready_for_training"
            crypto_training_readiness = {
                "decision": readiness_state,
                "reason": "heuristic_trigger_scorer_improved_but_residual_confusion_remains" if readiness_state != "ready_for_training" else "trigger_semantics_aligned",
                "eligible_row_count": eligible_after,
                "diagnostic_only_row_count": diag_after,
                "remaining_blockers": readiness_blockers,
                "prediction_semantics": _s(blind_diag.get("prediction_semantics", "")),
                "supplemental_only": True,
            }
        summary = {
            "schema_version": HISTORICAL_BLIND_SIM_SCHEMA_VERSION,
            "market": market,
            "enabled": True,
            "implemented": bool(market_report.get("implemented", False)),
            "supported": bool(market_report.get("supported", False)),
            "source_type": HISTORICAL_BLIND_SIM_SOURCE_TYPE,
            "lookback_days": int(lookback_days),
            "timeframe": timeframe,
            "simulation_rows_generated": int(len(trades)),
            "completed_mock_trades": int(len(trades)),
            "skipped_decisions": int(len(skips)),
            "training_eligible_rows": int(sum(1 for r in trades if bool(r.get("eligible_for_training", False)))),
            "diagnostic_only_rows": int(sum(1 for r in trades if bool(r.get("diagnostic_only", False)))),
            "historical_blind_simulation_used_as_primary": False,
            "historical_blind_simulation_used_as_supplemental": bool(trades),
            "walkforward_split_counts": {k: int(len(v)) for k, v in split_rows.items()},
            "validation_metrics": _historical_blind_metrics(split_rows.get("validation", [])),
            "test_metrics": _historical_blind_metrics(split_rows.get("test", [])),
            "metrics_by_symbol": {sym: _historical_blind_metrics([r for r in trades if _s(r.get("symbol", "")).upper() == sym]) for sym in candidate_symbols},
            "metrics_by_trigger": _count_by(trades, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
            "train_validation_test_rows": {k: int(len(v)) for k, v in split_rows.items()},
            "leakage_check_status": {
                "status": "pass",
                "details": "blind_sim_predictions_use_only_prior_completed_mock_trades",
            },
            "symbol_onboarding_counts": {
                "candidate_universe": int(len(candidate_symbols)),
                "onboarding_queue": int(len([r for r in onboarding_rows if _s(r.get('status', ''))])),
                "active_scan_set": int(len(active_rows)),
                "trade_eligible_set": int(len(eligible_rows)),
                "rejected_or_cooled_down": int(len(rejected_rows)),
            },
            "manual_watchlist_statuses": {
                sym: status_by_symbol.get(sym, {}).get("status", "")
                for sym, source in (
                    [(s, _s(status_by_symbol.get(s, {}).get("symbol_source", ""))) for s in candidate_symbols]
                )
                if source == "manual_watchlist"
            },
            "prediction_semantics": _s(blind_diag.get("prediction_semantics", "")),
            "prediction_semantics_warning": _s(blind_diag.get("prediction_semantics_warning", "")),
            "predicted_direction_counts": blind_diag.get("predicted_direction_counts", {}),
            "actual_direction_counts": blind_diag.get("actual_direction_counts", {}),
            "direction_confusion_matrix": blind_diag.get("direction_confusion_matrix", {}),
            "predicted_trigger_counts": blind_diag.get("predicted_trigger_counts", {}),
            "actual_trigger_counts": blind_diag.get("actual_trigger_counts", {}),
            "trigger_confusion_matrix": blind_diag.get("trigger_confusion_matrix", {}),
            "predicted_pnl_trend_counts": blind_diag.get("predicted_pnl_trend_counts", {}),
            "actual_pnl_trend_counts": blind_diag.get("actual_pnl_trend_counts", {}),
            "pnl_trend_confusion_matrix": blind_diag.get("pnl_trend_confusion_matrix", {}),
            "top_mismatch_buckets": blind_diag.get("top_mismatch_buckets", {}),
            "sample_mismatch_rows": blind_diag.get("sample_mismatch_rows", []),
            "trigger_score_distribution": blind_diag.get("trigger_score_distribution", {}),
            "selected_trigger_margin_buckets": blind_diag.get("selected_trigger_margin_buckets", {}),
            "training_eligible_counts_by_trigger": blind_diag.get("training_eligible_counts_by_trigger", {}),
            "diagnostic_only_reason_counts": blind_diag.get("diagnostic_only_reason_counts", {}),
            "second_stage_discriminator_applied_counts": blind_diag.get("second_stage_discriminator_applied_counts", {}),
            "second_stage_discriminator_reason_counts": blind_diag.get("second_stage_discriminator_reason_counts", {}),
            "residual_pair_counts": blind_diag.get("residual_pair_counts", {}),
            "top_risk_cut_to_trailing_residual_mismatches": blind_diag.get("top_risk_cut_to_trailing_residual_mismatches", []),
            "top_take_profit_to_trailing_residual_mismatches": blind_diag.get("top_take_profit_to_trailing_residual_mismatches", []),
            "top_stale_alignment_to_trailing_residual_mismatches": blind_diag.get("top_stale_alignment_to_trailing_residual_mismatches", []),
            "label_alignment_status": _s(blind_diag.get("label_alignment_status", "")),
            "label_alignment_blockers": blind_diag.get("label_alignment_blockers", []),
            "trade_eligible_set_blocker_counts": blind_diag.get("trade_eligible_set_blocker_counts", {}),
            "blocker": _s(market_report.get("blocker", "")),
        }
        if market == "crypto":
            summary["crypto_trigger_root_cause"] = "blind_sim_used_generic_crypto_replay_predictor_without_crypto-specific_sequence_trigger_scorer_biasing_predictions_toward_trailing"
            summary["crypto_actual_trigger_assignment_audit"] = {
                "actual_trigger_generated_in": "app.crypto_historical_replay._simulate_strategy_rows",
                "actual_trigger_assignment_method": "ordered_exit_sequence",
                "exit_condition_priority": ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"],
                "derived_from_final_pnl_only": False,
            }
            summary["crypto_predicted_trigger_generation_audit"] = {
                "predicted_trigger_generated_in": "app.model_quality_pass._crypto_blind_trigger_predict",
                "baseline_predictor_path": "app.model_quality_pass._predict_one",
                "baseline_trigger_mode": "generic_historical_replay_candidate_variant",
                "improved_trigger_mode": "blind_sequence_trigger_scorer_v1",
                "prediction_source_feature_availability": "entry_features_only",
            }
            summary.update(crypto_before_after)
            summary["residual_mismatch_artifact_path"] = _s((crypto_residual_artifacts.get("paths", {}) if isinstance(crypto_residual_artifacts.get("paths", {}), dict) else {}).get("rows", ""))
            summary["residual_mismatch_summary_path"] = _s((crypto_residual_artifacts.get("paths", {}) if isinstance(crypto_residual_artifacts.get("paths", {}), dict) else {}).get("summary", ""))
            summary["residual_mismatch_artifact_counts"] = (crypto_residual_artifacts.get("summary", {}) if isinstance(crypto_residual_artifacts.get("summary", {}), dict) else {})
            summary["crypto_training_readiness_decision"] = crypto_training_readiness
            if int(crypto_before_after.get("training_eligible_rows_after", 0) or 0) < int(crypto_before_after.get("training_eligible_rows_before", 0) or 0) - 20:
                summary["blocker"] = (_s(summary.get("blocker", "")) + (" | " if _s(summary.get("blocker", "")) else "") + "training_eligible_rows_dropped_materially_after_trigger_semantics_pass").strip()
        _write_json_atomic(paths["summary"], summary)
        _write_json_atomic(bucket_paths["candidate_universe"], {"market": market, "symbols": candidate_symbols, "count": int(len(candidate_symbols)), "ts": ts_now})
        _write_json_atomic(bucket_paths["onboarding_queue"], {"market": market, "symbols": onboarding_rows, "count": int(len(onboarding_rows)), "ts": ts_now})
        _write_json_atomic(bucket_paths["active_scan_set"], {"market": market, "symbols": active_rows, "count": int(len(active_rows)), "ts": ts_now})
        _write_json_atomic(bucket_paths["trade_eligible_set"], {"market": market, "symbols": eligible_rows, "count": int(len(eligible_rows)), "ts": ts_now})
        _write_json_atomic(bucket_paths["rejected_or_cooled_down_symbols"], {"market": market, "symbols": rejected_rows, "count": int(len(rejected_rows)), "ts": ts_now})
        market_report["summary"] = summary
        market_report["artifact_paths"] = paths
        market_report["symbol_bucket_artifacts"] = bucket_paths
        report["markets"][market] = market_report
        report["artifact_paths"][market] = paths
        report["symbol_bucket_artifacts"][market] = bucket_paths
        unified_trades.extend(trades)
        unified_decisions.extend(decisions)
        unified_skips.extend(skips)

    report["historical_blind_simulation_used_as_supplemental"] = bool(unified_trades)
    unified_paths = {
        "trades": os.path.join(hub_dir, "historical_blind_simulation_trades.jsonl"),
        "decisions": os.path.join(hub_dir, "historical_blind_simulation_decisions.jsonl"),
        "skips": os.path.join(hub_dir, "historical_blind_simulation_skips.jsonl"),
        "summary": os.path.join(hub_dir, "historical_blind_simulation_summary.json"),
    }
    _write_jsonl(unified_paths["trades"], unified_trades)
    _write_jsonl(unified_paths["decisions"], unified_decisions)
    _write_jsonl(unified_paths["skips"], unified_skips)
    _write_json_atomic(
        unified_paths["summary"],
        {
            "schema_version": HISTORICAL_BLIND_SIM_SCHEMA_VERSION,
            "enabled": True,
            "historical_blind_simulation_used_as_primary": False,
            "historical_blind_simulation_used_as_supplemental": bool(unified_trades),
            "completed_mock_trades": int(len(unified_trades)),
            "skipped_decisions": int(len(unified_skips)),
            "training_eligible_rows": int(sum(1 for r in unified_trades if bool(r.get("eligible_for_training", False)))),
            "diagnostic_only_rows": int(sum(1 for r in unified_trades if bool(r.get("diagnostic_only", False)))),
            "markets": {mk: (report["markets"].get(mk, {}).get("summary", {})) for mk in report["markets"]},
        },
    )
    report["unified_artifacts"] = unified_paths
    return report


def _stock_symbol_candidates(hub_dir: str, closed_rows: List[Dict[str, Any]], limit: int = 18) -> List[str]:
    out: List[str] = []
    for row in sorted(list(closed_rows or []), key=lambda r: int(_f(r.get("exit_ts", 0.0), 0.0)), reverse=True):
        sym = _s(row.get("symbol", "")).upper()
        if sym and sym not in out:
            out.append(sym)
        if len(out) >= limit:
            return out
    cache = _safe_read_json(os.path.join(hub_dir, "stocks", "stock_universe_cache.json"))
    for sym in list(cache.get("symbols", []) or []):
        ss = _s(sym).upper()
        if ss and ss not in out:
            out.append(ss)
        if len(out) >= limit:
            return out
    rank_rows = _safe_read_jsonl(os.path.join(hub_dir, "stocks", "scanner_rankings.jsonl"), max_lines=120)
    for row in reversed(rank_rows):
        top = row.get("top", []) if isinstance(row.get("top", []), list) else []
        for cand in top:
            if not isinstance(cand, dict):
                continue
            ss = _s(cand.get("symbol", "")).upper()
            if ss and ss not in out:
                out.append(ss)
            if len(out) >= limit:
                return out
    return out[:limit]


def _stock_provider_client(settings: Dict[str, Any], base_dir: str) -> Tuple[str, Any, Dict[str, Any]]:
    provider = _s(settings.get("stock_data_provider", "")).lower() or "alpaca"
    if provider not in {"alpaca", "twelvedata"}:
        provider = "alpaca"
    diagnostics = {"provider": provider, "credentials_available": False}
    if provider == "alpaca":
        key, secret = get_alpaca_creds(settings, base_dir)
        diagnostics["credentials_available"] = bool(key and secret)
        if key and secret:
            return provider, AlpacaBrokerClient(key, secret, "https://paper-api.alpaca.markets"), diagnostics
        td = get_twelvedata_api_key(settings, base_dir)
        if td:
            diagnostics["provider"] = "twelvedata"
            diagnostics["credentials_available"] = True
            diagnostics["fallback_used"] = "twelvedata"
            return "twelvedata", TwelveDataClient(td), diagnostics
        diagnostics["reason"] = "missing_stock_provider_credentials"
        return provider, None, diagnostics
    td = get_twelvedata_api_key(settings, base_dir)
    diagnostics["credentials_available"] = bool(td)
    if td:
        return provider, TwelveDataClient(td), diagnostics
    key, secret = get_alpaca_creds(settings, base_dir)
    if key and secret:
        diagnostics["provider"] = "alpaca"
        diagnostics["credentials_available"] = True
        diagnostics["fallback_used"] = "alpaca"
        return "alpaca", AlpacaBrokerClient(key, secret, "https://paper-api.alpaca.markets"), diagnostics
    diagnostics["reason"] = "missing_stock_provider_credentials"
    return provider, None, diagnostics


def _normalize_stock_ticker(value: Any) -> str:
    raw = _s(value).upper()
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Z0-9._-]+", "", raw)
    return cleaned[:16]


def _stock_manual_watchlist_path(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "manual_watchlist.json")


def _stock_watchlist_preview_dir(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "watchlist_previews")


def _stock_watchlist_preview_path(hub_dir: str, symbol: str) -> str:
    return os.path.join(_stock_watchlist_preview_dir(hub_dir), f"{_normalize_stock_ticker(symbol)}.json")


def validate_stock_watchlist_symbol(
    *,
    symbol: str,
    settings: Dict[str, Any],
    base_dir: str,
    hub_dir: str = "",
) -> Dict[str, Any]:
    normalized = _normalize_stock_ticker(symbol)
    provider, client, provider_diag = _stock_provider_client(settings, base_dir)
    result: Dict[str, Any] = {
        "symbol": normalized,
        "provider": provider,
        "valid": False,
        "status": "invalid_symbol" if normalized else "empty_symbol",
        "error": "",
        "already_in_watchlist": False,
        "stock_watchlist_search_query": _s(symbol),
        "stock_watchlist_search_provider": provider,
        "stock_watchlist_search_valid": False,
        "stock_watchlist_search_error": "",
    }
    existing_watch = {_normalize_stock_ticker(s) for s in _stock_symbol_candidates(hub_dir, [], limit=500)} if hub_dir else set()
    settings_watch = {_normalize_stock_ticker(tok) for tok in _s(settings.get("stock_universe_symbols", "")).replace("\n", ",").split(",")}
    if normalized and (normalized in existing_watch or normalized in settings_watch):
        result["already_in_watchlist"] = True
        result["valid"] = True
        result["status"] = "already_in_watchlist"
        result["stock_watchlist_search_valid"] = True
        return result
    if not normalized:
        result["stock_watchlist_search_error"] = "empty_symbol"
        return result
    if client is None:
        result["status"] = "provider_unavailable"
        result["error"] = _s(provider_diag.get("reason", "")) or "provider_unavailable"
        result["stock_watchlist_search_error"] = result["error"]
        return result
    try:
        valid = False
        now_ts = int(time.time())
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts - (120 * 86400)))
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))
        if provider == "alpaca":
            try:
                assets = client.list_tradable_assets()
            except Exception:
                assets = []
            if assets:
                valid = any(
                    _normalize_stock_ticker((row or {}).get("symbol", "")) == normalized
                    and str((row or {}).get("tradable", True)).lower() != "false"
                    for row in assets
                    if isinstance(row, dict)
                )
            if not valid:
                valid = len(client.get_stock_bars(
                    normalized,
                    timeframe="1Day",
                    limit=30,
                    feed="iex",
                    start_iso=start_iso,
                    end_iso=end_iso,
                )) > 0
        else:
            valid = len((client.get_time_series_batch([normalized], interval="1day", outputsize=5) or {}).get(normalized, [])) > 0
        result["valid"] = bool(valid)
        result["status"] = "valid_symbol" if valid else "invalid_symbol"
    except Exception as exc:
        result["status"] = "provider_unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["stock_watchlist_search_error"] = result["error"]
    result["stock_watchlist_search_valid"] = bool(result["valid"])
    return result


def _read_stock_manual_watchlist(hub_dir: str) -> Dict[str, Any]:
    payload = _safe_read_json(_stock_manual_watchlist_path(hub_dir))
    if not isinstance(payload.get("symbols", {}), dict):
        payload["symbols"] = {}
    return payload


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def add_stock_to_manual_watchlist(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    symbol: str,
    validation_provider: str,
) -> Dict[str, Any]:
    normalized = _normalize_stock_ticker(symbol)
    payload = _read_stock_manual_watchlist(hub_dir)
    symbols_map = payload.get("symbols", {}) if isinstance(payload.get("symbols", {}), dict) else {}
    now_ts = int(time.time())
    existing = symbols_map.get(normalized, {}) if isinstance(symbols_map.get(normalized, {}), dict) else {}
    symbols_map[normalized] = {
        "symbol": normalized,
        "added_at": int(existing.get("added_at", now_ts) or now_ts),
        "updated_at": now_ts,
        "added_by_user": True,
        "source": "manual_watchlist_search",
        "validation_provider": validation_provider,
        "historical_warmup_status": _s(existing.get("historical_warmup_status", "")) or "pending",
        "prediction_preview_status": _s(existing.get("prediction_preview_status", "")) or "pending",
    }
    payload["ts"] = now_ts
    payload["symbols"] = symbols_map
    _write_json_atomic(_stock_manual_watchlist_path(hub_dir), payload)

    watch = []
    for tok in _s(settings.get("stock_universe_symbols", "")).replace("\n", ",").split(","):
        sym = _normalize_stock_ticker(tok)
        if sym and sym not in watch:
            watch.append(sym)
    if normalized and normalized not in watch:
        watch.append(normalized)
    settings["stock_universe_symbols"] = ",".join(watch)
    return {"symbol": normalized, "symbols": watch, "storage_path": _stock_manual_watchlist_path(hub_dir)}


def _bar_close(row: Dict[str, Any]) -> float:
    for key in ("c", "close"):
        val = _f(row.get(key, 0.0), 0.0)
        if val > 0.0:
            return val
    return 0.0


def _bar_open(row: Dict[str, Any]) -> float:
    for key in ("o", "open"):
        val = _f(row.get(key, 0.0), 0.0)
        if val > 0.0:
            return val
    return _bar_close(row)


def _bar_high(row: Dict[str, Any]) -> float:
    for key in ("h", "high"):
        val = _f(row.get(key, 0.0), 0.0)
        if val > 0.0:
            return val
    return _bar_close(row)


def _bar_low(row: Dict[str, Any]) -> float:
    for key in ("l", "low"):
        val = _f(row.get(key, 0.0), 0.0)
        if val > 0.0:
            return val
    return _bar_close(row)


def _bar_ts(row: Dict[str, Any]) -> int:
    txt = _s(row.get("t", row.get("datetime", "")))
    if not txt:
        return 0
    try:
        txt = txt.replace(" ", "T")
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return int(time.mktime(time.strptime(txt[:19], "%Y-%m-%dT%H:%M:%S")))
    except Exception:
        try:
            return int(_f(txt, 0.0))
        except Exception:
            return 0


def _stock_cache_root(hub_dir: str) -> str:
    return os.path.join(hub_dir, "stocks", "historical_replay_cache")


def _stock_cache_path(hub_dir: str, symbol: str, timeframe: str) -> str:
    safe_symbol = _s(symbol).upper().replace("/", "_")
    return os.path.join(_stock_cache_root(hub_dir), f"{safe_symbol}_{timeframe}.json")


def _load_stock_cached_bars(hub_dir: str, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    payload = _safe_read_json(_stock_cache_path(hub_dir, symbol, timeframe))
    rows = payload.get("bars", []) if isinstance(payload.get("bars", []), list) else []
    return [dict(r) for r in rows if isinstance(r, dict)]


def _save_stock_cached_bars(hub_dir: str, symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> None:
    path = _stock_cache_path(hub_dir, symbol, timeframe)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"bars": list(bars or [])}, f)
    os.replace(tmp, path)


def warm_stock_historical_cache(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
    symbol: str,
    lookback_days: int = 365,
    timeframe: str = "1Hour",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    normalized = _normalize_stock_ticker(symbol)
    provider, client, _diag = _stock_provider_client(settings, base_dir)
    cache_before = _load_stock_cached_bars(hub_dir, normalized, timeframe) if normalized and (not force_refresh) else []
    out: Dict[str, Any] = {
        "symbol": normalized,
        "provider": provider,
        "timeframe": timeframe,
        "lookback_days": int(lookback_days),
        "cache_path": _stock_cache_root(hub_dir),
        "cache_rows_before": int(len(cache_before)),
        "remote_rows_fetched": 0,
        "cache_rows_after": int(len(cache_before)),
        "date_range": {},
        "warmup_status": "provider_unavailable" if client is None else "pending",
        "warmup_error": "",
    }
    if not normalized:
        out["warmup_status"] = "invalid_symbol"
        out["warmup_error"] = "invalid_symbol"
        return out
    if client is None:
        out["warmup_error"] = _s(_diag.get("reason", "")) or "provider_unavailable"
        return out
    now_ts = int(time.time())
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts - (int(lookback_days) * 86400)))
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))
    bars = list(cache_before)
    needs_fetch = force_refresh or len(bars) < 48
    if bars and not needs_fetch:
        ts_values = [_bar_ts(r) for r in bars if _bar_ts(r) > 0]
        if ts_values:
            out["date_range"] = {"start_ts": int(min(ts_values)), "end_ts": int(max(ts_values))}
            oldest_ok = min(ts_values) <= (now_ts - (int(lookback_days) * 86400) + 86400)
            newest_ok = max(ts_values) >= (now_ts - 172800)
            needs_fetch = not (oldest_ok and newest_ok)
    try:
        if needs_fetch:
            fetched: List[Dict[str, Any]] = []
            if provider == "alpaca":
                fetched = client.get_stock_bars(
                    normalized,
                    timeframe=timeframe,
                    limit=max(480, int((lookback_days * 24) * 1.2)),
                    feed="iex",
                    start_iso=start_iso,
                    end_iso=end_iso,
                )
                if len(fetched) < 48 and timeframe == "1Hour":
                    fetched = client.get_stock_bars(
                        normalized,
                        timeframe="1Day",
                        limit=max(180, int(lookback_days)),
                        feed="iex",
                        start_iso=start_iso,
                        end_iso=end_iso,
                    )
            else:
                interval = "1h" if timeframe.lower() == "1hour" else "1day"
                fetched = list((client.get_time_series_batch([normalized], interval=interval, outputsize=max(480, int((lookback_days * 24) * 1.2))) or {}).get(normalized, []))
            out["remote_rows_fetched"] = int(len(fetched))
            if fetched:
                merged: Dict[str, Dict[str, Any]] = {}
                for row in list(cache_before) + list(fetched):
                    if isinstance(row, dict):
                        merged[_s(row.get("t", row.get("datetime", "")))] = dict(row)
                bars = sorted(merged.values(), key=_bar_ts)
                _save_stock_cached_bars(hub_dir, normalized, timeframe, bars)
        out["cache_rows_after"] = int(len(bars))
        ts_values = [_bar_ts(r) for r in bars if _bar_ts(r) > 0]
        if ts_values:
            out["date_range"] = {"start_ts": int(min(ts_values)), "end_ts": int(max(ts_values))}
        out["warmup_status"] = "ready" if len(bars) >= 48 else "insufficient_history"
    except Exception as exc:
        out["warmup_status"] = "provider_error"
        out["warmup_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _build_stock_preview_candidate(symbol: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [dict(r) for r in list(bars or []) if _bar_close(r) > 0.0]
    rows.sort(key=_bar_ts)
    if len(rows) < 25:
        return {}
    idx = len(rows) - 1
    px = _bar_close(rows[idx])
    prev6 = _bar_close(rows[idx - 6])
    prev24 = _bar_close(rows[idx - 24])
    if px <= 0.0 or prev6 <= 0.0 or prev24 <= 0.0:
        return {}
    recent_returns: List[float] = []
    for j in range(max(1, idx - 12), idx + 1):
        prev_px = _bar_close(rows[j - 1])
        cur_px = _bar_close(rows[j])
        if prev_px > 0.0 and cur_px > 0.0:
            recent_returns.append(((cur_px / prev_px) - 1.0) * 100.0)
    mom6 = ((px / prev6) - 1.0) * 100.0
    mom24 = ((px / prev24) - 1.0) * 100.0
    trend_momentum_score = (0.55 * mom6) + (0.45 * mom24)
    recent_volatility = _stddev_local(recent_returns)
    signal_margin = max(0.0, (mom6 / 100.0)) + max(0.0, (mom24 / 100.0))
    return {
        "symbol": normalized if (normalized := _normalize_stock_ticker(symbol)) else _normalize_stock_ticker(symbol),
        "market": "stocks",
        "source_type": "historical_api_replay",
        "provider": "alpaca_or_twelvedata",
        "candle_timeframe": "1Hour",
        "entry_price": round(float(px), 10),
        "entry_ts": int(_bar_ts(rows[idx])),
        "recent_return_6": round(float(mom6), 6),
        "recent_return_24": round(float(mom24), 6),
        "recent_volatility": round(float(recent_volatility), 6),
        "trend_momentum_score": round(float(trend_momentum_score), 6),
        "signal_margin": round(float(signal_margin), 6),
        "signal_side": "long",
        "regime": "historical_preview",
    }


def build_stock_watchlist_prediction_preview(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
    symbol: str,
    timeframe: str = "1Hour",
    lookback_days: int = 365,
) -> Dict[str, Any]:
    normalized = _normalize_stock_ticker(symbol)
    bars = _load_stock_cached_bars(hub_dir, normalized, timeframe)
    replay_rows = _simulate_stock_trades_from_bars(normalized, bars)
    candidate = _build_stock_preview_candidate(normalized, bars)
    preview: Dict[str, Any] = {
        "symbol": normalized,
        "provider": _s(settings.get("stock_data_provider", "alpaca")) or "alpaca",
        "rows_available": int(len(bars)),
        "replay_rows_generated": int(len(replay_rows)),
        "source_type": "historical_api_replay",
        "timeframe": timeframe,
        "lookback_days": int(lookback_days),
        "predicted_direction": "warming / insufficient data",
        "predicted_exit_trigger": "warming / insufficient data",
        "predicted_pnl_trend": "warming / insufficient data",
        "confidence": 0.0,
        "stock_pnl_quality_score": 0.0,
        "stock_pnl_quality_reason": "",
        "stock_trade_quality_score": 0.0,
        "trade_quality_gate_applied": False,
        "direction_scores": {},
        "trigger_scores": {},
        "stock_readiness": {
            "sufficient_history": bool(len(bars) >= 48),
            "sufficient_replay_rows": bool(len(replay_rows) >= 12),
            "trade_eligible": False,
            "trade_blockers": [],
        },
        "common_historical_outcomes": {
            "trigger_distribution": _count_by(replay_rows, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
            "direction_distribution": _count_by(replay_rows, lambda r: _s(r.get("actual_direction", _trade_direction(r))).lower() or "unknown"),
            "average_pnl_pct": round(sum(_trade_return_pct(r) for r in replay_rows) / max(1, len(replay_rows)), 6) if replay_rows else 0.0,
        },
        "explanation": "",
    }
    blockers: List[str] = []
    if len(bars) < 48:
        blockers.append("insufficient_history")
    if len(replay_rows) < 12:
        blockers.append("insufficient_replay_rows")
    if not candidate:
        blockers.append("candidate_features_unavailable")
    if candidate and replay_rows:
        pred = _stock_predict_one(train_rows=replay_rows[:-1] if len(replay_rows) > 1 else replay_rows, candidate=candidate, regime="historical_preview", predictor_variant="stock_pnl_quality_v2")
        preview["predicted_direction"] = _s(pred.get("predicted_direction", "")) or "unknown"
        preview["predicted_exit_trigger"] = _s(pred.get("predicted_exit_trigger", "")) or "Unknown"
        preview["predicted_pnl_trend"] = _s(pred.get("predicted_pnl_trend", "")) or ("up" if _s(pred.get("predicted_direction", "")).lower() == "up" else "down")
        preview["confidence"] = round(_f(pred.get("predicted_confidence", 0.0), 0.0), 6)
        preview["stock_pnl_quality_score"] = round(_f(pred.get("stock_pnl_quality_score", 0.0), 0.0), 6)
        preview["stock_pnl_quality_reason"] = _s(pred.get("stock_pnl_quality_reason", ""))
        preview["stock_trade_quality_score"] = round(_f(pred.get("stock_trade_quality_score", 0.0), 0.0), 6)
        preview["trade_quality_gate_applied"] = bool(pred.get("stock_trade_quality_gate_applied", False))
        preview["direction_scores"] = dict(pred.get("direction_scores", {}) if isinstance(pred.get("direction_scores", {}), dict) else {})
        preview["trigger_scores"] = dict(pred.get("trigger_scores", {}) if isinstance(pred.get("trigger_scores", {}), dict) else {})
        if _s(preview.get("predicted_direction", "")).lower() != "up":
            blockers.append("predicted_direction_not_up")
        if _s(preview.get("predicted_pnl_trend", "")).lower() != "up":
            blockers.append("predicted_pnl_trend_not_up")
        if _f(preview.get("confidence", 0.0), 0.0) < 0.30:
            blockers.append("preview_confidence_low")
        if _f(preview.get("stock_pnl_quality_score", 0.0), 0.0) < 0.50:
            blockers.append("stock_pnl_quality_weak")
        if bool(preview.get("trade_quality_gate_applied", False)):
            blockers.append("trade_quality_gate_applied")
    rollout = _safe_read_json(os.path.join(hub_dir, "model_quality_full_pass.json"))
    stocks_rollout = rollout.get("controlled_rollout_readiness", {}).get("stocks", {}) if isinstance(rollout.get("controlled_rollout_readiness", {}), dict) else {}
    rollout_eligible = False
    if stocks_rollout:
        rollout_eligible = bool(
            stocks_rollout.get("controlled_rollout_eligible", stocks_rollout.get("eligible", False))
        )
    if stocks_rollout and not rollout_eligible:
        blockers.append("market_rollout_not_ready")
    preview["stock_readiness"]["trade_blockers"] = blockers
    preview["stock_readiness"]["trade_eligible"] = not blockers
    preview["manual_watchlist_trade_eligible"] = bool(not blockers)
    preview["manual_watchlist_trade_blockers"] = list(blockers)
    preview["explanation"] = (
        "Historical replay preview is ready and the symbol passes local readiness checks."
        if not blockers
        else "Historical replay preview is available, but live trading stays blocked until readiness checks pass: " + ", ".join(blockers)
    )
    os.makedirs(_stock_watchlist_preview_dir(hub_dir), exist_ok=True)
    _write_json_atomic(_stock_watchlist_preview_path(hub_dir, normalized), preview)
    return preview


def _simulate_stock_trades_from_bars(symbol: str, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in list(bars or []) if _bar_close(r) > 0.0]
    rows.sort(key=_bar_ts)
    if len(rows) < 48:
        return []
    out: List[Dict[str, Any]] = []
    in_pos = False
    entry_px = 0.0
    entry_ts = 0
    entry_idx = -1
    peak_px = 0.0
    armed = False
    gap_pct = 0.018
    max_hold = 18
    stock_exit_shape_thresholds = {
        "trailing_arm_pct": 1.5,
        "trailing_pullback_pct": 1.8,
        "stale_hold_bars": float(max_hold),
    }
    for idx in range(24, len(rows)):
        px = _bar_close(rows[idx])
        ts = _bar_ts(rows[idx])
        prev6 = _bar_close(rows[idx - 6])
        prev24 = _bar_close(rows[idx - 24])
        if prev6 <= 0.0 or prev24 <= 0.0 or px <= 0.0 or ts <= 0:
            continue
        mom6 = (px / prev6) - 1.0
        mom24 = (px / prev24) - 1.0
        recent_returns: List[float] = []
        for j in range(max(1, idx - 12), idx):
            prev_px = _bar_close(rows[j - 1])
            cur_px = _bar_close(rows[j])
            if prev_px > 0.0 and cur_px > 0.0:
                recent_returns.append(((cur_px / prev_px) - 1.0) * 100.0)
        recent_volatility = _stddev_local(recent_returns)
        trend_momentum_score = ((0.55 * (mom6 * 100.0)) + (0.45 * (mom24 * 100.0)))
        if not in_pos:
            if mom6 >= 0.012 and mom24 >= 0.02:
                in_pos = True
                entry_px = px
                entry_ts = ts
                entry_idx = idx
                peak_px = px
                armed = False
            continue
        peak_px = max(peak_px, px)
        gain = (px / entry_px) - 1.0 if entry_px > 0.0 else 0.0
        hold_bars = idx - max(0, entry_idx)
        if gain >= 0.015:
            armed = True
        trigger = ""
        if armed and px <= (peak_px * (1.0 - gap_pct)):
            trigger = "Trailing"
        elif hold_bars >= max_hold and (mom6 <= -0.004 or gain <= 0.004):
            trigger = "Stale Alignment"
        if not trigger:
            continue
        trade_slice = rows[max(0, entry_idx): idx + 1]
        highs = [_bar_high(r) for r in trade_slice if _bar_high(r) > 0.0]
        lows = [_bar_low(r) for r in trade_slice if _bar_low(r) > 0.0]
        closes = [_bar_close(r) for r in trade_slice if _bar_close(r) > 0.0]
        opens = [_bar_open(r) for r in trade_slice if _bar_open(r) > 0.0]
        peak_px = max(highs or [px])
        trough_px = min(lows or [px])
        peak_idx = 0
        peak_profit_pct = ((peak_px / entry_px) - 1.0) * 100.0 if entry_px > 0.0 else 0.0
        worst_drawdown_pct = ((trough_px / entry_px) - 1.0) * 100.0 if entry_px > 0.0 else 0.0
        for local_idx, r in enumerate(trade_slice):
            if abs(_bar_high(r) - peak_px) <= 1e-9:
                peak_idx = local_idx
        bars_in_trade = len(trade_slice)
        bars_since_peak = max(0, bars_in_trade - 1 - peak_idx)
        drawdown_from_peak_pct = ((px / peak_px) - 1.0) * 100.0 if peak_px > 0.0 else 0.0
        trailing_armed = bool(peak_profit_pct >= stock_exit_shape_thresholds["trailing_arm_pct"])
        trailing_pullback_pct = abs(drawdown_from_peak_pct)
        favorable_then_softened_flag = bool(trailing_armed and trailing_pullback_pct >= stock_exit_shape_thresholds["trailing_pullback_pct"])
        stale_hold_profile = bool(bars_in_trade >= int(stock_exit_shape_thresholds["stale_hold_bars"]) and peak_profit_pct < 2.0 and abs(mom6 * 100.0) < 1.5)
        pre_slice = rows[max(0, entry_idx - 12): entry_idx]
        pre_rets: List[float] = []
        for j in range(1, len(pre_slice)):
            prev_px = _bar_close(pre_slice[j - 1])
            cur_px = _bar_close(pre_slice[j])
            if prev_px > 0.0 and cur_px > 0.0:
                pre_rets.append(((cur_px / prev_px) - 1.0) * 100.0)
        pre_vol = _stddev_local(pre_rets)
        volatility_expansion_pct = ((recent_volatility - pre_vol) / max(0.05, pre_vol or 0.05)) * 100.0 if recent_volatility > 0.0 else 0.0
        peak_close = closes[peak_idx] if peak_idx < len(closes) else px
        trend_decay_after_peak = ((px / peak_close) - 1.0) * 100.0 if peak_close > 0.0 else 0.0
        prev_close = _bar_close(rows[idx - 1]) if idx > 0 else px
        gap_against_position_pct = max(0.0, ((prev_close - opens[-1]) / max(1e-9, prev_close)) * 100.0)
        gap_with_position_pct = max(0.0, ((opens[-1] - prev_close) / max(1e-9, prev_close)) * 100.0)
        if len(closes) >= 4 and closes[-4] > 0.0:
            exit_m3 = ((closes[-1] / closes[-4]) - 1.0) * 100.0
        else:
            exit_m3 = 0.0
        if len(closes) >= 7 and closes[-7] > 0.0:
            exit_m6 = ((closes[-1] / closes[-7]) - 1.0) * 100.0
        else:
            exit_m6 = 0.0
        pnl_usd = px - entry_px
        out.append(
            {
                "symbol": symbol,
                "market": "stocks",
                "source_type": "historical_api_replay",
                "provider": "alpaca_or_twelvedata",
                "candle_timeframe": "1Hour",
                "entry_ts": int(entry_ts),
                "exit_ts": int(ts),
                "qty": 1.0,
                "entry_price": round(float(entry_px), 10),
                "exit_price": round(float(px), 10),
                "hold_hours": round(float(max(0.0, (ts - entry_ts) / 3600.0)), 6),
                "pnl_usd": round(float(pnl_usd), 8),
                "actual_direction": "up" if pnl_usd > 1e-9 else "down" if pnl_usd < -1e-9 else "flat",
                "actual_exit_trigger": trigger,
                "event_exit_tag": f"historical_api_replay:{trigger.lower().replace(' ', '_')}",
                "raw_rule_reason": trigger.lower().replace(" ", "_"),
                "recent_return_6": round(float(mom6 * 100.0), 6),
                "recent_return_24": round(float(mom24 * 100.0), 6),
                "recent_volatility": round(float(recent_volatility), 6),
                "trend_momentum_score": round(float(trend_momentum_score), 6),
                "signal_side": "long",
                "signal_margin": round(float(max(mom6, 0.0) + max(mom24, 0.0)), 6),
                "max_favorable_excursion_pct": round(float(max(0.0, peak_profit_pct)), 6),
                "max_adverse_excursion_pct": round(float(min(0.0, worst_drawdown_pct)), 6),
                "peak_profit_pct": round(float(max(0.0, peak_profit_pct)), 6),
                "worst_drawdown_pct": round(float(worst_drawdown_pct), 6),
                "drawdown_from_peak_pct": round(float(drawdown_from_peak_pct), 6),
                "bars_since_peak": int(bars_since_peak),
                "bars_in_trade": int(bars_in_trade),
                "trailing_armed": bool(trailing_armed),
                "trailing_pullback_pct": round(float(trailing_pullback_pct), 6),
                "favorable_then_softened_flag": bool(favorable_then_softened_flag),
                "stale_hold_profile": bool(stale_hold_profile),
                "volatility_expansion_pct": round(float(volatility_expansion_pct), 6),
                "trend_decay_after_peak": round(float(trend_decay_after_peak), 6),
                "gap_against_position_pct": round(float(gap_against_position_pct), 6),
                "gap_with_position_pct": round(float(gap_with_position_pct), 6),
                "exit_momentum_3": round(float(exit_m3), 6),
                "exit_momentum_6": round(float(exit_m6), 6),
                "stock_exit_shape_thresholds": dict(stock_exit_shape_thresholds),
            }
        )
        in_pos = False
        entry_px = 0.0
        entry_ts = 0
        entry_idx = -1
        peak_px = 0.0
        armed = False
    return out


def _generate_stock_historical_replay_closed_trades(
    *,
    hub_dir: str,
    base_dir: str,
    settings: Dict[str, Any],
    existing_closed_rows: List[Dict[str, Any]],
    lookback_days: int = 90,
    max_symbols: int = 24,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    provider, client, diag = _stock_provider_client(settings, base_dir)
    diagnostics: Dict[str, Any] = {
        "source_type": "historical_api_replay",
        "provider": provider,
        "symbols_covered": [],
        "candle_timeframe": "1Hour",
        "lookback_days": int(lookback_days),
        "hour_bar_limit": 480,
        "daily_bar_limit": 180,
        "date_range": {},
        "rows_generated": 0,
        "skipped": [],
        "stock_backfill_force_refresh": bool(force_refresh),
        "stock_backfill_cache_path": _stock_cache_root(hub_dir),
        "stock_exit_shape_features_enabled": True,
        "stock_exit_shape_rows_with_features": 0,
        "stock_exit_shape_rows_missing_features": 0,
        "stock_exit_shape_feature_names": [
            "max_favorable_excursion_pct",
            "max_adverse_excursion_pct",
            "peak_profit_pct",
            "worst_drawdown_pct",
            "drawdown_from_peak_pct",
            "bars_since_peak",
            "bars_in_trade",
            "trailing_armed",
            "trailing_pullback_pct",
            "favorable_then_softened_flag",
            "stale_hold_profile",
            "volatility_expansion_pct",
            "trend_decay_after_peak",
            "gap_against_position_pct",
            "gap_with_position_pct",
            "exit_momentum_3",
            "exit_momentum_6",
        ],
        "stock_exit_shape_thresholds": {
            "trailing_arm_pct": 1.5,
            "trailing_pullback_pct": 1.8,
            "stale_hold_bars": 18,
        },
    }
    if client is None:
        diagnostics["reason_unavailable"] = _s(diag.get("reason", "")) or "provider_unavailable"
        return {"rows": list(existing_closed_rows or []), "diagnostics": diagnostics}
    symbols = _stock_symbol_candidates(hub_dir, existing_closed_rows, limit=max_symbols)
    generated: List[Dict[str, Any]] = []
    ts_values: List[int] = []
    now_ts = int(time.time())
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts - (int(lookback_days) * 86400)))
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))
    diagnostics["stock_backfill_symbols_requested"] = int(len(symbols))
    cache_rows_loaded = 0
    remote_rows_fetched = 0
    errors: List[Dict[str, Any]] = []
    for sym in symbols:
        bars: List[Dict[str, Any]] = []
        if not force_refresh:
            bars = _load_stock_cached_bars(hub_dir, sym, "1Hour")
            if bars:
                cache_rows_loaded += len(bars)
        try:
            if len(list(bars or [])) < 48:
                if provider == "alpaca":
                    bars = client.get_stock_bars(
                        sym,
                        timeframe="1Hour",
                        limit=max(480, int((lookback_days * 24) * 1.2)),
                        feed="iex",
                        start_iso=start_iso,
                        end_iso=end_iso,
                    )
                    if len(list(bars or [])) < 48:
                        bars = client.get_stock_bars(
                            sym,
                            timeframe="1Day",
                            limit=max(180, int(lookback_days)),
                            feed="iex",
                            start_iso=start_iso,
                            end_iso=end_iso,
                        )
                else:
                    bars_map = client.get_time_series_batch([sym], interval="1h", outputsize=max(480, int((lookback_days * 24) * 1.2)))
                    bars = list((bars_map.get(sym, []) if isinstance(bars_map, dict) else []) or [])
                remote_rows_fetched += len(list(bars or []))
                if bars:
                    _save_stock_cached_bars(hub_dir, sym, "1Hour", list(bars or []))
        except Exception as exc:
            diagnostics["skipped"].append({"symbol": sym, "reason": f"fetch_error:{type(exc).__name__}"})
            errors.append({"symbol": sym, "reason": f"fetch_error:{type(exc).__name__}"})
            continue
        if len(list(bars or [])) < 48:
            diagnostics["skipped"].append({"symbol": sym, "reason": "insufficient_bars", "bars": int(len(list(bars or [])))})
            continue
        trade_rows = _simulate_stock_trades_from_bars(sym, list(bars or []))
        if not trade_rows:
            diagnostics["skipped"].append({"symbol": sym, "reason": "no_replay_trades"})
            continue
        diagnostics["symbols_covered"].append(sym)
        generated.extend(trade_rows)
        ts_values.extend([int(_f(r.get("entry_ts", 0.0), 0.0)) for r in trade_rows])
        ts_values.extend([int(_f(r.get("exit_ts", 0.0), 0.0)) for r in trade_rows])
    generated.sort(key=lambda r: (int(_f(r.get("entry_ts", 0.0), 0.0)), _s(r.get("symbol", ""))))
    diagnostics["rows_generated"] = int(len(generated))
    diagnostics["stock_exit_shape_rows_with_features"] = int(sum(1 for r in generated if "bars_in_trade" in r))
    diagnostics["stock_exit_shape_rows_missing_features"] = int(sum(1 for r in generated if "bars_in_trade" not in r))
    diagnostics["stock_backfill_symbols_completed"] = int(len(diagnostics["symbols_covered"]))
    diagnostics["stock_backfill_remote_rows_fetched"] = int(remote_rows_fetched)
    diagnostics["stock_backfill_cache_rows_loaded"] = int(cache_rows_loaded)
    diagnostics["stock_backfill_errors"] = errors
    if ts_values:
        diagnostics["date_range"] = {"start_ts": int(min(ts_values)), "end_ts": int(max(ts_values))}
    return {"rows": generated if generated else list(existing_closed_rows or []), "diagnostics": diagnostics}


def build_synthetic_replay_artifact(
    hub_dir: str,
    market: str,
    closed_rows: List[Dict[str, Any]],
    deterministic_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    m = _s(market).lower()
    det_cfg = dict(deterministic_config or {})
    det_enabled = bool(det_cfg.get("enabled", False))
    rows = list(closed_rows or [])
    if det_enabled:
        rows = _apply_replay_deterministic_filters(
            rows,
            start_ts=int(_f(det_cfg.get("start_ts", 0), 0.0)),
            cutoff_ts=int(_f(det_cfg.get("cutoff_ts", 0), 0.0)),
            locked_symbols=list(det_cfg.get("locked_symbols", []) or []),
        )
    else:
        rows = sorted(rows, key=_stable_row_sort_key)
    # Bound per-run replay cost while keeping enough coverage for stable estimates.
    if len(rows) > 260:
        rows = rows[-260:]
    if len(rows) < 30:
        return {
            "status": "no_data",
            "summary": f"Not enough closed trades for {m} replay.",
            "meta": {"market": m, "closed_trades_total": int(len(rows)), "train_trades": 0, "test_trades": 0},
            "best_test_metrics": {"directional_accuracy_pct": 0.0, "trigger_match_pct": 0.0, "pnl_trend_match_pct": 0.0},
            "hybrid_test_metrics": {"directional_accuracy_pct": 0.0, "trigger_match_pct": 0.0, "pnl_trend_match_pct": 0.0},
            "iterations": [],
        }

    for r in rows:
        r["actual_direction"] = _trade_direction(r)
        r["replay_row_id"] = _replay_row_id(r)

    n = len(rows)
    test_n = max(8, min(40, int(round(n * 0.20))))
    train_n_min = max(20, min(120, int(round(n * 0.60))))
    step_n = max(5, int(test_n // 2))
    windows: List[Dict[str, Any]] = []
    admitted_all: List[Dict[str, Any]] = []
    abstained_all: List[Dict[str, Any]] = []
    preds_all: List[Dict[str, Any]] = []
    thr_values: List[float] = []
    policy_snapshots: List[Dict[str, Any]] = []
    validation_policy_quality: List[Dict[str, Any]] = []
    max_windows = 10
    win_count = 0
    safe_selection_snapshots: List[Dict[str, Any]] = []
    for end in range(train_n_min, n - test_n + 1, step_n):
        train = rows[:end]
        test = rows[end : end + test_n]
        if len(test) < 4:
            continue
        current_window_index = win_count + 1
        predictor_variants = ["candidate"]
        if m == "crypto" and any(_is_historical_strategy_replay_row(r) for r in train):
            predictor_variants = ["baseline", "candidate", "label_compatible_v2", "label_compatible_v3_sequence"]
        elif m == "forex":
            predictor_variants = ["baseline", "candidate"]
        elif m == "stocks":
            predictor_variants = ["candidate", "stock_pnl_quality_v2"]
        variant_predictions: Dict[str, List[Dict[str, Any]]] = {}
        variant_abstain: Dict[str, Dict[str, Any]] = {}
        variant_scored: Dict[str, Dict[str, Any]] = {}
        variant_summaries: Dict[str, Dict[str, Any]] = {}
        for predictor_variant in predictor_variants:
            abstain = _calibrate_abstain_threshold(train, m, predictor_variant=predictor_variant)
            threshold = _f(abstain.get("threshold", 0.6), 0.6)
            selected_policy = abstain.get("policy", {}) if isinstance(abstain.get("policy", {}), dict) else {}
            test_preds: List[Dict[str, Any]] = []
            prior = list(train)
            for r in test:
                regime = _regime_from_prior(prior)
                rr = dict(r)
                rr["regime"] = regime
                if _s(rr.get("source_type", "")) == "completed_live_decision_snapshot" and _s(rr.get("predicted_direction", "")):
                    pred = {
                        "predicted_direction": _s(rr.get("predicted_direction", "flat")).lower() or "flat",
                        "predicted_exit_trigger": _s(rr.get("predicted_exit_trigger", "Unknown")) or "Unknown",
                        "predicted_hold_hours": _f(rr.get("hold_hours", 0.0), 0.0),
                        "predicted_exit_price": _f(rr.get("exit_price", rr.get("actual_exit_price", 0.0)), 0.0),
                        "predicted_confidence": _f(rr.get("predicted_confidence", 0.0), 0.0),
                        "trigger_scores": dict(rr.get("trigger_scores", {}) if isinstance(rr.get("trigger_scores", {}), dict) else {}),
                        "direction_scores": dict(rr.get("direction_scores", {}) if isinstance(rr.get("direction_scores", {}), dict) else {}),
                        "predictor_source_mode": "completed_live_decision_snapshot",
                        "predictor_variant": _s(rr.get("predictor_variant", "")) or predictor_variant,
                        "predictor_mode": _s(rr.get("predictor_mode", "")) or _s(rr.get("selected_predictor_name", "")) or "live_snapshot",
                        "trigger_margin": _f(rr.get("trigger_margin", 0.0), 0.0),
                        "direction_margin": _f(rr.get("direction_margin", 0.0), 0.0),
                    }
                else:
                    pred = _predict_one(train_rows=prior, candidate=rr, regime=regime, market=m, predictor_variant=predictor_variant)
                merged = dict(rr)
                merged.update(pred)
                merged["predictor_variant"] = _s(merged.get("predictor_variant", "")) or predictor_variant
                merged["walkforward_window_index"] = int(current_window_index)
                merged["predicted_exit_ts"] = int(
                    _f(rr.get("entry_ts", 0.0), 0.0) + int(round(_f(pred.get("predicted_hold_hours", 0.0), 0.0) * 3600.0))
                )
                merged["replay_row_id"] = _replay_row_id(merged)
                test_preds.append(merged)
                prior.append(rr)
            scored = _score_crypto_with_policy(test_preds, selected_policy) if m == "crypto" else _score_with_threshold(test_preds, threshold, m)
            variant_predictions[predictor_variant] = test_preds
            variant_abstain[predictor_variant] = abstain
            variant_scored[predictor_variant] = scored
            policy_name = (
                _s(abstain.get("policy_name", ""))
                or _s(scored.get("policy_name", ""))
                or ("no_abstain_full_universe" if threshold <= 0.0 else f"threshold_{threshold:.2f}")
            )
            variant_summaries[predictor_variant] = _selection_option_summary(
                policy_name,
                scored,
                predictor_variant=predictor_variant,
                policy=selected_policy,
            )
        baseline_variant = "baseline" if "baseline" in variant_predictions else predictor_variants[0]
        baseline_summary = variant_summaries[baseline_variant]
        candidate_variants = [v for v in predictor_variants if v != baseline_variant]
        candidate_evaluations: List[Dict[str, Any]] = []
        selected_variant = baseline_variant
        selected_scored = variant_scored[baseline_variant]
        selected_preds = variant_predictions[baseline_variant]
        selected_abstain = variant_abstain[baseline_variant]
        selected_failures: List[str] = []
        winning_candidate_summary: Dict[str, Any] = {}
        baseline_unknown = len([r for r in variant_predictions[baseline_variant] if _s(r.get("predicted_exit_trigger", "")) == "Unknown"])
        for cand_variant in candidate_variants:
            cand_summary = variant_summaries[cand_variant]
            candidate_unknown = len([r for r in variant_predictions[cand_variant] if _s(r.get("predicted_exit_trigger", "")) == "Unknown"])
            guardrail_failures = _safe_selection_guardrails(
                market=m,
                baseline=baseline_summary,
                candidate=cand_summary,
                full_test_rows=len(variant_predictions[cand_variant]),
                baseline_unknown_count=baseline_unknown,
                candidate_unknown_count=candidate_unknown,
            )
            candidate_evaluations.append(
                {
                    "variant": cand_variant,
                    "summary": cand_summary,
                    "guardrail_failures": list(guardrail_failures),
                }
            )
        passing_candidates = [c for c in candidate_evaluations if not c.get("guardrail_failures")]
        if passing_candidates:
            passing_candidates.sort(key=lambda c: (_f(c["summary"].get("score", 0.0), 0.0), _f(c["summary"].get("admission_rate_pct", 0.0), 0.0)), reverse=True)
            winner = passing_candidates[0]
            selected_variant = _s(winner.get("variant", "")) or baseline_variant
            selected_scored = variant_scored[selected_variant]
            selected_preds = variant_predictions[selected_variant]
            selected_abstain = variant_abstain[selected_variant]
            winning_candidate_summary = winner.get("summary", {})
        else:
            selected_failures = list(candidate_evaluations[0].get("guardrail_failures", [])) if candidate_evaluations else []
        threshold = _f(selected_abstain.get("threshold", 0.6), 0.6)
        thr_values.append(float(threshold))
        selected_policy = selected_abstain.get("policy", {}) if isinstance(selected_abstain.get("policy", {}), dict) else {}
        if selected_policy:
            policy_snapshots.append(dict(selected_policy))
        validation_policy_quality.append(
            {
                "policy_name": _s(selected_abstain.get("policy_name", "")) or _s(selected_scored.get("policy_name", "")),
                "coverage": round(_f(selected_scored.get("coverage", 0.0), 0.0), 6),
                "metrics": selected_scored.get("metrics", {}),
                "class_precision": selected_scored.get("class_precision", {}),
                "reason_counts": selected_scored.get("reason_counts", {}),
                "policy_frontier": selected_abstain.get("policy_frontier", []),
            }
        )
        safe_selection_snapshots.append(
            {
                "candidate_policy_evaluated": bool(candidate_evaluations),
                "candidate_policy_selected": bool(selected_variant != baseline_variant),
                "fallback_to_baseline": bool(selected_variant == baseline_variant and bool(candidate_evaluations)),
                "fallback_reason": ",".join(selected_failures) if selected_failures else "",
                "baseline_metrics": baseline_summary.get("metrics", {}),
                "candidate_metrics": (winning_candidate_summary or (candidate_evaluations[0].get("summary", {}) if candidate_evaluations else {})).get("metrics", {}),
                "baseline_full_metrics": baseline_summary.get("full_metrics", {}),
                "candidate_full_metrics": (winning_candidate_summary or (candidate_evaluations[0].get("summary", {}) if candidate_evaluations else {})).get("full_metrics", {}),
                "baseline_admitted_trades": int(baseline_summary.get("admitted_trades", 0)),
                "candidate_admitted_trades": int((winning_candidate_summary or (candidate_evaluations[0].get("summary", {}) if candidate_evaluations else {})).get("admitted_trades", 0)),
                "baseline_admission_rate_pct": _f(baseline_summary.get("admission_rate_pct", 0.0), 0.0),
                "candidate_admission_rate_pct": _f((winning_candidate_summary or (candidate_evaluations[0].get("summary", {}) if candidate_evaluations else {})).get("admission_rate_pct", 0.0), 0.0),
                "selection_guardrails_failed": list(selected_failures),
                "selected_predictor_variant": selected_variant,
                "baseline_policy_name": baseline_summary.get("policy_name", ""),
                "candidate_policy_name": (winning_candidate_summary or (candidate_evaluations[0].get("summary", {}) if candidate_evaluations else {})).get("policy_name", ""),
                "candidate_evaluations": candidate_evaluations,
            }
        )
        preds_all.extend(selected_preds)
        scored = selected_scored
        admitted = scored.get("admitted", []) if isinstance(scored.get("admitted", []), list) else []
        abstained = scored.get("abstained", []) if isinstance(scored.get("abstained", []), list) else []
        admitted_all.extend(admitted)
        abstained_all.extend(abstained)
        windows.append(
            {
                "train_rows": int(len(train)),
                "test_rows": int(len(test_preds)),
                "admitted_rows": int(len(admitted)),
                "abstained_rows": int(len(abstained)),
                "threshold": round(float(threshold), 4),
                "coverage": round(_f(scored.get("coverage", 0.0), 0.0), 6),
                "metrics": scored.get("metrics", {}),
                "full_universe_metrics": scored.get("full_metrics", {}),
                "abstained_metrics": scored.get("abstained_metrics", {}),
                "admission_reason_counts": scored.get("reason_counts", {}),
                "selected_policy": selected_policy,
                "safe_selection": safe_selection_snapshots[-1],
            }
        )
        win_count += 1
        if win_count >= max_windows:
            break

    if not windows:
        return {
            "status": "no_data",
            "summary": f"{m} synthetic replay could not build enough walk-forward windows.",
            "meta": {"market": m, "closed_trades_total": int(len(rows)), "train_trades": 0, "test_trades": 0},
            "best_test_metrics": {"directional_accuracy_pct": 0.0, "trigger_match_pct": 0.0, "pnl_trend_match_pct": 0.0},
            "hybrid_test_metrics": {"directional_accuracy_pct": 0.0, "trigger_match_pct": 0.0, "pnl_trend_match_pct": 0.0},
            "iterations": [],
        }

    metrics = _metrics_for_rows(admitted_all)
    coverage = (len(admitted_all) / max(1, len(preds_all))) if preds_all else 0.0
    worst = {"directional_accuracy_pct": 100.0, "trigger_match_pct": 100.0, "pnl_trend_match_pct": 100.0}
    for w in windows:
        wm = w.get("metrics", {}) if isinstance(w.get("metrics", {}), dict) else {}
        for key in list(worst.keys()):
            worst[key] = min(worst[key], _f(wm.get(key, 0.0), 0.0))
    if not windows:
        worst = {"directional_accuracy_pct": 0.0, "trigger_match_pct": 0.0, "pnl_trend_match_pct": 0.0}
    ci = {
        "directional_accuracy_pct": _bootstrap_ci(admitted_all, "directional_accuracy_pct"),
        "trigger_match_pct": _bootstrap_ci(admitted_all, "trigger_match_pct"),
        "pnl_trend_match_pct": _bootstrap_ci(admitted_all, "pnl_trend_match_pct"),
    }

    by_trigger: Dict[str, List[Dict[str, Any]]] = {}
    by_regime: Dict[str, List[Dict[str, Any]]] = {}
    for r in admitted_all:
        by_trigger.setdefault(_s(r.get("actual_exit_trigger", "Unknown")) or "Unknown", []).append(r)
        by_regime.setdefault(_s(r.get("regime", "unknown")) or "unknown", []).append(r)
    seg_trigger = {k: _metrics_for_rows(v) for k, v in by_trigger.items()}
    seg_regime = {k: _metrics_for_rows(v) for k, v in by_regime.items()}
    population_diag = _population_diagnostics(preds_all, admitted_all, abstained_all)
    crypto_diag = _crypto_replay_diagnostics(admitted_all, full_rows=preds_all, abstained_rows=abstained_all) if m == "crypto" else {}
    market_diag = _market_predictor_diagnostics(m, admitted_all, full_rows=preds_all, abstained_rows=abstained_all) if m in {"stocks", "forex"} else {}
    selected_variant_distribution = _count_by(
        [w.get("safe_selection", {}) for w in windows if isinstance(w.get("safe_selection", {}), dict)],
        lambda s: _s(s.get("selected_predictor_variant", "")) or "unknown",
    )
    final_scoring_variant_name = (
        next(iter(selected_variant_distribution.keys()))
        if len(selected_variant_distribution) == 1
        else "mixed_selected_variants_aggregate"
    )
    if crypto_diag:
        crypto_diag["population_diagnostics"] = population_diag
        crypto_diag["crypto_label_feature_alignment_diagnostics"] = _crypto_label_feature_alignment_diagnostics(preds_all)
        crypto_diag["selected_crypto_admission_policy"] = policy_snapshots[-1] if policy_snapshots else {}
        crypto_diag["selected_policy_validation_quality"] = validation_policy_quality[-1] if validation_policy_quality else {}
        crypto_diag["crypto_policy_frontier"] = (
            validation_policy_quality[-1].get("policy_frontier", []) if validation_policy_quality else []
        )
        if _s(crypto_diag.get("crypto_predictor_source_mode", "")) == "" and admitted_all:
            crypto_diag["crypto_predictor_source_mode"] = _count_by(
                admitted_all,
                lambda r: _s(r.get("predictor_source_mode", "")) or "legacy",
            )
        if admitted_all and all(_is_historical_strategy_replay_row(r) for r in preds_all):
            crypto_diag["historical_replay_predictor_diagnostics"] = {
                "selected_source_type": "historical_strategy_replay",
                "actual_trigger_distribution": crypto_diag.get("actual_trigger_distribution", {}),
                "predicted_trigger_distribution": crypto_diag.get("predicted_trigger_distribution", {}),
                "actual_direction_distribution": crypto_diag.get("actual_direction_distribution", {}),
                "predicted_direction_distribution": crypto_diag.get("predicted_direction_distribution", {}),
                "trigger_confusion_matrix": crypto_diag.get("trigger_confusion_matrix", {}),
                "direction_confusion_matrix": crypto_diag.get("direction_confusion_matrix", {}),
            }
            crypto_diag["historical_replay_admission_policy"] = policy_snapshots[-1] if policy_snapshots else {}
            crypto_diag["historical_replay_policy_frontier"] = (
                validation_policy_quality[-1].get("policy_frontier", []) if validation_policy_quality else []
            )
            crypto_diag["historical_replay_source_metrics"] = {
                "full_universe_metrics": population_diag.get("full_universe_metrics", {}),
                "admitted_metrics": population_diag.get("admitted_metrics", {}),
                "abstained_metrics": population_diag.get("abstained_metrics", {}),
            }
        latest_selection = safe_selection_snapshots[-1] if safe_selection_snapshots else {}
        latest_candidate_evals = latest_selection.get("candidate_evaluations", []) if isinstance(latest_selection.get("candidate_evaluations", []), list) else []
        latest_candidate_variant = "label_compatible_v2" if any(_s(ev.get("variant", "")) == "label_compatible_v2" for ev in latest_candidate_evals) else ""
        latest_candidate_eval = next(
            (
                ev
                for ev in latest_candidate_evals
                if _s(ev.get("variant", "")) == latest_candidate_variant
            ),
            {},
        ) if latest_candidate_variant else {}
        latest_candidate_summary = latest_candidate_eval.get("summary", {}) if isinstance(latest_candidate_eval.get("summary", {}), dict) else {}
        latest_candidate_rows = variant_predictions.get(latest_candidate_variant, []) if latest_candidate_variant else []
        latest_candidate_admitted = (
            variant_scored.get(latest_candidate_variant, {}).get("admitted", [])
            if latest_candidate_variant and isinstance(variant_scored.get(latest_candidate_variant, {}), dict)
            else []
        )
        selected_latest_rows = selected_preds if isinstance(selected_preds, list) else []
        selected_latest_admitted = selected_scored.get("admitted", []) if isinstance(selected_scored.get("admitted", []), list) else []
        divergence_reason = "candidate_and_final_metrics_align"
        if _s(latest_selection.get("selected_predictor_variant", "")) != latest_candidate_variant and latest_candidate_variant:
            divergence_reason = "latest_candidate_not_selected_due_to_safe_selection_guardrails"
        elif final_scoring_variant_name == "mixed_selected_variants_aggregate":
            divergence_reason = "final_metrics_are_aggregate_across_windows_with_mixed_selected_variants"
        elif int(len(admitted_all)) != int(len(selected_latest_admitted)) or int(len(preds_all)) != int(len(selected_latest_rows)):
            divergence_reason = "final_metrics_are_aggregate_across_all_selected_windows_not_latest_window_only"
        elif _s(latest_selection.get("selected_predictor_variant", "")) != _s(final_scoring_variant_name):
            divergence_reason = "final_metrics_use_selected_window_aggregate_not_latest_selected_variant_only"
        crypto_diag["candidate_vs_final_reconciliation"] = {
            "candidate_variant_name": latest_candidate_variant or "none",
            "selected_variant_name": _s(latest_selection.get("selected_predictor_variant", "")) or "none",
            "final_scoring_variant_name": final_scoring_variant_name,
            "final_metrics_basis": "selected_admitted_rows_aggregate",
            "candidate_full_rows_count": int(len(latest_candidate_rows)),
            "candidate_admitted_rows_count": int(len(latest_candidate_admitted)),
            "final_full_rows_count": int(len(preds_all)),
            "final_admitted_rows_count": int(len(admitted_all)),
            "latest_selected_window_full_rows_count": int(len(selected_latest_rows)),
            "latest_selected_window_admitted_rows_count": int(len(selected_latest_admitted)),
            "overlap_count_between_candidate_and_final_admitted": _row_overlap_count(latest_candidate_admitted, admitted_all),
            "candidate_metrics": latest_candidate_summary.get("metrics", {}),
            "candidate_full_metrics": latest_candidate_summary.get("full_metrics", {}),
            "final_metrics": metrics,
            "final_full_metrics": population_diag.get("full_universe_metrics", {}),
            "latest_selected_window_metrics": selected_scored.get("metrics", {}) if isinstance(selected_scored.get("metrics", {}), dict) else {},
            "latest_selected_window_full_metrics": selected_scored.get("full_metrics", {}) if isinstance(selected_scored.get("full_metrics", {}), dict) else {},
            "selected_variant_distribution": selected_variant_distribution,
            "reason_final_metrics_differ_from_candidate_metrics": divergence_reason,
            "final_metrics_using": {
                "variant": final_scoring_variant_name,
                "row_set": "admitted_rows",
                "aggregation_scope": "all_selected_windows",
                "fallback_to_baseline": bool(latest_selection.get("fallback_to_baseline", False)),
            },
            "candidate_guardrail_failures": list(latest_candidate_eval.get("guardrail_failures", []) or []),
        }
        crypto_diag["admission_reason_counts"] = _count_by(
            admitted_all + abstained_all,
            lambda r: _s(r.get("admission_reason", "")) or "unknown",
        )
        crypto_diag["admitted_predicted_trigger_counts"] = _count_by(
            admitted_all,
            lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown",
        )
        crypto_diag["abstained_predicted_trigger_counts"] = _count_by(
            abstained_all,
            lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown",
        )
        crypto_diag["confidence_bucket_admission_counts"] = {
            "admitted": _count_by(admitted_all, lambda r: _bucket_probability(_f(r.get("predicted_confidence", 0.0), 0.0))),
            "abstained": _count_by(abstained_all, lambda r: _bucket_probability(_f(r.get("predicted_confidence", 0.0), 0.0))),
        }
        crypto_diag["trigger_margin_bucket_admission_counts"] = {
            "admitted": _count_by(admitted_all, lambda r: _bucket_margin_value(_f(r.get("trigger_margin", 0.0), 0.0))),
            "abstained": _count_by(abstained_all, lambda r: _bucket_margin_value(_f(r.get("trigger_margin", 0.0), 0.0))),
        }
        crypto_diag["direction_margin_bucket_admission_counts"] = {
            "admitted": _count_by(admitted_all, lambda r: _bucket_margin_value(_f(r.get("direction_margin", 0.0), 0.0))),
            "abstained": _count_by(abstained_all, lambda r: _bucket_margin_value(_f(r.get("direction_margin", 0.0), 0.0))),
        }
        crypto_diag["manual_separability_audit"] = {
            "full_universe": _manual_prediction_diagnostics(preds_all),
            "admitted": _manual_prediction_diagnostics(admitted_all),
        }
        best_frontier = validation_policy_quality[-1].get("policy_frontier", []) if validation_policy_quality else []
        if best_frontier:
            best_row = best_frontier[0]
            if (
                _f(best_row.get("trigger_match_pct", 0.0), 0.0) < 90.0
                or int(best_row.get("admitted_trades", 0) or 0) < 40
                or _f(best_row.get("worst_window_metrics", {}).get("trigger_match_pct", 0.0), 0.0) < 85.0
            ):
                crypto_diag["next_required_features"] = _next_required_features()
                crypto_diag["stabilization_conclusion"] = (
                    "Current closed-trade-only replay features are insufficient to achieve both >=90% admitted quality "
                    "and >=40 admitted rows / 40% coverage honestly."
                )

    payload = {
        "status": "ok",
        "summary": (
            f"{m} synthetic replay: admitted {len(admitted_all)}/{len(preds_all)} "
            f"({coverage * 100.0:.1f}% coverage) with directional {_f(metrics.get('directional_accuracy_pct', 0.0), 0.0):.2f}%, "
            f"trigger {_f(metrics.get('trigger_match_pct', 0.0), 0.0):.2f}%, pnl-trend {_f(metrics.get('pnl_trend_match_pct', 0.0), 0.0):.2f}%."
        ),
        "meta": {
            "market": m,
            "closed_trades_total": int(len(rows)),
            "train_trades": int(max(w.get("train_rows", 0) for w in windows)),
            "test_trades": int(len(admitted_all)),
            "full_test_trades": int(len(preds_all)),
            "abstained_test_trades": int(len(abstained_all)),
            "admitted_test_trades": int(len(admitted_all)),
            "walkforward_windows": int(len(windows)),
            "deterministic_mode_enabled": bool(det_enabled),
            "rows_hash": _stable_rows_hash(rows),
        },
        "abstain_policy": {
            "threshold_median": round(_median([float(v) for v in thr_values], default=0.6), 4),
            "test_coverage": round(float(coverage), 6),
            "selected_crypto_policy": policy_snapshots[-1] if policy_snapshots else {},
            "selected_crypto_policy_name": (
                validation_policy_quality[-1].get("policy_name", "") if validation_policy_quality else ""
            ),
        },
        "best_test_metrics": metrics,
        "hybrid_test_metrics": metrics,
        "population_diagnostics": population_diag,
        "safe_selection_diagnostics": {
            "windows": safe_selection_snapshots,
            "latest": safe_selection_snapshots[-1] if safe_selection_snapshots else {},
        },
        "worst_window_metrics": worst,
        "confidence_intervals": ci,
        "walkforward_windows": windows,
        "segmented_by_actual_trigger": seg_trigger,
        "segmented_by_regime": seg_regime,
        "iterations": [{"iteration": 1, "test_predictions": admitted_all}],
    }
    if crypto_diag:
        crypto_diag["safe_selection"] = latest_selection
        crypto_diag["exit_shape_predictive_mode"] = (
            "active" if _s(latest_selection.get("selected_predictor_variant", "")) in {"label_compatible_v2", "label_compatible_v3_sequence"} else "diagnostic_only"
        )
        label_v2_eval = {}
        for ev in latest_selection.get("candidate_evaluations", []) if isinstance(latest_selection.get("candidate_evaluations", []), list) else []:
            if _s(ev.get("variant", "")) == "label_compatible_v2":
                label_v2_eval = ev
                break
        if label_v2_eval:
            summary = label_v2_eval.get("summary", {}) if isinstance(label_v2_eval.get("summary", {}), dict) else {}
            crypto_diag["label_compatible_v2_rejection_diagnostics"] = summary.get("rejection_diagnostics", {})
            crypto_diag["label_compatible_v2_near_miss_rows_count"] = int(summary.get("near_miss_rows_count", 0) or 0)
            crypto_diag["label_compatible_v2_correct_rejected_count"] = int(summary.get("correct_rejected_count", 0) or 0)
            crypto_diag["label_compatible_v2_incorrect_admitted_count"] = int(summary.get("incorrect_admitted_count", 0) or 0)
            crypto_diag["crypto_clean_expansion_attempted"] = bool(summary.get("crypto_clean_expansion_attempted", False))
            crypto_diag["crypto_clean_expansion_rows_added"] = int(summary.get("crypto_clean_expansion_rows_added", 0) or 0)
            crypto_diag["crypto_clean_expansion_correct_rejected_added_estimate"] = int(
                summary.get("crypto_clean_expansion_correct_rejected_added_estimate", 0) or 0
            )
            crypto_diag["crypto_clean_expansion_guardrails_passed"] = bool(
                summary.get("crypto_clean_expansion_guardrails_passed", False)
            )
            crypto_diag["crypto_clean_expansion_rejection_reason"] = _s(
                summary.get("crypto_clean_expansion_rejection_reason", "")
            )
        payload["crypto_classifier_diagnostics"] = crypto_diag
    if market_diag:
        if m == "stocks":
            payload["stock_predictor_diagnostics"] = market_diag
            payload["stock_predictor_mode"] = market_diag.get("predictor_mode", {})
            payload["stock_policy_frontier"] = validation_policy_quality[-1].get("policy_frontier", []) if validation_policy_quality else []
            payload["stock_safe_selection"] = safe_selection_snapshots[-1] if safe_selection_snapshots else {}
        if m == "forex":
            payload["forex_predictor_diagnostics"] = market_diag
            payload["forex_predictor_mode"] = market_diag.get("predictor_mode", {})
            payload["forex_policy_frontier"] = validation_policy_quality[-1].get("policy_frontier", []) if validation_policy_quality else []
            payload["forex_safe_selection"] = safe_selection_snapshots[-1] if safe_selection_snapshots else {}
    if det_enabled and m == "crypto":
        payload["crypto_deterministic_evaluation"] = {
            "enabled": True,
            "start_ts": int(_f(det_cfg.get("start_ts", 0), 0.0)),
            "cutoff_ts": int(_f(det_cfg.get("cutoff_ts", 0), 0.0)),
            "symbols_locked": bool(det_cfg.get("symbols_locked", False)),
            "symbol_list_hash": _s(det_cfg.get("symbol_list_hash", "")),
            "cache_frozen": bool(det_cfg.get("cache_frozen", False)),
            "rows_hash": _stable_rows_hash(rows),
        }
    return payload


def _metrics_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n <= 0:
        return {
            "trades": 0.0,
            "directional_accuracy_pct": 0.0,
            "trigger_match_pct": 0.0,
            "pnl_trend_match_pct": 0.0,
        }
    d_hits = 0
    t_hits = 0
    t_scored = 0
    p_hits = 0
    for r in rows:
        if _s(r.get("predicted_direction", "")).lower() == _s(r.get("actual_direction", "")).lower():
            d_hits += 1
        act_trig = _s(r.get("actual_exit_trigger", "")) or "Unknown"
        pred_trig = _s(r.get("predicted_exit_trigger", "")) or "Unknown"
        if act_trig != "Unknown":
            t_scored += 1
            if pred_trig == act_trig:
                t_hits += 1
        entry = _f(r.get("entry_price", 0.0), 0.0)
        act = _f(r.get("actual_exit_price", r.get("exit_price", 0.0)), 0.0)
        pred = _f(r.get("predicted_exit_price", 0.0), 0.0)
        if entry > 0.0:
            ar = (act - entry) / entry
            pr = (pred - entry) / entry
            eps = 1e-9
            if (abs(ar) <= eps and abs(pr) <= eps) or (ar > eps and pr > eps) or (ar < -eps and pr < -eps):
                p_hits += 1
    return {
        "trades": float(n),
        "directional_accuracy_pct": round(100.0 * d_hits / max(1, n), 4),
        "trigger_match_pct": round(100.0 * t_hits / max(1, t_scored), 4),
        "trigger_scored_trades": float(t_scored),
        "trigger_coverage_pct": round(100.0 * t_scored / max(1, n), 4),
        "pnl_trend_match_pct": round(100.0 * p_hits / max(1, n), 4),
    }


def _bucket_hold_hours(value: float) -> str:
    v = max(0.0, float(value))
    if v < 1.0:
        return "<1h"
    if v < 4.0:
        return "1-4h"
    if v < 12.0:
        return "4-12h"
    if v < 24.0:
        return "12-24h"
    return "24h+"


def _bucket_return_mag_pct(entry_price: float, exit_price: float) -> str:
    if entry_price <= 0.0:
        return "unknown"
    mag = abs(((exit_price / entry_price) - 1.0) * 100.0)
    if mag < 0.5:
        return "<0.5%"
    if mag < 1.5:
        return "0.5-1.5%"
    if mag < 3.0:
        return "1.5-3.0%"
    return "3.0%+"


def _confusion_matrix(rows: List[Dict[str, Any]], actual_key: str, pred_key: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        actual = _s(row.get(actual_key, "Unknown")) or "Unknown"
        pred = _s(row.get(pred_key, "Unknown")) or "Unknown"
        if actual not in out:
            out[actual] = {}
        out[actual][pred] = int(out[actual].get(pred, 0) + 1)
    return out


def _top_counter(values: Dict[str, int], limit: int = 12) -> Dict[str, int]:
    items = sorted(values.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return {str(k): int(v) for k, v in items[:limit]}


def _crypto_replay_diagnostics(
    rows: List[Dict[str, Any]],
    *,
    full_rows: Optional[List[Dict[str, Any]]] = None,
    abstained_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not rows:
        return {}
    full_population = list(full_rows or rows)
    abstained_population = list(abstained_rows or [])
    dir_conf = _confusion_matrix(rows, "actual_direction", "predicted_direction")
    trig_conf = _confusion_matrix(rows, "actual_exit_trigger", "predicted_exit_trigger")
    miss_symbol: Dict[str, int] = {}
    miss_regime: Dict[str, int] = {}
    miss_actual_trigger: Dict[str, int] = {}
    miss_pred_trigger: Dict[str, int] = {}
    miss_rule_reason: Dict[str, int] = {}
    miss_strategy_adapter: Dict[str, int] = {}
    miss_signal_side: Dict[str, int] = {}
    miss_hold: Dict[str, int] = {}
    miss_ret: Dict[str, int] = {}
    miss_trend_bucket: Dict[str, int] = {}
    miss_vol_bucket: Dict[str, int] = {}
    miss_signal_margin_bucket: Dict[str, int] = {}
    miss_recent3_bucket: Dict[str, int] = {}
    miss_recent6_bucket: Dict[str, int] = {}
    miss_recent12_bucket: Dict[str, int] = {}
    miss_recent24_bucket: Dict[str, int] = {}
    actual_up_by_symbol: Dict[str, int] = {}
    actual_up_pred_down_by_symbol: Dict[str, int] = {}
    actual_up_by_trigger: Dict[str, int] = {}
    actual_up_by_pred_trigger: Dict[str, int] = {}
    actual_up_by_signal_side: Dict[str, int] = {}
    actual_up_by_trend_bucket: Dict[str, int] = {}
    actual_up_by_recent3_bucket: Dict[str, int] = {}
    actual_up_by_recent6_bucket: Dict[str, int] = {}
    actual_up_by_recent12_bucket: Dict[str, int] = {}
    actual_up_by_recent24_bucket: Dict[str, int] = {}
    actual_up_by_signal_margin_bucket: Dict[str, int] = {}
    actual_up_by_up_boundary_bucket: Dict[str, int] = {}
    actual_up_by_down_boundary_bucket: Dict[str, int] = {}
    take_profit_rows: List[Dict[str, Any]] = []
    trailing_rows: List[Dict[str, Any]] = []
    risk_cut_as_stale: List[Dict[str, Any]] = []
    take_profit_predicted = 0
    take_profit_correct = 0
    risk_cut_predicted = 0
    risk_cut_correct = 0
    upside_recovery_counts: Dict[str, int] = {}
    conf_hits: List[float] = []
    conf_miss: List[float] = []
    trig_margin_hits: List[float] = []
    trig_margin_miss: List[float] = []
    dir_margin_hits: List[float] = []
    dir_margin_miss: List[float] = []
    uptrail_downstale: List[Dict[str, Any]] = []
    manual_mispreds: List[Dict[str, Any]] = []
    stale_vs_trailing_examples: List[Dict[str, Any]] = []
    stale_vs_manual_examples: List[Dict[str, Any]] = []
    trigger_pair_misses: Dict[str, int] = {
        "Risk Cut->Trailing": 0,
        "Risk Cut->Stale Alignment": 0,
        "Trailing->Risk Cut": 0,
        "Trailing->Stale Alignment": 0,
        "Stale Alignment->Risk Cut": 0,
        "Stale Alignment->Trailing": 0,
    }
    miss_bars_in_trade: Dict[str, int] = {}
    miss_drawdown_from_peak: Dict[str, int] = {}
    miss_max_adverse_excursion: Dict[str, int] = {}
    miss_trailing_armed: Dict[str, int] = {}
    miss_risk_cut_touched: Dict[str, int] = {}
    miss_favorable_then_softened: Dict[str, int] = {}
    actual_risk_cut_by_sequence: Dict[str, int] = {}
    actual_trailing_by_sequence: Dict[str, int] = {}
    risk_cut_to_trailing_sequence: Dict[str, int] = {}
    trailing_to_risk_cut_sequence: Dict[str, int] = {}
    miss_exit_close_position_bucket: Dict[str, int] = {}
    miss_peak_to_exit_velocity_bucket: Dict[str, int] = {}
    miss_bars_from_trailing_arm_to_risk_cut_bucket: Dict[str, int] = {}
    proto_support: Dict[str, Dict[str, int]] = {}
    manual_predicted = 0
    manual_correct = 0
    trailing_predicted = 0
    trailing_correct = 0
    stale_predicted = 0
    stale_correct = 0
    for row in rows:
        pred_trig = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
        act_trig = _s(row.get("actual_exit_trigger", "")) or "Unknown"
        hit = (
            _s(row.get("actual_direction", "")).lower() == _s(row.get("predicted_direction", "")).lower()
            and act_trig == pred_trig
        )
        conf = _f(row.get("predicted_confidence", 0.0), 0.0)
        trig_margin = _f(row.get("trigger_margin", 0.0), 0.0)
        dir_margin = _f(row.get("direction_margin", 0.0), 0.0)
        if _s(row.get("actual_direction", "")).lower() == "up":
            actual_up_by_symbol[_s(row.get("symbol", "")) or "UNKNOWN"] = int(actual_up_by_symbol.get(_s(row.get("symbol", "")) or "UNKNOWN", 0) + 1)
            actual_up_by_trigger[act_trig] = int(actual_up_by_trigger.get(act_trig, 0) + 1)
            actual_up_by_pred_trigger[pred_trig] = int(actual_up_by_pred_trigger.get(pred_trig, 0) + 1)
            actual_up_by_signal_side[_s(row.get("signal_side", "")) or "unknown"] = int(actual_up_by_signal_side.get(_s(row.get("signal_side", "")) or "unknown", 0) + 1)
            actual_up_by_trend_bucket[_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", 0.0), 0.0))] = int(actual_up_by_trend_bucket.get(_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", 0.0), 0.0)), 0) + 1)
            actual_up_by_recent3_bucket[_historical_replay_bucket("recent_return_3", _f(row.get("recent_return_3", 0.0), 0.0))] = int(actual_up_by_recent3_bucket.get(_historical_replay_bucket("recent_return_3", _f(row.get("recent_return_3", 0.0), 0.0)), 0) + 1)
            actual_up_by_recent6_bucket[_historical_replay_bucket("recent_return_6", _f(row.get("recent_return_6", 0.0), 0.0))] = int(actual_up_by_recent6_bucket.get(_historical_replay_bucket("recent_return_6", _f(row.get("recent_return_6", 0.0), 0.0)), 0) + 1)
            actual_up_by_recent12_bucket[_historical_replay_bucket("recent_return_12", _f(row.get("recent_return_12", 0.0), 0.0))] = int(actual_up_by_recent12_bucket.get(_historical_replay_bucket("recent_return_12", _f(row.get("recent_return_12", 0.0), 0.0)), 0) + 1)
            actual_up_by_recent24_bucket[_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", 0.0), 0.0))] = int(actual_up_by_recent24_bucket.get(_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", 0.0), 0.0)), 0) + 1)
            actual_up_by_signal_margin_bucket[_historical_replay_bucket("signal_margin", _f(row.get("signal_margin", 0.0), 0.0))] = int(actual_up_by_signal_margin_bucket.get(_historical_replay_bucket("signal_margin", _f(row.get("signal_margin", 0.0), 0.0)), 0) + 1)
            actual_up_by_up_boundary_bucket[_historical_replay_bucket("recent_return_3", _f(row.get("upside_boundary_distance_pct", 0.0), 0.0))] = int(actual_up_by_up_boundary_bucket.get(_historical_replay_bucket("recent_return_3", _f(row.get("upside_boundary_distance_pct", 0.0), 0.0)), 0) + 1)
            actual_up_by_down_boundary_bucket[_historical_replay_bucket("recent_return_3", _f(row.get("downside_boundary_distance_pct", 0.0), 0.0))] = int(actual_up_by_down_boundary_bucket.get(_historical_replay_bucket("recent_return_3", _f(row.get("downside_boundary_distance_pct", 0.0), 0.0)), 0) + 1)
            if _s(row.get("predicted_direction", "")).lower() == "down":
                actual_up_pred_down_by_symbol[_s(row.get("symbol", "")) or "UNKNOWN"] = int(actual_up_pred_down_by_symbol.get(_s(row.get("symbol", "")) or "UNKNOWN", 0) + 1)
        if act_trig == "Take Profit" and len(take_profit_rows) < 12:
            take_profit_rows.append(
                {
                    "symbol": _s(row.get("symbol", "")),
                    "predicted_trigger": pred_trig,
                    "signal_margin": round(_f(row.get("signal_margin", 0.0), 0.0), 6),
                    "trend_momentum_score": round(_f(row.get("trend_momentum_score", 0.0), 0.0), 6),
                    "recent_return_3": round(_f(row.get("recent_return_3", 0.0), 0.0), 6),
                    "recent_return_6": round(_f(row.get("recent_return_6", 0.0), 0.0), 6),
                    "recent_return_12": round(_f(row.get("recent_return_12", 0.0), 0.0), 6),
                    "recent_return_24": round(_f(row.get("recent_return_24", 0.0), 0.0), 6),
                    "recent_volatility": round(_f(row.get("recent_volatility", 0.0), 0.0), 6),
                    "upside_score": round(_f(row.get("upside_score", 0.0), 0.0), 6),
                    "take_profit_score": round(_f(row.get("take_profit_score", 0.0), 0.0), 6),
                    "stale_score": round(_f(row.get("stale_score", 0.0), 0.0), 6),
                    "risk_cut_score": round(_f(row.get("risk_cut_score", 0.0), 0.0), 6),
                }
            )
        if act_trig == "Trailing" and len(trailing_rows) < 18:
            trailing_rows.append(
                {
                    "symbol": _s(row.get("symbol", "")),
                    "predicted_trigger": pred_trig,
                    "predicted_direction": _s(row.get("predicted_direction", "")),
                    "upside_score": round(_f(row.get("upside_score", 0.0), 0.0), 6),
                    "signal_margin": round(_f(row.get("signal_margin", 0.0), 0.0), 6),
                    "trend_momentum_score": round(_f(row.get("trend_momentum_score", 0.0), 0.0), 6),
                    "recent_volatility": round(_f(row.get("recent_volatility", 0.0), 0.0), 6),
                    "trailing_score": round(_f(row.get("trailing_score", 0.0), 0.0), 6),
                    "stale_score": round(_f(row.get("stale_score", 0.0), 0.0), 6),
                    "risk_cut_score": round(_f(row.get("risk_cut_score", 0.0), 0.0), 6),
                }
            )
        if act_trig == "Risk Cut" and pred_trig == "Stale Alignment" and len(risk_cut_as_stale) < 18:
            risk_cut_as_stale.append(
                {
                    "symbol": _s(row.get("symbol", "")),
                    "trend_bucket": _historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", 0.0), 0.0)),
                    "signal_margin_bucket": _historical_replay_bucket("signal_margin", _f(row.get("signal_margin", 0.0), 0.0)),
                    "recent_volatility_bucket": _historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0)),
                    "upside_score": round(_f(row.get("upside_score", 0.0), 0.0), 6),
                    "downside_score": round(_f(row.get("downside_score", 0.0), 0.0), 6),
                    "risk_cut_score": round(_f(row.get("risk_cut_score", 0.0), 0.0), 6),
                    "stale_score": round(_f(row.get("stale_score", 0.0), 0.0), 6),
                }
            )
        if hit:
            conf_hits.append(conf)
            trig_margin_hits.append(trig_margin)
            dir_margin_hits.append(dir_margin)
        else:
            conf_miss.append(conf)
            trig_margin_miss.append(trig_margin)
            dir_margin_miss.append(dir_margin)
            sym = _s(row.get("symbol", "")) or "UNKNOWN"
            reg = _s(row.get("regime", "")) or "unknown"
            atr = act_trig
            ptr = pred_trig
            miss_symbol[sym] = int(miss_symbol.get(sym, 0) + 1)
            miss_regime[reg] = int(miss_regime.get(reg, 0) + 1)
            miss_actual_trigger[atr] = int(miss_actual_trigger.get(atr, 0) + 1)
            miss_pred_trigger[ptr] = int(miss_pred_trigger.get(ptr, 0) + 1)
            miss_rule_reason[_s(row.get("raw_rule_reason", "")) or "unknown"] = int(
                miss_rule_reason.get(_s(row.get("raw_rule_reason", "")) or "unknown", 0) + 1
            )
            miss_strategy_adapter[_s(row.get("strategy_adapter_used", "")) or "unknown"] = int(
                miss_strategy_adapter.get(_s(row.get("strategy_adapter_used", "")) or "unknown", 0) + 1
            )
            miss_signal_side[_s(row.get("signal_side", "")) or "unknown"] = int(
                miss_signal_side.get(_s(row.get("signal_side", "")) or "unknown", 0) + 1
            )
            miss_hold[_bucket_hold_hours(_f(row.get("hold_hours", row.get("actual_hold_hours", 0.0)), 0.0))] = int(
                miss_hold.get(_bucket_hold_hours(_f(row.get("hold_hours", row.get("actual_hold_hours", 0.0)), 0.0)), 0) + 1
            )
            miss_ret[_bucket_return_mag_pct(_f(row.get("entry_price", 0.0), 0.0), _f(row.get("exit_price", row.get("actual_exit_price", 0.0)), 0.0))] = int(
                miss_ret.get(
                    _bucket_return_mag_pct(
                        _f(row.get("entry_price", 0.0), 0.0),
                        _f(row.get("exit_price", row.get("actual_exit_price", 0.0)), 0.0),
                    ),
                    0,
                )
                + 1
            )
            miss_trend_bucket[_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", 0.0), 0.0))] = int(
                miss_trend_bucket.get(_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", 0.0), 0.0)), 0) + 1
            )
            miss_vol_bucket[_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0))] = int(
                miss_vol_bucket.get(_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0)), 0) + 1
            )
            miss_signal_margin_bucket[_historical_replay_bucket("signal_margin", _f(row.get("signal_margin", 0.0), 0.0))] = int(
                miss_signal_margin_bucket.get(_historical_replay_bucket("signal_margin", _f(row.get("signal_margin", 0.0), 0.0)), 0) + 1
            )
            miss_recent3_bucket[_historical_replay_bucket("recent_return_3", _f(row.get("recent_return_3", 0.0), 0.0))] = int(
                miss_recent3_bucket.get(_historical_replay_bucket("recent_return_3", _f(row.get("recent_return_3", 0.0), 0.0)), 0) + 1
            )
            miss_recent6_bucket[_historical_replay_bucket("recent_return_6", _f(row.get("recent_return_6", 0.0), 0.0))] = int(
                miss_recent6_bucket.get(_historical_replay_bucket("recent_return_6", _f(row.get("recent_return_6", 0.0), 0.0)), 0) + 1
            )
            miss_recent12_bucket[_historical_replay_bucket("recent_return_12", _f(row.get("recent_return_12", 0.0), 0.0))] = int(
                miss_recent12_bucket.get(_historical_replay_bucket("recent_return_12", _f(row.get("recent_return_12", 0.0), 0.0)), 0) + 1
            )
            miss_recent24_bucket[_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", 0.0), 0.0))] = int(
                miss_recent24_bucket.get(_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", 0.0), 0.0)), 0) + 1
            )
            pair_key = f"{act_trig}->{pred_trig}"
            if pair_key in trigger_pair_misses:
                trigger_pair_misses[pair_key] = int(trigger_pair_misses.get(pair_key, 0) + 1)
            miss_bars_in_trade[_bucket_hold_hours(_f(row.get("bars_in_trade", 0.0), 0.0))] = int(
                miss_bars_in_trade.get(_bucket_hold_hours(_f(row.get("bars_in_trade", 0.0), 0.0)), 0) + 1
            )
            miss_drawdown_from_peak[_bucket_margin_value(abs(_f(row.get("drawdown_from_peak_pct", 0.0), 0.0)))] = int(
                miss_drawdown_from_peak.get(_bucket_margin_value(abs(_f(row.get("drawdown_from_peak_pct", 0.0), 0.0))), 0) + 1
            )
            miss_max_adverse_excursion[_bucket_margin_value(abs(_f(row.get("max_adverse_excursion_pct", 0.0), 0.0)))] = int(
                miss_max_adverse_excursion.get(_bucket_margin_value(abs(_f(row.get("max_adverse_excursion_pct", 0.0), 0.0))), 0) + 1
            )
            miss_trailing_armed[str(bool(row.get("trailing_armed", False))).lower()] = int(
                miss_trailing_armed.get(str(bool(row.get("trailing_armed", False))).lower(), 0) + 1
            )
            miss_risk_cut_touched[str(bool(row.get("risk_cut_touched", False))).lower()] = int(
                miss_risk_cut_touched.get(str(bool(row.get("risk_cut_touched", False))).lower(), 0) + 1
            )
            miss_favorable_then_softened[str(bool(row.get("favorable_then_softened_flag", False))).lower()] = int(
                miss_favorable_then_softened.get(str(bool(row.get("favorable_then_softened_flag", False))).lower(), 0) + 1
            )
            sequence_bucket = (
                "risk_before_trailing" if bool(row.get("risk_cut_before_trailing", False))
                else "trailing_before_risk" if bool(row.get("trailing_before_risk_cut", False))
                else "same_candle" if bool(row.get("same_candle_risk_and_trailing", False))
                else "risk_after_trailing" if bool(row.get("risk_cut_after_trailing_arm", False))
                else "other"
            )
            if act_trig == "Risk Cut" and pred_trig == "Trailing":
                risk_cut_to_trailing_sequence[sequence_bucket] = int(risk_cut_to_trailing_sequence.get(sequence_bucket, 0) + 1)
            if act_trig == "Trailing" and pred_trig == "Risk Cut":
                trailing_to_risk_cut_sequence[sequence_bucket] = int(trailing_to_risk_cut_sequence.get(sequence_bucket, 0) + 1)
            miss_exit_close_position_bucket[_bucket_probability(_f(row.get("exit_close_position_in_candle_range", 0.5), 0.5))] = int(
                miss_exit_close_position_bucket.get(_bucket_probability(_f(row.get("exit_close_position_in_candle_range", 0.5), 0.5)), 0) + 1
            )
            miss_peak_to_exit_velocity_bucket[_bucket_margin_value(abs(_f(row.get("peak_to_exit_velocity_pct_per_bar", 0.0), 0.0)))] = int(
                miss_peak_to_exit_velocity_bucket.get(_bucket_margin_value(abs(_f(row.get("peak_to_exit_velocity_pct_per_bar", 0.0), 0.0))), 0) + 1
            )
            miss_bars_from_trailing_arm_to_risk_cut_bucket[_bucket_margin_value(_f(row.get("bars_from_trailing_arm_to_risk_cut", -1.0), -1.0))] = int(
                miss_bars_from_trailing_arm_to_risk_cut_bucket.get(_bucket_margin_value(_f(row.get("bars_from_trailing_arm_to_risk_cut", -1.0), -1.0)), 0) + 1
            )
        if act_trig == "Risk Cut":
            sequence_bucket = (
                "risk_before_trailing" if bool(row.get("risk_cut_before_trailing", False))
                else "trailing_before_risk" if bool(row.get("trailing_before_risk_cut", False))
                else "same_candle" if bool(row.get("same_candle_risk_and_trailing", False))
                else "risk_after_trailing" if bool(row.get("risk_cut_after_trailing_arm", False))
                else "other"
            )
            actual_risk_cut_by_sequence[sequence_bucket] = int(actual_risk_cut_by_sequence.get(sequence_bucket, 0) + 1)
        if act_trig == "Trailing":
            sequence_bucket = (
                "risk_before_trailing" if bool(row.get("risk_cut_before_trailing", False))
                else "trailing_before_risk" if bool(row.get("trailing_before_risk_cut", False))
                else "same_candle" if bool(row.get("same_candle_risk_and_trailing", False))
                else "risk_after_trailing" if bool(row.get("risk_cut_after_trailing_arm", False))
                else "other"
            )
            actual_trailing_by_sequence[sequence_bucket] = int(actual_trailing_by_sequence.get(sequence_bucket, 0) + 1)
        if (
            _s(row.get("actual_direction", "")).lower() == "up"
            and _s(row.get("actual_exit_trigger", "")) == "Trailing"
            and _s(row.get("predicted_direction", "")).lower() == "down"
            and _s(row.get("predicted_exit_trigger", "")) == "Stale Alignment"
            and len(uptrail_downstale) < 15
        ):
            uptrail_downstale.append(
                {
                    "symbol": _s(row.get("symbol", "")),
                    "entry_ts": int(_f(row.get("entry_ts", 0.0), 0.0)),
                    "hold_hours": round(_f(row.get("hold_hours", row.get("actual_hold_hours", 0.0)), 0.0), 4),
                    "predicted_confidence": round(conf, 6),
                }
            )
        if act_trig == "Manual" and pred_trig in {"Stale Alignment", "Trailing"} and len(manual_mispreds) < 15:
            manual_mispreds.append(
                {
                    "symbol": _s(row.get("symbol", "")),
                    "entry_ts": int(_f(row.get("entry_ts", 0.0), 0.0)),
                    "predicted_exit_trigger": pred_trig,
                    "predicted_confidence": round(conf, 6),
                }
            )
        if pred_trig == "Manual":
            manual_predicted += 1
            if act_trig == "Manual":
                manual_correct += 1
        if pred_trig == "Trailing":
            trailing_predicted += 1
            if act_trig == "Trailing":
                trailing_correct += 1
        if pred_trig == "Take Profit":
            take_profit_predicted += 1
            if act_trig == "Take Profit":
                take_profit_correct += 1
        if pred_trig == "Risk Cut":
            risk_cut_predicted += 1
            if act_trig == "Risk Cut":
                risk_cut_correct += 1
        if pred_trig == "Stale Alignment":
            stale_predicted += 1
            if act_trig == "Stale Alignment":
                stale_correct += 1
        if bool(row.get("up_recovery_applied", False)):
            reason = _s(row.get("up_recovery_reason", "")) or "applied"
            upside_recovery_counts[reason] = int(upside_recovery_counts.get(reason, 0) + 1)
        if not proto_support and isinstance(row.get("trigger_prototype_support", {}), dict):
            for k, v in (row.get("trigger_prototype_support", {}) or {}).items():
                if isinstance(v, dict):
                    proto_support[str(k)] = {
                        "support_count": int(_f(v.get("support_count", 0), 0.0)),
                        "symbol_support_count": int(_f(v.get("symbol_support_count", 0), 0.0)),
                        "symbol_regime_support_count": int(_f(v.get("symbol_regime_support_count", 0), 0.0)),
                    }
        if len(stale_vs_trailing_examples) < 12 and isinstance(row.get("trigger_scores", {}), dict):
            ts = row.get("trigger_scores", {})
            if isinstance(ts, dict):
                stale_vs_trailing_examples.append(
                    {
                        "symbol": _s(row.get("symbol", "")),
                        "actual_trigger": act_trig,
                        "predicted_trigger": pred_trig,
                        "stale_score": round(_f(ts.get("Stale Alignment", 0.0), 0.0), 6),
                        "trailing_score": round(_f(ts.get("Trailing", 0.0), 0.0), 6),
                        "trigger_margin": round(trig_margin, 6),
                    }
                )
        if len(stale_vs_manual_examples) < 12 and isinstance(row.get("trigger_scores", {}), dict):
            ts = row.get("trigger_scores", {})
            if isinstance(ts, dict):
                stale_vs_manual_examples.append(
                    {
                        "symbol": _s(row.get("symbol", "")),
                        "actual_trigger": act_trig,
                        "predicted_trigger": pred_trig,
                        "stale_score": round(_f(ts.get("Stale Alignment", 0.0), 0.0), 6),
                        "manual_score": round(_f(ts.get("Manual", 0.0), 0.0), 6),
                        "trigger_margin": round(trig_margin, 6),
                    }
                )
    return {
        "direction_confusion_matrix": dir_conf,
        "trigger_confusion_matrix": trig_conf,
        "actual_trigger_distribution": _count_by(full_population, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
        "predicted_trigger_distribution": _count_by(full_population, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
        "actual_direction_distribution": _count_by(full_population, lambda r: _s(r.get("actual_direction", "")).lower() or "unknown"),
        "predicted_direction_distribution": _count_by(full_population, lambda r: _s(r.get("predicted_direction", "")).lower() or "unknown"),
        "trigger_class_prototypes_summary": proto_support,
        "class_precision": {
            "Manual": _trigger_precision(rows, "Manual"),
            "Trailing": _trigger_precision(rows, "Trailing"),
            "Stale Alignment": _trigger_precision(rows, "Stale Alignment"),
            "Risk Cut": _trigger_precision(rows, "Risk Cut"),
            "Take Profit": _trigger_precision(rows, "Take Profit"),
        },
        "misses_by_symbol": _top_counter(miss_symbol),
        "misses_by_regime": _top_counter(miss_regime),
        "misses_by_actual_trigger": _top_counter(miss_actual_trigger),
        "misses_by_trigger": _top_counter(miss_actual_trigger),
        "misses_by_predicted_trigger": _top_counter(miss_pred_trigger),
        "misses_by_rule_reason": _top_counter(miss_rule_reason),
        "misses_by_strategy_adapter_used": _top_counter(miss_strategy_adapter),
        "misses_by_signal_side": _top_counter(miss_signal_side),
        "misses_by_hold_bucket": _top_counter(miss_hold),
        "misses_by_return_magnitude_bucket": _top_counter(miss_ret),
        "misses_by_trend_momentum_bucket": _top_counter(miss_trend_bucket),
        "misses_by_recent_volatility_bucket": _top_counter(miss_vol_bucket),
        "misses_by_signal_margin_bucket": _top_counter(miss_signal_margin_bucket),
        "misses_by_recent_return_3_bucket": _top_counter(miss_recent3_bucket),
        "misses_by_recent_return_6_bucket": _top_counter(miss_recent6_bucket),
        "misses_by_recent_return_12_bucket": _top_counter(miss_recent12_bucket),
        "misses_by_recent_return_24_bucket": _top_counter(miss_recent24_bucket),
        "trigger_pair_miss_counts": trigger_pair_misses,
        "trigger_miss_feature_diagnostics": {
            "misses_by_bars_in_trade": _top_counter(miss_bars_in_trade),
            "misses_by_drawdown_from_peak_pct": _top_counter(miss_drawdown_from_peak),
            "misses_by_max_adverse_excursion_pct": _top_counter(miss_max_adverse_excursion),
            "misses_by_trailing_armed": _top_counter(miss_trailing_armed),
            "misses_by_risk_cut_touched": _top_counter(miss_risk_cut_touched),
            "misses_by_favorable_then_softened_flag": _top_counter(miss_favorable_then_softened),
            "misses_by_exit_close_position_in_candle_range": _top_counter(miss_exit_close_position_bucket),
            "misses_by_peak_to_exit_velocity_pct_per_bar": _top_counter(miss_peak_to_exit_velocity_bucket),
            "misses_by_bars_from_trailing_arm_to_risk_cut": _top_counter(miss_bars_from_trailing_arm_to_risk_cut_bucket),
        },
        "risk_trailing_sequence_diagnostics": {
            "actual_risk_cut_by_sequence": _top_counter(actual_risk_cut_by_sequence),
            "actual_trailing_by_sequence": _top_counter(actual_trailing_by_sequence),
            "risk_cut_predicted_trailing_by_sequence": _top_counter(risk_cut_to_trailing_sequence),
            "trailing_predicted_risk_cut_by_sequence": _top_counter(trailing_to_risk_cut_sequence),
        },
        "confidence_distribution": _confidence_summary(rows),
        "full_universe_confidence_distribution": _confidence_summary(full_population),
        "abstained_confidence_distribution": _confidence_summary(abstained_population),
        "margin_summary": {
            "trigger_hit_median": round(_median(trig_margin_hits, default=0.0), 6),
            "trigger_miss_median": round(_median(trig_margin_miss, default=0.0), 6),
            "direction_hit_median": round(_median(dir_margin_hits, default=0.0), 6),
            "direction_miss_median": round(_median(dir_margin_miss, default=0.0), 6),
        },
        "manual_support_prediction_summary": {
            "predicted_count": int(manual_predicted),
            "correct_count": int(manual_correct),
        },
        "trailing_support_prediction_summary": {
            "predicted_count": int(trailing_predicted),
            "correct_count": int(trailing_correct),
        },
        "take_profit_support_prediction_summary": {
            "predicted_count": int(take_profit_predicted),
            "correct_count": int(take_profit_correct),
        },
        "risk_cut_support_prediction_summary": {
            "predicted_count": int(risk_cut_predicted),
            "correct_count": int(risk_cut_correct),
        },
        "stale_support_prediction_summary": {
            "predicted_count": int(stale_predicted),
            "correct_count": int(stale_correct),
        },
        "manual_score_summary": {
            "predicted_manual_count": int(manual_predicted),
            "predicted_manual_correct_count": int(manual_correct),
            "same_symbol_support_max": int(
                max([int(_f(r.get("manual_same_symbol_support_count", 0.0), 0.0)) for r in rows] or [0])
            ),
            "same_regime_support_max": int(
                max([int(_f(r.get("manual_same_regime_support_count", 0.0), 0.0)) for r in rows] or [0])
            ),
        },
        "manual_prediction_diagnostics": _manual_prediction_diagnostics(rows),
        "admitted_vs_abstained_by_predicted_trigger": {
            "admitted": _count_by(rows, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
            "abstained": _count_by(abstained_population, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
        },
        "admitted_vs_abstained_by_actual_trigger": {
            "admitted": _count_by(rows, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
            "abstained": _count_by(abstained_population, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
        },
        "admitted_vs_abstained_by_source_type": {
            "admitted": _count_by(rows, lambda r: _s(r.get("source_type", "")) or "unknown"),
            "abstained": _count_by(abstained_population, lambda r: _s(r.get("source_type", "")) or "unknown"),
        },
        "actual_up_summary": {
            "by_symbol": _top_counter(actual_up_by_symbol),
            "predicted_down_by_symbol": _top_counter(actual_up_pred_down_by_symbol),
            "by_actual_trigger": _top_counter(actual_up_by_trigger),
            "by_predicted_trigger": _top_counter(actual_up_by_pred_trigger),
            "by_signal_side": _top_counter(actual_up_by_signal_side),
            "by_trend_bucket": _top_counter(actual_up_by_trend_bucket),
            "by_recent_return_3_bucket": _top_counter(actual_up_by_recent3_bucket),
            "by_recent_return_6_bucket": _top_counter(actual_up_by_recent6_bucket),
            "by_recent_return_12_bucket": _top_counter(actual_up_by_recent12_bucket),
            "by_recent_return_24_bucket": _top_counter(actual_up_by_recent24_bucket),
            "by_signal_margin_bucket": _top_counter(actual_up_by_signal_margin_bucket),
            "by_up_boundary_distance_bucket": _top_counter(actual_up_by_up_boundary_bucket),
            "by_down_boundary_distance_bucket": _top_counter(actual_up_by_down_boundary_bucket),
        },
        "take_profit_rows": take_profit_rows,
        "trailing_rows": trailing_rows,
        "risk_cut_predicted_stale_rows": risk_cut_as_stale,
        "upside_recovery_summary": {
            "applied_count_by_reason": _top_counter(upside_recovery_counts),
            "actual_up_predicted_up": int(sum(1 for r in full_population if _s(r.get("actual_direction", "")).lower() == "up" and _s(r.get("predicted_direction", "")).lower() == "up")),
            "actual_up_predicted_down": int(sum(1 for r in full_population if _s(r.get("actual_direction", "")).lower() == "up" and _s(r.get("predicted_direction", "")).lower() == "down")),
        },
        "crypto_predictor_source_mode": _count_by(full_population, lambda r: _s(r.get("predictor_source_mode", "")) or "legacy"),
        "stale_vs_trailing_score_margin_examples": stale_vs_trailing_examples,
        "stale_vs_manual_score_margin_examples": stale_vs_manual_examples,
        "actual_up_trailing_pred_down_stale_examples": uptrail_downstale,
        "actual_manual_pred_directional_examples": manual_mispreds,
    }


def _market_predictor_diagnostics(
    market: str,
    rows: List[Dict[str, Any]],
    *,
    full_rows: Optional[List[Dict[str, Any]]] = None,
    abstained_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    population = list(full_rows or rows)
    abstained = list(abstained_rows or [])
    if not population:
        return {}
    miss_symbol: Dict[str, int] = {}
    miss_reason: Dict[str, int] = {}
    miss_return_bucket: Dict[str, int] = {}
    miss_trend_bucket: Dict[str, int] = {}
    miss_vol_bucket: Dict[str, int] = {}
    miss_side: Dict[str, int] = {}
    miss_hold: Dict[str, int] = {}
    pnl_conf: Dict[str, Dict[str, int]] = {}
    actual_down_predicted_up_after_guard = 0
    stale_as_trailing = 0
    trailing_as_stale = 0
    stock_pnl_miss_symbol: Dict[str, int] = {}
    stock_pnl_miss_trigger: Dict[str, int] = {}
    stock_pnl_miss_pred_trigger: Dict[str, int] = {}
    stock_pnl_miss_direction: Dict[str, int] = {}
    stock_pnl_miss_pred_direction: Dict[str, int] = {}
    stock_pnl_miss_hold: Dict[str, int] = {}
    stock_pnl_miss_vol: Dict[str, int] = {}
    stock_pnl_miss_ret6: Dict[str, int] = {}
    stock_pnl_miss_ret24: Dict[str, int] = {}
    stock_pnl_miss_mfe: Dict[str, int] = {}
    stock_pnl_miss_mae: Dict[str, int] = {}
    stock_pnl_miss_drawdown: Dict[str, int] = {}
    stock_pnl_miss_bars_since_peak: Dict[str, int] = {}
    stock_pnl_miss_bars_in_trade: Dict[str, int] = {}
    stock_pnl_miss_trailing_armed: Dict[str, int] = {}
    stock_pnl_miss_softened: Dict[str, int] = {}
    stock_pnl_miss_stale_profile: Dict[str, int] = {}
    stock_pnl_miss_window: Dict[str, int] = {}
    stock_weak_window_reason_counts: Dict[str, int] = {}
    stock_direction_correct_pnl_wrong = 0
    stock_trigger_correct_pnl_wrong = 0
    stock_direction_trigger_correct_pnl_wrong = 0
    stock_pnl_false_positive = 0
    stock_pnl_false_negative = 0
    for row in population:
        actual_dir = _s(row.get("actual_direction", "")).lower() or "unknown"
        pred_dir = _s(row.get("predicted_direction", "")).lower() or "unknown"
        actual_trigger = _s(row.get("actual_exit_trigger", "")) or "Unknown"
        predicted_trigger = _s(row.get("predicted_exit_trigger", "")) or "Unknown"
        actual_pnl = "up" if _trade_return_pct(row) > 1e-9 else "down" if _trade_return_pct(row) < -1e-9 else "flat"
        pred_pnl = "up" if (_f(row.get("predicted_exit_price", 0.0), 0.0) - _f(row.get("entry_price", 0.0), 0.0)) > 1e-9 else "down" if (_f(row.get("predicted_exit_price", 0.0), 0.0) - _f(row.get("entry_price", 0.0), 0.0)) < -1e-9 else "flat"
        pnl_conf.setdefault(actual_pnl, {})
        pnl_conf[actual_pnl][pred_pnl] = int(pnl_conf[actual_pnl].get(pred_pnl, 0) + 1)
        if _s(market).lower() == "stocks":
            if actual_dir == "down" and pred_dir == "up" and bool(row.get("stock_down_case_guard_applied", False)):
                actual_down_predicted_up_after_guard += 1
            if actual_trigger == "Stale Alignment" and predicted_trigger == "Trailing":
                stale_as_trailing += 1
            if actual_trigger == "Trailing" and predicted_trigger == "Stale Alignment":
                trailing_as_stale += 1
            pnl_wrong = actual_pnl != pred_pnl
            if pnl_wrong:
                stock_pnl_miss_symbol[_s(row.get("symbol", "")) or "UNKNOWN"] = int(stock_pnl_miss_symbol.get(_s(row.get("symbol", "")) or "UNKNOWN", 0) + 1)
                stock_pnl_miss_trigger[actual_trigger] = int(stock_pnl_miss_trigger.get(actual_trigger, 0) + 1)
                stock_pnl_miss_pred_trigger[predicted_trigger] = int(stock_pnl_miss_pred_trigger.get(predicted_trigger, 0) + 1)
                stock_pnl_miss_direction[actual_dir] = int(stock_pnl_miss_direction.get(actual_dir, 0) + 1)
                stock_pnl_miss_pred_direction[pred_dir] = int(stock_pnl_miss_pred_direction.get(pred_dir, 0) + 1)
                stock_pnl_miss_hold[_bucket_hold_hours(_f(row.get("hold_hours", 0.0), 0.0))] = int(stock_pnl_miss_hold.get(_bucket_hold_hours(_f(row.get("hold_hours", 0.0), 0.0)), 0) + 1)
                stock_pnl_miss_vol[_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0))] = int(stock_pnl_miss_vol.get(_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0)), 0) + 1)
                stock_pnl_miss_ret6[_historical_replay_bucket("recent_return_6", abs(_f(row.get("recent_return_6", 0.0), 0.0)))] = int(stock_pnl_miss_ret6.get(_historical_replay_bucket("recent_return_6", abs(_f(row.get("recent_return_6", 0.0), 0.0))), 0) + 1)
                stock_pnl_miss_ret24[_historical_replay_bucket("recent_return_24", abs(_f(row.get("recent_return_24", 0.0), 0.0)))] = int(stock_pnl_miss_ret24.get(_historical_replay_bucket("recent_return_24", abs(_f(row.get("recent_return_24", 0.0), 0.0))), 0) + 1)
                stock_pnl_miss_mfe[_stock_shape_bucket(_f(row.get("max_favorable_excursion_pct", 0.0), 0.0), thresholds=(1.0, 2.0, 4.0))] = int(stock_pnl_miss_mfe.get(_stock_shape_bucket(_f(row.get("max_favorable_excursion_pct", 0.0), 0.0), thresholds=(1.0, 2.0, 4.0)), 0) + 1)
                stock_pnl_miss_mae[_stock_shape_bucket(_f(row.get("max_adverse_excursion_pct", 0.0), 0.0), thresholds=(0.5, 1.5, 3.0))] = int(stock_pnl_miss_mae.get(_stock_shape_bucket(_f(row.get("max_adverse_excursion_pct", 0.0), 0.0), thresholds=(0.5, 1.5, 3.0)), 0) + 1)
                stock_pnl_miss_drawdown[_stock_shape_bucket(_f(row.get("drawdown_from_peak_pct", 0.0), 0.0), thresholds=(0.5, 1.5, 3.0))] = int(stock_pnl_miss_drawdown.get(_stock_shape_bucket(_f(row.get("drawdown_from_peak_pct", 0.0), 0.0), thresholds=(0.5, 1.5, 3.0)), 0) + 1)
                stock_pnl_miss_bars_since_peak[_bucket_margin_value(_f(row.get("bars_since_peak", 0.0), 0.0))] = int(stock_pnl_miss_bars_since_peak.get(_bucket_margin_value(_f(row.get("bars_since_peak", 0.0), 0.0)), 0) + 1)
                stock_pnl_miss_bars_in_trade[_bucket_margin_value(_f(row.get("bars_in_trade", 0.0), 0.0))] = int(stock_pnl_miss_bars_in_trade.get(_bucket_margin_value(_f(row.get("bars_in_trade", 0.0), 0.0)), 0) + 1)
                stock_pnl_miss_trailing_armed[str(bool(row.get("trailing_armed", False))).lower()] = int(stock_pnl_miss_trailing_armed.get(str(bool(row.get("trailing_armed", False))).lower(), 0) + 1)
                stock_pnl_miss_softened[str(bool(row.get("favorable_then_softened_flag", False))).lower()] = int(stock_pnl_miss_softened.get(str(bool(row.get("favorable_then_softened_flag", False))).lower(), 0) + 1)
                stock_pnl_miss_stale_profile[str(bool(row.get("stale_hold_profile", False))).lower()] = int(stock_pnl_miss_stale_profile.get(str(bool(row.get("stale_hold_profile", False))).lower(), 0) + 1)
                stock_pnl_miss_window[str(int(_f(row.get("walkforward_window_index", 0.0), 0.0)) or 0)] = int(stock_pnl_miss_window.get(str(int(_f(row.get("walkforward_window_index", 0.0), 0.0)) or 0), 0) + 1)
                weak_reason = _s(row.get("stock_weak_window_guard_reason", "")) or "none"
                stock_weak_window_reason_counts[weak_reason] = int(stock_weak_window_reason_counts.get(weak_reason, 0) + 1)
                if actual_dir == pred_dir:
                    stock_direction_correct_pnl_wrong += 1
                if actual_trigger == predicted_trigger:
                    stock_trigger_correct_pnl_wrong += 1
                if actual_dir == pred_dir and actual_trigger == predicted_trigger:
                    stock_direction_trigger_correct_pnl_wrong += 1
                if pred_pnl == "up" and actual_pnl != "up":
                    stock_pnl_false_positive += 1
                if pred_pnl != "up" and actual_pnl == "up":
                    stock_pnl_false_negative += 1
        if actual_dir != pred_dir or actual_trigger != predicted_trigger:
            miss_symbol[_s(row.get("symbol", "")) or "UNKNOWN"] = int(miss_symbol.get(_s(row.get("symbol", "")) or "UNKNOWN", 0) + 1)
            miss_reason[_s(row.get("raw_rule_reason", row.get("event_exit_tag", ""))) or "unknown"] = int(miss_reason.get(_s(row.get("raw_rule_reason", row.get("event_exit_tag", ""))) or "unknown", 0) + 1)
            miss_return_bucket[_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", _trade_return_pct(row)), _trade_return_pct(row)))] = int(miss_return_bucket.get(_historical_replay_bucket("recent_return_24", _f(row.get("recent_return_24", _trade_return_pct(row)), _trade_return_pct(row))), 0) + 1)
            miss_trend_bucket[_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", _trade_return_pct(row)), _trade_return_pct(row)))] = int(miss_trend_bucket.get(_historical_replay_bucket("trend_momentum_score", _f(row.get("trend_momentum_score", _trade_return_pct(row)), _trade_return_pct(row))), 0) + 1)
            miss_vol_bucket[_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0))] = int(miss_vol_bucket.get(_historical_replay_bucket("recent_volatility", _f(row.get("recent_volatility", 0.0), 0.0)), 0) + 1)
            miss_side[_s(row.get("side", row.get("signal_side", ""))) or "unknown"] = int(miss_side.get(_s(row.get("side", row.get("signal_side", ""))) or "unknown", 0) + 1)
            miss_hold[_bucket_hold_hours(_f(row.get("hold_hours", 0.0), 0.0))] = int(miss_hold.get(_bucket_hold_hours(_f(row.get("hold_hours", 0.0), 0.0)), 0) + 1)
    out = {
        "predictor_mode": _count_by(population, lambda r: _s(r.get("predictor_mode", "")) or "generic"),
        "actual_direction_distribution": _count_by(population, lambda r: _s(r.get("actual_direction", "")).lower() or "unknown"),
        "predicted_direction_distribution": _count_by(population, lambda r: _s(r.get("predicted_direction", "")).lower() or "unknown"),
        "actual_trigger_distribution": _count_by(population, lambda r: _s(r.get("actual_exit_trigger", "")) or "Unknown"),
        "predicted_trigger_distribution": _count_by(population, lambda r: _s(r.get("predicted_exit_trigger", "")) or "Unknown"),
        "direction_confusion_matrix": _confusion_matrix(population, "actual_direction", "predicted_direction"),
        "trigger_confusion_matrix": _confusion_matrix(population, "actual_exit_trigger", "predicted_exit_trigger"),
        "pnl_trend_confusion_matrix": pnl_conf,
        "misses_by_symbol": _top_counter(miss_symbol),
        "misses_by_rule_reason": _top_counter(miss_reason),
        "misses_by_recent_return_bucket": _top_counter(miss_return_bucket),
        "misses_by_trend_bucket": _top_counter(miss_trend_bucket),
        "misses_by_volatility_bucket": _top_counter(miss_vol_bucket),
        "misses_by_side": _top_counter(miss_side),
        "misses_by_hold_bucket": _top_counter(miss_hold),
        "admitted_vs_abstained_by_source_type": {
            "admitted": _count_by(rows, lambda r: _s(r.get("source_type", "")) or "unknown"),
            "abstained": _count_by(abstained, lambda r: _s(r.get("source_type", "")) or "unknown"),
        },
        "provider_counts": _count_by(population, lambda r: _s(r.get("provider", "")) or "unknown"),
        "feature_source_used": _count_by(population, lambda r: _s(r.get("source_type", "")) or "unknown"),
        "market": market,
    }
    if _s(market).lower() == "stocks":
        out["stock_actual_down_predicted_up_after_guard"] = int(actual_down_predicted_up_after_guard)
        out["stock_stale_trailing_confusion_after_exit_shape"] = {
            "stale_predicted_trailing": int(stale_as_trailing),
            "trailing_predicted_stale": int(trailing_as_stale),
        }
        out["stock_direction_correct_pnl_wrong_count"] = int(stock_direction_correct_pnl_wrong)
        out["stock_trigger_correct_pnl_wrong_count"] = int(stock_trigger_correct_pnl_wrong)
        out["stock_direction_trigger_correct_pnl_wrong_count"] = int(stock_direction_trigger_correct_pnl_wrong)
        out["stock_pnl_trend_miss_diagnostics"] = {
            "pnl_trend_false_positives": int(stock_pnl_false_positive),
            "pnl_trend_false_negatives": int(stock_pnl_false_negative),
            "misses_by_symbol": _top_counter(stock_pnl_miss_symbol),
            "misses_by_trigger": _top_counter(stock_pnl_miss_trigger),
            "misses_by_predicted_trigger": _top_counter(stock_pnl_miss_pred_trigger),
            "misses_by_direction": _top_counter(stock_pnl_miss_direction),
            "misses_by_predicted_direction": _top_counter(stock_pnl_miss_pred_direction),
            "misses_by_hold_bucket": _top_counter(stock_pnl_miss_hold),
            "misses_by_volatility_bucket": _top_counter(stock_pnl_miss_vol),
            "misses_by_return_6_bucket": _top_counter(stock_pnl_miss_ret6),
            "misses_by_return_24_bucket": _top_counter(stock_pnl_miss_ret24),
            "misses_by_max_favorable_excursion_pct": _top_counter(stock_pnl_miss_mfe),
            "misses_by_max_adverse_excursion_pct": _top_counter(stock_pnl_miss_mae),
            "misses_by_drawdown_from_peak_pct": _top_counter(stock_pnl_miss_drawdown),
            "misses_by_bars_since_peak": _top_counter(stock_pnl_miss_bars_since_peak),
            "misses_by_bars_in_trade": _top_counter(stock_pnl_miss_bars_in_trade),
            "misses_by_trailing_armed": _top_counter(stock_pnl_miss_trailing_armed),
            "misses_by_favorable_then_softened_flag": _top_counter(stock_pnl_miss_softened),
            "misses_by_stale_hold_profile": _top_counter(stock_pnl_miss_stale_profile),
            "misses_by_walkforward_window": _top_counter(stock_pnl_miss_window),
        }
        out["stock_weak_window_diagnostics"] = {
            "guard_reasons": _top_counter(stock_weak_window_reason_counts),
            "guard_applied_rows": int(sum(1 for r in population if bool(r.get("stock_weak_window_guard_applied", False)))),
            "high_vol_long_hold_penalty_rows": int(sum(1 for r in population if _f(r.get("stock_high_vol_long_hold_penalty", 0.0), 0.0) > 0.0)),
            "drawdown_quality_penalty_rows": int(sum(1 for r in population if _f(r.get("stock_drawdown_quality_penalty", 0.0), 0.0) > 0.0)),
            "symbol_pnl_history_penalty_rows": int(sum(1 for r in population if _f(r.get("stock_symbol_pnl_history_penalty", 0.0), 0.0) > 0.0)),
            "weak_window_metrics_by_window": {
                str(window): _metrics_for_rows([r for r in population if str(int(_f(r.get("walkforward_window_index", 0.0), 0.0)) or 0) == str(window)])
                for window in sorted({str(int(_f(r.get("walkforward_window_index", 0.0), 0.0)) or 0) for r in population})
            },
        }
    return out


def build_replay_diagnostics(hub_dir: str, market: str) -> Dict[str, Any]:
    m = _s(market).lower()
    path = _latest_replay_path(hub_dir, m)
    if not path:
        return {"market": m, "state": "NO_DATA", "msg": "no replay artifact found"}
    payload = _safe_read_json(path)
    iterations = payload.get("iterations", []) if isinstance(payload.get("iterations", []), list) else []
    test_rows: List[Dict[str, Any]] = []
    if iterations:
        it = iterations[-1] if isinstance(iterations[-1], dict) else {}
        test_rows = it.get("test_predictions", []) if isinstance(it.get("test_predictions", []), list) else []
    if not test_rows:
        return {
            "market": m,
            "state": "NO_TEST_ROWS",
            "source": path,
            "summary": _s(payload.get("summary", "")),
        }

    by_trigger: Dict[str, List[Dict[str, Any]]] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    for row in test_rows:
        act = _s(row.get("actual_exit_trigger", "Unknown")) or "Unknown"
        pred = _s(row.get("predicted_exit_trigger", "Unknown")) or "Unknown"
        by_trigger.setdefault(act, []).append(row)
        if act not in confusion:
            confusion[act] = {}
        confusion[act][pred] = int(confusion[act].get(pred, 0) + 1)
    seg = {}
    for trig, rows in by_trigger.items():
        seg[trig] = _metrics_for_rows(rows)

    headline = payload.get("hybrid_test_metrics", {}) if isinstance(payload.get("hybrid_test_metrics", {}), dict) else {}
    if not headline:
        headline = payload.get("best_test_metrics", {}) if isinstance(payload.get("best_test_metrics", {}), dict) else {}
    return {
        "market": m,
        "state": "READY",
        "source": path,
        "test_rows": int(len(test_rows)),
        "headline_metrics": {
            "directional_accuracy_pct": round(_f(headline.get("directional_accuracy_pct", 0.0), 0.0), 4),
            "trigger_match_pct": round(_f(headline.get("trigger_match_pct", 0.0), 0.0), 4),
            "trigger_scored_trades": round(_f(headline.get("trigger_scored_trades", 0.0), 0.0), 4),
            "trigger_coverage_pct": round(_f(headline.get("trigger_coverage_pct", 0.0), 0.0), 4),
            "pnl_trend_match_pct": round(_f(headline.get("pnl_trend_match_pct", 0.0), 0.0), 4),
        },
        "segmented_by_actual_trigger": seg,
        "trigger_confusion_matrix": confusion,
    }


def run_model_quality_full_pass(
    *,
    base_dir: str,
    hub_dir: str,
    settings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    crypto_backfill_enabled = _env_flag("CRYPTO_REPLAY_BACKFILL")
    crypto_backfill_lookback_days = _env_int("CRYPTO_REPLAY_BACKFILL_LOOKBACK_DAYS", 365 if crypto_backfill_enabled else 90)
    crypto_backfill_timeframe = _s(os.environ.get("CRYPTO_REPLAY_BACKFILL_TIMEFRAME", "1hour" if crypto_backfill_enabled else "1hour")) or "1hour"
    crypto_backfill_max_symbols = _env_int("CRYPTO_REPLAY_BACKFILL_MAX_SYMBOLS", 20 if crypto_backfill_enabled else 12)
    crypto_backfill_force_refresh = _env_flag("CRYPTO_REPLAY_BACKFILL_FORCE_REFRESH")
    crypto_deterministic = _env_flag("CRYPTO_REPLAY_DETERMINISTIC")
    crypto_freeze_symbols = _env_flag("CRYPTO_REPLAY_FREEZE_SYMBOLS")
    crypto_freeze_cache = _env_flag("CRYPTO_REPLAY_FREEZE_CACHE")
    crypto_repeat_evals = max(1, _env_int("CRYPTO_REPLAY_REPEAT_EVALS", 1))
    crypto_eval_start_ts = _env_int("CRYPTO_REPLAY_EVAL_START_TS", 0)
    crypto_eval_cutoff_ts = _env_int("CRYPTO_REPLAY_EVAL_CUTOFF_TS", 0)
    crypto_symbols_lock_file = _env_str(
        "CRYPTO_REPLAY_SYMBOLS_LOCK_FILE",
        os.path.join(hub_dir, "crypto", "historical_replay_cache", "symbols.lock.json"),
    )
    stock_backfill_enabled = _env_flag("STOCK_REPLAY_BACKFILL")
    stock_backfill_lookback_days = _env_int("STOCK_REPLAY_BACKFILL_LOOKBACK_DAYS", 365 if stock_backfill_enabled else 90)
    stock_backfill_timeframe = _s(os.environ.get("STOCK_REPLAY_BACKFILL_TIMEFRAME", "1Hour" if stock_backfill_enabled else "1Hour")) or "1Hour"
    stock_backfill_max_symbols = _env_int("STOCK_REPLAY_BACKFILL_MAX_SYMBOLS", 50 if stock_backfill_enabled else 24)
    stock_backfill_force_refresh = _env_flag("STOCK_REPLAY_BACKFILL_FORCE_REFRESH")
    use_live_rows_as_primary = _env_flag("MODEL_QUALITY_USE_LIVE_ROWS_AS_PRIMARY")
    use_completed_live_priors = _env_flag("MODEL_QUALITY_USE_COMPLETED_LIVE_PRIORS")
    use_legacy_replay_as_primary = _env_flag("MODEL_QUALITY_USE_LEGACY_REPLAY_AS_PRIMARY")
    ts = int(time.time())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    out_dir = os.path.join(hub_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)

    markets = ["crypto", "stocks", "forex"]
    dataset_reports: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    closed_by_market: Dict[str, List[Dict[str, Any]]] = {}
    completed_live_rows_by_market: Dict[str, List[Dict[str, Any]]] = {}
    replay_generation: Dict[str, Any] = {}
    crypto_artifact_report: Dict[str, Any] = {}
    existing_runtime_settings = _existing_runtime_settings_diagnostics(cfg, base_dir)
    for m in markets:
        loaded = load_market_trade_events(hub_dir, m)
        events = loaded.get("events", []) if isinstance(loaded.get("events", []), list) else []
        closed = build_closed_trades(events, m).get("closed_trades", [])
        live_rows, live_diag = _completed_live_decision_rows(market=m, events=events, closed_rows=list(closed or []))
        completed_live_rows_by_market[m] = list(live_rows or [])
        if m == "crypto":
            crypto_rows_before = int(len(list(closed or [])))
            historical_replay = build_crypto_historical_strategy_replay(
                hub_dir=hub_dir,
                settings=cfg,
                timeframe=crypto_backfill_timeframe,
                lookback_days=crypto_backfill_lookback_days,
                max_symbols=crypto_backfill_max_symbols,
                force_refresh=crypto_backfill_force_refresh,
                start_ts=crypto_eval_start_ts if crypto_eval_start_ts > 0 else None,
                cutoff_ts=crypto_eval_cutoff_ts if crypto_eval_cutoff_ts > 0 else None,
                symbols_lock_file=crypto_symbols_lock_file,
                freeze_symbols=crypto_freeze_symbols,
                freeze_cache=crypto_freeze_cache,
                deterministic=crypto_deterministic,
            )
            hist_ok, hist_reason = _crypto_historical_replay_sufficient(historical_replay if isinstance(historical_replay, dict) else {})
            crypto_artifact_report = _attach_crypto_artifact_features(
                list(closed if isinstance(closed, list) else []),
                base_dir=base_dir,
                settings=cfg,
            )
            artifact_feature_diag = crypto_artifact_report.get("feature_source", {}) if isinstance(crypto_artifact_report.get("feature_source", {}), dict) else {}
            artifact_primary_ok = int(_f(artifact_feature_diag.get("artifact_rows", 0), 0.0)) > 0
            live_primary_ok = bool(use_live_rows_as_primary and live_rows and len(live_rows) >= 40)
            source_selection_reason = ""
            primary_source_used = "closed_trade_only"
            if live_primary_ok:
                closed = live_rows
                primary_source_used = "completed_live_decision_snapshot"
                source_selection_reason = "live_rows_primary_override_enabled_and_sufficient"
            elif hist_ok:
                hist_rows = historical_replay.get("rows", [])
                if isinstance(hist_rows, list) and hist_rows:
                    closed = hist_rows
                primary_source_used = "historical_strategy_replay"
                source_selection_reason = "historical_strategy_replay_available_and_selected_by_default"
            elif artifact_primary_ok:
                primary_source_used = "trained_artifact"
                source_selection_reason = "historical_strategy_replay_insufficient_using_trained_artifact_enriched_closed_trades"
            else:
                source_selection_reason = "historical_strategy_replay_insufficient_falling_back_to_closed_trade_only"
            replay_generation[m] = {
                "source_type": primary_source_used,
                "model_quality_source_priority": [
                    "historical_strategy_replay",
                    "trained_artifact",
                    "closed_trade_only",
                ],
                "model_quality_default_source": "historical_strategy_replay",
                "model_quality_primary_source_used": primary_source_used,
                "model_quality_live_rows_supplemental_only": not bool(use_live_rows_as_primary),
                "completed_live_rows_used_as_primary": bool(primary_source_used == "completed_live_decision_snapshot"),
                "completed_live_rows_used_as_supplemental": bool(live_rows),
                "completed_live_rows_found": int(_f(live_diag.get("live_decision_rows_found", 0), 0.0)),
                "completed_live_rows_completed": int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)),
                "completed_live_join_rate_pct": round(_f(live_diag.get("live_decision_join_rate_pct", 0.0), 0.0), 4),
                "completed_live_rows_by_market": dict(live_diag.get("live_decision_rows_by_market", {}) if isinstance(live_diag.get("live_decision_rows_by_market", {}), dict) else {}),
                "completed_live_rows_by_predictor": dict(live_diag.get("live_decision_rows_by_predictor", {}) if isinstance(live_diag.get("live_decision_rows_by_predictor", {}), dict) else {}),
                "completed_live_priors_enabled": bool(use_completed_live_priors),
                "completed_live_prior_rows_used": 0,
                "completed_live_learning_ready": bool(int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)) >= 20),
                "source_selection_reason": source_selection_reason,
                "rows_generated": int(len(list(closed or []))),
                "historical_strategy_replay": (historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}),
                "historical_strategy_replay_available": bool(hist_ok),
                "historical_strategy_replay_reason": "" if hist_ok else (hist_reason or _s((historical_replay.get("state", "") if isinstance(historical_replay, dict) else "")) or "historical_strategy_replay_unavailable"),
                "live_decision_source_diagnostics": live_diag,
                "trained_artifact_diagnostics": crypto_artifact_report.get("discovery", {}),
                "crypto_feature_source_diagnostics": crypto_artifact_report.get("feature_source", {}),
                "crypto_backfill_mode_enabled": bool(crypto_backfill_enabled),
                "crypto_backfill_lookback_days": int(crypto_backfill_lookback_days),
                "crypto_backfill_timeframe": crypto_backfill_timeframe,
                "crypto_backfill_max_symbols": int(crypto_backfill_max_symbols),
                "crypto_backfill_force_refresh": bool(crypto_backfill_force_refresh),
                "crypto_backfill_symbols_requested": int(len(list((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("historical_strategy_replay_symbols", []) or []))) if hist_ok else 0,
                "crypto_backfill_symbols_completed": int(len(list((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("symbols_covered", []) or []))),
                "crypto_backfill_remote_rows_fetched": int(_f((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("remote_rows_fetched", 0), 0.0)),
                "crypto_backfill_cache_rows_loaded": int(_f((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("cache_rows_loaded", 0), 0.0)),
                "crypto_backfill_cache_path": _s((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("replay_cache_path", "")),
                "crypto_backfill_provider": "kucoin",
                "crypto_backfill_errors": list((historical_replay.get("skipped", []) if isinstance(historical_replay, dict) else []) or []),
                "crypto_replay_rows_before": int(crypto_rows_before),
                "crypto_replay_rows_after": int(len(list(closed or []))),
                "crypto_replay_deterministic_mode_enabled": bool(crypto_deterministic),
                "crypto_replay_eval_start_ts": int(_f((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_eval_start_ts", crypto_eval_start_ts), 0.0)),
                "crypto_replay_eval_cutoff_ts": int(_f((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_eval_cutoff_ts", crypto_eval_cutoff_ts), 0.0)),
                "crypto_replay_symbols_locked": bool((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_symbols_locked", crypto_freeze_symbols)),
                "crypto_replay_symbol_list_hash": _s((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_symbol_list_hash", "")),
                "crypto_replay_cache_frozen": bool((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_cache_frozen", crypto_freeze_cache)),
                "crypto_replay_rows_hash": _s((historical_replay.get("diagnostics", {}) if isinstance(historical_replay, dict) else {}).get("crypto_replay_rows_hash", "")),
                "crypto_replay_repeat_evals": int(crypto_repeat_evals),
            }
            replay_generation[m]["model_quality_source_priority_used"] = primary_source_used
        if m == "stocks" and (stock_backfill_enabled or len(list(closed or [])) < 40):
            stock_rows_before = int(len(list(closed or [])))
            stock_gen = _generate_stock_historical_replay_closed_trades(
                hub_dir=hub_dir,
                base_dir=base_dir,
                settings=cfg,
                existing_closed_rows=list(closed or []),
                lookback_days=stock_backfill_lookback_days,
                max_symbols=stock_backfill_max_symbols,
                force_refresh=stock_backfill_force_refresh,
            )
            replay_generation[m] = stock_gen.get("diagnostics", {})
            stock_rows = stock_gen.get("rows", [])
            live_primary_ok = bool(use_live_rows_as_primary and live_rows and len(live_rows) >= 40)
            if live_primary_ok:
                closed = live_rows
                primary_source_used = "completed_live_decision_snapshot"
                source_selection_reason = "live_rows_primary_override_enabled_and_sufficient"
            elif isinstance(stock_rows, list) and stock_rows:
                closed = stock_rows
                primary_source_used = "historical_api_replay"
                source_selection_reason = "historical_stock_replay_available_and_selected_by_default"
                dataset_reports.setdefault(m, {})
            else:
                primary_source_used = "closed_trade_only"
                source_selection_reason = "historical_stock_replay_unavailable_falling_back_to_closed_trade_only"
            replay_generation[m]["stock_backfill_mode_enabled"] = bool(stock_backfill_enabled)
            replay_generation[m]["stock_backfill_lookback_days"] = int(stock_backfill_lookback_days)
            replay_generation[m]["stock_backfill_timeframe"] = stock_backfill_timeframe
            replay_generation[m]["stock_backfill_max_symbols"] = int(stock_backfill_max_symbols)
            replay_generation[m]["stock_backfill_force_refresh"] = bool(stock_backfill_force_refresh)
            replay_generation[m]["stock_replay_rows_before"] = int(stock_rows_before)
            replay_generation[m]["stock_replay_rows_after"] = int(len(list(closed or [])))
            replay_generation[m]["live_decision_source_diagnostics"] = live_diag
            replay_generation[m]["model_quality_source_priority"] = [
                "historical_api_replay",
                "closed_trade_only",
            ]
            replay_generation[m]["model_quality_default_source"] = "historical_api_replay"
            replay_generation[m]["model_quality_primary_source_used"] = primary_source_used
            replay_generation[m]["model_quality_live_rows_supplemental_only"] = not bool(use_live_rows_as_primary)
            replay_generation[m]["completed_live_rows_used_as_primary"] = bool(primary_source_used == "completed_live_decision_snapshot")
            replay_generation[m]["completed_live_rows_used_as_supplemental"] = bool(live_rows)
            replay_generation[m]["completed_live_rows_found"] = int(_f(live_diag.get("live_decision_rows_found", 0), 0.0))
            replay_generation[m]["completed_live_rows_completed"] = int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0))
            replay_generation[m]["completed_live_join_rate_pct"] = round(_f(live_diag.get("live_decision_join_rate_pct", 0.0), 0.0), 4)
            replay_generation[m]["completed_live_rows_by_market"] = dict(live_diag.get("live_decision_rows_by_market", {}) if isinstance(live_diag.get("live_decision_rows_by_market", {}), dict) else {})
            replay_generation[m]["completed_live_rows_by_predictor"] = dict(live_diag.get("live_decision_rows_by_predictor", {}) if isinstance(live_diag.get("live_decision_rows_by_predictor", {}), dict) else {})
            replay_generation[m]["completed_live_priors_enabled"] = bool(use_completed_live_priors)
            replay_generation[m]["completed_live_prior_rows_used"] = 0
            replay_generation[m]["completed_live_learning_ready"] = bool(int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)) >= 20)
            replay_generation[m]["source_selection_reason"] = source_selection_reason
            replay_generation[m]["model_quality_source_priority_used"] = primary_source_used
        elif m != "crypto":
            replay_generation[m] = {
                "source_type": "execution_log",
                "rows_generated": int(len(list(closed or []))),
                "live_decision_source_diagnostics": live_diag,
                "model_quality_source_priority": [
                    "completed_live_decision_snapshot",
                    "execution_log",
                ],
                "model_quality_default_source": "execution_log",
                "model_quality_primary_source_used": "execution_log",
                "model_quality_live_rows_supplemental_only": True,
                "completed_live_rows_used_as_primary": False,
                "completed_live_rows_used_as_supplemental": bool(live_rows),
                "completed_live_rows_found": int(_f(live_diag.get("live_decision_rows_found", 0), 0.0)),
                "completed_live_rows_completed": int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)),
                "completed_live_join_rate_pct": round(_f(live_diag.get("live_decision_join_rate_pct", 0.0), 0.0), 4),
                "completed_live_rows_by_market": dict(live_diag.get("live_decision_rows_by_market", {}) if isinstance(live_diag.get("live_decision_rows_by_market", {}), dict) else {}),
                "completed_live_rows_by_predictor": dict(live_diag.get("live_decision_rows_by_predictor", {}) if isinstance(live_diag.get("live_decision_rows_by_predictor", {}), dict) else {}),
                "completed_live_priors_enabled": bool(use_completed_live_priors),
                "completed_live_prior_rows_used": 0,
                "completed_live_learning_ready": bool(int(_f(live_diag.get("live_decision_rows_completed", 0), 0.0)) >= 20),
                "source_selection_reason": "forex_execution_log_default_primary",
                "model_quality_source_priority_used": "execution_log",
            }
        closed_by_market[m] = list(closed if isinstance(closed, list) else [])
        snap_path = os.path.join(out_dir, f"{m}_closed_trades_v1_{stamp}.jsonl")
        _write_jsonl(snap_path, closed if isinstance(closed, list) else [])
        snapshots[m] = {
            "path": snap_path,
            "sha256": _sha256_file(snap_path),
            "rows": int(len(closed) if isinstance(closed, list) else 0),
        }
        dataset_reports[m] = build_market_dataset_quality(hub_dir, m)
        if isinstance(dataset_reports.get(m, {}), dict):
            dataset_reports[m]["replay_source_diagnostics"] = replay_generation.get(m, {})
            if m == "crypto":
                dataset_reports[m]["trained_artifact_diagnostics"] = crypto_artifact_report.get("discovery", {})
                feature_diag = dict(crypto_artifact_report.get("feature_source", {}) if isinstance(crypto_artifact_report.get("feature_source", {}), dict) else {})
                hist_diag = replay_generation.get(m, {}).get("historical_strategy_replay", {}) if isinstance(replay_generation.get(m, {}), dict) else {}
                hist_used = bool(replay_generation.get(m, {}).get("historical_strategy_replay_available", False))
                primary_source_used = _s(replay_generation.get(m, {}).get("model_quality_primary_source_used", "closed_trade_only")) or "closed_trade_only"
                feature_diag["crypto_feature_source_priority"] = [
                    "historical_strategy_replay",
                    "trained_artifact",
                    "closed_trade_only",
                ]
                feature_diag["completed_live_rows_available"] = bool(live_rows)
                feature_diag["completed_live_rows_used_as_primary"] = bool(replay_generation.get(m, {}).get("completed_live_rows_used_as_primary", False))
                feature_diag["completed_live_rows_used_as_supplemental"] = bool(replay_generation.get(m, {}).get("completed_live_rows_used_as_supplemental", False))
                feature_diag["model_quality_live_rows_supplemental_only"] = bool(replay_generation.get(m, {}).get("model_quality_live_rows_supplemental_only", True))
                feature_diag["model_quality_default_source"] = _s(replay_generation.get(m, {}).get("model_quality_default_source", "historical_strategy_replay"))
                feature_diag["model_quality_primary_source_used"] = primary_source_used
                feature_diag["source_selection_reason"] = _s(replay_generation.get(m, {}).get("source_selection_reason", ""))
                feature_diag["feature_source_rows"] = {
                    "decision_snapshot": int(_f(feature_diag.get("feature_source_rows", {}).get("decision_snapshot", 0), 0.0)) if isinstance(feature_diag.get("feature_source_rows", {}), dict) else 0,
                    "historical_strategy_replay": int(_f(hist_diag.get("historical_strategy_replay_rows", 0), 0.0)),
                    "trained_artifact": int(_f(feature_diag.get("feature_source_rows", {}).get("trained_artifact", 0), 0.0)) if isinstance(feature_diag.get("feature_source_rows", {}), dict) else 0,
                    "closed_trade_only": int(_f(feature_diag.get("feature_source_rows", {}).get("closed_trade_only", 0), 0.0)) if isinstance(feature_diag.get("feature_source_rows", {}), dict) else 0,
                }
                if primary_source_used == "completed_live_decision_snapshot":
                    feature_diag["crypto_feature_source_used"] = "completed_live_decision_snapshot"
                elif primary_source_used == "historical_strategy_replay" and hist_used:
                    feature_diag["crypto_feature_source_used"] = "historical_strategy_replay"
                elif primary_source_used == "trained_artifact":
                    feature_diag["crypto_feature_source_used"] = "trained_artifact"
                else:
                    feature_diag["crypto_feature_source_used"] = "closed_trade_only"
                    feature_diag["historical_strategy_replay_reason"] = ""
                feature_diag["historical_strategy_replay_available"] = bool(hist_used)
                feature_diag["historical_strategy_replay_rows"] = int(_f(hist_diag.get("historical_strategy_replay_rows", 0), 0.0))
                feature_diag["historical_strategy_replay_symbols"] = list(hist_diag.get("historical_strategy_replay_symbols", []) or [])
                feature_diag["historical_strategy_replay_timeframe"] = _s(hist_diag.get("historical_strategy_replay_timeframe", ""))
                feature_diag["historical_strategy_replay_date_range"] = hist_diag.get("historical_strategy_replay_date_range", {})
                feature_diag["historical_strategy_replay_provider"] = _s(hist_diag.get("historical_strategy_replay_provider", ""))
                feature_diag["historical_strategy_replay_cache_path"] = _s(hist_diag.get("historical_strategy_replay_cache_path", ""))
                feature_diag["historical_strategy_replay_skipped_reasons"] = list(hist_diag.get("historical_strategy_replay_skipped_reasons", []) or [])
                feature_diag["replay_used_remote_api"] = bool(hist_diag.get("replay_used_remote_api", False))
                feature_diag["replay_used_local_cache"] = bool(hist_diag.get("replay_used_local_cache", False))
                feature_diag["replay_mutated_live_artifacts"] = bool(hist_diag.get("replay_mutated_live_artifacts", False))
                if not hist_used:
                    feature_diag["fallback_reason"] = (
                        _s(replay_generation.get(m, {}).get("historical_strategy_replay_reason", ""))
                        or feature_diag.get("fallback_reason", {})
                    )
                dataset_reports[m]["feature_source_diagnostics"] = feature_diag
                replay_generation[m]["crypto_feature_source_diagnostics"] = feature_diag

    completed_live_decision_artifacts = _write_completed_live_decision_artifacts(
        hub_dir,
        completed_live_rows_by_market,
    )
    synthetic_completed_live_verification = _verify_completed_live_synthetic_paths()
    completed_live_rows_found: Dict[str, int] = {}
    completed_live_rows_completed: Dict[str, int] = {}
    completed_live_join_rate_pct: Dict[str, float] = {}
    completed_live_missing_fields_by_market: Dict[str, Dict[str, int]] = {}
    completed_live_eligible_rows_by_market: Dict[str, int] = {}
    completed_live_ineligible_rows_by_market: Dict[str, int] = {}
    completed_live_waiting_for_real_trade_completion: Dict[str, bool] = {}
    completed_live_no_post_model_completed_trades_yet: Dict[str, bool] = {}
    for m in markets:
        replay_meta = replay_generation.get(m, {}) if isinstance(replay_generation.get(m, {}), dict) else {}
        live_diag = replay_meta.get("live_decision_source_diagnostics", {}) if isinstance(replay_meta.get("live_decision_source_diagnostics", {}), dict) else {}
        completed_live_rows_found[m] = int(_f(replay_meta.get("completed_live_rows_found", 0), 0.0))
        completed_live_rows_completed[m] = int(_f(replay_meta.get("completed_live_rows_completed", 0), 0.0))
        completed_live_join_rate_pct[m] = round(_f(replay_meta.get("completed_live_join_rate_pct", 0.0), 0.0), 4)
        missing_by_market = live_diag.get("completed_live_missing_fields_by_market", {}) if isinstance(live_diag.get("completed_live_missing_fields_by_market", {}), dict) else {}
        eligible_by_market = live_diag.get("completed_live_eligible_rows_by_market", {}) if isinstance(live_diag.get("completed_live_eligible_rows_by_market", {}), dict) else {}
        ineligible_by_market = live_diag.get("completed_live_ineligible_rows_by_market", {}) if isinstance(live_diag.get("completed_live_ineligible_rows_by_market", {}), dict) else {}
        completed_live_missing_fields_by_market[m] = dict(missing_by_market.get(m, {}) if isinstance(missing_by_market.get(m, {}), dict) else {})
        completed_live_eligible_rows_by_market[m] = int(_f(eligible_by_market.get(m, 0), 0.0))
        completed_live_ineligible_rows_by_market[m] = int(_f(ineligible_by_market.get(m, 0), 0.0))
        completed_live_waiting_for_real_trade_completion[m] = bool(live_diag.get("completed_live_waiting_for_real_trade_completion", True))
        completed_live_no_post_model_completed_trades_yet[m] = bool(live_diag.get("completed_live_no_post_model_completed_trades_yet", True))
    completed_live_verification = {
        "completed_live_rows_found": completed_live_rows_found,
        "completed_live_rows_completed": completed_live_rows_completed,
        "completed_live_join_rate_pct": completed_live_join_rate_pct,
        "completed_live_missing_fields_by_market": completed_live_missing_fields_by_market,
        "completed_live_eligible_rows_by_market": completed_live_eligible_rows_by_market,
        "completed_live_ineligible_rows_by_market": completed_live_ineligible_rows_by_market,
        "completed_live_waiting_for_real_trade_completion": completed_live_waiting_for_real_trade_completion,
        "completed_live_no_post_model_completed_trades_yet": completed_live_no_post_model_completed_trades_yet,
        "completed_live_synthetic_path_verified_by_market": dict(synthetic_completed_live_verification.get("completed_live_synthetic_path_verified_by_market", {})),
        "completed_live_synthetic_blockers_by_market": dict(synthetic_completed_live_verification.get("completed_live_synthetic_blockers_by_market", {})),
    }

    # Build/refresh replay artifacts for all markets from normalized closed trades.
    openai_dir = os.path.join(hub_dir, "openai")
    os.makedirs(openai_dir, exist_ok=True)
    synthetic_paths: Dict[str, str] = {}
    for m in markets:
        deterministic_cfg = None
        if m == "crypto" and crypto_deterministic:
            replay_diag = replay_generation.get(m, {}) if isinstance(replay_generation.get(m, {}), dict) else {}
            deterministic_cfg = {
                "enabled": True,
                "start_ts": int(_f(replay_diag.get("crypto_replay_eval_start_ts", crypto_eval_start_ts), 0.0)),
                "cutoff_ts": int(_f(replay_diag.get("crypto_replay_eval_cutoff_ts", crypto_eval_cutoff_ts), 0.0)),
                "locked_symbols": list(
                    ((replay_diag.get("historical_strategy_replay", {}) if isinstance(replay_diag.get("historical_strategy_replay", {}), dict) else {}).get("historical_strategy_replay_symbols", []))
                    or _load_locked_symbol_list(crypto_symbols_lock_file)
                ),
                "symbols_locked": bool(replay_diag.get("crypto_replay_symbols_locked", crypto_freeze_symbols)),
                "symbol_list_hash": _s(replay_diag.get("crypto_replay_symbol_list_hash", "")),
                "cache_frozen": bool(replay_diag.get("crypto_replay_cache_frozen", crypto_freeze_cache)),
            }
        payload = build_synthetic_replay_artifact(hub_dir, m, closed_by_market.get(m, []), deterministic_cfg)
        if m == "crypto" and crypto_deterministic and isinstance(payload, dict):
            repeat_results: List[Dict[str, Any]] = []
            for idx in range(max(1, crypto_repeat_evals)):
                rep = build_synthetic_replay_artifact(hub_dir, m, closed_by_market.get(m, []), deterministic_cfg)
                latest = rep.get("safe_selection_diagnostics", {}).get("latest", {}) if isinstance(rep.get("safe_selection_diagnostics", {}), dict) else {}
                evals = latest.get("candidate_evaluations", []) if isinstance(latest.get("candidate_evaluations", []), list) else []
                v2_summary = {}
                baseline_summary = {
                    "admitted_trades": int(latest.get("baseline_admitted_trades", 0) or 0),
                    "metrics": latest.get("baseline_metrics", {}),
                    "full_metrics": latest.get("baseline_full_metrics", {}),
                }
                for ev in evals:
                    if _s(ev.get("variant", "")) == "label_compatible_v2":
                        v2_summary = ev.get("summary", {}) if isinstance(ev.get("summary", {}), dict) else {}
                        break
                repeat_results.append(
                    {
                        "repeat": int(idx + 1),
                        "rows_hash": _s(rep.get("meta", {}).get("rows_hash", "")) if isinstance(rep.get("meta", {}), dict) else "",
                        "replay_rows": int(rep.get("meta", {}).get("closed_trades_total", 0) or 0) if isinstance(rep.get("meta", {}), dict) else 0,
                        "full_test_trades": int(rep.get("meta", {}).get("full_test_trades", 0) or 0) if isinstance(rep.get("meta", {}), dict) else 0,
                        "selected_mode": _s(latest.get("selected_predictor_variant", "")),
                        "selection_result": "v2" if _s(latest.get("selected_predictor_variant", "")) == "label_compatible_v2" else "baseline",
                        "guardrail_failures": list(next((ev.get("guardrail_failures", []) for ev in evals if _s(ev.get("variant", "")) == "label_compatible_v2"), [])),
                        "baseline_metrics": baseline_summary.get("metrics", {}),
                        "v2_metrics": v2_summary.get("metrics", {}),
                        "baseline_admitted_trades": int(baseline_summary.get("admitted_trades", 0) or 0),
                        "v2_admitted_trades": int(v2_summary.get("admitted_trades", 0) or 0),
                        "v2_admission_rate_pct": _f(v2_summary.get("admission_rate_pct", 0.0), 0.0),
                    }
                )
            selected_modes = {_s(r.get("selected_mode", "")) for r in repeat_results}
            rows_hashes = {_s(r.get("rows_hash", "")) for r in repeat_results}
            metric_signatures = {
                json.dumps(
                    {
                        "selected_mode": _s(r.get("selected_mode", "")),
                        "v2_admitted_trades": int(r.get("v2_admitted_trades", 0) or 0),
                        "v2_admission_rate_pct": round(_f(r.get("v2_admission_rate_pct", 0.0), 0.0), 6),
                        "guardrail_failures": list(r.get("guardrail_failures", [])),
                    },
                    sort_keys=True,
                )
                for r in repeat_results
            }
            payload.setdefault("crypto_classifier_diagnostics", {})
            payload["crypto_classifier_diagnostics"]["crypto_replay_repeat_evals"] = int(crypto_repeat_evals)
            payload["crypto_classifier_diagnostics"]["crypto_replay_repeat_eval_results"] = repeat_results
            payload["crypto_classifier_diagnostics"]["crypto_replay_repeat_activation_consistent"] = bool(len(selected_modes) == 1)
            payload["crypto_classifier_diagnostics"]["crypto_replay_repeat_metrics_consistent"] = bool(len(rows_hashes) == 1 and len(metric_signatures) == 1)
            payload["crypto_classifier_diagnostics"]["crypto_v2_promotion_eligible"] = bool(
                repeat_results
                and len(selected_modes) == 1
                and "label_compatible_v2" in selected_modes
                and len(rows_hashes) == 1
                and len(metric_signatures) == 1
            )
            payload.setdefault("replay_source_diagnostics", {})
            payload["replay_source_diagnostics"]["crypto_replay_repeat_evals"] = int(crypto_repeat_evals)
            payload["replay_source_diagnostics"]["crypto_replay_repeat_eval_results"] = repeat_results
            payload["replay_source_diagnostics"]["crypto_replay_repeat_activation_consistent"] = bool(len(selected_modes) == 1)
            payload["replay_source_diagnostics"]["crypto_replay_repeat_metrics_consistent"] = bool(len(rows_hashes) == 1 and len(metric_signatures) == 1)
            payload["replay_source_diagnostics"]["crypto_v2_promotion_eligible"] = bool(
                repeat_results
                and len(selected_modes) == 1
                and "label_compatible_v2" in selected_modes
                and len(rows_hashes) == 1
                and len(metric_signatures) == 1
            )
        if isinstance(payload, dict) and isinstance(replay_generation.get(m, {}), dict):
            merged_replay_diag = dict(replay_generation.get(m, {}))
            if isinstance(payload.get("replay_source_diagnostics", {}), dict):
                merged_replay_diag.update(payload.get("replay_source_diagnostics", {}))
            payload["replay_source_diagnostics"] = merged_replay_diag
            if isinstance(dataset_reports.get(m, {}), dict):
                dataset_reports[m]["replay_source_diagnostics"] = dict(merged_replay_diag)
        if m == "crypto" and isinstance(payload, dict):
            payload["trained_artifact_diagnostics"] = crypto_artifact_report.get("discovery", {})
            payload["feature_source_diagnostics"] = (
                (dataset_reports.get(m, {}).get("feature_source_diagnostics", {}) if isinstance(dataset_reports.get(m, {}), dict) else {})
                or crypto_artifact_report.get("feature_source", {})
            )
            crypto_diag = payload.get("crypto_classifier_diagnostics", {}) if isinstance(payload.get("crypto_classifier_diagnostics", {}), dict) else {}
            source_diag = payload.get("feature_source_diagnostics", {}) if isinstance(payload.get("feature_source_diagnostics", {}), dict) else {}
            crypto_diag["crypto_feature_source_priority"] = source_diag.get(
                "crypto_feature_source_priority",
                [],
            )
            crypto_diag["crypto_feature_source_used"] = source_diag.get(
                "crypto_feature_source_used",
                "closed_trade_only",
            )
            crypto_diag["feature_source_rows"] = source_diag.get("feature_source_rows", {})
            crypto_diag["fallback_reason"] = source_diag.get("fallback_reason", {})
            crypto_diag["missing_feature_reason"] = source_diag.get("missing_feature_reason", "")
            payload["crypto_classifier_diagnostics"] = crypto_diag
        rpath = os.path.join(openai_dir, f"{m}_historical_replay_synthetic.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(payload if isinstance(payload, dict) else {}, f, indent=2)
        synthetic_paths[m] = rpath

    regimes = build_all_market_regimes(hub_dir)
    walk = build_walkforward_report(hub_dir)
    calibration = build_confidence_calibration_payload(hub_dir, cfg)
    shadow = build_shadow_scorecards(hub_dir)
    replay_diag = {m: build_replay_diagnostics(hub_dir, m) for m in markets}
    legacy_trade_replay = build_legacy_trade_model_replay(hub_dir=hub_dir, base_dir=base_dir, settings=cfg)
    blind_simulation = _build_historical_blind_simulation(
        hub_dir=hub_dir,
        base_dir=base_dir,
        settings=cfg,
    )
    for m in markets:
        replay_generation.setdefault(m, {})
        blind_market = blind_simulation.get("markets", {}).get(m, {}) if isinstance(blind_simulation.get("markets", {}), dict) else {}
        blind_summary = blind_market.get("summary", {}) if isinstance(blind_market.get("summary", {}), dict) else {}
        replay_generation[m]["historical_blind_simulation_used_as_primary"] = False
        replay_generation[m]["historical_blind_simulation_used_as_supplemental"] = bool(blind_summary.get("completed_mock_trades", 0))
        replay_generation[m]["historical_blind_simulation_rows_generated"] = int(_f(blind_summary.get("completed_mock_trades", 0), 0.0))
        replay_generation[m]["historical_blind_simulation_skipped_decisions"] = int(_f(blind_summary.get("skipped_decisions", 0), 0.0))
        replay_generation[m]["historical_blind_simulation_training_eligible_rows"] = int(_f(blind_summary.get("training_eligible_rows", 0), 0.0))
        replay_generation[m]["historical_blind_simulation_leakage_check_status"] = (
            blind_summary.get("leakage_check_status", {}) if isinstance(blind_summary.get("leakage_check_status", {}), dict) else {}
        )
        replay_generation[m]["historical_blind_simulation_artifact_paths"] = (
            blind_market.get("artifact_paths", {}) if isinstance(blind_market.get("artifact_paths", {}), dict) else {}
        )
        replay_generation[m]["historical_blind_simulation_symbol_bucket_artifacts"] = (
            blind_market.get("symbol_bucket_artifacts", {}) if isinstance(blind_market.get("symbol_bucket_artifacts", {}), dict) else {}
        )
        replay_generation[m]["historical_blind_simulation_blocker"] = _s(blind_summary.get("blocker", blind_market.get("blocker", "")))

    promotion_thresholds = {"directional_accuracy_pct": 90.0, "trigger_match_pct": 90.0, "pnl_trend_match_pct": 90.0}
    promotion_minima = {
        "ci_lower_bound_pct": 85.0,
        "worst_window_min_pct": 85.0,
        "min_windows": 3,
        "min_test_trades": 40,
        "min_trigger_scored_trades": 20,
        "min_trigger_coverage_pct": 40.0,
    }
    controlled_rollout_readiness: Dict[str, Any] = {}
    promotion_readiness: Dict[str, Any] = {}
    for m in markets:
        row = replay_diag.get(m, {}) if isinstance(replay_diag.get(m, {}), dict) else {}
        metrics = row.get("headline_metrics", {}) if isinstance(row.get("headline_metrics", {}), dict) else {}
        payload = _safe_read_json(_s(row.get("source", "")))
        ci = payload.get("confidence_intervals", {}) if isinstance(payload.get("confidence_intervals", {}), dict) else {}
        worst = payload.get("worst_window_metrics", {}) if isinstance(payload.get("worst_window_metrics", {}), dict) else {}
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        population_diag = payload.get("population_diagnostics", {}) if isinstance(payload.get("population_diagnostics", {}), dict) else {}
        n_windows = int(_f(meta.get("walkforward_windows", 0), 0.0))
        n_test = int(_f(meta.get("test_trades", 0), 0.0))
        n_full_test = int(_f(meta.get("full_test_trades", n_test), 0.0))
        trig_scored = _f(metrics.get("trigger_scored_trades", 0.0), 0.0)
        trig_cov = _f(metrics.get("trigger_coverage_pct", 0.0), 0.0)
        admission_rate = _f(population_diag.get("admission_rate_pct", 100.0 if n_test > 0 else 0.0), 0.0)
        blockers: List[str] = []
        for key, need in promotion_thresholds.items():
            got = _f(metrics.get(key, 0.0), 0.0)
            if got < float(need):
                blockers.append(f"{key}_below_target({got:.2f}<{need:.2f})")
            row_ci = ci.get(key, {}) if isinstance(ci.get(key, {}), dict) else {}
            ci_lb = _f(row_ci.get("p05", 0.0), 0.0)
            if ci_lb < float(promotion_minima["ci_lower_bound_pct"]):
                blockers.append(
                    f"{key}_ci_p05_below_floor({ci_lb:.2f}<{float(promotion_minima['ci_lower_bound_pct']):.2f})"
                )
            worst_val = _f(worst.get(key, 0.0), 0.0)
            if worst_val < float(promotion_minima["worst_window_min_pct"]):
                blockers.append(
                    f"{key}_worst_window_below_floor({worst_val:.2f}<{float(promotion_minima['worst_window_min_pct']):.2f})"
                )
        if n_windows < int(promotion_minima["min_windows"]):
            blockers.append(f"walkforward_windows_insufficient({n_windows}<{int(promotion_minima['min_windows'])})")
        if n_test < int(promotion_minima["min_test_trades"]):
            blockers.append(f"test_trades_insufficient({n_test}<{int(promotion_minima['min_test_trades'])})")
        if trig_scored < float(promotion_minima["min_trigger_scored_trades"]):
            blockers.append(
                f"trigger_scored_trades_insufficient({trig_scored:.0f}<{float(promotion_minima['min_trigger_scored_trades']):.0f})"
            )
        if trig_cov < float(promotion_minima["min_trigger_coverage_pct"]):
            blockers.append(
                f"trigger_coverage_insufficient({trig_cov:.2f}%<{float(promotion_minima['min_trigger_coverage_pct']):.2f}%)"
            )
        if m == "crypto" and admission_rate < float(promotion_minima["min_trigger_coverage_pct"]):
            blockers.append(
                f"admission_rate_low({admission_rate:.2f}%<{float(promotion_minima['min_trigger_coverage_pct']):.2f}%)"
            )
        state = "PASS" if not blockers and _s(row.get("state", "")) == "READY" else "BLOCK"
        rollout = _controlled_rollout_status(
            market=m,
            metrics=metrics,
            ci=ci,
            worst=worst,
            windows=n_windows,
            test_trades=n_test,
            trigger_scored=trig_scored,
            trigger_cov=trig_cov,
            admission_rate=admission_rate,
            payload=payload,
        )
        controlled_rollout_readiness[m] = {
            "eligible": bool(rollout.get("eligible", False)),
            "reason": _s(rollout.get("reason", "")),
            "blockers": list(rollout.get("blockers", []) or []),
            "risk_multiplier_recommended": float(_f(rollout.get("risk_multiplier_recommended", 0.0), 0.0)),
            "thresholds": dict(rollout.get("thresholds", {}) if isinstance(rollout.get("thresholds", {}), dict) else {}),
        }
        promotion_readiness[m] = {
            "state": state,
            "blockers": blockers,
            "metrics": metrics,
            "source": _s(row.get("source", "")),
            "windows": int(n_windows),
            "test_trades": int(n_test),
            "full_test_trades": int(n_full_test),
            "admission_rate_pct": round(float(admission_rate), 4),
            "controlled_rollout_eligible": bool(rollout.get("eligible", False)),
            "controlled_rollout_reason": _s(rollout.get("reason", "")),
            "risk_multiplier_recommended": float(_f(rollout.get("risk_multiplier_recommended", 0.0), 0.0)),
            "full_promotion_eligible": bool(state == "PASS"),
            "full_promotion_blockers": list(blockers),
        }

    market_readiness = _build_market_readiness_artifact(
        base_dir=base_dir,
        hub_dir=hub_dir,
        settings=cfg,
        replay_generation=replay_generation,
        replay_diag=replay_diag,
        promotion_readiness=promotion_readiness,
        completed_live_artifacts=completed_live_decision_artifacts,
        existing_runtime_settings=existing_runtime_settings,
    )

    return {
        "ts": ts,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
        "base_dir": base_dir,
        "hub_dir": hub_dir,
        "existing_runtime_settings": existing_runtime_settings,
        "dataset_quality": dataset_reports,
        "dataset_snapshots": snapshots,
        "completed_live_decision_artifacts": completed_live_decision_artifacts,
        "completed_live_verification": completed_live_verification,
        "completed_live_synthetic_verification": synthetic_completed_live_verification,
        "synthetic_replay_artifacts": synthetic_paths,
        "replay_generation": replay_generation,
        "market_regimes": regimes,
        "walkforward_report": walk,
        "confidence_calibration": calibration,
        "shadow_scorecards": shadow,
        "replay_diagnostics": replay_diag,
        "legacy_trade_model_replay": legacy_trade_replay,
        "legacy_trade_model_replay_used_as_primary": bool(use_legacy_replay_as_primary and legacy_trade_replay.get("legacy_rows_replayed", 0)),
        "legacy_trade_model_replay_used_as_supplemental": True,
        "historical_blind_simulation": blind_simulation,
        "crypto_trained_artifact_diagnostics": crypto_artifact_report.get("discovery", {}),
        "crypto_feature_source_diagnostics": (
            dataset_reports.get("crypto", {}).get("feature_source_diagnostics", {})
            if isinstance(dataset_reports.get("crypto", {}), dict)
            else crypto_artifact_report.get("feature_source", {})
        ),
        "promotion_thresholds": promotion_thresholds,
        "promotion_minima": promotion_minima,
        "controlled_rollout_readiness": controlled_rollout_readiness,
        "promotion_readiness": promotion_readiness,
        "model_quality_market_readiness": market_readiness,
    }
