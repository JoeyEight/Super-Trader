from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.confidence_calibration import build_confidence_calibration_payload
from app.crypto_artifacts import discover_crypto_trained_artifacts, load_crypto_artifact_features
from app.crypto_historical_replay import build_crypto_historical_strategy_replay
from app.credential_utils import get_alpaca_creds, get_twelvedata_api_key
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
    found = 0
    completed = 0
    if mk != "crypto":
        return [], {
            "live_decision_source_available": False,
            "live_decision_rows_found": 0,
            "live_decision_rows_completed": 0,
            "live_decision_join_rate_pct": 0.0,
            "live_decision_rows_by_market": by_market,
            "live_decision_rows_by_predictor": {},
            "live_decision_missing_reason": "completed_live_decision_snapshots_not_available_for_market",
            "model_quality_source_priority_used": "",
        }
    for row in list(closed_rows or []):
        if not isinstance(row, dict):
            continue
        pred_dir = _s(row.get("predicted_direction", "")) or _predicted_direction_from_action(_s(row.get("entry_snapshot_selected_action", "")), mk)
        pred_trigger = _s(row.get("predicted_exit_trigger", "")) or _s(row.get("entry_snapshot_normalized_trigger", ""))
        pred_conf = _extract_live_prediction_field(row, ("predicted_confidence", "entry_snapshot_ai_confidence", "entry_snapshot_strategy_score"))
        predictor_name = _s(_extract_live_prediction_field(row, ("predictor_name", "entry_snapshot_policy_mode", "entry_snapshot_profile"))) or "live_snapshot"
        predictor_variant = _s(_extract_live_prediction_field(row, ("predictor_variant", "entry_snapshot_entry_alignment_mode", "entry_snapshot_signal_gate_mode"))) or "live"
        trigger_scores = _extract_live_prediction_field(row, ("trigger_scores",))
        direction_scores = _extract_live_prediction_field(row, ("direction_scores",))
        if pred_dir or pred_trigger or pred_conf is not None:
            found += 1
        if not pred_dir or pred_conf is None:
            continue
        completed += 1
        item = dict(row)
        item["predicted_direction"] = pred_dir
        item["predicted_exit_trigger"] = pred_trigger or "Unknown"
        item["predicted_confidence"] = round(_f(pred_conf, 0.0), 6)
        if isinstance(trigger_scores, dict):
            item["trigger_scores"] = dict(trigger_scores)
        if isinstance(direction_scores, dict):
            item["direction_scores"] = dict(direction_scores)
        item["selected_predictor_name"] = predictor_name
        item["predictor_mode"] = predictor_name
        item["predictor_variant"] = predictor_variant
        item["source_type"] = "completed_live_decision_snapshot"
        item["raw_rule_reason"] = _s(row.get("entry_snapshot_raw_rule_reason", "")) or _s(row.get("raw_rule_reason", ""))
        item["event_exit_tag"] = _s(row.get("event_exit_tag", ""))
        item["actual_direction"] = _trade_direction(item)
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
        "live_decision_missing_reason": missing_reason,
        "model_quality_source_priority_used": "completed_live_decision_snapshot" if rows_out else "",
    }


def _closed_trades_from_exits(events: Iterable[Dict[str, Any]], market: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(events or []):
        if not isinstance(row, dict) or _s(row.get("event", "")).lower() != "exit":
            continue
        ts = int(_f(row.get("ts", 0.0), 0.0))
        hold_s = int(max(0.0, _f(row.get("hold_s", 0.0), 0.0)))
        sym = _s(row.get("symbol", "")).upper()
        qty = max(0.0, _f(row.get("qty", 0.0), 0.0))
        exit_px = _f(row.get("price", 0.0), 0.0)
        if ts <= 0 or (not sym) or qty <= 0.0 or exit_px <= 0.0:
            continue
        entry_px = _derive_entry_price_from_exit(row, market)
        if entry_px <= 0.0:
            continue
        entry_ts = int(ts - hold_s) if hold_s > 0 else int(max(0, ts - 3600))
        pnl_usd = (exit_px - entry_px) * qty
        if _s(market).lower() == "forex" and _s(row.get("side", "")).lower() != "long":
            pnl_usd = (entry_px - exit_px) * qty
        out.append(
            {
                "symbol": sym,
                "market": _s(market).lower(),
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
            }
        )
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
    use_exit_shape = variant in {"candidate", "active", "exit_shape_enhanced", "label_compatible_v2"}
    use_label_compatible_v2 = variant == "label_compatible_v2"
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

    if use_label_compatible_v2:
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
        "exit_shape_predictive_mode": "active" if variant == "label_compatible_v2" else ("active" if use_exit_shape else "diagnostic_only"),
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
    stats = _weighted_trade_stats(filtered)
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
    exit_px = entry_px * (1.0 + abs(med_ret)) if pred_dir == "up" else entry_px * (1.0 - abs(med_ret))
    conf = max(0.18, min(0.95, 0.20 + (0.30 * max(up_score, down_score)) + (0.25 * max(trigger_scores.values()))))
    sorted_triggers = sorted(trigger_scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "predicted_direction": pred_dir,
        "predicted_exit_trigger": pred_trigger,
        "predicted_hold_hours": round(float(hold_h), 6),
        "predicted_exit_price": round(float(max(1e-12, exit_px)), 10),
        "predicted_confidence": round(float(conf), 6),
        "trigger_scores": {k: round(float(v), 6) for k, v in trigger_scores.items()},
        "direction_scores": {"up": round(float(up_score), 6), "down": round(float(down_score), 6)},
        "trigger_margin": round(float(sorted_triggers[0][1] - sorted_triggers[1][1] if len(sorted_triggers) > 1 else sorted_triggers[0][1]), 6),
        "direction_margin": round(float(abs(up_score - down_score)), 6),
        "stock_direction_score_up": round(float(up_score), 6),
        "stock_direction_score_down": round(float(down_score), 6),
        "stock_direction_score_gap": round(float(up_score - down_score), 6),
        "predictor_mode": "stock_historical_replay",
        "predictor_variant": _s(predictor_variant).lower() or "candidate",
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
        predictor_variants = ["candidate"]
        if m == "crypto" and any(_is_historical_strategy_replay_row(r) for r in train):
            predictor_variants = ["baseline", "candidate", "label_compatible_v2"]
        elif m == "forex":
            predictor_variants = ["baseline", "candidate"]
        elif m == "stocks":
            predictor_variants = ["candidate"]
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
                merged["predicted_exit_ts"] = int(
                    _f(rr.get("entry_ts", 0.0), 0.0) + int(round(_f(pred.get("predicted_hold_hours", 0.0), 0.0) * 3600.0))
                )
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
        latest_selection = safe_selection_snapshots[-1] if safe_selection_snapshots else {}
        crypto_diag["safe_selection"] = latest_selection
        crypto_diag["exit_shape_predictive_mode"] = (
            "active" if _s(latest_selection.get("selected_predictor_variant", "")) == "label_compatible_v2" else "diagnostic_only"
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
    ts = int(time.time())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    out_dir = os.path.join(hub_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)

    markets = ["crypto", "stocks", "forex"]
    dataset_reports: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    closed_by_market: Dict[str, List[Dict[str, Any]]] = {}
    replay_generation: Dict[str, Any] = {}
    crypto_artifact_report: Dict[str, Any] = {}
    for m in markets:
        loaded = load_market_trade_events(hub_dir, m)
        events = loaded.get("events", []) if isinstance(loaded.get("events", []), list) else []
        closed = build_closed_trades(events, m).get("closed_trades", [])
        live_rows, live_diag = _completed_live_decision_rows(market=m, events=events, closed_rows=list(closed or []))
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

    return {
        "ts": ts,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
        "base_dir": base_dir,
        "hub_dir": hub_dir,
        "dataset_quality": dataset_reports,
        "dataset_snapshots": snapshots,
        "synthetic_replay_artifacts": synthetic_paths,
        "replay_generation": replay_generation,
        "market_regimes": regimes,
        "walkforward_report": walk,
        "confidence_calibration": calibration,
        "shadow_scorecards": shadow,
        "replay_diagnostics": replay_diag,
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
    }
