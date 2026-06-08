from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List


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


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _s(f.read())
    except Exception:
        return ""


def _parse_memory_rows(raw: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for row in str(raw or "").split("~"):
        txt = _s(row)
        if not txt:
            continue
        parts = txt.split("{}")
        if len(parts) < 3:
            continue
        memory_pattern = parts[0].replace("'", "").replace(",", "").replace('"', "").replace("]", "").replace("[", "").split()
        if len(memory_pattern) < 2:
            continue
        try:
            out.append(
                {
                    "memory_candle": float(memory_pattern[0]),
                    "move_pct": float(memory_pattern[-1]),
                    "high_diff": float(parts[1].replace("'", "").replace(",", "").replace('"', "").replace("]", "").replace("[", "").replace(" ", "")) / 100.0,
                    "low_diff": float(parts[2].replace("'", "").replace(",", "").replace('"', "").replace("]", "").replace("[", "").replace(" ", "")) / 100.0,
                }
            )
        except Exception:
            continue
    return out


def _parse_weight_list(raw: str, expected_len: int) -> List[float]:
    vals: List[float] = []
    for tok in str(raw or "").replace("'", "").replace(",", "").replace('"', "").replace("]", "").replace("[", "").split():
        try:
            vals.append(abs(float(tok)))
        except Exception:
            continue
    if len(vals) < expected_len:
        vals.extend([1.0] * (expected_len - len(vals)))
    return vals[:expected_len]


def _weighted_mean(vals: Iterable[float], weights: Iterable[float]) -> float:
    num = 0.0
    den = 0.0
    for value, weight in zip(vals, weights):
        ww = _f(weight, 0.0)
        if ww <= 0.0:
            continue
        num += _f(value, 0.0) * ww
        den += ww
    return (num / den) if den > 0.0 else 0.0


def derive_crypto_original_signal_dry_run(
    *,
    symbol: str,
    current_price: float,
    timeframe_predictions: list[dict],
    thresholds: dict | None = None,
) -> dict:
    audit = {
        "bound_signal_dry_run_available": True,
        "bound_signal_dry_run_used": True,
        "bound_signal_replay_safe": True,
        "bound_signal_called_robinhood": False,
        "bound_signal_called_kucoin_live": False,
        "bound_signal_wrote_live_signal_files": False,
        "bound_signal_wrote_bound_files": False,
        "bound_signal_changed_cwd": False,
        "bound_signal_imported_pt_thinker": False,
        "bound_signal_blockers": [],
    }
    distance_pct = _f((thresholds or {}).get("distance_pct", 0.0), 0.0)
    preds = [dict(p) for p in list(timeframe_predictions or []) if isinstance(p, dict)]
    messages: List[str] = []
    tf_sides: List[str] = []
    margins: List[float] = []
    per_timeframe: List[Dict[str, Any]] = []
    for pred in preds:
        timeframe = _s(pred.get("timeframe", "")) or "1hour"
        active_state = _s(pred.get("active_model_state", "")).lower() or "inactive"
        predicted_low = _f(pred.get("predicted_low_boundary", pred.get("low_new_price", 0.0)), 0.0)
        predicted_high = _f(pred.get("predicted_high_boundary", pred.get("high_new_price", 0.0)), 0.0)
        if predicted_low <= 0.0:
            predicted_low = _f(pred.get("low_new_price", 0.0), 0.0)
        if predicted_high <= 0.0:
            predicted_high = _f(pred.get("high_new_price", 0.0), 0.0)
        low_tf_price = _f(pred.get("low_new_price", predicted_low if predicted_low > 0.0 else current_price), current_price)
        high_tf_price = _f(pred.get("high_new_price", predicted_high if predicted_high > 0.0 else current_price), current_price)
        if active_state != "active":
            low_bound = 0.01
            high_bound = 99999999999999999.0
        else:
            low_bound = predicted_low - (predicted_low * (distance_pct / 100.0)) if predicted_low > 0.0 else 0.01
            high_bound = predicted_high + (predicted_high * (distance_pct / 100.0)) if predicted_high > 0.0 else 99999999999999999.0
        side = "none"
        margin = 0.0
        message_type = "INACTIVE" if active_state != "active" else "WITHIN"
        if active_state == "active" and current_price > high_bound and abs(high_tf_price - low_tf_price) > 1e-12:
            side = "short"
            margin = ((high_tf_price - current_price) / abs(current_price)) * 100.0 if current_price != 0.0 else 0.0
            message_type = "SHORT"
            message = "SHORT on " + timeframe + " timeframe. " + format(((high_bound - current_price) / abs(current_price)) * 100.0 if current_price != 0.0 else 0.0, ".8f") + " High Boundary: " + str(high_bound)
        elif active_state == "active" and current_price < low_bound and abs(high_tf_price - low_tf_price) > 1e-12:
            side = "long"
            margin = ((low_tf_price - current_price) / abs(current_price)) * 100.0 if current_price != 0.0 else 0.0
            message_type = "LONG"
            message = "LONG on " + timeframe + " timeframe. " + format(((low_bound - current_price) / abs(current_price)) * 100.0 if current_price != 0.0 else 0.0, ".8f") + " Low Boundary: " + str(low_bound)
        else:
            if active_state != "active":
                message = "INACTIVE on " + timeframe + " timeframe." + " Low Boundary: " + str(low_bound) + " High Boundary: " + str(high_bound)
            else:
                message = "WITHIN on " + timeframe + " timeframe." + " Low Boundary: " + str(low_bound) + " High Boundary: " + str(high_bound)
        messages.append(message)
        tf_sides.append(side)
        margins.append(float(margin))
        per_timeframe.append(
            {
                "timeframe": timeframe,
                "message": message,
                "message_type": message_type,
                "side": side,
                "margin": round(float(margin), 6),
                "low_bound": round(float(low_bound), 10),
                "high_bound": round(float(high_bound), 10),
                "active_model_state": active_state,
            }
        )
    longs = int(sum(1 for s in tf_sides if s == "long"))
    shorts = int(sum(1 for s in tf_sides if s == "short"))
    current_pms = [float(m) for m in margins if abs(_f(m, 0.0)) > 1e-12]
    pm = sum(current_pms) / len(current_pms) if current_pms else 0.25
    if pm < 0.25:
        pm = 0.25
    if longs > shorts:
        signal_side = "long"
        message_type = "LONG"
    elif shorts > longs:
        signal_side = "short"
        message_type = "SHORT"
    elif any(_s(p.get("active_model_state", "")).lower() == "active" for p in per_timeframe):
        signal_side = "within"
        message_type = "WITHIN"
    else:
        signal_side = "inactive"
        message_type = "INACTIVE"
    return {
        "symbol": _s(symbol).upper(),
        "timeframe_predictions": per_timeframe,
        "messages": list(messages),
        "tf_sides": list(tf_sides),
        "margins": [round(float(m), 6) for m in margins],
        "original_signal_side": signal_side,
        "original_signal_margin": round(float(sum(current_pms) / len(current_pms) if current_pms else 0.0), 6),
        "original_long_signal_count": longs,
        "original_short_signal_count": shorts,
        "original_long_profit_margin": round(float(pm), 6),
        "original_short_profit_margin": round(float(abs(pm)), 6),
        "original_message_type": message_type,
        "original_bound_position": signal_side,
        "original_signal_active": bool(longs > 0 or shorts > 0),
        "original_bounds_active": bool(any(_s(p.get("active_model_state", "")).lower() == "active" for p in per_timeframe)),
        "original_trigger_semantics_status": "unavailable",
        "trigger_semantics_blocker": "original_artifacts_do_not_encode_exit_trigger_class",
        "predicted_exit_trigger": "Unknown",
        **audit,
    }


def predict_crypto_original_dry_run(
    *,
    symbol: str,
    timeframe: str,
    candle_snapshot: list,
    artifact_dir: str,
    as_of_ts: int,
    thresholds: dict | None = None,
) -> dict:
    default_audit = {
        "original_predictor_dry_run_available": False,
        "original_predictor_dry_run_used": False,
        "original_predictor_replay_safe": True,
        "original_predictor_imported_pt_thinker": False,
        "original_predictor_called_step_coin": False,
        "original_predictor_called_robinhood": False,
        "original_predictor_called_kucoin_live": False,
        "original_predictor_wrote_live_signal_files": False,
        "original_predictor_changed_cwd": False,
        "original_predictor_mutated_live_artifacts": False,
        "original_predictor_blockers": [],
    }
    out: Dict[str, Any] = {
        "symbol": _s(symbol).upper(),
        "timeframe": _s(timeframe) or "1hour",
        "as_of_ts": int(_f(as_of_ts, 0.0)),
        "selected_predictor": "original_crypto_artifact_model",
        "selected_predictor_name": "original_crypto_artifact_model",
        "predictor_variant": "original_strategy_dry_run_v1",
        "source_used": "trained_artifact_original_memory_files",
        "prediction_semantics": "original_strategy_dry_run_direction_only",
        "prediction_semantics_warning": "",
        "predicted_exit_trigger": "Unknown",
        "trigger_scores": {},
        **default_audit,
    }
    folder = _s(artifact_dir)
    rows = [list(r) for r in list(candle_snapshot or []) if isinstance(r, (list, tuple)) and len(r) >= 5]
    if len(rows) <= 0:
        out["original_predictor_blockers"] = ["missing_candle_snapshot"]
        return out
    last = rows[-1]
    try:
        open_px = float(last[1])
        close_px = float(last[2])
    except Exception:
        out["original_predictor_blockers"] = ["invalid_latest_candle"]
        return out
    if open_px <= 0.0 or close_px <= 0.0:
        out["original_predictor_blockers"] = ["non_positive_latest_candle"]
        return out
    tf = _s(timeframe) or "1hour"
    required = {
        "threshold": os.path.join(folder, f"neural_perfect_threshold_{tf}.txt"),
        "memories": os.path.join(folder, f"memories_{tf}.txt"),
        "weights": os.path.join(folder, f"memory_weights_{tf}.txt"),
        "weights_high": os.path.join(folder, f"memory_weights_high_{tf}.txt"),
        "weights_low": os.path.join(folder, f"memory_weights_low_{tf}.txt"),
    }
    missing = [name for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        out["original_predictor_blockers"] = [f"missing_artifact:{name}" for name in missing]
        return out
    try:
        perfect_threshold = float(_safe_read_text(required["threshold"]) or "0")
    except Exception:
        out["original_predictor_blockers"] = ["invalid_threshold_file"]
        return out
    memory_rows = _parse_memory_rows(_safe_read_text(required["memories"]))
    if not memory_rows:
        out["original_predictor_blockers"] = ["empty_memory_rows"]
        return out
    move_weights = _parse_weight_list(_safe_read_text(required["weights"]), len(memory_rows))
    high_weights = _parse_weight_list(_safe_read_text(required["weights_high"]), len(memory_rows))
    low_weights = _parse_weight_list(_safe_read_text(required["weights_low"]), len(memory_rows))
    current_candle = 100.0 * ((close_px - open_px) / open_px)
    candidates: List[Dict[str, float]] = []
    for idx, row in enumerate(memory_rows):
        memory_candle = _f(row.get("memory_candle", 0.0), 0.0)
        if current_candle == 0.0 and memory_candle == 0.0:
            diff_avg = 0.0
        else:
            try:
                diff_avg = abs((abs(current_candle - memory_candle) / ((current_candle + memory_candle) / 2.0)) * 100.0)
            except Exception:
                diff_avg = 0.0
        candidates.append(
            {
                "diff": float(diff_avg),
                "move_pct": _f(row.get("move_pct", 0.0), 0.0),
                "high_diff": _f(row.get("high_diff", 0.0), 0.0),
                "low_diff": _f(row.get("low_diff", 0.0), 0.0),
                "w_move": _f(move_weights[idx], 1.0),
                "w_high": _f(high_weights[idx], 1.0),
                "w_low": _f(low_weights[idx], 1.0),
            }
        )
    perfect = sorted([c for c in candidates if c["diff"] <= float(perfect_threshold)], key=lambda c: c["diff"])
    sorted_all = sorted(candidates, key=lambda c: c["diff"])
    min_perfect_count = 3
    top_k_perfect = 24
    top_k_fallback = 12
    shrink = 1.0
    selected = perfect[:top_k_perfect]
    active_model = len(selected) >= min_perfect_count
    fallback_used = False
    if not active_model:
        selected = sorted_all[:top_k_fallback]
        avg_diff = (sum(c["diff"] for c in selected) / len(selected)) if selected else 999.0
        active_model = (len(selected) >= 6) and (avg_diff <= max(1.0, float(perfect_threshold) * 1.5))
        shrink = 0.35
        fallback_used = True
    else:
        avg_diff = (sum(c["diff"] for c in selected) / len(selected)) if selected else 0.0
    sim = [1.0 / (1.0 + max(0.0, float(c["diff"]))) for c in selected]
    move_ws = [max(0.001, float(c["w_move"])) * s for c, s in zip(selected, sim)]
    high_ws = [max(0.001, float(c["w_high"])) * s for c, s in zip(selected, sim)]
    low_ws = [max(0.001, float(c["w_low"])) * s for c, s in zip(selected, sim)]
    final_moves = _weighted_mean([c["move_pct"] for c in selected], move_ws) * shrink if selected else 0.0
    high_final_moves = _weighted_mean([c["high_diff"] for c in selected], high_ws) * shrink if selected else 0.0
    low_final_moves = _weighted_mean([c["low_diff"] for c in selected], low_ws) * shrink if selected else 0.0
    final_moves = _clamp(final_moves, -8.0, 8.0)
    high_final_moves = _clamp(high_final_moves, -0.08, 0.08)
    low_final_moves = _clamp(low_final_moves, -0.08, 0.08)
    start_price = float(close_px)
    high_new_price = start_price + (start_price * high_final_moves)
    low_new_price = start_price + (start_price * low_final_moves)
    edge_up_pct = ((high_new_price / start_price) - 1.0) * 100.0 if start_price > 0.0 else 0.0
    edge_down_pct = ((low_new_price / start_price) - 1.0) * 100.0 if start_price > 0.0 else 0.0
    directional_move = final_moves if abs(final_moves) > 1e-9 else ((edge_up_pct + edge_down_pct) / 2.0)
    predicted_direction = "up" if directional_move > 0.02 else "down" if directional_move < -0.02 else "flat"
    predicted_pnl_trend = "up" if predicted_direction == "up" else "down" if predicted_direction == "down" else "flat"
    up_score = max(0.0, directional_move, edge_up_pct)
    down_score = max(0.0, -directional_move, -edge_down_pct)
    direction_scores = {
        "up": round(float(up_score), 6),
        "down": round(float(down_score), 6),
    }
    confidence = _clamp(
        0.18
        + (0.22 if active_model else 0.0)
        + (0.12 if not fallback_used else 0.0)
        + min(0.22, max(0.0, abs(directional_move)) / 8.0)
        + min(0.18, max(0.0, 1.0 - (avg_diff / max(1.0, perfect_threshold if perfect_threshold > 0 else 1.0))) * 0.18),
        0.05,
        0.95,
    )
    pnl_quality_score = _clamp((abs(directional_move) / 4.0) + (0.20 if active_model else 0.0), 0.0, 1.0)
    trade_quality_score = _clamp((confidence * 0.7) + (0.15 if active_model else 0.0) + (0.10 if len(selected) >= 6 else 0.0), 0.0, 1.0)
    signal_layer = derive_crypto_original_signal_dry_run(
        symbol=symbol,
        current_price=start_price,
        timeframe_predictions=[
            {
                "timeframe": tf,
                "active_model_state": "active" if active_model else "inactive",
                "predicted_low_boundary": low_new_price,
                "predicted_high_boundary": high_new_price,
                "low_new_price": low_new_price,
                "high_new_price": high_new_price,
            }
        ],
        thresholds=thresholds,
    )
    semantics = "original_strategy_dry_run_direction_only"
    semantics_warning = "original_direction_and_bounds_from_artifacts_only_exit_trigger_class_not_encoded"
    out.update(
        {
            "original_predictor_dry_run_available": True,
            "original_predictor_dry_run_used": True,
            "current_candle_pct_move": round(float(current_candle), 6),
            "memory_rows_total": int(len(memory_rows)),
            "memory_rows_selected": int(len(selected)),
            "memory_rows_perfect_matches": int(len(perfect)),
            "perfect_threshold": round(float(perfect_threshold), 6),
            "selected_average_diff": round(float(avg_diff), 6),
            "active_model_state": "active" if active_model else "inactive",
            "active_model": bool(active_model),
            "fallback_used": bool(fallback_used),
            "final_moves": round(float(final_moves), 6),
            "high_final_moves": round(float(high_final_moves), 6),
            "low_final_moves": round(float(low_final_moves), 6),
            "high_new_price": round(float(high_new_price), 10),
            "low_new_price": round(float(low_new_price), 10),
            "predicted_direction": predicted_direction,
            "predicted_pnl_trend": predicted_pnl_trend,
            "predicted_confidence": round(float(confidence), 6),
            "confidence": round(float(confidence), 6),
            "direction_scores": direction_scores,
            "trade_quality_score": round(float(trade_quality_score), 6),
            "pnl_quality_score": round(float(pnl_quality_score), 6),
            "prediction_semantics": semantics,
            "prediction_semantics_warning": semantics_warning,
            "original_trigger_semantics_status": _s(signal_layer.get("original_trigger_semantics_status", "")) or "unavailable",
            "trigger_semantics_blocker": _s(signal_layer.get("trigger_semantics_blocker", "")) or "original_artifacts_do_not_encode_exit_trigger_class",
            "original_predictor_blockers": [],
            **signal_layer,
        }
    )
    return out
