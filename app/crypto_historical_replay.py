from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import requests

from app.crypto_artifacts import TRAINING_STALE_SECONDS, discover_crypto_trained_artifacts, load_crypto_artifact_features
from app.crypto_original_predictor import predict_crypto_original_dry_run
from app.trigger_normalization import normalize_exit_trigger


TIMEFRAME_MINUTES = {
    "1hour": 60,
    "2hour": 120,
    "4hour": 240,
    "8hour": 480,
    "12hour": 720,
    "1day": 1440,
    "1week": 10080,
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _performance_limits() -> Dict[str, int]:
    return {
        "max_symbols_per_replay": max(1, _env_int("MODEL_QUALITY_MAX_SYMBOLS_PER_REPLAY", 50)),
        "max_lookback_days": max(30, _env_int("MODEL_QUALITY_MAX_LOOKBACK_DAYS", 365)),
        "max_network_retry_attempts": max(1, _env_int("MODEL_QUALITY_MAX_NETWORK_RETRY_ATTEMPTS", 2)),
        "max_cached_candle_rows": max(200, _env_int("MODEL_QUALITY_MAX_CACHED_CANDLE_ROWS", 30000)),
    }


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


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _s(f.read())
    except Exception:
        return ""


def _safe_read_json_rows(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            return []
        return payload[-int(_performance_limits()["max_cached_candle_rows"]):]
    except Exception:
        return []


def _cache_root(hub_dir: str) -> str:
    return os.path.join(hub_dir, "crypto", "historical_replay_cache")


def _derive_cached_cutoff_ts(hub_dir: str, symbols: List[str], timeframe: str) -> int:
    latest = 0
    for symbol in list(symbols or []):
        rows = _load_cached_candles(hub_dir, symbol, timeframe)
        if rows:
            try:
                latest = max(latest, max(int(r[0]) for r in rows if isinstance(r, list) and len(r) >= 1))
            except Exception:
                continue
    return int(latest / 1000) if latest > 0 else 0


def _cache_candle_path(hub_dir: str, symbol: str, timeframe: str) -> str:
    base = _s(symbol).upper().split("-")[0]
    return os.path.join(_cache_root(hub_dir), "candles", base, f"{_s(timeframe)}.json")


def _symbol_dir(main_neural_dir: str, symbol: str) -> str:
    base = _s(symbol).upper().split("-")[0]
    return os.path.join(main_neural_dir, base)


def _parse_bound_file(path: str) -> List[float]:
    txt = _safe_read_text(path).replace(",", " ").replace("\n", " ")
    vals: List[float] = []
    for tok in txt.split():
        try:
            vals.append(float(tok))
        except Exception:
            continue
    return vals


def _artifact_context(main_neural_dir: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    folder = _symbol_dir(main_neural_dir, symbol)
    features = load_crypto_artifact_features(folder, as_of_ts=0)
    discovery = discover_crypto_trained_artifacts(main_neural_dir, symbols=[symbol])
    per_symbol = (discovery.get("per_symbol", {}) if isinstance(discovery.get("per_symbol", {}), dict) else {}).get(_s(symbol).upper().split("-")[0], {})
    paths = (per_symbol.get("paths", {}) if isinstance(per_symbol, dict) else {}) if isinstance(per_symbol, dict) else {}
    lows = _parse_bound_file(_s((paths.get("bounds", []) or [None])[0])) if len(paths.get("bounds", []) or []) > 0 else []
    highs = _parse_bound_file(_s((paths.get("bounds", []) or [None, None])[-1])) if len(paths.get("bounds", []) or []) > 0 else []
    low_bound = min(lows) if lows else 0.0
    high_bound = max(highs) if highs else 0.0
    long_signal = int(_f(_safe_read_text(os.path.join(folder, "long_dca_signal.txt")), 0.0))
    short_signal = int(_f(_safe_read_text(os.path.join(folder, "short_dca_signal.txt")), 0.0))
    long_margin = _f(_safe_read_text(os.path.join(folder, "futures_long_profit_margin.txt")), 0.0)
    short_margin = _f(_safe_read_text(os.path.join(folder, "futures_short_profit_margin.txt")), 0.0)
    side = "long" if (long_signal + max(0.0, long_margin)) >= (short_signal + max(0.0, short_margin)) else "short"
    signal_margin = (float(long_signal) - float(short_signal)) + (long_margin - short_margin)
    return {
        "trained_artifacts_found": int(discovery.get("trained_artifacts_found", 0) or 0),
        "trained_artifacts_fresh": bool(features.get("trained_freshness_flag", False)),
        "artifact_training_time": int(_f(features.get("trainer_last_training_time", 0), 0.0)),
        "artifact_paths_used": sorted(sum((list(v) for v in (paths or {}).values()), [])) if isinstance(paths, dict) else [],
        "artifact_timeframes_used": [timeframe] if timeframe in list(features.get("trained_timeframes", []) or []) else list(features.get("trained_timeframes", []) or []),
        "artifact_missing_reasons": [] if bool(features.get("usable", False)) else [_s(features.get("reason_if_not_used", "")) or "artifact_unusable"],
        "active_timeframe_count": int(_f(features.get("active_timeframe_count", 0), 0.0)),
        "predicted_low_boundary": round(float(low_bound), 10) if low_bound > 0.0 else 0.0,
        "predicted_high_boundary": round(float(high_bound), 10) if high_bound > 0.0 else 0.0,
        "signal_side": side,
        "signal_margin": round(float(signal_margin), 6),
        "long_signal_count": int(long_signal),
        "short_signal_count": int(short_signal),
        "long_profit_margin": round(float(long_margin), 6),
        "short_profit_margin": round(float(short_margin), 6),
        "threshold_mean": round(_f(features.get("threshold_mean", 0.0), 0.0), 6),
        "pattern_memory_count": int(_f(features.get("pattern_memory_count", 0.0), 0.0)),
        "usable": bool(features.get("usable", False)),
    }


def _normalize_kucoin_rows(rows: Iterable[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    by_ts: Dict[int, List[float]] = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            ts_ms = int(float(row[0]))
            if ts_ms < 1_000_000_000_000:
                ts_ms *= 1000
            open_px = float(row[1])
            close_px = float(row[2])
            high_px = float(row[3])
            low_px = float(row[4])
            vol = float(row[5]) if len(row) > 5 else 0.0
        except Exception:
            continue
        by_ts[ts_ms] = [ts_ms, open_px, close_px, high_px, low_px, vol]
    for ts in sorted(by_ts.keys()):
        out.append(by_ts[ts])
    return out


def _load_cached_candles(hub_dir: str, symbol: str, timeframe: str) -> List[List[float]]:
    return _normalize_kucoin_rows(_safe_read_json_rows(_cache_candle_path(hub_dir, symbol, timeframe)))


def _save_cached_candles(hub_dir: str, symbol: str, timeframe: str, rows: List[List[float]]) -> None:
    path = _cache_candle_path(hub_dir, symbol, timeframe)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    os.replace(tmp, path)


def _fetch_kucoin_candles(symbol: str, timeframe: str, start_ts_ms: int, end_ts_ms: int) -> List[List[float]]:
    tf_minutes = int(TIMEFRAME_MINUTES.get(timeframe, 60))
    interval_s = max(60, tf_minutes * 60)
    pair = f"{_s(symbol).upper().split('-')[0]}-USDT"
    session = requests.Session()
    out: List[List[float]] = []
    cursor_end = int(end_ts_ms / 1000)
    start_s = int(start_ts_ms / 1000)
    max_retries = int(_performance_limits()["max_network_retry_attempts"])
    while cursor_end >= start_s:
        cursor_start = max(start_s, cursor_end - (1500 * interval_s))
        params = {"symbol": pair, "type": timeframe, "startAt": cursor_start, "endAt": cursor_end}
        payload: Dict[str, Any] = {}
        last_exc: Exception | None = None
        for _attempt in range(max_retries):
            try:
                resp = session.get("https://api.kucoin.com/api/v1/market/candles", params=params, timeout=12)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.15 * (_attempt + 1))
        if last_exc is not None:
            raise last_exc
        data = payload.get("data", []) if isinstance(payload, dict) else []
        rows = _normalize_kucoin_rows(data if isinstance(data, list) else [])
        if not rows:
            break
        out = _normalize_kucoin_rows(out + rows)
        if cursor_start <= start_s:
            break
        cursor_end = cursor_start - interval_s
        time.sleep(0.15)
    return out


def _crypto_symbols(settings: Dict[str, Any], main_neural_dir: str, max_symbols: int | None = None) -> List[str]:
    discovery = discover_crypto_trained_artifacts(main_neural_dir)
    out: List[str] = [f"{sym}-USD" for sym in list(discovery.get("trained_symbols", []) or [])]
    raw_pool = _s((settings or {}).get("crypto_dynamic_pool_symbols", ""))
    for tok in raw_pool.split(","):
        sym = _s(tok).upper()
        if sym and f"{sym}-USD" not in out:
            out.append(f"{sym}-USD")
    if max_symbols is not None and max_symbols > 0:
        return out[: int(max_symbols)]
    return out


def _stddev(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / max(1, len(vals))
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
    return math.sqrt(max(0.0, var))


def _simulate_strategy_rows(
    symbol: str,
    candles: List[List[float]],
    artifact_ctx: Dict[str, Any],
    timeframe: str,
    thresholds: Dict[str, float],
    artifact_dir: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(candles) < 48:
        return rows
    in_pos = False
    entry_idx = -1
    entry_px = 0.0
    peak_px = 0.0
    trough_px = 0.0
    entry_features: Dict[str, Any] = {}
    trailing_armed = False
    take_profit_touched = False
    risk_cut_touched = False
    bars_since_peak = 0
    trailing_armed_bar = -1
    risk_cut_touched_bar = -1
    take_profit_touched_bar = -1
    peak_bar = 0
    trough_bar = 0
    label_rule_version = "historical_replay_v2"
    exit_condition_priority = ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"]

    def _preview_state(entry_i: int, preview_i: int, start_px: float) -> Dict[str, Any]:
        if entry_i < 0 or preview_i <= entry_i or start_px <= 0.0:
            return {
                "preview_bars_used": 0,
                "bars_since_entry": 0,
                "current_unrealized_pnl_pct": 0.0,
                "max_favorable_excursion_pct_so_far": 0.0,
                "max_adverse_excursion_pct_so_far": 0.0,
                "drawdown_from_peak_pct_so_far": 0.0,
                "trailing_armed_so_far": False,
                "bars_since_trailing_armed": -1,
                "risk_cut_distance_pct": float(thresholds["risk_cut_pct"]),
                "take_profit_distance_pct": float(thresholds["take_profit_pct"]),
                "risk_cut_touched_so_far": False,
                "take_profit_touched_so_far": False,
                "peak_to_current_reversal_pct": 0.0,
                "favorable_then_softened_flag_so_far": False,
            }
        preview_peak = start_px
        preview_trough = start_px
        preview_close = start_px
        preview_trailing_armed = False
        preview_take_profit_touched = False
        preview_risk_cut_touched = False
        preview_trailing_armed_bar = -1
        for pidx in range(entry_i + 1, preview_i + 1):
            _ts, _open_px, close_px_i, high_px_i, low_px_i, _vol_i = candles[pidx]
            preview_peak = max(preview_peak, high_px_i, close_px_i)
            preview_trough = min(preview_trough, low_px_i, close_px_i)
            preview_close = close_px_i
            preview_mfe = ((preview_peak / start_px) - 1.0) * 100.0 if start_px > 0.0 else 0.0
            preview_drawdown = ((preview_close / preview_peak) - 1.0) * 100.0 if preview_peak > 0.0 else 0.0
            if (not preview_trailing_armed) and preview_mfe >= thresholds["trailing_arm_pct"]:
                preview_trailing_armed_bar = max(0, pidx - entry_i)
            preview_trailing_armed = preview_trailing_armed or (preview_mfe >= thresholds["trailing_arm_pct"])
            preview_take_profit_touched = preview_take_profit_touched or (high_px_i >= (start_px * (1.0 + (thresholds["take_profit_pct"] / 100.0))))
            preview_risk_cut_touched = preview_risk_cut_touched or (low_px_i <= (start_px * (1.0 - (thresholds["risk_cut_pct"] / 100.0))))
        preview_mfe = ((preview_peak / start_px) - 1.0) * 100.0 if start_px > 0.0 else 0.0
        preview_mae = ((preview_trough / start_px) - 1.0) * 100.0 if start_px > 0.0 else 0.0
        preview_drawdown = ((preview_close / preview_peak) - 1.0) * 100.0 if preview_peak > 0.0 else 0.0
        preview_pullback_pct = abs(min(0.0, preview_drawdown))
        preview_bar = max(0, preview_i - entry_i)
        return {
            "preview_bars_used": int(preview_bar),
            "bars_since_entry": int(preview_bar),
            "current_unrealized_pnl_pct": round(float(((preview_close / start_px) - 1.0) * 100.0 if start_px > 0.0 else 0.0), 6),
            "max_favorable_excursion_pct_so_far": round(float(max(0.0, preview_mfe)), 6),
            "max_adverse_excursion_pct_so_far": round(float(min(0.0, preview_mae)), 6),
            "drawdown_from_peak_pct_so_far": round(float(preview_drawdown), 6),
            "trailing_armed_so_far": bool(preview_trailing_armed),
            "bars_since_trailing_armed": int(preview_bar - preview_trailing_armed_bar) if preview_trailing_armed_bar >= 0 else -1,
            "risk_cut_distance_pct": round(float(max(0.0, thresholds["risk_cut_pct"] - max(0.0, abs(min(0.0, preview_mae))))), 6),
            "take_profit_distance_pct": round(float(max(0.0, thresholds["take_profit_pct"] - max(0.0, preview_mfe))), 6),
            "risk_cut_touched_so_far": bool(preview_risk_cut_touched),
            "take_profit_touched_so_far": bool(preview_take_profit_touched),
            "peak_to_current_reversal_pct": round(float(preview_pullback_pct), 6),
            "favorable_then_softened_flag_so_far": bool(preview_mfe >= thresholds["trailing_arm_pct"] and preview_pullback_pct >= max(0.35, thresholds["trailing_drawdown_pct"] * 0.50)),
        }

    def _late_reversal_preview_state(entry_i: int, preview_i: int, start_px: float) -> Dict[str, Any]:
        preview = _preview_state(entry_i, preview_i, start_px)
        if entry_i < 0 or preview_i <= entry_i or start_px <= 0.0:
            return {
                "late_favorable_reversal_score": 0.0,
                "favorable_then_reversed": False,
                "max_favorable_before_exit_pct": 0.0,
                "reversal_from_peak_before_exit_pct": 0.0,
                "bars_from_peak_to_exit_preview": 0,
                "trailing_arm_to_exit_bars_preview": -1,
                "risk_pressure_after_peak": 0.0,
                "take_profit_pressure_before_reversal": 0.0,
                "target_near_before_reversal": False,
                "reversal_velocity_pct_per_bar": 0.0,
                "post_peak_momentum_decay": 0.0,
                "drawdown_after_favorable_move_pct": 0.0,
                "favorable_move_quality_score": 0.0,
                "late_risk_after_favorable_move": False,
            }
        preview_peak = start_px
        preview_peak_bar = 0
        preview_close = start_px
        for pidx in range(entry_i + 1, preview_i + 1):
            _ts, _open_px, close_px_i, high_px_i, _low_px_i, _vol_i = candles[pidx]
            if max(high_px_i, close_px_i) > preview_peak + 1e-12:
                preview_peak = max(high_px_i, close_px_i)
                preview_peak_bar = max(0, pidx - entry_i)
            preview_close = close_px_i
        max_favorable_before_exit_pct = max(0.0, _f(preview.get("max_favorable_excursion_pct_so_far", 0.0), 0.0))
        reversal_from_peak_before_exit_pct = max(0.0, abs(min(0.0, _f(preview.get("drawdown_from_peak_pct_so_far", 0.0), 0.0))))
        bars_from_peak_to_exit_preview = max(0, int(_f(preview.get("bars_since_entry", 0), 0.0)) - int(preview_peak_bar))
        trailing_arm_to_exit_bars_preview = int(_f(preview.get("bars_since_trailing_armed", -1), -1.0))
        target_near_before_reversal = bool(
            _f(preview.get("take_profit_distance_pct", thresholds["take_profit_pct"]), thresholds["take_profit_pct"]) <= max(0.75, thresholds["take_profit_pct"] * 0.35)
            or bool(preview.get("take_profit_touched_so_far", False))
        )
        favorable_then_reversed = bool(max_favorable_before_exit_pct >= thresholds["trailing_arm_pct"] and reversal_from_peak_before_exit_pct >= max(0.35, thresholds["trailing_drawdown_pct"] * 0.45))
        favorable_move_quality_score = (
            max(0.0, max_favorable_before_exit_pct / max(1e-6, thresholds["trailing_arm_pct"]))
            + (0.50 if bool(preview.get("trailing_armed_so_far", False)) else 0.0)
            + max(0.0, (1.0 - _f(preview.get("risk_cut_distance_pct", thresholds["risk_cut_pct"]), thresholds["risk_cut_pct"]) / max(1e-6, thresholds["risk_cut_pct"])))
        )
        risk_pressure_after_peak = (
            max(0.0, reversal_from_peak_before_exit_pct / max(0.5, thresholds["trailing_drawdown_pct"]))
            + max(0.0, abs(min(0.0, _f(preview.get("max_adverse_excursion_pct_so_far", 0.0), 0.0))) / max(0.75, thresholds["risk_cut_pct"]))
            + (0.75 if bool(preview.get("risk_cut_touched_so_far", False)) else 0.0)
        )
        take_profit_pressure_before_reversal = (
            max(0.0, max_favorable_before_exit_pct / max(1.0, thresholds["take_profit_pct"]))
            + (0.80 if target_near_before_reversal else 0.0)
            + max(0.0, (1.0 - (_f(preview.get("take_profit_distance_pct", thresholds["take_profit_pct"]), thresholds["take_profit_pct"]) / max(1.0, thresholds["take_profit_pct"]))))
        )
        reversal_velocity_pct_per_bar = reversal_from_peak_before_exit_pct / max(1, bars_from_peak_to_exit_preview)
        post_peak_momentum_decay = max(0.0, _f(preview.get("momentum_decay", 0.0), 0.0))
        drawdown_after_favorable_move_pct = reversal_from_peak_before_exit_pct if max_favorable_before_exit_pct > 0.0 else 0.0
        late_favorable_reversal_score = (
            max(0.0, favorable_move_quality_score)
            + max(0.0, reversal_from_peak_before_exit_pct / max(0.5, thresholds["trailing_drawdown_pct"]))
            + (0.60 if favorable_then_reversed else 0.0)
            - (0.35 if bool(preview.get("risk_cut_touched_so_far", False)) else 0.0)
        )
        return {
            "late_favorable_reversal_score": round(float(late_favorable_reversal_score), 6),
            "favorable_then_reversed": favorable_then_reversed,
            "max_favorable_before_exit_pct": round(float(max_favorable_before_exit_pct), 6),
            "reversal_from_peak_before_exit_pct": round(float(reversal_from_peak_before_exit_pct), 6),
            "bars_from_peak_to_exit_preview": int(bars_from_peak_to_exit_preview),
            "trailing_arm_to_exit_bars_preview": int(trailing_arm_to_exit_bars_preview),
            "risk_pressure_after_peak": round(float(risk_pressure_after_peak), 6),
            "take_profit_pressure_before_reversal": round(float(take_profit_pressure_before_reversal), 6),
            "target_near_before_reversal": bool(target_near_before_reversal),
            "reversal_velocity_pct_per_bar": round(float(reversal_velocity_pct_per_bar), 6),
            "post_peak_momentum_decay": round(float(post_peak_momentum_decay), 6),
            "drawdown_after_favorable_move_pct": round(float(drawdown_after_favorable_move_pct), 6),
            "favorable_move_quality_score": round(float(favorable_move_quality_score), 6),
            "late_risk_after_favorable_move": bool(max_favorable_before_exit_pct >= thresholds["trailing_arm_pct"] and bool(preview.get("risk_cut_touched_so_far", False))),
        }

    for idx in range(24, len(candles)):
        ts_ms, open_px, close_px, high_px, low_px, _vol = candles[idx]
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
        recent_volatility = _stddev(recent_returns)
        trend_momentum_score = (
            (0.40 * recent_return_3)
            + (0.30 * recent_return_6)
            + (0.20 * recent_return_12)
            + (0.10 * recent_return_24)
        )
        predicted_low = _f(artifact_ctx.get("predicted_low_boundary", 0.0), 0.0)
        predicted_high = _f(artifact_ctx.get("predicted_high_boundary", 0.0), 0.0)
        low_edge = ((predicted_low - close_px) / close_px) * 100.0 if predicted_low > 0 and close_px > 0 else 0.0
        high_edge = ((predicted_high - close_px) / close_px) * 100.0 if predicted_high > 0 and close_px > 0 else 0.0
        signal_margin = _f(artifact_ctx.get("signal_margin", 0.0), 0.0) + (0.18 * trend_momentum_score) + (0.08 * high_edge) - (0.04 * max(0.0, recent_volatility - 2.5))
        signal_side = "long" if signal_margin >= 0.0 else "short"
        feature_payload = {
            "market": "crypto",
            "symbol": symbol,
            "source_type": "historical_strategy_replay",
            "provider": "local_cache_or_kucoin",
            "candle_timeframe": timeframe,
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
            "strategy_adapter_used": "minimal_artifact_replay_v1",
        }
        if not in_pos:
            entry_ok = (
                bool(artifact_ctx.get("usable", False))
                and signal_side == "long"
                and signal_margin >= thresholds["entry_signal_margin_min"]
                and trend_momentum_score >= thresholds["entry_trend_score_min"]
                and recent_return_6 >= thresholds["entry_return6_min"]
            )
            if entry_ok:
                in_pos = True
                entry_idx = idx
                entry_px = close_px
                peak_px = close_px
                trough_px = close_px
                trailing_armed = False
                take_profit_touched = False
                risk_cut_touched = False
                bars_since_peak = 0
                trailing_armed_bar = -1
                risk_cut_touched_bar = -1
                take_profit_touched_bar = -1
                peak_bar = 0
                trough_bar = 0
                entry_features = dict(feature_payload)
                try:
                    original_pred = predict_crypto_original_dry_run(
                        symbol=symbol,
                        timeframe=timeframe,
                        candle_snapshot=candles[: idx + 1],
                        artifact_dir=artifact_dir,
                        as_of_ts=int(ts_ms / 1000),
                    )
                except Exception as exc:
                    original_pred = {
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
                        "original_predictor_blockers": [f"predictor_exception:{type(exc).__name__}"],
                    }
                entry_features.update(
                    {
                        "original_predicted_direction": _s(original_pred.get("predicted_direction", "")),
                        "original_predicted_exit_trigger": _s(original_pred.get("predicted_exit_trigger", "")),
                        "original_predicted_pnl_trend": _s(original_pred.get("predicted_pnl_trend", "")),
                        "original_confidence": round(_f(original_pred.get("confidence", original_pred.get("predicted_confidence", 0.0)), 0.0), 6),
                        "original_direction_scores": dict(original_pred.get("direction_scores", {}) if isinstance(original_pred.get("direction_scores", {}), dict) else {}),
                        "original_trigger_scores": dict(original_pred.get("trigger_scores", {}) if isinstance(original_pred.get("trigger_scores", {}), dict) else {}),
                        "original_trade_quality_score": round(_f(original_pred.get("trade_quality_score", 0.0), 0.0), 6),
                        "original_pnl_quality_score": round(_f(original_pred.get("pnl_quality_score", 0.0), 0.0), 6),
                        "original_selected_predictor": _s(original_pred.get("selected_predictor", "")),
                        "original_predictor_variant": _s(original_pred.get("predictor_variant", "")),
                        "original_source_used": _s(original_pred.get("source_used", "")),
                        "original_prediction_semantics": _s(original_pred.get("prediction_semantics", "")),
                        "original_prediction_semantics_warning": _s(original_pred.get("prediction_semantics_warning", "")),
                        "original_current_candle_pct_move": round(_f(original_pred.get("current_candle_pct_move", 0.0), 0.0), 6),
                        "original_final_moves": round(_f(original_pred.get("final_moves", 0.0), 0.0), 6),
                        "original_high_final_moves": round(_f(original_pred.get("high_final_moves", 0.0), 0.0), 6),
                        "original_low_final_moves": round(_f(original_pred.get("low_final_moves", 0.0), 0.0), 6),
                        "original_high_new_price": round(_f(original_pred.get("high_new_price", 0.0), 0.0), 10),
                        "original_low_new_price": round(_f(original_pred.get("low_new_price", 0.0), 0.0), 10),
                        "original_active_model_state": _s(original_pred.get("active_model_state", "")),
                        "original_signal_side": _s(original_pred.get("original_signal_side", "")),
                        "original_signal_margin": round(_f(original_pred.get("original_signal_margin", 0.0), 0.0), 6),
                        "original_long_signal_count": int(_f(original_pred.get("original_long_signal_count", 0), 0.0)),
                        "original_short_signal_count": int(_f(original_pred.get("original_short_signal_count", 0), 0.0)),
                        "original_long_profit_margin": round(_f(original_pred.get("original_long_profit_margin", 0.0), 0.0), 6),
                        "original_short_profit_margin": round(_f(original_pred.get("original_short_profit_margin", 0.0), 0.0), 6),
                        "original_message_type": _s(original_pred.get("original_message_type", "")),
                        "original_bound_position": _s(original_pred.get("original_bound_position", "")),
                        "original_signal_active": bool(original_pred.get("original_signal_active", False)),
                        "original_bounds_active": bool(original_pred.get("original_bounds_active", False)),
                        "original_trigger_semantics_status": _s(original_pred.get("original_trigger_semantics_status", "")),
                        "trigger_semantics_blocker": _s(original_pred.get("trigger_semantics_blocker", "")),
                        "original_memory_rows_selected": int(_f(original_pred.get("memory_rows_selected", 0), 0.0)),
                        "original_memory_rows_perfect_matches": int(_f(original_pred.get("memory_rows_perfect_matches", 0), 0.0)),
                        "original_predictor_dry_run_available": bool(original_pred.get("original_predictor_dry_run_available", False)),
                        "original_predictor_dry_run_used": bool(original_pred.get("original_predictor_dry_run_used", False)),
                        "original_predictor_replay_safe": bool(original_pred.get("original_predictor_replay_safe", True)),
                        "original_predictor_imported_pt_thinker": bool(original_pred.get("original_predictor_imported_pt_thinker", False)),
                        "original_predictor_called_step_coin": bool(original_pred.get("original_predictor_called_step_coin", False)),
                        "original_predictor_called_robinhood": bool(original_pred.get("original_predictor_called_robinhood", False)),
                        "original_predictor_called_kucoin_live": bool(original_pred.get("original_predictor_called_kucoin_live", False)),
                        "original_predictor_wrote_live_signal_files": bool(original_pred.get("original_predictor_wrote_live_signal_files", False)),
                        "original_predictor_changed_cwd": bool(original_pred.get("original_predictor_changed_cwd", False)),
                        "original_predictor_mutated_live_artifacts": bool(original_pred.get("original_predictor_mutated_live_artifacts", False)),
                        "bound_signal_dry_run_available": bool(original_pred.get("bound_signal_dry_run_available", False)),
                        "bound_signal_dry_run_used": bool(original_pred.get("bound_signal_dry_run_used", False)),
                        "bound_signal_replay_safe": bool(original_pred.get("bound_signal_replay_safe", True)),
                        "bound_signal_called_robinhood": bool(original_pred.get("bound_signal_called_robinhood", False)),
                        "bound_signal_called_kucoin_live": bool(original_pred.get("bound_signal_called_kucoin_live", False)),
                        "bound_signal_wrote_live_signal_files": bool(original_pred.get("bound_signal_wrote_live_signal_files", False)),
                        "bound_signal_wrote_bound_files": bool(original_pred.get("bound_signal_wrote_bound_files", False)),
                        "bound_signal_changed_cwd": bool(original_pred.get("bound_signal_changed_cwd", False)),
                        "bound_signal_imported_pt_thinker": bool(original_pred.get("bound_signal_imported_pt_thinker", False)),
                        "bound_signal_blockers": list(original_pred.get("bound_signal_blockers", []) or []),
                        "original_predictor_blockers": list(original_pred.get("original_predictor_blockers", []) or []),
                    }
                )
            continue
        prior_peak = peak_px
        peak_px = max(peak_px, high_px, close_px)
        trough_px = min(trough_px, low_px, close_px) if trough_px > 0.0 else min(low_px, close_px)
        if peak_px > prior_peak + 1e-12:
            bars_since_peak = 0
            peak_bar = max(0, idx - entry_idx)
        else:
            bars_since_peak += 1
        pnl_pct = ((close_px / entry_px) - 1.0) * 100.0 if entry_px > 0 else 0.0
        mfe_pct = ((peak_px / entry_px) - 1.0) * 100.0 if entry_px > 0 and peak_px > 0 else 0.0
        mae_pct = ((trough_px / entry_px) - 1.0) * 100.0 if entry_px > 0 and trough_px > 0 else 0.0
        if abs(((low_px / entry_px) - 1.0) * 100.0 if entry_px > 0 else 0.0) >= abs(mae_pct) - 1e-9:
            trough_bar = max(0, idx - entry_idx)
        drawdown_from_peak_pct = ((close_px / peak_px) - 1.0) * 100.0 if peak_px > 0 else 0.0
        hold_hours = ((ts_ms - candles[entry_idx][0]) / 3_600_000.0) if entry_idx >= 0 else 0.0
        bars_in_trade = max(1, idx - entry_idx + 1) if entry_idx >= 0 else 0
        current_bar = max(0, idx - entry_idx)
        if (not trailing_armed) and (mfe_pct >= thresholds["trailing_arm_pct"]):
            trailing_armed_bar = current_bar
        trailing_armed = trailing_armed or (mfe_pct >= thresholds["trailing_arm_pct"])
        trailing_pullback_pct = abs(drawdown_from_peak_pct)
        if (not take_profit_touched) and (high_px >= (entry_px * (1.0 + (thresholds["take_profit_pct"] / 100.0)))):
            take_profit_touched_bar = current_bar
        take_profit_touched = take_profit_touched or (high_px >= (entry_px * (1.0 + (thresholds["take_profit_pct"] / 100.0))))
        if (not risk_cut_touched) and (low_px <= (entry_px * (1.0 - (thresholds["risk_cut_pct"] / 100.0)))):
            risk_cut_touched_bar = current_bar
        risk_cut_touched = risk_cut_touched or (low_px <= (entry_px * (1.0 - (thresholds["risk_cut_pct"] / 100.0))))
        take_profit_distance_pct = max(0.0, thresholds["take_profit_pct"] - max(0.0, mfe_pct))
        risk_cut_distance_pct = max(0.0, thresholds["risk_cut_pct"] - max(0.0, abs(min(0.0, mae_pct))))
        exit_momentum_3 = recent_return_3
        exit_momentum_6 = recent_return_6
        favorable_then_softened = bool(mfe_pct >= thresholds["trailing_arm_pct"] and trailing_pullback_pct >= max(0.35, thresholds["trailing_drawdown_pct"] * 0.50))
        risk_cut_now = bool(low_px <= (entry_px * (1.0 - (thresholds["risk_cut_pct"] / 100.0))))
        take_profit_now = bool(high_px >= (entry_px * (1.0 + (thresholds["take_profit_pct"] / 100.0))))
        trailing_now = bool(trailing_armed and trailing_pullback_pct >= thresholds["trailing_drawdown_pct"])
        stale_now = bool(hold_hours >= thresholds["stale_hold_hours"] and trend_momentum_score <= thresholds["stale_trend_score_max"])
        conditions = [
            ("Risk Cut", risk_cut_now),
            ("Take Profit", take_profit_now),
            ("Trailing", trailing_now),
            ("Stale Alignment", stale_now),
        ]
        same_candle_multi_exit_condition_count = int(sum(1 for _name, active in conditions if active))
        reason = ""
        for trigger_name, active in conditions:
            if not active:
                continue
            if trigger_name == "Risk Cut":
                reason = "risk_cut"
            elif trigger_name == "Take Profit":
                reason = "take_profit"
            elif trigger_name == "Trailing":
                reason = "trailing"
            elif trigger_name == "Stale Alignment":
                reason = "stale_alignment"
            break
        if not reason:
            continue
        normalized_trigger = normalize_exit_trigger(reason)
        first_hard_exit_condition = ""
        first_hard_exit_bar = -1
        for hard_name, hard_bar in (
            ("Risk Cut", risk_cut_touched_bar),
            ("Take Profit", take_profit_touched_bar),
            ("Trailing", trailing_armed_bar if trailing_now else -1),
        ):
            if hard_bar >= 0 and (first_hard_exit_bar < 0 or hard_bar < first_hard_exit_bar):
                first_hard_exit_condition = hard_name
                first_hard_exit_bar = int(hard_bar)
        if first_hard_exit_bar < 0 and stale_now:
            first_hard_exit_condition = "Stale Alignment"
            first_hard_exit_bar = int(current_bar)
        candle_range = max(1e-9, high_px - low_px)
        exit_close_position_in_candle_range = max(0.0, min(1.0, (close_px - low_px) / candle_range))
        bars_from_entry_to_trailing_arm = int(trailing_armed_bar) if trailing_armed_bar >= 0 else -1
        bars_from_entry_to_risk_cut = int(risk_cut_touched_bar) if risk_cut_touched_bar >= 0 else -1
        bars_from_trailing_arm_to_risk_cut = int(risk_cut_touched_bar - trailing_armed_bar) if trailing_armed_bar >= 0 and risk_cut_touched_bar >= 0 else -1
        bars_from_peak_to_exit = int(max(0, current_bar - peak_bar))
        reversal_from_peak_pct = abs(min(0.0, drawdown_from_peak_pct))
        peak_to_exit_velocity_pct_per_bar = (
            -reversal_from_peak_pct / max(1, bars_from_peak_to_exit)
            if bars_from_peak_to_exit > 0
            else 0.0
        )
        entry_to_trough_velocity_pct_per_bar = (
            float(mae_pct) / max(1, trough_bar)
            if trough_bar > 0
            else float(mae_pct)
        )
        trade_slice = candles[entry_idx : idx + 1]
        closes_in_trade = [float(c[2]) for c in trade_slice if isinstance(c, list) and len(c) >= 3 and _f(c[2], 0.0) > 0.0]
        def _drawdown_over_last_n(n: int) -> float:
            if len(closes_in_trade) < 2:
                return 0.0
            slice_vals = closes_in_trade[-max(2, n):]
            peak_local = max(slice_vals) if slice_vals else 0.0
            if peak_local <= 0.0:
                return 0.0
            return ((slice_vals[-1] / peak_local) - 1.0) * 100.0
        last_3_bar_drawdown_pct = _drawdown_over_last_n(3)
        last_6_bar_drawdown_pct = _drawdown_over_last_n(6)
        risk_breach_depth_pct = max(0.0, abs(min(0.0, mae_pct)) - float(thresholds["risk_cut_pct"]))
        preview_idx = entry_idx if current_bar <= 1 else min(idx - 1, entry_idx + 6)
        preview_state = _preview_state(entry_idx, preview_idx, entry_px)
        late_preview_state = _late_reversal_preview_state(entry_idx, max(entry_idx, idx - 1), entry_px)
        row = dict(entry_features)
        row.update(
            {
                "entry_ts": int(candles[entry_idx][0] / 1000),
                "exit_ts": int(ts_ms / 1000),
                "entry_price": round(float(entry_px), 10),
                "exit_price": round(float(close_px), 10),
                "hold_hours": round(float(max(0.0, hold_hours)), 6),
                "qty": 1.0,
                "pnl_usd": round(float(close_px - entry_px), 8),
                "pnl_pct": round(float(pnl_pct), 6),
                "actual_direction": "up" if pnl_pct > 1e-9 else "down" if pnl_pct < -1e-9 else "flat",
                "actual_exit_trigger": normalized_trigger,
                "event_exit_tag": f"historical_strategy_replay:{reason}",
                "raw_rule_reason": reason,
                "normalized_trigger": normalized_trigger,
                "max_favorable_excursion_pct": round(float(max(0.0, mfe_pct)), 6),
                "max_adverse_excursion_pct": round(float(min(0.0, mae_pct)), 6),
                "peak_profit_pct": round(float(max(0.0, mfe_pct)), 6),
                "worst_drawdown_pct": round(float(min(0.0, mae_pct)), 6),
                "drawdown_from_peak_pct": round(float(drawdown_from_peak_pct), 6),
                "bars_since_peak": int(max(0, bars_since_peak)),
                "bars_in_trade": int(max(1, bars_in_trade)),
                "trailing_armed_bar": int(trailing_armed_bar),
                "risk_cut_touched_bar": int(risk_cut_touched_bar),
                "take_profit_touched_bar": int(take_profit_touched_bar),
                "peak_bar": int(peak_bar),
                "exit_bar": int(current_bar),
                "first_hard_exit_condition": first_hard_exit_condition,
                "first_hard_exit_bar": int(first_hard_exit_bar),
                "bars_from_entry_to_trailing_arm": int(bars_from_entry_to_trailing_arm),
                "bars_from_entry_to_risk_cut": int(bars_from_entry_to_risk_cut),
                "bars_from_trailing_arm_to_risk_cut": int(bars_from_trailing_arm_to_risk_cut),
                "bars_from_peak_to_exit": int(bars_from_peak_to_exit),
                "risk_cut_before_trailing": bool(risk_cut_touched_bar >= 0 and (trailing_armed_bar < 0 or risk_cut_touched_bar < trailing_armed_bar)),
                "trailing_before_risk_cut": bool(trailing_armed_bar >= 0 and (risk_cut_touched_bar < 0 or trailing_armed_bar < risk_cut_touched_bar)),
                "risk_cut_after_trailing_arm": bool(risk_cut_touched_bar >= 0 and trailing_armed_bar >= 0 and risk_cut_touched_bar > trailing_armed_bar),
                "risk_cut_after_peak": bool(risk_cut_touched_bar >= 0 and peak_bar >= 0 and risk_cut_touched_bar >= peak_bar),
                "trailing_valid_before_risk": bool(trailing_armed_bar >= 0 and favorable_then_softened and (risk_cut_touched_bar < 0 or trailing_armed_bar < risk_cut_touched_bar)),
                "same_candle_risk_and_trailing": bool(risk_cut_now and trailing_now),
                "trailing_armed": bool(trailing_armed),
                "trailing_pullback_pct": round(float(trailing_pullback_pct), 6),
                "take_profit_touched": bool(take_profit_touched),
                "take_profit_distance_pct": round(float(take_profit_distance_pct), 6),
                "risk_cut_touched": bool(risk_cut_touched),
                "risk_cut_distance_pct": round(float(risk_cut_distance_pct), 6),
                "exit_close_position_in_candle_range": round(float(exit_close_position_in_candle_range), 6),
                "peak_to_exit_velocity_pct_per_bar": round(float(peak_to_exit_velocity_pct_per_bar), 6),
                "entry_to_trough_velocity_pct_per_bar": round(float(entry_to_trough_velocity_pct_per_bar), 6),
                "last_3_bar_drawdown_pct": round(float(last_3_bar_drawdown_pct), 6),
                "last_6_bar_drawdown_pct": round(float(last_6_bar_drawdown_pct), 6),
                "reversal_from_peak_pct": round(float(reversal_from_peak_pct), 6),
                "risk_breach_depth_pct": round(float(risk_breach_depth_pct), 6),
                "exit_momentum_3": round(float(exit_momentum_3), 6),
                "exit_momentum_6": round(float(exit_momentum_6), 6),
                "favorable_then_softened_flag": bool(favorable_then_softened),
                "label_rule_version": label_rule_version,
                "exit_condition_priority_used": list(exit_condition_priority),
                "same_candle_multi_exit_condition_count": int(same_candle_multi_exit_condition_count),
                **preview_state,
                **late_preview_state,
            }
        )
        rows.append(row)
        in_pos = False
        entry_idx = -1
        entry_px = 0.0
        peak_px = 0.0
        entry_features = {}
    return rows


def build_crypto_historical_strategy_replay(
    hub_dir: str,
    settings: Dict[str, Any] | None = None,
    symbols: Iterable[str] | None = None,
    timeframe: str = "1hour",
    lookback_days: int = 90,
    max_symbols: int | None = None,
    force_refresh: bool = False,
    start_ts: int | None = None,
    cutoff_ts: int | None = None,
    symbols_lock_file: str = "",
    freeze_symbols: bool = False,
    freeze_cache: bool = False,
    deterministic: bool = False,
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    limits = _performance_limits()
    main_neural_dir = _s(cfg.get("main_neural_dir", ""))
    if not main_neural_dir:
        main_neural_dir = os.path.join(os.path.dirname(hub_dir), "market_data", "coins")
    cache_path = _cache_root(hub_dir)
    os.makedirs(cache_path, exist_ok=True)
    requested_max_symbols = int(max_symbols) if max_symbols is not None else int(limits["max_symbols_per_replay"])
    requested_lookback_days = int(lookback_days)
    capped_max_symbols = min(int(limits["max_symbols_per_replay"]), max(1, requested_max_symbols))
    capped_lookback_days = min(int(limits["max_lookback_days"]), max(1, requested_lookback_days))
    replay_symbols = [str(s) for s in list(symbols or []) if _s(s)] or _crypto_symbols(cfg, main_neural_dir, max_symbols=capped_max_symbols)
    lock_path = _s(symbols_lock_file) or os.path.join(hub_dir, "crypto", "historical_replay_cache", "symbols.lock.json")
    if freeze_symbols and os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                locked = json.load(f)
            locked_symbols = [str(s) for s in list(locked.get("symbols", []) or []) if _s(s)]
            if locked_symbols:
                replay_symbols = locked_symbols
        except Exception:
            pass
    if capped_max_symbols > 0:
        replay_symbols = replay_symbols[: int(capped_max_symbols)]
    if freeze_symbols and (not os.path.exists(lock_path)):
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump({"symbols": list(replay_symbols)}, f, indent=2)
        except Exception:
            pass
    derived_cutoff_ts = int(cutoff_ts) if cutoff_ts is not None else 0
    if deterministic and freeze_cache and derived_cutoff_ts <= 0:
        derived_cutoff_ts = _derive_cached_cutoff_ts(hub_dir, replay_symbols, timeframe)
    if derived_cutoff_ts <= 0:
        derived_cutoff_ts = int(time.time())
    end_ts_ms = int(derived_cutoff_ts * 1000)
    start_ts_ms = int((start_ts if start_ts is not None else int((end_ts_ms / 1000) - (max(7, int(capped_lookback_days)) * 86400))) * 1000)
    thresholds = {
        "entry_signal_margin_min": 0.35,
        "entry_trend_score_min": 0.12,
        "entry_return6_min": -0.25,
        "risk_cut_pct": 2.25,
        "take_profit_pct": 4.25,
        "trailing_arm_pct": 1.6,
        "trailing_drawdown_pct": 1.1,
        "stale_hold_hours": 18.0,
        "stale_trend_score_max": 0.05,
    }
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    used_local_cache = False
    used_remote_api = False
    cache_rows_loaded = 0
    remote_rows_fetched = 0
    candle_rows_final = 0
    symbols_covered: List[str] = []
    artifact_paths_used: List[str] = []
    artifact_timeframes_used: List[str] = []
    artifact_missing_reasons: Dict[str, int] = {}
    for symbol in replay_symbols:
        artifact_ctx = _artifact_context(main_neural_dir, symbol, timeframe)
        artifact_paths_used.extend(list(artifact_ctx.get("artifact_paths_used", []) or []))
        artifact_timeframes_used.extend(list(artifact_ctx.get("artifact_timeframes_used", []) or []))
        for reason in list(artifact_ctx.get("artifact_missing_reasons", []) or []):
            artifact_missing_reasons[_s(reason) or "unknown"] = int(artifact_missing_reasons.get(_s(reason) or "unknown", 0) + 1)
        effective_force_refresh = bool(force_refresh and not freeze_cache)
        cached = [] if effective_force_refresh else _load_cached_candles(hub_dir, symbol, timeframe)
        if cached:
            used_local_cache = True
            cache_rows_loaded += len(cached)
        final_candles = [row for row in cached if int(row[0]) >= start_ts_ms and int(row[0]) <= end_ts_ms]
        if len(final_candles) < 48:
            try:
                fetched = [] if freeze_cache else _fetch_kucoin_candles(symbol, timeframe, start_ts_ms, end_ts_ms)
                if fetched:
                    used_remote_api = True
                    remote_rows_fetched += len(fetched)
                    final_candles = _normalize_kucoin_rows(cached + fetched)
                    _save_cached_candles(hub_dir, symbol, timeframe, final_candles)
            except Exception as exc:
                skipped.append({"symbol": symbol, "reason": f"remote_fetch_failed:{type(exc).__name__}"})
        final_candles = [row for row in final_candles if int(row[0]) >= start_ts_ms and int(row[0]) <= end_ts_ms]
        candle_rows_final += len(final_candles)
        if len(final_candles) < 48:
            skipped.append({"symbol": symbol, "reason": f"insufficient_candles:{len(final_candles)}"})
            continue
        if not bool(artifact_ctx.get("usable", False)):
            skipped.append({"symbol": symbol, "reason": _s((artifact_ctx.get("artifact_missing_reasons", []) or ["artifact_unusable"])[0]) or "artifact_unusable"})
            continue
        trade_rows = _simulate_strategy_rows(symbol, final_candles, artifact_ctx, timeframe, thresholds, artifact_dir=_symbol_dir(main_neural_dir, symbol))
        if not trade_rows:
            skipped.append({"symbol": symbol, "reason": "no_strategy_trades"})
            continue
        symbols_covered.append(symbol)
        rows.extend(trade_rows)
    rows.sort(key=lambda r: (
        int(_f(r.get("entry_ts", 0.0), 0.0)),
        int(_f(r.get("exit_ts", 0.0), 0.0)),
        _s(r.get("symbol", "")),
        _s(r.get("actual_exit_trigger", "")),
    ))
    symbol_hash = hashlib.sha256("|".join(sorted(set(replay_symbols))).encode("utf-8")).hexdigest() if replay_symbols else ""
    row_hash = hashlib.sha256(
        json.dumps(
            [
                [
                    _s(r.get("symbol", "")),
                    int(_f(r.get("entry_ts", 0.0), 0.0)),
                    int(_f(r.get("exit_ts", 0.0), 0.0)),
                    round(_f(r.get("entry_price", 0.0), 0.0), 8),
                    round(_f(r.get("exit_price", 0.0), 0.0), 8),
                    _s(r.get("actual_exit_trigger", "")),
                ]
                for r in rows
            ],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    diagnostics = {
        "replay_mutated_live_artifacts": False,
        "replay_cache_path": cache_path,
        "live_artifact_read_only": True,
        "replay_used_local_cache": bool(used_local_cache),
        "replay_used_remote_api": bool(used_remote_api),
        "remote_provider": "kucoin",
        "credentials_required": False,
        "cache_rows_loaded": int(cache_rows_loaded),
        "remote_rows_fetched": int(remote_rows_fetched),
        "final_candle_rows": int(candle_rows_final),
        "date_range": {"start_ts": int(start_ts_ms / 1000), "end_ts": int(end_ts_ms / 1000)},
        "timeframe": timeframe,
        "symbols_covered": sorted(set(symbols_covered)),
        "historical_strategy_replay_rows": int(len(rows)),
        "historical_strategy_replay_symbols": sorted(set(symbols_covered)),
        "historical_strategy_replay_timeframe": timeframe,
        "historical_strategy_replay_date_range": {"start_ts": int(start_ts_ms / 1000), "end_ts": int(end_ts_ms / 1000)},
        "historical_strategy_replay_provider": "kucoin" if used_remote_api else "local_cache",
        "historical_strategy_replay_cache_path": cache_path,
        "historical_strategy_replay_skipped_reasons": skipped,
        "trained_artifacts_found": int(discover_crypto_trained_artifacts(main_neural_dir).get("trained_artifacts_found", 0) or 0),
        "trained_artifacts_fresh": True,
        "artifact_training_time": max([int(_f(r.get("artifact_training_time", 0.0), 0.0)) for r in rows] or [0]),
        "artifact_paths_used": sorted(set(artifact_paths_used)),
        "artifact_timeframes_used": sorted(set(artifact_timeframes_used)),
        "artifact_missing_reasons": artifact_missing_reasons,
        "strategy_adapter_used": "minimal_artifact_replay_v1",
        "thresholds": thresholds,
        "exit_shape_features_enabled": True,
        "exit_shape_feature_names": [
            "max_favorable_excursion_pct",
            "max_adverse_excursion_pct",
            "peak_profit_pct",
            "worst_drawdown_pct",
            "drawdown_from_peak_pct",
            "bars_since_peak",
            "bars_in_trade",
            "trailing_armed_bar",
            "risk_cut_touched_bar",
            "take_profit_touched_bar",
            "peak_bar",
            "exit_bar",
            "first_hard_exit_condition",
            "first_hard_exit_bar",
            "bars_from_entry_to_trailing_arm",
            "bars_from_entry_to_risk_cut",
            "bars_from_trailing_arm_to_risk_cut",
            "bars_from_peak_to_exit",
            "risk_cut_before_trailing",
            "trailing_before_risk_cut",
            "risk_cut_after_trailing_arm",
            "risk_cut_after_peak",
            "trailing_valid_before_risk",
            "same_candle_risk_and_trailing",
            "trailing_armed",
            "trailing_pullback_pct",
            "take_profit_touched",
            "take_profit_distance_pct",
            "risk_cut_touched",
            "risk_cut_distance_pct",
            "exit_close_position_in_candle_range",
            "peak_to_exit_velocity_pct_per_bar",
            "entry_to_trough_velocity_pct_per_bar",
            "last_3_bar_drawdown_pct",
            "last_6_bar_drawdown_pct",
            "reversal_from_peak_pct",
            "risk_breach_depth_pct",
            "exit_momentum_3",
            "exit_momentum_6",
            "favorable_then_softened_flag",
        ],
        "exit_shape_thresholds": {
            "trailing_arm_pct": thresholds["trailing_arm_pct"],
            "trailing_pullback_pct": thresholds["trailing_drawdown_pct"],
            "take_profit_pct": thresholds["take_profit_pct"],
            "risk_cut_pct": thresholds["risk_cut_pct"],
        },
        "rows_with_exit_shape_features": int(sum(1 for r in rows if "bars_in_trade" in r)),
        "rows_missing_exit_shape_features": int(sum(1 for r in rows if "bars_in_trade" not in r)),
        "crypto_backfill_force_refresh": bool(force_refresh),
        "crypto_replay_deterministic_mode_enabled": bool(deterministic),
        "crypto_replay_eval_start_ts": int(start_ts_ms / 1000),
        "crypto_replay_eval_cutoff_ts": int(end_ts_ms / 1000),
        "crypto_replay_symbols_locked": bool(freeze_symbols),
        "crypto_replay_symbol_list_hash": symbol_hash,
        "crypto_replay_cache_frozen": bool(freeze_cache),
        "crypto_replay_rows_hash": row_hash,
        "crypto_replay_symbols_lock_file": lock_path,
        "performance_caps": {
            "max_symbols_per_replay": int(limits["max_symbols_per_replay"]),
            "requested_max_symbols": int(requested_max_symbols),
            "max_symbols_capped": bool(requested_max_symbols > capped_max_symbols),
            "max_lookback_days": int(limits["max_lookback_days"]),
            "requested_lookback_days": int(requested_lookback_days),
            "lookback_days_capped": bool(requested_lookback_days > capped_lookback_days),
            "max_network_retry_attempts": int(limits["max_network_retry_attempts"]),
            "max_cached_candle_rows": int(limits["max_cached_candle_rows"]),
        },
        "live_api_calls_avoided": {
            "robinhood_called": False,
            "kucoin_live_called_only_when_cache_missing": bool(used_remote_api),
        },
    }
    state = "READY" if rows else "NO_DATA"
    return {"state": state, "rows": rows, "diagnostics": diagnostics, "skipped": skipped, "provider_cache_details": diagnostics}
