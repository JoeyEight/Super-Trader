from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import requests

from app.crypto_artifacts import TRAINING_STALE_SECONDS, discover_crypto_trained_artifacts, load_crypto_artifact_features
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
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _cache_root(hub_dir: str) -> str:
    return os.path.join(hub_dir, "crypto", "historical_replay_cache")


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
    while cursor_end >= start_s:
        cursor_start = max(start_s, cursor_end - (1500 * interval_s))
        params = {"symbol": pair, "type": timeframe, "startAt": cursor_start, "endAt": cursor_end}
        resp = session.get("https://api.kucoin.com/api/v1/market/candles", params=params, timeout=12)
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
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
    label_rule_version = "historical_replay_v2"
    exit_condition_priority = ["Risk Cut", "Take Profit", "Trailing", "Stale Alignment"]
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
                entry_features = dict(feature_payload)
            continue
        prior_peak = peak_px
        peak_px = max(peak_px, high_px, close_px)
        trough_px = min(trough_px, low_px, close_px) if trough_px > 0.0 else min(low_px, close_px)
        if peak_px > prior_peak + 1e-12:
            bars_since_peak = 0
        else:
            bars_since_peak += 1
        pnl_pct = ((close_px / entry_px) - 1.0) * 100.0 if entry_px > 0 else 0.0
        mfe_pct = ((peak_px / entry_px) - 1.0) * 100.0 if entry_px > 0 and peak_px > 0 else 0.0
        mae_pct = ((trough_px / entry_px) - 1.0) * 100.0 if entry_px > 0 and trough_px > 0 else 0.0
        drawdown_from_peak_pct = ((close_px / peak_px) - 1.0) * 100.0 if peak_px > 0 else 0.0
        hold_hours = ((ts_ms - candles[entry_idx][0]) / 3_600_000.0) if entry_idx >= 0 else 0.0
        bars_in_trade = max(1, idx - entry_idx + 1) if entry_idx >= 0 else 0
        trailing_armed = trailing_armed or (mfe_pct >= thresholds["trailing_arm_pct"])
        trailing_pullback_pct = abs(drawdown_from_peak_pct)
        take_profit_touched = take_profit_touched or (high_px >= (entry_px * (1.0 + (thresholds["take_profit_pct"] / 100.0))))
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
                "trailing_armed": bool(trailing_armed),
                "trailing_pullback_pct": round(float(trailing_pullback_pct), 6),
                "take_profit_touched": bool(take_profit_touched),
                "take_profit_distance_pct": round(float(take_profit_distance_pct), 6),
                "risk_cut_touched": bool(risk_cut_touched),
                "risk_cut_distance_pct": round(float(risk_cut_distance_pct), 6),
                "exit_momentum_3": round(float(exit_momentum_3), 6),
                "exit_momentum_6": round(float(exit_momentum_6), 6),
                "favorable_then_softened_flag": bool(favorable_then_softened),
                "label_rule_version": label_rule_version,
                "exit_condition_priority_used": list(exit_condition_priority),
                "same_candle_multi_exit_condition_count": int(same_candle_multi_exit_condition_count),
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
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    main_neural_dir = _s(cfg.get("main_neural_dir", ""))
    if not main_neural_dir:
        main_neural_dir = os.path.join(os.path.dirname(hub_dir), "market_data", "coins")
    cache_path = _cache_root(hub_dir)
    os.makedirs(cache_path, exist_ok=True)
    replay_symbols = [str(s) for s in list(symbols or []) if _s(s)] or _crypto_symbols(cfg, main_neural_dir, max_symbols=max_symbols)
    if max_symbols is not None and max_symbols > 0:
        replay_symbols = replay_symbols[: int(max_symbols)]
    end_ts_ms = int(time.time() * 1000)
    start_ts_ms = int(end_ts_ms - (max(7, int(lookback_days)) * 86400 * 1000))
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
        cached = _load_cached_candles(hub_dir, symbol, timeframe)
        if cached:
            used_local_cache = True
            cache_rows_loaded += len(cached)
        final_candles = [row for row in cached if int(row[0]) >= start_ts_ms and int(row[0]) <= end_ts_ms]
        if len(final_candles) < 48:
            try:
                fetched = _fetch_kucoin_candles(symbol, timeframe, start_ts_ms, end_ts_ms)
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
        trade_rows = _simulate_strategy_rows(symbol, final_candles, artifact_ctx, timeframe, thresholds)
        if not trade_rows:
            skipped.append({"symbol": symbol, "reason": "no_strategy_trades"})
            continue
        symbols_covered.append(symbol)
        rows.extend(trade_rows)
    rows.sort(key=lambda r: (int(_f(r.get("entry_ts", 0.0), 0.0)), _s(r.get("symbol", ""))))
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
            "trailing_armed",
            "trailing_pullback_pct",
            "take_profit_touched",
            "take_profit_distance_pct",
            "risk_cut_touched",
            "risk_cut_distance_pct",
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
    }
    state = "READY" if rows else "NO_DATA"
    return {"state": state, "rows": rows, "diagnostics": diagnostics, "skipped": skipped, "provider_cache_details": diagnostics}
