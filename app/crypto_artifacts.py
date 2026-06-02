from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, Iterable, List, Tuple


TRAINING_STALE_SECONDS = 14 * 24 * 60 * 60
TIMEFRAME_SUFFIXES = ("1hour", "2hour", "4hour", "8hour", "12hour", "1day", "1week")


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


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _s(f.read())
    except Exception:
        return ""


def _symbol_candidates(symbol: str) -> List[str]:
    raw = _s(symbol).upper()
    if not raw:
        return []
    out: List[str] = []
    for cand in [raw, raw.replace("/", "_"), raw.replace("-", "_")]:
        cc = _s(cand).upper()
        if cc and cc not in out:
            out.append(cc)
    for sep in ("-", "/", "_"):
        if sep in raw:
            base = _s(raw.split(sep)[0]).upper()
            quote = _s(raw.split(sep)[-1]).upper()
            for cand in [base, f"{base}_{quote}", f"{base}-{quote}"]:
                cc = _s(cand).upper()
                if cc and cc not in out:
                    out.append(cc)
    return out


def _resolve_symbol_dir(base_dir: str, symbol: str) -> str:
    root = os.path.abspath(_s(base_dir))
    for cand in _symbol_candidates(symbol):
        path = os.path.join(root, cand)
        if os.path.isdir(path):
            return path
    return os.path.join(root, _s(symbol))


def list_crypto_artifact_patterns(symbol_dir: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "trainer_status": [],
        "training_timestamps": [],
        "thresholds": [],
        "memories": [],
        "memory_weights": [],
        "memory_weights_high": [],
        "memory_weights_low": [],
        "signal_state": [],
        "bounds": [],
    }
    names = [os.path.join(symbol_dir, name) for name in os.listdir(symbol_dir)] if os.path.isdir(symbol_dir) else []
    for path in names:
        base = os.path.basename(path)
        if base == "trainer_status.json":
            out["trainer_status"].append(path)
        elif base in {"trainer_last_training_time.txt", "trainer_last_start_time.txt"}:
            out["training_timestamps"].append(path)
        elif base.startswith("neural_perfect_threshold_"):
            out["thresholds"].append(path)
        elif base.startswith("memories_"):
            out["memories"].append(path)
        elif base.startswith("memory_weights_high_"):
            out["memory_weights_high"].append(path)
        elif base.startswith("memory_weights_low_"):
            out["memory_weights_low"].append(path)
        elif base.startswith("memory_weights_"):
            out["memory_weights"].append(path)
        elif base in {"long_dca_signal.txt", "short_dca_signal.txt", "futures_long_profit_margin.txt", "futures_short_profit_margin.txt"}:
            out["signal_state"].append(path)
        elif base in {"high_bound_prices.html", "low_bound_prices.html"}:
            out["bounds"].append(path)
    return out


def _extract_timeframe(path: str) -> str:
    base = os.path.basename(path).lower()
    for suffix in TIMEFRAME_SUFFIXES:
        if suffix in base:
            return suffix
    return ""


def discover_crypto_trained_artifacts(main_dir: str, symbols: Iterable[str] | None = None) -> Dict[str, Any]:
    base_dir = os.path.abspath(_s(main_dir))
    found_symbols: List[str] = []
    requested = [_s(sym).upper() for sym in list(symbols or []) if _s(sym)]
    dirs: List[str] = []
    if requested:
        dirs = [os.path.join(base_dir, sym) for sym in requested]
    else:
        dirs = [p for p in sorted(glob.glob(os.path.join(base_dir, "*"))) if os.path.isdir(p)]
    artifact_paths: List[str] = []
    trained_timeframes: Dict[str, List[str]] = {}
    freshness: Dict[str, Dict[str, Any]] = {}
    stale_artifacts: List[str] = []
    missing_artifacts: List[str] = []
    per_symbol: Dict[str, Any] = {}
    now_ts = int(time.time())
    for folder in dirs:
        symbol = os.path.basename(folder).upper()
        if not symbol:
            continue
        patterns = list_crypto_artifact_patterns(folder)
        thresholds = sorted(patterns.get("thresholds", []))
        memories = sorted(patterns.get("memories", []))
        weight_sets = sorted(patterns.get("memory_weights", []))
        timeframes = sorted({tf for tf in [_extract_timeframe(p) for p in thresholds + memories + weight_sets] if tf})
        train_txt = os.path.join(folder, "trainer_last_training_time.txt")
        train_ts = int(_f(_safe_read_text(train_txt), 0.0))
        is_stale = bool(train_ts <= 0 or (now_ts - train_ts) > TRAINING_STALE_SECONDS)
        if is_stale:
            stale_artifacts.append(symbol)
        if not thresholds or not memories or not weight_sets:
            missing_artifacts.append(symbol)
        if any(patterns.values()):
            found_symbols.append(symbol)
            artifact_paths.extend(sum((list(v) for v in patterns.values()), []))
        trained_timeframes[symbol] = timeframes
        freshness[symbol] = {
            "trainer_last_training_time": int(train_ts),
            "fresh": bool(not is_stale),
            "stale_after_seconds": int(TRAINING_STALE_SECONDS),
        }
        per_symbol[symbol] = {
            "artifact_groups": {k: int(len(v)) for k, v in patterns.items()},
            "timeframes": timeframes,
            "paths": patterns,
            "trainer_last_training_time": int(train_ts),
            "fresh": bool(not is_stale),
        }
    return {
        "base_dir": base_dir,
        "trained_artifacts_found": int(len(found_symbols)),
        "trained_symbols": sorted(found_symbols),
        "trained_timeframes": trained_timeframes,
        "artifact_paths": sorted(artifact_paths),
        "artifact_freshness": freshness,
        "stale_artifacts": sorted(set(stale_artifacts)),
        "missing_artifacts": sorted(set(missing_artifacts)),
        "per_symbol": per_symbol,
        "freshness_rule": f"trainer_last_training_time.txt must be <= {TRAINING_STALE_SECONDS}s old",
        "live_consumer_function": "engines.pt_thinker.step_coin",
    }


def load_crypto_artifact_features(symbol_dir: str, as_of_ts: int = 0) -> Dict[str, Any]:
    base_dir = os.path.dirname(symbol_dir)
    symbol = os.path.basename(symbol_dir).upper()
    resolved_dir = _resolve_symbol_dir(base_dir, symbol)
    if not os.path.isdir(resolved_dir):
        return {"usable": False, "reason_if_not_used": "symbol_dir_missing"}
    discovery = discover_crypto_trained_artifacts(os.path.dirname(resolved_dir), symbols=[os.path.basename(resolved_dir)])
    symbol = os.path.basename(resolved_dir).upper()
    meta = (discovery.get("per_symbol", {}) if isinstance(discovery.get("per_symbol", {}), dict) else {}).get(symbol, {})
    if not isinstance(meta, dict):
        return {"usable": False, "reason_if_not_used": "artifact_discovery_missing"}
    train_ts = int(_f(meta.get("trainer_last_training_time", 0), 0.0))
    if as_of_ts > 0 and train_ts > 0 and train_ts > int(as_of_ts):
        return {
            "usable": False,
            "reason_if_not_used": "artifact_training_time_after_candidate",
            "trainer_last_training_time": int(train_ts),
        }
    thresholds = list((meta.get("paths", {}) if isinstance(meta.get("paths", {}), dict) else {}).get("thresholds", []) or [])
    threshold_vals: List[float] = []
    for path in thresholds:
        txt = _safe_read_text(path)
        if txt:
            threshold_vals.append(_f(txt, 0.0))
    memory_count_total = 0
    for path in list((meta.get("paths", {}) if isinstance(meta.get("paths", {}), dict) else {}).get("memories", []) or []):
        txt = _safe_read_text(path)
        if txt:
            memory_count_total += int(len([part for part in txt.split("~") if _s(part)]))
    return {
        "usable": True,
        "trained_freshness_flag": bool(meta.get("fresh", False)),
        "active_timeframe_count": int(len(list(meta.get("timeframes", []) or []))),
        "trained_timeframes": list(meta.get("timeframes", []) or []),
        "trainer_last_training_time": int(train_ts),
        "threshold_mean": round(sum(threshold_vals) / max(1, len(threshold_vals)), 6) if threshold_vals else 0.0,
        "threshold_min": round(min(threshold_vals), 6) if threshold_vals else 0.0,
        "threshold_max": round(max(threshold_vals), 6) if threshold_vals else 0.0,
        "pattern_memory_count": int(memory_count_total),
        "artifact_feature_names": [
            "trained_freshness_flag",
            "active_timeframe_count",
            "trained_timeframes",
            "trainer_last_training_time",
            "threshold_mean",
            "threshold_min",
            "threshold_max",
            "pattern_memory_count",
        ],
        "reason_if_not_used": "",
    }
