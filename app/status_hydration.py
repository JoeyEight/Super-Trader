from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List

from app.json_codec import load as json_load
from app.json_codec import loads as json_loads
from app.scan_diagnostics_schema import normalize_scan_diagnostics

_JSON_DICT_CACHE: Dict[str, tuple[tuple[int, int], Dict[str, Any], float]] = {}
_JSONL_CACHE: Dict[tuple[str, int], tuple[tuple[int, int], List[Dict[str, Any]], float]] = {}
_CACHE_MAX_ENTRIES = 256


def _file_sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
        return int(getattr(st, "st_mtime_ns", 0) or 0), int(getattr(st, "st_size", 0) or 0)
    except Exception:
        return None


def _trim_cache(cache: Dict[Any, tuple[Any, Any, float]], max_entries: int = _CACHE_MAX_ENTRIES) -> None:
    if len(cache) <= int(max_entries):
        return
    try:
        drop = max(1, len(cache) - int(max_entries))
        oldest = sorted(cache.items(), key=lambda item: float((item[1][2] if isinstance(item[1], tuple) and len(item[1]) >= 3 else 0.0)))[:drop]
        for key, _ in oldest:
            cache.pop(key, None)
    except Exception:
        # If trimming fails, keep cache best-effort without breaking callers.
        pass


def safe_read_json_dict(path: str) -> Dict[str, Any]:
    if not str(path or "").strip():
        return {}
    sig = _file_sig(path)
    if sig is None:
        return {}
    key = os.path.abspath(str(path))
    cached = _JSON_DICT_CACHE.get(key)
    if isinstance(cached, tuple) and len(cached) >= 3 and cached[0] == sig:
        payload = cached[1] if isinstance(cached[1], dict) else {}
        # Return a shallow copy so callers don't mutate cache entries.
        return dict(payload)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json_load(f, default={})
        out = payload if isinstance(payload, dict) else {}
        _JSON_DICT_CACHE[key] = (sig, dict(out), float(time.time()))
        _trim_cache(_JSON_DICT_CACHE)
        return out
    except Exception:
        return {}


def safe_read_jsonl_dicts(path: str, limit: int = 200) -> List[Dict[str, Any]]:
    if not str(path or "").strip():
        return []
    lim = max(1, int(limit or 200))
    sig = _file_sig(path)
    if sig is None:
        return []
    key = (os.path.abspath(str(path)), int(lim))
    cached = _JSONL_CACHE.get(key)
    if isinstance(cached, tuple) and len(cached) >= 3 and cached[0] == sig:
        rows = cached[1] if isinstance(cached[1], list) else []
        return [dict(row) for row in rows if isinstance(row, dict)]
    lines: List[str] = []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = int(f.tell() or 0)
            if size <= 0:
                return []
            window = min(size, max(8192, (512 * lim), (512 * 1024)))
            f.seek(-window, os.SEEK_END)
            blob = f.read(window)
        lines = str(blob.decode("utf-8", errors="ignore")).splitlines()
        if window < size and lines:
            lines = lines[1:]
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for ln in lines[-lim:]:
        txt = str(ln or "").strip()
        if not txt:
            continue
        try:
            obj = json_loads(txt, default=None)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    _JSONL_CACHE[key] = (sig, [dict(row) for row in rows], float(time.time()))
    _trim_cache(_JSONL_CACHE)
    return rows


def _is_missing_metric(value: Any) -> bool:
    txt = str(value or "").strip().lower()
    return txt in {"", "n/a", "pending account link", "none", "null"}


def payload_age_seconds(payload: Dict[str, Any], now_ts: float | None = None) -> float:
    if not isinstance(payload, dict):
        return float("inf")
    try:
        now = float(now_ts if now_ts is not None else time.time())
    except Exception:
        now = float(time.time())
    try:
        ts = float(payload.get("ts", payload.get("updated_at", 0)) or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0.0:
        return float("inf")
    return max(0.0, now - ts)


def market_status_has_account_snapshot(status_payload: Dict[str, Any]) -> bool:
    if not isinstance(status_payload, dict):
        return False
    for key in ("buying_power", "margin_available", "cash", "equity", "nav"):
        if not _is_missing_metric(status_payload.get(key)):
            return True
    raw_positions = status_payload.get("raw_positions", [])
    if isinstance(raw_positions, list) and raw_positions:
        return True
    positions_preview = status_payload.get("positions_preview", [])
    if isinstance(positions_preview, list) and positions_preview:
        return True
    return False


def needs_market_snapshot_refresh(
    status_payload: Dict[str, Any],
    loop_payload: Dict[str, Any],
    market_key: str,
    *,
    now_ts: float | None = None,
    stale_after_s: float = 45.0,
) -> bool:
    status = status_payload if isinstance(status_payload, dict) else {}
    loop = loop_payload if isinstance(loop_payload, dict) else {}
    market = str(market_key or "").strip().lower()
    threshold = max(5.0, float(stale_after_s or 45.0))
    status_age = payload_age_seconds(status, now_ts=now_ts)
    if math.isinf(status_age) or status_age > threshold:
        return True
    if not market_status_has_account_snapshot(status):
        return True
    state = str(status.get("state", "") or "").upper().strip()
    if state != "READY":
        return True
    try:
        loop_heartbeat_ts = float(loop.get("heartbeat_ts", loop.get("ts", 0)) or 0.0)
    except Exception:
        loop_heartbeat_ts = 0.0
    if market in {"stocks", "forex"} and loop_heartbeat_ts > 0.0:
        heartbeat_age = payload_age_seconds({"ts": loop_heartbeat_ts}, now_ts=now_ts)
        if heartbeat_age > threshold:
            return True
    return False


def load_market_status_bundle(
    *,
    status_path: str,
    trader_path: str,
    thinker_path: str,
    scan_diag_path: str,
    history_path: str = "",
    history_limit: int = 120,
    market_key: str = "",
) -> Dict[str, Any]:
    guessed_market = str(market_key or "").strip().lower()
    if not guessed_market:
        path_low = str(scan_diag_path or "").lower()
        if "/stocks/" in path_low or "\\stocks\\" in path_low:
            guessed_market = "stocks"
        elif "/forex/" in path_low or "\\forex\\" in path_low:
            guessed_market = "forex"
    return {
        "status": safe_read_json_dict(status_path),
        "trader": safe_read_json_dict(trader_path),
        "thinker": safe_read_json_dict(thinker_path),
        "scan_diagnostics": normalize_scan_diagnostics(safe_read_json_dict(scan_diag_path), market=guessed_market),
        "history": safe_read_jsonl_dicts(history_path, limit=history_limit),
    }
