from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

_BINS: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50, 5.00, 999.0)


def _safe_read_jsonl(path: str, max_lines: int = 6000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return out
    for ln in lines[-max(1, int(max_lines)):]:
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _trade_history_to_execution_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        side = str(row.get("side", "") or "").strip().lower()
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        score = _f(row.get("score", row.get("dynamic_score", 0.0)), 0.0)
        ts = int(_f(row.get("ts", 0.0), 0.0))
        if side == "buy":
            out.append(
                {
                    "ts": ts,
                    "event": "entry",
                    "ok": True,
                    "symbol": symbol,
                    "score": float(score),
                    "payload": dict(row),
                }
            )
            continue
        if side != "sell":
            continue
        pnl_pct = _f(row.get("pnl_pct", 0.0), 0.0)
        realized = _f(row.get("realized_profit_usd", 0.0), 0.0)
        ok = bool((realized >= 0.0) or (pnl_pct >= 0.0))
        out.append(
            {
                "ts": ts,
                "event": "exit",
                "ok": bool(ok),
                "symbol": symbol,
                "score": float(score),
                "pnl_pct": float(pnl_pct),
                "pnl_usd": float(realized),
                "payload": dict(row),
            }
        )
    return out


def _calibration_rows_for_market(hub_dir: str, market: str) -> List[Dict[str, Any]]:
    mk = str(market or "").strip().lower()
    if mk in {"stocks", "forex"}:
        return _safe_read_jsonl(os.path.join(hub_dir, mk, "execution_audit.jsonl"), max_lines=8000)
    if mk == "crypto":
        rows = _safe_read_jsonl(os.path.join(hub_dir, "crypto", "execution_audit.jsonl"), max_lines=10000)
        if rows:
            return rows
        legacy = _safe_read_jsonl(os.path.join(hub_dir, "trade_history.jsonl"), max_lines=12000)
        return _trade_history_to_execution_rows(legacy)
    return []


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _row_score(row: Dict[str, Any]) -> float:
    score = abs(_f(row.get("score", 0.0), 0.0))
    if score <= 0.0:
        payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
        score = abs(_f(payload.get("score", 0.0), 0.0))
    return float(score)


def _row_outcome(row: Dict[str, Any]) -> int:
    evt = str(row.get("event", "") or "").strip().lower()
    if evt != "exit":
        return -1
    # Calibrate against realized exit outcomes (not entry admissions).
    # Prefer explicit PnL fields because historical "ok" flags can be stale/noisy.
    pnl_usd = row.get("pnl_usd", row.get("realized_pnl_usd", None))
    if pnl_usd is None and isinstance(row.get("payload"), dict):
        pnl_usd = row["payload"].get("realized_profit_usd", None)
    pnl_pct = row.get("pnl_pct", None)
    if pnl_pct is None and isinstance(row.get("payload"), dict):
        pnl_pct = row["payload"].get("pnl_pct", None)

    has_usd = pnl_usd is not None
    has_pct = pnl_pct is not None
    if has_usd or has_pct:
        usd_v = _f(pnl_usd, 0.0) if has_usd else 0.0
        pct_v = _f(pnl_pct, 0.0) if has_pct else 0.0
        return 1 if ((has_usd and usd_v >= 0.0) or (has_pct and pct_v >= 0.0)) else 0

    ok = bool(row.get("ok", False))
    if ok:
        return 1
    if evt in {"exit_fail", "shadow_live_divergence"}:
        return 0
    if not ok:
        return 0
    return -1


def _bin_label(lo: float, hi: float) -> str:
    if hi >= 999.0:
        return f">={lo:.2f}"
    return f"{lo:.2f}-{hi:.2f}"


def _build_curve(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: List[List[int]] = [[] for _ in range(len(_BINS) - 1)]
    for row in rows:
        if not isinstance(row, dict):
            continue
        outcome = _row_outcome(row)
        if outcome < 0:
            continue
        score = _row_score(row)
        for i in range(len(_BINS) - 1):
            lo = float(_BINS[i])
            hi = float(_BINS[i + 1])
            if score < lo:
                continue
            if score >= hi:
                continue
            buckets[i].append(outcome)
            break

    curve: List[Dict[str, Any]] = []
    for i in range(len(_BINS) - 1):
        lo = float(_BINS[i])
        hi = float(_BINS[i + 1])
        vals = buckets[i]
        n = len(vals)
        wins = sum(1 for x in vals if int(x) > 0)
        success = (100.0 * wins / max(1, n)) if n > 0 else 0.0
        curve.append(
            {
                "bin": _bin_label(lo, hi),
                "min_score": round(lo, 4),
                "max_score": (None if hi >= 999.0 else round(hi, 4)),
                "samples": int(n),
                "wins": int(wins),
                "success_rate_pct": round(float(success), 4),
            }
        )
    return curve


def _recommended_threshold(curve: List[Dict[str, Any]], base_threshold: float, min_samples: int = 18, target_success_pct: float = 55.0) -> Dict[str, Any]:
    base = max(0.0, float(base_threshold or 0.0))
    rec = base
    reason = "insufficient_data"
    for row in curve:
        n = int(row.get("samples", 0) or 0)
        success = float(row.get("success_rate_pct", 0.0) or 0.0)
        lo = float(row.get("min_score", 0.0) or 0.0)
        if n < int(min_samples):
            continue
        if success >= float(target_success_pct):
            rec = max(base * 0.75, lo)
            reason = f"first_bin_meets_target({success:.1f}% >= {target_success_pct:.1f}%)"
            break
    drift = rec - base
    return {
        "base_threshold": round(base, 6),
        "recommended_threshold": round(float(rec), 6),
        "delta": round(float(drift), 6),
        "reason": reason,
        "min_samples": int(min_samples),
        "target_success_pct": round(float(target_success_pct), 4),
    }


def build_market_confidence_calibration(
    hub_dir: str,
    market: str,
    base_threshold: float,
    min_samples: int = 18,
    target_success_pct: float = 55.0,
) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    if m not in {"stocks", "forex", "crypto"}:
        return {"market": m, "state": "ERROR", "msg": "unsupported market"}

    rows = _calibration_rows_for_market(hub_dir, m)
    curve = _build_curve(rows)
    rec = _recommended_threshold(curve, base_threshold=base_threshold, min_samples=min_samples, target_success_pct=target_success_pct)

    counted = 0
    wins = 0
    for row in rows:
        outcome = _row_outcome(row if isinstance(row, dict) else {})
        if outcome < 0:
            continue
        counted += 1
        if outcome > 0:
            wins += 1

    return {
        "ts": int(time.time()),
        "market": m,
        "state": "READY",
        "samples": int(counted),
        "wins": int(wins),
        "win_rate_pct": round((100.0 * wins / max(1, counted)) if counted > 0 else 0.0, 4),
        "curve": curve,
        "recommendation": rec,
    }


def build_confidence_calibration_payload(hub_dir: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    s = settings if isinstance(settings, dict) else {}
    stock_thr = _f(s.get("stock_score_threshold", 0.2), 0.2)
    fx_thr = _f(s.get("forex_score_threshold", 0.2), 0.2)
    crypto_thr = _f(s.get("crypto_dynamic_min_projected_edge_pct", s.get("crypto_allocator_signal_floor", 0.15)), 0.15)
    min_samples = max(6, int(_f(s.get("adaptive_confidence_min_samples", 18), 18)))
    target_success = max(30.0, min(90.0, _f(s.get("adaptive_confidence_target_success_pct", 55.0), 55.0)))
    return {
        "ts": int(time.time()),
        "crypto": build_market_confidence_calibration(
            hub_dir,
            "crypto",
            base_threshold=crypto_thr,
            min_samples=min_samples,
            target_success_pct=target_success,
        ),
        "stocks": build_market_confidence_calibration(
            hub_dir,
            "stocks",
            base_threshold=stock_thr,
            min_samples=min_samples,
            target_success_pct=target_success,
        ),
        "forex": build_market_confidence_calibration(
            hub_dir,
            "forex",
            base_threshold=fx_thr,
            min_samples=min_samples,
            target_success_pct=target_success,
        ),
    }
