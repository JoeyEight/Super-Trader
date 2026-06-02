from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Tuple

from app.confidence_calibration import build_confidence_calibration_payload
from app.regime_classifier import build_all_market_regimes
from app.settings_utils import sanitize_settings
from app.shadow_scorecard import build_shadow_scorecards
from app.walkforward_report import build_walkforward_report


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


def _norm_exit_trigger(tag: str) -> str:
    txt = _s(tag).upper()
    if not txt:
        return "Unknown"
    if "MANUAL" in txt:
        return "Manual"
    if "TRAIL" in txt:
        return "Trailing"
    if "STALE" in txt:
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
    for path in paths:
        src_rows = _safe_read_jsonl(path)
        if not src_rows:
            continue
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
    return {"market": m, "state": "READY" if rows else "NO_DATA", "source": source, "events": rows}


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
                "entry_ts": int(entry_ts),
                "exit_ts": int(ts),
                "qty": float(qty),
                "entry_price": round(float(entry_px), 10),
                "exit_price": round(float(exit_px), 10),
                "hold_hours": round(float(max(0.0, hold_s / 3600.0)), 6),
                "pnl_usd": round(float(pnl_usd), 8),
                "actual_exit_trigger": _norm_exit_trigger(_s(row.get("tag", ""))),
                "event_exit_tag": _s(row.get("tag", "")),
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
            open_lots[sym].append(
                {
                    "ts": int(_f(row.get("ts", 0), 0.0)),
                    "qty": float(_f(row.get("qty", 0.0), 0.0)),
                    "price": float(_f(row.get("price", 0.0), 0.0)),
                    "side": _s(row.get("side", "")).lower(),
                }
            )
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
            closed.append(
                {
                    "symbol": sym,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "qty": float(take),
                    "entry_price": round(entry_px, 10),
                    "exit_price": round(exit_px, 10),
                    "hold_hours": round(hold_h, 6),
                    "pnl_usd": round(pnl_usd, 8),
                    "actual_exit_trigger": _norm_exit_trigger(_s(row.get("tag", ""))),
                    "event_exit_tag": _s(row.get("tag", "")),
                }
            )
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
    holds_non_positive = sum(1 for r in closed if _f(r.get("hold_hours", 0.0), 0.0) <= 0.0)
    return {
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


def _predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
) -> Dict[str, Any]:
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


def _score_with_threshold(rows: List[Dict[str, Any]], threshold: float, market: str) -> Dict[str, Any]:
    admitted = [r for r in rows if _f(r.get("predicted_confidence", 0.0), 0.0) >= float(threshold)]
    metrics = _metrics_for_rows(admitted)
    coverage = (len(admitted) / max(1, len(rows))) if rows else 0.0
    score = _objective_score(metrics, market)
    if coverage < 0.30:
        score -= (0.30 - coverage) * 120.0
    return {"score": score, "coverage": coverage, "metrics": metrics, "admitted": admitted}


def _calibrate_abstain_threshold(train_rows: List[Dict[str, Any]], market: str) -> Dict[str, Any]:
    if len(train_rows) < 24:
        return {"threshold": 0.6, "coverage": 1.0, "metrics": _metrics_for_rows(train_rows)}
    split = max(12, int(len(train_rows) * 0.70))
    base = train_rows[:split]
    val = train_rows[split:]
    pred_val: List[Dict[str, Any]] = []
    running = list(base)
    for row in val:
        regime = _regime_from_prior(running)
        pred = _predict_one(train_rows=running, candidate=row, regime=regime)
        merged = dict(row)
        merged.update(pred)
        pred_val.append(merged)
        running.append(row)
    best = {"threshold": 0.6, "score": -1e9, "coverage": 0.0, "metrics": _metrics_for_rows(pred_val)}
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        row = _score_with_threshold(pred_val, thr, market)
        score = _f(row.get("score", -1e9), -1e9)
        cov = _f(row.get("coverage", 0.0), 0.0)
        if score > _f(best.get("score", -1e9), -1e9) or (
            abs(score - _f(best.get("score", -1e9), -1e9)) <= 1e-9 and cov > _f(best.get("coverage", 0.0), 0.0)
        ):
            best = {"threshold": thr, "score": score, "coverage": cov, "metrics": row.get("metrics", {})}
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


def build_synthetic_replay_artifact(hub_dir: str, market: str, closed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = _s(market).lower()
    rows = sorted(list(closed_rows or []), key=lambda r: int(_f(r.get("exit_ts", 0), 0.0)))
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
    preds_all: List[Dict[str, Any]] = []
    thr_values: List[float] = []
    max_windows = 10
    win_count = 0
    for end in range(train_n_min, n - test_n + 1, step_n):
        train = rows[:end]
        test = rows[end : end + test_n]
        if len(test) < 4:
            continue
        abstain = _calibrate_abstain_threshold(train, m)
        threshold = _f(abstain.get("threshold", 0.6), 0.6)
        thr_values.append(float(threshold))
        test_preds: List[Dict[str, Any]] = []
        prior = list(train)
        for r in test:
            regime = _regime_from_prior(prior)
            rr = dict(r)
            rr["regime"] = regime
            pred = _predict_one(train_rows=prior, candidate=rr, regime=regime)
            merged = dict(rr)
            merged.update(pred)
            merged["predicted_exit_ts"] = int(
                _f(rr.get("entry_ts", 0.0), 0.0) + int(round(_f(pred.get("predicted_hold_hours", 0.0), 0.0) * 3600.0))
            )
            test_preds.append(merged)
            prior.append(rr)
        preds_all.extend(test_preds)
        scored = _score_with_threshold(test_preds, threshold, m)
        admitted = scored.get("admitted", []) if isinstance(scored.get("admitted", []), list) else []
        admitted_all.extend(admitted)
        windows.append(
            {
                "train_rows": int(len(train)),
                "test_rows": int(len(test_preds)),
                "admitted_rows": int(len(admitted)),
                "threshold": round(float(threshold), 4),
                "coverage": round(_f(scored.get("coverage", 0.0), 0.0), 6),
                "metrics": scored.get("metrics", {}),
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

    return {
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
            "test_trades": int(sum(int(w.get("test_rows", 0) or 0) for w in windows)),
            "admitted_test_trades": int(len(admitted_all)),
            "walkforward_windows": int(len(windows)),
        },
        "abstain_policy": {
            "threshold_median": round(_median([float(v) for v in thr_values], default=0.6), 4),
            "test_coverage": round(float(coverage), 6),
        },
        "best_test_metrics": metrics,
        "hybrid_test_metrics": metrics,
        "worst_window_metrics": worst,
        "confidence_intervals": ci,
        "walkforward_windows": windows,
        "segmented_by_actual_trigger": seg_trigger,
        "segmented_by_regime": seg_regime,
        "iterations": [{"iteration": 1, "test_predictions": admitted_all}],
    }


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
    ts = int(time.time())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    out_dir = os.path.join(hub_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)

    markets = ["crypto", "stocks", "forex"]
    dataset_reports: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    closed_by_market: Dict[str, List[Dict[str, Any]]] = {}
    for m in markets:
        loaded = load_market_trade_events(hub_dir, m)
        events = loaded.get("events", []) if isinstance(loaded.get("events", []), list) else []
        closed = build_closed_trades(events, m).get("closed_trades", [])
        closed_by_market[m] = list(closed if isinstance(closed, list) else [])
        snap_path = os.path.join(out_dir, f"{m}_closed_trades_v1_{stamp}.jsonl")
        _write_jsonl(snap_path, closed if isinstance(closed, list) else [])
        snapshots[m] = {
            "path": snap_path,
            "sha256": _sha256_file(snap_path),
            "rows": int(len(closed) if isinstance(closed, list) else 0),
        }
        dataset_reports[m] = build_market_dataset_quality(hub_dir, m)

    # Build/refresh replay artifacts for all markets from normalized closed trades.
    openai_dir = os.path.join(hub_dir, "openai")
    os.makedirs(openai_dir, exist_ok=True)
    synthetic_paths: Dict[str, str] = {}
    for m in markets:
        payload = build_synthetic_replay_artifact(hub_dir, m, closed_by_market.get(m, []))
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
    promotion_readiness: Dict[str, Any] = {}
    for m in markets:
        row = replay_diag.get(m, {}) if isinstance(replay_diag.get(m, {}), dict) else {}
        metrics = row.get("headline_metrics", {}) if isinstance(row.get("headline_metrics", {}), dict) else {}
        payload = _safe_read_json(_s(row.get("source", "")))
        ci = payload.get("confidence_intervals", {}) if isinstance(payload.get("confidence_intervals", {}), dict) else {}
        worst = payload.get("worst_window_metrics", {}) if isinstance(payload.get("worst_window_metrics", {}), dict) else {}
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        n_windows = int(_f(meta.get("walkforward_windows", 0), 0.0))
        n_test = int(_f(meta.get("test_trades", 0), 0.0))
        trig_scored = _f(metrics.get("trigger_scored_trades", 0.0), 0.0)
        trig_cov = _f(metrics.get("trigger_coverage_pct", 0.0), 0.0)
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
        state = "PASS" if not blockers and _s(row.get("state", "")) == "READY" else "BLOCK"
        promotion_readiness[m] = {
            "state": state,
            "blockers": blockers,
            "metrics": metrics,
            "source": _s(row.get("source", "")),
            "windows": int(n_windows),
            "test_trades": int(n_test),
        }

    return {
        "ts": ts,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
        "base_dir": base_dir,
        "hub_dir": hub_dir,
        "dataset_quality": dataset_reports,
        "dataset_snapshots": snapshots,
        "synthetic_replay_artifacts": synthetic_paths,
        "market_regimes": regimes,
        "walkforward_report": walk,
        "confidence_calibration": calibration,
        "shadow_scorecards": shadow,
        "replay_diagnostics": replay_diag,
        "promotion_thresholds": promotion_thresholds,
        "promotion_minima": promotion_minima,
        "promotion_readiness": promotion_readiness,
    }
