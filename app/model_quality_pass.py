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
        "manual_same_regime_support_count": float(len(regime_manual_rows)),
        "manual_symbol_recent_support_count": float(len(symbol_recent_manual)),
        "manual_recent_density_40": ratio(len(recent_manual_40), len(recent40)),
        "manual_recent_density_80": ratio(len(recent_manual_80), len(recent80)),
        "manual_symbol_long_hold_ratio": ratio(len(long_symbol_manual), len(long_symbol_rows)),
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
            score -= 0.30 * p_manual
            if manual_support_count >= 3 and candidate_hold_h > (1.35 * max(1.0, _f(proto.get("q75_hold_h", 0.0), 0.0))):
                score -= 0.40
        elif cls == "Stale Alignment":
            score += 0.80 * max(0.0, p_down - p_up)
            score += 0.65 * max(0.0, symbol_down - symbol_up)
            score += 0.55 * p_stale
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
        "up": (1.10 * p_up) + (0.45 * max(0.0, symbol_momentum)) + (0.35 * max(0.0, trig_scores.get("Trailing", 0.0) - trig_scores.get("Stale Alignment", 0.0))),
        "down": (1.10 * p_down) + (0.45 * max(0.0, -symbol_momentum)) + (0.35 * max(0.0, trig_scores.get("Stale Alignment", 0.0) - trig_scores.get("Trailing", 0.0))),
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
        "manual_same_symbol_support_count": int(_f(manual_signals.get("manual_same_symbol_support_count", 0.0), 0.0)),
        "manual_same_regime_support_count": int(_f(manual_signals.get("manual_same_regime_support_count", 0.0), 0.0)),
        "trigger_prototype_support": {
            k: {
                "support_count": int(v.get("support_count", 0) or 0),
                "symbol_support_count": int(v.get("symbol_support_count", 0) or 0),
                "symbol_regime_support_count": int(v.get("symbol_regime_support_count", 0) or 0),
            }
            for k, v in prototypes.items()
        },
    }


def _predict_one(
    *,
    train_rows: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    regime: str,
    market: str = "",
) -> Dict[str, Any]:
    if _s(market).lower() == "crypto":
        return _crypto_predict_one(train_rows=train_rows, candidate=candidate, regime=regime)
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
        pred = _predict_one(train_rows=running, candidate=row, regime=regime, market=market)
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
    abstained_all: List[Dict[str, Any]] = []
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
            pred = _predict_one(train_rows=prior, candidate=rr, regime=regime, market=m)
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
    if crypto_diag:
        crypto_diag["population_diagnostics"] = population_diag

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
        },
        "abstain_policy": {
            "threshold_median": round(_median([float(v) for v in thr_values], default=0.6), 4),
            "test_coverage": round(float(coverage), 6),
        },
        "best_test_metrics": metrics,
        "hybrid_test_metrics": metrics,
        "population_diagnostics": population_diag,
        "worst_window_metrics": worst,
        "confidence_intervals": ci,
        "walkforward_windows": windows,
        "segmented_by_actual_trigger": seg_trigger,
        "segmented_by_regime": seg_regime,
        "iterations": [{"iteration": 1, "test_predictions": admitted_all}],
    }
    if crypto_diag:
        payload["crypto_classifier_diagnostics"] = crypto_diag
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
    miss_hold: Dict[str, int] = {}
    miss_ret: Dict[str, int] = {}
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
        if pred_trig == "Stale Alignment":
            stale_predicted += 1
            if act_trig == "Stale Alignment":
                stale_correct += 1
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
        "trigger_class_prototypes_summary": proto_support,
        "misses_by_symbol": _top_counter(miss_symbol),
        "misses_by_regime": _top_counter(miss_regime),
        "misses_by_actual_trigger": _top_counter(miss_actual_trigger),
        "misses_by_predicted_trigger": _top_counter(miss_pred_trigger),
        "misses_by_hold_bucket": _top_counter(miss_hold),
        "misses_by_return_magnitude_bucket": _top_counter(miss_ret),
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
        "stale_vs_trailing_score_margin_examples": stale_vs_trailing_examples,
        "stale_vs_manual_score_margin_examples": stale_vs_manual_examples,
        "actual_up_trailing_pred_down_stale_examples": uptrail_downstale,
        "actual_manual_pred_directional_examples": manual_mispreds,
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
        promotion_readiness[m] = {
            "state": state,
            "blockers": blockers,
            "metrics": metrics,
            "source": _s(row.get("source", "")),
            "windows": int(n_windows),
            "test_trades": int(n_test),
            "full_test_trades": int(n_full_test),
            "admission_rate_pct": round(float(admission_rate), 4),
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
