from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl(path: str, max_lines: int = 400) -> List[Dict[str, Any]]:
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


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _symbol_of(row: Dict[str, Any], market: str) -> str:
    mk = str(market or "").strip().lower()
    if mk in {"stocks", "crypto"}:
        return str(row.get("symbol", "") or "").strip().upper()
    return str(row.get("pair", row.get("instrument", row.get("symbol", ""))) or "").strip().upper()


def _score_of(row: Dict[str, Any]) -> float:
    score = _f(row.get("score", 0.0), 0.0)
    if abs(score) <= 0.0:
        score = _f(row.get("score_raw", 0.0), 0.0)
    return float(score)


def _threshold_grid(base_threshold: float) -> List[float]:
    base = max(0.01, float(base_threshold))
    raw = [
        base * 0.50,
        base * 0.75,
        base,
        base * 1.10,
        base * 1.25,
        base * 1.50,
        base * 2.00,
    ]
    out = sorted({round(max(0.01, min(10.0, float(v))), 6) for v in raw})
    return out


def _is_entry_side(side: str) -> bool:
    s = str(side or "").strip().lower()
    return s in {"long", "short", "buy", "sell"}


def _scenario(rows: Iterable[Dict[str, Any]], market: str, threshold: float) -> Dict[str, Any]:
    t = max(0.01, float(threshold))
    considered = 0
    actionable = 0
    entry_ready = 0
    long_count = 0
    short_count = 0
    watch_count = 0
    abs_sum = 0.0
    picked: List[Tuple[float, str]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        score = _score_of(row)
        if abs(score) <= 0.0:
            continue
        considered += 1
        if abs(score) < t:
            continue
        actionable += 1
        abs_sum += abs(score)
        side = str(row.get("side", "watch") or "watch").strip().lower()
        if side in {"short", "sell"}:
            short_count += 1
        elif side in {"long", "buy"}:
            long_count += 1
        else:
            watch_count += 1
        if _is_entry_side(side) and bool(row.get("eligible_for_entry", True)):
            entry_ready += 1
        symbol = _symbol_of(row, market)
        if symbol:
            picked.append((abs(score), symbol))

    picked.sort(key=lambda it: it[0], reverse=True)
    preview = [sym for _, sym in picked[:8]]
    avg_abs = (abs_sum / max(1, actionable)) if actionable > 0 else 0.0
    return {
        "threshold": round(float(t), 6),
        "considered": int(considered),
        "actionable": int(actionable),
        "entry_ready": int(entry_ready),
        "long": int(long_count),
        "short": int(short_count),
        "watch": int(watch_count),
        "avg_abs_score": round(float(avg_abs), 6),
        "top_symbols": preview,
    }


def _pick_recommendation(scenarios: List[Dict[str, Any]], current_threshold: float, target_entries: int) -> Dict[str, Any]:
    if not scenarios:
        return {
            "recommended_threshold": round(float(current_threshold), 6),
            "reason": "no_scenarios",
            "target_entries": int(target_entries),
        }

    cur = max(0.01, float(current_threshold))
    target = max(1, int(target_entries))

    def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        entry_ready = int(row.get("entry_ready", 0) or 0)
        actionable = int(row.get("actionable", 0) or 0)
        avg_abs = float(row.get("avg_abs_score", 0.0) or 0.0)
        threshold = float(row.get("threshold", cur) or cur)
        return (
            abs(entry_ready - target),
            abs(actionable - target),
            -avg_abs,
            abs(threshold - cur),
        )

    best = sorted(scenarios, key=_rank_key)[0]
    rec = float(best.get("threshold", cur) or cur)
    delta = rec - cur
    reason = (
        f"target_entries={target} | current={cur:.4f} -> recommended={rec:.4f} "
        f"(entry_ready={int(best.get('entry_ready', 0) or 0)}, actionable={int(best.get('actionable', 0) or 0)})"
    )
    return {
        "recommended_threshold": round(float(rec), 6),
        "current_threshold": round(float(cur), 6),
        "delta": round(float(delta), 6),
        "reason": reason,
        "target_entries": int(target),
    }


def _reason_breakdown(rank_rows: Iterable[Dict[str, Any]], max_items: int = 12) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    total = 0
    for row in rank_rows:
        if not isinstance(row, dict):
            continue
        rejected = row.get("rejected", []) if isinstance(row.get("rejected", []), list) else []
        for rej in rejected:
            if not isinstance(rej, dict):
                continue
            reason = str(rej.get("reason", "unknown") or "unknown").strip().lower()
            if not reason:
                reason = "unknown"
            counts[reason] = int(counts.get(reason, 0) or 0) + 1
            total += 1
    rows = sorted(counts.items(), key=lambda it: (it[1], it[0]), reverse=True)
    out: List[Dict[str, Any]] = []
    for reason, count in rows[: max(1, int(max_items))]:
        pct = (100.0 * float(count) / float(max(1, total))) if total > 0 else 0.0
        out.append({"reason": reason, "count": int(count), "pct": round(float(pct), 4)})
    return out


def _extract_scored_rows(thinker: Dict[str, Any], rank_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scores = thinker.get("all_scores", []) if isinstance(thinker.get("all_scores", []), list) else []
    if scores:
        return [row for row in scores if isinstance(row, dict)]
    ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
    if ranked:
        return [row for row in ranked if isinstance(row, dict)]
    if rank_rows:
        latest = rank_rows[-1]
        top = latest.get("top", []) if isinstance(latest.get("top", []), list) else []
        return [row for row in top if isinstance(row, dict)]
    return []


def replay_target_entries_for_market(settings: Dict[str, Any] | None, market: str) -> int:
    cfg = settings if isinstance(settings, dict) else {}
    m = str(market or "").strip().lower()
    if m == "stocks":
        raw = cfg.get("replay_target_entries_stocks", 3)
        fallback = 3
    elif m == "forex":
        raw = cfg.get("replay_target_entries_forex", 4)
        fallback = 4
    else:
        raw = cfg.get("replay_target_entries_crypto", 5)
        fallback = 5
    return max(1, min(20, int(_f(raw, fallback))))


def recommend_threshold_from_scores(
    scored_rows: List[Dict[str, Any]],
    market: str,
    current_threshold: float,
    target_entries: int,
) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    rows = [row for row in list(scored_rows or []) if isinstance(row, dict)]
    cur = max(0.01, float(current_threshold))
    target = max(1, min(20, int(target_entries)))
    grid = _threshold_grid(cur)
    scenarios = [_scenario(rows, m, th) for th in grid]
    rec = _pick_recommendation(scenarios, current_threshold=cur, target_entries=target)
    return {
        "market": m,
        "current_threshold": round(float(cur), 6),
        "target_entries": int(target),
        "scored_rows": int(len(rows)),
        "recommendation": rec,
        "scenarios": scenarios,
    }


def build_market_rejection_replay(
    hub_dir: str,
    market: str,
    settings: Dict[str, Any] | None = None,
    max_scan_rows: int = 240,
) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    if m not in {"stocks", "forex", "crypto"}:
        return {"market": m, "state": "ERROR", "msg": "unsupported market"}

    cfg = settings if isinstance(settings, dict) else {}
    if m == "crypto":
        thinker = _safe_read_json(os.path.join(hub_dir, "crypto_dynamic_status.json"))
        rank_rows = _safe_read_jsonl(os.path.join(hub_dir, "crypto", "scanner_rankings.jsonl"), max_lines=max_scan_rows)
    else:
        mdir = os.path.join(hub_dir, m)
        thinker_name = "stock_thinker_status.json" if m == "stocks" else "forex_thinker_status.json"
        thinker = _safe_read_json(os.path.join(mdir, thinker_name))
        rank_rows = _safe_read_jsonl(os.path.join(mdir, "scanner_rankings.jsonl"), max_lines=max_scan_rows)
    scored_rows = _extract_scored_rows(thinker, rank_rows)

    if m == "stocks":
        current_threshold = _f(cfg.get("stock_score_threshold", 0.2), 0.2)
    elif m == "forex":
        current_threshold = _f(cfg.get("forex_score_threshold", 0.2), 0.2)
    else:
        current_threshold = _f(
            cfg.get(
                "crypto_dynamic_min_projected_edge_pct",
                cfg.get("crypto_allocator_signal_floor", 0.15),
            ),
            0.15,
        )
    target_entries = replay_target_entries_for_market(cfg, m)
    replay = recommend_threshold_from_scores(
        scored_rows,
        market=m,
        current_threshold=current_threshold,
        target_entries=target_entries,
    )
    rec = replay.get("recommendation", {}) if isinstance(replay.get("recommendation", {}), dict) else {}
    scenarios = replay.get("scenarios", []) if isinstance(replay.get("scenarios", []), list) else []
    reasons = _reason_breakdown(rank_rows, max_items=10)

    symbol_rows: List[Dict[str, Any]] = []
    for row in scored_rows:
        score = _score_of(row)
        symbol = _symbol_of(row, m)
        if not symbol:
            continue
        symbol_rows.append(
            {
                "symbol": symbol,
                "score": round(float(score), 6),
                "abs_score": round(abs(float(score)), 6),
                "side": str(row.get("side", "watch") or "watch"),
                "eligible_for_entry": bool(row.get("eligible_for_entry", False)),
                "reason_logic": str(row.get("reason_logic", row.get("reason", "")) or ""),
            }
        )
    symbol_rows = sorted(symbol_rows, key=lambda it: (float(it.get("abs_score", 0.0) or 0.0), str(it.get("symbol", ""))), reverse=True)

    state = "READY" if scored_rows else "NO_DATA"
    msg = (
        "Replay complete."
        if scored_rows
        else "No scored candidates available yet. Run scanner first to generate replay scenarios."
    )
    return {
        "ts": int(time.time()),
        "market": m,
        "state": state,
        "msg": msg,
        "current_threshold": round(float(current_threshold), 6),
        "target_entries": int(target_entries),
        "scored_rows": int(len(scored_rows)),
        "scan_rows": int(len(rank_rows)),
        "recommendation": rec,
        "scenarios": scenarios,
        "rejected_reason_breakdown": reasons,
        "top_scored_symbols": symbol_rows[:20],
    }


def build_rejection_replay_report(hub_dir: str, settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "ts": int(time.time()),
        "crypto": build_market_rejection_replay(hub_dir, "crypto", settings=settings),
        "stocks": build_market_rejection_replay(hub_dir, "stocks", settings=settings),
        "forex": build_market_rejection_replay(hub_dir, "forex", settings=settings),
    }
