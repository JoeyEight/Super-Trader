from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List

from app.scan_diagnostics_schema import normalize_scan_diagnostics
from app.scanner_quality import effective_reject_pressure


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tail_text_lines(path: str, max_lines: int = 2000, max_bytes: int = 2 * 1024 * 1024) -> List[str]:
    lim = max(1, int(max_lines or 1))
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = int(f.tell() or 0)
            if size <= 0:
                return []
            window = min(size, max(8192, int(max_bytes or 0), lim * 512))
            f.seek(-window, os.SEEK_END)
            blob = f.read(window)
    except Exception:
        return []
    try:
        lines = str(blob.decode("utf-8", errors="ignore")).splitlines()
    except Exception:
        return []
    if window < size and lines:
        # The first line may be partial when reading from a tail window.
        lines = lines[1:]
    out: List[str] = []
    for ln in lines:
        txt = str(ln or "").strip()
        if txt:
            out.append(txt)
    return out[-lim:]


def _safe_read_jsonl(path: str, max_lines: int = 2000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in _tail_text_lines(path, max_lines=max_lines):
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(x) for x in values)
    idx = int(round((len(arr) - 1) * max(0.0, min(1.0, float(q)))))
    idx = max(0, min(idx, len(arr) - 1))
    return float(arr[idx])


def parse_stale_signal_seconds(msg: Any) -> int:
    text = str(msg or "")
    m = re.search(r"signal\s+stale\s*\((\d+)s\s*>\s*(\d+)s\)", text, flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except Exception:
        return 0


def _audit_event_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        evt = str(row.get("event", "") or "").strip().lower()
        if not evt:
            continue
        out[evt] = int(out.get(evt, 0)) + 1
    return out


def _recent_rows(rows: List[Dict[str, Any]], horizon_s: int = 86400) -> List[Dict[str, Any]]:
    now = int(time.time())
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = int(row.get("ts", 0) or 0)
        if ts <= 0:
            continue
        if (now - ts) <= int(horizon_s):
            out.append(row)
    return out


def _spread_series_from_rankings(rows: List[Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for row in rows:
        top = row.get("top", [])
        if not isinstance(top, list):
            continue
        for cand in top[:5]:
            if not isinstance(cand, dict):
                continue
            try:
                spr = float(cand.get("spread_bps", 0.0) or 0.0)
            except Exception:
                spr = 0.0
            if spr > 0.0:
                vals.append(spr)
    return vals


def _data_source_reliability(scan_diag: Dict[str, Any], quality_report: Dict[str, Any], thinker_status: Dict[str, Any]) -> Dict[str, Any]:
    diag = scan_diag if isinstance(scan_diag, dict) else {}
    report = quality_report if isinstance(quality_report, dict) else {}
    thinker = thinker_status if isinstance(thinker_status, dict) else {}
    feed_health = diag.get("feed_health", {}) if isinstance(diag.get("feed_health", {}), dict) else {}
    reject_summary = diag.get("reject_summary", {}) if isinstance(diag.get("reject_summary", {}), dict) else {}
    fallback_cached = bool(thinker.get("fallback_cached", False))
    fallback_age_s = int(float(thinker.get("fallback_age_s", 0) or 0)) if fallback_cached else 0
    market_open = bool(diag.get("market_open", True))
    market_closed_mode = (not market_open) and (str(reject_summary.get("dominant_reason", "") or "").strip().lower() == "market_closed")

    if market_closed_mode:
        return {
            "score": 100.0,
            "level": "high",
            "reject_rate_pct": 0.0,
            "fallback_cached": bool(fallback_cached),
            "fallback_age_s": int(fallback_age_s),
            "feed_health": {
                "errors": int(feed_health.get("errors", 0) or 0),
                "stale": int(feed_health.get("stale", 0) or 0),
                "total": int(feed_health.get("total", 0) or 0),
            },
        }

    reject_rate = float(report.get("reject_rate_pct", reject_summary.get("reject_rate_pct", 0.0)) or 0.0)
    quality_penalty = max(0.0, min(60.0, reject_rate * 0.45))
    fallback_penalty = min(30.0, (float(fallback_age_s) / 1800.0) * 12.0) if fallback_cached else 0.0
    feed_penalty = 0.0
    if feed_health:
        try:
            total = max(1, int(feed_health.get("total", 0) or 0))
            fail = int(feed_health.get("errors", 0) or 0) + int(feed_health.get("stale", 0) or 0)
            feed_penalty = min(30.0, (100.0 * fail / total) * 0.20)
        except Exception:
            feed_penalty = 0.0
    score = max(0.0, min(100.0, 100.0 - quality_penalty - fallback_penalty - feed_penalty))
    level = "high" if score >= 85.0 else ("medium" if score >= 65.0 else "low")
    return {
        "score": round(score, 3),
        "level": level,
        "reject_rate_pct": round(reject_rate, 3),
        "fallback_cached": bool(fallback_cached),
        "fallback_age_s": int(fallback_age_s),
        "feed_health": {
            "errors": int(feed_health.get("errors", 0) or 0),
            "stale": int(feed_health.get("stale", 0) or 0),
            "total": int(feed_health.get("total", 0) or 0),
        },
    }


def _fill_quality_by_hour(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_hour: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        evt = str(row.get("event", "") or "").strip().lower()
        if evt not in {"entry", "entry_fail", "exit", "exit_fail"}:
            continue
        ts = int(row.get("ts", 0) or 0)
        if ts <= 0:
            continue
        hour = int(time.localtime(ts).tm_hour)
        bucket = by_hour.get(hour, {"samples": 0, "ok": 0, "spread_bps": [], "pnl_usd": 0.0})
        bucket["samples"] = int(bucket.get("samples", 0) or 0) + 1
        if bool(row.get("ok", False)):
            bucket["ok"] = int(bucket.get("ok", 0) or 0) + 1
        try:
            spr = float(row.get("spread_bps", 0.0) or 0.0)
        except Exception:
            spr = 0.0
        if spr > 0.0:
            arr = list(bucket.get("spread_bps", []) or [])
            arr.append(spr)
            bucket["spread_bps"] = arr
        try:
            bucket["pnl_usd"] = float(bucket.get("pnl_usd", 0.0) or 0.0) + float(row.get("pnl_usd", 0.0) or 0.0)
        except Exception:
            pass
        by_hour[hour] = bucket

    rows_out: List[Dict[str, Any]] = []
    for hour in sorted(by_hour):
        row = by_hour[hour]
        spreads = [float(v) for v in list(row.get("spread_bps", []) or []) if float(v) > 0.0]
        samples = int(row.get("samples", 0) or 0)
        ok = int(row.get("ok", 0) or 0)
        ok_rate = (100.0 * ok / max(1, samples))
        rows_out.append(
            {
                "hour": int(hour),
                "samples": int(samples),
                "ok_rate_pct": round(ok_rate, 3),
                "spread_bps_avg": round((sum(spreads) / max(1, len(spreads))), 4) if spreads else 0.0,
                "pnl_usd": round(float(row.get("pnl_usd", 0.0) or 0.0), 4),
            }
        )

    best_hour = None
    worst_hour = None
    if rows_out:
        ranked = sorted(rows_out, key=lambda r: (float(r.get("ok_rate_pct", 0.0)), float(r.get("pnl_usd", 0.0))), reverse=True)
        best_hour = ranked[0]
        worst_hour = ranked[-1]
    return {"hours": rows_out, "best_hour": best_hour or {}, "worst_hour": worst_hour or {}}


def _strategy_attribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_event: Dict[str, int] = {}
    pnl_by_event: Dict[str, float] = {}
    for row in rows:
        evt = str(row.get("event", "") or "").strip().lower()
        if not evt:
            continue
        by_event[evt] = int(by_event.get(evt, 0) or 0) + 1
        try:
            pnl_by_event[evt] = float(pnl_by_event.get(evt, 0.0) or 0.0) + float(row.get("pnl_usd", 0.0) or 0.0)
        except Exception:
            pass
    top_events = sorted(by_event.items(), key=lambda it: it[1], reverse=True)[:6]
    return {
        "events_total": int(sum(by_event.values())),
        "top_events": [{"event": k, "count": int(v), "pnl_usd": round(float(pnl_by_event.get(k, 0.0) or 0.0), 4)} for k, v in top_events],
    }


def _discrepancy_tracker(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    shadow = 0
    entries = 0
    ok_entries = 0
    for row in rows:
        evt = str(row.get("event", "") or "").strip().lower()
        if evt == "shadow_live_divergence":
            shadow += 1
        if evt in {"entry", "entry_fail", "shadow_entry"}:
            entries += 1
            if bool(row.get("ok", False)) and evt == "entry":
                ok_entries += 1
    ratio = (100.0 * shadow / max(1, shadow + ok_entries)) if (shadow + ok_entries) > 0 else 0.0
    level = "high" if ratio >= 75.0 else ("medium" if ratio >= 40.0 else "low")
    return {
        "shadow_divergence_24h": int(shadow),
        "entry_attempts_24h": int(entries),
        "entry_ok_24h": int(ok_entries),
        "divergence_pressure_pct": round(ratio, 3),
        "level": level,
    }


def _why_not_traded(thinker_status: Dict[str, Any], trader_status: Dict[str, Any], audit_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    t_status = thinker_status if isinstance(thinker_status, dict) else {}
    tr_status = trader_status if isinstance(trader_status, dict) else {}
    top_pick = t_status.get("top_pick", {}) if isinstance(t_status.get("top_pick", {}), dict) else {}
    ident = str(top_pick.get("symbol") or top_pick.get("pair") or "").strip().upper()
    top_reason = str(tr_status.get("entry_eval_top_reason", "") or "").strip()
    if top_reason:
        return {"candidate": ident, "reason": top_reason, "source": "trader_entry_gate"}
    for row in reversed(list(audit_rows[-40:])):
        if not isinstance(row, dict):
            continue
        evt = str(row.get("event", "") or "").strip().lower()
        if evt != "shadow_live_divergence":
            continue
        rid = str(row.get("symbol") or row.get("instrument") or "").strip().upper()
        if ident and rid and ident != rid:
            continue
        msg = str(row.get("msg", "") or "").strip()
        if msg:
            return {"candidate": (ident or rid), "reason": msg, "source": "shadow_divergence"}
    return {"candidate": ident, "reason": "", "source": "none"}


def _quality_aggregate(scan_diag: Dict[str, Any], quality_report: Dict[str, Any]) -> Dict[str, Any]:
    diag = scan_diag if isinstance(scan_diag, dict) else {}
    report = quality_report if isinstance(quality_report, dict) else {}
    reject_summary = diag.get("reject_summary", {}) if isinstance(diag.get("reject_summary", {}), dict) else {}
    report_reasons = report.get("rejection_reasons", []) if isinstance(report.get("rejection_reasons", []), list) else []
    dominant_reason = ""
    if report_reasons and isinstance(report_reasons[0], dict):
        dominant_reason = str(report_reasons[0].get("reason", "") or "").strip().lower()
    if not dominant_reason:
        dominant_reason = str(reject_summary.get("dominant_reason", "") or "").strip().lower()
    reject_rate_raw = float(
        report.get(
            "reject_rate_raw_pct",
            report.get("reject_rate_pct", reject_summary.get("reject_rate_pct", 0.0)),
        )
        or 0.0
    )
    candidate_churn = float(report.get("candidate_churn_pct", diag.get("candidate_churn_pct", 0.0)) or 0.0)
    leader_churn = float(report.get("leader_churn_pct", diag.get("leader_churn_pct", 0.0)) or 0.0)
    gate_pass_pct = float(report.get("gate_pass_pct", 0.0) or 0.0)
    leaders_total = int(report.get("leaders_total", diag.get("leaders_total", 0)) or 0)
    scores_total = int(report.get("scores_total", diag.get("scores_total", 0)) or 0)
    dominant_ratio_pct = float(reject_summary.get("dominant_ratio_pct", 0.0) or 0.0)
    market_open = bool(diag.get("market_open", True))
    market_closed_mode = (not market_open) and (str(reject_summary.get("dominant_reason", "") or "").strip().lower() == "market_closed")
    if market_closed_mode:
        reject_rate_raw = float(reject_summary.get("reject_rate_pct", 0.0) or 0.0)
        reject_rate = 0.0
        dominant_reason = "market_closed"
        dominant_ratio_pct = 0.0
        candidate_churn = float(diag.get("candidate_churn_pct", 0.0) or 0.0)
        leader_churn = float(diag.get("leader_churn_pct", 0.0) or 0.0)
        gate_pass_pct = 100.0 if (leaders_total <= 0 and scores_total <= 0 and reject_rate_raw <= 0.0) else 0.0
        return {
            "reject_rate_pct": round(max(0.0, min(100.0, reject_rate)), 3),
            "reject_rate_raw_pct": round(max(0.0, min(100.0, reject_rate_raw)), 3),
            "candidate_churn_pct": round(max(0.0, min(100.0, candidate_churn)), 3),
            "leader_churn_pct": round(max(0.0, min(100.0, leader_churn)), 3),
            "gate_pass_pct": round(max(0.0, min(100.0, gate_pass_pct)), 3),
            "dominant_reason": dominant_reason,
            "reject_dominant_ratio_pct": round(max(0.0, min(100.0, dominant_ratio_pct)), 3),
            "leaders_total": max(0, leaders_total),
            "scores_total": max(0, scores_total),
        }
    reject_rate = effective_reject_pressure(
        reject_rate_raw,
        dominant_reason=dominant_reason,
        dominant_ratio_pct=dominant_ratio_pct,
        leaders_total=leaders_total,
        scores_total=scores_total,
    )
    return {
        "reject_rate_pct": round(max(0.0, min(100.0, reject_rate)), 3),
        "reject_rate_raw_pct": round(max(0.0, min(100.0, reject_rate_raw)), 3),
        "candidate_churn_pct": round(max(0.0, min(100.0, candidate_churn)), 3),
        "leader_churn_pct": round(max(0.0, min(100.0, leader_churn)), 3),
        "gate_pass_pct": round(max(0.0, min(100.0, gate_pass_pct)), 3),
        "dominant_reason": dominant_reason,
        "reject_dominant_ratio_pct": round(max(0.0, min(100.0, dominant_ratio_pct)), 3),
        "leaders_total": max(0, leaders_total),
        "scores_total": max(0, scores_total),
    }


def _cadence_aggregate(hub_dir: str, market: str) -> Dict[str, Any]:
    cadence = _safe_read_json(os.path.join(hub_dir, "scanner_cadence_drift.json"))
    markets = cadence.get("markets", {}) if isinstance(cadence.get("markets", {}), dict) else {}
    row = markets.get(market, {}) if isinstance(markets.get(market, {}), dict) else {}
    active = cadence.get("active", []) if isinstance(cadence.get("active", []), list) else []
    active_rows = [
        a
        for a in active
        if isinstance(a, dict) and str(a.get("market", "") or "").strip().lower() == str(market or "").strip().lower()
    ]
    level = str(row.get("level", "ok") or "ok").strip().lower()
    late_pct = float(row.get("late_pct", 0.0) or 0.0)
    observed_s = float(row.get("observed_s", 0.0) or 0.0)
    expected_s = float(row.get("expected_s", 0.0) or 0.0)
    return {
        "level": level,
        "late_pct": round(max(0.0, late_pct), 3),
        "observed_s": round(max(0.0, observed_s), 3),
        "expected_s": round(max(0.0, expected_s), 3),
        "active": bool(active_rows or level in {"warning", "critical"}),
        "active_alerts_total": int(len(active_rows)),
    }


def build_market_trend_summary(hub_dir: str, market: str) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    if m not in {"stocks", "forex"}:
        return {"market": m, "state": "ERROR", "msg": "unsupported market"}

    mdir = os.path.join(hub_dir, m)
    audit_rows = _safe_read_jsonl(os.path.join(mdir, "execution_audit.jsonl"), max_lines=3000)
    rank_rows = _safe_read_jsonl(os.path.join(mdir, "scanner_rankings.jsonl"), max_lines=1200)
    trader_status = _safe_read_json(os.path.join(mdir, f"{m[:-1] if m.endswith('s') else m}_trader_status.json"))
    thinker_status = _safe_read_json(os.path.join(mdir, f"{m[:-1] if m.endswith('s') else m}_thinker_status.json"))
    scan_diag = normalize_scan_diagnostics(_safe_read_json(os.path.join(mdir, "scan_diagnostics.json")), market=m)
    quality_report = _safe_read_json(os.path.join(mdir, "universe_quality.json"))

    recent_audit = _recent_rows(audit_rows, horizon_s=86400)
    stale_secs = [parse_stale_signal_seconds(row.get("msg", "")) for row in recent_audit]
    stale_secs = [int(x) for x in stale_secs if int(x) > 0]

    audit_spreads: List[float] = []
    for row in recent_audit:
        try:
            spr = float(row.get("spread_bps", 0.0) or 0.0)
        except Exception:
            spr = 0.0
        if spr > 0.0:
            audit_spreads.append(spr)
    ranking_spreads = _spread_series_from_rankings(rank_rows[-200:])
    spread_series = list(audit_spreads) + list(ranking_spreads)

    scores: List[float] = []
    for row in rank_rows[-200:]:
        top = row.get("top", [])
        if not isinstance(top, list):
            continue
        for cand in top[:5]:
            if not isinstance(cand, dict):
                continue
            try:
                s = abs(float(cand.get("score", 0.0) or 0.0))
            except Exception:
                s = 0.0
            if s > 0.0:
                scores.append(s)

    event_counts_24h = _audit_event_counts(recent_audit)
    all_event_counts = _audit_event_counts(audit_rows)
    divergence_24h = int(event_counts_24h.get("shadow_live_divergence", 0))
    quality_agg = _quality_aggregate(scan_diag, quality_report)
    cadence_agg = _cadence_aggregate(hub_dir, m)
    reliability = _data_source_reliability(scan_diag, quality_report, thinker_status)
    fill_quality = _fill_quality_by_hour(recent_audit)
    discrepancy = _discrepancy_tracker(recent_audit)
    attribution = _strategy_attribution(recent_audit)
    why_not = _why_not_traded(thinker_status, trader_status, recent_audit)
    chart_map = thinker_status.get("top_chart_map", {}) if isinstance(thinker_status.get("top_chart_map", {}), dict) else {}
    chart_coverage = {
        "symbols_cached": int(len([k for k, v in chart_map.items() if str(k).strip() and isinstance(v, list)])),
        "bars_total": int(sum(len(list(v or [])) for v in chart_map.values() if isinstance(v, list))),
        "fallback_cached": bool(thinker_status.get("fallback_cached", False)),
    }

    return {
        "market": m,
        "ts": int(time.time()),
        "event_counts_24h": event_counts_24h,
        "event_counts_total": all_event_counts,
        "divergence_24h": int(divergence_24h),
        "stale_signal": {
            "count_24h": int(len(stale_secs)),
            "avg_s": round((sum(stale_secs) / max(1, len(stale_secs))), 3) if stale_secs else 0.0,
            "p95_s": round(_percentile([float(x) for x in stale_secs], 0.95), 3),
            "max_s": int(max(stale_secs) if stale_secs else 0),
        },
        "spread_bps": {
            "samples": int(len(spread_series)),
            "avg": round((sum(spread_series) / max(1, len(spread_series))), 4) if spread_series else 0.0,
            "p95": round(_percentile(spread_series, 0.95), 4),
        },
        "signal_score_abs": {
            "samples": int(len(scores)),
            "p50": round(_percentile(scores, 0.50), 6),
            "p95": round(_percentile(scores, 0.95), 6),
        },
        "quality_aggregates": quality_agg,
        "cadence_aggregates": cadence_agg,
        "data_source_reliability": reliability,
        "fill_quality_by_hour": fill_quality,
        "discrepancy_tracker": discrepancy,
        "strategy_attribution": attribution,
        "why_not_traded": why_not,
        "chart_coverage": chart_coverage,
        "trader_state": str(trader_status.get("state", "") or ""),
        "trader_msg": str(trader_status.get("msg", "") or "")[:180],
    }


def build_trends_payload(hub_dir: str) -> Dict[str, Any]:
    stocks = build_market_trend_summary(hub_dir, "stocks")
    forex = build_market_trend_summary(hub_dir, "forex")
    return {
        "ts": int(time.time()),
        "stocks": stocks,
        "forex": forex,
    }
