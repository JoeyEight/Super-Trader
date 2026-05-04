from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from app.scan_diagnostics_schema import normalize_scan_diagnostics
from app.scanner_quality import build_universe_quality_report


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(text or ""))
    os.replace(tmp, path)


def _default_quality_report(market: str, scan_diag: Dict[str, Any], now_ts: int) -> Dict[str, Any]:
    universe_total = int(scan_diag.get("universe_total", 0) or 0)
    candidates_total = int(scan_diag.get("candidates_total", 0) or 0)
    scores_total = int(scan_diag.get("scores_total", 0) or 0)
    leaders_total = int(scan_diag.get("leaders_total", 0) or 0)
    reject_summary = scan_diag.get("reject_summary", {}) if isinstance(scan_diag.get("reject_summary", {}), dict) else {}
    report = build_universe_quality_report(
        market=str(market),
        ts=int(now_ts),
        mode=str(scan_diag.get("mode", "") or ""),
        universe_total=universe_total,
        candidates_total=candidates_total,
        scores_total=scores_total,
        leaders_total=leaders_total,
        reject_summary=reject_summary,
        rejected_rows=[],
        scored_rows=[],
        candidate_churn_pct=float(scan_diag.get("candidate_churn_pct", 0.0) or 0.0),
        leader_churn_pct=float(scan_diag.get("leader_churn_pct", 0.0) or 0.0),
    )
    if not str(report.get("summary", "") or "").strip():
        report["summary"] = "Pending first scanner cycle."
    return report


def bootstrap_runtime_artifacts(hub_dir: str, force: bool = False, now_ts: int | None = None) -> Dict[str, Any]:
    ts_now = int(time.time() if now_ts is None else now_ts)
    stats: Dict[str, Any] = {
        "hub_dir": str(hub_dir),
        "ts": int(ts_now),
        "updated": 0,
        "updated_files": [],
    }
    os.makedirs(hub_dir, exist_ok=True)
    for market in ("stocks", "forex"):
        market_dir = os.path.join(hub_dir, market)
        os.makedirs(market_dir, exist_ok=True)
        scan_path = os.path.join(market_dir, "scan_diagnostics.json")
        quality_path = os.path.join(market_dir, "universe_quality.json")

        scan_raw = _safe_read_json(scan_path)
        scan_norm = normalize_scan_diagnostics(scan_raw, market=market)
        write_scan = bool(force) or (scan_raw != scan_norm) or (not scan_raw)
        if write_scan:
            _safe_write_json(scan_path, scan_norm)
            stats["updated"] = int(stats["updated"]) + 1
            stats["updated_files"].append(scan_path)

        quality_raw = _safe_read_json(quality_path)
        if force or (not quality_raw):
            quality_payload = _default_quality_report(market, scan_norm, ts_now)
            _safe_write_json(quality_path, quality_payload)
            stats["updated"] = int(stats["updated"]) + 1
            stats["updated_files"].append(quality_path)

    cadence_path = os.path.join(hub_dir, "scanner_cadence_drift.json")
    cadence_raw = _safe_read_json(cadence_path)
    cadence_payload = cadence_raw if isinstance(cadence_raw, dict) else {}
    if force or (not cadence_payload):
        cadence_payload = {"ts": int(ts_now), "active": [], "markets": {}}
        _safe_write_json(cadence_path, cadence_payload)
        stats["updated"] = int(stats["updated"]) + 1
        stats["updated_files"].append(cadence_path)

    trends_path = os.path.join(hub_dir, "market_trends.json")
    trends_raw = _safe_read_json(trends_path)
    trends_payload = dict(trends_raw) if isinstance(trends_raw, dict) else {}
    updated_trends = False
    for market in ("stocks", "forex"):
        row = trends_payload.get(market, {})
        if not isinstance(row, dict):
            row = {}
        if not isinstance(row.get("quality_aggregates", {}), dict):
            row["quality_aggregates"] = {
                "reject_rate_pct": 0.0,
                "candidate_churn_pct": 0.0,
                "leader_churn_pct": 0.0,
                "gate_pass_pct": 0.0,
                "dominant_reason": "",
                "leaders_total": 0,
                "scores_total": 0,
            }
            updated_trends = True
        if not isinstance(row.get("cadence_aggregates", {}), dict):
            row["cadence_aggregates"] = {
                "level": "ok",
                "late_pct": 0.0,
                "observed_s": 0.0,
                "expected_s": 0.0,
                "active": False,
                "active_alerts_total": 0,
            }
            updated_trends = True
        trends_payload[market] = row
    trends_payload["ts"] = int(trends_payload.get("ts", ts_now) or ts_now)
    if force or (not trends_raw) or updated_trends:
        _safe_write_json(trends_path, trends_payload)
        stats["updated"] = int(stats["updated"]) + 1
        stats["updated_files"].append(trends_path)

    for rel, payload in (
        (
            "market_regimes.json",
            {
                "ts": int(ts_now),
                "stocks": {"market": "stocks", "dominant_regime": "unknown", "symbols": [], "samples": 0},
                "forex": {"market": "forex", "dominant_regime": "unknown", "symbols": [], "samples": 0},
            },
        ),
        (
            "walkforward_report.json",
            {
                "ts": int(ts_now),
                "stocks": {"market": "stocks", "state": "READY", "stability": "insufficient", "events_considered": 0, "windows": []},
                "forex": {"market": "forex", "state": "READY", "stability": "insufficient", "events_considered": 0, "windows": []},
            },
        ),
        (
            "confidence_calibration.json",
            {
                "ts": int(ts_now),
                "crypto": {"market": "crypto", "state": "READY", "samples": 0, "curve": [], "recommendation": {}},
                "stocks": {"market": "stocks", "state": "READY", "samples": 0, "curve": [], "recommendation": {}},
                "forex": {"market": "forex", "state": "READY", "samples": 0, "curve": [], "recommendation": {}},
            },
        ),
        (
            "shadow_deployment_scorecards.json",
            {
                "ts": int(ts_now),
                "stocks": {"market": "stocks", "promotion_gate": "WARN", "readiness_score": 0.0, "blockers": []},
                "forex": {"market": "forex", "promotion_gate": "WARN", "readiness_score": 0.0, "blockers": []},
                "all_markets_pass": False,
            },
        ),
        (
            "notification_center.json",
            {"ts": int(ts_now), "total": 0, "by_market": {}, "by_severity": {"critical": 0, "warning": 0, "info": 0}, "items": []},
        ),
        (
            "rejection_replay.json",
            {
                "ts": int(ts_now),
                "crypto": {"market": "crypto", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
                "stocks": {"market": "stocks", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
                "forex": {"market": "forex", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
            },
        ),
        (
            "rejection_replay_stocks.json",
            {
                "ts": int(ts_now),
                "stocks": {"market": "stocks", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
            },
        ),
        (
            "rejection_replay_crypto.json",
            {
                "ts": int(ts_now),
                "crypto": {"market": "crypto", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
            },
        ),
        (
            "rejection_replay_forex.json",
            {
                "ts": int(ts_now),
                "forex": {"market": "forex", "state": "NO_DATA", "msg": "Run scanner to generate replay report.", "scenarios": [], "recommendation": {}},
            },
        ),
    ):
        path = os.path.join(hub_dir, rel)
        cur = _safe_read_json(path)
        if force or (not cur):
            _safe_write_json(path, payload)
            stats["updated"] = int(stats["updated"]) + 1
            stats["updated_files"].append(path)

    notes_md = os.path.join(hub_dir, "operator_notes.md")
    notes_log = os.path.join(hub_dir, "operator_notes_log.jsonl")
    if force or (not os.path.isfile(notes_md)):
        _safe_write_text(
            notes_md,
            "# Operator Notes\n\n"
            "Use this file for shift handoff, risk decisions, and incident context.\n\n",
        )
        stats["updated"] = int(stats["updated"]) + 1
        stats["updated_files"].append(notes_md)
    if force or (not os.path.isfile(notes_log)):
        _safe_write_text(notes_log, "")
        stats["updated"] = int(stats["updated"]) + 1
        stats["updated_files"].append(notes_log)

    runtime_state_path = os.path.join(hub_dir, "runtime_state.json")
    runtime_state = _safe_read_json(runtime_state_path)
    if runtime_state:
        alerts = runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}
        metrics = alerts.get("metrics", {}) if isinstance(alerts.get("metrics", {}), dict) else {}
        if "market_loop_age_s" not in metrics:
            market_loop = runtime_state.get("market_loop", {}) if isinstance(runtime_state.get("market_loop", {}), dict) else {}
            metrics["market_loop_age_s"] = int(market_loop.get("age_s", -1) or -1)
            alerts["metrics"] = metrics
            runtime_state["alerts"] = alerts
            _safe_write_json(runtime_state_path, runtime_state)
            stats["updated"] = int(stats["updated"]) + 1
            stats["updated_files"].append(runtime_state_path)

    return stats
