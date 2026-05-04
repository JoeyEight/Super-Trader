from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_MARKETS = ("crypto", "stocks", "forex")
_ALLOWED_ASSESSMENT = {"normal", "caution", "degraded", "critical"}
_ALLOWED_THROTTLE = {"none", "monitor", "throttle", "pause"}


OPENAI_ROOT_CAUSE_SYSTEM_PROMPT = """
You are the anomaly and root-cause analysis advisor for an automated multi-market trading app.

Your job is to review structured runtime anomalies and diagnostics, then return strict JSON with likely causes and mitigations.

You do NOT place trades.
You do NOT directly disable trading.
You do NOT override local hard safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must remain conservative when data is stale or incomplete.

Primary objective:
Diagnose likely causes behind incident spikes, reject pressure, stale exits, alignment drift, trust degradation, and scan-latency changes.

Analysis principles:
1. Prioritize explainable likely causes grounded in provided evidence.
2. Distinguish normal transient behavior from persistent degradation.
3. Recommend practical mitigations and monitoring actions.
4. Suggest throttle/pause only when evidence is sufficiently strong.
5. Prefer monitor/none when uncertainty is high.

Return only JSON matching the schema.
""".strip()


OPENAI_ROOT_CAUSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "overall_assessment",
        "market_diagnoses",
        "global_risks",
        "throttle_recommendations",
    ],
    "properties": {
        "summary": {"type": "string"},
        "overall_assessment": {"type": "string", "enum": sorted(_ALLOWED_ASSESSMENT)},
        "market_diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "assessment", "likely_causes", "recommended_actions"],
                "properties": {
                    "market": {"type": "string", "enum": list(_MARKETS)},
                    "assessment": {"type": "string", "enum": sorted(_ALLOWED_ASSESSMENT)},
                    "likely_causes": {"type": "array", "items": {"type": "string"}},
                    "recommended_actions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "global_risks": {"type": "array", "items": {"type": "string"}},
        "throttle_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "recommendation", "confidence", "reason"],
                "properties": {
                    "market": {"type": "string", "enum": list(_MARKETS)},
                    "recommendation": {"type": "string", "enum": sorted(_ALLOWED_THROTTLE)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if txt in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _s(value: Any) -> str:
    return str(value or "").strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _market(value: Any) -> str:
    mk = _s(value).lower()
    if mk == "stock":
        return "stocks"
    return mk if mk in _MARKETS else "crypto"


def _trading_mode(settings: Dict[str, Any]) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    if bool(cfg.get("alpaca_paper_mode", False)) or bool(cfg.get("oanda_practice_mode", False)):
        return "paper"
    return "live"


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl_tail(path: str, limit: int = 4000) -> List[Dict[str, Any]]:
    lim = max(1, int(limit or 1))
    out: List[Dict[str, Any]] = []
    try:
        buf: deque[str] = deque(maxlen=lim)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = str(line or "").strip()
                if txt:
                    buf.append(txt)
        for txt in list(buf):
            try:
                row = json.loads(txt)
                if isinstance(row, dict):
                    out.append(row)
            except Exception:
                continue
    except Exception:
        return []
    return out


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_state_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_state.json")
    if mk == "forex":
        return os.path.join(hub_dir, "forex", "forex_trader_state.json")
    return ""


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _recent_performance_from_audit(rows: List[Dict[str, Any]], *, now_ts: int) -> Dict[str, Any]:
    cutoff = int(max(0, int(now_ts) - (7 * 86400)))
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    realized = 0.0
    stale_exits = 0
    churn_count = 0
    blocked_reason_counts: Counter[str] = Counter()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts < cutoff:
            continue
        event = _s(row.get("event")).lower()
        msg = _s(row.get("msg", row.get("tag", row.get("reason", row.get("source", ""))))).lower()
        if event == "entry":
            entries += 1
        if event == "exit":
            exits += 1
            pnl = _f(row.get("realized_pnl_usd", row.get("realized_pnl", row.get("pnl_usd", 0.0))), 0.0)
            realized += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
            hold_s = int(max(0.0, _f(row.get("hold_s", 0), 0.0)))
            if hold_s > 0 and hold_s <= 6 * 3600:
                churn_count += 1
        if ("stale" in msg) or ("policy_stale_exit" in msg) or ("misalign" in msg):
            stale_exits += 1
        if any(tok in msg for tok in ("reject", "blocked", "compliance", "pdt", "cooldown")):
            token = _s(row.get("tag", row.get("reason", row.get("source", msg))))[:80] or msg[:80]
            if token:
                blocked_reason_counts[token] += 1
    return {
        "entries_7d": int(entries),
        "exits_7d": int(exits),
        "wins_7d": int(wins),
        "losses_7d": int(losses),
        "realized_pnl_7d_usd": round(float(realized), 6),
        "stale_exit_count_7d": int(stale_exits),
        "churn_count_7d": int(churn_count),
        "top_blocked_reasons_7d": [str(k)[:80] for k, _ in blocked_reason_counts.most_common(6)],
    }


def _alignment_drift_counts(*, market: str, status: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    mk = _market(market)
    if mk == "crypto":
        positions = status.get("positions", {}) if isinstance(status.get("positions", {}), dict) else {}
        open_positions = 0
        drift = 0
        for _, row in positions.items():
            if not isinstance(row, dict):
                continue
            qty = _f(row.get("quantity", 0.0), 0.0)
            if qty <= 0.0:
                continue
            open_positions += 1
            if not bool(row.get("aligned_with_strategy", True)):
                drift += 1
        return {"open_positions": int(open_positions), "alignment_drift_count": int(drift)}

    open_meta = state.get("open_meta", {}) if isinstance(state.get("open_meta", {}), dict) else {}
    stale = state.get("stale_alignment_streaks", {}) if isinstance(state.get("stale_alignment_streaks", {}), dict) else {}
    drift = 0
    max_streak = 0
    for symbol in list(open_meta.keys()):
        sym = _s(symbol)
        if not sym:
            continue
        streak = int(max(0.0, _f(stale.get(sym, stale.get(symbol, 0)), 0.0)))
        max_streak = max(max_streak, streak)
        if streak > 0:
            drift += 1
    return {"open_positions": int(len(open_meta)), "alignment_drift_count": int(drift), "alignment_drift_max_streak": int(max_streak)}


def _incident_compact(row: Dict[str, Any]) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    return {
        "ts": int(max(0.0, _f(src.get("ts", 0), 0.0))),
        "severity": _s(src.get("severity", "info")).lower()[:16],
        "event": _s(src.get("event", ""))[:80],
        "msg": _s(src.get("msg", ""))[:180],
        "component": _s((src.get("context", {}) if isinstance(src.get("context", {}), dict) else {}).get("component", ""))[:40],
    }


def build_openai_root_cause_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    max_incidents = max(50, min(2000, int(_f(cfg.get("openai_root_cause_max_incidents", 300), 300))))
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    incidents = _safe_read_jsonl_tail(os.path.join(hub_dir, "incidents.jsonl"), limit=max_incidents * 2)
    compact_incidents = [_incident_compact(row) for row in incidents[-max_incidents:] if isinstance(row, dict)]
    per_market: Dict[str, Dict[str, Any]] = {}
    for mk in _MARKETS:
        status = _safe_read_json(_market_status_path(hub_dir, mk))
        state_path = _market_state_path(hub_dir, mk)
        state = _safe_read_json(state_path) if state_path else {}
        audit = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=3000)
        policy = status.get("automation_policy", {}) if isinstance(status.get("automation_policy", {}), dict) else {}
        trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
        gate = status.get("entry_gate_flags", {}) if isinstance(status.get("entry_gate_flags", {}), dict) else {}
        trade_quality = status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}
        alignment = _alignment_drift_counts(market=mk, status=status, state=state)
        perf = _recent_performance_from_audit(audit, now_ts=now_i)
        if int(perf.get("stale_exit_count_7d", 0) or 0) <= 0:
            perf["stale_exit_count_7d"] = int(max(0.0, _f(status.get("stale_exit_count", 0), 0.0)))
        entry_eval_counts = status.get("entry_eval_reason_counts", {}) if isinstance(status.get("entry_eval_reason_counts", {}), dict) else {}
        blocked_reasons = [
            str(k)[:80]
            for k, _ in sorted(
                [(str(k), int(max(0.0, _f(v, 0.0)))) for k, v in entry_eval_counts.items()],
                key=lambda item: int(item[1]),
                reverse=True,
            )[:6]
            if str(k).strip()
        ]
        scan_health = runtime_state.get("scan_health", {}) if isinstance(runtime_state.get("scan_health", {}), dict) else {}
        scan_row = scan_health.get(mk, {}) if isinstance(scan_health.get(mk, {}), dict) else {}
        sla = runtime_state.get("sla_metrics", {}) if isinstance(runtime_state.get("sla_metrics", {}), dict) else {}
        scan_key = f"{mk}_scan"
        sla_scan = sla.get(scan_key, {}) if isinstance(sla.get(scan_key, {}), dict) else {}
        per_market[mk] = {
            "market": mk,
            "policy_summary": _s(policy.get("summary", ""))[:220],
            "allow_new_entries": bool(policy.get("allow_new_entries", True)),
            "runtime_trust_score": round(float(max(0.0, _f(trust.get("score", 0.0), 0.0))), 6),
            "trade_quality_confidence_score": round(float(max(0.0, _f(trade_quality.get("confidence_score", 0.0), 0.0))), 6),
            "trade_quality_decision": _s(trade_quality.get("decision", ""))[:24].lower(),
            "loss_streak": int(max(0.0, _f(gate.get("loss_streak", status.get("loss_streak", 0)), 0.0))),
            "entry_eval_failed": int(max(0.0, _f(status.get("entry_eval_failed", 0), 0.0))),
            "entry_eval_total": int(max(0.0, _f(status.get("entry_eval_total", 0), 0.0))),
            "entry_eval_top_reason": _s(status.get("entry_eval_top_reason", ""))[:180],
            "blocked_reasons": blocked_reasons,
            "exposure_usd": round(float(max(0.0, _f(status.get("exposure_usd", 0.0), 0.0))), 6),
            "account_value_usd": round(float(max(0.0, _f(status.get("account_value_usd", 0.0), 0.0))), 6),
            "reject_rate_pct": round(float(max(0.0, _f(gate.get("reject_rate_pct", scan_row.get("reject_rate_pct", 0.0)), 0.0))), 6),
            "stale_exit_count": int(max(0.0, _f(status.get("stale_exit_count", perf.get("stale_exit_count_7d", 0)), 0.0))),
            "alignment_drift": alignment,
            "scan_latency_ms": {
                "last_ms": round(float(max(0.0, _f(sla_scan.get("last_ms", 0.0), 0.0))), 3),
                "p95_ms": round(float(max(0.0, _f(sla_scan.get("p95_ms", 0.0), 0.0))), 3),
            },
            "recent_performance": perf,
        }

    alerts = runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}
    trends = runtime_state.get("market_trends", {}) if isinstance(runtime_state.get("market_trends", {}), dict) else {}
    policy = runtime_state.get("automation_policy", {}) if isinstance(runtime_state.get("automation_policy", {}), dict) else {}
    scan_health = runtime_state.get("scan_health", {}) if isinstance(runtime_state.get("scan_health", {}), dict) else {}
    exposure = runtime_state.get("exposure_map", {}) if isinstance(runtime_state.get("exposure_map", {}), dict) else {}
    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "policy_context": {
            "market_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))), 6),
            "crypto_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("stock_max_total_exposure_pct", 0.0), 0.0))), 6),
            "forex_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("forex_max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("stock_max_daily_loss_pct", 0.0), 0.0))), 6),
            "forex_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("forex_max_daily_loss_pct", 0.0), 0.0))), 6),
        },
        "alerts": {
            "severity": _s(alerts.get("severity", "info")).lower(),
            "reasons": [_s(x)[:160] for x in list(alerts.get("reasons", []) or [])[:12] if _s(x)],
            "hints": [_s(x)[:160] for x in list(alerts.get("hints", []) or [])[:12] if _s(x)],
        },
        "scan_health": dict(scan_health),
        "market_trends": dict(trends),
        "automation_policy_runtime": dict(policy),
        "exposure_map": dict(exposure),
        "incident_trend": dict(runtime_state.get("incident_trend", {}) or {}) if isinstance(runtime_state.get("incident_trend", {}), dict) else {},
        "recent_incidents": compact_incidents,
        "per_market": per_market,
    }


def _extract_json_text(response_json: Dict[str, Any]) -> str:
    if not isinstance(response_json, dict):
        return ""
    out_text = response_json.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()
    chunks: List[str] = []
    output = response_json.get("output", [])
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "output_text", "value"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return "\n".join(chunks).strip()


def _trim_json_object(text: str) -> str:
    src = _s(text)
    if not src:
        return ""
    if src.startswith("{") and src.endswith("}"):
        return src
    start = src.find("{")
    end = src.rfind("}")
    if start >= 0 and end > start:
        return src[start : end + 1]
    return src


def _normalize_market_diagnosis(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    assessment = _s(src.get("assessment")).lower()
    if assessment not in _ALLOWED_ASSESSMENT:
        assessment = "caution"
    likely = src.get("likely_causes", [])
    if not isinstance(likely, list):
        likely = []
    actions = src.get("recommended_actions", [])
    if not isinstance(actions, list):
        actions = []
    return {
        "market": market,
        "assessment": assessment,
        "likely_causes": [_s(x)[:160] for x in likely[:8] if _s(x)],
        "recommended_actions": [_s(x)[:160] for x in actions[:10] if _s(x)],
    }


def _normalize_throttle_recommendation(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    rec = _s(src.get("recommendation")).lower()
    if rec not in _ALLOWED_THROTTLE:
        rec = "none"
    return {
        "market": market,
        "recommendation": rec,
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason"))[:220],
    }


def _normalize_root_cause_payload(raw: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Root-cause response was not a JSON object"
    overall = _s(raw.get("overall_assessment")).lower()
    if overall not in _ALLOWED_ASSESSMENT:
        return None, "Root-cause overall_assessment enum was invalid"
    market_src = raw.get("market_diagnoses", [])
    if not isinstance(market_src, list):
        market_src = []
    market_rows = [_normalize_market_diagnosis(row) for row in market_src[:12] if isinstance(row, dict)]
    by_market = {str(item.get("market", "")): item for item in market_rows if isinstance(item, dict)}
    for mk in _MARKETS:
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "assessment": "normal",
                "likely_causes": [],
                "recommended_actions": ["No strong diagnosis generated."],
            }
    throttle_src = raw.get("throttle_recommendations", [])
    if not isinstance(throttle_src, list):
        throttle_src = []
    throttle_rows = [_normalize_throttle_recommendation(row) for row in throttle_src[:12] if isinstance(row, dict)]
    throttle_by_market = {str(item.get("market", "")): item for item in throttle_rows if isinstance(item, dict)}
    for mk in _MARKETS:
        if mk not in throttle_by_market:
            throttle_by_market[mk] = {
                "market": mk,
                "recommendation": "none",
                "confidence": 0.0,
                "reason": "No explicit throttle recommendation.",
            }
    risks = raw.get("global_risks", [])
    if not isinstance(risks, list):
        risks = []
    return {
        "summary": _s(raw.get("summary"))[:260],
        "overall_assessment": overall,
        "market_diagnoses": [by_market[mk] for mk in _MARKETS],
        "global_risks": [_s(x)[:120] for x in risks[:12] if _s(x)],
        "throttle_recommendations": [throttle_by_market[mk] for mk in _MARKETS],
    }, ""


def _root_cause_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_root_cause_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_root_cause_live_enabled"), True)):
        return False, "live_disabled"
    return True, "enabled"


def request_openai_root_cause_analysis(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    analysis_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _root_cause_enabled(cfg, mode=_s(analysis_packet.get("mode", "")))
    model = _s(cfg.get("openai_root_cause_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_root_cause_timeout_s", 8.0), 8.0), 1.0, 30.0)
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "",
            "error": "",
            "analysis": {},
            "latency_ms": 0,
        }

    api_key = get_openai_api_key(cfg, base_dir=base_dir)
    if not api_key:
        return {
            "enabled": True,
            "active": False,
            "status": "missing_api_key",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "analysis": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_ROOT_CAUSE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "root_cause_analysis", "root_cause_input": analysis_packet},
                            separators=(",", ":"),
                            ensure_ascii=True,
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "root_cause_analysis",
                "strict": True,
                "schema": OPENAI_ROOT_CAUSE_SCHEMA,
            }
        },
    }

    started = time.time()
    try:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            timeout=(2.0, float(timeout_s)),
        )
    except requests.Timeout:
        return {
            "enabled": True,
            "active": False,
            "status": "timeout",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis timed out; local diagnostics remain active.",
            "error": "OpenAI request timed out",
            "analysis": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis request failed; local diagnostics remain active.",
            "error": _s(exc)[:180],
            "analysis": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }

    latency_ms = int(round((time.time() - started) * 1000.0))
    if int(resp.status_code) >= 400:
        return {
            "enabled": True,
            "active": False,
            "status": "http_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis unavailable from OpenAI API; local diagnostics remain active.",
            "error": f"HTTP {int(resp.status_code)}",
            "analysis": {},
            "latency_ms": latency_ms,
        }

    try:
        response_json = resp.json()
    except Exception:
        return {
            "enabled": True,
            "active": False,
            "status": "invalid_json",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis returned invalid JSON; local diagnostics remain active.",
            "error": "OpenAI response could not be parsed as JSON",
            "analysis": {},
            "latency_ms": latency_ms,
        }

    text = _trim_json_object(_extract_json_text(response_json))
    if not text:
        return {
            "enabled": True,
            "active": False,
            "status": "empty_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis returned no structured output; local diagnostics remain active.",
            "error": "No output text found in OpenAI response",
            "analysis": {},
            "latency_ms": latency_ms,
        }

    try:
        raw = json.loads(text)
    except Exception:
        return {
            "enabled": True,
            "active": False,
            "status": "malformed_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis output was malformed; local diagnostics remain active.",
            "error": "Could not decode AI root-cause JSON payload",
            "analysis": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_root_cause_payload(raw)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI root-cause analysis schema validation failed; local diagnostics remain active.",
            "error": _s(err)[:180],
            "analysis": {},
            "latency_ms": latency_ms,
        }
    return {
        "enabled": True,
        "active": True,
        "status": "ok",
        "model": model,
        "timeout_s": float(timeout_s),
        "summary": _s(normalized.get("summary", ""))[:220],
        "error": "",
        "analysis": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def run_openai_root_cause_analysis(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    hub_dir: str,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    date_local = time.strftime("%Y-%m-%d", time.localtime(now_i))
    openai_dir = os.path.join(hub_dir, "openai")
    os.makedirs(openai_dir, exist_ok=True)
    report_path = os.path.join(openai_dir, "root_cause_analysis.json")
    status_path = os.path.join(openai_dir, "root_cause_analysis_status.json")

    enabled, enabled_reason = _root_cause_enabled(cfg, mode=_trading_mode(cfg))
    interval_s = _clamp(_f(cfg.get("openai_root_cause_interval_s", 240.0), 240.0), 30.0, 86400.0)
    timeout_s = _clamp(_f(cfg.get("openai_root_cause_timeout_s", 8.0), 8.0), 1.0, 30.0)
    max_incidents = max(50, min(2000, int(_f(cfg.get("openai_root_cause_max_incidents", 300), 300))))
    model = _s(cfg.get("openai_root_cause_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI root-cause analysis in progress." if enabled else "AI root-cause analysis disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_incidents": int(max_incidents),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "overall_assessment": "",
        "market_diagnoses": [],
        "throttle_recommendations": [],
        "global_risks": [],
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_root_cause_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_root_cause_analysis(settings=cfg, base_dir=base_dir, analysis_packet=packet)
    analysis = result.get("analysis", {}) if isinstance(result.get("analysis", {}), dict) else {}
    market_rows = analysis.get("market_diagnoses", []) if isinstance(analysis.get("market_diagnoses", []), list) else []
    throttle_rows = analysis.get("throttle_recommendations", []) if isinstance(analysis.get("throttle_recommendations", []), list) else []
    risk_rows = analysis.get("global_risks", []) if isinstance(analysis.get("global_risks", []), list) else []

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(analysis.get("summary", result.get("summary", "")))[:260],
        "overall_assessment": _s(analysis.get("overall_assessment", ""))[:24].lower(),
        "market_diagnoses": [dict(row) for row in market_rows[:12] if isinstance(row, dict)],
        "throttle_recommendations": [dict(row) for row in throttle_rows[:12] if isinstance(row, dict)],
        "global_risks": [_s(x)[:120] for x in risk_rows[:12] if _s(x)],
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "interval_s": float(interval_s),
            "max_incidents": int(max_incidents),
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "advisory_only": True,
        },
        "input_packet": {
            "alerts": dict(packet.get("alerts", {}) or {}) if isinstance(packet.get("alerts", {}), dict) else {},
            "incident_trend": dict(packet.get("incident_trend", {}) or {}) if isinstance(packet.get("incident_trend", {}), dict) else {},
            "per_market": dict(packet.get("per_market", {}) or {}) if isinstance(packet.get("per_market", {}), dict) else {},
        },
    }
    atomic_write_json(report_path, report)

    status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": False,
        "active": bool(result.get("active", False)),
        "status": _s(result.get("status", enabled_reason if not enabled else "ok")).lower(),
        "summary": _s(analysis.get("summary", result.get("summary", "")))[:260],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_incidents": int(max_incidents),
        "advisory_only": True,
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": True,
        "overall_assessment": _s(analysis.get("overall_assessment", ""))[:24].lower(),
        "market_diagnoses": [dict(row) for row in market_rows[:12] if isinstance(row, dict)],
        "market_diagnoses_by_market": {
            _market(row.get("market", "")): dict(row)
            for row in market_rows[:12]
            if isinstance(row, dict)
        },
        "throttle_recommendations": [dict(row) for row in throttle_rows[:12] if isinstance(row, dict)],
        "throttle_by_market": {
            _market(row.get("market", "")): dict(row)
            for row in throttle_rows[:12]
            if isinstance(row, dict)
        },
        "global_risks": [_s(x)[:120] for x in risk_rows[:12] if _s(x)],
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_root_cause_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "root_cause_analysis_status.json"))


def load_latest_openai_root_cause_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "root_cause_analysis.json"))


def load_current_settings_for_root_cause(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_root_cause_analysis") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
