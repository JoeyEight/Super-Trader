from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key

_ALLOWED_DECISION = {"no_trade", "allow", "deprioritize", "block"}
_ALLOWED_MARKET = {"crypto", "stocks", "forex", "none"}
_ALLOWED_ACTION = {"wait", "prefer_crypto", "prefer_stocks", "prefer_forex", "manage_existing", "allow_top_candidate"}
_ALLOWED_CANDIDATE_ACTION = {"allow", "deprioritize", "block", "no_trade"}
_ALLOWED_POSITION_ACTION = {"hold", "reduce", "increase", "exit", "block_add", "monitor"}

OPENAI_PORTFOLIO_SYSTEM_PROMPT = """
You are the portfolio decision engine for an automated multi-market trading app.

Your job is to evaluate:
1. structured candidate trades across crypto, stocks, and forex
2. currently open positions across crypto, stocks, and forex

and return a strict JSON decision object.

You do NOT place trades.
You do NOT override hard safety, compliance, broker, or execution rules.
You do NOT invent missing data.
You must be conservative when important data is missing or stale.

Primary objective:
Maximize expected portfolio growth by selecting the best current opportunities across markets and improving management of existing positions, while respecting the constraints already computed by the host application.

Decision principles:
1. Prefer higher expected opportunity quality over raw trade count.
2. Consider cross-market tradeoffs, not just local candidate strength.
3. Penalize poor execution conditions, stale data, weak confirmation, high exposure pressure, elevated loss-streak conditions, and churn risk.
4. Prefer waiting over forcing a weak trade.
5. Treat the host app’s hard gates as authoritative.
6. When data quality is weak, reduce confidence and size.
7. Avoid recommending increases to stale or misaligned positions.
8. Consider whether managing existing positions is better than opening new ones.
9. Be explainable and concise.

You must use only the data provided.
Do not assume news, market conditions, prices, or broker state not included in the input.
If all candidates and current positions are weak, recommend no_trade / hold / monitor instead of forcing action.

Return only JSON matching the schema.
""".strip()

OPENAI_PORTFOLIO_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "best_market",
        "portfolio_action",
        "portfolio_confidence",
        "capital_constrained",
        "top_recommendation",
        "ranked_candidates",
        "position_actions",
        "global_risks",
        "explanation",
    ],
    "properties": {
        "decision": {"type": "string", "enum": sorted(_ALLOWED_DECISION)},
        "best_market": {"type": "string", "enum": sorted(_ALLOWED_MARKET)},
        "portfolio_action": {"type": "string", "enum": sorted(_ALLOWED_ACTION)},
        "portfolio_confidence": {"type": "number"},
        "capital_constrained": {"type": "boolean"},
        "top_recommendation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["market", "symbol", "action", "confidence", "size_multiplier", "opportunity_score", "reason"],
            "properties": {
                "market": {"type": "string", "enum": ["crypto", "stocks", "forex"]},
                "symbol": {"type": "string"},
                "action": {"type": "string", "enum": sorted(_ALLOWED_CANDIDATE_ACTION)},
                "confidence": {"type": "number"},
                "size_multiplier": {"type": "number"},
                "opportunity_score": {"type": "number"},
                "reason": {"type": "string"},
            },
        },
        "ranked_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "market",
                    "symbol",
                    "action",
                    "confidence",
                    "size_multiplier",
                    "opportunity_score",
                    "risk_flags",
                    "reason",
                ],
                "properties": {
                    "market": {"type": "string", "enum": ["crypto", "stocks", "forex"]},
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(_ALLOWED_CANDIDATE_ACTION)},
                    "confidence": {"type": "number"},
                    "size_multiplier": {"type": "number"},
                    "opportunity_score": {"type": "number"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
        "position_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "market",
                    "symbol",
                    "action",
                    "confidence",
                    "size_multiplier",
                    "reduce_fraction",
                    "reason",
                    "risk_flags",
                ],
                "properties": {
                    "market": {"type": "string", "enum": ["crypto", "stocks", "forex"]},
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(_ALLOWED_POSITION_ACTION)},
                    "confidence": {"type": "number"},
                    "size_multiplier": {"type": "number"},
                    "reduce_fraction": {"type": "number"},
                    "reason": {"type": "string"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "global_risks": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
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
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _s(value: Any) -> str:
    return str(value or "").strip()


def _market(value: Any) -> str:
    mk = _s(value).lower()
    if mk == "stock":
        return "stocks"
    return mk


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
    if not chunks:
        return ""
    return "\n".join(chunks).strip()


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload if isinstance(payload, dict) else {}, f, indent=2, ensure_ascii=True)
    except Exception:
        pass


def _decision_rate_paths(base_dir: str) -> Tuple[str, str]:
    hub_dir = os.path.join(base_dir, "hub_data", "openai")
    state_path = os.path.join(hub_dir, "portfolio_decision_rate_state.json")
    cache_path = os.path.join(hub_dir, "portfolio_decision_last_result.json")
    return state_path, cache_path


def _decision_cached_result(
    *,
    cfg: Dict[str, Any],
    status: str,
    summary: str,
    cache_payload: Dict[str, Any],
) -> Dict[str, Any]:
    model = _s(cfg.get("openai_model")) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_timeout_s"), 6.0), 1.0, 20.0)
    cached_decision = cache_payload.get("decision", {}) if isinstance(cache_payload.get("decision", {}), dict) else {}
    cached_summary = _s(cache_payload.get("summary", ""))
    return {
        "enabled": True,
        "active": bool(cached_decision),
        "status": status,
        "model": model,
        "timeout_s": float(timeout_s),
        "summary": (cached_summary or summary)[:220],
        "error": "",
        "decision": dict(cached_decision) if cached_decision else {},
        "latency_ms": 0,
        "cached": True,
    }


def _trim_json_object(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    if src.startswith("{") and src.endswith("}"):
        return src
    start = src.find("{")
    end = src.rfind("}")
    if start >= 0 and end > start:
        return src[start : end + 1]
    return src


def _normalize_candidate(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    if market not in {"crypto", "stocks", "forex"}:
        market = "crypto"
    action = _s(src.get("action")).lower()
    if action not in _ALLOWED_CANDIDATE_ACTION:
        action = "no_trade"
    risk_flags = src.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    return {
        "market": market,
        "symbol": _s(src.get("symbol"))[:48],
        "action": action,
        "confidence": round(_clamp(_f(src.get("confidence"), 0.0), 0.0, 1.0), 6),
        "size_multiplier": round(_clamp(_f(src.get("size_multiplier"), 1.0), 0.25, 1.25), 6),
        "opportunity_score": round(_clamp(_f(src.get("opportunity_score"), 0.0), 0.0, 100.0), 6),
        "risk_flags": [_s(x)[:72] for x in risk_flags[:8] if _s(x)],
        "reason": _s(src.get("reason"))[:220],
    }


def _normalize_position_action(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    if market not in {"crypto", "stocks", "forex"}:
        market = "crypto"
    action = _s(src.get("action")).lower()
    if action not in _ALLOWED_POSITION_ACTION:
        action = "monitor"
    risk_flags = src.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    return {
        "market": market,
        "symbol": _s(src.get("symbol"))[:48],
        "action": action,
        "confidence": round(_clamp(_f(src.get("confidence"), 0.0), 0.0, 1.0), 6),
        "size_multiplier": round(_clamp(_f(src.get("size_multiplier"), 1.0), 0.25, 1.25), 6),
        "reduce_fraction": round(_clamp(_f(src.get("reduce_fraction"), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason"))[:220],
        "risk_flags": [_s(x)[:72] for x in risk_flags[:8] if _s(x)],
    }


def _normalize_decision_payload(raw: Dict[str, Any], *, max_ranked: int) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "AI response was not a JSON object"

    decision = _s(raw.get("decision")).lower()
    if decision not in _ALLOWED_DECISION:
        return None, "AI response decision enum was invalid"
    best_market = _market(raw.get("best_market"))
    if best_market not in _ALLOWED_MARKET:
        best_market = "none"
    portfolio_action = _s(raw.get("portfolio_action")).lower()
    if portfolio_action not in _ALLOWED_ACTION:
        portfolio_action = "wait"

    ranked_src = raw.get("ranked_candidates", [])
    if not isinstance(ranked_src, list):
        ranked_src = []
    ranked = [_normalize_candidate(row) for row in ranked_src[: max(1, int(max_ranked))] if isinstance(row, dict)]

    top_rec = _normalize_candidate(raw.get("top_recommendation") if isinstance(raw.get("top_recommendation"), dict) else {})
    if not _s(top_rec.get("symbol")):
        if ranked:
            top_rec = dict(ranked[0])
        else:
            top_rec = {
                "market": "crypto",
                "symbol": "",
                "action": "no_trade",
                "confidence": 0.0,
                "size_multiplier": 1.0,
                "opportunity_score": 0.0,
                "risk_flags": [],
                "reason": "No high-confidence candidate identified.",
            }

    global_risks = raw.get("global_risks", [])
    if not isinstance(global_risks, list):
        global_risks = []
    position_actions_src = raw.get("position_actions", [])
    if not isinstance(position_actions_src, list):
        position_actions_src = []
    position_actions = [
        _normalize_position_action(row)
        for row in position_actions_src[: max(8, max_ranked * 4)]
        if isinstance(row, dict)
    ]

    payload = {
        "decision": decision,
        "best_market": best_market,
        "portfolio_action": portfolio_action,
        "portfolio_confidence": round(_clamp(_f(raw.get("portfolio_confidence"), 0.0), 0.0, 1.0), 6),
        "capital_constrained": _b(raw.get("capital_constrained"), False),
        "top_recommendation": {
            "market": _market(top_rec.get("market")),
            "symbol": _s(top_rec.get("symbol"))[:48],
            "action": _s(top_rec.get("action")).lower() if _s(top_rec.get("action")).lower() in _ALLOWED_CANDIDATE_ACTION else "no_trade",
            "confidence": round(_clamp(_f(top_rec.get("confidence"), 0.0), 0.0, 1.0), 6),
            "size_multiplier": round(_clamp(_f(top_rec.get("size_multiplier"), 1.0), 0.25, 1.25), 6),
            "opportunity_score": round(_clamp(_f(top_rec.get("opportunity_score"), 0.0), 0.0, 100.0), 6),
            "reason": _s(top_rec.get("reason"))[:220],
        },
        "ranked_candidates": ranked,
        "position_actions": position_actions,
        "global_risks": [_s(x)[:96] for x in global_risks[:10] if _s(x)],
        "explanation": _s(raw.get("explanation"))[:240],
    }
    return payload, ""


def _decision_enabled(settings: Dict[str, Any] | None, *, broker_mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_decision_enabled"), False):
        return False, "disabled"
    mode = _s(broker_mode).lower()
    if mode not in {"live", "paper"}:
        mode = "paper" if (_b(cfg.get("alpaca_paper_mode"), False) or _b(cfg.get("oanda_practice_mode"), False)) else "live"
    if mode == "live" and not _b(cfg.get("openai_decision_live_enabled"), True):
        return False, "live_disabled"
    if mode == "paper" and not _b(cfg.get("openai_decision_paper_enabled"), True):
        return False, "paper_disabled"
    return True, "enabled"


def request_openai_portfolio_decision(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    decision_packet: Dict[str, Any],
    broker_mode: str = "",
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    enabled, enabled_reason = _decision_enabled(cfg, broker_mode=broker_mode)
    model = _s(cfg.get("openai_model")) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_timeout_s"), 6.0), 1.0, 20.0)
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "",
            "error": "",
            "decision": {},
            "latency_ms": 0,
        }

    min_interval_s = _clamp(_f(cfg.get("openai_decision_min_interval_s"), 0.0), 0.0, 604800.0)
    max_calls_day = max(1, int(_f(cfg.get("openai_decision_max_calls_per_day"), 24)))
    max_calls_week = max(1, int(_f(cfg.get("openai_decision_max_calls_per_week"), 120)))
    now_i = int(time.time())
    state_path, cache_path = _decision_rate_paths(base_dir)
    rate_state = _safe_read_json(state_path)
    cache_payload = _safe_read_json(cache_path)

    day_start_ts = int(_f(rate_state.get("day_start_ts", 0), 0.0))
    week_start_ts = int(_f(rate_state.get("week_start_ts", 0), 0.0))
    calls_day = int(_f(rate_state.get("calls_day", 0), 0.0))
    calls_week = int(_f(rate_state.get("calls_week", 0), 0.0))
    last_call_ts = int(_f(rate_state.get("last_call_ts", 0), 0.0))

    if day_start_ts <= 0 or (now_i - day_start_ts) >= 86400:
        day_start_ts = int(now_i)
        calls_day = 0
    if week_start_ts <= 0 or (now_i - week_start_ts) >= 604800:
        week_start_ts = int(now_i)
        calls_week = 0

    if min_interval_s > 0.0 and last_call_ts > 0 and (now_i - last_call_ts) < int(min_interval_s):
        wait_s = max(1, int(min_interval_s) - max(0, now_i - last_call_ts))
        return _decision_cached_result(
            cfg=cfg,
            status="throttled_min_interval",
            summary=f"AI portfolio decision throttled for cost control ({wait_s}s until next request window). Local allocator remains active.",
            cache_payload=cache_payload,
        )

    if calls_day >= max_calls_day:
        return _decision_cached_result(
            cfg=cfg,
            status="rate_limited_daily_budget",
            summary="AI portfolio decision daily budget reached; local allocator remains active.",
            cache_payload=cache_payload,
        )
    if calls_week >= max_calls_week:
        return _decision_cached_result(
            cfg=cfg,
            status="rate_limited_weekly_budget",
            summary="AI portfolio decision weekly budget reached; local allocator remains active.",
            cache_payload=cache_payload,
        )

    api_key = get_openai_api_key(cfg, base_dir=base_dir)
    if not api_key:
        return {
            "enabled": True,
            "active": False,
            "status": "missing_api_key",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision unavailable (missing OpenAI API key); local allocator is active.",
            "error": "OpenAI API key was not configured",
            "decision": {},
            "latency_ms": 0,
        }

    max_per_market = max(1, int(_f(cfg.get("openai_decision_max_candidates_per_market"), 3)))
    max_ranked = max(3, (max_per_market * 3) + 3)
    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": OPENAI_PORTFOLIO_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "portfolio_decision", "decision_input": decision_packet},
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
                "name": "portfolio_decision",
                "strict": True,
                "schema": OPENAI_PORTFOLIO_DECISION_SCHEMA,
            }
        },
    }

    # Persist budget counters before launching network request so all attempted
    # OpenAI calls are counted toward spend protection.
    calls_day += 1
    calls_week += 1
    last_call_ts = int(time.time())
    _safe_write_json(
        state_path,
        {
            "ts": int(last_call_ts),
            "day_start_ts": int(day_start_ts),
            "week_start_ts": int(week_start_ts),
            "calls_day": int(calls_day),
            "calls_week": int(calls_week),
            "last_call_ts": int(last_call_ts),
            "last_status": "",
            "last_error": "",
            "min_interval_s": float(min_interval_s),
            "max_calls_per_day": int(max_calls_day),
            "max_calls_per_week": int(max_calls_week),
        },
    )

    started = time.time()
    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
            timeout=(2.0, float(timeout_s)),
        )
    except requests.Timeout:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "timeout",
                "last_error": "OpenAI request timed out",
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "timeout",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision timed out; local allocator is active.",
            "error": "OpenAI request timed out",
            "decision": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "request_error",
                "last_error": _s(exc)[:160],
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision request failed; local allocator is active.",
            "error": _s(exc)[:160],
            "decision": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }

    latency_ms = int(round((time.time() - started) * 1000.0))
    if int(resp.status_code) >= 400:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "http_error",
                "last_error": f"HTTP {int(resp.status_code)}",
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "http_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision unavailable from OpenAI API; local allocator is active.",
            "error": f"HTTP {int(resp.status_code)}",
            "decision": {},
            "latency_ms": latency_ms,
        }

    try:
        response_json = resp.json()
    except Exception:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "invalid_json",
                "last_error": "OpenAI response could not be parsed as JSON",
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "invalid_json",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision returned invalid JSON; local allocator is active.",
            "error": "OpenAI response could not be parsed as JSON",
            "decision": {},
            "latency_ms": latency_ms,
        }

    text = _trim_json_object(_extract_json_text(response_json))
    if not text:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "empty_response",
                "last_error": "No output text found in OpenAI response",
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "empty_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision returned no structured output; local allocator is active.",
            "error": "No output text found in OpenAI response",
            "decision": {},
            "latency_ms": latency_ms,
        }

    try:
        raw_decision = json.loads(text)
    except Exception:
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "malformed_response",
                "last_error": "Could not decode AI decision JSON payload",
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "malformed_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision output did not match JSON schema; local allocator is active.",
            "error": "Could not decode AI decision JSON payload",
            "decision": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_decision_payload(raw_decision, max_ranked=max_ranked)
    if not isinstance(normalized, dict):
        _safe_write_json(
            state_path,
            {
                "ts": int(time.time()),
                "day_start_ts": int(day_start_ts),
                "week_start_ts": int(week_start_ts),
                "calls_day": int(calls_day),
                "calls_week": int(calls_week),
                "last_call_ts": int(last_call_ts),
                "last_status": "schema_validation_failed",
                "last_error": _s(err)[:180],
                "min_interval_s": float(min_interval_s),
                "max_calls_per_day": int(max_calls_day),
                "max_calls_per_week": int(max_calls_week),
            },
        )
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI portfolio decision schema validation failed; local allocator is active.",
            "error": _s(err)[:180],
            "decision": {},
            "latency_ms": latency_ms,
        }

    _safe_write_json(
        cache_path,
        {
            "ts": int(time.time()),
            "status": "ok",
            "summary": _s(normalized.get("explanation"))[:220],
            "decision": dict(normalized),
        },
    )
    _safe_write_json(
        state_path,
        {
            "ts": int(time.time()),
            "day_start_ts": int(day_start_ts),
            "week_start_ts": int(week_start_ts),
            "calls_day": int(calls_day),
            "calls_week": int(calls_week),
            "last_call_ts": int(last_call_ts),
            "last_status": "ok",
            "last_error": "",
            "min_interval_s": float(min_interval_s),
            "max_calls_per_day": int(max_calls_day),
            "max_calls_per_week": int(max_calls_week),
        },
    )

    return {
        "enabled": True,
        "active": True,
        "status": "ok",
        "model": model,
        "timeout_s": float(timeout_s),
        "summary": _s(normalized.get("explanation"))[:220],
        "error": "",
        "decision": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }
