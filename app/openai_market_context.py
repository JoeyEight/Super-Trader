from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_MARKETS = ("crypto", "stocks", "forex")
_ALLOWED_CONTEXT_STATE = {"supportive", "neutral", "adverse", "unclear"}


OPENAI_MARKET_CONTEXT_SYSTEM_PROMPT = """
You are the market context summarization and context-scoring advisor for an automated multi-market trading app.

Your job is to evaluate structured local event/news/context text and return strict JSON context scores.

You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must stay conservative when evidence is weak, stale, or incomplete.

Primary objective:
Provide concise, explainable context scoring that can be used as one input into local candidate ranking and confidence decisions.

Rules:
1. Use only the data supplied in the input payload.
2. Prefer neutral/unclear when evidence quality is weak.
3. Keep reasons concise and operational.
4. Do not recommend direct execution actions.
5. Return JSON only and follow the schema strictly.
""".strip()


OPENAI_MARKET_CONTEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "market_context_scores", "symbol_context_scores"],
    "properties": {
        "summary": {"type": "string"},
        "market_context_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "context_state", "confidence", "reason"],
                "properties": {
                    "market": {"type": "string", "enum": list(_MARKETS)},
                    "context_state": {"type": "string", "enum": sorted(_ALLOWED_CONTEXT_STATE)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "symbol_context_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "symbol", "context_state", "confidence", "reason"],
                "properties": {
                    "market": {"type": "string", "enum": list(_MARKETS)},
                    "symbol": {"type": "string"},
                    "context_state": {"type": "string", "enum": sorted(_ALLOWED_CONTEXT_STATE)},
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


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_thinker_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto_dynamic_status.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _candidate_notes_for_market(*, market: str, thinker: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    mk = _market(market)
    lim = max(1, int(limit or 1))
    out: List[Dict[str, Any]] = []
    if mk == "crypto":
        ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
        for idx, row in enumerate(ranked[:lim]):
            if not isinstance(row, dict):
                continue
            symbol = _s(row.get("symbol")).upper()
            if not symbol:
                continue
            out.append(
                {
                    "market": mk,
                    "symbol": symbol,
                    "side": ("long" if _f(row.get("score", 0.0), 0.0) > 0.0 else "watch"),
                    "local_rank": int(idx + 1),
                    "signal_score": round(_f(row.get("score", 0.0), 0.0), 6),
                    "reason": _s(row.get("reason_logic", row.get("reason", "")))[:180],
                }
            )
        return out

    rows = thinker.get("leaders", []) if isinstance(thinker.get("leaders", []), list) else []
    if not rows:
        top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        if top:
            rows = [top]
    symbol_key = "pair" if mk == "forex" else "symbol"
    for idx, row in enumerate(rows[:lim]):
        if not isinstance(row, dict):
            continue
        symbol = _s(row.get(symbol_key)).upper()
        if not symbol:
            continue
        out.append(
            {
                "market": mk,
                "symbol": symbol,
                "side": _s(row.get("side", "watch")).lower() or "watch",
                "local_rank": int(idx + 1),
                "signal_score": round(_f(row.get("score", 0.0), 0.0), 6),
                "reason": _s(row.get("reason_logic", row.get("reason", "")))[:180],
            }
        )
    return out


def build_openai_market_context_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    max_items = max(6, min(120, int(_f(cfg.get("openai_market_context_max_items", 24), 24))))
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    alerts = runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}
    trends = runtime_state.get("market_trends", {}) if isinstance(runtime_state.get("market_trends", {}), dict) else {}
    event_items: List[Dict[str, Any]] = []
    reasons = alerts.get("reasons", []) if isinstance(alerts.get("reasons", []), list) else []
    hints = alerts.get("hints", []) if isinstance(alerts.get("hints", []), list) else []
    for idx, reason in enumerate(reasons[:8]):
        txt = _s(reason)
        if not txt:
            continue
        hint = _s(hints[idx]) if idx < len(hints) else ""
        event_items.append({"kind": "runtime_alert", "text": txt[:180], "hint": hint[:180]})

    per_market: Dict[str, Dict[str, Any]] = {}
    symbol_notes: List[Dict[str, Any]] = []
    per_market_limit = max(2, min(24, int(max_items // 3)))
    for mk in _MARKETS:
        trader = _safe_read_json(_market_status_path(hub_dir, mk))
        thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
        policy = trader.get("automation_policy", {}) if isinstance(trader.get("automation_policy", {}), dict) else {}
        trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
        quality = trader.get("trade_quality", {}) if isinstance(trader.get("trade_quality", {}), dict) else {}
        candidates = _candidate_notes_for_market(market=mk, thinker=thinker, limit=per_market_limit)
        symbol_notes.extend(candidates)
        trend_row = trends.get(mk, {}) if isinstance(trends.get(mk, {}), dict) else {}
        why_not = trend_row.get("why_not_traded", {}) if isinstance(trend_row.get("why_not_traded", {}), dict) else {}
        per_market[mk] = {
            "market": mk,
            "policy_summary": _s(policy.get("summary", ""))[:220],
            "allow_new_entries": bool(policy.get("allow_new_entries", True)),
            "runtime_trust_score": round(float(max(0.0, _f(trust.get("score", 0.0), 0.0))), 6),
            "trade_quality_confidence_score": round(float(max(0.0, _f(quality.get("confidence_score", 0.0), 0.0))), 6),
            "trade_quality_decision": _s(quality.get("decision", ""))[:24].lower(),
            "entry_eval_top_reason": _s(trader.get("entry_eval_top_reason", ""))[:180],
            "why_not_traded": _s(why_not.get("reason", ""))[:180],
            "candidates": candidates[:per_market_limit],
        }

    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "alerts": {
            "severity": _s(alerts.get("severity", "info")).lower(),
            "reasons": [_s(x)[:160] for x in reasons[:8] if _s(x)],
            "hints": [_s(x)[:160] for x in hints[:8] if _s(x)],
        },
        "event_items": event_items[:max_items],
        "per_market": per_market,
        "symbol_notes": symbol_notes[:max_items],
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
                val = part.get(key)
                if isinstance(val, str) and val.strip():
                    chunks.append(val.strip())
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


def _normalize_market_context_row(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market", ""))
    state = _s(src.get("context_state", "")).lower()
    if state not in _ALLOWED_CONTEXT_STATE:
        state = "unclear"
    return {
        "market": market,
        "context_state": state,
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason", ""))[:220],
    }


def _normalize_symbol_context_row(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market", ""))
    symbol = _s(src.get("symbol", "")).upper()
    state = _s(src.get("context_state", "")).lower()
    if state not in _ALLOWED_CONTEXT_STATE:
        state = "unclear"
    return {
        "market": market,
        "symbol": symbol[:32],
        "context_state": state,
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason", ""))[:220],
    }


def _normalize_market_context_payload(raw: Dict[str, Any], *, max_items: int) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Market-context response was not a JSON object"
    market_rows_src = raw.get("market_context_scores", [])
    if not isinstance(market_rows_src, list):
        market_rows_src = []
    normalized_market = [_normalize_market_context_row(row) for row in market_rows_src[:12] if isinstance(row, dict)]
    market_by_key = {str(item.get("market", "")): item for item in normalized_market if isinstance(item, dict)}
    for mk in _MARKETS:
        if mk not in market_by_key:
            market_by_key[mk] = {
                "market": mk,
                "context_state": "unclear",
                "confidence": 0.0,
                "reason": "No explicit context score returned.",
            }

    symbol_rows_src = raw.get("symbol_context_scores", [])
    if not isinstance(symbol_rows_src, list):
        symbol_rows_src = []
    normalized_symbol: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in symbol_rows_src[: max(6, min(240, int(max_items or 24)))]:
        if not isinstance(row, dict):
            continue
        item = _normalize_symbol_context_row(row)
        symbol = _s(item.get("symbol", "")).upper()
        if not symbol:
            continue
        key = f"{_market(item.get('market', ''))}:{symbol}"
        if key in seen:
            continue
        seen.add(key)
        normalized_symbol.append(item)

    return {
        "summary": _s(raw.get("summary", ""))[:260],
        "market_context_scores": [market_by_key[mk] for mk in _MARKETS],
        "symbol_context_scores": normalized_symbol,
    }, ""


def _market_context_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_market_context_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_market_context_live_enabled"), True)):
        return False, "live_disabled"
    return True, "enabled"


def request_openai_market_context(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    context_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _market_context_enabled(cfg, mode=_s(context_packet.get("mode", "")))
    model = _s(cfg.get("openai_market_context_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_market_context_timeout_s", 6.0), 6.0), 1.0, 30.0)
    max_items = max(6, min(120, int(_f(cfg.get("openai_market_context_max_items", 24), 24))))
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "summary": "",
            "error": "",
            "context": {},
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
            "max_items": int(max_items),
            "summary": "AI market context unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "context": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_MARKET_CONTEXT_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "market_context_scoring", "market_context_input": context_packet},
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
                "name": "market_context_scoring",
                "strict": True,
                "schema": OPENAI_MARKET_CONTEXT_SCHEMA,
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
            "max_items": int(max_items),
            "summary": "AI market context request timed out; local logic remains active.",
            "error": "OpenAI request timed out",
            "context": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "summary": "AI market context request failed; local logic remains active.",
            "error": _s(exc)[:180],
            "context": {},
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
            "max_items": int(max_items),
            "summary": "AI market context unavailable from OpenAI API; local logic remains active.",
            "error": f"HTTP {int(resp.status_code)}",
            "context": {},
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
            "max_items": int(max_items),
            "summary": "AI market context returned invalid JSON; local logic remains active.",
            "error": "OpenAI response could not be parsed as JSON",
            "context": {},
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
            "max_items": int(max_items),
            "summary": "AI market context returned no structured output; local logic remains active.",
            "error": "No output text found in OpenAI response",
            "context": {},
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
            "max_items": int(max_items),
            "summary": "AI market context output was malformed; local logic remains active.",
            "error": "Could not decode AI market-context JSON payload",
            "context": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_market_context_payload(raw, max_items=max_items)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "summary": "AI market-context schema validation failed; local logic remains active.",
            "error": _s(err)[:180],
            "context": {},
            "latency_ms": latency_ms,
        }

    return {
        "enabled": True,
        "active": True,
        "status": "ok",
        "model": model,
        "timeout_s": float(timeout_s),
        "max_items": int(max_items),
        "summary": _s(normalized.get("summary", ""))[:220],
        "error": "",
        "context": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def run_openai_market_context(
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
    report_path = os.path.join(openai_dir, "market_context.json")
    status_path = os.path.join(openai_dir, "market_context_status.json")

    enabled, enabled_reason = _market_context_enabled(cfg, mode=_trading_mode(cfg))
    interval_s = _clamp(_f(cfg.get("openai_market_context_interval_s", 300.0), 300.0), 30.0, 86400.0)
    timeout_s = _clamp(_f(cfg.get("openai_market_context_timeout_s", 6.0), 6.0), 1.0, 30.0)
    max_items = max(6, min(120, int(_f(cfg.get("openai_market_context_max_items", 24), 24))))
    model = _s(cfg.get("openai_market_context_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI market context scoring in progress." if enabled else "AI market context scoring disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_items": int(max_items),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "market_context_scores": [],
        "symbol_context_scores": [],
        "by_market": {},
        "by_symbol": {},
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_market_context_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_market_context(settings=cfg, base_dir=base_dir, context_packet=packet)
    context = result.get("context", {}) if isinstance(result.get("context", {}), dict) else {}
    market_scores = context.get("market_context_scores", []) if isinstance(context.get("market_context_scores", []), list) else []
    symbol_scores = context.get("symbol_context_scores", []) if isinstance(context.get("symbol_context_scores", []), list) else []
    by_market = {
        _market(row.get("market", "")): dict(row)
        for row in market_scores[:12]
        if isinstance(row, dict) and _market(row.get("market", "")) in _MARKETS
    }
    for mk in _MARKETS:
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "context_state": "unclear",
                "confidence": 0.0,
                "reason": "No explicit context score returned.",
            }
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in symbol_scores[:max_items]:
        if not isinstance(row, dict):
            continue
        mk = _market(row.get("market", ""))
        symbol = _s(row.get("symbol", "")).upper()
        if not symbol:
            continue
        key = f"{mk}:{symbol}"
        if key in by_symbol:
            continue
        by_symbol[key] = dict(row)

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(context.get("summary", result.get("summary", "")))[:260],
        "market_context_scores": [dict(row) for row in market_scores[:12] if isinstance(row, dict)],
        "symbol_context_scores": [dict(row) for row in symbol_scores[:max_items] if isinstance(row, dict)],
        "by_market": dict(by_market),
        "by_symbol": dict(by_symbol),
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "interval_s": float(interval_s),
            "max_items": int(max_items),
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "advisory_only": True,
        },
        "input_packet": {
            "alerts": dict(packet.get("alerts", {}) or {}) if isinstance(packet.get("alerts", {}), dict) else {},
            "event_items": list(packet.get("event_items", []) or []) if isinstance(packet.get("event_items", []), list) else [],
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
        "summary": _s(context.get("summary", result.get("summary", "")))[:260],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_items": int(max_items),
        "advisory_only": True,
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": True,
        "market_context_scores": [dict(row) for row in market_scores[:12] if isinstance(row, dict)],
        "symbol_context_scores": [dict(row) for row in symbol_scores[:max_items] if isinstance(row, dict)],
        "by_market": dict(by_market),
        "by_symbol": dict(by_symbol),
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_market_context_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "market_context_status.json"))


def load_latest_openai_market_context_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "market_context.json"))


def load_current_settings_for_market_context(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_market_context") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
