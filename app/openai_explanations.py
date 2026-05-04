from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_ALLOWED_KIND = {
    "trade_block",
    "trade_allow",
    "position_review",
    "daily_summary",
    "portfolio_decision",
}
_MARKETS = ("crypto", "stocks", "forex")

OPENAI_EXPLANATIONS_SYSTEM_PROMPT = """
You are the explanation generator for an automated multi-market trading app.

Your job is to turn structured local app facts into concise, non-technical explanations for non-expert users.

You do NOT place trades.
You do NOT change execution logic.
You do NOT invent missing facts.
You must stay grounded in the provided fields.

Output principles:
1. Keep short_text concise and operational.
2. Use plain language for non-expert users.
3. If data is weak or incomplete, acknowledge uncertainty in plain terms.
4. Keep reason_bullets factual and bounded to input evidence.
5. Prefer clarity over jargon.

Return only JSON matching the schema.
""".strip()


OPENAI_EXPLANATIONS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "target", "short_text", "reason_bullets"],
                "properties": {
                    "kind": {"type": "string", "enum": sorted(_ALLOWED_KIND)},
                    "target": {"type": "string"},
                    "short_text": {"type": "string"},
                    "reason_bullets": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
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


def _safe_read_jsonl_tail(path: str, limit: int = 300) -> List[Dict[str, Any]]:
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


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _s(market).strip().lower()
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_state_path(hub_dir: str, market: str) -> str:
    mk = _s(market).strip().lower()
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_state.json")
    if mk == "forex":
        return os.path.join(hub_dir, "forex", "forex_trader_state.json")
    return ""


def _collect_prompt_items(*, hub_dir: str, runtime_state: Dict[str, Any], max_items: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    alerts = runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}
    reasons = alerts.get("reasons", []) if isinstance(alerts.get("reasons", []), list) else []
    hints = alerts.get("hints", []) if isinstance(alerts.get("hints", []), list) else []
    for idx, reason in enumerate(reasons[:6]):
        reason_txt = _s(reason)[:160]
        if not reason_txt:
            continue
        hint_txt = _s(hints[idx])[:160] if idx < len(hints) else ""
        out.append(
            {
                "kind": "trade_block",
                "target": f"runtime_alert:{idx}",
                "facts": {
                    "reason": reason_txt,
                    "hint": hint_txt,
                    "severity": _s(alerts.get("severity", "info")).lower(),
                },
            }
        )

    allocator = runtime_state.get("cross_market_opportunity", {}) if isinstance(runtime_state.get("cross_market_opportunity", {}), dict) else {}
    selected = allocator.get("selected_candidate", {}) if isinstance(allocator.get("selected_candidate", {}), dict) else {}
    if selected:
        out.append(
            {
                "kind": "portfolio_decision",
                "target": f"{_s(selected.get('market', 'global')).lower()}:{_s(selected.get('symbol', 'candidate')).upper()}",
                "facts": {
                    "summary": _s(allocator.get("summary", ""))[:220],
                    "decision": _s(allocator.get("decision", ""))[:32].lower(),
                    "market": _s(selected.get("market", ""))[:24].lower(),
                    "symbol": _s(selected.get("symbol", ""))[:32].upper(),
                    "opportunity_score": round(_f(selected.get("opportunity_score", 0.0), 0.0), 6),
                    "reason": _s(selected.get("reason", selected.get("explanation", "")))[:220],
                },
            }
        )
    elif _s(allocator.get("summary", "")):
        out.append(
            {
                "kind": "portfolio_decision",
                "target": "allocator:summary",
                "facts": {
                    "summary": _s(allocator.get("summary", ""))[:220],
                    "decision": _s(allocator.get("decision", ""))[:32].lower(),
                },
            }
        )

    position_review = runtime_state.get("openai_position_review", {}) if isinstance(runtime_state.get("openai_position_review", {}), dict) else {}
    for row in list(position_review.get("position_actions", []) or [])[:6]:
        if not isinstance(row, dict):
            continue
        action = _s(row.get("effective_action", row.get("action", ""))).lower()
        symbol = _s(row.get("symbol", "")).upper()
        market = _s(row.get("market", "")).lower()
        if (not symbol) or (not action):
            continue
        out.append(
            {
                "kind": "position_review",
                "target": f"{market}:{symbol}",
                "facts": {
                    "action": action,
                    "confidence": round(_f(row.get("confidence", 0.0), 0.0), 6),
                    "reason": _s(row.get("reason", ""))[:220],
                },
            }
        )

    for market in _MARKETS:
        status = _safe_read_json(_market_status_path(hub_dir, market))
        state_path = _market_state_path(hub_dir, market)
        state = _safe_read_json(state_path) if state_path else {}
        top_reason = _s(status.get("entry_eval_top_reason", ""))[:180]
        entry_failed = int(max(0.0, _f(status.get("entry_eval_failed", 0), 0.0)))
        entry_total = int(max(0.0, _f(status.get("entry_eval_total", 0), 0.0)))
        stale_count = int(max(0.0, _f(status.get("stale_exit_count", 0), 0.0)))
        if top_reason and (entry_failed > 0 or entry_total > 0):
            out.append(
                {
                    "kind": "trade_block",
                    "target": f"{market}:entry",
                    "facts": {
                        "market": market,
                        "entry_eval_failed": entry_failed,
                        "entry_eval_total": entry_total,
                        "top_reason": top_reason,
                    },
                }
            )
        if stale_count > 0:
            out.append(
                {
                    "kind": "position_review",
                    "target": f"{market}:stale_positions",
                    "facts": {
                        "market": market,
                        "stale_exit_count": stale_count,
                        "alignment_streaks": dict(state.get("stale_alignment_streaks", {}) or {})
                        if isinstance(state.get("stale_alignment_streaks", {}), dict)
                        else {},
                    },
                }
            )

    pnl = runtime_state.get("pnl_decomposition", {}) if isinstance(runtime_state.get("pnl_decomposition", {}), dict) else {}
    if pnl:
        out.append(
            {
                "kind": "daily_summary",
                "target": "portfolio:session",
                "facts": {
                    "realized_total_usd": round(_f(pnl.get("realized_total_usd", pnl.get("realized_usd", 0.0)), 0.0), 6),
                    "unrealized_total_usd": round(_f(pnl.get("unrealized_total_usd", pnl.get("unrealized_usd", 0.0)), 0.0), 6),
                    "trade_count": int(max(0.0, _f(pnl.get("trade_count", 0), 0.0))),
                    "summary": _s(pnl.get("summary", ""))[:220],
                },
            }
        )

    incidents = _safe_read_jsonl_tail(os.path.join(hub_dir, "incidents.jsonl"), limit=100)
    for row in incidents[-6:]:
        sev = _s(row.get("severity", "info")).lower()
        event = _s(row.get("event", ""))[:80]
        msg = _s(row.get("msg", ""))[:180]
        if not (event or msg):
            continue
        out.append(
            {
                "kind": "trade_block",
                "target": f"incident:{event or 'runtime'}",
                "facts": {"severity": sev, "event": event, "message": msg},
            }
        )

    compact: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in out:
        if not isinstance(row, dict):
            continue
        kind = _s(row.get("kind", "")).lower()
        target = _s(row.get("target", ""))
        if kind not in _ALLOWED_KIND:
            continue
        if not target:
            continue
        key = (kind, target)
        if key in seen:
            continue
        seen.add(key)
        compact.append(
            {
                "kind": kind,
                "target": target[:120],
                "facts": row.get("facts", {}) if isinstance(row.get("facts", {}), dict) else {},
            }
        )
        if len(compact) >= int(max_items):
            break
    return compact


def build_openai_explanations_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    max_items = max(4, min(64, int(_f(cfg.get("openai_explanations_max_items", 18), 18))))
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    prompts = _collect_prompt_items(hub_dir=hub_dir, runtime_state=runtime_state, max_items=max_items)
    exposure = runtime_state.get("exposure_map", {}) if isinstance(runtime_state.get("exposure_map", {}), dict) else {}
    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "context": {
            "alert_severity": _s((runtime_state.get("alerts", {}) if isinstance(runtime_state.get("alerts", {}), dict) else {}).get("severity", "info")).lower(),
            "account_value_usd": round(_f(exposure.get("account_value_usd", 0.0), 0.0), 6),
            "total_exposure_pct": round(_f(exposure.get("total_exposure_pct", 0.0), 0.0), 6),
            "market_max_total_exposure_pct": round(_f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0), 6),
            "policy_summary": _s((runtime_state.get("automation_policy", {}) if isinstance(runtime_state.get("automation_policy", {}), dict) else {}).get("summary", ""))[:220],
        },
        "explanation_requests": prompts[:max_items],
    }


def _normalize_explanations_payload(raw: Dict[str, Any], *, max_items: int) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Explanations response was not a JSON object"
    src_items = raw.get("items", [])
    if not isinstance(src_items, list):
        return None, "Explanations payload missing items list"
    items: List[Dict[str, Any]] = []
    for row in src_items[: max(1, int(max_items))]:
        if not isinstance(row, dict):
            continue
        kind = _s(row.get("kind", "")).lower()
        target = _s(row.get("target", ""))[:120]
        short_text = _s(row.get("short_text", ""))[:220]
        reasons = row.get("reason_bullets", [])
        if not isinstance(reasons, list):
            reasons = []
        reason_rows = [_s(x)[:140] for x in reasons[:6] if _s(x)]
        if kind not in _ALLOWED_KIND:
            continue
        if not target:
            continue
        if not short_text:
            short_text = "Explanation unavailable from AI output; using local context."
        items.append(
            {
                "kind": kind,
                "target": target,
                "short_text": short_text,
                "reason_bullets": reason_rows,
            }
        )
    return {"items": items}, ""


def _explanations_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_explanations_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_explanations_live_enabled"), True)):
        return False, "live_disabled"
    return True, "enabled"


def request_openai_explanations(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    explanation_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _explanations_enabled(cfg, mode=_s(explanation_packet.get("mode", "")))
    model = _s(cfg.get("openai_explanations_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_explanations_timeout_s", 6.0), 6.0), 1.0, 30.0)
    max_items = max(4, min(64, int(_f(cfg.get("openai_explanations_max_items", 18), 18))))
    requests_list = explanation_packet.get("explanation_requests", []) if isinstance(explanation_packet.get("explanation_requests", []), list) else []
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
            "explanations": {"items": []},
            "latency_ms": 0,
        }
    if not requests_list:
        return {
            "enabled": True,
            "active": False,
            "status": "no_candidates",
            "model": model,
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "summary": "No explanation candidates were available from local runtime facts.",
            "error": "",
            "explanations": {"items": []},
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
            "summary": "AI explanations unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "explanations": {"items": []},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_EXPLANATIONS_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "explanations", "explanations_input": explanation_packet},
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
                "name": "openai_explanations",
                "strict": True,
                "schema": OPENAI_EXPLANATIONS_SCHEMA,
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
            "summary": "AI explanations timed out; local text remains active.",
            "error": "OpenAI request timed out",
            "explanations": {"items": []},
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
            "summary": "AI explanations request failed; local text remains active.",
            "error": _s(exc)[:180],
            "explanations": {"items": []},
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
            "summary": "AI explanations unavailable from OpenAI API; local text remains active.",
            "error": f"HTTP {int(resp.status_code)}",
            "explanations": {"items": []},
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
            "summary": "AI explanations returned invalid JSON; local text remains active.",
            "error": "OpenAI response could not be parsed as JSON",
            "explanations": {"items": []},
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
            "summary": "AI explanations returned no structured output; local text remains active.",
            "error": "No output text found in OpenAI response",
            "explanations": {"items": []},
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
            "summary": "AI explanations output was malformed; local text remains active.",
            "error": "Could not decode AI explanations JSON payload",
            "explanations": {"items": []},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_explanations_payload(raw, max_items=max_items)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "summary": "AI explanations schema validation failed; local text remains active.",
            "error": _s(err)[:180],
            "explanations": {"items": []},
            "latency_ms": latency_ms,
        }
    items = normalized.get("items", []) if isinstance(normalized.get("items", []), list) else []
    summary = f"Generated {len(items)} AI explanation item(s)." if items else "No AI explanations were generated for current runtime context."
    return {
        "enabled": True,
        "active": True,
        "status": "ok",
        "model": model,
        "timeout_s": float(timeout_s),
        "max_items": int(max_items),
        "summary": summary[:220],
        "error": "",
        "explanations": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def run_openai_explanations(
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
    report_path = os.path.join(openai_dir, "explanations.json")
    status_path = os.path.join(openai_dir, "explanations_status.json")

    enabled, enabled_reason = _explanations_enabled(cfg, mode=_trading_mode(cfg))
    timeout_s = _clamp(_f(cfg.get("openai_explanations_timeout_s", 6.0), 6.0), 1.0, 30.0)
    max_items = max(4, min(64, int(_f(cfg.get("openai_explanations_max_items", 18), 18))))
    model = _s(cfg.get("openai_explanations_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI explanations generation in progress." if enabled else "AI explanations disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "max_items": int(max_items),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "items_count": 0,
        "items": [],
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_explanations_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_explanations(settings=cfg, base_dir=base_dir, explanation_packet=packet)
    explanations = result.get("explanations", {}) if isinstance(result.get("explanations", {}), dict) else {}
    items = explanations.get("items", []) if isinstance(explanations.get("items", []), list) else []
    concise_items = [dict(row) for row in items[: max_items] if isinstance(row, dict)]

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "enabled": bool(enabled),
        "status": _s(result.get("status", "")),
        "summary": _s(result.get("summary", ""))[:260],
        "items_count": int(len(concise_items)),
        "items": concise_items,
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "max_items": int(max_items),
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "advisory_only": True,
        },
        "input_packet": {
            "context": dict(packet.get("context", {}) or {}) if isinstance(packet.get("context", {}), dict) else {},
            "explanation_requests": list(packet.get("explanation_requests", []) or [])[: max_items]
            if isinstance(packet.get("explanation_requests", []), list)
            else [],
        },
    }
    atomic_write_json(report_path, report)

    status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": False,
        "active": bool(result.get("active", False)),
        "status": _s(result.get("status", enabled_reason if not enabled else "ok")).lower(),
        "summary": _s(result.get("summary", ""))[:260],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "max_items": int(max_items),
        "advisory_only": True,
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": True,
        "items_count": int(len(concise_items)),
        "items": concise_items,
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_explanations_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "explanations_status.json"))


def load_latest_openai_explanations_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "explanations.json"))


def load_current_settings_for_openai_explanations(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_explanations") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
