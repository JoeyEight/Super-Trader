from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import requests

from app.credential_utils import get_openai_api_key
from app.openai_trade_review import (
    apply_validated_tuning_suggestions,
    validate_low_risk_tuning_suggestions,
)
from app.path_utils import read_settings_file, resolve_settings_path
from app.runtime_logging import atomic_write_json
from app.settings_utils import sanitize_settings

_MARKETS = ("crypto", "stocks", "forex")


OPENAI_POSTMORTEM_SYSTEM_PROMPT = """
You are the offline backtest/postmortem analysis advisor for an automated multi-market trading app.

Your job is to review structured historical execution/session data and return strict JSON insights.

You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must be conservative when evidence is weak or incomplete.

Primary objective:
Identify the strongest drags/strengths and provide practical improvement recommendations for what to skip, improve in exits, and how to reallocate capital.

Rules:
1. Use only data supplied in the payload.
2. Prefer bounded, explainable recommendations.
3. If confidence is weak, recommend conservative/no-change guidance.
4. Return JSON only and follow the schema strictly.
""".strip()


OPENAI_POSTMORTEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "main_drags",
        "main_strengths",
        "skip_recommendations",
        "exit_improvement_recommendations",
        "capital_reallocation_recommendations",
        "tuning_suggestions",
    ],
    "properties": {
        "summary": {"type": "string"},
        "main_drags": {"type": "array", "items": {"type": "string"}},
        "main_strengths": {"type": "array", "items": {"type": "string"}},
        "skip_recommendations": {"type": "array", "items": {"type": "string"}},
        "exit_improvement_recommendations": {"type": "array", "items": {"type": "string"}},
        "capital_reallocation_recommendations": {"type": "array", "items": {"type": "string"}},
        "tuning_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["setting_key", "suggested_value", "confidence", "reason"],
                "properties": {
                    "setting_key": {"type": "string"},
                    "suggested_value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "integer"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
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


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _event_symbol(market: str, row: Dict[str, Any]) -> str:
    mk = _market(market)
    if mk == "forex":
        return _s(row.get("instrument", row.get("pair", row.get("symbol", "")))).upper()
    return _s(row.get("symbol", row.get("pair", row.get("instrument", "")))).upper()


def _event_reason(row: Dict[str, Any]) -> str:
    for key in ("msg", "tag", "reason", "source"):
        val = _s(row.get(key, ""))
        if val:
            return val
    return ""


def _event_setup_family(row: Dict[str, Any]) -> str:
    for key in ("setup_family", "strategy", "signal_family", "setup", "reason"):
        val = _s(row.get(key, ""))
        if val:
            raw = val.lower()
            out = "".join((ch if ch.isalnum() else "_") for ch in raw)
            while "__" in out:
                out = out.replace("__", "_")
            out = out.strip("_")
            if out:
                return out[:56]
    return "unknown"


def _realized_pnl(row: Dict[str, Any]) -> float:
    for key in ("realized_pnl_usd", "realized_pnl", "realized_pl", "realized", "pnl_usd", "pnl"):
        if key in row:
            return float(_f(row.get(key, 0.0), 0.0))
    return 0.0


def _summarize_market_audit(*, market: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    realized = 0.0
    stale_exit_count = 0
    rapid_turnover_count = 0
    exit_reasons: Counter[str] = Counter()
    setup_families: Counter[str] = Counter()
    symbol_drag: Dict[str, float] = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        event = _s(row.get("event", "")).lower()
        reason = _event_reason(row).lower()
        setup_families[_event_setup_family(row)] += 1
        if event in {"entry", "order_entry", "buy"}:
            entries += 1
            continue
        if event in {"exit", "sell", "close"}:
            exits += 1
            pnl = _realized_pnl(row)
            realized += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
                symbol = _event_symbol(market, row)
                if symbol:
                    symbol_drag[symbol] = float(symbol_drag.get(symbol, 0.0) + pnl)
            hold_s = int(max(0.0, _f(row.get("hold_s", 0), 0.0)))
            if hold_s > 0 and hold_s <= 6 * 3600:
                rapid_turnover_count += 1
            if ("stale" in reason) or ("policy_stale_exit" in reason) or ("misalign" in reason):
                stale_exit_count += 1
            exit_reasons[reason[:80] if reason else "unknown"] += 1
            continue

    top_underperformers = sorted(symbol_drag.items(), key=lambda item: float(item[1]))[:8]
    return {
        "market": _market(market),
        "entries": int(entries),
        "exits": int(exits),
        "wins": int(wins),
        "losses": int(losses),
        "realized_pnl_usd": round(float(realized), 6),
        "stale_exit_count": int(stale_exit_count),
        "rapid_turnover_count": int(rapid_turnover_count),
        "top_exit_reasons": [{"reason": str(k)[:80], "count": int(v)} for k, v in exit_reasons.most_common(8)],
        "setup_family_counts": [{"family": str(k)[:80], "count": int(v)} for k, v in setup_families.most_common(10)],
        "top_underperformers": [{"symbol": str(sym), "realized_pnl_usd": round(float(val), 6)} for sym, val in top_underperformers],
    }


def build_openai_postmortem_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    max_events = max(200, min(50000, int(_f(cfg.get("openai_postmortem_max_events", 5000), 5000))))
    per_market_limit = max(80, int(max_events // max(1, len(_MARKETS))))
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))

    per_market: Dict[str, Dict[str, Any]] = {}
    total_realized = 0.0
    total_entries = 0
    total_exits = 0
    total_wins = 0
    total_losses = 0
    stale_exit_count = 0
    rapid_turnover_count = 0
    common_exit_reasons: Counter[str] = Counter()
    setup_families: Counter[str] = Counter()
    underperformers: List[Dict[str, Any]] = []

    for mk in _MARKETS:
        rows = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=per_market_limit)
        summary = _summarize_market_audit(market=mk, rows=rows)
        per_market[mk] = summary
        total_realized += float(_f(summary.get("realized_pnl_usd", 0.0), 0.0))
        total_entries += int(max(0.0, _f(summary.get("entries", 0), 0.0)))
        total_exits += int(max(0.0, _f(summary.get("exits", 0), 0.0)))
        total_wins += int(max(0.0, _f(summary.get("wins", 0), 0.0)))
        total_losses += int(max(0.0, _f(summary.get("losses", 0), 0.0)))
        stale_exit_count += int(max(0.0, _f(summary.get("stale_exit_count", 0), 0.0)))
        rapid_turnover_count += int(max(0.0, _f(summary.get("rapid_turnover_count", 0), 0.0)))
        for row in list(summary.get("top_exit_reasons", []) or []):
            if not isinstance(row, dict):
                continue
            key = _s(row.get("reason", ""))[:80]
            if key:
                common_exit_reasons[key] += int(max(0.0, _f(row.get("count", 0), 0.0)))
        for row in list(summary.get("setup_family_counts", []) or []):
            if not isinstance(row, dict):
                continue
            key = _s(row.get("family", ""))[:80]
            if key:
                setup_families[key] += int(max(0.0, _f(row.get("count", 0), 0.0)))
        for row in list(summary.get("top_underperformers", []) or []):
            if isinstance(row, dict):
                underperformers.append(
                    {
                        "market": mk,
                        "symbol": _s(row.get("symbol", "")).upper(),
                        "realized_pnl_usd": round(float(_f(row.get("realized_pnl_usd", 0.0), 0.0)), 6),
                    }
                )

    underperformers = sorted(underperformers, key=lambda item: float(item.get("realized_pnl_usd", 0.0)))[:12]
    allocator_ctx = runtime_state.get("cross_market_opportunity", {}) if isinstance(runtime_state.get("cross_market_opportunity", {}), dict) else {}
    pos_review_ctx = runtime_state.get("openai_position_review", {}) if isinstance(runtime_state.get("openai_position_review", {}), dict) else {}
    capital_planner_ctx = runtime_state.get("openai_capital_planner", {}) if isinstance(runtime_state.get("openai_capital_planner", {}), dict) else {}
    strategy_ctx = runtime_state.get("openai_strategy_optimizer", {}) if isinstance(runtime_state.get("openai_strategy_optimizer", {}), dict) else {}

    position_summary: Dict[str, Any] = {}
    for mk in _MARKETS:
        trader = _safe_read_json(_market_status_path(hub_dir, mk))
        position_summary[mk] = {
            "open_positions": int(max(0.0, _f(trader.get("open_positions", 0), 0.0))),
            "stale_exit_count": int(max(0.0, _f(trader.get("stale_exit_count", 0), 0.0))),
            "entry_eval_top_reason": _s(trader.get("entry_eval_top_reason", ""))[:120],
            "entry_eval_failed": int(max(0.0, _f(trader.get("entry_eval_failed", 0), 0.0))),
            "entry_eval_total": int(max(0.0, _f(trader.get("entry_eval_total", 0), 0.0))),
        }

    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "max_events": int(max_events),
        "trade_history_summary": {
            "entries": int(total_entries),
            "exits": int(total_exits),
            "wins": int(total_wins),
            "losses": int(total_losses),
            "realized_pnl_usd": round(float(total_realized), 6),
        },
        "execution_audit_summary": {
            "stale_exit_count": int(stale_exit_count),
            "rapid_turnover_count": int(rapid_turnover_count),
            "common_exit_reasons": [{"reason": str(k), "count": int(v)} for k, v in common_exit_reasons.most_common(12)],
            "setup_family_summaries": [{"family": str(k), "count": int(v)} for k, v in setup_families.most_common(16)],
            "underperformers": underperformers,
        },
        "position_management_summary": {
            "by_market": position_summary,
            "position_review_summary": _s(pos_review_ctx.get("summary", ""))[:220],
            "position_review_status": _s(pos_review_ctx.get("status", "")).lower(),
            "position_review_actions_count": int(max(0.0, _f(pos_review_ctx.get("actions_count", 0), 0.0))),
        },
        "per_market_pnl_summary": {
            mk: {
                "realized_pnl_usd": round(float(_f((per_market.get(mk, {}) if isinstance(per_market.get(mk, {}), dict) else {}).get("realized_pnl_usd", 0.0), 0.0)), 6),
                "wins": int(max(0.0, _f((per_market.get(mk, {}) if isinstance(per_market.get(mk, {}), dict) else {}).get("wins", 0), 0.0))),
                "losses": int(max(0.0, _f((per_market.get(mk, {}) if isinstance(per_market.get(mk, {}), dict) else {}).get("losses", 0), 0.0))),
                "stale_exit_count": int(max(0.0, _f((per_market.get(mk, {}) if isinstance(per_market.get(mk, {}), dict) else {}).get("stale_exit_count", 0), 0.0))),
            }
            for mk in _MARKETS
        },
        "allocator_and_ai_context": {
            "allocator_summary": _s(allocator_ctx.get("summary", ""))[:220],
            "allocator_best_market": _s(allocator_ctx.get("best_market", "")).lower(),
            "capital_planner_summary": _s(capital_planner_ctx.get("summary", ""))[:220],
            "capital_planner_status": _s(capital_planner_ctx.get("status", "")).lower(),
            "strategy_optimizer_summary": _s(strategy_ctx.get("summary", ""))[:220],
            "strategy_optimizer_status": _s(strategy_ctx.get("status", "")).lower(),
        },
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


def _normalize_tuning_suggestion(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    return {
        "setting_key": _s(src.get("setting_key", ""))[:96],
        "suggested_value": src.get("suggested_value"),
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason", ""))[:220],
    }


def _normalize_postmortem_payload(raw: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Postmortem response was not a JSON object"
    tuning_src = raw.get("tuning_suggestions", [])
    if not isinstance(tuning_src, list):
        tuning_src = []
    tuning_rows = [_normalize_tuning_suggestion(row) for row in tuning_src[:20] if isinstance(row, dict)]
    return {
        "summary": _s(raw.get("summary", ""))[:260],
        "main_drags": [_s(x)[:140] for x in list(raw.get("main_drags", []) or [])[:16] if _s(x)],
        "main_strengths": [_s(x)[:140] for x in list(raw.get("main_strengths", []) or [])[:16] if _s(x)],
        "skip_recommendations": [_s(x)[:140] for x in list(raw.get("skip_recommendations", []) or [])[:16] if _s(x)],
        "exit_improvement_recommendations": [
            _s(x)[:160]
            for x in list(raw.get("exit_improvement_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "capital_reallocation_recommendations": [
            _s(x)[:160]
            for x in list(raw.get("capital_reallocation_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "tuning_suggestions": [
            row
            for row in tuning_rows
            if _s(row.get("setting_key", "")) and _s(row.get("reason", ""))
        ],
    }, ""


def _postmortem_enabled(settings: Dict[str, Any] | None) -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_postmortem_enabled"), False):
        return False, "disabled"
    return True, "enabled"


def request_openai_postmortem_analysis(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    postmortem_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _postmortem_enabled(cfg)
    model = _s(cfg.get("openai_postmortem_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_postmortem_timeout_s", 12.0), 12.0), 1.0, 60.0)
    max_events = max(200, min(50000, int(_f(cfg.get("openai_postmortem_max_events", 5000), 5000))))
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "max_events": int(max_events),
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
            "max_events": int(max_events),
            "summary": "AI postmortem unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "analysis": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_POSTMORTEM_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "postmortem_analysis", "postmortem_input": postmortem_packet},
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
                "name": "postmortem_analysis",
                "strict": True,
                "schema": OPENAI_POSTMORTEM_SCHEMA,
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
            "max_events": int(max_events),
            "summary": "AI postmortem timed out; local analytics remain active.",
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
            "max_events": int(max_events),
            "summary": "AI postmortem request failed; local analytics remain active.",
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
            "max_events": int(max_events),
            "summary": "AI postmortem unavailable from OpenAI API; local analytics remain active.",
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
            "max_events": int(max_events),
            "summary": "AI postmortem returned invalid JSON; local analytics remain active.",
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
            "max_events": int(max_events),
            "summary": "AI postmortem returned no structured output; local analytics remain active.",
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
            "max_events": int(max_events),
            "summary": "AI postmortem output was malformed; local analytics remain active.",
            "error": "Could not decode AI postmortem JSON payload",
            "analysis": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_postmortem_payload(raw)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "max_events": int(max_events),
            "summary": "AI postmortem schema validation failed; local analytics remain active.",
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
        "max_events": int(max_events),
        "summary": _s(normalized.get("summary", ""))[:220],
        "error": "",
        "analysis": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def run_openai_postmortem_analysis(
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
    report_path = os.path.join(openai_dir, "postmortem_analysis.json")
    status_path = os.path.join(openai_dir, "postmortem_analysis_status.json")

    enabled, enabled_reason = _postmortem_enabled(cfg)
    timeout_s = _clamp(_f(cfg.get("openai_postmortem_timeout_s", 12.0), 12.0), 1.0, 60.0)
    max_events = max(200, min(50000, int(_f(cfg.get("openai_postmortem_max_events", 5000), 5000))))
    model = _s(cfg.get("openai_postmortem_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    write_report_enabled = bool(cfg.get("openai_postmortem_write_report_enabled", True))
    auto_apply_enabled = bool(cfg.get("openai_postmortem_auto_apply_tuning_enabled", False))

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI postmortem analysis in progress." if enabled else "AI postmortem analysis disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "max_events": int(max_events),
        "write_report_enabled": bool(write_report_enabled),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "applied_tuning_count": 0,
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_postmortem_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_postmortem_analysis(settings=cfg, base_dir=base_dir, postmortem_packet=packet)
    analysis = result.get("analysis", {}) if isinstance(result.get("analysis", {}), dict) else {}
    tuning_rows = analysis.get("tuning_suggestions", []) if isinstance(analysis.get("tuning_suggestions", []), list) else []
    validated = {"validated": [], "skipped": [], "allowlist_size": 0}
    apply_details = {
        "applied": [],
        "applied_count": 0,
        "persisted_verified_count": 0,
        "settings_path": "",
        "changed": False,
    }
    if bool(result.get("active", False)) and bool(auto_apply_enabled) and tuning_rows:
        validated = validate_low_risk_tuning_suggestions(settings=cfg, suggestions=tuning_rows, min_confidence=0.82)
        apply_details = apply_validated_tuning_suggestions(
            base_dir=base_dir,
            settings=cfg,
            validated_rows=validated.get("validated", []) if isinstance(validated.get("validated", []), list) else [],
        )

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(analysis.get("summary", result.get("summary", "")))[:260],
        "main_drags": [_s(x)[:140] for x in list(analysis.get("main_drags", []) or [])[:16] if _s(x)],
        "main_strengths": [_s(x)[:140] for x in list(analysis.get("main_strengths", []) or [])[:16] if _s(x)],
        "skip_recommendations": [_s(x)[:140] for x in list(analysis.get("skip_recommendations", []) or [])[:16] if _s(x)],
        "exit_improvement_recommendations": [
            _s(x)[:160]
            for x in list(analysis.get("exit_improvement_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "capital_reallocation_recommendations": [
            _s(x)[:160]
            for x in list(analysis.get("capital_reallocation_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "tuning_suggestions": [dict(row) for row in tuning_rows[:20] if isinstance(row, dict)],
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "max_events": int(max_events),
            "write_report_enabled": bool(write_report_enabled),
            "auto_apply_enabled": bool(auto_apply_enabled),
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "advisory_only": (not bool(auto_apply_enabled)),
            "validated_suggestions_count": int(len([row for row in list(validated.get("validated", []) or []) if isinstance(row, dict)])),
            "applied_tuning_count": int(max(0.0, _f(apply_details.get("applied_count", 0), 0.0))),
            "persisted_verified_count": int(max(0.0, _f(apply_details.get("persisted_verified_count", 0), 0.0))),
            "settings_path": str(apply_details.get("settings_path", "") or ""),
        },
        "input_packet": {
            "trade_history_summary": dict(packet.get("trade_history_summary", {}) or {})
            if isinstance(packet.get("trade_history_summary", {}), dict)
            else {},
            "execution_audit_summary": dict(packet.get("execution_audit_summary", {}) or {})
            if isinstance(packet.get("execution_audit_summary", {}), dict)
            else {},
            "position_management_summary": dict(packet.get("position_management_summary", {}) or {})
            if isinstance(packet.get("position_management_summary", {}), dict)
            else {},
        },
    }
    report_written = False
    if write_report_enabled:
        atomic_write_json(report_path, report)
        report_written = True

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
        "max_events": int(max_events),
        "write_report_enabled": bool(write_report_enabled),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "advisory_only": (not bool(auto_apply_enabled)),
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": bool(report_written),
        "main_drags": [_s(x)[:140] for x in list(analysis.get("main_drags", []) or [])[:16] if _s(x)],
        "main_strengths": [_s(x)[:140] for x in list(analysis.get("main_strengths", []) or [])[:16] if _s(x)],
        "skip_recommendations": [_s(x)[:140] for x in list(analysis.get("skip_recommendations", []) or [])[:16] if _s(x)],
        "exit_improvement_recommendations": [
            _s(x)[:160]
            for x in list(analysis.get("exit_improvement_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "capital_reallocation_recommendations": [
            _s(x)[:160]
            for x in list(analysis.get("capital_reallocation_recommendations", []) or [])[:16]
            if _s(x)
        ],
        "tuning_suggestions": [dict(row) for row in tuning_rows[:20] if isinstance(row, dict)],
        "tuning_suggestions_count": int(len([row for row in tuning_rows[:20] if isinstance(row, dict)])),
        "validated_suggestions_count": int(len([row for row in list(validated.get("validated", []) or []) if isinstance(row, dict)])),
        "applied_tuning_count": int(max(0.0, _f(apply_details.get("applied_count", 0), 0.0))),
        "persisted_verified_count": int(max(0.0, _f(apply_details.get("persisted_verified_count", 0), 0.0))),
        "applied_tuning": [dict(row) for row in list(apply_details.get("applied", []) or [])[:20] if isinstance(row, dict)],
        "skipped_suggestions": [dict(row) for row in list(validated.get("skipped", []) or [])[:20] if isinstance(row, dict)],
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_postmortem_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "postmortem_analysis_status.json"))


def load_latest_openai_postmortem_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "postmortem_analysis.json"))


def load_current_settings_for_postmortem(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_postmortem_analysis") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
