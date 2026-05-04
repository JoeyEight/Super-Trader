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
_ALLOWED_MODE = {"wait", "prefer_existing", "prefer_new_entries", "mixed"}
_ALLOWED_MARKET_ACTION = {"prefer", "neutral", "deprioritize", "block"}

OPENAI_CAPITAL_PLANNER_SYSTEM_PROMPT = """
You are the cross-market capital allocation planner for an automated trading app.

Your job is to produce a structured portfolio allocation plan across crypto, stocks, and forex.

You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.
You must be conservative when data is stale or weak.

Primary objective:
Improve next-allocation decision quality by prioritizing where the next available capital and position slots should go, while respecting host application constraints.

Planning principles:
1. Prefer quality opportunities over raw trade count.
2. Consider concentration and capital constraints explicitly.
3. Recommend reserving capital when near-term opportunity quality is weak.
4. Prefer explainable, bounded recommendations.
5. When uncertain, use neutral/wait bias rather than forced preference.

Return only JSON matching the schema.
""".strip()


OPENAI_CAPITAL_PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "portfolio_plan", "market_actions", "global_risks"],
    "properties": {
        "summary": {"type": "string"},
        "portfolio_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "preferred_market_order", "reserve_capital_pct", "capital_constrained", "reason"],
            "properties": {
                "mode": {"type": "string", "enum": sorted(_ALLOWED_MODE)},
                "preferred_market_order": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(_MARKETS)},
                },
                "reserve_capital_pct": {"type": "number"},
                "capital_constrained": {"type": "boolean"},
                "reason": {"type": "string"},
            },
        },
        "market_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "action", "confidence", "suggested_capital_share_pct", "reason"],
                "properties": {
                    "market": {"type": "string", "enum": list(_MARKETS)},
                    "action": {"type": "string", "enum": sorted(_ALLOWED_MARKET_ACTION)},
                    "confidence": {"type": "number"},
                    "suggested_capital_share_pct": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "global_risks": {"type": "array", "items": {"type": "string"}},
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


def _safe_read_jsonl_tail(path: str, limit: int = 3000) -> List[Dict[str, Any]]:
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


def _market_thinker_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto_dynamic_status.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = _market(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _top_candidates_for_market(market: str, thinker: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
    mk = _market(market)
    lim = max(1, int(max_candidates or 1))
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
                    "spread_bps": round(max(0.0, _f(row.get("spread_bps", 0.0), 0.0)), 6),
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
                "spread_bps": round(max(0.0, _f(row.get("spread_bps", 0.0), 0.0)), 6),
                "reason": _s(row.get("reason_logic", row.get("reason", "")))[:180],
            }
        )
    return out


def _recent_perf_from_audit(rows: List[Dict[str, Any]], now_ts: int) -> Dict[str, Any]:
    cutoff = int(max(0, int(now_ts) - (7 * 24 * 3600)))
    realized = 0.0
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    stale_exit_count = 0
    churn_count = 0
    drag_reasons: Counter[str] = Counter()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts < cutoff:
            continue
        event = _s(row.get("event")).lower()
        msg = _s(row.get("msg", row.get("tag", row.get("reason", "")))).lower()
        if event == "entry":
            entries += 1
        elif event == "exit":
            exits += 1
            pnl = _f(row.get("realized_pnl_usd", row.get("realized_pnl", row.get("pnl_usd", 0.0))), 0.0)
            realized += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            hold_s = int(max(0.0, _f(row.get("hold_s", 0), 0.0)))
            if hold_s > 0 and hold_s <= 6 * 3600:
                churn_count += 1
        if ("stale" in msg) or ("policy_stale_exit" in msg) or ("misalign" in msg):
            stale_exit_count += 1
            drag_reasons["stale_or_misaligned_exits"] += 1
        if "cooldown" in msg:
            drag_reasons["cooldown_pressure"] += 1
        if "risk cap" in msg:
            drag_reasons["risk_cap_pressure"] += 1
        if "confidence" in msg:
            drag_reasons["confidence_gate_pressure"] += 1
    return {
        "realized_pnl_7d_usd": round(float(realized), 6),
        "entries_7d": int(entries),
        "exits_7d": int(exits),
        "wins_7d": int(wins),
        "losses_7d": int(losses),
        "stale_exit_count_7d": int(stale_exit_count),
        "churn_count_7d": int(churn_count),
        "drag_reasons": [str(k) for k, _ in drag_reasons.most_common(5)],
    }


def build_openai_capital_planner_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    max_candidates = max(1, min(8, int(_f(cfg.get("openai_capital_planner_max_candidates_per_market", 3), 3))))
    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    alloc_ctx = runtime_state.get("cross_market_opportunity", {}) if isinstance(runtime_state.get("cross_market_opportunity", {}), dict) else {}
    pos_review_ctx = runtime_state.get("openai_position_review", {}) if isinstance(runtime_state.get("openai_position_review", {}), dict) else {}

    per_market: Dict[str, Dict[str, Any]] = {}
    total_exposure = 0.0
    account_value = 0.0
    buying_power = 0.0
    margin_available = 0.0
    total_open_positions = 0

    for mk in _MARKETS:
        status = _safe_read_json(_market_status_path(hub_dir, mk))
        thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
        audit_rows = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=4000)
        top_candidates = _top_candidates_for_market(mk, thinker, max_candidates=max_candidates)
        gate = status.get("entry_gate_flags", {}) if isinstance(status.get("entry_gate_flags", {}), dict) else {}
        policy = status.get("automation_policy", {}) if isinstance(status.get("automation_policy", {}), dict) else {}
        trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}
        trade_quality = status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}
        exposure_usd = max(0.0, _f(status.get("exposure_usd", 0.0), 0.0))
        account_value_usd = max(
            0.0,
            _f(status.get("account_value_usd", 0.0), 0.0),
            _f((status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}).get("total_account_value", 0.0), 0.0),
        )
        buying_power_usd = max(
            0.0,
            _f(status.get("buying_power_usd", 0.0), 0.0),
            _f((status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}).get("buying_power", 0.0), 0.0),
            _f(status.get("margin_available_usd", 0.0), 0.0),
        )
        margin_available_usd = max(0.0, _f(status.get("margin_available_usd", buying_power_usd), buying_power_usd))
        open_positions = int(max(0.0, _f(status.get("open_positions", 0), 0.0)))
        if open_positions <= 0:
            positions = status.get("positions", {})
            if isinstance(positions, dict):
                open_positions = len([k for k, row in positions.items() if isinstance(row, dict)])
            elif isinstance(positions, list):
                open_positions = len([row for row in positions if isinstance(row, dict)])
        perf = _recent_perf_from_audit(audit_rows, now_ts=now_i)
        per_market[mk] = {
            "market": mk,
            "exposure_usd": round(float(exposure_usd), 6),
            "open_positions": int(max(0, open_positions)),
            "allow_new_entries": bool(policy.get("allow_new_entries", True)),
            "runtime_trust_score": round(float(max(0.0, _f(trust.get("score", 0.0), 0.0))), 6),
            "trade_quality_confidence_score": round(float(max(0.0, _f(trade_quality.get("confidence_score", 0.0), 0.0))), 6),
            "trade_quality_decision": _s(trade_quality.get("decision"))[:24].lower(),
            "loss_streak": int(max(0.0, _f(gate.get("loss_streak", status.get("loss_streak", 0)), 0.0))),
            "top_candidates": top_candidates,
            "recent_performance": perf,
        }
        total_exposure += exposure_usd
        account_value = max(account_value, account_value_usd)
        buying_power = max(buying_power, buying_power_usd)
        margin_available = max(margin_available, margin_available_usd)
        total_open_positions += int(max(0, open_positions))

    total_exposure_pct = ((total_exposure / max(1e-6, account_value)) * 100.0) if account_value > 0.0 else 0.0
    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "portfolio_context": {
            "account_value_usd": round(float(max(0.0, account_value)), 6),
            "buying_power_usd": round(float(max(0.0, buying_power)), 6),
            "margin_available_usd": round(float(max(0.0, margin_available)), 6),
            "total_exposure_usd": round(float(max(0.0, total_exposure)), 6),
            "total_exposure_pct": round(float(max(0.0, total_exposure_pct)), 6),
            "total_open_positions": int(max(0, total_open_positions)),
            "market_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))), 6),
            "crypto_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("stock_max_total_exposure_pct", 0.0), 0.0))), 6),
            "forex_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("forex_max_total_exposure_pct", 0.0), 0.0))), 6),
            "stock_max_open_positions": int(max(0.0, _f(cfg.get("stock_max_open_positions", 0), 0.0))),
            "forex_max_open_positions": int(max(0.0, _f(cfg.get("forex_max_open_positions", 0), 0.0))),
            "crypto_max_open_positions": int(max(0.0, _f(cfg.get("crypto_max_open_positions", 0), 0.0))),
        },
        "current_allocator_context": {
            "summary": _s(alloc_ctx.get("summary", ""))[:220],
            "best_market": _s(alloc_ctx.get("best_market", "")).lower(),
            "decisions": dict(alloc_ctx.get("decisions", {}) or {}) if isinstance(alloc_ctx.get("decisions", {}), dict) else {},
            "scores": dict(alloc_ctx.get("scores", {}) or {}) if isinstance(alloc_ctx.get("scores", {}), dict) else {},
        },
        "position_review_context": {
            "summary": _s(pos_review_ctx.get("summary", ""))[:220],
            "status": _s(pos_review_ctx.get("status", "")).lower(),
            "actions_count": int(max(0.0, _f(pos_review_ctx.get("actions_count", 0), 0.0))),
            "blocked_count": int(max(0.0, _f(pos_review_ctx.get("blocked_count", 0), 0.0))),
            "portfolio_risks": [
                _s(x)[:96]
                for x in list(pos_review_ctx.get("portfolio_risks", []) or [])[:8]
                if _s(x)
            ],
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


def _normalize_market_action(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _market(src.get("market"))
    action = _s(src.get("action")).lower()
    if action not in _ALLOWED_MARKET_ACTION:
        action = "neutral"
    return {
        "market": market,
        "action": action,
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "suggested_capital_share_pct": round(_clamp(_f(src.get("suggested_capital_share_pct", 0.0), 0.0), 0.0, 100.0), 6),
        "reason": _s(src.get("reason"))[:220],
    }


def _normalize_planner_payload(raw: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Capital planner response was not a JSON object"

    plan = raw.get("portfolio_plan", {}) if isinstance(raw.get("portfolio_plan", {}), dict) else {}
    mode = _s(plan.get("mode")).lower()
    if mode not in _ALLOWED_MODE:
        mode = "wait"
    order_src = plan.get("preferred_market_order", [])
    if not isinstance(order_src, list):
        order_src = []
    seen = set()
    order: List[str] = []
    for item in order_src:
        mk = _market(item)
        if mk in _MARKETS and mk not in seen:
            seen.add(mk)
            order.append(mk)
    for mk in _MARKETS:
        if mk not in seen:
            order.append(mk)

    actions_src = raw.get("market_actions", [])
    if not isinstance(actions_src, list):
        actions_src = []
    actions = [_normalize_market_action(row) for row in actions_src[:12] if isinstance(row, dict)]
    by_market = {str(row.get("market", "")): row for row in actions if isinstance(row, dict)}
    for mk in _MARKETS:
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "action": "neutral",
                "confidence": 0.0,
                "suggested_capital_share_pct": 0.0,
                "reason": "No explicit planner recommendation for this market.",
            }
    actions = [by_market[mk] for mk in _MARKETS]
    risks_src = raw.get("global_risks", [])
    if not isinstance(risks_src, list):
        risks_src = []
    return {
        "summary": _s(raw.get("summary"))[:260],
        "portfolio_plan": {
            "mode": mode,
            "preferred_market_order": order,
            "reserve_capital_pct": round(_clamp(_f(plan.get("reserve_capital_pct", 0.0), 0.0), 0.0, 95.0), 6),
            "capital_constrained": _b(plan.get("capital_constrained"), False),
            "reason": _s(plan.get("reason"))[:220],
        },
        "market_actions": actions,
        "global_risks": [_s(x)[:120] for x in risks_src[:12] if _s(x)],
    }, ""


def _planner_enabled(settings: Dict[str, Any] | None, *, mode: str = "") -> Tuple[bool, str]:
    cfg = settings if isinstance(settings, dict) else {}
    if not _b(cfg.get("openai_capital_planner_enabled"), False):
        return False, "disabled"
    cur_mode = _s(mode).lower()
    if cur_mode not in {"live", "paper"}:
        cur_mode = _trading_mode(cfg)
    if cur_mode == "live" and (not _b(cfg.get("openai_capital_planner_live_enabled"), True)):
        return False, "live_disabled"
    if cur_mode == "paper" and (not _b(cfg.get("openai_capital_planner_paper_enabled"), True)):
        return False, "paper_disabled"
    return True, "enabled"


def request_openai_capital_planner(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    planner_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled, enabled_reason = _planner_enabled(cfg, mode=_s(planner_packet.get("mode", "")))
    model = _s(cfg.get("openai_capital_planner_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_capital_planner_timeout_s", 6.0), 6.0), 1.0, 30.0)
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": enabled_reason,
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "",
            "error": "",
            "plan": {},
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
            "summary": "AI capital planner unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "plan": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": OPENAI_CAPITAL_PLANNER_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "capital_planner", "capital_planner_input": planner_packet},
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
                "name": "capital_planner",
                "strict": True,
                "schema": OPENAI_CAPITAL_PLANNER_SCHEMA,
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
            "summary": "AI capital planner timed out; local allocator remains active.",
            "error": "OpenAI request timed out",
            "plan": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI capital planner request failed; local allocator remains active.",
            "error": _s(exc)[:180],
            "plan": {},
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
            "summary": "AI capital planner unavailable from OpenAI API; local allocator remains active.",
            "error": f"HTTP {int(resp.status_code)}",
            "plan": {},
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
            "summary": "AI capital planner returned invalid JSON; local allocator remains active.",
            "error": "OpenAI response could not be parsed as JSON",
            "plan": {},
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
            "summary": "AI capital planner returned no structured output; local allocator remains active.",
            "error": "No output text found in OpenAI response",
            "plan": {},
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
            "summary": "AI capital planner output was malformed; local allocator remains active.",
            "error": "Could not decode AI capital planner JSON payload",
            "plan": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_planner_payload(raw)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "AI capital planner schema validation failed; local allocator remains active.",
            "error": _s(err)[:180],
            "plan": {},
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
        "plan": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def run_openai_capital_planner(
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
    report_path = os.path.join(openai_dir, "capital_planner.json")
    status_path = os.path.join(openai_dir, "capital_planner_status.json")

    enabled, enabled_reason = _planner_enabled(cfg, mode=_trading_mode(cfg))
    interval_s = _clamp(_f(cfg.get("openai_capital_planner_interval_s", 180.0), 180.0), 30.0, 86400.0)
    timeout_s = _clamp(_f(cfg.get("openai_capital_planner_timeout_s", 6.0), 6.0), 1.0, 30.0)
    max_candidates = max(1, min(8, int(_f(cfg.get("openai_capital_planner_max_candidates_per_market", 3), 3))))
    model = _s(cfg.get("openai_capital_planner_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": bool(enabled),
        "active": False,
        "status": ("running" if enabled else enabled_reason),
        "summary": ("AI capital planner in progress." if enabled else "AI capital planner disabled."),
        "last_attempt_ts": int(now_i),
        "last_completed_ts": 0,
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_candidates_per_market": int(max_candidates),
        "advisory_only": True,
        "error": "",
        "latency_ms": 0,
        "report_path": str(report_path),
        "report_written": False,
        "portfolio_plan": {},
        "market_actions": [],
        "global_risks": [],
    }
    atomic_write_json(status_path, pre_status)

    packet = build_openai_capital_planner_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_capital_planner(settings=cfg, base_dir=base_dir, planner_packet=packet)
    plan = result.get("plan", {}) if isinstance(result.get("plan", {}), dict) else {}

    report = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": _s(result.get("status", "")),
        "enabled": bool(enabled),
        "summary": _s(plan.get("summary", result.get("summary", "")))[:260],
        "portfolio_plan": dict(plan.get("portfolio_plan", {}) or {}) if isinstance(plan.get("portfolio_plan", {}), dict) else {},
        "market_actions": [dict(row) for row in list(plan.get("market_actions", []) or [])[:12] if isinstance(row, dict)],
        "global_risks": [_s(x)[:120] for x in list(plan.get("global_risks", []) or [])[:12] if _s(x)],
        "meta": {
            "model": str(model),
            "timeout_s": float(timeout_s),
            "interval_s": float(interval_s),
            "max_candidates_per_market": int(max_candidates),
            "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
            "response_status": _s(result.get("status", "")),
            "response_error": _s(result.get("error", ""))[:180],
            "mode": _s(packet.get("mode", "")),
            "advisory_only": True,
        },
        "input_packet": {
            "portfolio_context": dict(packet.get("portfolio_context", {}) or {}) if isinstance(packet.get("portfolio_context", {}), dict) else {},
            "current_allocator_context": dict(packet.get("current_allocator_context", {}) or {}) if isinstance(packet.get("current_allocator_context", {}), dict) else {},
            "position_review_context": dict(packet.get("position_review_context", {}) or {}) if isinstance(packet.get("position_review_context", {}), dict) else {},
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
        "summary": _s(plan.get("summary", result.get("summary", "")))[:260],
        "last_attempt_ts": int(now_i),
        "last_completed_ts": int(now_i),
        "date_local": str(date_local),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "interval_s": float(interval_s),
        "max_candidates_per_market": int(max_candidates),
        "advisory_only": True,
        "latency_ms": int(max(0.0, _f(result.get("latency_ms", 0), 0.0))),
        "error": _s(result.get("error", ""))[:180],
        "report_path": str(report_path),
        "report_written": True,
        "portfolio_plan": dict(plan.get("portfolio_plan", {}) or {}) if isinstance(plan.get("portfolio_plan", {}), dict) else {},
        "market_actions": [dict(row) for row in list(plan.get("market_actions", []) or [])[:12] if isinstance(row, dict)],
        "market_actions_by_market": {
            _market(row.get("market", "")): dict(row)
            for row in list(plan.get("market_actions", []) or [])[:12]
            if isinstance(row, dict)
        },
        "global_risks": [_s(x)[:120] for x in list(plan.get("global_risks", []) or [])[:12] if _s(x)],
    }
    atomic_write_json(status_path, status)
    return status


def load_latest_openai_capital_planner_status(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "capital_planner_status.json"))


def load_latest_openai_capital_planner_report(hub_dir: str) -> Dict[str, Any]:
    return _safe_read_json(os.path.join(hub_dir, "openai", "capital_planner.json"))


def load_current_settings_for_capital_planner(base_dir: str) -> Dict[str, Any]:
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_capital_planner") or {}
    return sanitize_settings(raw if isinstance(raw, dict) else {})
