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
from app.settings_utils import PROFILE_MANUAL_OVERRIDE_ALLOWLIST, sanitize_settings

_ALLOWED_ASSESSMENT = {"healthy", "caution", "unhealthy"}
_ALLOWED_MARKET = {"crypto", "stocks", "forex"}


OPENAI_TRADE_REVIEW_SYSTEM_PROMPT = """
You are the nightly trade-review and tuning advisor for an automated multi-market trading app.

Your job is to evaluate recent structured trading behavior for crypto, stocks, and forex and return strict JSON.

You are NOT a live trade executor.
You do NOT place trades.
You do NOT override hard local safety, compliance, broker, or execution controls.
You do NOT invent missing data.

Primary objective:
Improve next-day decision quality and reduce avoidable churn, stale exits, and weak entries while preserving risk discipline.

Review principles:
1. Prefer practical, bounded, explainable tuning recommendations.
2. Be conservative when evidence is weak or data is stale/incomplete.
3. Prioritize churn reduction and drawdown control over increasing trade count.
4. Highlight concentrated loss patterns, stale-exit pressure, and weak setup families.
5. Suggest adjustments only when supported by observed data in the payload.
6. If no clear tuning change is justified, recommend no changes.

Return only JSON matching the schema.
""".strip()


OPENAI_NIGHTLY_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "overall_assessment",
        "market_reviews",
        "tuning_suggestions",
        "risk_flags",
        "next_day_guidance",
    ],
    "properties": {
        "summary": {"type": "string"},
        "overall_assessment": {"type": "string", "enum": sorted(_ALLOWED_ASSESSMENT)},
        "market_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["market", "assessment", "main_drags", "main_strengths", "recommended_actions"],
                "properties": {
                    "market": {"type": "string", "enum": ["crypto", "stocks", "forex"]},
                    "assessment": {"type": "string", "enum": sorted(_ALLOWED_ASSESSMENT)},
                    "main_drags": {"type": "array", "items": {"type": "string"}},
                    "main_strengths": {"type": "array", "items": {"type": "string"}},
                    "recommended_actions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "tuning_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["setting_key", "current_value", "suggested_value", "confidence", "reason"],
                "properties": {
                    "setting_key": {"type": "string"},
                    "current_value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "integer"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
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
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "next_day_guidance": {"type": "array", "items": {"type": "string"}},
    },
}


# Low-risk setting bounds allowed for optional auto-apply.
_LOW_RISK_TUNING_RULES: Dict[str, Dict[str, Any]] = {
    "max_dca_buys_per_24h": {"type": "int", "min": 0, "max": 8},
    "dca_multiplier": {"type": "float", "min": 1.0, "max": 4.0, "precision": 3},
    "crypto_dynamic_target_count": {"type": "int", "min": 4, "max": 20},
    "crypto_dynamic_scan_interval_s": {"type": "float", "min": 15.0, "max": 900.0, "precision": 2},
    "crypto_dynamic_rotation_cooldown_s": {"type": "float", "min": 120.0, "max": 10800.0, "precision": 2},
    "crypto_dynamic_min_projected_edge_pct": {"type": "float", "min": 0.05, "max": 2.5, "precision": 4},
    "stock_score_threshold": {"type": "float", "min": 0.05, "max": 1.5, "precision": 4},
    "forex_score_threshold": {"type": "float", "min": 0.05, "max": 1.5, "precision": 4},
    "stock_scan_max_symbols": {"type": "int", "min": 16, "max": 220},
    "forex_scan_max_pairs": {"type": "int", "min": 8, "max": 80},
    "stock_max_open_positions": {"type": "int", "min": 1, "max": 8},
    "forex_max_open_positions": {"type": "int", "min": 1, "max": 8},
    "stock_stale_alignment_grace_cycles": {"type": "int", "min": 1, "max": 8},
    "forex_stale_alignment_grace_cycles": {"type": "int", "min": 1, "max": 8},
    "max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "stock_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "forex_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
    "market_max_total_exposure_pct": {"type": "float", "min": 10.0, "max": 90.0, "precision": 2},
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _s(value: Any) -> str:
    return str(value or "").strip()


def _trading_mode(settings: Dict[str, Any]) -> str:
    cfg = settings if isinstance(settings, dict) else {}
    if bool(cfg.get("alpaca_paper_mode", False)) or bool(cfg.get("oanda_practice_mode", False)):
        return "paper"
    return "live"


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl_tail(path: str, limit: int = 5000) -> List[Dict[str, Any]]:
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
    mk = str(market or "").strip().lower()
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_state_path(hub_dir: str, market: str) -> str:
    mk = str(market or "").strip().lower()
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_state.json")
    if mk == "forex":
        return os.path.join(hub_dir, "forex", "forex_trader_state.json")
    return ""


def _market_audit_path(hub_dir: str, market: str) -> str:
    mk = str(market or "").strip().lower()
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "execution_audit.jsonl")
    return os.path.join(hub_dir, "forex", "execution_audit.jsonl")


def _event_symbol(market: str, row: Dict[str, Any]) -> str:
    mk = str(market or "").strip().lower()
    if mk == "forex":
        return _s(row.get("instrument", row.get("pair", row.get("symbol", "")))).upper()
    return _s(row.get("symbol", row.get("pair", row.get("instrument", "")))).upper()


def _event_reason(row: Dict[str, Any]) -> str:
    for key in ("tag", "source", "reason", "msg"):
        val = _s(row.get(key))
        if val:
            return val
    return ""


def _realized_pnl(row: Dict[str, Any]) -> float:
    for key in ("realized_pnl_usd", "realized_pnl", "realized_pl", "realized", "pnl_usd", "pnl"):
        if key in row:
            return float(_f(row.get(key, 0.0), 0.0))
    return 0.0


def _trade_hold_s(row: Dict[str, Any]) -> int:
    return int(max(0.0, _f(row.get("hold_s", 0), 0.0)))


def _reason_token(text: str) -> str:
    raw = _s(text).lower()
    if not raw:
        return "unknown"
    out = "".join((ch if ch.isalnum() else "_") for ch in raw)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out[:56] if out else "unknown"


def _is_stale_reason(reason: str) -> bool:
    text = _s(reason).lower()
    if not text:
        return False
    markers = ("stale", "policy_stale_exit", "misalign", "alignment")
    return any(tok in text for tok in markers)


def _is_dca_or_add_reason(reason: str) -> bool:
    text = _s(reason).lower()
    if not text:
        return False
    markers = ("dca", "add", "averag")
    return any(tok in text for tok in markers)


def _open_position_alignment_counts(market: str, status: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, int]:
    mk = str(market or "").strip().lower()
    if mk == "crypto":
        positions = status.get("positions", {}) if isinstance(status.get("positions", {}), dict) else {}
        aligned = 0
        misaligned = 0
        for _, row in positions.items():
            if not isinstance(row, dict):
                continue
            qty = _f(row.get("quantity", 0.0), 0.0)
            if qty <= 0.0:
                continue
            if bool(row.get("aligned_with_strategy", True)):
                aligned += 1
            else:
                misaligned += 1
        return {
            "open_positions": int(aligned + misaligned),
            "aligned_positions": int(aligned),
            "misaligned_positions": int(misaligned),
        }

    open_meta = state.get("open_meta", {}) if isinstance(state.get("open_meta", {}), dict) else {}
    stale = state.get("stale_alignment_streaks", {}) if isinstance(state.get("stale_alignment_streaks", {}), dict) else {}
    aligned = 0
    misaligned = 0
    for symbol in list(open_meta.keys()):
        sym = _s(symbol).upper()
        if not sym:
            continue
        streak = int(max(0.0, _f(stale.get(sym, stale.get(symbol, 0)), 0.0)))
        if streak > 0:
            misaligned += 1
        else:
            aligned += 1
    return {
        "open_positions": int(aligned + misaligned),
        "aligned_positions": int(aligned),
        "misaligned_positions": int(misaligned),
    }


def _market_metrics(
    *,
    market: str,
    status: Dict[str, Any],
    state: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
    cutoff_ts: int,
    max_events: int,
) -> Dict[str, Any]:
    mk = str(market or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    for row in list(audit_rows or []):
        if not isinstance(row, dict):
            continue
        ts = int(max(0.0, _f(row.get("ts", 0), 0.0)))
        if ts <= 0 or ts < int(cutoff_ts):
            continue
        rows.append(dict(row))
    if len(rows) > int(max_events):
        rows = rows[-int(max_events):]

    entries = 0
    exits = 0
    wins = 0
    losses = 0
    realized_pnl = 0.0
    stale_exit_count = 0
    churn_count = 0
    dca_add_count = 0
    hold_samples: List[int] = []
    reject_reason_counts: Counter[str] = Counter()
    exit_reason_counts: Counter[str] = Counter()
    loss_by_symbol: Dict[str, float] = {}
    loss_exits_by_symbol: Dict[str, int] = {}

    churn_max_s = 6 * 3600

    for row in rows:
        evt = _s(row.get("event")).lower()
        reason = _event_reason(row)
        reason_token = _reason_token(reason)
        if evt in {"entry", "order_entry", "buy"}:
            entries += 1
            if _is_dca_or_add_reason(reason):
                dca_add_count += 1
            continue
        if evt in {"exit", "sell", "close"}:
            exits += 1
            pnl = float(_realized_pnl(row))
            realized_pnl += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
                sym = _event_symbol(mk, row)
                if sym:
                    loss_by_symbol[sym] = float(loss_by_symbol.get(sym, 0.0) + pnl)
                    loss_exits_by_symbol[sym] = int(loss_exits_by_symbol.get(sym, 0) + 1)
            hold_s = _trade_hold_s(row)
            if hold_s > 0:
                hold_samples.append(int(hold_s))
                if hold_s <= churn_max_s:
                    churn_count += 1
            if _is_stale_reason(reason):
                stale_exit_count += 1
            exit_reason_counts[reason_token] += 1
            if _is_dca_or_add_reason(reason):
                dca_add_count += 1
            continue
        if evt in {"shadow_live_divergence", "entry_reject", "reject"}:
            reject_reason_counts[reason_token] += 1

    entry_eval_reasons = status.get("entry_eval_reason_counts", {}) if isinstance(status.get("entry_eval_reason_counts", {}), dict) else {}
    for key, value in entry_eval_reasons.items():
        k = _reason_token(_s(key))
        v = int(max(0.0, _f(value, 0.0)))
        if k and v > 0:
            reject_reason_counts[k] += v

    underperformers = sorted(loss_by_symbol.items(), key=lambda item: float(item[1]))[:5]
    top_underperformers = [
        {
            "symbol": str(sym),
            "realized_pnl_usd": round(float(val), 6),
            "loss_exits": int(loss_exits_by_symbol.get(sym, 0) or 0),
        }
        for sym, val in underperformers
    ]

    align = _open_position_alignment_counts(mk, status, state)
    avg_hold_s = int(round(sum(hold_samples) / max(1, len(hold_samples)))) if hold_samples else 0

    account_value_usd = max(
        _f(status.get("account_value_usd", 0.0), 0.0),
        _f((status.get("account", {}) if isinstance(status.get("account", {}), dict) else {}).get("total_account_value", 0.0), 0.0),
    )

    return {
        "market": mk,
        "events_analyzed": int(len(rows)),
        "entries": int(entries),
        "exits": int(exits),
        "wins": int(wins),
        "losses": int(losses),
        "realized_pnl_usd": round(float(realized_pnl), 6),
        "stale_exit_count": int(stale_exit_count),
        "churn_count": int(churn_count),
        "dca_add_count": int(dca_add_count),
        "avg_hold_s": int(avg_hold_s),
        "top_exit_reasons": [
            {"reason": str(key), "count": int(val)}
            for key, val in exit_reason_counts.most_common(6)
        ],
        "top_reject_reasons": [
            {"reason": str(key), "count": int(val)}
            for key, val in reject_reason_counts.most_common(6)
        ],
        "open_position_alignment": align,
        "top_underperformers": top_underperformers,
        "loss_streak": int(max(0.0, _f((status.get("entry_gate_flags", {}) if isinstance(status.get("entry_gate_flags", {}), dict) else {}).get("loss_streak", status.get("loss_streak", 0)), 0.0))),
        "automation_policy_summary": _s((status.get("automation_policy", {}) if isinstance(status.get("automation_policy", {}), dict) else {}).get("summary", ""))[:220],
        "trade_quality_decision": _s((status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}).get("decision", ""))[:24].lower(),
        "trade_quality_confidence": round(_f((status.get("trade_quality", {}) if isinstance(status.get("trade_quality", {}), dict) else {}).get("confidence_score", 0.0), 0.0), 4),
        "allocator_summary": _s((status.get("opportunity_allocator", {}) if isinstance(status.get("opportunity_allocator", {}), dict) else {}).get("summary", ""))[:220],
        "account_value_usd": round(float(max(0.0, account_value_usd)), 6),
        "exposure_usd": round(float(max(0.0, _f(status.get("exposure_usd", 0.0), 0.0))), 6),
    }


def build_nightly_review_packet(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    now_ts_value: int | None = None,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    now_i = int(now_ts_value if isinstance(now_ts_value, int) and now_ts_value > 0 else int(time.time()))
    lookback_days = max(1, min(30, int(_f(cfg.get("openai_nightly_review_lookback_days", 7), 7))))
    max_events = max(200, min(50000, int(_f(cfg.get("openai_nightly_review_max_events", 5000), 5000))))
    cutoff_ts = int(now_i - (lookback_days * 86400))

    market_data: Dict[str, Dict[str, Any]] = {}
    portfolio_value = 0.0
    total_exposure = 0.0

    for mk in ("crypto", "stocks", "forex"):
        status = _safe_read_json(_market_status_path(hub_dir, mk))
        state_path = _market_state_path(hub_dir, mk)
        state = _safe_read_json(state_path) if state_path else {}
        audit = _safe_read_jsonl_tail(_market_audit_path(hub_dir, mk), limit=max_events * 2)
        metrics = _market_metrics(
            market=mk,
            status=status,
            state=state,
            audit_rows=audit,
            cutoff_ts=cutoff_ts,
            max_events=max_events,
        )
        market_data[mk] = metrics
        portfolio_value += max(0.0, _f(metrics.get("account_value_usd", 0.0), 0.0))
        total_exposure += max(0.0, _f(metrics.get("exposure_usd", 0.0), 0.0))

    runtime_state = _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    cross_market = runtime_state.get("cross_market_opportunity", {}) if isinstance(runtime_state.get("cross_market_opportunity", {}), dict) else {}

    return {
        "timestamp": int(now_i),
        "date_local": time.strftime("%Y-%m-%d", time.localtime(now_i)),
        "mode": _trading_mode(cfg),
        "settings_profile": _s(cfg.get("settings_profile", "balanced")).lower(),
        "settings_control_mode": _s(cfg.get("settings_control_mode", "self_managed")).lower(),
        "lookback_days": int(lookback_days),
        "max_events": int(max_events),
        "portfolio_context": {
            "portfolio_value_usd_est": round(float(max(0.0, portfolio_value)), 6),
            "total_exposure_usd_est": round(float(max(0.0, total_exposure)), 6),
            "total_exposure_pct_est": round(
                float(((total_exposure / max(1e-6, portfolio_value)) * 100.0) if portfolio_value > 0.0 else 0.0),
                4,
            ),
            "market_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))), 4),
            "crypto_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("max_total_exposure_pct", 0.0), 0.0))), 4),
            "stock_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("stock_max_total_exposure_pct", 0.0), 0.0))), 4),
            "forex_max_total_exposure_pct": round(float(max(0.0, _f(cfg.get("forex_max_total_exposure_pct", 0.0), 0.0))), 4),
            "stock_max_open_positions": int(max(0, _f(cfg.get("stock_max_open_positions", 1), 1))),
            "forex_max_open_positions": int(max(0, _f(cfg.get("forex_max_open_positions", 1), 1))),
            "crypto_max_open_positions": int(max(0, _f(cfg.get("crypto_max_open_positions", 8), 8))),
            "stock_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("stock_max_daily_loss_pct", 0.0), 0.0))), 4),
            "forex_max_daily_loss_pct": round(float(max(0.0, _f(cfg.get("forex_max_daily_loss_pct", 0.0), 0.0))), 4),
        },
        "market_metrics": {
            "crypto": market_data.get("crypto", {}),
            "stocks": market_data.get("stocks", {}),
            "forex": market_data.get("forex", {}),
        },
        "cross_market_allocator": {
            "summary": _s(cross_market.get("summary", ""))[:220],
            "best_market": _s(cross_market.get("best_market", "")).lower(),
            "decisions": dict(cross_market.get("decisions", {}) or {}) if isinstance(cross_market.get("decisions", {}), dict) else {},
        },
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


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return _s(value)[:96]


def _normalize_market_review(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    market = _s(src.get("market")).lower()
    if market not in _ALLOWED_MARKET:
        market = "crypto"
    assessment = _s(src.get("assessment")).lower()
    if assessment not in _ALLOWED_ASSESSMENT:
        assessment = "caution"

    def _arr(key: str, cap: int) -> List[str]:
        items = src.get(key, [])
        if not isinstance(items, list):
            items = []
        return [_s(x)[:160] for x in items[:cap] if _s(x)]

    return {
        "market": market,
        "assessment": assessment,
        "main_drags": _arr("main_drags", 6),
        "main_strengths": _arr("main_strengths", 6),
        "recommended_actions": _arr("recommended_actions", 8),
    }


def _normalize_tuning_suggestion(row: Dict[str, Any] | None) -> Dict[str, Any]:
    src = row if isinstance(row, dict) else {}
    return {
        "setting_key": _s(src.get("setting_key"))[:80],
        "current_value": _normalize_scalar(src.get("current_value")),
        "suggested_value": _normalize_scalar(src.get("suggested_value")),
        "confidence": round(_clamp(_f(src.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "reason": _s(src.get("reason"))[:220],
    }


def _normalize_review_payload(raw: Dict[str, Any], *, max_suggestions: int) -> Tuple[Dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "Nightly review response was not a JSON object"

    overall = _s(raw.get("overall_assessment")).lower()
    if overall not in _ALLOWED_ASSESSMENT:
        return None, "Nightly review overall_assessment enum was invalid"

    market_rows_src = raw.get("market_reviews", [])
    if not isinstance(market_rows_src, list):
        market_rows_src = []
    market_rows = [_normalize_market_review(row) for row in market_rows_src[:12] if isinstance(row, dict)]

    # Ensure every market has at least one review row for deterministic consumers.
    by_market = {str(row.get("market", "")): row for row in market_rows if isinstance(row, dict)}
    for mk in ("crypto", "stocks", "forex"):
        if mk not in by_market:
            by_market[mk] = {
                "market": mk,
                "assessment": "caution",
                "main_drags": [],
                "main_strengths": [],
                "recommended_actions": ["No strong recommendation generated."],
            }
    market_rows = [by_market[mk] for mk in ("crypto", "stocks", "forex")]

    sugg_src = raw.get("tuning_suggestions", [])
    if not isinstance(sugg_src, list):
        sugg_src = []
    suggestions = [
        _normalize_tuning_suggestion(row)
        for row in sugg_src[: max(1, int(max_suggestions))]
        if isinstance(row, dict)
    ]

    risk_flags = raw.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    next_day = raw.get("next_day_guidance", [])
    if not isinstance(next_day, list):
        next_day = []

    return {
        "summary": _s(raw.get("summary"))[:260],
        "overall_assessment": overall,
        "market_reviews": market_rows,
        "tuning_suggestions": suggestions,
        "risk_flags": [_s(x)[:120] for x in risk_flags[:12] if _s(x)],
        "next_day_guidance": [_s(x)[:180] for x in next_day[:12] if _s(x)],
    }, ""


def request_openai_nightly_trade_review(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    review_packet: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    enabled = bool(cfg.get("openai_nightly_review_enabled", False))
    model = _s(cfg.get("openai_nightly_review_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = _clamp(_f(cfg.get("openai_nightly_review_timeout_s", 12.0), 12.0), 1.0, 60.0)
    if not enabled:
        return {
            "enabled": False,
            "active": False,
            "status": "disabled",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "",
            "error": "",
            "review": {},
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
            "summary": "Nightly AI trade review unavailable (missing OpenAI API key).",
            "error": "OpenAI API key was not configured",
            "review": {},
            "latency_ms": 0,
        }

    endpoint = _s(cfg.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    max_suggestions = max(3, min(20, int(_f(cfg.get("openai_nightly_review_max_suggestions", 12), 12))))
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": OPENAI_TRADE_REVIEW_SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"task": "nightly_trade_review", "review_input": review_packet},
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
                "name": "nightly_trade_review",
                "strict": True,
                "schema": OPENAI_NIGHTLY_REVIEW_SCHEMA,
            }
        },
    }

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
        return {
            "enabled": True,
            "active": False,
            "status": "timeout",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "Nightly AI trade review timed out; local behavior remains unchanged.",
            "error": "OpenAI request timed out",
            "review": {},
            "latency_ms": int(round((time.time() - started) * 1000.0)),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "status": "request_error",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "Nightly AI trade review request failed; local behavior remains unchanged.",
            "error": _s(exc)[:180],
            "review": {},
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
            "summary": "Nightly AI trade review unavailable from OpenAI API; local behavior remains unchanged.",
            "error": f"HTTP {int(resp.status_code)}",
            "review": {},
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
            "summary": "Nightly AI trade review returned invalid JSON; local behavior remains unchanged.",
            "error": "OpenAI response could not be parsed as JSON",
            "review": {},
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
            "summary": "Nightly AI trade review returned no structured output; local behavior remains unchanged.",
            "error": "No output text found in OpenAI response",
            "review": {},
            "latency_ms": latency_ms,
        }

    try:
        raw_review = json.loads(text)
    except Exception:
        return {
            "enabled": True,
            "active": False,
            "status": "malformed_response",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "Nightly AI trade review output was malformed; local behavior remains unchanged.",
            "error": "Could not decode AI nightly review JSON payload",
            "review": {},
            "latency_ms": latency_ms,
        }

    normalized, err = _normalize_review_payload(raw_review, max_suggestions=max_suggestions)
    if not isinstance(normalized, dict):
        return {
            "enabled": True,
            "active": False,
            "status": "schema_validation_failed",
            "model": model,
            "timeout_s": float(timeout_s),
            "summary": "Nightly AI trade review schema validation failed; local behavior remains unchanged.",
            "error": _s(err)[:180],
            "review": {},
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
        "review": normalized,
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id")),
    }


def _coerce_tuning_value(rule: Dict[str, Any], value: Any) -> Tuple[Any, bool]:
    kind = _s(rule.get("type")).lower()
    lo = _f(rule.get("min", 0.0), 0.0)
    hi = _f(rule.get("max", 0.0), 0.0)
    if kind == "int":
        try:
            val = int(round(_f(value, lo)))
        except Exception:
            return None, False
        val = int(max(int(lo), min(int(hi), int(val))))
        return int(val), True
    if kind == "float":
        try:
            val_f = _f(value, lo)
        except Exception:
            return None, False
        val_f = float(max(lo, min(hi, float(val_f))))
        precision = int(max(0, _f(rule.get("precision", 4), 4.0)))
        return round(float(val_f), precision), True
    return None, False


def validate_low_risk_tuning_suggestions(
    *,
    settings: Dict[str, Any],
    suggestions: List[Dict[str, Any]] | None,
    min_confidence: float = 0.80,
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    rows = suggestions if isinstance(suggestions, list) else []
    validated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _s(row.get("setting_key"))
        conf = _clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0)
        reason = _s(row.get("reason"))[:220]
        if key not in _LOW_RISK_TUNING_RULES:
            skipped.append({
                "setting_key": key,
                "reason": reason or "setting not in low-risk allowlist",
                "skip_code": "not_allowlisted",
                "confidence": round(conf, 6),
            })
            continue
        if conf < float(min_confidence):
            skipped.append({
                "setting_key": key,
                "reason": reason or "confidence below apply threshold",
                "skip_code": "low_confidence",
                "confidence": round(conf, 6),
            })
            continue
        rule = _LOW_RISK_TUNING_RULES.get(key, {})
        suggested_value, ok = _coerce_tuning_value(rule, row.get("suggested_value"))
        if not ok:
            skipped.append({
                "setting_key": key,
                "reason": reason or "suggested value could not be coerced",
                "skip_code": "bad_value",
                "confidence": round(conf, 6),
            })
            continue
        current_value = cfg.get(key)
        validated.append(
            {
                "setting_key": key,
                "current_value": _normalize_scalar(current_value),
                "suggested_value": _normalize_scalar(suggested_value),
                "confidence": round(conf, 6),
                "reason": reason,
            }
        )

    return {
        "validated": validated,
        "skipped": skipped,
        "allowlist_size": int(len(_LOW_RISK_TUNING_RULES)),
    }


def _parse_profile_override_keys(value: Any) -> set[str]:
    if isinstance(value, str):
        seq = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        seq = [str(part).strip() for part in value if str(part).strip()]
    else:
        seq = []
    return {key for key in seq if key in PROFILE_MANUAL_OVERRIDE_ALLOWLIST}


def apply_validated_tuning_suggestions(
    *,
    base_dir: str,
    settings: Dict[str, Any],
    validated_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    src_settings = settings if isinstance(settings, dict) else {}
    rows = validated_rows if isinstance(validated_rows, list) else []
    if not rows:
        return {
            "applied": [],
            "applied_count": 0,
            "persisted_verified_count": 0,
            "settings_path": "",
            "changed": False,
        }

    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw = read_settings_file(settings_path, module_name="openai_trade_review") or {}
    merged = dict(raw if isinstance(raw, dict) else {})
    changed = False
    applied: List[Dict[str, Any]] = []
    preset_mode = str(merged.get("settings_control_mode", src_settings.get("settings_control_mode", "self_managed")) or "self_managed").strip().lower()
    manual_overrides = _parse_profile_override_keys(merged.get("profile_manual_overrides", src_settings.get("profile_manual_overrides", [])))
    original_manual_overrides = set(manual_overrides)

    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _s(row.get("setting_key"))
        if not key:
            continue
        new_val = row.get("suggested_value")
        old_val = merged.get(key, src_settings.get(key))
        if old_val == new_val:
            continue
        merged[key] = new_val
        changed = True
        if preset_mode == "preset_managed" and key in PROFILE_MANUAL_OVERRIDE_ALLOWLIST:
            manual_overrides.add(key)
        applied.append(
            {
                "setting_key": key,
                "old_value": _normalize_scalar(old_val),
                "new_value": _normalize_scalar(new_val),
                "confidence": round(_clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
                "reason": _s(row.get("reason"))[:220],
            }
        )

    if manual_overrides != original_manual_overrides:
        merged["profile_manual_overrides"] = sorted(manual_overrides)
        changed = True

    if changed:
        clean = sanitize_settings(merged)
        atomic_write_json(settings_path, clean)

    persisted_settings = sanitize_settings(
        read_settings_file(settings_path, module_name="openai_tuning_verify") or {}
    )
    persisted_verified = 0
    for row in applied:
        if not isinstance(row, dict):
            continue
        key = _s(row.get("setting_key"))
        if not key:
            continue
        persisted_val = _normalize_scalar(persisted_settings.get(key))
        row["persisted_value"] = persisted_val
        row["persisted_match"] = bool(persisted_val == _normalize_scalar(row.get("new_value")))
        if bool(row.get("persisted_match", False)):
            persisted_verified += 1

    return {
        "applied": applied,
        "applied_count": int(len(applied)),
        "persisted_verified_count": int(persisted_verified),
        "settings_path": str(settings_path),
        "changed": bool(changed),
    }


def _apply_validated_tuning(
    *,
    base_dir: str,
    settings: Dict[str, Any],
    validated_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Backward-compatible alias for older call sites.
    return apply_validated_tuning_suggestions(
        base_dir=base_dir,
        settings=settings,
        validated_rows=validated_rows,
    )


def run_openai_nightly_trade_review(
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

    report_path = os.path.join(openai_dir, "nightly_trade_review.json")
    status_path = os.path.join(openai_dir, "nightly_trade_review_status.json")

    enabled = bool(cfg.get("openai_nightly_review_enabled", False))
    write_report = bool(cfg.get("openai_nightly_review_write_report_enabled", True))
    apply_enabled = bool(cfg.get("openai_nightly_review_apply_tuning_enabled", False))
    lookback_days = max(1, min(30, int(_f(cfg.get("openai_nightly_review_lookback_days", 7), 7))))
    max_events = max(200, min(50000, int(_f(cfg.get("openai_nightly_review_max_events", 5000), 5000))))
    timeout_s = _clamp(_f(cfg.get("openai_nightly_review_timeout_s", 12.0), 12.0), 1.0, 60.0)
    model = _s(cfg.get("openai_nightly_review_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"

    pre_status = {
        "ts": int(now_i),
        "status": "running" if enabled else "disabled",
        "enabled": bool(enabled),
        "running": bool(enabled),
        "summary": "Nightly review in progress" if enabled else "Nightly review disabled",
        "overall_assessment": "",
        "last_attempt_date_local": str(date_local),
        "completed_date_local": "",
        "lookback_days": int(lookback_days),
        "max_events": int(max_events),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "write_report_enabled": bool(write_report),
        "apply_tuning_enabled": bool(apply_enabled),
        "report_path": str(report_path),
        "report_written": False,
        "tuning_suggestions_count": 0,
        "applied_tuning_count": 0,
        "error": "",
    }
    atomic_write_json(status_path, pre_status)

    packet = build_nightly_review_packet(hub_dir=hub_dir, settings=cfg, now_ts_value=now_i)
    result = request_openai_nightly_trade_review(settings=cfg, base_dir=base_dir, review_packet=packet)
    review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}

    applied_details: Dict[str, Any] = {
        "applied": [],
        "applied_count": 0,
        "persisted_verified_count": 0,
        "settings_path": "",
        "changed": False,
    }
    validated = {"validated": [], "skipped": [], "allowlist_size": int(len(_LOW_RISK_TUNING_RULES))}

    if bool(result.get("active", False)) and review:
        suggestions = review.get("tuning_suggestions", []) if isinstance(review.get("tuning_suggestions", []), list) else []
        validated = validate_low_risk_tuning_suggestions(settings=cfg, suggestions=suggestions, min_confidence=0.80)
        if apply_enabled:
            applied_details = apply_validated_tuning_suggestions(
                base_dir=base_dir,
                settings=cfg,
                validated_rows=validated.get("validated", []) if isinstance(validated.get("validated", []), list) else [],
            )

    report_payload: Dict[str, Any] = {
        "ts": int(now_i),
        "date_local": str(date_local),
        "status": str(result.get("status", "") or ""),
        "enabled": bool(enabled),
        "summary": _s((review.get("summary", "") if review else result.get("summary", "")))[:260],
        "overall_assessment": _s(review.get("overall_assessment", "")).lower() if review else "",
        "market_reviews": [dict(row) for row in list(review.get("market_reviews", []) or [])[:3] if isinstance(row, dict)],
        "tuning_suggestions": [dict(row) for row in list(review.get("tuning_suggestions", []) or [])[:24] if isinstance(row, dict)],
        "risk_flags": [_s(x)[:120] for x in list(review.get("risk_flags", []) or [])[:12] if _s(x)] if review else [],
        "next_day_guidance": [_s(x)[:180] for x in list(review.get("next_day_guidance", []) or [])[:12] if _s(x)] if review else [],
        "meta": {
            "model": str(model),
            "latency_ms": int(max(0, _f(result.get("latency_ms", 0), 0.0))),
            "lookback_days": int(lookback_days),
            "max_events": int(max_events),
            "timeout_s": float(timeout_s),
            "mode": str(packet.get("mode", "")),
            "write_report_enabled": bool(write_report),
            "apply_tuning_enabled": bool(apply_enabled),
            "response_status": str(result.get("status", "") or ""),
            "response_error": _s(result.get("error", ""))[:180],
            "tuning_validation": {
                "allowlist_size": int(validated.get("allowlist_size", 0) or 0),
                "validated_count": int(len([r for r in list(validated.get("validated", []) or []) if isinstance(r, dict)])),
                "skipped_count": int(len([r for r in list(validated.get("skipped", []) or []) if isinstance(r, dict)])),
            },
            "applied_tuning": {
                "applied_count": int(applied_details.get("applied_count", 0) or 0),
                "persisted_verified_count": int(applied_details.get("persisted_verified_count", 0) or 0),
                "changed": bool(applied_details.get("changed", False)),
                "settings_path": str(applied_details.get("settings_path", "") or ""),
                "applied": [
                    dict(row)
                    for row in list(applied_details.get("applied", []) or [])[:16]
                    if isinstance(row, dict)
                ],
            },
        },
        # keep payload compact but include enough context for explainability/debugging
        "input_packet": {
            "lookback_days": int(packet.get("lookback_days", lookback_days) or lookback_days),
            "mode": str(packet.get("mode", "")),
            "settings_profile": str(packet.get("settings_profile", "")),
            "portfolio_context": dict(packet.get("portfolio_context", {}) or {}) if isinstance(packet.get("portfolio_context", {}), dict) else {},
            "market_metrics": dict(packet.get("market_metrics", {}) or {}) if isinstance(packet.get("market_metrics", {}), dict) else {},
            "cross_market_allocator": dict(packet.get("cross_market_allocator", {}) or {}) if isinstance(packet.get("cross_market_allocator", {}), dict) else {},
        },
    }

    if write_report:
        atomic_write_json(report_path, report_payload)

    final_status = {
        "ts": int(now_i),
        "enabled": bool(enabled),
        "running": False,
        "status": str(result.get("status", "disabled") or "disabled"),
        "summary": _s((review.get("summary", "") if review else result.get("summary", "")))[:260],
        "overall_assessment": _s(review.get("overall_assessment", "")).lower() if review else "",
        "last_attempt_date_local": str(date_local),
        "completed_date_local": str(date_local),
        "lookback_days": int(lookback_days),
        "max_events": int(max_events),
        "model": str(model),
        "timeout_s": float(timeout_s),
        "write_report_enabled": bool(write_report),
        "apply_tuning_enabled": bool(apply_enabled),
        "report_path": str(report_path),
        "report_written": bool(write_report),
        "tuning_suggestions_count": int(len([r for r in list(review.get("tuning_suggestions", []) or []) if isinstance(r, dict)]) if review else 0),
        "applied_tuning_count": int(applied_details.get("applied_count", 0) or 0),
        "persisted_verified_count": int(applied_details.get("persisted_verified_count", 0) or 0),
        "risk_flags_count": int(len([x for x in list(review.get("risk_flags", []) or []) if _s(x)]) if review else 0),
        "next_day_guidance": [_s(x)[:180] for x in list(review.get("next_day_guidance", []) or [])[:6] if _s(x)] if review else [],
        "market_reviews": [
            {
                "market": _s(row.get("market", "")).lower(),
                "assessment": _s(row.get("assessment", "")).lower(),
                "recommended_actions": [_s(x)[:160] for x in list(row.get("recommended_actions", []) or [])[:4] if _s(x)],
            }
            for row in list(review.get("market_reviews", []) or [])[:3]
            if isinstance(row, dict)
        ] if review else [],
        "error": _s(result.get("error", ""))[:180],
        "latency_ms": int(max(0, _f(result.get("latency_ms", 0), 0.0))),
        "applied_tuning": [
            {
                "setting_key": _s(row.get("setting_key", "")),
                "old_value": _normalize_scalar(row.get("old_value")),
                "new_value": _normalize_scalar(row.get("new_value")),
                "confidence": round(_clamp(_f(row.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
                "reason": _s(row.get("reason", ""))[:180],
            }
            for row in list(applied_details.get("applied", []) or [])[:12]
            if isinstance(row, dict)
        ],
    }

    atomic_write_json(status_path, final_status)
    return final_status
