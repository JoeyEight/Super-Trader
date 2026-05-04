from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List

from app.scanner_quality import effective_reject_pressure

_TRANSIENT_INCIDENT_TTL_S: Dict[str, int] = {
    "ui_market_panel_desync": 300,
    "market_panel_refresh_failed": 300,
    "runner_watchdog_restart": 600,
    "runner_child_exit": 600,
    "runner_script_path_changed": 600,
    "runner_script_hot_reload": 600,
    "runner_forced_shutdown": 600,
    "runner_sleep_resume_detected": 600,
    "runner_child_start_failed": 900,
    "runner_child_crash_loop": 900,
    "runner_missing_script": 900,
    "stocks_snapshot_failed": 900,
    "forex_snapshot_failed": 900,
    "stocks_thinker_error": 900,
    "stocks_thinker_failed": 900,
    "stocks_trader_error": 900,
    "stocks_trader_failed": 900,
    "forex_thinker_error": 900,
    "forex_thinker_failed": 900,
    "forex_trader_error": 900,
    "forex_trader_failed": 900,
    "market_trends_update_failed": 900,
}

_STARTUP_CHECK_INFO_WARNINGS = {
    "stale_pid_file_removed",
}

_OPENAI_ERROR_STATUSES = {
    "timeout",
    "request_error",
    "http_error",
    "invalid_json",
    "empty_response",
    "malformed_response",
    "schema_validation_failed",
    "runner_exception",
}

# Transient/unavailable OpenAI statuses should not page the user when the app
# already falls back to local logic. Reserve warning-level severity for hard
# integration failures that likely need code/config intervention.
_OPENAI_HARD_ERROR_STATUSES = {
    "malformed_response",
    "schema_validation_failed",
    "runner_exception",
}

_OPENAI_INCIDENT_RUNTIME_KEY = {
    "openai_nightly_review_status": "openai_nightly_review",
    "openai_position_review_status": "openai_position_review",
    "openai_capital_planner_status": "openai_capital_planner",
    "openai_root_cause_status": "openai_root_cause_analysis",
    "openai_strategy_optimizer_status": "openai_strategy_optimizer",
    "openai_explanations_status": "openai_explanations",
    "openai_market_context_status": "openai_market_context",
    "openai_postmortem_status": "openai_postmortem_analysis",
}

_OPENAI_INFO_DISABLED_STATUSES = {
    "",
    "disabled",
    "live_disabled",
    "paper_disabled",
    "idle",
    "not_requested",
}


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


def _sev(v: str) -> str:
    txt = str(v or "").strip().lower()
    if txt in {"critical", "error", "high"}:
        return "critical"
    if txt in {"warn", "warning", "medium"}:
        return "warning"
    return "info"


def _openai_status_is_error(status: Any) -> bool:
    return str(status or "").strip().lower() in _OPENAI_ERROR_STATUSES


def _openai_status_is_hard_error(status: Any) -> bool:
    return str(status or "").strip().lower() in _OPENAI_HARD_ERROR_STATUSES


def _openai_incident_is_actionable(row: Dict[str, Any], runtime_state: Dict[str, Any]) -> bool:
    evt = str(row.get("event", "") or "").strip().lower()
    if evt in _OPENAI_INCIDENT_RUNTIME_KEY:
        key = _OPENAI_INCIDENT_RUNTIME_KEY[evt]
        payload = runtime_state.get(key, {}) if isinstance(runtime_state.get(key, {}), dict) else {}
        if not payload:
            return False
        status = str(payload.get("status", "") or "").strip().lower()
        running = bool(payload.get("running", False))
        return _openai_status_is_hard_error(status) and (not running)
    if evt.startswith("openai_"):
        return False
    return True


def _severity_rank(v: str) -> int:
    sev = _sev(v)
    if sev == "critical":
        return 3
    if sev == "warning":
        return 2
    if sev == "ok":
        return 0
    return 1


def _openai_should_emit_status_row(
    *,
    status: Any,
    running: bool,
    change_applied: bool = False,
) -> bool:
    st = str(status or "").strip().lower()
    if bool(running):
        return False
    if _openai_status_is_hard_error(st):
        return True
    if st in _OPENAI_INFO_DISABLED_STATUSES:
        return False
    if st == "ok" and bool(change_applied):
        return True
    return False


def _row_is_actionable_or_issue(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    severity = _sev(str(row.get("severity", "info") or "info"))
    if severity in {"critical", "warning"}:
        return True
    action = row.get("action", {}) if isinstance(row.get("action", {}), dict) else {}
    if str(action.get("setting_key", "") or "").strip():
        return True
    return bool(row.get("change_applied", False))


def _market_from_incident(row: Dict[str, Any]) -> str:
    details = row.get("details", {}) if isinstance(row.get("details", {}), dict) else {}
    market = str(details.get("market", "") or "").strip().lower()
    if market in {"stocks", "forex", "crypto"}:
        return market
    evt = str(row.get("event", "") or "").strip().lower()
    if "stock" in evt:
        return "stocks"
    if "forex" in evt:
        return "forex"
    if "kucoin" in evt or "crypto" in evt:
        return "crypto"
    return "global"


def _runtime_now_ts(runtime_state: Dict[str, Any]) -> int:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    try:
        ts = int(rs.get("ts", 0) or 0)
    except Exception:
        ts = 0
    return ts if ts > 0 else int(time.time())


def _incident_is_recent(row: Dict[str, Any], runtime_state: Dict[str, Any], ttl_s: int) -> bool:
    try:
        ts = int(float(row.get("ts", 0) or 0))
    except Exception:
        ts = 0
    if ts <= 0:
        return False
    return (_runtime_now_ts(runtime_state) - ts) <= max(30, int(ttl_s or 0))


def _runtime_alert_reasons(runtime_state: Dict[str, Any]) -> set[str]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    alerts = rs.get("alerts", {}) if isinstance(rs.get("alerts", {}), dict) else {}
    out: set[str] = set()
    for row in list(alerts.get("reasons", []) or []):
        key = str(row or "").strip().lower()
        if key:
            out.add(key)
    return out


def _active_cadence_markets(runtime_state: Dict[str, Any]) -> set[str]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    out: set[str] = set()
    scan_cadence = rs.get("scan_cadence", {}) if isinstance(rs.get("scan_cadence", {}), dict) else {}
    for row in list(scan_cadence.get("active", []) or []):
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "") or "").strip().lower()
        if market in {"stocks", "forex", "crypto"}:
            out.add(market)
    return out


def _active_drift_markets(runtime_state: Dict[str, Any]) -> set[str]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    out: set[str] = set()
    scan_drift = rs.get("scan_drift", {}) if isinstance(rs.get("scan_drift", {}), dict) else {}
    for row in list(scan_drift.get("active", []) or []):
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "") or "").strip().lower()
        if market in {"stocks", "forex", "crypto"}:
            out.add(market)
    return out


def _startup_checks_active(runtime_state: Dict[str, Any]) -> bool:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    checks = rs.get("checks", {}) if isinstance(rs.get("checks", {}), dict) else {}
    if not bool(checks.get("ok", False)):
        return True
    warnings = list(checks.get("warnings", []) or []) if isinstance(checks.get("warnings", []), list) else []
    errors = list(checks.get("errors", []) or []) if isinstance(checks.get("errors", []), list) else []
    warnings = [str(row or "").strip().lower() for row in warnings if str(row or "").strip()]
    warnings = [row for row in warnings if row not in _STARTUP_CHECK_INFO_WARNINGS]
    return bool(warnings or errors)


def _market_loop_issue_active(runtime_state: Dict[str, Any]) -> bool:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    if "market_loop_stale" in _runtime_alert_reasons(rs):
        return True
    alerts = rs.get("alerts", {}) if isinstance(rs.get("alerts", {}), dict) else {}
    metrics = alerts.get("metrics", {}) if isinstance(alerts.get("metrics", {}), dict) else {}
    return bool(metrics.get("market_loop_stale", False))


def _runner_child_pid(runtime_state: Dict[str, Any], child: str) -> int:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    runner = rs.get("runner", {}) if isinstance(rs.get("runner", {}), dict) else {}
    children = runner.get("children", {}) if isinstance(runner.get("children", {}), dict) else {}
    try:
        pid = int(children.get(child, 0) or 0)
    except Exception:
        pid = 0
    return pid if pid > 0 else 0


def _autopilot_issue_active(runtime_state: Dict[str, Any]) -> bool:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    autopilot = rs.get("autopilot", {}) if isinstance(rs.get("autopilot", {}), dict) else {}
    if not autopilot:
        return False
    if bool(autopilot.get("issue_open", False)):
        return True
    if bool(autopilot.get("api_unstable", False)):
        return True
    if not bool(autopilot.get("markets_healthy", True)):
        return True
    try:
        status_ts = int(autopilot.get("ts", 0) or 0)
    except Exception:
        status_ts = 0
    if status_ts > 0 and (_runtime_now_ts(rs) - status_ts) > 240:
        return True
    return False


def _runner_restart_incident_active(row: Dict[str, Any], runtime_state: Dict[str, Any]) -> bool:
    details = row.get("details", {}) if isinstance(row.get("details", {}), dict) else {}
    child = str(details.get("child", "") or "").strip().lower()
    ttl_s = int(_TRANSIENT_INCIDENT_TTL_S.get(str(row.get("event", "") or "").strip().lower(), 0) or 0)
    if not child:
        return _incident_is_recent(row, runtime_state, ttl_s)
    child_pid = _runner_child_pid(runtime_state, child)
    if child == "autopilot":
        return child_pid <= 0 or _autopilot_issue_active(runtime_state)
    if child == "markets":
        return child_pid <= 0 or _market_loop_issue_active(runtime_state)
    if child in {"thinker", "trader"}:
        return child_pid <= 0
    return _incident_is_recent(row, runtime_state, ttl_s)


def _incident_is_active(row: Dict[str, Any], runtime_state: Dict[str, Any]) -> bool:
    evt = str(row.get("event", "") or "").strip().lower()
    market = _market_from_incident(row)
    if evt == "scanner_cadence_drift":
        return market in _active_cadence_markets(runtime_state)
    if evt == "scanner_reject_spike":
        return market in _active_drift_markets(runtime_state)
    if evt == "runner_startup_check":
        return _startup_checks_active(runtime_state)
    if evt in {"runner_watchdog_restart", "runner_child_exit", "runner_child_crash_loop", "runner_child_start_failed"}:
        return _runner_restart_incident_active(row, runtime_state)
    if evt in {"runner_market_loop_status_stale", "runner_market_loop_restart"}:
        return _market_loop_issue_active(runtime_state)
    ttl_s = int(_TRANSIENT_INCIDENT_TTL_S.get(evt, 0) or 0)
    if ttl_s > 0:
        return _incident_is_recent(row, runtime_state, ttl_s)
    return True


def _dedupe_notification_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("market", "global") or "global").strip().lower(),
            str(row.get("source", "") or "").strip().lower(),
            _sev(str(row.get("severity", "info") or "info")),
            str(row.get("title", "") or "").strip(),
            str(row.get("message", "") or "").strip(),
        )
        prev = keep.get(key)
        if prev is None:
            keep[key] = row
            continue
        prev_key = (int(prev.get("ts", 0) or 0), _severity_rank(str(prev.get("severity", "info") or "info")))
        next_key = (int(row.get("ts", 0) or 0), _severity_rank(str(row.get("severity", "info") or "info")))
        if next_key >= prev_key:
            keep[key] = row
    return list(keep.values())


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _market_score_threshold_key(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "stock_score_threshold"
    if m == "forex":
        return "forex_score_threshold"
    return ""


def _market_scan_interval_key(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "market_bg_stocks_interval_s"
    if m == "forex":
        return "market_bg_forex_interval_s"
    return ""


def _market_trade_size_key(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "stock_trade_notional_usd"
    if m == "forex":
        return "forex_trade_units"
    return ""


def _market_min_samples_key(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "stock_min_samples_live_guarded"
    if m == "forex":
        return "forex_min_samples_live_guarded"
    return ""


def _market_max_open_positions_key(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "stock_max_open_positions"
    if m == "forex":
        return "forex_max_open_positions"
    if m == "crypto":
        return "crypto_max_open_positions"
    return ""


def _market_label(market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return "stocks"
    if m == "forex":
        return "forex"
    if m == "crypto":
        return "crypto"
    return "market"


def _notification_action(
    action_id: str,
    label: str,
    kind: str,
    setting_key: str,
    *,
    reason: str = "",
    minimum: Any | None = None,
    maximum: Any | None = None,
    step: Any | None = None,
    factor: Any | None = None,
    precision: Any | None = None,
    value: Any | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(action_id or "").strip(),
        "label": str(label or "").strip() or "Auto-adjust setting",
        "kind": str(kind or "").strip().lower(),
        "setting_key": str(setting_key or "").strip(),
    }
    if reason:
        payload["reason"] = str(reason or "").strip()
    if minimum is not None:
        payload["min"] = minimum
    if maximum is not None:
        payload["max"] = maximum
    if step is not None:
        payload["step"] = step
    if factor is not None:
        payload["factor"] = factor
    if precision is not None:
        payload["precision"] = precision
    if value is not None:
        payload["value"] = value
    return payload


def _dominant_reject_market(runtime_state: Dict[str, Any]) -> str:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    trends = rs.get("market_trends", {}) if isinstance(rs.get("market_trends", {}), dict) else {}
    best_market = ""
    best_pressure = -1.0
    for market in ("stocks", "forex"):
        row = trends.get(market, {}) if isinstance(trends.get(market, {}), dict) else {}
        quality = row.get("quality_aggregates", {}) if isinstance(row.get("quality_aggregates", {}), dict) else {}
        reject_raw = _f(quality.get("reject_rate_raw_pct", quality.get("reject_rate_pct", 0.0)), 0.0)
        reject = effective_reject_pressure(
            reject_raw,
            dominant_reason=quality.get("dominant_reason", ""),
            dominant_ratio_pct=quality.get("reject_dominant_ratio_pct", 0.0),
            leaders_total=quality.get("leaders_total", 0),
            scores_total=quality.get("scores_total", 0),
        )
        if reject > best_pressure:
            best_market = market
            best_pressure = reject
    return best_market


def _dominant_cadence_market(runtime_state: Dict[str, Any]) -> str:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    scan_cadence = rs.get("scan_cadence", {}) if isinstance(rs.get("scan_cadence", {}), dict) else {}
    active = scan_cadence.get("active", []) if isinstance(scan_cadence.get("active", []), list) else []
    best_market = ""
    best_late_pct = -1.0
    for row in active:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "") or "").strip().lower()
        if market not in {"stocks", "forex"}:
            continue
        late_pct = _f(row.get("late_pct", 0.0), 0.0)
        if late_pct > best_late_pct:
            best_late_pct = late_pct
            best_market = market
    return best_market


def _top_exposure_market(runtime_state: Dict[str, Any]) -> str:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    exposure = rs.get("exposure_map", {}) if isinstance(rs.get("exposure_map", {}), dict) else {}
    top = exposure.get("top_positions", []) if isinstance(exposure.get("top_positions", []), list) else []
    if not top or not isinstance(top[0], dict):
        return ""
    market = str(top[0].get("market", "") or "").strip().lower()
    return market if market in {"stocks", "forex", "crypto"} else ""


def _action_for_runtime_reason(reason: str, runtime_state: Dict[str, Any], hint: str = "") -> Dict[str, Any]:
    key = str(reason or "").strip().lower()
    if not key:
        return {}

    if key in {"scan_reject_pressure", "scanner_reject_spike"}:
        market = _dominant_reject_market(runtime_state) or "stocks"
        setting_key = _market_score_threshold_key(market)
        if not setting_key:
            return {}
        return _notification_action(
            f"{key}_{market}_threshold",
            f"Loosen {str(_market_label(market)).title()} score threshold",
            "float_scale_down",
            setting_key,
            reason=hint or "Lower threshold slightly to reduce reject pressure.",
            minimum=0.0,
            factor=0.92,
            precision=4,
        )

    if key == "cadence_drift_pressure":
        market = _dominant_cadence_market(runtime_state) or "stocks"
        setting_key = _market_scan_interval_key(market)
        if not setting_key:
            return {}
        return _notification_action(
            f"{key}_{market}_interval",
            f"Reduce {str(_market_label(market)).title()} scan pressure",
            "float_scale_up",
            setting_key,
            reason=hint or "Increase scanner interval to reduce cadence drift pressure.",
            minimum=0.0,
            maximum=300.0,
            factor=1.15,
            precision=2,
        )

    if key == "exposure_concentration":
        market = _top_exposure_market(runtime_state)
        setting_key = _market_max_open_positions_key(market)
        if not setting_key:
            return {}
        minimum = 1 if market == "crypto" else 0
        return _notification_action(
            f"{key}_{market}_max_open",
            f"Tighten {str(_market_label(market)).title()} concentration cap",
            "int_step_down",
            setting_key,
            reason=hint or "Reduce max concurrent positions for the concentrated market.",
            minimum=minimum,
            step=1,
        )

    if key == "execution_temporarily_disabled":
        return _notification_action(
            f"{key}_cooldown",
            "Increase broker failure cooldown",
            "int_scale_up",
            "broker_failure_disable_cooldown_s",
            reason=hint or "Increase cooldown before retrying execution after repeated failures.",
            minimum=60,
            maximum=86400,
            factor=1.25,
        )

    if key == "api_unstable":
        return _notification_action(
            f"{key}_retry_cap",
            "Increase API retry-after cap",
            "float_scale_up",
            "broker_order_retry_after_cap_s",
            reason=hint or "Increase retry-after cap to reduce request thrash while unstable.",
            minimum=1.0,
            maximum=3600.0,
            factor=1.2,
            precision=1,
        )

    return {}


def _action_for_market_trend_row(market: str, title: str, message: str) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    t = str(title or "").strip().lower()
    msg = str(message or "").strip()
    if m not in {"stocks", "forex"}:
        return {}

    if t == "high scanner rejection pressure":
        setting_key = _market_score_threshold_key(m)
        if not setting_key:
            return {}
        return _notification_action(
            f"{m}_reject_pressure_threshold",
            f"Loosen {str(_market_label(m)).title()} score threshold",
            "float_scale_down",
            setting_key,
            reason=msg or "Lower threshold slightly to improve candidate flow.",
            minimum=0.0,
            factor=0.92,
            precision=4,
        )

    if t == "data reliability degraded":
        setting_key = _market_scan_interval_key(m)
        if not setting_key:
            return {}
        return _notification_action(
            f"{m}_reliability_interval",
            f"Slow {str(_market_label(m)).title()} scan cadence",
            "float_scale_up",
            setting_key,
            reason=msg or "Increase scanner interval to reduce data provider pressure.",
            minimum=0.0,
            maximum=300.0,
            factor=1.12,
            precision=2,
        )

    return {}


def _action_for_execution_gate_row(market: str, message: str) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    msg = str(message or "").strip()
    msg_l = msg.lower()
    if m not in {"stocks", "forex"}:
        return {}

    if "risk cap:" in msg_l:
        size_key = _market_trade_size_key(m)
        if not size_key:
            return {}
        if m == "forex":
            return _notification_action(
                f"{m}_risk_cap_trade_size",
                "Reduce forex trade units",
                "int_scale_down",
                size_key,
                reason=msg,
                minimum=1,
                factor=0.8,
            )
        return _notification_action(
            f"{m}_risk_cap_trade_size",
            "Reduce stock trade notional",
            "float_scale_down",
            size_key,
            reason=msg,
            minimum=1.0,
            factor=0.85,
            precision=2,
        )

    if "max open positions reached" in msg_l:
        key = _market_max_open_positions_key(m)
        if not key:
            return {}
        return _notification_action(
            f"{m}_max_open_positions_up",
            f"Increase {str(_market_label(m)).title()} max open positions",
            "int_scale_up",
            key,
            reason=msg,
            minimum=0,
            maximum=500,
            factor=1.2,
        )

    if "reject-pressure gate active" in msg_l:
        threshold_key = _market_score_threshold_key(m)
        if not threshold_key:
            return {}
        return _notification_action(
            f"{m}_execution_gate_threshold",
            f"Loosen {str(_market_label(m)).title()} score threshold",
            "float_scale_down",
            threshold_key,
            reason=msg,
            minimum=0.0,
            factor=0.92,
            precision=4,
        )

    if "calibration sample gate" in msg_l or "calibration history insufficient" in msg_l:
        sample_key = _market_min_samples_key(m)
        if not sample_key:
            return {}
        return _notification_action(
            f"{m}_calibration_samples",
            f"Reduce {str(_market_label(m)).title()} minimum calibration samples",
            "int_step_down",
            sample_key,
            reason=msg,
            minimum=0,
            step=1,
        )

    return {}


def _action_for_incident_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    evt = str(row.get("event", "") or "").strip().lower()
    market = _market_from_incident(row)
    details = row.get("details", {}) if isinstance(row.get("details", {}), dict) else {}
    message = str(row.get("msg", "") or "").strip()
    if evt == "scanner_cadence_drift":
        setting_key = _market_scan_interval_key(market)
        if not setting_key:
            return {}
        late_pct = _f(details.get("late_pct", 0.0), 0.0)
        factor = 1.15 if late_pct < 150.0 else 1.25
        return _notification_action(
            f"{market}_incident_cadence",
            f"Slow {str(_market_label(market)).title()} scan cadence",
            "float_scale_up",
            setting_key,
            reason=message,
            minimum=0.0,
            maximum=300.0,
            factor=factor,
            precision=2,
        )
    if evt == "scanner_reject_spike":
        setting_key = _market_score_threshold_key(market)
        if not setting_key:
            return {}
        return _notification_action(
            f"{market}_incident_reject",
            f"Loosen {str(_market_label(market)).title()} score threshold",
            "float_scale_down",
            setting_key,
            reason=message,
            minimum=0.0,
            factor=0.9,
            precision=4,
        )
    return {}


def _automation_policy_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("automation_policy", {}) if isinstance(rs.get("automation_policy", {}), dict) else {}
    rows: List[Dict[str, Any]] = []
    for market in ("crypto", "stocks", "forex"):
        row = payload.get(market, {}) if isinstance(payload.get(market, {}), dict) else {}
        summary = str(row.get("summary", "") or "").strip()
        if not summary:
            continue
        trust_score = _f(row.get("runtime_trust_score", 0.0), 0.0)
        compliance_status = str(row.get("compliance_status", "") or "").strip()
        severity = "info"
        if trust_score > 0.0 and trust_score < 35.0:
            severity = "critical"
        title = f"{str(_market_label(market)).title()} automation policy"
        message_bits = [summary]
        if compliance_status and compliance_status.lower() not in summary.lower():
            message_bits.append(compliance_status)
        message = " | ".join([part for part in message_bits if str(part or "").strip()][:2])
        rows.append(
            {
                "id": f"{market}_automation_policy_{ts_now}",
                "ts": int(ts_now),
                "severity": severity,
                "market": market,
                "source": "automation_policy",
                "title": title,
                "message": message[:220],
            }
        )
    return rows


def _cross_market_opportunity_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("cross_market_opportunity", {}) if isinstance(rs.get("cross_market_opportunity", {}), dict) else {}
    if not payload or not bool(payload.get("active", False)):
        return []
    summary = str(payload.get("summary", "") or "").strip()
    best_market = str(payload.get("best_market", "") or "").strip().lower()
    decisions = payload.get("decisions", {}) if isinstance(payload.get("decisions", {}), dict) else {}
    deprioritized = payload.get("deprioritized_markets", []) if isinstance(payload.get("deprioritized_markets", []), list) else []
    severity = "info"
    if not summary and best_market:
        summary = f"Best current opportunity: {str(_market_label(best_market)).title()}"
    if not summary:
        return []
    best_label = str(_market_label(best_market)).title() if best_market else "Portfolio"
    rows: List[Dict[str, Any]] = [
        {
            "id": f"cross_market_opportunity_{ts_now}",
            "ts": int(ts_now),
            "severity": severity,
            "market": "global",
            "source": "opportunity_allocator",
            "title": f"{best_label} opportunity priority",
            "message": summary[:220],
        }
    ]
    ai = payload.get("openai_decision", {}) if isinstance(payload.get("openai_decision", {}), dict) else {}
    ai_enabled = bool(ai.get("enabled", False))
    ai_active = bool(ai.get("active", False))
    ai_status = str(ai.get("status", "") or "").strip().lower()
    ai_summary = str(ai.get("summary", "") or "").strip()
    if ai_enabled and ai_status and not ai_summary and ai_status not in {"ok", "not_requested"}:
        ai_summary = "AI portfolio decision unavailable; local allocator is active."
    ai_applied = bool(ai_active and bool(ai.get("applied", False)))
    if ai_enabled and ai_summary and (_openai_status_is_hard_error(ai_status) or ai_applied):
        ai_severity = "info"
        if ai_applied:
            ai_title = "AI portfolio decision applied"
        elif ai_active:
            ai_title = "AI portfolio decision advisory"
        else:
            ai_title = "AI portfolio decision fallback"
            if _openai_status_is_hard_error(ai_status):
                ai_severity = "warning"
        rows.append(
            {
                "id": f"cross_market_openai_{ts_now}",
                "ts": int(ts_now),
                "severity": ai_severity,
                "market": "global",
                "source": "opportunity_allocator",
                "title": ai_title,
                "message": ai_summary[:220],
                "change_applied": bool(ai_applied),
            }
        )
    ai_position_actions = ai.get("position_actions", []) if isinstance(ai.get("position_actions", []), list) else []
    if ai_enabled and ai_position_actions and ai_applied:
        for idx, prow in enumerate(ai_position_actions[:3]):
            if not isinstance(prow, dict):
                continue
            action = str(prow.get("action", "") or "").strip().lower()
            symbol = str(prow.get("symbol", "") or "").strip().upper()
            market = str(prow.get("market", "") or "").strip().lower()
            reason = str(prow.get("reason", "") or "").strip()
            if not symbol or not action:
                continue
            sev = "info"
            title = f"AI position review: {action.upper()} {symbol}"
            if action == "block_add":
                title = f"AI position review: BLOCK ADD {symbol}"
            msg_bits: List[str] = []
            if reason:
                msg_bits.append(reason[:170])
            conf = _f(prow.get("confidence", 0.0), 0.0)
            if conf > 0.0:
                msg_bits.append(f"confidence {conf:.2f}")
            rows.append(
                {
                    "id": f"cross_market_openai_position_{idx}_{ts_now}",
                    "ts": int(ts_now),
                    "severity": sev,
                    "market": market or "global",
                    "source": "opportunity_allocator",
                    "title": title,
                    "message": " | ".join(msg_bits)[:220] if msg_bits else f"AI recommends {action} for {symbol}.",
                    "change_applied": True,
                }
            )
    for row in deprioritized[:2]:
        if not isinstance(row, dict):
            continue
        mk = str(row.get("market", "") or "").strip().lower()
        decision = str(row.get("decision", "") or "").strip().lower()
        message = str(row.get("summary", "") or "").strip()
        if not mk or not message:
            continue
        sev = "info"
        rows.append(
            {
                "id": f"cross_market_{mk}_{ts_now}",
                "ts": int(ts_now),
                "severity": sev,
                "market": mk,
                "source": "opportunity_allocator",
                "title": f"{str(_market_label(mk)).title()} candidate {decision or 'deprioritized'}",
                "message": message[:220],
            }
        )
    # Include decisions map only when there is no explicit deprioritized row.
    if not deprioritized and decisions:
        top_decision = str(decisions.get(best_market, "") or "").strip().lower() if best_market else ""
        if top_decision:
            rows[0]["message"] = f"{rows[0]['message']} | Decision: {top_decision}"
    return rows


def _openai_nightly_review_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_nightly_review", {}) if isinstance(rs.get("openai_nightly_review", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    applied_count = int(max(0.0, _f(payload.get("applied_tuning_count", 0), 0.0)))
    should_emit = _openai_should_emit_status_row(
        status=status,
        running=running,
        change_applied=(applied_count > 0),
    )
    if not should_emit:
        return []

    rows: List[Dict[str, Any]] = []
    if _openai_status_is_hard_error(status):
        message = summary or "Nightly AI trade review failed; local logic remains active."
        rows.append(
            {
                "id": f"openai_nightly_review_{ts_now}",
                "ts": int(ts_now),
                "severity": "warning",
                "market": "global",
                "source": "openai_nightly_review",
                "title": "AI nightly trade review fallback",
                "message": message[:220],
            }
        )
        return rows

    # Status OK and changes applied.
    rows.append(
        {
            "id": f"openai_nightly_review_tuning_{ts_now}",
            "ts": int(ts_now),
            "severity": "info",
            "market": "global",
            "source": "openai_nightly_review",
            "title": "AI nightly tuning applied",
            "message": (
                f"{applied_count} low-risk setting change(s) were auto-applied by nightly review"
                + (
                    f"; verified {int(max(0.0, _f(payload.get('persisted_verified_count', 0), 0.0)))} persisted."
                    if int(max(0.0, _f(payload.get("persisted_verified_count", 0), 0.0))) > 0
                    else "."
                )
            )[:220],
            "change_applied": True,
        }
    )
    applied_rows = payload.get("applied_tuning", []) if isinstance(payload.get("applied_tuning", []), list) else []
    for idx, row in enumerate(applied_rows[:4]):
        if not isinstance(row, dict):
            continue
        key = str(row.get("setting_key", "") or "").strip()
        old_val = row.get("old_value")
        new_val = row.get("new_value")
        reason = str(row.get("reason", "") or "").strip()
        if not key:
            continue
        msg = f"{key}: {old_val} -> {new_val}"
        if reason:
            msg = f"{msg} | {reason[:120]}"
        rows.append(
            {
                "id": f"openai_nightly_review_tuning_item_{idx}_{ts_now}",
                "ts": int(ts_now),
                "severity": "info",
                "market": "global",
                "source": "openai_nightly_review",
                "title": f"Applied: {key}"[:120],
                "message": msg[:220],
                "change_applied": True,
            }
        )
    return rows


def _openai_position_review_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_position_review", {}) if isinstance(rs.get("openai_position_review", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    if not _openai_should_emit_status_row(status=status, running=running, change_applied=False):
        return []
    if not _openai_status_is_hard_error(status):
        return []
    return [
        {
            "id": f"openai_position_review_{ts_now}",
            "ts": int(ts_now),
            "severity": "warning",
            "market": "global",
            "source": "openai_position_review",
            "title": "AI position review fallback",
            "message": (summary or "AI position review failed; local logic remains active.")[:220],
        }
    ]


def _openai_capital_planner_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_capital_planner", {}) if isinstance(rs.get("openai_capital_planner", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    if not _openai_should_emit_status_row(status=status, running=running, change_applied=False):
        return []
    if not _openai_status_is_hard_error(status):
        return []
    return [
        {
            "id": f"openai_capital_planner_{ts_now}",
            "ts": int(ts_now),
            "severity": "warning",
            "market": "global",
            "source": "openai_capital_planner",
            "title": "AI capital planner fallback",
            "message": (summary or "AI capital planner failed; local allocator remains active.")[:220],
        }
    ]


def _openai_root_cause_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_root_cause_analysis", {}) if isinstance(rs.get("openai_root_cause_analysis", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    if not _openai_should_emit_status_row(status=status, running=running, change_applied=False):
        return []
    if not _openai_status_is_hard_error(status):
        return []
    return [
        {
            "id": f"openai_root_cause_{ts_now}",
            "ts": int(ts_now),
            "severity": "warning",
            "market": "global",
            "source": "openai_root_cause_analysis",
            "title": "AI root-cause analysis fallback",
            "message": (summary or "AI root-cause analysis failed; local diagnostics remain active.")[:220],
        }
    ]


def _openai_strategy_optimizer_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_strategy_optimizer", {}) if isinstance(rs.get("openai_strategy_optimizer", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    applied_count = int(max(0.0, _f(payload.get("applied_tuning_count", 0), 0.0)))
    should_emit = _openai_should_emit_status_row(
        status=status,
        running=running,
        change_applied=(applied_count > 0),
    )
    if not should_emit:
        return []

    rows: List[Dict[str, Any]] = []
    if _openai_status_is_hard_error(status):
        rows.append(
            {
                "id": f"openai_strategy_optimizer_{ts_now}",
                "ts": int(ts_now),
                "severity": "warning",
                "market": "global",
                "source": "openai_strategy_optimizer",
                "title": "AI strategy optimizer fallback",
                "message": (summary or "AI strategy optimizer failed; local strategy remains active.")[:220],
            }
        )
        return rows

    rows.append(
        {
            "id": f"openai_strategy_optimizer_{ts_now}",
            "ts": int(ts_now),
            "severity": "info",
            "market": "global",
            "source": "openai_strategy_optimizer",
            "title": "AI strategy tuning applied",
            "message": (
                f"{applied_count} low-risk strategy setting change(s) were auto-applied"
                + (
                    f"; verified {int(max(0.0, _f(payload.get('persisted_verified_count', 0), 0.0)))} persisted."
                    if int(max(0.0, _f(payload.get("persisted_verified_count", 0), 0.0))) > 0
                    else "."
                )
            )[:220],
            "change_applied": True,
        }
    )
    applied_rows = payload.get("applied_tuning", []) if isinstance(payload.get("applied_tuning", []), list) else []
    for idx, row in enumerate(applied_rows[:4]):
        if not isinstance(row, dict):
            continue
        key = str(row.get("setting_key", "") or "").strip()
        old_val = row.get("old_value")
        new_val = row.get("new_value")
        reason = str(row.get("reason", "") or "").strip()
        if not key:
            continue
        msg = f"{key}: {old_val} -> {new_val}"
        if reason:
            msg = f"{msg} | {reason[:120]}"
        rows.append(
            {
                "id": f"openai_strategy_optimizer_item_{idx}_{ts_now}",
                "ts": int(ts_now),
                "severity": "info",
                "market": "global",
                "source": "openai_strategy_optimizer",
                "title": f"Applied: {key}"[:120],
                "message": msg[:220],
                "change_applied": True,
            }
        )
    return rows


def _openai_market_context_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_market_context", {}) if isinstance(rs.get("openai_market_context", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    if not _openai_should_emit_status_row(status=status, running=running, change_applied=False):
        return []
    if not _openai_status_is_hard_error(status):
        return []
    return [
        {
            "id": f"openai_market_context_{ts_now}",
            "ts": int(ts_now),
            "severity": "warning",
            "market": "global",
            "source": "openai_market_context",
            "title": "AI market context fallback",
            "message": (summary or "AI market context scoring failed; local context logic remains active.")[:220],
        }
    ]


def _openai_postmortem_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_postmortem_analysis", {}) if isinstance(rs.get("openai_postmortem_analysis", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    applied_count = int(max(0.0, _f(payload.get("applied_tuning_count", 0), 0.0)))
    should_emit = _openai_should_emit_status_row(
        status=status,
        running=running,
        change_applied=(applied_count > 0),
    )
    if not should_emit:
        return []
    rows: List[Dict[str, Any]] = []
    if _openai_status_is_hard_error(status):
        rows.append(
            {
                "id": f"openai_postmortem_{ts_now}",
                "ts": int(ts_now),
                "severity": "warning",
                "market": "global",
                "source": "openai_postmortem_analysis",
                "title": "AI postmortem analysis fallback",
                "message": (summary or "AI postmortem analysis failed; local postmortem remains available.")[:220],
            }
        )
        return rows

    rows.append(
        {
            "id": f"openai_postmortem_{ts_now}",
            "ts": int(ts_now),
            "severity": "info",
            "market": "global",
            "source": "openai_postmortem_analysis",
            "title": "AI postmortem tuning applied",
            "message": (
                f"{applied_count} low-risk postmortem tuning change(s) were auto-applied"
                + (
                    f"; verified {int(max(0.0, _f(payload.get('persisted_verified_count', 0), 0.0)))} persisted."
                    if int(max(0.0, _f(payload.get("persisted_verified_count", 0), 0.0))) > 0
                    else "."
                )
            )[:220],
            "change_applied": True,
        }
    )
    applied_rows = payload.get("applied_tuning", []) if isinstance(payload.get("applied_tuning", []), list) else []
    for idx, row in enumerate(applied_rows[:4]):
        if not isinstance(row, dict):
            continue
        key = str(row.get("setting_key", "") or "").strip()
        old_val = row.get("old_value")
        new_val = row.get("new_value")
        reason = str(row.get("reason", "") or "").strip()
        if not key:
            continue
        msg = f"{key}: {old_val} -> {new_val}"
        if reason:
            msg = f"{msg} | {reason[:120]}"
        rows.append(
            {
                "id": f"openai_postmortem_tuning_{idx}_{ts_now}",
                "ts": int(ts_now),
                "severity": "info",
                "market": "global",
                "source": "openai_postmortem_analysis",
                "title": f"Applied: {key}"[:120],
                "message": msg[:220],
                "change_applied": True,
            }
        )
    return rows


def _openai_explanation_rows(runtime_state: Dict[str, Any], ts_now: int) -> List[Dict[str, Any]]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    payload = rs.get("openai_explanations", {}) if isinstance(rs.get("openai_explanations", {}), dict) else {}
    if not payload:
        return []
    running = bool(payload.get("running", False))
    status = str(payload.get("status", "") or "").strip().lower()
    summary = str(payload.get("summary", "") or "").strip()
    if not _openai_should_emit_status_row(status=status, running=running, change_applied=False):
        return []
    if not _openai_status_is_hard_error(status):
        return []
    return [
        {
            "id": f"openai_explanations_{ts_now}",
            "ts": int(ts_now),
            "severity": "warning",
            "market": "global",
            "source": "openai_explanations",
            "title": "AI explanations fallback",
            "message": (summary or "AI explanations failed; local explanation text remains active.")[:220],
        }
    ]


def build_notification_center_payload(
    runtime_state: Dict[str, Any],
    incidents_rows: Iterable[Dict[str, Any]] | None = None,
    max_items: int = 220,
) -> Dict[str, Any]:
    rs = runtime_state if isinstance(runtime_state, dict) else {}
    out_rows: List[Dict[str, Any]] = []

    ts_now = int(rs.get("ts", 0) or 0) or int(time.time())
    alerts = rs.get("alerts", {}) if isinstance(rs.get("alerts", {}), dict) else {}
    reasons = [str(x or "").strip() for x in list(alerts.get("reasons", []) or []) if str(x or "").strip()]
    hints = [str(x or "").strip() for x in list(alerts.get("hints", []) or []) if str(x or "").strip()]
    sev = _sev(str(alerts.get("severity", "info") or "info"))
    for i, reason in enumerate(reasons[:8]):
        hint = hints[i] if i < len(hints) else ""
        action = _action_for_runtime_reason(reason, rs, hint=hint)
        row: Dict[str, Any] = {
            "id": f"alert_{i}_{ts_now}",
            "ts": int(ts_now),
            "severity": sev,
            "market": "global",
            "source": "runtime_alerts",
            "title": reason,
            "message": hint or reason,
        }
        if action:
            row["action"] = action
        out_rows.append(
            row
        )
    out_rows.extend(_automation_policy_rows(rs, ts_now))
    out_rows.extend(_cross_market_opportunity_rows(rs, ts_now))
    out_rows.extend(_openai_nightly_review_rows(rs, ts_now))
    out_rows.extend(_openai_position_review_rows(rs, ts_now))
    out_rows.extend(_openai_capital_planner_rows(rs, ts_now))
    out_rows.extend(_openai_root_cause_rows(rs, ts_now))
    out_rows.extend(_openai_strategy_optimizer_rows(rs, ts_now))
    out_rows.extend(_openai_market_context_rows(rs, ts_now))
    out_rows.extend(_openai_postmortem_rows(rs, ts_now))
    out_rows.extend(_openai_explanation_rows(rs, ts_now))

    trends = rs.get("market_trends", {}) if isinstance(rs.get("market_trends", {}), dict) else {}
    for market in ("stocks", "forex"):
        row = trends.get(market, {}) if isinstance(trends.get(market, {}), dict) else {}
        quality = row.get("quality_aggregates", {}) if isinstance(row.get("quality_aggregates", {}), dict) else {}
        rel = row.get("data_source_reliability", {}) if isinstance(row.get("data_source_reliability", {}), dict) else {}
        why = row.get("why_not_traded", {}) if isinstance(row.get("why_not_traded", {}), dict) else {}
        reject_raw = float(quality.get("reject_rate_raw_pct", quality.get("reject_rate_pct", 0.0)) or 0.0)
        reject = effective_reject_pressure(
            reject_raw,
            dominant_reason=quality.get("dominant_reason", ""),
            dominant_ratio_pct=quality.get("reject_dominant_ratio_pct", 0.0),
            leaders_total=quality.get("leaders_total", 0),
            scores_total=quality.get("scores_total", 0),
        )
        rel_score = float(rel.get("score", 0.0) or 0.0)
        why_reason = str(why.get("reason", "") or "").strip()
        if reject >= 90.0:
            message = f"Reject rate {reject:.1f}% is suppressing candidate flow."
            action = _action_for_market_trend_row(market, "High scanner rejection pressure", message)
            row: Dict[str, Any] = {
                "id": f"{market}_reject_{ts_now}",
                "ts": int(ts_now),
                "severity": "warning",
                "market": market,
                "source": "market_trends",
                "title": "High scanner rejection pressure",
                "message": message,
            }
            if action:
                row["action"] = action
            out_rows.append(
                row
            )
        has_rel_signal = isinstance(rel, dict) and any(
            key in rel for key in ("score", "samples", "source_count", "source_failures", "as_of_ts")
        )
        if has_rel_signal and rel_score < 70.0:
            message = f"Reliability score {rel_score:.1f}/100."
            action = _action_for_market_trend_row(market, "Data reliability degraded", message)
            row = {
                "id": f"{market}_reliability_{ts_now}",
                "ts": int(ts_now),
                "severity": ("critical" if rel_score < 55.0 else "warning"),
                "market": market,
                "source": "market_trends",
                "title": "Data reliability degraded",
                "message": message,
            }
            if action:
                row["action"] = action
            out_rows.append(
                row
            )
        if why_reason:
            action = _action_for_execution_gate_row(market, why_reason)
            row = {
                "id": f"{market}_why_not_{ts_now}",
                "ts": int(ts_now),
                "severity": "info",
                "market": market,
                "source": "execution_gate",
                "title": "Why top candidate was not traded",
                "message": why_reason,
            }
            if action:
                row["action"] = action
            out_rows.append(
                row
            )

    for row in list(incidents_rows or []):
        if not isinstance(row, dict):
            continue
        if not _openai_incident_is_actionable(row, rs):
            continue
        if not _incident_is_active(row, rs):
            continue
        ts = int(float(row.get("ts", 0) or 0))
        if ts <= 0:
            continue
        severity = _sev(str(row.get("severity", "info") or "info"))
        if severity == "info":
            continue
        msg = str(row.get("msg", "") or "").strip()
        evt = str(row.get("event", "") or "").strip()
        out_row = {
            "id": f"inc_{ts}_{evt[:24]}",
            "ts": int(ts),
            "severity": severity,
            "market": _market_from_incident(row),
            "source": "incidents",
            "title": evt or "runtime_incident",
            "message": msg[:220],
        }
        action = _action_for_incident_row(row)
        if action:
            out_row["action"] = action
        out_rows.append(out_row)

    # Keep notifications focused on actual issues or concrete changes:
    # - warning/critical issues always show
    # - info rows show only when they carry a quick-action or confirmed applied change
    out_rows = [row for row in out_rows if _row_is_actionable_or_issue(row)]

    out_rows = _dedupe_notification_rows(out_rows)
    out_rows = sorted(
        out_rows,
        key=lambda r: (
            int(r.get("ts", 0) or 0),
            _severity_rank(str(r.get("severity", "info") or "info")),
        ),
        reverse=True,
    )
    out_rows = out_rows[: max(10, int(max_items))]

    by_market: Dict[str, Dict[str, int]] = {}
    by_sev: Dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for row in out_rows:
        market = str(row.get("market", "global") or "global").strip().lower()
        severity = _sev(str(row.get("severity", "info") or "info"))
        by_market.setdefault(market, {"critical": 0, "warning": 0, "info": 0, "total": 0})
        by_market[market][severity] = int(by_market[market].get(severity, 0) or 0) + 1
        by_market[market]["total"] = int(by_market[market].get("total", 0) or 0) + 1
        by_sev[severity] = int(by_sev.get(severity, 0) or 0) + 1

    return {
        "ts": int(ts_now),
        "total": int(len(out_rows)),
        "by_market": by_market,
        "by_severity": by_sev,
        "items": out_rows,
    }


def build_notification_center_from_hub(hub_dir: str, runtime_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rs = runtime_state if isinstance(runtime_state, dict) else _safe_read_json(os.path.join(hub_dir, "runtime_state.json"))
    if runtime_state is None:
        try:
            from app.health_rules import evaluate_runtime_alerts

            settings = _safe_read_json(os.path.join(os.path.dirname(str(hub_dir or "")), "gui_settings.json"))
            if isinstance(rs, dict):
                rs = dict(rs)
                rs["alerts"] = evaluate_runtime_alerts(rs, settings if isinstance(settings, dict) else {})
        except Exception:
            pass
    incidents = _safe_read_jsonl(os.path.join(hub_dir, "incidents.jsonl"), max_lines=500)
    return build_notification_center_payload(rs, incidents_rows=incidents, max_items=220)
