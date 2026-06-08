from __future__ import annotations

from typing import Any, Dict, List

from app.scanner_quality import effective_reject_pressure

_QUICKFIX_MAP: Dict[str, str] = {
    "startup_checks_failed": "Open Runtime Checks and resolve missing scripts/permissions first.",
    "startup_warnings": "Review startup warnings; fix credential or path hygiene before live execution.",
    "scan_reject_pressure": "Lower score/quality gates slightly and verify data source freshness.",
    "error_incidents": "Open incidents and logs; address top recurring runtime error first.",
    "api_unstable": "Reduce scan/step frequency and keep paper/practice mode until stable.",
    "scanner_reject_spike": "Check scanner diagnostics dominant reject reason and tune that gate.",
    "cadence_drift_pressure": "Reduce scanner cadence pressure or network latency; align loop intervals with broker/data capacity.",
    "market_loop_stale": "Check runtime/markets process health and loop heartbeat freshness before trusting scanner status.",
    "exposure_concentration": "Lower per-asset exposure caps or rotate into broader symbol set.",
    "execution_temporarily_disabled": "Wait for cooldown, then verify broker connectivity and quotas before resuming.",
    "key_rotation_due": "Rotate API keys/secrets and update credentials before switching to live mode.",
    "drawdown_guard_triggered": "Review drawdown guard payload and account-value history before restarting trading.",
    "stop_flag_active": "Clear stop flag only after root cause is resolved and checks are green.",
    "shadow_scorecard_blocked": "Keep shadow mode; resolve scorecard blockers before enabling live rollout.",
    "notification_center_critical": "Open Alerts panel and clear critical notifications before proceeding.",
}

_RUNBOOK_LINK_MAP: Dict[str, str] = {
    "startup_checks_failed": "docs/RUNBOOK.md#1-runtime-not-starting",
    "startup_warnings": "docs/RUNBOOK.md#0-preflight-before-shadowlive",
    "scan_reject_pressure": "docs/RUNBOOK.md#8-universe-quality-tuning",
    "error_incidents": "docs/RUNBOOK.md#6-core-logs",
    "api_unstable": "docs/RUNBOOK.md#2-brokerapi-instability",
    "scanner_reject_spike": "docs/RUNBOOK.md#8-universe-quality-tuning",
    "cadence_drift_pressure": "docs/RUNBOOK.md#7-scanner-cadence-troubleshooting",
    "market_loop_stale": "docs/RUNBOOK.md#7-scanner-cadence-troubleshooting",
    "exposure_concentration": "docs/RUNBOOK.md#4-trading-disabled-in-live-mode",
    "execution_temporarily_disabled": "docs/RUNBOOK.md#2-brokerapi-instability",
    "key_rotation_due": "docs/RUNBOOK.md#5-key-hygiene",
    "drawdown_guard_triggered": "docs/RUNBOOK.md#4-trading-disabled-in-live-mode",
    "stop_flag_active": "docs/RUNBOOK.md#4-trading-disabled-in-live-mode",
    "shadow_scorecard_blocked": "docs/RUNBOOK.md#0-preflight-before-shadowlive",
    "notification_center_critical": "docs/RUNBOOK.md#6-core-logs",
}

_NON_ACTIONABLE_STARTUP_WARNING_PREFIXES = (
    "stale_pid_file_removed",
    "key_rotation_due:",
)


def evaluate_runtime_alerts(runtime_state: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    scan_health = runtime_state.get("scan_health", {}) if isinstance(runtime_state.get("scan_health", {}), dict) else {}
    stocks_enabled = bool(settings.get("market_stocks_enabled", True))
    forex_enabled = bool(settings.get("market_forex_enabled", True))
    stocks = (
        scan_health.get("stocks", {})
        if stocks_enabled and isinstance(scan_health.get("stocks", {}), dict)
        else {}
    )
    forex = (
        scan_health.get("forex", {})
        if forex_enabled and isinstance(scan_health.get("forex", {}), dict)
        else {}
    )
    checks = runtime_state.get("checks", {}) if isinstance(runtime_state.get("checks", {}), dict) else {}
    incidents = runtime_state.get("incidents_last_200", {}) if isinstance(runtime_state.get("incidents_last_200", {}), dict) else {}
    sev = incidents.get("by_severity", {}) if isinstance(incidents.get("by_severity", {}), dict) else {}
    sev_1h = incidents.get("by_severity_1h", {}) if isinstance(incidents.get("by_severity_1h", {}), dict) else {}
    by_event_sev_1h = (
        incidents.get("by_event_severity_1h", {}) if isinstance(incidents.get("by_event_severity_1h", {}), dict) else {}
    )
    by_event_sev = incidents.get("by_event_severity", {}) if isinstance(incidents.get("by_event_severity", {}), dict) else {}
    autopilot = runtime_state.get("autopilot", {}) if isinstance(runtime_state.get("autopilot", {}), dict) else {}
    scan_drift = runtime_state.get("scan_drift", {}) if isinstance(runtime_state.get("scan_drift", {}), dict) else {}
    active_drift_raw = scan_drift.get("active", []) if isinstance(scan_drift.get("active", []), list) else []
    scan_cadence = runtime_state.get("scan_cadence", {}) if isinstance(runtime_state.get("scan_cadence", {}), dict) else {}
    active_cadence_raw = scan_cadence.get("active", []) if isinstance(scan_cadence.get("active", []), list) else []
    enabled_markets = {mk for mk, on in {"stocks": stocks_enabled, "forex": forex_enabled}.items() if bool(on)}
    active_drift: List[Dict[str, Any]] = []
    for row in active_drift_raw:
        if not isinstance(row, dict):
            continue
        mk = str(row.get("market", "") or "").strip().lower()
        if mk in {"", "global"} or (mk in enabled_markets):
            active_drift.append(row)
    active_cadence: List[Dict[str, Any]] = []
    for row in active_cadence_raw:
        if not isinstance(row, dict):
            continue
        mk = str(row.get("market", "") or "").strip().lower()
        if mk in {"", "global"} or (mk in enabled_markets):
            active_cadence.append(row)
    execution_guard = runtime_state.get("execution_guard", {}) if isinstance(runtime_state.get("execution_guard", {}), dict) else {}
    guard_markets = execution_guard.get("markets", {}) if isinstance(execution_guard.get("markets", {}), dict) else {}
    market_loop = runtime_state.get("market_loop", {}) if isinstance(runtime_state.get("market_loop", {}), dict) else {}
    exposure_map = runtime_state.get("exposure_map", {}) if isinstance(runtime_state.get("exposure_map", {}), dict) else {}
    top_positions = exposure_map.get("top_positions", []) if isinstance(exposure_map.get("top_positions", []), list) else []
    drawdown_guard = runtime_state.get("drawdown_guard", {}) if isinstance(runtime_state.get("drawdown_guard", {}), dict) else {}
    stop_flag = runtime_state.get("stop_flag", {}) if isinstance(runtime_state.get("stop_flag", {}), dict) else {}
    shadow_scorecards = runtime_state.get("shadow_scorecards", {}) if isinstance(runtime_state.get("shadow_scorecards", {}), dict) else {}
    notification_center = runtime_state.get("notification_center", {}) if isinstance(runtime_state.get("notification_center", {}), dict) else {}

    reject_warn = float(settings.get("runtime_alert_scan_reject_warn_pct", 65.0) or 65.0)
    reject_crit = float(settings.get("runtime_alert_scan_reject_crit_pct", 85.0) or 85.0)
    incident_warn = int(float(settings.get("runtime_alert_incident_warn_count", 8) or 8))
    incident_crit = int(float(settings.get("runtime_alert_incident_crit_count", 20) or 20))
    error_warn = int(float(settings.get("runtime_alert_error_incident_warn_count", 2) or 2))
    error_crit = int(float(settings.get("runtime_alert_error_incident_crit_count", 6) or 6))
    startup_warn = int(float(settings.get("runtime_alert_startup_warning_warn_count", 2) or 2))
    drift_warn = int(float(settings.get("runtime_alert_drift_spike_warn_count", 1) or 1))
    drift_crit = int(float(settings.get("runtime_alert_drift_spike_crit_count", 3) or 3))
    cadence_warn = int(float(settings.get("runtime_alert_cadence_warn_count", 1) or 1))
    cadence_crit = int(float(settings.get("runtime_alert_cadence_crit_count", 2) or 2))
    market_loop_stale_s = float(settings.get("runtime_alert_market_loop_stale_s", 90.0) or 90.0)
    exposure_warn_pct = float(settings.get("runtime_alert_exposure_concentration_warn_pct", 55.0) or 55.0)
    exposure_crit_pct = float(settings.get("runtime_alert_exposure_concentration_crit_pct", 75.0) or 75.0)
    exposure_min_market_account_pct = float(
        settings.get("runtime_alert_exposure_concentration_min_market_account_pct", 5.0) or 5.0
    )
    rollout_stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    shadow_scorecard_gate_relevant = rollout_stage in {"legacy", "scan_expanded", "risk_caps", "shadow_only"}

    s_reject_raw = float(stocks.get("reject_rate_pct", 0.0) or 0.0)
    f_reject_raw = float(forex.get("reject_rate_pct", 0.0) or 0.0)
    s_dom = str(stocks.get("reject_dominant_reason", "") or "").strip().lower()
    f_dom = str(forex.get("reject_dominant_reason", "") or "").strip().lower()
    s_dom_ratio = float(stocks.get("reject_dominant_ratio_pct", 0.0) or 0.0)
    f_dom_ratio = float(forex.get("reject_dominant_ratio_pct", 0.0) or 0.0)
    s_leaders = int(stocks.get("leaders_total", 0) or 0)
    f_leaders = int(forex.get("leaders_total", 0) or 0)
    s_scores = int(stocks.get("scores_total", 0) or 0)
    f_scores = int(forex.get("scores_total", 0) or 0)
    unknown_dom_cap = max(0.0, float(settings.get("runtime_alert_reject_unknown_dom_cap_pct", 64.0) or 64.0))

    s_reject = effective_reject_pressure(
        s_reject_raw,
        dominant_reason=s_dom,
        dominant_ratio_pct=s_dom_ratio,
        leaders_total=s_leaders,
        scores_total=s_scores,
        unknown_dom_cap_pct=unknown_dom_cap,
    )
    f_reject = effective_reject_pressure(
        f_reject_raw,
        dominant_reason=f_dom,
        dominant_ratio_pct=f_dom_ratio,
        leaders_total=f_leaders,
        scores_total=f_scores,
        unknown_dom_cap_pct=unknown_dom_cap,
    )
    max_reject = max((s_reject if stocks_enabled else 0.0), (f_reject if forex_enabled else 0.0))
    incident_count_total = int(incidents.get("count", 0) or 0)
    sev_src = sev_1h if bool(sev_1h) else sev
    warn_count = int((sev_src.get("warning", 0) or 0) + (sev_src.get("warn", 0) or 0))
    err_count_raw = int((sev_src.get("error", 0) or 0) + (sev_src.get("critical", 0) or 0) + (sev_src.get("high", 0) or 0))
    evt_src = by_event_sev_1h if bool(by_event_sev_1h) else by_event_sev
    cadence_evt = evt_src.get("scanner_cadence_drift", {}) if isinstance(evt_src.get("scanner_cadence_drift", {}), dict) else {}
    cadence_err_count = int(
        (cadence_evt.get("error", 0) or 0) + (cadence_evt.get("critical", 0) or 0) + (cadence_evt.get("high", 0) or 0)
    )
    # Cadence drift already contributes through scan_cadence metrics; avoid double-counting it as generic runtime errors.
    err_count = max(0, int(err_count_raw - cadence_err_count))
    incident_count = int(warn_count + err_count)
    raw_startup_warnings = [
        str(x or "").strip()
        for x in list(checks.get("warnings", []) or [])
        if str(x or "").strip()
    ]
    startup_warnings = [
        row
        for row in raw_startup_warnings
        if not any(str(row).lower().startswith(prefix) for prefix in _NON_ACTIONABLE_STARTUP_WARNING_PREFIXES)
    ]
    warns = int(len(startup_warnings))
    startup_errors = len(list(checks.get("errors", []) or []))
    checks_ok_raw = bool(checks.get("ok", False))
    checks_indeterminate = bool((not checks_ok_raw) and startup_errors <= 0 and warns <= 0)
    checks_ok = bool(checks_ok_raw or checks_indeterminate)
    api_unstable = bool(autopilot.get("api_unstable", False))
    drift_count = int(len(active_drift))
    cadence_count = int(len(active_cadence))
    cadence_critical = 0
    for row in active_cadence:
        if not isinstance(row, dict):
            continue
        if str(row.get("level", "") or "").strip().lower() == "critical":
            cadence_critical += 1
    loop_age_s = int(float(market_loop.get("age_s", -1) or -1))
    try:
        now_ts = int(runtime_state.get("ts", 0) or 0)
    except Exception:
        now_ts = 0
    check_ts = 0
    if isinstance(checks, dict):
        try:
            check_ts = int(checks.get("ts", 0) or 0)
        except Exception:
            check_ts = 0
    startup_age_s = max(0, (now_ts - check_ts)) if (now_ts > 0 and check_ts > 0 and now_ts >= check_ts) else 0
    startup_grace_s = max(0.0, float(settings.get("runtime_alert_startup_grace_s", 180.0) or 180.0))
    loop_stale = bool(loop_age_s >= int(max(10.0, market_loop_stale_s)))
    loop_crit = bool(loop_age_s >= int(max(20.0, market_loop_stale_s * 3.0)))
    if startup_age_s > 0 and startup_age_s < startup_grace_s:
        loop_stale = False
        loop_crit = False

    def _event_severity_count(event_map: Dict[str, Dict[str, Any]], event: str, severities: List[str]) -> int:
        row = event_map.get(event, {}) if isinstance(event_map.get(event, {}), dict) else {}
        total = 0
        for sev_key in severities:
            total += int(row.get(sev_key, 0) or 0)
        return int(total)

    startup_checks_active = (not checks_ok) or warns > 0 or startup_errors > 0
    inactive_warn_sub = 0
    inactive_err_sub = 0
    inactive_cadence_err_sub = 0
    # Sleep-resume diagnostics are environmental telemetry, not runtime health regressions.
    inactive_warn_sub += _event_severity_count(evt_src, "runner_sleep_resume_detected", ["warning", "warn"])
    inactive_err_sub += _event_severity_count(evt_src, "runner_sleep_resume_detected", ["critical", "error", "high"])
    if not startup_checks_active:
        inactive_warn_sub += _event_severity_count(evt_src, "runner_startup_check", ["warning", "warn"])
        inactive_err_sub += _event_severity_count(evt_src, "runner_startup_check", ["critical", "error", "high"])
    if cadence_count <= 0:
        inactive_warn_sub += _event_severity_count(evt_src, "scanner_cadence_drift", ["warning", "warn"])
        inactive_cadence_err_sub += _event_severity_count(evt_src, "scanner_cadence_drift", ["critical", "error", "high"])
        inactive_err_sub += inactive_cadence_err_sub
    if not loop_stale:
        for event_name in ("runner_market_loop_status_stale", "runner_market_loop_restart"):
            inactive_warn_sub += _event_severity_count(evt_src, event_name, ["warning", "warn"])
            inactive_err_sub += _event_severity_count(evt_src, event_name, ["critical", "error", "high"])

    warn_count = max(0, int(warn_count - inactive_warn_sub))
    err_count_raw = max(0, int(err_count_raw - inactive_err_sub))
    cadence_err_count = max(0, int(cadence_err_count - inactive_cadence_err_sub))
    # Cadence drift already contributes through scan_cadence metrics; avoid double-counting it as generic runtime errors.
    err_count = max(0, int(err_count_raw - cadence_err_count))
    incident_count = int(warn_count + err_count)
    cadence_critical_effective = int(cadence_critical)
    cadence_critical_suppressed = False
    # Cadence alerts can stay high when scanners are intentionally configured faster than the
    # achievable loop throughput. Escalate to critical only when paired with other instability signals.
    if cadence_critical_effective > 0 and (not loop_stale) and (not api_unstable) and (err_count < error_warn):
        cadence_critical_effective = 0
        cadence_critical_suppressed = True
    guard_active = 0
    for row in guard_markets.values():
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("disabled_until", 0) or 0) > int(now_ts or 0):
                guard_active += 1
        except Exception:
            continue
    top_exposure_pct = 0.0
    top_exposure_market_account_pct = 0.0
    if top_positions and isinstance(top_positions[0], dict):
        top_exposure_pct = float(top_positions[0].get("pct_of_total_exposure", 0.0) or 0.0)
        top_exposure_market_account_pct = float(top_positions[0].get("pct_of_market_account", 0.0) or 0.0)
    exposure_concentration_warn = bool(
        top_exposure_pct >= exposure_warn_pct and top_exposure_market_account_pct >= exposure_min_market_account_pct
    )
    exposure_concentration_crit = bool(
        top_exposure_pct >= exposure_crit_pct and top_exposure_market_account_pct >= exposure_min_market_account_pct
    )
    stocks_score_gate = str((shadow_scorecards.get("stocks", {}) if isinstance(shadow_scorecards.get("stocks", {}), dict) else {}).get("promotion_gate", "") or "").strip().upper()
    forex_score_gate = str((shadow_scorecards.get("forex", {}) if isinstance(shadow_scorecards.get("forex", {}), dict) else {}).get("promotion_gate", "") or "").strip().upper()
    notif_by_sev = notification_center.get("by_severity", {}) if isinstance(notification_center.get("by_severity", {}), dict) else {}
    notif_critical = int(notif_by_sev.get("critical", 0) or 0)
    shadow_scorecard_blocked = shadow_scorecard_gate_relevant and (
        (stocks_enabled and stocks_score_gate == "BLOCK")
        or (forex_enabled and forex_score_gate == "BLOCK")
    )

    reasons: List[str] = []
    hints: List[str] = []
    severity = "ok"

    def bump(to: str) -> None:
        nonlocal severity
        order = {"ok": 0, "warn": 1, "critical": 2}
        if order.get(to, 0) > order.get(severity, 0):
            severity = to

    if (
        (not checks_ok)
        or err_count >= error_crit
        or max_reject >= reject_crit
        or drift_count >= drift_crit
        or cadence_critical_effective >= cadence_crit
        or loop_crit
        or exposure_concentration_crit
        or shadow_scorecard_blocked
    ):
        bump("critical")
    if bool(drawdown_guard.get("triggered_recent", False)) or bool(stop_flag.get("active", False)):
        bump("critical")
    if warns >= startup_warn or err_count >= error_warn or incident_count >= incident_warn or max_reject >= reject_warn or api_unstable or drift_count >= drift_warn or cadence_count >= cadence_warn or loop_stale or exposure_concentration_warn or guard_active > 0:
        bump("warn")

    if not checks_ok:
        reasons.append("startup_checks_failed")
        hints.append("Fix runtime_startup_checks errors before enabling unattended trading.")
    if warns >= startup_warn:
        reasons.append("startup_warnings")
        hints.append("Review startup warnings in runtime_startup_checks.json.")
    if any(str(w).lower().startswith("key_rotation_due:") for w in raw_startup_warnings):
        reasons.append("key_rotation_due")
        hints.append("API key age exceeded rotation threshold; rotate credentials.")
    if max_reject >= reject_warn:
        reasons.append("scan_reject_pressure")
        hints.append("High scanner rejection rate: loosen gates or improve input data quality.")
    if err_count >= error_warn:
        reasons.append("error_incidents")
        hints.append("Recent runtime errors elevated; inspect incidents and broker health.")
    if api_unstable:
        reasons.append("api_unstable")
        hints.append("Autopilot detected API instability; keep request pace conservative.")
    if drift_count >= drift_warn:
        reasons.append("scanner_reject_spike")
        hints.append("Scanner reject spike detected; review data gates and universe quality filters.")
    if cadence_count >= cadence_warn:
        reasons.append("cadence_drift_pressure")
        hints.append("Scanner cadence drift active; scanner loops are running slower than configured cadence.")
    if loop_stale:
        reasons.append("market_loop_stale")
        hints.append("Market loop heartbeat is stale; verify markets runner process and loop heartbeat output.")
    if exposure_concentration_warn:
        reasons.append("exposure_concentration")
        hints.append("Exposure concentration is high; diversify or tighten per-asset caps.")
    if guard_active > 0:
        reasons.append("execution_temporarily_disabled")
        hints.append("Execution is temporarily paused due to repeated broker failures; cooldown in progress.")
    if bool(drawdown_guard.get("triggered_recent", False)):
        reasons.append("drawdown_guard_triggered")
        hints.append("Global drawdown guard recently triggered; stop trading remains in effect.")
    if bool(stop_flag.get("active", False)):
        reasons.append("stop_flag_active")
        hints.append("Stop flag file is present; trading should remain paused until reviewed.")
    if shadow_scorecard_blocked:
        reasons.append("shadow_scorecard_blocked")
        hints.append("Shadow scorecard gate is BLOCK for at least one market; keep rollout in shadow mode.")

    # Avoid non-actionable warning banners when we do not have concrete reasons to present.
    if severity in {"warn", "critical"} and (not reasons):
        severity = "ok"

    quickfix: List[str] = []
    for r in reasons:
        tip = str(_QUICKFIX_MAP.get(r, "") or "").strip()
        if tip and tip not in quickfix:
            quickfix.append(tip)
    runbook_links: List[Dict[str, str]] = []
    for r in reasons:
        link = str(_RUNBOOK_LINK_MAP.get(r, "") or "").strip()
        if not link:
            continue
        if any(link == str(x.get("path", "") or "") for x in runbook_links):
            continue
        runbook_links.append({"reason": r, "path": link})

    return {
        "severity": severity,
        "reasons": reasons[:8],
        "hints": hints[:8],
        "quickfix_suggestions": quickfix[:5],
        "runbook_links": runbook_links[:5],
        "metrics": {
            "stocks_reject_rate_pct": round(s_reject, 3),
            "forex_reject_rate_pct": round(f_reject, 3),
            "stocks_reject_rate_raw_pct": round(s_reject_raw, 3),
            "forex_reject_rate_raw_pct": round(f_reject_raw, 3),
            "stocks_enabled": bool(stocks_enabled),
            "forex_enabled": bool(forex_enabled),
            "incident_count_last_200": int(incident_count),
            "incident_count_total_last_200": int(incident_count_total),
            "warning_incidents_last_200": int(warn_count),
            "error_incidents_last_200": int(err_count),
            "error_incidents_raw_last_1h": int(err_count_raw),
            "error_incidents_cadence_last_1h": int(cadence_err_count),
            "error_incidents_non_cadence_last_1h": int(err_count),
            "startup_warning_count": int(warns),
            "startup_checks_indeterminate": bool(checks_indeterminate),
            "checks_ok": bool(checks_ok),
            "api_unstable": bool(api_unstable),
            "drift_spike_active_count": int(drift_count),
            "scan_cadence_active_count": int(cadence_count),
            "scan_cadence_critical_count": int(cadence_critical),
            "scan_cadence_critical_effective_count": int(cadence_critical_effective),
            "scan_cadence_critical_suppressed": bool(cadence_critical_suppressed),
            "market_loop_age_s": int(loop_age_s),
            "market_loop_stale": bool(loop_stale),
            "execution_guard_active_markets": int(guard_active),
            "top_exposure_pct_of_total": round(top_exposure_pct, 4),
            "top_exposure_pct_of_market_account": round(top_exposure_market_account_pct, 4),
            "drawdown_guard_triggered_recent": bool(drawdown_guard.get("triggered_recent", False)),
            "stop_flag_active": bool(stop_flag.get("active", False)),
            "shadow_scorecard_stocks_gate": stocks_score_gate,
            "shadow_scorecard_forex_gate": forex_score_gate,
            "notification_critical_count": int(notif_critical),
        },
        "thresholds": {
            "scan_reject_warn_pct": float(reject_warn),
            "scan_reject_crit_pct": float(reject_crit),
            "reject_unknown_dom_cap_pct": float(unknown_dom_cap),
            "incident_warn_count": int(incident_warn),
            "incident_crit_count": int(incident_crit),
            "error_incident_warn_count": int(error_warn),
            "error_incident_crit_count": int(error_crit),
            "startup_warning_warn_count": int(startup_warn),
            "drift_spike_warn_count": int(drift_warn),
            "drift_spike_crit_count": int(drift_crit),
            "cadence_warn_count": int(cadence_warn),
            "cadence_crit_count": int(cadence_crit),
            "market_loop_stale_s": float(market_loop_stale_s),
            "startup_grace_s": float(startup_grace_s),
            "exposure_concentration_warn_pct": float(exposure_warn_pct),
            "exposure_concentration_crit_pct": float(exposure_crit_pct),
            "exposure_concentration_min_market_account_pct": float(exposure_min_market_account_pct),
        },
    }
