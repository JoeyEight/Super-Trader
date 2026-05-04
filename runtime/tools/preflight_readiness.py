from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.api_endpoint_validation import validate_alpaca_endpoints, validate_oanda_endpoints
from app.credential_utils import (
    get_alpaca_creds,
    get_oanda_creds,
    get_robinhood_creds_from_env,
    get_robinhood_creds_from_files,
    key_file_permission_issues,
    key_rotation_reminder_issues,
)
from app.path_utils import read_settings_file, resolve_settings_path
from app.settings_utils import sanitize_settings


def _resolve_path(base_dir: str, path_value: Any) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(base_dir, raw))


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_is_alive(pid: int) -> bool:
    try:
        if int(pid) <= 0:
            return False
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _writable_dir_status(path: str) -> Tuple[bool, str]:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".preflight_write_probe.tmp")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _mode_string(path: str) -> str:
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        return oct(mode)
    except Exception:
        return ""


def _issue(level: str, code: str, message: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "level": str(level or "info").lower(),
        "code": str(code or "").strip() or "issue",
        "message": str(message or "").strip(),
        "details": dict(details or {}),
    }


def build_preflight_report(project_dir: str, now_ts: int | None = None) -> Dict[str, Any]:
    base_dir = os.path.abspath(str(project_dir or os.getcwd()))
    settings_path = resolve_settings_path(base_dir) or os.path.join(base_dir, "gui_settings.json")
    raw_settings = read_settings_file(settings_path, module_name="preflight_readiness")
    settings = sanitize_settings(raw_settings if isinstance(raw_settings, dict) else {})
    ts_now = int(time.time() if now_ts is None else now_ts)

    issues: List[Dict[str, Any]] = []

    settings_exists = bool(os.path.isfile(settings_path))
    if not settings_exists:
        issues.append(
            _issue(
                "warning",
                "settings_missing",
                "Settings file was not found; defaults were used for readiness checks.",
                {"settings_path": settings_path},
            )
        )

    hub_dir = _resolve_path(base_dir, settings.get("hub_data_dir", "")) or os.path.join(base_dir, "hub_data")
    hub_writable, hub_err = _writable_dir_status(hub_dir)
    if not hub_writable:
        issues.append(_issue("critical", "hub_data_unwritable", "Hub data directory is not writable.", {"path": hub_dir, "error": hub_err}))

    logs_dir = os.path.join(hub_dir, "logs")
    logs_writable, logs_err = _writable_dir_status(logs_dir)
    if not logs_writable:
        issues.append(_issue("critical", "logs_unwritable", "Runtime logs directory is not writable.", {"path": logs_dir, "error": logs_err}))

    script_keys = (
        "script_neural_runner2",
        "script_neural_trainer",
        "script_trader",
        "script_markets_runner",
        "script_autopilot",
    )
    scripts: Dict[str, Dict[str, Any]] = {}
    for key in script_keys:
        resolved = _resolve_path(base_dir, settings.get(key, ""))
        exists = bool(resolved and os.path.isfile(resolved))
        scripts[key] = {"configured": str(settings.get(key, "")), "resolved_path": resolved, "exists": exists}
        if not exists:
            issues.append(
                _issue(
                    "critical",
                    "script_missing",
                    f"Configured script is missing for `{key}`.",
                    {"key": key, "path": resolved, "configured": str(settings.get(key, ""))},
                )
            )

    rollout_stage = str(settings.get("market_rollout_stage", "legacy") or "legacy").strip().lower()
    if rollout_stage == "live_guarded":
        rollout_stage = "live"
    paper_only_guard = bool(settings.get("paper_only_unless_checklist_green", True))
    alpaca_paper = bool(settings.get("alpaca_paper_mode", True))
    oanda_practice = bool(settings.get("oanda_practice_mode", True))
    stock_auto = bool(settings.get("stock_auto_trade_enabled", False))
    forex_auto = bool(settings.get("forex_auto_trade_enabled", False))

    rh_key_env, rh_secret_env = get_robinhood_creds_from_env()
    rh_key_file, rh_secret_file = get_robinhood_creds_from_files(base_dir)
    rh_ok = bool((rh_key_env and rh_secret_env) or (rh_key_file and rh_secret_file))

    alpaca_key, alpaca_secret = get_alpaca_creds(settings, base_dir=base_dir)
    oanda_account, oanda_token = get_oanda_creds(settings, base_dir=base_dir)
    alpaca_ok = bool(alpaca_key and alpaca_secret)
    oanda_ok = bool(oanda_account and oanda_token)
    alpaca_endpoint_check = validate_alpaca_endpoints(
        settings.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
        settings.get("alpaca_data_url", "https://data.alpaca.markets"),
        paper_mode=bool(alpaca_paper),
    )
    oanda_endpoint_check = validate_oanda_endpoints(
        settings.get("oanda_rest_url", "https://api-fxpractice.oanda.com"),
        settings.get("oanda_stream_url", ""),
        practice_mode=bool(oanda_practice),
    )
    for row in list(alpaca_endpoint_check.get("issues", []) or []):
        if isinstance(row, dict):
            issues.append(row)
    for row in list(oanda_endpoint_check.get("issues", []) or []):
        if isinstance(row, dict):
            issues.append(row)

    if stock_auto and (not alpaca_ok):
        issues.append(_issue("critical", "alpaca_creds_missing", "Stocks auto-trade is enabled but Alpaca credentials are missing."))
    if forex_auto and (not oanda_ok):
        issues.append(_issue("critical", "oanda_creds_missing", "Forex auto-trade is enabled but OANDA credentials are missing."))
    if not rh_ok:
        issues.append(_issue("warning", "robinhood_creds_missing", "Crypto credentials were not found in env or keys files."))
    runner_pid_path = os.path.join(hub_dir, "runner.pid")
    runner_pid = 0
    runner_alive = False
    try:
        if os.path.isfile(runner_pid_path):
            with open(runner_pid_path, "r", encoding="utf-8") as f:
                runner_pid = int(float((f.read() or "0").strip() or 0))
    except Exception:
        runner_pid = 0
    runner_alive = _pid_is_alive(runner_pid) if runner_pid > 0 else False

    market_loop_path = os.path.join(hub_dir, "market_loop_status.json")
    market_loop = _safe_read_json(market_loop_path)
    market_loop_ts = 0
    try:
        market_loop_ts = int(float(market_loop.get("ts", 0) or 0))
    except Exception:
        market_loop_ts = 0
    market_loop_age_s = (max(0, int(ts_now) - int(market_loop_ts)) if market_loop_ts > 0 else -1)
    try:
        market_loop_stale_after_s = max(10, int(float(settings.get("runtime_alert_market_loop_stale_s", 90.0) or 90.0)))
    except Exception:
        market_loop_stale_after_s = 90
    if runner_alive and (not os.path.isfile(market_loop_path)):
        issues.append(
            _issue(
                "warning",
                "market_loop_status_missing",
                "Runner is active but market loop heartbeat file is missing; restart runner to reload latest runtime modules.",
                {"runner_pid": int(runner_pid), "path": market_loop_path},
            )
        )
    elif runner_alive and market_loop_age_s >= market_loop_stale_after_s:
        issues.append(
            _issue(
                "warning",
                "market_loop_status_stale",
                "Runner is active but market loop heartbeat is stale; markets process may be hung or running stale code.",
                {
                    "runner_pid": int(runner_pid),
                    "path": market_loop_path,
                    "age_s": int(market_loop_age_s),
                    "stale_after_s": int(market_loop_stale_after_s),
                },
            )
        )

    stocks_thinker = _safe_read_json(os.path.join(hub_dir, "stocks", "stock_thinker_status.json"))
    forex_thinker = _safe_read_json(os.path.join(hub_dir, "forex", "forex_thinker_status.json"))
    stock_state = str(stocks_thinker.get("state", "") or "").strip().upper()
    forex_state = str(forex_thinker.get("state", "") or "").strip().upper()
    if runner_alive and alpaca_ok and stock_state == "NOT CONFIGURED":
        issues.append(
            _issue(
                "warning",
                "stocks_cred_runtime_mismatch",
                "Alpaca credentials were found, but Stocks thinker still reports NOT CONFIGURED; restart runner/markets.",
                {"runner_pid": int(runner_pid), "thinker_path": os.path.join(hub_dir, "stocks", "stock_thinker_status.json")},
            )
        )
    if runner_alive and oanda_ok and forex_state == "NOT CONFIGURED":
        issues.append(
            _issue(
                "warning",
                "forex_cred_runtime_mismatch",
                "OANDA credentials were found, but Forex thinker still reports NOT CONFIGURED; restart runner/markets.",
                {"runner_pid": int(runner_pid), "thinker_path": os.path.join(hub_dir, "forex", "forex_thinker_status.json")},
            )
        )

    scorecards_path = os.path.join(hub_dir, "shadow_deployment_scorecards.json")
    scorecards = _safe_read_json(scorecards_path)
    stock_gate = str(((scorecards.get("stocks", {}) if isinstance(scorecards.get("stocks", {}), dict) else {}).get("promotion_gate", "N/A") or "N/A")).strip().upper()
    forex_gate = str(((scorecards.get("forex", {}) if isinstance(scorecards.get("forex", {}), dict) else {}).get("promotion_gate", "N/A") or "N/A")).strip().upper()
    if not scorecards:
        issues.append(
            _issue(
                "warning",
                "shadow_scorecards_missing",
                "Shadow deployment scorecards are missing; wait for markets loop to generate readiness scorecards.",
                {"path": scorecards_path},
            )
        )
    elif rollout_stage in {"execution_v2", "live"}:
        if stock_gate == "BLOCK" or forex_gate == "BLOCK":
            issues.append(
                _issue(
                    "critical",
                    "shadow_scorecard_blocked",
                    "Shadow deployment scorecard gate is BLOCK for one or more markets.",
                    {"stocks_gate": stock_gate, "forex_gate": forex_gate, "path": scorecards_path},
                )
            )
        elif stock_gate == "WARN" or forex_gate == "WARN":
            issues.append(
                _issue(
                    "warning",
                    "shadow_scorecard_warn",
                    "Shadow deployment scorecard gate is WARN; review scorecard blockers before live rollout.",
                    {"stocks_gate": stock_gate, "forex_gate": forex_gate, "path": scorecards_path},
                )
            )

    notif_path = os.path.join(hub_dir, "notification_center.json")
    notif = _safe_read_json(notif_path)
    notif_by_sev = notif.get("by_severity", {}) if isinstance(notif.get("by_severity", {}), dict) else {}
    crit_notif = int(notif_by_sev.get("critical", 0) or 0)
    if crit_notif > 0 and rollout_stage in {"execution_v2", "live"}:
        issues.append(
            _issue(
                "warning",
                "notification_center_critical_items",
                "Notification center currently has active critical items.",
                {"critical_items": int(crit_notif), "path": notif_path},
            )
        )

    if rollout_stage == "live" and alpaca_paper:
        issues.append(_issue("warning", "alpaca_still_paper", "Rollout is `live` but Alpaca is still in paper mode."))
    if rollout_stage == "live" and oanda_practice:
        issues.append(_issue("warning", "oanda_still_practice", "Rollout is `live` but OANDA is still in practice mode."))

    if (rollout_stage in {"execution_v2", "live"}) and (not stock_auto) and (not forex_auto):
        issues.append(_issue("warning", "market_auto_trade_off", "Rollout stage is execution-capable but Stocks/Forex auto-trade are both disabled."))

    if not paper_only_guard:
        issues.append(
            _issue(
                "warning",
                "live_guard_bypass",
                "`paper_only_unless_checklist_green` is disabled; verify this is intentional before live rollout.",
            )
        )

    if not bool(settings.get("stock_block_entries_on_cached_scan", True)):
        issues.append(
            _issue(
                "warning",
                "stock_cached_scan_entries_allowed",
                "Stocks entries are allowed during cached fallback mode.",
                {
                    "hard_block_age_s": int(settings.get("stock_cached_scan_hard_block_age_s", 1800) or 1800),
                    "entry_size_mult": float(settings.get("stock_cached_scan_entry_size_mult", 0.60) or 0.60),
                },
            )
        )
    if not bool(settings.get("forex_block_entries_on_cached_scan", True)):
        issues.append(
            _issue(
                "warning",
                "forex_cached_scan_entries_allowed",
                "Forex entries are allowed during cached fallback mode.",
                {
                    "hard_block_age_s": int(settings.get("forex_cached_scan_hard_block_age_s", 1200) or 1200),
                    "entry_size_mult": float(settings.get("forex_cached_scan_entry_size_mult", 0.65) or 0.65),
                },
            )
        )

    if not bool(settings.get("stock_require_data_quality_ok_for_entries", True)):
        issues.append(_issue("warning", "stock_data_quality_gate_disabled", "Stocks data-quality entry gate is disabled."))
    if not bool(settings.get("forex_require_data_quality_ok_for_entries", True)):
        issues.append(_issue("warning", "forex_data_quality_gate_disabled", "Forex data-quality entry gate is disabled."))

    try:
        stock_reject_gate = float(settings.get("stock_require_reject_rate_max_pct", 92.0) or 92.0)
    except Exception:
        stock_reject_gate = 92.0
    try:
        forex_reject_gate = float(settings.get("forex_require_reject_rate_max_pct", 92.0) or 92.0)
    except Exception:
        forex_reject_gate = 92.0
    if stock_reject_gate >= 99.0:
        issues.append(_issue("warning", "stock_reject_gate_loose", "Stocks reject-pressure gate is effectively very loose.", {"value_pct": stock_reject_gate}))
    if forex_reject_gate >= 99.0:
        issues.append(_issue("warning", "forex_reject_gate_loose", "Forex reject-pressure gate is effectively very loose.", {"value_pct": forex_reject_gate}))

    perm_issues = key_file_permission_issues(base_dir)
    for item in perm_issues:
        issues.append(_issue("warning", "key_permissions", "Key file permissions are weaker than recommended.", {"detail": item}))
    rotation_days = int(settings.get("key_rotation_warn_days", 0) or 0)
    rotation_issues: List[str] = []
    # Endpoint-managed mode: warn_days <= 0 disables local age-based rotation reminders.
    if rotation_days > 0:
        rotation_issues = key_rotation_reminder_issues(base_dir, max_age_days=rotation_days)
        for item in rotation_issues[:10]:
            issues.append(_issue("warning", "key_rotation_due", "A key file is past rotation reminder age.", {"detail": item}))

    critical_count = sum(1 for it in issues if str(it.get("level", "")).lower() == "critical")
    warning_count = sum(1 for it in issues if str(it.get("level", "")).lower() == "warning")

    summary_lines: List[str] = []
    summary_lines.append(f"stage={rollout_stage} | stock_auto={stock_auto} | forex_auto={forex_auto}")
    summary_lines.append(f"alpaca={'OK' if alpaca_ok else 'MISSING'} | oanda={'OK' if oanda_ok else 'MISSING'} | crypto_keys={'OK' if rh_ok else 'MISSING'}")
    summary_lines.append(
        "alpaca_endpoint="
        + str(alpaca_endpoint_check.get("normalized_base_url", "") or "")
        + " | oanda_endpoint="
        + str(oanda_endpoint_check.get("normalized_rest_url", "") or "")
    )
    summary_lines.append(f"hub_dir={hub_dir}")
    summary_lines.append(
        f"runner={'ACTIVE' if runner_alive else 'INACTIVE'}"
        + (
            f" | market_loop_age_s={int(market_loop_age_s)}"
            if market_loop_age_s >= 0
            else " | market_loop_age_s=N/A"
        )
    )
    summary_lines.append(f"shadow_scorecards: stocks={stock_gate} forex={forex_gate}")
    summary_lines.append(f"issues: critical={critical_count} warning={warning_count}")

    return {
        "ts": int(ts_now),
        "project_dir": base_dir,
        "settings_path": settings_path,
        "settings_exists": bool(settings_exists),
        "hub_dir": hub_dir,
        "hub_writable": bool(hub_writable),
        "logs_writable": bool(logs_writable),
        "runtime": {
            "runner_pid": int(runner_pid),
            "runner_alive": bool(runner_alive),
            "runner_pid_path": runner_pid_path,
            "market_loop_path": market_loop_path,
            "market_loop_ts": int(market_loop_ts),
            "market_loop_age_s": int(market_loop_age_s),
            "market_loop_stale_after_s": int(market_loop_stale_after_s),
            "stock_thinker_state": stock_state,
            "forex_thinker_state": forex_state,
        },
        "scripts": scripts,
        "rollout": {
            "stage": rollout_stage,
            "paper_only_unless_checklist_green": bool(paper_only_guard),
            "alpaca_paper_mode": bool(alpaca_paper),
            "oanda_practice_mode": bool(oanda_practice),
            "stock_auto_trade_enabled": bool(stock_auto),
            "forex_auto_trade_enabled": bool(forex_auto),
        },
        "credentials": {
            "crypto_robinhood_ok": bool(rh_ok),
            "stocks_alpaca_ok": bool(alpaca_ok),
            "forex_oanda_ok": bool(oanda_ok),
            "key_permission_issues": list(perm_issues),
            "key_rotation_issues": list(rotation_issues),
        },
        "endpoint_validation": {
            "alpaca": alpaca_endpoint_check,
            "oanda": oanda_endpoint_check,
        },
        "issues": issues,
        "counts": {"critical": int(critical_count), "warning": int(warning_count)},
        "pass": bool(critical_count == 0),
        "summary": summary_lines,
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight readiness checks for shadow/live validation.")
    ap.add_argument("--project-dir", default=os.getcwd())
    ap.add_argument("--output", default="")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Fail if warnings exist (not only critical issues).")
    args = ap.parse_args()

    report = build_preflight_report(str(args.project_dir or os.getcwd()))

    output_path = str(args.output or "").strip()
    if not output_path:
        output_path = os.path.join(str(report.get("hub_dir", "")), "preflight_readiness.json")
    try:
        _write_json(output_path, report)
    except Exception:
        pass

    if bool(args.json_only):
        print(json.dumps(report, indent=2))
    else:
        print("Super Trader Preflight Readiness")
        print(f"project: {report.get('project_dir')}")
        print(f"settings: {report.get('settings_path')}")
        for line in list(report.get("summary", []) or []):
            print(f"- {line}")
        print(f"- report: {output_path}")
        issues = list(report.get("issues", []) or [])
        if issues:
            print("\nIssues:")
            for row in issues[:40]:
                lvl = str(row.get("level", "info")).upper()
                msg = str(row.get("message", "") or "")
                code = str(row.get("code", "") or "")
                print(f"  [{lvl}] {code}: {msg}")
        else:
            print("\nNo issues detected.")

    critical = int((report.get("counts", {}) or {}).get("critical", 0) or 0)
    warning = int((report.get("counts", {}) or {}).get("warning", 0) or 0)
    if critical > 0:
        return 1
    if bool(args.strict) and warning > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
