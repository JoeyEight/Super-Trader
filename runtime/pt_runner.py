from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional, TextIO

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.api_quota import summarize_quota_events
from app.automation_policy import summarize_policy_snapshot
from app.cache_maintenance import prune_data_cache, prune_scanner_quality_artifacts
from app.credential_utils import (
    get_alpaca_creds,
    get_oanda_creds,
    key_file_permission_issues,
    key_rotation_reminder_issues,
)
from app.exposure_analytics import build_exposure_payload
from app.feature_flags import build_feature_flag_snapshot
from app.health_rules import evaluate_runtime_alerts
from app.http_utils import parse_retry_after_value
from app.opportunity_allocator import summarize_allocator_snapshot
from app.json_codec import load as json_load
from app.json_codec import loads as json_loads
from app.notification_center import build_notification_center_payload
from app.path_utils import read_settings_file, resolve_runtime_paths, resolve_settings_path
from app.runtime_insights import (
    build_broker_latency_histogram,
    build_incident_trend,
    build_pnl_decomposition,
    detect_equity_anomaly,
    detect_stale_history,
)
from app.runtime_logging import append_jsonl, atomic_write_json, cleanup_logs, runtime_event, trim_jsonl_max_lines
from app.scan_diagnostics_schema import normalize_scan_diagnostics
from app.settings_utils import sanitize_settings
from app.time_utils import now_date_local, now_datetime_local, now_ts

BASE_DIR, _SETTINGS_PATH, HUB_DATA_DIR, _BOOT_SETTINGS = resolve_runtime_paths(__file__, "pt_runner")
RUNNER_PID_PATH = os.path.join(HUB_DATA_DIR, "runner.pid")
STOP_FLAG_PATH = os.path.join(HUB_DATA_DIR, "stop_trading.flag")
TRADER_STATUS_PATH = os.path.join(HUB_DATA_DIR, "trader_status.json")
CRYPTO_TRADER_DETAIL_PATH = os.path.join(HUB_DATA_DIR, "trader_data.json")
LOG_DIR = os.path.join(HUB_DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

RUNNER_LOG_PATH = os.path.join(LOG_DIR, "runner.log")
THINKER_LOG_PATH = os.path.join(LOG_DIR, "thinker.log")
TRADER_LOG_PATH = os.path.join(LOG_DIR, "trader.log")
MARKETS_LOG_PATH = os.path.join(LOG_DIR, "markets.log")
AUTOPILOT_LOG_PATH = os.path.join(LOG_DIR, "autopilot.log")
RUNTIME_CHECKS_PATH = os.path.join(HUB_DATA_DIR, "runtime_startup_checks.json")
KEY_ROTATION_STATUS_PATH = os.path.join(HUB_DATA_DIR, "key_rotation_status.json")
INCIDENTS_PATH = os.path.join(HUB_DATA_DIR, "incidents.jsonl")
RUNTIME_EVENTS_PATH = os.path.join(HUB_DATA_DIR, "runtime_events.jsonl")
RUNTIME_STATE_PATH = os.path.join(HUB_DATA_DIR, "runtime_state.json")
NOTIFICATION_CENTER_PATH = os.path.join(HUB_DATA_DIR, "notification_center.json")
MARKET_REGIMES_PATH = os.path.join(HUB_DATA_DIR, "market_regimes.json")
WALKFORWARD_REPORT_PATH = os.path.join(HUB_DATA_DIR, "walkforward_report.json")
CONFIDENCE_CALIBRATION_PATH = os.path.join(HUB_DATA_DIR, "confidence_calibration.json")
SHADOW_SCORECARDS_PATH = os.path.join(HUB_DATA_DIR, "shadow_deployment_scorecards.json")
MARKET_LOOP_STATUS_PATH = os.path.join(HUB_DATA_DIR, "market_loop_status.json")
CADENCE_DRIFT_PATH = os.path.join(HUB_DATA_DIR, "scanner_cadence_drift.json")

HEARTBEAT_INTERVAL_S = 2.0
MAX_BACKOFF_S = 30.0
CRASH_WINDOW_S = 600.0
CRASH_THRESHOLD = 10
CRASH_LOCKOUT_S = 180.0
LOG_ROTATE_MAX_BYTES = 25 * 1024 * 1024
LOG_ROTATE_KEEP = 8
LOG_RETENTION_AGE_DAYS = 14.0
LOG_RETENTION_MAX_TOTAL_BYTES = 200 * 1024 * 1024
LOG_RETENTION_INTERVAL_S = 600.0
WATCHDOG_INTERVAL_S = 15.0
MARKETS_STALE_MULT = 4.0
AUTOPILOT_STALE_MULT = 6.0
MARKET_LOOP_RESTART_COOLDOWN_S = 180.0
MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_S = 180.0
MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_MULT = 1.25
WATCHDOG_SLEEP_GAP_DETECT_S = 90.0
WATCHDOG_SLEEP_RESUME_GRACE_S = 180.0
SCRIPT_WATCH_INTERVAL_S = 3.0
SCRIPT_CHANGE_MIN_UPTIME_S = 3.0
SCRIPT_CHANGE_RESTART_COOLDOWN_S = 10.0
SLEEP_GUARD_RESTART_COOLDOWN_S = 60.0
DRAWDOWN_GUARD_PATH = os.path.join(HUB_DATA_DIR, "global_drawdown_guard.json")
SAFETY_ACK_PATH = os.path.join(HUB_DATA_DIR, "safety_ack.json")

RUNTIME_STATE_SCHEMA_VERSION = 3
RUNTIME_STATE_MIN_READER_VERSION = 1


def _detached_subprocess_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | int(getattr(subprocess, "DETACHED_PROCESS", 0))
        return {"creationflags": flags} if flags else {}
    return {"start_new_session": True}


def _rotate_log_file(path: str, max_bytes: int = LOG_ROTATE_MAX_BYTES, keep: int = LOG_ROTATE_KEEP) -> None:
    try:
        if not os.path.isfile(path):
            return
        if os.path.getsize(path) <= int(max_bytes):
            return
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        rotated = f"{path}.{ts}"
        os.replace(path, rotated)
        prefix = os.path.basename(path) + "."
        base_dir = os.path.dirname(path)
        olds = sorted([os.path.join(base_dir, n) for n in os.listdir(base_dir) if n.startswith(prefix)])
        if len(olds) > int(keep):
            for old in olds[:-keep]:
                try:
                    os.remove(old)
                except Exception:
                    pass
    except Exception:
        pass


def _runner_log(msg: str) -> None:
    line = f"{now_datetime_local()} {msg}\n"
    try:
        _rotate_log_file(RUNNER_LOG_PATH)
        with open(RUNNER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        pass
    runtime_event(RUNTIME_EVENTS_PATH, component="runner", event="log", level="info", msg=msg)


def _append_incident(severity: str, event: str, msg: str, details: Optional[Dict[str, Any]] = None) -> None:
    append_jsonl(
        INCIDENTS_PATH,
        {
            "ts": now_ts(),
            "date": now_date_local(),
            "severity": str(severity or "info").lower(),
            "event": str(event or "").strip() or "runtime_event",
            "msg": str(msg or "").strip(),
            "details": (details or {}),
        },
    )
    runtime_event(
        RUNTIME_EVENTS_PATH,
        component="runner",
        event=str(event or "incident"),
        level=str(severity or "info"),
        msg=str(msg or ""),
        details=details or {},
    )


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    atomic_write_json(path, data)


def _check_writable_dir(path: str) -> Optional[str]:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe.tmp")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json_load(f, default={})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _intraday_drawdown_pct(history_path: str, lookback_hours: int = 24) -> float:
    now = time.time()
    cutoff = now - (max(1, int(lookback_hours)) * 3600.0)
    vals: list[float] = []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json_loads(ln, default=None)
                except Exception:
                    continue
                try:
                    ts = float(row.get("ts", 0.0) or 0.0)
                    v = float(row.get("total_account_value", 0.0) or 0.0)
                except Exception:
                    ts = 0.0
                    v = 0.0
                if ts < cutoff or v <= 0.0:
                    continue
                vals.append(v)
    except Exception:
        return 0.0
    if len(vals) < 2:
        return 0.0
    peak = max(vals)
    cur = vals[-1]
    if peak <= 0.0:
        return 0.0
    return ((cur - peak) / peak) * 100.0


def _read_jsonl_tail(path: str, limit: int = 600) -> list[Dict[str, Any]]:
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []
    out: list[Dict[str, Any]] = []
    for ln in lines[-max(1, int(limit)):]:
        try:
            row = json_loads(ln, default=None)
            if isinstance(row, dict):
                out.append(row)
        except Exception:
            continue
    return out


def _summarize_broker_backoff_events(rows: list[Dict[str, Any]], now_ts_value: float | None = None) -> Dict[str, Any]:
    now_val = float(time.time() if now_ts_value is None else now_ts_value)
    cutoff_24h = now_val - 86400.0
    waits: list[float] = []
    by_component: Dict[str, int] = {}
    last_wait_s = 0.0
    last_wait_ts = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        evt = str(row.get("event", "") or "").strip().lower()
        if evt != "broker_retry_after_wait":
            continue
        try:
            ts = float(row.get("ts", 0.0) or 0.0)
        except Exception:
            ts = 0.0
        if ts < cutoff_24h:
            continue
        details = row.get("details", {}) if isinstance(row.get("details", {}), dict) else {}
        wait_s = 0.0
        try:
            wait_s = float(details.get("wait_s", 0.0) or 0.0)
        except Exception:
            wait_s = 0.0
        if wait_s <= 0.0:
            wait_s = parse_retry_after_value(str(row.get("msg", "") or ""), max_wait_s=3600.0)
        if wait_s <= 0.0:
            continue
        waits.append(wait_s)
        comp = str(row.get("component", "runtime") or "runtime").strip().lower()
        by_component[comp] = int(by_component.get(comp, 0)) + 1
        if int(ts) >= int(last_wait_ts):
            last_wait_ts = int(ts)
            last_wait_s = float(wait_s)
    return {
        "count_24h": int(len(waits)),
        "avg_wait_s": round((sum(waits) / max(1, len(waits))), 3) if waits else 0.0,
        "max_wait_s": round((max(waits) if waits else 0.0), 3),
        "last_wait_s": round(float(last_wait_s), 3),
        "last_wait_ts": int(last_wait_ts),
        "by_component": by_component,
    }


def _stop_flag_payload(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"active": False, "ts": 0, "age_s": 0, "reason": "", "details": {}}
    ts = 0
    reason = ""
    details: Dict[str, Any] = {}
    raw = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = str(f.read() or "").strip()
        if raw.startswith("{"):
            obj = json_loads(raw, default=None)
            if isinstance(obj, dict):
                ts = int(float(obj.get("ts", 0) or 0))
                reason = str(obj.get("reason", "") or "").strip().lower()
                details = obj.get("details", {}) if isinstance(obj.get("details", {}), dict) else {}
        elif raw:
            ts = int(float(raw))
    except Exception:
        ts = 0
    try:
        mtime = int(os.path.getmtime(path))
    except Exception:
        mtime = int(time.time())
    if ts <= 0:
        ts = mtime
    age_s = max(0, int(time.time()) - int(ts))
    return {"active": True, "ts": int(ts), "age_s": int(age_s), "reason": reason, "details": details}


def _write_stop_flag(path: str, ts_value: int, reason: str = "", details: Dict[str, Any] | None = None) -> None:
    payload = {
        "ts": int(ts_value),
        "reason": str(reason or "").strip().lower(),
        "details": (details if isinstance(details, dict) else {}),
    }
    try:
        _atomic_write_json(path, payload)
    except Exception:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(int(ts_value)))
        except Exception:
            pass


def _clear_stop_flag(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False


def _run_startup_checks(scripts: Dict[str, str], settings: Dict[str, Any], stale_pid_removed: bool) -> Dict[str, Any]:
    errors = []
    warnings = []

    for key, path in scripts.items():
        if not os.path.isfile(path):
            errors.append(f"missing_script:{key}:{path}")

    hub_write = _check_writable_dir(HUB_DATA_DIR)
    if hub_write:
        errors.append(f"hub_data_not_writable:{hub_write}")
    log_write = _check_writable_dir(LOG_DIR)
    if log_write:
        errors.append(f"log_dir_not_writable:{log_write}")

    try:
        a_key, a_secret = get_alpaca_creds(settings, base_dir=BASE_DIR)
        if not (str(a_key or "").strip() and str(a_secret or "").strip()):
            warnings.append("alpaca_credentials_missing")
    except Exception:
        warnings.append("alpaca_credentials_check_failed")
    try:
        o_id, o_tok = get_oanda_creds(settings, base_dir=BASE_DIR)
        if not (str(o_id or "").strip() and str(o_tok or "").strip()):
            warnings.append("oanda_credentials_missing")
    except Exception:
        warnings.append("oanda_credentials_check_failed")

    if stale_pid_removed:
        warnings.append("stale_pid_file_removed")
    try:
        warnings.extend(list(key_file_permission_issues(BASE_DIR)))
    except Exception:
        warnings.append("key_permission_check_failed")
    try:
        max_age_days = int(float(settings.get("key_rotation_warn_days", 90) or 90))
        key_rot = list(key_rotation_reminder_issues(BASE_DIR, max_age_days=max_age_days))
        warnings.extend(key_rot)
        _atomic_write_json(
            KEY_ROTATION_STATUS_PATH,
            {
                "ts": now_ts(),
                "warn_days": int(max_age_days),
                "due": key_rot,
                "due_count": int(len(key_rot)),
            },
        )
    except Exception:
        warnings.append("key_rotation_check_failed")

    payload = {
        "ts": now_ts(),
        "ok": bool((not errors)),
        "errors": list(errors),
        "warnings": list(warnings),
        "scripts": {k: os.path.abspath(v) for k, v in scripts.items()},
    }
    try:
        _atomic_write_json(RUNTIME_CHECKS_PATH, payload)
    except Exception:
        pass

    for msg in errors:
        _append_incident("error", "runner_startup_check", msg, {"component": "runner"})
    for msg in warnings:
        _append_incident("warning", "runner_startup_check", msg, {"component": "runner"})
    return payload


def _pid_is_alive(pid: Optional[int]) -> bool:
    try:
        if not pid or int(pid) <= 0:
            return False
        pid_i = int(pid)
        os.kill(pid_i, 0)
        # On Unix, zombie processes still pass kill(pid, 0). Treat zombies as
        # dead so stale runner.pid files self-heal instead of blocking relaunch.
        if os.name != "nt":
            try:
                out = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pid_i)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
                stat = str((out.stdout or "").strip())
                if stat.upper().startswith("Z"):
                    return False
            except Exception:
                pass
        return True
    except OSError:
        return False


def _read_pid_file(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = (f.read() or "").strip()
        pid = int(raw)
        return pid if pid > 0 else None
    except Exception:
        return None


def _write_pid_file(path: str, pid: int) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(pid)))
    os.replace(tmp, path)


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _ps_process_rows() -> list[Dict[str, Any]]:
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        lines = (out.stdout or "").splitlines()
    except Exception:
        lines = []
    rows: list[Dict[str, Any]] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        command = str(parts[2] or "").strip()
        if not command:
            continue
        rows.append({"pid": int(pid), "ppid": int(ppid), "command": command})
    return rows


def _discover_runtime_processes(scripts: Dict[str, str]) -> Dict[str, Any]:
    runner_script = os.path.abspath(os.path.join(BASE_DIR, "runtime", "pt_runner.py"))
    targets = {
        "thinker": os.path.abspath(str(scripts.get("thinker", ""))),
        "trader": os.path.abspath(str(scripts.get("trader", ""))),
        "markets": os.path.abspath(str(scripts.get("markets", ""))),
        "autopilot": os.path.abspath(str(scripts.get("autopilot", ""))),
    }
    rows = _ps_process_rows()
    pid_to_command: Dict[int, str] = {}
    runner_pids: set[int] = set()
    child_rows: list[Dict[str, Any]] = []
    for row in rows:
        try:
            pid = int(row.get("pid", 0) or 0)
            ppid = int(row.get("ppid", 0) or 0)
            command = str(row.get("command", "") or "")
        except Exception:
            continue
        if pid <= 0 or (not command):
            continue
        pid_to_command[pid] = command
        if runner_script and (runner_script in command):
            runner_pids.add(pid)
        for role, script_path in targets.items():
            if script_path and (script_path in command):
                child_rows.append(
                    {
                        "role": str(role),
                        "pid": int(pid),
                        "ppid": int(ppid),
                        "command": command,
                    }
                )
                break
    return {
        "runner_script": runner_script,
        "runner_pids": sorted(runner_pids),
        "children": list(child_rows),
        "pid_to_command": pid_to_command,
    }


def _terminate_pid(pid: int, force: bool = False) -> bool:
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.kill(int(pid), sig)
        return True
    except Exception:
        return False


def _cleanup_orphan_runtime_children(scripts: Dict[str, str], keep_runner_pids: Optional[set[int]] = None) -> Dict[str, Any]:
    keep = {int(p) for p in (keep_runner_pids or set()) if int(p) > 0}
    proc_map = _discover_runtime_processes(scripts)
    runner_pids = {int(p) for p in list(proc_map.get("runner_pids", []) or []) if int(p) > 0}
    keep |= runner_pids
    pid_to_command = proc_map.get("pid_to_command", {}) if isinstance(proc_map.get("pid_to_command"), dict) else {}
    children = list(proc_map.get("children", []) or [])
    stale_rows: list[Dict[str, Any]] = []
    for row in children:
        try:
            pid = int(row.get("pid", 0) or 0)
            ppid = int(row.get("ppid", 0) or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        if ppid in keep:
            continue
        parent_cmd = str(pid_to_command.get(ppid, "") or "")
        parent_is_runner = ("runtime/pt_runner.py" in parent_cmd) if parent_cmd else False
        if parent_is_runner:
            continue
        parent_alive = _pid_is_alive(ppid) if ppid > 0 else False
        if (ppid <= 1) or (not parent_alive) or (not parent_is_runner):
            stale_rows.append(dict(row))

    terminated = 0
    forced = 0
    stale_pids = [int(r.get("pid", 0) or 0) for r in stale_rows if int(r.get("pid", 0) or 0) > 0]
    for pid in stale_pids:
        if _terminate_pid(pid, force=False):
            terminated += 1
    if stale_pids:
        time.sleep(0.3)
    for pid in stale_pids:
        if _pid_is_alive(pid):
            if _terminate_pid(pid, force=True):
                forced += 1

    return {
        "stale_count": int(len(stale_rows)),
        "terminated": int(terminated),
        "forced": int(forced),
        "runner_pids": sorted(int(p) for p in runner_pids),
        "stale_children": [
            {
                "role": str(r.get("role", "") or ""),
                "pid": int(r.get("pid", 0) or 0),
                "ppid": int(r.get("ppid", 0) or 0),
            }
            for r in stale_rows
        ],
    }


def _live_runner_pids_from_discovery(discovered: Dict[str, Any], self_pid: Optional[int] = None) -> list[int]:
    try:
        self_pid_i = int(os.getpid() if self_pid is None else self_pid)
    except Exception:
        self_pid_i = -1
    out: set[int] = set()
    for raw_pid in list((discovered or {}).get("runner_pids", []) or []):
        try:
            pid = int(raw_pid)
        except Exception:
            continue
        if pid <= 0 or pid == self_pid_i:
            continue
        if _pid_is_alive(pid):
            out.add(pid)
    return sorted(int(p) for p in out)


def _settings_scripts() -> Dict[str, str]:
    settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
    data = read_settings_file(settings_path, module_name="pt_runner") or {}
    data = sanitize_settings(data if isinstance(data, dict) else {})
    thinker_name = str(data.get("script_neural_runner2", "engines/pt_thinker.py") or "engines/pt_thinker.py").strip()
    trader_name = str(data.get("script_trader", "engines/pt_trader.py") or "engines/pt_trader.py").strip()
    markets_name = str(data.get("script_markets_runner", "runtime/pt_markets.py") or "runtime/pt_markets.py").strip()
    autopilot_name = str(data.get("script_autopilot", "runtime/pt_autopilot.py") or "runtime/pt_autopilot.py").strip()
    return {
        "thinker": os.path.abspath(os.path.join(BASE_DIR, thinker_name)),
        "trader": os.path.abspath(os.path.join(BASE_DIR, trader_name)),
        "markets": os.path.abspath(os.path.join(BASE_DIR, markets_name)),
        "autopilot": os.path.abspath(os.path.join(BASE_DIR, autopilot_name)),
    }


def _terminate_process(proc: Optional[subprocess.Popen], name: str, force: bool = False) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except Exception as exc:
        _runner_log(f"{name}: terminate error {type(exc).__name__}: {exc}")


class ChildSpec:
    def __init__(self, name: str, script_path: str, log_path: str) -> None:
        self.name = name
        self.script_path = script_path
        self.log_path = log_path
        self.log_handle: Optional[TextIO] = None
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.backoff_s = 1.0
        self.next_restart_at = 0.0
        self.lockout_until = 0.0
        self.crash_times: list[float] = []
        self.last_exit: Dict[str, Any] = {}
        self.started_at = 0.0
        self.loaded_script_mtime = 0.0
        self.last_script_change_restart_at = 0.0

    def pid(self) -> Optional[int]:
        if self.proc and self.proc.poll() is None:
            return int(self.proc.pid)
        return None


class Runner:
    def __init__(self) -> None:
        scripts = _settings_scripts()
        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        sdata = read_settings_file(settings_path, module_name="pt_runner") or {}
        sdata = sanitize_settings(sdata if isinstance(sdata, dict) else {})
        try:
            self.crash_lockout_s = max(30.0, float(sdata.get("runner_crash_lockout_s", CRASH_LOCKOUT_S) or CRASH_LOCKOUT_S))
        except Exception:
            self.crash_lockout_s = float(CRASH_LOCKOUT_S)
        self.children = {
            "thinker": ChildSpec("thinker", scripts["thinker"], THINKER_LOG_PATH),
            "trader": ChildSpec("trader", scripts["trader"], TRADER_LOG_PATH),
            "markets": ChildSpec("markets", scripts["markets"], MARKETS_LOG_PATH),
            "autopilot": ChildSpec("autopilot", scripts["autopilot"], AUTOPILOT_LOG_PATH),
        }
        self.state = "RUNNING"
        self.msg = "Supervisor starting"
        self.running = True
        self.current_backoff_s = 0.0
        self._last_log_cleanup_at = 0.0
        self._last_watchdog_at = 0.0
        self._last_market_loop_stale_note_at = 0.0
        self._last_market_loop_restart_at = 0.0
        self._last_market_loop_phase_timeout_sig = ""
        self._last_market_loop_phase_timeout_at = 0.0
        self._watchdog_resume_grace_until = 0.0
        self._last_script_watch_at = 0.0
        self._sleep_guard_proc: Optional[subprocess.Popen] = None
        self._sleep_guard_last_start_at = 0.0
        self._sleep_guard_warned_unavailable = False
        self._sleep_guard_warned_disabled = False

    def __del__(self) -> None:
        # Best-effort cleanup for tests and short-lived invocations that do not call run().
        try:
            self._stop_sleep_guard()
        except Exception:
            pass

    def write_heartbeat(self) -> None:
        payload = {
            "state": self.state,
            "ts": int(time.time()),
            "runner_pid": int(os.getpid()),
            "thinker_pid": self.children["thinker"].pid(),
            "trader_pid": self.children["trader"].pid(),
            "markets_pid": self.children["markets"].pid(),
            "autopilot_pid": self.children["autopilot"].pid(),
            "restarts": {
                "thinker": int(self.children["thinker"].restarts),
                "trader": int(self.children["trader"].restarts),
                "markets": int(self.children["markets"].restarts),
                "autopilot": int(self.children["autopilot"].restarts),
            },
            "last_exit": {
                "thinker": self.children["thinker"].last_exit or None,
                "trader": self.children["trader"].last_exit or None,
                "markets": self.children["markets"].last_exit or None,
                "autopilot": self.children["autopilot"].last_exit or None,
            },
            "backoff_s": float(self.current_backoff_s),
            "msg": self.msg,
        }
        try:
            _atomic_write_json(TRADER_STATUS_PATH, payload)
        except Exception as exc:
            _runner_log(f"heartbeat write failed {type(exc).__name__}: {exc}")
        try:
            self._write_runtime_state(payload)
        except Exception:
            pass

    def _incident_summary(self, limit: int = 200) -> Dict[str, Any]:
        sev_counts: Dict[str, int] = {}
        sev_counts_1h: Dict[str, int] = {}
        event_counts: Dict[str, int] = {}
        event_sev_counts: Dict[str, Dict[str, int]] = {}
        event_sev_counts_1h: Dict[str, Dict[str, int]] = {}
        lines: list[str] = []
        now_ts = int(time.time())
        count_1h = 0
        count_24h = 0
        try:
            with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            return {"count": 0, "by_severity": {}, "top_events": []}

        for ln in lines[-max(1, int(limit)):]:
            try:
                row = json_loads(ln, default=None)
            except Exception:
                continue
            sev = str(row.get("severity", "info") or "info").strip().lower()
            evt = str(row.get("event", "") or "").strip().lower()
            sev_counts[sev] = int(sev_counts.get(sev, 0)) + 1
            if evt:
                event_counts[evt] = int(event_counts.get(evt, 0)) + 1
                evt_sev = event_sev_counts.get(evt, {})
                if not isinstance(evt_sev, dict):
                    evt_sev = {}
                evt_sev[sev] = int(evt_sev.get(sev, 0) or 0) + 1
                event_sev_counts[evt] = evt_sev
            try:
                ts = int(float(row.get("ts", 0) or 0))
            except Exception:
                ts = 0
            if ts > 0 and (now_ts - ts) <= 3600:
                count_1h += 1
                sev_counts_1h[sev] = int(sev_counts_1h.get(sev, 0) or 0) + 1
                if evt:
                    evt_sev_1h = event_sev_counts_1h.get(evt, {})
                    if not isinstance(evt_sev_1h, dict):
                        evt_sev_1h = {}
                    evt_sev_1h[sev] = int(evt_sev_1h.get(sev, 0) or 0) + 1
                    event_sev_counts_1h[evt] = evt_sev_1h
            if ts > 0 and (now_ts - ts) <= 86400:
                count_24h += 1
        top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        return {
            "count": int(len(lines[-max(1, int(limit)):])),
            "by_severity": sev_counts,
            "by_severity_1h": sev_counts_1h,
            "by_event_severity": event_sev_counts,
            "by_event_severity_1h": event_sev_counts_1h,
            "top_events": [{"event": k, "count": int(v)} for k, v in top_events],
            "count_1h": int(count_1h),
            "count_24h": int(count_24h),
        }

    def _write_runtime_state(self, heartbeat_payload: Dict[str, Any]) -> None:
        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_runner") or {})
        checks = _safe_read_json(RUNTIME_CHECKS_PATH)
        check_errors = list(checks.get("errors", []) or []) if isinstance(checks.get("errors", []), list) else []
        check_warnings = list(checks.get("warnings", []) or []) if isinstance(checks.get("warnings", []), list) else []
        check_ok = bool(checks.get("ok", False))
        # Self-heal missing/corrupt startup-check artifacts so alerts do not stay stuck on a
        # false critical state after data/cache cleanup.
        if (not check_ok) and (not check_errors) and (not check_warnings):
            checks = {"ts": now_ts(), "ok": True, "errors": [], "warnings": []}
            try:
                _atomic_write_json(RUNTIME_CHECKS_PATH, checks)
            except Exception:
                pass
        stock_diag = normalize_scan_diagnostics(
            _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "scan_diagnostics.json")),
            market="stocks",
        )
        forex_diag = normalize_scan_diagnostics(
            _safe_read_json(os.path.join(HUB_DATA_DIR, "forex", "scan_diagnostics.json")),
            market="forex",
        )
        sla = _safe_read_json(os.path.join(HUB_DATA_DIR, "market_sla_metrics.json"))
        scan_drift = _safe_read_json(os.path.join(HUB_DATA_DIR, "scan_drift_alerts.json"))
        scan_cadence = _safe_read_json(CADENCE_DRIFT_PATH)
        trends = _safe_read_json(os.path.join(HUB_DATA_DIR, "market_trends.json"))
        regimes = _safe_read_json(MARKET_REGIMES_PATH)
        walkforward = _safe_read_json(WALKFORWARD_REPORT_PATH)
        confidence_calibration = _safe_read_json(CONFIDENCE_CALIBRATION_PATH)
        shadow_scorecards = _safe_read_json(SHADOW_SCORECARDS_PATH)
        exec_guard = _safe_read_json(os.path.join(HUB_DATA_DIR, "broker_execution_guard.json"))
        drawdown_guard = _safe_read_json(DRAWDOWN_GUARD_PATH)
        market_loop = _safe_read_json(MARKET_LOOP_STATUS_PATH)
        key_rotation = _safe_read_json(KEY_ROTATION_STATUS_PATH)
        if not isinstance(key_rotation, dict) or (not key_rotation):
            key_rotation = {"ts": now_ts(), "warn_days": int(settings.get("key_rotation_warn_days", 90) or 90), "due": [], "due_count": 0}
            try:
                _atomic_write_json(KEY_ROTATION_STATUS_PATH, key_rotation)
            except Exception:
                pass
        autopilot = _safe_read_json(os.path.join(HUB_DATA_DIR, "autopilot_status.json"))
        stock_broker = _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "alpaca_status.json"))
        forex_broker = _safe_read_json(os.path.join(HUB_DATA_DIR, "forex", "oanda_status.json"))
        stock_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "stocks", "stock_trader_status.json"))
        forex_trader = _safe_read_json(os.path.join(HUB_DATA_DIR, "forex", "forex_trader_status.json"))
        crypto_trader = _safe_read_json(CRYPTO_TRADER_DETAIL_PATH)
        automation_policy = summarize_policy_snapshot(
            stock_trader if isinstance(stock_trader, dict) else {},
            forex_trader if isinstance(forex_trader, dict) else {},
            crypto_trader if isinstance(crypto_trader, dict) else {},
        )
        cross_market_opportunity = summarize_allocator_snapshot(
            stock_trader if isinstance(stock_trader, dict) else {},
            forex_trader if isinstance(forex_trader, dict) else {},
            crypto_trader if isinstance(crypto_trader, dict) else {},
        )
        status = _safe_read_json(TRADER_STATUS_PATH)
        incidents_rows = _read_jsonl_tail(INCIDENTS_PATH, limit=800)
        runtime_event_rows = _read_jsonl_tail(RUNTIME_EVENTS_PATH, limit=2000)
        account_history_rows = _read_jsonl_tail(os.path.join(HUB_DATA_DIR, "account_value_history.jsonl"), limit=5000)
        stock_audit_rows = _read_jsonl_tail(os.path.join(HUB_DATA_DIR, "stocks", "execution_audit.jsonl"), limit=3000)
        forex_audit_rows = _read_jsonl_tail(os.path.join(HUB_DATA_DIR, "forex", "execution_audit.jsonl"), limit=3000)
        try:
            quota_warn = max(1, int(float(settings.get("runtime_api_quota_warn_15m", 4) or 4)))
        except Exception:
            quota_warn = 4
        try:
            quota_crit = max(quota_warn, int(float(settings.get("runtime_api_quota_crit_15m", 10) or 10)))
        except Exception:
            quota_crit = 10
        api_quota = summarize_quota_events(
            incidents_rows,
            now_ts=time.time(),
            warn_15m=quota_warn,
            crit_15m=quota_crit,
        )
        api_quota["thresholds"] = {"warn_15m": int(quota_warn), "crit_15m": int(quota_crit)}
        broker_backoff = _summarize_broker_backoff_events(runtime_event_rows, now_ts_value=time.time())
        latency_hist = build_broker_latency_histogram(
            runtime_event_rows,
            market_audit_rows={"stocks": stock_audit_rows, "forex": forex_audit_rows},
            now_ts_value=time.time(),
        )
        incident_trend = build_incident_trend(incidents_rows, now_ts_value=time.time())
        pnl_decomposition = build_pnl_decomposition(HUB_DATA_DIR)
        equity_anomaly = detect_equity_anomaly(
            account_history_rows,
            now_ts_value=time.time(),
            spike_pct=max(1.0, float(settings.get("equity_curve_anomaly_spike_pct", 3.0) or 3.0)),
        )
        stale_history = detect_stale_history(
            account_history_rows,
            now_ts_value=time.time(),
            stale_after_s=max(60, int(float(settings.get("runtime_alert_history_stale_s", 900) or 900))),
        )
        feature_flags = build_feature_flag_snapshot(settings)

        def _broker_state(name: str, payload: Dict[str, Any], quota_row: Dict[str, Any]) -> Dict[str, Any]:
            st = str(payload.get("state", "") or "").upper().strip()
            msg = str(payload.get("msg", "") or "").strip()
            quota_state = str(quota_row.get("status", "ok") or "ok").strip().lower()
            state = "ok"
            if st in {"ERROR", "NOT CONFIGURED"}:
                state = "error"
            elif quota_state == "critical":
                state = "error"
            elif quota_state == "warning":
                state = "warning"
            return {
                "name": name,
                "state": state,
                "status_text": st or "UNKNOWN",
                "quota_15m": int(quota_row.get("count_15m", 0) or 0),
                "quota_60m": int(quota_row.get("count_60m", 0) or 0),
                "quota_last_ts": int(quota_row.get("last_ts", 0) or 0),
                "msg": msg[:240],
            }

        qmap = api_quota.get("by_component", {}) if isinstance(api_quota.get("by_component", {}), dict) else {}
        broker_health = {
            "alpaca": _broker_state("Alpaca", stock_broker, qmap.get("alpaca", {}) if isinstance(qmap.get("alpaca", {}), dict) else {}),
            "oanda": _broker_state("OANDA", forex_broker, qmap.get("oanda", {}) if isinstance(qmap.get("oanda", {}), dict) else {}),
            "kucoin": _broker_state(
                "KuCoin",
                {"state": ("ERROR" if bool(autopilot.get("api_unstable", False)) else "READY"), "msg": ""},
                qmap.get("kucoin", {}) if isinstance(qmap.get("kucoin", {}), dict) else {},
            ),
        }

        payload = {
            "runtime_state_schema": {
                "version": int(RUNTIME_STATE_SCHEMA_VERSION),
                "min_reader_version": int(RUNTIME_STATE_MIN_READER_VERSION),
            },
            "ts": int(time.time()),
            "runner": {
                "state": str(heartbeat_payload.get("state", "") or ""),
                "msg": str(heartbeat_payload.get("msg", "") or ""),
                "pid": int(os.getpid()),
                "children": {
                    "thinker": status.get("thinker_pid"),
                    "trader": status.get("trader_pid"),
                    "markets": status.get("markets_pid"),
                    "autopilot": status.get("autopilot_pid"),
                },
                "restarts": dict(heartbeat_payload.get("restarts", {}) or {}),
                "last_exit": dict(heartbeat_payload.get("last_exit", {}) or {}),
            },
            "checks": {
                "ok": bool(checks.get("ok", False)),
                "errors": list(checks.get("errors", []) or []),
                "warnings": list(checks.get("warnings", []) or []),
            },
            "scan_health": {
                "stocks": {
                    "state": str(stock_diag.get("state", "") or ""),
                    "leaders_total": int(stock_diag.get("leaders_total", 0) or 0),
                    "scores_total": int(stock_diag.get("scores_total", 0) or 0),
                    "reject_rate_pct": float(((stock_diag.get("reject_summary", {}) or {}).get("reject_rate_pct", 0.0) or 0.0)),
                    "reject_dominant_reason": str(((stock_diag.get("reject_summary", {}) or {}).get("dominant_reason", "") or "")),
                    "reject_dominant_ratio_pct": float(((stock_diag.get("reject_summary", {}) or {}).get("dominant_ratio_pct", 0.0) or 0.0)),
                },
                "forex": {
                    "state": str(forex_diag.get("state", "") or ""),
                    "leaders_total": int(forex_diag.get("leaders_total", 0) or 0),
                    "scores_total": int(forex_diag.get("scores_total", 0) or 0),
                    "reject_rate_pct": float(((forex_diag.get("reject_summary", {}) or {}).get("reject_rate_pct", 0.0) or 0.0)),
                    "reject_dominant_reason": str(((forex_diag.get("reject_summary", {}) or {}).get("dominant_reason", "") or "")),
                    "reject_dominant_ratio_pct": float(((forex_diag.get("reject_summary", {}) or {}).get("dominant_ratio_pct", 0.0) or 0.0)),
                },
            },
            "sla_metrics": dict((sla.get("metrics", {}) if isinstance(sla.get("metrics", {}), dict) else {})),
            "scan_drift": {
                "active": list(scan_drift.get("active", []) or []) if isinstance(scan_drift.get("active", []), list) else [],
                "markets": dict(scan_drift.get("markets", {}) or {}) if isinstance(scan_drift.get("markets", {}), dict) else {},
                "ts": int(scan_drift.get("ts", 0) or 0),
            },
            "scan_cadence": {
                "active": list(scan_cadence.get("active", []) or []) if isinstance(scan_cadence.get("active", []), list) else [],
                "markets": dict(scan_cadence.get("markets", {}) or {}) if isinstance(scan_cadence.get("markets", {}), dict) else {},
                "ts": int(scan_cadence.get("ts", 0) or 0),
            },
            "execution_guard": {
                "markets": dict(exec_guard.get("markets", {}) or {}) if isinstance(exec_guard.get("markets", {}), dict) else {},
                "ts": int(exec_guard.get("ts", 0) or 0),
            },
            "drawdown_guard": {
                "triggered_ts": int(drawdown_guard.get("triggered_ts", 0) or 0),
                "drawdown_pct": float(drawdown_guard.get("drawdown_pct", 0.0) or 0.0),
                "limit_pct": float(drawdown_guard.get("limit_pct", 0.0) or 0.0),
                "lookback_hours": int(drawdown_guard.get("lookback_hours", 0) or 0),
                "triggered_recent": bool(
                    int(drawdown_guard.get("triggered_ts", 0) or 0) > 0
                    and (int(time.time()) - int(drawdown_guard.get("triggered_ts", 0) or 0)) <= 86400
                ),
            },
            "stop_flag": _stop_flag_payload(STOP_FLAG_PATH),
            "key_rotation": {
                "warn_days": int(key_rotation.get("warn_days", 0) or 0),
                "due": list(key_rotation.get("due", []) or []) if isinstance(key_rotation.get("due", []), list) else [],
                "due_count": int(key_rotation.get("due_count", 0) or 0),
                "ts": int(key_rotation.get("ts", 0) or 0),
            },
            "market_loop": {
                "ts": int(market_loop.get("ts", 0) or 0),
                "age_s": (
                    max(0, int(time.time()) - int(market_loop.get("ts", 0) or 0))
                    if int(market_loop.get("ts", 0) or 0) > 0
                    else -1
                ),
                "stocks_last_scan_ts": int(market_loop.get("stocks_last_scan_ts", 0) or 0),
                "forex_last_scan_ts": int(market_loop.get("forex_last_scan_ts", 0) or 0),
                "stocks_last_step_ts": int(market_loop.get("stocks_last_step_ts", 0) or 0),
                "forex_last_step_ts": int(market_loop.get("forex_last_step_ts", 0) or 0),
                "stocks_cadence": dict(
                    ((market_loop.get("stocks_cycle", {}) if isinstance(market_loop.get("stocks_cycle", {}), dict) else {}).get("cadence", {}) or {})
                    if isinstance((market_loop.get("stocks_cycle", {}) if isinstance(market_loop.get("stocks_cycle", {}), dict) else {}).get("cadence", {}), dict)
                    else {}
                ),
                "forex_cadence": dict(
                    ((market_loop.get("forex_cycle", {}) if isinstance(market_loop.get("forex_cycle", {}), dict) else {}).get("cadence", {}) or {})
                    if isinstance((market_loop.get("forex_cycle", {}) if isinstance(market_loop.get("forex_cycle", {}), dict) else {}).get("cadence", {}), dict)
                    else {}
                ),
            },
            "market_trends": {
                "stocks": dict(trends.get("stocks", {}) or {}) if isinstance(trends.get("stocks", {}), dict) else {},
                "forex": dict(trends.get("forex", {}) or {}) if isinstance(trends.get("forex", {}), dict) else {},
                "ts": int(trends.get("ts", 0) or 0),
            },
            "market_regimes": {
                "stocks": dict(regimes.get("stocks", {}) or {}) if isinstance(regimes.get("stocks", {}), dict) else {},
                "forex": dict(regimes.get("forex", {}) or {}) if isinstance(regimes.get("forex", {}), dict) else {},
                "ts": int(regimes.get("ts", 0) or 0),
            },
            "walkforward_report": {
                "stocks": dict(walkforward.get("stocks", {}) or {}) if isinstance(walkforward.get("stocks", {}), dict) else {},
                "forex": dict(walkforward.get("forex", {}) or {}) if isinstance(walkforward.get("forex", {}), dict) else {},
                "ts": int(walkforward.get("ts", 0) or 0),
            },
            "confidence_calibration": {
                "stocks": dict(confidence_calibration.get("stocks", {}) or {})
                if isinstance(confidence_calibration.get("stocks", {}), dict)
                else {},
                "forex": dict(confidence_calibration.get("forex", {}) or {})
                if isinstance(confidence_calibration.get("forex", {}), dict)
                else {},
                "ts": int(confidence_calibration.get("ts", 0) or 0),
            },
            "shadow_scorecards": {
                "stocks": dict(shadow_scorecards.get("stocks", {}) or {}) if isinstance(shadow_scorecards.get("stocks", {}), dict) else {},
                "forex": dict(shadow_scorecards.get("forex", {}) or {}) if isinstance(shadow_scorecards.get("forex", {}), dict) else {},
                "all_markets_pass": bool(shadow_scorecards.get("all_markets_pass", False)),
                "ts": int(shadow_scorecards.get("ts", 0) or 0),
            },
            "exposure_map": build_exposure_payload(HUB_DATA_DIR),
            "autopilot": {
                "stable_cycles": int(autopilot.get("stable_cycles", 0) or 0),
                "api_unstable": bool(autopilot.get("api_unstable", False)),
                "markets_healthy": bool(autopilot.get("markets_healthy", False)),
                "issue_open": bool(autopilot.get("issue_open", False)),
            },
            "api_quota": api_quota,
            "broker_backoff": broker_backoff,
            "broker_latency_histogram": latency_hist,
            "broker_health": broker_health,
            "incidents_last_200": self._incident_summary(limit=200),
            "incident_trend": incident_trend,
            "pnl_decomposition": pnl_decomposition,
            "equity_curve_anomaly": equity_anomaly,
            "stale_history": stale_history,
            "feature_flags": feature_flags,
            "automation_policy": {
                "ts": int(time.time()),
                "crypto": dict(automation_policy.get("crypto", {}) or {}) if isinstance(automation_policy.get("crypto", {}), dict) else {},
                "stocks": dict(automation_policy.get("stocks", {}) or {}) if isinstance(automation_policy.get("stocks", {}), dict) else {},
                "forex": dict(automation_policy.get("forex", {}) or {}) if isinstance(automation_policy.get("forex", {}), dict) else {},
            },
            "cross_market_opportunity": (
                dict(cross_market_opportunity)
                if isinstance(cross_market_opportunity, dict)
                else {}
            ),
        }
        payload["alerts"] = evaluate_runtime_alerts(payload, settings)
        notifications = build_notification_center_payload(payload, incidents_rows=incidents_rows, max_items=240)
        payload["notification_center"] = notifications
        _atomic_write_json(NOTIFICATION_CENTER_PATH, notifications)
        _atomic_write_json(RUNTIME_STATE_PATH, payload)

    def _status_file_stale(self, path: str, max_age_s: float) -> bool:
        try:
            st = os.stat(path)
            age = time.time() - float(st.st_mtime)
            return age > float(max_age_s)
        except Exception:
            return True

    def _market_loop_watchdog_stale_after(self, settings: Dict[str, Any], floor_s: float = 150.0) -> float:
        try:
            configured_s = float(settings.get("runner_market_loop_watchdog_stale_after_s", 0.0) or 0.0)
        except Exception:
            configured_s = 0.0
        try:
            snapshot_interval_s = max(5.0, float(settings.get("market_bg_snapshot_interval_s", 15.0) or 15.0))
        except Exception:
            snapshot_interval_s = 15.0
        try:
            stock_interval_s = max(8.0, float(settings.get("market_bg_stocks_interval_s", 18.0) or 18.0))
        except Exception:
            stock_interval_s = 18.0
        try:
            forex_interval_s = max(6.0, float(settings.get("market_bg_forex_interval_s", 12.0) or 12.0))
        except Exception:
            forex_interval_s = 12.0

        derived_s = snapshot_interval_s + stock_interval_s + forex_interval_s + 10.0
        runtime_state = _safe_read_json(RUNTIME_STATE_PATH)
        sla_metrics = runtime_state.get("sla_metrics", {}) if isinstance(runtime_state.get("sla_metrics", {}), dict) else {}
        phase_budget_s = 0.0
        for key in (
            "stocks_snapshot",
            "forex_snapshot",
            "stocks_scan",
            "forex_scan",
            "stocks_trader_step",
            "forex_trader_step",
        ):
            row = sla_metrics.get(key, {}) if isinstance(sla_metrics, dict) else {}
            if not isinstance(row, dict):
                continue
            samples_ms: list[float] = []
            for field in ("p95_ms", "last_ms"):
                try:
                    val = float(row.get(field, 0.0) or 0.0)
                except Exception:
                    val = 0.0
                if val > 0.0:
                    samples_ms.append(val)
            if not samples_ms:
                continue
            phase_budget_s += min(max(samples_ms) / 1000.0, 180.0)
        if phase_budget_s > 0.0:
            derived_s = max(derived_s, phase_budget_s + 10.0)

        base_s = max(
            float(floor_s or 0.0),
            forex_interval_s * MARKETS_STALE_MULT,
            float(configured_s if configured_s > 0.0 else 0.0),
            float(derived_s),
        )
        return min(max(30.0, base_s), 600.0)

    def _market_loop_phase_timeout_s(
        self,
        phase: str,
        runtime_state: Dict[str, Any],
        base_stale_after: float,
    ) -> float:
        phase_key = str(phase or "").strip().lower()
        metric_map = {
            "snapshots": ("stocks_snapshot", "forex_snapshot"),
            "stocks_scan": ("stocks_scan",),
            "stocks_step": ("stocks_trader_step",),
            "forex_scan": ("forex_scan",),
            "forex_step": ("forex_trader_step",),
            "intelligence": (),
        }
        metrics = runtime_state.get("sla_metrics", {}) if isinstance(runtime_state.get("sla_metrics", {}), dict) else {}
        phase_budget_s = 0.0
        for metric_key in metric_map.get(phase_key, ()):
            row = metrics.get(metric_key, {}) if isinstance(metrics, dict) else {}
            if not isinstance(row, dict):
                continue
            samples_ms: list[float] = []
            for field in ("p95_ms", "last_ms"):
                try:
                    val = float(row.get(field, 0.0) or 0.0)
                except Exception:
                    val = 0.0
                if val > 0.0:
                    samples_ms.append(val)
            if samples_ms:
                phase_budget_s += min(max(samples_ms) / 1000.0, 300.0)
        if phase_key == "intelligence":
            phase_budget_s = max(phase_budget_s, 120.0)
        if phase_budget_s <= 0.0:
            return min(max(float(base_stale_after or 0.0), 240.0), 900.0)
        return min(max(float(base_stale_after or 0.0), (phase_budget_s * 2.0) + 30.0), 900.0)

    def _market_loop_watchdog_state(
        self,
        settings: Dict[str, Any],
        now: float,
        base_stale_after: float,
    ) -> tuple[bool, float, str, str, int, float]:
        loop_status = _safe_read_json(MARKET_LOOP_STATUS_PATH)
        phase = str(loop_status.get("phase", "idle") or "idle").strip().lower() if isinstance(loop_status, dict) else "idle"
        phase_started_ts = int(loop_status.get("phase_started_ts", 0) or 0) if isinstance(loop_status, dict) else 0
        phase_age_s = max(0.0, float(now) - float(phase_started_ts)) if phase_started_ts > 0 else 0.0
        if self._status_file_stale(MARKET_LOOP_STATUS_PATH, base_stale_after):
            return True, float(base_stale_after), phase, "heartbeat_stale", int(phase_started_ts), float(phase_age_s)
        if not isinstance(loop_status, dict) or not loop_status:
            return False, float(base_stale_after), phase, "ok", int(phase_started_ts), float(phase_age_s)
        if phase in {"", "idle"} or phase_started_ts <= 0:
            return False, float(base_stale_after), phase, "ok", int(phase_started_ts), float(phase_age_s)
        runtime_state = _safe_read_json(RUNTIME_STATE_PATH)
        phase_timeout_s = self._market_loop_phase_timeout_s(phase, runtime_state, base_stale_after)
        if phase_age_s >= phase_timeout_s:
            return True, float(phase_timeout_s), phase, "phase_timeout", int(phase_started_ts), float(phase_age_s)
        return False, float(phase_timeout_s), phase, "ok", int(phase_started_ts), float(phase_age_s)

    def _child_uptime_s(self, child: Optional[ChildSpec], now: float) -> float:
        if child is None or child.proc is None or child.proc.poll() is not None:
            return 0.0
        started_at = float(getattr(child, "started_at", 0.0) or 0.0)
        if started_at <= 0.0:
            return 1e9
        return max(0.0, float(now) - started_at)

    def _watchdog_tick(self, now: float) -> None:
        prev_watchdog_at = float(self._last_watchdog_at or 0.0)
        if (prev_watchdog_at > 0.0) and ((now - prev_watchdog_at) < WATCHDOG_INTERVAL_S):
            return
        tick_gap_s = max(0.0, float(now) - prev_watchdog_at) if prev_watchdog_at > 0.0 else 0.0
        self._last_watchdog_at = now

        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_runner") or {})
        self._ensure_sleep_guard(settings, now)
        try:
            sleep_gap_detect_s = max(
                30.0,
                float(settings.get("runner_sleep_gap_detect_s", WATCHDOG_SLEEP_GAP_DETECT_S) or WATCHDOG_SLEEP_GAP_DETECT_S),
            )
        except Exception:
            sleep_gap_detect_s = float(WATCHDOG_SLEEP_GAP_DETECT_S)
        try:
            sleep_resume_grace_s = max(
                30.0,
                float(
                    settings.get("runner_sleep_resume_grace_s", WATCHDOG_SLEEP_RESUME_GRACE_S) or WATCHDOG_SLEEP_RESUME_GRACE_S
                ),
            )
        except Exception:
            sleep_resume_grace_s = float(WATCHDOG_SLEEP_RESUME_GRACE_S)
        if prev_watchdog_at > 0.0 and tick_gap_s >= sleep_gap_detect_s:
            self._watchdog_resume_grace_until = max(self._watchdog_resume_grace_until, float(now) + float(sleep_resume_grace_s))
            msg = (
                f"system sleep/standby gap detected ({int(tick_gap_s)}s); "
                f"suppressing watchdog restarts for {int(sleep_resume_grace_s)}s"
            )
            self.msg = msg
            _runner_log(msg)
            _append_incident(
                "info",
                "runner_sleep_resume_detected",
                msg,
                {
                    "gap_s": round(float(tick_gap_s), 3),
                    "grace_s": round(float(sleep_resume_grace_s), 3),
                },
            )
        if float(now) < float(self._watchdog_resume_grace_until):
            return

        market_interval = max(6.0, float(settings.get("market_bg_forex_interval_s", 12.0) or 12.0))
        autopilot_interval = 30.0
        try:
            market_startup_grace_s = max(
                20.0,
                float(settings.get("runner_market_watchdog_startup_grace_s", 120.0) or 120.0),
            )
        except Exception:
            market_startup_grace_s = 120.0
        try:
            autopilot_startup_grace_s = max(
                20.0,
                float(settings.get("runner_autopilot_watchdog_startup_grace_s", 120.0) or 120.0),
            )
        except Exception:
            autopilot_startup_grace_s = 120.0
        try:
            market_loop_startup_grace_s = max(
                market_startup_grace_s,
                float(settings.get("runner_market_loop_startup_grace_s", 240.0) or 240.0),
            )
        except Exception:
            market_loop_startup_grace_s = max(market_startup_grace_s, 240.0)
        market_loop_stale_after = self._market_loop_watchdog_stale_after(settings, floor_s=market_loop_startup_grace_s)
        try:
            raw_phase_timeout_restart_grace_s = settings.get(
                "runner_market_loop_phase_timeout_restart_grace_s",
                MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_S,
            )
            phase_timeout_restart_grace_s = max(
                0.0,
                float(
                    MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_S
                    if raw_phase_timeout_restart_grace_s in (None, "")
                    else raw_phase_timeout_restart_grace_s
                ),
            )
        except Exception:
            phase_timeout_restart_grace_s = float(MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_S)
        try:
            raw_phase_timeout_restart_grace_mult = settings.get(
                "runner_market_loop_phase_timeout_restart_grace_mult",
                MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_MULT,
            )
            phase_timeout_restart_grace_mult = max(
                0.0,
                float(
                    MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_MULT
                    if raw_phase_timeout_restart_grace_mult in (None, "")
                    else raw_phase_timeout_restart_grace_mult
                ),
            )
        except Exception:
            phase_timeout_restart_grace_mult = float(MARKET_LOOP_PHASE_TIMEOUT_RESTART_GRACE_MULT)

        targets = [
            (
                "autopilot",
                os.path.join(HUB_DATA_DIR, "autopilot_status.json"),
                autopilot_interval * AUTOPILOT_STALE_MULT,
            ),
        ]

        for key, path, stale_after in targets:
            child = self.children.get(key)
            if not child or not child.proc or child.proc.poll() is not None:
                continue
            startup_grace_s = market_startup_grace_s if key == "markets" else autopilot_startup_grace_s
            if self._child_uptime_s(child, now) < startup_grace_s:
                continue
            if not self._status_file_stale(path, stale_after):
                continue
            msg = f"{key} appears hung (stale status>{int(stale_after)}s); restarting"
            self.msg = msg
            _runner_log(msg)
            _append_incident("warning", "runner_watchdog_restart", msg, {"child": key, "status_path": path})
            _terminate_process(child.proc, key, force=False)

        # Extra signal: markets loop heartbeat is stale even if process has not yet been restarted.
        markets_child = self.children.get("markets")
        if markets_child and markets_child.proc and markets_child.proc.poll() is None:
            loop_stale_after = market_loop_stale_after
            if self._child_uptime_s(markets_child, now) < market_loop_startup_grace_s:
                return
            if (not os.path.isfile(MARKET_LOOP_STATUS_PATH)) and self._child_uptime_s(markets_child, now) < (
                market_loop_startup_grace_s + loop_stale_after
            ):
                return
            stale, loop_limit_s, phase, stale_reason, phase_started_ts, phase_age_s = self._market_loop_watchdog_state(
                settings,
                now,
                loop_stale_after,
            )
            if stale:
                phase_txt = f" phase={phase}" if phase and phase != "idle" else ""
                restart_deferred_reason = ""
                if stale_reason == "phase_timeout":
                    timeout_sig = f"{phase}:{int(phase_started_ts)}" if int(phase_started_ts) > 0 else str(phase or "")
                    if timeout_sig != self._last_market_loop_phase_timeout_sig:
                        self._last_market_loop_phase_timeout_sig = timeout_sig
                        self._last_market_loop_phase_timeout_at = float(now)
                    timeout_persist_s = max(0.0, float(now) - float(self._last_market_loop_phase_timeout_at or now))
                    timeout_grace_s = min(
                        1200.0,
                        max(
                            float(phase_timeout_restart_grace_s),
                            float(loop_limit_s) * float(phase_timeout_restart_grace_mult),
                        ),
                    )
                    if timeout_persist_s < timeout_grace_s:
                        restart_deferred_reason = (
                            f"phase-timeout grace {int(timeout_persist_s)}s/{int(timeout_grace_s)}s"
                        )
                else:
                    self._last_market_loop_phase_timeout_sig = ""
                    self._last_market_loop_phase_timeout_at = 0.0
                if stale_reason == "phase_timeout" and phase_txt:
                    stale_msg = f"market loop phase timeout>{int(loop_limit_s)}s ({phase_txt.strip()}; markets child alive)"
                else:
                    stale_msg = f"market loop status stale>{int(loop_limit_s)}s ({phase_txt.strip() + '; ' if phase_txt else ''}markets child alive)"
                if restart_deferred_reason:
                    stale_msg = f"{stale_msg} | restart deferred ({restart_deferred_reason})"
                if (now - self._last_market_loop_stale_note_at) >= 60.0:
                    self._last_market_loop_stale_note_at = now
                    _runner_log(stale_msg)
                    _append_incident(
                        "warning",
                        "runner_market_loop_status_stale",
                        stale_msg,
                        {
                            "status_path": MARKET_LOOP_STATUS_PATH,
                            "stale_after_s": int(loop_limit_s),
                            "phase": phase,
                            "reason": stale_reason,
                        },
                    )
                restart_cooldown_s = max(float(MARKET_LOOP_RESTART_COOLDOWN_S), float(market_interval) * 3.0)
                if (not restart_deferred_reason) and ((now - self._last_market_loop_restart_at) >= restart_cooldown_s):
                    self._last_market_loop_restart_at = now
                    restart_msg = (
                        f"market loop stale>{int(loop_limit_s)}s; restarting markets child "
                        f"(cooldown {int(restart_cooldown_s)}s)"
                    )
                    if phase and phase != "idle":
                        restart_msg = restart_msg[:-1] + f" | phase={phase})"
                    self.msg = restart_msg
                    _runner_log(restart_msg)
                    _append_incident(
                        "warning",
                        "runner_market_loop_restart",
                        restart_msg,
                        {
                            "status_path": MARKET_LOOP_STATUS_PATH,
                            "stale_after_s": int(loop_limit_s),
                            "cooldown_s": int(restart_cooldown_s),
                            "phase": phase,
                            "reason": stale_reason,
                        },
                    )
                    _terminate_process(markets_child.proc, "markets", force=False)
            else:
                self._last_market_loop_phase_timeout_sig = ""
                self._last_market_loop_phase_timeout_at = 0.0

    def _sleep_guard_should_run(self, settings: Dict[str, Any]) -> bool:
        if sys.platform != "darwin":
            return False
        raw = settings.get("runner_prevent_system_sleep", True)
        return bool(raw)

    def _stop_sleep_guard(self) -> None:
        proc = self._sleep_guard_proc
        self._sleep_guard_proc = None
        if not proc or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_sleep_guard(self, settings: Dict[str, Any], now: float) -> None:
        if not self._sleep_guard_should_run(settings):
            if not self._sleep_guard_warned_disabled:
                self._sleep_guard_warned_disabled = True
                _runner_log("sleep guard disabled by settings (runner_prevent_system_sleep=false)")
            self._stop_sleep_guard()
            return
        self._sleep_guard_warned_disabled = False

        proc = self._sleep_guard_proc
        if proc and proc.poll() is None:
            return

        if (float(now) - float(self._sleep_guard_last_start_at or 0.0)) < float(SLEEP_GUARD_RESTART_COOLDOWN_S):
            return

        caffeinate_bin = shutil.which("caffeinate")
        if not caffeinate_bin:
            if not self._sleep_guard_warned_unavailable:
                self._sleep_guard_warned_unavailable = True
                _runner_log("sleep guard unavailable: 'caffeinate' not found on PATH")
                runtime_event(
                    RUNTIME_EVENTS_PATH,
                    component="runner",
                    event="runner_sleep_guard_unavailable",
                    level="warning",
                    msg="Sleep guard unavailable; caffeinate not found. Host may enter standby while unattended.",
                    details={"platform": sys.platform},
                )
            return

        self._sleep_guard_warned_unavailable = False
        self._sleep_guard_last_start_at = float(now)
        try:
            proc = subprocess.Popen(
                [str(caffeinate_bin), "-dims", "-w", str(int(os.getpid()))],
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                **_detached_subprocess_kwargs(),
            )
            self._sleep_guard_proc = proc
            msg = f"sleep guard active via caffeinate (pid={int(proc.pid)})"
            _runner_log(msg)
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="runner_sleep_guard_active",
                level="info",
                msg=msg,
                details={"caffeinate_pid": int(proc.pid)},
            )
        except Exception as exc:
            self._sleep_guard_proc = None
            err = f"failed to start sleep guard: {type(exc).__name__}: {exc}"
            _runner_log(err)
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="runner_sleep_guard_start_failed",
                level="warning",
                msg=err,
                details={"platform": sys.platform},
            )

    def _retention_tick(self, now: float) -> None:
        if (now - self._last_log_cleanup_at) < LOG_RETENTION_INTERVAL_S:
            return
        self._last_log_cleanup_at = now
        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_runner") or {})
        stats = cleanup_logs(
            LOG_DIR,
            keep_patterns=("runner.log", "thinker.log", "trader.log", "markets.log", "autopilot.log"),
            max_age_days=LOG_RETENTION_AGE_DAYS,
            max_total_bytes=LOG_RETENTION_MAX_TOTAL_BYTES,
        )
        try:
            cache_stats = prune_data_cache(
                HUB_DATA_DIR,
                max_age_days=float(settings.get("data_cache_max_age_days", 14.0) or 14.0),
                max_total_bytes=int(float(settings.get("data_cache_max_total_mb", 300) or 300) * 1024 * 1024),
            )
        except Exception:
            cache_stats = {"removed": 0, "removed_bytes": 0}
        try:
            quality_stats = prune_scanner_quality_artifacts(
                HUB_DATA_DIR,
                max_age_days=float(settings.get("scanner_quality_max_age_days", 14.0) or 14.0),
            )
        except Exception:
            quality_stats = {"removed": 0, "removed_bytes": 0}
        try:
            incidents_limit = int(float(settings.get("runtime_incidents_max_lines", 25000) or 25000))
        except Exception:
            incidents_limit = 25000
        try:
            events_limit = int(float(settings.get("runtime_events_max_lines", 50000) or 50000))
        except Exception:
            events_limit = 50000
        incidents_trim = trim_jsonl_max_lines(INCIDENTS_PATH, max_lines=incidents_limit)
        events_trim = trim_jsonl_max_lines(RUNTIME_EVENTS_PATH, max_lines=events_limit)
        if int(stats.get("removed", 0) or 0) > 0:
            _runner_log(
                f"log cleanup removed={int(stats.get('removed', 0))} removed_bytes={int(stats.get('removed_bytes', 0))}"
            )
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="log_cleanup",
                level="info",
                msg="log cleanup completed",
                details=stats,
            )
        if int(cache_stats.get("removed", 0) or 0) > 0:
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="cache_cleanup",
                level="info",
                msg="data cache cleanup completed",
                details=cache_stats,
            )
        if int(quality_stats.get("removed", 0) or 0) > 0:
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="quality_cleanup",
                level="info",
                msg="scanner quality artifact cleanup completed",
                details=quality_stats,
            )
        if bool(incidents_trim.get("trimmed", False)) or bool(events_trim.get("trimmed", False)):
            runtime_event(
                RUNTIME_EVENTS_PATH,
                component="runner",
                event="jsonl_trim",
                level="info",
                msg="runtime jsonl retention trim completed",
                details={
                    "incidents": incidents_trim,
                    "events": events_trim,
                },
            )

    def _drawdown_guard_tick(self, now: float) -> None:
        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_runner") or {})
        max_dd = float(settings.get("global_max_drawdown_pct", 0.0) or 0.0)
        if max_dd <= 0.0:
            return
        lookback_h = int(float(settings.get("global_drawdown_lookback_hours", 24) or 24))
        dd_pct = _intraday_drawdown_pct(os.path.join(HUB_DATA_DIR, "account_value_history.jsonl"), lookback_hours=lookback_h)
        if dd_pct > (-max_dd):
            return
        guard = _safe_read_json(DRAWDOWN_GUARD_PATH)
        last_ts = int(guard.get("triggered_ts", 0) or 0) if isinstance(guard, dict) else 0
        if (now - float(last_ts)) < 300.0:
            return
        payload = {
            "triggered_ts": int(now),
            "drawdown_pct": float(round(dd_pct, 6)),
            "limit_pct": float(max_dd),
            "lookback_hours": int(lookback_h),
        }
        _atomic_write_json(DRAWDOWN_GUARD_PATH, payload)
        _append_incident(
            "critical",
            "global_drawdown_guard",
            f"Global drawdown guard triggered ({dd_pct:.2f}% <= -{max_dd:.2f}%). Stopping trading.",
            payload,
        )
        _write_stop_flag(
            STOP_FLAG_PATH,
            ts_value=int(now),
            reason="drawdown_guard",
            details={
                "drawdown_pct": float(round(dd_pct, 6)),
                "limit_pct": float(max_dd),
                "lookback_hours": int(lookback_h),
            },
        )

    def _safety_ack_payload(self) -> Dict[str, Any]:
        data = _safe_read_json(SAFETY_ACK_PATH)
        return data if isinstance(data, dict) else {}

    def _maybe_resume_drawdown_stop_flag(self, now: float) -> bool:
        stop_row = _stop_flag_payload(STOP_FLAG_PATH)
        if not bool(stop_row.get("active", False)):
            return False
        reason = str(stop_row.get("reason", "") or "").strip().lower()
        if reason != "drawdown_guard":
            return False

        settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
        settings = sanitize_settings(read_settings_file(settings_path, module_name="pt_runner") or {})
        if not bool(settings.get("global_drawdown_auto_resume_enabled", True)):
            return False

        guard = _safe_read_json(DRAWDOWN_GUARD_PATH)
        triggered_ts = int(guard.get("triggered_ts", stop_row.get("ts", 0)) or 0)
        max_dd = float(settings.get("global_max_drawdown_pct", 0.0) or 0.0)
        if max_dd <= 0.0:
            return False

        cooloff_s = max(60, int(float(settings.get("global_drawdown_resume_cooloff_s", 14400) or 14400)))
        if triggered_ts > 0 and (int(now) - int(triggered_ts)) < cooloff_s:
            return False

        ack_required = bool(settings.get("global_drawdown_require_manual_ack", True))
        if ack_required:
            ack = self._safety_ack_payload()
            ack_ts = int(ack.get("drawdown_ack_ts", 0) or 0)
            if ack_ts <= 0 or ack_ts < triggered_ts:
                return False

        lookback_h = int(float(settings.get("global_drawdown_lookback_hours", 24) or 24))
        dd_pct = _intraday_drawdown_pct(os.path.join(HUB_DATA_DIR, "account_value_history.jsonl"), lookback_hours=lookback_h)
        recovery_buffer = max(0.0, float(settings.get("global_drawdown_resume_recovery_buffer_pct", 0.25) or 0.25))
        resume_threshold = -max(0.0, max_dd - recovery_buffer)
        if dd_pct <= resume_threshold:
            return False

        if not _clear_stop_flag(STOP_FLAG_PATH):
            return False
        _append_incident(
            "info",
            "drawdown_guard_auto_resume",
            "Drawdown stop flag auto-cleared after cooldown/recovery checks.",
            {
                "triggered_ts": int(triggered_ts),
                "dd_pct_now": float(round(dd_pct, 6)),
                "resume_threshold_pct": float(round(resume_threshold, 6)),
                "cooloff_s": int(cooloff_s),
                "ack_required": bool(ack_required),
            },
        )
        _runner_log("drawdown stop flag cleared automatically; resuming runner loop")
        return True

    def _script_watch_tick(self, now: float) -> None:
        if (now - self._last_script_watch_at) < float(SCRIPT_WATCH_INTERVAL_S):
            return
        self._last_script_watch_at = float(now)

        try:
            scripts = _settings_scripts()
        except Exception:
            scripts = {}

        for key, child in self.children.items():
            configured_path = os.path.abspath(str(scripts.get(key, child.script_path) or child.script_path))
            old_path = os.path.abspath(str(child.script_path or ""))
            if configured_path != old_path:
                child.script_path = configured_path
                child.loaded_script_mtime = 0.0
                if child.proc and child.proc.poll() is None:
                    msg = f"{key} script path changed; restarting child"
                    self.msg = msg
                    _runner_log(msg)
                    _append_incident(
                        "warning",
                        "runner_script_path_changed",
                        msg,
                        {"child": key, "old_path": old_path, "new_path": configured_path},
                    )
                    _terminate_process(child.proc, key, force=False)
                continue

            if not (child.proc and child.proc.poll() is None):
                continue
            if self._child_uptime_s(child, now) < float(SCRIPT_CHANGE_MIN_UPTIME_S):
                continue

            try:
                current_mtime = float(os.path.getmtime(child.script_path))
            except Exception:
                continue

            loaded_mtime = float(getattr(child, "loaded_script_mtime", 0.0) or 0.0)
            if loaded_mtime <= 0.0:
                child.loaded_script_mtime = current_mtime
                continue
            if current_mtime <= (loaded_mtime + 1e-6):
                continue
            if (now - float(getattr(child, "last_script_change_restart_at", 0.0) or 0.0)) < float(
                SCRIPT_CHANGE_RESTART_COOLDOWN_S
            ):
                continue

            child.last_script_change_restart_at = float(now)
            child.loaded_script_mtime = current_mtime
            msg = f"{key} script updated on disk; restarting child"
            self.msg = msg
            _runner_log(msg)
            _append_incident(
                "warning",
                "runner_script_hot_reload",
                msg,
                {"child": key, "script_path": child.script_path, "old_mtime": loaded_mtime, "new_mtime": current_mtime},
            )
            _terminate_process(child.proc, key, force=False)

    def start_child(self, key: str) -> None:
        child = self.children[key]
        if child.proc and child.proc.poll() is None:
            return
        if not os.path.isfile(child.script_path):
            self.state = "ERROR"
            self.msg = f"Missing script: {child.script_path}"
            _runner_log(self.msg)
            _append_incident("error", "runner_missing_script", self.msg, {"child": key, "path": child.script_path})
            return
        try:
            if child.log_handle is not None:
                try:
                    child.log_handle.close()
                except Exception:
                    pass
            _rotate_log_file(child.log_path)
            log_f = open(child.log_path, "a", encoding="utf-8")
            child.log_handle = log_f
            env = os.environ.copy()
            env["POWERTRADER_HUB_DIR"] = HUB_DATA_DIR
            env["POWERTRADER_PROJECT_DIR"] = BASE_DIR
            prev_pp = str(env.get("PYTHONPATH", "") or "").strip()
            env["PYTHONPATH"] = BASE_DIR if not prev_pp else (BASE_DIR + os.pathsep + prev_pp)
            proc = subprocess.Popen(
                [sys.executable, "-u", child.script_path],
                cwd=BASE_DIR,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                **_detached_subprocess_kwargs(),
            )
            child.proc = proc
            child.next_restart_at = 0.0
            child.started_at = float(time.time())
            try:
                child.loaded_script_mtime = float(os.path.getmtime(child.script_path))
            except Exception:
                child.loaded_script_mtime = 0.0
            self.msg = f"Started {key} pid={proc.pid}"
            _runner_log(self.msg)
        except Exception as exc:
            child.proc = None
            child.started_at = 0.0
            child.loaded_script_mtime = 0.0
            child.next_restart_at = time.time() + min(MAX_BACKOFF_S, child.backoff_s)
            self.state = "ERROR"
            self.msg = f"Failed to start {key}: {type(exc).__name__}: {exc}"
            _runner_log(self.msg)
            _append_incident("error", "runner_child_start_failed", self.msg, {"child": key})

    def handle_exit(self, key: str) -> None:
        child = self.children[key]
        if not child.proc:
            return
        code = child.proc.poll()
        if code is None:
            return
        now = time.time()
        child.last_exit = {"code": int(code), "ts": int(now)}
        child.crash_times = [ts for ts in child.crash_times if (now - ts) <= CRASH_WINDOW_S]
        child.crash_times.append(now)
        child.restarts += 1
        child.backoff_s = min(MAX_BACKOFF_S, child.backoff_s * 2.0 if child.backoff_s > 0 else 1.0)
        child.next_restart_at = now + child.backoff_s
        self.current_backoff_s = child.backoff_s
        child.proc = None
        child.started_at = 0.0
        if child.log_handle is not None:
            try:
                child.log_handle.close()
            except Exception:
                pass
            child.log_handle = None
        if len(child.crash_times) >= CRASH_THRESHOLD:
            self.state = "ERROR"
            child.lockout_until = now + float(self.crash_lockout_s)
            child.next_restart_at = max(child.next_restart_at, child.lockout_until)
            self.msg = f"{key} crash loop; lockout {self.crash_lockout_s:.0f}s"
            _append_incident(
                "error",
                "runner_child_crash_loop",
                self.msg,
                {
                    "child": key,
                    "crashes_in_window": int(len(child.crash_times)),
                    "backoff_s": float(child.backoff_s),
                    "lockout_s": float(self.crash_lockout_s),
                },
            )
        else:
            self.state = "RUNNING"
            self.msg = f"{key} exited code={code}; restarting in {child.backoff_s:.0f}s"
            _append_incident(
                "warning",
                "runner_child_exit",
                self.msg,
                {"child": key, "code": int(code), "backoff_s": float(child.backoff_s)},
            )
        _runner_log(self.msg)

    def stop_all(self, force: bool = False) -> None:
        self.state = "STOPPING"
        self.msg = "Stopping child processes"
        self.write_heartbeat()
        self._stop_sleep_guard()
        for child in self.children.values():
            _terminate_process(child.proc, child.name, force=force)

    def wait_for_children(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            alive = False
            for child in self.children.values():
                if child.proc and child.proc.poll() is None:
                    alive = True
                    break
            if not alive:
                return
            time.sleep(0.2)

    def graceful_shutdown(self) -> None:
        self.stop_all(force=False)
        self.wait_for_children(timeout_s=5.0)
        alive_after_term = [name for name, c in self.children.items() if c.proc and c.proc.poll() is None]
        if alive_after_term:
            _runner_log(f"graceful shutdown timeout; force-killing {len(alive_after_term)} child process(es)")
            _append_incident(
                "warning",
                "runner_forced_shutdown",
                "force kill required after graceful timeout",
                {"children": alive_after_term},
            )
        self.stop_all(force=True)

    def run(self) -> int:
        last_heartbeat = 0.0
        while self.running:
            if os.path.exists(STOP_FLAG_PATH):
                if self._maybe_resume_drawdown_stop_flag(time.time()):
                    time.sleep(0.2)
                    continue
                self.state = "STOPPING"
                self.msg = "Stop flag detected"
                self.graceful_shutdown()
                self.state = "STOPPED"
                self.msg = "Stopped by flag"
                self.write_heartbeat()
                return 0

            now = time.time()
            for key, child in self.children.items():
                if child.proc and child.proc.poll() is not None:
                    self.handle_exit(key)
                if child.lockout_until > now:
                    continue
                if (not child.proc) and now >= float(child.next_restart_at):
                    self.start_child(key)
            self._script_watch_tick(now)
            self._drawdown_guard_tick(now)
            self._watchdog_tick(now)
            self._retention_tick(now)

            if (now - last_heartbeat) >= HEARTBEAT_INTERVAL_S:
                if self.state not in ("ERROR", "STOPPING"):
                    self.state = "RUNNING"
                if not any(child.pid() for child in self.children.values()):
                    self.msg = "Waiting for child processes"
                self.write_heartbeat()
                last_heartbeat = now
            time.sleep(0.5)
        return 0


def _install_signal_handlers(runner: Runner) -> None:
    def _handle(_signum: int, _frame: Any) -> None:
        runner.running = False
        runner.state = "STOPPING"
        runner.msg = "Signal received"
        runner.graceful_shutdown()
        runner.state = "STOPPED"
        runner.msg = "Stopped by signal"
        runner.write_heartbeat()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main() -> int:
    scripts = _settings_scripts()
    discovered = _discover_runtime_processes(scripts)
    live_runner_pids = _live_runner_pids_from_discovery(discovered, self_pid=os.getpid())

    existing_pid = _read_pid_file(RUNNER_PID_PATH)
    if existing_pid and _pid_is_alive(existing_pid):
        _runner_log(f"runner already active pid={existing_pid}; exiting")
        return 0
    if live_runner_pids:
        active_pid = int(sorted(live_runner_pids)[0])
        _runner_log(f"runner already active pid={active_pid} (detected via process table); exiting")
        return 0

    stale_pid_removed = False
    if existing_pid and (not _pid_is_alive(existing_pid)):
        _runner_log(f"stale runner pid file detected pid={existing_pid}; removing")
        _remove_file(RUNNER_PID_PATH)
        stale_pid_removed = True

    orphan_cleanup = _cleanup_orphan_runtime_children(scripts, keep_runner_pids=set())
    if int(orphan_cleanup.get("stale_count", 0) or 0) > 0:
        _runner_log(
            "cleaned stale runtime children "
            f"(count={int(orphan_cleanup.get('stale_count', 0) or 0)} "
            f"term={int(orphan_cleanup.get('terminated', 0) or 0)} "
            f"force={int(orphan_cleanup.get('forced', 0) or 0)})"
        )

    settings_path = resolve_settings_path(BASE_DIR) or _SETTINGS_PATH or os.path.join(BASE_DIR, "gui_settings.json")
    settings = read_settings_file(settings_path, module_name="pt_runner") or {}
    settings = sanitize_settings(settings if isinstance(settings, dict) else {})
    checks = _run_startup_checks(scripts, settings, stale_pid_removed=stale_pid_removed)
    if not bool(checks.get("ok", False)):
        _runner_log("startup checks failed; refusing to start runner")
        return 1

    _write_pid_file(RUNNER_PID_PATH, os.getpid())
    runner = Runner()
    _install_signal_handlers(runner)

    try:
        runner.write_heartbeat()
        return runner.run()
    finally:
        runner.stop_all(force=True)
        for child in runner.children.values():
            if child.log_handle is not None:
                try:
                    child.log_handle.close()
                except Exception:
                    pass
                child.log_handle = None
        runner.state = "STOPPED"
        runner.msg = "Runner exiting"
        runner.write_heartbeat()
        _remove_file(RUNNER_PID_PATH)


if __name__ == "__main__":
    sys.exit(main())
