from __future__ import annotations

import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import runtime.pt_runner as pt_runner


class _AliveProc:
    def poll(self) -> None:
        return None


def _scripts(base_dir: str) -> dict[str, str]:
    return {
        "thinker": os.path.join(base_dir, "noop_thinker.py"),
        "trader": os.path.join(base_dir, "noop_trader.py"),
        "markets": os.path.join(base_dir, "noop_markets.py"),
        "autopilot": os.path.join(base_dir, "noop_autopilot.py"),
    }


class TestRunnerWatchdog(unittest.TestCase):
    def test_market_loop_watchdog_stale_after_uses_latency_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            runtime_state_path = os.path.join(hub, "runtime_state.json")
            with open(runtime_state_path, "w", encoding="utf-8") as f:
                f.write(
                    """
{
  "sla_metrics": {
    "stocks_snapshot": {"p95_ms": 400.0},
    "forex_snapshot": {"p95_ms": 300.0},
    "stocks_scan": {"p95_ms": 118000.0},
    "forex_scan": {"p95_ms": 11000.0},
    "stocks_trader_step": {"p95_ms": 600.0},
    "forex_trader_step": {"p95_ms": 500.0}
  }
}
""".strip()
                )
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", os.path.join(hub, "market_loop_status.json")))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_STATE_PATH", runtime_state_path))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "read_settings_file",
                        return_value={
                            "market_bg_snapshot_interval_s": 15.0,
                            "market_bg_stocks_interval_s": 12.0,
                            "market_bg_forex_interval_s": 8.0,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                stale_after = runner._market_loop_watchdog_stale_after(
                    {
                        "market_bg_snapshot_interval_s": 15.0,
                        "market_bg_stocks_interval_s": 12.0,
                        "market_bg_forex_interval_s": 8.0,
                    },
                    floor_s=90.0,
                )

            self.assertGreaterEqual(float(stale_after), 140.0)

    def test_market_loop_phase_timeout_uses_active_phase_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            runtime_state_path = os.path.join(hub, "runtime_state.json")
            with open(runtime_state_path, "w", encoding="utf-8") as f:
                f.write(
                    """
{
  "sla_metrics": {
    "stocks_scan": {"p95_ms": 110000.0, "last_ms": 100000.0}
  }
}
""".strip()
                )
            market_loop_status_path = os.path.join(hub, "market_loop_status.json")
            with open(market_loop_status_path, "w", encoding="utf-8") as f:
                f.write('{"phase":"stocks_scan","phase_started_ts":1000,"ts":1200,"heartbeat_ts":1200}')
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", market_loop_status_path))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_STATE_PATH", runtime_state_path))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "read_settings_file",
                        return_value={
                            "market_bg_forex_interval_s": 12.0,
                            "runner_market_loop_phase_timeout_restart_grace_s": 0.0,
                            "runner_market_loop_phase_timeout_restart_grace_mult": 0.0,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                runner.children["markets"].proc = _AliveProc()  # type: ignore[assignment]

                with patch.object(runner, "_status_file_stale", return_value=False), patch.object(
                    pt_runner, "_append_incident"
                ) as mock_incident, patch.object(pt_runner, "_runner_log"):
                    runner._watchdog_tick(1260.0)

                events = [str(call.args[1]) for call in mock_incident.call_args_list if len(call.args) >= 2]
                self.assertIn("runner_market_loop_status_stale", events)
                self.assertIn("runner_market_loop_restart", events)

    def test_market_loop_phase_timeout_restart_is_deferred_by_grace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            runtime_state_path = os.path.join(hub, "runtime_state.json")
            with open(runtime_state_path, "w", encoding="utf-8") as f:
                f.write(
                    """
{
  "sla_metrics": {
    "stocks_scan": {"p95_ms": 110000.0, "last_ms": 100000.0}
  }
}
""".strip()
                )
            market_loop_status_path = os.path.join(hub, "market_loop_status.json")
            with open(market_loop_status_path, "w", encoding="utf-8") as f:
                f.write('{"phase":"stocks_scan","phase_started_ts":1000,"ts":1200,"heartbeat_ts":1200}')
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", market_loop_status_path))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_STATE_PATH", runtime_state_path))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "read_settings_file",
                        return_value={"market_bg_forex_interval_s": 12.0},
                    )
                )
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                runner.children["markets"].proc = _AliveProc()  # type: ignore[assignment]

                with patch.object(runner, "_status_file_stale", return_value=False), patch.object(
                    pt_runner, "_append_incident"
                ) as mock_incident, patch.object(pt_runner, "_runner_log"):
                    runner._watchdog_tick(1260.0)

                events = [str(call.args[1]) for call in mock_incident.call_args_list if len(call.args) >= 2]
                self.assertIn("runner_market_loop_status_stale", events)
                self.assertNotIn("runner_market_loop_restart", events)

    def test_market_loop_stale_note_emitted_and_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", os.path.join(hub, "market_loop_status.json")))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "read_settings_file",
                        return_value={"market_bg_forex_interval_s": 12.0},
                    )
                )
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                runner.children["markets"].proc = _AliveProc()  # type: ignore[assignment]

                def _stale(path: str, _max_age_s: float) -> bool:
                    if str(path) == str(pt_runner.MARKET_LOOP_STATUS_PATH):
                        return True
                    return False

                with patch.object(runner, "_status_file_stale", side_effect=_stale), patch.object(
                    pt_runner, "_append_incident"
                ) as mock_incident, patch.object(pt_runner, "_runner_log"):
                    base_now = time.time() + 100.0
                    runner._watchdog_tick(base_now)
                    calls_after_first = int(mock_incident.call_count)
                    runner._watchdog_tick(base_now + 10.0)
                    calls_after_second = int(mock_incident.call_count)
                    runner._watchdog_tick(base_now + 80.0)
                    calls_after_third = int(mock_incident.call_count)

                self.assertGreaterEqual(calls_after_first, 1)
                self.assertEqual(calls_after_second, calls_after_first)
                self.assertGreaterEqual(calls_after_third, calls_after_first + 1)
                events = [str(call.args[1]) for call in mock_incident.call_args_list if len(call.args) >= 2]
                self.assertIn("runner_market_loop_status_stale", events)
                self.assertIn("runner_market_loop_restart", events)

    def test_script_watch_restarts_child_when_script_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", os.path.join(hub, "market_loop_status.json")))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(patch.object(pt_runner, "read_settings_file", return_value={}))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                child = runner.children["trader"]
                child.proc = _AliveProc()  # type: ignore[assignment]
                child.started_at = time.time() - 20.0
                child.loaded_script_mtime = os.path.getmtime(child.script_path)
                os.utime(child.script_path, None)

                with patch.object(pt_runner, "_append_incident") as mock_incident, patch.object(
                    pt_runner, "_runner_log"
                ), patch.object(pt_runner, "_terminate_process") as mock_term:
                    runner._script_watch_tick(time.time())

                self.assertGreaterEqual(int(mock_term.call_count), 1)
                events = [str(call.args[1]) for call in mock_incident.call_args_list if len(call.args) >= 2]
                self.assertIn("runner_script_hot_reload", events)

    def test_script_watch_restarts_child_when_script_path_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts_old = _scripts(td)
            scripts_new = dict(scripts_old)
            scripts_new["trader"] = os.path.join(td, "new_trader.py")
            for path in set(list(scripts_old.values()) + [scripts_new["trader"]]):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", os.path.join(hub, "market_loop_status.json")))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(patch.object(pt_runner, "read_settings_file", return_value={}))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", side_effect=[scripts_old, scripts_new]))
                runner = pt_runner.Runner()
                child = runner.children["trader"]
                old_path = child.script_path
                child.proc = _AliveProc()  # type: ignore[assignment]
                child.started_at = time.time() - 20.0

                with patch.object(pt_runner, "_append_incident") as mock_incident, patch.object(
                    pt_runner, "_runner_log"
                ), patch.object(pt_runner, "_terminate_process") as mock_term:
                    runner._script_watch_tick(time.time())

                self.assertEqual(os.path.abspath(child.script_path), os.path.abspath(scripts_new["trader"]))
                self.assertNotEqual(os.path.abspath(old_path), os.path.abspath(child.script_path))
                self.assertGreaterEqual(int(mock_term.call_count), 1)
                events = [str(call.args[1]) for call in mock_incident.call_args_list if len(call.args) >= 2]
                self.assertIn("runner_script_path_changed", events)

    def test_market_watchdog_startup_grace_skips_early_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub = os.path.join(td, "hub_data")
            logs = os.path.join(hub, "logs")
            os.makedirs(logs, exist_ok=True)
            settings_path = os.path.join(td, "gui_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                f.write("{}")

            scripts = _scripts(td)
            for path in scripts.values():
                with open(path, "w", encoding="utf-8") as f:
                    f.write("print('noop')\n")

            with ExitStack() as stack:
                stack.enter_context(patch.object(pt_runner, "BASE_DIR", td))
                stack.enter_context(patch.object(pt_runner, "HUB_DATA_DIR", hub))
                stack.enter_context(patch.object(pt_runner, "LOG_DIR", logs))
                stack.enter_context(patch.object(pt_runner, "RUNNER_LOG_PATH", os.path.join(logs, "runner.log")))
                stack.enter_context(patch.object(pt_runner, "THINKER_LOG_PATH", os.path.join(logs, "thinker.log")))
                stack.enter_context(patch.object(pt_runner, "TRADER_LOG_PATH", os.path.join(logs, "trader.log")))
                stack.enter_context(patch.object(pt_runner, "MARKETS_LOG_PATH", os.path.join(logs, "markets.log")))
                stack.enter_context(patch.object(pt_runner, "AUTOPILOT_LOG_PATH", os.path.join(logs, "autopilot.log")))
                stack.enter_context(patch.object(pt_runner, "MARKET_LOOP_STATUS_PATH", os.path.join(hub, "market_loop_status.json")))
                stack.enter_context(patch.object(pt_runner, "RUNTIME_EVENTS_PATH", os.path.join(hub, "runtime_events.jsonl")))
                stack.enter_context(patch.object(pt_runner, "INCIDENTS_PATH", os.path.join(hub, "incidents.jsonl")))
                stack.enter_context(patch.object(pt_runner, "TRADER_STATUS_PATH", os.path.join(hub, "trader_status.json")))
                stack.enter_context(patch.object(pt_runner, "resolve_settings_path", return_value=settings_path))
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "read_settings_file",
                        return_value={
                            "market_bg_forex_interval_s": 12.0,
                            "runner_market_watchdog_startup_grace_s": 120.0,
                            "runner_market_loop_startup_grace_s": 150.0,
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        pt_runner,
                        "sanitize_settings",
                        side_effect=lambda x: (x if isinstance(x, dict) else {}),
                    )
                )
                stack.enter_context(patch.object(pt_runner, "_settings_scripts", return_value=scripts))
                runner = pt_runner.Runner()
                runner.children["markets"].proc = _AliveProc()  # type: ignore[assignment]
                runner.children["markets"].started_at = time.time()

                with patch.object(runner, "_status_file_stale", return_value=True), patch.object(
                    pt_runner, "_append_incident"
                ) as mock_incident, patch.object(pt_runner, "_runner_log"), patch.object(
                    pt_runner, "_terminate_process"
                ) as mock_term:
                    runner._watchdog_tick(time.time() + 10.0)

                self.assertEqual(int(mock_incident.call_count), 0)
                self.assertEqual(int(mock_term.call_count), 0)


if __name__ == "__main__":
    unittest.main()
