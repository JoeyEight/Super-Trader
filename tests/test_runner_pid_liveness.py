from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import runtime.pt_runner as pt_runner


class TestRunnerPidLiveness(unittest.TestCase):
    def test_pid_is_alive_treats_zombie_as_dead(self) -> None:
        if str(getattr(pt_runner.os, "name", "")).lower() == "nt":
            self.skipTest("Zombie-state probing uses ps and is Unix-specific")
        with patch.object(pt_runner.os, "kill", return_value=None) as mock_kill, patch.object(
            pt_runner.subprocess,
            "run",
            return_value=SimpleNamespace(stdout="Z+", returncode=0),
        ) as mock_run:
            self.assertFalse(pt_runner._pid_is_alive(43210))
            mock_kill.assert_called_once_with(43210, 0)
            self.assertTrue(mock_run.called)

    def test_pid_is_alive_true_for_non_zombie(self) -> None:
        with patch.object(pt_runner.os, "kill", return_value=None) as mock_kill:
            if str(getattr(pt_runner.os, "name", "")).lower() != "nt":
                with patch.object(
                    pt_runner.subprocess,
                    "run",
                    return_value=SimpleNamespace(stdout="S", returncode=0),
                ) as mock_run:
                    self.assertTrue(pt_runner._pid_is_alive(12345))
                    self.assertTrue(mock_run.called)
            else:
                self.assertTrue(pt_runner._pid_is_alive(12345))
            mock_kill.assert_called_once_with(12345, 0)

    def test_discover_runtime_processes_detects_runner_and_children(self) -> None:
        scripts = {
            "thinker": f"{pt_runner.BASE_DIR}/engines/pt_thinker.py",
            "trader": f"{pt_runner.BASE_DIR}/engines/pt_trader.py",
            "markets": f"{pt_runner.BASE_DIR}/runtime/pt_markets.py",
            "autopilot": f"{pt_runner.BASE_DIR}/runtime/pt_autopilot.py",
        }
        rows = [
            {"pid": 10, "ppid": 1, "command": f"python -u {pt_runner.BASE_DIR}/runtime/pt_runner.py"},
            {"pid": 11, "ppid": 10, "command": f"python -u {scripts['thinker']}"},
            {"pid": 12, "ppid": 10, "command": f"python -u {scripts['trader']}"},
            {"pid": 13, "ppid": 10, "command": f"python -u {scripts['markets']}"},
        ]
        with patch.object(pt_runner, "_ps_process_rows", return_value=rows):
            out = pt_runner._discover_runtime_processes(scripts)
        self.assertEqual(out.get("runner_pids"), [10])
        children = list(out.get("children", []) or [])
        self.assertEqual(len(children), 3)
        self.assertEqual({str(c.get("role")) for c in children}, {"thinker", "trader", "markets"})

    def test_cleanup_orphan_runtime_children_terminates_only_orphans(self) -> None:
        scripts = {
            "thinker": f"{pt_runner.BASE_DIR}/engines/pt_thinker.py",
            "trader": f"{pt_runner.BASE_DIR}/engines/pt_trader.py",
            "markets": f"{pt_runner.BASE_DIR}/runtime/pt_markets.py",
            "autopilot": f"{pt_runner.BASE_DIR}/runtime/pt_autopilot.py",
        }
        discovered = {
            "runner_pids": [222],
            "pid_to_command": {
                222: f"python -u {pt_runner.BASE_DIR}/runtime/pt_runner.py",
                333: "/bin/zsh -lc python worker.py",
            },
            "children": [
                {"role": "thinker", "pid": 100, "ppid": 1, "command": f"python -u {scripts['thinker']}"},
                {"role": "trader", "pid": 101, "ppid": 222, "command": f"python -u {scripts['trader']}"},
                {"role": "markets", "pid": 102, "ppid": 333, "command": f"python -u {scripts['markets']}"},
            ],
        }
        terminated = []

        def _fake_alive(pid: int) -> bool:
            return int(pid) == 333

        def _fake_term(pid: int, force: bool = False) -> bool:
            terminated.append((int(pid), bool(force)))
            return True

        with patch.object(pt_runner, "_discover_runtime_processes", return_value=discovered), patch.object(
            pt_runner, "_pid_is_alive", side_effect=_fake_alive
        ), patch.object(pt_runner, "_terminate_pid", side_effect=_fake_term):
            out = pt_runner._cleanup_orphan_runtime_children(scripts, keep_runner_pids=set())
        self.assertEqual(int(out.get("stale_count", 0)), 2)
        self.assertEqual(int(out.get("terminated", 0)), 2)
        self.assertEqual(int(out.get("forced", 0)), 0)
        self.assertEqual(sorted([pid for pid, forced in terminated if not forced]), [100, 102])

    def test_live_runner_pids_excludes_self(self) -> None:
        discovered = {"runner_pids": [500, 501, 777]}

        def _fake_alive(pid: int) -> bool:
            return int(pid) in {500, 777}

        with patch.object(pt_runner, "_pid_is_alive", side_effect=_fake_alive):
            out = pt_runner._live_runner_pids_from_discovery(discovered, self_pid=500)
        self.assertEqual(out, [777])

    def test_live_runner_pids_filters_dead_and_invalid(self) -> None:
        discovered = {"runner_pids": [0, "bad", 42, 99]}

        def _fake_alive(pid: int) -> bool:
            return int(pid) == 99

        with patch.object(pt_runner, "_pid_is_alive", side_effect=_fake_alive):
            out = pt_runner._live_runner_pids_from_discovery(discovered, self_pid=12345)
        self.assertEqual(out, [99])
