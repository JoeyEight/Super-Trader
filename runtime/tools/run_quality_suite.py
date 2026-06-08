from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List


def _run(cmd: List[str], cwd: str) -> int:
    print(f"[quality] {' '.join(cmd)}")
    timeout_s = max(30, int(float(os.environ.get("QUALITY_SUITE_STEP_TIMEOUT_S", 1800) or 1800)))
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=False, timeout=timeout_s)
        print(f"[quality] step_seconds={time.time() - started:.2f}")
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        print(f"[quality] step_timeout after {timeout_s}s")
        return 124


def main() -> int:
    ap = argparse.ArgumentParser(description="Run lint/type/tests plus optional artifact and stability checks.")
    ap.add_argument("--project-dir", default=os.getcwd())
    ap.add_argument("--skip-lint", action="store_true")
    ap.add_argument("--skip-type", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-artifacts", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--require-artifacts", action="store_true", help="Fail if runtime artifacts are missing.")
    ap.add_argument(
        "--require-stability",
        action="store_true",
        help="Run strict runtime stability audit and fail if it does not pass.",
    )
    ap.add_argument(
        "--require-preflight",
        action="store_true",
        help="Run strict preflight readiness checks and fail if warnings/critical issues are found.",
    )
    ap.add_argument(
        "--bootstrap-artifacts",
        action="store_true",
        help="Run runtime artifact bootstrap before artifact checks.",
    )
    args = ap.parse_args()

    project_dir = os.path.abspath(str(args.project_dir or os.getcwd()))
    failures: List[str] = []

    if not bool(args.skip_lint):
        rc = _run([sys.executable, "-m", "ruff", "check", "app", "runtime", "engines", "tests"], cwd=project_dir)
        if rc != 0:
            failures.append("ruff")
    if not bool(args.skip_type):
        rc = _run([sys.executable, "-m", "mypy", "app", "runtime", "engines", "tests"], cwd=project_dir)
        if rc != 0:
            failures.append("mypy")
    if not bool(args.skip_tests):
        rc = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=project_dir)
        if rc != 0:
            failures.append("unittest")
    if not bool(args.skip_artifacts):
        if bool(args.require_artifacts):
            if bool(args.bootstrap_artifacts):
                rc = _run([sys.executable, "runtime/tools/bootstrap_runtime_artifacts.py"], cwd=project_dir)
                if rc != 0:
                    failures.append("bootstrap_runtime_artifacts")
            rc = _run([sys.executable, "runtime/tools/check_pass3_artifacts.py"], cwd=project_dir)
            if rc != 0:
                failures.append("check_pass3_artifacts")
        else:
            print("[quality] skipping artifact check (use --require-artifacts to enforce runtime artifact validation)")
    if not bool(args.skip_stability):
        if bool(args.require_stability):
            rc = _run([sys.executable, "runtime/tools/stability_audit.py", "--strict"], cwd=project_dir)
            if rc != 0:
                failures.append("stability_audit")
        else:
            print("[quality] skipping stability audit (use --require-stability to enforce runtime stability checks)")
    if not bool(args.skip_preflight):
        if bool(args.require_preflight):
            rc = _run([sys.executable, "runtime/tools/preflight_readiness.py", "--strict"], cwd=project_dir)
            if rc != 0:
                failures.append("preflight_readiness")
        else:
            print("[quality] skipping preflight readiness (use --require-preflight to enforce environment readiness checks)")

    if failures:
        print(f"[quality] failed: {', '.join(failures)}")
        return 1
    print("[quality] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
