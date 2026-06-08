from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __package__ in (None, ""):
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from app.model_quality_pass import run_model_quality_full_pass  # noqa: E402
from app.path_utils import read_settings_file, resolve_runtime_paths, resolve_settings_path  # noqa: E402
from app.runtime_logging import atomic_write_json  # noqa: E402
from app.settings_utils import sanitize_settings  # noqa: E402


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _default_output(hub_dir: str) -> str:
    return os.path.join(hub_dir, "model_quality_full_pass.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-pass predictive model quality pipeline.")
    parser.add_argument("--output", default="")
    parser.add_argument("--no-sync-core-artifacts", action="store_true")
    args = parser.parse_args()

    probe = os.path.join(_ROOT, "runtime", "pt_runner.py")
    base_dir, _settings, hub_dir, _ = resolve_runtime_paths(probe, "model_quality_full_pass")
    settings_path = resolve_settings_path(base_dir)
    raw = read_settings_file(settings_path, module_name="model_quality_full_pass") or {}
    settings = sanitize_settings(raw if isinstance(raw, dict) else {})

    payload = run_model_quality_full_pass(base_dir=base_dir, hub_dir=hub_dir, settings=settings)

    out = str(args.output or "").strip() or _default_output(hub_dir)
    if not os.path.isabs(out):
        out = os.path.abspath(os.path.join(base_dir, out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_write_json(out, payload if isinstance(payload, dict) else {})

    if not bool(args.no_sync_core_artifacts):
        if isinstance(payload, dict):
            core_map = {
                os.path.join(hub_dir, "market_regimes.json"): payload.get("market_regimes", {}),
                os.path.join(hub_dir, "walkforward_report.json"): payload.get("walkforward_report", {}),
                os.path.join(hub_dir, "confidence_calibration.json"): payload.get("confidence_calibration", {}),
                os.path.join(hub_dir, "shadow_deployment_scorecards.json"): payload.get("shadow_scorecards", {}),
            }
            for path, obj in core_map.items():
                atomic_write_json(path, obj if isinstance(obj, dict) else {})

    crypto_ready = payload.get("promotion_readiness", {}).get("crypto", {}) if isinstance(payload, dict) else {}
    stocks_ready = payload.get("promotion_readiness", {}).get("stocks", {}) if isinstance(payload, dict) else {}
    forex_ready = payload.get("promotion_readiness", {}).get("forex", {}) if isinstance(payload, dict) else {}
    c_diag = payload.get("replay_diagnostics", {}).get("crypto", {}) if isinstance(payload, dict) else {}
    c_metrics = c_diag.get("headline_metrics", {}) if isinstance(c_diag.get("headline_metrics", {}), dict) else {}

    console: Dict[str, Any] = {
        "status": "ok",
        "output": out,
        "performance_diagnostics_path": str(payload.get("performance_diagnostics_path", "")) if isinstance(payload, dict) else "",
        "runtime_seconds": round(_f((payload.get("performance_diagnostics", {}) if isinstance(payload.get("performance_diagnostics", {}), dict) else {}).get("runtime_seconds", 0.0), 0.0), 4),
        "crypto": {
            "state": str(crypto_ready.get("state", "")),
            "directional_accuracy_pct": round(_f(c_metrics.get("directional_accuracy_pct", 0.0), 0.0), 4),
            "trigger_match_pct": round(_f(c_metrics.get("trigger_match_pct", 0.0), 0.0), 4),
            "pnl_trend_match_pct": round(_f(c_metrics.get("pnl_trend_match_pct", 0.0), 0.0), 4),
            "blockers": list(crypto_ready.get("blockers", []) or []),
        },
        "stocks": {"state": str(stocks_ready.get("state", "")), "blockers": list(stocks_ready.get("blockers", []) or [])},
        "forex": {"state": str(forex_ready.get("state", "")), "blockers": list(forex_ready.get("blockers", []) or [])},
    }
    print(json.dumps(console, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
