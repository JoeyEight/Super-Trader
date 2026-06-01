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

from app.crypto_ai_historical_replay import run_crypto_historical_replay  # noqa: E402
from app.path_utils import read_settings_file, resolve_runtime_paths, resolve_settings_path  # noqa: E402
from app.runtime_logging import atomic_write_json  # noqa: E402
from app.settings_utils import sanitize_settings  # noqa: E402


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _default_output(hub_dir: str) -> str:
    return os.path.join(hub_dir, "openai", "crypto_historical_replay.json")


def _metric_block(report: Dict[str, Any]) -> Dict[str, float]:
    best = report.get("best_test_metrics", {}) if isinstance(report.get("best_test_metrics", {}), dict) else {}
    hybrid = report.get("hybrid_test_metrics", {}) if isinstance(report.get("hybrid_test_metrics", {}), dict) else {}
    use = hybrid if hybrid else best
    return {
        "directional_accuracy_pct": _f(use.get("directional_accuracy_pct", 0.0), 0.0),
        "trigger_match_pct": _f(use.get("trigger_match_pct", 0.0), 0.0),
        "pnl_trend_match_pct": _f(use.get("pnl_trend_match_pct", 0.0), 0.0),
        "mean_abs_timing_error_hours": _f(use.get("mean_abs_timing_error_hours", 0.0), 0.0),
        "mean_abs_exit_price_error_pct": _f(use.get("mean_abs_exit_price_error_pct", 0.0), 0.0),
    }


def _median(values: list[float]) -> float:
    vals = sorted(float(v) for v in list(values or []))
    n = len(vals)
    if n <= 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _run_once(
    *,
    settings: Dict[str, Any],
    base_dir: str,
    hub_dir: str,
    max_trades: int,
    test_ratio: float,
    iterations: int,
    model: str,
    timeout_s: float,
    enable_train_calibration: bool,
    trigger_override_conf_min_floor: float,
) -> Dict[str, Any]:
    return run_crypto_historical_replay(
        settings=settings,
        base_dir=base_dir,
        hub_dir=hub_dir,
        max_trades=max_trades,
        test_ratio=test_ratio,
        max_iterations=iterations,
        model_override=model,
        timeout_s_override=timeout_s,
        enable_train_calibration=enable_train_calibration,
        trigger_override_conf_min_floor=trigger_override_conf_min_floor,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind historical crypto replay: OpenAI predicts exit timing/price from pre-entry context only."
    )
    parser.add_argument("--max-trades", type=int, default=24)
    parser.add_argument("--test-ratio", type=float, default=0.35)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument("--enable-train-calibration", action="store_true")
    parser.add_argument("--trigger-override-floor", type=float, default=0.60)
    parser.add_argument("--stability-trials", type=int, default=3)
    parser.add_argument("--promotion-min-direction-lift", type=float, default=2.0)
    parser.add_argument("--promotion-min-trigger-lift", type=float, default=5.0)
    parser.add_argument("--promotion-min-pnl-trend-lift", type=float, default=3.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    probe = os.path.join(_ROOT, "runtime", "pt_runner.py")
    base_dir, _settings, hub_dir, _ = resolve_runtime_paths(probe, "crypto_ai_historical_replay")
    settings_path = resolve_settings_path(base_dir)
    raw = read_settings_file(settings_path, module_name="crypto_ai_historical_replay") or {}
    settings = sanitize_settings(raw if isinstance(raw, dict) else {})

    max_trades = max(4, int(args.max_trades or 24))
    test_ratio = max(0.1, min(0.8, _f(args.test_ratio, 0.35)))
    iterations = max(1, int(args.iterations or 3))
    model = str(args.model or "").strip()
    timeout_s = max(0.0, _f(args.timeout_s, 0.0))
    enable_train_calibration = bool(args.enable_train_calibration)
    trigger_floor = max(0.0, min(1.0, _f(args.trigger_override_floor, 0.60)))
    trials = max(1, int(args.stability_trials or 1))
    min_dir_lift = max(0.0, _f(args.promotion_min_direction_lift, 2.0))
    min_trig_lift = max(0.0, _f(args.promotion_min_trigger_lift, 5.0))
    min_pnl_trend_lift = max(0.0, _f(args.promotion_min_pnl_trend_lift, 3.0))

    if trials <= 1:
        report = _run_once(
            settings=settings,
            base_dir=base_dir,
            hub_dir=hub_dir,
            max_trades=max_trades,
            test_ratio=test_ratio,
            iterations=iterations,
            model=model,
            timeout_s=timeout_s,
            enable_train_calibration=enable_train_calibration,
            trigger_override_conf_min_floor=trigger_floor,
        )
    else:
        baseline_runs: list[Dict[str, Any]] = []
        candidate_runs: list[Dict[str, Any]] = []
        for _ in range(trials):
            baseline_runs.append(
                _run_once(
                    settings=settings,
                    base_dir=base_dir,
                    hub_dir=hub_dir,
                    max_trades=max_trades,
                    test_ratio=test_ratio,
                    iterations=iterations,
                    model=model,
                    timeout_s=timeout_s,
                    enable_train_calibration=enable_train_calibration,
                    trigger_override_conf_min_floor=0.0,
                )
            )
            candidate_runs.append(
                _run_once(
                    settings=settings,
                    base_dir=base_dir,
                    hub_dir=hub_dir,
                    max_trades=max_trades,
                    test_ratio=test_ratio,
                    iterations=iterations,
                    model=model,
                    timeout_s=timeout_s,
                    enable_train_calibration=enable_train_calibration,
                    trigger_override_conf_min_floor=trigger_floor,
                )
            )

        base_metrics = [_metric_block(r if isinstance(r, dict) else {}) for r in baseline_runs]
        cand_metrics = [_metric_block(r if isinstance(r, dict) else {}) for r in candidate_runs]
        base_median = {
            "directional_accuracy_pct": round(_median([m["directional_accuracy_pct"] for m in base_metrics]), 4),
            "trigger_match_pct": round(_median([m["trigger_match_pct"] for m in base_metrics]), 4),
            "pnl_trend_match_pct": round(_median([m["pnl_trend_match_pct"] for m in base_metrics]), 4),
            "mean_abs_timing_error_hours": round(_median([m["mean_abs_timing_error_hours"] for m in base_metrics]), 4),
            "mean_abs_exit_price_error_pct": round(_median([m["mean_abs_exit_price_error_pct"] for m in base_metrics]), 4),
        }
        cand_median = {
            "directional_accuracy_pct": round(_median([m["directional_accuracy_pct"] for m in cand_metrics]), 4),
            "trigger_match_pct": round(_median([m["trigger_match_pct"] for m in cand_metrics]), 4),
            "pnl_trend_match_pct": round(_median([m["pnl_trend_match_pct"] for m in cand_metrics]), 4),
            "mean_abs_timing_error_hours": round(_median([m["mean_abs_timing_error_hours"] for m in cand_metrics]), 4),
            "mean_abs_exit_price_error_pct": round(_median([m["mean_abs_exit_price_error_pct"] for m in cand_metrics]), 4),
        }

        promoted = (
            (cand_median["directional_accuracy_pct"] >= (base_median["directional_accuracy_pct"] + float(min_dir_lift)))
            and (cand_median["trigger_match_pct"] >= (base_median["trigger_match_pct"] + float(min_trig_lift)))
            and (cand_median["pnl_trend_match_pct"] >= (base_median["pnl_trend_match_pct"] + float(min_pnl_trend_lift)))
        )
        chosen_label = "candidate_floored_gate" if promoted else "baseline_unfloored"
        chosen_runs = candidate_runs if promoted else baseline_runs
        chosen_report = max(
            list(chosen_runs or []),
            key=lambda r: (
                _metric_block(r).get("directional_accuracy_pct", 0.0)
                + _metric_block(r).get("trigger_match_pct", 0.0)
            ),
        ) if chosen_runs else {}
        report = dict(chosen_report if isinstance(chosen_report, dict) else {})
        report["stability"] = {
            "trials": int(trials),
            "trigger_override_floor_candidate": round(float(trigger_floor), 4),
            "promotion_min_direction_lift": round(float(min_dir_lift), 4),
            "promotion_min_trigger_lift": round(float(min_trig_lift), 4),
            "promotion_min_pnl_trend_lift": round(float(min_pnl_trend_lift), 4),
            "baseline_median_metrics": base_median,
            "candidate_median_metrics": cand_median,
            "promoted": bool(promoted),
            "chosen": str(chosen_label),
        }
        report["summary"] = (
            f"Stability layer ({trials} trials): chosen={chosen_label}; "
            f"baseline med dir {base_median['directional_accuracy_pct']:.2f}%/trig {base_median['trigger_match_pct']:.2f}%/pnl-trend {base_median['pnl_trend_match_pct']:.2f}%, "
            f"candidate med dir {cand_median['directional_accuracy_pct']:.2f}%/trig {cand_median['trigger_match_pct']:.2f}%/pnl-trend {cand_median['pnl_trend_match_pct']:.2f}%."
        )

    out = str(args.output or "").strip() or _default_output(hub_dir)
    if not os.path.isabs(out):
        out = os.path.abspath(os.path.join(base_dir, out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_write_json(out, report if isinstance(report, dict) else {})

    status = str((report.get("status", "") if isinstance(report, dict) else "") or "")
    summary = str((report.get("summary", "") if isinstance(report, dict) else "") or "")
    best_metrics = report.get("best_test_metrics", {}) if isinstance(report.get("best_test_metrics", {}), dict) else {}
    hybrid_metrics = report.get("hybrid_test_metrics", {}) if isinstance(report.get("hybrid_test_metrics", {}), dict) else {}
    stability = report.get("stability", {}) if isinstance(report.get("stability", {}), dict) else {}
    meta = report.get("meta", {}) if isinstance(report.get("meta", {}), dict) else {}

    console: Dict[str, Any] = {
        "status": status,
        "summary": summary,
        "output": out,
        "model": str(meta.get("model", "")),
        "closed_trades_total": int(_f(meta.get("closed_trades_total", 0), 0.0)),
        "train_trades": int(_f(meta.get("train_trades", 0), 0.0)),
        "test_trades": int(_f(meta.get("test_trades", 0), 0.0)),
        "best_iteration": int(_f(report.get("best_iteration", 0), 0.0)),
        "best_direction_iteration": int(_f(report.get("best_direction_iteration", 0), 0.0)),
        "best_trigger_iteration": int(_f(report.get("best_trigger_iteration", 0), 0.0)),
        "best_trigger_override_conf_min": round(_f(report.get("best_trigger_override_conf_min", 0.0), 0.0), 4),
        "directional_accuracy_pct": round(_f(best_metrics.get("directional_accuracy_pct", 0.0), 0.0), 4),
        "mean_abs_timing_error_hours": round(_f(best_metrics.get("mean_abs_timing_error_hours", 0.0), 0.0), 4),
        "mean_abs_exit_price_error_pct": round(_f(best_metrics.get("mean_abs_exit_price_error_pct", 0.0), 0.0), 4),
        "trigger_match_pct": round(_f(best_metrics.get("trigger_match_pct", 0.0), 0.0), 4),
        "pnl_trend_match_pct": round(_f(best_metrics.get("pnl_trend_match_pct", 0.0), 0.0), 4),
        "hybrid_directional_accuracy_pct": round(_f(hybrid_metrics.get("directional_accuracy_pct", 0.0), 0.0), 4),
        "hybrid_trigger_match_pct": round(_f(hybrid_metrics.get("trigger_match_pct", 0.0), 0.0), 4),
        "hybrid_pnl_trend_match_pct": round(_f(hybrid_metrics.get("pnl_trend_match_pct", 0.0), 0.0), 4),
        "hybrid_mean_abs_timing_error_hours": round(_f(hybrid_metrics.get("mean_abs_timing_error_hours", 0.0), 0.0), 4),
        "hybrid_mean_abs_exit_price_error_pct": round(_f(hybrid_metrics.get("mean_abs_exit_price_error_pct", 0.0), 0.0), 4),
        "stability_trials": int(_f(stability.get("trials", 1), 1.0)),
        "stability_promoted": bool(stability.get("promoted", False)),
        "stability_chosen": str(stability.get("chosen", "")),
    }
    print(json.dumps(console, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
