from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable


def load_replay_trigger_reliability(
    hub_dir: str,
    market: str,
    patterns: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Best-effort trigger reliability from latest replay artifact(s).
    Falls back to a neutral 0.50 when no replay metrics are available.
    """
    default = {"value": 0.50, "samples": 0, "source": "default_neutral"}
    openai_dir = os.path.join(str(hub_dir or "").strip(), "openai")
    if not os.path.isdir(openai_dir):
        return default
    m = str(market or "").strip().lower()
    if not m:
        return default
    search_patterns = list(patterns or [f"{m}_*replay*.json", f"{m}_historical_replay*.json"])
    candidates: list[str] = []
    for pat in search_patterns:
        try:
            candidates.extend(glob.glob(os.path.join(openai_dir, pat)))
        except Exception:
            continue
    if not candidates:
        return default
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        hybrid = payload.get("hybrid_test_metrics", {}) if isinstance(payload.get("hybrid_test_metrics", {}), dict) else {}
        best = payload.get("best_test_metrics", {}) if isinstance(payload.get("best_test_metrics", {}), dict) else {}
        metric_source = hybrid if hybrid else best
        trigger_match_pct = float(metric_source.get("trigger_match_pct", payload.get("trigger_match_pct", 0.0)) or 0.0)
        trigger_match_pct = max(0.0, min(100.0, trigger_match_pct))
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        test_trades = int(float(meta.get("test_trades", payload.get("test_trades", 0)) or 0))
        if trigger_match_pct <= 0.0 and test_trades <= 0:
            continue
        return {
            "value": round(trigger_match_pct / 100.0, 6),
            "samples": int(max(0, test_trades)),
            "source": f"replay:{os.path.basename(path)}",
        }
    return default


def evaluate_entry_calibration_gate(
    *,
    calib_prob: float,
    calib_samples: int,
    trigger_reliability: float,
    min_calib_prob: float,
    min_calib_samples: int,
    min_trigger_reliability: float,
    min_entry_score: float,
    weight_calib: float = 0.65,
    weight_trigger: float = 0.35,
    require_samples: bool = True,
) -> Dict[str, Any]:
    cp = max(0.0, min(1.0, float(calib_prob or 0.0)))
    cs = max(0, int(calib_samples or 0))
    tr = max(0.0, min(1.0, float(trigger_reliability or 0.0)))
    min_cp = max(0.0, min(1.0, float(min_calib_prob or 0.0)))
    min_cs = max(0, int(min_calib_samples or 0))
    min_tr = max(0.0, min(1.0, float(min_trigger_reliability or 0.0)))
    min_es = max(0.0, min(1.0, float(min_entry_score or 0.0)))
    wc = max(0.0, min(1.0, float(weight_calib or 0.0)))
    wt = max(0.0, min(1.0, float(weight_trigger or 0.0)))
    if (wc + wt) <= 1e-9:
        wc, wt = 0.65, 0.35
    else:
        den = float(wc + wt)
        wc, wt = (wc / den), (wt / den)
    score = (cp * wc) + (tr * wt)
    samples_ok = (cs >= min_cs) if bool(require_samples) else True
    prob_ok = cp >= min_cp
    trig_ok = tr >= min_tr
    score_ok = score >= min_es
    passed = bool(samples_ok and prob_ok and trig_ok and score_ok)
    reason = ""
    if not passed:
        if not samples_ok:
            reason = f"calibration samples too low ({cs} < {min_cs})"
        elif not prob_ok:
            reason = f"calibrated confidence too low ({cp:.3f} < {min_cp:.3f})"
        elif not trig_ok:
            reason = f"trigger reliability too low ({tr:.3f} < {min_tr:.3f})"
        else:
            reason = f"entry calibration score too low ({score:.3f} < {min_es:.3f})"
    return {
        "passed": bool(passed),
        "reason": str(reason),
        "score": round(float(score), 6),
        "calib_prob": round(float(cp), 6),
        "calib_samples": int(cs),
        "trigger_reliability": round(float(tr), 6),
        "thresholds": {
            "min_calib_prob": round(float(min_cp), 6),
            "min_calib_samples": int(min_cs),
            "min_trigger_reliability": round(float(min_tr), 6),
            "min_entry_score": round(float(min_es), 6),
        },
        "weights": {
            "calib": round(float(wc), 6),
            "trigger": round(float(wt), 6),
        },
        "require_samples": bool(require_samples),
    }
