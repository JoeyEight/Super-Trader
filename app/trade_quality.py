from __future__ import annotations

from typing import Any, Dict, List


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def evaluate_trade_quality(
    *,
    market: str,
    signal_score: float,
    required_score: float,
    data_quality_ok: bool,
    broker_ok: bool,
    runtime_trust_score: float,
    runtime_alert_severity: str,
    compliance_allowed: bool,
    compliance_reason: str = "",
    fallback_active: bool = False,
    fallback_age_s: int = 0,
    fallback_hard_block_age_s: int = 0,
    reject_rate_pct: float = 0.0,
    reject_rate_limit_pct: float = 0.0,
    spread_bps: float = 0.0,
    max_slippage_bps: float = 0.0,
    loss_streak: int = 0,
    max_loss_streak: int = 0,
    exposure_usage_pct: float = 0.0,
    min_runtime_trust_score: float = 35.0,
    min_confidence_score: float = 40.0,
) -> Dict[str, Any]:
    mk = str(market or "").strip().lower()
    sev = str(runtime_alert_severity or "ok").strip().lower()
    abs_signal = abs(float(signal_score or 0.0))
    req_signal = max(0.0, float(required_score or 0.0))
    signal_ratio = 1.0 if req_signal <= 0.0 else _clamp(abs_signal / max(1e-6, req_signal), 0.0, 1.4)
    signal_component = _clamp(signal_ratio * 100.0, 0.0, 100.0)
    data_component = 100.0 if bool(data_quality_ok) else 20.0
    runtime_component = _clamp(float(runtime_trust_score or 0.0), 0.0, 100.0)
    if sev in {"critical", "error"}:
        runtime_component = min(runtime_component, 35.0)
    reject_rate = _clamp(float(reject_rate_pct or 0.0), 0.0, 100.0)
    reject_limit = _clamp(float(reject_rate_limit_pct or 0.0), 0.0, 100.0)
    reject_component = 100.0
    if reject_limit > 0.0:
        reject_component = _clamp(100.0 - ((reject_rate / max(1e-6, reject_limit)) * 55.0), 0.0, 100.0)
    spread = max(0.0, float(spread_bps or 0.0))
    max_slip = max(0.0, float(max_slippage_bps or 0.0))
    spread_component = 100.0
    if max_slip > 0.0:
        spread_penalty = 45.0 if mk == "crypto" else 60.0
        spread_component = _clamp(100.0 - ((spread / max(1e-6, max_slip)) * spread_penalty), 0.0, 100.0)
    discipline_component = 100.0
    if max_loss_streak > 0:
        streak_ratio = _clamp(float(loss_streak) / float(max(1, max_loss_streak)), 0.0, 1.5)
        discipline_component -= 45.0 * min(1.0, streak_ratio)
    discipline_component -= _clamp(float(exposure_usage_pct or 0.0), 0.0, 100.0) * 0.18
    if bool(fallback_active):
        discipline_component -= 12.0
    discipline_component = _clamp(discipline_component, 0.0, 100.0)
    if mk == "crypto":
        confidence_score = (
            (signal_component * 0.34)
            + (data_component * 0.14)
            + (runtime_component * 0.18)
            + (reject_component * 0.12)
            + (spread_component * 0.11)
            + (discipline_component * 0.11)
        )
    else:
        confidence_score = (
            (signal_component * 0.30)
            + (data_component * 0.16)
            + (runtime_component * 0.18)
            + (reject_component * 0.14)
            + (spread_component * 0.12)
            + (discipline_component * 0.10)
        )
    layers = {
        "signal_quality": bool(signal_ratio >= 1.0 and data_quality_ok),
        "execution_quality": bool(
            broker_ok
            and (max_slip <= 0.0 or spread <= max_slip)
            and (reject_limit <= 0.0 or reject_rate < reject_limit)
            and (not bool(fallback_active) or int(fallback_age_s) <= int(max(0, fallback_hard_block_age_s)))
        ),
        "compliance_permission": bool(compliance_allowed),
        "runtime_trust": bool(float(runtime_component) >= float(min_runtime_trust_score) and sev not in {"critical", "error"}),
    }
    block_reasons: List[str] = []
    if not bool(layers["signal_quality"]):
        if signal_ratio < 1.0:
            block_reasons.append(f"Signal quality below requirement ({abs_signal:.4f} < {req_signal:.4f})")
        if not bool(data_quality_ok):
            block_reasons.append("Signal blocked because data quality is degraded")
    if not bool(layers["execution_quality"]):
        if not bool(broker_ok):
            block_reasons.append("Execution blocked because broker/API health is degraded")
        if max_slip > 0.0 and spread > max_slip:
            block_reasons.append(f"Execution blocked by slippage guard ({spread:.2f}bps > {max_slip:.2f}bps)")
        if reject_limit > 0.0 and reject_rate >= reject_limit:
            block_reasons.append(f"Execution blocked by reject-pressure gate ({reject_rate:.1f}% >= {reject_limit:.1f}%)")
        if bool(fallback_active) and int(fallback_age_s) > int(max(0, fallback_hard_block_age_s)):
            block_reasons.append("Execution blocked because cached fallback data is stale")
    if not bool(layers["compliance_permission"]):
        block_reasons.append(str(compliance_reason or f"{mk.title()} compliance protection active"))
    if not bool(layers["runtime_trust"]):
        block_reasons.append("Runtime trust is too low for new entries")
    confidence_gate_pass = bool(float(confidence_score) >= float(min_confidence_score))
    if not confidence_gate_pass:
        block_reasons.append(f"Trade confidence score too low ({confidence_score:.1f} < {float(min_confidence_score):.1f})")
    all_layers_pass = bool(all(layers.values()) and confidence_gate_pass)
    size_cap = 1.40 if mk == "crypto" else 1.25
    if all_layers_pass:
        size_multiplier = _clamp(0.90 + ((confidence_score / 100.0) * 0.40), 0.90, size_cap)
        if confidence_gate_pass:
            size_multiplier = max(1.0, size_multiplier)
    else:
        size_multiplier = 0.0
    decision = "allow" if all_layers_pass else "block"
    return {
        "market": mk,
        "decision": decision,
        "confidence_score": round(float(_clamp(confidence_score, 0.0, 100.0)), 2),
        "size_multiplier": round(float(_clamp(size_multiplier, 0.0, size_cap)), 4),
        "confidence_gate_pass": bool(confidence_gate_pass),
        "layers": layers,
        "block_reasons": block_reasons[:6],
        "components": {
            "signal": round(float(signal_component), 2),
            "data": round(float(data_component), 2),
            "runtime": round(float(runtime_component), 2),
            "reject_pressure": round(float(reject_component), 2),
            "spread": round(float(spread_component), 2),
            "discipline": round(float(discipline_component), 2),
        },
    }
