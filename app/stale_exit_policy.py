from __future__ import annotations

from typing import Any, Dict


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def evaluate_stale_profit_hold(
    *,
    pnl_pct: float,
    stale_streak: int,
    grace_cycles: int,
    position_age_s: int,
    hard_reverse: bool,
    trend_support: bool,
    profit_hold_min_pct: float,
    profit_hold_extra_cycles: int,
    profit_hold_max_s: int,
    pullback_pct: float,
    profit_hold_max_pullback_pct: float,
) -> Dict[str, Any]:
    profit = float(_f(pnl_pct, 0.0))
    streak = max(0, int(stale_streak or 0))
    grace = max(1, int(grace_cycles or 1))
    age_s = max(0, int(position_age_s or 0))
    min_profit = max(0.0, float(_f(profit_hold_min_pct, 0.0)))
    extra_cycles = max(0, int(profit_hold_extra_cycles or 0))
    max_s = max(0, int(profit_hold_max_s or 0))
    pb = max(0.0, float(_f(pullback_pct, 0.0)))
    max_pb = max(0.0, float(_f(profit_hold_max_pullback_pct, 0.0)))

    if profit < min_profit:
        return {
            "hold": False,
            "reason": "profit_below_guard_threshold",
            "detail": f"PnL {profit:+.3f}% is below stale-profit hold threshold +{min_profit:.3f}%.",
        }

    if bool(hard_reverse):
        return {
            "hold": False,
            "reason": "hard_reverse_override",
            "detail": "Hard reverse evidence is active; stale-profit hold guard not applied.",
        }

    if not bool(trend_support):
        return {
            "hold": False,
            "reason": "trend_support_missing",
            "detail": "Trend support check failed; stale-profit hold guard not applied.",
        }

    if max_s > 0 and age_s >= max_s:
        return {
            "hold": False,
            "reason": "profit_hold_max_age_exceeded",
            "detail": (
                f"Position age {age_s}s exceeded stale-profit hold max window {max_s}s."
            ),
        }

    max_streak = int(grace + extra_cycles)
    if streak >= max_streak:
        return {
            "hold": False,
            "reason": "profit_hold_streak_budget_exhausted",
            "detail": (
                f"Stale streak {streak} reached stale-profit hold budget {max_streak}."
            ),
        }

    if pb > max_pb:
        return {
            "hold": False,
            "reason": "profit_hold_pullback_exceeded",
            "detail": (
                f"Pullback {pb:.3f}% exceeded stale-profit hold max pullback {max_pb:.3f}%"
            ),
        }

    return {
        "hold": True,
        "reason": "stale_profit_hold_guard",
        "detail": (
            f"Holding stale-aligned winner (PnL {profit:+.3f}%, streak {streak}/{max_streak}, "
            f"age {age_s}s, pullback {pb:.3f}% <= {max_pb:.3f}%)."
        ),
    }
