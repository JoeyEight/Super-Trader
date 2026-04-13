from __future__ import annotations

from typing import Any, Dict

from app.settings_utils import normalize_settings_profile

_PROFILE_LABELS: Dict[str, str] = {
    "safe": "Safe",
    "balanced": "Balanced",
    "aggressive": "Aggressive",
    "max_growth": "Max Growth",
}

_PROFILE_AGGRESSION: Dict[str, float] = {
    "safe": 0.90,
    "balanced": 1.00,
    "aggressive": 1.10,
    "max_growth": 1.20,
}

_MARKET_AGGRESSION: Dict[str, float] = {
    "crypto": 1.16,
    "stocks": 1.00,
    "forex": 1.00,
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _capital_bucket(account_value_usd: float) -> str:
    val = max(0.0, float(account_value_usd or 0.0))
    if val >= 50_000.0:
        return "large"
    if val >= 10_000.0:
        return "mid"
    if val >= 2_500.0:
        return "small"
    return "micro"


def profile_label(profile_key: Any) -> str:
    key = normalize_settings_profile(profile_key)
    return str(_PROFILE_LABELS.get(key, "Balanced"))


def runtime_trust_snapshot(runtime_alerts: Dict[str, Any] | None, market_health: Dict[str, Any] | None = None) -> Dict[str, Any]:
    alerts = runtime_alerts if isinstance(runtime_alerts, dict) else {}
    health = market_health if isinstance(market_health, dict) else {}
    severity = str(alerts.get("severity", "ok") or "ok").strip().lower()
    score = 84.0
    reasons: list[str] = []
    if severity in {"critical", "error"}:
        score = 36.0
        reasons.append("Runtime alerts are critical")
    elif severity in {"warn", "warning"}:
        score = 62.0
        reasons.append("Runtime alerts are elevated")
    if not bool(health.get("data_ok", True)):
        score -= 20.0
        reasons.append("Data quality degraded")
    if not bool(health.get("broker_ok", True)):
        score -= 22.0
        reasons.append("Broker/API health degraded")
    if not bool(health.get("orders_ok", True)):
        score -= 12.0
        reasons.append("Order reliability degraded")
    if bool(health.get("drift_warning", False)):
        score -= 8.0
        reasons.append("Execution drift warning active")
    score = _clamp(score, 0.0, 100.0)
    if score < 30.0:
        mode = "restricted"
        size_mult = 0.40
        allow_entries = False
    elif score < 65.0:
        mode = "cautious"
        size_mult = 0.72
        allow_entries = True
    else:
        mode = "normal"
        size_mult = 1.0
        allow_entries = True
    return {
        "score": round(float(score), 2),
        "severity": severity if severity else "ok",
        "mode": mode,
        "size_multiplier": round(float(size_mult), 4),
        "allow_new_entries": bool(allow_entries),
        "reasons": reasons[:4],
    }


def stock_compliance_state(
    account_summary: Dict[str, Any] | None,
    equity_usd: float,
    pdt_equity_threshold_usd: float,
    day_trades_rolling_5d: int,
    pdt_max_day_trades_rolling_5d: int,
) -> Dict[str, Any]:
    acct = account_summary if isinstance(account_summary, dict) else {}
    equity = max(0.0, float(equity_usd or 0.0))
    pdt_threshold = max(0.0, float(pdt_equity_threshold_usd or 0.0))
    rolling = max(0, int(day_trades_rolling_5d or 0))
    limit = max(0, int(pdt_max_day_trades_rolling_5d or 0))
    multiplier = _f(acct.get("multiplier", 0.0), 0.0)
    acct_type = str(acct.get("account_type", acct.get("type", "")) or "").strip().lower()
    daytrade_count = max(0, int(_f(acct.get("daytrade_count", 0), 0.0)))
    pattern_day_trader = bool(acct.get("pattern_day_trader", False))
    trading_blocked = bool(acct.get("trading_blocked", False))
    if acct_type == "cash":
        account_mode = "cash"
    elif acct_type in {"margin", "limited_margin"}:
        account_mode = "margin"
    elif multiplier > 1.0 or pattern_day_trader or daytrade_count > 0:
        account_mode = "margin"
    elif multiplier > 0.0:
        account_mode = "cash"
    else:
        account_mode = "margin"
    under_threshold = bool(pdt_threshold > 0.0 and equity > 0.0 and equity < pdt_threshold)
    pdt_restricted = bool(account_mode == "margin" and under_threshold)
    remaining = max(0, limit - rolling) if limit > 0 else 0
    entry_blocked = False
    entry_block_reason = ""
    if trading_blocked:
        entry_blocked = True
        entry_block_reason = "Broker account reports stock trading is blocked."
    elif pdt_restricted and limit > 0 and rolling >= limit:
        entry_blocked = True
        entry_block_reason = "Blocked to avoid PDT violation risk"
    if account_mode == "cash":
        mode = "cash_account_protection"
        status_text = "Stock automation is running in cash-account compliance mode"
    elif pdt_restricted:
        mode = "under_25k_pdt_protection"
        status_text = "Stock automation is running in under-$25k compliance mode"
    else:
        mode = "standard_margin"
        status_text = "Stock automation is running in standard margin compliance mode"
    return {
        "mode": mode,
        "account_mode": account_mode,
        "pattern_day_trader": bool(pattern_day_trader),
        "pdt_restricted": bool(pdt_restricted),
        "under_pdt_equity_threshold": bool(under_threshold),
        "pdt_equity_threshold_usd": round(float(pdt_threshold), 4),
        "day_trades_rolling_5d": int(rolling),
        "pdt_max_day_trades_rolling_5d": int(limit),
        "remaining_day_trades_rolling_5d": int(remaining),
        "entry_blocked": bool(entry_blocked),
        "entry_block_reason": str(entry_block_reason),
        "status_text": status_text,
    }


def build_market_automation_policy(
    market: str,
    settings: Dict[str, Any] | None,
    profile_key: Any,
    broker_mode: str,
    account_value_usd: float,
    buying_power_usd: float,
    open_positions: int,
    runtime_alerts: Dict[str, Any] | None = None,
    market_health: Dict[str, Any] | None = None,
    compliance_state: Dict[str, Any] | None = None,
    reject_rate_pct: float = 0.0,
    reject_rate_limit_pct: float = 0.0,
    fallback_active: bool = False,
    fallback_age_s: int = 0,
    fallback_hard_block_age_s: int = 0,
    loss_streak: int = 0,
    max_loss_streak: int = 0,
) -> Dict[str, Any]:
    mk = str(market or "").strip().lower()
    cfg = settings if isinstance(settings, dict) else {}
    profile = normalize_settings_profile(profile_key)
    trust = runtime_trust_snapshot(runtime_alerts, market_health=market_health)
    base_mult = _PROFILE_AGGRESSION.get(profile, 0.90) * _MARKET_AGGRESSION.get(mk, 1.0)
    size_mult = float(base_mult) * float(trust.get("size_multiplier", 1.0) or 1.0)
    summary_bits: list[str] = [f"{profile_label(profile)} preset"]
    effective_limits: Dict[str, Any] = {}
    if mk in {"stocks", "forex"}:
        scan_key = "market_bg_stocks_interval_s" if mk == "stocks" else "market_bg_forex_interval_s"
        try:
            scan_interval_s = float(cfg.get(scan_key, 0.0) or 0.0)
        except Exception:
            scan_interval_s = 0.0
    else:
        scan_interval_s = float(_f(cfg.get("crypto_dynamic_scan_interval_s", 0.0), 0.0))
    if scan_interval_s > 0.0:
        summary_bits.append(f"scan every {int(max(1, round(scan_interval_s)))}s")
    if mk == "crypto":
        capital_tier = _capital_bucket(account_value_usd)
        target_count_raw = max(1, int(_f(cfg.get("crypto_dynamic_target_count", 8), 8.0)))
        max_new_raw = max(1, int(_f(cfg.get("crypto_dynamic_max_new_per_scan", 1), 1.0)))
        rotation_cooldown_raw = max(30.0, float(_f(cfg.get("crypto_dynamic_rotation_cooldown_s", 900.0), 900.0)))
        min_edge_raw = max(0.0, float(_f(cfg.get("crypto_dynamic_min_projected_edge_pct", 0.25), 0.25)))
        max_spread_raw = max(1.0, float(_f(cfg.get("crypto_max_spread_bps", 150.0), 150.0)))
        max_open_raw = max(1, int(_f(cfg.get("crypto_max_open_positions", 8), 8.0)))
        tier_target_bias = {"micro": -1, "small": 0, "mid": 1, "large": 2}.get(capital_tier, 0)
        tier_new_bias = {"micro": 0, "small": 0, "mid": 1, "large": 1}.get(capital_tier, 0)
        tier_rotation_mult = {"micro": 1.10, "small": 1.0, "mid": 0.92, "large": 0.85}.get(capital_tier, 1.0)
        effective_target_count = max(3, min(30, int(target_count_raw + tier_target_bias)))
        effective_max_new = max(1, min(8, int(max_new_raw + tier_new_bias)))
        effective_rotation_cooldown_s = max(45.0, float(rotation_cooldown_raw) * float(tier_rotation_mult))
        effective_min_edge_pct = float(max(0.01, min(5.0, min_edge_raw)))
        effective_max_spread_bps = float(max_spread_raw)
        trust_mode = str(trust.get("mode", "normal") or "normal").strip().lower()
        if profile == "max_growth":
            if trust_mode == "normal":
                effective_max_spread_bps = max(effective_max_spread_bps, min(250.0, float(max_spread_raw) + 25.0))
            elif trust_mode == "cautious":
                effective_max_spread_bps = max(effective_max_spread_bps, min(230.0, float(max_spread_raw) + 10.0))
        if trust_mode == "restricted":
            effective_max_spread_bps = max(40.0, min(float(effective_max_spread_bps), float(max_spread_raw) * 0.90))
        if bool(fallback_active):
            effective_max_new = max(1, min(effective_max_new, 2))
            effective_rotation_cooldown_s *= 1.15
        if str(trust.get("mode", "normal")) == "cautious":
            effective_max_new = max(1, min(effective_max_new, 2))
            effective_rotation_cooldown_s *= 1.15
        if str(trust.get("mode", "normal")) == "restricted":
            effective_max_new = 1
            effective_rotation_cooldown_s *= 1.30
        summary_bits.append(f"rotation every {int(max(1, round(effective_rotation_cooldown_s)))}s")
        summary_bits.append(f"up to {effective_target_count} symbols")
        summary_bits.append(f"spread guard <= {int(max(1, round(effective_max_spread_bps)))}bps")
        effective_limits = {
            "capital_tier": str(capital_tier),
            "target_symbols": int(effective_target_count),
            "max_new_entries_per_scan": int(effective_max_new),
            "rotation_cooldown_s": round(float(effective_rotation_cooldown_s), 3),
            "min_projected_edge_pct": round(float(effective_min_edge_pct), 4),
            "max_spread_bps": round(float(effective_max_spread_bps), 4),
            "max_open_positions": int(max_open_raw),
        }
    if bool(fallback_active):
        summary_bits.append("cached scan fallback active")
    if int(fallback_hard_block_age_s) > 0 and int(fallback_age_s) > int(fallback_hard_block_age_s):
        size_mult *= 0.70
        summary_bits.append("fallback data is aging")
    rej_rate = max(0.0, min(100.0, float(reject_rate_pct or 0.0)))
    rej_limit = max(0.0, min(100.0, float(reject_rate_limit_pct or 0.0)))
    if rej_limit > 0.0 and rej_rate >= (rej_limit * 0.9):
        size_mult *= 0.86
        summary_bits.append("scanner reject pressure elevated")
    if max_loss_streak > 0 and loss_streak > 0:
        loss_ratio = _clamp(float(loss_streak) / float(max(1, max_loss_streak)), 0.0, 1.5)
        size_mult *= max(0.60, 1.0 - (0.30 * min(1.0, loss_ratio)))
        summary_bits.append("loss-streak scaling active")
    compliance = compliance_state if isinstance(compliance_state, dict) else {}
    allow_entries = bool(trust.get("allow_new_entries", True))
    compliance_block = bool(compliance.get("entry_blocked", False))
    if compliance_block:
        allow_entries = False
        summary_bits.append(str(compliance.get("entry_block_reason", "compliance protection active")))
    if mk == "stocks" and compliance:
        summary_bits.append(str(compliance.get("status_text", "")).strip())
    size_mult_ceiling = 1.55 if mk == "crypto" else 1.35
    size_mult = _clamp(size_mult, 0.20, size_mult_ceiling)
    if not allow_entries:
        mode = "restricted"
        size_mult = min(size_mult, 0.35)
    elif mk == "crypto" and profile in {"aggressive", "max_growth"}:
        mode = "aggressive_rotation" if str(trust.get("mode", "normal")) == "normal" else "aggressive_guarded"
    elif profile in {"aggressive", "max_growth"}:
        mode = "aggressive" if str(trust.get("mode", "normal")) == "normal" else "aggressive_guarded"
    elif profile == "safe":
        mode = "safe"
    else:
        mode = "balanced"
    exposure_cap_key = {
        "stocks": "stock_max_total_exposure_pct",
        "forex": "forex_max_total_exposure_pct",
    }.get(mk, "max_total_exposure_pct")
    exposure_cap_pct = max(0.0, float(_f(cfg.get(exposure_cap_key, 0.0), 0.0)))
    global_exposure_cap_pct = max(0.0, float(_f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0)))
    effective_limits = dict(effective_limits or {})
    effective_limits["market_exposure_cap_pct"] = round(float(exposure_cap_pct), 4)
    effective_limits["global_exposure_cap_pct"] = round(float(global_exposure_cap_pct), 4)
    if mk in {"stocks", "forex"}:
        daily_loss_pct_key = "stock_max_daily_loss_pct" if mk == "stocks" else "forex_max_daily_loss_pct"
        daily_loss_usd_key = "stock_max_daily_loss_usd" if mk == "stocks" else "forex_max_daily_loss_usd"
        effective_limits["daily_loss_pct"] = round(max(0.0, float(_f(cfg.get(daily_loss_pct_key, 0.0), 0.0))), 4)
        effective_limits["daily_loss_usd"] = round(max(0.0, float(_f(cfg.get(daily_loss_usd_key, 0.0), 0.0))), 4)
    summary = "; ".join([part for part in summary_bits if str(part or "").strip()][:4])
    if not summary:
        summary = f"{profile_label(profile)} policy active"
    return {
        "market": mk,
        "profile": profile,
        "profile_label": profile_label(profile),
        "mode": mode,
        "broker_mode": str(broker_mode or "").strip().lower(),
        "account_value_usd": round(max(0.0, float(account_value_usd or 0.0)), 4),
        "buying_power_usd": round(max(0.0, float(buying_power_usd or 0.0)), 4),
        "open_positions": max(0, int(open_positions or 0)),
        "scan_interval_s": round(max(0.0, float(scan_interval_s)), 3),
        "exposure_cap_pct": round(float(exposure_cap_pct), 4),
        "runtime_trust": trust,
        "compliance": compliance if isinstance(compliance, dict) else {},
        "allow_new_entries": bool(allow_entries),
        "size_multiplier": round(float(size_mult), 4),
        "effective_limits": effective_limits if isinstance(effective_limits, dict) else {},
        "summary": summary,
        "status_chips": [chip for chip in [profile_label(profile), mode.replace("_", " ").title(), str(trust.get("mode", "")).title()] if chip],
    }


def summarize_policy_snapshot(
    stocks: Dict[str, Any] | None,
    forex: Dict[str, Any] | None,
    crypto: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _one(row: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = row if isinstance(row, dict) else {}
        pol = payload.get("automation_policy", {}) if isinstance(payload.get("automation_policy", {}), dict) else {}
        trust = pol.get("runtime_trust", {}) if isinstance(pol.get("runtime_trust", {}), dict) else {}
        compliance = pol.get("compliance", {}) if isinstance(pol.get("compliance", {}), dict) else {}
        return {
            "mode": str(pol.get("mode", "") or ""),
            "profile": str(pol.get("profile", "") or ""),
            "summary": str(pol.get("summary", "") or ""),
            "allow_new_entries": bool(pol.get("allow_new_entries", True)),
            "size_multiplier": float(_f(pol.get("size_multiplier", 1.0), 1.0)),
            "runtime_trust_score": float(_f(trust.get("score", 0.0), 0.0)),
            "compliance_mode": str(compliance.get("mode", "") or ""),
            "compliance_status": str(compliance.get("status_text", "") or ""),
            "effective_limits": dict(pol.get("effective_limits", {}) or {}) if isinstance(pol.get("effective_limits", {}), dict) else {},
        }

    return {
        "crypto": _one(crypto),
        "stocks": _one(stocks),
        "forex": _one(forex),
    }
