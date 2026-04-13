from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List


_MARKETS = ("crypto", "stocks", "forex")


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _market_key(market: Any) -> str:
    mk = str(market or "").strip().lower()
    if mk == "stock":
        return "stocks"
    return mk if mk in _MARKETS else "stocks"


def _required_score(settings: Dict[str, Any], market: str) -> float:
    mk = _market_key(market)
    if mk == "stocks":
        return max(0.01, _f(settings.get("stock_score_threshold", 0.2), 0.2))
    if mk == "forex":
        return max(0.01, _f(settings.get("forex_score_threshold", 0.2), 0.2))
    # Crypto scores are dynamic rank values and are not normalized to one fixed threshold.
    return max(0.01, _f(settings.get("crypto_allocator_signal_floor", 0.15), 0.15))


def _market_status_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "trader_data.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_trader_status.json")
    return os.path.join(hub_dir, "forex", "forex_trader_status.json")


def _market_thinker_path(hub_dir: str, market: str) -> str:
    mk = _market_key(market)
    if mk == "crypto":
        return os.path.join(hub_dir, "crypto_dynamic_status.json")
    if mk == "stocks":
        return os.path.join(hub_dir, "stocks", "stock_thinker_status.json")
    return os.path.join(hub_dir, "forex", "forex_thinker_status.json")


def _side_actionable(market: str, side: str) -> bool:
    mk = _market_key(market)
    sd = str(side or "").strip().lower()
    if mk == "stocks":
        return sd == "long"
    if mk == "forex":
        return sd in {"long", "short"}
    return sd == "long"


def _soft_confidence(signal_score: float, required_score: float) -> float:
    req = max(0.01, float(required_score or 0.01))
    ratio = abs(float(signal_score or 0.0)) / req
    return _clamp(42.0 + (ratio * 18.0), 20.0, 72.0)


def _top_candidate_from_market_files(
    *,
    hub_dir: str,
    settings: Dict[str, Any],
    market: str,
    now_ts: int,
) -> Dict[str, Any]:
    mk = _market_key(market)
    thinker = _safe_read_json(_market_thinker_path(hub_dir, mk))
    trader = _safe_read_json(_market_status_path(hub_dir, mk))
    policy = trader.get("automation_policy", {}) if isinstance(trader.get("automation_policy", {}), dict) else {}
    quality = trader.get("trade_quality", {}) if isinstance(trader.get("trade_quality", {}), dict) else {}
    gate_flags = trader.get("entry_gate_flags", {}) if isinstance(trader.get("entry_gate_flags", {}), dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}

    candidate_id = ""
    side = "watch"
    signal_score = 0.0
    spread_bps = 0.0
    updated_at = int(_f(thinker.get("updated_at", thinker.get("ts", 0) or 0), 0.0))
    required = _required_score(settings, mk)
    if mk == "crypto":
        ranked = thinker.get("ranked", []) if isinstance(thinker.get("ranked", []), list) else []
        top = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
        candidate_id = str(top.get("symbol", "") or "").strip().upper()
        signal_score = _f(top.get("score", 0.0), 0.0)
        side = "long" if signal_score > 0.0 else "watch"
    elif mk == "stocks":
        top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        candidate_id = str(top.get("symbol", "") or "").strip().upper()
        side = str(top.get("side", "watch") or "watch").strip().lower()
        signal_score = _f(top.get("score", 0.0), 0.0)
        spread_bps = _f(top.get("spread_bps", 0.0), 0.0)
    else:
        top = thinker.get("top_pick", {}) if isinstance(thinker.get("top_pick", {}), dict) else {}
        candidate_id = str(top.get("pair", "") or "").strip().upper()
        side = str(top.get("side", "watch") or "watch").strip().lower()
        signal_score = _f(top.get("score", 0.0), 0.0)
        spread_bps = _f(top.get("spread_bps", 0.0), 0.0)

    if updated_at <= 0:
        updated_at = int(_f(trader.get("updated_at", trader.get("timestamp", 0) or 0), 0.0))
    candidate_age_s = max(0, int(now_ts - updated_at)) if updated_at > 0 else 10_000
    confidence = _f(quality.get("confidence_score", 0.0), 0.0)
    if confidence <= 0.0:
        confidence = _soft_confidence(signal_score, required)

    account = trader.get("account", {}) if isinstance(trader.get("account", {}), dict) else {}
    market_exposure_usd = float(max(0.0, _f(trader.get("exposure_usd", 0.0), 0.0)))
    account_value_usd = float(max(0.0, _f(trader.get("account_value_usd", 0.0), 0.0)))
    buying_power_usd = float(
        max(
            0.0,
            _f(trader.get("buying_power_usd", 0.0), 0.0),
            _f(account.get("buying_power", 0.0), 0.0),
        )
    )
    if mk == "crypto":
        market_exposure_usd = float(
            max(
                market_exposure_usd,
                _f(account.get("holdings_sell_value", 0.0), 0.0),
                _f(account.get("holdings_value", 0.0), 0.0),
                _f(account.get("holdings_usd", 0.0), 0.0),
            )
        )
        account_value_usd = float(
            max(
                account_value_usd,
                _f(account.get("total_account_value", 0.0), 0.0),
                _f(account.get("account_value_usd", 0.0), 0.0),
                _f(account.get("equity", 0.0), 0.0),
                _f(account.get("nav", 0.0), 0.0),
            )
        )

    return {
        "market": mk,
        "candidate_id": candidate_id,
        "side": side,
        "signal_score": float(signal_score),
        "required_score": float(required),
        "spread_bps": float(max(0.0, spread_bps)),
        "max_slippage_bps": float(max(0.0, _f(settings.get(f"{mk[:-1] if mk.endswith('s') else mk}_max_slippage_bps", 0.0), 0.0))),
        "candidate_age_s": int(candidate_age_s),
        "trade_quality_confidence_score": float(confidence),
        "trade_quality_decision": str(quality.get("decision", "") or ""),
        "runtime_trust_score": float(_f(trust.get("score", 0.0), _f(gate_flags.get("runtime_trust_score", 0.0), 0.0))),
        "allow_new_entries": bool(policy.get("allow_new_entries", True)),
        "market_exposure_usd": float(market_exposure_usd),
        "account_value_usd": float(account_value_usd),
        "buying_power_usd": float(buying_power_usd),
        "reason_hint": str(trader.get("entry_eval_top_reason", "") or "").strip(),
    }


def _score_candidate(snapshot: Dict[str, Any], *, stale_soft_s: int, stale_hard_s: int) -> Dict[str, Any]:
    mk = _market_key(snapshot.get("market", ""))
    side = str(snapshot.get("side", "watch") or "watch").strip().lower()
    signal = float(snapshot.get("signal_score", 0.0) or 0.0)
    required = max(0.01, float(snapshot.get("required_score", 0.01) or 0.01))
    confidence = _clamp(float(snapshot.get("trade_quality_confidence_score", 0.0) or 0.0), 0.0, 100.0)
    trust = _clamp(float(snapshot.get("runtime_trust_score", 0.0) or 0.0), 0.0, 100.0)
    spread_bps = max(0.0, float(snapshot.get("spread_bps", 0.0) or 0.0))
    max_slippage_bps = max(0.0, float(snapshot.get("max_slippage_bps", 0.0) or 0.0))
    age_s = max(0, int(snapshot.get("candidate_age_s", 0) or 0))
    allow_entries = bool(snapshot.get("allow_new_entries", True))
    quality_decision = str(snapshot.get("trade_quality_decision", "") or "").strip().lower()
    actionable = _side_actionable(mk, side)

    signal_ratio = abs(signal) / required
    signal_points = _clamp(signal_ratio / 1.6, 0.0, 1.0) * 34.0
    confidence_points = (confidence / 100.0) * 32.0
    trust_points = (trust / 100.0) * 16.0

    freshness_points = 10.0
    if age_s > stale_soft_s:
        freshness_points = max(0.0, 10.0 - ((float(age_s - stale_soft_s) / max(1.0, float(stale_hard_s - stale_soft_s))) * 10.0))

    spread_penalty = 0.0
    if max_slippage_bps > 0.0:
        spread_penalty = _clamp((spread_bps / max(1e-6, max_slippage_bps)) * 9.0, 0.0, 12.0)

    projected_total_exposure_pct = _clamp(float(snapshot.get("projected_total_exposure_pct", 0.0) or 0.0), 0.0, 300.0)
    projected_market_share_pct = _clamp(float(snapshot.get("projected_market_share_pct", 0.0) or 0.0), 0.0, 100.0)
    exposure_penalty = _clamp((projected_total_exposure_pct * 0.06) + max(0.0, projected_market_share_pct - 50.0) * 0.12, 0.0, 14.0)

    loss_streak = max(0, int(snapshot.get("loss_streak", 0) or 0))
    max_loss_streak = max(1, int(snapshot.get("max_loss_streak", 1) or 1))
    loss_ratio = _clamp(float(loss_streak) / float(max_loss_streak), 0.0, 1.5)
    discipline_penalty = 8.0 * min(1.0, loss_ratio)

    gate_penalty = 0.0
    reasons: List[str] = []
    if not actionable:
        gate_penalty += 22.0
        reasons.append("top candidate side is not actionable")
    if not allow_entries:
        gate_penalty += 20.0
        reasons.append("market policy currently blocks new entries")
    if quality_decision == "block":
        gate_penalty += 16.0
        reasons.append("trade-quality gate is blocking entry")
    if age_s > stale_hard_s:
        gate_penalty += 16.0
        reasons.append(f"candidate is stale ({age_s}s old)")

    score = signal_points + confidence_points + trust_points + freshness_points - spread_penalty - exposure_penalty - discipline_penalty - gate_penalty
    score = _clamp(score, 0.0, 100.0)
    eligible = bool(actionable and allow_entries and quality_decision != "block" and age_s <= stale_hard_s and score >= 8.0)

    return {
        "score": float(round(score, 4)),
        "eligible": bool(eligible),
        "reasons": reasons[:4],
    }


def evaluate_cross_market_allocation(
    *,
    hub_dir: str,
    settings: Dict[str, Any] | None,
    market: str,
    candidate_id: str,
    candidate_side: str,
    signal_score: float,
    required_score: float,
    trade_quality: Dict[str, Any] | None,
    automation_policy: Dict[str, Any] | None,
    projected_trade_value_usd: float,
    market_exposure_usd: float,
    account_value_usd: float,
    buying_power_usd: float,
    spread_bps: float = 0.0,
    max_slippage_bps: float = 0.0,
    candidate_age_s: int = 0,
    loss_streak: int = 0,
    max_loss_streak: int = 0,
    now_ts: int | None = None,
) -> Dict[str, Any]:
    cfg = settings if isinstance(settings, dict) else {}
    market_key = _market_key(market)
    now = int(now_ts if isinstance(now_ts, int) and now_ts > 0 else int(time.time()))
    policy = automation_policy if isinstance(automation_policy, dict) else {}
    quality = trade_quality if isinstance(trade_quality, dict) else {}
    trust = policy.get("runtime_trust", {}) if isinstance(policy.get("runtime_trust", {}), dict) else {}

    stale_soft_s = max(20, int(_f(cfg.get("portfolio_allocator_stale_soft_s", 120), 120)))
    stale_hard_s = max(stale_soft_s + 30, int(_f(cfg.get("portfolio_allocator_stale_hard_s", 420), 420)))

    current_snapshot: Dict[str, Any] = {
        "market": market_key,
        "candidate_id": str(candidate_id or "").strip().upper(),
        "side": str(candidate_side or "watch").strip().lower(),
        "signal_score": float(signal_score or 0.0),
        "required_score": float(max(0.01, float(required_score or 0.01))),
        "spread_bps": float(max(0.0, spread_bps)),
        "max_slippage_bps": float(max(0.0, max_slippage_bps)),
        "candidate_age_s": max(0, int(candidate_age_s or 0)),
        "trade_quality_confidence_score": float(max(0.0, _f(quality.get("confidence_score", 0.0), 0.0))),
        "trade_quality_decision": str(quality.get("decision", "allow") or "allow"),
        "runtime_trust_score": float(max(0.0, _f(trust.get("score", 0.0), 0.0))),
        "allow_new_entries": bool(policy.get("allow_new_entries", True)),
        "market_exposure_usd": float(max(0.0, market_exposure_usd)),
        "account_value_usd": float(max(0.0, account_value_usd)),
        "buying_power_usd": float(max(0.0, buying_power_usd)),
        "projected_trade_value_usd": float(max(0.0, projected_trade_value_usd)),
        "loss_streak": int(max(0, loss_streak)),
        "max_loss_streak": int(max(1, max_loss_streak)),
        "reason_hint": "",
    }
    if current_snapshot["trade_quality_confidence_score"] <= 0.0:
        current_snapshot["trade_quality_confidence_score"] = _soft_confidence(
            current_snapshot["signal_score"],
            current_snapshot["required_score"],
        )

    snapshots: Dict[str, Dict[str, Any]] = {market_key: current_snapshot}
    for mk in _MARKETS:
        if mk == market_key:
            continue
        snapshots[mk] = _top_candidate_from_market_files(hub_dir=hub_dir, settings=cfg, market=mk, now_ts=now)
        snapshots[mk]["projected_trade_value_usd"] = 0.0
        snapshots[mk]["loss_streak"] = int(max(0, _f((snapshots[mk].get("entry_gate_flags", {}) or {}).get("loss_streak", 0), 0.0)))
        snapshots[mk]["max_loss_streak"] = 3

    projected_total_exposure_usd = 0.0
    for mk in _MARKETS:
        row = snapshots.get(mk, {})
        if not isinstance(row, dict):
            continue
        projected_total_exposure_usd += max(0.0, _f(row.get("market_exposure_usd", 0.0), 0.0))
    projected_total_exposure_usd += max(0.0, float(projected_trade_value_usd or 0.0))
    portfolio_account_value_usd = 0.0
    for mk in _MARKETS:
        row = snapshots.get(mk, {})
        if not isinstance(row, dict):
            continue
        portfolio_account_value_usd += max(0.0, _f(row.get("account_value_usd", 0.0), 0.0))
    if portfolio_account_value_usd <= 0.0:
        portfolio_account_value_usd = max(0.0, float(account_value_usd or 0.0))

    projected_market_exposure_usd = max(0.0, float(market_exposure_usd or 0.0)) + max(0.0, float(projected_trade_value_usd or 0.0))
    projected_total_exposure_pct = (
        ((projected_total_exposure_usd / max(1e-6, float(portfolio_account_value_usd or 0.0))) * 100.0)
        if float(portfolio_account_value_usd or 0.0) > 0.0
        else 0.0
    )
    projected_market_share_pct = (
        ((projected_market_exposure_usd / max(1e-6, projected_total_exposure_usd)) * 100.0)
        if projected_total_exposure_usd > 0.0
        else 0.0
    )
    current_snapshot["projected_total_exposure_pct"] = float(projected_total_exposure_pct)
    current_snapshot["projected_market_share_pct"] = float(projected_market_share_pct)

    scores: Dict[str, float] = {}
    eligibility: Dict[str, bool] = {}
    market_reasons: Dict[str, List[str]] = {}
    for mk in _MARKETS:
        snap = snapshots.get(mk, {})
        if not isinstance(snap, dict):
            continue
        scored = _score_candidate(snap, stale_soft_s=stale_soft_s, stale_hard_s=stale_hard_s)
        scores[mk] = float(scored.get("score", 0.0) or 0.0)
        eligibility[mk] = bool(scored.get("eligible", False))
        market_reasons[mk] = [str(x or "").strip() for x in list(scored.get("reasons", []) or []) if str(x or "").strip()]

    ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    best_market = ranked[0][0] if ranked else market_key
    best_score = float(ranked[0][1]) if ranked else 0.0
    current_score = float(scores.get(market_key, 0.0))
    score_gap = float(best_score - current_score)

    global_cap_pct = max(0.0, _f(cfg.get("market_max_total_exposure_pct", 0.0), 0.0))
    concentration_warn_pct = _clamp(_f(cfg.get("portfolio_allocator_concentration_warn_pct", 68.0), 68.0), 40.0, 95.0)
    concentration_block_pct = _clamp(_f(cfg.get("portfolio_allocator_concentration_block_pct", 82.0), 82.0), concentration_warn_pct + 4.0, 99.0)
    priority_gap_min = _clamp(_f(cfg.get("portfolio_allocator_priority_gap_min", 8.0), 8.0), 2.0, 30.0)
    buying_power_buffer_mult = _clamp(_f(cfg.get("portfolio_allocator_buying_power_buffer_mult", 1.1), 1.1), 1.0, 2.0)

    buying_power_tight = bool(float(buying_power_usd or 0.0) > 0.0 and float(projected_trade_value_usd or 0.0) > 0.0 and float(buying_power_usd or 0.0) <= (float(projected_trade_value_usd or 0.0) * buying_power_buffer_mult))
    near_global_cap = bool(global_cap_pct > 0.0 and projected_total_exposure_pct >= (global_cap_pct * 0.9))
    concentration_warn = bool(projected_market_share_pct >= concentration_warn_pct)
    concentration_block = bool(projected_market_share_pct >= concentration_block_pct)
    capital_constrained = bool(buying_power_tight or near_global_cap or concentration_warn)

    decision = "allow"
    reasons: List[str] = []

    if not bool(eligibility.get(market_key, False)):
        decision = "block"
        reasons.extend(market_reasons.get(market_key, []))
        if not reasons:
            reasons.append(f"{market_key.title()} candidate is not currently eligible")
    else:
        best_eligible = bool(eligibility.get(best_market, False))
        if best_market != market_key and best_eligible and score_gap >= priority_gap_min and capital_constrained:
            decision = "deprioritize"
            reasons.append(
                f"{best_market.title()} has higher opportunity quality ({best_score:.1f} vs {current_score:.1f}) while capital is constrained"
            )
        elif best_market != market_key and best_eligible and concentration_block and score_gap >= 3.0:
            decision = "block"
            reasons.append(
                f"Blocked due to concentration risk ({projected_market_share_pct:.1f}% in {market_key}) while {best_market} is comparable"
            )
        elif concentration_warn and best_market == market_key:
            reasons.append(f"Portfolio concentration in {market_key} is elevated ({projected_market_share_pct:.1f}%)")

    if near_global_cap:
        reasons.append(f"Projected global exposure is near cap ({projected_total_exposure_pct:.1f}% of {global_cap_pct:.1f}%)")
    if buying_power_tight:
        reasons.append("Remaining buying power is tight for this entry")

    if decision == "allow":
        summary = (
            f"{market_key.title()} selected as best current opportunity"
            if best_market == market_key
            else f"{market_key.title()} allowed despite {best_market.title()} leading score"
        )
    elif decision == "deprioritize":
        summary = (
            f"{market_key.title()} candidate deprioritized because {best_market.title()} has higher opportunity quality and capital is constrained"
        )
    else:
        summary = reasons[0] if reasons else f"{market_key.title()} candidate blocked by portfolio allocator"

    return {
        "ts": int(now),
        "market": market_key,
        "candidate_id": str(candidate_id or "").strip().upper(),
        "decision": str(decision),
        "summary": str(summary)[:220],
        "reasons": [str(x or "").strip() for x in reasons[:6] if str(x or "").strip()],
        "best_market": str(best_market),
        "best_market_score": round(float(best_score), 4),
        "current_market_score": round(float(current_score), 4),
        "score_gap": round(float(score_gap), 4),
        "capital_constrained": bool(capital_constrained),
        "portfolio_account_value_usd": round(float(max(0.0, portfolio_account_value_usd)), 4),
        "projected_trade_value_usd": round(float(max(0.0, projected_trade_value_usd)), 4),
        "projected_total_exposure_pct": round(float(max(0.0, projected_total_exposure_pct)), 4),
        "projected_market_share_pct": round(float(max(0.0, projected_market_share_pct)), 4),
        "scores": {mk: round(float(scores.get(mk, 0.0)), 4) for mk in _MARKETS},
        "eligibility": {mk: bool(eligibility.get(mk, False)) for mk in _MARKETS},
        "market_reasons": {mk: list(market_reasons.get(mk, [])) for mk in _MARKETS},
    }


def summarize_allocator_snapshot(
    stocks: Dict[str, Any] | None,
    forex: Dict[str, Any] | None,
    crypto: Dict[str, Any] | None,
) -> Dict[str, Any]:
    market_rows = {
        "stocks": stocks if isinstance(stocks, dict) else {},
        "forex": forex if isinstance(forex, dict) else {},
        "crypto": crypto if isinstance(crypto, dict) else {},
    }

    allocator_rows: Dict[str, Dict[str, Any]] = {}
    latest: Dict[str, Any] = {}
    latest_ts = 0
    for mk, row in market_rows.items():
        alloc = row.get("opportunity_allocator", {}) if isinstance(row.get("opportunity_allocator", {}), dict) else {}
        if alloc:
            allocator_rows[mk] = dict(alloc)
            ts = int(_f(alloc.get("ts", 0), 0.0))
            if ts >= latest_ts:
                latest = dict(alloc)
                latest_ts = ts

    if not allocator_rows:
        return {"active": False, "ts": int(time.time()), "summary": "", "scores": {}, "decisions": {}, "best_market": ""}

    scores = latest.get("scores", {}) if isinstance(latest.get("scores", {}), dict) else {}
    if not scores:
        scores = {}
        for mk in _MARKETS:
            row = allocator_rows.get(mk, {})
            if not isinstance(row, dict):
                continue
            scores[mk] = _f(row.get("current_market_score", 0.0), 0.0)
    best_market = str(latest.get("best_market", "") or "")
    if not best_market and scores:
        ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
        best_market = str(ranked[0][0])
    best_score = _f(((scores or {}).get(best_market, 0.0) if best_market else 0.0), 0.0)

    decisions: Dict[str, str] = {}
    deprioritized: List[Dict[str, Any]] = []
    for mk in _MARKETS:
        row = allocator_rows.get(mk, {})
        decision = str(row.get("decision", "") or "").strip().lower()
        if not decision:
            decision = "unknown"
        decisions[mk] = decision
        if decision in {"deprioritize", "block"}:
            deprioritized.append(
                {
                    "market": mk,
                    "decision": decision,
                    "summary": str(row.get("summary", "") or "")[:220],
                }
            )

    summary = str(latest.get("summary", "") or "").strip()
    if not summary and best_market:
        summary = f"Best current opportunity: {best_market.title()} ({best_score:.1f})"

    return {
        "active": True,
        "ts": int(max(latest_ts, int(time.time()))),
        "best_market": str(best_market),
        "best_score": round(float(best_score), 4),
        "scores": {mk: round(float(_f(scores.get(mk, 0.0), 0.0)), 4) for mk in _MARKETS},
        "decisions": decisions,
        "deprioritized_markets": deprioritized[:6],
        "summary": summary[:220],
    }
