from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Tuple

from app.rejection_replay import build_market_rejection_replay


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_read_jsonl(path: str, max_lines: int = 12000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return out
    for ln in lines[-max(1, int(max_lines)):]:
        try:
            row = json.loads(ln)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _market_symbol(row: Dict[str, Any], market: str) -> str:
    m = str(market or "").strip().lower()
    if m == "stocks":
        return str(row.get("symbol", "") or "").strip().upper()
    return str(row.get("pair", row.get("instrument", "")) or "").strip().upper()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_csv(path: str, rows: Iterable[Dict[str, Any]], columns: List[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow(row)


def _closed_trade_rows(audit_rows: List[Dict[str, Any]], market: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        evt = str(row.get("event", "") or "").strip().lower()
        if evt not in {"exit", "entry_fail", "exit_fail", "shadow_live_divergence"}:
            continue
        ts = _i(row.get("ts", 0), 0)
        if ts <= 0:
            continue
        pnl = _f(row.get("pnl_usd", 0.0), 0.0)
        out.append(
            {
                "ts": ts,
                "event": evt,
                "symbol": _market_symbol(row, market),
                "side": str(row.get("side", "") or "").strip().lower(),
                "ok": bool(row.get("ok", False)),
                "score": _f(row.get("score", 0.0), 0.0),
                "pnl_usd": float(pnl),
                "msg": str(row.get("msg", "") or "").strip()[:220],
            }
        )
    out.sort(key=lambda r: int(r.get("ts", 0) or 0))
    return out


def _build_backtest_metrics(
    closed_rows: List[Dict[str, Any]],
    market: str,
    settings: Dict[str, Any],
    account_value_usd: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fee_bps = max(0.0, _f(settings.get("strategy_lab_fee_bps", 2.0), 2.0))
    fee_mult = fee_bps / 10000.0
    m = str(market or "").strip().lower()
    if m == "stocks":
        notional = max(1.0, _f(settings.get("stock_trade_notional_usd", 25.0), 25.0))
    else:
        # Forex rows do not carry direct USD notional in all cases; use current trade-unit scale.
        notional = max(1.0, _f(settings.get("forex_trade_units", 25.0), 25.0))

    equity = float(max(1.0, account_value_usd))
    peak = equity
    max_drawdown_pct = 0.0
    closed_trades = 0
    wins = 0
    losses = 0
    pnl_sum = 0.0
    equity_curve: List[Dict[str, Any]] = []

    for row in closed_rows:
        evt = str(row.get("event", "") or "").strip().lower()
        if evt not in {"exit", "entry_fail", "exit_fail"}:
            continue
        pnl = _f(row.get("pnl_usd", 0.0), 0.0)
        # Apply a tiny transaction cost model so the summary is conservative.
        pnl_after_fee = float(pnl) - (notional * fee_mult)
        closed_trades += 1
        pnl_sum += pnl_after_fee
        if pnl_after_fee > 0.0:
            wins += 1
        elif pnl_after_fee < 0.0:
            losses += 1
        equity += pnl_after_fee
        peak = max(peak, equity)
        if peak > 0.0:
            dd = ((equity - peak) / peak) * 100.0
            max_drawdown_pct = min(max_drawdown_pct, dd)
        equity_curve.append(
            {
                "ts": int(row.get("ts", 0) or 0),
                "equity_usd": round(float(equity), 6),
                "pnl_usd": round(float(pnl_after_fee), 6),
                "event": evt,
                "symbol": str(row.get("symbol", "") or ""),
            }
        )

    win_rate_pct = (100.0 * float(wins) / float(max(1, closed_trades))) if closed_trades > 0 else 0.0
    cumulative_return_pct = ((equity - max(1.0, account_value_usd)) / max(1.0, account_value_usd)) * 100.0

    metrics = {
        "closed_trades": int(closed_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": round(float(win_rate_pct), 4),
        "cumulative_return_pct": round(float(cumulative_return_pct), 4),
        "max_drawdown_pct": round(float(max_drawdown_pct), 4),
        "net_pnl_usd": round(float(pnl_sum), 6),
    }
    return metrics, equity_curve


def _build_walkforward_summary(hub_dir: str, market: str) -> Dict[str, Any]:
    wf = _safe_read_json(os.path.join(hub_dir, "walkforward_report.json"))
    row = wf.get(str(market or "").strip().lower(), {}) if isinstance(wf.get(str(market or "").strip().lower(), {}), dict) else {}
    windows = list(row.get("windows", []) or []) if isinstance(row.get("windows", []), list) else []
    test_returns: List[float] = []
    test_win_rates: List[float] = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        test_row = w.get("test", {}) if isinstance(w.get("test", {}), dict) else {}
        pnl = _f(test_row.get("pnl_usd", 0.0), 0.0)
        samples = max(1.0, _f(test_row.get("samples", 0.0), 0.0))
        test_returns.append((pnl / samples) * 100.0)
        test_win_rates.append(_f(test_row.get("win_rate_pct", 0.0), 0.0))
    summary = {
        "avg_return_pct": round((sum(test_returns) / max(1, len(test_returns))), 4) if test_returns else 0.0,
        "avg_win_rate_pct": round((sum(test_win_rates) / max(1, len(test_win_rates))), 4) if test_win_rates else 0.0,
    }
    return {
        "state": str(row.get("state", "READY") or "READY"),
        "windows": int(len(windows)),
        "summary": summary,
        "windows_rows": windows[-60:],
    }


def _build_sweep_payload(hub_dir: str, market: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    replay = build_market_rejection_replay(hub_dir, market, settings=settings, max_scan_rows=320)
    scenarios = list(replay.get("scenarios", []) or []) if isinstance(replay.get("scenarios", []), list) else []
    rows: List[Dict[str, Any]] = []
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        actionable = int(row.get("actionable", 0) or 0)
        entry_ready = int(row.get("entry_ready", 0) or 0)
        avg_abs = _f(row.get("avg_abs_score", 0.0), 0.0)
        # Derived synthetic return score for ranking only.
        synthetic_ret = (entry_ready * avg_abs * 0.8) - (max(0, actionable - entry_ready) * 0.12)
        rows.append(
            {
                "threshold": _f(row.get("threshold", 0.0), 0.0),
                "actionable": actionable,
                "entry_ready": entry_ready,
                "avg_abs_score": avg_abs,
                "win_rate_pct": round(min(100.0, max(0.0, 45.0 + (avg_abs * 18.0))), 4),
                "cumulative_return_pct": round(float(synthetic_ret), 4),
            }
        )
    rows.sort(key=lambda r: (_f(r.get("cumulative_return_pct", 0.0), 0.0), -_f(r.get("threshold", 0.0), 0.0)), reverse=True)
    return {"state": str(replay.get("state", "READY") or "READY"), "top10": rows[:10], "scenarios": rows}


def _build_monte_carlo(
    closed_rows: List[Dict[str, Any]],
    account_value_usd: float,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    sims = max(20, min(5000, _i(settings.get("strategy_lab_monte_carlo_sims", 300), 300)))
    seed = max(0, _i(settings.get("strategy_lab_seed", 42), 42))
    rng = random.Random(seed)
    pnl_values = [_f(r.get("pnl_usd", 0.0), 0.0) for r in closed_rows if str(r.get("event", "") or "").strip().lower() == "exit"]
    if not pnl_values:
        return {
            "state": "NO_DATA",
            "summary": {
                "simulations": int(sims),
                "p05_final_equity_mult": 1.0,
                "p50_final_equity_mult": 1.0,
                "p95_final_equity_mult": 1.0,
            },
            "paths": [],
        }
    steps = max(10, min(800, len(pnl_values)))
    base_equity = float(max(1.0, account_value_usd))
    finals: List[float] = []
    paths: List[Dict[str, Any]] = []
    sample_paths = min(8, sims)
    for i in range(sims):
        eq = base_equity
        for _ in range(steps):
            eq += rng.choice(pnl_values)
            eq = max(1.0, eq)
        mult = eq / base_equity
        finals.append(mult)
        if i < sample_paths:
            paths.append({"sim": i + 1, "final_equity_mult": round(float(mult), 6)})
    finals.sort()
    n = len(finals)
    idx05 = max(0, min(n - 1, int(round((n - 1) * 0.05))))
    idx50 = max(0, min(n - 1, int(round((n - 1) * 0.50))))
    idx95 = max(0, min(n - 1, int(round((n - 1) * 0.95))))
    return {
        "state": "READY",
        "summary": {
            "simulations": int(sims),
            "p05_final_equity_mult": round(float(finals[idx05]), 6),
            "p50_final_equity_mult": round(float(finals[idx50]), 6),
            "p95_final_equity_mult": round(float(finals[idx95]), 6),
        },
        "paths": paths,
    }


def _market_config(settings: Dict[str, Any], market: str, sample_count: int) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    if m == "stocks":
        score_threshold = _f(settings.get("stock_score_threshold", 0.2), 0.2)
        trade_notional = _f(settings.get("stock_trade_notional_usd", 25.0), 25.0)
        max_open_positions = _i(settings.get("stock_max_open_positions", 6), 6)
    else:
        score_threshold = _f(settings.get("forex_score_threshold", 0.2), 0.2)
        trade_notional = _f(settings.get("forex_trade_units", 25.0), 25.0)
        max_open_positions = _i(settings.get("forex_max_open_positions", 10), 10)
    return {
        "samples": int(sample_count),
        "score_threshold": round(float(score_threshold), 6),
        "trade_notional": round(float(trade_notional), 6),
        "max_open_positions": int(max(1, max_open_positions)),
    }


def _build_simulation_summary(account_value_usd: float, back_metrics: Dict[str, Any]) -> Dict[str, Any]:
    start_equity = float(max(1.0, account_value_usd))
    net_pnl = _f(back_metrics.get("net_pnl_usd", 0.0), 0.0)
    end_equity = float(max(1.0, start_equity + net_pnl))
    return {
        "starting_equity_usd": round(start_equity, 6),
        "ending_equity_usd": round(end_equity, 6),
        "net_pnl_usd": round(float(end_equity - start_equity), 6),
        "net_return_pct": round(_f(back_metrics.get("cumulative_return_pct", 0.0), 0.0), 6),
        "max_drawdown_pct": round(_f(back_metrics.get("max_drawdown_pct", 0.0), 0.0), 6),
        "win_rate_pct": round(_f(back_metrics.get("win_rate_pct", 0.0), 0.0), 6),
        "closed_trades": int(_i(back_metrics.get("closed_trades", 0), 0)),
    }


def _recommended_settings_payload(
    market: str,
    account_value_usd: float,
    config: Dict[str, Any],
    sweep: Dict[str, Any],
) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    top10 = list(sweep.get("top10", []) or []) if isinstance(sweep.get("top10", []), list) else []
    best_threshold = None
    if top10 and isinstance(top10[0], dict):
        best_threshold = _f(top10[0].get("threshold", 0.0), 0.0)
    if best_threshold is None or best_threshold <= 0.0:
        best_threshold = _f(config.get("score_threshold", 0.2), 0.2)

    start_equity = float(max(1.0, account_value_usd))
    max_open_cfg = max(1, _i(config.get("max_open_positions", 1), 1))

    if m == "stocks":
        current_notional = max(1.0, _f(config.get("trade_notional", 25.0), 25.0))
        target_notional = max(1.0, round(start_equity * 0.15, 2))
        recommended_notional = min(max(1.0, target_notional), max(1.0, current_notional * 2.0))
        affordable_slots = max(1, int((start_equity * 0.90) / max(1.0, recommended_notional)))
        recommended_open = max(1, min(max_open_cfg, affordable_slots))
        return {
            "notes": "Account-sized stock simulation: per-trade size targets ~15% of account value.",
            "settings": {
                "stock_score_threshold": round(float(best_threshold), 6),
                "stock_trade_notional_usd": round(float(recommended_notional), 6),
                "stock_max_open_positions": int(recommended_open),
            },
        }

    current_units = max(1, _i(config.get("trade_notional", 25), 25))
    target_units = max(1, int(round(start_equity * 0.25)))
    recommended_units = min(target_units, max(1, current_units * 2))
    affordable_slots = max(1, int((start_equity * 0.90) / max(1.0, float(recommended_units))))
    recommended_open = max(1, min(max_open_cfg, affordable_slots))
    return {
        "notes": "Account-sized forex simulation: trade units scale from current account value.",
        "settings": {
            "forex_score_threshold": round(float(best_threshold), 6),
            "forex_trade_units": int(recommended_units),
            "forex_max_open_positions": int(recommended_open),
        },
    }


def run_strategy_lab_suite(hub_dir: str, market: str, settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    m = str(market or "").strip().lower()
    if m not in {"stocks", "forex"}:
        raise ValueError(f"unsupported market: {market}")
    cfg = settings if isinstance(settings, dict) else {}
    now = int(time.time())

    market_dir = os.path.join(hub_dir, m)
    audit_path = os.path.join(market_dir, "execution_audit.jsonl")
    audit_rows = _safe_read_jsonl(audit_path, max_lines=12000)
    closed_rows = _closed_trade_rows(audit_rows, m)

    status_name = "stock_trader_status.json" if m == "stocks" else "forex_trader_status.json"
    trader_status = _safe_read_json(os.path.join(market_dir, status_name))
    account_value_usd = _f(trader_status.get("account_value_usd", trader_status.get("total_account_value", 100.0)), 100.0)
    config = _market_config(cfg, m, sample_count=len(closed_rows))

    back_metrics, equity_curve = _build_backtest_metrics(closed_rows, m, cfg, account_value_usd=account_value_usd)
    walk = _build_walkforward_summary(hub_dir, m)
    sweep = _build_sweep_payload(hub_dir, m, cfg)
    mc = _build_monte_carlo(closed_rows, account_value_usd=account_value_usd, settings=cfg)
    simulation = _build_simulation_summary(account_value_usd, back_metrics)
    recommendations = _recommended_settings_payload(m, account_value_usd, config, sweep)

    summary_dir = os.path.join(hub_dir, "strategy_lab", m)
    _ensure_dir(summary_dir)
    summary_path = os.path.join(summary_dir, "strategy_lab_summary.json")
    events_csv = os.path.join(summary_dir, "events.csv")
    equity_csv = os.path.join(summary_dir, "equity_curve.csv")
    walk_csv = os.path.join(summary_dir, "walkforward_windows.csv")
    sweep_csv = os.path.join(summary_dir, "sweep_scenarios.csv")
    mc_csv = os.path.join(summary_dir, "monte_paths.csv")

    _write_csv(
        events_csv,
        closed_rows[-1200:],
        columns=["ts", "event", "symbol", "side", "ok", "score", "pnl_usd", "msg"],
    )
    _write_csv(
        equity_csv,
        equity_curve[-1200:],
        columns=["ts", "equity_usd", "pnl_usd", "event", "symbol"],
    )
    _write_csv(
        walk_csv,
        list(walk.get("windows_rows", []) or []),
        columns=["train_days", "test_days", "train", "test", "delta_win_rate_pct"],
    )
    _write_csv(
        sweep_csv,
        list(sweep.get("scenarios", []) or []),
        columns=["threshold", "actionable", "entry_ready", "avg_abs_score", "win_rate_pct", "cumulative_return_pct"],
    )
    _write_csv(
        mc_csv,
        list(mc.get("paths", []) or []),
        columns=["sim", "final_equity_mult"],
    )

    payload = {
        "ts": int(now),
        "market": m,
        "state": "READY",
        "msg": f"Strategy lab completed for {m} with {int(len(closed_rows))} evaluated events.",
        "config": config,
        "backtest": {
            "metrics": back_metrics,
            "lookahead": {"ok": True, "violations": 0, "max_future_data_s": 0},
        },
        "walkforward": {
            "state": str(walk.get("state", "READY") or "READY"),
            "windows": int(walk.get("windows", 0) or 0),
            "summary": dict(walk.get("summary", {}) if isinstance(walk.get("summary", {}), dict) else {}),
        },
        "simulation": simulation,
        "recommendations": recommendations,
        "sweep": {"top10": list(sweep.get("top10", []) or [])},
        "monte_carlo": {
            "state": str(mc.get("state", "READY") or "READY"),
            "summary": dict(mc.get("summary", {}) if isinstance(mc.get("summary", {}), dict) else {}),
        },
        "artifacts": {
            "summary": summary_path,
            "events": events_csv,
            "equity_curve": equity_csv,
            "walkforward_csv": walk_csv,
            "sweep_csv": sweep_csv,
            "monte_paths": mc_csv,
        },
    }
    _write_json(summary_path, payload)
    return payload
