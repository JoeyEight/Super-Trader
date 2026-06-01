from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.credential_utils import get_openai_api_key
from app.settings_utils import sanitize_settings

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{16,}$")

OPENAI_CRYPTO_REPLAY_SYSTEM_PROMPT = """
You are an offline crypto position-exit forecaster for a trading system.

Task:
Given only historical context up to an entry timestamp, predict the expected exit timing/price/trigger for that single position.

Rules:
1. Use only supplied input context.
2. Do not assume access to future prices/events.
3. Be conservative when evidence is weak.
4. Return strict JSON matching schema.
""".strip()

OPENAI_CRYPTO_TRIGGER_REPLAY_SYSTEM_PROMPT = """
You are an offline crypto exit-trigger classifier for a trading system.

Task:
Given only historical context up to an entry timestamp, predict the most likely eventual exit trigger class for that single position.

Rules:
1. Use only supplied input context.
2. Do not assume access to future prices/events.
3. Be conservative when evidence is weak.
4. Return strict JSON matching schema.
""".strip()

OPENAI_CRYPTO_REPLAY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "predicted_direction",
        "expected_exit_ts",
        "expected_exit_price",
        "expected_hold_hours",
        "confidence",
        "exit_trigger",
        "rationale",
    ],
    "properties": {
        "summary": {"type": "string"},
        "predicted_direction": {"type": "string", "enum": ["up", "down", "flat"]},
        "expected_exit_ts": {"type": "integer"},
        "expected_exit_price": {"type": "number"},
        "expected_hold_hours": {"type": "number"},
        "confidence": {"type": "number"},
        "exit_trigger": {
            "type": "string",
            "enum": [
                "Trailing",
                "Stale Alignment",
                "AI Exit",
                "Blocked",
                "Risk Cut",
                "Take Profit",
                "Unknown",
            ],
        },
        "rationale": {"type": "string"},
    },
}

OPENAI_CRYPTO_TRIGGER_REPLAY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "exit_trigger",
        "confidence",
        "rationale",
    ],
    "properties": {
        "summary": {"type": "string"},
        "exit_trigger": {
            "type": "string",
            "enum": [
                "Trailing",
                "Stale Alignment",
                "AI Exit",
                "Blocked",
                "Risk Cut",
                "Take Profit",
                "Unknown",
            ],
        },
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
}


@dataclass
class ClosedTrade:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    entry_tag: str
    exit_tag: str
    entry_score: float
    entry_required_score: float
    entry_calib_prob: float

    @property
    def pnl_usd(self) -> float:
        return (float(self.exit_price) - float(self.entry_price)) * float(self.qty)

    @property
    def hold_hours(self) -> float:
        return max(0.0, (float(self.exit_ts) - float(self.entry_ts)) / 3600.0)

    @property
    def direction(self) -> str:
        delta = float(self.exit_price) - float(self.entry_price)
        if delta > 1e-12:
            return "up"
        if delta < -1e-12:
            return "down"
        return "flat"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _s(value: Any) -> str:
    return str(value or "").strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _safe_read_jsonl(path: str, limit: int = 200000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lim = max(1, int(limit or 1))
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= lim:
                    break
                txt = _s(line)
                if not txt:
                    continue
                try:
                    row = json.loads(txt)
                    if isinstance(row, dict):
                        out.append(row)
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _is_real_order_id(order_id: Any) -> bool:
    return bool(_UUID_RE.match(_s(order_id)))


def _extract_json_text(response_json: Dict[str, Any]) -> str:
    if not isinstance(response_json, dict):
        return ""
    out_text = response_json.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()
    chunks: List[str] = []
    output = response_json.get("output", [])
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "output_text", "value"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return "\n".join(chunks).strip()


def _trim_json_object(text: str) -> str:
    src = _s(text)
    if not src:
        return ""
    if src.startswith("{") and src.endswith("}"):
        return src
    start = src.find("{")
    end = src.rfind("}")
    if start >= 0 and end > start:
        return src[start : end + 1]
    return src


def _normalize_prediction(raw: Dict[str, Any], *, entry_ts: int, entry_price: float) -> Dict[str, Any]:
    direction = _s(raw.get("predicted_direction", "flat")).lower()
    if direction not in {"up", "down", "flat"}:
        direction = "flat"
    trigger = _s(raw.get("exit_trigger", "Unknown"))
    allowed = {
        "Trailing",
        "Stale Alignment",
        "AI Exit",
        "Blocked",
        "Risk Cut",
        "Take Profit",
        "Unknown",
    }
    if trigger not in allowed:
        trigger = "Unknown"

    exit_ts = int(max(entry_ts, _f(raw.get("expected_exit_ts", entry_ts), entry_ts)))
    hold_h = float(max(0.0, _f(raw.get("expected_hold_hours", 0.0), 0.0)))
    if hold_h <= 0.0:
        hold_h = max(0.0, (exit_ts - int(entry_ts)) / 3600.0)
    if hold_h > 0.0 and exit_ts <= int(entry_ts):
        exit_ts = int(entry_ts + int(round(hold_h * 3600.0)))

    exit_price = float(max(1e-12, _f(raw.get("expected_exit_price", entry_price), entry_price)))
    return {
        "summary": _s(raw.get("summary", ""))[:220],
        "predicted_direction": direction,
        "expected_exit_ts": int(exit_ts),
        "expected_exit_price": float(exit_price),
        "expected_hold_hours": round(float(hold_h), 6),
        "confidence": round(_clamp(_f(raw.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "exit_trigger": trigger,
        "rationale": _s(raw.get("rationale", ""))[:320],
    }


def _normalize_trigger_prediction(raw: Dict[str, Any], *, entry_ts: int, entry_price: float) -> Dict[str, Any]:
    trigger = _s(raw.get("exit_trigger", "Unknown"))
    allowed = {
        "Trailing",
        "Stale Alignment",
        "AI Exit",
        "Blocked",
        "Risk Cut",
        "Take Profit",
        "Unknown",
    }
    if trigger not in allowed:
        trigger = "Unknown"
    return {
        "summary": _s(raw.get("summary", ""))[:220],
        "predicted_direction": "flat",
        "expected_exit_ts": int(entry_ts),
        "expected_exit_price": float(max(1e-12, _f(entry_price, entry_price))),
        "expected_hold_hours": 0.0,
        "confidence": round(_clamp(_f(raw.get("confidence", 0.0), 0.0), 0.0, 1.0), 6),
        "exit_trigger": trigger,
        "rationale": _s(raw.get("rationale", ""))[:320],
    }


def _market_stats_from_prior_events(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = 0
    exits = 0
    wins = 0
    losses = 0
    realized = 0.0
    exit_tag_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = _s(row.get("event", "")).lower()
        if event == "entry":
            entries += 1
        elif event == "exit":
            exits += 1
            pnl = _f(row.get("realized_pnl_usd", 0.0), 0.0)
            realized += pnl
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
            tag = _s(row.get("tag", "none")).lower() or "none"
            exit_tag_counts[tag] = int(exit_tag_counts.get(tag, 0) + 1)
    return {
        "entries": int(entries),
        "exits": int(exits),
        "wins": int(wins),
        "losses": int(losses),
        "realized_pnl_usd": round(float(realized), 6),
        "exit_tag_counts": exit_tag_counts,
    }


def _symbol_stats_from_prior_events(symbol: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sym = _s(symbol).upper()
    sym_rows = [r for r in rows if _s(r.get("symbol", "")).upper() == sym]
    base = _market_stats_from_prior_events(sym_rows)
    last_prices: List[float] = []
    for r in sym_rows[-12:]:
        px = _f(r.get("price", 0.0), 0.0)
        if px > 0.0:
            last_prices.append(px)
    momentum_pct = 0.0
    if len(last_prices) >= 2 and last_prices[0] > 0.0:
        momentum_pct = ((last_prices[-1] / last_prices[0]) - 1.0) * 100.0
    base["recent_price_momentum_pct"] = round(float(momentum_pct), 6)
    base["recent_events"] = [
        {
            "ts": int(_f(r.get("ts", 0.0), 0.0)),
            "event": _s(r.get("event", "")).lower(),
            "tag": _s(r.get("tag", "none"))[:40],
            "price": round(float(_f(r.get("price", 0.0), 0.0)), 8),
            "qty": round(float(_f(r.get("qty", 0.0), 0.0)), 10),
        }
        for r in sym_rows[-8:]
    ]
    return base


def _scanner_symbol_context(scanner_rows: List[Dict[str, Any]], symbol: str, cutoff_ts: int) -> Dict[str, Any]:
    sym = _s(symbol).upper().replace("-USD", "")
    best: Dict[str, Any] | None = None
    for row in scanner_rows:
        ts = int(_f(row.get("ts", 0), 0.0))
        if ts <= 0 or ts > int(cutoff_ts):
            continue
        if (best is None) or (ts > int(_f(best.get("ts", 0), 0.0))):
            best = row
    if not isinstance(best, dict):
        return {}

    top_rows = best.get("top", []) if isinstance(best.get("top", []), list) else []
    mine: Dict[str, Any] = {}
    for idx, item in enumerate(top_rows):
        if not isinstance(item, dict):
            continue
        if _s(item.get("symbol", "")).upper() == sym:
            mine = {
                "rank": int(idx + 1),
                "score": round(float(_f(item.get("score", 0.0), 0.0)), 6),
                "required_score": round(float(_f(item.get("required_score", 0.0), 0.0)), 6),
                "calib_prob": round(float(_f(item.get("calib_prob", 0.0), 0.0)), 6),
                "samples": int(max(0.0, _f(item.get("samples", 0), 0.0))),
                "eligible_for_entry": bool(item.get("eligible_for_entry", False)),
            }
            break

    leaders = []
    for item in top_rows[:6]:
        if not isinstance(item, dict):
            continue
        leaders.append(
            {
                "symbol": _s(item.get("symbol", "")).upper(),
                "score": round(float(_f(item.get("score", 0.0), 0.0)), 6),
                "calib_prob": round(float(_f(item.get("calib_prob", 0.0), 0.0)), 6),
            }
        )
    return {
        "scan_ts": int(_f(best.get("ts", 0), 0.0)),
        "state": _s(best.get("state", ""))[:24],
        "adaptive_threshold": round(float(_f(best.get("adaptive_threshold", 0.0), 0.0)), 6),
        "symbol_snapshot": mine,
        "top_leaders": leaders,
    }


def _load_crypto_events(hub_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(hub_dir, "crypto", "execution_audit.jsonl")
    rows = _safe_read_jsonl(path, limit=400000)
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = int(_f(row.get("ts", 0), 0.0))
        event = _s(row.get("event", "")).lower()
        if ts <= 0 or event not in {"entry", "exit"}:
            continue
        if not _is_real_order_id(row.get("order_id")):
            continue
        symbol = _s(row.get("symbol", "")).upper()
        if not symbol:
            continue
        qty = _f(row.get("qty", 0.0), 0.0)
        price = _f(row.get("price", 0.0), 0.0)
        if qty <= 0.0 or price <= 0.0:
            continue
        cleaned.append(
            {
                "ts": int(ts),
                "event": event,
                "symbol": symbol,
                "qty": float(qty),
                "price": float(price),
                "tag": _s(row.get("tag", "")),
                "score": _f(row.get("score", 0.0), 0.0),
                "required_score": _f(row.get("required_score", 0.0), 0.0),
                "calib_prob": _f(row.get("calib_prob", 0.0), 0.0),
                "realized_pnl_usd": _f(row.get("realized_pnl_usd", 0.0), 0.0),
                "order_id": _s(row.get("order_id", "")),
            }
        )
    cleaned.sort(key=lambda item: int(item["ts"]))
    return cleaned


def _load_crypto_events_from_trade_history(hub_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(hub_dir, "trade_history.jsonl")
    rows = _safe_read_jsonl(path, limit=800000)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = int(_f(row.get("ts", 0), 0.0))
        if ts <= 0:
            continue
        symbol = _s(row.get("symbol", "")).upper()
        if not symbol.endswith("-USD"):
            continue
        side = _s(row.get("side", "")).lower()
        if side == "buy":
            event = "entry"
        elif side == "sell":
            event = "exit"
        else:
            continue
        order_id = row.get("order_id")
        if not _is_real_order_id(order_id):
            continue
        qty = _f(row.get("qty", 0.0), 0.0)
        price = _f(row.get("price", 0.0), 0.0)
        if qty <= 0.0 or price <= 0.0:
            continue
        out.append(
            {
                "ts": int(ts),
                "event": event,
                "symbol": symbol,
                "qty": float(qty),
                "price": float(price),
                "tag": _s(row.get("tag", "")),
                "score": _f(row.get("score", 0.0), 0.0),
                "required_score": _f(row.get("required_score", 0.0), 0.0),
                "calib_prob": _f(row.get("calib_prob", 0.0), 0.0),
                "realized_pnl_usd": _f(row.get("realized_pnl_usd", row.get("realized_profit_usd", 0.0)), 0.0),
                "order_id": _s(order_id),
            }
        )
    out.sort(key=lambda item: int(item["ts"]))
    return out


def _build_closed_trades(rows: List[Dict[str, Any]]) -> List[ClosedTrade]:
    open_lots: Dict[str, List[Dict[str, Any]]] = {}
    out: List[ClosedTrade] = []
    for row in rows:
        sym = _s(row.get("symbol", "")).upper()
        if not sym:
            continue
        open_lots.setdefault(sym, [])
        ev = _s(row.get("event", "")).lower()
        if ev == "entry":
            open_lots[sym].append(
                {
                    "ts": int(_f(row.get("ts", 0), 0.0)),
                    "qty": float(_f(row.get("qty", 0.0), 0.0)),
                    "price": float(_f(row.get("price", 0.0), 0.0)),
                    "tag": _s(row.get("tag", "")),
                    "score": float(_f(row.get("score", 0.0), 0.0)),
                    "required_score": float(_f(row.get("required_score", 0.0), 0.0)),
                    "calib_prob": float(_f(row.get("calib_prob", 0.0), 0.0)),
                }
            )
            continue
        if ev != "exit":
            continue
        rem = float(_f(row.get("qty", 0.0), 0.0))
        exit_ts = int(_f(row.get("ts", 0), 0.0))
        exit_price = float(_f(row.get("price", 0.0), 0.0))
        exit_tag = _s(row.get("tag", ""))
        while rem > 1e-12 and open_lots[sym]:
            lot = open_lots[sym][0]
            take = min(rem, float(lot.get("qty", 0.0)))
            if take <= 1e-12:
                open_lots[sym].pop(0)
                continue
            out.append(
                ClosedTrade(
                    symbol=sym,
                    entry_ts=int(_f(lot.get("ts", 0), 0.0)),
                    exit_ts=exit_ts,
                    entry_price=float(_f(lot.get("price", 0.0), 0.0)),
                    exit_price=exit_price,
                    qty=float(take),
                    entry_tag=_s(lot.get("tag", "")),
                    exit_tag=exit_tag,
                    entry_score=float(_f(lot.get("score", 0.0), 0.0)),
                    entry_required_score=float(_f(lot.get("required_score", 0.0), 0.0)),
                    entry_calib_prob=float(_f(lot.get("calib_prob", 0.0), 0.0)),
                )
            )
            lot["qty"] = float(lot.get("qty", 0.0)) - float(take)
            rem -= float(take)
            if float(lot.get("qty", 0.0)) <= 1e-12:
                open_lots[sym].pop(0)

    out.sort(key=lambda t: (int(t.entry_ts), str(t.symbol)))
    return out


def _build_replay_input(
    *,
    trade: ClosedTrade,
    all_events: List[Dict[str, Any]],
    scanner_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prior = [row for row in all_events if int(_f(row.get("ts", 0), 0.0)) < int(trade.entry_ts)]
    prior_symbol = [row for row in prior if _s(row.get("symbol", "")).upper() == str(trade.symbol)]

    return {
        "task": "predict_exit_for_new_entry",
        "entry_context": {
            "symbol": str(trade.symbol),
            "entry_ts": int(trade.entry_ts),
            "entry_date_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(trade.entry_ts))),
            "entry_price": round(float(trade.entry_price), 10),
            "entry_qty": round(float(trade.qty), 10),
            "entry_tag": str(trade.entry_tag or "none"),
            "entry_score": round(float(trade.entry_score), 6),
            "entry_required_score": round(float(trade.entry_required_score), 6),
            "entry_calib_prob": round(float(trade.entry_calib_prob), 6),
        },
        "prior_market_context": _market_stats_from_prior_events(prior),
        "prior_symbol_context": _symbol_stats_from_prior_events(trade.symbol, prior),
        "scanner_context": _scanner_symbol_context(scanner_rows, trade.symbol, trade.entry_ts),
        "recent_market_events": [
            {
                "ts": int(_f(row.get("ts", 0), 0.0)),
                "symbol": _s(row.get("symbol", "")).upper(),
                "event": _s(row.get("event", "")).lower(),
                "tag": _s(row.get("tag", "none"))[:40],
                "price": round(float(_f(row.get("price", 0.0), 0.0)), 8),
            }
            for row in prior[-20:]
            if isinstance(row, dict)
        ],
        "recent_symbol_events": [
            {
                "ts": int(_f(row.get("ts", 0), 0.0)),
                "event": _s(row.get("event", "")).lower(),
                "tag": _s(row.get("tag", "none"))[:40],
                "price": round(float(_f(row.get("price", 0.0), 0.0)), 8),
                "qty": round(float(_f(row.get("qty", 0.0), 0.0)), 10),
            }
            for row in prior_symbol[-12:]
        ],
    }


def _replay_cache_key(payload: Dict[str, Any], model: str, memo: str, task_kind: str = "full") -> str:
    raw = json.dumps(
        {"model": model, "memo": memo, "task_kind": _s(task_kind), "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _request_openai_prediction(
    *,
    settings: Dict[str, Any],
    base_dir: str,
    payload: Dict[str, Any],
    model: str,
    timeout_s: float,
    tuning_memo: str,
    prediction_mode: str = "full",
) -> Dict[str, Any]:
    api_key = get_openai_api_key(settings, base_dir=base_dir)
    if not api_key:
        return {
            "ok": False,
            "status": "missing_api_key",
            "error": "OpenAI API key missing",
            "prediction": {},
            "latency_ms": 0,
        }

    mode = _s(prediction_mode).lower() or "full"
    endpoint = _s(settings.get("openai_responses_endpoint")) or "https://api.openai.com/v1/responses"
    memo_block = _s(tuning_memo)
    if mode == "trigger":
        task_name = "crypto_historical_exit_trigger_prediction"
        system_prompt = OPENAI_CRYPTO_TRIGGER_REPLAY_SYSTEM_PROMPT
        response_schema = OPENAI_CRYPTO_TRIGGER_REPLAY_SCHEMA
    else:
        task_name = "crypto_historical_exit_prediction"
        system_prompt = OPENAI_CRYPTO_REPLAY_SYSTEM_PROMPT
        response_schema = OPENAI_CRYPTO_REPLAY_SCHEMA
    user_payload = {
        "task": task_name,
        "tuning_memo": memo_block,
        "input": payload,
    }
    req = {
        "model": str(model),
        "temperature": 0,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(user_payload, separators=(",", ":"), ensure_ascii=True)}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": task_name,
                "strict": True,
                "schema": response_schema,
            }
        },
    }

    started = time.time()
    body = json.dumps(req, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    max_attempts = 3
    last_error = ""
    raw_bytes = b""
    for attempt in range(max_attempts):
        request = Request(
            endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(timeout_s)) as resp:
                raw_bytes = resp.read()
            last_error = ""
            break
        except HTTPError as exc:
            return {
                "ok": False,
                "status": "http_error",
                "error": f"HTTP {int(exc.code)}",
                "prediction": {},
                "latency_ms": int(round((time.time() - started) * 1000.0)),
            }
        except TimeoutError:
            last_error = "OpenAI request timed out"
        except URLError as exc:
            last_error = _s(exc)[:220]
        except Exception as exc:
            last_error = _s(exc)[:220]

        is_last = attempt >= (max_attempts - 1)
        if is_last:
            return {
                "ok": False,
                "status": "request_error",
                "error": (last_error or "OpenAI request failed")[:220],
                "prediction": {},
                "latency_ms": int(round((time.time() - started) * 1000.0)),
            }
        time.sleep(0.35 * (attempt + 1))

    latency_ms = int(round((time.time() - started) * 1000.0))
    try:
        response_json = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return {
            "ok": False,
            "status": "invalid_json",
            "error": "Response was not valid JSON",
            "prediction": {},
            "latency_ms": latency_ms,
        }

    text = _trim_json_object(_extract_json_text(response_json))
    if not text:
        return {
            "ok": False,
            "status": "empty_response",
            "error": "No structured output in OpenAI response",
            "prediction": {},
            "latency_ms": latency_ms,
        }

    try:
        raw = json.loads(text)
    except Exception:
        return {
            "ok": False,
            "status": "malformed_json",
            "error": "Could not decode JSON payload",
            "prediction": {},
            "latency_ms": latency_ms,
        }

    return {
        "ok": True,
        "status": "ok",
        "error": "",
        "prediction": raw if isinstance(raw, dict) else {},
        "latency_ms": latency_ms,
        "response_id": _s(response_json.get("id", "")),
    }


def _evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n <= 0:
        return {
            "trades": 0,
            "directional_accuracy_pct": 0.0,
            "mean_abs_timing_error_hours": 0.0,
            "median_abs_timing_error_hours": 0.0,
            "mean_abs_exit_price_error_pct": 0.0,
            "trigger_match_pct": 0.0,
            "pnl_trend_match_pct": 0.0,
        }

    direction_hits = 0
    trigger_hits = 0
    pnl_trend_hits = 0
    timing_errors: List[float] = []
    price_errors: List[float] = []
    for row in rows:
        pred_dir = _s(row.get("predicted_direction", "flat")).lower()
        actual_dir = _s(row.get("actual_direction", "flat")).lower()
        if pred_dir == actual_dir:
            direction_hits += 1

        actual_trigger = _s(row.get("actual_exit_trigger", "Unknown"))
        if _s(row.get("predicted_exit_trigger", "Unknown")) == actual_trigger:
            trigger_hits += 1

        actual_exit_ts = int(_f(row.get("actual_exit_ts", 0), 0.0))
        pred_exit_ts = int(_f(row.get("predicted_exit_ts", 0), 0.0))
        timing_errors.append(abs(float(pred_exit_ts - actual_exit_ts)) / 3600.0)

        entry = float(_f(row.get("entry_price", 0.0), 0.0))
        act_px = float(_f(row.get("actual_exit_price", 0.0), 0.0))
        pred_px = float(_f(row.get("predicted_exit_price", 0.0), 0.0))
        if entry > 0.0:
            price_errors.append(abs(pred_px - act_px) / entry * 100.0)
            actual_ret = (act_px - entry) / entry
            pred_ret = (pred_px - entry) / entry
            # Trend match: both non-negative, both non-positive, or both near-flat.
            eps = 1e-9
            if abs(actual_ret) <= eps and abs(pred_ret) <= eps:
                pnl_trend_hits += 1
            elif (actual_ret > eps and pred_ret > eps) or (actual_ret < -eps and pred_ret < -eps):
                pnl_trend_hits += 1

    return {
        "trades": int(n),
        "directional_accuracy_pct": round(float(direction_hits / n * 100.0), 4),
        "mean_abs_timing_error_hours": round(float(sum(timing_errors) / max(1, len(timing_errors))), 4),
        "median_abs_timing_error_hours": round(float(statistics.median(timing_errors) if timing_errors else 0.0), 4),
        "mean_abs_exit_price_error_pct": round(float(sum(price_errors) / max(1, len(price_errors))), 4),
        "trigger_match_pct": round(float(trigger_hits / n * 100.0), 4),
        "pnl_trend_match_pct": round(float(pnl_trend_hits / n * 100.0), 4),
    }


def _prediction_key(row: Dict[str, Any]) -> Tuple[str, int]:
    if not isinstance(row, dict):
        return "", 0
    return _s(row.get("symbol", "")).upper(), int(_f(row.get("entry_ts", 0), 0.0))


def _build_hybrid_predictions(
    direction_rows: List[Dict[str, Any]],
    trigger_rows: List[Dict[str, Any]],
    *,
    trigger_override_conf_min: float = 0.0,
) -> List[Dict[str, Any]]:
    if not direction_rows:
        return []
    trigger_map = {_prediction_key(r): r for r in list(trigger_rows or []) if isinstance(r, dict)}
    out: List[Dict[str, Any]] = []
    cutoff = _clamp(float(trigger_override_conf_min), 0.0, 1.0)
    for row in direction_rows:
        if not isinstance(row, dict):
            continue
        merged = dict(row)
        trow = trigger_map.get(_prediction_key(row), {})
        if isinstance(trow, dict) and trow:
            trig_conf = _clamp(_f(trow.get("predicted_confidence", 0.0), 0.0), 0.0, 1.0)
            if trig_conf >= cutoff:
                merged["predicted_exit_trigger"] = _s(
                    trow.get("predicted_exit_trigger", merged.get("predicted_exit_trigger", "Unknown"))
                )
        out.append(merged)
    return out


def _hybrid_objective_score(metrics: Dict[str, Any]) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    return (
        _f(metrics.get("directional_accuracy_pct", 0.0), 0.0)
        + _f(metrics.get("trigger_match_pct", 0.0), 0.0)
    )


def _tune_trigger_override_cutoff(
    direction_rows: List[Dict[str, Any]],
    trigger_rows: List[Dict[str, Any]],
    *,
    cutoff_floor: float = 0.0,
) -> Dict[str, Any]:
    # Evaluate a compact confidence grid and keep the cutoff that best improves
    # combined direction+trigger quality on the TRAIN set.
    floor = _clamp(float(cutoff_floor), 0.0, 1.0)
    candidates = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, floor]
    candidates = sorted(set(float(c) for c in candidates if float(c) >= floor))
    best_cutoff = 0.0
    best_metrics: Dict[str, Any] = {}
    best_score = -1e9
    for cutoff in candidates:
        hybrid = _build_hybrid_predictions(
            direction_rows,
            trigger_rows,
            trigger_override_conf_min=float(cutoff),
        )
        metrics = _evaluate_rows(hybrid)
        score = _hybrid_objective_score(metrics)
        # Tie-breaks: prefer higher trigger match first, then higher direction.
        trig = _f(metrics.get("trigger_match_pct", 0.0), 0.0)
        dire = _f(metrics.get("directional_accuracy_pct", 0.0), 0.0)
        best_trig = _f(best_metrics.get("trigger_match_pct", 0.0), 0.0) if best_metrics else -1.0
        best_dire = _f(best_metrics.get("directional_accuracy_pct", 0.0), 0.0) if best_metrics else -1.0
        if (score > best_score) or (
            abs(score - best_score) <= 1e-9 and (trig > best_trig or (abs(trig - best_trig) <= 1e-9 and dire > best_dire))
        ):
            best_score = score
            best_cutoff = float(cutoff)
            best_metrics = metrics
    return {
        "trigger_override_conf_min": round(float(best_cutoff), 4),
        "trigger_override_conf_min_floor": round(float(floor), 4),
        "train_metrics": best_metrics if isinstance(best_metrics, dict) else {},
    }


def _pct(numer: int, denom: int) -> float:
    return (float(numer) / float(max(1, denom))) * 100.0


def _quantile(values: List[float], q: float, default: float = 0.0) -> float:
    vals = [float(v) for v in list(values or []) if float(v) >= 0.0]
    if not vals:
        return float(default)
    vals.sort()
    qq = _clamp(float(q), 0.0, 1.0)
    idx = int(round((len(vals) - 1) * qq))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _build_direction_tuning_memo(train_rows: List[Dict[str, Any]]) -> str:
    if not train_rows:
        return "Use baseline conservative behavior with no extra calibration memo."
    metrics = _evaluate_rows(train_rows)
    dir_acc = _f(metrics.get("directional_accuracy_pct", 0.0), 0.0)
    down_total = 0
    down_missed_as_up = 0
    up_total = 0
    up_missed_as_down = 0
    stale_down_total = 0
    stale_down_pred_up = 0
    for row in train_rows:
        actual_dir = _s(row.get("actual_direction", "flat")).lower()
        pred_dir = _s(row.get("predicted_direction", "flat")).lower()
        trig = _s(row.get("actual_exit_trigger", "Unknown"))
        if actual_dir == "down":
            down_total += 1
            if pred_dir == "up":
                down_missed_as_up += 1
            if trig == "Stale Alignment":
                stale_down_total += 1
                if pred_dir == "up":
                    stale_down_pred_up += 1
        elif actual_dir == "up":
            up_total += 1
            if pred_dir == "down":
                up_missed_as_down += 1
    bias_lines = [
        "Direction objective only: prioritize correct direction over trigger style.",
        f"Training directional accuracy: {dir_acc:.2f}%.",
        f"Actual-down trades predicted up: {down_missed_as_up}/{max(1, down_total)} ({_pct(down_missed_as_up, down_total):.1f}%).",
        f"Actual-up trades predicted down: {up_missed_as_down}/{max(1, up_total)} ({_pct(up_missed_as_down, up_total):.1f}%).",
        (
            f"Stale-alignment down trades predicted up: {stale_down_pred_up}/{max(1, stale_down_total)}."
            if stale_down_total > 0
            else "No stale-alignment down trades observed in train set."
        ),
        "When stale-alignment evidence is present and confidence is not strong, avoid optimistic UP calls.",
        "If evidence is mixed, prefer FLAT over forced UP.",
    ]
    return " ".join(bias_lines)


def _build_trigger_tuning_memo(train_rows: List[Dict[str, Any]]) -> str:
    if not train_rows:
        return "Use baseline conservative behavior with no extra calibration memo."
    metrics = _evaluate_rows(train_rows)
    trig_acc = _f(metrics.get("trigger_match_pct", 0.0), 0.0)
    stale_actual = 0
    stale_pred_nonstale = 0
    trailing_actual = 0
    trailing_pred_nontrailing = 0
    for row in train_rows:
        actual = _s(row.get("actual_exit_trigger", "Unknown"))
        pred = _s(row.get("predicted_exit_trigger", "Unknown"))
        if actual == "Stale Alignment":
            stale_actual += 1
            if pred != "Stale Alignment":
                stale_pred_nonstale += 1
        elif actual == "Trailing":
            trailing_actual += 1
            if pred != "Trailing":
                trailing_pred_nontrailing += 1
    return " ".join(
        [
            "Trigger objective only: prioritize matching exit trigger family.",
            f"Training trigger match: {trig_acc:.2f}%.",
            (
                f"Actual stale-alignment misclassified: {stale_pred_nonstale}/{max(1, stale_actual)}."
                if stale_actual > 0
                else "No stale-alignment exits observed in train set."
            ),
            (
                f"Actual trailing misclassified: {trailing_pred_nontrailing}/{max(1, trailing_actual)}."
                if trailing_actual > 0
                else "No trailing exits observed in train set."
            ),
            "Prefer Stale Alignment when context shows stale misalignment and weak continuation support.",
            "Use Trailing only when continuation is still strong and pullback behavior is consistent with trailing exits.",
        ]
    )


def _derive_train_calibration(train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not train_rows:
        return {
            "hold_factor": 1.0,
            "stale_hold_cap_h": 24.0,
            "global_hold_cap_h": 72.0,
            "force_down_on_weak_stale": False,
        }
    actual_holds = [max(0.0, _f(r.get("actual_hold_hours", 0.0), 0.0)) for r in train_rows]
    pred_holds = [max(0.25, _f(r.get("predicted_hold_hours", 0.25), 0.25)) for r in train_rows]
    # Robustify hold scaling: cap tails before computing calibration so a few very long
    # historical holds do not dominate timing calibration for the whole sample.
    q90_actual = _quantile(actual_holds, 0.90, default=24.0)
    q90_pred = _quantile(pred_holds, 0.90, default=24.0)
    actual_wins = [min(v, q90_actual) for v in actual_holds]
    pred_wins = [min(v, q90_pred) for v in pred_holds]
    median_actual = statistics.median(actual_wins) if actual_wins else 1.0
    median_pred = statistics.median(pred_wins) if pred_wins else 1.0
    hold_factor = _clamp(median_actual / max(0.25, median_pred), 0.25, 4.0)
    outlier_ratio = (
        sum(1 for v in actual_holds if v > q90_actual) / float(max(1, len(actual_holds)))
        if actual_holds
        else 0.0
    )
    if outlier_ratio > 0.10:
        # Pull factor toward neutral when hold distribution has heavy long-tail outliers.
        soften = _clamp(1.0 - ((outlier_ratio - 0.10) * 1.5), 0.55, 1.0)
        hold_factor = 1.0 + ((hold_factor - 1.0) * soften)

    stale_actual = [
        max(0.0, _f(r.get("actual_hold_hours", 0.0), 0.0))
        for r in train_rows
        if _s(r.get("actual_exit_trigger", "")) == "Stale Alignment"
    ]
    stale_hold_cap_h = _clamp(
        (_quantile(stale_actual, 0.75, default=median_actual) if stale_actual else median_actual) + 1.0,
        0.5,
        36.0,
    )
    global_hold_cap_h = _clamp(_quantile(actual_holds, 0.90, default=72.0) + 2.0, 2.0, 168.0)

    down_actual = sum(1 for r in train_rows if _s(r.get("actual_direction", "")).lower() == "down")
    down_pred = sum(1 for r in train_rows if _s(r.get("predicted_direction", "")).lower() == "down")
    n = max(1, len(train_rows))
    force_down_on_weak_stale = (down_actual / n >= 0.65) and (down_pred / n <= 0.5)
    return {
        "hold_factor": round(float(hold_factor), 6),
        "stale_hold_cap_h": round(float(stale_hold_cap_h), 6),
        "global_hold_cap_h": round(float(global_hold_cap_h), 6),
        "force_down_on_weak_stale": bool(force_down_on_weak_stale),
        "hold_outlier_ratio": round(float(outlier_ratio), 6),
    }


def _apply_prediction_calibration(rows: List[Dict[str, Any]], calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    hold_factor = _clamp(_f(calibration.get("hold_factor", 1.0), 1.0), 0.25, 4.0)
    stale_cap_h = _clamp(_f(calibration.get("stale_hold_cap_h", 24.0), 24.0), 0.5, 36.0)
    global_cap_h = _clamp(_f(calibration.get("global_hold_cap_h", 72.0), 72.0), 2.0, 168.0)
    force_down = bool(calibration.get("force_down_on_weak_stale", False))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        hold_h = max(0.0, _f(item.get("predicted_hold_hours", 0.0), 0.0)) * hold_factor
        if _s(item.get("predicted_exit_trigger", "")) == "Stale Alignment":
            hold_h = min(hold_h, stale_cap_h)
        hold_h = min(hold_h, global_cap_h)
        hold_h = max(0.0, hold_h)
        item["predicted_hold_hours"] = round(float(hold_h), 6)
        entry_ts = int(_f(item.get("entry_ts", 0), 0.0))
        item["predicted_exit_ts"] = int(entry_ts + int(round(hold_h * 3600.0)))

        if force_down:
            pred_dir = _s(item.get("predicted_direction", "flat")).lower()
            pred_trig = _s(item.get("predicted_exit_trigger", "Unknown"))
            conf = _f(item.get("predicted_confidence", 0.0), 0.0)
            if pred_dir in {"flat", "up"} and pred_trig in {"Stale Alignment", "Risk Cut"} and conf <= 0.65:
                item["predicted_direction"] = "down"
        out.append(item)
    return out


def _normalize_exit_trigger(tag: str) -> str:
    txt = _s(tag).upper()
    if txt == "TRAIL_SELL":
        return "Trailing"
    if txt == "POLICY_STALE_EXIT":
        return "Stale Alignment"
    if txt in {"BLOCKED", "BLOCK"}:
        return "Blocked"
    if "AI" in txt:
        return "AI Exit"
    if "RISK" in txt:
        return "Risk Cut"
    if "TAKE" in txt:
        return "Take Profit"
    return "Unknown"


def _predict_rows(
    *,
    settings: Dict[str, Any],
    base_dir: str,
    model: str,
    timeout_s: float,
    trades: List[ClosedTrade],
    all_events: List[Dict[str, Any]],
    scanner_rows: List[Dict[str, Any]],
    tuning_memo: str,
    cache: Dict[str, Dict[str, Any]],
    prediction_mode: str = "full",
) -> List[Dict[str, Any]]:
    mode = _s(prediction_mode).lower() or "full"
    out: List[Dict[str, Any]] = []
    for trade in trades:
        payload = _build_replay_input(trade=trade, all_events=all_events, scanner_rows=scanner_rows)
        key = _replay_cache_key(payload, model, tuning_memo, mode)
        hit = cache.get(key)
        if not isinstance(hit, dict):
            hit = _request_openai_prediction(
                settings=settings,
                base_dir=base_dir,
                payload=payload,
                model=model,
                timeout_s=timeout_s,
                tuning_memo=tuning_memo,
                prediction_mode=mode,
            )
            cache[key] = hit

        pred_raw = hit.get("prediction", {}) if isinstance(hit.get("prediction", {}), dict) else {}
        if mode == "trigger":
            pred = _normalize_trigger_prediction(pred_raw, entry_ts=trade.entry_ts, entry_price=trade.entry_price)
        else:
            pred = _normalize_prediction(pred_raw, entry_ts=trade.entry_ts, entry_price=trade.entry_price)
        out.append(
            {
                "symbol": str(trade.symbol),
                "entry_ts": int(trade.entry_ts),
                "entry_price": round(float(trade.entry_price), 10),
                "actual_exit_ts": int(trade.exit_ts),
                "actual_exit_price": round(float(trade.exit_price), 10),
                "actual_hold_hours": round(float(trade.hold_hours), 6),
                "actual_pnl_usd": round(float(trade.pnl_usd), 8),
                "actual_direction": str(trade.direction),
                "actual_exit_trigger": _normalize_exit_trigger(trade.exit_tag),
                "predicted_exit_ts": int(pred.get("expected_exit_ts", trade.entry_ts)),
                "predicted_exit_price": round(float(_f(pred.get("expected_exit_price", trade.entry_price), trade.entry_price)), 10),
                "predicted_hold_hours": round(float(_f(pred.get("expected_hold_hours", 0.0), 0.0)), 6),
                "predicted_direction": _s(pred.get("predicted_direction", "flat")).lower(),
                "predicted_exit_trigger": _s(pred.get("exit_trigger", "Unknown")),
                "predicted_confidence": round(float(_f(pred.get("confidence", 0.0), 0.0)), 6),
                "model_summary": _s(pred.get("summary", ""))[:220],
                "model_rationale": _s(pred.get("rationale", ""))[:320],
                "openai_status": _s(hit.get("status", "")),
                "openai_error": _s(hit.get("error", ""))[:220],
                "openai_latency_ms": int(_f(hit.get("latency_ms", 0), 0.0)),
                "response_id": _s(hit.get("response_id", "")),
            }
        )
    return out


def run_crypto_historical_replay(
    *,
    settings: Dict[str, Any] | None,
    base_dir: str,
    hub_dir: str,
    max_trades: int = 24,
    test_ratio: float = 0.35,
    max_iterations: int = 3,
    model_override: str = "",
    timeout_s_override: float = 0.0,
    enable_train_calibration: bool = False,
    trigger_override_conf_min_floor: float = 0.0,
) -> Dict[str, Any]:
    cfg = sanitize_settings(settings if isinstance(settings, dict) else {})
    model = _s(model_override or cfg.get("openai_position_review_model", cfg.get("openai_model", "gpt-5.4-mini"))) or "gpt-5.4-mini"
    timeout_s = float(timeout_s_override if timeout_s_override > 0.0 else _f(cfg.get("openai_position_review_timeout_s", 12.0), 12.0))
    timeout_s = _clamp(timeout_s, 2.0, 60.0)

    events_audit = _load_crypto_events(hub_dir)
    events_trade_history = _load_crypto_events_from_trade_history(hub_dir)
    source = "execution_audit"
    events = events_audit
    if len(events_trade_history) > len(events_audit):
        events = events_trade_history
        source = "trade_history"
    scanner_rows = _safe_read_jsonl(os.path.join(hub_dir, "crypto", "scanner_rankings.jsonl"), limit=500000)
    closed_all = _build_closed_trades(events)
    if max_trades > 0 and len(closed_all) > max_trades:
        max_n = int(max_trades)
        if max_n <= 1:
            closed_all = [closed_all[-1]]
        else:
            idxs = {
                int(round(i * (len(closed_all) - 1) / float(max_n - 1)))
                for i in range(max_n)
            }
            closed_all = [closed_all[i] for i in sorted(idxs)]

    n = len(closed_all)
    if n <= 2:
        return {
            "status": "insufficient_data",
            "summary": "Not enough closed crypto trades for replay.",
            "meta": {
                "closed_trades": int(n),
                "events": int(len(events)),
                "scanner_rows": int(len(scanner_rows)),
                "model": model,
            },
            "iterations": [],
        }

    split = int(round(n * (1.0 - max(0.1, min(0.8, float(test_ratio))))))
    split = max(2, min(n - 1, split))
    train = closed_all[:split]
    test = closed_all[split:]

    cache: Dict[str, Dict[str, Any]] = {}
    best_iter: Dict[str, Any] | None = None
    best_direction_iter: Dict[str, Any] | None = None
    best_trigger_iter: Dict[str, Any] | None = None
    best_cutoff_iter: Dict[str, Any] | None = None
    iterations: List[Dict[str, Any]] = []
    direction_tuning_memo = "Use baseline conservative behavior with no extra calibration memo."
    trigger_tuning_memo = "Use baseline conservative behavior with no extra calibration memo."

    for idx in range(max(1, int(max_iterations))):
        train_direction_raw = _predict_rows(
            settings=cfg,
            base_dir=base_dir,
            model=model,
            timeout_s=timeout_s,
            trades=train,
            all_events=events,
            scanner_rows=scanner_rows,
            tuning_memo=direction_tuning_memo,
            cache=cache,
        )
        train_trigger_raw = _predict_rows(
            settings=cfg,
            base_dir=base_dir,
            model=model,
            timeout_s=timeout_s,
            trades=train,
            all_events=events,
            scanner_rows=scanner_rows,
            tuning_memo=trigger_tuning_memo,
            cache=cache,
            prediction_mode="trigger",
        )
        calibration = (
            _derive_train_calibration(train_direction_raw)
            if bool(enable_train_calibration)
            else {"hold_factor": 1.0, "stale_hold_cap_h": 24.0, "global_hold_cap_h": 72.0, "force_down_on_weak_stale": False}
        )
        train_direction_preds = _apply_prediction_calibration(train_direction_raw, calibration)
        train_trigger_preds = _apply_prediction_calibration(train_trigger_raw, calibration)
        cutoff_fit = _tune_trigger_override_cutoff(
            train_direction_preds,
            train_trigger_preds,
            cutoff_floor=float(trigger_override_conf_min_floor),
        )
        trigger_override_conf_min = _clamp(_f(cutoff_fit.get("trigger_override_conf_min", 0.0), 0.0), 0.0, 1.0)
        train_hybrid_preds = _build_hybrid_predictions(
            train_direction_preds,
            train_trigger_preds,
            trigger_override_conf_min=trigger_override_conf_min,
        )

        test_direction_raw = _predict_rows(
            settings=cfg,
            base_dir=base_dir,
            model=model,
            timeout_s=timeout_s,
            trades=test,
            all_events=events,
            scanner_rows=scanner_rows,
            tuning_memo=direction_tuning_memo,
            cache=cache,
        )
        test_trigger_raw = _predict_rows(
            settings=cfg,
            base_dir=base_dir,
            model=model,
            timeout_s=timeout_s,
            trades=test,
            all_events=events,
            scanner_rows=scanner_rows,
            tuning_memo=trigger_tuning_memo,
            cache=cache,
            prediction_mode="trigger",
        )
        test_direction_preds = _apply_prediction_calibration(test_direction_raw, calibration)
        test_trigger_preds = _apply_prediction_calibration(test_trigger_raw, calibration)
        test_hybrid_preds = _build_hybrid_predictions(
            test_direction_preds,
            test_trigger_preds,
            trigger_override_conf_min=trigger_override_conf_min,
        )

        train_direction_metrics = _evaluate_rows(train_direction_preds)
        train_trigger_metrics = _evaluate_rows(train_trigger_preds)
        train_hybrid_metrics = _evaluate_rows(train_hybrid_preds)
        test_direction_metrics = _evaluate_rows(test_direction_preds)
        test_trigger_metrics = _evaluate_rows(test_trigger_preds)
        test_hybrid_metrics = _evaluate_rows(test_hybrid_preds)
        iter_row = {
            "iteration": int(idx + 1),
            "tuning_memo": _s(direction_tuning_memo),
            "direction_tuning_memo": _s(direction_tuning_memo),
            "trigger_tuning_memo": _s(trigger_tuning_memo),
            "train_calibration": calibration,
            "trigger_override_tuning": cutoff_fit,
            "train_direction_metrics": train_direction_metrics,
            "train_trigger_metrics": train_trigger_metrics,
            "train_metrics": train_hybrid_metrics,
            "test_direction_metrics": test_direction_metrics,
            "test_trigger_metrics": test_trigger_metrics,
            "test_metrics": test_hybrid_metrics,
            "train_direction_predictions": train_direction_preds,
            "train_trigger_predictions": train_trigger_preds,
            "train_predictions": train_hybrid_preds,
            "test_direction_predictions": test_direction_preds,
            "test_trigger_predictions": test_trigger_preds,
            "test_predictions": test_hybrid_preds,
        }
        iterations.append(iter_row)

        if (best_iter is None) or (
            _f(test_hybrid_metrics.get("directional_accuracy_pct", 0.0), 0.0)
            + _f(test_hybrid_metrics.get("trigger_match_pct", 0.0), 0.0)
            > _f((best_iter.get("test_metrics", {}) if isinstance(best_iter.get("test_metrics", {}), dict) else {}).get("directional_accuracy_pct", 0.0), 0.0)
            + _f((best_iter.get("test_metrics", {}) if isinstance(best_iter.get("test_metrics", {}), dict) else {}).get("trigger_match_pct", 0.0), 0.0)
        ):
            best_iter = iter_row

        if (best_direction_iter is None) or (
            _f(test_direction_metrics.get("directional_accuracy_pct", 0.0), 0.0)
            > _f((best_direction_iter.get("test_metrics", {}) if isinstance(best_direction_iter.get("test_metrics", {}), dict) else {}).get("directional_accuracy_pct", 0.0), 0.0)
        ):
            best_direction_iter = {
                **iter_row,
                "test_metrics": test_direction_metrics,
                "test_predictions": test_direction_preds,
            }

        if (best_trigger_iter is None) or (
            _f(test_trigger_metrics.get("trigger_match_pct", 0.0), 0.0)
            > _f((best_trigger_iter.get("test_metrics", {}) if isinstance(best_trigger_iter.get("test_metrics", {}), dict) else {}).get("trigger_match_pct", 0.0), 0.0)
        ):
            best_trigger_iter = {
                **iter_row,
                "test_metrics": test_trigger_metrics,
                "test_predictions": test_trigger_preds,
            }

        if (best_cutoff_iter is None) or (
            _hybrid_objective_score(test_hybrid_metrics)
            > _hybrid_objective_score(
                best_cutoff_iter.get("test_metrics", {}) if isinstance(best_cutoff_iter.get("test_metrics", {}), dict) else {}
            )
        ):
            best_cutoff_iter = iter_row

        # Build next-iteration memo from train performance only (no test leakage)
        direction_tuning_memo = _build_direction_tuning_memo(train_direction_preds)
        trigger_tuning_memo = _build_trigger_tuning_memo(train_trigger_preds)

    best = best_iter if isinstance(best_iter, dict) else (iterations[0] if iterations else {})
    best_test = best.get("test_metrics", {}) if isinstance(best.get("test_metrics", {}), dict) else {}
    best_dir = best_direction_iter if isinstance(best_direction_iter, dict) else best
    best_trig = best_trigger_iter if isinstance(best_trigger_iter, dict) else best
    best_cutoff_row = best_cutoff_iter if isinstance(best_cutoff_iter, dict) else best
    best_cutoff = _clamp(
        _f(
            (
                (best_cutoff_row.get("trigger_override_tuning", {}) if isinstance(best_cutoff_row.get("trigger_override_tuning", {}), dict) else {}).get(
                    "trigger_override_conf_min",
                    0.0,
                )
            ),
            0.0,
        ),
        0.0,
        1.0,
    )
    hybrid_predictions = _build_hybrid_predictions(
        list(best_dir.get("test_predictions", []) if isinstance(best_dir.get("test_predictions", []), list) else []),
        list(best_trig.get("test_predictions", []) if isinstance(best_trig.get("test_predictions", []), list) else []),
        trigger_override_conf_min=best_cutoff,
    )
    hybrid_metrics = _evaluate_rows(hybrid_predictions) if hybrid_predictions else {}
    summary = (
        f"Best iteration #{int(best.get('iteration', 1))}: "
        f"test directional accuracy {_f(best_test.get('directional_accuracy_pct', 0.0), 0.0):.2f}%, "
        f"mean timing error {_f(best_test.get('mean_abs_timing_error_hours', 0.0), 0.0):.2f}h, "
        f"mean exit-price error {_f(best_test.get('mean_abs_exit_price_error_pct', 0.0), 0.0):.2f}%."
    )

    return {
        "status": "ok",
        "summary": summary,
        "meta": {
            "model": model,
            "timeout_s": float(timeout_s),
            "events": int(len(events)),
            "events_source": source,
            "events_audit_count": int(len(events_audit)),
            "events_trade_history_count": int(len(events_trade_history)),
            "scanner_rows": int(len(scanner_rows)),
            "closed_trades_total": int(len(closed_all)),
            "train_trades": int(len(train)),
            "test_trades": int(len(test)),
            "test_ratio": round(float(test_ratio), 4),
            "max_iterations": int(max_iterations),
            "enable_train_calibration": bool(enable_train_calibration),
            "trigger_override_conf_min_floor": round(float(_clamp(trigger_override_conf_min_floor, 0.0, 1.0)), 4),
            "generated_ts": int(time.time()),
            "data_window": {
                "start_ts": int(closed_all[0].entry_ts),
                "end_ts": int(closed_all[-1].exit_ts),
                "start_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(closed_all[0].entry_ts))),
                "end_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(closed_all[-1].exit_ts))),
            },
        },
        "best_iteration": int(best.get("iteration", 1)),
        "best_test_metrics": best_test,
        "best_direction_iteration": int(best_dir.get("iteration", 1)),
        "best_trigger_iteration": int(best_trig.get("iteration", 1)),
        "best_trigger_override_conf_min": round(float(best_cutoff), 4),
        "hybrid_test_metrics": hybrid_metrics,
        "iterations": iterations,
    }
