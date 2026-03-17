from __future__ import annotations

import email.utils
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List


YAHOO_NEWS_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"

_POSITIVE_KEYWORDS = (
    "beat",
    "beats",
    "upgrade",
    "upgrades",
    "bullish",
    "surge",
    "rally",
    "record",
    "strong",
    "growth",
    "approval",
    "adoption",
    "partnership",
    "buyback",
    "etf inflow",
    "raises guidance",
)
_NEGATIVE_KEYWORDS = (
    "downgrade",
    "downgrades",
    "miss",
    "misses",
    "lawsuit",
    "investigation",
    "hack",
    "exploit",
    "breach",
    "fraud",
    "bankruptcy",
    "default",
    "liquidation",
    "ban",
    "warning",
    "cuts guidance",
    "suspend",
)
_EVENT_RISK_KEYWORDS = (
    "earnings",
    "fomc",
    "fed",
    "cpi",
    "ppi",
    "jobs report",
    "nfp",
    "sec",
    "regulatory",
    "halving",
    "fork",
    "token unlock",
)


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _cache_path(hub_dir: str) -> str:
    return os.path.join(str(hub_dir or "."), "news_event_provider_cache.json")


def _norm_symbols(symbols: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for row in list(symbols or []):
        sym = str(row or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _provider_symbol(market: str, symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    m = str(market or "").strip().lower()
    if m != "crypto":
        return sym
    if sym.endswith("-USD"):
        return sym
    if "-" in sym:
        return sym
    return f"{sym}-USD"


def _parse_pub_date_ts(raw: Any) -> int:
    text = str(raw or "").strip()
    if not text:
        return 0
    try:
        dt = email.utils.parsedate_to_datetime(text)
        if dt is not None:
            return int(float(dt.timestamp()))
    except Exception:
        pass
    return 0


def _fetch_symbol_headlines(symbol_query: str, timeout_s: float = 8.0, max_items: int = 6) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "s": str(symbol_query or "").strip(),
            "region": "US",
            "lang": "en-US",
        }
    )
    url = f"{YAHOO_NEWS_RSS_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SuperTrader-AI/1.0",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=max(3.0, float(timeout_s))) as resp:
        raw = resp.read() if resp else b""
    if not raw:
        return []
    root = ET.fromstring(raw)
    rows: List[Dict[str, Any]] = []
    for item in root.findall(".//item")[: max(1, int(max_items))]:
        title = str(item.findtext("title", default="") or "").strip()
        if not title:
            continue
        pub_raw = str(item.findtext("pubDate", default="") or "").strip()
        ts = _parse_pub_date_ts(pub_raw)
        rows.append(
            {
                "title": title,
                "pub_date": pub_raw,
                "ts": int(ts),
            }
        )
    return rows


def _headline_scores(headlines: List[Dict[str, Any]], now_ts: int) -> Dict[str, Any]:
    pos = 0.0
    neg = 0.0
    evt = 0.0
    total = 0
    latest_title = ""
    latest_ts = 0
    for row in list(headlines or []):
        title = str(row.get("title", "") or "").strip()
        if not title:
            continue
        total += 1
        txt = title.lower()
        ts = int(row.get("ts", 0) or 0)
        if ts <= 0:
            ts = int(now_ts)
        age_s = max(0, int(now_ts - ts))
        if age_s <= 6 * 3600:
            recency_w = 1.0
        elif age_s <= 24 * 3600:
            recency_w = 0.7
        else:
            recency_w = 0.45
        if ts > latest_ts:
            latest_ts = ts
            latest_title = title
        pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in txt)
        neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in txt)
        evt_hits = sum(1 for kw in _EVENT_RISK_KEYWORDS if kw in txt)
        pos += float(pos_hits) * recency_w
        neg += float(neg_hits) * recency_w
        evt += float(evt_hits) * recency_w

    directional = pos + neg
    sentiment = (pos - neg) / directional if directional > 0.0 else 0.0
    impact = min(1.0, (directional + (evt * 0.60)) / max(1.0, float(max(1, total))))
    confidence = min(1.0, (directional + (evt * 0.40)) / 3.0)
    score = max(-1.0, min(1.0, sentiment * max(0.20, impact)))
    event_risk = bool(evt >= 1.0)
    if score >= 0.20:
        bias = "positive"
    elif score <= -0.20:
        bias = "negative"
    else:
        bias = "neutral"
    return {
        "score": round(float(score), 6),
        "impact": round(float(impact), 6),
        "confidence": round(float(confidence), 6),
        "bias": str(bias),
        "headline_count": int(total),
        "event_risk": bool(event_risk),
        "top_headline": str(latest_title or ""),
    }


def blend_score_with_news(
    base_score: float,
    news_score: float,
    confidence: float,
    impact: float,
    weight: float,
) -> float:
    base = float(base_score)
    news = max(-1.0, min(1.0, float(news_score)))
    conf = max(0.0, min(1.0, float(confidence)))
    imp = max(0.0, min(1.0, float(impact)))
    w = max(0.0, min(1.0, float(weight)))
    if w <= 0.0 or conf <= 0.0:
        return base
    direction = 1.0 if base >= 0.0 else -1.0
    alignment = direction * news
    mult = max(0.65, min(1.35, 1.0 + (w * conf * alignment)))
    blended = (base * mult) + (w * 0.25 * news * max(0.25, imp))
    return float(blended)


def build_unified_news_event_context(
    hub_dir: str,
    settings: Dict[str, Any],
    market: str,
    symbols: Iterable[Any],
    now_ts: int | None = None,
) -> Dict[str, Any]:
    now = int(time.time() if now_ts is None else now_ts)
    market_key = str(market or "").strip().lower()
    symbols_norm = _norm_symbols(symbols)
    enabled = _as_bool((settings or {}).get("news_event_enabled", True), True)
    refresh_s = max(60.0, _as_float((settings or {}).get("news_event_refresh_s", 900.0), 900.0))
    stale_max_s = max(refresh_s, _as_float((settings or {}).get("news_event_stale_max_s", 21600.0), 21600.0))
    timeout_s = max(3.0, min(30.0, _as_float((settings or {}).get("news_event_timeout_s", 8.0), 8.0)))
    max_symbols = max(1, _as_int((settings or {}).get("news_event_max_symbols_per_market", 20), 20))
    max_heads = max(1, _as_int((settings or {}).get("news_event_max_headlines_per_symbol", 6), 6))
    symbols_take = list(symbols_norm[:max_symbols])

    payload = {
        "enabled": bool(enabled),
        "market": market_key,
        "source": "yahoo_rss",
        "state": "disabled",
        "state_code": "disabled",
        "fetched_ts": 0,
        "retry_after_s": 0,
        "next_retry_ts": 0,
        "symbols": {},
        "errors": {},
    }
    if (not enabled) or (not symbols_take):
        return payload

    path = _cache_path(hub_dir)
    cache = _safe_read_json(path)
    markets = cache.get("markets", {}) if isinstance(cache.get("markets", {}), dict) else {}
    row = markets.get(market_key, {}) if isinstance(markets.get(market_key, {}), dict) else {}
    row_symbols = row.get("symbols", {}) if isinstance(row.get("symbols", {}), dict) else {}
    fetched_ts = int(row.get("fetched_ts", 0) or 0)
    next_retry_ts = int(row.get("next_retry_ts", 0) or 0)
    age_s = max(0, now - fetched_ts) if fetched_ts > 0 else 9_999_999
    has_all_cached = all(str(sym) in row_symbols for sym in symbols_take)

    if (next_retry_ts > now) and row_symbols:
        subset = {sym: dict(row_symbols.get(sym, {})) for sym in symbols_take if isinstance(row_symbols.get(sym, {}), dict)}
        return {
            **payload,
            "state": "cooldown",
            "state_code": "retry_cooldown",
            "fetched_ts": int(fetched_ts),
            "retry_after_s": int(max(0, next_retry_ts - now)),
            "next_retry_ts": int(next_retry_ts),
            "symbols": subset,
            "errors": {},
        }

    if has_all_cached and age_s <= refresh_s:
        subset = {sym: dict(row_symbols.get(sym, {})) for sym in symbols_take if isinstance(row_symbols.get(sym, {}), dict)}
        return {
            **payload,
            "state": "cached",
            "state_code": "cache_fresh",
            "fetched_ts": int(fetched_ts),
            "symbols": subset,
            "errors": {},
        }

    merged_symbols: Dict[str, Any] = {k: dict(v) for k, v in row_symbols.items() if isinstance(v, dict)}
    errors: Dict[str, str] = {}
    ok_count = 0
    for sym in symbols_take:
        query = _provider_symbol(market_key, sym)
        try:
            headlines = _fetch_symbol_headlines(query, timeout_s=timeout_s, max_items=max_heads)
            scored = _headline_scores(headlines, now)
            merged_symbols[sym] = {
                "query": str(query),
                "updated_ts": int(now),
                "headlines": list(headlines),
                "score": float(scored.get("score", 0.0) or 0.0),
                "impact": float(scored.get("impact", 0.0) or 0.0),
                "confidence": float(scored.get("confidence", 0.0) or 0.0),
                "bias": str(scored.get("bias", "neutral") or "neutral"),
                "headline_count": int(scored.get("headline_count", 0) or 0),
                "event_risk": bool(scored.get("event_risk", False)),
                "top_headline": str(scored.get("top_headline", "") or ""),
            }
            ok_count += 1
        except Exception as exc:
            errors[sym] = f"{type(exc).__name__}: {exc}"

    subset: Dict[str, Any] = {}
    for sym in symbols_take:
        row_sym = merged_symbols.get(sym, {}) if isinstance(merged_symbols.get(sym, {}), dict) else {}
        updated_ts = int(row_sym.get("updated_ts", 0) or 0)
        row_age = max(0, now - updated_ts) if updated_ts > 0 else 9_999_999
        if row_sym and row_age <= stale_max_s:
            subset[sym] = dict(row_sym)

    if ok_count > 0:
        state = "live"
        state_code = "live_ok"
        retry_after_s = 0
        next_retry = 0
        stored_fetched_ts = now
        last_error = ""
    elif subset:
        state = "cached_stale"
        state_code = "cache_fallback"
        retry_after_s = int(refresh_s)
        next_retry = int(now + retry_after_s)
        stored_fetched_ts = fetched_ts if fetched_ts > 0 else now
        last_error = "; ".join(list(errors.values())[:3])[:320]
    else:
        state = "unavailable"
        state_code = "fetch_failed"
        retry_after_s = int(refresh_s)
        next_retry = int(now + retry_after_s)
        stored_fetched_ts = fetched_ts
        last_error = "; ".join(list(errors.values())[:3])[:320]

    markets[market_key] = {
        "fetched_ts": int(stored_fetched_ts),
        "next_retry_ts": int(next_retry),
        "state": str(state),
        "state_code": str(state_code),
        "last_error_ts": int(now if errors else 0),
        "last_error": str(last_error),
        "symbols": merged_symbols,
    }
    _safe_write_json(path, {"ts": int(now), "markets": markets})
    return {
        **payload,
        "state": str(state),
        "state_code": str(state_code),
        "fetched_ts": int(stored_fetched_ts),
        "retry_after_s": int(retry_after_s),
        "next_retry_ts": int(next_retry),
        "symbols": subset,
        "errors": {k: str(v)[:220] for k, v in list(errors.items())[:6]},
    }
