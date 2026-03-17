from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from app.api_endpoint_validation import normalize_endpoint_url
from app.backoff_policy import BackoffPolicy
from app.http_utils import retry_after_from_urllib_http_error


def _retry_after_seconds_from_http_error(exc: urllib.error.HTTPError) -> float:
    return retry_after_from_urllib_http_error(exc, max_wait_s=300.0)


class OandaBrokerClient:
    def __init__(self, account_id: str, api_token: str, rest_url: str) -> None:
        self.account_id = str(account_id or "").strip()
        self.api_token = str(api_token or "").strip()
        norm_rest, rest_ok, _ = normalize_endpoint_url(rest_url, default="https://api-fxpractice.oanda.com")
        self.rest_url = str((norm_rest if rest_ok else "https://api-fxpractice.oanda.com") or "").strip().rstrip("/")

    def configured(self) -> bool:
        return bool(self.account_id and self.api_token and self.rest_url)

    @staticmethod
    def _as_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _retryable_http(code: int) -> bool:
        return int(code or 0) in {408, 425, 429, 500, 502, 503, 504}

    def _open_with_retry(self, req: urllib.request.Request, timeout: float, max_attempts: int = 3) -> str:
        attempts = max(1, int(max_attempts or 1))
        backoff = BackoffPolicy(base_delay_s=0.2, max_delay_s=8.0, jitter_s=0.25, max_retry_after_s=300.0)
        last_exc: Exception | None = None
        for att in range(1, attempts + 1):
            retry_after_s = 0.0
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    if isinstance(body, bytes):
                        return body.decode("utf-8")
                    return str(body or "")
            except urllib.error.HTTPError as exc:
                last_exc = exc
                retry_after_s = _retry_after_seconds_from_http_error(exc)
                if (not self._retryable_http(int(getattr(exc, "code", 0) or 0))) or att >= attempts:
                    raise
            except urllib.error.URLError as exc:
                last_exc = exc
                if att >= attempts:
                    raise
            except Exception as exc:
                last_exc = exc
                if att >= attempts:
                    raise
            wait_s = backoff.wait_seconds(att, retry_after_s=retry_after_s)
            time.sleep(wait_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OANDA request failed")

    def _request_json(self, path: str, timeout: float = 8.0, max_attempts: int = 3) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.rest_url}{path}",
            headers={"Authorization": f"Bearer {self.api_token}"},
        )
        raw = self._open_with_retry(req, timeout=timeout, max_attempts=max_attempts)
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}

    def _request(
        self,
        method: str,
        path: str,
        body: Dict[str, Any] | None = None,
        timeout: float = 8.0,
        max_attempts: int = 1,
    ) -> Dict[str, Any]:
        raw = None
        headers = {"Authorization": f"Bearer {self.api_token}"}
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self.rest_url}{path}",
            headers=headers,
            data=raw,
            method=method,
        )
        txt = self._open_with_retry(req, timeout=timeout, max_attempts=max_attempts)
        payload = json.loads(txt or "{}")
        return payload if isinstance(payload, dict) else {}

    def test_connection(self) -> Tuple[bool, str]:
        if not self.configured():
            return False, "OANDA credentials missing"
        try:
            payload = self._request_json(f"/v3/accounts/{self.account_id}/summary")
            acct = payload.get("account", {}) or {}
            nav = acct.get("NAV", "N/A")
            currency = acct.get("currency", "")
            return True, f"Connected | NAV={nav} {currency}".strip()
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            return False, f"Network error: {exc.reason}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def fetch_snapshot(self) -> Dict[str, Any]:
        if not self.configured():
            return {
                "state": "NOT CONFIGURED",
                "ai_state": "Credentials missing",
                "trader_state": "Idle",
                "msg": "Add OANDA practice credentials in Settings",
                "buying_power": "Pending account link",
                "open_positions": "0",
                "realized_pnl": "N/A",
                "positions_preview": [],
            }

        try:
            summary_payload = self._request_json(f"/v3/accounts/{self.account_id}/summary")
            positions_payload = self._request_json(f"/v3/accounts/{self.account_id}/openPositions")
            acct = summary_payload.get("account", {}) or {}
            raw_positions = positions_payload.get("positions", []) or []
            if not isinstance(raw_positions, list):
                raw_positions = []

            preview: List[str] = []
            for pos in raw_positions[:8]:
                if not isinstance(pos, dict):
                    continue
                inst = str(pos.get("instrument", "") or "").strip()
                long_units = str(((pos.get("long") or {}).get("units", "")) or "").strip()
                short_units = str(((pos.get("short") or {}).get("units", "")) or "").strip()
                upl = str(((pos.get("long") or {}).get("unrealizedPL", "")) or ((pos.get("short") or {}).get("unrealizedPL", "")) or "").strip()
                side = f"L {long_units}" if long_units and long_units not in ("0", "0.0") else f"S {short_units}"
                preview.append(f"{inst} | {side} | uPnL {upl or '0'}")

            currency = str(acct.get("currency", "") or "").strip()
            nav = acct.get("NAV", "N/A")
            pl = acct.get("pl", "N/A")
            nav_f = self._as_float(nav, 0.0)
            pl_f = self._as_float(pl, 0.0)
            margin_available_f = self._as_float(acct.get("marginAvailable", 0.0), 0.0)
            margin_rate_f = self._as_float(acct.get("marginRate", 0.0), 0.0)
            return {
                "state": "READY",
                "ai_state": "Broker linked",
                "trader_state": "Practice mode ready",
                "msg": f"NAV {nav} {currency}".strip(),
                "buying_power": f"{acct.get('marginAvailable', 'N/A')} {currency}".strip(),
                "open_positions": str(len(raw_positions)),
                "realized_pnl": f"{pl} {currency}".strip(),
                "positions_preview": preview,
                "raw_positions": raw_positions,
                "nav": nav_f,
                "pl_value": pl_f,
                "margin_available": margin_available_f,
                "margin_rate": margin_rate_f,
                "currency": currency,
            }
        except urllib.error.HTTPError as exc:
            return {
                "state": "ERROR",
                "ai_state": "Broker error",
                "trader_state": "Idle",
                "msg": f"HTTP {exc.code}: {exc.reason}",
                "buying_power": "N/A",
                "open_positions": "0",
                "realized_pnl": "N/A",
                "positions_preview": [],
                "raw_positions": [],
                "nav": 0.0,
                "pl_value": 0.0,
                "margin_available": 0.0,
                "margin_rate": 0.0,
                "currency": "",
            }
        except urllib.error.URLError as exc:
            return {
                "state": "ERROR",
                "ai_state": "Network error",
                "trader_state": "Idle",
                "msg": f"Network error: {exc.reason}",
                "buying_power": "N/A",
                "open_positions": "0",
                "realized_pnl": "N/A",
                "positions_preview": [],
                "raw_positions": [],
                "nav": 0.0,
                "pl_value": 0.0,
                "margin_available": 0.0,
                "margin_rate": 0.0,
                "currency": "",
            }
        except Exception as exc:
            return {
                "state": "ERROR",
                "ai_state": "Load failed",
                "trader_state": "Idle",
                "msg": f"{type(exc).__name__}: {exc}",
                "buying_power": "N/A",
                "open_positions": "0",
                "realized_pnl": "N/A",
                "positions_preview": [],
                "raw_positions": [],
                "nav": 0.0,
                "pl_value": 0.0,
                "margin_available": 0.0,
                "margin_rate": 0.0,
                "currency": "",
            }

    def get_mid_prices(self, instruments: List[str]) -> Dict[str, float]:
        if not self.configured():
            return {}
        items = [str(x).strip().upper() for x in (instruments or []) if str(x).strip()]
        if not items:
            return {}
        qs = ",".join(items)
        try:
            payload = self._request_json(f"/v3/accounts/{self.account_id}/pricing?instruments={qs}", timeout=8.0)
        except Exception:
            return {}
        prices = payload.get("prices", []) or []
        out: Dict[str, float] = {}
        if not isinstance(prices, list):
            return out
        for row in prices:
            if not isinstance(row, dict):
                continue
            inst = str(row.get("instrument", "") or "").strip().upper()
            bids = row.get("bids", []) or []
            asks = row.get("asks", []) or []
            try:
                bid = float((bids[0] or {}).get("price", 0.0)) if bids else 0.0
                ask = float((asks[0] or {}).get("price", 0.0)) if asks else 0.0
            except Exception:
                bid = 0.0
                ask = 0.0
            if bid > 0 and ask > 0:
                out[inst] = (bid + ask) / 2.0
            elif ask > 0:
                out[inst] = ask
            elif bid > 0:
                out[inst] = bid
        return out

    def get_pricing_details(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        if not self.configured():
            return {}
        items = [str(x).strip().upper() for x in (instruments or []) if str(x).strip()]
        if not items:
            return {}
        qs = ",".join(items)
        try:
            payload = self._request_json(f"/v3/accounts/{self.account_id}/pricing?instruments={qs}", timeout=8.0)
        except Exception:
            return {}
        prices = payload.get("prices", []) or []
        out: Dict[str, Dict[str, float]] = {}
        if not isinstance(prices, list):
            return out
        for row in prices:
            if not isinstance(row, dict):
                continue
            inst = str(row.get("instrument", "") or "").strip().upper()
            bids = row.get("bids", []) or []
            asks = row.get("asks", []) or []
            try:
                bid = float((bids[0] or {}).get("price", 0.0)) if bids else 0.0
                ask = float((asks[0] or {}).get("price", 0.0)) if asks else 0.0
            except Exception:
                bid = 0.0
                ask = 0.0
            mid = (bid + ask) * 0.5 if bid > 0 and ask > 0 else max(bid, ask, 0.0)
            spread_bps = (((ask - bid) / mid) * 10000.0) if (mid > 0 and ask > 0 and bid > 0) else 0.0
            quote_ccy = ""
            if "_" in inst:
                try:
                    quote_ccy = str(inst.split("_", 1)[1] or "").strip().upper()
                except Exception:
                    quote_ccy = ""
            qh = row.get("quoteHomeConversionFactors", {}) or {}
            try:
                qh_pos = float(qh.get("positiveUnits", 0.0) or 0.0)
            except Exception:
                qh_pos = 0.0
            try:
                qh_neg = abs(float(qh.get("negativeUnits", 0.0) or 0.0))
            except Exception:
                qh_neg = 0.0
            quote_to_home = max(abs(qh_pos), abs(qh_neg), 0.0)
            home_conversions = row.get("homeConversions", []) or []
            if quote_to_home <= 0.0 and quote_ccy and isinstance(home_conversions, list):
                for conv in home_conversions:
                    if not isinstance(conv, dict):
                        continue
                    if str(conv.get("currency", "") or "").strip().upper() != quote_ccy:
                        continue
                    try:
                        quote_to_home = max(
                            quote_to_home,
                            abs(float(conv.get("positionValue", 0.0) or 0.0)),
                            abs(float(conv.get("accountGain", 0.0) or 0.0)),
                            abs(float(conv.get("accountLoss", 0.0) or 0.0)),
                        )
                    except Exception:
                        pass
                    break
            out[inst] = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_bps": spread_bps,
                "quote_currency": quote_ccy,
                "quote_to_home": quote_to_home,
                "quote_to_home_positive": qh_pos,
                "quote_to_home_negative": qh_neg,
            }
        return out

    def list_tradeable_instruments(self) -> List[str]:
        if not self.configured():
            return []
        try:
            payload = self._request_json(f"/v3/accounts/{self.account_id}/instruments", timeout=10.0)
        except Exception:
            return []
        rows = payload.get("instruments", []) or []
        out: List[str] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("type", "") or "").upper() != "CURRENCY":
                continue
            if str(row.get("tradeable", "true")).lower() in {"false", "0", "no"}:
                continue
            name = str(row.get("name", "") or "").strip().upper()
            if name and name not in out:
                out.append(name)
        return out

    def get_candles(self, instrument: str, granularity: str = "H1", count: int = 120) -> List[Dict[str, Any]]:
        if not self.configured():
            return []
        inst = str(instrument or "").strip().upper()
        if not inst:
            return []
        g = str(granularity or "H1").strip().upper()
        cnt = max(10, min(5000, int(count or 120)))
        try:
            payload = self._request_json(
                f"/v3/instruments/{inst}/candles?price=M&granularity={g}&count={cnt}",
                timeout=10.0,
            )
            rows = payload.get("candles", []) or []
            return [row for row in rows if isinstance(row, dict)]
        except Exception:
            return []

    def place_market_order(
        self,
        instrument: str,
        units: int,
        client_order_id: str = "",
        max_retries: int = 2,
        retry_delay_s: float = 0.35,
        max_retry_after_s: float = 300.0,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        inst = str(instrument or "").strip().upper()
        if not self.configured():
            return False, "Not configured", {}
        if not inst:
            return False, "Missing instrument", {}
        body: Dict[str, Any] = {
            "order": {
                "type": "MARKET",
                "instrument": inst,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if str(client_order_id or "").strip():
            body["order"]["clientExtensions"] = {"id": str(client_order_id).strip()[:48]}
        attempts = max(1, int(max_retries))
        backoff = BackoffPolicy(
            base_delay_s=max(0.05, float(retry_delay_s)),
            max_delay_s=8.0,
            jitter_s=0.35,
            max_retry_after_s=max(1.0, float(max_retry_after_s or 300.0)),
        )
        last_msg = "Order not accepted"
        for att in range(1, attempts + 1):
            retry_after_s = 0.0
            try:
                payload = self._request(
                    "POST",
                    f"/v3/accounts/{self.account_id}/orders",
                    body=body,
                    timeout=10.0,
                )
                ok = bool(payload.get("orderFillTransaction") or payload.get("orderCreateTransaction"))
                msg = "Order submitted" if ok else str(payload.get("errorMessage", "Order not accepted") or "Order not accepted")
                if ok:
                    return True, msg, payload
                last_msg = msg
            except urllib.error.HTTPError as exc:
                last_msg = f"HTTP {exc.code}: {exc.reason}"
                if int(exc.code) != 429:
                    break
                retry_after_s = _retry_after_seconds_from_http_error(exc)
            except urllib.error.URLError as exc:
                last_msg = f"Network error: {exc.reason}"
            except Exception as exc:
                last_msg = f"{type(exc).__name__}: {exc}"
            if att < attempts:
                try:
                    import time as _t
                    wait_s = backoff.wait_seconds(att, retry_after_s=retry_after_s)
                    if retry_after_s > 0.0:
                        last_msg = f"{last_msg} | retry_after={wait_s:.2f}s".strip()
                    _t.sleep(wait_s)
                except Exception:
                    pass
        return False, last_msg, {}

    def close_position(self, instrument: str, side: str = "all") -> Tuple[bool, str, Dict[str, Any]]:
        inst = str(instrument or "").strip().upper()
        which = str(side or "all").strip().lower()
        if not self.configured():
            return False, "Not configured", {}
        if not inst:
            return False, "Missing instrument", {}
        body: Dict[str, Any] = {}
        if which == "long":
            body["longUnits"] = "ALL"
        elif which == "short":
            body["shortUnits"] = "ALL"
        else:
            body["longUnits"] = "ALL"
            body["shortUnits"] = "ALL"
        try:
            payload = self._request(
                "PUT",
                f"/v3/accounts/{self.account_id}/positions/{inst}/close",
                body=body,
                timeout=10.0,
            )
            ok = bool(payload)
            msg = "Position close submitted" if ok else "Close not accepted"
            return ok, msg, payload
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}: {exc.reason}", {}
        except urllib.error.URLError as exc:
            return False, f"Network error: {exc.reason}", {}
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", {}
