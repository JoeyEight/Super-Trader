from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List


class TwelveDataClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.twelvedata.com",
        timeout: float = 15.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "https://api.twelvedata.com").strip().rstrip("/")
        self.timeout = float(timeout or 15.0)

    def _request_json(self, path: str, params: Dict[str, Any], timeout: float | None = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing Twelve Data API key")
        payload = dict(params or {})
        payload["apikey"] = self.api_key
        url = f"{self.base_url}/{str(path or '').lstrip('/')}"
        url = f"{url}?{urllib.parse.urlencode(payload)}"
        # Twelve Data sits behind Cloudflare; explicit headers avoid 403/1010 blocks.
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SuperTrader/1.0 (+local trading bot)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout or self.timeout)) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = (exc.read() or b"").decode("utf-8", "replace").strip()
            except Exception:
                body = ""
            msg = f"HTTP {int(getattr(exc, 'code', 0) or 0)} {str(getattr(exc, 'reason', '') or '').strip()}".strip()
            if body:
                msg = f"{msg} | {body[:200]}"
            raise RuntimeError(msg) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _normalize_series_rows(values: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in list(values or []):
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "t": row.get("datetime") or row.get("date") or row.get("timestamp") or "",
                    "o": row.get("open"),
                    "h": row.get("high"),
                    "l": row.get("low"),
                    "c": row.get("close"),
                    "v": row.get("volume"),
                }
            )
        out.sort(key=lambda r: str(r.get("t", "") or ""))
        return out

    def get_time_series_batch(
        self,
        symbols: List[str],
        interval: str = "1h",
        outputsize: int = 96,
    ) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        sym_list = [str(s or "").strip().upper() for s in list(symbols or []) if str(s or "").strip()]
        if not sym_list:
            return out
        params = {
            "symbol": ",".join(sym_list),
            "interval": str(interval or "1h"),
            "outputsize": str(int(max(1, outputsize))),
            "format": "JSON",
        }
        payload = self._request_json("/time_series", params)
        if not payload:
            return out
        if str(payload.get("status", "") or "").lower() == "error":
            return out
        if "values" in payload:
            out[sym_list[0]] = self._normalize_series_rows(payload.get("values"))
            return out
        for key, block in payload.items():
            if not isinstance(block, dict):
                continue
            values = block.get("values")
            if isinstance(values, list):
                sym = str(key or "").strip().upper()
                if sym:
                    out[sym] = self._normalize_series_rows(values)
        return out
