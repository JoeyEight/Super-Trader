from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
import urllib.parse
from email.message import Message
from unittest.mock import patch

from brokers.broker_alpaca import AlpacaBrokerClient
from engines import stock_thinker


class _UniverseClient:
    def __init__(self, api_key_id: str, secret_key: str, base_url: str, data_url: str) -> None:
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = data_url

    def list_tradable_assets(self) -> list[dict]:
        return [
            {"symbol": "AAPL", "tradable": True, "status": "active", "class": "us_equity", "exchange": "NASDAQ", "marginable": True, "fractionable": True},
            {"symbol": "TSLA", "tradable": True, "status": "active", "class": "us_equity", "exchange": "NASDAQ", "marginable": True, "fractionable": True},
            {"symbol": "BRK.B", "tradable": True, "status": "active", "class": "us_equity", "exchange": "NYSE", "marginable": True, "fractionable": True},
            {"symbol": "KMTUY", "tradable": True, "status": "active", "class": "us_equity", "exchange": "OTC", "marginable": True, "fractionable": True},
            {"symbol": "ABCD1", "tradable": True, "status": "active", "class": "us_equity", "exchange": "NASDAQ", "marginable": True, "fractionable": True},
            {"symbol": "ZZZZ", "tradable": True, "status": "active", "class": "us_equity", "exchange": "NASDAQ", "marginable": False, "fractionable": True},
            {"symbol": "QQQ", "tradable": True, "status": "active", "class": "us_equity", "exchange": "ARCA", "marginable": True, "fractionable": True},
        ]


class TestStockScanApiAlignment(unittest.TestCase):
    def test_all_tradable_universe_filters_non_scannable_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = {
                "market_rollout_stage": "scan_expanded",
                "stock_universe_mode": "all_tradable_filtered",
                "stock_universe_symbols": "MSFT",
                "alpaca_base_url": "https://paper-api.alpaca.markets",
                "alpaca_data_url": "https://data.alpaca.markets",
            }
            with patch.object(stock_thinker, "AlpacaBrokerClient", _UniverseClient):
                out = stock_thinker._select_universe(settings, td, api_key="k", secret="s")
            # Watchlist symbol should be kept at front, then liquid scannable symbols.
            self.assertEqual(out[:4], ["MSFT", "QQQ", "AAPL", "TSLA"])
            self.assertNotIn("BRK.B", out)
            self.assertNotIn("KMTUY", out)
            self.assertNotIn("ABCD1", out)
            self.assertNotIn("ZZZZ", out)

    def test_feed_order_defaults_to_iex_first(self) -> None:
        self.assertEqual(stock_thinker._parse_feed_order({}), ["iex", "sip"])
        self.assertEqual(stock_thinker._parse_feed_order({"stock_data_feeds": "sip,iex"}), ["sip", "iex"])

    def test_stock_bars_request_uses_recent_sort_and_normalizes_order(self) -> None:
        client = AlpacaBrokerClient(api_key_id="k", secret_key="s", base_url="https://paper-api.alpaca.markets", data_url="https://data.alpaca.markets")
        calls: list[str] = []

        def _fake_request(path: str, timeout: float = 8.0, max_attempts: int = 3) -> dict:
            calls.append(path)
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-03-06T15:00:00Z", "c": 110.0},
                        {"t": "2026-03-06T14:00:00Z", "c": 109.0},
                        {"t": "2026-03-06T13:00:00Z", "c": 108.0},
                    ]
                }
            }

        client._request_data_json = _fake_request  # type: ignore[method-assign]
        rows = client.get_stock_bars("AAPL", timeframe="1Hour", limit=120, feed="iex")
        self.assertEqual([str(r.get("t", "")) for r in rows], ["2026-03-06T13:00:00Z", "2026-03-06T14:00:00Z", "2026-03-06T15:00:00Z"])
        self.assertTrue(any("sort=desc" in call for call in calls))

    def test_stock_bars_feed_fallback_from_sip_to_iex(self) -> None:
        client = AlpacaBrokerClient(api_key_id="k", secret_key="s", base_url="https://paper-api.alpaca.markets", data_url="https://data.alpaca.markets")
        calls: list[str] = []

        def _fake_request(path: str, timeout: float = 8.0, max_attempts: int = 3) -> dict:
            calls.append(path)
            if "feed=sip" in path:
                raise urllib.error.HTTPError(
                    url="https://data.alpaca.markets",
                    code=403,
                    msg="Forbidden",
                    hdrs=Message(),
                    fp=io.BytesIO(b'{"message":"subscription does not permit requested feed"}'),
                )
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-03-06T15:00:00Z", "c": 110.0},
                        {"t": "2026-03-06T14:00:00Z", "c": 109.0},
                    ]
                }
            }

        client._request_data_json = _fake_request  # type: ignore[method-assign]
        rows = client.get_stock_bars("AAPL", timeframe="1Hour", limit=120, feed="sip")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any("feed=sip" in call for call in calls))
        self.assertTrue(any("feed=iex" in call for call in calls))

    def test_stock_snapshot_does_not_report_equity_as_realized_pnl(self) -> None:
        client = AlpacaBrokerClient(
            api_key_id="k",
            secret_key="s",
            base_url="https://paper-api.alpaca.markets",
            data_url="https://data.alpaca.markets",
        )
        client.get_account_summary = lambda: {"status": "ACTIVE", "buying_power": "2000.00", "equity": "1000.00"}  # type: ignore[method-assign]
        client.list_positions = lambda: [{"symbol": "AAPL", "qty": "1", "market_value": "182.50", "unrealized_pl": "2.10"}]  # type: ignore[method-assign]
        payload = client.fetch_snapshot()
        self.assertEqual(payload.get("realized_pnl"), "N/A")
        self.assertEqual(payload.get("equity"), "1000.00")

    def test_batch_bars_fetch_uses_docs_aligned_limit_and_pagination(self) -> None:
        calls: list[str] = []

        def _fake_request(url: str, headers: dict, timeout: float = 15.0) -> dict:
            del headers, timeout
            calls.append(url)
            if "page_token=tok-2" in url:
                return {
                    "bars": {
                        "MSFT": [
                            {"t": "2026-03-06T15:00:00Z", "c": 301.0},
                            {"t": "2026-03-06T14:00:00Z", "c": 300.0},
                        ]
                    }
                }
            return {
                "bars": {
                    "AAPL": [
                        {"t": "2026-03-06T15:00:00Z", "c": 201.0},
                        {"t": "2026-03-06T14:00:00Z", "c": 200.0},
                    ]
                },
                "next_page_token": "tok-2",
            }

        with patch.object(stock_thinker, "_request_json", side_effect=_fake_request):
            out = stock_thinker._fetch_bars_for_symbols(
                base_url="https://data.alpaca.markets",
                headers={"APCA-API-KEY-ID": "k"},
                symbols=["AAPL", "MSFT"],
                start_iso="2026-03-01T00:00:00Z",
                end_iso="2026-03-06T23:59:59Z",
                feed="iex",
                min_bars_hint=24,
            )

        self.assertIn("AAPL", out)
        self.assertIn("MSFT", out)
        self.assertEqual([str(r.get("t", "")) for r in out["AAPL"]], ["2026-03-06T14:00:00Z", "2026-03-06T15:00:00Z"])
        self.assertEqual([str(r.get("t", "")) for r in out["MSFT"]], ["2026-03-06T14:00:00Z", "2026-03-06T15:00:00Z"])
        self.assertTrue(any("page_token=tok-2" in c for c in calls))
        self.assertTrue(any("sort=desc" in c for c in calls))
        first_qs = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0]).query)
        first_limit = int((first_qs.get("limit", ["0"]) or ["0"])[0])
        self.assertGreaterEqual(first_limit, 500)


if __name__ == "__main__":
    unittest.main()
