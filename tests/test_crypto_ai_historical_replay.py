from __future__ import annotations

import unittest

from app.crypto_ai_historical_replay import _build_closed_trades, _build_hybrid_predictions, _normalize_prediction


class TestCryptoAiHistoricalReplay(unittest.TestCase):
    def test_normalize_prediction_canonicalizes_trigger_labels(self) -> None:
        out = _normalize_prediction(
            {
                "predicted_direction": "up",
                "exit_trigger": "trailing sell",
                "expected_exit_ts": 1_700_000_100,
                "expected_exit_price": 110.0,
                "expected_hold_hours": 2.0,
                "confidence": 0.8,
            },
            entry_ts=1_700_000_000,
            entry_price=100.0,
        )
        self.assertEqual(str(out.get("exit_trigger", "")), "Trailing")

    def test_build_closed_trades_skips_manual_exits_for_replay_training(self) -> None:
        rows = [
            {"ts": 1_700_000_000, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0, "tag": "", "score": 1.0, "required_score": 0.2, "calib_prob": 0.7},
            {"ts": 1_700_000_010, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 99.0, "tag": "MANUAL_SELL_USD"},
            {"ts": 1_700_000_020, "event": "entry", "symbol": "BTC-USD", "qty": 1.0, "price": 100.0, "tag": "", "score": 1.2, "required_score": 0.2, "calib_prob": 0.8},
            {"ts": 1_700_000_030, "event": "exit", "symbol": "BTC-USD", "qty": 1.0, "price": 101.0, "tag": "POLICY_STALE_EXIT"},
        ]
        closed = _build_closed_trades(rows)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].exit_tag, "POLICY_STALE_EXIT")

    def test_hybrid_override_blocks_conflicting_bearish_trigger_on_up_direction(self) -> None:
        direction_rows = [
            {
                "symbol": "BTC-USD",
                "entry_ts": 1_700_000_000,
                "predicted_direction": "up",
                "predicted_exit_trigger": "Trailing",
            }
        ]
        trigger_rows = [
            {
                "symbol": "BTC-USD",
                "entry_ts": 1_700_000_000,
                "predicted_exit_trigger": "Stale Alignment",
                "predicted_confidence": 0.95,
            }
        ]
        out = _build_hybrid_predictions(direction_rows, trigger_rows, trigger_override_conf_min=0.6)
        self.assertEqual(str(out[0].get("predicted_exit_trigger", "")), "Trailing")

    def test_hybrid_override_applies_non_conflicting_trigger(self) -> None:
        direction_rows = [
            {
                "symbol": "ETH-USD",
                "entry_ts": 1_700_000_100,
                "predicted_direction": "down",
                "predicted_exit_trigger": "Unknown",
            }
        ]
        trigger_rows = [
            {
                "symbol": "ETH-USD",
                "entry_ts": 1_700_000_100,
                "predicted_exit_trigger": "Stale Alignment",
                "predicted_confidence": 0.95,
            }
        ]
        out = _build_hybrid_predictions(direction_rows, trigger_rows, trigger_override_conf_min=0.6)
        self.assertEqual(str(out[0].get("predicted_exit_trigger", "")), "Stale Alignment")


if __name__ == "__main__":
    unittest.main()
