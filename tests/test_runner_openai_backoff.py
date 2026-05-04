import unittest

from runtime.pt_runner import (
    OPENAI_ERROR_BACKOFF_MAX_S,
    OPENAI_GLOBAL_MIN_GAP_S,
    OPENAI_RATE_LIMIT_BACKOFF_BASE_S,
    Runner,
    _openai_backoff_interval_s,
    _openai_status_is_error,
)


class OpenAIRunnerBackoffTests(unittest.TestCase):
    def test_error_status_detection(self) -> None:
        self.assertTrue(_openai_status_is_error("http_error"))
        self.assertTrue(_openai_status_is_error("request_error"))
        self.assertFalse(_openai_status_is_error("ok"))
        self.assertFalse(_openai_status_is_error("disabled"))

    def test_rate_limit_429_prefers_rate_backoff(self) -> None:
        out = _openai_backoff_interval_s(
            base_interval_s=60.0,
            status_value="http_error",
            error_value="HTTP 429",
            failure_count=1,
        )
        self.assertGreaterEqual(out, OPENAI_RATE_LIMIT_BACKOFF_BASE_S)

    def test_non_rate_error_uses_transient_backoff(self) -> None:
        out = _openai_backoff_interval_s(
            base_interval_s=120.0,
            status_value="request_error",
            error_value="network timeout",
            failure_count=2,
        )
        self.assertGreaterEqual(out, 360.0)
        self.assertLessEqual(out, OPENAI_ERROR_BACKOFF_MAX_S)

    def test_retry_after_can_raise_backoff(self) -> None:
        out = _openai_backoff_interval_s(
            base_interval_s=120.0,
            status_value="http_error",
            error_value="HTTP 429",
            failure_count=1,
            retry_after_s=1200.0,
        )
        self.assertGreaterEqual(out, 1200.0)
        self.assertLessEqual(out, OPENAI_ERROR_BACKOFF_MAX_S)

    def test_success_status_keeps_base_interval(self) -> None:
        out = _openai_backoff_interval_s(
            base_interval_s=95.0,
            status_value="ok",
            error_value="",
            failure_count=5,
        )
        self.assertEqual(out, 95.0)

    def test_global_gap_blocks_immediate_burst_launches(self) -> None:
        runner = Runner()
        now = 1000.0
        runner._openai_mark_service_launch(service="position_review", now=now)
        self.assertFalse(
            runner._openai_can_launch_service(
                service="capital_planner",
                now=now + (OPENAI_GLOBAL_MIN_GAP_S * 0.5),
                settings={},
            )
        )
        self.assertTrue(
            runner._openai_can_launch_service(
                service="capital_planner",
                now=now + OPENAI_GLOBAL_MIN_GAP_S + 0.1,
                settings={},
            )
        )

    def test_global_gap_respects_setting_override(self) -> None:
        runner = Runner()
        now = 2000.0
        runner._openai_mark_service_launch(service="explanations", now=now)
        self.assertFalse(
            runner._openai_can_launch_service(
                service="market_context",
                now=now + 1.5,
                settings={"openai_global_min_gap_s": 3.0},
            )
        )
        self.assertTrue(
            runner._openai_can_launch_service(
                service="market_context",
                now=now + 3.1,
                settings={"openai_global_min_gap_s": 3.0},
            )
        )


if __name__ == "__main__":
    unittest.main()
