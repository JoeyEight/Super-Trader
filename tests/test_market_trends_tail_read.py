from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.market_trends import _safe_read_jsonl


class MarketTrendsTailReadTests(unittest.TestCase):
    def test_safe_read_jsonl_returns_recent_rows(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                for idx in range(2500):
                    f.write(json.dumps({"idx": idx}) + "\n")
            rows = _safe_read_jsonl(path, max_lines=7)
            self.assertEqual(len(rows), 7)
            self.assertEqual([int(r.get("idx", -1)) for r in rows], [2493, 2494, 2495, 2496, 2497, 2498, 2499])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def test_safe_read_jsonl_skips_invalid_tail_rows(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("not-json\n")
                f.write(json.dumps({"idx": 1}) + "\n")
                f.write("[]\n")
                f.write(json.dumps({"idx": 2}) + "\n")
            rows = _safe_read_jsonl(path, max_lines=10)
            self.assertEqual([int(r.get("idx", -1)) for r in rows], [1, 2])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
