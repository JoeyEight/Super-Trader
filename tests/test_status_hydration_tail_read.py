from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.status_hydration import safe_read_jsonl_dicts


class StatusHydrationTailReadTests(unittest.TestCase):
    def test_safe_read_jsonl_dicts_returns_recent_rows(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                for idx in range(1200):
                    f.write(json.dumps({"idx": idx}) + "\n")
            rows = safe_read_jsonl_dicts(path, limit=5)
            self.assertEqual([int(r.get("idx", -1)) for r in rows], [1195, 1196, 1197, 1198, 1199])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def test_safe_read_jsonl_dicts_ignores_non_dict_rows(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("[]\n")
                f.write(json.dumps({"ok": 1}) + "\n")
                f.write("{\"bad\":\n")
                f.write(json.dumps({"ok": 2}) + "\n")
            rows = safe_read_jsonl_dicts(path, limit=10)
            self.assertEqual([int(r.get("ok", -1)) for r in rows], [1, 2])
        finally:
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
