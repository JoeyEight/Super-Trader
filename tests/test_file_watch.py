from __future__ import annotations

import os

from app.file_watch import FileChangeWatch


def test_file_watch_mark_and_pop() -> None:
    watcher = FileChangeWatch()
    path = os.path.abspath("hub_data/runtime_state.json")
    watcher.mark(path)
    changed = watcher.pop_changed_paths()
    assert path in changed
    assert watcher.pop_changed_paths() == []


def test_file_watch_consume_matcher() -> None:
    watcher = FileChangeWatch()
    a = os.path.abspath("hub_data/logs/runner.log")
    b = os.path.abspath("hub_data/runtime_state.json")
    watcher.mark(a)
    watcher.mark(b)

    matched = watcher.consume(lambda p: p.endswith(".log"))
    assert matched is True
    remaining = watcher.pop_changed_paths()
    assert b in remaining
    assert a not in remaining
