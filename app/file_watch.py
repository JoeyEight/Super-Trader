from __future__ import annotations

import os
import threading
from typing import Callable, Iterable, List, Optional, Set

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    Observer = None  # type: ignore[assignment]

    class FileSystemEvent:  # type: ignore[no-redef]
        is_directory: bool = False
        src_path: str = ""
        dest_path: str = ""

    class FileSystemEventHandler:  # type: ignore[no-redef]
        pass


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def on_any_event(self, event: FileSystemEvent) -> None:
        try:
            if bool(getattr(event, "is_directory", False)):
                return
            src = str(getattr(event, "src_path", "") or "").strip()
            if src:
                self._callback(src)
            dst = str(getattr(event, "dest_path", "") or "").strip()
            if dst:
                self._callback(dst)
        except Exception:
            return


class FileChangeWatch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed: Set[str] = set()
        self._observer = None
        self._roots: List[str] = []

    @property
    def available(self) -> bool:
        return Observer is not None

    def _record(self, path: str) -> None:
        p = os.path.abspath(str(path or "")).strip()
        if not p:
            return
        with self._lock:
            self._changed.add(p)

    def mark(self, path: str) -> None:
        self._record(path)

    def start(self, roots: Iterable[str]) -> bool:
        if Observer is None:
            return False
        normalized: List[str] = []
        seen: Set[str] = set()
        for root in roots or []:
            rp = os.path.abspath(str(root or "")).strip()
            if (not rp) or (rp in seen) or (not os.path.isdir(rp)):
                continue
            seen.add(rp)
            normalized.append(rp)
        if not normalized:
            return False
        self.stop()
        observer = Observer()
        handler = _ChangeHandler(self._record)
        scheduled = False
        for root in normalized:
            try:
                observer.schedule(handler, root, recursive=True)
                scheduled = True
            except Exception:
                continue
        if not scheduled:
            return False
        try:
            observer.start()
        except Exception:
            try:
                observer.stop()
            except Exception:
                pass
            return False
        self._observer = observer
        self._roots = normalized
        for root in normalized:
            self._record(root)
        return True

    def stop(self) -> None:
        obs = self._observer
        self._observer = None
        self._roots = []
        if obs is None:
            return
        try:
            obs.stop()
        except Exception:
            pass
        try:
            obs.join(timeout=1.5)
        except Exception:
            pass

    def pop_changed_paths(self) -> List[str]:
        with self._lock:
            if not self._changed:
                return []
            out = sorted(self._changed)
            self._changed.clear()
            return out

    def consume(self, matcher: Optional[Callable[[str], bool]] = None) -> bool:
        with self._lock:
            if not self._changed:
                return False
            if matcher is None:
                self._changed.clear()
                return True
            matched = [p for p in self._changed if matcher(p)]
            if not matched:
                return False
            for p in matched:
                self._changed.discard(p)
            return True

