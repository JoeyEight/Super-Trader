from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List

from app.json_codec import dump as json_dump
from app.json_codec import dumps as json_dumps
from app.time_utils import now_ts


def _atomic_tmp_path(path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    else:
        parent = "."
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
    os.close(fd)
    return str(tmp)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json_dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


_SENSITIVE_KEYS = ("api_key", "secret", "token", "private", "password", "passphrase")
_ASYNC_LOCK = threading.Lock()
_ASYNC_WRITERS: Dict[str, "_AsyncJsonlWriter"] = {}


class _AsyncJsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=4096)
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def enqueue(self, line: str) -> bool:
        try:
            self.q.put_nowait(str(line))
            return True
        except Exception:
            return False

    def _run(self) -> None:
        while not self._stop:
            try:
                line = self.q.get(timeout=0.25)
            except Exception:
                continue
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass


def _get_async_writer(path: str) -> _AsyncJsonlWriter:
    p = str(path)
    with _ASYNC_LOCK:
        hit = _ASYNC_WRITERS.get(p)
        if hit is not None:
            return hit
        writer = _AsyncJsonlWriter(p)
        _ASYNC_WRITERS[p] = writer
        return writer


def append_jsonl(path: str, payload: Dict[str, Any], async_mode: bool = False) -> None:
    try:
        clean_payload = redact_payload(payload)
        line = json_dumps(clean_payload, separators=(",", ":"), ensure_ascii=False) + "\n"
        if bool(async_mode):
            if _get_async_writer(path).enqueue(line):
                return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def append_jsonl_async(path: str, payload: Dict[str, Any]) -> None:
    append_jsonl(path, payload, async_mode=True)


def redact_text(text: str) -> str:
    s = str(text or "")
    needles = _SENSITIVE_KEYS
    low = s.lower()
    if any(n in low for n in needles):
        return "[redacted-sensitive]"
    return s


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k or "")
            low = key.lower()
            if any(n in low for n in _SENSITIVE_KEYS):
                out[key] = "[redacted-sensitive]"
            else:
                out[key] = redact_payload(v)
        return out
    if isinstance(value, list):
        return [redact_payload(v) for v in value]
    if isinstance(value, tuple):
        return [redact_payload(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def runtime_event(
    path: str,
    component: str,
    event: str,
    level: str = "info",
    msg: str = "",
    details: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "ts": now_ts(),
        "component": str(component or "").strip() or "runtime",
        "event": str(event or "").strip() or "event",
        "level": str(level or "info").strip().lower(),
        "msg": redact_text(str(msg or "").strip()),
        "details": redact_payload(dict(details or {})),
    }
    append_jsonl(path, payload, async_mode=False)


def cleanup_logs(log_dir: str, keep_patterns: Iterable[str] | None = None, max_age_days: float = 14.0, max_total_bytes: int = 200 * 1024 * 1024) -> Dict[str, Any]:
    keep = {str(x).strip() for x in (keep_patterns or ()) if str(x).strip()}
    now = time.time()
    max_age_s = max(1.0, float(max_age_days)) * 86400.0
    files: List[Dict[str, Any]] = []
    removed = 0
    removed_bytes = 0

    try:
        names = os.listdir(log_dir)
    except Exception:
        return {"removed": 0, "removed_bytes": 0, "total_bytes": 0, "candidates": 0}

    for name in names:
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        if keep and any(name.startswith(prefix) for prefix in keep):
            continue
        try:
            st = os.stat(path)
        except Exception:
            continue
        files.append({"path": path, "name": name, "size": int(st.st_size), "mtime": float(st.st_mtime)})

    total_bytes = sum(int(x["size"]) for x in files)

    # Age pass first.
    for row in sorted(files, key=lambda x: float(x["mtime"])):
        if (now - float(row["mtime"])) <= max_age_s:
            continue
        try:
            os.remove(str(row["path"]))
            removed += 1
            removed_bytes += int(row["size"])
            total_bytes -= int(row["size"])
        except Exception:
            pass

    # Size-budget pass after age cleanup.
    if total_bytes > int(max_total_bytes):
        survivors = []
        for row in files:
            if not os.path.isfile(str(row["path"])):
                continue
            survivors.append(row)
        for row in sorted(survivors, key=lambda x: float(x["mtime"])):
            if total_bytes <= int(max_total_bytes):
                break
            try:
                os.remove(str(row["path"]))
                removed += 1
                removed_bytes += int(row["size"])
                total_bytes -= int(row["size"])
            except Exception:
                pass

    return {
        "removed": int(removed),
        "removed_bytes": int(removed_bytes),
        "total_bytes": int(max(0, total_bytes)),
        "candidates": int(len(files)),
    }


def trim_jsonl_max_lines(path: str, max_lines: int = 20000) -> Dict[str, Any]:
    target = max(1, int(max_lines or 0))
    tmp = ""
    try:
        if not os.path.isfile(path):
            return {"trimmed": False, "kept": 0, "dropped": 0}
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = int(len(lines))
        if total <= target:
            return {"trimmed": False, "kept": total, "dropped": 0}
        kept = lines[-target:]
        dropped = total - target
        tmp = _atomic_tmp_path(path)
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, path)
        return {"trimmed": True, "kept": int(len(kept)), "dropped": int(dropped)}
    except Exception:
        return {"trimmed": False, "kept": 0, "dropped": 0}
    finally:
        if tmp:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
