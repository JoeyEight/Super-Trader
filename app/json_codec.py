from __future__ import annotations

import json
from typing import Any, IO

_MISSING = object()

try:
    import orjson as _orjson  # type: ignore[import-not-found]
except Exception:
    _orjson = None


def has_orjson() -> bool:
    return _orjson is not None


def loads(data: Any, *, default: Any = _MISSING) -> Any:
    raw = data
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)

    if isinstance(raw, str):
        if not raw.strip():
            if default is _MISSING:
                raise json.JSONDecodeError("Expecting value", raw, 0)
            return default
    elif isinstance(raw, (bytes, bytearray)):
        if not bytes(raw).strip():
            if default is _MISSING:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return default

    try:
        if _orjson is not None:
            return _orjson.loads(raw)
        return json.loads(raw)
    except Exception:
        if default is _MISSING:
            raise
        return default


def load(fp: IO[str], *, default: Any = _MISSING) -> Any:
    try:
        raw = fp.read()
    except Exception:
        if default is _MISSING:
            raise
        return default
    return loads(raw, default=default)


def dumps(
    payload: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    if _orjson is not None and (not ensure_ascii) and separators is None and (indent in (None, 2)):
        options = 0
        if indent == 2:
            options |= int(getattr(_orjson, "OPT_INDENT_2", 0))
        if sort_keys:
            options |= int(getattr(_orjson, "OPT_SORT_KEYS", 0))
        try:
            encoded = _orjson.dumps(payload, option=options)
            return encoded.decode("utf-8")
        except Exception:
            pass
    return json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
    )


def dump(
    payload: Any,
    fp: IO[str],
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> None:
    txt = dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
    )
    fp.write(txt)
