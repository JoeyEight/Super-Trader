from __future__ import annotations

import io
import json

from app.json_codec import dump, dumps, has_orjson, load, loads


def test_json_codec_roundtrip_text_and_file() -> None:
    payload = {"a": 1, "b": [1, 2, 3], "nested": {"x": True}}
    txt = dumps(payload, indent=2)
    parsed = loads(txt)
    assert isinstance(parsed, dict)
    assert parsed == payload

    buf = io.StringIO()
    dump(payload, buf, indent=2)
    buf.seek(0)
    parsed_file = load(buf)
    assert isinstance(parsed_file, dict)
    assert parsed_file == payload


def test_json_codec_loads_default_on_invalid() -> None:
    default = {"ok": False}
    assert loads("not-json", default=default) == default


def test_json_codec_loads_raises_without_default() -> None:
    try:
        loads("not-json")
    except Exception as exc:
        assert isinstance(exc, Exception)
    else:
        raise AssertionError("loads() should raise for invalid JSON without default")


def test_json_codec_ensure_ascii_respected() -> None:
    payload = {"status": "🆗"}
    txt = dumps(payload, ensure_ascii=True)
    assert "\\ud83c" in txt.lower()
    assert json.loads(txt) == payload


def test_json_codec_has_orjson_boolean() -> None:
    assert isinstance(has_orjson(), bool)
