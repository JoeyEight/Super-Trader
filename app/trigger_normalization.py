from __future__ import annotations

from typing import Any


def _s(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def normalize_exit_trigger(*texts: Any) -> str:
    txt = " ".join(_s(text) for text in texts if _s(text)).upper()
    if not txt:
        return "Unknown"
    if "MANUAL" in txt:
        return "Manual"
    if "TRAIL" in txt:
        return "Trailing"
    if "STALE" in txt or "MISALIGN" in txt or "ALIGNMENT" in txt:
        return "Stale Alignment"
    if "BLOCK" in txt:
        return "Blocked"
    if ("AI" in txt) and ("EXIT" in txt or "CLOSE" in txt):
        return "AI Exit"
    if ("RISK" in txt) or ("STOP" in txt):
        return "Risk Cut"
    if ("TAKE" in txt) or ("PROFIT" in txt):
        return "Take Profit"
    return "Unknown"
