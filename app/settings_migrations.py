from __future__ import annotations

from typing import Any, Dict, List, Tuple

CURRENT_SETTINGS_VERSION = 3


def _migrate_v1_to_v2(data: Dict[str, Any], notes: List[str]) -> None:
    # Moved legacy root script paths into package layout.
    mapping = {
        "pt_thinker.py": "engines/pt_thinker.py",
        "pt_trader.py": "engines/pt_trader.py",
        "pt_trainer.py": "engines/pt_trainer.py",
        "pt_markets.py": "runtime/pt_markets.py",
        "pt_autopilot.py": "runtime/pt_autopilot.py",
    }
    for key in ("script_neural_runner2", "script_trader", "script_neural_trainer", "script_markets_runner", "script_autopilot"):
        cur = str(data.get(key, "") or "").strip().replace("\\", "/")
        if cur in mapping:
            data[key] = mapping[cur]
            notes.append(f"{key}: migrated {cur} -> {mapping[cur]}")


def _migrate_v2_to_v3(data: Dict[str, Any], notes: List[str]) -> None:
    # Normalize optional boolean safety guards if missing.
    if "paper_only_unless_checklist_green" not in data:
        data["paper_only_unless_checklist_green"] = True
        notes.append("paper_only_unless_checklist_green: defaulted true")
    if "key_rotation_warn_days" not in data:
        data["key_rotation_warn_days"] = 0
        notes.append("key_rotation_warn_days: defaulted 0 (endpoint-managed key lifecycle)")


def migrate_settings(raw: Dict[str, Any] | None) -> Tuple[Dict[str, Any], List[str], int, int]:
    data = dict(raw or {})
    notes: List[str] = []
    try:
        start_version = int(float(data.get("settings_schema_version", 1) or 1))
    except Exception:
        start_version = 1
    v = max(1, start_version)
    if v < 2:
        _migrate_v1_to_v2(data, notes)
        v = 2
    if v < 3:
        _migrate_v2_to_v3(data, notes)
        v = 3
    data["settings_schema_version"] = int(CURRENT_SETTINGS_VERSION)
    if notes:
        data["settings_upgrade_notes"] = notes[-20:]
    return data, notes, int(start_version), int(CURRENT_SETTINGS_VERSION)
