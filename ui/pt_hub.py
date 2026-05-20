from __future__ import annotations

import os
import sys as _sys

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)

from ui.hub import main as _hub_main

# Compatibility alias: keep legacy import path (`ui.pt_hub`) working while the
# implementation now lives under `ui.hub.main`.
_sys.modules[__name__] = _hub_main

if __name__ == "__main__":
    app = _hub_main.PowerTraderHub()
    app.mainloop()
