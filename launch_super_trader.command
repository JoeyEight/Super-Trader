#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR" || exit 1

# Ensure runtime directories exist.
mkdir -p \
  "$PROJECT_DIR/hub_data/logs" \
  "$PROJECT_DIR/hub_data/.mplconfig"

# Resolve settings and runtime paths for hub + children.
export POWERTRADER_PROJECT_DIR="$PROJECT_DIR"
export POWERTRADER_GUI_SETTINGS="$PROJECT_DIR/gui_settings.json"
export POWERTRADER_HUB_DIR="$PROJECT_DIR/hub_data"
export MPLCONFIGDIR="$PROJECT_DIR/hub_data/.mplconfig"

# Keep package imports stable when launching from Finder/Terminal.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="$PROJECT_DIR/venv"
PY_BIN="$VENV_DIR/bin/python3"

# Keep startup deterministic: always use project venv.
if [[ ! -x "$PY_BIN" ]]; then
  echo "[launch] venv not found; creating at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# If core deps are missing/corrupt, bootstrap from requirements.
if ! "$PY_BIN" -c "import matplotlib" >/dev/null 2>&1; then
  echo "[launch] installing dependencies from requirements.txt"
  "$PY_BIN" -m pip install --upgrade pip setuptools wheel
  "$PY_BIN" -m pip install -r "$PROJECT_DIR/requirements.txt"
fi

if command -v caffeinate >/dev/null 2>&1; then
  # Keep the trading runtime active while the app is open:
  # -i: prevent idle system sleep
  # -s: prevent full system sleep while on AC power
  # -m: prevent disk sleep
  # -d: prevent display sleep side-effects from pausing active visual workflows
  exec caffeinate -dims "$PY_BIN" -m ui.pt_hub
fi

exec "$PY_BIN" -m ui.pt_hub
