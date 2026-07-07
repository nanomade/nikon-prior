#!/bin/bash
# ── Nikon-Prior Microscope Control ── launch wrapper
# Activates the Python venv and starts the application.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# ── Locate the Python venv ──────────────────────────────────────────────────────
# Override with NIKON_PRIOR_VENV=/path/to/venv; otherwise try common locations.
# In --dev installs the project itself lives at $HOME/nikon-prior, so the venv is
# kept separately (default $HOME/venv) rather than inside the project dir.
VENV_DIR="${NIKON_PRIOR_VENV:-}"
if [ -z "$VENV_DIR" ]; then
    for cand in "$HOME/venv" "$PROJECT_DIR/venv" "$PROJECT_DIR/.venv" \
                "$HOME/nikon-prior-venv" "/opt/nikon-prior-venv"; do
        if [ -f "$cand/bin/activate" ]; then VENV_DIR="$cand"; break; fi
    done
fi

# ── Check venv exists ──────────────────────────────────────────────────────────
if [ -z "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    zenity --error \
        --title="Nikon-Prior: venv not found" \
        --text="Python virtual environment not found.\n\nLooked in: \$HOME/venv, $PROJECT_DIR/venv, $PROJECT_DIR/.venv\n\nCreate one with:\n  python3 -m venv ~/venv\n  source ~/venv/bin/activate\n  pip install -r $PROJECT_DIR/requirements.txt\n\nOr set NIKON_PRIOR_VENV=/path/to/venv" \
        2>/dev/null \
    || echo "ERROR: venv not found (set NIKON_PRIOR_VENV or create ~/venv)" >&2
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Environment ───────────────────────────────────────────────────────────────
export QT_QPA_PLATFORM=xcb   # suppress Wayland warnings under XWayland
cd "$PROJECT_DIR"

# ── Launch ────────────────────────────────────────────────────────────────────
exec python main.py "$@"
