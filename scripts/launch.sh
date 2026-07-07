#!/bin/bash
# ── Nikon-Prior Microscope Control ── launch wrapper
# Activates the Python venv and starts the application.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HOME/nikon-prior"

# ── Check venv exists ──────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    zenity --error \
        --title="Nikon-Prior: venv not found" \
        --text="Python virtual environment not found at:\n  $VENV_DIR\n\nPlease create it with:\n  python3 -m venv ~/nikon-prior\n  source ~/nikon-prior/bin/activate\n  pip install -r $PROJECT_DIR/requirements.txt" \
        2>/dev/null \
    || echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Environment ───────────────────────────────────────────────────────────────
export QT_QPA_PLATFORM=xcb   # suppress Wayland warnings under XWayland
cd "$PROJECT_DIR"

# ── Launch ────────────────────────────────────────────────────────────────────
exec python main.py "$@"
