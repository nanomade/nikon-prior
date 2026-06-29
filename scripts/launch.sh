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

# Cap native math thread-pools — unbounded BLAS/OpenMP pools spawn ~170 native
# threads, amplifying heap pressure and instability for no benefit here.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

cd "$PROJECT_DIR"

# ── Launch ────────────────────────────────────────────────────────────────────
# Keep a rolling log so GUI-launched crashes aren't silent.
LOG_DIR="$PROJECT_DIR/crash_logs"
mkdir -p "$LOG_DIR"
exec python main.py "$@" >>"$LOG_DIR/launcher.log" 2>&1
