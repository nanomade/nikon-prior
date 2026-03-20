#!/bin/bash
# ── Nikon-Prior Help ── opens the quick-start reference in the default browser.
# Shift+click (or run with --manual) to open the full manual instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")/docs"

if [[ "${1:-}" == "--manual" ]]; then
    TARGET="$DOCS_DIR/manual.html"
else
    TARGET="$DOCS_DIR/quickstart.html"
fi

if [ ! -f "$TARGET" ]; then
    zenity --error --title="Help file missing" \
        --text="Could not find:\n  $TARGET" 2>/dev/null \
    || echo "ERROR: $TARGET not found" >&2
    exit 1
fi

xdg-open "$TARGET"
