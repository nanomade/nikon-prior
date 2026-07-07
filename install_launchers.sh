#!/bin/bash
# ── Nikon-Prior launcher installer ──────────────────────────────────────────
# Installs desktop icons so the app appears in the application menu and
# optionally on the desktop.
#
# Usage:
#   ./install_launchers.sh            # system-wide install to /opt (needs sudo)
#   ./install_launchers.sh --dev      # point launchers at THIS working copy —
#                                      #   no /opt copy, no reinstall after edits
#   ./install_launchers.sh --desktop  # also add icons to the Desktop
#   ./install_launchers.sh --remove   # uninstall
#
# Flags combine, e.g.  ./install_launchers.sh --dev --desktop
#
# NOTE on --dev: the launchers run code straight from this working copy and the
# menu entry is installed for the *current* user only. Run it as the user whose
# home holds the checkout (and who will launch the app); it needs no sudo.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/nikon-prior"
APP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="$HOME/Desktop"

# ─── helpers ────────────────────────────────────────────────────────────────

green()  { echo -e "\033[1;32m$*\033[0m"; }
yellow() { echo -e "\033[1;33m$*\033[0m"; }
red()    { echo -e "\033[1;31m$*\033[0m"; }
info()   { echo "  $*"; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { red "ERROR: '$1' not found — please install it."; exit 1; }; }

# ─── parse flags ─────────────────────────────────────────────────────────────

REMOVE=0; DESKTOP=0; DEV=0
for arg in "$@"; do
    case "$arg" in
        --remove)  REMOVE=1 ;;
        --desktop) DESKTOP=1 ;;
        --dev)     DEV=1 ;;
        -h|--help)
            echo "Usage: $0 [--dev] [--desktop] [--remove]"
            exit 0 ;;
        *) red "ERROR: unknown option '$arg' (try --help)"; exit 1 ;;
    esac
done

# ─── uninstall mode ─────────────────────────────────────────────────────────

if [[ $REMOVE -eq 1 ]]; then
    yellow "Removing Nikon-Prior launchers…"
    rm -f "$APP_DIR/nikon-prior.desktop" "$APP_DIR/nikon-prior-help.desktop"
    rm -f "$ICON_DIR/nikon-prior.svg" "$ICON_DIR/nikon-prior-help.svg"
    rm -f "$DESKTOP_DIR/nikon-prior.desktop" "$DESKTOP_DIR/nikon-prior-help.desktop"
    update-desktop-database "$APP_DIR" 2>/dev/null || true
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    green "Uninstalled."
    exit 0
fi

# ─── preflight ──────────────────────────────────────────────────────────────

echo ""
if [[ $DEV -eq 1 ]]; then
    green "═══ Nikon-Prior Launcher Installer (dev mode) ═══"
else
    green "═══ Nikon-Prior Launcher Installer ═══"
fi
echo ""

require_cmd xdg-open
require_cmd update-desktop-database

# APP_ROOT is the directory the launchers exec from; ICON_MAIN/ICON_HELP are the
# icon paths the .desktop entries reference.
if [[ $DEV -eq 1 ]]; then
    APP_ROOT="$PROJECT_DIR"
    ICON_MAIN="$PROJECT_DIR/assets/nikon-prior.svg"
    ICON_HELP="$PROJECT_DIR/assets/nikon-prior-help.svg"
    info "Mode        : dev (launchers point at the working copy)"
    info "Project dir : $PROJECT_DIR"
    info "App dir     : $APP_DIR"
else
    APP_ROOT="$INSTALL_DIR"
    ICON_MAIN="$ICON_DIR/nikon-prior.svg"
    ICON_HELP="$ICON_DIR/nikon-prior-help.svg"
    info "Mode        : system-wide"
    info "Project dir : $PROJECT_DIR"
    info "Install dir : $INSTALL_DIR"
    info "App dir     : $APP_DIR"
fi

# ─── system-wide copy + icon install (skipped in dev mode) ──────────────────

if [[ $DEV -eq 0 ]]; then
    require_cmd rsync

    yellow "\nCopying project to $INSTALL_DIR (requires sudo)…"
    sudo mkdir -p "$INSTALL_DIR"
    sudo rsync -a --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$PROJECT_DIR/" "$INSTALL_DIR/"
    sudo chmod -R a+rX "$INSTALL_DIR"
    sudo chmod a+x "$INSTALL_DIR/scripts/launch.sh" "$INSTALL_DIR/scripts/open_help.sh"
    info "Done."

    yellow "\nInstalling icons…"
    mkdir -p "$ICON_DIR"
    cp "$PROJECT_DIR/assets/nikon-prior.svg"      "$ICON_DIR/nikon-prior.svg"
    cp "$PROJECT_DIR/assets/nikon-prior-help.svg" "$ICON_DIR/nikon-prior-help.svg"
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    info "Icons installed to $ICON_DIR"
else
    yellow "\nDev mode: skipping /opt copy and icon install."
    info "Launchers will run code directly from $PROJECT_DIR."
    info "Icons are referenced in place from $PROJECT_DIR/assets."
fi

# ─── .desktop files ─────────────────────────────────────────────────────────

yellow "\nInstalling application launchers…"
mkdir -p "$APP_DIR"

# Rewrite Exec and Icon to the chosen location. The /bin/bash wrapper runs the
# target scripts regardless of their executable bit.
sed \
    -e "s|Exec=.*|Exec=/bin/bash $APP_ROOT/scripts/launch.sh|" \
    -e "s|Icon=.*|Icon=$ICON_MAIN|" \
    "$PROJECT_DIR/nikon-prior.desktop" > "$APP_DIR/nikon-prior.desktop"

sed \
    -e "s|Exec=.*|Exec=/bin/bash $APP_ROOT/scripts/open_help.sh|" \
    -e "s|Icon=.*|Icon=$ICON_HELP|" \
    "$PROJECT_DIR/nikon-prior-help.desktop" > "$APP_DIR/nikon-prior-help.desktop"

chmod +x "$APP_DIR/nikon-prior.desktop" "$APP_DIR/nikon-prior-help.desktop"
update-desktop-database "$APP_DIR"
info "Launchers installed to $APP_DIR"

# ─── optional desktop shortcuts ─────────────────────────────────────────────

if [[ $DESKTOP -eq 1 ]]; then
    yellow "\nAdding Desktop shortcuts…"
    if [ -d "$DESKTOP_DIR" ]; then
        cp "$APP_DIR/nikon-prior.desktop"      "$DESKTOP_DIR/"
        cp "$APP_DIR/nikon-prior-help.desktop" "$DESKTOP_DIR/"
        chmod +x "$DESKTOP_DIR/nikon-prior.desktop" "$DESKTOP_DIR/nikon-prior-help.desktop"
        # Mark as trusted (GNOME)
        gio set "$DESKTOP_DIR/nikon-prior.desktop"      metadata::trusted true 2>/dev/null || true
        gio set "$DESKTOP_DIR/nikon-prior-help.desktop" metadata::trusted true 2>/dev/null || true
        info "Desktop shortcuts added."
    else
        yellow "WARNING: No ~/Desktop directory found — skipping."
    fi
fi

# ─── done ───────────────────────────────────────────────────────────────────

echo ""
green "Installation complete!"
echo ""
info "• Search for 'Nikon-Prior' in your application menu"
if [[ $DESKTOP -eq 1 ]]; then
info "• Two icons have been placed on your Desktop"
fi
if [[ $DEV -eq 1 ]]; then
info "• Dev mode: launchers run directly from $PROJECT_DIR — no reinstall after edits"
fi
info "• The quick-reference guide opens separately from the help icon"
info ""
info "To uninstall: ./install_launchers.sh --remove"
echo ""
