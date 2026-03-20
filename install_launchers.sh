#!/bin/bash
# ── Nikon-Prior launcher installer ──────────────────────────────────────────
# Installs desktop icons so the app appears in the application menu and
# optionally on the desktop.
#
# Usage:
#   ./install_launchers.sh            # install for current user
#   ./install_launchers.sh --desktop  # also add icons to Desktop
#   ./install_launchers.sh --remove   # uninstall

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

# ─── uninstall mode ─────────────────────────────────────────────────────────

if [[ "${1:-}" == "--remove" ]]; then
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
green "═══ Nikon-Prior Launcher Installer ═══"
echo ""
info "Project dir : $PROJECT_DIR"
info "Install dir : $INSTALL_DIR"
info "App dir     : $APP_DIR"

require_cmd xdg-open
require_cmd update-desktop-database

# ─── system-wide copy (needs sudo) ─────────────────────────────────────────

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

# ─── icons ──────────────────────────────────────────────────────────────────

yellow "\nInstalling icons…"
mkdir -p "$ICON_DIR"
cp "$PROJECT_DIR/assets/nikon-prior.svg"      "$ICON_DIR/nikon-prior.svg"
cp "$PROJECT_DIR/assets/nikon-prior-help.svg" "$ICON_DIR/nikon-prior-help.svg"
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
info "Icons installed to $ICON_DIR"

# ─── .desktop files ─────────────────────────────────────────────────────────

yellow "\nInstalling application launchers…"
mkdir -p "$APP_DIR"

# Rewrite Exec and Icon paths to use the installed location
sed \
    -e "s|Exec=.*|Exec=$INSTALL_DIR/scripts/launch.sh|" \
    -e "s|Icon=.*|Icon=$ICON_DIR/nikon-prior.svg|" \
    "$PROJECT_DIR/nikon-prior.desktop" > "$APP_DIR/nikon-prior.desktop"

sed \
    -e "s|Exec=.*|Exec=$INSTALL_DIR/scripts/open_help.sh|" \
    -e "s|Icon=.*|Icon=$ICON_DIR/nikon-prior-help.svg|" \
    "$PROJECT_DIR/nikon-prior-help.desktop" > "$APP_DIR/nikon-prior-help.desktop"

chmod +x "$APP_DIR/nikon-prior.desktop" "$APP_DIR/nikon-prior-help.desktop"
update-desktop-database "$APP_DIR"
info "Launchers installed to $APP_DIR"

# ─── optional desktop shortcuts ─────────────────────────────────────────────

if [[ "${1:-}" == "--desktop" ]]; then
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
if [[ "${1:-}" == "--desktop" ]]; then
info "• Two icons have been placed on your Desktop"
fi
info "• The quick-reference guide opens separately from the help icon"
info ""
info "To uninstall: ./install_launchers.sh --remove"
echo ""
