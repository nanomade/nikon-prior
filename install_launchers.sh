#!/bin/bash
# ── Nikon-Prior launcher installer ──────────────────────────────────────────
# Installs desktop icons so the app appears in the application menu and
# optionally on the desktop.
#
# Usage:
#   ./install_launchers.sh            # system-wide install: /opt copy + menu
#                                      #   entries for ALL users (needs sudo)
#   ./install_launchers.sh --dev      # point launchers at THIS working copy —
#                                      #   current user only, no sudo,
#                                      #   no reinstall after edits
#   ./install_launchers.sh --desktop  # also add icons to the Desktop
#   ./install_launchers.sh --remove   # uninstall (user + system locations)
#
# Flags combine, e.g.  ./install_launchers.sh --dev --desktop
#
# NOTE on --dev: the launchers run code straight from this working copy and the
# menu entry is installed for the *current* user only. Run it as the user whose
# home holds the checkout (and who will launch the app); it needs no sudo.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/nikon-prior"

# Per-user locations (dev mode + --desktop)
USER_APP_DIR="$HOME/.local/share/applications"
USER_ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="$HOME/Desktop"

# System-wide locations (default mode) — menu entries visible to every user
SYS_APP_DIR="/usr/share/applications"
SYS_ICON_DIR="/usr/share/icons/hicolor/scalable/apps"

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
    rm -f "$USER_APP_DIR/nikon-prior.desktop" "$USER_APP_DIR/nikon-prior-help.desktop"
    rm -f "$USER_ICON_DIR/nikon-prior.svg" "$USER_ICON_DIR/nikon-prior-help.svg"
    rm -f "$DESKTOP_DIR/nikon-prior.desktop" "$DESKTOP_DIR/nikon-prior-help.desktop"
    if [ -f "$SYS_APP_DIR/nikon-prior.desktop" ] || [ -f "$SYS_ICON_DIR/nikon-prior.svg" ]; then
        yellow "Removing system-wide launchers (requires sudo)…"
        sudo rm -f "$SYS_APP_DIR/nikon-prior.desktop" "$SYS_APP_DIR/nikon-prior-help.desktop"
        sudo rm -f "$SYS_ICON_DIR/nikon-prior.svg" "$SYS_ICON_DIR/nikon-prior-help.svg"
        sudo update-desktop-database "$SYS_APP_DIR" 2>/dev/null || true
    fi
    update-desktop-database "$USER_APP_DIR" 2>/dev/null || true
    gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    green "Uninstalled."
    exit 0
fi

# ─── preflight ──────────────────────────────────────────────────────────────

echo ""
if [[ $DEV -eq 1 ]]; then
    green "═══ Nikon-Prior Launcher Installer (dev mode) ═══"
else
    green "═══ Nikon-Prior Launcher Installer (system-wide) ═══"
fi
echo ""

require_cmd xdg-open
require_cmd update-desktop-database

# APP_ROOT is the directory the launchers exec from; APP_DIR/ICON_DIR are where
# the .desktop entries and icons are installed.
if [[ $DEV -eq 1 ]]; then
    APP_ROOT="$PROJECT_DIR"
    APP_DIR="$USER_APP_DIR"
    ICON_DIR="$USER_ICON_DIR"
    info "Mode        : dev (current user; launchers point at the working copy)"
    info "Project dir : $PROJECT_DIR"
    info "App dir     : $APP_DIR"
else
    APP_ROOT="$INSTALL_DIR"
    APP_DIR="$SYS_APP_DIR"
    ICON_DIR="$SYS_ICON_DIR"
    info "Mode        : system-wide (all users)"
    info "Project dir : $PROJECT_DIR"
    info "Install dir : $INSTALL_DIR"
    info "App dir     : $APP_DIR"
fi

ICON_MAIN="$ICON_DIR/nikon-prior.svg"
ICON_HELP="$ICON_DIR/nikon-prior-help.svg"
if [[ $DEV -eq 1 ]]; then
    # Dev mode references the icons in place — no copy step.
    ICON_MAIN="$PROJECT_DIR/assets/nikon-prior.svg"
    ICON_HELP="$PROJECT_DIR/assets/nikon-prior-help.svg"
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

    yellow "\nInstalling icons system-wide…"
    sudo mkdir -p "$ICON_DIR"
    sudo cp "$PROJECT_DIR/assets/nikon-prior.svg"      "$ICON_DIR/nikon-prior.svg"
    sudo cp "$PROJECT_DIR/assets/nikon-prior-help.svg" "$ICON_DIR/nikon-prior-help.svg"
    sudo gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
    info "Icons installed to $ICON_DIR"
else
    yellow "\nDev mode: skipping /opt copy and icon install."
    info "Launchers will run code directly from $PROJECT_DIR."
    info "Icons are referenced in place from $PROJECT_DIR/assets."
fi

# ─── .desktop files ─────────────────────────────────────────────────────────

yellow "\nInstalling application launchers…"

# Rewrite Exec and Icon to the chosen location. The /bin/bash wrapper runs the
# target scripts regardless of their executable bit.
TMPDIR_DESKTOP="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_DESKTOP"' EXIT

sed \
    -e "s|Exec=.*|Exec=/bin/bash $APP_ROOT/scripts/launch.sh|" \
    -e "s|Icon=.*|Icon=$ICON_MAIN|" \
    "$PROJECT_DIR/nikon-prior.desktop" > "$TMPDIR_DESKTOP/nikon-prior.desktop"

sed \
    -e "s|Exec=.*|Exec=/bin/bash $APP_ROOT/scripts/open_help.sh|" \
    -e "s|Icon=.*|Icon=$ICON_HELP|" \
    "$PROJECT_DIR/nikon-prior-help.desktop" > "$TMPDIR_DESKTOP/nikon-prior-help.desktop"

if [[ $DEV -eq 1 ]]; then
    mkdir -p "$APP_DIR"
    install -m 755 "$TMPDIR_DESKTOP/nikon-prior.desktop"      "$APP_DIR/"
    install -m 755 "$TMPDIR_DESKTOP/nikon-prior-help.desktop" "$APP_DIR/"
    update-desktop-database "$APP_DIR" 2>/dev/null || true
else
    sudo install -m 644 "$TMPDIR_DESKTOP/nikon-prior.desktop"      "$APP_DIR/"
    sudo install -m 644 "$TMPDIR_DESKTOP/nikon-prior-help.desktop" "$APP_DIR/"
    sudo update-desktop-database "$APP_DIR" 2>/dev/null || true
fi
info "Launchers installed to $APP_DIR"

# ─── optional desktop shortcuts (current user) ──────────────────────────────

if [[ $DESKTOP -eq 1 ]]; then
    yellow "\nAdding Desktop shortcuts…"
    if [ -d "$DESKTOP_DIR" ]; then
        cp "$TMPDIR_DESKTOP/nikon-prior.desktop"      "$DESKTOP_DIR/"
        cp "$TMPDIR_DESKTOP/nikon-prior-help.desktop" "$DESKTOP_DIR/"
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
info "• Search for 'Nikon-Prior' in the application menu"
if [[ $DEV -eq 0 ]]; then
info "• Menu entries are system-wide: every user on this machine sees them"
info "• Each user needs a Python venv the launcher can find:"
info "    ~/venv, or the shared /opt/nikon-prior-venv (see launch.sh)"
fi
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
