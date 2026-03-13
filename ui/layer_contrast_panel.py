# ui/layer_contrast_panel.py
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QSlider, QFrame,
    QLineEdit, QColorDialog,
)

_LAYER_NAMES = ["Background", "Monolayer", "Bilayer", "Trilayer"]

# Default overlay colours BGR — protanopia-friendly (blue / orange / purple)
_DEFAULT_OVERLAY_BGR = {
    "Monolayer": (187, 119,   0),   # #0077bb  blue
    "Bilayer":   ( 51, 119, 238),   # #ee7733  orange
    "Trilayer":  (204,  68, 170),   # #aa44cc  purple
}


def _mode_color_bgr(image, subsample=4):
    """Return the most common colour in image (BGR) using quantised binning."""
    pixels = image[::subsample, ::subsample]           # subsample for speed
    q = (pixels.astype(np.uint32) >> 3)               # quantise to 32 levels/channel
    packed = (q[:,:,0] << 16) | (q[:,:,1] << 8) | q[:,:,2]
    vals, counts = np.unique(packed.ravel(), return_counts=True)
    mode = vals[np.argmax(counts)]
    b = int(((mode >> 16) & 0x1F) * 8 + 4)
    g = int(((mode >>  8) & 0x1F) * 8 + 4)
    r = int(( mode        & 0x1F) * 8 + 4)
    return (b, g, r)                                   # BGR


def _bgr_to_qcolor(bgr):
    b, g, r = bgr
    return QColor(r, g, b)


def _qcolor_to_bgr(qc):
    return (qc.blue(), qc.green(), qc.red())


def _css(r, g, b):
    return f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"


class LayerContrastPanel(QWidget):
    def __init__(self, preview, parent=None):
        super().__init__(parent)
        self.preview = preview
        self.setWindowTitle("Layer Contrast")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # Effective target sample colour (r, g, b) for each layer, or None
        self._colors = {name: None for name in _LAYER_NAMES}
        # Overlay draw colour (BGR) — user-adjustable
        self._overlay_bgr = dict(_DEFAULT_OVERLAY_BGR)
        self._picking = None

        preview.color_sampled.connect(self._on_color_sampled)
        preview.layer_panel = self

        layout = QVBoxLayout()
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)

        self._swatches        = {}
        self._contrast_edits  = {}   # QLineEdit per non-background layer
        self._overlay_btns    = {}   # QPushButton coloured square per non-bg layer
        self._overlay_checks  = {}
        self._auto_bg_check   = None

        for row, name in enumerate(_LAYER_NAMES):
            grid.addWidget(QLabel(name + ":"), row, 0)

            # Sample colour swatch
            swatch = QLabel()
            swatch.setFixedSize(36, 20)
            swatch.setFrameShape(QFrame.Box)
            swatch.setStyleSheet("background-color: #404040; border: 1px solid #888;")
            self._swatches[name] = swatch
            grid.addWidget(swatch, row, 1)

            # Eyedropper pick button
            pick_btn = QPushButton("Pick")
            pick_btn.setFixedWidth(42)
            pick_btn.clicked.connect(lambda checked, n=name: self._start_pick(n))
            grid.addWidget(pick_btn, row, 2)

            if name == "Background":
                auto_chk = QCheckBox("Auto")
                auto_chk.setChecked(True)
                auto_chk.setToolTip("Automatically detect background as the most common colour in the image")
                auto_chk.stateChanged.connect(lambda s, b=pick_btn: b.setEnabled(s != Qt.Checked))
                pick_btn.setEnabled(False)   # disabled while Auto is on
                self._auto_bg_check = auto_chk
                grid.addWidget(auto_chk, row, 3)

            if name != "Background":
                # Editable contrast field
                edit = QLineEdit()
                edit.setPlaceholderText("e.g. -3.5")
                edit.setFixedWidth(70)
                edit.setToolTip("Optical contrast ΔI/I % vs background.\n"
                                "Edit to set target colour from background + this contrast.")
                edit.editingFinished.connect(lambda n=name: self._on_contrast_edited(n))
                self._contrast_edits[name] = edit
                grid.addWidget(edit, row, 3)

                # Overlay colour picker button
                ov_btn = QPushButton()
                ov_btn.setFixedSize(28, 20)
                b, g, r = self._overlay_bgr[name]
                ov_btn.setStyleSheet(_css(r, g, b))
                ov_btn.setToolTip("Click to change overlay colour")
                ov_btn.clicked.connect(lambda checked, n=name: self._pick_overlay_color(n))
                self._overlay_btns[name] = ov_btn
                grid.addWidget(ov_btn, row, 4)

                # Show/hide overlay toggle
                chk = QCheckBox("Show")
                chk.setChecked(True)
                self._overlay_checks[name] = chk
                grid.addWidget(chk, row, 5)

        layout.addLayout(grid)

        # Tolerance slider
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance:"))
        self._tol_slider = QSlider(Qt.Horizontal)
        self._tol_slider.setRange(1, 80)
        self._tol_slider.setValue(20)
        self._tol_label = QLabel("20")
        self._tol_slider.valueChanged.connect(lambda v: self._tol_label.setText(str(v)))
        tol_row.addWidget(self._tol_slider)
        tol_row.addWidget(self._tol_label)
        layout.addLayout(tol_row)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Sample colour picking (eyedropper)
    # ------------------------------------------------------------------

    def _start_pick(self, name):
        self._picking = name
        self.preview.start_color_pick()

    def _on_color_sampled(self, r, g, b):
        if self._picking is None:
            return
        name = self._picking
        self._picking = None
        self._set_layer_color(name, r, g, b)

    def _set_layer_color(self, name, r, g, b):
        """Store colour, update swatch, and refresh contrast edit if possible."""
        self._colors[name] = (r, g, b)
        self._swatches[name].setStyleSheet(_css(r, g, b))
        if name != "Background":
            self._update_contrast_edit(name)

    # ------------------------------------------------------------------
    # Contrast text input
    # ------------------------------------------------------------------

    def _update_contrast_edit(self, name):
        """Recompute contrast from current colours and push to the text box."""
        edit = self._contrast_edits.get(name)
        if edit is None:
            return
        bg  = self._colors.get("Background")
        col = self._colors.get(name)
        if bg is None or col is None:
            return
        bg_i  = sum(bg)  / 3.0
        col_i = sum(col) / 3.0
        if bg_i > 0:
            contrast = (col_i - bg_i) / bg_i * 100.0
            edit.setText(f"{contrast:.2f}")

    def _on_contrast_edited(self, name):
        """User typed a contrast value — derive target colour from background."""
        edit = self._contrast_edits[name]
        text = edit.text().strip().rstrip('%')
        try:
            contrast = float(text)
        except ValueError:
            return
        bg = self._colors.get("Background")
        if bg is None:
            return  # need background to compute absolute colour
        factor = 1.0 + contrast / 100.0
        r = int(max(0, min(255, bg[0] * factor)))
        g = int(max(0, min(255, bg[1] * factor)))
        b = int(max(0, min(255, bg[2] * factor)))
        self._colors[name] = (r, g, b)
        self._swatches[name].setStyleSheet(_css(r, g, b))

    # ------------------------------------------------------------------
    # Overlay colour picker
    # ------------------------------------------------------------------

    def _pick_overlay_color(self, name):
        initial = _bgr_to_qcolor(self._overlay_bgr[name])
        qc = QColorDialog.getColor(initial, self, f"Overlay colour — {name}")
        if not qc.isValid():
            return
        self._overlay_bgr[name] = _qcolor_to_bgr(qc)
        r, g, b = qc.red(), qc.green(), qc.blue()
        self._overlay_btns[name].setStyleSheet(_css(r, g, b))

    # ------------------------------------------------------------------
    # Overlay (called each frame from preview.update_frame)
    # ------------------------------------------------------------------

    def apply_overlay(self, display):
        """
        Return display with thin contour outlines for each active layer.
        Operates on the already-resized display image for performance.
        """
        # Auto background detection
        if self._auto_bg_check is not None and self._auto_bg_check.isChecked():
            b, g, r = _mode_color_bgr(display)
            if self._colors.get("Background") != (r, g, b):
                self._colors["Background"] = (r, g, b)
                self._swatches["Background"].setStyleSheet(_css(r, g, b))
                for name in self._contrast_edits:
                    self._update_contrast_edit(name)

        tol = self._tol_slider.value()
        any_active = any(
            chk.isChecked() and self._colors.get(name) is not None
            for name, chk in self._overlay_checks.items()
        )
        if not any_active:
            return display

        result = display.copy()
        display_i16 = display.astype(np.int16)
        kernel = np.ones((3, 3), np.uint8)

        for name, chk in self._overlay_checks.items():
            if not chk.isChecked():
                continue
            col = self._colors.get(name)
            if col is None:
                continue
            r, g, b = col
            target = np.array([b, g, r], dtype=np.int16)           # BGR
            mask = (np.max(np.abs(display_i16 - target), axis=2) <= tol).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckle
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, self._overlay_bgr[name], 1)

        return result
