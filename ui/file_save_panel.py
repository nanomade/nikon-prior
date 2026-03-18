# ui/file_save_panel.py
"""File-saving panel: working directory, sample name, metadata in filename.

Provides a richer save interface than the basic buttons in controls.py:
  • Working directory selection (persists across sessions via JSON sidecar)
  • Sample name field
  • Selectable metadata tags: timestamp, magnification, exposure, X/Y position
  • Live filename preview
  • Save View (PNG, saves the rendered preview with overlays)
  • Capture Frame (PNG, saves raw camera frame without overlays)
"""

import datetime
import json
import os

import cv2
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

_SETTINGS_FILE = "file_save_settings.json"


class FileSavePanel(QWidget):
    def __init__(self, preview, motor_manager, controls):
        super().__init__()
        self.setWindowTitle("File Save")
        self.preview = preview
        self.motor_manager = motor_manager
        self.controls = controls

        self._settings = self._load_settings()

        layout = QVBoxLayout(self)

        # ── Working directory ──────────────────────────────────────────────
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout()
        self._dir_label = QLabel()
        self._dir_label.setWordWrap(True)
        dir_layout.addWidget(self._dir_label, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # ── Sample name ────────────────────────────────────────────────────
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Sample name:"))
        self._name_edit = QLineEdit(self._settings.get("sample_name", "sample"))
        self._name_edit.setPlaceholderText("e.g. wafer_A1")
        self._name_edit.textChanged.connect(self._update_preview)
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # ── Metadata checkboxes ────────────────────────────────────────────
        meta_group = QGroupBox("Include in filename")
        meta_grid = QGridLayout()

        def _chk(label, key, row, col):
            cb = QCheckBox(label)
            cb.setChecked(self._settings.get(key, True))
            cb.stateChanged.connect(self._update_preview)
            meta_grid.addWidget(cb, row, col)
            return cb

        self._chk_timestamp = _chk("Timestamp",    "chk_timestamp", 0, 0)
        self._chk_mag       = _chk("Magnification","chk_mag",       0, 1)
        self._chk_exposure  = _chk("Exposure (ms)","chk_exposure",  1, 0)
        self._chk_xy        = _chk("X / Y pos",    "chk_xy",        1, 1)
        meta_group.setLayout(meta_grid)
        layout.addWidget(meta_group)

        # ── Filename preview ───────────────────────────────────────────────
        self._preview_label = QLabel()
        self._preview_label.setWordWrap(True)
        self._preview_label.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self._preview_label)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        save_view_btn = QPushButton("Save View (PNG)")
        save_view_btn.setToolTip("Save the rendered preview image (with overlays, scale bar, etc.)")
        save_view_btn.clicked.connect(self._save_view)
        btn_row.addWidget(save_view_btn)

        capture_btn = QPushButton("Capture Frame (PNG)")
        capture_btn.setToolTip("Save the raw camera frame (no overlays)")
        capture_btn.clicked.connect(self._capture_frame)
        btn_row.addWidget(capture_btn)
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        layout.addStretch()

        # Initialise directory display
        wd = self._settings.get("working_dir", os.path.expanduser("~"))
        if not os.path.isdir(wd):
            wd = os.path.expanduser("~")
        self._settings["working_dir"] = wd
        self._dir_label.setText(wd)
        self._update_preview()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE) as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_settings(self):
        self._settings.update({
            "sample_name": self._name_edit.text(),
            "chk_timestamp": self._chk_timestamp.isChecked(),
            "chk_mag":       self._chk_mag.isChecked(),
            "chk_exposure":  self._chk_exposure.isChecked(),
            "chk_xy":        self._chk_xy.isChecked(),
        })
        try:
            with open(_SETTINGS_FILE, "w") as fh:
                json.dump(self._settings, fh, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Directory browse
    # ------------------------------------------------------------------

    def _browse_dir(self):
        current = self._settings.get("working_dir", os.path.expanduser("~"))
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Directory", current)
        if chosen:
            self._settings["working_dir"] = chosen
            self._dir_label.setText(chosen)
            self._save_settings()
            self._update_preview()

    # ------------------------------------------------------------------
    # Filename construction
    # ------------------------------------------------------------------

    def _build_filename(self, ext="png"):
        parts = [self._name_edit.text().strip() or "capture"]

        if self._chk_timestamp.isChecked():
            parts.append(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        if self._chk_mag.isChecked():
            parts.append(getattr(self.preview, "magnification", "?x").replace("/", "-"))

        if self._chk_exposure.isChecked():
            try:
                exp_ms = self.controls.get_exposure_ms()
                parts.append(f"{exp_ms:.0f}ms")
            except Exception:
                pass

        if self._chk_xy.isChecked():
            try:
                x = self.motor_manager.get_position_units("X")
                y = self.motor_manager.get_position_units("Y")
                if x is not None and y is not None:
                    parts.append(f"X{x:+.3f}")
                    parts.append(f"Y{y:+.3f}")
            except Exception:
                pass

        return "_".join(parts) + f".{ext}"

    def _update_preview(self):
        name = self._build_filename()
        wd = self._settings.get("working_dir", "~")
        self._preview_label.setText(f"→ {os.path.join(wd, name)}")

    # ------------------------------------------------------------------
    # Save actions
    # ------------------------------------------------------------------

    def _full_path(self, filename):
        wd = self._settings.get("working_dir", os.path.expanduser("~"))
        os.makedirs(wd, exist_ok=True)
        return os.path.join(wd, filename)

    def _save_view(self):
        self._save_settings()
        filename = self._build_filename("png")
        path = self._full_path(filename)
        pixmap = self.preview.image_label.pixmap()
        if pixmap:
            pixmap.save(path, "PNG")
            self._status_label.setText(f"Saved: {filename}")
        else:
            self._status_label.setText("No frame to save.")

    def _capture_frame(self):
        self._save_settings()
        filename = self._build_filename("png")
        path = self._full_path(filename)
        frame = self.preview.get_frame()
        if frame is not None:
            cv2.imwrite(path, frame)
            self._status_label.setText(f"Captured: {filename}")
        else:
            self._status_label.setText("No frame available.")
