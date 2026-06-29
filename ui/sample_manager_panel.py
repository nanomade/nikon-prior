# ui/sample_manager_panel.py
"""
Sample Manager panel — Phase 1
--------------------------------
Users → Samples → Flakes workflow.

Requires core/sample_data.py (pure-Python data model).
Motor manager + preview are optional references for "Add Flake" and "Navigate To".
"""

import math
import os

import core.sample_data as sd
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from ui import APP_DIR

_LOCKED_BG   = QBrush(QColor(255, 250, 200))   # soft amber for locked rows
_UNLOCKED_BG = QBrush(QColor(0, 0, 0, 0))      # transparent (default)

# Approximate half-FOV in mm per magnification — used as the position-guard
# threshold for "Capture Frame" / "Save View".  If the stage has moved further
# than this from the flake's recorded position the user is warned.
_HALF_FOV_MM = {
    '5x':   1.00,
    '10x':  0.50,
    '20x':  0.25,
    '50x':  0.10,
    '100x': 0.05,
}
_HALF_FOV_DEFAULT_MM = 0.50   # fallback when magnification is unknown

_IMAGE_THUMB_PX = 64   # thumbnail size in the image list

# Root directory for all user samples — sibling to this package
_USERS_ROOT = APP_DIR / 'users'


def _ro_item(text):
    """Non-editable, non-selectable QTableWidgetItem."""
    item = QTableWidgetItem(str(text) if text is not None else '')
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item


_ICON_DIR = APP_DIR / 'assets' / 'icons'


def _btn_icon(name: str) -> QIcon:
    """Load a bundled SVG button icon by stem; empty QIcon if the file is missing."""
    path = _ICON_DIR / f'{name}.svg'
    return QIcon(str(path)) if path.exists() else QIcon()


# Layer-count choices shared by the table column and the detail dropdown.
_LAYER_CHOICES = ['', '1', '2', '3', '4', '5', '6']


def _layer_str(lc) -> str:
    """Compact layer-count label for the table, e.g. 1 -> '1L', None -> ''."""
    return f"{int(lc)}L" if lc else ''


def _metric_str(v, fmt="{:.2f}") -> str:
    """Format a shape metric for the table; blank when missing (None)."""
    if v is None:
        return ''
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return ''


class SampleManagerPanel(QWidget):
    sample_changed       = pyqtSignal()  # emitted when sample is opened, created, or closed
    registration_changed = pyqtSignal()  # emitted when placement/registration state changes

    def __init__(self, preview=None, motor_manager=None, controls=None,
                 index_mark_panel=None, stage_controls=None,
                 edge_detection_panel=None):
        super().__init__()
        self.setWindowTitle("Sample Manager")
        self.preview          = preview
        self.mm               = motor_manager
        self.controls         = controls
        self.index_mark_panel = index_mark_panel
        self.stage_controls   = stage_controls
        self.edge_detection_panel = edge_detection_panel
        self._sample          = None   # currently loaded sample dict
        self._ref_marks       = [None, None]  # two reference marks for coord system
        self._transform_fresh = False  # True only after _compute_transform succeeds this session
        self._session_flake_ids: set = set()  # IDs of flakes added since sample was opened

        # Register extents callback so wafer extents are auto-saved to the sample
        if stage_controls is not None:
            stage_controls._on_extents_updated = self._save_extents_to_sample

        self._build_ui()
        self._refresh_users()
        self._set_locked(True)
        self._set_sample_locked(True)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── User row ──────────────────────────────────────────────────────
        user_group = QGroupBox("User")
        user_row   = QHBoxLayout()
        self._user_combo = QComboBox()
        self._user_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._user_combo.currentTextChanged.connect(self._on_user_changed)
        user_row.addWidget(self._user_combo)
        add_user_btn = QPushButton("Add…")
        add_user_btn.setFixedWidth(50)
        add_user_btn.clicked.connect(self._add_user)
        user_row.addWidget(add_user_btn)
        user_group.setLayout(user_row)
        root.addWidget(user_group)

        # ── Everything below is disabled until a user is chosen ──────────
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        # ── Sample row ────────────────────────────────────────────────────
        sample_group = QGroupBox("Sample")
        s_layout     = QVBoxLayout()

        # New sample — shown first
        new_row = QHBoxLayout()
        new_row.addWidget(QLabel("New:"))
        self._new_name_edit = QLineEdit()
        self._new_name_edit.setPlaceholderText("short name, e.g. MoS2_A1")
        new_row.addWidget(self._new_name_edit)
        self._new_btn = QPushButton("Create")
        self._new_btn.setFixedWidth(55)
        self._new_btn.clicked.connect(self._create_sample)
        new_row.addWidget(self._new_btn)
        s_layout.addLayout(new_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        s_layout.addWidget(sep)

        # Open existing
        open_row = QHBoxLayout()
        open_row.addWidget(QLabel("Open:"))
        self._sample_combo = QComboBox()
        self._sample_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        open_row.addWidget(self._sample_combo)
        self._open_btn = QPushButton("Open")
        self._open_btn.setFixedWidth(50)
        self._open_btn.clicked.connect(self._open_sample)
        open_row.addWidget(self._open_btn)
        s_layout.addLayout(open_row)

        # Active sample label + wafer extents button + close button
        active_row = QHBoxLayout()
        self._active_label = QLabel("No sample open.")
        self._active_label.setStyleSheet("font-weight: bold;")
        active_row.addWidget(self._active_label, stretch=1)
        self._extents_btn = QPushButton("Wafer Extents…")
        self._extents_btn.setFixedWidth(110)
        self._extents_btn.setToolTip("Map the wafer edge and save extents to the sample file.")
        self._extents_btn.clicked.connect(self._find_wafer_extents)
        active_row.addWidget(self._extents_btn)
        self._close_sample_btn = QPushButton("Close")
        self._close_sample_btn.setFixedWidth(50)
        self._close_sample_btn.setToolTip("Deselect the current sample.")
        self._close_sample_btn.clicked.connect(self._close_sample)
        active_row.addWidget(self._close_sample_btn)
        s_layout.addLayout(active_row)

        # Substrate selector + thickness measurement
        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Substrate:"))
        self._substrate_combo = QComboBox()
        self._substrate_combo.addItems(sd._SUBSTRATES)
        self._substrate_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._substrate_combo.currentTextChanged.connect(self._on_substrate_changed)
        sub_row.addWidget(self._substrate_combo)
        self._measure_thick_btn = QPushButton("Measure…")
        self._measure_thick_btn.setFixedWidth(72)
        self._measure_thick_btn.setToolTip(
            "Estimate oxide thickness from the current frame colour.\n"
            "Point the objective at a bare substrate region first.")
        self._measure_thick_btn.clicked.connect(self._measure_substrate_thickness)
        sub_row.addWidget(self._measure_thick_btn)
        s_layout.addLayout(sub_row)

        self._thickness_label = QLabel("")
        self._thickness_label.setStyleSheet(
            "color: #444; font-family: monospace; font-size: 10px;")
        self._thickness_label.setWordWrap(True)
        s_layout.addWidget(self._thickness_label)

        sample_group.setLayout(s_layout)
        content_layout.addWidget(sample_group)

        # ── Coordinate system ─────────────────────────────────────────────
        cs_group  = QWidget()
        cs_layout = QVBoxLayout()
        cs_layout.setSpacing(4)

        # Two reference marks
        _QUADRANT_ITEMS = ["—", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self._ref_labels     = []
        self._ref_set_btns   = []
        self._ref_quadrants  = []   # per-mark quadrant combo boxes
        for n in range(2):
            row = QHBoxLayout()
            lbl = QLabel(f"Mark {n+1}: not set")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            lbl.setStyleSheet("font-family: monospace;")
            self._ref_labels.append(lbl)
            q_combo = QComboBox()
            q_combo.addItems(_QUADRANT_ITEMS)
            q_combo.setFixedWidth(54)
            q_combo.setToolTip(
                "Quadrant dot position for this reference mark.\n"
                "'—' = no dot (centre / reference quadrant).\n"
                "Pre-filled from the stored mark; change if you moved\n"
                "to a different quadrant tile when re-registering.")
            self._ref_quadrants.append(q_combo)
            btn = QPushButton("Set from current mark")
            btn.setFixedWidth(170)
            btn.clicked.connect(lambda checked, idx=n: self._set_ref_mark(idx))
            self._ref_set_btns.append(btn)
            row.addWidget(lbl)
            row.addWidget(q_combo)
            row.addWidget(btn)
            cs_layout.addLayout(row)

        # Compute button + status
        comp_row = QHBoxLayout()
        self._compute_transform_btn = QPushButton("Compute local coordinate system")
        self._compute_transform_btn.setEnabled(False)
        self._compute_transform_btn.setToolTip(
            "Fit a rigid-body transform from the two reference marks.\n"
            "Both marks keep their absolute XX/YY grid coordinates.\n"
            "Mark 1 should be bottom-left of Mark 2 (smaller XX and YY).")
        self._compute_transform_btn.clicked.connect(self._compute_transform)
        comp_row.addWidget(self._compute_transform_btn)
        self._cs_status = QLabel("")
        self._cs_status.setWordWrap(True)
        comp_row.addWidget(self._cs_status, 1)
        cs_layout.addLayout(comp_row)

        cs_group.setLayout(cs_layout)
        self._cs_group = cs_group
        self.coord_system_widget = cs_group   # exposed for placement in app_shell

        # ── Splitter: flake table (top) + detail (bottom) ─────────────────
        splitter = QSplitter(Qt.Vertical)

        # Flake table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(4)

        self._flake_table = QTableWidget(0, 11)
        self._flake_table.setHorizontalHeaderLabels(
            ["ID", "Name", "Area (µm²)", "Layer", "Cleanliness", "Isolation",
             "Status", "Locked", "Circ.", "Aspect", "Solidity"])
        self._flake_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._flake_table.setEditTriggers(QAbstractItemView.DoubleClicked
                                          | QAbstractItemView.SelectedClicked)
        self._flake_table.setAlternatingRowColors(True)
        self._flake_table.horizontalHeader().setStretchLastSection(False)
        self._flake_table.setColumnWidth(0, 40)
        self._flake_table.setColumnWidth(1, 110)
        self._flake_table.setColumnWidth(2, 75)
        self._flake_table.setColumnWidth(3, 50)    # Layer
        self._flake_table.setColumnWidth(4, 90)
        self._flake_table.setColumnWidth(5, 80)
        self._flake_table.setColumnWidth(6, 80)
        self._flake_table.setColumnWidth(7, 55)    # Locked
        self._flake_table.setColumnWidth(8, 55)    # Circularity (read-only metric)
        self._flake_table.setColumnWidth(9, 55)    # Aspect ratio (read-only metric)
        self._flake_table.setColumnWidth(10, 60)   # Solidity (read-only metric)
        self._flake_table.currentCellChanged.connect(self._on_flake_selected)
        self._flake_table.itemChanged.connect(self._on_flake_item_changed)
        table_layout.addWidget(self._flake_table)

        # Flake action buttons — two rows
        # Row 1: primary capture-first workflow + navigation + delete
        btn_row1 = QHBoxLayout()
        _icon_sz = QSize(16, 16)
        self._capture_add_btn = QPushButton("Capture && Add Flake")
        self._capture_add_btn.setIcon(_btn_icon("book"))          # book
        self._capture_add_btn.setToolTip(
            "Grab a clean frame, record the current stage position,\n"
            "create a new flake entry, and link the image — all at once.")
        self._capture_add_btn.clicked.connect(self._capture_and_add)
        btn_row1.addWidget(self._capture_add_btn)
        self._nav_btn = QPushButton("Navigate To")
        self._nav_btn.setIcon(_btn_icon("compass"))              # compass
        self._nav_btn.clicked.connect(self._navigate_to)
        btn_row1.addWidget(self._nav_btn)
        self._nudge_btn = QPushButton("Nudge to Here")
        self._nudge_btn.setIcon(_btn_icon("hand-pointer"))       # fingertip
        self._nudge_btn.setToolTip(
            "Update the selected flake's recorded position to the current\n"
            "stage position (re-derives wafer/chip coords and corrects for the\n"
            "objective paraxial offset if the current mag differs from the\n"
            "flake's recorded mag).")
        self._nudge_btn.clicked.connect(self._nudge_to_here)
        btn_row1.addWidget(self._nudge_btn)
        self._delete_flake_btn = QPushButton("Delete Flake")
        self._delete_flake_btn.setIcon(_btn_icon("trash"))       # bin
        self._delete_flake_btn.clicked.connect(self._delete_flake)
        btn_row1.addWidget(self._delete_flake_btn)
        table_layout.addLayout(btn_row1)

        # Row 2: image actions for the selected flake + fallback add
        btn_row2 = QHBoxLayout()
        self._capture_frame_btn = QPushButton("Capture Frame")
        self._capture_frame_btn.setIcon(_btn_icon("camera"))     # camera
        self._capture_frame_btn.setToolTip(
            "Save a clean (no-overlay) camera frame\n"
            "and link it to the selected flake.")
        self._capture_frame_btn.clicked.connect(self._capture_frame)
        btn_row2.addWidget(self._capture_frame_btn)
        self._save_view_btn = QPushButton("Save View")
        self._save_view_btn.setIcon(_btn_icon("camera-edit"))    # camera + pencil
        self._save_view_btn.setToolTip(
            "Save the annotated preview display exactly as shown\n"
            "(with overlays) and link it to the selected flake.")
        self._save_view_btn.clicked.connect(self._save_view)
        btn_row2.addWidget(self._save_view_btn)
        self._add_flake_btn = QPushButton("Add Flake (no image)")
        self._add_flake_btn.setIcon(_btn_icon("pin"))            # thumbtack
        self._add_flake_btn.setToolTip("Record current position as a new flake without capturing an image.")
        self._add_flake_btn.clicked.connect(self._add_flake)
        btn_row2.addWidget(self._add_flake_btn)
        table_layout.addLayout(btn_row2)

        for _b in (self._capture_add_btn, self._nav_btn, self._nudge_btn,
                   self._delete_flake_btn, self._capture_frame_btn,
                   self._save_view_btn, self._add_flake_btn):
            _b.setIconSize(_icon_sz)
        splitter.addWidget(table_widget)

        # Flake detail panel
        detail_widget = QGroupBox("Flake Details")
        detail_vbox   = QVBoxLayout()
        detail_vbox.setSpacing(4)

        detail_layout = QFormLayout()
        detail_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self._det_name  = QLineEdit()
        self._det_name.setPlaceholderText("optional label")
        detail_layout.addRow("Name:", self._det_name)

        self._det_area  = QDoubleSpinBox()
        self._det_area.setRange(0, 1e6)
        self._det_area.setDecimals(1)
        self._det_area.setSuffix(" µm²")
        self._det_area.setSpecialValueText("—")
        detail_layout.addRow("Area:", self._det_area)

        self._det_clean = QComboBox()
        self._det_clean.addItems(sd._CLEANLINESS)
        detail_layout.addRow("Cleanliness:", self._det_clean)

        self._det_iso   = QComboBox()
        self._det_iso.addItems(sd._ISOLATION)
        detail_layout.addRow("Isolation:", self._det_iso)

        self._det_status = QComboBox()
        self._det_status.addItems(sd._STATUSES)
        detail_layout.addRow("Status:", self._det_status)

        self._det_layer = QComboBox()
        self._det_layer.addItems(_LAYER_CHOICES)   # '' = unknown, else 1..6
        detail_layout.addRow("Layer:", self._det_layer)

        self._det_notes = QTextEdit()
        self._det_notes.setFixedHeight(50)
        self._det_notes.setPlaceholderText("Notes…")
        detail_layout.addRow("Notes:", self._det_notes)

        save_det_btn = QPushButton("Save Details")
        save_det_btn.clicked.connect(self._save_flake_details)
        detail_layout.addRow("", save_det_btn)

        detail_vbox.addLayout(detail_layout)

        # ── Images sub-section ────────────────────────────────────────────
        imgs_hdr = QLabel("Images")
        imgs_hdr.setStyleSheet("font-weight: bold; margin-top: 2px;")
        detail_vbox.addWidget(imgs_hdr)

        self._det_images_list = QListWidget()
        self._det_images_list.setIconSize(QSize(_IMAGE_THUMB_PX, _IMAGE_THUMB_PX))
        self._det_images_list.setViewMode(QListWidget.IconMode)
        self._det_images_list.setResizeMode(QListWidget.Adjust)
        self._det_images_list.setWrapping(True)
        self._det_images_list.setFixedHeight(_IMAGE_THUMB_PX + 28)
        self._det_images_list.setSpacing(4)
        self._det_images_list.setToolTip("Double-click an image to open it.")
        self._det_images_list.itemDoubleClicked.connect(self._open_image)
        detail_vbox.addWidget(self._det_images_list)

        detail_widget.setLayout(detail_vbox)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self._splitter = splitter
        self.flake_catalogue_widget = splitter  # exposed for placement in app_shell

        root.addWidget(self._content)

    # ── Lock / unlock ──────────────────────────────────────────────────────

    def _set_locked(self, locked: bool):
        self._content.setEnabled(not locked)
        if locked:
            self._set_sample_locked(True)

    def _set_sample_locked(self, locked: bool):
        """Grey out everything that requires an open sample."""
        enabled = not locked
        self._extents_btn.setEnabled(enabled)
        self._close_sample_btn.setEnabled(enabled)
        self._substrate_combo.setEnabled(enabled)
        self._measure_thick_btn.setEnabled(enabled)
        self._cs_group.setEnabled(enabled)
        self._splitter.setEnabled(enabled)

    # ── User management ────────────────────────────────────────────────────

    def _refresh_users(self):
        self._user_combo.blockSignals(True)
        prev = self._user_combo.currentText()
        self._user_combo.clear()
        self._user_combo.addItem("")          # blank = "no selection"
        for u in sd.load_users():
            self._user_combo.addItem(u)
        idx = self._user_combo.findText(prev)
        if idx >= 0:
            self._user_combo.setCurrentIndex(idx)
        self._user_combo.blockSignals(False)

    def _add_user(self):
        name, ok = QInputDialog.getText(self, "Add User", "User name:")
        if ok and name.strip():
            sd.add_user(name.strip())
            self._refresh_users()
            idx = self._user_combo.findText(name.strip())
            if idx >= 0:
                self._user_combo.setCurrentIndex(idx)

    def _on_user_changed(self, user: str):
        if user:
            self._set_locked(False)
            self._set_sample_locked(True)
            self._refresh_sample_list()
            self._sample = None
            self._active_label.setText("No sample open.")
            self._flake_table.setRowCount(0)
            self._det_images_list.clear()
        else:
            self._set_locked(True)

    # ── Sample management ──────────────────────────────────────────────────

    def _active_sample_dir(self) -> str | None:
        """Return the absolute path to the active sample's root directory, or None."""
        if self._sample is None:
            return None
        return os.path.abspath(os.path.join(
            str(_USERS_ROOT), self._sample['user'], self._sample['folder']))

    def scan_output_info(self):
        """Return (scans_dir, sample_name) for the active sample, or (None, None)."""
        sdir = self._active_sample_dir()
        if sdir is None:
            return None, None
        return os.path.join(sdir, 'mapping', 'scans'), self._sample.get('name', '')

    def wafer_extents_mm(self):
        """Return (x_neg, x_pos, y_neg, y_pos) in current stage-mm, or None.

        Uses the already-projected extents stored in stage_controls (populated
        by _restore_extents on sample open and update_extents_from_registration
        after apply), so it reflects the placement transform automatically.
        """
        ep = getattr(self.stage_controls, '_edge_positions', None) if self.stage_controls else None
        if ep:
            vals = (ep.get('x_negative'), ep.get('x_positive'),
                    ep.get('y_negative'), ep.get('y_positive'))
            if all(v is not None for v in vals):
                return vals
        return None

    def corner_focus_points(self):
        """Return [(x_mm, y_mm, z_mm)] in current stage coords from registered corners.

        Only corners with a recorded Z value are included.  XY are transformed
        from reference-stage to current-stage using the best available placement
        hint; Z is mount-invariant (substrate height doesn't change between mounts).
        Returns an empty list if no sample / no corners / fewer than 1 Z-bearing corner.
        """
        if self._sample is None:
            return []
        reg = (self._sample.get('placement') or {}).get('registration') or {}
        corners = reg.get('corners') or []
        tf = sd.get_placement_transform_hint(self._sample)
        pts = []
        for c in corners:
            x_ref, y_ref, z = c.get('x_mm'), c.get('y_mm'), c.get('z_mm')
            if x_ref is None or y_ref is None or z is None:
                continue
            if tf is not None:
                from vision.registration import apply_placement_transform
                x, y = apply_placement_transform(tf, x_ref, y_ref)
            else:
                x, y = x_ref, y_ref
            pts.append((float(x), float(y), float(z)))
        return pts

    def timelapse_output_dir(self) -> str | None:
        """Return the timelapse directory for the active sample, or None."""
        sdir = self._active_sample_dir()
        return os.path.join(sdir, 'timelapse') if sdir else None

    def frames_output_dir(self) -> str | None:
        """Return the frames directory for the active sample, or None."""
        sdir = self._active_sample_dir()
        return os.path.join(sdir, 'frames') if sdir else None

    def _current_user(self) -> str:
        return self._user_combo.currentText()

    def _refresh_sample_list(self):
        user = self._current_user()
        folders = sd.list_samples(_USERS_ROOT, user) if user else []
        self._sample_combo.blockSignals(True)
        self._sample_combo.clear()
        for f in folders:
            self._sample_combo.addItem(f)
        self._sample_combo.blockSignals(False)

    def _open_sample(self):
        folder = self._sample_combo.currentText()
        if not folder:
            return
        try:
            self._sample = sd.load_sample(_USERS_ROOT, self._current_user(), folder)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not open sample:\n{exc}")
            return
        self._transform_fresh = False
        self._session_flake_ids = set()
        self._active_label.setText(f"Sample: {self._sample['name']}  [{folder}]")
        self._set_sample_locked(False)
        self._populate_flake_table()
        self._restore_coord_system()
        self._restore_substrate()
        self._restore_extents()
        self._update_stage_markers()
        self.sample_changed.emit()

        # Ask whether the sample is physically on the stage now
        self._prompt_load_state()

    def _prompt_load_state(self):
        """Ask the user if the sample is currently on the stage and handle registration."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        reg = sd.registration_state(self._sample)
        was_loaded = self._sample.get('placement', {}).get('loaded', False)

        reply = QMessageBox.question(
            self, "Sample on stage?",
            f"Is  '{self._sample['name']}'  currently mounted on the stage?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply != QMessageBox.Yes:
            return  # not on stage — leave placement state as-is

        if was_loaded and reg in ('registered', 'extents'):
            # Was previously registered — ask if this is the same physical placement
            remount = QMessageBox.question(
                self, "Remounted?",
                "Has this sample been removed and remounted since last time?\n"
                "(Choose No if it has not been touched.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            same_placement = (remount == QMessageBox.No)
        else:
            same_placement = False   # first load or no prior registration

        needs_registration = sd.mark_loaded(self._sample) if not same_placement \
            else self._mark_loaded_same_placement()
        sd.save_sample(_USERS_ROOT, self._sample)
        self.registration_changed.emit()

        if needs_registration:
            QMessageBox.information(
                self, "Registration needed",
                "The sample has been remounted.  Open the Registration panel "
                "(Find section) to re-confirm the coordinate alignment.")

    def _mark_loaded_same_placement(self) -> bool:
        """Mark loaded without invalidating an existing registration."""
        import core.sample_data as sd_mod
        p = sd_mod._placement(self._sample)
        from datetime import datetime
        p['loaded']    = True
        p['loaded_at'] = datetime.now().isoformat(timespec='seconds')
        return False  # no re-registration needed

    def _create_sample(self):
        name = self._new_name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Name required",
                                    "Enter a short sample name before creating.")
            return
        user = self._current_user()
        try:
            self._sample = sd.new_sample(_USERS_ROOT, user, name)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not create sample:\n{exc}")
            return
        self._transform_fresh = False
        self._session_flake_ids = set()
        self._new_name_edit.clear()
        self._refresh_sample_list()
        folder = self._sample['folder']
        idx = self._sample_combo.findText(folder)
        if idx >= 0:
            self._sample_combo.setCurrentIndex(idx)
        self._active_label.setText(f"Sample: {self._sample['name']}  [{folder}]")
        self._set_sample_locked(False)
        self._flake_table.setRowCount(0)
        self._det_images_list.clear()
        self._restore_coord_system()
        self._restore_substrate()
        self.sample_changed.emit()

    def _close_sample(self):
        """Deselect the current sample, returning to the no-sample state."""
        self._sample = None
        self._transform_fresh = False
        self._session_flake_ids = set()
        self._ref_marks = [None, None]
        self._active_label.setText("No sample open.")
        self._flake_table.setRowCount(0)
        self._det_images_list.clear()
        self._restore_coord_system()
        self._thickness_label.setText("")
        self._set_sample_locked(True)
        if self.stage_controls is not None:
            self.stage_controls.clear_edge_positions()
        self._update_stage_markers()
        self.sample_changed.emit()

    # ── Flake table ────────────────────────────────────────────────────────

    def _populate_flake_table(self):
        self._flake_table.blockSignals(True)
        self._flake_table.setRowCount(0)
        self._det_images_list.clear()
        if self._sample is None:
            self._flake_table.blockSignals(False)
            return
        for flake in self._sample.get('flakes', []):
            if not flake.get('deleted'):
                self._append_flake_row(flake)
        self._flake_table.blockSignals(False)

    def _append_flake_row(self, flake: dict):
        """Insert one flake row; caller must manage blockSignals around bulk inserts."""
        row = self._flake_table.rowCount()
        self._flake_table.insertRow(row)
        locked = flake.get('locked', False)
        bg = _LOCKED_BG if locked else _UNLOCKED_BG
        area = flake.get('area_um2')

        # Col 0: ID — always read-only
        self._flake_table.setItem(row, 0, _ro_item(flake.get('id', '')))

        # Editable text cells (col index → value); col 3 (Layer) is read-only,
        # edited via the detail dropdown, so it is skipped here.
        editable_vals = {
            1: flake.get('name', ''),
            2: f"{area:.1f}" if area else '',
            4: flake.get('cleanliness', ''),
            5: flake.get('isolation', ''),
            6: flake.get('status', ''),
        }
        for col, val in editable_vals.items():
            item = QTableWidgetItem(str(val))
            if locked:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setBackground(bg)
            self._flake_table.setItem(row, col, item)

        # Col 3: layer count — read-only display (edit via the detail dropdown)
        lyr = _ro_item(_layer_str(flake.get('layer_count')))
        lyr.setBackground(bg)
        self._flake_table.setItem(row, 3, lyr)

        # Col 7: lock checkbox
        chk = QCheckBox()
        chk.setChecked(locked)
        chk.setStyleSheet("margin-left: 12px;")
        chk.stateChanged.connect(lambda state, r=row: self._set_flake_lock(r, state))
        self._flake_table.setCellWidget(row, 7, chk)

        # Cols 8-10: shape metrics — read-only (sourced from detection)
        for col, key, fmt in ((8, 'circularity', "{:.2f}"),
                              (9, 'aspect_ratio', "{:.2f}"),
                              (10, 'solidity', "{:.2f}")):
            item = _ro_item(_metric_str(flake.get(key), fmt))
            item.setBackground(bg)
            self._flake_table.setItem(row, col, item)

    def _selected_flake(self):
        """Return the flake dict for the currently selected row, or None."""
        if self._sample is None:
            return None
        row = self._flake_table.currentRow()
        if row < 0 or row >= len(self._sample.get('flakes', [])):
            return None
        return self._sample['flakes'][row]

    def _on_flake_item_changed(self, item):
        """Auto-save when the user edits a cell inline."""
        if self._sample is None:
            return
        row = item.row()
        col = item.column()
        flakes = self._sample.get('flakes', [])
        if row < 0 or row >= len(flakes):
            return
        flake = flakes[row]
        if flake.get('locked', False):
            return  # locked rows must not be modified

        _COL_FIELD = {
            1: 'name',
            4: 'cleanliness',
            5: 'isolation',
            6: 'status',
        }
        text = item.text().strip()
        if col in _COL_FIELD:
            flake[_COL_FIELD[col]] = text
        elif col == 2:   # area_um2 — parse float
            try:
                flake['area_um2'] = float(text) if text else None
            except ValueError:
                return  # ignore bad input; don't save
        else:
            return  # col 0 (ID), col 3 (Layer, read-only) or col 7 (lock widget)

        from datetime import datetime
        flake['updated_at'] = datetime.now().isoformat(timespec='seconds')
        sd.save_sample(_USERS_ROOT, self._sample)
        # Keep detail panel in sync if this row is selected
        if self._flake_table.currentRow() == row:
            self._flake_table.blockSignals(True)
            self._on_flake_selected(row)
            self._flake_table.blockSignals(False)

    def _set_flake_lock(self, row: int, state: int):
        """Toggle the locked state for a flake and update cell editability."""
        if self._sample is None:
            return
        flakes = self._sample.get('flakes', [])
        if row < 0 or row >= len(flakes):
            return
        locked = (state == Qt.Checked)
        flakes[row]['locked'] = locked
        sd.save_sample(_USERS_ROOT, self._sample)

        bg = _LOCKED_BG if locked else _UNLOCKED_BG
        self._flake_table.blockSignals(True)
        for col in range(1, 7):          # data cells: Name,Area,Layer,Clean,Iso,Status
            it = self._flake_table.item(row, col)
            if not it:
                continue
            if col != 3:                 # col 3 (Layer) stays read-only always
                if locked:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                else:
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
            it.setBackground(bg)
        self._flake_table.blockSignals(False)

    def _on_flake_selected(self, row, *_):
        flake = self._selected_flake()
        if flake is None:
            return
        self._det_name.setText(flake.get('name', ''))
        area = flake.get('area_um2')
        self._det_area.setValue(area if area else 0.0)
        self._det_clean.setCurrentText(flake.get('cleanliness', ''))
        self._det_iso.setCurrentText(flake.get('isolation', ''))
        self._det_status.setCurrentText(flake.get('status', 'Candidate'))
        lc = flake.get('layer_count')
        self._det_layer.setCurrentText(str(int(lc)) if lc else '')
        self._det_notes.setPlainText(flake.get('notes', ''))
        self._update_image_list(flake)

    def _stamp_grid_coords(self, flake: dict):
        """Attach local-frame grid coordinates to the flake in-place.

        Only stamps when the transform has been confirmed fresh this session —
        a stale (previous-session) transform must not be used here because the
        wafer may be at a different stage position.
        """
        if not self._transform_fresh:
            return
        t = (self._sample or {}).get('transform')
        if t is None:
            return
        try:
            gx, gy = sd.stage_to_grid(t, flake['stage_x_mm'], flake['stage_y_mm'])
            flake['grid_x_eff'] = round(gx, 6)
            flake['grid_y_eff'] = round(gy, 6)
        except Exception:
            pass

    def _stamp_chip_coords(self, flake: dict):
        """Attach chip-local (cx_mm, cy_mm) to the flake using corner registration.

        Only stamps when placement is 'registered' — a stale or absent placement
        transform would produce wrong chip coords (missing the inverse placement
        transform).  Flakes created before registration is confirmed are
        backfilled by backfill_chip_coords() when _apply() completes.
        """
        if self._sample is None:
            return
        if sd.registration_state(self._sample) != 'registered':
            return
        try:
            result = sd.stage_to_chip(
                self._sample, flake['stage_x_mm'], flake['stage_y_mm'])
            if result is not None:
                flake['chip_x_mm'] = round(result[0], 6)
                flake['chip_y_mm'] = round(result[1], 6)
        except Exception:
            pass

    def _add_flake(self):
        if self._sample is None:
            QMessageBox.information(self, "No sample", "Open or create a sample first.")
            return
        # Read current stage position
        x_mm = y_mm = z_mm = r_deg = 0.0
        mag = ''
        if self.mm is not None:
            try:
                x_mm  = self.mm.get_position_units_cached('X') or 0.0
                y_mm  = self.mm.get_position_units_cached('Y') or 0.0
                z_mm  = self.mm.get_position_units_cached('Z') or 0.0
                r_deg = self.mm.get_position_units_cached('R') or 0.0
            except Exception:
                pass
        if self.controls is not None:
            try:
                mag = self.controls._mag_selector.currentText()
            except Exception:
                pass

        fid = sd.next_flake_id(self._sample['flakes'])
        flake = sd.new_flake(fid, '', x_mm, y_mm, z_mm, mag, r_deg=r_deg)
        flake['source'] = 'app'          # provenance: marked at the scope
        self._stamp_grid_coords(flake)
        self._stamp_chip_coords(flake)
        self._session_flake_ids.add(fid)
        self._sample['flakes'].append(flake)
        sd.save_sample(_USERS_ROOT, self._sample)
        self._flake_table.blockSignals(True)
        self._append_flake_row(flake)
        self._flake_table.blockSignals(False)
        new_row = self._flake_table.rowCount() - 1
        self._flake_table.setCurrentCell(new_row, 0)
        self._update_stage_markers()
        import core.logbook as _lb
        if _lb.get():
            _lb.get().log('flake', f'Flake #{fid} saved (no image)',
                          detail=f'X={x_mm:+.3f}  Y={y_mm:+.3f}  Z={z_mm:.4f} mm  {mag}')

    def import_map_markers(self, markers: list):
        """Add a list of {x_mm, y_mm, note, layer?} dicts as new Candidate flakes."""
        if self._sample is None:
            return
        self._flake_table.blockSignals(True)
        try:
            for m in markers:
                fid = sd.next_flake_id(self._sample['flakes'])
                lc = m.get('layer')
                lc = int(lc) if lc is not None else None
                flake = sd.new_flake(fid, '',
                                     m['x_mm'], m['y_mm'], 0.0, '',
                                     layer_count=lc)
                flake['notes'] = m.get('note', '') or m.get('notes', '')
                flake['source'] = 'map'
                self._stamp_grid_coords(flake)
                self._stamp_chip_coords(flake)
                self._attach_scan_crop(flake)
                self._sample['flakes'].append(flake)
                self._append_flake_row(flake)
        finally:
            self._flake_table.blockSignals(False)
        sd.save_sample(_USERS_ROOT, self._sample)
        self._update_stage_markers()
        import core.logbook as _lb
        if _lb.get():
            _lb.get().log('flake', f'Imported {len(markers)} flake(s) from map',
                          detail=', '.join(f'X={m["x_mm"]:+.3f} Y={m["y_mm"]:+.3f}'
                                           for m in markers))

    def _attach_scan_crop(self, flake: dict):
        """Crop this flake from the nearest area-scan tile and register it as a
        catalogue thumbnail (so map flakes get an image like app/auto flakes)."""
        try:
            import os
            import cv2
            from vision.scan_crops import find_scan_folder, crop_at_stage
            images_dir = sd.images_dir_for_sample(_USERS_ROOT, self._sample)
            sample_dir = os.path.dirname(images_dir)
            scan = find_scan_folder(sample_dir, flake.get('magnification') or None)
            if not scan:
                return
            crop, crop_um = crop_at_stage(scan, flake['stage_x_mm'], flake['stage_y_mm'],
                                          flake.get('area_um2'))
            if crop is None:
                return
            os.makedirs(images_dir, exist_ok=True)
            fname = f"{flake['id']}_map_crop.png"
            cv2.imwrite(os.path.join(images_dir, fname), crop)
            flake.setdefault('images', []).append(
                {'file': fname, 'mag': flake.get('magnification', ''),
                 'type': 'crop', 'crop_um': crop_um})
        except Exception as e:
            print(f"[catalogue] map crop failed for {flake.get('id')}: {e}")

    def update_flake_fields(self, flake_id: str, updates: dict):
        """Persist field updates (e.g. layer_count) from the HTML map to a catalogue flake."""
        if self._sample is None:
            return
        for fl in self._sample.get('flakes', []):
            if fl.get('id') == flake_id:
                for k, v in updates.items():
                    fl[k] = v
                import datetime as _dt
                fl['updated_at'] = _dt.datetime.now().isoformat()[:19]
                sd.save_sample(_USERS_ROOT, self._sample)
                self._refresh_table()
                return

    def delete_flake_by_id(self, flake_id: str):
        """Soft-delete a flake (sets deleted=True) so it can be restored from the JSON."""
        if self._sample is None:
            return
        import datetime as _dt
        for fl in self._sample.get('flakes', []):
            if fl.get('id') == flake_id:
                fl['deleted'] = True
                fl['deleted_at'] = _dt.datetime.now().isoformat()[:19]
                sd.save_sample(_USERS_ROOT, self._sample)
                self._refresh_table()
                return

    def _navigate_to(self):
        flake = self._selected_flake()
        if flake is None:
            return
        if self.mm is None:
            QMessageBox.information(self, "No stage", "No motor manager available.")
            return
        fx = flake['stage_x_mm']
        fy = flake['stage_y_mm']
        flake_r = flake.get('r_deg', None)
        used_chip = False
        used_grid = False

        # Priority 1: chip-local coords via corner registration.
        # Chip coords are stable across remounts — chip_to_stage applies the
        # current placement_transform to map them to the current stage frame.
        chip_x = flake.get('chip_x_mm')
        chip_y = flake.get('chip_y_mm')
        if chip_x is not None and chip_y is not None:
            result = None
            try:
                result = sd.chip_to_stage(self._sample, chip_x, chip_y)
            except Exception:
                pass
            if result is not None:
                fx, fy = result
                used_chip = True
            else:
                # Chip coords exist but no transform — warn and abort.
                QMessageBox.warning(
                    self, "Registration required",
                    "This flake has chip-local coordinates but no placement\n"
                    "transform is set. Please apply corner registration first\n"
                    "(Registration panel → Capture & Match → Apply).")
                return

        # Priority 2: index-mark grid coords if chip coords unavailable.
        if not used_chip:
            gx_eff = flake.get('grid_x_eff')
            gy_eff = flake.get('grid_y_eff')
            t = (self._sample or {}).get('transform')
            if (gx_eff is not None and gy_eff is not None
                    and t is not None and self._transform_fresh):
                try:
                    fx, fy = sd.grid_to_stage(t, gx_eff, gy_eff)
                    used_grid = True
                except Exception:
                    pass

        # Eucentric R correction: rotate around the eucentric centre to account
        # for R drift since calibration.
        t = (self._sample or {}).get('transform')
        current_r = self.mm.get_position_units_cached('R')
        if used_chip:
            r_ref = flake_r   # chip_to_stage gives stage position at save-time R
        elif used_grid:
            r_ref = t.get('r_deg_at_calibration', 0.0) if t else 0.0
        else:
            r_ref = flake_r
        if current_r is not None and r_ref is not None:
            delta_r = math.radians(current_r - r_ref)
            if abs(delta_r) > 1e-6:
                xc, yc = (
                    self.stage_controls._eucentric_center_mm
                    if self.stage_controls
                    and self.stage_controls._eucentric_center_mm
                    else (0.0, 0.0)
                )
                dx, dy = fx - xc, fy - yc
                cos_r, sin_r = math.cos(delta_r), math.sin(delta_r)
                fx = xc + dx * cos_r + dy * sin_r
                fy = yc - dx * sin_r + dy * cos_r

        move = self.mm.move_absolute_units
        z = flake.get('z_mm')                       # 0/None = no real focus (e.g. raw map mark)
        try:
            move('X', fx)
            move('Y', fy)
            if z:                                   # only drive Z to a real focus height
                move('Z', z)
        except Exception as exc:
            QMessageBox.warning(self, "Navigation error",
                                f"Stage move failed:\n{exc}")
            return
        if self.stage_controls is not None:
            z_set = z if z else (self.mm.get_position_units_cached('Z') or 0.0)
            self.stage_controls.set_move_setpoint(fx, fy, z_set)

        # Logbook
        import core.logbook as _lb
        if _lb.get():
            fid   = flake.get('id', '?')
            fname = flake.get('name', '')
            title = f"Navigate · #{fid}" + (f'  "{fname}"' if fname else '')
            src   = 'chip' if used_chip else ('grid' if used_grid else 'stage')
            pos   = (f'X {fx:+.3f}  Y {fy:+.3f}  '
                     + (f'Z {z:+.3f} mm' if z else 'Z held') + f'  [{src}]')
            cx = flake.get('chip_x_mm')
            cy = flake.get('chip_y_mm')
            if cx is not None and cy is not None:
                pos += f'  chip {cx:+.3f} {cy:+.3f}'
            _lb.get().log('navigate', title, detail=pos)

    def _nudge_to_here(self):
        """Re-record the selected flake's position at the current stage position.

        Stores the new position in the flake's own magnification frame: the live
        stage reading is taken at the *current* objective, so any difference from
        the flake's recorded objective is removed via the paraxial offset before
        storing.  Wafer-relative (chip-local) and grid coords are re-derived from
        the corrected stage position, keeping the flake portable across remounts.
        """
        flake = self._selected_flake()
        if flake is None:
            QMessageBox.information(self, "No flake", "Select a flake first.")
            return
        if self.mm is None:
            QMessageBox.information(self, "No stage", "No motor manager available.")
            return
        if flake.get('locked', False):
            QMessageBox.information(self, "Locked",
                                    f"{flake.get('id', '')} is locked. Unlock it to nudge.")
            return
        try:
            cx = self.mm.get_position_units_cached('X') or 0.0
            cy = self.mm.get_position_units_cached('Y') or 0.0
            cz = self.mm.get_position_units_cached('Z') or 0.0
            cr = self.mm.get_position_units_cached('R') or 0.0
        except Exception as exc:
            QMessageBox.warning(self, "Read error", f"Could not read stage:\n{exc}")
            return

        cur_mag   = self._current_mag()
        # The flake's stored position lives in its own mag frame; if it never had
        # a recorded mag (e.g. a raw map mark), adopt the current mag as its frame.
        frame_mag = flake.get('magnification') or cur_mag
        sdx, sdy  = self._paraxial_shift(cur_mag, frame_mag)
        new_sx = cx + sdx
        new_sy = cy + sdy

        # How far is this move?  Compare against the flake's current expected
        # stage position so the user can see (and confirm) the displacement.
        ex, ey = self._expected_stage_xy(flake)
        delta_um = math.hypot(new_sx - ex, new_sy - ey) * 1000.0
        para_note = ''
        if abs(sdx) > 1e-6 or abs(sdy) > 1e-6:
            para_note = (f"\n(includes paraxial correction "
                         f"{cur_mag or '?'}→{frame_mag or '?'}: "
                         f"ΔX {sdx * 1000:+.0f}  ΔY {sdy * 1000:+.0f} µm)")
        ans = QMessageBox.question(
            self, "Nudge flake position",
            f"Move {flake.get('id', '')}'s recorded position by "
            f"<b>{delta_um:.0f} µm</b> to the current stage position?{para_note}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans != QMessageBox.Yes:
            return

        flake['stage_x_mm']   = round(new_sx, 6)
        flake['stage_y_mm']   = round(new_sy, 6)
        flake['z_mm']         = round(cz, 6)
        flake['r_deg']        = round(cr, 4)
        if not flake.get('magnification'):
            flake['magnification'] = cur_mag
        # Re-derive wafer-relative coords from the corrected stage position.
        # Clear stale values first so a failed re-stamp doesn't leave old coords.
        flake.pop('chip_x_mm', None)
        flake.pop('chip_y_mm', None)
        flake.pop('grid_x_eff', None)
        flake.pop('grid_y_eff', None)
        self._stamp_chip_coords(flake)
        self._stamp_grid_coords(flake)

        from datetime import datetime
        flake['updated_at'] = datetime.now().isoformat(timespec='seconds')
        sd.save_sample(_USERS_ROOT, self._sample)
        self._update_stage_markers()

        import core.logbook as _lb
        if _lb.get():
            fid = flake.get('id', '?')
            detail = (f'X {new_sx:+.3f}  Y {new_sy:+.3f}  Z {cz:.4f} mm  '
                      f'{cur_mag or "?"}  (Δ {delta_um:.0f} µm)')
            ch = (flake.get('chip_x_mm'), flake.get('chip_y_mm'))
            if ch[0] is not None and ch[1] is not None:
                detail += f'  chip {ch[0]:+.3f} {ch[1]:+.3f}'
            _lb.get().log('flake', f'Flake #{fid} position nudged', detail=detail)

    # ── Image capture helpers ──────────────────────────────────────────────

    def _current_mag(self) -> str:
        if self.controls is not None:
            try:
                return self.controls._mag_selector.currentText()
            except Exception:
                pass
        return ''

    # ── Objective paraxial offsets ──────────────────────────────────────────
    # Each objective's optical axis points at a slightly different stage XY;
    # the runtime auto-compensates on a *mag switch* (stagecontrol._on_mag_changed_cal)
    # so the same feature stays centred.  Relationship: a feature centred at
    # magnification M reads at stage position  S(M) = S_ref + off[M]  (off[100x]=0).
    # The catalogue stores positions in each flake's own mag frame; these helpers
    # convert a stage reading taken at one mag into another mag's frame.

    def _objective_offset(self, mag: str):
        """Calibrated XY offset (mm) of *mag* relative to the 100x reference.

        Returns (dx, dy), or None if the objective is uncalibrated (unknown).
        """
        if mag == '100x':
            return (0.0, 0.0)
        try:
            from ui.calibration_panel import load_calibration
            off = load_calibration().get('objective_offsets', {}).get(mag)
        except Exception:
            off = None
        if not off:
            return None
        return (float(off[0]), float(off[1]))

    def _paraxial_shift(self, from_mag: str, to_mag: str):
        """Stage shift (dx, dy) mm to convert a reading at *from_mag* into the
        *to_mag* frame:  S(to) = S(from) + (off[to] - off[from]).

        Returns (0, 0) when either objective is uncalibrated — applying a guessed
        offset would be worse than none (cf. stagecontrol._on_mag_changed_cal).
        """
        a = self._objective_offset(to_mag)
        b = self._objective_offset(from_mag)
        if a is None or b is None:
            return (0.0, 0.0)
        return (a[0] - b[0], a[1] - b[1])

    def _expected_stage_xy(self, flake: dict):
        """Flake's recorded position mapped into the *current* stage frame.

        Mirrors _navigate_to's priority — chip-local → index-mark grid → raw
        saved stage — so the result tracks the wafer across remounts instead of
        trusting the (possibly stale) saved stage coordinates.  Result is in the
        flake's own magnification frame (see _paraxial_shift)."""
        chip_x = flake.get('chip_x_mm')
        chip_y = flake.get('chip_y_mm')
        if chip_x is not None and chip_y is not None:
            try:
                r = sd.chip_to_stage(self._sample, chip_x, chip_y)
                if r is not None:
                    return r
            except Exception:
                pass
        gx = flake.get('grid_x_eff')
        gy = flake.get('grid_y_eff')
        t = (self._sample or {}).get('transform')
        if (gx is not None and gy is not None
                and t is not None and self._transform_fresh):
            try:
                return sd.grid_to_stage(t, gx, gy)
            except Exception:
                pass
        return flake['stage_x_mm'], flake['stage_y_mm']

    def _update_image_list(self, flake: dict):
        """Repopulate the image thumbnail list for *flake*."""
        self._det_images_list.clear()
        if self._sample is None:
            return
        imgs_dir = sd.images_dir_for_sample(_USERS_ROOT, self._sample)
        for img in flake.get('images', []):
            fname = img.get('file', '')
            fpath = os.path.join(imgs_dir, fname)
            # Thumbnail
            px = QPixmap(fpath)
            if px.isNull():
                # File missing or unreadable — show a placeholder icon
                px = QPixmap(_IMAGE_THUMB_PX, _IMAGE_THUMB_PX)
                px.fill(QColor(180, 180, 180))
            else:
                px = px.scaled(_IMAGE_THUMB_PX, _IMAGE_THUMB_PX,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label = fname
            if img.get('type') == 'view':
                label += '\n[view]'
            item = QListWidgetItem(QIcon(px), label)
            item.setData(Qt.UserRole, fpath)   # full path for double-click open
            item.setToolTip(f"{fname}\n{img.get('mag', '')}  {img.get('type', 'frame')}")
            self._det_images_list.addItem(item)

    def _open_image(self, item: QListWidgetItem):
        """Open the double-clicked image with the system default viewer."""
        fpath = item.data(Qt.UserRole)
        if fpath and os.path.isfile(fpath):
            import subprocess
            import sys
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', fpath])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', fpath])
            else:
                os.startfile(fpath)

    def _check_at_flake(self, flake: dict) -> bool:
        """Return True if the stage is close enough to the flake's position.

        If the stage has moved further than the approximate half-FOV for the
        current magnification, shows a warning and returns False (or True if
        the user confirms they still want to capture).
        """
        if self.mm is None:
            return True
        try:
            cx = self.mm.get_position_units_cached('X') or 0.0
            cy = self.mm.get_position_units_cached('Y') or 0.0
        except Exception:
            return True
        mag = self._current_mag()
        # Compare the flake's *wafer* position (mapped through the current
        # transform) against where the stage actually is — not the raw saved
        # stage coords, which go stale after a remount.  Bring the live reading
        # into the flake's mag frame so the paraxial objective offset doesn't
        # masquerade as drift.
        ex, ey = self._expected_stage_xy(flake)
        sdx, sdy = self._paraxial_shift(mag, flake.get('magnification', '') or mag)
        dist_mm = math.hypot((cx + sdx) - ex, (cy + sdy) - ey)
        threshold = _HALF_FOV_MM.get(mag, _HALF_FOV_DEFAULT_MM)
        if dist_mm <= threshold:
            return True
        ans = QMessageBox.warning(
            self, "Stage has moved",
            f"Current position is <b>{dist_mm * 1000:.0f} µm</b> from "
            f"{flake['id']}'s recorded position "
            f"(limit: {threshold * 1000:.0f} µm at {mag or 'unknown mag'}).\n\n"
            "The flake may not be in the field of view.\n"
            "Capture and link to this flake anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return ans == QMessageBox.Yes

    def _capture_and_add(self):
        """Capture a clean frame, create a new flake at current position, link image."""
        if self._sample is None:
            QMessageBox.information(self, "No sample", "Open or create a sample first.")
            return
        if self.preview is None:
            QMessageBox.information(self, "No preview", "No camera preview available.")
            return
        frame = None
        try:
            frame = self.preview.get_clean_frame()
        except Exception:
            pass
        if frame is None:
            QMessageBox.warning(self, "No frame", "No camera frame available.")
            return

        # Read current stage position
        x_mm = y_mm = z_mm = r_deg = 0.0
        if self.mm is not None:
            try:
                x_mm  = self.mm.get_position_units_cached('X') or 0.0
                y_mm  = self.mm.get_position_units_cached('Y') or 0.0
                z_mm  = self.mm.get_position_units_cached('Z') or 0.0
                r_deg = self.mm.get_position_units_cached('R') or 0.0
            except Exception:
                pass
        mag = self._current_mag()

        fid   = sd.next_flake_id(self._sample['flakes'])
        flake = sd.new_flake(fid, '', x_mm, y_mm, z_mm, mag, r_deg=r_deg)
        flake['source'] = 'app'          # provenance: marked at the scope
        self._stamp_grid_coords(flake)
        self._stamp_chip_coords(flake)
        self._session_flake_ids.add(fid)
        self._sample['flakes'].append(flake)
        sd.save_sample(_USERS_ROOT, self._sample)

        exp_ms = self.controls.get_exposure_ms() if self.controls is not None else None
        fname = sd.add_image_to_flake(_USERS_ROOT, self._sample, fid, frame, mag,
                                      exposure_ms=exp_ms)

        self._flake_table.blockSignals(True)
        self._append_flake_row(flake)
        self._flake_table.blockSignals(False)
        new_row = self._flake_table.rowCount() - 1
        self._flake_table.setCurrentCell(new_row, 0)
        self._update_image_list(flake)
        self._update_stage_markers()

        import core.logbook as _lb
        if _lb.get():
            exp_s = f'  {exp_ms:.2f} ms' if exp_ms is not None else ''
            img_rel = f'images/{fname}' if fname else ''
            _lb.get().log('flake', f'Flake #{fid} saved  {mag}{exp_s}',
                          detail=(f'X={x_mm:+.3f}  Y={y_mm:+.3f}  Z={z_mm:.4f} mm'
                                  + (f'  {img_rel}' if img_rel else '')),
                          image_rel=img_rel)

        if fname is None:
            QMessageBox.warning(self, "Partial save",
                                f"Flake {fid} created but image capture failed.")

    def _capture_frame(self):
        """Save a clean (no-overlay) camera frame linked to the selected flake."""
        flake = self._selected_flake()
        if flake is None:
            QMessageBox.information(self, "No flake", "Select a flake first.")
            return
        if self._sample is None or self.preview is None:
            return
        if not self._check_at_flake(flake):
            return
        frame = None
        try:
            frame = self.preview.get_clean_frame()
        except Exception:
            pass
        if frame is None:
            QMessageBox.warning(self, "No frame", "No camera frame available.")
            return
        mag    = self._current_mag()
        exp_ms = self.controls.get_exposure_ms() if self.controls is not None else None
        fname  = sd.add_image_to_flake(_USERS_ROOT, self._sample, flake['id'], frame, mag,
                                       exposure_ms=exp_ms)
        if fname:
            self._update_image_list(flake)
            import core.logbook as _lb
            if _lb.get():
                fid    = flake.get('id', '?')
                exp_s  = f'{exp_ms:.2f} ms' if exp_ms is not None else ''
                title  = f'Capture · #{fid}  {mag}' + (f'  {exp_s}' if exp_s else '')
                img_rel = f'images/{fname}'
                _lb.get().log('capture', title, detail=img_rel, image_rel=img_rel)
        else:
            QMessageBox.warning(self, "Error", "Failed to save image.")

    def _save_view(self):
        """Save the annotated preview display (as-seen) linked to the selected flake."""
        flake = self._selected_flake()
        if flake is None:
            QMessageBox.information(self, "No flake", "Select a flake first.")
            return
        if self._sample is None or self.preview is None:
            return
        if not self._check_at_flake(flake):
            return
        import os
        from datetime import datetime
        mag    = self._current_mag()
        ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname  = f"{flake['id']}_{mag}_{ts}_view.png"
        fpath  = os.path.join(
            sd.images_dir_for_sample(_USERS_ROOT, self._sample), fname)
        pixmap = self.preview.grab()
        if not pixmap.save(fpath):
            QMessageBox.warning(self, "Error", "Failed to save view image.")
            return
        if sd.register_image_for_flake(_USERS_ROOT, self._sample, flake['id'],
                                        fname, mag, image_type='view'):
            self._update_image_list(flake)
        else:
            QMessageBox.warning(self, "Error", "Image saved but could not link to flake.")

    def _delete_flake(self):
        flake = self._selected_flake()
        if flake is None:
            return
        if flake.get('locked', False):
            QMessageBox.information(self, "Locked",
                                    f"{flake['id']} is locked — unlock it first.")
            return
        ans = QMessageBox.question(self, "Delete flake",
                                   f"Delete {flake['id']}? This cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._sample['flakes'] = [f for f in self._sample['flakes']
                                   if f['id'] != flake['id']]
        sd.save_sample(_USERS_ROOT, self._sample)
        self._populate_flake_table()
        self._update_stage_markers()

    def _save_flake_details(self):
        flake = self._selected_flake()
        if flake is None:
            return
        row = self._flake_table.currentRow()
        flake['name']        = self._det_name.text().strip()
        area = self._det_area.value()
        flake['area_um2']    = area if area > 0 else None
        flake['cleanliness'] = self._det_clean.currentText()
        flake['isolation']   = self._det_iso.currentText()
        flake['status']      = self._det_status.currentText()
        _lyr = self._det_layer.currentText().strip()
        flake['layer_count'] = int(_lyr) if _lyr else None
        flake['notes']       = self._det_notes.toPlainText()
        from datetime import datetime
        flake['updated_at']  = datetime.now().isoformat(timespec='seconds')
        sd.save_sample(_USERS_ROOT, self._sample)
        # Refresh the row in the table (cols: 1 Name, 2 Area, 3 Layer, 4 Clean,
        # 5 Iso, 6 Status; 0 ID and 7 Lock unchanged here).
        self._flake_table.setItem(row, 1, _ro_item(flake['name']))
        self._flake_table.setItem(row, 2, _ro_item(
            f"{flake['area_um2']:.1f}" if flake['area_um2'] else ''))
        self._flake_table.setItem(row, 3, _ro_item(_layer_str(flake['layer_count'])))
        self._flake_table.setItem(row, 4, _ro_item(flake['cleanliness']))
        self._flake_table.setItem(row, 5, _ro_item(flake['isolation']))
        self._flake_table.setItem(row, 6, _ro_item(flake['status']))

    # ── Coordinate system ──────────────────────────────────────────────────

    # ── Wafer extents ──────────────────────────────────────────────────────

    @staticmethod
    def _corners_to_polygon(corners: list[dict],
                             transform: dict | None = None) -> list[tuple] | None:
        """Build a convex polygon from registration corner dicts.

        corners: list of dicts with 'x_mm', 'y_mm' (reference stage positions).
        transform: placement transform dict (hint or exact); None → use reference coords.
        Returns a list of (x, y) tuples sorted clockwise from the centroid,
        or None if fewer than 3 corners are available.
        """
        if len(corners) < 3:
            return None
        from vision.registration import apply_placement_transform
        import math
        pts = []
        for c in corners:
            x, y = c['x_mm'], c['y_mm']
            if transform is not None:
                x, y = apply_placement_transform(transform, x, y)
            pts.append((x, y))
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        return pts

    def _restore_extents(self):
        """Push saved wafer extents from the sample JSON into the stage map.

        Priority:
        1. Registration corners + current placement transform (re-projects on every load)
        2. Index-mark grid polygon + current transform (when transform is fresh)
        3. Raw stage-mm polygon from the extents dict (stale after remount)
        """
        if self.stage_controls is None:
            return
        self.stage_controls.clear_edge_positions()   # always clear on sample open

        # --- priority 1: re-project registration corners via placement transform ---
        s = self._sample
        if s is not None:
            reg = (s.get('placement') or {}).get('registration') or {}
            corners = reg.get('corners') or []
            tf = sd.get_placement_transform_hint(s)
            polygon = self._corners_to_polygon(corners, transform=tf)
            if polygon:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                edges = {
                    'x_negative': min(xs), 'x_positive': max(xs),
                    'y_negative': min(ys), 'y_positive': max(ys),
                }
                self.stage_controls.update_edge_positions(edges, 0.0, polygon=polygon)
                return

        ext = (s or {}).get('extents')
        if not ext:
            return

        # --- priority 2: index-mark grid polygon (fresh transform only) ---
        poly_grid = ext.get('polygon_grid')
        t = (s or {}).get('transform')
        if poly_grid and t and self._transform_fresh:
            try:
                polygon = [sd.grid_to_stage(t, g[0], g[1]) for g in poly_grid]
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                edges = {
                    'x_negative': min(xs), 'x_positive': max(xs),
                    'y_negative': min(ys), 'y_positive': max(ys),
                }
                r_deg = t.get('r_deg_at_calibration', 0.0)
                self.stage_controls.update_edge_positions(edges, r_deg, polygon=polygon)
                return
            except Exception:
                pass

        # --- priority 3: raw stage-mm polygon (stale after remount) ---
        edges = {
            'x_negative': ext.get('x_negative_mm'),
            'x_positive': ext.get('x_positive_mm'),
            'y_negative': ext.get('y_negative_mm'),
            'y_positive': ext.get('y_positive_mm'),
        }
        if any(v is None for v in edges.values()):
            return
        r_deg   = ext.get('r_deg_at_detection', 0.0)
        polygon = ext.get('polygon_mm')
        if polygon:
            polygon = [tuple(p) for p in polygon]
        self.stage_controls.update_edge_positions(edges, r_deg, polygon=polygon)

    def update_extents_from_registration(self):
        """Recompute and save the chip boundary polygon from registration corners.

        Called after registration is applied.  Derives the polygon from the
        freshly-fitted placement transform so the stage map shows the chip
        boundary in its current position without requiring a separate edge walk.
        """
        if self._sample is None or self.stage_controls is None:
            return
        reg = (self._sample.get('placement') or {}).get('registration') or {}
        corners = reg.get('corners') or []
        tf = sd.get_placement_transform(self._sample)   # strict — just applied
        polygon = self._corners_to_polygon(corners, transform=tf)
        if not polygon:
            return
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        edges = {
            'x_negative': min(xs), 'x_positive': max(xs),
            'y_negative': min(ys), 'y_positive': max(ys),
        }
        # update_edge_positions fires _on_extents_updated → _save_extents_to_sample
        self.stage_controls.update_edge_positions(edges, 0.0, polygon=polygon)

    # ── Substrate / oxide thickness ────────────────────────────────────────

    def _restore_substrate(self):
        """Populate the substrate combo from the loaded sample (no save triggered)."""
        sub = (self._sample or {}).get('substrate', sd._SUBSTRATES[0])
        self._substrate_combo.blockSignals(True)
        idx = self._substrate_combo.findText(sub)
        if idx >= 0:
            self._substrate_combo.setCurrentIndex(idx)
        self._substrate_combo.blockSignals(False)
        self._thickness_label.setText("")

    def _on_substrate_changed(self, text: str):
        if self._sample is not None and text:
            self._sample['substrate'] = text
            sd.save_sample(_USERS_ROOT, self._sample)

    def _find_wafer_extents(self):
        """Open wafer-extents detection and save results to the sample file.

        Wired to the existing edge-detection panel for now. Per the agreed
        rework this should become a corner-finding routine (move to each
        expected corner and detect it) rather than the edge-walk — tracked as
        a follow-up; the extents callback (_save_extents_to_sample via
        stage_controls._on_extents_updated) already persists whatever it finds.
        """
        if self._sample is None:
            QMessageBox.information(self, "No sample open",
                                    "Open or create a sample first — "
                                    "extents will be saved into its JSON file.")
            return
        if self.edge_detection_panel is None:
            QMessageBox.information(self, "Unavailable",
                                    "Edge detection is not available.")
            return
        self.edge_detection_panel.show()
        self.edge_detection_panel.raise_()
        self.edge_detection_panel.activateWindow()

    def _measure_substrate_thickness(self):
        """Show SiO₂ thickness from the optical calibration result.

        Optical contrast / oxide spectrometry (vision.calibrate) is not yet
        ported to the Nikon-Prior build, so report that and let the user pick a
        substrate manually for now.
        """
        cal_path = APP_DIR / 'optical_calibration.json'
        try:
            from vision.calibrate import load as _cal_load
            cal = _cal_load(cal_path)
            self._thickness_label.setText(
                f"Calibrated: {cal.sio2_nm:.1f} nm SiO₂\n"
                f"(from optical_calibration.json, {cal.timestamp[:10]})"
            )
        except ImportError:
            self._thickness_label.setText(
                "Oxide-thickness measurement is not available in this build.\n"
                "Select the substrate manually from the dropdown."
            )
        except FileNotFoundError:
            self._thickness_label.setText(
                "No optical calibration found.\n"
                "Select the substrate manually from the dropdown."
            )
        except Exception as exc:
            self._thickness_label.setText(f"Could not read calibration: {exc}")

    def _set_ref_mark(self, n: int):
        """Record current confirmed index mark + stage XY as reference mark n."""
        if self._sample is None:
            QMessageBox.information(self, "No sample", "Open or create a sample first.")
            return

        # Grid coordinates (XX, YY) from the index mark panel's confirmed mark.
        # Quadrant comes from THIS panel's per-mark combo box (not the detector
        # combo) so that it's always visible and survives session reloads.
        xx = yy = None
        if self.index_mark_panel is not None:
            cm = getattr(self.index_mark_panel, '_confirmed_mark', None)
            if cm is not None:
                xx = cm.get('xx')
                yy = cm.get('yy')
        if xx is None or yy is None:
            QMessageBox.information(
                self, "No confirmed mark",
                "Confirm a reference mark in the Index Mark Navigator first.")
            return

        q_text   = self._ref_quadrants[n].currentText()
        quadrant = None if q_text == "—" else q_text

        # Stage XY + R
        sx = sy = r_deg = 0.0
        if self.mm is not None:
            try:
                sx    = self.mm.get_position_units_cached('X') or 0.0
                sy    = self.mm.get_position_units_cached('Y') or 0.0
                r_deg = self.mm.get_position_units_cached('R') or 0.0
            except Exception:
                pass

        ref = {'grid_xx': int(xx), 'grid_yy': int(yy), 'quadrant': quadrant,
               'stage_x_mm': sx, 'stage_y_mm': sy, 'r_deg': r_deg}
        self._ref_marks[n] = ref
        q_str = f" [{quadrant}]" if quadrant else ""
        self._ref_labels[n].setText(
            f"Mark {n+1}: XX={xx:02d} YY={yy:02d}{q_str}  →  "
            f"stage ({sx:.4f}, {sy:.4f}) mm  R={r_deg:.3f}°")
        self._compute_transform_btn.setEnabled(
            self._ref_marks[0] is not None and self._ref_marks[1] is not None)

    def _compute_transform(self):
        if self._ref_marks[0] is None or self._ref_marks[1] is None:
            return
        try:
            t = sd.compute_transform(self._ref_marks[0], self._ref_marks[1])
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid reference marks", str(exc))
            return
        # Store R at calibration so grid_to_stage can correct for later rotation
        t['r_deg_at_calibration'] = self._ref_marks[0].get('r_deg', 0.0)

        # Derive local-frame grid coords for every flake from its original
        # stage_x_mm/stage_y_mm.  Historical flakes use t_old_corrected (their
        # stage coords were recorded in the previous placement); session flakes
        # use t (recorded in the current placement).
        #
        # t_old_corrected is t_old recomputed from its stored ref_marks using the
        # current formula.  This ensures stage_to_grid gives correct UV coords even
        # when t_old was saved with an earlier (potentially wrong) formula.
        t_old = self._sample.get('transform')
        t_old_corrected = None
        if t_old and 'ref_marks' in t_old and len(t_old.get('ref_marks', [])) == 2:
            try:
                t_old_corrected = sd.compute_transform(
                    t_old['ref_marks'][0], t_old['ref_marks'][1])
                t_old_corrected['r_deg_at_calibration'] = t_old.get('r_deg_at_calibration', 0.0)
            except Exception:
                t_old_corrected = None  # fall back to raw t_old if recompute fails

        for flake in self._sample.get('flakes', []):
            is_session = flake['id'] in self._session_flake_ids
            if t_old is None or is_session:
                t_stamp = t
            else:
                t_stamp = t_old_corrected or t_old
            try:
                gx, gy = sd.stage_to_grid(t_stamp,
                                           flake['stage_x_mm'],
                                           flake['stage_y_mm'])
                flake['grid_x_eff'] = round(gx, 6)
                flake['grid_y_eff'] = round(gy, 6)
            except Exception:
                pass

        # Re-project stage coords from local coords using the new transform so
        # that stage_x_mm/stage_y_mm stay meaningful for this placement.
        for flake in self._sample.get('flakes', []):
            gx = flake.get('grid_x_eff')
            gy = flake.get('grid_y_eff')
            if gx is not None and gy is not None:
                try:
                    sx, sy = sd.grid_to_stage(t, gx, gy)
                    flake['stage_x_mm'] = round(sx, 6)
                    flake['stage_y_mm'] = round(sy, 6)
                except Exception:
                    pass

        # Re-derive extents polygon in wafer-frame UV using the corrected old
        # transform, then re-project to stage coords for the current placement.
        ext = self._sample.get('extents', {})
        if ext and 'polygon_mm' in ext and t_old is not None:
            try:
                t_for_poly = t_old_corrected or t_old
                grid_poly = [[round(gx, 6), round(gy, 6)]
                             for gx, gy in (sd.stage_to_grid(t_for_poly, x, y)
                                            for x, y in ext['polygon_mm'])]
                ext['polygon_grid'] = grid_poly
                ext['polygon_mm'] = [[round(x, 4), round(y, 4)]
                                     for x, y in (sd.grid_to_stage(t, gx, gy)
                                                  for gx, gy in grid_poly)]
                self._sample['extents'] = ext
            except Exception:
                pass

        self._sample['transform'] = t
        self._transform_fresh = True
        self._session_flake_ids = set()
        sd.save_sample(_USERS_ROOT, self._sample)
        self._update_stage_markers()
        self._restore_extents()   # re-project extents to current placement
        rot   = t['rotation_deg']
        scale = t['scale']
        r_cal = t['r_deg_at_calibration']
        self._cs_status.setStyleSheet("")
        self._cs_status.setText(
            f"Saved.  Rotation {rot:+.2f}°  |  Scale {scale:.4f}"
            f"  |  R={r_cal:.3f}°"
            f"{'  ✓' if abs(scale - 1.0) < 0.05 else '  ⚠ scale unexpected'}")

    def _restore_coord_system(self):
        """Populate coordinate-system widgets from the loaded sample's transform."""
        self._ref_marks = [None, None]
        t = (self._sample or {}).get('transform')
        if t and 'ref_marks' in t:
            for n, ref in enumerate(t['ref_marks'][:2]):
                self._ref_marks[n] = ref
                xx    = ref['grid_xx']
                yy    = ref['grid_yy']
                q     = ref.get('quadrant')
                sx    = ref['stage_x_mm']
                sy    = ref['stage_y_mm']
                r_ref = ref.get('r_deg')
                q_str = f" [{q}]" if q else ""
                r_str = f"  R={r_ref:.3f}°" if r_ref is not None else ""
                self._ref_labels[n].setText(
                    f"Mark {n+1}: XX={xx:02d} YY={yy:02d}{q_str}  →  "
                    f"stage ({sx:.4f}, {sy:.4f}) mm{r_str}")
                # Pre-populate the quadrant combo so the user doesn't have to
                # remember to re-select it when de-staling the transform.
                q_combo = self._ref_quadrants[n]
                q_combo.blockSignals(True)
                q_combo.setCurrentText(q if q else "—")
                q_combo.blockSignals(False)
            rot   = t['rotation_deg']
            scale = t['scale']
            r_cal = t.get('r_deg_at_calibration')
            r_str = f"  |  R={r_cal:.3f}°" if r_cal is not None else ""
            self._cs_status.setText(
                f"⚠ Stale — re-set both marks at current positions and recompute."
                f"  (Last: Rotation {rot:+.2f}°  |  Scale {scale:.4f}{r_str})")
            self._cs_status.setStyleSheet("color: #cc7700;")
        else:
            for n, lbl in enumerate(self._ref_labels):
                lbl.setText(f"Mark {n+1}: not set")
                self._ref_quadrants[n].blockSignals(True)
                self._ref_quadrants[n].setCurrentText("—")
                self._ref_quadrants[n].blockSignals(False)
            self._cs_status.setText("")
            self._cs_status.setStyleSheet("")
        self._compute_transform_btn.setEnabled(
            self._ref_marks[0] is not None and self._ref_marks[1] is not None)

    def backfill_chip_coords(self):
        """Stamp chip_x_mm/chip_y_mm for flakes that have no chip-local coordinates.

        Only touches flakes where chip_x_mm is None.  Existing chip coords are
        mount-invariant and must not be overwritten: stage_x_mm is recorded in
        whatever mount frame was active when the flake was saved, so applying the
        current mount's inverse placement transform to it would give wrong results.
        """
        if self._sample is None:
            return
        updated = 0
        for flake in self._sample.get('flakes', []):
            if flake.get('chip_x_mm') is not None:
                continue   # already have chip coords — leave them alone
            self._stamp_chip_coords(flake)
            if flake.get('chip_x_mm') is not None:
                updated += 1
        if updated:
            import core.sample_data as _sd
            _sd.save_sample(_USERS_ROOT, self._sample)

    def _update_stage_markers(self):
        """Push current sample's flake positions and transform to the stage map."""
        if self.stage_controls is None:
            return
        flakes = (self._sample or {}).get('flakes', [])
        t = (self._sample or {}).get('transform')
        name = (self._sample or {}).get('name', '')

        # For flakes with chip-local coords, project to the current mount's stage
        # frame via the placement transform so the stage map stays correct after
        # remount.  The stage map's fallback path uses stage_x_mm/stage_y_mm, which
        # are first-mount values and go stale after any subsequent registration.
        resolved = []
        for flake in flakes:
            cx = flake.get('chip_x_mm')
            cy = flake.get('chip_y_mm')
            if cx is not None and cy is not None:
                try:
                    sx, sy = sd.chip_to_stage(self._sample, cx, cy)
                    if sx is not None:
                        flake = dict(flake)
                        flake['stage_x_mm'] = sx
                        flake['stage_y_mm'] = sy
                except Exception:
                    pass
            resolved.append(flake)

        self.stage_controls.set_flake_markers(resolved, transform=t,
                                               transform_fresh=self._transform_fresh,
                                               sample_name=name)

    def import_detected_candidates(self, candidates: list) -> int:
        """Import flake candidates from the detector into the active sample.

        Skips candidates whose detector `id` already appears in any flake's
        `detected_id` field.  Returns the number of newly-added flakes.
        """
        if self._sample is None:
            return 0

        existing_ids = {
            f.get('detected_id') for f in self._sample.get('flakes', [])
        }

        _N_LABEL = {1: "1L", 2: "2L", 3: "3L"}
        added = 0
        self._flake_table.blockSignals(True)
        for c in candidates:
            det_id = c.get('id', '')
            if det_id and det_id in existing_ids:
                continue

            fid   = sd.next_flake_id(self._sample['flakes'])
            N     = c.get('target_N', 1)
            bgr   = c.get('mean_contrast_bgr', [None, None, None])

            def _pct(v):
                return f"{v * 100:.1f}%" if v is not None else "?"

            notes = (
                f"Detected: {_N_LABEL.get(N, f'{N}L')} "
                f"score={c.get('score', 0):.3f} "
                f"sol={c.get('solidity', 0):.2f} "
                f"B={_pct(bgr[0])} G={_pct(bgr[1])} R={_pct(bgr[2])}"
            )

            flake = sd.new_flake(
                fid,
                _N_LABEL.get(N, f'{N}L'),
                c.get('x_mm', 0.0),
                c.get('y_mm', 0.0),
                c.get('z_mm', 0.0),
                c.get('mag', ''),
            )
            flake['source']     = 'auto'   # provenance: automatic detection
            flake['area_um2']    = c.get('area_um2')
            flake['circularity'] = c.get('circularity')
            flake['aspect_ratio'] = c.get('aspect_ratio')
            flake['solidity']    = c.get('solidity')
            flake['notes']      = notes
            flake['detected_id'] = det_id

            self._stamp_grid_coords(flake)
            self._stamp_chip_coords(flake)
            self._session_flake_ids.add(fid)
            self._sample['flakes'].append(flake)
            existing_ids.add(det_id)
            self._append_flake_row(flake)
            added += 1

        self._flake_table.blockSignals(False)
        if added > 0:
            sd.save_sample(_USERS_ROOT, self._sample)
            self._update_stage_markers()
        return added

    def _save_extents_to_sample(self, edges: dict, r_deg: float, polygon=None):
        """Called by stage_controls when wafer extents are detected.  Saves to sample JSON."""
        if self._sample is None:
            return
        extents = {
            'x_negative_mm':      edges.get('x_negative'),
            'x_positive_mm':      edges.get('x_positive'),
            'y_negative_mm':      edges.get('y_negative'),
            'y_positive_mm':      edges.get('y_positive'),
            'r_deg_at_detection': r_deg,
        }
        if polygon:
            extents['polygon_mm'] = [[round(x, 4), round(y, 4)] for x, y in polygon]
            # Only stamp local-frame coords if the transform is confirmed fresh —
            # a stale transform would map the current boundary to the wrong wafer position.
            if self._transform_fresh:
                t = self._sample.get('transform')
                if t:
                    try:
                        grid_poly = [[round(gx, 6), round(gy, 6)]
                                     for gx, gy in (sd.stage_to_grid(t, x, y) for x, y in polygon)]
                        extents['polygon_grid'] = grid_poly
                    except Exception:
                        pass
        self._sample['extents'] = extents
        sd.save_sample(_USERS_ROOT, self._sample)
