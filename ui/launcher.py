import sys

from camera_manager import create_camera_manager
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget,
)


def _heading(text):
    lbl = QLabel(text)
    font = QFont()
    font.setBold(True)
    font.setPointSize(8)
    lbl.setFont(font)
    lbl.setStyleSheet("color: #aaaaaa; margin-top: 6px;")
    return lbl


def _separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class LauncherWindow(QWidget):
    def __init__(self, preview, controls, stage_controls, position_manager,
                 focus_panel=None, autofocus_panel=None, focus_map_panel=None,
                 wafer_mapping_panel=None, index_mark_panel=None,
                 gamepad_panel=None, pixel_panel=None,
                 layer_contrast_panel=None, flat_field_panel=None,
                 edge_detection_panel=None, file_save_panel=None):
        super().__init__()
        self.setWindowTitle("Launcher")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.preview = preview
        self.controls = controls
        self.stage_controls = stage_controls
        self.position_manager = position_manager
        self.focus_panel = focus_panel
        self.autofocus_panel = autofocus_panel
        self.focus_map_panel = focus_map_panel
        self.wafer_mapping_panel = wafer_mapping_panel
        self.index_mark_panel = index_mark_panel
        self.gamepad_panel = gamepad_panel
        self.pixel_panel = pixel_panel
        self.layer_contrast_panel = layer_contrast_panel
        self.flat_field_panel = flat_field_panel
        self.edge_detection_panel = edge_detection_panel
        self.file_save_panel = file_save_panel

        layout = QVBoxLayout()
        layout.setSpacing(2)

        def _btn(label, slot):
            b = QPushButton(label)
            b.clicked.connect(slot)
            layout.addWidget(b)

        layout.addWidget(_heading("IMAGING"))
        _btn("Open Preview",             self._open_preview)
        _btn("Open Imaging Controls",    self.controls.show)
        if self.file_save_panel:
            _btn("File Save",            self.file_save_panel.show)
        if self.layer_contrast_panel:
            _btn("Open Layer Contrast",  self.layer_contrast_panel.show)
        if self.flat_field_panel:
            _btn("Flat-Field Correction", self.flat_field_panel.show)
        if self.pixel_panel:
            _btn("Pixel Intensity",      self.pixel_panel.show)

        layout.addWidget(_separator())
        layout.addWidget(_heading("STAGE"))
        _btn("Open Stage Controls",      self.stage_controls.show)
        _btn("Open Position Manager",    self._open_positions)
        if self.gamepad_panel:
            _btn("Open Gamepad",         self.gamepad_panel.show)

        layout.addWidget(_separator())
        layout.addWidget(_heading("FOCUS"))
        if self.focus_panel:
            _btn("Open Focus Panel",     self.focus_panel.show)
        if self.autofocus_panel:
            _btn("Open Autofocus",       self.autofocus_panel.show)
        if self.focus_map_panel:
            _btn("Open Focus Map",       self.focus_map_panel.show)

        layout.addWidget(_separator())
        layout.addWidget(_heading("SAMPLE"))
        if self.edge_detection_panel:
            _btn("Find Wafer Extents",   self.edge_detection_panel.show)
        if self.wafer_mapping_panel:
            _btn("Open Wafer Mapping",   self.wafer_mapping_panel.show)
        if self.index_mark_panel:
            _btn("Index Mark Navigator", self.index_mark_panel.show)

        layout.addWidget(_separator())
        _btn("Close All and Exit",       self._close_all)

        self.setLayout(layout)
        self.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width(), screen.top())

    def _open_preview(self):
        if self.preview.cap is None or not self.preview.cap.connected():
            self.preview.cap = create_camera_manager()
            self.preview.native_width  = self.preview.cap.native_width
            self.preview.native_height = self.preview.cap.native_height
        if not hasattr(self.preview, "timer"):
            self.preview.timer = QTimer()
            self.preview.timer.timeout.connect(self.preview.update_frame)
        if not self.preview.timer.isActive():
            fps = getattr(self.preview.controller, "target_fps", 30)
            self.preview.timer.start(int(1000 / max(1, int(fps))))
        self.preview.show()

    def _open_positions(self):
        self.position_manager.show()
        self.position_manager.raise_()
        self.position_manager.activateWindow()

    def _close_all(self):
        for p in [
            self.preview, self.controls, self.stage_controls,
            self.position_manager, self.focus_panel, self.autofocus_panel,
            self.focus_map_panel, self.wafer_mapping_panel,
            self.index_mark_panel, self.gamepad_panel, self.pixel_panel,
            self.layer_contrast_panel, self.flat_field_panel,
            self.edge_detection_panel, self.file_save_panel,
        ]:
            if p is not None:
                try:
                    p.close()
                except Exception:
                    pass
        self.close()
        sys.exit(0)
