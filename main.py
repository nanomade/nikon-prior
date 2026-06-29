"""Nikon / Prior ProScan III microscope control application.

Entry point.  Run with:
    python main.py

Note on Qt / cv2 conflict
--------------------------
opencv-python bundles its own Qt and registers cv2/qt/plugins as a Qt
platform-plugin search path at import time.  If cv2 is imported before
QApplication is created, its bundled (often ABI-incompatible) xcb plugin
can win the search and crash the app with:
  "Could not load the Qt platform plugin xcb … Aborted (core dumped)"
The moveToThread warning at startup is a related symptom (cv2 creates Qt
objects internally during import when no QApplication exists yet).

Fix applied here: create QApplication before importing any module that
imports cv2 at module level (stage_controls, autofocus_panel, etc.).
Once QApplication is constructed, PyQt5's platform plugin is already
loaded and cv2's path registration has no effect.

Permanent alternative: pip install opencv-python-headless
(headless OpenCV has no bundled Qt, so the conflict never arises).
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

# ── Create QApplication NOW, before any cv2-importing module is loaded ──────
# This must come before all other local imports.
_qapp = QApplication(sys.argv)

from PyQt5.QtCore import QPoint

from controller import Controller
from motors.factory import create_motor_manager
from pixel_intensity_panel import PixelIntensityPanel
from ui.app_shell import AppShell
from ui.autofocus_panel import AutoFocusPanel
from ui.controls import ControlWindow
from ui.edge_detection_panel import EdgeDetectionPanel
from ui.file_save_panel import FileSavePanel
from ui.flat_field_panel import FlatFieldPanel
from ui.histogram_widget import LiveHistogram
from ui.focus_map_panel import FocusMapPanel
from ui.focus_panel import FocusPanel
from ui.gamepad_panel import GamepadPanel
from ui.index_mark_panel import IndexMarkPanel
from ui.launcher import LauncherWindow
from ui.layer_contrast_panel import LayerContrastPanel
from ui.preview import PreviewWindow
from ui.sample_manager_panel import SampleManagerPanel
from ui.stage_controls import PositionManagerWindow, StageControlWindow
from ui.wafer_mapping_panel import WaferMappingPanel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tile_windows(widgets, start=QPoint(40, 40), gap=20):
    """Lay out widgets left-to-right, wrapping within screen bounds."""
    screen_rect = QApplication.primaryScreen().availableGeometry()
    x, y, row_h = start.x(), start.y(), 0
    for w in widgets:
        hint = w.sizeHint()
        if not hint.isEmpty():
            w.resize(hint)
        if x + w.width() > screen_rect.right() - gap:
            x = start.x()
            y += row_h + gap
            row_h = 0
        w.move(x, y)
        x += w.width() + gap
        row_h = max(row_h, w.height())


class Application:
    def __init__(self):
        self.app = QApplication.instance()  # reuse the one created at module level
        self.motor_manager = create_motor_manager()
        self.controller = Controller()
        self._init_windows()

    def _init_windows(self):
        mm = self.motor_manager

        self.preview = PreviewWindow(self.controller)
        self.preview.motor_manager = mm
        self.controls = ControlWindow(self.controller, self.preview)

        self.stage_controls = StageControlWindow(self.preview, mm)
        self.preview.stage_controls = self.stage_controls
        self.position_manager = PositionManagerWindow(self.stage_controls)
        self.stage_controls.position_manager = self.position_manager

        self.focus_panel = FocusPanel(self.stage_controls)
        self.stage_controls.focus_panel = self.focus_panel

        self.autofocus_panel = AutoFocusPanel(
            motor_manager=mm,
            preview_obj=self.preview,
            stage_controls=self.stage_controls,
        )
        self.preview.autofocus_panel = self.autofocus_panel  # Enter = autofocus
        self.controller.magnification_changed.connect(
            self.autofocus_panel.apply_defaults_for_mag
        )

        self.focus_map_panel = FocusMapPanel(
            motor_manager=mm,
            preview=self.preview,
            autofocus_panel=self.autofocus_panel,
            wafer_mapping_panel=None,
        )

        self.wafer_mapping_panel = WaferMappingPanel(
            self.preview, mm, self.stage_controls,
            autofocus_panel=self.autofocus_panel,
            focus_map_panel=self.focus_map_panel,
        )
        self.focus_map_panel.wafer_mapping_panel = self.wafer_mapping_panel

        self.histogram = LiveHistogram(self.preview)
        self.layer_contrast_panel = LayerContrastPanel(self.preview)
        self.flat_field_panel = FlatFieldPanel(self.preview, mm)
        self.pixel_panel = PixelIntensityPanel(mm, self.preview)
        self.index_mark_panel = IndexMarkPanel(
            self.preview, mm, self.stage_controls
        )

        self.edge_detection_panel = EdgeDetectionPanel(
            self.preview, mm,
            wafer_mapping_panel=self.wafer_mapping_panel,
        )

        self.file_save_panel = FileSavePanel(
            self.preview, mm, self.controls,
        )

        self.sample_manager_panel = SampleManagerPanel(
            preview=self.preview, motor_manager=mm, controls=self.controls,
            index_mark_panel=self.index_mark_panel,
            stage_controls=self.stage_controls,
            edge_detection_panel=self.edge_detection_panel,
        )

        self.gamepad_panel = GamepadPanel(
            self.stage_controls, self.controller,
            autofocus_panel=self.autofocus_panel,
            preview=self.preview,
            controls=self.controls,
        )

        self.launcher = LauncherWindow(
            self.preview,
            self.controls,
            self.stage_controls,
            self.position_manager,
            focus_panel=self.focus_panel,
            autofocus_panel=self.autofocus_panel,
            focus_map_panel=self.focus_map_panel,
            wafer_mapping_panel=self.wafer_mapping_panel,
            index_mark_panel=self.index_mark_panel,
            gamepad_panel=self.gamepad_panel,
            pixel_panel=self.pixel_panel,
            layer_contrast_panel=self.layer_contrast_panel,
            flat_field_panel=self.flat_field_panel,
            edge_detection_panel=self.edge_detection_panel,
            file_save_panel=self.file_save_panel,
        )

        # Three-column shell is the primary window.  The Launcher and all
        # floating panels remain available (opened on demand from the shell).
        self.shell = AppShell(
            preview=self.preview,
            launcher=self.launcher,
            controls=self.controls,
            stage_controls=self.stage_controls,
            position_manager=self.position_manager,
            autofocus_panel=self.autofocus_panel,
            focus_panel=self.focus_panel,
            focus_map_panel=self.focus_map_panel,
            flat_field_panel=self.flat_field_panel,
            layer_contrast_panel=self.layer_contrast_panel,
            index_mark_panel=self.index_mark_panel,
            edge_detection_panel=self.edge_detection_panel,
            wafer_mapping_panel=self.wafer_mapping_panel,
            file_save_panel=self.file_save_panel,
            gamepad_panel=self.gamepad_panel,
            pixel_panel=self.pixel_panel,
            sample_manager_panel=self.sample_manager_panel,
            histogram=self.histogram,
        )

    def run(self):
        self.app.setQuitOnLastWindowClosed(False)

        # App-level input routing: keyboard stage jog / focus / autofocus work
        # regardless of which widget has focus, and the wheel doesn't hijack
        # spin boxes in the side panels.  Keep references so they aren't GC'd.
        from ui.input_routing import GlobalKeyRouter, SpinBoxWheelGuard
        self._key_router = GlobalKeyRouter(
            self.preview, motor_manager=self.motor_manager,
            gamepad_panel=self.gamepad_panel,
        )
        self._wheel_guard = SpinBoxWheelGuard()
        self.app.installEventFilter(self._key_router)
        self.app.installEventFilter(self._wheel_guard)

        self.shell.destroyed.connect(self.app.quit)
        screen = QApplication.primaryScreen().availableGeometry()
        self.shell.move(screen.left(), screen.top())
        self.shell.resize(min(1700, screen.width()), min(1000, screen.height()))
        self.shell.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    Application().run()
