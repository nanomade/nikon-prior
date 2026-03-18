# ui/edge_detection_panel.py
"""Automated wafer-edge detection panel.

Steps the XY stage outward from centre in four directions, watches frame
intensity, and detects where brightness drops below a threshold (wafer edge).
Results are passed directly to WaferMappingPanel.wafer_boundaries.

Ported from standa-stacker (no R axis on this microscope).
"""

import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class AutomatedSequenceWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str, int)

    def __init__(self, motor_manager, preview, step_size=0.1,
                 threshold=None, auto_threshold=True):
        super().__init__()
        self.motor_manager = motor_manager
        self.preview = preview
        self.step_size = step_size
        self.intensity_threshold = threshold
        self.auto_threshold = auto_threshold
        self.should_stop = False
        self.move_delay = 0.15   # seconds between step and intensity sample

    def stop(self):
        self.should_stop = True

    def _goto_stage_center(self):
        self.motor_manager.move_absolute_units('X', 0.0)
        self.motor_manager.move_absolute_units('Y', 0.0)

    def get_frame_intensity(self):
        try:
            frame = self.preview.get_frame()
            if frame is None:
                return 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            return float(np.mean(gray))
        except Exception:
            return 0

    def test_wafer_vs_background_intensity(self):
        self.progress.emit("Measuring centre intensity…", 5)
        try:
            self._goto_stage_center()
            time.sleep(2)
            center_intensity = self.get_frame_intensity()
            self.progress.emit(f"Centre intensity: {center_intensity:.1f}", 10)

            self.motor_manager.move_absolute_units('X', 10.0)
            time.sleep(1)
            outside_intensity = self.get_frame_intensity()
            self.progress.emit(f"X=10 intensity: {outside_intensity:.1f}", 15)
            self._goto_stage_center()
            time.sleep(1)

            if abs(center_intensity - outside_intensity) > 50:
                optimal = (center_intensity + outside_intensity) / 2
                self.progress.emit(
                    f"Auto threshold: {optimal:.1f} "
                    f"(wafer={center_intensity:.0f}, bg={outside_intensity:.0f})", 20)
            else:
                optimal = center_intensity * 0.5
                self.progress.emit(
                    f"Auto threshold: {optimal:.1f} (relative; background not found at X=10)", 20)

            return {'optimal_threshold': optimal,
                    'wafer_intensity': center_intensity,
                    'background_intensity': outside_intensity}
        except Exception as exc:
            print(f"[EdgeDetect] threshold test error: {exc}")
            return None

    def get_position_mm(self, axis):
        try:
            return self.motor_manager.get_position_units(axis)
        except Exception:
            return 0.0

    def find_edge(self, axis, step_size_mm):
        """Step along *axis* by *step_size_mm* (signed) until intensity drops.
        Returns edge position in mm, or None if not found."""
        step_count = 0
        max_total_steps = 300
        consecutive_dark = 0
        required_consecutive = 3
        edge_detected = False
        intensities = []
        positions = []

        while not self.should_stop and step_count < max_total_steps:
            intensity = self.get_frame_intensity()
            current_pos = self.get_position_mm(axis)
            intensities.append(intensity)
            positions.append(current_pos)

            if step_count % 20 == 0:
                print(f"  [{axis}] step {step_count}: pos={current_pos:.3f} mm  I={intensity:.1f}")

            if intensity < self.intensity_threshold:
                consecutive_dark += 1
                if consecutive_dark >= required_consecutive and not edge_detected:
                    edge_pos = current_pos - (required_consecutive - 1) * step_size_mm
                    edge_detected = True
                    self.progress.emit(f"{axis} edge at {edge_pos:.3f} mm", 0)
                    # Step a few more times to confirm solid background
                    for _ in range(5):
                        if self.should_stop:
                            break
                        self.motor_manager.move_units(axis, step_size_mm)
                        time.sleep(self.move_delay)
                    return edge_pos
            else:
                consecutive_dark = 0

            if not self.should_stop and not edge_detected:
                self.motor_manager.move_units(axis, step_size_mm)
                time.sleep(self.move_delay)
                step_count += 1

        if not edge_detected and positions and step_count > 50:
            return positions[-1]
        return None

    def run(self):
        try:
            self.progress.emit("Moving to stage centre…", 10)
            self._goto_stage_center()
            time.sleep(2)

            center_intensity = self.get_frame_intensity()
            self.progress.emit(f"Centre intensity: {center_intensity:.1f}", 15)

            if self.auto_threshold and self.intensity_threshold is None:
                self.progress.emit("Calculating auto threshold…", 25)
                result = self.test_wafer_vs_background_intensity()
                if result:
                    self.intensity_threshold = result['optimal_threshold']
                    self.progress.emit(f"Threshold: {self.intensity_threshold:.1f}", 30)
                else:
                    self.intensity_threshold = 100
                    self.progress.emit(f"Using default threshold: {self.intensity_threshold}", 30)
            else:
                self.progress.emit(f"Threshold: {self.intensity_threshold}", 30)

            edges_found = {}
            search_pattern = [
                ('X', 'x_positive',  abs(self.step_size)),
                ('X', 'x_negative', -abs(self.step_size)),
                ('Y', 'y_positive',  abs(self.step_size)),
                ('Y', 'y_negative', -abs(self.step_size)),
            ]
            progress_per_axis = 60 // len(search_pattern)
            current_progress = 40

            for axis, edge_key, step_size in search_pattern:
                if self.should_stop:
                    break
                self.progress.emit(f"Searching {edge_key}…", current_progress)
                edge_pos = self.find_edge(axis, step_size)
                if edge_pos is not None:
                    edges_found[edge_key] = edge_pos
                    self.progress.emit(f"{edge_key}: {edge_pos:+.3f} mm", current_progress + 10)
                else:
                    self.progress.emit(f"{edge_key}: not found", current_progress + 10)
                self._goto_stage_center()
                time.sleep(1)
                current_progress += progress_per_axis

            if not self.should_stop:
                self._goto_stage_center()
                time.sleep(1)

            all_keys = ('x_negative', 'x_positive', 'y_negative', 'y_positive')
            if all(e in edges_found for e in all_keys):
                wafer_width  = abs(edges_found['x_positive'] - edges_found['x_negative'])
                wafer_height = abs(edges_found['y_positive'] - edges_found['y_negative'])
            else:
                wafer_width = wafer_height = 0.0

            self.progress.emit("Complete.", 100)
            self.finished.emit({
                'status': 'success' if not self.should_stop else 'stopped',
                'edges_found': edges_found,
                'threshold_used': self.intensity_threshold,
                'wafer_width': wafer_width,
                'wafer_height': wafer_height,
                'auto_threshold': self.auto_threshold,
            })

        except Exception as exc:
            self.finished.emit({'status': 'error', 'error': str(exc)})


# ---------------------------------------------------------------------------
# Modal progress dialog
# ---------------------------------------------------------------------------

class FindEdgesDialog(QDialog):
    def __init__(self, motor_manager, preview, wafer_mapping_panel,
                 step_size, use_auto_threshold, manual_threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Finding Wafer Extents")
        self.setModal(True)
        self.wafer_mapping_panel = wafer_mapping_panel
        self.result = None
        self.worker = None

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Starting…")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(110)
        layout.addWidget(self.log)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)

        self.resize(400, 280)

        threshold = None if use_auto_threshold else manual_threshold
        self.worker = AutomatedSequenceWorker(
            motor_manager=motor_manager,
            preview=preview,
            step_size=step_size,
            threshold=threshold,
            auto_threshold=use_auto_threshold,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        QTimer.singleShot(200, self.worker.start)

    @pyqtSlot(str, int)
    def _on_progress(self, message, value):
        self.status_label.setText(message)
        if value > 0:
            self.progress_bar.setValue(value)

    @pyqtSlot(dict)
    def _on_finished(self, result):
        self.result = result
        self.cancel_btn.setText("Close")

        if result.get('status') == 'success':
            edges = result['edges_found']
            wafer_width  = result.get('wafer_width', 0)
            wafer_height = result.get('wafer_height', 0)

            self.status_label.setText("Complete.")
            name_map = {'x_positive': 'X+', 'x_negative': 'X−',
                        'y_positive': 'Y+', 'y_negative': 'Y−'}
            for k, v in edges.items():
                self.log.append(f"  {name_map.get(k, k)}: {v:+.3f} mm")
            if wafer_width > 0:
                self.log.append(f"  Size: {wafer_width:.2f} × {wafer_height:.2f} mm")

            all_found = all(e in edges for e in
                            ('x_negative', 'x_positive', 'y_negative', 'y_positive'))
            if all_found and self.wafer_mapping_panel is not None:
                self.wafer_mapping_panel.wafer_boundaries = (
                    edges['x_negative'], edges['x_positive'],
                    edges['y_negative'], edges['y_positive'],
                )
                self.wafer_mapping_panel.status_label.setText(
                    f"Boundaries set: X[{edges['x_negative']:+.2f}, {edges['x_positive']:+.2f}]  "
                    f"Y[{edges['y_negative']:+.2f}, {edges['y_positive']:+.2f}]"
                )
        elif result.get('status') == 'stopped':
            self.status_label.setText("Stopped.")
        else:
            self.status_label.setText(f"Error: {result.get('error', 'Unknown error')}")

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self.accept()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Launcher panel
# ---------------------------------------------------------------------------

class EdgeDetectionPanel(QWidget):
    def __init__(self, preview_window, motor_manager, wafer_mapping_panel=None):
        super().__init__()
        self.setWindowTitle("Find Wafer Extents")
        self.preview = preview_window
        self.motor_manager = motor_manager
        self.wafer_mapping_panel = wafer_mapping_panel

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Find Wafer Extents</b>"))

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step size:"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.05, 2.0)
        self.step_spin.setValue(0.1)
        self.step_spin.setSuffix(" mm")
        self.step_spin.setDecimals(2)
        step_row.addWidget(self.step_spin)
        step_row.addStretch()
        layout.addLayout(step_row)

        thresh_row = QHBoxLayout()
        self.auto_threshold_check = QCheckBox("Auto threshold")
        self.auto_threshold_check.setChecked(False)
        self.auto_threshold_check.toggled.connect(self._on_auto_toggled)
        thresh_row.addWidget(self.auto_threshold_check)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(100)
        thresh_row.addWidget(QLabel("Manual:"))
        thresh_row.addWidget(self.threshold_spin)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        self.start_btn = QPushButton("Find Wafer Extents")
        self.start_btn.clicked.connect(self._start)
        layout.addWidget(self.start_btn)

        self.result_label = QLabel("No edges detected yet.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        layout.addStretch()

    def _on_auto_toggled(self, checked):
        self.threshold_spin.setEnabled(not checked)

    def _start(self):
        dlg = FindEdgesDialog(
            motor_manager=self.motor_manager,
            preview=self.preview,
            wafer_mapping_panel=self.wafer_mapping_panel,
            step_size=self.step_spin.value(),
            use_auto_threshold=self.auto_threshold_check.isChecked(),
            manual_threshold=self.threshold_spin.value(),
            parent=self,
        )
        dlg.exec_()

        if dlg.result and dlg.result.get('status') == 'success':
            w = dlg.result.get('wafer_width', 0)
            h = dlg.result.get('wafer_height', 0)
            self.result_label.setText(
                f"Last result: {w:.2f} × {h:.2f} mm"
                if w > 0 else "Complete (edges not all found)."
            )
        elif dlg.result and dlg.result.get('status') == 'error':
            self.result_label.setText(f"Error: {dlg.result.get('error', '?')}")
