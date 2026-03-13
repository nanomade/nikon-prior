from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
import time
import numpy as np
import os

class PixelIntensityPanel(QWidget):
    """
    QWidget panel that lets you measure a 5×5 ROI intensity
    at the centre of the preview *only when you click a button*.
    - Displays mean & centre pixel intensity
    - Saves all data to data/pixel_data.csv
    """
    def __init__(self, motor_manager, preview_obj, parent=None):
        super().__init__(parent)
        self.mm = motor_manager
        self.preview = preview_obj

        self.setWindowTitle("Pixel Intensity Panel")
        self.label = QLabel("Click 'Measure Intensity' to record data")
        self.measure_btn = QPushButton("Measure Intensity")

        lay = QVBoxLayout()
        lay.addWidget(self.label)
        lay.addWidget(self.measure_btn)
        self.setLayout(lay)

        self.measure_btn.clicked.connect(self.measure_intensity)

        # Data folder & CSV
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, "pixel_data.csv")
        self.csv = open(self.csv_path, "w")

        header = ["time", "stage_x", "stage_y", "stage_z",
                  "mean_intensity", "centre_intensity"]
        for i in range(25):            # 25 pixels
            for c in ["B", "G", "R"]:
                header.append(f"p{i}_{c}")
        self.csv.write(",".join(header) + "\n")
        self.csv.flush()

    def measure_intensity(self):
        """Grab a frame and save intensity data."""
        frame = self.preview.get_frame()
        if frame is None:
            self.label.setText("No frame available.")
            return

        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        half = 6  # 5×5
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)

        roi = frame[y0:y1, x0:x1]
        mean_val = roi.mean()
        b, g, r = frame[cy, cx]
        centre_val = (int(b) + int(g) + int(r)) / 3.0

        pos = tuple(self.mm.get_position_units(ax) for ax in ("X", "Y", "Z"))

        self.label.setText(
            f"Z={pos[2]:.3f} mm | Mean={mean_val:.1f} | Centre={centre_val:.1f}"
        )

        flat_roi = roi.reshape(-1, 3)
        row = [f"{time.time():.3f}",
               f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
               f"{mean_val:.2f}", f"{centre_val:.2f}"]
        row += [f"{v:.0f}" for v in flat_roi.flatten()]
        self.csv.write(",".join(row) + "\n")
        self.csv.flush()


    def closeEvent(self, ev):
        if self.csv:
            self.csv.close()
        super().closeEvent(ev)
