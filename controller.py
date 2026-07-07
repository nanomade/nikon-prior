# camera_stage_refactor/controller.py

from PyQt5.QtCore import QObject, pyqtSignal
import time

class Controller(QObject):
    exposure_changed = pyqtSignal(float)
    gain_changed = pyqtSignal(float)
    auto_exposure_changed = pyqtSignal(bool)
    magnification_changed = pyqtSignal(str)
    show_scale_bar_changed = pyqtSignal(bool)
    color_changed = pyqtSignal(str)
    measure_mode_changed = pyqtSignal(bool)
    measure_area_mode_changed = pyqtSignal(bool)
    crosshair_visible_changed = pyqtSignal(bool)
    center_crosshair_changed = pyqtSignal(bool)
    full_crosshair_changed = pyqtSignal(bool)
    brightness_changed = pyqtSignal(int)    # display LUT: additive offset -64..64
    contrast_changed = pyqtSignal(int)      # display LUT: /32 multiplier, 32 = neutral
    gamma_changed = pyqtSignal(int)         # display LUT: /100 gamma, 100 = neutral
    sharpness_changed = pyqtSignal(int)     # reserved — not implemented on the Alvium
    hud_changed = pyqtSignal(bool)
    resolution_changed = pyqtSignal(tuple)
    native_zoom_toggled = pyqtSignal(bool)
    wb_red_changed = pyqtSignal(float)      # manual Red BalanceRatio (0.25–8.0)
    wb_blue_changed = pyqtSignal(float)     # manual Blue BalanceRatio (0.25–8.0)
    binning_changed = pyqtSignal(int)
    zoom_under_cursor_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.magnification = "10x"
        self.show_scale_bar = True
        self.scale_bar_color = "White"
        self.measure_mode = False
        self.native_zoom = False

        self.last_time = time.time()
        self.frame_count = 0
        self.measured_fps = 0
