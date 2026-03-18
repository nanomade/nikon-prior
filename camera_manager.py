"""Allied Vision Alvium USB3 camera manager.

Wraps the Vimba X / VmbPy SDK and exposes two operating modes:

  Live mode   — free-running, BayerRG8, reduced resolution via decimation
                for real-time preview in the Qt UI.
  Capture mode — software-triggered, BayerRG12 (or Mono12), full resolution
                 for high-quality image acquisition.

The public API is intentionally close to what preview.py expects from
cv2.VideoCapture, so the existing UI requires minimal changes:

    cam = AlviumCameraManager()
    cam.connect()

    # Live (replaces cap.read()):
    cam.set_live_mode()
    ok, frame = cam.read()          # returns (True, H×W×3 uint8 BGR numpy array)

    # Capture (replaces a triggered grab):
    cam.set_capture_mode()
    ok, frame = cam.read()          # returns (True, H×W×3 uint16 BGR numpy array)

    cam.close()

Requires: pip install vmbpy
Install the USB transport layer first (Linux):
    sudo ./VimbaX_<ver>/cti/VimbaUSBTL_Install.sh
"""

import logging
import threading
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Sensor native resolution
SENSOR_WIDTH  = 2464
SENSOR_HEIGHT = 2056

# Default live-feed reduction factor
DEFAULT_LIVE_FACTOR = 4   # 4× → 616×514

try:
    import vmbpy
    VMBPY_AVAILABLE = True
except ImportError:
    VMBPY_AVAILABLE = False
    logger.warning("vmbpy not available — AlviumCameraManager will not function")


class AlviumCameraManager:
    """Interface to an Allied Vision Alvium 1800 U-508c via Vimba X."""

    def __init__(self):
        self._vmb: "vmbpy.VmbSystem | None" = None
        self._cam: "vmbpy.Camera | None" = None
        self._mode: str = "live"
        self._live_factor: int = DEFAULT_LIVE_FACTOR

        # Thread-safe frame buffer for streaming live mode
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._streaming: bool = False

        # Expose resolution so preview.py can read native_width / native_height
        self.native_width  = SENSOR_WIDTH  // DEFAULT_LIVE_FACTOR
        self.native_height = SENSOR_HEIGHT // DEFAULT_LIVE_FACTOR

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the Vimba system and the first detected Alvium camera."""
        if not VMBPY_AVAILABLE:
            raise RuntimeError("vmbpy is not installed")

        self._vmb = vmbpy.VmbSystem.get_instance()
        self._vmb.__enter__()

        cameras = self._vmb.get_all_cameras()
        if not cameras:
            raise RuntimeError("No Allied Vision cameras found")

        self._cam = cameras[0]
        self._cam.__enter__()

        logger.info(
            "Alvium connected: %s  firmware %s",
            self._cam.get_id(),
            self._cam.get_feature_by_name('DeviceFirmwareVersion').get(),
        )

        self.set_live_mode()
        return True

    def connected(self) -> bool:
        return self._cam is not None

    def close(self):
        self._stop_streaming()
        if self._cam:
            try:
                self._cam.__exit__(None, None, None)
            except Exception:
                pass
            self._cam = None
        if self._vmb:
            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass
            self._vmb = None

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_live_mode(self, factor: int | None = None):
        """Free-running, 8-bit Bayer, reduced resolution for preview.

        factor: reduction factor (1=full, 2=2×, 4=4×).
                If None, reuse the last-set factor.
        """
        cam = self._cam
        if cam is None:
            return

        self._stop_streaming()

        if factor is not None:
            self._live_factor = factor

        cam.get_feature_by_name("TriggerMode").set("Off")

        # Pixel format: raw 8-bit Bayer (debayered in read())
        try:
            cam.get_feature_by_name("PixelFormat").set("BayerRG8")
        except Exception:
            cam.get_feature_by_name("PixelFormat").set("Rgb8")

        self._apply_live_resolution(cam, self._live_factor)

        # Start continuous streaming for maximum FPS
        self._latest_frame = None
        try:
            cam.start_streaming(self._frame_callback, buffer_count=10)
            self._streaming = True
            logger.info(
                "Live mode: %dx%d BayerRG8 streaming (factor=%d)",
                self.native_width, self.native_height, self._live_factor,
            )
        except Exception as exc:
            logger.warning(
                "start_streaming failed (%s) — falling back to get_frame()", exc
            )
            self._streaming = False

        self._mode = "live"

    def set_live_binning(self, factor: int):
        """Change the live preview reduction factor on the fly."""
        if self._mode == "live":
            self.set_live_mode(factor=factor)
        else:
            self._live_factor = factor

    def set_capture_mode(self):
        """Software-triggered, 12-bit Bayer, full resolution for acquisition."""
        cam = self._cam
        if cam is None:
            return

        self._stop_streaming()

        # Reset decimation and binning to full resolution
        for feat in ("DecimationHorizontal", "DecimationVertical",
                     "BinningHorizontal",   "BinningVertical"):
            try:
                cam.get_feature_by_name(feat).set(1)
            except Exception:
                pass

        self.native_width  = SENSOR_WIDTH
        self.native_height = SENSOR_HEIGHT

        # 12-bit Bayer
        try:
            cam.get_feature_by_name("PixelFormat").set("BayerRG12")
        except Exception:
            logger.warning("BayerRG12 not available, falling back to BayerRG8")
            cam.get_feature_by_name("PixelFormat").set("BayerRG8")

        # Software trigger
        cam.get_feature_by_name("TriggerSelector").set("FrameStart")
        cam.get_feature_by_name("TriggerMode").set("On")
        cam.get_feature_by_name("TriggerSource").set("Software")

        self._mode = "capture"
        logger.info(
            "Capture mode: %dx%d BayerRG12 software-triggered",
            self.native_width, self.native_height,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_streaming(self):
        if self._streaming and self._cam:
            try:
                self._cam.stop_streaming()
            except Exception:
                pass
            self._streaming = False

    def _apply_live_resolution(self, cam, factor: int):
        """Try decimation then binning for the requested factor."""
        if factor == 1:
            for feat in ("DecimationHorizontal", "DecimationVertical",
                         "BinningHorizontal",   "BinningVertical"):
                try:
                    cam.get_feature_by_name(feat).set(1)
                except Exception:
                    pass
            self.native_width  = SENSOR_WIDTH
            self.native_height = SENSOR_HEIGHT
            logger.info("Live mode: full resolution (no binning)")
            return

        # Try decimation first
        try:
            cam.get_feature_by_name("DecimationHorizontal").set(factor)
            cam.get_feature_by_name("DecimationVertical").set(factor)
            self.native_width  = SENSOR_WIDTH  // factor
            self.native_height = SENSOR_HEIGHT // factor
            logger.info("Decimation %dx applied for live preview", factor)
            return
        except Exception:
            logger.warning("Decimation %dx not supported, trying binning", factor)

        # Fall back to binning
        try:
            cam.get_feature_by_name("BinningHorizontal").set(factor)
            cam.get_feature_by_name("BinningVertical").set(factor)
            self.native_width  = SENSOR_WIDTH  // factor
            self.native_height = SENSOR_HEIGHT // factor
            logger.info("Binning %dx applied for live preview", factor)
            return
        except Exception:
            logger.warning("Binning %dx not supported either — full resolution", factor)

        self.native_width  = SENSOR_WIDTH
        self.native_height = SENSOR_HEIGHT

    def _frame_callback(self, cam, stream, frame):
        """Vimba streaming callback — runs in Vimba's internal thread."""
        if frame.get_status() == vmbpy.FrameStatus.Complete:
            try:
                debayered = self._debayer(frame)
                with self._frame_lock:
                    self._latest_frame = debayered
            except Exception as exc:
                logger.warning("Frame callback error: %s", exc)
        cam.queue_frame(frame)

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab one frame and return (success, BGR numpy array).

        Live mode:   uint8,  shape (H, W, 3)
        Capture mode: uint16, shape (H, W, 3)  — scaled from 12-bit
        """
        if self._cam is None:
            return False, None

        if self._mode == "live" and self._streaming:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy()
            return False, None

        # Capture mode (or live fallback without streaming)
        try:
            timeout = 5000 if self._mode == "capture" else 2000
            frame = self._cam.get_frame(timeout_ms=timeout)
            return True, self._debayer(frame)
        except Exception as exc:
            logger.warning("Frame grab failed: %s", exc)
            return False, None

    def _debayer(self, frame: "vmbpy.Frame") -> np.ndarray:
        """Convert a raw Bayer frame to a BGR numpy array."""
        raw = frame.as_numpy_ndarray()

        fmt = frame.get_pixel_format()

        if fmt == vmbpy.PixelFormat.BayerRG8:
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
            return bgr

        elif fmt == vmbpy.PixelFormat.BayerRG12:
            # raw arrives as uint16 with values in [0, 4095]
            # Debayer at 16-bit then scale to full uint16 range for consistency
            bgr16 = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
            bgr16 = (bgr16 * 16).astype(np.uint16)   # 12-bit → 16-bit
            return bgr16

        elif fmt in (vmbpy.PixelFormat.Rgb8, vmbpy.PixelFormat.Bgr8):
            # Already debayered by camera
            if fmt == vmbpy.PixelFormat.Rgb8:
                return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            return raw

        else:
            logger.warning("Unhandled pixel format %s — returning raw", fmt)
            return raw

    # ------------------------------------------------------------------
    # Exposure / gain helpers
    # ------------------------------------------------------------------

    def set_exposure_us(self, exposure_us: float):
        """Set exposure time in microseconds."""
        if self._cam:
            self._cam.get_feature_by_name("ExposureTime").set(float(exposure_us))

    def get_exposure_us(self) -> float:
        if self._cam:
            return self._cam.get_feature_by_name("ExposureTime").get()
        return 0.0

    def set_gain_db(self, gain_db: float):
        """Set analogue gain in dB."""
        if self._cam:
            self._cam.get_feature_by_name("Gain").set(float(gain_db))

    def get_gain_db(self) -> float:
        if self._cam:
            return self._cam.get_feature_by_name("Gain").get()
        return 0.0

    def set_auto_exposure(self, enabled: bool):
        mode = "Continuous" if enabled else "Off"
        if self._cam:
            self._cam.get_feature_by_name("ExposureAuto").set(mode)

    # ------------------------------------------------------------------
    # Convenience: save a capture-mode frame to disk
    # ------------------------------------------------------------------

    def capture_to_file(self, path: str) -> bool:
        """Switch to capture mode, grab one frame, save as 16-bit TIFF."""
        was_live = self._mode == "live"
        if was_live:
            self.set_capture_mode()

        ok, frame = self.read()
        if ok and frame is not None:
            cv2.imwrite(path, frame)
            logger.info("Saved capture to %s", path)

        if was_live:
            self.set_live_mode()

        return ok


class MockCameraManager:
    """Drop-in replacement when no Alvium camera is available.

    Generates a synthetic noise frame so the rest of the UI can run offline.
    """

    def __init__(self):
        self._live_factor = DEFAULT_LIVE_FACTOR
        self.native_width  = SENSOR_WIDTH  // DEFAULT_LIVE_FACTOR
        self.native_height = SENSOR_HEIGHT // DEFAULT_LIVE_FACTOR
        self._mode = "live"

    def connect(self) -> bool:
        logger.info("MockCameraManager active (no hardware)")
        return True

    def connected(self) -> bool:
        return True

    def close(self):
        pass

    def set_live_mode(self, factor: int | None = None):
        if factor is not None:
            self._live_factor = factor
        self.native_width  = SENSOR_WIDTH  // self._live_factor
        self.native_height = SENSOR_HEIGHT // self._live_factor
        self._mode = "live"

    def set_live_binning(self, factor: int):
        self._live_factor = factor
        if self._mode == "live":
            self.native_width  = SENSOR_WIDTH  // factor
            self.native_height = SENSOR_HEIGHT // factor

    def set_capture_mode(self):
        self.native_width  = SENSOR_WIDTH
        self.native_height = SENSOR_HEIGHT
        self._mode = "capture"

    def read(self) -> tuple[bool, np.ndarray]:
        h, w = self.native_height, self.native_width
        dtype = np.uint8 if self._mode == "live" else np.uint16
        frame = np.random.randint(80, 180, (h, w, 3), dtype=dtype)
        return True, frame

    def set_exposure_us(self, _): pass
    def get_exposure_us(self): return 10000.0
    def set_gain_db(self, _): pass
    def get_gain_db(self): return 0.0
    def set_auto_exposure(self, _): pass
    def capture_to_file(self, path): return False


def create_camera_manager():
    """Return a real AlviumCameraManager if hardware is found, else Mock."""
    try:
        cam = AlviumCameraManager()
        cam.connect()
        return cam
    except Exception as exc:
        logger.warning("Alvium camera not available (%s) — using mock camera", exc)
        return MockCameraManager()
