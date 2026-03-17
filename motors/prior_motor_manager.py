"""Prior Scientific ProScan III motor manager.

Speaks the ProScan III ASCII serial protocol and exposes the same API as
MockMotorManager so all UI panels work without modification.

ProScan III serial defaults: 9600 baud, 8N1, no flow control.
(Can be changed on the controller front panel — match whatever is set.)

Key commands used:
  P            — query position → "x,y,z"  (encoder counts)
  G x,y,z      — absolute move (all axes)
  GR dx,dy,dz  — relative move (all axes)
  VS n         — set speed (1–100, percent of configured max)
  V            — query speed
  H            — hard halt (immediate stop)
  K            — soft stop (decelerate to stop)
  Z            — zero current position (set 0,0,0)
  !            — busy? → "R" if moving, "0" if idle
  STAT         — status byte

All positions are in encoder counts internally; the step_config
calibration (mm per count) converts to physical mm for the UI.
"""

import json
import os
import time
import re
import threading

import serial
import serial.tools.list_ports

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "step_config.json")
_AXES = ("X", "Y", "Z")


class PriorMotorManager:
    """Interface to a Prior ProScan III XYZ stage."""

    def __init__(self, port: str | None = None, baudrate: int = 9600,
                 timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: serial.Serial | None = None
        self._lock = threading.Lock()

        self.step_config = self._load_config()
        # Cached positions (counts) — updated from hardware on connect
        self._pos = {ax: 0 for ax in _AXES}

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH) as fh:
                return json.load(fh)
        # Fallback defaults: 0.1 µm per encoder count (Prior standard)
        return {
            "X": {"step": 0.0001, "invert": 1,  "unit": "mm",
                  "min_mm": -50.0, "max_mm": 50.0},
            "Y": {"step": 0.0001, "invert": -1, "unit": "mm",
                  "min_mm": -37.0, "max_mm": 37.0},
            "Z": {"step": 0.0001, "invert": 1,  "unit": "mm",
                  "min_mm":   0.0, "max_mm": 25.0},
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the serial connection and verify the controller responds."""
        port = self.port or self._autodetect_port()
        if port is None:
            raise RuntimeError("ProScan III: no serial port found")
        self.ser = serial.Serial(
            port=port, baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout, write_timeout=1.0,
        )
        self.port = port
        self.ser.reset_input_buffer()
        # Ping: read position; raises if no response
        self._refresh_positions()
        return True

    def _autodetect_port(self) -> str | None:
        """Prefer ports whose description mentions Prior or FTDI/USB-serial."""
        for p in serial.tools.list_ports.comports():
            desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
            if "prior" in desc or "proscan" in desc:
                return p.device
        # Broader fallback: any USB-serial adapter
        for p in serial.tools.list_ports.comports():
            desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
            if "usb serial" in desc or "ftdi" in desc or "cp210" in desc:
                return p.device
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ports[0] if ports else None

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ------------------------------------------------------------------
    # Low-level serial
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> str | None:
        if not self.connected():
            return None
        with self._lock:
            try:
                self.ser.write((cmd.strip() + "\r").encode("ascii"))
                self.ser.flush()
                raw = self.ser.read_until(b'\r')
                if not raw:
                    return None
                return raw.decode("ascii", errors="ignore").strip()
            except Exception as exc:
                print(f"[ProScan] serial error on '{cmd}': {exc}")
                return None

    def _wait_idle(self, poll_ms: int = 50, timeout_s: float = 60.0):
        """Block until the stage reports idle or timeout expires."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = self._send("!")
            if resp == "0":
                return
            time.sleep(poll_ms / 1000.0)

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def _refresh_positions(self):
        """Read hardware position and update internal cache."""
        resp = self._send("P")
        if resp is None:
            return
        parts = resp.split(",")
        if len(parts) >= 3:
            try:
                self._pos["X"] = int(parts[0])
                self._pos["Y"] = int(parts[1])
                self._pos["Z"] = int(parts[2])
            except ValueError:
                pass

    def get_position(self, axis: str) -> int:
        """Return current position in encoder counts."""
        self._refresh_positions()
        return self._pos.get(axis.upper(), 0)

    def get_position_units(self, axis: str) -> float:
        """Return current position in mm."""
        ax = axis.upper()
        cfg = self.step_config.get(ax, {})
        step = cfg.get("step", 0.0001)
        invert = cfg.get("invert", 1)
        return self.get_position(ax) * step * invert

    # ------------------------------------------------------------------
    # Movement — internal (counts)
    # ------------------------------------------------------------------

    def _move_absolute_counts(self, x: int | None = None,
                               y: int | None = None,
                               z: int | None = None,
                               wait: bool = True):
        """Send G or T command for absolute moves in counts."""
        if x is None:
            x = self._pos["X"]
        if y is None:
            y = self._pos["Y"]
        if z is None:
            z = self._pos["Z"]
        self._send(f"G {x},{y},{z}")
        if wait:
            self._wait_idle()
        self._refresh_positions()

    def _move_relative_counts(self, dx: int = 0, dy: int = 0, dz: int = 0,
                               wait: bool = True):
        self._send(f"GR {dx},{dy},{dz}")
        if wait:
            self._wait_idle()
        self._refresh_positions()

    # ------------------------------------------------------------------
    # Public movement API  (mirrors MotorManager / MockMotorManager)
    # ------------------------------------------------------------------

    def move(self, axis: str, steps: int, wait: bool = True):
        """Relative move in encoder counts."""
        ax = axis.upper()
        if ax == "X":
            self._move_relative_counts(dx=steps, wait=wait)
        elif ax == "Y":
            self._move_relative_counts(dy=steps, wait=wait)
        elif ax == "Z":
            self._move_relative_counts(dz=steps, wait=wait)

    def move_absolute(self, axis: str, steps: int, wait: bool = True):
        """Absolute move in encoder counts."""
        ax = axis.upper()
        if ax == "X":
            self._move_absolute_counts(x=steps, wait=wait)
        elif ax == "Y":
            self._move_absolute_counts(y=steps, wait=wait)
        elif ax == "Z":
            self._move_absolute_counts(z=steps, wait=wait)

    def move_units(self, axis: str, amount_mm: float, wait: bool = True):
        """Relative move in mm."""
        ax = axis.upper()
        cfg = self.step_config.get(ax, {})
        step = cfg.get("step", 0.0001)
        invert = cfg.get("invert", 1)
        counts = int(round(amount_mm / step * invert))
        self.move(ax, counts, wait=wait)

    def move_absolute_units(self, axis: str, pos_mm: float, wait: bool = True):
        """Absolute move in mm."""
        ax = axis.upper()
        cfg = self.step_config.get(ax, {})
        step = cfg.get("step", 0.0001)
        invert = cfg.get("invert", 1)
        counts = int(round(pos_mm / step * invert))
        self.move_absolute(ax, counts, wait=wait)

    def home(self, axis: str | None = None):
        """Zero position without moving (set current as origin).
        Prior 'Z' command sets current position as 0,0,0.
        Physical homing (move to hardware limit) is not supported here.
        """
        self._send("Z")
        self._pos = {ax: 0 for ax in _AXES}

    def stop(self):
        """Soft stop (decelerate)."""
        self._send("K")

    def halt(self):
        """Hard stop (immediate)."""
        self._send("H")

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    def get_speed(self, axis: str | None = None) -> int:
        """Return current speed (1–100 percent)."""
        resp = self._send("V")
        if resp:
            m = re.search(r"\d+", resp)
            if m:
                return int(m.group())
        return 50  # default

    def set_speed(self, speed_pct: int, axis: str | None = None):
        """Set speed as a percentage of maximum (1–100).
        ProScan III applies the same speed to all axes simultaneously.
        """
        speed_pct = max(1, min(100, int(speed_pct)))
        self._send(f"VS {speed_pct}")
