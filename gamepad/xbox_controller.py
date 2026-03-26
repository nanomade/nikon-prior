"""Xbox controller input using direct /dev/input/js0 for buttons and pygame for axes.

Handles connect/disconnect gracefully. Deadzone is applied in get_axis().
Triggers are reported as 0..1 (SDL2 returns -1..1 for triggers on axis 4/5).
"""

import os
import struct
import sys

if sys.platform == "win32":
    fcntl = None
else:
    import fcntl

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

_JS_DEV = '/dev/input/js0'
_JS_EVENT_FMT = 'IhBB'   # time(u32), value(s16), type(u8), number(u8)
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_INIT   = 0x80

# SDL2 axis indices for Xbox One S Controller on Linux (xpad/SDL2)
# Triggers rest at -1.0 (released) and go to +1.0 (fully pressed)
_AXIS_MAP = {
    "left_x": 0,
    "left_y": 1,
    "trigger_l": 2,
    "right_x": 3,
    "right_y": 4,
    "trigger_r": 5,
}

# SDL2 button indices for Xbox One S on Linux (11 buttons, index 8 = guide)
_BUTTON_MAP = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "lb": 4,
    "rb": 5,
    "back": 6,
    "start": 7,
    "ls": 9,   # left stick click
    "rs": 10,  # right stick click
}

_TRIGGER_AXES = {"trigger_l", "trigger_r"}


class XboxController:
    """Manages a single Xbox-compatible joystick via pygame."""

    def __init__(self, deadzone: float = 0.4):
        self._joystick = None
        self._button_state = {}
        self._js_fd = None
        self.deadzone = deadzone
        self._init_pygame()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_pygame(self):
        if not _PYGAME_AVAILABLE:
            return
        # Initialise the display subsystem with a dummy driver so that
        # pygame.event.pump() works without a real display (which would
        # conflict with Qt).  Then initialise joystick.
        import os
        if not pygame.display.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

    def connect(self) -> bool:
        """Try to connect to joystick 0. Returns True on success."""
        self._button_state = {}
        # Open js0 directly for reliable button state
        try:
            fd = os.open(_JS_DEV, os.O_RDONLY | os.O_NONBLOCK)
            if self._js_fd is not None:
                try:
                    os.close(self._js_fd)
                except Exception:
                    pass
            self._js_fd = fd
            self._read_js_events()  # drain initial-state events
        except Exception:
            self._js_fd = None

        if not _PYGAME_AVAILABLE:
            return self._js_fd is not None
        self._joystick = None
        try:
            if pygame.joystick.get_count() > 0:
                js = pygame.joystick.Joystick(0)
                js.init()
                self._joystick = js
                return True
        except Exception:
            self._joystick = None
        return self._js_fd is not None

    def disconnect(self):
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
            self._joystick = None
        if self._js_fd is not None:
            try:
                os.close(self._js_fd)
            except Exception:
                pass
            self._js_fd = None

    def connected(self) -> bool:
        if not _PYGAME_AVAILABLE or self._joystick is None:
            return False
        try:
            # Re-check that the joystick handle is still valid
            return self._joystick.get_init()
        except Exception:
            return False

    def name(self) -> str:
        if self._joystick is not None:
            try:
                return self._joystick.get_name()
            except Exception:
                pass
        return ""

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _read_js_events(self):
        """Read all pending events from js0 and update button state."""
        if self._js_fd is None:
            return
        try:
            while True:
                data = os.read(self._js_fd, _JS_EVENT_SIZE)
                if len(data) < _JS_EVENT_SIZE:
                    break
                _, value, etype, number = struct.unpack(_JS_EVENT_FMT, data)
                if (etype & ~_JS_EVENT_INIT) == _JS_EVENT_BUTTON:
                    self._button_state[number] = bool(value)
        except BlockingIOError:
            pass
        except Exception:
            pass

    def poll(self):
        """Update joystick state: pump pygame for axes, read js0 for buttons."""
        self._read_js_events()
        if not _PYGAME_AVAILABLE:
            return
        try:
            pygame.event.pump()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Axis / button reads
    # ------------------------------------------------------------------

    def _raw_axis(self, index: int) -> float:
        if self._joystick is None:
            return 0.0
        try:
            return self._joystick.get_axis(index)
        except Exception:
            return 0.0

    def get_axis(self, name: str) -> float:
        """Return deadzone-applied axis value.

        For left/right sticks: returns -1..1.
        For triggers: SDL2 reports -1 (released) to +1 (full). We remap to 0..1.
        """
        index = _AXIS_MAP.get(name)
        if index is None:
            return 0.0
        raw = self._raw_axis(index)

        if name in _TRIGGER_AXES:
            # Remap -1..1 → 0..1
            value = (raw + 1.0) / 2.0
            return value if value > self.deadzone else 0.0

        return self._apply_deadzone(raw)

    def get_button(self, name: str) -> bool:
        index = _BUTTON_MAP.get(name)
        if index is None:
            return False
        return self._button_state.get(index, False)

    def get_hat(self) -> tuple:
        """Return D-pad as (x, y) where y=+1 is up, y=-1 is down."""
        if self._joystick is None:
            return (0, 0)
        try:
            return self._joystick.get_hat(0)
        except Exception:
            return (0, 0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_deadzone(self, raw: float) -> float:
        dz = self.deadzone
        if abs(raw) < dz:
            return 0.0
        sign = 1.0 if raw > 0 else -1.0
        return sign * (abs(raw) - dz) / (1.0 - dz)
