"""Application-level input routing.

Ported from standa-stacker's stagecontrol.py (the _GlobalKeyRouter /
spinbox-wheel-guard event filters) and trimmed to the Prior/Nikon rig
(XYZ stage, no manipulator/rotation).

- GlobalKeyRouter: routes stage jog / focus / autofocus shortcuts to the
  preview window regardless of which widget has focus, so the user never has
  to click the preview first.  Escape is a panic stop; X cycles magnification.
- SpinBoxWheelGuard: stops the mouse wheel from changing an unfocused spin
  box / combo box value when scrolling a side panel — the wheel is forwarded
  to the enclosing scroll area instead.
"""

from PyQt5.QtCore import QObject, QEvent, Qt
from PyQt5.QtWidgets import (
    QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox,
    QLineEdit, QTextEdit,
)


class GlobalKeyRouter(QObject):
    """Route stage keyboard shortcuts to the preview regardless of focus.

    Intercepts key presses at the QApplication level.  Keys are suppressed
    when a text-entry widget has focus or a modal dialog is open.
    """

    _TEXT_TYPES = (QLineEdit, QAbstractSpinBox, QTextEdit)
    _ROUTED_KEYS = {
        Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
        Qt.Key_PageUp, Qt.Key_PageDown,
        Qt.Key_Return, Qt.Key_Enter,
        Qt.Key_Plus, Qt.Key_Minus,
        # Numpad XY jog (1–9) is handled by the preview while the cursor is
        # over it — not routed globally, to avoid hijacking numeric entry.
    }

    def __init__(self, preview_win, motor_manager=None, gamepad_panel=None,
                 parent=None):
        super().__init__(parent)
        self._preview = preview_win
        self._motor_manager = motor_manager
        self._gamepad_panel = gamepad_panel

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False

        # Escape = panic stop: halt stage motion regardless of focus or modal
        # state.  The event is NOT consumed, so Escape keeps its normal role
        # (close dialog, cancel edit).
        if event.key() == Qt.Key_Escape:
            if self._motor_manager is not None:
                try:
                    self._motor_manager.stop()
                except Exception:
                    pass
            return False

        if QApplication.activeModalWidget() is not None:
            return False
        fw = QApplication.focusWidget()
        if fw is not None and isinstance(fw, self._TEXT_TYPES):
            return False

        # X = cycle magnification (same action as the gamepad X button).
        if event.key() == Qt.Key_X and self._gamepad_panel is not None:
            try:
                self._gamepad_panel._cycle_magnification()
            except Exception:
                pass
            return True

        if event.key() not in self._ROUTED_KEYS:
            return False
        self._preview._dispatch_global_key(event)
        return True


class SpinBoxWheelGuard(QObject):
    """Stop the wheel from changing an unfocused spin/combo box value.

    Without focus, a wheel event over a spin box is forwarded to the nearest
    enclosing scroll area so the side panel scrolls instead of the value
    jumping.  A focused widget keeps normal wheel behaviour.
    """

    def eventFilter(self, obj, event):
        if isinstance(obj, (QAbstractSpinBox, QComboBox)):
            if obj.focusPolicy() != Qt.StrongFocus:
                obj.setFocusPolicy(Qt.StrongFocus)
            if event.type() == QEvent.Wheel and not obj.hasFocus():
                ancestor = obj.parent()
                while ancestor is not None:
                    if isinstance(ancestor, QAbstractScrollArea):
                        QApplication.sendEvent(ancestor.viewport(), event)
                        break
                    ancestor = ancestor.parent()
                return True
        return False
