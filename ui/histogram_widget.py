"""Live BGR image histogram (CLAUDE.md #40).

A lightweight QWidget that samples the preview's latest clean frame and draws
per-channel B/G/R histograms with mean readouts and clipping flags.  Intended
to sit in a collapsible rollout next to the exposure / white-balance controls so
the operator can set exposure and WB by eye — keeping the substrate mid-range
and avoiding channel clipping (the underexposed-scan problem of 2026-06-22).

Self-throttling: the refresh timer only runs while the widget is visible
(i.e. its rollout is expanded), so it costs nothing when collapsed.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QWidget

# BGR channel order (cv2 / preview convention) → draw colours.
_CH = [("B", QColor(80, 140, 255)), ("G", QColor(60, 200, 90)), ("R", QColor(255, 90, 90))]
_BINS = 128
_CLIP_FRAC = 0.005   # >0.5% of pixels in the top bin → flag clipping


class LiveHistogram(QWidget):
    def __init__(self, preview, parent=None):
        super().__init__(parent)
        self._preview = preview
        self._hist = None          # (3, _BINS) normalised heights
        self._means8 = [0, 0, 0]   # per-channel mean in 8-bit equivalent
        self._clip = [False, False, False]
        self.setMinimumHeight(140)
        self.setToolTip(
            "Live B/G/R histogram. Raise exposure until the BRIGHTEST channel's peak "
            "is well to the right but no channel shows CLIP (no spike at the right edge). "
            "WB is fixed by the white reference — do NOT rebalance channels here; the "
            "substrate's blue cast is real and the contrast targets depend on it.")
        self._timer = QTimer(self)
        self._timer.setInterval(125)   # ~8 Hz
        self._timer.timeout.connect(self._tick)

    # Only refresh while actually on screen (rollout expanded).
    def showEvent(self, e):
        super().showEvent(e)
        self._timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._timer.stop()

    def _tick(self):
        if not self.isVisible():
            return
        frame = None
        getf = getattr(self._preview, "get_clean_frame", None)
        if callable(getf):
            frame = getf()
        if frame is None:
            frame = getattr(self._preview, "_last_raw_frame", None)
        self._update_from_frame(frame)

    def _update_from_frame(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return
        # Subsample for speed; histogram is statistically identical.
        s = frame[::4, ::4, :3]
        maxval = 4095 if s.dtype == np.uint16 else 255
        scale8 = 255.0 / maxval
        hist = np.empty((3, _BINS), np.float64)
        means8 = [0, 0, 0]
        clip = [False, False, False]
        for c in range(3):
            ch = s[:, :, c].ravel()
            h, _ = np.histogram(ch, bins=_BINS, range=(0, maxval))
            hist[c] = h
            means8[c] = float(ch.mean()) * scale8
            clip[c] = (h[-1] / max(ch.size, 1)) > _CLIP_FRAC
        # sqrt compression + shared normalisation so all channels are comparable
        hist = np.sqrt(hist)
        peak = hist.max() or 1.0
        self._hist = hist / peak
        self._means8 = means8
        self._clip = clip
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(18, 18, 18))
        plot_h = h - 22                      # leave a strip at the bottom for text
        # "hot zone" near the top end — push the brightest channel toward here
        # but keep it OUT of the clip region. WB is fixed; we expose to headroom.
        x235 = int(235 / 255 * w)
        p.fillRect(x235, 0, max(1, w - x235), plot_h, QColor(70, 30, 30))
        # right-edge clip line
        p.setPen(QPen(QColor(160, 90, 90), 1, Qt.DashLine))
        p.drawLine(w - 1, 0, w - 1, plot_h)
        if self._hist is not None:
            for c, (_, col) in enumerate(_CH):
                pen = QPen(col, 1.6)
                p.setPen(pen)
                poly = QPolygonF()
                for b in range(_BINS):
                    x = b / (_BINS - 1) * (w - 1)
                    y = plot_h - self._hist[c, b] * (plot_h - 2)
                    poly.append(QPointF(x, y))
                p.drawPolyline(poly)
        # readouts
        p.setPen(QColor(210, 210, 210))
        txt = "  ".join(
            f"{name}:{int(self._means8[c])}{'!' if self._clip[c] else ''}"
            for c, (name, _) in enumerate(_CH))
        p.drawText(6, h - 6, txt + ("   CLIP" if any(self._clip) else ""))
        p.end()
