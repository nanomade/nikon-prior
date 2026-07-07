"""Unified scale-bar geometry + drawing for SAVED imaging artifacts.

One source of truth so every saved image (catalogue crops, contact-sheet cells,
map thumbnails) shows the *same* scale bar: a black-haloed white bar + "N um"
label, bottom-left, with the bar length the largest 1/2/5 step that fits under a
fraction of the field of view.

Backends: `draw_cv2` (BGR numpy arrays) and `draw_pil` (PIL ImageDraw).  Callers
tune only `max_frac` (how big) — the style is fixed here.

NOT used by the live camera preview (`ui/preview.py`), which keeps its own
fixed-length-per-magnification bar with a user colour toggle: a different,
interactive purpose, deliberately separate.
"""
from __future__ import annotations

BAR_STEPS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)   # µm, 1/2/5 decade steps


def nice_bar_um(field_um: float, max_frac: float = 0.22):
    """Largest BAR_STEPS value ≤ max_frac × field width (µm), or None if no field."""
    if not field_um or field_um <= 0:
        return None
    return next((s for s in reversed(BAR_STEPS) if s <= max_frac * field_um), BAR_STEPS[0])


def draw_cv2(img, px_per_um: float, *, max_frac: float = 0.22, margin: int = 6):
    """Draw the scale bar onto a BGR numpy image in place.  Returns the bar length
    in µm (or None if it couldn't be drawn).  Line/'font' scale gently with image
    size so the bar stays small and unobtrusive on large crops."""
    if img is None or not px_per_um or px_per_um <= 0:
        return None
    import cv2
    h, w = img.shape[:2]
    bar = nice_bar_um(w / px_per_um, max_frac)
    if bar is None:
        return None
    blen = int(round(bar * px_per_um))
    if blen < 5:
        return None
    t = max(1, round(h / 500))                    # thin bar; grows slowly with size
    x0, y0 = margin, h - margin - 3
    cv2.line(img, (x0, y0), (x0 + blen, y0), (0, 0, 0), t + 2, cv2.LINE_AA)
    cv2.line(img, (x0, y0), (x0 + blen, y0), (255, 255, 255), t, cv2.LINE_AA)
    fs = max(0.3, min(0.45, h / 1100.0))
    for col, th in (((0, 0, 0), t + 2), ((255, 255, 255), max(1, t - 1))):
        cv2.putText(img, f"{bar}um", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, col, th, cv2.LINE_AA)
    return bar


def draw_pil(draw, w: int, h: int, px_per_um: float, *, max_frac: float = 0.33,
             margin: int = 6):
    """Draw the scale bar via a PIL ImageDraw (for small thumbnails).  px_per_um is
    in the THUMBNAIL's pixel space.  Returns the bar length in µm, or None."""
    if not px_per_um or px_per_um <= 0:
        return None
    bar = nice_bar_um(w / px_per_um, max_frac)
    if bar is None:
        return None
    blen = int(round(bar * px_per_um))
    if not (4 <= blen <= w - 8):
        return None
    x0, y0 = margin, h - 8
    draw.rectangle([x0 - 1, y0 - 1, x0 + blen + 1, y0 + 3], fill=(0, 0, 0))
    draw.rectangle([x0, y0, x0 + blen, y0 + 2], fill=(255, 255, 255))
    draw.text((x0, y0 - 11), f"{bar}um", fill=(255, 255, 255))
    return bar
