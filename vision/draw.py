"""Shared drawing geometry (plan task L6).

Neither cv2 nor PIL has a dashed stroke, so the dash/gap walk along a
polyline was implemented twice (tools/import_found_flakes cv2 crops,
tools/make_map PIL thumbnails). The geometry is the common core; consumers
draw the returned segments with their own backend.
"""


def dash_segments(pts, dash: float = 9.0, gap: float = 6.0, closed: bool = True):
    """Yield ((x0, y0), (x1, y1)) dash segments along a polyline.

    pts: sequence of (x, y). closed: connect the last point back to the first.
    """
    n = len(pts)
    if n < 2:
        return
    last = n if closed else n - 1
    for i in range(last):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        seg = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if seg < 1e-6:
            continue
        ux, uy = (bx - ax) / seg, (by - ay) / seg
        d = 0.0
        while d < seg:
            e = min(d + dash, seg)
            yield (ax + ux * d, ay + uy * d), (ax + ux * e, ay + uy * e)
            d += dash + gap
