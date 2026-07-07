"""vision/calibrate.py — Filter-kit optical calibration for the ForwardModel.

Procedure
---------
1. With each bandpass filter in the light path, image a clean area of bare
   SiO₂/Si substrate (no flakes, stage stationary) and record the median
   R, G, B pixel values.
2. Call ``fit(measurements)`` to determine:
   - SiO₂ oxide thickness  ``d_nm``  (from the reflectance fringe pattern)
   - Effective source × camera response  S_R(λ), S_G(λ), S_B(λ)
3. Save the result and pass ``camera=camera_from_calibration(cal)`` to
   ``ForwardModel``; set ``source=flat_spectrum``.

The calibration encodes illumination spectrum, camera QE, and Bayer filter
shape in one empirical measurement — no separate spectrophotometry needed.
Re-calibrate whenever the lamp, camera, or neutral-density filters change.

Physics
-------
For a narrow bandpass filter f (FWHM « fringe period ≈ 350 nm for 300 nm SiO₂):

    P_ch^f  ≈  S_ch(λ_f) · R(λ_f, d)

where S_ch(λ) = I_source(λ) · QE_ch(λ) is the unknown source×camera product
and R(λ, d) is the full Fresnel TM reflectance computed by optical_contrast.

Fitting: grid-search d ∈ [50, 500] nm.  For each trial d, the optimal
per-channel amplitude a_ch is solved analytically (least squares).  The d that
minimises the total squared residual is selected, then refined with Brent.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize_scalar

from vision.optical_contrast import LAM_NM, flat_spectrum, n_si, n_sio2, reflectance

# ─────────────────────────────────────────────────────────────────────────────
# Filter specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterSpec:
    """One bandpass interference filter."""
    name:      str
    center_nm: float
    fwhm_nm:   float

    def transmission(self, lam_nm: np.ndarray) -> np.ndarray:
        """Gaussian transmission profile, peak = 1."""
        sigma = self.fwhm_nm / 2.355
        return np.exp(-0.5 * ((np.asarray(lam_nm, float) - self.center_nm) / sigma) ** 2)

    @property
    def area_nm(self) -> float:
        """∫ T(λ) dλ for the Gaussian profile."""
        return self.fwhm_nm * np.sqrt(np.pi / (4.0 * np.log(2.0)))


# ── Built-in filter kits ──────────────────────────────────────────────────────

THORLABS_FB_10NM: list[FilterSpec] = [
    FilterSpec("FB450-10", 450, 10),
    FilterSpec("FB500-10", 500, 10),
    FilterSpec("FB550-10", 550, 10),
    FilterSpec("FB600-10", 600, 10),
    FilterSpec("FB650-10", 650, 10),
    FilterSpec("FB700-10", 700, 10),
]

THORLABS_FBH_40NM: list[FilterSpec] = [
    FilterSpec("FBH450-40", 450, 40),
    FilterSpec("FBH500-40", 500, 40),
    FilterSpec("FBH550-40", 550, 40),
    FilterSpec("FBH600-40", 600, 40),
    FilterSpec("FBH650-40", 650, 40),
    FilterSpec("FBH700-40", 700, 40),
]

FILTER_KITS: dict[str, list[FilterSpec]] = {
    "thorlabs_fb_10nm":  THORLABS_FB_10NM,
    "thorlabs_fbh_40nm": THORLABS_FBH_40NM,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-filter measurement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterMeasurement:
    """Median R, G, B pixel intensities (0–255) through one bandpass filter."""
    filter_name: str
    center_nm:   float
    fwhm_nm:     float
    median_r:    float
    median_g:    float
    median_b:    float
    frame_path:  str = ""

    @classmethod
    def from_frame(cls, filt: FilterSpec, frame_bgr: np.ndarray,
                   roi: tuple[int, int, int, int] | None = None,
                   frame_path: str = "") -> "FilterMeasurement":
        """Extract measurement from a BGR uint8 camera frame.

        Args:
            filt:       FilterSpec for the filter currently in the light path.
            frame_bgr:  OpenCV BGR frame (uint8 or float).
            roi:        (x, y, w, h) crop in pixels; None uses the central 60%.
            frame_path: Optional path string recorded for provenance.
        """
        arr = np.asarray(frame_bgr, dtype=float)
        h, w = arr.shape[:2]
        if roi is None:
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            crop = arr[margin_y:h - margin_y, margin_x:w - margin_x]
        else:
            x, y, rw, rh = roi
            crop = arr[y:y + rh, x:x + rw]
        return cls(
            filter_name=filt.name,
            center_nm=filt.center_nm,
            fwhm_nm=filt.fwhm_nm,
            median_r=float(np.median(crop[:, :, 2])),
            median_g=float(np.median(crop[:, :, 1])),
            median_b=float(np.median(crop[:, :, 0])),
            frame_path=frame_path,
        )

    @property
    def rgb(self) -> np.ndarray:
        """(R, G, B) as a float array."""
        return np.array([self.median_r, self.median_g, self.median_b])

    def filter_spec(self) -> FilterSpec:
        return FilterSpec(self.filter_name, self.center_nm, self.fwhm_nm)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Fitted optical calibration: SiO₂ thickness + effective source×camera spectra."""
    sio2_nm:            float
    fit_residual:       float
    source_wavelengths: list[float]   # filter centre λ (nm)
    source_r:           list[float]   # S_R at each filter wavelength (arb. units)
    source_g:           list[float]
    source_b:           list[float]
    filter_kit:         str
    measurements:       list[dict]
    wb_red:             float = 1.0   # BalanceRatio Red at capture time
    wb_blue:            float = 1.0   # BalanceRatio Blue at capture time
    timestamp:          str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def _reflectance_at_centers(centers_nm: np.ndarray, d_nm: float) -> np.ndarray:
    """TM reflectance of bare SiO₂/Si at given wavelengths and SiO₂ thickness."""
    n2 = n_sio2(centers_nm)
    n3 = n_si(centers_nm)
    n_air = np.ones(len(centers_nm), complex)
    return reflectance([n2], [d_nm], centers_nm, n_in=n_air, n_sub=n3)


def _gaussian_smooth_S(
    centers: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    sat_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian to S_ch(λ) for each channel, excluding saturated/dead filters.

    Saturated P values (≥ sat_threshold) are lower bounds on S — exclude them.
    Dead-lamp filters (S < 3% of channel max) are excluded too.
    Falls back to linear interpolation for channels where < 3 valid points remain.

    Returns (lam_fine, S_r, S_g, S_b) on a 10 nm grid from 400–750 nm.
    """
    lam_fine = np.arange(400.0, 751.0, 10.0)
    ch_out = []

    def _gauss(x, A, mu, sig):
        return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    for ch in range(3):
        p_ch = P[:, ch]
        s_ch = S[:, ch]
        s_max = s_ch.max()

        valid = (p_ch < sat_threshold) & (s_ch > s_max * 0.03)

        if valid.sum() >= 3:
            lv, sv = centers[valid], s_ch[valid]
            p0 = [sv.max(), float(lv[np.argmax(sv)]), 60.0]
            bounds = ([0.0, 350.0, 20.0], [sv.max() * 3.0, 750.0, 200.0])
            try:
                popt, _ = curve_fit(_gauss, lv, sv, p0=p0, bounds=bounds, maxfev=2000)
                ch_out.append(np.maximum(_gauss(lam_fine, *popt), 0.0))
                continue
            except Exception:
                pass

        # Fallback: linear interpolation of non-dead points
        mask = valid if valid.sum() >= 2 else (s_ch > 0)
        if mask.sum() < 2:
            ch_out.append(np.zeros_like(lam_fine))
            continue
        f = interp1d(centers[mask], s_ch[mask], kind='linear',
                     fill_value=(float(s_ch[mask][0]), float(s_ch[mask][-1])),
                     bounds_error=False)
        ch_out.append(np.maximum(f(lam_fine), 0.0))

    return lam_fine, ch_out[0], ch_out[1], ch_out[2]


def fit(
    measurements: list[FilterMeasurement],
    references: list[FilterMeasurement] | None = None,
    fixed_d: float | None = None,
    d_range: tuple[float, float] = (50.0, 500.0),
    d_step: float = 1.0,
    filter_kit: str = "custom",
    camera=None,
    wb_red: float = 1.0,
    wb_blue: float = 1.0,
    gaussian_smooth: bool = False,
) -> CalibrationResult:
    """Fit SiO₂ thickness and source×camera spectra from filter measurements.

    Three fitting modes (in order of preference):

    **Reference mode** (``references`` provided):
      Divide oxide measurements by bare-Si reference measurements at the same
      filter wavelengths.  Lamp spectrum and camera QE cancel exactly:
        ratio(λ_f) = P_oxide(λ_f) / P_Si(λ_f) = R_SiO₂(λ_f, d) / R_Si(λ_f)
      Fit d by minimising the weighted deviation of measured ratios from the
      theoretical Fresnel ratio.  No camera model needed.

    **Fixed-d mode** (``fixed_d`` provided, no references):
      Skip the thickness search; use the supplied d (e.g. from wafer datasheet).
      Fit only the source×camera amplitudes S_ch(λ_f).

    **Camera-weighted mode** (default):
      Grid-search + Brent minimisation of camera-QE-weighted residual.
      Requires a camera prior (default: IMX219).

    At least 3 measurements are required (6 recommended).
    ``references``, if supplied, must cover the same filter wavelengths as
    ``measurements`` (order need not match; matched by center_nm).
    """
    if len(measurements) < 3:
        raise ValueError("Need ≥ 3 filter measurements to fit SiO₂ thickness")

    centers = np.array([m.center_nm for m in measurements])
    P = np.array([m.rgb for m in measurements])  # (N_f, 3)

    if P.max() < 5.0:
        raise ValueError("Pixel values are too dark (max < 5) — check exposure")

    # ── determine d_best and res_best ─────────────────────────────────────────

    if references is not None:
        # Align references to measurement order by center_nm
        ref_by_lam = {round(r.center_nm): r for r in references}
        P_si = np.array([
            ref_by_lam[round(c)].rgb if round(c) in ref_by_lam
            else np.zeros(3)
            for c in centers
        ])
        # Weighted ratio: sum(P_ox * P_si) / sum(P_si²) — weight by Si signal²
        si2 = np.sum(P_si ** 2, axis=1)                         # (N_f,)
        ratios  = np.where(si2 > 1.0, np.sum(P * P_si, axis=1) / si2, 0.0)
        weights = si2
        R_si = _reflectance_at_centers(centers, 0.0)            # bare Si

        def ref_residual(d: float) -> float:
            R_ox = _reflectance_at_centers(centers, d)
            expected = R_ox / np.clip(R_si, 1e-9, None)
            return float(np.sum(weights * (ratios - expected) ** 2))

        if fixed_d is not None:
            d_best, res_best = float(fixed_d), ref_residual(float(fixed_d))
        else:
            d_vals = np.arange(d_range[0], d_range[1] + d_step * 0.5, d_step)
            rg = [ref_residual(d) for d in d_vals]
            d0 = float(d_vals[int(np.argmin(rg))])
            lo = max(d_range[0], d0 - max(d_step * 3, 5.0))
            hi = min(d_range[1], d0 + max(d_step * 3, 5.0))
            opt = minimize_scalar(ref_residual, bounds=(lo, hi), method="bounded")
            d_best, res_best = float(opt.x), float(opt.fun)

    else:
        # Camera-weighted fit (with optional fixed_d shortcut)
        from vision.optical_contrast import camera_imx219
        if camera is None:
            camera = camera_imx219
        cam_r, cam_g, cam_b = camera(centers)
        W = np.stack([cam_r, cam_g, cam_b], axis=1).astype(float)
        for ch in range(3):
            mx = W[:, ch].max()
            if mx > 0:
                W[:, ch] /= mx

        def cam_residual(d: float) -> float:
            Rf = _reflectance_at_centers(centers, d)
            if Rf.max() < 1e-6:
                return 1e9
            total = 0.0
            for ch in range(3):
                Rw = Rf * W[:, ch]
                rr = float(np.dot(Rw, Rw))
                if rr < 1e-12:
                    continue
                a = float(np.dot(P[:, ch], Rw)) / rr
                total += float(np.sum((P[:, ch] - a * Rw) ** 2))
            return total

        if fixed_d is not None:
            d_best, res_best = float(fixed_d), cam_residual(float(fixed_d))
        else:
            d_vals = np.arange(d_range[0], d_range[1] + d_step * 0.5, d_step)
            rg = [cam_residual(d) for d in d_vals]
            d0 = float(d_vals[int(np.argmin(rg))])
            lo = max(d_range[0], d0 - max(d_step * 3, 5.0))
            hi = min(d_range[1], d0 + max(d_step * 3, 5.0))
            opt = minimize_scalar(cam_residual, bounds=(lo, hi), method="bounded")
            d_best, res_best = float(opt.x), float(opt.fun)

    # ── source × camera spectra S_ch(λ_f) = P / R ────────────────────────────
    Rf_best = _reflectance_at_centers(centers, d_best)
    S = np.where(Rf_best[:, None] > 1e-6, P / Rf_best[:, None], P)

    if gaussian_smooth:
        sat_thr = 4080.0 if P.max() > 300 else 250.0
        lam_out, S_r, S_g, S_b = _gaussian_smooth_S(centers, P, S, sat_thr)
    else:
        lam_out, S_r, S_g, S_b = centers, S[:, 0], S[:, 1], S[:, 2]

    return CalibrationResult(
        sio2_nm=round(d_best, 1),
        fit_residual=round(res_best, 4),
        source_wavelengths=[float(c) for c in lam_out],
        source_r=[float(v) for v in S_r],
        source_g=[float(v) for v in S_g],
        source_b=[float(v) for v in S_b],
        filter_kit=filter_kit,
        wb_red=round(float(wb_red), 4),
        wb_blue=round(float(wb_blue), 4),
        measurements=[
            {"filter": m.filter_name, "center_nm": m.center_nm,
             "fwhm_nm": m.fwhm_nm,
             "r": round(m.median_r, 2), "g": round(m.median_g, 2),
             "b": round(m.median_b, 2), "path": m.frame_path}
            for m in measurements
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ForwardModel integration
# ─────────────────────────────────────────────────────────────────────────────

def camera_from_calibration(cal: CalibrationResult) -> Callable:
    """Return a camera-response callable for ``ForwardModel``.

    Interpolates the measured S_ch(λ) to LAM_NM and normalises each channel
    to unit integral.  Use with ``source=flat_spectrum`` in ForwardModel so
    that ``source(λ) × camera_ch(λ) ∝ S_ch(λ)`` (contrast is a ratio, so
    absolute normalisation does not matter).

    Example::

        cal  = load("optical_calibration.json")
        model = ForwardModel(
            materials=[GRAPHENE, HBN],
            sio2_nm=cal.sio2_nm,
            source=flat_spectrum,
            camera=camera_from_calibration(cal),
        )
    """
    lam_cal = np.array(cal.source_wavelengths)
    lam_out = LAM_NM

    def _interp_normalise(values: list[float]) -> np.ndarray:
        v = np.array(values, float)
        # Clamp negatives (measurement noise at very low R)
        v = np.clip(v, 0.0, None)
        if v.max() < 1e-9:
            return np.ones(len(lam_out)) / (lam_out[-1] - lam_out[0])
        # Linear interpolation; flat extrapolation beyond measured range
        f = interp1d(lam_cal, v, kind="linear", fill_value=(v[0], v[-1]),
                     bounds_error=False)
        out = np.maximum(f(lam_out), 0.0)
        from scipy.integrate import trapezoid
        area = trapezoid(out, lam_out)
        return out / area if area > 1e-12 else out

    r_arr = _interp_normalise(cal.source_r)
    g_arr = _interp_normalise(cal.source_g)
    b_arr = _interp_normalise(cal.source_b)

    def _camera(lam_nm: np.ndarray):
        # ForwardModel will use its own lam grid; re-interpolate if needed
        lam = np.asarray(lam_nm, float)
        if len(lam) == len(lam_out) and np.allclose(lam, lam_out):
            return r_arr.copy(), g_arr.copy(), b_arr.copy()
        fr = interp1d(lam_out, r_arr, fill_value=(r_arr[0], r_arr[-1]),
                      bounds_error=False)
        fg = interp1d(lam_out, g_arr, fill_value=(g_arr[0], g_arr[-1]),
                      bounds_error=False)
        fb = interp1d(lam_out, b_arr, fill_value=(b_arr[0], b_arr[-1]),
                      bounds_error=False)
        return fr(lam), fg(lam), fb(lam)

    return _camera


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save(cal: CalibrationResult, path: str) -> None:
    """Write CalibrationResult to JSON."""
    with open(path, "w") as f:
        json.dump(asdict(cal), f, indent=2)


def load(path: str) -> CalibrationResult:
    """Load CalibrationResult from JSON."""
    with open(path) as f:
        d = json.load(f)
    # Tolerate extra keys added in future versions
    fields = {k for k in CalibrationResult.__dataclass_fields__}
    return CalibrationResult(**{k: v for k, v in d.items() if k in fields})


def save_measurements(measurements: list[FilterMeasurement], path: str) -> None:
    """Save raw filter measurements to JSON (no fit required)."""
    data = {
        "type": "filter_measurements",
        "measurements": [
            {"filter": m.filter_name, "center_nm": m.center_nm,
             "fwhm_nm": m.fwhm_nm,
             "r": round(m.median_r, 2), "g": round(m.median_g, 2),
             "b": round(m.median_b, 2), "path": m.frame_path}
            for m in measurements
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_measurements(path: str) -> list[FilterMeasurement]:
    """Load raw filter measurements from a measurements JSON or a CalibrationResult JSON."""
    with open(path) as f:
        d = json.load(f)
    meas_list = d.get("measurements", [])
    return [
        FilterMeasurement(
            filter_name=m["filter"], center_nm=m["center_nm"],
            fwhm_nm=m["fwhm_nm"], median_r=m["r"],
            median_g=m["g"], median_b=m["b"],
            frame_path=m.get("path", ""))
        for m in meas_list
    ]
