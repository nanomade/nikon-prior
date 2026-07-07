"""vision/optical_contrast.py — Forward contrast model for 2D-material identification.

Computes expected RGB contrast for (material, layer-count) combinations on
SiO₂/Si substrates using the Fresnel transfer-matrix formalism.

Physics references:
  Blake et al. arXiv:0705.0259      — graphene visibility model
  Jessen et al. Sci. Rep. 8, 6381  — full pipeline, eq. (1a-c)
  Gorbachev et al. Small 7, 465    — hBN contrast and NA correction
  Byrnes arXiv:1603.02720          — transfer-matrix implementation
  Weber et al. APL 97, 2010        — graphene refractive index
  Beal/Liang/Hughes J. Phys. C 8   — WSe₂ Kramers-Kronig values
  Malitson J. Opt. Soc. Am. 55     — SiO₂ Sellmeier
  Palik Handbook of Optical Constants — Si tabulation

Usage:
    from vision.optical_contrast import ForwardModel, GRAPHENE, SIO2_300NM
    model = ForwardModel([GRAPHENE], SIO2_300NM)
    lut = model.build_lut()  # {('graphene', 1): (C_R, C_G, C_B), ...}
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import trapezoid

# ── Wavelength grid ───────────────────────────────────────────────────────────
LAM_NM = np.arange(400.0, 701.0, 2.0)  # 400–700 nm, 2 nm step.
# NOTE (2026-07-01): the IMX250 has real NIR QE lobes (B ~800 nm, G/R 700–800 nm),
# but integrating them with a broadband (blackbody) source COLLAPSES the predicted
# contrast (graphene-on-SiO₂/Si contrast washes out in the NIR) — far below the
# strong empirical contrast.  That means the real optical path is effectively
# visible-only (white-LED illumination and/or an IR-cut filter).  So this grid
# encodes that IR-limited passband.  Once the lamp S(λ) is measured (Ocean Optics
# Red Tide) — and any IR-cut characterised — widen this to 1000 nm and let the
# measured S(λ)·QE weighting (near-zero in the NIR) handle the cutoff naturally.
# Si NIR indices are already tabulated below, ready for that.


# ─────────────────────────────────────────────────────────────────────────────
# Refractive index functions  n̂(λ) = n(λ) + i·k(λ),  k > 0 absorbing
# ─────────────────────────────────────────────────────────────────────────────

def _tab(lam: np.ndarray, lam_t, n_t, k_t) -> np.ndarray:
    """Interpolate (n, k) from a tabulated dataset."""
    return (np.interp(lam, np.asarray(lam_t, float), np.asarray(n_t, float))
            + 1j * np.interp(lam, np.asarray(lam_t, float), np.asarray(k_t, float)))


def n_sio2(lam_nm: np.ndarray) -> np.ndarray:
    """SiO₂ (fused silica): Malitson 1965 Sellmeier, non-absorbing in visible."""
    u = lam_nm / 1000.0  # µm
    n2 = (1
          + 0.6961663 * u**2 / (u**2 - 0.0684043**2)
          + 0.4079426 * u**2 / (u**2 - 0.1162414**2)
          + 0.8974794 * u**2 / (u**2 - 9.896161**2))
    return np.sqrt(np.maximum(n2, 1.0)).astype(complex)


def n_si(lam_nm: np.ndarray) -> np.ndarray:
    """Si: Palik Handbook of Optical Constants, visible range."""
    _l = [380, 400, 420, 440, 460, 480, 500, 520, 540, 560,
          580, 600, 620, 640, 660, 680, 700, 720,
          750, 800, 850, 900, 950, 1000]
    _n = [6.08, 5.57, 5.04, 4.73, 4.55, 4.38, 4.29, 4.21, 4.14, 4.10,
          4.06, 4.02, 3.99, 3.97, 3.95, 3.92, 3.90, 3.88,
          3.73, 3.69, 3.65, 3.62, 3.60, 3.58]          # NIR: Green 2008 / Aspnes
    _k = [0.800, 0.387, 0.224, 0.113, 0.0624, 0.0279, 0.0136, 5.78e-3,
          2.02e-3, 5.91e-4, 1.33e-4, 1.49e-5, 0, 0, 0, 0, 0, 0,
          7.0e-3, 3.7e-3, 2.1e-3, 1.2e-3, 7e-4, 5e-5]   # weak NIR absorption
    return _tab(lam_nm, _l, _n, _k)


# ── Dispersive monolayer-graphene optical model ───────────────────────────────
# Treat the monolayer as a d=0.335 nm film with
#     ε(E) = ε∞ + i·Im_uni(λ) + A·E0²/(E0² − E² − iΓE)
#   • Im_uni = σ0/(ε0·ω·d): the UNIVERSAL interband continuum (gapless Dirac
#     bands) → the flat ~2.3%/layer absorbance.  σ0 = e²/4ℏ (no free parameter).
#   • Lorentz at E0 = 4.6 eV: the M-point van Hove / saddle-exciton resonance
#     (Mak/Chae 2011) whose visible-side tail gives graphene its (weak) dispersion
#     — strongest in the blue, which is exactly where the old constant erred.
# ε∞ and A are solved ONCE so n̂(550 nm) = 2.6+1.3i — the Blake/Weber visible value
# our contrast calibration is anchored to — leaving the dispersion physical.
_GR_E0_EV     = 4.6      # saddle-exciton (van Hove) energy
_GR_GAMMA_EV  = 2.0      # effective resonance width
_GR_D_M       = 0.335e-9
_GR_SIGMA0    = 6.083e-5    # e²/4ℏ  (universal optical sheet conductivity, S)
_EPS0, _C_LIGHT = 8.854e-12, 2.998e8


def _gr_im_uni(lam_nm):
    omega = 2.0 * np.pi * _C_LIGHT / (np.asarray(lam_nm, float) * 1e-9)
    return _GR_SIGMA0 / (_EPS0 * omega * _GR_D_M)


def _gr_lorentz(lam_nm):
    E = 1240.0 / np.asarray(lam_nm, float)
    return _GR_E0_EV**2 / (_GR_E0_EV**2 - E**2 - 1j * _GR_GAMMA_EV * E)


# Solve (ε∞, A) at import by anchoring ε(550) = (2.6+1.3i)².
_eps_t   = (2.6 + 1.3j) ** 2
_L550    = _gr_lorentz(550.0)
_GR_A    = float((_eps_t.imag - _gr_im_uni(550.0)) / _L550.imag)
_GR_EPS_INF = float(_eps_t.real - _GR_A * _L550.real)


def n_graphene(lam_nm: np.ndarray, *, _warn: bool = False) -> np.ndarray:
    """Monolayer graphene: dispersive n̂(λ) (universal continuum + 4.6 eV van Hove).

    Anchored to n̂(550 nm)=2.6+1.3i (Blake/Weber); dispersion across 400–700 nm
    is weak (~2.6–2.8 + 1.2–1.5i) but physical — it carries the blue rise toward
    the π→π* saddle resonance that a flat index misses.  Replaces the former
    bulk-graphite constant.
    """
    eps = _GR_EPS_INF + 1j * _gr_im_uni(lam_nm) + _GR_A * _gr_lorentz(lam_nm)
    return np.sqrt(eps).astype(complex)


def n_graphite(lam_nm: np.ndarray) -> np.ndarray:
    """Bulk graphite: Blake et al. constant n̂=2.6+1.3i."""
    return np.full(len(np.asarray(lam_nm)), 2.6 + 1.3j, dtype=complex)


def n_hbn(lam_nm: np.ndarray) -> np.ndarray:
    """hBN: Gorbachev et al. Small 7, 465.

    n ≈ 2.2 across visible, slight upshift below 500 nm (bandgap tail).
    k ≈ 0 throughout visible.
    """
    lam = np.asarray(lam_nm, float)
    n = np.where(lam < 500.0, 2.2 + 0.1 * (500.0 - lam) / 120.0, 2.2)
    return n.astype(complex)


def n_wse2(lam_nm: np.ndarray) -> np.ndarray:
    """WSe₂ bulk: Beal, Liang & Hughes J. Phys. C 8, 4234 (1975).

    Kramers-Kronig values.  Excitonic peaks: A at ~640 nm, B at ~585 nm.
    NOTE: bulk values used per Jessen et al. Sci. Rep. 8, 6381 recommendation.
    A warning is logged when layer count = 1 (largest bulk/monolayer deviation).
    """
    _l = [400, 420, 440, 460, 480, 500, 520, 540, 560, 580,
          600, 620, 640, 660, 680, 700]
    _n = [4.3, 4.0, 3.8, 3.6, 3.7, 4.0, 4.4, 4.5, 4.4, 4.1,
          3.7, 3.4, 3.2, 3.3, 3.3, 3.2]
    _k = [2.6, 2.4, 2.2, 1.9, 1.7, 1.7, 1.8, 2.0, 2.1, 2.1,
          1.9, 1.5, 1.0, 0.6, 0.3, 0.2]
    return _tab(lam_nm, _l, _n, _k)


# Named registry for string lookup
N_REGISTRY: dict[str, Callable] = {
    "graphene": n_graphene,
    "graphite": n_graphite,
    "hbn":      n_hbn,
    "wse2":     n_wse2,
    "sio2":     n_sio2,
    "si":       n_si,
}


# ─────────────────────────────────────────────────────────────────────────────
# Transfer-matrix reflectance  (Byrnes arXiv:1603.02720)
# ─────────────────────────────────────────────────────────────────────────────

def _intf(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """(NL, 2, 2) interface transfer matrix  D_a^{-1} · D_b."""
    r = (na - nb) / (na + nb)
    t = 2.0 * na / (na + nb)
    NL = len(na)
    M = np.zeros((NL, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0 / t
    M[:, 0, 1] = r / t
    M[:, 1, 0] = r / t
    M[:, 1, 1] = 1.0 / t
    return M


def _prop(n: np.ndarray, d_nm: float, lam_nm: np.ndarray) -> np.ndarray:
    """(NL, 2, 2) propagation matrix through layer of thickness d_nm.

    Signs: exp(-iδ) forward / exp(+iδ) backward — the TM is built right-to-left
    (exit→entrance), so this matrix *undoes* the forward phase accumulation.
    Using exp(+iδ) for the forward slot gives R > 1 for thick absorbers (wrong).
    """
    delta = 2.0 * np.pi * n * d_nm / lam_nm
    NL = len(n)
    M = np.zeros((NL, 2, 2), dtype=complex)
    M[:, 0, 0] = np.exp(-1j * delta)
    M[:, 1, 1] = np.exp(+1j * delta)
    return M


def _mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched 2×2 matrix multiply: (NL,2,2) @ (NL,2,2)."""
    return np.einsum("...ij,...jk->...ik", A, B)


def reflectance(
    n_layers: list[np.ndarray],
    d_nm: list[float],
    lam_nm: np.ndarray,
    n_in: np.ndarray | None = None,
    n_sub: np.ndarray | None = None,
) -> np.ndarray:
    """Normal-incidence reflectance for a planar multilayer stack.

    Physical order: n_in → [n_layers[0], d_nm[0]] → … → n_sub (semi-∞)

    Args:
        n_layers: Complex n̂(λ) for each inner layer (len NL each).
        d_nm:     Thickness of each inner layer in nm.
        lam_nm:   Wavelength array in nm (length NL).
        n_in:     Incident medium (default: air = 1).
        n_sub:    Exit substrate (default: Si from table).

    Returns:
        R(λ), real array in [0, 1], shape (NL,).
    """
    lam = np.asarray(lam_nm, float)
    NL = len(lam)
    n0 = n_in  if n_in  is not None else np.ones(NL, complex)
    nS = n_sub if n_sub is not None else n_si(lam)

    assert len(n_layers) == len(d_nm), "n_layers and d_nm must match in length"

    # Build transfer matrix: M = M_01 · P_1 · M_12 · P_2 · … · M_(last→sub)
    M = _intf(n0, n_layers[0] if n_layers else nS)
    for i, (ni, di) in enumerate(zip(n_layers, d_nm)):
        M = _mm(M, _prop(ni, di, lam))
        n_next = n_layers[i + 1] if i + 1 < len(n_layers) else nS
        M = _mm(M, _intf(ni, n_next))

    if not n_layers:
        # bare substrate: just one interface
        pass  # M already = _intf(n0, nS)

    r = M[:, 1, 0] / M[:, 0, 0]
    return np.abs(r) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Source spectra
# ─────────────────────────────────────────────────────────────────────────────

def blackbody(lam_nm: np.ndarray, T_K: float = 3200.0) -> np.ndarray:
    """Planckian spectral radiance, normalised to unit integral."""
    lam_m = np.asarray(lam_nm, float) * 1e-9
    h, c, kB = 6.626e-34, 2.998e8, 1.381e-23
    B = 2 * h * c**2 / lam_m**5 / (np.exp(h * c / (lam_m * kB * T_K)) - 1)
    return B / trapezoid(B, lam_nm)


def flat_spectrum(lam_nm: np.ndarray) -> np.ndarray:
    """Uniform (white-light) source, normalised."""
    s = np.ones(len(lam_nm), float)
    return s / trapezoid(s, lam_nm)


# ─────────────────────────────────────────────────────────────────────────────
# Optical-path transmission T(λ)  (IR-cut filter, etc.) — 0..1, NOT normalised.
# The ForwardModel integrand is R(λ)·S(λ)·T(λ)·QE(λ); T lumps everything between
# source and sensor that isn't in S or QE (IR-cut filter, coatings, beamsplitter).
# The location of a cut is irrelevant — only the net T(λ) matters.
# ─────────────────────────────────────────────────────────────────────────────

def unity_transmission(lam_nm: np.ndarray) -> np.ndarray:
    """All-pass (no extra optics)."""
    return np.ones(len(np.asarray(lam_nm)), float)


def ir_cut_filter(cut_nm: float = 660.0, edge_nm: float = 30.0) -> Callable:
    """Visible-passband IR-cut: T≈1 below cut_nm, smoothly →0 above (sigmoid edge).

    Stand-in for the colour-camera IR-cut until a real transmission is measured.
    cut_nm = 50% point; edge_nm = rolloff steepness (smaller = sharper).
    """
    def _T(lam_nm):
        lam = np.asarray(lam_nm, float)
        return 1.0 / (1.0 + np.exp((lam - cut_nm) / max(edge_nm, 1e-3)))
    return _T


def measured_transmission(lam_t, T_t) -> Callable:
    """Wrap a measured transmission curve (λ_nm, T 0..1) as a source-path optics term."""
    lam_t = np.asarray(lam_t, float); T_t = np.asarray(T_t, float)
    return lambda lam_nm: np.interp(np.asarray(lam_nm, float), lam_t, T_t,
                                    left=T_t[0], right=T_t[-1])


def load_E(path) -> tuple[np.ndarray, np.ndarray]:
    """Load a measured E(λ) CSV (wavelength_nm, E_norm) written by
    tools/measure_spectrum.py → (λ_nm, E) arrays.  Feed into a ForwardModel as
    the source: ``ForwardModel([...], source=measured_transmission(*load_E(p)))``.

    Note E = (sig−dark)/R_Si still carries the *spectrometer's* detector QE; it
    is S(λ)·T_path(λ)·QE_spectrometer, not the bare lamp×path.  Divide by
    redtide_ilx511b_qe() for the physically-correct source S·T (the camera QE is
    applied separately by the model).  Empirically that correction does NOT close
    the residual blue-contrast gap vs the Graphene_Q targets — it slightly widens
    it — so the B deficit is downstream (imx250 blue QE / graphene n̂(blue) /
    camera white-balance), not the lamp or the Red Tide's own blue rolloff.
    """
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]


# Sony ILX511B relative spectral sensitivity, digitized from the datasheet
# (spectrecology.com SONY-ILX511B.pdf p.12, "Spectral Sensitivity Characteristics",
# Ta=25°C) — the linear CCD inside the Ocean Optics USB650 "Red Tide".  Peak≈1.0
# at ~595 nm; blue end ~0.42 (400 nm), so a Red-Tide-measured E(λ) under-reads blue
# by ~2× vs its green peak until this is divided out.
_ILX511B_QE = np.array([
    (400, 0.42), (410, 0.45), (420, 0.48), (430, 0.53), (440, 0.60), (450, 0.67),
    (460, 0.72), (470, 0.76), (480, 0.79), (490, 0.82), (500, 0.83), (510, 0.835),
    (520, 0.84), (530, 0.855), (540, 0.875), (550, 0.895), (560, 0.915), (570, 0.905),
    (580, 0.94), (590, 0.985), (600, 1.0), (610, 0.98), (620, 0.955), (630, 0.96),
    (640, 0.945), (650, 0.92), (660, 0.885), (670, 0.845), (680, 0.805), (690, 0.80),
    (700, 0.79), (710, 0.74), (720, 0.66), (730, 0.58), (740, 0.52), (750, 0.49),
    (760, 0.50), (770, 0.47), (780, 0.42), (790, 0.37), (800, 0.31), (810, 0.26),
    (820, 0.225), (830, 0.21), (840, 0.205), (850, 0.215), (860, 0.225), (870, 0.22),
    (880, 0.20), (890, 0.165), (900, 0.13), (920, 0.085), (940, 0.06), (960, 0.048),
    (980, 0.04), (1000, 0.035),
])


def redtide_ilx511b_qe(lam_nm: np.ndarray) -> np.ndarray:
    """Relative spectral sensitivity of the Red Tide's ILX511B CCD (0..1, peak≈1).

    Divide a Red-Tide-measured E(λ) by this to recover the true source×path S·T
    the ForwardModel wants (it applies the *camera* QE itself).  Constant outside
    400–1000 nm (held at the endpoints).
    """
    q = _ILX511B_QE
    return np.interp(np.asarray(lam_nm, float), q[:, 0], q[:, 1],
                     left=q[0, 1], right=q[-1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Camera spectral responses
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_response(lam_nm, peak, sigma, lam_range):
    """Gaussian approximation; zero outside lam_range."""
    lam = np.asarray(lam_nm, float)
    g = np.exp(-0.5 * ((lam - peak) / sigma) ** 2)
    g[(lam < lam_range[0]) | (lam > lam_range[1])] = 0.0
    return g / trapezoid(g, lam)


def camera_icx282aq(lam_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sony ICX282AQ spectral sensitivities (R, G, B) — Gaussian approximation.

    Factory curves are proprietary; Gaussian fits to the published QE graphs
    are accurate to ~5% across 450–650 nm (sufficient for contrast ratios).
    """
    lam = np.asarray(lam_nm, float)
    r = _gaussian_response(lam, peak=610, sigma=75, lam_range=(500, 720))
    g = _gaussian_response(lam, peak=545, sigma=55, lam_range=(450, 640))
    b = _gaussian_response(lam, peak=455, sigma=50, lam_range=(380, 550))
    return r, g, b


def camera_generic_rgb(lam_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic Bayer-pattern RGB response (wide-band approximation)."""
    lam = np.asarray(lam_nm, float)
    r = _gaussian_response(lam, peak=600, sigma=90, lam_range=(470, 720))
    g = _gaussian_response(lam, peak=540, sigma=65, lam_range=(420, 660))
    b = _gaussian_response(lam, peak=450, sigma=65, lam_range=(380, 570))
    return r, g, b


_imx219_data: np.ndarray | None = None

def camera_imx219(lam_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sony IMX219 spectral sensitivities (R, G, B) from measured data.

    Tabulated at 1 nm steps, 400–700 nm; linearly interpolated.
    Source: bluegreen-labs/raspberry_pi_camera_responses (AGPL-3.0).
    """
    global _imx219_data
    if _imx219_data is None:
        _csv = os.path.join(os.path.dirname(__file__), "imx219_spectral_response.csv")
        _imx219_data = np.loadtxt(_csv, delimiter=",", skiprows=1)
    lam = np.asarray(lam_nm, float)
    r = np.interp(lam, _imx219_data[:, 0], _imx219_data[:, 1], left=0.0, right=0.0)
    g = np.interp(lam, _imx219_data[:, 0], _imx219_data[:, 2], left=0.0, right=0.0)
    b = np.interp(lam, _imx219_data[:, 0], _imx219_data[:, 3], left=0.0, right=0.0)
    return r, g, b


_imx250_data = None


def camera_imx250(lam_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sony IMX250 (colour) absolute QE — the sensor in our Alvium 1800 U-508c.

    Programmatically digitized (5 nm steps, 400–1000 nm) from the Allied Vision
    U-508 datasheet QE plot (alliedvision.com/pdf/datasheet/1145) — curves traced
    by colour-segmentation against the auto-calibrated gridlines, verified by
    overlay.  Captures the real Pregius features (B secondary lobe ~800 nm, G
    notch ~650 nm, R shoulder ~700 nm).  Effective RGB QE incl. Bayer+microlens.
    Only per-channel spectral SHAPE affects predicted contrast (height cancels in
    the Weber ratio).
    """
    global _imx250_data
    if _imx250_data is None:
        _csv = os.path.join(os.path.dirname(__file__), "imx250_spectral_response.csv")
        _imx250_data = np.loadtxt(_csv, delimiter=",", skiprows=1)
    lam = np.asarray(lam_nm, float)
    return (np.interp(lam, _imx250_data[:, 0], _imx250_data[:, 1], left=0.0, right=0.0) / 100.0,
            np.interp(lam, _imx250_data[:, 0], _imx250_data[:, 2], left=0.0, right=0.0) / 100.0,
            np.interp(lam, _imx250_data[:, 0], _imx250_data[:, 3], left=0.0, right=0.0) / 100.0)


CAMERA_PRESETS: dict[str, Callable] = {
    "imx250":      camera_imx250,
    "imx219":      camera_imx219,
    "icx282aq":    camera_icx282aq,
    "generic_rgb": camera_generic_rgb,
}


# ─────────────────────────────────────────────────────────────────────────────
# Material specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MaterialSpec:
    """Specification for one material's optical properties and layer structure.

    Args:
        name:          Human-readable name ('graphene', 'hbn', 'wse2', …)
        n_func:        Callable(lam_nm) → complex n̂(λ) array.
                       Use built-ins like n_graphene, or supply your own table.
        d_layer_nm:    Monolayer thickness in nm (graphene 0.335, hBN 0.333,
                       WSe₂ 0.648).
        max_layers:    Maximum layer count to include in the LUT.
        is_bulk_approx: True if bulk n̂ is used for monolayers — triggers a
                        warning in the log per Jessen et al. recommendation.
    """
    name:          str
    n_func:        Callable
    d_layer_nm:    float
    max_layers:    int  = 4
    is_bulk_approx: bool = False


# ── Convenience presets ───────────────────────────────────────────────────────
GRAPHENE = MaterialSpec("graphene",  n_graphene,  d_layer_nm=0.335, max_layers=6)
GRAPHITE = MaterialSpec("graphite",  n_graphite,  d_layer_nm=0.335, max_layers=6)
HBN      = MaterialSpec("hbn",       n_hbn,       d_layer_nm=0.333, max_layers=6)
WSE2     = MaterialSpec("wse2",      n_wse2,      d_layer_nm=0.648, max_layers=4,
                        is_bulk_approx=True)

# Registry keyed by lowercase name — the material menu the oxide-first calibration
# offers.  Each supplies its n̂(λ) and monolayer thickness; targets derive from the
# SAME measured oxide + lamp, per material (see analytical_calibration).
MATERIALS = {m.name: m for m in (GRAPHENE, GRAPHITE, HBN, WSE2)}


# ─────────────────────────────────────────────────────────────────────────────
# Forward model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForwardModel:
    """Compute expected RGB pixel contrast for 2D materials on SiO₂/Si.

    Implements Jessen et al. eq. (1a-c) integrated over the source spectrum
    and camera response.  Output is a lookup table used by the classifier.

    Args:
        materials:      List of MaterialSpec objects.
        sio2_nm:        SiO₂ thickness in nm (90 or 300 are common standards).
        source:         Source spectrum function(lam_nm) → normalised array.
                        Default: 3200 K blackbody.
        optics:         Path transmission T(λ)→[0,1] between source and sensor
                        (IR-cut filter, coatings…).  Default: all-pass.  Use
                        ir_cut_filter() or measured_transmission().
        camera:         Camera response preset name or callable returning
                        (r, g, b) sensitivity arrays.
        lam_nm:         Wavelength grid for integration.
        na:             Numerical aperture.  For NA ≥ 0.7, angle-averaging is
                        applied (Gaussian weight σ = θ_NA/3) per Gorbachev
                        et al.  Below 0.7, normal-incidence approximation used.
    """
    materials:  list[MaterialSpec]
    sio2_nm:    float = 300.0
    source:     Callable = field(default_factory=lambda: blackbody)
    optics:     Callable = field(default_factory=lambda: unity_transmission)
    camera:     str | Callable = "imx250"   # the Alvium 1800 U-508c sensor
    lam_nm:     np.ndarray = field(default_factory=lambda: LAM_NM.copy())
    na:         float = 0.0

    def __post_init__(self):
        self._lam = np.asarray(self.lam_nm, float)
        self._src = self.source(self._lam) * self.optics(self._lam)

        if callable(self.camera):
            self._rgb = self.camera(self._lam)
        else:
            fn = CAMERA_PRESETS.get(self.camera)
            if fn is None:
                raise ValueError(f"Unknown camera preset '{self.camera}'. "
                                 f"Options: {list(CAMERA_PRESETS)}")
            self._rgb = fn(self._lam)  # (r, g, b) each shape (NL,)

        # Substrate (bare) reflectance
        n2 = n_sio2(self._lam)
        n3 = n_si(self._lam)
        self._R_sub = self._stack_reflectance([], [], n2, n3)
        self._I_sub = self._to_rgb(self._R_sub)  # (3,) floats

    def _stack_reflectance(
        self,
        n_inner: list[np.ndarray],
        d_nm: list[float],
        n_sio2_: np.ndarray,
        n_si_: np.ndarray,
    ) -> np.ndarray:
        """Compute R(λ) for air → inner layers → SiO₂ → Si."""
        if self.na >= 0.7:
            return self._angle_averaged_reflectance(n_inner, d_nm, n_sio2_, n_si_)
        n_air = np.ones(len(self._lam), complex)
        return reflectance(
            [*n_inner, n_sio2_], [*d_nm, self.sio2_nm],
            self._lam, n_in=n_air, n_sub=n_si_,
        )

    def _angle_averaged_reflectance(
        self, n_inner, d_nm, n_sio2_, n_si_
    ) -> np.ndarray:
        """Integrate R over acceptance cone; Gaussian weight σ = θ_NA/3.

        Gorbachev et al. Small 7, 465, Fig. 2 caption.
        """
        theta_max = np.arcsin(self.na)
        sigma_theta = theta_max / 3.0
        thetas = np.linspace(0, theta_max, 7)
        weights = np.exp(-0.5 * (thetas / sigma_theta) ** 2)
        weights /= weights.sum()
        n_air = np.ones(len(self._lam), complex)
        R_avg = np.zeros(len(self._lam))
        for theta, w in zip(thetas, weights):
            cos_t = np.cos(theta)
            # Scale layer thicknesses by 1/cos(theta) for oblique incidence
            d_eff = [d / cos_t for d in d_nm] + [self.sio2_nm / cos_t]
            R_avg += w * reflectance(
                [*n_inner, n_sio2_], d_eff, self._lam,
                n_in=n_air, n_sub=n_si_,
            )
        return R_avg

    def _to_rgb(self, R_lam: np.ndarray) -> np.ndarray:
        """Integrate R(λ) against source × camera sensitivities → (R, G, B)."""
        r, g, b = self._rgb
        src = self._src
        return np.array([
            trapezoid(R_lam * src * r, self._lam),
            trapezoid(R_lam * src * g, self._lam),
            trapezoid(R_lam * src * b, self._lam),
        ])

    def build_lut(self) -> dict[tuple[str, int], tuple[float, float, float]]:
        """Build contrast lookup table for all (material, layer-count) pairs.

        Returns:
            Dict mapping (material_name, n_layers) → (C_R, C_G, C_B).
            Contrast C_ch = (I_BG_ch − I_film_ch) / I_BG_ch (positive =
            flake is darker than substrate, following Blake et al.).
        """
        n2 = n_sio2(self._lam)
        n3 = n_si(self._lam)
        lut: dict[tuple[str, int], tuple[float, float, float]] = {}

        for mat in self.materials:
            if mat.is_bulk_approx:
                warnings.warn(
                    f"{mat.name}: bulk refractive indices used for monolayer — "
                    "contrast accuracy may be reduced for N=1. "
                    "Jessen et al. note this is acceptable for WSe₂ layer discrimination "
                    "but flag it here for awareness.",
                    stacklevel=2,
                )
            n_mat = mat.n_func(self._lam)
            for n_layers in range(1, mat.max_layers + 1):
                d_tot = n_layers * mat.d_layer_nm
                R_film = self._stack_reflectance([n_mat], [d_tot], n2, n3)
                I_film = self._to_rgb(R_film)
                contrast = tuple(
                    float((bg - film) / bg) if bg > 0 else 0.0
                    for bg, film in zip(self._I_sub, I_film)
                )
                lut[(mat.name, n_layers)] = contrast  # type: ignore[assignment]

        return lut

    def substrate_rgb(self) -> tuple[float, float, float]:
        """Normalised RGB for clean substrate at this configuration."""
        return tuple(float(v) for v in self._I_sub)  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Substrate self-calibration:  measured bare-substrate BGR  →  oxide d  →  targets
# The global modal substrate the scanner already computes is the in-situ integral
# ∫ R_sub(λ,d)·S(λ)·T(λ)·QE(λ) dλ through the REAL path — so it captures whatever
# IR-cut / transmission the optics impose, wherever they sit, and auto-recalibrates
# every scan (robust to lamp/camera disturbance).  Given S, T, QE, the substrate
# CHROMATICITY pins d (exposure/gain/brightness cancel); d then yields the per-layer
# contrast targets analytically — replacing the empirical contrast_calibration.json.
# ─────────────────────────────────────────────────────────────────────────────

def fit_oxide_from_substrate(measured_bgr, *, materials=None, source=None,
                             optics=None, camera="imx250", na=0.0,
                             d_grid=None):
    """Best-fit SiO₂ thickness whose modelled bare-substrate colour matches the
    measured global-modal substrate BGR (by chromaticity — exposure-invariant).

    measured_bgr: (B, G, R) of the clean substrate (any scale).
    Returns (best_d_nm, fitted_ForwardModel).
    """
    if source is None:
        source = blackbody
    if optics is None:
        optics = unity_transmission
    if materials is None:
        materials = [GRAPHENE]
    if d_grid is None:
        d_grid = np.arange(250.0, 340.0, 1.0)
    meas = np.asarray(measured_bgr, float)
    meas_chroma = meas / max(meas.sum(), 1e-9)
    best = None
    for d in d_grid:
        fm = ForwardModel(materials, sio2_nm=float(d), source=source,
                          optics=optics, camera=camera, na=na)
        r, g, b = fm.substrate_rgb()
        bgr = np.array([b, g, r], float)
        chroma = bgr / max(bgr.sum(), 1e-9)
        err = float(np.sum((chroma - meas_chroma) ** 2))
        if best is None or err < best[0]:
            best = (err, float(d), fm)
    return best[1], best[2]


# ─────────────────────────────────────────────────────────────────────────────
# Spectroscopic reflectometry:  measured R_oxide(λ)  →  oxide d  (direct, ±nm)
# A fibre spectrometer (Red Tide) on the oxide region vs bare Si gives an ABSOLUTE
# reflectance spectrum whose interference fringes pin d far more tightly than RGB
# colour — the fringe *positions* fix d; an overall scale (spot-to-spot coupling)
# is fitted out.  This is the #19 measurement done in-house.
# ─────────────────────────────────────────────────────────────────────────────

def oxide_reflectance_from_spectra(wl, sig_oxide, sig_si, dark=None):
    """Absolute oxide reflectance from three Red-Tide spectra (source cancels):

        R_oxide(λ) = (sig_oxide − dark)/(sig_si − dark) · R_Si(λ)

    R_Si is the smooth bare-Si Fresnel reflectance |(1−n)/(1+n)|² (no oxide, no
    fringes).  Returns (wl, R_oxide).  Points where the Si signal is ≤0 are dropped.
    """
    wl = np.asarray(wl, float)
    ox = np.asarray(sig_oxide, float)
    si = np.asarray(sig_si, float)
    d = np.zeros_like(wl) if dark is None else np.asarray(dark, float)
    n = n_si(wl)
    R_si = np.abs((1.0 - n) / (1.0 + n)) ** 2
    net_si = si - d
    ok = net_si > 1e-6
    R = np.where(ok, (ox - d) / np.where(ok, net_si, 1.0) * R_si, np.nan)
    return wl[ok], R[ok]


def fit_oxide_from_reflectance(wl, R_meas, *, d_grid=None, lo=430.0, hi=750.0):
    """Fit SiO₂ thickness to a measured absolute reflectance spectrum R_oxide(λ).

    The fringe positions fix d; an overall scale α (spot-to-spot fibre coupling,
    focus) is solved analytically per candidate d, so only the spectral *shape*
    matters.  Restricts to [lo,hi] nm where the lamp+CCD have signal and R_Si is
    well behaved.

    Returns (best_d_nm, model_dict) where model_dict has 'wl', 'R_model' (scaled
    to the data), 'residual' (normalised RMS), and 'd_nm'.
    """
    wl = np.asarray(wl, float)
    R_meas = np.asarray(R_meas, float)
    m = np.isfinite(R_meas) & (wl >= lo) & (wl <= hi)
    wl_f, R_f = wl[m], R_meas[m]
    if wl_f.size < 8:
        raise ValueError("too few valid reflectance points in the fit band")
    if d_grid is None:
        d_grid = np.arange(50.0, 500.0, 0.5)
    n_ox = n_sio2(wl_f)
    n_sub = n_si(wl_f)
    n_air = np.ones(wl_f.size, complex)
    best = None
    denom = float(np.sum(R_f * R_f))
    for d in d_grid:
        R_mod = reflectance([n_ox], [float(d)], wl_f, n_in=n_air, n_sub=n_sub)
        num = float(np.sum(R_f * R_mod))
        den = float(np.sum(R_mod * R_mod))
        alpha = num / den if den > 1e-12 else 0.0          # analytic best scale
        resid = float(np.sqrt(np.mean((R_f - alpha * R_mod) ** 2)))
        if best is None or resid < best[0]:
            best = (resid, float(d), alpha, R_mod)
    resid, d_best, alpha, R_mod = best
    rms_norm = resid / max(np.sqrt(np.mean(R_f ** 2)), 1e-9)
    return d_best, {"wl": wl_f, "R_model": alpha * R_mod, "R_meas": R_f,
                    "residual": rms_norm, "d_nm": d_best}


def contrast_targets(fm, material="graphene", layers=(1, 2, 3)):
    """Per-layer BGR contrast in the contrast_calibration.json convention
    (mean_bgr_pct = signed Weber %, NEGATIVE = darker), from a ForwardModel."""
    lut = fm.build_lut()
    out = {}
    for N in layers:
        key = (material, N)
        if key in lut:
            C_R, C_G, C_B = lut[key]        # positive = darker
            out[N] = [round(-C_B * 100, 3), round(-C_G * 100, 3), round(-C_R * 100, 3)]
    return out


def layer_ladder(oxide_nm, *, source=None, optics=None, camera="imx250", na=0.0,
                 max_layers=30, noise_pct=1.5, k=2.0,
                 material_n=n_graphene, d_layer_nm=0.335):
    """Graphene layer-number contrast ladder + resolvable bands for one oxide, from
    the forward model alone.

    The BGR Weber-contrast vector traces a curve vs layer number N.  Because
    reflectance is non-monotonic in thickness (it turns at the reflectance node),
    the curve folds back on itself, so a scalar (e.g. projection) ALIASES — but the
    full BGR vector is locally unique except at the fold.  A rung N is "resolvable"
    iff its nearest OTHER rung in BGR space exceeds the measurement uncertainty
    ``k · noise_pct · √3`` (this single test catches both poor adjacent spacing and
    self-intersection).  Resolvability, and the range of countable layers, is a
    FUNCTION OF THE OXIDE — thin oxide (~100 nm) resolves many few-layer rungs plus a
    thick branch; ~130–210 nm sits near graphene-invisibility and resolves nothing.

    Returns dict:
        oxide_nm, noise_pct, k, threshold_pct
        targets:          {N: [B, G, R]}   signed Weber %, negative = darker
        nn_dist_pct:      {N: float}       distance to nearest other rung (BGR %)
        magnitude_pct:    {N: float}       |contrast| (detectability)
        layer_sigma:      {N: float}       layer-number uncertainty (LAYERS) =
                          noise_pct / local |dBGR/dN|.  ~0.1 L at monolayer, ~1 L at
                          the reflectance turn — a per-flake confidence like area.
        resolvable:       {N: bool}
        resolvable_bands: [(N_lo, N_hi), ...]  contiguous resolvable ranges
    """
    if source is None:
        source = blackbody
    spec = MaterialSpec("graphene", material_n, d_layer_nm=d_layer_nm,
                        max_layers=int(max_layers))
    kw = dict(sio2_nm=float(oxide_nm), source=source, camera=camera, na=na)
    if optics is not None:
        kw["optics"] = optics
    fm = ForwardModel([spec], **kw)
    layers = tuple(range(1, int(max_layers) + 1))
    targets = contrast_targets(fm, material="graphene", layers=layers)
    ns = sorted(targets)
    V = np.array([targets[N] for N in ns], float)              # (M, 3) BGR
    thr = float(k * noise_pct * np.sqrt(3))
    nn = {}
    resolvable = {}
    magnitude = {}
    layer_sigma = {}
    for i, N in enumerate(ns):
        others = np.delete(V, i, axis=0)
        d = float(np.linalg.norm(others - V[i], axis=1).min()) if len(others) else np.inf
        nn[N] = round(d, 3)
        resolvable[N] = bool(d > thr)
        magnitude[N] = round(float(np.linalg.norm(V[i])), 3)
        # Layer-number uncertainty (in LAYERS), honest through the reflectance turn:
        #  (a) local resolution: a per-channel error σ projects onto the ladder tangent
        #      → σ_local = noise_pct / |dBGR/dN|.  Tiny at monolayer.
        #  (b) aliasing spread: the σ must also cover EVERY rung whose BGR is
        #      indistinguishable from this one within the threshold — at the fold that
        #      includes non-adjacent rungs, so σ blows up (a ~12 L flake is ~12 ± a
        #      few, not ± 0.2).  Take the larger; the guess is still the nearest rung.
        steps = []
        if i > 0:
            steps.append(float(np.linalg.norm(V[i] - V[i - 1])))
        if i + 1 < len(V):
            steps.append(float(np.linalg.norm(V[i + 1] - V[i])))
        slope = (sum(steps) / len(steps)) if steps else 1e-6
        sigma_local = noise_pct / max(slope, 1e-6)
        alias = np.array(ns)[np.linalg.norm(V - V[i], axis=1) <= thr]   # rungs within noise
        sigma_alias = float(np.std(alias)) if alias.size > 1 else 0.0
        layer_sigma[N] = round(max(sigma_local, sigma_alias), 2)
    # contiguous resolvable bands
    bands = []
    start = None
    for N in ns:
        if resolvable[N] and start is None:
            start = N
        if (not resolvable[N]) and start is not None:
            bands.append((start, N - 1)); start = None
    if start is not None:
        bands.append((start, ns[-1]))
    return {
        "oxide_nm": round(float(oxide_nm), 1),
        "noise_pct": noise_pct, "k": k, "threshold_pct": round(thr, 3),
        "targets": targets, "nn_dist_pct": nn, "magnitude_pct": magnitude,
        "layer_sigma": layer_sigma, "resolvable": resolvable, "resolvable_bands": bands,
    }


def analytical_calibration(oxide_nm, material="graphene", *, source=None, optics=None,
                           camera="imx250", na=0.0, max_layers=50,
                           noise_pct=1.5, k=2.0):
    """Full contrast-calibration dict for a MEASURED oxide + a chosen material — the
    oxide-first, per-material derivation (roadmap #61).  The oxide is measured once
    (substrate property); this turns it into that material's BGR targets, resolvable
    layer bands, and per-rung layer_sigma.  Works for any entry in MATERIALS
    (graphene / hbn / wse2 / …) — each supplies its own n̂ and monolayer thickness.

    Returns the contrast_calibration.json shape the detector consumes:
      method, sio2_nm, material, source, substrate_bgr, resolvable_layer_bands,
      monolayer_contrast_pct, targets (thin resolvable band → segmentation),
      ladder (all rungs, each with mean_bgr_pct + layer_sigma + resolvable).
    """
    if source is None:
        source = blackbody
    spec = MATERIALS.get(str(material).lower())
    if spec is None:
        raise ValueError(f"unknown material '{material}'; options: {sorted(MATERIALS)}")

    lad = layer_ladder(oxide_nm, source=source, optics=optics, camera=camera, na=na,
                       max_layers=max_layers, noise_pct=noise_pct, k=k,
                       material_n=spec.n_func, d_layer_nm=spec.d_layer_nm)
    tg, bands = lad["targets"], lad["resolvable_bands"]

    # bare-substrate colour (for reference / the map), from a matching ForwardModel
    kw = dict(sio2_nm=float(oxide_nm), source=source, camera=camera, na=na)
    if optics is not None:
        kw["optics"] = optics
    b, g, r = None, None, None
    try:
        rr, gg, bb = ForwardModel([spec], **kw).substrate_rgb()
        b, g, r = round(bb, 4), round(gg, 4), round(rr, 4)
    except Exception:  # noqa: BLE001 — substrate colour is optional metadata
        pass

    thin_hi = next((hi for lo, hi in bands if lo == 1), 3)
    return {
        "method": "analytical_forward_model",
        "sio2_nm": round(float(oxide_nm), 1),
        "material": spec.name,
        "resolvable_layer_bands": bands,
        "monolayer_contrast_pct": round(lad["magnitude_pct"].get(1, 0.0), 2),
        "substrate_bgr": [b, g, r] if b is not None else None,
        "targets": {str(N): {"layer_count": N, "mean_bgr_pct": [round(v, 3) for v in tg[N]]}
                    for N in range(1, thin_hi + 1) if N in tg},
        "ladder": {str(N): {"layer_count": N,
                            "mean_bgr_pct": [round(v, 3) for v in tg[N]],
                            "layer_sigma": lad["layer_sigma"].get(N),
                            "resolvable": bool(lad["resolvable"].get(N))}
                   for N in sorted(tg)},
    }


def calibration_from_substrate(measured_bgr, *, material="graphene", **fit_kwargs):
    """One-shot analytical calibration: measured substrate BGR → fit d → targets.

    Returns a dict shaped like contrast_calibration.json (method 'forward_model'),
    a drop-in for load_targets()/the detector — no marked flakes required.
    """
    d, fm = fit_oxide_from_substrate(measured_bgr, **fit_kwargs)
    targets = contrast_targets(fm, material=material)
    return {
        "method": "forward_model",
        "sio2_nm": round(d, 1),
        "substrate_bgr": [round(float(v), 1) for v in measured_bgr],
        "targets": {str(N): {"layer_count": N, "mean_bgr_pct": t}
                    for N, t in targets.items()},
    }
