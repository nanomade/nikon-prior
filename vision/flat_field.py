"""vision/flat_field.py — scan flat-field (vignetting) correction.

Extracted verbatim from vision/flake_detect.py so the legacy v2 detector module
can eventually retire without taking the flat-field logic with it. Builds a
per-pixel flat field from sampled scan frames (cached as flat_field.npy in the
scan folder) and applies it as a luminance-only gain map that preserves colour
balance.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from core.scan_io import load_metadata


def compute_flat_field(
    scan_folder,
    *,
    method: str = "median",
    n_sample: int = 128,
    recompute: bool = False,
    focused_only: bool = False,
    progress_cb=None,
) -> np.ndarray:
    """Build a per-pixel flat field from scan frames for vignetting correction.

    Args:
        scan_folder:  Scan folder containing scan_metadata.json and frames.
        method:       'median' — robust sample-based median (recommended);
                      'mean'   — running mean of all frames (slightly faster).
        n_sample:     Number of frames to sample for the median estimate.
                      128 is statistically equivalent to using all frames when
                      flake coverage is low (typical for wafer scans).
        recompute:    If True, ignore any cached flat_field.npy and recompute.
        focused_only: If True, use only frames with focus_ok=True.  Better
                      flat field when autofocus was reliable for some frames.
        progress_cb:  Optional callable(current, total) for progress reporting.

    Returns:
        float32 BGR array (H, W, 3).  Also saved to flat_field.npy for reuse.
    """
    folder = Path(scan_folder)
    cache_path = folder / "flat_field.npy"
    if cache_path.exists() and not recompute:
        if progress_cb:
            progress_cb(1, 1)
        return np.load(str(cache_path))

    meta = load_metadata(folder)

    entries = meta.get("images", [])
    if focused_only:
        entries = [e for e in entries if e.get("focus_ok", True)]
    all_paths = [folder / e["filename"] for e in entries if e.get("filename")]
    all_paths = [p for p in all_paths if p.exists()]
    if not all_paths:
        raise FileNotFoundError("No frame images found in scan folder")

    if method == "median":
        sample = random.sample(all_paths, min(n_sample, len(all_paths)))
        frames = []
        for i, p in enumerate(sample):
            f = cv2.imread(str(p))
            if f is not None:
                frames.append(f)
            if progress_cb:
                progress_cb(i + 1, len(sample))
        flat = np.median(np.stack(frames), axis=0).astype(np.float32)
    else:  # mean — single pass, O(1) frame memory
        acc = None
        n = 0
        for i, p in enumerate(all_paths):
            f = cv2.imread(str(p))
            if f is None:
                continue
            acc = f.astype(np.float64) if acc is None else acc + f.astype(np.float64)
            n += 1
            if progress_cb:
                progress_cb(i + 1, len(all_paths))
        flat = (acc / n).astype(np.float32)

    np.save(str(cache_path), flat)
    return flat


def apply_flat_field(frame: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Apply luminance-based vignetting correction, preserving colour balance.

    Uses the flat-field luminance (mean across channels) as a single gain map
    applied equally to all three channels — same method as flat_field_panel.py.
    Per-channel division would remove the substrate hue and break ForwardModel
    contrast classification.
    """
    flat_lum = flat.mean(axis=2, keepdims=True)
    gain = flat_lum.max() / np.clip(flat_lum, 1.0, None)
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
