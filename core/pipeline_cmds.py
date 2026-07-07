"""Subprocess argv builders for the flake-detection pipeline (plan L8).

This is the ONLY place subprocess argv for the detection pipeline is built.
Both surfaces call these functions so they cannot drift:

  * CLI  — ``graphene_detection_pipeline.py`` (detect → map)
  * GUI  — ``ui/area_wafer_scan_panel.py`` Detect & Map steps + Quick Stitch

Each function returns a plain ``list[str]`` ready for ``subprocess`` /
``QProcess``:  ``[python, ('-u',) script, args...]``.  ``python`` defaults to
``sys.executable``; pass ``unbuffered=True`` to insert ``-u`` (the GUI does,
so progress lines stream in real time).  Tool paths resolve repo-relative
from this file, matching the old inline ``_tool()`` / ``REPO`` lookups.

Flag ORDER within an argv is canonical here (anchored to the CLI pipeline's
historical order, which is echoed to the user).  The GUI call sites
historically used slightly different orders for ``make_map.py``; argparse is
order-insensitive so behaviour is identical — tests freeze both the exact
canonical argv and flag-set equality with the legacy argv.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _tool(name: str) -> str:
    return str(_REPO / "tools" / name)


def _base(python: str | None, unbuffered: bool, script: str) -> list[str]:
    cmd = [python or sys.executable]
    if unbuffered:
        cmd.append("-u")   # force unbuffered stdout → real-time progress lines
    cmd.append(script)
    return cmd


def detect_cmd(scan, calibration, *, timing: bool = False, extra=(),
               python: str | None = None, unbuffered: bool = False) -> list[str]:
    """vision/flake_detect_v3.py — detect candidates against calibrated targets."""
    cmd = _base(python, unbuffered, str(_REPO / "vision" / "flake_detect_v3.py"))
    cmd += [str(scan), "--calibration", str(calibration)]
    if timing:
        cmd.append("--timing")
    cmd += [str(a) for a in extra]
    return cmd


def ladder_cmd(scan, *, rank: str, per_layer: int, dedup_um: float,
               min_area: float | None = None, max_area: float | None = None,
               min_circularity: float | None = None,
               python: str | None = None, unbuffered: bool = False) -> list[str]:
    """tools/calibrate_ladder.py — self-calibrating detect (#49 wand)."""
    cmd = _base(python, unbuffered, _tool("calibrate_ladder.py"))
    cmd += [str(scan), "--rank", str(rank),
            "--per-layer", str(per_layer), "--dedup-um", str(dedup_um)]
    if min_area is not None:
        cmd += ["--min-area", str(min_area)]
    if max_area is not None:
        cmd += ["--max-area", str(max_area)]
    if min_circularity is not None:
        cmd += ["--min-circularity", str(min_circularity)]
    return cmd


def import_cmd(scan, *, dedup_um: float, per_layer: int | None = None,
               rank: str | None = None, min_area: float | None = None,
               max_area: float | None = None, ids=None,
               python: str | None = None, unbuffered: bool = False) -> list[str]:
    """tools/import_found_flakes.py — import candidates into the sample catalogue.

    ``per_layer``/``rank`` (+ area gates) shortlist v3's every-candidate output
    at import time (the analytical detect path); leave None to import as-is.
    ``ids`` (sequence of candidate-id strings, or a pre-joined comma string)
    imports exactly those candidates, bypassing the shortlist filters — the
    Flake Results panel's "Add checked to Catalogue" path.
    """
    cmd = _base(python, unbuffered, _tool("import_found_flakes.py"))
    cmd += [str(scan), "--dedup-um", str(dedup_um)]
    if per_layer is not None:
        cmd += ["--per-layer", str(per_layer)]
    if rank is not None:
        cmd += ["--rank", str(rank)]
    if min_area is not None:
        cmd += ["--min-area", str(min_area)]
    if max_area is not None:
        cmd += ["--max-area", str(max_area)]
    if ids is not None:
        joined = ids if isinstance(ids, str) else ",".join(str(i) for i in ids)
        cmd += ["--ids", joined]
    return cmd


def map_cmd(scan, *, rotation_model: bool = True, name: str | None = None,
            open_browser: bool = False, nav_port: int = 0,
            sample_json: str | None = None, cand_sort: str | None = None,
            cand_top: int | None = None, correct_bg: bool = False,
            timing: bool = False,
            python: str | None = None, unbuffered: bool = False) -> list[str]:
    """tools/make_map.py — stitch + HTML viewer (candidate overlay optional).

    Conditional flags mirror the historical call sites: ``--nav-port`` only
    when nonzero, ``--sample-json`` only when found, ``--correct-bg`` only
    when chosen, ``--cand-sort``/``--cand-top`` only when given (CLI always
    passes them; the GUI never does).
    """
    cmd = _base(python, unbuffered, _tool("make_map.py"))
    if rotation_model:
        cmd.append("--rotation-model")
    cmd.append(str(scan))
    if cand_sort is not None:
        cmd += ["--cand-sort", str(cand_sort)]
    if cand_top is not None:
        cmd += ["--cand-top", str(cand_top)]
    if timing:
        cmd.append("--timing")
    if correct_bg:
        cmd.append("--correct-bg")
    if sample_json:
        cmd += ["--sample-json", str(sample_json)]
    if nav_port:
        cmd += ["--nav-port", str(nav_port)]
    if name:
        cmd += ["--name", str(name)]
    if open_browser:
        cmd.append("--open")
    return cmd


def sheet_cmd(sample_dir, *, backfill: bool = True,
              python: str | None = None, unbuffered: bool = False) -> list[str]:
    """tools/catalogue_contact_sheet.py — contact-sheet PNG for the catalogue."""
    cmd = _base(python, unbuffered, _tool("catalogue_contact_sheet.py"))
    cmd.append(str(sample_dir))
    if backfill:
        cmd.append("--backfill")
    return cmd
