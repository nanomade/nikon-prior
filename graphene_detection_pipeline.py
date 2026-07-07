#!/usr/bin/env python3
"""End-to-end graphene flake detection: area scan → zoomable map with detected,
layer-grouped, area-sorted flake candidates (thumbnail list + scale bars).

One command, no manual steps. Runs the two committed tools in order:

    1. vision/flake_detect_v3.py   — detect candidates against the global modal
       substrate, classify by layer, write flake_candidates_v3.json.
    2. tools/make_map.py           — stitch (rotation model) + build the HTML
       viewer with the candidate overlay, thumbnails and scale bars, selecting
       the top-N per layer by area.

The ONLY external input is the oxide.  Two equivalent ways to give it:
    --oxide-nm 100     derive BGR targets analytically (ForwardModel + measured
                       lamp E(λ)) from the SiO₂ thickness you measured (e.g. Red
                       Tide reflectometry).  No labelled flakes, no prior file.
    --calibration P    a contrast_calibration.json (BGR targets) from this or any
                       SAME-OXIDE chip — targets transfer across identical oxide.
If neither is given and the scan folder already has a contrast_calibration.json,
it is used automatically.

The detector rejects non-flake junk itself (achromatic grey shadows / tape halos
via a chromaticity gate, glints via a bright-fraction gate, blurry/non-uniform
regions via uniformity + solidity gates), so the top-N-per-layer map is clean
without any manual post-filtering.

Usage:
    .venv/bin/python graphene_detection_pipeline.py <scan_folder> \\
        (--oxide-nm NM | --calibration PATH) [--source-csv E.csv] [--name NAME] \\
        [--sort-by area|score] [--top-per-layer N] [--open]
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

from core.pipeline_cmds import detect_cmd, map_cmd

REPO = Path(__file__).resolve().parent

# The running app's MapNavServer prefers this port (ui/map_nav_server.py
# _PREFERRED_PORT). If it answers /ping, the app is up and its maps can drive
# the stage — so we bake the port in and shift+click navigate works.
_NAV_PREFERRED_PORT = 57373


def _detect_nav_port() -> int:
    """Return the live MapNavServer port if the app is running, else 0."""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{_NAV_PREFERRED_PORT}/ping", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=0.5):
            return _NAV_PREFERRED_PORT
    except Exception:
        return 0


def _run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"{label} failed (exit {r.returncode}) — pipeline aborted.")


def _calibration_from_oxide(oxide_nm: float, source_csv: str | None, scan: Path,
                            material: str = "graphene") -> str:
    """Write a contrast_calibration.json analytically from an OXIDE THICKNESS +
    MATERIAL.  The physics lives in `vision.optical_contrast.analytical_calibration`
    (oxide-first, per-material — roadmap #61); this wrapper just resolves the lamp
    E(λ), calls it, writes the file, and prints the resolvable range.  Give it the
    thickness you measured (Red Tide reflectometry) — no labelled flakes needed.
    """
    from vision.optical_contrast import (
        analytical_calibration, measured_transmission, load_E, blackbody, ir_cut_filter)
    src_csv = source_csv or str(REPO / "E_measured.csv")
    if Path(src_csv).exists():
        source, optics, src_name = measured_transmission(*load_E(src_csv)), None, Path(src_csv).name
    else:
        print(f"  ! measured lamp {src_csv} not found — falling back to "
              f"blackbody 3200K + IR-cut (less accurate).", flush=True)
        source, optics, src_name = blackbody, ir_cut_filter(660, 30), "blackbody3200+ir_cut"

    out = analytical_calibration(oxide_nm, material, source=source, optics=optics, camera="imx250")
    out["source"] = src_name
    bands = out["resolvable_layer_bands"]
    p = scan / f"contrast_calibration_{material}_oxide{oxide_nm:g}nm.json"
    from vision.contrast_cal import write_contrast_calibration
    write_contrast_calibration(
        p, out,
        inputs=({'lamp_E': src_csv} if Path(src_csv).exists() else None),
        params={'oxide_nm': oxide_nm, 'material': material, 'camera': 'imx250'})
    print(f"  built analytical {material} calibration @ {oxide_nm:g} nm ({src_name}) → {p.name}", flush=True)
    for N in (1, 2, 3):
        if str(N) in out["targets"]:
            print(f"    {N}L BGR% = {out['targets'][str(N)]['mean_bgr_pct']}", flush=True)
    band_txt = ", ".join(f"{lo}-{hi}" for lo, hi in bands) or "NONE"
    print(f"  resolvable layer bands @ {oxide_nm:g} nm ({material}): {band_txt}  "
          f"(1L contrast {out['monolayer_contrast_pct']:.1f}%)", flush=True)
    if not bands:
        print(f"  ! WARNING: no {material} layers resolvable at this oxide — near an "
              "invisibility node. Layer counts here are unreliable.", flush=True)
    return str(p)


def run_pipeline(scan: Path, calibration: str, *, name: str | None = None,
                 sort_by: str = "area", top_per_layer: int = 20,
                 open_browser: bool = False, timing: bool = False,
                 nav_port: int | None = None) -> None:
    """Importable core of the CLI: detect → map on a resolved scan folder.

    argv for both steps comes from `core.pipeline_cmds` (the single source of
    subprocess argv for the pipeline), so the GUI call sites cannot drift from
    this CLI.  ``nav_port=None`` auto-detects the running app; 0 disables.
    """
    # detect_cmd/map_cmd default to sys.executable — the active (venv)
    # interpreter propagates to both tools.
    _run(detect_cmd(scan, calibration, timing=timing),
         "1/2  Detect flake candidates")

    port = nav_port if nav_port is not None else _detect_nav_port()
    if port:
        print(f"  Navigate enabled — MapNavServer on port {port}", flush=True)
    else:
        print("  Navigate disabled — app not detected (shift+click inert; "
              "re-run with the app open or pass --nav-port)", flush=True)

    _run(map_cmd(scan, cand_sort=sort_by, cand_top=top_per_layer,
                 timing=timing, nav_port=port or 0, name=name,
                 open_browser=open_browser),
         "2/2  Stitch map + candidate overlay")

    print("\nPipeline complete.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end graphene flake detection (scan → annotated map).")
    ap.add_argument("scan_folder", help="Area-scan folder (contains scan_metadata.json + tiles)")
    ap.add_argument("--calibration", default=None,
                    help="contrast_calibration.json (BGR targets) from this or any "
                         "SAME-OXIDE chip (default: <scan>/contrast_calibration.json)")
    ap.add_argument("--oxide-nm", type=float, default=None,
                    help="SiO₂ thickness (nm): derive BGR targets analytically from "
                         "the ForwardModel + measured lamp E(λ). Alternative to "
                         "--calibration; e.g. --oxide-nm 100 for a 100 nm wafer.")
    ap.add_argument("--source-csv", default=None,
                    help="measured lamp E(λ) CSV for --oxide-nm (default: E_measured.csv)")
    ap.add_argument("--material", default="graphene",
                    help="material for --oxide-nm target derivation "
                         "(graphene | hbn | wse2; default graphene)")
    ap.add_argument("--name", default=None, help="Output map file stem")
    ap.add_argument("--sort-by", choices=["area", "score"], default="area",
                    help="Candidate ranking (default: area)")
    ap.add_argument("--top-per-layer", type=int, default=20, metavar="N",
                    help="Keep top N candidates per layer count on the map (default: 20; 0 = all)")
    ap.add_argument("--open", action="store_true", help="Open the map in a browser when done")
    ap.add_argument("--timing", action="store_true",
                    help="Print a per-stage wall-clock breakdown for both the detector "
                         "and the map build (find the real bottleneck before GPU work — #58)")
    ap.add_argument("--nav-port", type=int, default=None, metavar="PORT",
                    help="MapNavServer port to bake in for shift+click navigate "
                         "(default: auto-detect the running app; 0 disables)")
    args = ap.parse_args()

    scan = Path(args.scan_folder).resolve()
    if not (scan / "scan_metadata.json").exists():
        sys.exit(f"Not a scan folder (no scan_metadata.json): {scan}")

    # Calibration source, in precedence order: explicit --calibration file, then
    # --oxide-nm (analytical targets), then a calibration already in the scan folder.
    cal = args.calibration
    if cal is None and args.oxide_nm is not None:
        cal = _calibration_from_oxide(args.oxide_nm, args.source_csv, scan, args.material)
    if cal is None:
        local = scan / "contrast_calibration.json"
        if local.exists():
            cal = str(local)
    if cal is None:
        # Default: the sample's latest derived optical calibration (from the Sample
        # panel's "Derive targets" — Red Tide oxide + ForwardModel, roadmap #61).
        try:
            from core.sample_data import find_sample_dir_upwards, latest_optical_calibration
            sdir = find_sample_dir_upwards(str(scan))
            if sdir:
                latest = latest_optical_calibration(sdir)
                if latest:
                    cal = latest
                    print(f"  using sample optical calibration → "
                          f"{Path(cal).relative_to(Path(sdir).parent)}", flush=True)
        except Exception:
            pass
    if cal is None:
        sys.exit("No calibration. Pass --oxide-nm <thickness> to derive targets "
                 "analytically, --calibration <contrast_calibration.json> from a "
                 "same-oxide chip, or derive targets in the Sample panel first "
                 "(saved under the sample's /optical/).")
    if not Path(cal).exists():
        sys.exit(f"Calibration file not found: {cal}")

    run_pipeline(scan, cal, name=args.name, sort_by=args.sort_by,
                 top_per_layer=args.top_per_layer, open_browser=args.open,
                 timing=args.timing, nav_port=args.nav_port)


if __name__ == "__main__":
    main()
