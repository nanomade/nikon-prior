# nikon-prior

Microscope imaging and stage control software for a **Nikon** optical
microscope with a **Prior Scientific ProScan III** motorised XYZ stage.

Derived from [standa-stacker](https://github.com/nanomade/standa_stacker)
but stripped of stacking-specific hardware (heater, manipulator, eucentric
rotation) and adapted for the ProScan III serial protocol.

## Features

- Live camera preview with scale bar, HUD, full-screen crosshair
- Per-objective exposure and focus presets
- Two-phase autofocus (Laplacian variance / Tenengrad)
- RBF focus height map
- Automated wafer mosaic scanning
- Index mark navigation
- Flat-field vignetting correction
- Layer contrast overlay (monolayer / bilayer / trilayer detection)
- Xbox controller fly-by-wire stage control
- Named position manager with thumbnail breadcrumbs

## Hardware requirements

- Prior Scientific ProScan III controller (XYZ stage)
- Allied Vision Alvium 1800 U-508c camera (USB3 Vision, 2464×2056)
- Windows 10/11 or Linux (Ubuntu 22.04+)

## Installation

### Python dependencies

```bash
pip install pyqt5 opencv-python-headless numpy pyserial pygame vmbpy
```

### Allied Vision Vimba X SDK

`vmbpy` requires the Vimba X runtime to be installed first.

1. Download **Vimba X** from the [Allied Vision website](https://www.alliedvision.com/en/products/vimba-sdk/)
2. Run the installer (Windows) or follow the Linux steps below
3. Install the USB transport layer (Linux only):
   ```bash
   sudo ./VimbaX_<ver>/cti/VimbaUSBTL_Install.sh
   ```
4. Add your user to the `dialout` group (Linux, for serial port access):
   ```bash
   sudo usermod -aG dialout $USER
   ```

If no Alvium camera is detected at startup, the application falls back to
a synthetic `MockCameraManager` so all UI features remain usable offline.

If the ProScan III is not detected, the application likewise falls back to
a software mock motor manager.

## Serial connection

Default: 9600 baud, 8N1, no flow control.  If your controller is set
to a different baud rate, edit `PriorMotorManager(baudrate=...)` in
`motors/factory.py` or update the controller front-panel setting.

Ensure your user is in the `dialout` group:
```bash
sudo usermod -aG dialout $USER
```

## Calibration

1. **Step size**: `motors/step_config.json` — default 0.1 µm/count
   (Prior standard).  Verify with a stage micrometer.
2. **Scale bar**: update `SCALE_BAR_TABLE` in `ui/preview.py` for
   your camera + tube lens combination.
3. **Axis direction**: set `invert` flags in `step_config.json` so
   that positive X/Y correspond to the expected stage directions.
