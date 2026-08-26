# Kalibr Workflow Guide

End-to-end guide for preparing mobile scan data for camera-IMU extrinsic/intrinsic calibration using [Kalibr](https://github.com/ethz-asl/kalibr).

## Overview

Kalibr's `kalibr_calibrate_imu_camera` needs three independent inputs, only two of which `yuv_sensor` produces directly:

```
Static IMU-only capture (phone lying still)
    ↓
auto_trim_static_imu (imu_trim.py)
    ↓
Trimmed imu.csv ──→ external Allan variance tool ──→ imu.yaml (noise_density, random_walk)

Checkerboard/AprilGrid capture (phone moving in view of the target)
    ↓
KalibrExporter (--export_kalibr)
    ↓
images/ + camchain.yaml (seed intrinsics) + imu.csv
    ↓
kalibr_bagcreater (images/ + imu.csv → rosbag)
    ↓
kalibr_calibrate_imu_camera (bag + camchain.yaml + imu.yaml + target.yaml)
    ↓
Output: camchain-imucam.yaml (refined camera-IMU extrinsics/intrinsics)
```

Two separate capture sessions are involved: a static one for `imu.yaml`, and a calibration-target one for `camchain.yaml` + the rosbag. Don't try to derive both from the same session.

## Prerequisites

- **yuv_sensor** installed: `pip install -e .`
- **Kalibr** installed (Docker image is the path of least resistance): see [Kalibr installation docs](https://github.com/ethz-asl/kalibr/wiki/installation). No prebuilt image is published — `git clone https://github.com/ethz-asl/kalibr.git && cd kalibr && docker build -t kalibr -f Dockerfile_ros1_20_04 .` (pick the Dockerfile matching the Ubuntu base you want; `_20_04` targets Noetic). Confirmed by building it: the image's `ENTRYPOINT` is a fixed `cd $WORKSPACE && /bin/bash` that ignores whatever `CMD` you pass — `docker run -it ... kalibr` for an interactive shell, or `docker run --rm --entrypoint bash ... kalibr -c '...'` to run one command non-interactively without hitting that.
- A physical calibration target (checkerboard or AprilGrid) and its `target.yaml` — `yuv_sensor` does not generate this
- An Allan variance tool to turn the trimmed static capture into `imu.yaml` (e.g. [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) or [kalibr_allan](https://github.com/rpng/kalibr_allan)) — not part of this package, and not part of Kalibr either: a built Kalibr image ships exactly three executables (`kalibr_calibrate_cameras`, `kalibr_calibrate_imu_camera`, `kalibr_calibrate_rs_cameras`) — no `kalibr_calibrate_imu`, despite the name pattern suggesting otherwise.

## Step 1: Trim the Static IMU Capture

Record a session with the phone completely motionless for as long as practical (hours, ideally) — this is what Allan variance analysis needs. Handling the phone to start/stop recording leaves real motion at both edges that has to be cut before analysis.

```bash
yuv-sensor --session_dir data/session_static_capture \
           --trim_imu_static both
```

This writes to `data/session_static_capture/imu_trimmed/`:

- `imu.csv` — motion trimmed off both ends
- `imu_raw.csv` — untrimmed copy, for comparison
- `trim_report.json` — baseline accel magnitude, seconds cut from each end, kept/dropped row counts

The trim works by scanning accelerometer magnitude deviation from the session's own median in 1-second bins, and only trusting a boundary once 30 consecutive seconds stay under threshold (see [`auto_trim_static_imu`](api_reference.md#auto_trim_static_imu)). If the mapper prints "no clean start/end boundary found," the capture likely has motion (or a sensor glitch) that never settles — check `imu_raw.csv` by hand.

Feed `imu_trimmed/imu.csv` into your Allan variance tool of choice to produce `imu.yaml`.

## Step 2: Export the Calibration-Target Capture

With the phone moving through frame in view of the checkerboard/AprilGrid, run the export on that session:

```bash
yuv-sensor --session_dir data/session_calib_target \
           --export_kalibr \
           --kalibr_output_dir output/kalibr
```

This generates:

- `output/kalibr/images/` — raw RGB frames, not undistorted (Kalibr fits its own distortion model)
- `output/kalibr/camchain.yaml` — seed intrinsics reformatted from `session.json`'s own calibration
- `output/kalibr/imu.csv` — this session's IMU log, copied as-is
- `output/kalibr/kalibr_export_metadata.json` — export metadata

Sanity-check `camchain.yaml` before using it: it's reformatted from the phone's manufacturer-reported calibration, not re-derived from the checkerboard frames. Comments in the file flag anything dropped in translation (`session.json`'s 5th distortion coefficient, lens skew) or worth double-checking (wide FOV, where `equidistant` may fit better than `pinhole`+`radtan`).

### Choosing the camera model

`pinhole`+`radtan` is the default because it matches the manufacturer's own distortion model, but a wide-FOV lens (check `fov_deg` in `session.json`) is exactly where `pinhole` tends to struggle — the projection itself gets numerically unstable as the half-FOV approaches 90°, which `equidistant` (fisheye) doesn't have a problem with. `KalibrExporter` flags this in `camchain.yaml`'s own comments whenever FOV exceeds 90°, but doesn't pick for you, because it isn't a decision a heuristic should make blind — run both and compare:

1. Write two `camchain.yaml` variants against the same intrinsics — `distortion_model: radtan` vs `distortion_model: equidistant` (equidistant uses a different coefficient set; drop `p1`/`p2` and see the [Kalibr wiki](https://github.com/ethz-asl/kalibr/wiki/supported-models) for its parameterization).
2. Run `kalibr_calibrate_cameras` (or just the intrinsics stage of `kalibr_calibrate_imu_camera`) once per variant, against the *same* image set and `target.yaml`.
3. Compare the `report-cam.pdf`/`.txt` each run produces:

   | What to check | How to read it |
   | --- | --- |
   | Mean/RMS reprojection error (px) | Lower wins. ~0.1-0.3px is a good fit; approaching or past 1px means the model doesn't fit the data |
   | Residual scatter plot | Should look like small, roughly uniform noise across the whole frame. Error that grows toward the frame **edges** specifically is the signature of a wide-FOV model mismatch — pinhole's usual failure mode |
   | Fitted parameters | Focal length should stay close to the manufacturer seed (`session.json`'s `intrinsics`); distortion coefficients blown up far past typical radtan/equidistant magnitudes means the model is straining to compensate for a shape it can't represent |

4. Pick the model with lower reprojection error **and** no edge-growing residual pattern. The two usually agree, but a low mean error with a clear edge pattern is still the worse choice — that pattern biases the geometry Kalibr recovers even though the average looks fine. For a lens this wide, `equidistant` more often wins, but that's a prior this comparison is meant to override when the data disagrees.

## Step 3: Get a target.yaml

Kalibr needs the physical dimensions of your calibration target. Follow [Kalibr's target format docs](https://github.com/ethz-asl/kalibr/wiki/calibration-targets) to write `target.yaml` for whichever board/grid you printed. This isn't something `yuv_sensor` can generate — it depends on your physical target.

## Step 4: Pack into a rosbag

Kalibr's calibration tools consume a rosbag, not a raw image + CSV directory:

```bash
kalibr_bagcreater \
    --folder output/kalibr \
    --output-bag output/kalibr/calib.bag
```

This expects `output/kalibr/images/` and `output/kalibr/imu.csv` in the layout `KalibrExporter` already produces.

## Step 5: Run kalibr_calibrate_imu_camera

```bash
kalibr_calibrate_imu_camera \
    --bag output/kalibr/calib.bag \
    --cam output/kalibr/camchain.yaml \
    --imu path/to/imu.yaml \
    --target path/to/target.yaml
```

**Output**: `camchain-imucam.yaml` with refined camera intrinsics/distortion and camera-IMU extrinsics (rotation + translation between the two sensors), plus a PDF report with reprojection error plots.

## Troubleshooting

### "No clean start/end boundary found" during `--trim_imu_static`

**Causes**:

- The static capture isn't actually static somewhere in the scanned window (someone bumped the table, vibration from a nearby device)
- `--max_scan_s` (default 1800s = 30 min) is shorter than how long the motion actually takes to settle

**Solutions**:

1. Inspect `imu_raw.csv` by hand around the expected settle point
2. If motion genuinely takes longer than 30 min to settle, this isn't exposed as a CLI flag currently — call `auto_trim_static_imu()` directly from Python with a larger `max_scan_s`

### Reprojection error stays high after calibration

**Causes**:

- `camchain.yaml`'s seed intrinsics are far enough off that Kalibr's optimizer doesn't converge well
- Wide-FOV lens forced into `pinhole`+`radtan` when `equidistant` fits better

**Solutions**:

1. Check for `equidistant` comments in `camchain.yaml` (added automatically when FOV > 90°) and re-run with `distortion_model: equidistant` if present
2. Re-run `kalibr_calibrate_camera` (camera-only, no IMU) first to validate intrinsics converge before adding IMU calibration on top

### IMU calibration looks noisy / extrinsics don't stabilize

**Causes**:

- `imu.yaml` noise parameters underestimate real sensor noise (Allan variance run on too-short or not-fully-static data)
- Not enough IMU excitation during the calibration-target capture (phone moved too slowly/smoothly)

**Solutions**:

1. Re-derive `imu.yaml` from a longer static capture — several hours gives a more reliable Allan variance curve than a few minutes
2. Re-capture the calibration-target session with more varied, deliberate rotation on all three axes (not just translation)

## Next Steps

- Feed `camchain-imucam.yaml`'s extrinsics into a downstream SLAM/VIO pipeline that fuses this camera with this IMU
- Re-validate calibration periodically — a phone's IMU and lens mounting can drift after drops or temperature swings
