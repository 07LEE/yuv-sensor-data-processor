# COLMAP Workflow Guide

End-to-end guide for preparing mobile scan data for 3D reconstruction using COLMAP, with IMU-estimated poses exported alongside as reference data.

## Overview

The workflow exports mobile sensor data (images + IMU) in COLMAP format and runs COLMAP's structure-from-motion pipeline on the images:

```text
Mobile Scan Data
    ↓
YUV Frames + IMU Logs (SessionDataLoader)
    ↓
Estimate IMU Trajectory (PoseEstimator) -> camera poses (only with T_cam_imu)
    ↓
Export COLMAP Format (ColmapExporter)
    ├─ images/ (RGB frames)            ← read by COLMAP
    ├─ cameras.txt (intrinsics)        ← reference, not read by the steps below
    ├─ images.txt (camera poses, only with T_cam_imu) ← reference, not read by the steps below
    └─ pose_priors.json (IMU trajectory) ← reference, not read by the steps below
    ↓
COLMAP: Feature Extraction & Matching (images/ only)
    ↓
COLMAP: Incremental Mapper (estimates poses from the images alone)
    ↓
Output: Sparse Point Cloud + Camera Poses
```

### What the exported pose files are for

The commands in Steps 2-4 (`feature_extractor`, `sequential_matcher`, `mapper`) read only `images/`. They do not read `cameras.txt`, `images.txt`, or `pose_priors.json`, so the IMU-estimated poses have no effect on the reconstruction. The mapper starts from no pose information and estimates every pose from feature matches.

The poses are dead-reckoned by integrating gyro and accelerometer data, so position error grows quickly with time. Treat them as reference data, for example to sanity-check the scan trajectory or to compare against the poses COLMAP recovers, not as ground truth.

## Prerequisites

- yuv_sensor installed: `pip install -e .`
- COLMAP installed: See [COLMAP documentation](https://colmap.github.io/)

  ```bash
  # macOS
  brew install colmap
  
  # Ubuntu/Debian
  sudo apt-get install colmap
  
  # Docker (alternative)
  docker pull colmap/colmap
  ```

## Step 1: Export Session Data to COLMAP Format

### Using CLI

```bash
yuv-sensor --session_dir data/session_419864820 \
           --export_colmap \
           --colmap_output_dir output/colmap \
           --format jpg \
           --undistort
```

This generates:

- `output/colmap/images/` — RGB frames (named by timestamp)
- `output/colmap/cameras.txt` — Camera intrinsics (PINHOLE model) matching the exported upright images, reference only
- `output/colmap/images.txt` — Image list with camera poses derived from the IMU trajectory, reference only; written only with `--camera_imu_extrinsics` (see below)
- `output/colmap/pose_priors.json` — IMU trajectory (IMU body frame), reference only; with `--camera_imu_extrinsics` each frame also gets `camera_position` and `camera_quaternion_xyzw` (camera-to-world), and `camera_poses_included` is `true`
- `output/colmap/colmap_export_metadata.json` — Export metadata
- `output/colmap/frame_quality_report.json` — Per-frame sharpness and exposure/white-balance convergence state, always written (see below)

### Camera-IMU extrinsics and `images.txt`

The integrated trajectory is the pose of the IMU, not of the camera. Writing it as a camera pose is only correct if the two frames coincide, so `images.txt` is written only when you pass the camera-IMU extrinsic:

```bash
yuv-sensor --session_dir data/session_419864820 \
           --export_colmap \
           --camera_imu_extrinsics extrinsics.json
```

`extrinsics.json` holds a 4x4 `T_cam_imu` matrix, with `x_cam = T_cam_imu @ x_imu`:

```json
{"T_cam_imu": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
```

The extrinsic is expected in the raw sensor frame by default; the exporter rotates it for `sensor_orientation` so the poses match the upright `images/`. Without `--camera_imu_extrinsics`, `images.txt` is not written (and one left by an earlier run is removed), and `pose_priors.json` carries the IMU body pose, labelled `"pose_frame": "imu_body"`.

#### Using a Kalibr result

`T_cam_imu` from this repo's [Kalibr Workflow](kalibr_workflow.md) is in the upright frame (`cam0/` holds upright images), not the raw sensor frame. Pass `--camera_imu_extrinsics_frame upright` (`t_cam_imu_frame="upright"` in Python). Without it the matrix is read as raw and rotated a second time unless `sensor_orientation` is 0.

`--camera_imu_extrinsics` reads JSON, not the YAML Kalibr writes, so extract `cam0.T_cam_imu` first (PyYAML is not a yuv_sensor dependency: `pip install pyyaml`):

```python
import json
import yaml

with open("camchain-imucam.yaml") as f:
    t_cam_imu = yaml.safe_load(f)["cam0"]["T_cam_imu"]
with open("extrinsics.json", "w") as f:
    json.dump({"T_cam_imu": t_cam_imu}, f)
```

```bash
yuv-sensor --session_dir data/session_419864820 \
           --export_colmap \
           --camera_imu_extrinsics extrinsics.json \
           --camera_imu_extrinsics_frame upright
```

### Optional: filter out blurry / not-yet-converged frames first

`frame_quality_report.json` above is generated either way, so you can inspect it before deciding whether to filter anything. If you do want to exclude frames, add:

```bash
yuv-sensor --session_dir data/session_419864820 \
           --export_colmap \
           --colmap_output_dir output/colmap \
           --min_sharpness 0.3 \
           --require_converged
```

`--min_sharpness` drops frames below that frames.csv `sharpness` value; `--require_converged` also drops frames whose nearest capture.csv row wasn't at `ae_state`/`awb_state` CONVERGED/LOCKED yet. Both are opt-in (default: exclude nothing) and apply consistently across `images/`, `images.txt`, and `pose_priors.json`. A filter that excludes 50%+ of frames prints a warning — COLMAP needs enough overlapping views to reconstruct, so check `frame_quality_report.json`'s per-frame entries before committing to an aggressive threshold. See [API Reference](api_reference.md#frame-quality-filtering).

### Using Python API

```python
from yuv_sensor import SessionDataLoader, ColmapExporter
from yuv_sensor.io_utils import load_t_cam_imu

loader = SessionDataLoader("data/session_419864820")
exporter = ColmapExporter(loader)

# raw sensor frame T_cam_imu; without it images.txt is not written
t_cam_imu = load_t_cam_imu("extrinsics.json")

result = exporter.export_to_directory(
    output_dir="output/colmap",
    extract_images=True,
    undistort=True,
    image_format="jpg",
    image_quality=92,
    t_cam_imu=t_cam_imu,
)

print(f"Exported to: {result['images_dir']}")
print(f"Next: cd {result['images_dir'].parent} && colmap feature_extractor ...")
```

## Step 2: Run COLMAP Feature Extraction

Extract visual features (SIFT by default) from all images:

```bash
cd output/colmap

colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1
```

Parameters:

- `--database_path database.db` — COLMAP database file
- `--image_path images` — Directory with extracted RGB images
- `--ImageReader.camera_model PINHOLE` — Match our camera model
- `--ImageReader.single_camera 1` — All images use same camera model (intrinsics)

Output: `database.db` populated with image metadata and keypoints.

## Step 3: Image Matching

Match visual features between nearby images:

```bash
colmap sequential_matcher \
    --database_path database.db \
    --SequentialMatching.overlap 5 \
    --SequentialMatching.quadratic_overlap 1
```

Why sequential matching: since images follow a camera trajectory (sequential scan), sequential matching is faster than exhaustive matching.

Alternative for small datasets (< 100 images, dense sampling):

```bash
colmap exhaustive_matcher --database_path database.db
```

Output: Feature matches stored in `database.db`.

## Step 4: Incremental Mapper (Main SfM)

Run the incremental mapper, which:

1. Initializes with two best-matched images
2. Incrementally adds more images
3. Refines camera poses via bundle adjustment
4. Triangulates points into 3D point cloud

```bash
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse \
    --Mapper.ignore_watermark 1
```

Parameters:

- `--output_path sparse` — Directory for output models
- `--Mapper.ignore_watermark 1` — Skip watermark check

Expected output:

- `sparse/0/` — Model directory with:
  - `images.bin` — Refined camera poses (world-to-camera, quaternion in QW QX QY QZ order)
  - `points3D.bin` — 3D point cloud
  - `cameras.bin` — Refined intrinsics

The mapper writes binary models. Convert them to text to read them, see [Verifying Output](#verifying-output).

Runtime: Depends on image count. ~10-30s for 100-500 images on modern hardware.

## Step 5: (Optional) Bundle Adjustment Refinement

For tighter geometry, run bundle adjustment:

```bash
colmap bundle_adjuster \
    --input_path sparse/0 \
    --output_path sparse/0
```

## Step 6: Convert to Dense Reconstruction (Optional)

For dense point clouds, use stereo reconstruction:

```bash
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense

colmap stereo_fusion \
    --workspace_path dense \
    --output_path dense/fused.ply
```

This produces:

- `dense/fused.ply` — Dense point cloud (millions of points)

## Troubleshooting

### "Not enough matches" / Mapper fails to initialize

Causes:

- Images too different (large baseline, rotation)
- Insufficient visual overlap
- Low image quality or motion blur

Solutions:

1. Check pose_priors.json for a sanity check of the scan trajectory (COLMAP does not use it, so it cannot cause this failure)
2. Verify images are in focus and well-lit — check `frame_quality_report.json` (always generated in Step 1) for blurry or not-yet-converged frames, and re-export with `--min_sharpness`/`--require_converged` if it finds a lot of them
3. Try exhaustive matching instead of sequential
4. Reduce `--SequentialMatching.overlap` to 3 or 1

### "Intrinsics refinement failed"

Cause: Camera intrinsics in cameras.txt might be slightly off.

Solution:

1. Double-check session.json intrinsics
2. Verify image resolution matches (width/height in cameras.txt)
3. Let COLMAP estimate intrinsics:

   ```bash
   colmap bundle_adjuster --refine_principal_point 1 --input_path sparse/0
   ```

### Output looks distorted or wrong rotation

Check:

1. Session orientation: Verify `sensor_orientation` in session.json matches device mounting
2. Image undistortion: Try `--undistort` flag in yuv-sensor export if not already applied
3. Pose priors: Inspect `pose_priors.json` — poses should show smooth trajectory

### Out of memory

If dataset is large (> 1000 images):

1. Reduce image resolution (save as smaller JPEGs)
2. Use `--Mapper.abs_pose_max_error 10` to be less strict on outliers
3. Process in batches (split session into sub-scans)

## Verifying Output

### Check reconstruction quality

The mapper writes binary models, so convert them to text first:

```bash
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0_txt \
    --output_type TXT
```

```python
import numpy as np
from scipy.spatial.transform import Rotation

# Read camera poses. TX, TY, TZ are the world-to-camera translation t,
# so the camera center is C = -R^T t. Every second line is POINTS2D.
with open("sparse/0_txt/images.txt") as f:
    rows = [line.split() for line in f if not line.startswith("#")]
for parts in rows[::2]:
    image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
    r_cw = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
    c = -r_cw.T @ np.array([float(tx), float(ty), float(tz)])
    print(f"{name}: pos=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

# Read point cloud stats
with open("sparse/0_txt/points3D.txt") as f:
    points = [l for l in f if not l.startswith("#")]
    print(f"Total 3D points: {len(points)}")
```

### Visualize in COLMAP GUI

```bash
colmap gui
# File → Open Model → sparse/0/
```

## Tips for Best Results

1. Overlap: Aim for 50-80% image overlap (not every other frame)
2. Motion: Smooth, continuous motion (avoid sudden jumps/rotation)
3. Lighting: Consistent lighting; avoid moving shadows
4. Camera Settings: Fixed focus, exposure; auto-focus can hurt matching
5. Resolution: Higher resolution = more features but slower processing
6. Unique geometry: Avoid featureless surfaces (blank walls, reflective floors)

## Performance Benchmarks

| Dataset Size | Processing Time | Output Size |
| -------------- | ----------------- | ------------- |
| 50 images (object) | 5-10s | 10K-50K points |
| 200 images (room) | 30-60s | 100K-500K points |
| 1000 images (indoor scene) | 3-10 min | 1M+ points |

Times on RTX 3070, M1 Pro times are similar.

## Next Steps

- Mesh reconstruction: Use dense point clouds with Poisson reconstruction
- Camera alignment: Use refined poses for downstream AR/robotics applications
- Streaming/SfM: Use COLMAP's MVS (multi-view stereo) for denser geometry
