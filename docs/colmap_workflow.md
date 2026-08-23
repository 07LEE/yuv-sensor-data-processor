# COLMAP Workflow Guide

End-to-end guide for preparing mobile scan data for 3D reconstruction using COLMAP with IMU-based pose priors.

## Overview

The workflow combines mobile sensor data (images + IMU) with COLMAP's structure-from-motion pipeline:

```
Mobile Scan Data
    ↓
YUV Frames + IMU Logs (SessionDataLoader)
    ↓
Estimate Camera Trajectory (PoseEstimator)
    ↓
Export COLMAP Format (ColmapExporter)
    ├─ images/ (RGB frames)
    ├─ cameras.txt (intrinsics)
    ├─ images.txt (initial poses)
    └─ pose_priors.json (IMU trajectory reference)
    ↓
COLMAP: Feature Extraction & Matching
    ↓
COLMAP: Incremental Mapper (refines poses via SfM)
    ↓
Output: Sparse Point Cloud + Refined Camera Poses
```

## Prerequisites

- **yuv_sensor** installed: `pip install -e .`
- **COLMAP** installed: See [COLMAP documentation](https://colmap.github.io/)

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
- `output/colmap/cameras.txt` — Camera intrinsics (PINHOLE model)
- `output/colmap/images.txt` — Image list with IMU-estimated poses
- `output/colmap/pose_priors.json` — IMU trajectory (reference only)
- `output/colmap/colmap_export_metadata.json` — Export metadata

### Using Python API

```python
from yuv_sensor import SessionDataLoader, ColmapExporter

loader = SessionDataLoader("data/session_419864820")
exporter = ColmapExporter(loader)

result = exporter.export_to_directory(
    output_dir="output/colmap",
    extract_images=True,
    undistort=True,
    image_format="jpg",
    image_quality=92
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

**Parameters**:

- `--database_path database.db` — COLMAP database file
- `--image_path images` — Directory with extracted RGB images
- `--ImageReader.camera_model PINHOLE` — Match our camera model
- `--ImageReader.single_camera 1` — All images use same camera model (intrinsics)

**Output**: `database.db` populated with image metadata and keypoints.

## Step 3: Image Matching

Match visual features between nearby images:

```bash
colmap sequential_matcher \
    --database_path database.db \
    --SequentialMatching.overlap 5 \
    --SequentialMatching.quadratic_overlap 1
```

**Why sequential matching?**: Since images follow a camera trajectory (sequential scan), sequential matching is faster than exhaustive matching.

**Alternative for small datasets** (< 100 images, dense sampling):

```bash
colmap exhaustive_matcher --database_path database.db
```

**Output**: Feature matches stored in `database.db`.

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

**Parameters**:

- `--output_path sparse` — Directory for output models
- `--Mapper.ignore_watermark 1` — Skip watermark check

**Expected output**:

- `sparse/0/` — Model directory with:
  - `images.txt` — Refined camera poses (xyzw quaternion format)
  - `points3D.txt` — 3D point cloud
  - `cameras.txt` — Refined intrinsics

**Runtime**: Depends on image count. ~10-30s for 100-500 images on modern hardware.

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

**Causes**:

- Images too different (large baseline, rotation)
- Insufficient visual overlap
- Low image quality or motion blur

**Solutions**:

1. Check pose_priors.json — do poses look reasonable?
2. Verify images are in focus and well-lit
3. Try exhaustive matching instead of sequential
4. Reduce `--SequentialMatching.overlap` to 3 or 1

### "Intrinsics refinement failed"

**Cause**: Camera intrinsics in cameras.txt might be slightly off.

**Solution**:

1. Double-check session.json intrinsics
2. Verify image resolution matches (width/height in cameras.txt)
3. Let COLMAP estimate intrinsics:

   ```bash
   colmap bundle_adjuster --refine_principal_point 1 --input_path sparse/0
   ```

### Output looks distorted or wrong rotation

**Check**:

1. Session orientation: Verify `sensor_orientation` in session.json matches device mounting
2. Image undistortion: Try `--undistort` flag in yuv-sensor export if not already applied
3. Pose priors: Inspect `pose_priors.json` — poses should show smooth trajectory

### Out of memory

**If dataset is large** (> 1000 images):

1. Reduce image resolution (save as smaller JPEGs)
2. Use `--Mapper.abs_pose_max_error 10` to be less strict on outliers
3. Process in batches (split session into sub-scans)

## Verifying Output

### Check reconstruction quality

```python
import json

# Read camera poses
with open("sparse/0/images.txt") as f:
    for line in f:
        if line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 10:
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
            print(f"{name}: pos=({tx}, {ty}, {tz})")

# Read point cloud stats
with open("sparse/0/points3D.txt") as f:
    points = [l for l in f if not l.startswith("#")]
    print(f"Total 3D points: {len(points)}")
```

### Visualize in COLMAP GUI

```bash
colmap gui
# File → Open Model → sparse/0/
```

## Tips for Best Results

1. **Overlap**: Aim for 50-80% image overlap (not every other frame)
2. **Motion**: Smooth, continuous motion (avoid sudden jumps/rotation)
3. **Lighting**: Consistent lighting; avoid moving shadows
4. **Camera Settings**: Fixed focus, exposure; auto-focus can hurt matching
5. **Resolution**: Higher resolution = more features but slower processing
6. **Unique geometry**: Avoid featureless surfaces (blank walls, reflective floors)

## Performance Benchmarks

| Dataset Size | Processing Time | Output Size |
| -------------- | ----------------- | ------------- |
| 50 images (object) | 5-10s | 10K-50K points |
| 200 images (room) | 30-60s | 100K-500K points |
| 1000 images (indoor scene) | 3-10 min | 1M+ points |

Times on RTX 3070, M1 Pro times are similar.

## Next Steps

- **Mesh reconstruction**: Use dense point clouds with Poisson reconstruction
- **Camera alignment**: Use refined poses for downstream AR/robotics applications
- **Streaming/SfM**: Use COLMAP's MVS (multi-view stereo) for denser geometry
