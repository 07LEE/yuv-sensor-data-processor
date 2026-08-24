# YUV Sensor Data Processor (`yuv_sensor`)

Toolkit for decoding raw YUV_420_888 camera frames, aligning IMU sensor logs, and preparing data for 3D reconstruction pipelines (e.g., COLMAP). Designed for mobile scanning workflows (object/indoor capture).

## 1. Installation

```bash
# Create virtual environment and install package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Quick Start

### Extract frames and synchronized metadata
```bash
yuv-sensor --session_dir data/session_419864820
```

### Export to COLMAP format with IMU-based pose priors
```bash
yuv-sensor --session_dir data/session_419864820 --export_colmap --colmap_output_dir output/colmap
```

Then run COLMAP:
```bash
cd output/colmap
colmap feature_extractor --database_path database.db --image_path images
colmap sequential_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images --output_path sparse
```

## 3. CLI Reference

```bash
yuv-sensor --session_dir <path> [options]

Options:
  --session_dir STR              Path to session directory (required)
  --output_dir STR               Output directory for extracted frames (default: session_dir/extracted_frames_raw)
  --format {jpg,png}             Output image format (default: jpg)
  --undistort                    Apply lens distortion correction
  --max_frames N                 Extract only first N frames
  --process_sync                 Generate synchronized_dataset.json (default: True)
  --export_colmap                Export in COLMAP format with IMU pose priors
  --colmap_output_dir STR        Output directory for COLMAP workspace (default: session_dir/colmap)
```

## 4. Python API Usage

### Basic frame extraction and metadata
```python
from yuv_sensor import SessionDataLoader, DataProcessor

loader = SessionDataLoader("data/session_419864820")

# Decode frame into RGB array with undistortion
rgb_frame = loader.get_decoded_frame(0, apply_undistort=True, apply_rotation=True)

# Query synchronized IMU samples around frame timestamp
imu_data = loader.get_synchronized_imu(timestamp_ns=419865219890714, time_window_ms=50.0)

# Export synchronized frame-to-IMU-to-exposure mapping
processor = DataProcessor(loader)
processor.export_synchronized_dataset()
```

### Export to COLMAP format
```python
from yuv_sensor import SessionDataLoader, ColmapExporter

loader = SessionDataLoader("data/session_419864820")
exporter = ColmapExporter(loader)

result = exporter.export_to_directory(
    output_dir="output/colmap",
    extract_images=True,
    undistort=False,
    image_format="jpg"
)

print(f"Images: {result['images_dir']}")
print(f"Camera config: {result['cameras_txt']}")
print(f"Image list with poses: {result['images_txt']}")
```

### Estimate camera pose from IMU
```python
from yuv_sensor import SessionDataLoader, PoseEstimator

loader = SessionDataLoader("data/session_419864820")
estimator = PoseEstimator()

trajectory = estimator.estimate_trajectory(loader.imu_df, loader.frames_df)

# trajectory[frame_idx] contains: position, rotation_matrix, quaternion, velocity
for idx, pose in trajectory.items():
    print(f"Frame {idx}: pos={pose['position']}, rotation={pose['rotation_matrix']}")
```

## 5. Core Features

- **YUV Decoding**: Decode raw YUV_420_888 frames using layout and stride metadata from frames.csv
- **Camera Calibration**: Apply lens distortion correction and sensor orientation rotation
- **Multi-sensor Synchronization**: Align camera frames, IMU (accel/gyro), and exposure metadata by nanosecond timestamps
- **IMU-based Pose Estimation**: Integrate accelerometer and gyroscope data to estimate initial camera trajectory
- **COLMAP Integration**: Export images and camera parameters in COLMAP-compatible format with pose priors for 3D reconstruction

## 6. Architecture

```
SessionDataLoader
  ├─ frames.csv (YUV frame metadata)
  ├─ imu.csv (accelerometer & gyroscope)
  ├─ capture.csv (exposure & control data)
  └─ session.json (camera calibration & config)

DataProcessor
  └─ Generates synchronized_dataset.json (frame → IMU → exposure mapping)

PoseEstimator
  └─ Estimates camera trajectory from IMU (initial prior for SfM)

ColmapExporter
  ├─ cameras.txt (camera intrinsics)
  ├─ images.txt (images + poses)
  ├─ pose_priors.json (IMU trajectory reference)
  └─ images/ (extracted RGB frames)
```

## 7. Workflow: Mobile Scan → COLMAP 3D Reconstruction

1. **Mobile App**: Captures image sequence + IMU/exposure logs (scans object/room)
2. **SessionDataLoader**: Loads raw YUV frames and metadata
3. **PoseEstimator**: Estimates camera trajectory from IMU (provides rotation hints)
4. **ColmapExporter**: Generates COLMAP workspace with images, intrinsics, and pose priors
5. **COLMAP**: Refines poses via feature matching and bundle adjustment
6. **Output**: Sparse point cloud + camera poses (in sparse/0/model directory)

**Note**: IMU poses are initialization hints only; COLMAP's feature-based SfM provides the authoritative geometry.

## Documentation

- [Data Specification](docs/data_spec.md): Session directory structure and metadata formats
- [COLMAP Workflow](docs/colmap_workflow.md): End-to-end example with COLMAP integration
- [API Reference](docs/api_reference.md): Detailed module and class documentation
