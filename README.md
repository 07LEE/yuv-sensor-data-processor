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

For the full COLMAP walkthrough (feature extraction, matching, mapping, troubleshooting), see [COLMAP Workflow](docs/colmap_workflow.md).

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
  --export_kalibr                Export Kalibr camera-IMU calibration input (images + camchain.yaml + imu.csv)
  --kalibr_output_dir STR        Output directory for Kalibr export (default: session_dir/kalibr)
  --trim_imu_static {start,end,both}
                                  Trim motion off a static IMU-only capture (for Allan variance / Kalibr imu.yaml)
  --imu_trim_output_dir STR      Output directory for --trim_imu_static (default: session_dir/imu_trimmed)
```

## 4. Core Features

- **YUV Decoding**: Decode raw YUV_420_888 frames using layout and stride metadata from frames.csv
- **Camera Calibration**: Apply lens distortion correction and sensor orientation rotation
- **Multi-sensor Synchronization**: Align camera frames, IMU (accel/gyro), and exposure metadata by nanosecond timestamps
- **IMU-based Pose Estimation**: Integrate accelerometer and gyroscope data to estimate initial camera trajectory
- **COLMAP Integration**: Export images and camera parameters in COLMAP-compatible format with pose priors for 3D reconstruction
- **Kalibr Export**: Export images, camchain.yaml, and imu.csv as input for Kalibr camera-IMU calibration
- **Static IMU Trim**: Auto-trim motion off the edges of a static IMU capture for Allan variance analysis

## 5. Architecture

```text
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

KalibrExporter
  ├─ camchain.yaml (seed intrinsics)
  ├─ imu.csv (copied as-is)
  └─ images/ (extracted RGB frames, not undistorted)

auto_trim_static_imu
  └─ Trims motion off a static IMU capture (imu_trimmed/imu.csv + trim_report.json)
```

## Documentation

- [Data Specification](docs/data_spec.md): Session directory structure and metadata formats
- [COLMAP Workflow](docs/colmap_workflow.md): End-to-end example with COLMAP integration
- [Kalibr Workflow](docs/kalibr_workflow.md): End-to-end example for Kalibr camera-IMU calibration
- [API Reference](docs/api_reference.md): Detailed module and class documentation
