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

For all command-line flags and defaults, see [CLI Reference](docs/cli_reference.md).

## 3. Core Features

- **YUV Decoding**: Decode raw YUV_420_888 frames using layout and stride metadata from frames.csv
- **Camera Calibration**: Apply lens distortion correction and sensor orientation rotation
- **Multi-sensor Synchronization**: Align camera frames, IMU (accel/gyro), and exposure metadata by nanosecond timestamps
- **IMU-based Pose Estimation**: Integrate accelerometer and gyroscope data to estimate initial camera trajectory
- **COLMAP Integration**: Export images and camera parameters in COLMAP-compatible format with pose priors for 3D reconstruction
- **Kalibr Export**: Export images, camchain.yaml, and imu.csv as input for Kalibr camera-IMU calibration
- **Static IMU Trim + imu.yaml**: Auto-trim motion off the edges of a static IMU capture and derive Kalibr's imu.yaml (noise_density / random_walk) from it via Allan variance
- **Checkerboard Camera Calibration**: Calibrate camera intrinsics directly from a checkerboard capture using OpenCV — no Kalibr/rosbag needed, a quick pre-check against session.json's own intrinsics

## Documentation

- [Data Specification](docs/data_spec.md): Session directory structure and metadata formats
- [CLI Reference](docs/cli_reference.md): All command-line flags and defaults
- [COLMAP Workflow](docs/colmap_workflow.md): End-to-end example with COLMAP integration
- [Kalibr Workflow](docs/kalibr_workflow.md): End-to-end example for Kalibr camera-IMU calibration
- [API Reference](docs/api_reference.md): Detailed module and class documentation

## License

Apache License 2.0 — see [LICENSE](LICENSE).
