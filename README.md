# YUV Sensor Data Processor (`yuv_sensor`)

Toolkit for decoding raw YUV_420_888 camera frames and aligning IMU sensor logs collected via mobile_sensor_logger.

## 1. Installation

```bash
# Create virtual environment and install package in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. CLI Usage

```bash
# Extract raw frames and generate synchronized_dataset.json
yuv-sensor --session_dir data/<session_directory>

# Extract lens-undistorted frames
yuv-sensor --session_dir data/<session_directory> --undistort
```

## 3. Python API Usage

```python
from yuv_sensor import SessionDataLoader, DataProcessor

# Load capture session
loader = SessionDataLoader("data/session_419864820")

# Decode frame into in-memory RGB array (index 0, with lens undistortion & rotation)
rgb_frame = loader.get_decoded_frame(0, apply_undistort=True)

# Query synchronized IMU samples (+/- 50ms window) for a frame timestamp
imu_data = loader.get_synchronized_imu(timestamp_ns=419865219890714, time_window_ms=50.0)

# Export synchronized dataset JSON
processor = DataProcessor(loader)
processor.export_synchronized_dataset()
```

## 4. Core Features

- YUV Decoding: Decode YUV_420_888 using layout and stride metadata in frames.csv.
- Calibration & Rotation: Rotate frame upright using sensor_orientation in session.json and apply OpenCV lens undistortion.
- Synchronization: Align camera, IMU, and capture metadata using timestamp_ns (nanoseconds).

## Documentation

- [Data Specification](docs/data_spec.md): Detailed session data directory structure, metadata formats, and processing directives.
