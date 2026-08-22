# YUV Sensor Data Processor

This project provides a raw sensor data processing pipeline for uncompressed YUV binary images, IMU logs, and exposure metadata collected via mobile_sensor_logger.

## 1. Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Usage

```bash
# Run session data processing pipeline
python3 src/main.py --session_dir data/<session_directory>
```

## 3. Core Directives

- YUV Decoding: Decode YUV_420_888 using layout and stride metadata in frames.csv.
- Calibration & Rotation: Rotate frame upright using sensor_orientation in session.json and apply OpenCV lens undistortion.
- Synchronization: Align camera, IMU, and capture metadata using timestamp_ns (nanoseconds).

## Documentation

- [Data Specification](docs/data_spec.md): Detailed session data directory structure, metadata formats, and processing directives.
