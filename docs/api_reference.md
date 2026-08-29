# API Reference

Complete reference for the `yuv_sensor` Python package.

## Table of Contents

1. [SessionDataLoader](#sessiondataloader)
2. [DataProcessor](#dataprocessor)
3. [PoseEstimator](#poseestimator)
4. [ColmapExporter](#colmapexporter)
5. [KalibrExporter](#kalibrexporter)
6. [auto_trim_static_imu](#auto_trim_static_imu)
7. [compute_imu_noise_params](#compute_imu_noise_params)
8. [calibrate_camera_from_checkerboard](#calibrate_camera_from_checkerboard)
9. [Utility Functions](#utility-functions)

---

## SessionDataLoader

Loads and provides access to a session's frame, IMU, and calibration data.

### Constructor

```python
SessionDataLoader(session_dir: str)
```

**Args**:

- `session_dir` (str): Path to session directory containing frames.csv, imu.csv, capture.csv, and session.json

**Raises**:

- `FileNotFoundError`: If session_dir does not exist or session.json missing

**Example**:

```python
from yuv_sensor import SessionDataLoader

loader = SessionDataLoader("data/session_419864820")
print(f"Frames: {loader.get_frame_count()}")
print(f"Device: {loader.session_config['device']}")
```

### Properties

| Property | Type | Description |
| ---------- | ------ | ------------- |
| `session_path` | `Path` | Session directory path |
| `session_config` | `dict` | Parsed session.json (device, intrinsics, distortion, sensor_orientation) |
| `frames_df` | `DataFrame` | Loaded frames.csv (columns: filename, width, height, timestamp_ns, ...) |
| `capture_df` | `DataFrame` | Loaded capture.csv or None (columns: exposure_ns, sensitivity, focus_diopters, ...) |
| `imu_df` | `DataFrame` | Loaded imu.csv or None (columns: timestamp_ns, sensor, x, y, z) |
| `camera_calib` | `CameraCalibration` | Camera calibration instance for undistortion and rotation |

### Methods

#### `get_frame_count() -> int`

Returns total number of frames in the session.

```python
total = loader.get_frame_count()
print(f"Session has {total} frames")
```

#### `load_raw_frame(index: int) -> Tuple[bytes, pd.Series]`

Loads raw YUV bytes and metadata for a frame.

**Args**:

- `index` (int): Frame row index (0-based)

**Returns**:

- `Tuple[bytes, pd.Series]`: Raw YUV bytes and frame metadata row

**Raises**:

- `FileNotFoundError`: If raw YUV file not found

**Example**:

```python
yuv_bytes, metadata = loader.load_raw_frame(0)
width, height = int(metadata["width"]), int(metadata["height"])
print(f"Frame size: {width}x{height}, YUV bytes: {len(yuv_bytes)}")
```

#### `get_decoded_frame(index: int, apply_undistort: bool = False, apply_rotation: bool = True) -> np.ndarray`

Loads, decodes, and optionally processes a frame into RGB.

**Args**:

- `index` (int): Frame row index
- `apply_undistort` (bool): Apply lens distortion correction via OpenCV
- `apply_rotation` (bool): Rotate image upright using sensor_orientation

**Returns**:

- `np.ndarray`: RGB image as numpy array (shape: height×width×3, dtype: uint8)

**Example**:

```python
# Get undistorted, upright RGB frame
rgb = loader.get_decoded_frame(42, apply_undistort=True, apply_rotation=True)
print(rgb.shape)  # (1080, 1440, 3)

# Display with PIL
from PIL import Image
img = Image.fromarray(rgb)
img.show()
```

#### `get_synchronized_imu(timestamp_ns: int, time_window_ms: float = 100.0) -> Dict[str, pd.DataFrame]`

Retrieves IMU samples within a time window around a frame timestamp.

**Args**:

- `timestamp_ns` (int): Frame timestamp in nanoseconds
- `time_window_ms` (float): Half-width of time window in milliseconds (default: 100ms)

**Returns**:

- `Dict` with keys "accel" and "gyro", each containing filtered DataFrame with columns: timestamp_ns, x, y, z
- Returns empty DataFrames if imu.csv not found

**Example**:

```python
# Get IMU samples 50ms before and after frame timestamp
imu = loader.get_synchronized_imu(frame_ts, time_window_ms=50.0)
print(f"Accel samples: {len(imu['accel'])}")
print(f"Gyro samples: {len(imu['gyro'])}")

# Access data
accel_mean = imu['accel'][['x', 'y', 'z']].mean()
print(f"Mean acceleration: {accel_mean.values}")
```

#### `get_nearest_capture_metadata(timestamp_ns: int) -> Optional[pd.Series]`

Finds nearest exposure/control metadata for a frame timestamp.

**Args**:

- `timestamp_ns` (int): Frame timestamp

**Returns**:

- `pd.Series`: Closest capture metadata row, or None if not available
- Columns: exposure_ns, sensitivity, focus_diopters, rolling_shutter_skew_ns, ...

**Example**:

```python
capture = loader.get_nearest_capture_metadata(frame_ts)
if capture is not None:
    print(f"Exposure: {capture['exposure_ns']} ns")
    print(f"ISO: {capture['sensitivity']}")
```

---

## DataProcessor

Generates synchronized datasets linking frames to IMU and exposure metadata.

### Constructor

```python
DataProcessor(loader: SessionDataLoader)
```

**Args**:

- `loader`: Loaded SessionDataLoader instance

**Example**:

```python
from yuv_sensor import SessionDataLoader, DataProcessor

loader = SessionDataLoader("data/session_419864820")
processor = DataProcessor(loader)
```

### Methods

#### `generate_synchronized_dataset(time_window_ms: float = 50.0) -> List[Dict]`

Generates synchronized data structure for all frames.

**Args**:

- `time_window_ms` (float): Time window half-width for IMU samples (default: 50ms)

**Returns**:

- `List[Dict]`: One entry per frame with structure:

  ```python
  {
    "frame_index": 0,
    "timestamp_ns": 419865219890714,
    "filename": "...",
    "width": 1440, "height": 1080,
    "sharpness": 0.82,
    "exposure": {
      "exposure_ns": 16000000,  # 16ms
      "sensitivity": 100,
      "focus_diopters": 0.5,
      "rolling_shutter_skew_ns": 33000000
    },
    "imu_window": {
      "window_ms": 50.0,
      "accel_samples_count": 3,
      "gyro_samples_count": 3,
      "accel": [{"timestamp_ns": ..., "x": 0.1, "y": 9.8, "z": -0.2}, ...],
      "gyro": [{"timestamp_ns": ..., "x": -0.01, "y": 0.02, "z": 0.0}, ...]
    },
    "interframe_imu": {
      "next_timestamp_ns": 419865219950000,
      "dt_ns": 59286,
      "accel": [...],
      "gyro": [...]
    }
  }
  ```

**Example**:

```python
dataset = processor.generate_synchronized_dataset(time_window_ms=50.0)
print(f"Generated {len(dataset)} synchronized frames")

# Access frame 0
frame0 = dataset[0]
print(f"Frame {frame0['frame_index']}: {len(frame0['imu_window']['accel'])} accel samples")
```

#### `export_synchronized_dataset(output_path: Optional[str] = None, time_window_ms: float = 50.0) -> Path`

Generates and exports synchronized dataset to JSON.

**Args**:

- `output_path` (str, optional): Output JSON path (default: session_dir/synchronized_dataset.json)
- `time_window_ms` (float): Time window for IMU (default: 50ms)

**Returns**:

- `Path`: Path to exported JSON file

**Example**:

```python
sync_file = processor.export_synchronized_dataset()
print(f"Exported to: {sync_file}")

# Read back as JSON
import json
with open(sync_file) as f:
    data = json.load(f)
    print(f"Session ID: {data['session_id']}")
    print(f"Total frames: {data['total_frames']}")
```

---

## PoseEstimator

Estimates camera pose trajectory from IMU accelerometer and gyroscope data.

### Constructor

```python
PoseEstimator(gravity_magnitude: float = 9.81)
```

**Args**:

- `gravity_magnitude` (float): Magnitude of gravity (m/s²), default 9.81

### Methods

#### `estimate_trajectory(imu_df: Optional[pd.DataFrame], frames_df: pd.DataFrame, initial_rotation: Optional[np.ndarray] = None) -> Dict[int, Dict]`

Estimates camera trajectory by integrating IMU data.

**Args**:

- `imu_df` (DataFrame, optional): IMU data with columns: timestamp_ns, sensor, x, y, z
- `frames_df` (DataFrame): Frame metadata with column: timestamp_ns
- `initial_rotation` (np.ndarray, optional): Initial 3×3 rotation matrix (default: identity)

**Returns**:

- `Dict[int, Dict]`: Maps frame_index to pose dict:

  ```python
  {
    "timestamp_ns": 419865219890714,
    "position": np.array([0.1, 0.2, -0.5]),  # [x, y, z] in meters
    "rotation_matrix": np.array([[...], [...]]),  # 3×3 rotation matrix
    "quaternion": np.array([0.0, 0.1, 0.05, 0.99]),  # [qx, qy, qz, qw]
    "velocity": np.array([0.05, 0.02, 0.0])  # [vx, vy, vz] m/s
  }
  ```

**Algorithm**:

1. Gyroscope → incremental rotation (small angle approximation)
2. Accelerometer → subtract gravity → acceleration in world frame
3. Integrate acceleration → velocity → position
4. Returns identity trajectory if IMU unavailable

**Example**:

```python
from yuv_sensor import SessionDataLoader, PoseEstimator

loader = SessionDataLoader("data/session_419864820")
estimator = PoseEstimator()

trajectory = estimator.estimate_trajectory(loader.imu_df, loader.frames_df)

# Print poses
for frame_idx, pose in trajectory.items():
    if frame_idx % 10 == 0:  # Print every 10th frame
        print(f"Frame {frame_idx}: pos={pose['position']}, quat={pose['quaternion']}")
```

**Note**: These are approximate trajectories useful as initialization hints. COLMAP's feature-based SfM will refine them significantly.

#### `get_frame_poses(trajectory: Dict[int, Dict], frames_df: pd.DataFrame) -> List[Tuple]`

Converts trajectory to list format for easy export.

**Args**:

- `trajectory`: Output from estimate_trajectory()
- `frames_df`: Frame metadata

**Returns**:

- `List[(frame_idx, position, quaternion)]`: Each entry is (int, np.array[3], np.array[4])

**Example**:

```python
poses = estimator.get_frame_poses(trajectory, loader.frames_df)
for idx, pos, quat in poses[:5]:
    print(f"Frame {idx}: {pos}, {quat}")
```

---

## ColmapExporter

Exports session data in COLMAP-compatible format with IMU-based pose priors.

### Constructor

```python
ColmapExporter(loader: SessionDataLoader)
```

**Args**:

- `loader`: Loaded SessionDataLoader instance

**Example**:

```python
from yuv_sensor import SessionDataLoader, ColmapExporter

loader = SessionDataLoader("data/session_419864820")
exporter = ColmapExporter(loader)
```

### Methods

#### `export_to_directory(output_dir: Path, extract_images: bool = True, undistort: bool = False, image_format: str = "jpg", image_quality: int = 92) -> Dict[str, Path]`

Exports all data to COLMAP workspace.

**Args**:

- `output_dir` (Path or str): Output directory
- `extract_images` (bool): Extract YUV → RGB images to output_dir/images/ (default: True)
- `undistort` (bool): Apply lens distortion correction (default: False)
- `image_format` (str): "jpg" or "png" (default: "jpg")
- `image_quality` (int): JPEG quality 1-100 (default: 92)

**Returns**:

- `Dict[str, Path]`:

  ```python
  {
    "cameras_txt": Path("output/cameras.txt"),
    "images_txt": Path("output/images.txt"),
    "pose_priors_json": Path("output/pose_priors.json"),
    "metadata_json": Path("output/colmap_export_metadata.json"),
    "images_dir": Path("output/images")
  }
  ```

**Generated Files**:

- `cameras.txt`: Camera intrinsics (PINHOLE model)
- `images.txt`: COLMAP format (IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME)
- `pose_priors.json`: IMU trajectory (reference)
- `colmap_export_metadata.json`: Export metadata and COLMAP commands
- `images/`: Extracted RGB frames

**Example**:

```python
loader = SessionDataLoader("data/session_419864820")
exporter = ColmapExporter(loader)

result = exporter.export_to_directory(
    output_dir="output/colmap",
    extract_images=True,
    undistort=True,
    image_format="jpg",
    image_quality=95
)

print(f"Images exported to: {result['images_dir']}")
print(f"Ready for COLMAP: cd {result['images_dir'].parent}")
```

---

## KalibrExporter

Exports session data as input for Kalibr's camera-IMU calibration. Unlike `ColmapExporter`, images are extracted without undistortion — Kalibr fits its own distortion model from the raw frames, and an already-undistorted image would get run through that fit a second time.

### Constructor

```python
KalibrExporter(loader: SessionDataLoader)
```

**Args**:

- `loader`: Loaded SessionDataLoader instance

**Example**:

```python
from yuv_sensor import SessionDataLoader, KalibrExporter

loader = SessionDataLoader("data/session_419864820")
exporter = KalibrExporter(loader)
```

### Methods

#### `export_to_directory(output_dir: Path, extract_images: bool = True, image_format: str = "png", max_frames: Optional[int] = None) -> Dict[str, Optional[Path]]`

Exports images, camchain.yaml, and imu.csv to a Kalibr input directory.

**Args**:

- `output_dir` (Path or str): Target directory for the Kalibr input set
- `extract_images` (bool): Extract YUV → RGB images to output_dir/images/ (default: True)
- `image_format` (str): "jpg" or "png" (default: "png" — avoids re-compressing frames Kalibr will run corner detection on)
- `max_frames` (int, optional): Cap on frames extracted, or None for all of them

**Returns**:

- `Dict[str, Optional[Path]]`: Value is `None` for a part that was skipped (e.g. no imu.csv on this session)

  ```python
  {
    "camchain_yaml": Path("output/camchain.yaml"),
    "imu_csv": Path("output/imu.csv"),
    "images_dir": Path("output/images"),
    "metadata_json": Path("output/kalibr_export_metadata.json")
  }
  ```

**Generated Files**:

- `camchain.yaml`: Seed intrinsics reformatted from session.json's own calibration (not re-derived from checkerboard frames). `session.json`'s 5-coefficient distortion and any lens skew don't fit Kalibr's 4-param `radtan` model — both are dropped, flagged in a comment in the written file rather than silently discarded. Wide-FOV sessions (>90°) also get a comment suggesting Kalibr's `equidistant` (fisheye) model as a comparison.
- `imu.csv`: Copied as-is from the session
- `kalibr_export_metadata.json`: Export metadata
- `images/`: Extracted RGB frames, not undistorted

**Example**:

```python
loader = SessionDataLoader("data/session_419864820")
exporter = KalibrExporter(loader)

result = exporter.export_to_directory(
    output_dir="output/kalibr",
    extract_images=True,
    image_format="png"
)

print(f"camchain.yaml: {result['camchain_yaml']}")
print(f"imu.csv: {result['imu_csv']}")
```

**Note**: Still needed before `kalibr_calibrate_imu_camera` can run: a `target.yaml` for the physical calibration board, an `imu.yaml` (see [`compute_imu_noise_params`](#compute_imu_noise_params) below), and packing `images/` + `imu.csv` into a rosbag. See [Kalibr Workflow](kalibr_workflow.md) for the full procedure.

---

## auto_trim_static_imu

Trims motion-contaminated edges off a static IMU capture. Deriving Kalibr's `imu.yaml` (noise_density / random_walk) via Allan variance requires input that is genuinely motionless — handling the phone to start or stop the recording leaves real motion at either edge that corrupts that analysis if left in.

Finds the cut point by scanning accelerometer magnitude deviation from the session's own median (a proxy for gravity) in 1-second bins, walking in from each requested end until 30 consecutive seconds all stay under threshold, then adding a safety margin past that point.

```python
auto_trim_static_imu(imu_df: pd.DataFrame, ends: str = "both", threshold: float = 0.07, settle_run_s: float = 30, margin_s: float = 15, max_scan_s: float = 1800) -> Dict
```

**Args**:

- `imu_df` (DataFrame): IMU data with columns timestamp_ns, sensor, x, y, z (accel + gyro interleaved, as loaded by SessionDataLoader)
- `ends` (str): Which end(s) to scan and trim — "start", "end", or "both" (default: "both")
- `threshold` (float): m/s² accel deviation from baseline treated as motion (default: 0.07)
- `settle_run_s` (float): Consecutive clean seconds required to trust a boundary (default: 30)
- `margin_s` (float): Extra cushion added past the confirmed-clean point (default: 15)
- `max_scan_s` (float): Give up looking for a clean boundary past this many seconds from the requested end (default: 1800)

**Returns**:

- `Dict` with `"trimmed"` (the cut DataFrame) and `"report"` (baseline magnitude, cut points found, kept/dropped row counts)

**Raises**:

- `ValueError`: No accel rows in imu_df, or no clean boundary found within max_scan_s on a requested end

**Example**:

```python
from yuv_sensor import SessionDataLoader, auto_trim_static_imu

loader = SessionDataLoader("data/session_static_capture")
result = auto_trim_static_imu(loader.imu_df, ends="both")

report = result["report"]
print(f"baseline |accel|: {report['baseline_mag']:.4f} m/s^2")
print(f"kept {report['kept_rows']} rows, dropped {report['dropped_rows']}")

result["trimmed"].to_csv("imu_trimmed.csv", index=False)
```

**Note**: This is exposed via the CLI as `--trim_imu_static {start,end,both}`, which also writes `imu_raw.csv` (the untrimmed copy), `trim_report.json`, and — via `compute_imu_noise_params`/`export_imu_yaml` below — `imu.yaml` and `imu_noise_report.json` alongside the trimmed `imu.csv`. See [Kalibr Workflow](kalibr_workflow.md).

---

## compute_imu_noise_params

Derives Kalibr's `imu.yaml` noise parameters (`accelerometer_noise_density`, `accelerometer_random_walk`, `gyroscope_noise_density`, `gyroscope_random_walk`) from a static IMU capture via Allan variance — the analysis step that follows `auto_trim_static_imu`, replacing the need for a separate external Allan variance tool.

The overlapping Allan deviation curve of a static IMU axis is convex on a log-log plot: it falls with slope -1/2 while white sensor noise dominates at short cluster time `tau`, bottoms out, then rises with slope +1/2 once bias random walk dominates at long `tau`. The curve's minimum splits it into the two regions each slope is fit in; the -1/2 line's value at `tau=1s` gives the noise density, and the +1/2 line's value at `tau=3s` gives the random walk (IEEE-STD-952 convention).

```python
compute_imu_noise_params(imu_df: pd.DataFrame) -> Dict
```

**Args**:

- `imu_df` (DataFrame): Static (motionless) IMU data with columns timestamp_ns, sensor, x, y, z — run `auto_trim_static_imu` on the raw capture first so edge motion doesn't corrupt the analysis

**Returns**:

- `Dict`:

  ```python
  {
    "accelerometer_noise_density": 0.0021,   # m/s^2 / sqrt(Hz), averaged across x/y/z
    "accelerometer_random_walk": 0.00015,
    "accelerometer_rate_hz": 200.0,
    "gyroscope_noise_density": 0.00014,       # rad/s / sqrt(Hz)
    "gyroscope_random_walk": 1.2e-05,
    "gyroscope_rate_hz": 200.0,
    "detail": {
      "accelerometer_x": {"noise_density": ..., "random_walk": ...},
      "accelerometer_y": {...}, "accelerometer_z": {...},
      "gyroscope_x": {...}, "gyroscope_y": {...}, "gyroscope_z": {...}
    }
  }
  ```

**Raises**:

- `ValueError`: `imu_df` has no accel or gyro rows, or a capture is too short for the Allan deviation curve to show a clear minimum on some axis (white noise and random walk regions can't be separated) — use a longer static capture (hours, not minutes)

**Example**:

```python
from yuv_sensor import SessionDataLoader, auto_trim_static_imu, compute_imu_noise_params, export_imu_yaml

loader = SessionDataLoader("data/session_static_capture")
trimmed = auto_trim_static_imu(loader.imu_df, ends="both")["trimmed"]

noise_params = compute_imu_noise_params(trimmed)
print(f"accel noise_density: {noise_params['accelerometer_noise_density']:.6g}")
print(f"gyro random_walk: {noise_params['gyroscope_random_walk']:.6g}")

export_imu_yaml(noise_params, "imu.yaml")
```

### `export_imu_yaml(noise_params: Dict, output_path: Path, rostopic: str = "/imu0") -> Path`

Writes Kalibr's `imu.yaml` from the dict returned by `compute_imu_noise_params`.

**Note**: This is exposed via the CLI as part of `--trim_imu_static {start,end,both}`, which runs the trim, then this analysis, and writes `imu.yaml` + `imu_noise_report.json` (the full dict, including the per-axis `detail`) into the trim output directory. Prints "Could not derive imu.yaml" and continues (the trim output is still written) if the capture is too short. See [Kalibr Workflow](kalibr_workflow.md).

---

## calibrate_camera_from_checkerboard

Calibrates camera intrinsics directly from a checkerboard capture using OpenCV's own `findChessboardCorners`/`calibrateCamera` — a lightweight alternative to the full Kalibr pipeline for a quick sanity check. No rosbag, no `target.yaml`, no Kalibr install. It calibrates the camera alone and won't jointly refine camera-IMU extrinsics the way `kalibr_calibrate_imu_camera` does, so treat its output as a fast pre-check against `session.json`'s own intrinsics, not a replacement for a full Kalibr calibration.

Frames are decoded without rotation or undistortion, matching the orientation `session.json`'s own intrinsics are defined in — so `fx`/`fy` can be compared directly against `session.json`.

```python
calibrate_camera_from_checkerboard(loader: SessionDataLoader, checkerboard_size: Tuple[int, int], square_size: float = 1.0, max_frames: Optional[int] = None, frame_stride: int = 1) -> Dict
```

**Args**:

- `loader` (SessionDataLoader): Loaded session for a calibration-target capture (phone moved through frame in view of a checkerboard)
- `checkerboard_size` (Tuple[int, int]): `(cols, rows)` of INNER corners on the board — a board with 10x7 squares has a `(9, 6)` inner-corner grid
- `square_size` (float): Physical side length of one checkerboard square. Only scales the (otherwise unused) per-frame translation vectors — the intrinsics/distortion returned are unaffected, so the default of `1.0` is fine if you only care about those
- `max_frames` (int, optional): Cap on frames scanned for corners, or None for all
- `frame_stride` (int): Use every Nth frame (default: 1). Adjacent video frames barely change viewpoint, so striding through a long capture gets similar angle coverage for a fraction of the corner-detection cost

**Returns**:

- `Dict`:

  ```python
  {
    "frames_scanned": 400, "frames_used": 37,
    "checkerboard_size": [9, 6], "square_size": 1.0,
    "image_size": [1440, 1080],
    "rms_reprojection_error_px": 0.31,
    "intrinsics": {"fx": 1441.2, "fy": 1440.8, "cx": 721.5, "cy": 538.9},
    "distortion": {"k1": -0.098, "k2": 0.047, "p1": 0.0003, "p2": -0.0001, "k3": 0.0},
    "per_frame_errors": [{"frame_index": 0, "reprojection_error_px": 0.28}, ...],
    "session_json_intrinsics": [1440.5, 1440.5, 720.0, 540.0, 0.0],
    "fx_delta_from_session_json": 0.7, "fy_delta_from_session_json": 0.3
  }
  ```

  `session_json_intrinsics`/`*_delta_from_session_json` are only present when `session.json` has its own `intrinsics`.

**Raises**:

- `ValueError`: Checkerboard detected in fewer than 4 scanned frames — `calibrateCamera` needs several views at different angles (15-20+ for a reliable fit); usually a wrong `checkerboard_size`, a target out of frame, or heavy motion blur

**Example**:

```python
from yuv_sensor import SessionDataLoader, calibrate_camera_from_checkerboard

loader = SessionDataLoader("data/session_calib_target")
result = calibrate_camera_from_checkerboard(loader, checkerboard_size=(9, 6), frame_stride=5)

print(f"RMS reprojection error: {result['rms_reprojection_error_px']:.3f} px")
print(f"fx={result['intrinsics']['fx']:.1f} fy={result['intrinsics']['fy']:.1f}")
```

**Note**: This is exposed via the CLI as `--calibrate_camera --checkerboard_size COLSxROWS`, which writes `checkerboard_calibration_report.json` (the full dict above) to `--camera_calib_output_dir` (default: `session_dir/camera_calibration`).

---

## Utility Functions

### `decode_yuv420_888(yuv_bytes, width, height, chroma_layout, luma_row_stride, chroma_row_stride, chroma_pixel_stride, segment0_length, segment1_length, segment2_length) -> np.ndarray`

Low-level YUV decoding function (usually called internally by SessionDataLoader).

**Args**:

- `yuv_bytes` (bytes): Raw YUV frame data
- `width`, `height` (int): Frame dimensions
- `chroma_layout` (str): "420" or other
- `luma_row_stride`, `chroma_row_stride`, `chroma_pixel_stride` (int): Memory layout parameters
- `segment0_length`, `segment1_length`, `segment2_length` (int): Segment sizes

**Returns**:

- `np.ndarray`: RGB image (height×width×3, uint8)

**Note**: Prefer SessionDataLoader.get_decoded_frame() which handles all parameters automatically.

---

## Data Format Reference

### Session Configuration (session.json)

```json
{
  "device": "Pixel 6 Pro",
  "intrinsics": [1440.5, 1440.5, 720.0, 540.0],
  "distortion": [-0.1, 0.05, 0.0, 0.0],
  "sensor_orientation": 270
}
```

- `intrinsics`: [fx, fy, cx, cy] in pixels
- `distortion`: [k1, k2, p1, p2] OpenCV barrel distortion coefficients
- `sensor_orientation`: Rotation in degrees (0, 90, 180, 270)

### Frames CSV

```csv
filename,width,height,timestamp_ns,chroma_layout,luma_row_stride,chroma_row_stride,chroma_pixel_stride,segment0_length,segment1_length,segment2_length,sharpness
frame_0.yuv,1440,1080,419865219890714,420,1440,720,1,1475280,368820,368820,0.82
```

### IMU CSV

```csv
timestamp_ns,sensor,x,y,z
419865219890000,accel,-0.05,9.81,0.02
419865219892000,gyro,-0.01,0.02,0.0
```

- `sensor`: "accel" or "gyro"
- Units: m/s² (accel), rad/s (gyro)

---

## Error Handling

### Common Exceptions

```python
from yuv_sensor import SessionDataLoader

try:
    loader = SessionDataLoader("nonexistent/path")
except FileNotFoundError as e:
    print(f"Session not found: {e}")

try:
    frame = loader.get_decoded_frame(999)  # Out of range
except IndexError as e:
    print(f"Frame index out of range: {e}")
```

---

## Performance Tips

1. **Batch processing**: Iterate over frames instead of loading all at once

   ```python
   for idx in range(loader.get_frame_count()):
       rgb = loader.get_decoded_frame(idx)
       # Process rgb...
   ```

2. **Skip undistortion** if not needed (expensive OpenCV operation)

3. **Limit time windows** in get_synchronized_imu() for faster queries

4. **Use NumPy operations** on batch data for speed:

   ```python
   accel_data = imu['accel'][['x', 'y', 'z']].to_numpy()  # Faster than iterating
   ```
