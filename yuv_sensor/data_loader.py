"""Dataset loader and nanosecond timestamp synchronizer for session capture data."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from yuv_sensor.yuv_decoder import decode_yuv420_888
from yuv_sensor.camera_calib import CameraCalibration


class SessionDataLoader:
    """Loader for session data including frames.csv, capture.csv, imu.csv, session.json, and raw .yuv files."""

    def __init__(self, session_dir: str):
        """Initialize loader with session directory path.

        Args:
            session_dir: Path to session directory (e.g. data/session_419864820).
        """
        self.session_path = Path(session_dir)
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

        # Load session.json
        session_json_path = self.session_path / "session.json"
        with open(session_json_path, "r", encoding="utf-8") as f:
            self.session_config: Dict[str, Any] = json.load(f)

        self.camera_calib = CameraCalibration(self.session_config)

        # Load metadata CSV files
        self.frames_df = pd.read_csv(self.session_path / "frames.csv")
        self.capture_df = pd.read_csv(self.session_path / "capture.csv") if (self.session_path / "capture.csv").exists() else None
        self.imu_df = pd.read_csv(self.session_path / "imu.csv") if (self.session_path / "imu.csv").exists() else None

    def get_frame_count(self) -> int:
        """Returns total number of frames in frames.csv."""
        return len(self.frames_df)

    def load_raw_frame(self, index: int) -> Tuple[bytes, pd.Series]:
        """Reads raw .yuv bytes and metadata row for frame at index.

        Args:
            index: Frame row index.

        Returns:
            Tuple of (yuv_bytes, frame_row).
        """
        row = self.frames_df.iloc[index]
        filename = row["filename"]
        frame_path = self.session_path / "frames" / filename
        if not frame_path.exists():
            raise FileNotFoundError(f"Raw YUV frame not found: {frame_path}")

        with open(frame_path, "rb") as f:
            yuv_bytes = f.read()

        return yuv_bytes, row

    def get_decoded_frame(self, index: int, apply_undistort: bool = False, apply_rotation: bool = True) -> np.ndarray:
        """Loads and decodes raw YUV frame at index into RGB image.

        Args:
            index: Frame row index.
            apply_undistort: If True, applies camera lens distortion correction.
            apply_rotation: If True, rotates image upright according to sensor_orientation.

        Returns:
            np.ndarray: Decoded RGB image.
        """
        yuv_bytes, row = self.load_raw_frame(index)

        rgb = decode_yuv420_888(
            yuv_bytes=yuv_bytes,
            width=int(row["width"]),
            height=int(row["height"]),
            chroma_layout=str(row["chroma_layout"]),
            luma_row_stride=int(row["luma_row_stride"]),
            chroma_row_stride=int(row["chroma_row_stride"]),
            chroma_pixel_stride=int(row["chroma_pixel_stride"]),
            segment0_length=int(row.get("segment0_length", 0)),
            segment1_length=int(row.get("segment1_length", 0)),
            segment2_length=int(row.get("segment2_length", 0))
        )

        if apply_undistort:
            rgb = self.camera_calib.undistort_frame(rgb)

        if apply_rotation:
            rgb = self.camera_calib.rotate_frame(rgb)

        return rgb

    def get_synchronized_imu(self, timestamp_ns: int, time_window_ms: float = 100.0) -> Dict[str, pd.DataFrame]:
        """Finds IMU samples within a time window surrounding frame timestamp_ns.

        Args:
            timestamp_ns: Target frame timestamp in nanoseconds.
            time_window_ms: Half-width of time window in milliseconds.

        Returns:
            Dict containing 'accel' and 'gyro' DataFrames centered around timestamp_ns.
        """
        if self.imu_df is None:
            return {"accel": pd.DataFrame(), "gyro": pd.DataFrame()}

        window_ns = int(time_window_ms * 1e6)
        min_ts = timestamp_ns - window_ns
        max_ts = timestamp_ns + window_ns

        sub_imu = self.imu_df[(self.imu_df["timestamp_ns"] >= min_ts) & (self.imu_df["timestamp_ns"] <= max_ts)]
        accel = sub_imu[sub_imu["sensor"] == "accel"].copy()
        gyro = sub_imu[sub_imu["sensor"] == "gyro"].copy()

        return {"accel": accel, "gyro": gyro}

    def get_nearest_capture_metadata(self, timestamp_ns: int) -> Optional[pd.Series]:
        """Finds the nearest exposure and control metadata row in capture.csv for given timestamp_ns.

        Args:
            timestamp_ns: Target frame timestamp in nanoseconds.

        Returns:
            Matching pd.Series row from capture.csv or None.
        """
        if self.capture_df is None or len(self.capture_df) == 0:
            return None

        idx = (self.capture_df["timestamp_ns"] - timestamp_ns).abs().idxmin()
        return self.capture_df.iloc[idx]
