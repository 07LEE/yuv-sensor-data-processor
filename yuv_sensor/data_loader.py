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

        # Built lazily on first use: per-sensor timestamp-sorted IMU tables
        # and their timestamp arrays (for get_imu_in_range's binary search),
        # and capture.csv sorted the same way (for get_nearest_capture_metadata).
        # Building these once instead of re-filtering the full table on every
        # call turns per-frame lookups from O(session length) into O(log
        # session length).
        self._imu_cache_built = False
        self._imu_cache: Optional[Dict[str, Any]] = None
        self._capture_cache_built = False
        self._capture_cache: Optional[Dict[str, Any]] = None

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

    def _ensure_imu_cache(self) -> None:
        """Builds the per-sensor timestamp-sorted IMU cache once, lazily."""
        if self._imu_cache_built:
            return
        self._imu_cache_built = True

        if self.imu_df is None:
            self._imu_cache = None
            return

        accel_df = self.imu_df[self.imu_df["sensor"] == "accel"].sort_values("timestamp_ns", kind="stable").reset_index(drop=True)
        gyro_df = self.imu_df[self.imu_df["sensor"] == "gyro"].sort_values("timestamp_ns", kind="stable").reset_index(drop=True)
        self._imu_cache = {
            "accel_df": accel_df,
            "accel_ts": accel_df["timestamp_ns"].to_numpy(),
            "gyro_df": gyro_df,
            "gyro_ts": gyro_df["timestamp_ns"].to_numpy(),
        }

    def get_imu_in_range(self, min_ts: int, max_ts: int) -> Dict[str, pd.DataFrame]:
        """Finds IMU samples with timestamp_ns in [min_ts, max_ts] (inclusive).

        Each sensor's samples are timestamp-sorted once and looked up here by
        binary search (np.searchsorted) instead of re-scanning the full
        table, so a per-frame caller costs O(log session length) rather than
        O(session length).

        Args:
            min_ts: Inclusive lower bound, in nanoseconds.
            max_ts: Inclusive upper bound, in nanoseconds.

        Returns:
            Dict containing 'accel' and 'gyro' DataFrames.
        """
        self._ensure_imu_cache()
        if self._imu_cache is None:
            return {"accel": pd.DataFrame(), "gyro": pd.DataFrame()}

        cache = self._imu_cache
        accel_lo = np.searchsorted(cache["accel_ts"], min_ts, side="left")
        accel_hi = np.searchsorted(cache["accel_ts"], max_ts, side="right")
        gyro_lo = np.searchsorted(cache["gyro_ts"], min_ts, side="left")
        gyro_hi = np.searchsorted(cache["gyro_ts"], max_ts, side="right")

        return {
            "accel": cache["accel_df"].iloc[accel_lo:accel_hi],
            "gyro": cache["gyro_df"].iloc[gyro_lo:gyro_hi],
        }

    def get_synchronized_imu(self, timestamp_ns: int, time_window_ms: float = 100.0) -> Dict[str, pd.DataFrame]:
        """Finds IMU samples within a time window surrounding frame timestamp_ns.

        Args:
            timestamp_ns: Target frame timestamp in nanoseconds.
            time_window_ms: Half-width of time window in milliseconds.

        Returns:
            Dict containing 'accel' and 'gyro' DataFrames centered around timestamp_ns.
        """
        window_ns = int(time_window_ms * 1e6)
        return self.get_imu_in_range(timestamp_ns - window_ns, timestamp_ns + window_ns)

    def get_nearest_capture_metadata(self, timestamp_ns: int) -> Optional[pd.Series]:
        """Finds the nearest exposure and control metadata row in capture.csv for given timestamp_ns.

        Looks up the nearest neighbor via binary search into a timestamp
        -sorted cache built once, instead of an abs-diff argmin over the
        full table on every call.

        Args:
            timestamp_ns: Target frame timestamp in nanoseconds.

        Returns:
            Matching pd.Series row from capture.csv or None.
        """
        if not self._capture_cache_built:
            self._capture_cache_built = True
            if self.capture_df is None or len(self.capture_df) == 0:
                self._capture_cache = None
            else:
                sorted_df = self.capture_df.sort_values("timestamp_ns", kind="stable").reset_index(drop=True)
                self._capture_cache = {"df": sorted_df, "ts": sorted_df["timestamp_ns"].to_numpy()}

        if self._capture_cache is None:
            return None

        ts = self._capture_cache["ts"]
        idx = int(np.searchsorted(ts, timestamp_ns))
        if idx == 0:
            best = 0
        elif idx == len(ts):
            best = len(ts) - 1
        else:
            before, after = ts[idx - 1], ts[idx]
            best = idx - 1 if (timestamp_ns - before) <= (after - timestamp_ns) else idx

        return self._capture_cache["df"].iloc[best]
