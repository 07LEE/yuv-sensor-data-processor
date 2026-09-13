"""Data processor module to build and export synchronized frame-to-IMU and exposure datasets."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from yuv_sensor.data_loader import SessionDataLoader


class DataProcessor:
    """Processor to aggregate and export synchronized multi-sensor datasets."""

    def __init__(self, loader: SessionDataLoader):
        """Initialize DataProcessor with a SessionDataLoader instance.

        Args:
            loader: Loaded SessionDataLoader instance.
        """
        self.loader = loader

    def generate_synchronized_dataset(self, time_window_ms: float = 50.0) -> List[Dict[str, Any]]:
        """Generates synchronized data structure linking each frame to exposure & IMU logs.

        Args:
            time_window_ms: Time window half-width in milliseconds for surrounding IMU samples.

        Returns:
            List of synchronized frame dictionaries.
        """
        sync_frames: List[Dict[str, Any]] = []
        total_frames = self.loader.get_frame_count()

        for idx in range(total_frames):
            frame_row = self.loader.frames_df.iloc[idx]
            timestamp_ns = int(frame_row["timestamp_ns"])

            # 1. Frame basic info
            frame_entry: Dict[str, Any] = {
                "frame_index": idx,
                "timestamp_ns": timestamp_ns,
                "filename": str(frame_row["filename"]),
                "width": int(frame_row["width"]),
                "height": int(frame_row["height"]),
                "sharpness": float(frame_row["sharpness"]) if "sharpness" in frame_row else None,
            }

            # 2. Exposure & control metadata matching
            capture_row = self.loader.get_nearest_capture_metadata(timestamp_ns)
            if capture_row is not None:
                frame_entry["exposure"] = {
                    "exposure_ns": int(capture_row.get("exposure_ns", 0)),
                    "sensitivity": int(capture_row.get("sensitivity", 0)),
                    "focus_diopters": float(capture_row.get("focus_diopters", 0.0)),
                    "rolling_shutter_skew_ns": int(capture_row.get("rolling_shutter_skew_ns", 0)),
                }

            # 3. Surrounding IMU window matching
            imu_sync = self.loader.get_synchronized_imu(timestamp_ns, time_window_ms=time_window_ms)
            
            accel_list = imu_sync["accel"][["timestamp_ns", "x", "y", "z"]].to_dict(orient="records") if not imu_sync["accel"].empty else []
            gyro_list = imu_sync["gyro"][["timestamp_ns", "x", "y", "z"]].to_dict(orient="records") if not imu_sync["gyro"].empty else []

            frame_entry["imu_window"] = {
                "window_ms": time_window_ms,
                "accel_samples_count": len(accel_list),
                "gyro_samples_count": len(gyro_list),
                "accel": accel_list,
                "gyro": gyro_list,
            }

            # 4. Inter-frame IMU segment (IMU samples between current frame and next frame)
            if idx < total_frames - 1:
                next_ts = int(self.loader.frames_df.iloc[idx + 1]["timestamp_ns"])
                if self.loader.imu_df is not None:
                    inter_imu = self.loader.get_imu_in_range(timestamp_ns, next_ts)
                    inter_accel = inter_imu["accel"][["timestamp_ns", "x", "y", "z"]].to_dict(orient="records")
                    inter_gyro = inter_imu["gyro"][["timestamp_ns", "x", "y", "z"]].to_dict(orient="records")

                    frame_entry["interframe_imu"] = {
                        "next_timestamp_ns": next_ts,
                        "dt_ns": next_ts - timestamp_ns,
                        "accel": inter_accel,
                        "gyro": inter_gyro,
                    }

            sync_frames.append(frame_entry)

        return sync_frames

    def export_synchronized_dataset(self, output_path: Optional[str] = None, time_window_ms: float = 50.0) -> Path:
        """Generates and exports synchronized dataset to JSON file.

        Args:
            output_path: Target path for output JSON file (defaults to session_dir/synchronized_dataset.json).
            time_window_ms: Half-width time window in milliseconds.

        Returns:
            Path of exported JSON file.
        """
        if output_path is None:
            out_file = self.loader.session_path / "synchronized_dataset.json"
        else:
            out_file = Path(output_path)

        out_file.parent.mkdir(parents=True, exist_ok=True)

        dataset_structure = {
            "session_id": self.loader.session_path.name,
            "session_config": self.loader.session_config,
            "total_frames": len(self.loader.frames_df),
            "frames": self.generate_synchronized_dataset(time_window_ms=time_window_ms)
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dataset_structure, f, indent=2)

        return out_file
