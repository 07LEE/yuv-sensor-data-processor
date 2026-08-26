"""Camera pose estimation from IMU data using numerical integration."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.spatial.transform import Rotation


class PoseEstimator:
    """Estimates camera pose trajectory from IMU accelerometer and gyroscope data."""

    def __init__(self, gravity_magnitude: float = 9.81):
        """Initialize PoseEstimator.

        Args:
            gravity_magnitude: Magnitude of gravity acceleration in m/s^2.
        """
        self.gravity = gravity_magnitude

    def estimate_trajectory(
        self,
        imu_df: Optional[pd.DataFrame],
        frames_df: pd.DataFrame,
        initial_rotation: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Estimate camera trajectory from IMU data.

        Args:
            imu_df: IMU dataframe with columns: timestamp_ns, sensor, x, y, z
            frames_df: Frames dataframe with column: timestamp_ns
            initial_rotation: Initial rotation matrix (default: identity)

        Returns:
            Dict mapping frame_index to {timestamp_ns, position, rotation_matrix, quaternion}
        """
        if imu_df is None or imu_df.empty:
            return self._zero_trajectory(frames_df)

        trajectory = {}
        accel_data = imu_df[imu_df["sensor"] == "accel"].sort_values("timestamp_ns")
        gyro_data = imu_df[imu_df["sensor"] == "gyro"].sort_values("timestamp_ns")

        if accel_data.empty or gyro_data.empty:
            return self._zero_trajectory(frames_df)

        # Initialize pose
        R_current = initial_rotation if initial_rotation is not None else np.eye(3)
        v_current = np.zeros(3)  # velocity
        p_current = np.zeros(3)  # position

        # Gravity vector in world frame
        g_world = np.array([0, 0, self.gravity])

        # Pre-extract sorted numpy arrays once: frames_df/accel_data/gyro_data
        # are each timestamp-ordered, so a frame-by-frame two-pointer walk
        # visits every IMU sample at most once (O(N+M)) instead of rescanning
        # the full accel/gyro tables per frame (O(N*M)).
        accel_ts = accel_data["timestamp_ns"].to_numpy()
        accel_xyz = accel_data[["x", "y", "z"]].to_numpy()
        gyro_ts = gyro_data["timestamp_ns"].to_numpy()
        gyro_xyz = gyro_data[["x", "y", "z"]].to_numpy()
        n_accel = len(accel_ts)
        n_gyro = len(gyro_ts)

        prev_timestamp_ns = None
        accel_idx = 0
        gyro_idx = 0

        for frame_idx, frame_row in frames_df.iterrows():
            frame_ts = int(frame_row["timestamp_ns"])

            a_start = accel_idx
            while accel_idx < n_accel and accel_ts[accel_idx] <= frame_ts:
                accel_idx += 1
            a_end = accel_idx

            g_start = gyro_idx
            while gyro_idx < n_gyro and gyro_ts[gyro_idx] <= frame_ts:
                gyro_idx += 1
            g_end = gyro_idx

            # Integrate IMU between frame timestamps
            if prev_timestamp_ns is not None and a_end > a_start and g_end > g_start:
                # Update rotation from gyro
                prev_gyro_ts = prev_timestamp_ns
                for i in range(g_start, g_end):
                    sample_ts = int(gyro_ts[i])
                    dt = (sample_ts - prev_gyro_ts) * 1e-9
                    prev_gyro_ts = sample_ts
                    omega = gyro_xyz[i]
                    angle = np.linalg.norm(omega) * dt
                    if angle > 1e-8:
                        axis = omega / np.linalg.norm(omega)
                        dR = Rotation.from_rotvec(axis * angle).as_matrix()
                        R_current = R_current @ dR

                # Update velocity and position from accel
                accel_mean = accel_xyz[a_start:a_end].mean(axis=0)
                accel_world = R_current @ accel_mean
                accel_world -= g_world  # Remove gravity

                dt_s = (frame_ts - prev_timestamp_ns) * 1e-9
                v_current += accel_world * dt_s
                p_current += v_current * dt_s

            trajectory[frame_idx] = {
                "timestamp_ns": frame_ts,
                "position": p_current.copy(),
                "rotation_matrix": R_current.copy(),
                "quaternion": Rotation.from_matrix(R_current).as_quat(),  # xyzw format
                "velocity": v_current.copy(),
            }

            prev_timestamp_ns = frame_ts

        return trajectory

    def _zero_trajectory(self, frames_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return identity trajectory when IMU is unavailable."""
        trajectory = {}
        for frame_idx, frame_row in frames_df.iterrows():
            trajectory[frame_idx] = {
                "timestamp_ns": int(frame_row["timestamp_ns"]),
                "position": np.zeros(3),
                "rotation_matrix": np.eye(3),
                "quaternion": np.array([0, 0, 0, 1]),  # identity quaternion xyzw
                "velocity": np.zeros(3),
            }
        return trajectory

    def get_frame_poses(
        self,
        trajectory: Dict[str, Dict],
        frames_df: pd.DataFrame
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Convert trajectory to list of (frame_index, position, quaternion).

        Args:
            trajectory: Output from estimate_trajectory()
            frames_df: Frames dataframe

        Returns:
            List of (frame_index, position [3], quaternion [4])
        """
        poses = []
        for frame_idx, pose_data in trajectory.items():
            poses.append((
                frame_idx,
                pose_data["position"],
                pose_data["quaternion"]
            ))
        return poses
