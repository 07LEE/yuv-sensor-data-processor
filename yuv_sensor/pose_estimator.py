"""Camera pose estimation from IMU data using numerical integration."""

import math

import numba
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.spatial.transform import Rotation

from yuv_sensor.data_loader import validate_timestamp_compatibility


@numba.njit(cache=True)
def _integrate_trajectory(
    frame_ts: np.ndarray,
    accel_ts: np.ndarray,
    accel_xyz: np.ndarray,
    gyro_ts: np.ndarray,
    gyro_xyz: np.ndarray,
    r_init: np.ndarray,
    gravity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled two-pointer IMU integration -- the numeric core of estimate_trajectory.

    Gyro integration composes one small rotation matrix per gyro sample
    (via a hand-rolled Rodrigues formula, since scipy's Rotation isn't
    usable from nopython code); at typical gyro rates this is thousands of
    tiny matrix ops per session, where per-call Python/scipy overhead used
    to dominate the actual FLOPs. Compiling the whole frame x gyro-sample
    loop removes that overhead entirely.

    Returns:
        Tuple of (positions, rotation_matrices, velocities), one row per
        frame in frame_ts.
    """
    n_frames = frame_ts.shape[0]
    n_accel = accel_ts.shape[0]
    n_gyro = gyro_ts.shape[0]

    positions = np.zeros((n_frames, 3))
    rotations = np.zeros((n_frames, 3, 3))
    velocities = np.zeros((n_frames, 3))

    r_current = r_init.copy()
    v_current = np.zeros(3)
    p_current = np.zeros(3)
    g_world = np.array([0.0, 0.0, gravity])
    dR = np.zeros((3, 3))

    accel_idx = 0
    gyro_idx = 0
    prev_timestamp_ns = frame_ts[0]
    has_prev = False

    for f in range(n_frames):
        frame_t = frame_ts[f]

        a_start = accel_idx
        while accel_idx < n_accel and accel_ts[accel_idx] <= frame_t:
            accel_idx += 1
        a_end = accel_idx

        g_start = gyro_idx
        while gyro_idx < n_gyro and gyro_ts[gyro_idx] <= frame_t:
            gyro_idx += 1
        g_end = gyro_idx

        if has_prev and a_end > a_start and g_end > g_start:
            prev_gyro_ts = prev_timestamp_ns
            for i in range(g_start, g_end):
                sample_ts = gyro_ts[i]
                dt = (sample_ts - prev_gyro_ts) * 1e-9
                prev_gyro_ts = sample_ts

                wx = gyro_xyz[i, 0]
                wy = gyro_xyz[i, 1]
                wz = gyro_xyz[i, 2]
                norm_w = math.sqrt(wx * wx + wy * wy + wz * wz)
                angle = norm_w * dt
                if angle > 1e-8:
                    ax = wx / norm_w
                    ay = wy / norm_w
                    az = wz / norm_w
                    s = math.sin(angle)
                    c = math.cos(angle)
                    one_c = 1.0 - c

                    dR[0, 0] = c + ax * ax * one_c
                    dR[0, 1] = ax * ay * one_c - az * s
                    dR[0, 2] = ax * az * one_c + ay * s
                    dR[1, 0] = ay * ax * one_c + az * s
                    dR[1, 1] = c + ay * ay * one_c
                    dR[1, 2] = ay * az * one_c - ax * s
                    dR[2, 0] = az * ax * one_c - ay * s
                    dR[2, 1] = az * ay * one_c + ax * s
                    dR[2, 2] = c + az * az * one_c

                    r_current = r_current @ dR

            accel_mean = np.zeros(3)
            for i in range(a_start, a_end):
                accel_mean[0] += accel_xyz[i, 0]
                accel_mean[1] += accel_xyz[i, 1]
                accel_mean[2] += accel_xyz[i, 2]
            accel_mean /= (a_end - a_start)

            accel_world = r_current @ accel_mean
            accel_world -= g_world

            dt_s = (frame_t - prev_timestamp_ns) * 1e-9
            v_current = v_current + accel_world * dt_s
            p_current = p_current + v_current * dt_s

        positions[f] = p_current
        rotations[f] = r_current
        velocities[f] = v_current

        prev_timestamp_ns = frame_t
        has_prev = True

    return positions, rotations, velocities


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

        accel_data = imu_df[imu_df["sensor"] == "accel"].sort_values("timestamp_ns")
        gyro_data = imu_df[imu_df["sensor"] == "gyro"].sort_values("timestamp_ns")

        if accel_data.empty or gyro_data.empty:
            return self._zero_trajectory(frames_df)

        validate_timestamp_compatibility(frames_df, imu_df)

        r_init = initial_rotation if initial_rotation is not None else np.eye(3)

        frame_ts = frames_df["timestamp_ns"].to_numpy(dtype=np.int64)
        accel_ts = accel_data["timestamp_ns"].to_numpy(dtype=np.int64)
        accel_xyz = accel_data[["x", "y", "z"]].to_numpy(dtype=np.float64)
        gyro_ts = gyro_data["timestamp_ns"].to_numpy(dtype=np.int64)
        gyro_xyz = gyro_data[["x", "y", "z"]].to_numpy(dtype=np.float64)

        positions, rotations, velocities = _integrate_trajectory(
            frame_ts, accel_ts, accel_xyz, gyro_ts, gyro_xyz,
            np.ascontiguousarray(r_init, dtype=np.float64), float(self.gravity),
        )
        # Batched, not per-frame: one scipy call converting every rotation
        # matrix to a quaternion instead of one Python-level call per frame.
        quaternions = Rotation.from_matrix(rotations).as_quat()  # xyzw format

        trajectory = {}
        for i, frame_idx in enumerate(frames_df.index):
            trajectory[frame_idx] = {
                "timestamp_ns": int(frame_ts[i]),
                "position": positions[i],
                "rotation_matrix": rotations[i],
                "quaternion": quaternions[i],
                "velocity": velocities[i],
            }

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


def validate_t_cam_imu(t_cam_imu: np.ndarray) -> np.ndarray:
    """Check that t_cam_imu is a rigid 4x4 transform and return it as float64.

    Raises:
        ValueError: If the shape, bottom row or rotation block is not rigid.
    """
    t = np.asarray(t_cam_imu, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"T_cam_imu must be 4x4, got shape {t.shape}")
    if not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("T_cam_imu bottom row must be [0, 0, 0, 1]")
    r = t[:3, :3]
    if not (np.allclose(r @ r.T, np.eye(3), atol=1e-3) and np.linalg.det(r) > 0):
        raise ValueError("T_cam_imu rotation block is not a proper rotation")
    return t


def imu_to_camera_trajectory(
    trajectory: Dict,
    t_cam_imu: np.ndarray,
    r_raw_up: Optional[np.ndarray] = None,
) -> Dict:
    """Convert an IMU/body trajectory into a camera trajectory.

    estimate_trajectory() tracks the IMU: position is the IMU origin in the
    world and rotation_matrix is R_wi (IMU-to-world). With the Kalibr
    convention x_cam = T_cam_imu @ x_imu, the camera pose is
    R_wc = R_wi @ R_ic and C = p_wi + R_wi @ t_ic, where
    R_ic = R_ci.T and t_ic = -R_ic @ t_ci.

    Args:
        trajectory: Output of PoseEstimator.estimate_trajectory().
        t_cam_imu: 4x4 transform taking IMU-frame points into the raw camera
            frame (OpenCV axes: x right, y down, z forward).
        r_raw_up: Optional rotation from the upright camera frame of the
            exported images to the raw camera frame (see
            upright_camera_rotation). None keeps the raw camera frame.

    Returns:
        Dict mapping frame_index to {timestamp_ns, position (camera center),
        rotation_matrix (camera-to-world), quaternion (xyzw)}.
    """
    t = validate_t_cam_imu(t_cam_imu)
    r_ci, t_ci = t[:3, :3], t[:3, 3]
    r_ic = r_ci.T
    t_ic = -r_ic @ t_ci
    if r_raw_up is not None:
        r_ic = r_ic @ r_raw_up

    frame_indices = list(trajectory.keys())
    r_wi = np.stack([trajectory[i]["rotation_matrix"] for i in frame_indices])
    p_wi = np.stack([trajectory[i]["position"] for i in frame_indices])
    r_wc = r_wi @ r_ic
    centers = p_wi + np.einsum("nij,j->ni", r_wi, t_ic)
    quats = Rotation.from_matrix(r_wc).as_quat()

    return {
        idx: {
            "timestamp_ns": trajectory[idx]["timestamp_ns"],
            "position": centers[n],
            "rotation_matrix": r_wc[n],
            "quaternion": quats[n],
        }
        for n, idx in enumerate(frame_indices)
    }
