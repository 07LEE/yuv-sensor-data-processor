import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.conftest import T0, make_frames, make_imu
from yuv_sensor.data_loader import TimestampDomainError
from yuv_sensor.pose_estimator import PoseEstimator


def test_no_imu_gives_identity_trajectory():
    frames = make_frames(count=3)
    traj = PoseEstimator().estimate_trajectory(None, frames)
    assert len(traj) == 3
    for pose in traj.values():
        np.testing.assert_array_equal(pose["position"], np.zeros(3))
        np.testing.assert_array_equal(pose["rotation_matrix"], np.eye(3))
        np.testing.assert_array_equal(pose["quaternion"], [0, 0, 0, 1])


def test_missing_gyro_gives_identity_trajectory():
    imu = make_imu()
    imu = imu[imu["sensor"] == "accel"]
    traj = PoseEstimator().estimate_trajectory(imu, make_frames(count=3))
    assert all(np.array_equal(p["rotation_matrix"], np.eye(3)) for p in traj.values())


def test_stationary_device_stays_put():
    """Accel reads +g on z and gyro is zero: no motion, no drift."""
    frames = make_frames(count=20)
    traj = PoseEstimator(gravity_magnitude=9.81).estimate_trajectory(make_imu(), frames)
    last = traj[frames.index[-1]]
    np.testing.assert_allclose(last["position"], 0.0, atol=1e-9)
    np.testing.assert_allclose(last["velocity"], 0.0, atol=1e-9)
    np.testing.assert_allclose(last["rotation_matrix"], np.eye(3), atol=1e-12)


def test_constant_yaw_rate_integrates_to_expected_angle():
    rate = 0.5  # rad/s about z
    frames = make_frames(count=20)
    imu = make_imu(gyro=(0.0, 0.0, rate), seconds=5.0)
    traj = PoseEstimator().estimate_trajectory(imu, frames)

    first, last = traj[frames.index[0]], traj[frames.index[-1]]
    elapsed_s = (last["timestamp_ns"] - first["timestamp_ns"]) * 1e-9
    r = last["rotation_matrix"]
    yaw = np.arctan2(r[1, 0], r[0, 0])

    assert yaw == pytest.approx(rate * elapsed_s, abs=0.02)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)  # still a rotation


def test_initial_rotation_is_respected():
    r_init = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
    frames = make_frames(count=5)
    traj = PoseEstimator().estimate_trajectory(make_imu(), frames, initial_rotation=r_init)
    np.testing.assert_allclose(traj[frames.index[-1]]["rotation_matrix"], r_init, atol=1e-9)


def test_constant_forward_acceleration_moves_along_x():
    """World-frame x accel of 1 m/s^2 over ~1 s => about 0.5 m traveled (frame-rate integration)."""
    frames = make_frames(count=10)
    imu = make_imu(accel=(1.0, 0.0, 9.81), seconds=3.0)
    traj = PoseEstimator().estimate_trajectory(imu, frames)
    last = traj[frames.index[-1]]
    assert last["position"][0] > 0.2
    assert last["velocity"][0] > 0.5
    assert abs(last["position"][1]) < 1e-9


def test_incompatible_clocks_raise():
    with pytest.raises(TimestampDomainError):
        PoseEstimator().estimate_trajectory(make_imu(t0=T0), make_frames(t0=10 ** 18))


def test_trajectory_has_one_entry_per_frame_with_quaternion():
    frames = make_frames(count=7)
    traj = PoseEstimator().estimate_trajectory(make_imu(), frames)
    assert list(traj.keys()) == list(frames.index)
    assert all(p["quaternion"].shape == (4,) for p in traj.values())


class TestImuToCameraTrajectory:
    @staticmethod
    def _traj(r_wi, p_wi):
        return {0: {"timestamp_ns": 1, "position": np.asarray(p_wi, float),
                    "rotation_matrix": np.asarray(r_wi, float), "quaternion": None}}

    @staticmethod
    def _t_cam_imu(r_ci, t_ci):
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r_ci, t_ci
        return t

    def test_identity_extrinsics_keep_imu_pose(self):
        from yuv_sensor.pose_estimator import imu_to_camera_trajectory
        r = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        cam = imu_to_camera_trajectory(self._traj(r, [1, 2, 3]), np.eye(4))[0]
        np.testing.assert_allclose(cam["rotation_matrix"], r)
        np.testing.assert_allclose(cam["position"], [1, 2, 3])

    def test_camera_pose_composes_with_extrinsics(self):
        from yuv_sensor.pose_estimator import imu_to_camera_trajectory
        r_wi = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        r_ci = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        t_ci = np.array([0.1, 0.0, 0.0])
        cam = imu_to_camera_trajectory(self._traj(r_wi, [1, 0, 0]), self._t_cam_imu(r_ci, t_ci))[0]

        r_ic = r_ci.T
        np.testing.assert_allclose(cam["rotation_matrix"], r_wi @ r_ic)
        # a point at the camera origin in the IMU frame is t_ic, in world R_wi @ t_ic + p
        np.testing.assert_allclose(cam["position"], np.array([1, 0, 0]) + r_wi @ (-r_ic @ t_ci))
        # round trip: IMU origin seen from the camera sits at t_ci
        r_cw, c = cam["rotation_matrix"].T, cam["position"]
        np.testing.assert_allclose(r_cw @ (np.array([1, 0, 0]) - c), t_ci, atol=1e-12)

    def test_upright_rotation_is_applied(self):
        from yuv_sensor.camera_calib import upright_camera_rotation
        from yuv_sensor.pose_estimator import imu_to_camera_trajectory
        up = upright_camera_rotation(90)
        cam = imu_to_camera_trajectory(self._traj(np.eye(3), [0, 0, 0]), np.eye(4), up)[0]
        np.testing.assert_allclose(cam["rotation_matrix"], up)
        # clockwise-rotated image: upright x axis is the raw -y axis
        np.testing.assert_allclose(up @ [1, 0, 0], [0, -1, 0], atol=1e-12)

    def test_invalid_extrinsics_rejected(self):
        from yuv_sensor.pose_estimator import imu_to_camera_trajectory
        with pytest.raises(ValueError):
            imu_to_camera_trajectory(self._traj(np.eye(3), [0, 0, 0]), np.eye(3))
        bad = np.eye(4)
        bad[0, 0] = 2.0
        with pytest.raises(ValueError):
            imu_to_camera_trajectory(self._traj(np.eye(3), [0, 0, 0]), bad)
