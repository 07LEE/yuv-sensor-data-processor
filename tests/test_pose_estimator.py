import numpy as np
import pytest

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
