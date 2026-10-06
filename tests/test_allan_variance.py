import numpy as np
import pytest

from tests.conftest import T0
from yuv_sensor.allan_variance import compute_imu_noise_params, overlapping_allan_deviation


def test_white_noise_has_minus_half_slope_and_known_level():
    rng = np.random.default_rng(1)
    sigma, dt = 0.05, 0.005
    data = rng.normal(0.0, sigma, 200_000)

    taus, adev = overlapping_allan_deviation(data, dt)

    # ADEV of white rate noise is sigma * sqrt(dt / tau); at tau = dt it equals sigma.
    assert adev[0] == pytest.approx(sigma, rel=0.05)
    mask = taus <= 1.0
    slope = np.polyfit(np.log(taus[mask]), np.log(adev[mask]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.05)


def test_constant_signal_has_zero_deviation():
    _, adev = overlapping_allan_deviation(np.full(1000, 3.0), 0.01)
    np.testing.assert_allclose(adev, 0.0, atol=1e-12)


def test_cluster_times_are_increasing_multiples_of_dt():
    taus, _ = overlapping_allan_deviation(np.random.default_rng(0).normal(size=3000), 0.01)
    assert np.all(np.diff(taus) > 0)
    assert taus[0] == pytest.approx(0.01)
    assert taus[-1] <= 3000 / 3 * 0.01 + 1e-9


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="not enough samples"):
        overlapping_allan_deviation(np.array([1.0, 2.0]), 0.01)


def _imu_with_noise(sigma_a, sigma_g, seconds, rate_hz=100.0, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    n = int(seconds * rate_hz)
    ts = T0 + (np.arange(n) * (1e9 / rate_hz)).astype(np.int64)
    frames = []
    for sensor, sigma, base in (("accel", sigma_a, 9.81), ("gyro", sigma_g, 0.0)):
        frames.append(pd.DataFrame({
            "timestamp_ns": ts, "sensor": sensor,
            "x": rng.normal(0.0, sigma, n),
            "y": rng.normal(0.0, sigma, n),
            "z": base + rng.normal(0.0, sigma, n),
        }))
    return pd.concat(frames, ignore_index=True)


def test_missing_sensor_rows_raise():
    imu = _imu_with_noise(0.01, 0.001, 60)
    with pytest.raises(ValueError, match="no 'accel' rows"):
        compute_imu_noise_params(imu[imu["sensor"] == "gyro"])


def test_too_short_capture_raises():
    with pytest.raises(ValueError):
        compute_imu_noise_params(_imu_with_noise(0.01, 0.001, 0.01))
