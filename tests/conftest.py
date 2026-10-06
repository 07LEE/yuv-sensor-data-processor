"""Shared fixtures: small synthetic sessions written to a temp directory."""

import json

import numpy as np
import pandas as pd
import pytest

from yuv_sensor.data_loader import SessionDataLoader

T0 = 1_000_000_000_000  # arbitrary session start, ns


def make_imu(t0: int = T0, seconds: float = 10.0, rate_hz: float = 200.0,
             accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0)) -> pd.DataFrame:
    """Interleaved accel + gyro rows at a constant rate, in imu.csv layout."""
    n = int(seconds * rate_hz)
    ts = t0 + (np.arange(n) * (1e9 / rate_hz)).astype(np.int64)
    frames = []
    for sensor, value in (("accel", accel), ("gyro", gyro)):
        frames.append(pd.DataFrame({
            "timestamp_ns": ts,
            "sensor": sensor,
            "x": value[0], "y": value[1], "z": value[2],
        }))
    return pd.concat(frames, ignore_index=True)


def make_frames(t0: int = T0, count: int = 10, period_ns: int = 100_000_000,
                width: int = 4000, height: int = 3000) -> pd.DataFrame:
    """frames.csv rows; first frame one period after t0 so it sits inside the IMU range."""
    ts = t0 + period_ns + np.arange(count, dtype=np.int64) * period_ns
    return pd.DataFrame({
        "timestamp_ns": ts,
        "filename": [f"{t}.yuv" for t in ts],
        "width": width,
        "height": height,
        "sharpness": 100.0,
    })


DEFAULT_CONFIG = {
    "device": "test",
    "sensor_orientation": 0,
    "intrinsics": [3100.0, 3050.0, 2000.0, 1480.0, 0.0],
    "distortion": [0.1, -0.05, 0.01, 0.002, -0.003],
    "pre_correction_active_array": [0, 0, 4032, 3024],
}


@pytest.fixture
def make_session(tmp_path):
    """Factory writing a session directory and returning its SessionDataLoader."""

    def _make(config=None, frames=None, imu=None, capture=None) -> SessionDataLoader:
        session = tmp_path / "session_1"
        session.mkdir(exist_ok=True)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        (session / "session.json").write_text(json.dumps(cfg))
        (frames if frames is not None else make_frames()).to_csv(session / "frames.csv", index=False)
        if imu is not False:
            (imu if imu is not None else make_imu()).to_csv(session / "imu.csv", index=False)
        if capture is not None:
            capture.to_csv(session / "capture.csv", index=False)
        return SessionDataLoader(str(session))

    return _make
