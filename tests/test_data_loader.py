import pandas as pd
import pytest

from tests.conftest import T0, make_frames, make_imu
from yuv_sensor.data_loader import TimestampDomainError, validate_timestamp_compatibility


class TestTimestampValidation:
    def test_same_clock_passes(self):
        validate_timestamp_compatibility(make_frames(), make_imu())

    def test_missing_data_is_skipped(self):
        validate_timestamp_compatibility(None, make_imu())
        validate_timestamp_compatibility(make_frames(), None)
        validate_timestamp_compatibility(make_frames(), make_imu().iloc[0:0])

    def test_disjoint_ranges_raise(self):
        with pytest.raises(TimestampDomainError, match="do not overlap"):
            validate_timestamp_compatibility(make_frames(t0=10 ** 18), make_imu(t0=T0))

    def test_frames_far_beyond_imu_raise(self):
        short_imu = make_imu(seconds=1.0)
        late_frames = make_frames(t0=T0 + 3_000_000_000, count=1)  # overlaps only in range terms
        with pytest.raises(TimestampDomainError):
            validate_timestamp_compatibility(late_frames, short_imu)

    def test_small_overrun_within_tolerance_passes(self):
        frames = make_frames(count=50)  # last frame ~5 s in, IMU is 5.5 s long
        validate_timestamp_compatibility(frames, make_imu(seconds=5.5))

    def test_is_a_value_error(self):
        assert issubclass(TimestampDomainError, ValueError)


class TestSessionDataLoader:
    def test_get_synchronized_imu_rejects_incompatible_clocks(self, make_session):
        loader = make_session(frames=make_frames(t0=10 ** 18))
        with pytest.raises(TimestampDomainError):
            loader.get_synchronized_imu(int(loader.frames_df["timestamp_ns"].iloc[0]))

    def test_validation_runs_once(self, make_session, monkeypatch):
        loader = make_session()
        calls = []
        monkeypatch.setattr("yuv_sensor.data_loader.validate_timestamp_compatibility",
                            lambda *a, **k: calls.append(1))
        for _ in range(3):
            loader.validate_timestamps()
        assert len(calls) == 1

    def test_imu_range_is_inclusive_and_per_sensor(self, make_session):
        loader = make_session()
        lo, hi = T0 + 10_000_000, T0 + 20_000_000  # 200 Hz -> samples at +10, +15, +20 ms
        result = loader.get_imu_in_range(lo, hi)
        assert list(result["accel"]["timestamp_ns"]) == [lo, lo + 5_000_000, hi]
        assert list(result["gyro"]["timestamp_ns"]) == [lo, lo + 5_000_000, hi]

    def test_synchronized_imu_window_is_centered(self, make_session):
        loader = make_session()
        center = T0 + 500_000_000
        result = loader.get_synchronized_imu(center, time_window_ms=10.0)
        ts = result["accel"]["timestamp_ns"]
        assert ts.min() == center - 10_000_000
        assert ts.max() == center + 10_000_000

    def test_no_imu_returns_empty_frames(self, make_session):
        loader = make_session(imu=False)
        result = loader.get_imu_in_range(0, 10 ** 18)
        assert result["accel"].empty and result["gyro"].empty

    def test_nearest_capture_metadata(self, make_session):
        capture = pd.DataFrame({"timestamp_ns": [T0, T0 + 100, T0 + 1000], "exposure_ns": [1, 2, 3]})
        loader = make_session(capture=capture)
        assert loader.get_nearest_capture_metadata(T0 + 90)["exposure_ns"] == 2
        assert loader.get_nearest_capture_metadata(T0 + 900)["exposure_ns"] == 3
        assert loader.get_nearest_capture_metadata(T0 - 50)["exposure_ns"] == 1

    def test_nearest_capture_metadata_without_capture_csv(self, make_session):
        assert make_session().get_nearest_capture_metadata(T0) is None
