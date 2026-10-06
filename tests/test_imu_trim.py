import numpy as np
import pytest

from tests.conftest import T0, make_imu
from yuv_sensor.imu_trim import auto_trim_static_imu

# Short settle/margin so the synthetic captures stay small.
FAST = dict(threshold=0.07, settle_run_s=3, margin_s=1, max_scan_s=60)


def _with_motion(imu, start_s=None, end_s=None):
    """Adds a large accel deviation to the first start_s / last end_s seconds."""
    imu = imu.copy()
    ts = imu["timestamp_ns"].to_numpy()
    rel = (ts - ts.min()) / 1e9
    total = rel.max()
    mask = np.zeros(len(imu), dtype=bool)
    if start_s:
        mask |= rel < start_s
    if end_s:
        mask |= rel > total - end_s
    mask &= (imu["sensor"] == "accel").to_numpy()
    imu.loc[mask, "x"] = 3.0
    return imu


def test_clean_capture_only_loses_the_margin():
    imu = make_imu(seconds=20)
    result = auto_trim_static_imu(imu, **FAST)
    assert result["report"]["start_cut_s"] == pytest.approx(1.0)
    assert result["report"]["end_cut_s"] == pytest.approx(1.0)
    assert result["report"]["baseline_mag"] == pytest.approx(9.81)


def test_start_motion_is_removed():
    imu = _with_motion(make_imu(seconds=30), start_s=8)
    result = auto_trim_static_imu(imu, ends="start", **FAST)
    trimmed = result["trimmed"]
    first_kept_s = (trimmed["timestamp_ns"].min() - T0) / 1e9
    assert first_kept_s >= 8
    assert result["report"]["dropped_rows"] > 0
    assert "end_cut_s" not in result["report"]


def test_end_motion_is_removed():
    imu = _with_motion(make_imu(seconds=30), end_s=8)
    result = auto_trim_static_imu(imu, ends="end", **FAST)
    assert (result["trimmed"]["timestamp_ns"].max() - T0) / 1e9 <= 30 - 8
    assert "start_cut_s" not in result["report"]


def test_trim_does_not_modify_input():
    imu = _with_motion(make_imu(seconds=30), start_s=8)
    before = imu.copy()
    auto_trim_static_imu(imu, **FAST)
    assert imu.equals(before)


def test_report_row_counts_add_up():
    imu = _with_motion(make_imu(seconds=30), start_s=8, end_s=5)
    result = auto_trim_static_imu(imu, **FAST)
    assert result["report"]["kept_rows"] + result["report"]["dropped_rows"] == len(imu)


def test_capture_too_short_to_confirm_quiet_raises():
    imu = make_imu(seconds=2)  # shorter than the 3 s settle run
    with pytest.raises(ValueError, match="no clean start boundary"):
        auto_trim_static_imu(imu, ends="start", **FAST)


def test_invalid_ends_raises():
    with pytest.raises(ValueError, match="ends must be"):
        auto_trim_static_imu(make_imu(seconds=5), ends="middle")


def test_no_accel_rows_raises():
    imu = make_imu(seconds=5)
    with pytest.raises(ValueError, match="no accel rows"):
        auto_trim_static_imu(imu[imu["sensor"] == "gyro"])
