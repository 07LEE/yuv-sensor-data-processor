import pandas as pd

from tests.conftest import T0, make_frames
from yuv_sensor.frame_quality import assess_frame_quality, get_usable_frame_indices


def _session(make_session, sharpness, ae_states=None, awb_states=None):
    frames = make_frames(count=len(sharpness))
    frames["sharpness"] = sharpness
    capture = None
    if ae_states is not None:
        capture = pd.DataFrame({
            "timestamp_ns": frames["timestamp_ns"],
            "ae_state": ae_states,
            "awb_state": awb_states if awb_states is not None else ae_states,
            "af_state": 0,
        })
    return make_session(frames=frames, capture=capture)


def test_filters_are_off_by_default(make_session):
    loader = _session(make_session, [1.0, 2.0, 3.0])
    assert get_usable_frame_indices(loader) == [0, 1, 2]


def test_min_sharpness_flags_blurry_frames(make_session):
    loader = _session(make_session, [100.0, 5.0, 80.0])
    results = assess_frame_quality(loader, min_sharpness=50.0)
    assert [r["blurry"] for r in results] == [False, True, False]
    assert get_usable_frame_indices(loader, min_sharpness=50.0) == [0, 2]


def test_require_converged_flags_searching_states(make_session):
    loader = _session(make_session, [100.0] * 4, ae_states=[2, 1, 3, 2], awb_states=[2, 2, 3, 1])
    results = assess_frame_quality(loader, require_converged=True)
    assert [r["unconverged"] for r in results] == [False, True, False, True]
    assert get_usable_frame_indices(loader, require_converged=True) == [0, 2]


def test_missing_capture_is_not_flagged_unconverged(make_session):
    loader = _session(make_session, [100.0, 100.0])  # no capture.csv
    results = assess_frame_quality(loader, require_converged=True)
    assert all(r["unconverged"] is None and r["usable"] for r in results)


def test_both_filters_must_pass(make_session):
    loader = _session(make_session, [10.0, 100.0, 100.0], ae_states=[2, 1, 2])
    assert get_usable_frame_indices(loader, min_sharpness=50.0, require_converged=True) == [2]


def test_max_frames_limits_assessment(make_session):
    loader = _session(make_session, [1.0] * 5)
    assert len(assess_frame_quality(loader, max_frames=2)) == 2
