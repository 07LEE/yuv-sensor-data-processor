"""Frame quality assessment for filtering blurry / not-yet-converged frames
before COLMAP/Kalibr export.

frames.csv already carries a per-frame `sharpness` score and capture.csv
carries Android Camera2 CONTROL_AE_STATE / CONTROL_AWB_STATE per capture --
both recorded at capture time but never read anywhere in the pipeline until
now. colmap_workflow.md's own troubleshooting section names exactly these
two failure modes ("Low image quality or motion blur", "Fixed focus,
exposure; auto-focus can hurt matching") as something to check by hand;
this turns that manual check into a report, and optionally a filter.

Camera2 state values (https://developer.android.com/reference/android/hardware/camera2/CaptureResult):
  CONTROL_AE_STATE:  0=INACTIVE 1=SEARCHING 2=CONVERGED 3=LOCKED 4=FLASH_REQUIRED 5=PRECAPTURE
  CONTROL_AWB_STATE: 0=INACTIVE 1=SEARCHING 2=CONVERGED 3=LOCKED
"An unconverged frame" below means neither CONVERGED nor LOCKED.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from yuv_sensor.io_utils import write_json

AE_STATE_CONVERGED = 2
AE_STATE_LOCKED = 3
AWB_STATE_CONVERGED = 2
AWB_STATE_LOCKED = 3
_CONVERGED_STATES = {AE_STATE_CONVERGED, AE_STATE_LOCKED}  # same two codes for AE and AWB

# A filter this aggressive risks leaving COLMAP too few overlapping views to
# reconstruct -- worth a loud warning rather than silently exporting a
# thinned-out dataset. This is a relative, not absolute, threshold: how many
# frames COLMAP actually needs depends on overlap and scene complexity, so we
# don't pretend to know a hard minimum count.
HIGH_EXCLUSION_WARNING_FRACTION = 0.5


def _clean_state(value: Any) -> Optional[int]:
    """None if value is missing or NaN (column absent or blank cell), else int(value)."""
    if value is None or value != value:
        return None
    return int(value)


def assess_frame_quality(
    loader,
    min_sharpness: Optional[float] = None,
    require_converged: bool = False,
    max_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Per-frame quality signals, plus a `usable` verdict under the given filters.

    Args:
        loader: Loaded SessionDataLoader.
        min_sharpness: Frames with frames.csv `sharpness` below this are
            flagged `blurry`. None disables the sharpness check.
        require_converged: If True, frames whose nearest capture.csv row
            has ae_state/awb_state outside {CONVERGED, LOCKED} are flagged
            `unconverged`. A frame with no capture.csv match, or missing
            ae_state/awb_state columns, is never flagged this way -- the
            signal being unavailable is not evidence of a problem.
        max_frames: Only assess the first N frames, or None for all.

    Returns:
        List of per-frame dicts (frame_index, timestamp_ns, sharpness,
        ae_state, awb_state, af_state, blurry, unconverged, usable).
        `usable` is True whenever neither active filter flags the frame
        (both filters default off, so by default every frame is usable).
    """
    total = len(loader.frames_df)
    limit = total if max_frames is None else min(max_frames, total)

    results: List[Dict[str, Any]] = []
    for idx in range(limit):
        row = loader.frames_df.iloc[idx]
        timestamp_ns = int(row["timestamp_ns"])
        sharpness = float(row["sharpness"]) if "sharpness" in row and row["sharpness"] == row["sharpness"] else None

        ae_state = awb_state = af_state = None
        capture_row = loader.get_nearest_capture_metadata(timestamp_ns)
        if capture_row is not None:
            ae_state = _clean_state(capture_row.get("ae_state"))
            awb_state = _clean_state(capture_row.get("awb_state"))
            af_state = _clean_state(capture_row.get("af_state"))

        blurry = None if (min_sharpness is None or sharpness is None) else sharpness < min_sharpness
        unconverged = None
        if require_converged and ae_state is not None and awb_state is not None:
            unconverged = ae_state not in _CONVERGED_STATES or awb_state not in _CONVERGED_STATES

        usable = not bool(blurry) and not bool(unconverged)

        results.append({
            "frame_index": idx,
            "timestamp_ns": timestamp_ns,
            "sharpness": sharpness,
            "ae_state": ae_state,
            "awb_state": awb_state,
            "af_state": af_state,
            "blurry": blurry,
            "unconverged": unconverged,
            "usable": usable,
        })

    return results


def get_usable_frame_indices(
    loader,
    min_sharpness: Optional[float] = None,
    require_converged: bool = False,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Frame indices left after applying the given quality filters.

    With both filters left at their default (off), this returns every
    frame index in range -- i.e. filtering is opt-in, never silently on.
    """
    quality = assess_frame_quality(loader, min_sharpness, require_converged, max_frames)
    return [entry["frame_index"] for entry in quality if entry["usable"]]


def build_quality_report(
    loader,
    min_sharpness: Optional[float] = None,
    require_converged: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Full report: per-frame quality entries plus a summary of what was flagged.

    Includes a `warning` string (else None) when an active filter
    (min_sharpness and/or require_converged) excludes at least
    HIGH_EXCLUSION_WARNING_FRACTION of frames -- filtering that aggressive
    can leave too little overlap for COLMAP to reconstruct from.
    """
    frames = assess_frame_quality(loader, min_sharpness, require_converged, max_frames)
    total_frames = len(frames)
    blurry_count = sum(1 for f in frames if f["blurry"])
    unconverged_count = sum(1 for f in frames if f["unconverged"])
    usable_count = sum(1 for f in frames if f["usable"])
    excluded_count = total_frames - usable_count

    warning = None
    filter_active = min_sharpness is not None or require_converged
    if filter_active and total_frames > 0 and excluded_count / total_frames >= HIGH_EXCLUSION_WARNING_FRACTION:
        warning = (
            f"{excluded_count}/{total_frames} frames ({excluded_count / total_frames:.0%}) excluded by "
            f"the current quality filter, leaving only {usable_count} usable. COLMAP needs enough "
            f"overlapping views to reconstruct -- if this looks too aggressive, loosen --min_sharpness "
            f"or drop --require_converged and check frame_quality_report.json's per-frame entries."
        )

    return {
        "min_sharpness_threshold": min_sharpness,
        "require_converged": require_converged,
        "total_frames": total_frames,
        "blurry_count": blurry_count,
        "unconverged_count": unconverged_count,
        "usable_count": usable_count,
        "excluded_count": excluded_count,
        "warning": warning,
        "frames": frames,
    }


def export_quality_report(
    loader,
    output_path: Path,
    min_sharpness: Optional[float] = None,
    require_converged: bool = False,
    max_frames: Optional[int] = None,
) -> Path:
    """Writes build_quality_report()'s result as JSON to output_path, printing its warning (if any)."""
    report = build_quality_report(loader, min_sharpness, require_converged, max_frames)
    if report["warning"]:
        print(f"WARNING: {report['warning']}")
    return write_json(output_path, report)
