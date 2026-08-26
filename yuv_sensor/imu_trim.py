"""Trims motion-contaminated edges off a static IMU capture.

For deriving Kalibr's imu.yaml (noise_density / random_walk) via Allan
variance, the input needs to be genuinely motionless -- handling the phone to
start or stop the recording leaves several seconds to tens of seconds of real
motion at either edge, which corrupts that analysis if left in.

Finds the cut point by scanning accelerometer magnitude deviation from the
session's own median (proxy for gravity) in 1-second bins, walking in from
each requested end until SETTLE_RUN_S consecutive seconds all stay under
threshold, then adding a safety margin past that point. This is the same
approach used by hand to trim a ~5.4 hour static capture, where a flat
"remove N seconds" guess undershot twice before this method found the real
boundary (motion had settled by t-59s; a naive 10s trim left the file dirty).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

SETTLE_RUN_S = 30         # consecutive clean seconds required before trusting a boundary
DEFAULT_THRESHOLD = 0.07  # m/s^2 above this is treated as real motion, not sensor noise
DEFAULT_MARGIN_S = 15     # extra cushion added past the confirmed-clean point
MAX_SCAN_S = 1800         # give up looking for a clean boundary past this


def _find_boundary_ns(
    accel: pd.DataFrame,
    from_start: bool,
    baseline: float,
    threshold: float,
    settle_run_s: float,
    margin_s: float,
    max_scan_s: float
) -> Optional[int]:
    """Nanoseconds to cut from the scanned end, or None if never found."""
    ts = accel["timestamp_ns"].to_numpy()
    mag = np.sqrt(accel["x"] ** 2 + accel["y"] ** 2 + accel["z"] ** 2).to_numpy()
    dev = np.abs(mag - baseline)

    rel = (ts - ts[0]) if from_start else (ts[-1] - ts)
    max_scan_ns = int(max_scan_s * 1e9)
    keep = rel <= max_scan_ns
    rel, dev = rel[keep], dev[keep]
    if len(rel) == 0:
        return None

    sec = (rel // 1_000_000_000).astype(np.int64)
    max_sec = int(sec.max())
    bin_max = np.zeros(max_sec + 1)
    np.maximum.at(bin_max, sec, dev)
    bin_count = np.zeros(max_sec + 1, dtype=np.int64)
    np.add.at(bin_count, sec, 1)
    bin_has_data = bin_count > 0

    settle_run = int(settle_run_s)
    for start in range(0, max_sec - settle_run + 2):
        window = slice(start, start + settle_run)
        # A second with no samples (sensor dropout) must not count as clean --
        # bin_max defaults to 0.0 there, which would otherwise look spotless.
        if np.all(bin_has_data[window]) and np.all(bin_max[window] <= threshold):
            return int((start + margin_s) * 1e9)
    return None


def auto_trim_static_imu(
    imu_df: pd.DataFrame,
    ends: str = "both",
    threshold: float = DEFAULT_THRESHOLD,
    settle_run_s: float = SETTLE_RUN_S,
    margin_s: float = DEFAULT_MARGIN_S,
    max_scan_s: float = MAX_SCAN_S
) -> Dict:
    """Trims motion off the start and/or end of a static IMU capture.

    Args:
        imu_df: IMU DataFrame with timestamp_ns, sensor, x, y, z columns
            (accel + gyro interleaved, as loaded by SessionDataLoader).
        ends: Which end(s) to scan and trim -- "start", "end", or "both".
        threshold: m/s^2 accel deviation from baseline treated as motion.
        settle_run_s: Consecutive clean seconds required to trust a boundary.
        margin_s: Extra cushion added past the confirmed-clean point.
        max_scan_s: Give up looking for a clean boundary past this many
            seconds from the requested end.

    Returns:
        Dict with "trimmed" (the cut DataFrame) and "report" (baseline,
        cut points found, and kept/dropped row counts).

    Raises:
        ValueError: no accel rows in imu_df, or no clean boundary found
            within max_scan_s on a requested end.
    """
    if ends not in ("start", "end", "both"):
        raise ValueError(f"ends must be 'start', 'end', or 'both', got {ends!r}")

    accel = imu_df[imu_df["sensor"] == "accel"].sort_values("timestamp_ns")
    if len(accel) == 0:
        raise ValueError("no accel rows in imu_df")

    baseline = float(np.median(np.sqrt(accel["x"] ** 2 + accel["y"] ** 2 + accel["z"] ** 2)))
    report = {"baseline_mag": baseline}

    lo_ns = int(imu_df["timestamp_ns"].min())
    hi_ns = int(imu_df["timestamp_ns"].max())

    if ends in ("start", "both"):
        cut = _find_boundary_ns(accel, True, baseline, threshold, settle_run_s, margin_s, max_scan_s)
        if cut is None:
            raise ValueError(f"no clean start boundary found within {max_scan_s}s")
        lo_ns += cut
        report["start_cut_s"] = cut / 1e9

    if ends in ("end", "both"):
        cut = _find_boundary_ns(accel, False, baseline, threshold, settle_run_s, margin_s, max_scan_s)
        if cut is None:
            raise ValueError(f"no clean end boundary found within {max_scan_s}s")
        hi_ns -= cut
        report["end_cut_s"] = cut / 1e9

    trimmed = imu_df[(imu_df["timestamp_ns"] >= lo_ns) & (imu_df["timestamp_ns"] <= hi_ns)].copy()
    report["kept_rows"] = len(trimmed)
    report["dropped_rows"] = len(imu_df) - len(trimmed)

    return {"trimmed": trimmed, "report": report}
