"""Allan variance analysis for deriving Kalibr's imu.yaml noise parameters.

Where auto_trim_static_imu (imu_trim.py) prepares the input -- long enough
and free of edge motion -- this module is the analysis step: it turns that
trimmed, static imu.csv into the accelerometer/gyroscope noise_density and
random_walk values kalibr_calibrate_imu_camera needs, so a separate external
Allan variance tool is no longer required for the common case.

Method (the same one tools like kalibr_allan and allan_variance_ros use): the
overlapping Allan deviation curve of a static IMU axis is convex on a log-log
plot -- it falls with slope -1/2 while white sensor noise dominates at short
cluster time tau, bottoms out, then rises with slope +1/2 once bias random
walk dominates at long tau. Reading the -1/2 line's value at tau=1s gives the
noise density; reading the +1/2 line's value at tau=3s gives the random walk
(IEEE-STD-952 convention).
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MAX_CLUSTER_FRACTION = 3  # cap cluster time at N/3 samples so each Allan point still averages several independent clusters
NUM_TAUS = 100             # log-spaced cluster times to evaluate the Allan deviation curve at


def overlapping_allan_deviation(data: np.ndarray, dt: float, num_taus: int = NUM_TAUS) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a uniformly-sampled static signal.

    Args:
        data: 1D array of sensor readings, assumed uniformly sampled at dt.
        dt: Sample period in seconds.
        num_taus: Number of log-spaced cluster times to evaluate.

    Returns:
        Tuple (taus, adev): cluster times in seconds and the Allan deviation
        at each.

    Raises:
        ValueError: Not enough samples to form even the shortest cluster.
    """
    n = len(data)
    theta = np.concatenate(([0.0], np.cumsum(data) * dt))  # integral of the rate signal; theta[0] = 0

    max_m = n // MAX_CLUSTER_FRACTION
    if max_m < 1:
        raise ValueError(f"not enough samples ({n}) for Allan variance analysis")

    ms = np.unique(np.logspace(0, np.log10(max_m), num_taus).astype(np.int64))
    taus = ms * dt

    adev = np.empty(len(ms))
    for i, m in enumerate(ms):
        k = n - 2 * m
        diffs = theta[2 * m:2 * m + k] - 2 * theta[m:m + k] + theta[0:k]
        adev[i] = np.sqrt(np.mean(diffs ** 2) / (2 * (m * dt) ** 2))

    return taus, adev


def _fixed_slope_value_at(log_tau: np.ndarray, log_adev: np.ndarray, idx: int, slope: float, eval_tau: float) -> float:
    """Value at eval_tau of the line with the given fixed slope passing through point idx."""
    intercept = log_adev[idx] - slope * log_tau[idx]
    return float(10 ** (intercept + slope * np.log10(eval_tau)))


def read_noise_and_random_walk(taus: np.ndarray, adev: np.ndarray) -> Tuple[float, float]:
    """Reads the white-noise and bias-random-walk lines off an Allan deviation curve.

    The curve's minimum splits it into the two regions each slope is searched
    in -- slope -1/2 (white noise) before the minimum, slope +1/2 (random
    walk) after it.

    Args:
        taus: Cluster times in seconds, ascending (as returned by
            overlapping_allan_deviation).
        adev: Allan deviation at each tau.

    Returns:
        Tuple (noise_density, random_walk) in the same units as the input
        signal (e.g. m/s^2/sqrt(Hz) for accel, rad/s/sqrt(Hz) for gyro).

    Raises:
        ValueError: The curve has no clear minimum within the analyzed tau
            range, meaning the capture is too short to separate the two
            regimes.
    """
    log_tau = np.log10(taus)
    log_adev = np.log10(adev)
    local_slope = np.gradient(log_adev, log_tau)

    idx_min = int(np.argmin(adev))
    if idx_min < 2 or idx_min > len(taus) - 3:
        raise ValueError(
            "Allan deviation curve has no clear minimum within the analyzed "
            "tau range -- capture is too short to separate white noise from "
            "random walk; use a longer static capture"
        )

    idx_n = 1 + int(np.argmin(np.abs(local_slope[1:idx_min] + 0.5)))
    idx_k = idx_min + 1 + int(np.argmin(np.abs(local_slope[idx_min + 1:-1] - 0.5)))

    noise_density = _fixed_slope_value_at(log_tau, log_adev, idx_n, -0.5, eval_tau=1.0)
    random_walk = _fixed_slope_value_at(log_tau, log_adev, idx_k, 0.5, eval_tau=3.0)

    return noise_density, random_walk


def compute_imu_noise_params(imu_df: pd.DataFrame) -> Dict:
    """Derives Kalibr imu.yaml noise parameters from a static IMU capture.

    Args:
        imu_df: Static (motionless) IMU DataFrame with timestamp_ns, sensor,
            x, y, z columns -- run auto_trim_static_imu on the raw capture
            first so edge motion doesn't corrupt the analysis.

    Returns:
        Dict with accelerometer_noise_density, accelerometer_random_walk,
        accelerometer_rate_hz, gyroscope_noise_density, gyroscope_random_walk,
        gyroscope_rate_hz (each noise value averaged across x/y/z), and
        "detail" with the per-axis breakdown.

    Raises:
        ValueError: imu_df is missing accel or gyro rows, or a capture is too
            short to resolve both Allan deviation regimes on some axis.
    """
    result: Dict = {"detail": {}}

    for sensor, prefix in (("accel", "accelerometer"), ("gyro", "gyroscope")):
        sub = imu_df[imu_df["sensor"] == sensor].sort_values("timestamp_ns")
        if len(sub) == 0:
            raise ValueError(f"imu_df has no {sensor!r} rows -- cannot derive {prefix} noise parameters")

        dt = float(np.median(np.diff(sub["timestamp_ns"].to_numpy()))) / 1e9
        rate_hz = 1.0 / dt

        noise_densities, random_walks = [], []
        for axis in ("x", "y", "z"):
            taus, adev = overlapping_allan_deviation(sub[axis].to_numpy(), dt)
            n, k = read_noise_and_random_walk(taus, adev)
            noise_densities.append(n)
            random_walks.append(k)
            result["detail"][f"{prefix}_{axis}"] = {"noise_density": n, "random_walk": k}

        result[f"{prefix}_noise_density"] = float(np.mean(noise_densities))
        result[f"{prefix}_random_walk"] = float(np.mean(random_walks))
        result[f"{prefix}_rate_hz"] = rate_hz

    return result


def export_imu_yaml(noise_params: Dict, output_path: Path, rostopic: str = "/imu0") -> Path:
    """Writes Kalibr's imu.yaml from noise parameters computed by compute_imu_noise_params.

    Args:
        noise_params: Dict returned by compute_imu_noise_params.
        output_path: Path to write imu.yaml to.
        rostopic: IMU topic name to record in the file (default: /imu0).

    Returns:
        Path: output_path.
    """
    update_rate = round(noise_params["accelerometer_rate_hz"], 2)

    lines = [
        "# Generated by yuv_sensor.allan_variance from a static IMU capture.",
        "# Derived from an overlapping Allan deviation curve, not run through",
        "# Kalibr yet -- sanity-check reprojection/IMU residuals after calibrating.",
        "",
        "#Accelerometers",
        f"accelerometer_noise_density: {noise_params['accelerometer_noise_density']:.6g}  #Noise density (continuous-time)",
        f"accelerometer_random_walk: {noise_params['accelerometer_random_walk']:.6g}   #Bias random walk",
        "",
        "#Gyroscopes",
        f"gyroscope_noise_density: {noise_params['gyroscope_noise_density']:.6g}     #Noise density (continuous-time)",
        f"gyroscope_random_walk: {noise_params['gyroscope_random_walk']:.6g}   #Bias random walk",
        "",
        f"rostopic: {rostopic}",
        f"update_rate: {update_rate}",
        "",
    ]

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))
    return output_path
