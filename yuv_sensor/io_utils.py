"""Small shared I/O helpers used across the CLI and export modules."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, data: Any, indent: int = 2) -> Path:
    """Write data as indented JSON to path, creating/overwriting the file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    return path


def load_t_cam_imu(path: Path) -> np.ndarray:
    """Read a camera-IMU extrinsic from a JSON file holding a 4x4 "T_cam_imu".

    The matrix is Kalibr's T_cam_imu (camchain-imucam.yaml): x_cam = T_cam_imu @ x_imu.
    A bare 4x4 nested list is accepted as well.

    Raises:
        ValueError: If the file has no 4x4 matrix.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = data.get("T_cam_imu") if isinstance(data, dict) else data
    if matrix is None:
        raise ValueError(f"{path}: expected a 4x4 matrix or a JSON object with a \"T_cam_imu\" key")
    t = np.asarray(matrix, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"{path}: T_cam_imu must be 4x4, got shape {t.shape}")
    return t
