"""Local OpenCV camera intrinsic calibration from a checkerboard capture.

A lightweight alternative to the full Kalibr pipeline (docs/kalibr_workflow.md)
for a quick sanity check: given a session where the phone was moved through
frame in view of a checkerboard, this runs OpenCV's own corner detection and
calibrateCamera directly on the session's frames -- no rosbag, no target.yaml,
no Kalibr install required. It calibrates the camera alone and won't jointly
refine camera-IMU extrinsics the way kalibr_calibrate_imu_camera does, so
treat its output as a fast pre-check against session.json's own intrinsics,
not a replacement for a full Kalibr calibration.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from yuv_sensor.data_loader import SessionDataLoader

MIN_DETECTED_FRAMES = 4  # calibrateCamera needs several views at different angles to be well-posed


def _object_points(checkerboard_size: Tuple[int, int], square_size: float) -> np.ndarray:
    """3D object points for one checkerboard view, board plane at z=0."""
    cols, rows = checkerboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def calibrate_camera_from_checkerboard(
    loader: SessionDataLoader,
    checkerboard_size: Tuple[int, int],
    square_size: float = 1.0,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> Dict:
    """Calibrates camera intrinsics from checkerboard frames using OpenCV.

    Args:
        loader: Loaded SessionDataLoader for a calibration-target capture
            (phone moved through frame in view of a checkerboard).
        checkerboard_size: (cols, rows) of INNER corners on the board -- a
            board with 10x7 squares has a (9, 6) inner-corner grid.
        square_size: Physical side length of one checkerboard square. Only
            scales the (otherwise unused) per-frame translation vectors --
            the intrinsics/distortion this returns are unaffected, so the
            default of 1.0 is fine if you only care about those.
        max_frames: Cap on frames scanned for corners, or None for all.
        frame_stride: Use every Nth frame (default 1 = every frame).
            Adjacent video frames barely change viewpoint, so striding
            through a long capture gets similar angle coverage for a
            fraction of the corner-detection cost.

    Returns:
        Dict with frames_scanned, frames_used, checkerboard_size,
        image_size, rms_reprojection_error_px, intrinsics (fx, fy, cx, cy),
        distortion (k1, k2, p1, p2, k3), per_frame_errors, and -- when
        session.json has its own intrinsics -- session_json_intrinsics plus
        the fx/fy delta against them (frames are decoded without rotation or
        undistortion, matching the orientation session.json's own intrinsics
        are defined in).

    Raises:
        ValueError: Fewer than MIN_DETECTED_FRAMES frames had a detectable
            checkerboard.
    """
    objp = _object_points(checkerboard_size, square_size)

    total = loader.get_frame_count()
    limit = total if max_frames is None else min(max_frames, total)
    indices = list(range(0, limit, frame_stride))

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    used_indices: List[int] = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for idx in indices:
        rgb = loader.get_decoded_frame(idx, apply_undistort=False, apply_rotation=False)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used_indices.append(idx)

    if len(obj_points) < MIN_DETECTED_FRAMES:
        raise ValueError(
            f"checkerboard detected in only {len(obj_points)}/{len(indices)} scanned "
            f"frames -- calibrateCamera needs at least {MIN_DETECTED_FRAMES} views at "
            f"different angles (15-20+ for a reliable fit); wrong --checkerboard_size, "
            f"a target out of frame, or heavy motion blur are the usual causes"
        )

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    per_frame_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        diff = img_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        per_frame_errors.append({"frame_index": used_indices[i], "reprojection_error_px": error})

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    k1, k2, p1, p2, k3 = [float(v) for v in dist_coeffs.flatten()[:5]]

    result = {
        "frames_scanned": len(indices),
        "frames_used": len(obj_points),
        "checkerboard_size": list(checkerboard_size),
        "square_size": square_size,
        "image_size": list(image_size),
        "rms_reprojection_error_px": float(rms_error),
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "distortion": {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3},
        "per_frame_errors": per_frame_errors,
    }

    session_intrinsics = loader.session_config.get("intrinsics")
    if session_intrinsics and len(session_intrinsics) >= 4:
        result["session_json_intrinsics"] = session_intrinsics
        result["fx_delta_from_session_json"] = fx - session_intrinsics[0]
        result["fy_delta_from_session_json"] = fy - session_intrinsics[1]

    return result
