"""Camera calibration and frame rotation utilities."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Tuple
import numpy as np
import cv2


@dataclass(frozen=True)
class CameraGeometry:
    """Pinhole intrinsics and radtan distortion for one specific image size."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float


def upright_camera_geometry(
    intrinsics: Sequence[float],
    distortion: Optional[Sequence[float]],
    frame_width: int,
    frame_height: int,
    sensor_orientation: int = 0,
    active_array: Optional[Sequence[float]] = None,
) -> CameraGeometry:
    """Describes session.json's camera for the upright frames the exporters write.

    session.json's intrinsics are in pre_correction_active_array pixels, and
    the raw frame is in sensor orientation. Exported images are the raw
    frame rotated clockwise by sensor_orientation, so the intrinsics have
    to be scaled to the frame resolution, then rotated the same way.

    Rotation maps a point (x, y) of a W x H frame to (H - y, x) for 90,
    (W - x, H - y) for 180 and (y, W - x) for 270 (continuous pixel
    coordinates, the same convention the input intrinsics use). Radial terms
    k1-k3 depend only on the distance from the principal point, which
    rotates with the image, so they are unchanged. The tangential terms
    (p1, p2) mix the two axes: (p2, -p1) for 90, (-p1, -p2) for 180 and
    (-p2, p1) for 270. Skew is ignored, as in the exporters.

    Args:
        intrinsics: [fx, fy, cx, cy, ...] from session.json.
        distortion: [k1, k2, k3, p1, p2] from session.json, or None for none.
        frame_width: Raw (un-rotated) frame width in pixels.
        frame_height: Raw (un-rotated) frame height in pixels.
        sensor_orientation: Clockwise rotation, in degrees, that makes the
            raw frame upright: 0, 90, 180 or 270.
        active_array: pre_correction_active_array [left, top, width, height]
            the intrinsics are expressed in, or None if they are already in
            frame pixels.

    Returns:
        CameraGeometry for the rotated image.

    Raises:
        ValueError: If sensor_orientation is not 0, 90, 180 or 270.
    """
    fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
    d = list(distortion or [])
    k1, k2, k3, p1, p2 = (float(d[i]) if len(d) > i else 0.0 for i in range(5))

    w, h = int(frame_width), int(frame_height)
    if active_array:
        sx = w / float(active_array[2])
        sy = h / float(active_array[3])
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy

    orientation = int(sensor_orientation) % 360
    if orientation == 0:
        return CameraGeometry(w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    if orientation == 90:
        return CameraGeometry(h, w, fy, fx, h - cy, cx, k1, k2, k3, p2, -p1)
    if orientation == 180:
        return CameraGeometry(w, h, fx, fy, w - cx, h - cy, k1, k2, k3, -p1, -p2)
    if orientation == 270:
        return CameraGeometry(h, w, fy, fx, cy, w - cx, k1, k2, k3, -p2, p1)
    raise ValueError(f"sensor_orientation must be 0, 90, 180 or 270, got {sensor_orientation}")


class CameraCalibration:
    """Class to manage camera intrinsic parameters, distortion correction, and orientation rotation."""

    def __init__(self, session_config: Dict[str, Any]):
        """Initialize camera calibration parameters from session.json configuration.

        Args:
            session_config: Dictionary parsed from session.json.
        """
        self.sensor_orientation = session_config.get("sensor_orientation", 0)
        
        intrinsics = session_config.get("intrinsics", [1.0, 1.0, 0.0, 0.0, 0.0])
        fx, fy, cx, cy, skew = intrinsics
        self.camera_matrix = np.array([
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        distortion = session_config.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
        # session.json distortion order: [k1, k2, k3, p1, p2]
        # OpenCV distCoeffs order: [k1, k2, p1, p2, k3]
        if len(distortion) >= 5:
            k1, k2, k3, p1, p2 = distortion[:5]
            self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        else:
            self.dist_coeffs = np.array(distortion, dtype=np.float64)

        self.pre_correction_active_array = session_config.get("pre_correction_active_array", None)

    def rotate_frame(self, image: np.ndarray) -> np.ndarray:
        """Rotates image clockwise according to sensor_orientation to make it upright.

        Args:
            image: Input RGB image.

        Returns:
            np.ndarray: Upright rotated image.
        """
        if self.sensor_orientation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.sensor_orientation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.sensor_orientation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def undistort_frame(self, image: np.ndarray) -> np.ndarray:
        """Applies lens distortion correction to image.

        Args:
            image: Input RGB image.

        Returns:
            np.ndarray: Lens distortion corrected RGB image.
        """
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
