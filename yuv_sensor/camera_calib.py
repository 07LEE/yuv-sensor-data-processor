"""Camera calibration and frame rotation utilities."""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


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
