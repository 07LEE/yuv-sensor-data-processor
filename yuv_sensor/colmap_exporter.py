"""Exporter for COLMAP-compatible input format with camera poses from IMU."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from yuv_sensor.data_loader import SessionDataLoader
from yuv_sensor.frame_extractor import extract_frames
from yuv_sensor.frame_quality import export_quality_report, get_usable_frame_indices
from yuv_sensor.io_utils import write_json
from yuv_sensor.pose_estimator import PoseEstimator


class ColmapExporter:
    """Exports session data in COLMAP-compatible format with IMU-based pose priors."""

    def __init__(self, loader: SessionDataLoader):
        """Initialize ColmapExporter with a SessionDataLoader.

        Args:
            loader: Loaded SessionDataLoader instance.
        """
        self.loader = loader
        self.pose_estimator = PoseEstimator()

    def export_to_directory(
        self,
        output_dir: Path,
        extract_images: bool = True,
        undistort: bool = False,
        image_format: str = "jpg",
        image_quality: int = 92,
        min_sharpness: Optional[float] = None,
        require_converged: bool = False,
    ) -> Dict[str, Path]:
        """Export images and COLMAP format files to directory.

        Args:
            output_dir: Target directory for COLMAP workspace.
            extract_images: Whether to extract images to output_dir/images/
            undistort: Apply lens distortion correction to images.
            image_format: Output image format (jpg or png).
            image_quality: JPEG quality (1-100).
            min_sharpness: Exclude frames with frames.csv `sharpness` below
                this from images/, images.txt and pose_priors.json (COLMAP
                matching suffers on motion-blurred frames). A
                frame_quality_report.json is always written regardless of
                whether this is set, so quality can be checked before
                deciding on a threshold. None (default) excludes nothing.
            require_converged: Also exclude frames whose nearest capture.csv
                row has ae_state/awb_state outside {CONVERGED, LOCKED}
                (exposure/white-balance still settling). Default False.

        Returns:
            Dict mapping file type to output path (cameras.txt, images.txt,
            etc.), plus quality_report_json.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / "images"
        if extract_images:
            images_dir.mkdir(parents=True, exist_ok=True)

        quality_report_path = export_quality_report(
            self.loader,
            output_dir / "frame_quality_report.json",
            min_sharpness=min_sharpness,
            require_converged=require_converged,
        )
        usable_indices = get_usable_frame_indices(
            self.loader, min_sharpness=min_sharpness, require_converged=require_converged
        )

        # Generate poses from IMU -- over every frame, regardless of the
        # quality filter: interframe integration needs the full, unbroken
        # timestamp sequence, and only which frames get *written* below is
        # filtered.
        trajectory = self.pose_estimator.estimate_trajectory(
            self.loader.imu_df,
            self.loader.frames_df
        )

        # Export COLMAP format files
        cameras_txt = self._export_cameras_txt(output_dir)
        images_txt = self._export_images_txt(output_dir, trajectory, usable_indices)
        pose_priors = self._export_pose_priors_json(output_dir, trajectory, usable_indices)

        # Extract images if requested
        if extract_images:
            self._extract_images(
                images_dir,
                trajectory,
                undistort,
                image_format,
                image_quality,
                usable_indices,
            )

        # Export metadata
        metadata = self._export_metadata_json(output_dir, undistort, image_format)

        return {
            "cameras_txt": cameras_txt,
            "images_txt": images_txt,
            "pose_priors_json": pose_priors,
            "metadata_json": metadata,
            "quality_report_json": quality_report_path,
            "images_dir": images_dir if extract_images else None,
        }

    def _export_cameras_txt(self, output_dir: Path) -> Path:
        """Export cameras.txt in COLMAP format.

        Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
        Using PINHOLE model: fx, fy, cx, cy
        """
        intrinsics = self.loader.session_config.get("intrinsics")
        width = self.loader.frames_df.iloc[0]["width"]
        height = self.loader.frames_df.iloc[0]["height"]

        cameras_path = output_dir / "cameras.txt"

        with open(cameras_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
            f.write("# Number of cameras: 1\n")

            if intrinsics:
                fx, fy, cx, cy = intrinsics[:4]
                f.write(f"1 PINHOLE {int(width)} {int(height)} {fx} {fy} {cx} {cy}\n")
            else:
                # Fallback: assume principal point at center, estimate focal length
                focal = width
                f.write(f"1 PINHOLE {int(width)} {int(height)} {focal} {focal} {width/2} {height/2}\n")

        return cameras_path

    def _export_images_txt(self, output_dir: Path, trajectory: Dict, usable_indices: List[int]) -> Path:
        """Export images.txt in COLMAP format.

        Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
        where (QW, QX, QY, QZ, TX, TY, TZ) is the world-to-camera transform
        COLMAP expects: R_cw (from the quaternion) and T = -R_cw @ C, with
        C the camera center in world coordinates. PoseEstimator tracks the
        camera-to-world rotation R_wc, so both must be inverted here.

        Only usable_indices are written -- IMAGE_ID stays frame_idx + 1
        (gaps from excluded frames are fine; COLMAP doesn't require
        contiguous IDs), so IDs still match up with pose_priors.json.
        """
        images_path = output_dir / "images.txt"

        usable_set = set(usable_indices)
        frame_indices = [idx for idx in self.loader.frames_df.index if idx in usable_set]
        filenames = [
            self.loader.frames_df.loc[idx, "filename"].replace(".yuv", ".jpg")
            for idx in frame_indices
        ]

        # Batched, not per-frame: one scipy call converting every frame's
        # world-to-camera rotation to a quaternion instead of one Python-level
        # call per frame (same fix as PoseEstimator's quaternion conversion).
        r_cw_stack = np.stack([trajectory[idx]["rotation_matrix"].T for idx in frame_indices])
        quats = Rotation.from_matrix(r_cw_stack).as_quat()  # (N, 4) xyzw
        positions = np.stack([trajectory[idx]["position"] for idx in frame_indices])
        translations = -np.einsum("nij,nj->ni", r_cw_stack, positions)

        with open(images_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(frame_indices)}\n")

            for i, frame_idx in enumerate(frame_indices):
                image_id = frame_idx + 1  # COLMAP uses 1-based IDs
                qx, qy, qz, qw = quats[i]
                tx, ty, tz = translations[i]

                f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {filenames[i]}\n")
                f.write("# Image features (empty until COLMAP runs)\n")

        return images_path

    def _export_pose_priors_json(self, output_dir: Path, trajectory: Dict, usable_indices: List[int]) -> Path:
        """Export pose_priors.json with IMU-derived poses for reference."""
        pose_priors_path = output_dir / "pose_priors.json"

        priors = {
            "source": "IMU-based trajectory estimation",
            "note": "These are initial pose estimates; COLMAP will refine them via feature matching.",
            "frames": []
        }

        usable_set = set(usable_indices)
        for frame_idx in self.loader.frames_df.index:
            if frame_idx not in usable_set:
                continue
            pose = trajectory[frame_idx]
            q = pose["quaternion"]
            priors["frames"].append({
                "frame_index": int(frame_idx),
                "timestamp_ns": int(pose["timestamp_ns"]),
                "position": pose["position"].tolist(),
                "quaternion_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "velocity": pose["velocity"].tolist(),
            })

        write_json(pose_priors_path, priors)

        return pose_priors_path

    def _extract_images(
        self,
        images_dir: Path,
        trajectory: Dict,
        undistort: bool,
        image_format: str,
        image_quality: int,
        usable_indices: List[int],
    ) -> None:
        """Extract images from YUV frames to images_dir."""
        extract_frames(
            self.loader,
            images_dir,
            image_format,
            undistort=undistort,
            quality=image_quality,
            indices=usable_indices,
        )

    def _export_metadata_json(self, output_dir: Path, undistort: bool, image_format: str) -> Path:
        """Export metadata about the COLMAP export."""
        metadata_path = output_dir / "colmap_export_metadata.json"

        metadata = {
            "session_id": self.loader.session_path.name,
            "session_config": self.loader.session_config,
            "total_frames": len(self.loader.frames_df),
            "image_format": image_format,
            "undistorted": undistort,
            "poses_source": "IMU trajectory estimation (initial priors for COLMAP)",
            "colmap_workflow": [
                "colmap feature_extractor --database_path database.db --image_path images",
                "colmap sequential_matcher --database_path database.db",
                "colmap mapper --database_path database.db --image_path images --output_path sparse"
            ]
        }

        write_json(metadata_path, metadata)

        return metadata_path
