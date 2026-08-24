"""Exporter for COLMAP-compatible input format with camera poses from IMU."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from yuv_sensor.data_loader import SessionDataLoader
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
        image_quality: int = 92
    ) -> Dict[str, Path]:
        """Export images and COLMAP format files to directory.

        Args:
            output_dir: Target directory for COLMAP workspace.
            extract_images: Whether to extract images to output_dir/images/
            undistort: Apply lens distortion correction to images.
            image_format: Output image format (jpg or png).
            image_quality: JPEG quality (1-100).

        Returns:
            Dict mapping file type to output path (cameras.txt, images.txt, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = output_dir / "images"
        if extract_images:
            images_dir.mkdir(parents=True, exist_ok=True)

        # Generate poses from IMU
        trajectory = self.pose_estimator.estimate_trajectory(
            self.loader.imu_df,
            self.loader.frames_df
        )

        # Export COLMAP format files
        cameras_txt = self._export_cameras_txt(output_dir)
        images_txt = self._export_images_txt(output_dir, trajectory)
        pose_priors = self._export_pose_priors_json(output_dir, trajectory)

        # Extract images if requested
        if extract_images:
            self._extract_images(
                images_dir,
                trajectory,
                undistort,
                image_format,
                image_quality
            )

        # Export metadata
        metadata = self._export_metadata_json(output_dir, undistort, image_format)

        return {
            "cameras_txt": cameras_txt,
            "images_txt": images_txt,
            "pose_priors_json": pose_priors,
            "metadata_json": metadata,
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
                fx, fy, cx, cy = intrinsics
                f.write(f"1 PINHOLE {int(width)} {int(height)} {fx} {fy} {cx} {cy}\n")
            else:
                # Fallback: assume principal point at center, estimate focal length
                focal = width
                f.write(f"1 PINHOLE {int(width)} {int(height)} {focal} {focal} {width/2} {height/2}\n")

        return cameras_path

    def _export_images_txt(self, output_dir: Path, trajectory: Dict) -> Path:
        """Export images.txt in COLMAP format.

        Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
        where (qw, qx, qy, qz) is rotation quaternion in xyzw format.
        """
        images_path = output_dir / "images.txt"

        with open(images_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.loader.frames_df)}\n")

            for frame_idx, frame_row in self.loader.frames_df.iterrows():
                image_id = frame_idx + 1  # COLMAP uses 1-based IDs
                timestamp_ns = int(frame_row["timestamp_ns"])
                filename = frame_row["filename"].replace(".yuv", ".jpg")

                pose = trajectory[frame_idx]
                q = pose["quaternion"]  # xyzw format
                qx, qy, qz, qw = q[0], q[1], q[2], q[3]
                tx, ty, tz = pose["position"]

                f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {filename}\n")
                f.write("# Image features (empty until COLMAP runs)\n")

        return images_path

    def _export_pose_priors_json(self, output_dir: Path, trajectory: Dict) -> Path:
        """Export pose_priors.json with IMU-derived poses for reference."""
        pose_priors_path = output_dir / "pose_priors.json"

        priors = {
            "source": "IMU-based trajectory estimation",
            "note": "These are initial pose estimates; COLMAP will refine them via feature matching.",
            "frames": []
        }

        for frame_idx, frame_row in self.loader.frames_df.iterrows():
            pose = trajectory[frame_idx]
            q = pose["quaternion"]
            priors["frames"].append({
                "frame_index": int(frame_idx),
                "timestamp_ns": int(pose["timestamp_ns"]),
                "position": pose["position"].tolist(),
                "quaternion_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "velocity": pose["velocity"].tolist(),
            })

        with open(pose_priors_path, "w") as f:
            json.dump(priors, f, indent=2)

        return pose_priors_path

    def _extract_images(
        self,
        images_dir: Path,
        trajectory: Dict,
        undistort: bool,
        image_format: str,
        image_quality: int
    ) -> None:
        """Extract images from YUV frames to images_dir."""
        from PIL import Image

        total = len(self.loader.frames_df)
        for idx, frame_row in self.loader.frames_df.iterrows():
            rgb = self.loader.get_decoded_frame(idx, apply_undistort=undistort, apply_rotation=True)

            filename = frame_row["filename"].replace(".yuv", f".{image_format}")
            save_path = images_dir / filename

            img = Image.fromarray(rgb)
            if image_format == "jpg":
                img.save(save_path, quality=image_quality)
            else:
                img.save(save_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"  Extracted images: {idx + 1} / {total}")

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

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path
