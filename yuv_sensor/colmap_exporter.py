"""Exporter for COLMAP-format images, cameras.txt, images.txt and IMU-estimated reference poses."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from yuv_sensor.camera_calib import upright_camera_geometry, upright_camera_rotation
from yuv_sensor.data_loader import SessionDataLoader
from yuv_sensor.frame_extractor import extract_frames
from yuv_sensor.frame_quality import export_quality_report, get_usable_frame_indices
from yuv_sensor.io_utils import write_json
from yuv_sensor.pose_estimator import PoseEstimator, imu_to_camera_trajectory, validate_t_cam_imu


class ColmapExporter:
    """Exports session data in COLMAP format alongside IMU-estimated reference poses.

    cameras.txt, images.txt and pose_priors.json are reference data: the
    documented COLMAP workflow (feature_extractor, sequential_matcher,
    mapper) does not read them, and the mapper estimates poses from the
    images alone. See docs/colmap_workflow.md.

    The IMU trajectory is a body-frame trajectory. images.txt holds camera
    poses, so it is only written when a camera-IMU extrinsic (T_cam_imu) is
    supplied; without one, pose_priors.json carries the IMU poses labelled
    as such and images.txt is not written.
    """

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
        t_cam_imu: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[Path]]:
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
            t_cam_imu: 4x4 camera-IMU extrinsic in the Kalibr convention
                (x_cam = T_cam_imu @ x_imu, raw sensor camera frame with
                OpenCV axes). Required to write images.txt; None writes
                IMU-frame poses to pose_priors.json only.

        Returns:
            Dict mapping file type to output path (cameras.txt, images.txt,
            etc.), plus quality_report_json. images_txt is None when no
            t_cam_imu was given.
        """
        if t_cam_imu is not None:
            t_cam_imu = validate_t_cam_imu(t_cam_imu)

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

        camera_trajectory = None
        if t_cam_imu is not None:
            camera_trajectory = imu_to_camera_trajectory(
                trajectory,
                t_cam_imu,
                upright_camera_rotation(self.loader.session_config.get("sensor_orientation", 0)),
            )

        # Export COLMAP format files
        cameras_txt = self._export_cameras_txt(output_dir)
        images_txt_path = output_dir / "images.txt"
        if camera_trajectory is not None:
            images_txt = self._export_images_txt(output_dir, camera_trajectory, usable_indices)
        else:
            # Never leave an images.txt from an earlier run: it would present
            # IMU poses as camera poses.
            images_txt_path.unlink(missing_ok=True)
            images_txt = None
            print("images.txt: no camera-IMU extrinsics (T_cam_imu) given, so IMU poses "
                  "cannot be written as camera poses; skipping. pose_priors.json holds "
                  "the IMU-frame trajectory.")
        pose_priors = self._export_pose_priors_json(
            output_dir, trajectory, usable_indices, camera_trajectory
        )

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
        metadata = self._export_metadata_json(
            output_dir, undistort, image_format, camera_trajectory is not None
        )

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
        cfg = self.loader.session_config
        intrinsics = cfg.get("intrinsics")
        raw_width = int(self.loader.frames_df.iloc[0]["width"])
        raw_height = int(self.loader.frames_df.iloc[0]["height"])

        cameras_path = output_dir / "cameras.txt"

        with open(cameras_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
            f.write("# Number of cameras: 1\n")

            if intrinsics:
                # images/ holds upright (rotated) frames, so describe that
                # geometry, not the raw sensor frame the intrinsics are in.
                geometry = upright_camera_geometry(
                    intrinsics,
                    cfg.get("distortion"),
                    raw_width,
                    raw_height,
                    sensor_orientation=cfg.get("sensor_orientation", 0),
                    active_array=cfg.get("pre_correction_active_array"),
                )
                f.write(
                    f"1 PINHOLE {geometry.width} {geometry.height} "
                    f"{geometry.fx} {geometry.fy} {geometry.cx} {geometry.cy}\n"
                )
            else:
                # Fallback: assume principal point at center, estimate focal length
                if int(cfg.get("sensor_orientation", 0)) % 180 == 90:
                    width, height = raw_height, raw_width
                else:
                    width, height = raw_width, raw_height
                focal = width
                f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")

        return cameras_path

    def _export_images_txt(self, output_dir: Path, trajectory: Dict, usable_indices: List[int]) -> Path:
        """Export images.txt in COLMAP format.

        Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
        where (QW, QX, QY, QZ, TX, TY, TZ) is the world-to-camera transform
        COLMAP expects: R_cw (from the quaternion) and T = -R_cw @ C, with
        C the camera center in world coordinates. trajectory must be a
        camera trajectory (see imu_to_camera_trajectory) holding the
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

    def _export_pose_priors_json(
        self,
        output_dir: Path,
        trajectory: Dict,
        usable_indices: List[int],
        camera_trajectory: Optional[Dict] = None,
    ) -> Path:
        """Export pose_priors.json with IMU-derived poses for reference only.

        position/quaternion_xyzw/velocity are always the IMU body frame
        (position = IMU origin, quaternion = IMU-to-world). When
        camera_trajectory is given, camera_position (camera center) and
        camera_quaternion_xyzw (camera-to-world, upright image frame) are
        added per frame.
        """
        pose_priors_path = output_dir / "pose_priors.json"

        priors = {
            "source": "IMU-based trajectory estimation",
            "pose_frame": "imu_body",
            "note": "Reference only: dead-reckoned from IMU integration, which drifts. The documented COLMAP workflow does not read this file; the mapper estimates poses from the images alone. position/quaternion_xyzw are the IMU body pose, not the camera pose.",
            "camera_poses_included": camera_trajectory is not None,
            "frames": []
        }
        if camera_trajectory is None:
            priors["note"] += " No camera-IMU extrinsics were supplied, so no camera poses are included."

        usable_set = set(usable_indices)
        for frame_idx in self.loader.frames_df.index:
            if frame_idx not in usable_set:
                continue
            pose = trajectory[frame_idx]
            q = pose["quaternion"]
            entry = {
                "frame_index": int(frame_idx),
                "timestamp_ns": int(pose["timestamp_ns"]),
                "position": pose["position"].tolist(),
                "quaternion_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "velocity": pose["velocity"].tolist(),
            }
            if camera_trajectory is not None:
                cam = camera_trajectory[frame_idx]
                cq = cam["quaternion"]
                entry["camera_position"] = cam["position"].tolist()
                entry["camera_quaternion_xyzw"] = [float(cq[0]), float(cq[1]), float(cq[2]), float(cq[3])]
            priors["frames"].append(entry)

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

    def _export_metadata_json(
        self, output_dir: Path, undistort: bool, image_format: str, camera_poses: bool = False
    ) -> Path:
        """Export metadata about the COLMAP export."""
        metadata_path = output_dir / "colmap_export_metadata.json"

        metadata = {
            "session_id": self.loader.session_path.name,
            "session_config": self.loader.session_config,
            "total_frames": len(self.loader.frames_df),
            "image_format": image_format,
            "undistorted": undistort,
            "poses_source": "IMU trajectory estimation (reference only, not read by the COLMAP commands below)",
            "camera_poses_from_extrinsics": camera_poses,
            "colmap_workflow": [
                "colmap feature_extractor --database_path database.db --image_path images",
                "colmap sequential_matcher --database_path database.db",
                "colmap mapper --database_path database.db --image_path images --output_path sparse"
            ]
        }

        write_json(metadata_path, metadata)

        return metadata_path
