"""Exporter for Kalibr camera-IMU calibration input (images + camchain.yaml + imu.csv)."""

import json
from pathlib import Path
from typing import Dict, Optional

from yuv_sensor.data_loader import SessionDataLoader


class KalibrExporter:
    """Exports session data as input for Kalibr's camera-IMU calibration.

    Unlike ColmapExporter, images are extracted WITHOUT undistortion: Kalibr
    fits its own distortion model from the raw (distorted) frames, and an
    already-undistorted image would get run through that fit a second time.
    """

    def __init__(self, loader: SessionDataLoader):
        """Initialize KalibrExporter with a SessionDataLoader.

        Args:
            loader: Loaded SessionDataLoader instance.
        """
        self.loader = loader

    def export_to_directory(
        self,
        output_dir: Path,
        extract_images: bool = True,
        image_format: str = "png",
        max_frames: Optional[int] = None
    ) -> Dict[str, Optional[Path]]:
        """Export images, camchain.yaml, and imu.csv to a Kalibr input directory.

        Args:
            output_dir: Target directory for the Kalibr input set.
            extract_images: Whether to extract images to output_dir/images/
            image_format: Output image format (jpg or png). png avoids
                re-compressing frames Kalibr will run corner detection on.
            max_frames: Cap on frames extracted, or None for all of them.

        Returns:
            Dict mapping output name to path (camchain_yaml, imu_csv,
            images_dir, metadata_json). A value is None if that part was
            skipped (e.g. no imu.csv on this session).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        camchain_path = self._export_camchain_yaml(output_dir)
        imu_path = self._export_imu_csv(output_dir)

        images_dir = None
        if extract_images:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._extract_images(images_dir, image_format, max_frames)

        metadata_path = self._export_metadata_json(output_dir, image_format)

        return {
            "camchain_yaml": camchain_path,
            "imu_csv": imu_path,
            "images_dir": images_dir,
            "metadata_json": metadata_path,
        }

    def _export_camchain_yaml(self, output_dir: Path) -> Optional[Path]:
        """Export camchain.yaml seeded from session.json's own intrinsics.

        Reformats what session.json already measured; doesn't re-derive
        anything from the images. Kalibr's radtan model is 4-coefficient
        while session.json carries 5 (with k3), so k3 is dropped -- flagged
        in a comment in the written file rather than silently discarded.
        """
        cfg = self.loader.session_config
        intrinsics = cfg.get("intrinsics")
        distortion = cfg.get("distortion")
        active_array = cfg.get("pre_correction_active_array")
        if not intrinsics or not distortion or not active_array:
            print("camchain.yaml: session.json missing intrinsics/distortion/"
                  "pre_correction_active_array, skipping")
            return None

        fx, fy, cx, cy = intrinsics[:4]
        skew = intrinsics[4] if len(intrinsics) > 4 else 0.0
        k1 = distortion[0] if len(distortion) > 0 else 0.0
        k2 = distortion[1] if len(distortion) > 1 else 0.0
        k3 = distortion[2] if len(distortion) > 2 else 0.0
        p1 = distortion[3] if len(distortion) > 3 else 0.0
        p2 = distortion[4] if len(distortion) > 4 else 0.0
        _, _, width, height = active_array
        fov = cfg.get("fov_deg")

        lines = [
            f"# Seed intrinsics for Kalibr, taken directly from "
            f"{self.loader.session_path.name}/session.json",
            "# (manufacturer-measured, not re-derived from checkerboard frames).",
            "# Not run through Kalibr yet -- sanity-check before use.",
        ]
        if k3:
            lines += [
                "#",
                f"# CAVEAT: session.json's 5-coefficient radtan distortion has k3={k3}",
                "# which Kalibr's 4-param `radtan` model has no slot for, so it's",
                "# dropped below -- an approximation, not the full fit.",
            ]
        if skew:
            lines += [
                "#",
                f"# CAVEAT: session.json reports nonzero skew ({skew}), which",
                "# pinhole/radtan has no term for either. Dropped below.",
            ]
        if fov and max(fov) > 90:
            lines += [
                "#",
                f"# CAVEAT: {fov[0]:.1f} x {fov[1]:.1f} deg FOV is wide enough that",
                "# pinhole+radtan may not fit well -- Kalibr's `equidistant` (fisheye)",
                "# model is worth trying too and comparing reprojection error, rather",
                "# than assuming pinhole is right.",
            ]

        lines += [
            "",
            "cam0:",
            "  camera_model: pinhole",
            f"  intrinsics: [{fx}, {fy}, {cx}, {cy}]   # fx, fy, cx, cy",
            "  distortion_model: radtan",
            f"  distortion_coeffs: [{k1}, {k2}, {p1}, {p2}]  # k1, k2, p1, p2",
            f"  resolution: [{int(width)}, {int(height)}]",
            "  rostopic: /cam0/image_raw",
            "",
        ]

        path = output_dir / "camchain.yaml"
        path.write_text("\n".join(lines))
        return path

    def _export_imu_csv(self, output_dir: Path) -> Optional[Path]:
        """Copy this session's imu.csv into the Kalibr input directory as-is."""
        if self.loader.imu_df is None:
            print("imu.csv: session has no imu.csv, skipping")
            return None
        path = output_dir / "imu.csv"
        self.loader.imu_df.to_csv(path, index=False)
        return path

    def _extract_images(self, images_dir: Path, image_format: str, max_frames: Optional[int]) -> None:
        """Extract raw (non-undistorted) images from YUV frames to images_dir."""
        from PIL import Image

        total = len(self.loader.frames_df)
        limit = total if max_frames is None else min(max_frames, total)

        for idx in range(limit):
            row = self.loader.frames_df.iloc[idx]
            rgb = self.loader.get_decoded_frame(idx, apply_undistort=False, apply_rotation=True)

            filename = row["filename"].replace(".yuv", f".{image_format}")
            save_path = images_dir / filename
            Image.fromarray(rgb).save(save_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == limit:
                print(f"  Extracted images: {idx + 1} / {limit}")

    def _export_metadata_json(self, output_dir: Path, image_format: str) -> Path:
        """Export metadata about the Kalibr export."""
        metadata_path = output_dir / "kalibr_export_metadata.json"

        metadata = {
            "session_id": self.loader.session_path.name,
            "session_config": self.loader.session_config,
            "total_frames": len(self.loader.frames_df),
            "image_format": image_format,
            "undistorted": False,
            "note": "Images are intentionally NOT undistorted -- Kalibr fits its own "
                    "distortion model from the raw frames.",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path
