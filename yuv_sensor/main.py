"""Main entry point script for YUV data processing, batch frame extraction, and synchronized dataset generation."""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image

from yuv_sensor.data_loader import SessionDataLoader
from yuv_sensor.data_processor import DataProcessor
from yuv_sensor.colmap_exporter import ColmapExporter


def main():
    parser = argparse.ArgumentParser(description="Process YUV raw sensor data session and extract frames.")
    parser.add_argument("--session_dir", type=str, default="data/session_419864820", help="Path to session directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save extracted frames (defaults to session_dir/extracted_frames_raw or extracted_frames_undistorted)")
    parser.add_argument("--extract_all", action="store_true", default=True, help="Extract all frames in session")
    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("--undistort", action="store_true", default=False, help="Apply lens distortion correction")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
    parser.add_argument("--process_sync", action="store_true", default=True, help="Generate synchronized_dataset.json mapping frames to IMU and exposure data")
    parser.add_argument("--export_colmap", action="store_true", default=False, help="Export data in COLMAP format with IMU-based pose priors")
    parser.add_argument("--colmap_output_dir", type=str, default=None, help="Directory for COLMAP export (defaults to session_dir/colmap)")
    args = parser.parse_args()

    session_path = Path(args.session_dir)
    print(f"Loading session directory: {session_path}")
    loader = SessionDataLoader(str(session_path))
    total_frames = loader.get_frame_count()

    print("\nSession Metadata:")
    print(f"  Device: {loader.session_config.get('device')}")
    print(f"  Intrinsics: {loader.session_config.get('intrinsics')}")
    print(f"  Distortion: {loader.session_config.get('distortion')}")
    print(f"  Sensor Orientation: {loader.session_config.get('sensor_orientation')} deg")
    print(f"  Total Frames in session: {total_frames}")

    if total_frames == 0:
        print("No frames found in session.")
        return

    # Export to COLMAP format if requested
    if args.export_colmap:
        print("\nExporting to COLMAP format with IMU-based pose priors...")
        if args.colmap_output_dir is None:
            colmap_dir = session_path / "colmap"
        else:
            colmap_dir = Path(args.colmap_output_dir)

        exporter = ColmapExporter(loader)
        export_result = exporter.export_to_directory(
            colmap_dir,
            extract_images=True,
            undistort=args.undistort,
            image_format=args.format,
            image_quality=92 if args.format == "jpg" else 100
        )

        print(f"\nCOLMAP export complete:")
        print(f"  Output directory: {colmap_dir}")
        print(f"  Cameras config: {export_result['cameras_txt']}")
        print(f"  Images list: {export_result['images_txt']}")
        print(f"  Pose priors: {export_result['pose_priors_json']}")
        print(f"  Images: {export_result['images_dir']}")
        print(f"\nNext steps:")
        print(f"  cd {colmap_dir}")
        print(f"  colmap feature_extractor --database_path database.db --image_path images")
        print(f"  colmap sequential_matcher --database_path database.db")
        print(f"  colmap mapper --database_path database.db --image_path images --output_path sparse")
        return

    # Process & export synchronized dataset JSON if enabled
    if args.process_sync:
        print("\nGenerating synchronized frame-to-IMU dataset...")
        processor = DataProcessor(loader)
        sync_file = processor.export_synchronized_dataset()
        print(f"  Exported synchronized dataset to: {sync_file}")

    # Set default session-relative output directory if not explicitly specified
    if args.output_dir is None:
        folder_name = "extracted_frames_undistorted" if args.undistort else "extracted_frames_raw"
        out_dir = session_path / folder_name
    else:
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    limit = total_frames if args.max_frames is None else min(args.max_frames, total_frames)

    print(f"\nExtracting {limit} / {total_frames} frames to: {out_dir} (Format: {args.format.upper()}, Undistort: {args.undistort})...")

    for idx in range(limit):
        row = loader.frames_df.iloc[idx]
        timestamp_ns = row["timestamp_ns"]

        rgb_frame = loader.get_decoded_frame(idx, apply_undistort=args.undistort, apply_rotation=True)

        file_name = f"{timestamp_ns}.{args.format}"
        save_path = out_dir / file_name
        img = Image.fromarray(rgb_frame)
        img.save(save_path, quality=92 if args.format == "jpg" else 100)

        if (idx + 1) % 50 == 0 or (idx + 1) == limit:
            print(f"  Progress: {idx + 1} / {limit} frames extracted...")

    # Write extraction_info.json metadata file into the output directory
    info_metadata = {
        "session_dir": str(session_path),
        "undistorted": args.undistort,
        "sensor_orientation": loader.session_config.get("sensor_orientation", 0),
        "format": args.format,
        "extracted_frames_count": limit,
        "total_session_frames": total_frames,
        "intrinsics": loader.session_config.get("intrinsics"),
        "distortion": loader.session_config.get("distortion")
    }

    info_path = out_dir / "extraction_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_metadata, f, indent=2)

    print(f"Saved extraction metadata to: {info_path}")
    print(f"\nSuccessfully extracted {limit} frames into: {out_dir}")


if __name__ == "__main__":
    main()
