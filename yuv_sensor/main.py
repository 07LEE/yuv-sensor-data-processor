"""Main entry point script for YUV data processing, batch frame extraction, and synchronized dataset generation."""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image

from yuv_sensor.data_loader import SessionDataLoader
from yuv_sensor.data_processor import DataProcessor
from yuv_sensor.colmap_exporter import ColmapExporter
from yuv_sensor.kalibr_exporter import KalibrExporter
from yuv_sensor.imu_trim import auto_trim_static_imu


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
    parser.add_argument("--export_kalibr", action="store_true", default=False, help="Export data as Kalibr camera-IMU calibration input (images + camchain.yaml + imu.csv)")
    parser.add_argument("--kalibr_output_dir", type=str, default=None, help="Directory for Kalibr export (defaults to session_dir/kalibr)")
    parser.add_argument("--trim_imu_static", type=str, choices=["start", "end", "both"], default=None, help="Trim motion off a static IMU-only capture (for Allan variance / Kalibr imu.yaml) instead of any other action; writes trimmed imu.csv, imu_raw.csv, and a trim report")
    parser.add_argument("--imu_trim_output_dir", type=str, default=None, help="Directory for --trim_imu_static output (defaults to session_dir/imu_trimmed)")
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

    # Trim a static IMU-only capture, independent of frame count
    if args.trim_imu_static:
        print(f"\nTrimming static IMU capture (ends={args.trim_imu_static})...")
        if loader.imu_df is None:
            print("No imu.csv in this session.")
            return

        result = auto_trim_static_imu(loader.imu_df, ends=args.trim_imu_static)
        report = result["report"]

        if args.imu_trim_output_dir is None:
            trim_dir = session_path / "imu_trimmed"
        else:
            trim_dir = Path(args.imu_trim_output_dir)
        trim_dir.mkdir(parents=True, exist_ok=True)

        result["trimmed"].to_csv(trim_dir / "imu.csv", index=False)
        loader.imu_df.to_csv(trim_dir / "imu_raw.csv", index=False)
        with open(trim_dir / "trim_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"  baseline |accel|: {report['baseline_mag']:.4f} m/s^2")
        if "start_cut_s" in report:
            print(f"  start: cut {report['start_cut_s']:.0f}s")
        if "end_cut_s" in report:
            print(f"  end: cut {report['end_cut_s']:.0f}s")
        print(f"  kept {report['kept_rows']} rows, dropped {report['dropped_rows']}")
        print(f"\nTrimmed output: {trim_dir}")
        return

    if total_frames == 0:
        print("No frames found in session.")
        return

    # Export as Kalibr camera-IMU calibration input if requested
    if args.export_kalibr:
        print("\nExporting Kalibr camera-IMU calibration input...")
        if args.kalibr_output_dir is None:
            kalibr_dir = session_path / "kalibr"
        else:
            kalibr_dir = Path(args.kalibr_output_dir)

        exporter = KalibrExporter(loader)
        export_result = exporter.export_to_directory(
            kalibr_dir,
            extract_images=True,
            image_format="png",
            max_frames=args.max_frames
        )

        print(f"\nKalibr export complete:")
        print(f"  Output directory: {kalibr_dir}")
        print(f"  camchain.yaml: {export_result['camchain_yaml']}")
        print(f"  imu.csv: {export_result['imu_csv']}")
        print(f"  Images: {export_result['images_dir']}")
        print(f"\nStill needed before kalibr_calibrate_imu_camera can run:")
        print(f"  - a target.yaml for the physical calibration board")
        print(f"  - an imu.yaml (see --trim_imu_static on a separate static capture)")
        print(f"  - packing images/ + imu.csv into a rosbag")
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
