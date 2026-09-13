"""Main entry point script for YUV data processing, batch frame extraction, and synchronized dataset generation."""

import sys
import argparse
from pathlib import Path

from yuv_sensor.data_loader import SessionDataLoader
from yuv_sensor.data_processor import DataProcessor
from yuv_sensor.colmap_exporter import ColmapExporter
from yuv_sensor.kalibr_exporter import KalibrExporter
from yuv_sensor.frame_extractor import extract_frames
from yuv_sensor.frame_quality import export_quality_report, get_usable_frame_indices
from yuv_sensor.imu_trim import auto_trim_static_imu
from yuv_sensor.allan_variance import compute_imu_noise_params, export_imu_yaml
from yuv_sensor.checkerboard_calib import calibrate_camera_from_checkerboard
from yuv_sensor.io_utils import write_json


def main():
    parser = argparse.ArgumentParser(description="Process YUV raw sensor data session and extract frames.")
    parser.add_argument("--session_dir", type=str, default="data/session_419864820", help="Path to session directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save extracted frames (defaults to session_dir/extracted_frames_raw or extracted_frames_undistorted)")
    parser.add_argument("--extract_all", action="store_true", default=True, help="Extract all frames in session")
    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("--undistort", action="store_true", default=False, help="Apply lens distortion correction")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
    parser.add_argument("--min_sharpness", type=float, default=None, help="Exclude frames with frames.csv sharpness below this (blurry frames hurt COLMAP/Kalibr matching); a frame_quality_report.json is always written regardless. Default: exclude nothing")
    parser.add_argument("--require_converged", action="store_true", default=False, help="Also exclude frames whose nearest capture.csv row has ae_state/awb_state outside {CONVERGED, LOCKED} (exposure/white-balance still settling)")
    parser.add_argument("--process_sync", action="store_true", default=True, help="Generate synchronized_dataset.json mapping frames to IMU and exposure data (skipped by --export_colmap/--export_kalibr/--trim_imu_static, which are standalone actions)")
    parser.add_argument("--colmap_output_dir", type=str, default=None, help="Directory for COLMAP export (defaults to session_dir/colmap)")
    parser.add_argument("--kalibr_output_dir", type=str, default=None, help="Directory for Kalibr export (defaults to session_dir/kalibr)")
    parser.add_argument("--imu_trim_output_dir", type=str, default=None, help="Directory for --trim_imu_static output (defaults to session_dir/imu_trimmed)")
    parser.add_argument("--checkerboard_size", type=str, default=None, help="Required by --calibrate_camera: inner-corner grid of the checkerboard as COLSxROWS, e.g. 9x6 for a 10x7-square board")
    parser.add_argument("--square_size", type=float, default=1.0, help="Physical checkerboard square side length, in whatever unit you want (only scales the unused per-frame translation vectors; default: 1.0)")
    parser.add_argument("--calib_frame_stride", type=int, default=5, help="Use every Nth frame for --calibrate_camera's corner detection (default: 5)")
    parser.add_argument("--camera_calib_output_dir", type=str, default=None, help="Directory for --calibrate_camera output (defaults to session_dir/camera_calibration)")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--export_colmap", action="store_true", default=False, help="Standalone action: export data in COLMAP format with IMU-based pose priors, skipping --process_sync and normal frame extraction")
    mode_group.add_argument("--export_kalibr", action="store_true", default=False, help="Standalone action: export data as Kalibr camera-IMU calibration input (images + camchain.yaml + imu.csv), skipping --process_sync and normal frame extraction")
    mode_group.add_argument("--trim_imu_static", type=str, choices=["start", "end", "both"], default=None, help="Standalone action: trim motion off a static IMU-only capture (for Allan variance / Kalibr imu.yaml), skipping --process_sync and normal frame extraction; writes trimmed imu.csv, imu_raw.csv, and a trim report")
    mode_group.add_argument("--calibrate_camera", action="store_true", default=False, help="Standalone action: calibrate camera intrinsics from a checkerboard capture using OpenCV directly (no Kalibr/rosbag needed), skipping --process_sync and normal frame extraction; requires --checkerboard_size")

    args = parser.parse_args()

    if args.calibrate_camera:
        if not args.checkerboard_size:
            parser.error("--calibrate_camera requires --checkerboard_size COLSxROWS (e.g. 9x6)")
        try:
            checkerboard_size = tuple(int(v) for v in args.checkerboard_size.lower().split("x"))
            if len(checkerboard_size) != 2:
                raise ValueError
        except ValueError:
            parser.error(f"--checkerboard_size must be COLSxROWS (e.g. 9x6), got {args.checkerboard_size!r}")

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
        print("(standalone action: skipping --process_sync and frame extraction for this run)")
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
        write_json(trim_dir / "trim_report.json", report)

        print(f"  baseline |accel|: {report['baseline_mag']:.4f} m/s^2")
        if "start_cut_s" in report:
            print(f"  start: cut {report['start_cut_s']:.0f}s")
        if "end_cut_s" in report:
            print(f"  end: cut {report['end_cut_s']:.0f}s")
        print(f"  kept {report['kept_rows']} rows, dropped {report['dropped_rows']}")

        print("\nDeriving imu.yaml (Allan variance)...")
        try:
            noise_params = compute_imu_noise_params(result["trimmed"])
            yaml_path = export_imu_yaml(noise_params, trim_dir / "imu.yaml")
            write_json(trim_dir / "imu_noise_report.json", noise_params)
            print(f"  accel noise_density: {noise_params['accelerometer_noise_density']:.6g}  random_walk: {noise_params['accelerometer_random_walk']:.6g}")
            print(f"  gyro  noise_density: {noise_params['gyroscope_noise_density']:.6g}  random_walk: {noise_params['gyroscope_random_walk']:.6g}")
            print(f"  imu.yaml: {yaml_path}")
        except ValueError as e:
            print(f"  Could not derive imu.yaml: {e}")

        print(f"\nTrimmed output: {trim_dir}")
        return

    if total_frames == 0:
        print("No frames found in session.")
        return

    # Calibrate camera intrinsics from a checkerboard capture if requested
    if args.calibrate_camera:
        print("\nCalibrating camera intrinsics from checkerboard frames...")
        print("(standalone action: skipping --process_sync and normal frame extraction for this run)")

        try:
            calib_result = calibrate_camera_from_checkerboard(
                loader,
                checkerboard_size=checkerboard_size,
                square_size=args.square_size,
                max_frames=args.max_frames,
                frame_stride=args.calib_frame_stride,
            )
        except ValueError as e:
            print(f"  Calibration failed: {e}")
            return

        if args.camera_calib_output_dir is None:
            calib_dir = session_path / "camera_calibration"
        else:
            calib_dir = Path(args.camera_calib_output_dir)
        calib_dir.mkdir(parents=True, exist_ok=True)

        report_path = write_json(calib_dir / "checkerboard_calibration_report.json", calib_result)

        intr = calib_result["intrinsics"]
        print(f"  Detected checkerboard in {calib_result['frames_used']}/{calib_result['frames_scanned']} scanned frames")
        print(f"  RMS reprojection error: {calib_result['rms_reprojection_error_px']:.4f} px")
        print(f"  fx={intr['fx']:.2f} fy={intr['fy']:.2f} cx={intr['cx']:.2f} cy={intr['cy']:.2f}")
        if "fx_delta_from_session_json" in calib_result:
            print(f"  Delta from session.json: fx={calib_result['fx_delta_from_session_json']:+.2f} fy={calib_result['fy_delta_from_session_json']:+.2f}")
        print(f"  Report: {report_path}")
        print(f"\nNote: this is a camera-only fit -- for camera-IMU extrinsics, run the full Kalibr workflow (see docs/kalibr_workflow.md)")
        return

    # Export as Kalibr camera-IMU calibration input if requested
    if args.export_kalibr:
        print("\nExporting Kalibr camera-IMU calibration input...")
        print("(standalone action: skipping --process_sync and normal frame extraction for this run)")
        if args.kalibr_output_dir is None:
            kalibr_dir = session_path / "kalibr"
        else:
            kalibr_dir = Path(args.kalibr_output_dir)

        exporter = KalibrExporter(loader)
        export_result = exporter.export_to_directory(
            kalibr_dir,
            extract_images=True,
            image_format="png",
            max_frames=args.max_frames,
            min_sharpness=args.min_sharpness,
            require_converged=args.require_converged,
        )

        print(f"\nKalibr export complete:")
        print(f"  Output directory: {kalibr_dir}")
        print(f"  camchain.yaml: {export_result['camchain_yaml']}")
        print(f"  imu.csv: {export_result['imu_csv']}")
        print(f"  Images: {export_result['images_dir']}")
        print(f"  Frame quality report: {export_result['quality_report_json']}")
        print(f"\nStill needed before kalibr_calibrate_imu_camera can run:")
        print(f"  - a target.yaml for the physical calibration board")
        print(f"  - an imu.yaml (see --trim_imu_static on a separate static capture)")
        print(f"  - packing images/ + imu.csv into a rosbag")
        return

    # Export to COLMAP format if requested
    if args.export_colmap:
        print("\nExporting to COLMAP format with IMU-based pose priors...")
        print("(standalone action: skipping --process_sync and normal frame extraction for this run)")
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
            image_quality=92 if args.format == "jpg" else 100,
            min_sharpness=args.min_sharpness,
            require_converged=args.require_converged,
        )

        print(f"\nCOLMAP export complete:")
        print(f"  Output directory: {colmap_dir}")
        print(f"  Cameras config: {export_result['cameras_txt']}")
        print(f"  Images list: {export_result['images_txt']}")
        print(f"  Pose priors: {export_result['pose_priors_json']}")
        print(f"  Images: {export_result['images_dir']}")
        print(f"  Frame quality report: {export_result['quality_report_json']}")
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

    quality_report_path = export_quality_report(
        loader,
        out_dir / "frame_quality_report.json",
        min_sharpness=args.min_sharpness,
        require_converged=args.require_converged,
        max_frames=args.max_frames,
    )
    usable_indices = get_usable_frame_indices(
        loader,
        min_sharpness=args.min_sharpness,
        require_converged=args.require_converged,
        max_frames=args.max_frames,
    )
    limit = len(usable_indices)

    print(f"\nExtracting {limit} / {total_frames} frames to: {out_dir} (Format: {args.format.upper()}, Undistort: {args.undistort})...")

    extract_frames(
        loader,
        out_dir,
        args.format,
        filename_for_row=lambda idx, row, fmt: f"{int(row['timestamp_ns'])}.{fmt}",
        undistort=args.undistort,
        indices=usable_indices,
    )

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
    write_json(info_path, info_metadata)

    print(f"Saved extraction metadata to: {info_path}")
    print(f"Frame quality report: {quality_report_path}")
    print(f"\nSuccessfully extracted {limit} frames into: {out_dir}")


if __name__ == "__main__":
    main()
