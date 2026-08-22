"""Main entry point script for YUV data processing, batch frame extraction, and verification."""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import SessionDataLoader


def main():
    parser = argparse.ArgumentParser(description="Process YUV raw sensor data session and extract frames.")
    parser.add_argument("--session_dir", type=str, default="data/session_419864820", help="Path to session directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save extracted frames (defaults to session_dir/extracted_frames_raw or extracted_frames_undistorted)")
    parser.add_argument("--extract_all", action="store_true", default=True, help="Extract all frames in session")
    parser.add_argument("--format", type=str, choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("--undistort", action="store_true", default=False, help="Apply lens distortion correction")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract")
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
