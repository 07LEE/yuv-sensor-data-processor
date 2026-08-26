"""Shared frame-extraction loop used by the CLI and the COLMAP/Kalibr exporters."""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image


def _default_filename(idx, row, image_format: str) -> str:
    return row["filename"].replace(".yuv", f".{image_format}")


def extract_frames(
    loader,
    output_dir: Path,
    image_format: str,
    *,
    filename_for_row: Callable = _default_filename,
    undistort: bool = False,
    apply_rotation: bool = True,
    max_frames: Optional[int] = None,
    quality: int = 92,
    progress_every: int = 50,
) -> int:
    """Decode and save frames from loader to output_dir.

    Args:
        loader: SessionDataLoader providing frames_df and get_decoded_frame().
        output_dir: Directory to save decoded images into.
        image_format: "jpg" or "png".
        filename_for_row: Maps (idx, frame_row, image_format) to an output
            filename; defaults to the frame's own filename with its
            extension swapped.
        undistort: Apply lens distortion correction.
        apply_rotation: Rotate frame upright per sensor_orientation.
        max_frames: Cap on frames extracted, or None for all of them.
        quality: JPEG quality (ignored for png).
        progress_every: Print progress every N frames.

    Returns:
        Number of frames extracted.
    """
    total = len(loader.frames_df)
    limit = total if max_frames is None else min(max_frames, total)

    for idx in range(limit):
        row = loader.frames_df.iloc[idx]
        rgb = loader.get_decoded_frame(idx, apply_undistort=undistort, apply_rotation=apply_rotation)

        filename = filename_for_row(idx, row, image_format)
        save_path = output_dir / filename

        img = Image.fromarray(rgb)
        if image_format == "jpg":
            img.save(save_path, quality=quality)
        else:
            img.save(save_path)

        if (idx + 1) % progress_every == 0 or (idx + 1) == limit:
            print(f"  Extracted images: {idx + 1} / {limit}")

    return limit
