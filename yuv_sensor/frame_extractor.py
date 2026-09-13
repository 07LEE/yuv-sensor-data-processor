"""Shared frame-extraction loop used by the CLI and the COLMAP/Kalibr exporters."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from yuv_sensor.data_loader import SessionDataLoader

# Set once per worker process by _init_worker; decoding is CPU/I/O-bound
# (raw YUV read + OpenCV decode/resize/undistort + JPEG/PNG encode), not
# something threads share well across a GIL, so each worker gets its own
# SessionDataLoader instead of one shared across threads.
_worker_loader: Optional[SessionDataLoader] = None


def _init_worker(loader: SessionDataLoader) -> None:
    global _worker_loader
    _worker_loader = loader


def _decode_and_save(
    idx: int,
    save_path: str,
    image_format: str,
    undistort: bool,
    apply_rotation: bool,
    quality: int,
) -> None:
    rgb = _worker_loader.get_decoded_frame(idx, apply_undistort=undistort, apply_rotation=apply_rotation)
    img = Image.fromarray(rgb)
    if image_format == "jpg":
        img.save(save_path, quality=quality)
    else:
        img.save(save_path)


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
    max_workers: Optional[int] = None,
) -> int:
    """Decode and save frames from loader to output_dir, in parallel across processes.

    Each frame's decode (raw YUV read + OpenCV resize/color-convert/undistort)
    and save (JPEG/PNG encode + disk write) is independent of every other
    frame, and dominated by I/O and native (OpenCV/libjpeg) calls rather than
    Python bytecode -- so this parallelizes across a process pool instead of
    a single-threaded loop, one worker per CPU core by default.

    Args:
        loader: SessionDataLoader providing frames_df and get_decoded_frame().
        output_dir: Directory to save decoded images into.
        image_format: "jpg" or "png".
        filename_for_row: Maps (idx, frame_row, image_format) to an output
            filename; defaults to the frame's own filename with its
            extension swapped. Called up front in the main process (not in
            workers), so it may be a closure/lambda.
        undistort: Apply lens distortion correction.
        apply_rotation: Rotate frame upright per sensor_orientation.
        max_frames: Cap on frames extracted, or None for all of them.
        quality: JPEG quality (ignored for png).
        progress_every: Print progress every N frames.
        max_workers: Number of worker processes, or None to use one per CPU
            core (capped at the number of frames).

    Returns:
        Number of frames extracted.
    """
    total = len(loader.frames_df)
    limit = total if max_frames is None else min(max_frames, total)

    save_paths = []
    for idx in range(limit):
        row = loader.frames_df.iloc[idx]
        filename = filename_for_row(idx, row, image_format)
        save_paths.append(str(output_dir / filename))

    if limit == 0:
        return 0

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, limit))

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(loader,),
    ) as executor:
        futures = [
            executor.submit(
                _decode_and_save, idx, save_paths[idx], image_format, undistort, apply_rotation, quality
            )
            for idx in range(limit)
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % progress_every == 0 or completed == limit:
                print(f"  Extracted images: {completed} / {limit}")

    return limit
