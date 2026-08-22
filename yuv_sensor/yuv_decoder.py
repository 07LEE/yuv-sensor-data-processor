"""YUV_420_888 raw binary image decoder.

Decodes raw binary YUV_420_888 camera frames using stride, layout, and segment metadata from frames.csv.
"""

from typing import Tuple
import numpy as np
import cv2


def decode_yuv420_888(
    yuv_bytes: bytes,
    width: int,
    height: int,
    chroma_layout: str,
    luma_row_stride: int,
    chroma_row_stride: int,
    chroma_pixel_stride: int,
    segment0_length: int = 0,
    segment1_length: int = 0,
    segment2_length: int = 0
) -> np.ndarray:
    """Decodes raw YUV_420_888 binary bytes into an RGB image.

    Args:
        yuv_bytes: Raw binary bytes of the frame.
        width: Image width in pixels.
        height: Image height in pixels.
        chroma_layout: Layout of chroma planes ('planar', 'semi_planar_uv', 'semi_planar_vu').
        luma_row_stride: Row stride (in bytes) for luma plane.
        chroma_row_stride: Row stride (in bytes) for chroma plane.
        chroma_pixel_stride: Pixel stride (in bytes) for chroma plane (1 or 2).
        segment0_length: Byte length of segment 0 (Luma plane).
        segment1_length: Byte length of segment 1 (Chroma plane 1 or interleaved).
        segment2_length: Byte length of segment 2 (Chroma plane 2 if planar).

    Returns:
        np.ndarray: Decoded RGB image of shape (height, width, 3) with dtype uint8.
    """
    # Helper to ensure raw buffer length matches full row_stride * rows
    def _pad_buffer(data: np.ndarray, target_size: int) -> np.ndarray:
        if data.size < target_size:
            padding = np.zeros(target_size - data.size, dtype=np.uint8)
            return np.concatenate([data, padding])
        return data[:target_size]

    # 1. Extract Luma (Y) plane
    y_end = segment0_length if segment0_length > 0 else luma_row_stride * height
    y_raw = np.frombuffer(yuv_bytes[:y_end], dtype=np.uint8)
    y_target_size = luma_row_stride * height
    y_raw_padded = _pad_buffer(y_raw, y_target_size)
    
    # Reshape according to luma_row_stride and slice out padding
    y_plane = y_raw_padded.reshape((height, luma_row_stride))[:, :width]

    # 2. Extract Chroma (U, V) planes
    chroma_bytes = yuv_bytes[y_end:]
    uv_height = height // 2
    uv_width = width // 2

    if chroma_layout == "planar":
        u_end = segment1_length if segment1_length > 0 else chroma_row_stride * uv_height
        u_raw = np.frombuffer(chroma_bytes[:u_end], dtype=np.uint8)
        v_raw = np.frombuffer(chroma_bytes[u_end:], dtype=np.uint8)

        uv_target_size = chroma_row_stride * uv_height
        u_padded = _pad_buffer(u_raw, uv_target_size)
        v_padded = _pad_buffer(v_raw, uv_target_size)

        u_plane = u_padded.reshape((uv_height, chroma_row_stride))[:, :uv_width]
        v_plane = v_padded.reshape((uv_height, chroma_row_stride))[:, :uv_width]

    elif chroma_layout in ("semi_planar_uv", "semi_planar_vu"):
        uv_raw = np.frombuffer(chroma_bytes, dtype=np.uint8)
        uv_stride_width = uv_width * chroma_pixel_stride
        uv_target_size = chroma_row_stride * uv_height
        uv_padded = _pad_buffer(uv_raw, uv_target_size)
        uv_plane = uv_padded.reshape((uv_height, chroma_row_stride))[:, :uv_stride_width]

        if chroma_layout == "semi_planar_uv":
            u_plane = uv_plane[:, 0::chroma_pixel_stride]
            v_plane = uv_plane[:, 1::chroma_pixel_stride]
        else:  # semi_planar_vu
            v_plane = uv_plane[:, 0::chroma_pixel_stride]
            u_plane = uv_plane[:, 1::chroma_pixel_stride]
    else:
        raise ValueError(f"Unsupported chroma layout: {chroma_layout}")

    # 3. Resize U and V to full Luma resolution
    u_resized = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_NEAREST)
    v_resized = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_NEAREST)

    # 4. Merge Y, U, V and convert to RGB
    yuv_merged = cv2.merge([y_plane, u_resized, v_resized])
    rgb = cv2.cvtColor(yuv_merged, cv2.COLOR_YUV2RGB)
    return rgb
