# Capture Data Session Specification and Processing Directive

## 1. Overview and Data Summary

This session contains raw sensor data captured via mobile_sensor_logger for 3D Reconstruction, Visual SLAM, and Gaussian Splatting applications. Instead of compressed video or JPEG files, uncompressed raw YUV binary files and dense sensor logs are preserved to support sensor fusion and accurate 3D reconstruction.

## 2. Session Data Directory Structure and File Details

1. frames/
Per-frame uncompressed raw YUV_420_888 binary files named after their nanosecond timestamp (timestamp_ns.yuv). Pixel data (Luma and Chroma) is written sequentially without image headers or encoding.

2. frames.csv
Memory layout and stride metadata required for decoding each .yuv binary file.
Columns: timestamp_ns, filename, width, height, sharpness, chroma_layout, luma_row_stride, chroma_row_stride, chroma_pixel_stride, segment0_length, segment1_length, segment2_length

3. session.json
Device specifications and camera physical and optical calibration parameters.
Key parameters: sensor_orientation (clockwise rotation angle for upright display), intrinsics [fx, fy, cx, cy, skew], distortion [k1, k2, k3, p1, p2], pre_correction_active_array.

4. capture.csv
Hardware sensor exposure and control metadata.
Columns: timestamp_ns, exposure_ns, sensitivity, focus_diopters, rolling_shutter_skew_ns, ae_state, awb_state, af_state, physical_id

5. imu.csv
Nanosecond timestamp-synchronized 200Hz accelerometer and gyroscope logs.
Columns: timestamp_ns, sensor, x, y, z (where sensor is accel or gyro)

6. synchronized_dataset.json (Generated via DataProcessor)
Pre-processed dataset mapping each frame timestamp to matched exposure settings and inter-frame IMU samples.

## 3. Mandatory Processing Directives

- YUV Decoding: Convert to RGB by reflecting chroma_layout (planar, semi_planar_uv, semi_planar_vu) and row strides (luma_row_stride, chroma_row_stride) with padding bytes recorded in frames.csv.
- Image Rotation and Undistortion: Apply sensor_orientation rotation angle from session.json and perform lens distortion correction using intrinsics and distortion parameters.
- Timestamp Synchronization: Use nanosecond timestamp (timestamp_ns) as the primary key when joining camera frames, IMU, and exposure metadata.
