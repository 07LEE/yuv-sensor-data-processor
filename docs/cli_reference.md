# CLI Reference

```bash
yuv-sensor --session_dir <path> [options]

Options:
  --session_dir STR              Path to session directory (required)
  --output_dir STR               Output directory for extracted frames
                                  (default: session_dir/extracted_frames_raw)
  --format {jpg,png}             Output image format (default: jpg)
  --undistort                    Apply lens distortion correction
  --max_frames N                 Extract only first N frames
  --min_sharpness FLOAT          Exclude frames with frames.csv sharpness below this from
                                  extraction/--export_colmap/--export_kalibr (a frame_quality_report.json
                                  is always written regardless). Default: exclude nothing
  --require_converged            Also exclude frames whose nearest capture.csv row has ae_state/awb_state
                                  outside {CONVERGED, LOCKED} (exposure/white-balance still settling)
  --process_sync                 Generate synchronized_dataset.json (default: True; skipped by
                                  --export_colmap/--export_kalibr/--trim_imu_static, which are
                                  standalone actions)
  --export_colmap                Standalone action: export in COLMAP format with IMU pose priors
  --colmap_output_dir STR        Output directory for COLMAP workspace (default: session_dir/colmap)
  --export_kalibr                Standalone action: export Kalibr camera-IMU calibration input
                                  (images + camchain.yaml + imu.csv)
  --kalibr_output_dir STR        Output directory for Kalibr export (default: session_dir/kalibr)
  --trim_imu_static {start,end,both}
                                  Standalone action: trim motion off a static IMU-only capture and
                                  derive Kalibr imu.yaml from it via Allan variance
  --imu_trim_output_dir STR      Output directory for --trim_imu_static
                                  (default: session_dir/imu_trimmed)
  --calibrate_camera             Standalone action: calibrate camera intrinsics from a checkerboard
                                  capture using OpenCV directly (no Kalibr/rosbag needed); requires
                                  --checkerboard_size
  --checkerboard_size COLSxROWS  Inner-corner grid of the checkerboard, e.g. 9x6 for a 10x7-square
                                  board (required by --calibrate_camera)
  --square_size FLOAT            Physical checkerboard square side length, any unit (default: 1.0;
                                  only scales unused per-frame translation vectors)
  --calib_frame_stride N         Use every Nth frame during --calibrate_camera corner detection
                                  (default: 5)
  --camera_calib_output_dir STR  Output directory for --calibrate_camera
                                  (default: session_dir/camera_calibration)
```

`--export_colmap`, `--export_kalibr`, `--trim_imu_static`, and `--calibrate_camera` are mutually
exclusive standalone actions — each performs its own export/trim/calibration and exits without
running `--process_sync` or normal frame extraction. See [COLMAP Workflow](colmap_workflow.md) and
[Kalibr Workflow](kalibr_workflow.md) for end-to-end examples.
