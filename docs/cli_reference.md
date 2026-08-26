# CLI Reference

```bash
yuv-sensor --session_dir <path> [options]

Options:
  --session_dir STR              Path to session directory (required)
  --output_dir STR               Output directory for extracted frames (default: session_dir/extracted_frames_raw)
  --format {jpg,png}             Output image format (default: jpg)
  --undistort                    Apply lens distortion correction
  --max_frames N                 Extract only first N frames
  --process_sync                 Generate synchronized_dataset.json (default: True; skipped by
                                  --export_colmap/--export_kalibr/--trim_imu_static, which are
                                  standalone actions)
  --export_colmap                Standalone action: export in COLMAP format with IMU pose priors
  --colmap_output_dir STR        Output directory for COLMAP workspace (default: session_dir/colmap)
  --export_kalibr                Standalone action: export Kalibr camera-IMU calibration input
                                  (images + camchain.yaml + imu.csv)
  --kalibr_output_dir STR        Output directory for Kalibr export (default: session_dir/kalibr)
  --trim_imu_static {start,end,both}
                                  Standalone action: trim motion off a static IMU-only capture
                                  (for Allan variance / Kalibr imu.yaml)
  --imu_trim_output_dir STR      Output directory for --trim_imu_static (default: session_dir/imu_trimmed)
```

`--export_colmap`, `--export_kalibr`, and `--trim_imu_static` are mutually exclusive standalone
actions — each performs its own export/trim and exits without running `--process_sync` or normal
frame extraction. See [COLMAP Workflow](colmap_workflow.md) and [Kalibr Workflow](kalibr_workflow.md)
for end-to-end examples.
