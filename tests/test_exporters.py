import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from tests.conftest import T0, make_frames, make_imu
from yuv_sensor.camera_calib import upright_camera_rotation
from yuv_sensor.colmap_exporter import ColmapExporter
from yuv_sensor.data_loader import TimestampDomainError
from yuv_sensor.kalibr_exporter import KalibrExporter


def _camchain_values(text):
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("intrinsics:", "distortion_coeffs:", "resolution:")):
            key, rest = stripped.split(":", 1)
            values[key] = [float(v) for v in rest.split("#")[0].strip().strip("[]").split(",")]
    return values


def _cameras_line(path):
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == 1
    return lines[0].split()


class TestKalibrCamchain:
    def test_unrotated_session_scales_to_frame_resolution(self, make_session, tmp_path):
        loader = make_session()  # 4000x3000 frames, 4032x3024 active array
        values = _camchain_values(KalibrExporter(loader)._export_camchain_yaml(tmp_path).read_text())

        sx, sy = 4000 / 4032, 3000 / 3024
        assert values["resolution"] == [4000, 3000]
        assert values["intrinsics"] == pytest.approx([3100 * sx, 3050 * sy, 2000 * sx, 1480 * sy])

    def test_rotated_session_describes_upright_images(self, make_session, tmp_path):
        loader = make_session(config={"sensor_orientation": 90})
        values = _camchain_values(KalibrExporter(loader)._export_camchain_yaml(tmp_path).read_text())

        sx, sy = 4000 / 4032, 3000 / 3024
        assert values["resolution"] == [3000, 4000]
        assert values["intrinsics"] == pytest.approx([3050 * sy, 3100 * sx, 3000 - 1480 * sy, 2000 * sx])
        # p1, p2 -> p2, -p1 for a 90 degree rotation
        assert values["distortion_coeffs"] == pytest.approx([0.1, -0.05, -0.003, -0.002])

    def test_k3_caveat_is_written(self, make_session, tmp_path):
        loader = make_session()
        text = KalibrExporter(loader)._export_camchain_yaml(tmp_path).read_text()
        assert "k3=0.01" in text

    def test_missing_intrinsics_skips(self, make_session, tmp_path):
        loader = make_session()
        del loader.session_config["intrinsics"]
        assert KalibrExporter(loader)._export_camchain_yaml(tmp_path) is None


class TestKalibrImuCsv:
    def test_one_row_per_gyro_sample_with_interpolated_accel(self, make_session, tmp_path):
        ts = T0 + np.arange(5) * 10_000_000
        imu = pd.concat([
            pd.DataFrame({"timestamp_ns": ts[::2], "sensor": "accel", "x": [0.0, 2.0, 4.0], "y": 0.0, "z": 9.81}),
            pd.DataFrame({"timestamp_ns": ts, "sensor": "gyro", "x": 0.1, "y": 0.2, "z": 0.3}),
        ], ignore_index=True)
        loader = make_session(frames=make_frames(count=1, period_ns=10_000_000), imu=imu)

        out = pd.read_csv(KalibrExporter(loader)._export_imu_csv(tmp_path))

        assert list(out.columns) == ["timestamp", "omega_x", "omega_y", "omega_z",
                                     "alpha_x", "alpha_y", "alpha_z"]
        assert len(out) == 5
        np.testing.assert_allclose(out["alpha_x"], [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(out["omega_z"], 0.3)

    def test_gyro_outside_accel_range_is_dropped(self, make_session, tmp_path):
        ts = T0 + np.arange(5) * 10_000_000
        imu = pd.concat([
            pd.DataFrame({"timestamp_ns": ts[1:4], "sensor": "accel", "x": 1.0, "y": 0.0, "z": 9.81}),
            pd.DataFrame({"timestamp_ns": ts, "sensor": "gyro", "x": 0.0, "y": 0.0, "z": 0.0}),
        ], ignore_index=True)
        loader = make_session(frames=make_frames(count=1, period_ns=10_000_000), imu=imu)

        out = pd.read_csv(KalibrExporter(loader)._export_imu_csv(tmp_path))
        assert list(out["timestamp"]) == list(ts[1:4])

    def test_no_imu_skips(self, make_session, tmp_path):
        assert KalibrExporter(make_session(imu=False))._export_imu_csv(tmp_path) is None

    def test_incompatible_clocks_raise(self, make_session, tmp_path):
        loader = make_session(frames=make_frames(t0=10 ** 18))
        with pytest.raises(TimestampDomainError):
            KalibrExporter(loader)._export_imu_csv(tmp_path)


class TestColmapCameras:
    def test_unrotated_session(self, make_session, tmp_path):
        parts = _cameras_line(ColmapExporter(make_session())._export_cameras_txt(tmp_path))
        sx, sy = 4000 / 4032, 3000 / 3024
        assert parts[:4] == ["1", "PINHOLE", "4000", "3000"]
        assert [float(p) for p in parts[4:]] == pytest.approx([3100 * sx, 3050 * sy, 2000 * sx, 1480 * sy])

    @pytest.mark.parametrize("orientation, size", [(90, ("3000", "4000")), (270, ("3000", "4000")), (180, ("4000", "3000"))])
    def test_rotated_session_swaps_dimensions(self, make_session, tmp_path, orientation, size):
        loader = make_session(config={"sensor_orientation": orientation})
        parts = _cameras_line(ColmapExporter(loader)._export_cameras_txt(tmp_path))
        assert tuple(parts[2:4]) == size

    def test_90_degree_rotation_transforms_principal_point(self, make_session, tmp_path):
        loader = make_session(config={"sensor_orientation": 90})
        parts = _cameras_line(ColmapExporter(loader)._export_cameras_txt(tmp_path))
        sx, sy = 4000 / 4032, 3000 / 3024
        fx, fy, cx, cy = (float(p) for p in parts[4:])
        assert (fx, fy) == pytest.approx((3050 * sy, 3100 * sx))
        assert (cx, cy) == pytest.approx((3000 - 1480 * sy, 2000 * sx))

    def test_fallback_without_intrinsics_uses_upright_size(self, make_session, tmp_path):
        loader = make_session(config={"sensor_orientation": 90})
        del loader.session_config["intrinsics"]
        parts = _cameras_line(ColmapExporter(loader)._export_cameras_txt(tmp_path))
        assert tuple(parts[2:4]) == ("3000", "4000")
        assert float(parts[6]) == pytest.approx(1500.0)  # cx at image center
        assert float(parts[7]) == pytest.approx(2000.0)


class TestColmapPoses:
    @staticmethod
    def _export(loader, tmp_path, **kwargs):
        return ColmapExporter(loader).export_to_directory(tmp_path, extract_images=False, **kwargs)

    def test_without_extrinsics_images_txt_is_not_written(self, make_session, tmp_path):
        result = self._export(make_session(), tmp_path)
        assert result["images_txt"] is None
        assert not (tmp_path / "images.txt").exists()

    def test_stale_images_txt_is_removed(self, make_session, tmp_path):
        (tmp_path / "images.txt").write_text("stale")
        self._export(make_session(), tmp_path)
        assert not (tmp_path / "images.txt").exists()

    def test_without_extrinsics_priors_are_labelled_imu(self, make_session, tmp_path):
        import json
        result = self._export(make_session(), tmp_path)
        priors = json.loads(result["pose_priors_json"].read_text())
        assert priors["pose_frame"] == "imu_body"
        assert priors["camera_poses_included"] is False
        assert "camera_position" not in priors["frames"][0]

    def test_with_extrinsics_images_txt_holds_camera_poses(self, make_session, tmp_path):
        import json
        t = np.eye(4)
        t[:3, 3] = [0.0, 0.0, -0.2]  # IMU origin sits 0.2 m behind the camera along its z
        result = self._export(make_session(), tmp_path, t_cam_imu=t)

        rows = [l.split() for l in result["images_txt"].read_text().splitlines()
                if l and not l.startswith("#")]
        assert len(rows) == 10
        # stationary device, identity rotation: T = -R_cw @ C = -C, C = p + t_ic = (0, 0, 0.2)
        np.testing.assert_allclose([float(v) for v in rows[0][5:8]], [0.0, 0.0, -0.2], atol=1e-5)

        priors = json.loads(result["pose_priors_json"].read_text())
        assert priors["camera_poses_included"] is True
        np.testing.assert_allclose(priors["frames"][0]["camera_position"], [0.0, 0.0, 0.2], atol=1e-5)

    @pytest.mark.parametrize("image_format", ["jpg", "png"])
    def test_images_txt_names_follow_image_format(self, make_session, tmp_path, image_format):
        loader = make_session()
        result = self._export(loader, tmp_path, t_cam_imu=np.eye(4), image_format=image_format)

        rows = [l.split() for l in result["images_txt"].read_text().splitlines()
                if l and not l.startswith("#")]
        expected = [name.replace(".yuv", f".{image_format}") for name in loader.frames_df["filename"]]
        assert [r[9] for r in rows] == expected

    @staticmethod
    def _image_rows(result):
        return np.array([
            [float(v) for v in l.split()[1:8]]
            for l in result["images_txt"].read_text().splitlines()
            if l and not l.startswith("#")
        ])

    @pytest.mark.parametrize("orientation", [0, 90, 180, 270])
    def test_upright_frame_extrinsics_match_converted_raw(self, make_session, tmp_path, orientation):
        loader = make_session(config={"sensor_orientation": orientation})
        t_up = np.eye(4)
        t_up[:3, :3] = Rotation.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
        t_up[:3, 3] = [0.03, -0.01, 0.07]
        t_raw_up = np.eye(4)
        t_raw_up[:3, :3] = upright_camera_rotation(orientation)

        raw = self._export(loader, tmp_path / "raw", t_cam_imu=t_raw_up @ t_up)
        upright = self._export(loader, tmp_path / "upright", t_cam_imu=t_up, t_cam_imu_frame="upright")
        misread = self._export(loader, tmp_path / "misread", t_cam_imu=t_up)

        np.testing.assert_allclose(self._image_rows(upright), self._image_rows(raw), atol=1e-5)
        if orientation:
            # the same matrix read as raw is rotated a second time
            assert not np.allclose(self._image_rows(misread), self._image_rows(upright), atol=1e-3)

    @pytest.mark.parametrize("image_format", ["jpg", "png"])
    def test_images_txt_names_match_extracted_files(self, make_session, tmp_path, image_format):
        frames = make_frames(count=3, width=4, height=4).assign(
            chroma_layout="planar", luma_row_stride=4, chroma_row_stride=2, chroma_pixel_stride=1,
            segment0_length=16, segment1_length=4, segment2_length=4,
        )
        loader = make_session(frames=frames)
        frames_dir = loader.session_path / "frames"
        frames_dir.mkdir()
        for name in frames["filename"]:
            (frames_dir / name).write_bytes(bytes(24))  # 4x4 planar YUV: 16 + 4 + 4 bytes

        result = ColmapExporter(loader).export_to_directory(
            tmp_path / "out", extract_images=True, image_format=image_format, t_cam_imu=np.eye(4)
        )

        rows = [l.split() for l in result["images_txt"].read_text().splitlines()
                if l and not l.startswith("#")]
        names = [r[9] for r in rows]
        assert len(names) == 3
        assert all((result["images_dir"] / name).is_file() for name in names)
        assert sorted(p.name for p in result["images_dir"].iterdir()) == sorted(names)

    def test_unknown_extrinsics_frame_raises(self, make_session, tmp_path):
        with pytest.raises(ValueError, match="t_cam_imu_frame"):
            self._export(make_session(), tmp_path, t_cam_imu=np.eye(4), t_cam_imu_frame="camera")

    def test_invalid_extrinsics_raise(self, make_session, tmp_path):
        with pytest.raises(ValueError):
            self._export(make_session(), tmp_path, t_cam_imu=np.eye(3))
