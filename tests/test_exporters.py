import numpy as np
import pandas as pd
import pytest

from tests.conftest import T0, make_frames, make_imu
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
