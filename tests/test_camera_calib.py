import numpy as np
import pytest

from yuv_sensor.camera_calib import CameraCalibration, upright_camera_geometry

W, H = 4000, 3000
ACTIVE = [0, 0, 4032, 3024]
K = [3100.0, 3050.0, 2000.0, 1480.0, 0.0]
D = [0.1, -0.05, 0.01, 0.002, -0.003]  # k1, k2, k3, p1, p2


def _project(pts, fx, fy, cx, cy, k1, k2, k3, p1, p2):
    """Pinhole + radtan projection of points with z == 1."""
    x, y = pts[:, 0], pts[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def _rotate_points(pts, orientation):
    """Camera-frame points after the camera is rotated clockwise by orientation."""
    x, y = pts[:, 0], pts[:, 1]
    rotated = {
        0: (x, y),
        90: (-y, x),
        180: (-x, -y),
        270: (y, -x),
    }[orientation]
    return np.stack([rotated[0], rotated[1], pts[:, 2]], axis=1)


def _rotate_pixels(px, orientation):
    x, y = px[:, 0], px[:, 1]
    return np.stack({
        0: (x, y),
        90: (H - y, x),
        180: (W - x, H - y),
        270: (y, W - x),
    }[orientation], axis=1)


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.4, 0.4, (200, 3))
    pts[:, 2] = 1.0
    return pts


@pytest.mark.parametrize("orientation", [0, 90, 180, 270])
def test_projection_matches_rotated_pixels(points, orientation):
    """Projecting with the transformed model equals rotating the raw pixels."""
    sx, sy = W / ACTIVE[2], H / ACTIVE[3]
    raw = _project(points, K[0] * sx, K[1] * sy, K[2] * sx, K[3] * sy, *D)
    expected = _rotate_pixels(raw, orientation)

    g = upright_camera_geometry(K, D, W, H, orientation, ACTIVE)
    got = _project(_rotate_points(points, orientation), g.fx, g.fy, g.cx, g.cy,
                   g.k1, g.k2, g.k3, g.p1, g.p2)

    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("orientation, size", [(0, (W, H)), (90, (H, W)), (180, (W, H)), (270, (H, W))])
def test_dimensions_swap_for_quarter_turns(orientation, size):
    g = upright_camera_geometry(K, D, W, H, orientation, ACTIVE)
    assert (g.width, g.height) == size


def test_scales_intrinsics_to_frame_resolution():
    g = upright_camera_geometry(K, D, 2016, 1512, 0, ACTIVE)  # exactly half the active array
    assert g.fx == pytest.approx(K[0] / 2)
    assert g.fy == pytest.approx(K[1] / 2)
    assert g.cx == pytest.approx(K[2] / 2)
    assert g.cy == pytest.approx(K[3] / 2)


def test_no_active_array_means_no_scaling():
    g = upright_camera_geometry(K, D, W, H, 0, None)
    assert (g.fx, g.fy, g.cx, g.cy) == (K[0], K[1], K[2], K[3])


def test_radial_terms_unchanged_by_rotation():
    g = upright_camera_geometry(K, D, W, H, 90, ACTIVE)
    assert (g.k1, g.k2, g.k3) == (D[0], D[1], D[2])


def test_missing_distortion_defaults_to_zero():
    g = upright_camera_geometry(K, None, W, H, 0, ACTIVE)
    assert (g.k1, g.k2, g.k3, g.p1, g.p2) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_four_quarter_turns_are_identity_for_tangential_terms():
    g = upright_camera_geometry(K, D, W, H, 90, None)
    g2 = upright_camera_geometry(
        [g.fx, g.fy, g.cx, g.cy], [g.k1, g.k2, g.k3, g.p1, g.p2], g.width, g.height, 270, None)
    assert (g2.p1, g2.p2) == pytest.approx((D[3], D[4]))
    assert (g2.fx, g2.fy, g2.cx, g2.cy) == pytest.approx((K[0], K[1], K[2], K[3]))


def test_orientation_wraps_modulo_360():
    assert upright_camera_geometry(K, D, W, H, 360, ACTIVE) == upright_camera_geometry(K, D, W, H, 0, ACTIVE)


def test_invalid_orientation_raises():
    with pytest.raises(ValueError):
        upright_camera_geometry(K, D, W, H, 45, ACTIVE)


@pytest.mark.parametrize("orientation, shape", [(0, (3, 4)), (90, (4, 3)), (180, (3, 4)), (270, (4, 3))])
def test_rotate_frame_shape(orientation, shape):
    cal = CameraCalibration({"sensor_orientation": orientation})
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    assert cal.rotate_frame(image).shape[:2] == shape


def test_rotate_frame_90_is_clockwise():
    cal = CameraCalibration({"sensor_orientation": 90})
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0, 0] = 255  # top-left
    assert cal.rotate_frame(image)[0, 1, 0] == 255  # clockwise: top-left moves to top-right


def test_distortion_reordered_for_opencv():
    cal = CameraCalibration({"distortion": [1.0, 2.0, 3.0, 4.0, 5.0]})  # k1 k2 k3 p1 p2
    assert list(cal.dist_coeffs) == [1.0, 2.0, 4.0, 5.0, 3.0]  # k1 k2 p1 p2 k3
