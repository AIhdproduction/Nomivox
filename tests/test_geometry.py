import numpy as np
import pytest

from e57cls.geometry.cameras import Camera, Pose, View, plan_views, scan_to_camera
from e57cls.geometry.images import bilinear, render_view
from e57cls.geometry.reference import project_point
from e57cls.geometry.visibility import DepthBuffer


def pose(w=1, x=0, y=0, z=0, tx=0, ty=0, tz=0):
    return {"rotation": dict(w=w, x=x, y=y, z=z), "translation": dict(x=tx, y=ty, z=tz)}


def spherical():
    return Camera("sphericalRepresentation", dict(imageWidth=360, imageHeight=180,
                                                 pixelWidth=np.pi/180, pixelHeight=np.pi/180))


def test_cardinal_axes_and_wrap():
    uv, depth, valid = spherical().project([[1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(uv, [[180, 90], [90, 90], [270, 90], [0, 90], [180, 0]])
    assert valid.all()
    np.testing.assert_allclose(depth, 1)


@pytest.mark.parametrize("kind,params", [
    ("sphericalRepresentation", dict(imageWidth=360, imageHeight=180, pixelWidth=np.pi/180, pixelHeight=np.pi/180)),
    ("cylindricalRepresentation", dict(imageWidth=360, imageHeight=180, pixelWidth=np.pi/180, pixelHeight=.01, radius=1., principalPointY=90.)),
    ("pinholeRepresentation", dict(imageWidth=400, imageHeight=300, pixelWidth=.001, pixelHeight=.001, focalLength=.1, principalPointX=200., principalPointY=150.))])
def test_independent_reference_and_camera_roundtrip(kind, params):
    camera = Camera(kind, params)
    pixels = np.array([[20.5, 30.5], [130.5, 90.5], [250.5, 140.5]])
    rays = camera.rays(pixels)
    actual, _, valid = camera.project(rays*3.7)
    np.testing.assert_allclose(actual, pixels, atol=1e-10)
    assert valid.all()
    s = pose(w=np.cos(.3), z=np.sin(.3), tx=2700000, ty=1200000, tz=400)
    c = pose(w=np.cos(.7), z=np.sin(.7), tx=2700000.13, ty=1200000.01, tz=400.03)
    scan, image = Pose.from_dict(s), Pose.from_dict(c)
    world = image.to_world(rays*3.7)
    local = scan.from_world(world)
    production, _, _ = camera.project(scan_to_camera(local, scan, image))
    reference = [project_point(p, s, c, kind, params) for p in local]
    np.testing.assert_allclose(production, reference, atol=1e-6)


def test_large_origin_camera_offset_is_applied_once():
    s, c = Pose.from_dict(pose(tx=2700000, ty=1200000)), Pose.from_dict(pose(tx=2700000.125, ty=1200000))
    np.testing.assert_allclose(scan_to_camera([[1, 0, 0]], s, c), [[.875, 0, 0]])


def test_view_projection_uses_axis_depth_not_radial():
    view = View("front", 0, 0, 90, 100)
    uv, depth, valid = view.project([[2, -1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(uv[0], [75, 50])
    assert depth[0] == 2
    assert np.linalg.norm([2, -1, 0]) > depth[0]
    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(view.rays(uv[:1])[0], np.array([2, -1, 0])/np.sqrt(5))
    assert len(plan_views(spherical())) == 17
    assert len(plan_views(spherical(), profile="fast")) == 9


def test_visibility_rejects_background_and_uses_max_tolerance():
    buffer = DepthBuffer(5, 5, radius=0)
    uv = np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]])
    depths = np.array([1., 2., 1.035])
    buffer.add(uv, depths, np.ones(3, bool))
    assert buffer.visible(uv, depths, np.ones(3, bool)).tolist() == [True, False, False]


def test_resampling_wrap_and_mask():
    image = np.zeros((180, 360, 3), np.uint8)
    image[:, :, 0] = 200
    image[:, 0, 1] = 100
    image[:, -1, 1] = 200
    assert bilinear(image, np.array([[0, 50.5]]), True)[0, 1] == 150
    view = View("front", 0, 0, 90, 32)
    rendered, valid = render_view(image, np.zeros((180, 360), np.uint8), spherical(), view)
    assert not rendered.any() and not valid.any()
    rendered, valid = render_view(image, np.full((180, 360), 255, np.uint8), spherical(), view)
    assert valid.all() and (rendered[:, :, 0] == 200).all()
