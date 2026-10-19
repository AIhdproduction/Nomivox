import numpy as np
import pytest

from e57cls.geometry.cameras import Camera, Pose, View
from e57cls.slam.cache import VisibilityCache, visibility_key
from e57cls.slam.index import build_spatial_index_from_arrays
from e57cls.pipeline.run import _slam_candidate_loader
from e57cls.slam.visibility import (_torch_depth, origin_depth_width, visibility_for_points,
                                    visibility_for_candidates, visibility_for_view, visibility_from_origin)


def _config(**changes):
    return {"slam_max_range": 20., "slam_max_range_indoor": 20., "slam_max_range_outdoor": 40.,
            "slam_depth_resolution": 256, "slam_splat_max_px": 6, "slam_depth_tol_abs": .02,
            "slam_depth_tol_rel": .01, "gpu_device": "cpu", **changes}


def _occluder_scene():
    # A dense box face at x=2 blocks the corresponding wall patch at x=5.
    # Its sampling is denser than the six-pixel maximum footprint so the test
    # models a closed foreground surface under the footprint-edge semantics.
    values = np.linspace(-.7, .7, 25)
    y, z = np.meshgrid(values, values)
    box = np.column_stack((np.full(y.size, 2.), y.ravel(), z.ravel()))
    wall = np.column_stack((np.full(y.size, 5.), y.ravel(), z.ravel()))
    return np.vstack((box, wall)), np.arange(len(box), len(box)+len(wall), dtype=np.int64)


def _radial_layers(count):
    """Return two spherical layers on identical rays for rule-d tests."""
    index = np.arange(count, dtype=np.float64)
    z = 1.-2.*(index+.5)/count
    radius = np.sqrt(np.maximum(0., 1.-z*z))
    angle = index*np.pi*(3.-np.sqrt(5.))
    rays = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))
    return np.vstack((rays*2., rays*9.))


def _origin_room_with_box():
    values = np.linspace(-6., 6., 55)
    a, b = np.meshgrid(values, values)
    walls = [np.column_stack((np.full(a.size, -6.), a.ravel(), b.ravel())),
             np.column_stack((np.full(a.size, 6.), a.ravel(), b.ravel())),
             np.column_stack((a.ravel(), np.full(a.size, -6.), b.ravel())),
             np.column_stack((a.ravel(), np.full(a.size, 6.), b.ravel())),
             np.column_stack((a.ravel(), b.ravel(), np.full(a.size, -2.))),
             np.column_stack((a.ravel(), b.ravel(), np.full(a.size, 4.)))]
    side, top = np.meshgrid(np.linspace(-.9, .9, 32), np.linspace(-1., 1., 32))
    box = np.column_stack((np.full(side.size, 1.5), side.ravel(), top.ravel()+.5))
    return np.vstack([*walls, box])


def test_occluded_wall_is_hidden_but_side_view_sees_it(tmp_path):
    points, wall_ids = _occluder_scene()
    index = build_spatial_index_from_arrays({1: points}, tmp_path/"index", tile_size=1., voxel_size=.1,
                                            ram_budget_bytes=64*1024)
    front = visibility_for_view(index, Pose(np.eye(3), np.zeros(3)), View("front", 0, 0, 80, 256), _config())
    front_rows = {int(value & ((1 << 48)-1)) for value in front.point_indices}
    assert not front_rows.intersection(wall_ids)
    side_pose = Pose(np.eye(3), np.array((0., -4., 0.)))
    side = visibility_for_view(index, side_pose, View("side", 25, 0, 100, 256), _config())
    side_rows = {int(value & ((1 << 48)-1)) for value in side.point_indices}
    assert side_rows.intersection(wall_ids)


def test_sparse_thin_surface_does_not_leak_far_points():
    # Spacing-derived splats deliberately bridge sub-pixel holes in the near surface.
    near = np.array([[2., -.04, 0.], [2., .04, 0.], [2., 0., -.04], [2., 0., .04]])
    far = np.array([[8., 0., 0.]])
    points = np.vstack((near, far))
    result = visibility_for_points(points, np.arange(len(points)), Pose(np.eye(3), np.zeros(3)),
                                   View("front", 0, 0, 90, 256), spacing=np.full(len(points), .45),
                                   max_range=20., depth_resolution=256, splat_max_px=6,
                                   tolerance_abs=.02, tolerance_rel=.01, device="cpu")
    assert len(points)-1 not in set(result.point_indices.tolist())


def test_capped_close_splat_closes_one_pixel_gap_before_label_transfer():
    # At 50 cm range this 6 cm box sampling projects wider than the configured
    # six-pixel splat cap.  The far point is geometrically behind the box but
    # used to slip through the one-pixel z-buffer gap and receive a box label.
    y, z = np.meshgrid(np.linspace(-.9, .9, 32), np.linspace(-.5, 1.5, 32))
    box = np.column_stack((np.full(y.size, 1.5), y.ravel(), z.ravel()))
    far = np.array([[6., -1.7647058823529411, .1176470588235294]])
    points = np.vstack((box, far))
    camera = Camera("pinholeRepresentation", {"imageWidth": 128, "imageHeight": 64,
                    "pixelWidth": 1., "pixelHeight": 1., "focalLength": 64.,
                    "principalPointX": 64., "principalPointY": 32.})
    pose = Pose.from_dict({"rotation": {"w": 2**-.5, "x": 0., "y": -2**-.5, "z": 0.},
                           "translation": {"x": 1., "y": 0., "z": .6}})
    result = visibility_for_points(points, np.arange(len(points)), pose, camera,
                                   spacing=np.full(len(points), .063), max_range=20.,
                                   depth_resolution=128, splat_max_px=6, tolerance_abs=.02,
                                   tolerance_rel=.01, device="cpu")
    assert len(points)-1 not in set(result.point_indices.tolist())


@pytest.mark.parametrize("total", [20_000, 200_000])
def test_origin_visibility_rejects_duplicate_radial_layers_at_both_sample_sizes(total):
    visible = visibility_from_origin(_radial_layers(total//2), device="cpu")
    assert visible.mean() < .60


def test_origin_visibility_keeps_dense_static_room_with_box_above_90_percent():
    visible = visibility_from_origin(_origin_room_with_box(), device="cpu")
    assert visible.mean() >= .90


def test_origin_depth_width_scales_with_sample_count():
    for count in (20_000, 200_000):
        width = origin_depth_width(count)
        pixel_count = width*max(1, width//2)
        if width < 1024:
            assert 6.8 <= pixel_count/count <= 7.1
        else:
            assert pixel_count < 7*count


def test_splat_size_is_a_footprint_edge_length():
    depth = _torch_depth(np.array([[10.2, 10.2]]), np.array([1.]), np.array([True]), np.array([6.]),
                         24, 24, 1., 6, __import__("torch").device("cpu"), lambda: None)
    finite = np.argwhere(np.isfinite(depth))
    assert len(finite) == 36
    assert finite[:, 0].min() == finite[:, 1].min() == 7
    assert finite[:, 0].max() == finite[:, 1].max() == 12


def test_batched_splats_match_single_portion_rendering():
    rng = np.random.default_rng(42)
    uv = rng.uniform(4., 60., (200, 2))
    depth = rng.uniform(.5, 15., 200)
    spacing = rng.uniform(.01, .5, 200)
    valid = np.ones(200, bool)
    torch = __import__("torch")
    single = _torch_depth(uv, depth, valid, spacing, 64, 64, 20., 6, torch.device("cpu"), lambda: None,
                          splat_batch_entries=1_000_000)
    batched = _torch_depth(uv, depth, valid, spacing, 64, 64, 20., 6, torch.device("cpu"), lambda: None,
                           splat_batch_entries=36)
    np.testing.assert_array_equal(single, batched)


@pytest.mark.parametrize(("kind", "parameters"), [
    ("sphericalRepresentation", {"pixelWidth": 2*np.pi/100, "pixelHeight": np.pi/50}),
    ("cylindricalRepresentation", {"pixelWidth": 2*np.pi/100, "pixelHeight": .02,
                                    "radius": 1., "principalPointY": 25.}),
])
def test_angular_cameras_use_angular_pixel_density_for_splats(kind, parameters):
    camera = Camera(kind, {"imageWidth": 100, "imageHeight": 50, **parameters})
    result = visibility_for_points(np.array([[1., 0., 0.]]), np.array([0]), Pose(np.eye(3), np.zeros(3)), camera,
                                   spacing=np.array([.1]), max_range=20., depth_resolution=100,
                                   splat_max_px=6, device="cpu")
    assert np.isfinite(result.depth_map).sum() == 4


def test_visibility_cache_ignores_prompts_but_invalidates_geometry(tmp_path):
    config = _config(prompts=["wall"], min_winning_share=.5)
    pose = Pose(np.eye(3), np.zeros(3))
    view = View("front", 0, 0, 90, 128)
    cache = VisibilityCache(tmp_path, "source")
    key = visibility_key("source", pose, {"kind": "perspective"}, {"id": view.id}, config)
    calls = []
    result, hit = cache.get_or_compute(key, lambda: (calls.append(1) or visibility_for_points(
        np.array([[2., 0., 0.]]), np.array([1]), pose, view, max_range=20., depth_resolution=128,
        splat_max_px=1, device="cpu")))
    assert not hit and len(result.point_indices) == 1
    prompt_only = _config(prompts=["ceiling"], min_winning_share=.9)
    assert visibility_key("source", pose, {"kind": "perspective"}, {"id": view.id}, prompt_only) == key
    assert key == "ff1a4b4a53c958f844f336b85aed602ad5b7793d06de3effd8d8c2e2f8ed8554"
    _, hit = cache.get_or_compute(key, lambda: (calls.append(1) or result))
    assert hit and calls == [1] and cache.recomputations == 1
    changed_pose = Pose(np.eye(3), np.array((.1, 0., 0.)))
    assert visibility_key("source", changed_pose, {"kind": "perspective"}, {"id": view.id}, config) != key
    changed_tolerance = _config(slam_depth_tol_abs=.03)
    assert visibility_key("source", pose, {"kind": "perspective"}, {"id": view.id}, changed_tolerance) != key


@pytest.mark.parametrize("splat_max_px", [1, 6])
def test_frustum_candidates_are_bit_identical_for_perspective_and_spherical_views(splat_max_px):
    rng = np.random.default_rng(91)
    world = rng.uniform(-12., 12., (2_000, 3))
    point_ids = np.arange(len(world), dtype=np.int64)
    candidates = np.empty(len(world), [("xyz", "<f8", (3,)), ("point_id", "<i8"),
                                       ("spacing", "<f4"), ("normal", "<f4", (3,))])
    candidates["xyz"], candidates["point_id"] = world, point_ids
    candidates["spacing"] = rng.uniform(.01, .3, len(world))
    candidates["normal"] = rng.normal(size=(len(world), 3))
    pose = Pose(np.eye(3), np.array((.3, -.2, .1)))
    projectors = [View("front", 25, -10, 95, 192),
                  Camera("sphericalRepresentation", {"imageWidth": 192, "imageHeight": 96,
                         "pixelWidth": 2*np.pi/192, "pixelHeight": np.pi/96}),
                  Camera("pinholeRepresentation", {"imageWidth": 192, "imageHeight": 96,
                         "pixelWidth": 1., "pixelHeight": 1., "focalLength": 120.,
                         "principalPointX": 96., "principalPointY": 48.})]
    config = _config(slam_splat_max_px=splat_max_px, slam_depth_resolution=192)
    for projector in projectors:
        old = visibility_for_points(world, point_ids, pose, projector, spacing=candidates["spacing"],
                                    normals=candidates["normal"], max_range=20., depth_resolution=192,
                                    splat_max_px=splat_max_px, tolerance_abs=.02, tolerance_rel=.01, device="cpu")
        new = visibility_for_candidates(candidates, pose, projector, config,
                                        camera_xyz=pose.from_world(world))
        for field in ("point_indices", "pixels", "distance", "incidence_angle", "radial_position", "depth_map"):
            assert np.array_equal(getattr(new, field), getattr(old, field))


def test_camera_candidates_are_loaded_once_and_not_for_full_visibility_cache_hits(tmp_path):
    points, _ = _occluder_scene()
    values = np.empty(len(points), [("xyz", "<f8", (3,)), ("point_id", "<i8"),
                                    ("spacing", "<f4"), ("normal", "<f4", (3,))])
    values["xyz"], values["point_id"], values["spacing"] = points, np.arange(len(points)), .1
    values["normal"] = np.array((1., 0., 0.), np.float32)
    pose = Pose(np.eye(3), np.zeros(3))
    cache = VisibilityCache(tmp_path, "source")
    calls = []

    class FakeIndex:
        def query_range(self, origin, maximum, checkpoint):
            calls.append(1)
            return values

    load = _slam_candidate_loader(FakeIndex(), pose, _config(slam_depth_resolution=128), lambda: None)

    for view in (View("first", 0, 0, 90, 128), View("second", 35, 0, 90, 128)):
        cache.visibility_for_view(None, pose, view, _config(slam_depth_resolution=128),
                                  intrinsics={"kind": "perspective"}, view={"id": view.id}, candidates_loader=load)
    assert calls == [1]
    for view in (View("first", 0, 0, 90, 128), View("second", 35, 0, 90, 128)):
        cache.visibility_for_view(None, pose, view, _config(slam_depth_resolution=128),
                                  intrinsics={"kind": "perspective"}, view={"id": view.id},
                                  candidates_loader=lambda: pytest.fail("cache hit must not load candidates"))


def test_index_respects_small_budget_and_uses_disk_tiles(tmp_path):
    points = np.column_stack((np.linspace(0, 100, 2000), np.zeros(2000), np.zeros(2000)))
    index = build_spatial_index_from_arrays({1: points}, tmp_path/"index", tile_size=1., voxel_size=.1,
                                            ram_budget_bytes=64*1024)
    assert index.meta["ram_budget_bytes"] == 64*1024
    assert index.meta["maximum_batch_bytes"] <= 64*1024
    assert len(index.query_range((0., 0., 0.), 1.)) < len(points)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA unavailable")
def test_cuda_and_cpu_visibility_sets_match():
    points, _ = _occluder_scene()
    view = View("front", 0, 0, 80, 256)
    arguments = dict(spacing=np.full(len(points), .1), max_range=20., depth_resolution=256,
                     splat_max_px=6, tolerance_abs=.02, tolerance_rel=.01)
    cpu = visibility_for_points(points, np.arange(len(points)), Pose(np.eye(3), np.zeros(3)), view, device="cpu", **arguments)
    cuda = visibility_for_points(points, np.arange(len(points)), Pose(np.eye(3), np.zeros(3)), view, device="cuda:0", **arguments)
    assert set(cpu.point_indices.tolist()) == set(cuda.point_indices.tolist())
