import numpy as np
from e57cls.mirror.evidence import Plane, Registry, DirectIndex, label_points
from e57cls.mirror.pipeline import fit_surfaces, fit_surfaces_many


def mirror():
    return Plane([0,0,0], [1,0,0], [[0,1,0],[0,0,1]], [[-1,-1],[-1,1],[1,1],[1,-1]],
                 [[0,y,z] for y in (-1,0,1) for z in (-1,0,1)], True)


def test_multi_scan_agreement_through_mirror_is_not_real_confirmation(tmp_path):
    registry = Registry([mirror()])
    direct = DirectIndex(tmp_path/'d.sqlite', np.zeros(3))
    # Both scanner origins are in front of mirror. The same virtual location is blocked for both.
    virtual = np.array([[-2.,0,0], [-2.,.2,0]])
    for scan, origin in [(1, np.array([2.,0,0])), (2,np.array([2.,.2,0]))]:
        _, states = registry.rays(origin, virtual)
        assert states.tolist() == [4,4]
        direct.add(virtual[states == 0], scan)
    assert label_points(registry, direct, [2,0,0], virtual, 1).tolist() == [2,2]
    # A third scanner behind the mirror provides a direct measurement for one location.
    direct.add(virtual[:1], 3)
    assert label_points(registry, direct, [2,0,0], virtual, 1).tolist() == [3,2]
    # No own mask is required: registry ray geometry alone handles another scan.
    assert label_points(registry, direct, [3,0,0], virtual, 4).tolist() == [3,2]
    points = np.array([[0,0,0], [.5,0,0], [-2,8,0]])
    assert label_points(registry, direct, [2,0,0], points, 1).tolist() == [1,0,0]
    assert label_points(Registry(), direct, [2,0,0], virtual, 1, unknown_mask=np.ones(2,bool)).tolist() == [4,4]
    direct.close()


def test_low_quality_plane_requires_mirrored_geometry(tmp_path):
    plane = mirror()
    plane.quality = False
    registry = Registry([plane])
    direct = DirectIndex(tmp_path/'d.sqlite',np.zeros(3))
    points = np.array([[-2.,0,0],[-2.,.3,0]])
    assert label_points(registry,direct,[2,0,0],points,1).tolist() == [4,4]
    # The pipeline keeps its temporary cross-scan index in one transaction;
    # reads on that connection must still see newly indexed direct points.
    direct.add(np.array([[2.,0,0]]),1,commit=False)
    assert label_points(registry,direct,[2,0,0],points,1).tolist() == [2,4]
    direct.close()


def test_registry_merges_overlapping_planes_only():
    one = mirror()
    one.frames = np.array([[0,y,z] for y in np.linspace(-1,1,6) for z in np.linspace(-1,1,6)])
    registry = Registry([one])
    other = mirror()
    other.center += [.01,.1,0]
    other.frames = one.frames+[.01,.1,0]
    registry.add(other)
    assert len(registry.planes) == 1
    far = mirror()
    far.center += [0,10,0]
    far.frames += [0,10,0]
    registry.add(far)
    assert len(registry.planes) == 2


def test_fused_surface_sampling_matches_the_independent_reference():
    class Camera:
        width = height = 100
        wraps = False

        @staticmethod
        def project(points):
            uv = np.column_stack((50 + points[:, 1] * 40, 50 + points[:, 2] * 40))
            return uv, np.linalg.norm(points, axis=1), np.ones(len(points), bool)

        @staticmethod
        def rays(uv):
            rays = np.column_stack((np.ones(len(uv)), (uv[:, 0]-50)/40, (uv[:, 1]-50)/40))
            return rays / np.linalg.norm(rays, axis=1)[:, None]

    class Store:
        def __init__(self):
            y, z = np.meshgrid(np.linspace(-1, 1, 61), np.linspace(-1, 1, 61))
            self.xyz = np.column_stack((np.full(y.size, 3.), y.ravel(), z.ravel()))
            self.meta = {'metadata': {'pose': {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.},
                                                'translation': {'x': 0., 'y': 0., 'z': 0.}}}}
            self.calls = 0

        def chunks(self):
            self.calls += 1
            yield 0, self.xyz, np.ones(len(self.xyz), bool)

    class Control:
        @staticmethod
        def checkpoint():
            pass

    mask = np.zeros((100, 100), bool)
    mask[42:58, 42:58] = True
    config = {'seed': 42, 'mirror_min_pixels': 20, 'mirror_frame_band': 8,
              'mirror_frame_band_max': 8, 'mirror_fit_tolerance': .02,
              'mirror_max_range': 30., 'mirror_shape_regularize': False,
              'mirror_shape_kinds': (), 'mirror_plane_tolerance': .03,
              'mirror_edge_search': .1, 'mirror_shape_min_iou': .75,
              'mirror_shape_max_growth': 1.35}
    camera, pose, control = Camera(), type('Pose', (), {'rotation': np.eye(3), 'translation': np.zeros(3),
                                                         'from_world': staticmethod(lambda points: points)})(), Control()
    baseline_store = Store()
    baseline = {kind: fit_surfaces(baseline_store, camera, pose, mask, config, control, kind=kind)
                for kind in ('mirror_mask', 'glass_mask', 'reflective_mask')}
    fused_store = Store()
    fused = fit_surfaces_many(fused_store, camera, pose,
                              {kind: mask for kind in ('mirror_mask', 'glass_mask', 'reflective_mask')}, config, control)
    assert baseline_store.calls == 3
    assert fused_store.calls == 1
    for kind in baseline:
        old_labels, old_planes, old_details = baseline[kind]
        labels, planes, details = fused[kind]
        np.testing.assert_array_equal(labels, old_labels)
        assert details == old_details
        assert set(planes) == set(old_planes)
        for region in planes:
            for field in ('center', 'normal', 'basis', 'outline', 'frames'):
                np.testing.assert_array_equal(getattr(planes[region], field), getattr(old_planes[region], field))
