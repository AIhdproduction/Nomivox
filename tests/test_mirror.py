import numpy as np
from e57cls.mirror.evidence import Plane, Registry, DirectIndex, label_points


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
    direct.add(np.array([[2.,0,0]]),1)
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
