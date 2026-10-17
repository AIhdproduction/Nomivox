import numpy as np
import laspy

from e57cls.io.store import ScanStore, reduced_store, reduction_mask
from e57cls.output.las import write_cloud


def make_store(tmp_path, count=1000):
    xyz = np.c_[np.arange(count, dtype=np.float64), np.zeros((count, 2))]
    valid = np.ones(count, bool)
    valid[::17] = False
    np.save(tmp_path/'xyz.npy', xyz)
    np.save(tmp_path/'valid.npy', valid)
    pose = {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.},
            'translation': {'x': 0., 'y': 0., 'z': 0.}}
    return ScanStore(tmp_path, {'record_count': count, 'valid_count': int(valid.sum()), 'scan_index': 1,
        'source_sha256': 'test', 'arrays': {'xyz_local': {'path': 'xyz.npy'}, 'valid': {'path': 'valid.npy'}},
        'metadata': {'pose': pose, 'points': {'fields': {}}},
        'bounds_local': {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()}})


def test_reduction_mask_is_exact_deterministic_and_uniform(tmp_path):
    store = make_store(tmp_path)
    path = reduction_mask(store, .4, 42)
    first = np.load(path)
    assert first.sum() == round(store.meta['valid_count']*.6)
    assert not np.any(first & ~store.read('valid', 0, store.meta['record_count']))
    assert path == reduction_mask(store, .4, 42)
    # Each tenth of record order receives the same approximately 60 % density.
    valid = store.read('valid', 0, store.meta['record_count'])
    for indices in np.array_split(np.arange(store.meta['record_count']), 10):
        assert abs(first[indices].sum()/valid[indices].sum()-.6) < .03
    reduced = reduced_store(store, .4, 42)
    one = np.concatenate([valid for _, _, valid in reduced.chunks(37)])
    many = np.concatenate([valid for _, _, valid in reduced.chunks(257)])
    np.testing.assert_array_equal(one, many)
    assert reduced_store(store, 0., 42) is store


def test_output_density_selects_retained_or_all_valid_points(tmp_path):
    store = make_store(tmp_path, 50)
    reduced = reduced_store(store, .4, 7)
    class Votes:
        def lookup(self, keys):
            return np.full(len(keys), 2, np.uint8), np.full(len(keys), 200, np.uint8)
    write_cloud(tmp_path/'reduced.las', [reduced], Votes(), np.zeros(3), 1., '', [1, 2], output_density='reduced')
    write_cloud(tmp_path/'original.las', [reduced], Votes(), np.zeros(3), 1., '', [1, 2], output_density='original')
    assert len(laspy.read(tmp_path/'reduced.las')) == round(store.meta['valid_count']*.6)
    assert len(laspy.read(tmp_path/'original.las')) == store.meta['valid_count']


def test_zero_reduction_keeps_las_point_data_identical(tmp_path):
    store = make_store(tmp_path, 30)
    class Votes:
        def lookup(self, keys):
            return np.full(len(keys), 2, np.uint8), np.full(len(keys), 99, np.uint8)
    write_cloud(tmp_path/'baseline.las', [store], Votes(), np.zeros(3), 1., '', [1, 2])
    write_cloud(tmp_path/'zero.las', [reduced_store(store, 0., 42)], Votes(), np.zeros(3), 1., '', [1, 2])
    baseline, zero = laspy.read(tmp_path/'baseline.las'), laspy.read(tmp_path/'zero.las')
    np.testing.assert_array_equal(baseline.points.array, zero.points.array)


def test_parallel_laz_readback_matches_las_point_data(tmp_path):
    store = make_store(tmp_path, 30)
    class Votes:
        def lookup(self, keys):
            return np.full(len(keys), 2, np.uint8), np.full(len(keys), 99, np.uint8)
    write_cloud(tmp_path/'cloud.las', [store], Votes(), np.zeros(3), 1., '', [1, 2])
    write_cloud(tmp_path/'cloud.laz', [store], Votes(), np.zeros(3), 1., '', [1, 2])
    las, laz = laspy.read(tmp_path/'cloud.las'), laspy.read(tmp_path/'cloud.laz')
    np.testing.assert_array_equal(las.points.array, laz.points.array)
