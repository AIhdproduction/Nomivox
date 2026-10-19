"""Regression coverage for optimizations whose results must stay exact."""

import os
import sqlite3
import time
from dataclasses import dataclass
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
from scipy.spatial import cKDTree

from e57cls.io.store import ScanStore
from e57cls.geometry.cameras import Camera, Pose
from e57cls.mirror.evidence import DirectIndex
from e57cls.output.classify import ClassifiedSource
from e57cls.output.las import ClassPointIndex, write_cloud
from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline import run as pipeline_run
from e57cls.vote.engine import VoteStore, voxel_keys
import e57cls.vote.engine as vote_engine


def _old_lookup(connection, keys):
    """The former SQLite/unique lookup, kept here as an independent oracle."""
    unique, inverse = np.unique(keys, return_inverse=True)
    rows = {}
    for start in range(0, len(unique), 500):
        part = [bytes(key) for key in unique[start:start+500]]
        statement = 'SELECT key,code,confidence FROM decisions WHERE key IN ('+','.join('?' for _ in part)+')'
        rows.update((key, (code, confidence)) for key, code, confidence in connection.execute(statement, part))
    values = np.asarray([rows.get(bytes(key), (1, 0)) for key in unique], np.uint8).reshape(-1, 2)
    return values[inverse, 0], values[inverse, 1]


def _old_decision(rows, count, order, minimum, share, priority):
    total = sum(weight for _, weight, _ in rows)
    code, maximum, _ = min(rows, key=lambda item: (-item[1], order.get(item[0], 1000+item[0])))
    override = False
    if priority and code in priority['over'] and count >= minimum:
        evidence = next((row for row in rows if row[0] == priority['code'] and row[2] >= priority['confidence']), None)
        if evidence:
            code, maximum, _ = evidence
            override = True
    fraction = maximum/total if total else 0
    decided = code if count >= minimum and (fraction >= share or override) else 1
    return decided, round(fraction*255) if decided != 1 else 0, override


def _decision_rows(votes):
    return [(bytes(key), code, confidence) for key, code, confidence in votes.db.execute(
        'SELECT key,code,confidence FROM decisions ORDER BY key')]


def _old_decisions(votes, minimum, share, class_order, priority):
    order = {int(code): index for index, code in enumerate(class_order)}
    grouped = {}
    cursor = votes.db.execute('SELECT v.key,v.class,v.weight,o.count,v.peak FROM votes v JOIN observations o ON o.key=v.key ORDER BY v.key,v.class')
    for key, code, weight, count, peak in cursor:
        grouped.setdefault(bytes(key), [count, []])[1].append((code, weight, peak))
    result, overrides = [], 0
    for key, (count, rows) in grouped.items():
        code, confidence, override = _old_decision(rows, count, order, minimum, share, priority)
        result.append((key, code, confidence))
        overrides += override
    return result, overrides


def test_decide_and_lookup_match_previous_sqlite_path(tmp_path):
    votes = VoteStore(tmp_path/'votes.sqlite')
    try:
        world = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0]], float)
        keys = voxel_keys(world, np.zeros(3), 1)
        first = np.array([[.8, .8, 0], [.1, .9, .7], [.6, 0, .7], [.2, .2, .8]], np.float32)
        second = np.array([[.1, .1, .9], [.1, .1, .7], [.2, .8, 0], [.2, .2, .8]], np.float32)
        votes.add_scan(1, 'same', [(keys, first)], [105, 102, 100])
        votes.add_scan(2, 'same', [(keys, second)], [105, 102, 100])
        order = {105: 0, 102: 1, 100: 2}
        priority = {'code': 105, 'over': [102], 'confidence': .65}
        expected, overrides = [], 0
        cursor = votes.db.execute('SELECT v.key,v.class,v.weight,o.count,v.peak FROM votes v JOIN observations o ON o.key=v.key ORDER BY v.key,v.class')
        grouped = {}
        for key, code, weight, count, peak in cursor:
            grouped.setdefault(bytes(key), [count, []])[1].append((code, weight, peak))
        for key, (count, rows) in grouped.items():
            code, confidence, override = _old_decision(rows, count, order, 2, .4, priority)
            expected.append((key, code, confidence))
            overrides += override
        votes.decide(2, .4, [105, 102, 100], priority=priority)
        assert _decision_rows(votes) == expected
        assert votes.priority_overrides == overrides
        probe = np.r_[keys[[2, 0, 1, 3, 0]], voxel_keys([[99, 0, 0]], np.zeros(3), 1)]
        old_codes, old_confidence = _old_lookup(votes.db, probe)
        codes, confidence = votes.lookup(probe)
        np.testing.assert_array_equal(codes, old_codes)
        np.testing.assert_array_equal(confidence, old_confidence)
    finally:
        votes.close()


def test_decide_matches_previous_path_for_seeded_edge_cases(tmp_path):
    priority = {'code': 105, 'over': [102, 100], 'confidence': .65}
    classes = [105, 102, 100]
    for seed, minimum, share in ((3, 1, .4), (7, 2, .5), (11, 3, .6)):
        votes = VoteStore(tmp_path/f'votes_{seed}.sqlite')
        try:
            rng = np.random.default_rng(seed)
            keys = voxel_keys(np.c_[np.arange(97)-48, np.zeros((97, 2))], np.zeros(3), 1)
            for scan in range(1, 4):
                weights = rng.random((len(keys), len(classes)), dtype=np.float32)
                weights[rng.random(weights.shape) < .35] = 0
                weights[scan-1, :2] = .5  # exact tie, class order resolves it
                weights[scan+4] = (0, .7, .7)  # secondary tie
                votes.add_scan(scan, 'same', [(keys, weights)], classes)
            expected, overrides = _old_decisions(votes, minimum, share, classes, priority)
            votes.decide(minimum, share, classes, priority=priority)
            assert _decision_rows(votes) == expected
            assert votes.priority_overrides == overrides
        finally:
            votes.close()
    votes = VoteStore(tmp_path/'boundary.sqlite')
    try:
        keys = voxel_keys([[0, 0, 0]], np.zeros(3), 1)
        votes.add_scan(1, 'same', [(keys, np.array([[.5, .5]], np.float64))], [10, 20])
        expected, overrides = _old_decisions(votes, 1, .5, [10, 20], None)
        votes.decide(1, .5, [10, 20])
        assert _decision_rows(votes) == expected == [(bytes(keys[0]), 10, 128)]
        assert votes.priority_overrides == overrides == 0
    finally:
        votes.close()


def test_lookup_snapshot_matches_blob_order_and_cleans_memmaps(tmp_path, monkeypatch):
    monkeypatch.setattr(vote_engine.psutil, 'virtual_memory', lambda: SimpleNamespace(available=64))
    votes = VoteStore(tmp_path/'lookup.sqlite')
    paths = ()
    try:
        negative = voxel_keys([[-1, 0, 0], [0, -1, 0]], np.zeros(3), 1)
        keys = np.asarray([np.void(bytes(23)+b'\x01'), np.void(bytes(23)+b'\x02'), *negative], dtype='V24')
        votes.db.executemany('INSERT INTO decisions VALUES (?,?,?)',
                             ((bytes(key), code, confidence) for code, confidence, key in zip((2, 3, 4, 5), (20, 30, 40, 50), keys)))
        votes.db.commit()
        probe = keys[[3, 1, 0, 2]]
        old_codes, old_confidence = _old_lookup(votes.db, probe)
        codes, confidence = votes.lookup(probe)
        np.testing.assert_array_equal(codes, old_codes)
        np.testing.assert_array_equal(confidence, old_confidence)
        paths = votes._lookup_paths
        assert len(paths) == 2 and all(path.is_file() for path in paths)
    finally:
        votes.close()
    assert all(not path.exists() for path in paths)


def test_cleanup_and_vegetation_keep_negative_voxel_results(tmp_path):
    votes = VoteStore(tmp_path/'cleanup.sqlite')
    try:
        points = np.array([[-3, -3, -3], [-2, -2, -2], [-1, -1, -1], [8, 0, 0],
                           [-10, -10, 0], [-10, -10, 1], [-10, -10, 2]], float)
        keys = voxel_keys(points, np.zeros(3), 1)
        weights = np.ones((len(keys), 1), np.float32)
        votes.add_scan(1, 'same', [(keys, weights)], [3])
        votes.decide(1, .1, [3])
        cleanup = votes.remove_small_components(1, 3, [(3, 3, 2)])
        assert cleanup == {'components': 3, 'removed_voxels': 1}
        vegetation = votes.vegetation_bases_to_ground(3, 2, 1, 3, 1)
        assert vegetation == {'moved_voxels': 4, 'cells': 2}
        assert votes.lookup(keys)[0].tolist() == [2, 2, 3, 1, 2, 2, 3]
    finally:
        votes.close()


def _old_nearest(points, sources, codes, radius):
    result = np.zeros(len(points), np.uint8)
    for index, point in enumerate(points):
        distances = np.linalg.norm(sources-point, axis=1)
        nearest = distances.argmin()
        if distances[nearest] <= radius:
            result[index] = codes[nearest]
    return result


def test_class_transfer_and_direct_index_match_reference_without_ties(tmp_path):
    origin = np.zeros(3)
    sources = np.array([[0, 0, 0], [.03, 0, 0], [1, 1, 1]], float)
    codes = np.array([2, 6, 9], np.uint8)
    queries = np.array([[.004, 0, 0], [.027, 0, 0], [.8, 1, 1], [4, 4, 4]], float)
    index = ClassPointIndex(tmp_path/'classes.sqlite', origin, block=.05)
    direct = DirectIndex(tmp_path/'direct.sqlite', origin, block=.05)
    try:
        index.add(sources, codes)
        index.db.commit()
        np.testing.assert_array_equal(index.nearest_codes(queries, .01), _old_nearest(queries, sources, codes, .01))
        direct.add(sources, 1)
        assert direct.nearby(queries, .01).tolist() == (_old_nearest(queries, sources, np.ones(len(sources), np.uint8), .01) != 0).tolist()
    finally:
        index.close()
        direct.close()


def _synthetic_store(tmp_path):
    xyz = np.array([[0., 0, 0], [.005, 0, 0], [1., 0, 0]], np.float64)
    valid = np.ones(len(xyz), bool)
    np.save(tmp_path/'xyz.npy', xyz)
    np.save(tmp_path/'valid.npy', valid)
    pose = {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.}, 'translation': {'x': 0., 'y': 0., 'z': 0.}}
    return ScanStore(tmp_path, {'record_count': len(xyz), 'valid_count': len(xyz), 'scan_index': 1,
        'arrays': {'xyz_local': {'path': 'xyz.npy'}, 'valid': {'path': 'valid.npy'}},
        'metadata': {'pose': pose, 'points': {'fields': []}},
        'bounds_local': {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()}})


def test_write_cloud_transfer_reuses_exact_predictions_and_mirror_states(tmp_path):
    class Votes:
        def lookup(self, keys):
            return np.array([2, 1, 1], np.uint8), np.array([20, 0, 0], np.uint8)
    def mirror(store, start, stop, valid, world, codes):
        return codes, np.full(len(world), 4, np.uint8)
    path = tmp_path/'cloud.las'
    result = write_cloud(path, [_synthetic_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2],
                         mirror=mirror, class_transfer_radius=.01)
    cloud = laspy.read(path)
    assert result['class_transfer_count'] == 1
    assert cloud.classification.tolist() == [2, 2, 1]
    assert cloud.class_conf.tolist() == [20, 128, 0]
    assert cloud.mirror_state.tolist() == [4, 4, 4]


def _distant_transfer_store(tmp_path):
    xyz = np.array([[10., 0, 0], [10.025, 0, 0]], np.float64)
    valid = np.ones(len(xyz), bool)
    np.save(tmp_path/'distant_xyz.npy', xyz)
    np.save(tmp_path/'distant_valid.npy', valid)
    pose = {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.}, 'translation': {'x': 0., 'y': 0., 'z': 0.}}
    return ScanStore(tmp_path, {'record_count': len(xyz), 'valid_count': len(xyz), 'scan_index': 1,
        'arrays': {'xyz_local': {'path': 'distant_xyz.npy'}, 'valid': {'path': 'distant_valid.npy'}},
        'metadata': {'pose': pose, 'points': {'fields': []}},
        'bounds_local': {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()}})


def test_transfer_radius_grows_with_own_scanner_distance(tmp_path):
    class Votes:
        def lookup(self, keys):
            return np.array([1, 2], np.uint8), np.array([0, 200], np.uint8)

    fixed = write_cloud(tmp_path/'fixed.las', [_distant_transfer_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2],
                        class_transfer_radius=.01, class_transfer_range_factor=0., class_transfer_radius_max=.05)
    adaptive = write_cloud(tmp_path/'adaptive.las', [_distant_transfer_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2],
                           class_transfer_radius=.01, class_transfer_range_factor=.003, class_transfer_radius_max=.05)
    assert fixed['class_transfer_count'] == 0
    assert adaptive['class_transfer_count'] == 1
    assert laspy.read(tmp_path/'adaptive.las').classification.tolist() == [2, 2]


def test_zero_range_factor_keeps_transfer_output_identical(tmp_path):
    class Votes:
        def lookup(self, keys):
            return np.array([1, 2], np.uint8), np.array([0, 200], np.uint8)

    old_path, fixed_path = tmp_path/'old.las', tmp_path/'zero-factor.las'
    write_cloud(old_path, [_distant_transfer_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2], class_transfer_radius=.01)
    write_cloud(fixed_path, [_distant_transfer_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2],
                class_transfer_radius=.01, class_transfer_range_factor=0., class_transfer_radius_max=.05)
    old, fixed = laspy.read(old_path), laspy.read(fixed_path)
    np.testing.assert_array_equal(old.classification, fixed.classification)
    np.testing.assert_array_equal(old.class_conf, fixed.class_conf)


def test_tiled_transfer_matches_reference_for_large_negative_boundary_data(tmp_path):
    """The tiled path retains the old nearest-neighbour result without ties."""
    rng = np.random.default_rng(103)
    sources = rng.uniform(-8, 8, (200_000, 3))
    # Exercise both sides of root-tile edges, including negative coordinates.
    queries = np.r_[sources[:4_000]+rng.uniform(-.015, .015, (4_000, 3)),
                    np.column_stack((np.r_[-2.001, -2., -1.999, 1.999, 2., 2.001], np.zeros(6), np.zeros(6)))]
    codes = rng.integers(2, 68, len(sources), dtype=np.uint8)
    radius = .04
    reference_tree = cKDTree(sources)
    distances, ids = reference_tree.query(queries, distance_upper_bound=radius)
    expected = np.zeros(len(queries), np.uint8)
    found = np.isfinite(distances)
    expected[found] = codes[ids[found]]
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=radius)
    try:
        index.add(sources, codes)
        np.testing.assert_array_equal(index.nearest_codes(queries, radius), expected)
    finally:
        index.close()


def test_tiled_transfer_matches_ckdtree_for_600k_sources_and_60k_queries(tmp_path):
    """A larger exact reference guards the vectorized tile construction."""
    rng = np.random.default_rng(20261018)
    sources = rng.uniform(-20, 20, (600_000, 3))
    codes = rng.integers(2, 68, len(sources), dtype=np.uint8)
    queries = np.r_[sources[:30_000]+rng.uniform(-.008, .008, (30_000, 3)),
                    rng.uniform(-20, 20, (30_000, 3))]
    radius = .02
    distances, ids = cKDTree(sources).query(queries, distance_upper_bound=radius)
    expected = np.zeros(len(queries), np.uint8)
    found = np.isfinite(distances)
    expected[found] = codes[ids[found]]
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=radius)
    try:
        for start in range(0, len(sources), 500_000):
            index.add(sources[start:start+500_000], codes[start:start+500_000])
        np.testing.assert_array_equal(index.nearest_codes(queries, radius), expected)
    finally:
        index.close()


def test_tiled_transfer_matches_ckdtree_with_lv95_coordinates(tmp_path):
    """Absolute projected coordinates are packed relative to the first tile."""
    rng = np.random.default_rng(20261019)
    centre = np.array([2_600_000., 1_200_000., 480.])
    sources = centre+rng.uniform(-20, 20, (20_000, 3))
    codes = rng.integers(2, 68, len(sources), dtype=np.uint8)
    queries = np.r_[sources[:4_000]+rng.uniform(-.008, .008, (4_000, 3)),
                    centre+rng.uniform(-20, 20, (4_000, 3))]
    radius = .02
    distances, ids = cKDTree(sources).query(queries, distance_upper_bound=radius)
    expected = np.zeros(len(queries), np.uint8)
    expected[np.isfinite(distances)] = codes[ids[np.isfinite(distances)]]
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=radius)
    try:
        index.add(sources, codes)
        np.testing.assert_array_equal(index.nearest_codes(queries, radius), expected)
    finally:
        index.close()


def test_tiled_transfer_falls_back_for_groups_more_than_3000_km_apart(tmp_path):
    """Far-apart tiles retain their exact int64 coordinates instead of failing."""
    rng = np.random.default_rng(20261020)
    first = rng.uniform(-10, 10, (500, 3))
    second = first+np.array([3_000_000., 0., 0.])
    sources = np.r_[first, second]
    codes = rng.integers(2, 68, len(sources), dtype=np.uint8)
    queries = np.r_[sources[:200]+rng.uniform(-.005, .005, (200, 3)),
                    sources[-200:]+rng.uniform(-.005, .005, (200, 3))]
    radius = .02
    distances, ids = cKDTree(sources).query(queries, distance_upper_bound=radius)
    expected = np.zeros(len(queries), np.uint8)
    expected[np.isfinite(distances)] = codes[ids[np.isfinite(distances)]]
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=radius)
    try:
        index.add(sources, codes)
        np.testing.assert_array_equal(index.nearest_codes(queries, radius), expected)
    finally:
        index.close()


def test_classified_source_closes_partial_resources_after_prepare_error(tmp_path):
    class BrokenStore:
        meta = {'scan_index': 7, 'record_count': 10, 'valid_count': 10}

        def chunks(self, capacity):
            raise RuntimeError('synthetic preparation error')
            yield

    class Votes:
        def lookup(self, keys):
            return np.zeros(len(keys), np.uint8), np.zeros(len(keys), np.uint8)

    with pytest.raises(RuntimeError, match='synthetic preparation error'):
        with ClassifiedSource(tmp_path, [BrokenStore()], Votes(), np.zeros(3), .02,
                              radius=.01):
            pass
    assert not list(tmp_path.glob('e57cls_class_neighbours_*'))


def test_tiled_transfer_uses_lowest_source_ordinal_for_exact_ties(tmp_path):
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=.1)
    try:
        index.add(np.array([[-.02, 0, 0], [.02, 0, 0]]), np.array([9, 2], np.uint8),
                  np.array([3, 8], np.int64))
        assert index.nearest_codes(np.array([[0., 0, 0]]), .1).tolist() == [9]
    finally:
        index.close()


def test_tiled_transfer_adaptively_subdivides_dense_root_tile(tmp_path):
    points = np.array([[x, y, z] for x in (.2, .7, 1.2, 1.7) for y in (.2, .7) for z in (.2, .7)])
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=.02, max_tile_points=5)
    try:
        index.add(points, np.full(len(points), 7, np.uint8))
        assert index.nearest_codes(points, .001).tolist() == [7]*len(points)
        assert index._children
    finally:
        index.close()


def test_output_prepare_events_and_timing_are_reported(tmp_path):
    class Votes:
        def lookup(self, keys):
            return np.array([2, 1, 1], np.uint8), np.array([20, 0, 0], np.uint8)
    events = []
    result = write_cloud(tmp_path/'cloud.las', [_synthetic_store(tmp_path)], Votes(), np.zeros(3), .01, '', [1, 2],
                         class_transfer_radius=.01, progress=lambda *event: events.append(event))
    assert any(event[0] == 'output_prepare' for event in events)
    # The terrain fill is a separate output stage, so existing timing keys must
    # remain stable while the new key is explicitly required.
    assert {'lookup_mirror', 'transfer', 'plane_fill', 'write', 'readback'} <= set(result['timing_seconds'])
    assert 'terrain_fill' in result['timing_seconds']


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get('NOMIVOX_PERF_BENCHMARK'), reason='set NOMIVOX_PERF_BENCHMARK=1 to run benchmarks')
def test_tiled_transfer_benchmark(tmp_path):
    """Report the requested end-to-end tile-index timings on synthetic data."""
    rng = np.random.default_rng(20261018)
    sources = rng.uniform(0, 40, (3_000_000, 3))
    codes = rng.integers(2, 68, len(sources), dtype=np.uint8)
    queries = rng.uniform(0, 40, (125_000, 3))
    radius = .01
    index = ClassPointIndex(tmp_path/'classes', np.zeros(3), radius=radius)
    try:
        started = time.perf_counter()
        for start in range(0, len(sources), 500_000):
            index.add(sources[start:start+500_000], codes[start:start+500_000])
        index._finalize()
        add_seconds = time.perf_counter()-started
        started = time.perf_counter()
        result = index.nearest_codes(queries, radius)
        query_seconds = time.perf_counter()-started
        print(f'tiled transfer benchmark: add={add_seconds:.2f}s query={query_seconds:.2f}s')
        assert result.shape == (len(queries),)
        assert add_seconds <= 12
    finally:
        index.close()


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get('NOMIVOX_PERF_BENCHMARK'), reason='set NOMIVOX_PERF_BENCHMARK=1 to run benchmarks')
def test_lookup_benchmark(tmp_path):
    votes = VoteStore(tmp_path/'benchmark.sqlite')
    try:
        keys = voxel_keys(np.c_[np.arange(100_000), np.zeros((100_000, 2))], np.zeros(3), 1)
        votes.add_scan(1, 'benchmark', [(keys, np.ones((len(keys), 1), np.float32))], [2])
        votes.decide(1, .1, [2])
        start = time.perf_counter()
        votes.lookup(keys)
        assert time.perf_counter()-start >= 0
    finally:
        votes.close()


@dataclass(frozen=True)
class _ProjectionView:
    id: str
    size: int = 16

    def project(self, camera_points):
        count = len(camera_points)
        uv = np.column_stack(((np.arange(count) % self.size)+.5,
                              ((np.arange(count)//self.size) % self.size)+.5))
        return uv, np.ones(count), np.ones(count, bool)


class _ProjectionDepthBuffer:
    width = 32
    height = 16

    @staticmethod
    def visible(uv, depth, valid, absolute, relative):
        return valid.copy()


class _ProjectionControl:
    def __init__(self):
        self.events = []

    @staticmethod
    def checkpoint():
        return None

    def event(self, kind, **payload):
        self.events.append((kind, payload))


class _ProjectionMirrors:
    @staticmethod
    def states(world, *args):
        states = np.zeros(len(world), np.uint8)
        states[::5] = 2
        return states


def _old_image_vote_reference(reps, camera, camera_pose, depth_buffer, views, labels, confidence,
                              classes, config, mirror_states):
    """Former per-view projection path, retained as an independent oracle."""
    weights = np.zeros((len(reps), len(classes)), np.float32)
    target_indices = {index for index, item in enumerate(classes) if item['id'] in config['depth_gate_classes']}
    for view in views:
        blocked = np.zeros(len(reps), bool)
        gate_ids, gate_pixels, gate_ranges, gate_world, gate_labels = [], [], [], [], []
        for start in range(0, len(reps), 50_000):
            world = reps['xyz'][start:start+50_000]
            camera_points = camera_pose.from_world(world)
            camera_uv, camera_range, camera_valid = camera.project(camera_points)
            visible = depth_buffer.visible(camera_uv*np.array([depth_buffer.width/camera.width, depth_buffer.height/camera.height]),
                                           camera_range, camera_valid, config['visibility_abs'], config['visibility_rel'])
            visible &= mirror_states[start:start+50_000] != 2
            uv, _, projected = view.project(camera_points)
            ids = np.flatnonzero(visible & projected)
            pixels = np.floor(uv[ids]*np.array(labels.shape[::-1])/view.size).astype(int)
            pixels[:, 0] = pixels[:, 0].clip(0, labels.shape[1]-1)
            pixels[:, 1] = pixels[:, 1].clip(0, labels.shape[0]-1)
            chosen = labels[pixels[:, 1], pixels[:, 0]]
            take = np.isin(chosen, list(target_indices))
            if take.any():
                gate_ids.append(start+ids[take]); gate_pixels.append(pixels[take]); gate_ranges.append(camera_range[ids[take]])
                gate_world.append(world[ids[take]]); gate_labels.append(chosen[take])
        if gate_ids:
            local, _ = pipeline_run.depth_gate_representatives(
                labels, np.concatenate(gate_pixels), np.concatenate(gate_ranges), np.concatenate(gate_world),
                np.concatenate(gate_labels), classes, config)
            blocked[np.concatenate(gate_ids)[local]] = True
        for start in range(0, len(reps), 50_000):
            world = reps['xyz'][start:start+50_000]
            camera_points = camera_pose.from_world(world)
            camera_uv, camera_range, camera_valid = camera.project(camera_points)
            visible = depth_buffer.visible(camera_uv*np.array([depth_buffer.width/camera.width, depth_buffer.height/camera.height]),
                                           camera_range, camera_valid, config['visibility_abs'], config['visibility_rel'])
            visible &= mirror_states[start:start+50_000] != 2
            uv, _, projected = view.project(camera_points)
            ids = np.flatnonzero(visible & projected)
            pixels = np.floor(uv[ids]*np.array(labels.shape[::-1])/view.size).astype(int)
            pixels[:, 0] = pixels[:, 0].clip(0, labels.shape[1]-1)
            pixels[:, 1] = pixels[:, 1].clip(0, labels.shape[0]-1)
            chosen = labels[pixels[:, 1], pixels[:, 0]]
            accepted = (chosen >= 0) & ~blocked[start+ids]
            weights[start+ids[accepted], chosen[accepted]] = np.maximum(
                weights[start+ids[accepted], chosen[accepted]], confidence[pixels[accepted, 1], pixels[accepted, 0]])
    return weights


def test_classify_scan_projection_cache_matches_old_path_with_door_and_mirrors(tmp_path, monkeypatch):
    count = 24
    dtype = np.dtype([('key', 'V24'), ('xyz', '<f8', 3)])
    reps = np.zeros(count, dtype=dtype)
    reps['key'] = np.arange(count, dtype='<i8').view('V8').repeat(3).view('V24')
    camera = Camera('sphericalRepresentation', {'imageWidth': 32, 'imageHeight': 16,
                    'pixelWidth': 2*np.pi/32, 'pixelHeight': np.pi/16})
    grid = np.array([(x+.5, y+.5) for y in range(4, 8) for x in range(6)], float)
    reps['xyz'] = 3*camera.rays(grid)
    scan_dir = tmp_path/'scan_0001'; scan_dir.mkdir()
    rep_path = scan_dir/'representatives.npy'; np.save(rep_path, reps)
    pose = {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.},
            'translation': {'x': 0., 'y': 0., 'z': 0.}}
    store = SimpleNamespace(meta={'scan_index': 1, 'metadata': {'pose': pose}})
    inspection = {'scan_count': 1, 'image_count': 1, 'scans': [{'image_indices': [1]}],
                  'images': [{'index': 1, 'pose_status': 'valid', 'representations': ['sphericalRepresentation'],
                              'metadata': {'pose': pose}}]}
    classes = [{'id': 'door', 'prompts': ('door',), 'min_mask_conf': .5}]
    config = {**DEFAULTS, 'voxel_origin': [0., 0., 0.], 'voxel_size': .02, 'depth_width': 32,
              'diagnostics': 'off', 'output': str(tmp_path), 'boundary_margin': 0,
              'depth_gate_classes': ['door'], 'door_max_width': 100., 'door_max_height': 100.}
    views = [_ProjectionView('first'), _ProjectionView('second')]
    labels = np.zeros((16, 16), np.int16)
    confidence = np.full((16, 16), .9, np.float32)
    depth_buffer = _ProjectionDepthBuffer()
    control = _ProjectionControl()
    monkeypatch.setattr(pipeline_run, 'representatives', lambda *args: (rep_path, count))
    monkeypatch.setattr(pipeline_run, 'image_assets', lambda *args: (camera, np.zeros((16, 16, 3), np.uint8), None, 'image', None))
    monkeypatch.setattr(pipeline_run, 'camera_depth', lambda *args: depth_buffer)
    monkeypatch.setattr(pipeline_run, 'plan_views', lambda *args: views)
    monkeypatch.setattr(pipeline_run, 'render_view', lambda *args: (np.zeros((16, 16, 3), np.uint8), np.ones((16, 16), bool)))
    monkeypatch.setattr(pipeline_run, 'segment_view', lambda *args, **kwargs: ([confidence], {'hits': 1, 'misses': 0}))
    monkeypatch.setattr(pipeline_run.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    _, weights_path, stats = pipeline_run.classify_scan(None, store, inspection, config, tmp_path, None, None,
                                                        'model', None, control, classes, classes, 'identity', _ProjectionMirrors())
    actual = np.load(weights_path, allow_pickle=False)
    states = _ProjectionMirrors.states(reps['xyz'])
    expected = _old_image_vote_reference(reps, camera, Pose.from_dict(pose), depth_buffer, views, labels, confidence,
                                         classes, config, states)
    np.testing.assert_array_equal(actual, expected)
    assert stats['depth_gate']['components'] == len(views)
    assert all(not path.exists() for path in scan_dir.glob('projection_*.npy'))
    assert len([event for event in control.events if event[0] == 'progress']) == len(views)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get('NOMIVOX_PERF_BENCHMARK'), reason='set NOMIVOX_PERF_BENCHMARK=1 to run benchmarks')
def test_image_projection_cache_benchmark():
    """Report CPU time for the former and shared-projection paths at production scale."""
    count = 200_000
    rng = np.random.default_rng(20261020)
    points = rng.normal(size=(count, 3))
    camera = Camera('sphericalRepresentation', {'imageWidth': 4096, 'imageHeight': 2048,
                    'pixelWidth': 2*np.pi/4096, 'pixelHeight': np.pi/2048})
    views = pipeline_run.plan_views(camera, 'balanced', 1512, 95)
    assert len(views) == 17
    pose = Pose(np.eye(3), np.zeros(3))
    depth = _ProjectionDepthBuffer()
    valid = np.ones(count, bool)
    started = time.process_time()
    for view in views:
        camera_points = pose.from_world(points)
        uv, ranges, projected = camera.project(camera_points)
        visible = depth.visible(uv*np.array([depth.width/camera.width, depth.height/camera.height]), ranges, projected, .03, .01)
        view.project(camera_points)
        assert visible.shape == valid.shape
    old_seconds = time.process_time()-started
    started = time.process_time()
    camera_points = pose.from_world(points)
    uv, ranges, projected = camera.project(camera_points)
    visible = depth.visible(uv*np.array([depth.width/camera.width, depth.height/camera.height]), ranges, projected, .03, .01)
    for view in views:
        view.project(camera_points)
    shared_seconds = time.process_time()-started
    print(f'image projection benchmark (17 views, 200000 representatives): old={old_seconds:.2f}s shared={shared_seconds:.2f}s')
    assert visible.shape == valid.shape
