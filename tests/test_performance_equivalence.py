"""Regression coverage for optimizations whose results must stay exact."""

import os
import sqlite3
import time
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

from e57cls.io.store import ScanStore
from e57cls.mirror.evidence import DirectIndex
from e57cls.output.las import ClassPointIndex, write_cloud
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
