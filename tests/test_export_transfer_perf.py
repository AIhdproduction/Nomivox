"""Regression and performance coverage for spatially batched export transfer."""

import cProfile
import importlib.util
import io
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import tempfile
import threading
import time

import laspy
import numpy as np
import psutil
import pytest

from e57cls.io.store import ScanStore
from e57cls.output import las as las_output
from e57cls.mirror.pipeline import MirrorAnalysis
from e57cls.output.classify import ClassifiedSource, classify_points
from e57cls.output.las import write_cloud
from e57cls.vote.engine import VoteStore, voxel_keys


_ORIGIN = np.asarray((2665953., 1207269., 440.))


def _store(tmp_path, scan, local, translation):
    valid = np.ones(len(local), bool)
    np.save(tmp_path/f'xyz_{scan}.npy', local)
    np.save(tmp_path/f'valid_{scan}.npy', valid)
    angle = np.deg2rad(23.)/2
    pose = {'rotation': {'w': float(np.cos(angle)), 'x': 0., 'y': 0., 'z': float(np.sin(angle))},
            'translation': {'x': float(translation[0]), 'y': float(translation[1]), 'z': float(translation[2])}}
    return ScanStore(tmp_path, {'record_count': len(local), 'valid_count': len(local), 'scan_index': scan,
        'arrays': {'xyz_local': {'path': f'xyz_{scan}.npy'}, 'valid': {'path': f'valid_{scan}.npy'}},
        'metadata': {'pose': pose, 'points': {'fields': []}},
        'bounds_local': {'min': local.min(axis=0).tolist(), 'max': local.max(axis=0).tolist()}})


def _overlapping_stores(tmp_path, scans=6, pairs=1_600, seed=20261025):
    rng = np.random.default_rng(seed)
    direct = rng.uniform((-5., -5., -1.), (5., 5., 2.), (pairs, 3))
    unknown = direct+rng.normal(0., .004, direct.shape)
    local = np.vstack((direct, unknown))
    return [_store(tmp_path, scan, local, _ORIGIN+np.asarray((.001*scan, -.0015*scan, 0.)))
            for scan in range(1, scans+1)]


class _PairedVotes:
    def lookup(self, keys):
        count = len(keys)
        half = count//2
        return np.r_[np.full(half, 2, np.uint8), np.ones(count-half, np.uint8)], np.r_[
            np.full(half, 200, np.uint8), np.zeros(count-half, np.uint8)]

    def lookup_station_ids(self, keys):
        return np.ones(len(keys), np.uint32)


class _PairedVoteStore(VoteStore):
    """File-backed VoteStore used by the slow benchmark's deterministic oracle."""
    lookup_is_key_pure = False

    def lookup(self, keys):
        return _PairedVotes().lookup(keys)


def _lookup_mirror_fixture(tmp_path, points_per_voxel=1_000):
    """Build repeated LV95 voxels and persisted mirror-mask states."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    local = np.repeat(np.asarray(((-.004, -.004, 0.), (.016, -.004, 0.),
                                  (-.004, .016, 0.), (.016, .016, 0.))), points_per_voxel, axis=0)
    store = _store(tmp_path, 81, local, _ORIGIN)
    masks = MirrorAnalysis(tmp_path/'mirrors', {'mirror_plane_tolerance': .03})
    states = np.zeros(len(local), np.uint8)
    states[::3] = 2
    state_path = tmp_path/'mirrors'/'states.npy'
    np.save(state_path, states)
    masks.paths[81] = state_path
    votes = VoteStore(tmp_path/'lookup.sqlite')
    world = _ORIGIN+local
    keys = voxel_keys(world, _ORIGIN, .02)
    unique = np.unique(keys)
    votes.db.executemany('INSERT INTO decisions(key,code,confidence) VALUES(?,?,?)',
                         ((bytes(key), 2+index, 180+index) for index, key in enumerate(unique)))
    votes.db.commit()
    return store, local, votes, masks.output


def test_lookup_mirror_coalesces_equal_voxel_lookups(tmp_path):
    """VoteStore lookup is once per voxel while persisted mirror states stay exact."""
    store, xyz, votes, mirror = _lookup_mirror_fixture(tmp_path)
    valid = np.ones(len(xyz), bool)
    original = votes.lookup
    calls = []
    def counted(keys):
        calls.append(len(keys))
        return original(keys)
    votes.lookup = counted
    try:
        world, codes, confidence, states = classify_points(store, 0, len(xyz), xyz, valid, votes, _ORIGIN, .02,
                                                            mirror=mirror)
        assert calls == [4]
        assert np.count_nonzero(states == 2) == len(xyz[::3])
        assert np.all(codes[states == 2] == 68)
        votes.lookup_is_key_pure = False
        calls.clear()
        legacy = classify_points(store, 0, len(xyz), xyz, valid, votes, _ORIGIN, .02, mirror=mirror)
        assert calls == [len(xyz)]
        for actual, expected in zip((world, codes, confidence, states), legacy):
            np.testing.assert_array_equal(actual, expected)
    finally:
        votes.close()


def test_transfer_bucket_buffer_coalesces_leaf_appends(tmp_path):
    """Several chunks for one leaf need one append open while they fit in RAM."""
    source = ClassifiedSource(tmp_path, (), object(), _ORIGIN, .02, radius=.02)
    source.temporary = tempfile.TemporaryDirectory(prefix='transfer_buffer_', dir=tmp_path)
    dtype = source._transfer_record_dtype
    records = np.zeros(11, dtype=dtype)
    records['scan'] = 1
    roots = np.zeros((len(records), 3), np.int64)
    paths = np.ones(len(records), np.uint64)
    try:
        source._buffer_transfer_records(records[:5], roots[:5], paths[:5])
        source._buffer_transfer_records(records[5:], roots[5:], paths[5:])
        source._flush_transfer_buffers()
        files = list(Path(source.temporary.name).glob('queries_*.bin'))
        assert len(files) == 1
        assert source.transfer_bucket_stats['transfer_bucket_open_calls'] == 1
        assert files[0].stat().st_size == len(records)*dtype.itemsize
    finally:
        source.close()


def _legacy_classify_module(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git is required to load the 4c0ba4c transfer oracle')
    source = subprocess.run(['git', 'show', '4c0ba4c:e57cls/output/classify.py'], cwd=Path(__file__).parents[1],
                            check=False, capture_output=True, text=True).stdout
    if not source:
        pytest.skip('4c0ba4c classify.py is unavailable')
    path = tmp_path/'legacy_classify.py'
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('legacy_export_classify', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_classified_source(tmp_path):
    return _legacy_classify_module(tmp_path).ClassifiedSource


def _write_with_source(monkeypatch, source, path, stores, votes=None, cache_gb=.00001, **kwargs):
    monkeypatch.setattr(las_output, 'ClassifiedSource', source)
    try:
        return write_cloud(path, stores, votes or _PairedVotes(), _ORIGIN, .02, '', [1, 2],
                           class_transfer_radius=.02, export_index_cache_gb=cache_gb,
                           export_index_max_tile_points=1_000_000, **kwargs)
    finally:
        monkeypatch.setattr(las_output, 'ClassifiedSource', ClassifiedSource)


def test_batched_reduced_transfer_is_byte_identical_and_cache_local(tmp_path, monkeypatch):
    """Six rotated LV95 scans exercise the scan-order cache-thrashing pattern."""
    stores = _overlapping_stores(tmp_path)
    legacy = _legacy_classified_source(tmp_path)
    old = _write_with_source(monkeypatch, legacy, tmp_path/'old.las', stores)
    new = _write_with_source(monkeypatch, ClassifiedSource, tmp_path/'new.las', stores)
    assert (tmp_path/'new.las').read_bytes() == (tmp_path/'old.las').read_bytes()
    angle = np.deg2rad(23.)
    rotation = np.asarray(((np.cos(angle), -np.sin(angle), 0.),
                           (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    world = np.vstack([store.read('xyz_local', 0, store.meta['record_count'])@rotation.T+
                       np.asarray([store.meta['metadata']['pose']['translation'][axis] for axis in ('x', 'y', 'z')])
                       for store in stores])
    leaves = max(1, len(set(map(tuple, np.floor((world-_ORIGIN)/2).astype(np.int64)))))
    assert (new['index_cache_misses'] <= leaves or
            new['index_cache_misses'] <= old['index_cache_misses']//10)


@pytest.mark.parametrize('stations', (None, {1: np.asarray([_ORIGIN])}))
def test_batched_transfer_keeps_adaptive_radius_codes_identical(tmp_path, monkeypatch, stations):
    stores = _overlapping_stores(tmp_path, scans=6, pairs=700)
    legacy = _legacy_classified_source(tmp_path)
    options = {'class_transfer_range_factor': .003, 'class_transfer_radius_max': .04,
               'transfer_station_positions': stations}
    _write_with_source(monkeypatch, legacy, tmp_path/'old.las', stores, **options)
    _write_with_source(monkeypatch, ClassifiedSource, tmp_path/'new.las', stores, **options)
    old, new = laspy.read(tmp_path/'old.las'), laspy.read(tmp_path/'new.las')
    np.testing.assert_array_equal(new.classification, old.classification)
    np.testing.assert_array_equal(new.class_conf, old.class_conf)
    np.testing.assert_array_equal(new.mirror_state, old.mirror_state)


def test_transfer_prepare_exception_releases_temp_resources(tmp_path, monkeypatch):
    stores = _overlapping_stores(tmp_path, scans=1, pairs=64)
    def broken_transfer(self):
        raise RuntimeError('synthetic transfer failure')
    monkeypatch.setattr(ClassifiedSource, '_run_transfer_buckets', broken_transfer)
    with pytest.raises(RuntimeError, match='synthetic transfer failure'):
        with ClassifiedSource(tmp_path, stores, _PairedVotes(), _ORIGIN, .02, radius=.02):
            pass
    assert not list(tmp_path.glob('e57cls_class_neighbours_*'))


def _measure(call):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    try:
        profiler.enable()
        result = call()
    finally:
        profiler.disable()
        elapsed = time.perf_counter()-started
        stop.set()
        watcher.join()
    listing = io.StringIO()
    pstats.Stats(profiler, stream=listing).sort_stats('cumulative').print_stats(20)
    return result, elapsed, max(0, peak[0]-baseline), listing.getvalue()


def _measure_lookup_mirror(classifier, tmp_path):
    store, xyz, votes, mirror = _lookup_mirror_fixture(tmp_path, points_per_voxel=125_000)
    valid = np.ones(len(xyz), bool)
    try:
        return _measure(lambda: classifier(store, 0, len(xyz), xyz, valid, votes, _ORIGIN, .02, mirror=mirror))
    finally:
        votes.close()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1', reason='set NOMIVOX_RUN_SLOW=1 to run export benchmark')
def test_batched_transfer_benchmark(tmp_path, monkeypatch):
    """9m realistic TLS points; 4 GB/roughly 300 GB is modeled as 12 MiB/900 MiB."""
    stores = _overlapping_stores(tmp_path, scans=30, pairs=150_000, seed=20261026)
    legacy_module = _legacy_classify_module(tmp_path)
    legacy = legacy_module.ClassifiedSource
    votes = _PairedVoteStore(tmp_path/'x.sqlite')
    def write(source, name):
        return _write_with_source(monkeypatch, source, tmp_path/name, stores, votes=votes, cache_gb=.012)
    try:
        old, old_seconds, old_rss, old_profile = _measure(lambda: write(legacy, 'old.las'))
        new, new_seconds, new_rss, new_profile = _measure(lambda: write(ClassifiedSource, 'new.las'))
    finally:
        votes.close()
    old_lookup, old_lookup_seconds, old_lookup_rss, _ = _measure_lookup_mirror(
        legacy_module.classify_points, tmp_path/'lookup_old')
    new_lookup, new_lookup_seconds, new_lookup_rss, _ = _measure_lookup_mirror(
        classify_points, tmp_path/'lookup_new')
    assert (tmp_path/'new.las').read_bytes() == (tmp_path/'old.las').read_bytes()
    for actual, expected in zip(new_lookup, old_lookup):
        np.testing.assert_array_equal(actual, expected)
    assert new['timing_seconds']['transfer'] <= old['timing_seconds']['transfer']/2
    assert new['index_cache_misses'] <= old['index_cache_misses']//10
    assert new_rss <= 1024**3
    assert new_lookup_rss <= 1024**3
    print('export transfer benchmark: '
          f'old_total={old_seconds:.2f}s new_total={new_seconds:.2f}s '
          f'old_transfer={old["timing_seconds"]["transfer"]:.2f}s '
          f'new_transfer={new["timing_seconds"]["transfer"]:.2f}s '
          f'old_lookup_mirror={old["timing_seconds"]["lookup_mirror"]:.2f}s '
          f'new_lookup_mirror={new["timing_seconds"]["lookup_mirror"]:.2f}s '
          f'old_lookup_mirror_probe={old_lookup_seconds:.2f}s '
          f'new_lookup_mirror_probe={new_lookup_seconds:.2f}s '
          f'old_lookup_rss_delta={old_lookup_rss/1024**2:.1f}MiB '
          f'new_lookup_rss_delta={new_lookup_rss/1024**2:.1f}MiB '
          f'old_cache_hits={old["index_cache_hits"]} old_cache_misses={old["index_cache_misses"]} '
          f'old_cache_evictions={old["index_cache_evictions"]} '
          f'new_cache_hits={new["index_cache_hits"]} new_cache_misses={new["index_cache_misses"]} '
          f'new_cache_evictions={new["index_cache_evictions"]} '
          f'old_bucket_files={old.get("transfer_bucket_files", 0)} '
          f'old_bucket_opens={old.get("transfer_bucket_open_calls", 0)} '
          f'new_bucket_files={new.get("transfer_bucket_files", 0)} '
          f'new_bucket_opens={new.get("transfer_bucket_open_calls", 0)} '
          f'old_rss_delta={old_rss/1024**2:.1f}MiB new_rss_delta={new_rss/1024**2:.1f}MiB')
    print('old cProfile top 20:\n'+old_profile)
    print('new cProfile top 20:\n'+new_profile)
