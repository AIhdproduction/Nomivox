"""File-SQLite equivalence and scale guards for shape-rule post-processing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import psutil
import pytest

from e57cls.pipeline.config import DEFAULTS
from e57cls.vote.engine import VoteStore, voxel_keys
from tests.shape_hslu_fixture import build_hslu_shape_fixture

SHIFT = np.asarray((2665953.0, 1207269.0, 440.0))
ANGLE = np.deg2rad(23.0)
ROTATION = np.asarray(((np.cos(ANGLE), -np.sin(ANGLE), 0.), (np.sin(ANGLE), np.cos(ANGLE), 0.), (0., 0., 1.)))
BY_ID = {'wall': 100, 'ceiling': 101, 'floor': 102, 'column': 108, 'beam': 109,
         'steel_structure': 110, 'pipe': 111, 'cable': 112, 'electrical': 113, 'hvac': 114,
         'ceiling_equipment': 116, 'furniture': 117, 'sanitary_fixtures': 63, 'railing': 107,
         'construction_material': 76, 'stairs': 105, 'door': 103, 'window': 104, 'reflection': 68,
         'building': 6, 'ground': 2, 'road': 11, 'sidewalk': 73, 'curb': 74, 'outdoor_wall': 78,
         'utility_pole': 84, 'traffic_sign': 81, 'fence': 77, 'vehicle': 91, 'power_line': 14,
         'tree': 5, 'tree_trunk': 95}
CODES = [1, *dict.fromkeys(BY_ID.values())]
TIMING_KEYS = {'layer_sigma_s', 'layer_s', 'candidates_s', 'objects_s', 'contacts_s', 'object_apply_s',
               'facade_apply_s', 'outdoor_apply_s', 'tree_apply_s', 'paved_apply_s'}
INDOOR_KEYS = ('layer_sigma_s', 'layer_s', 'candidates_s', 'objects_s', 'contacts_s', 'object_apply_s', 'facade_apply_s')


def _config():
    return {**DEFAULTS, 'geometry_tile_size': 2., 'geometry_halo': .3, 'geometry_knn_radius': .5}


def _world(local):
    return np.asarray(local, np.float64) @ ROTATION.T+SHIFT


def _frozen_planes(path, count):
    """Write ~1,300 rotated wall planes once, independent of either variant."""
    cells = np.asarray([(x, z) for x in range(41) for z in range(61)], np.int32)
    normal = ROTATION @ np.asarray((0., 1., 0.))
    basis = np.asarray((ROTATION @ np.asarray((1., 0., 0.)), ROTATION @ np.asarray((0., 0., 1.))))
    planes, data, columns = [], {}, int(np.ceil(np.sqrt(count)))
    for index in range(count):
        center = _world(((index % columns)*2.7, (index // columns)*3.1, (index % 3)*3.))
        span = np.abs(basis[0])*1.2+np.abs(basis[1])*1.7+np.abs(normal)*.12
        key = f'wall_{index}'
        planes.append({'class_id': 'wall', 'normal': normal.tolist(), 'center': center.tolist(),
                       'basis': basis.tolist(), 'area_m2': 6., 'cells': key,
                       'bbox_min': (center-span).tolist(), 'bbox_max': (center+span).tolist()})
        data[key] = cells
    np.savez(path, metadata=json.dumps({'cell_size': .05, 'planes': planes}), **data)
    return path


def _weights(winners):
    values = np.zeros((len(winners), len(CODES)), np.float32)
    for code in np.unique(winners):
        values[winners == code, CODES.index(int(code))] = 1.
    return values


def _indoor_points(scale):
    """Irregular three-storey rooms with noisy perforated 2--4 cm walls and 40k object mix."""
    rng, n = np.random.default_rng(16016), max(120, int(np.ceil(40_000*scale)))
    side, sequence = int(np.ceil(np.sqrt(n))), np.arange(n)
    objects = np.column_stack(((sequence % side)*.14+rng.uniform(-.018, .018, n),
                               (sequence // side)*.17+rng.uniform(-.018, .018, n),
                               (sequence % 3)*3.+.12+rng.uniform(0., 1.9, n)))
    mix = np.asarray((BY_ID['furniture'], BY_ID['construction_material'], BY_ID['pipe'],
                      BY_ID['ceiling_equipment'], BY_ID['hvac'], BY_ID['wall']))
    wall_n = max(1200, int(36_000*np.sqrt(scale)))
    wx, wz = rng.uniform(0., max(8., side*.14), wall_n), rng.uniform(0., 8.8, wall_n)
    wy = rng.integers(0, 7, wall_n)*4.1+rng.normal(0., .03, wall_n)
    keep = ((np.rint(wx/.02).astype(np.int64)+3*np.rint(wz/.02).astype(np.int64)) % 9) != 0
    walls = np.column_stack((wx[keep], wy[keep], wz[keep]))
    surface_n = max(900, int(8_000*np.sqrt(scale)))
    sx, sy = rng.uniform(0., max(8., side*.14), surface_n), rng.uniform(0., 18., surface_n)
    floors = np.column_stack((sx, sy, rng.choice((0., 3., 6.), surface_n)))
    return np.vstack((objects, walls, floors, floors+np.asarray((0., 0., 2.6)))), np.r_[mix[sequence % len(mix)],
        np.full(len(walls), BY_ID['wall']), np.full(len(floors), BY_ID['floor']), np.full(len(floors), BY_ID['ceiling'])]


def _outdoor_points(scale):
    """Clustered alleys, parking and forest, 2,000 stems, sloped terrain, road and sidewalk."""
    rng = np.random.default_rng(16017)
    n, stems_n = max(180, int(np.ceil(50_000*scale))), max(20, int(np.ceil(2_000*scale)))
    sequence, side = np.arange(n), int(np.ceil(np.sqrt(n)))
    cluster = sequence % 3
    x, y = (sequence % side)*.22+rng.normal(0., .025, n)+cluster*35., (sequence // side)*.27+rng.normal(0., .035, n)+(cluster == 2)*28.
    objects = np.column_stack((x, y, .025*x+.012*y+rng.uniform(.03, 1.8, n)))
    mix = np.asarray((BY_ID['vehicle'], BY_ID['fence'], BY_ID['traffic_sign'], BY_ID['outdoor_wall'], 1))
    trunk_id = np.arange(stems_n)
    sx, sy = (trunk_id % 50)*1.2+70.+rng.uniform(-.14, .14, stems_n), (trunk_id // 50)*1.3+15.+rng.uniform(-.14, .14, stems_n)
    # Four-centimetre vertical continuity is required for the same object-grid
    # connected-component path used by real trunk evidence.
    base, levels = .025*sx+.012*sy, np.arange(0., 3.201, .04)
    stem_angles = np.linspace(0., 2*np.pi, 8, endpoint=False)
    stem_z, stem_angle = np.meshgrid(levels, stem_angles, indexing='ij')
    stems = np.column_stack((np.repeat(sx, stem_z.size)+np.tile(.06*np.cos(stem_angle).ravel(), stems_n),
                             np.repeat(sy, stem_z.size)+np.tile(.06*np.sin(stem_angle).ravel(), stems_n),
                             np.repeat(base, stem_z.size)+np.tile(stem_z.ravel(), stems_n)))
    angles = np.linspace(0., 2*np.pi, 12, endpoint=False)
    crowns = np.column_stack((np.repeat(sx, len(angles))+np.tile(np.cos(angles)*1.5, stems_n),
                              np.repeat(sy, len(angles))+np.tile(np.sin(angles)*1.5, stems_n), np.repeat(base+4.2, len(angles))))
    road_x = np.linspace(-5., 105., max(400, int(4_000*np.sqrt(scale))))
    road = np.column_stack((road_x, np.full(len(road_x), -2.), .025*road_x-.024))
    sidewalk = road+np.asarray((0., .4, .12))
    return np.vstack((objects, stems, crowns, road, sidewalk)), np.r_[mix[sequence % len(mix)],
        np.full(len(stems), BY_ID['tree_trunk']), np.full(len(crowns), BY_ID['tree']),
        np.full(len(road), BY_ID['road']), np.full(len(sidewalk), BY_ID['sidewalk'])]


def build_shape_fixture(tmp_path, part, scale=1., name='fixture'):
    """Vectorised, deterministic fixture factory used by fast, slow and profile runs."""
    points, winners = _indoor_points(scale) if part == 'indoor' else _outdoor_points(scale)
    world = _world(points)
    origin = np.floor(world.min(axis=0)/.02)*.02-.04
    keys = voxel_keys(world, origin, .02)
    order = np.argsort(keys, kind='stable')
    selected = order[np.r_[True, keys[order][1:] != keys[order][:-1]]]
    store = VoteStore(tmp_path/f'{name}_{part}.sqlite')
    store.add_scan(1, '1', [(keys[selected], _weights(winners[selected]))], CODES)
    planes = _frozen_planes(tmp_path/f'{name}_{part}_planes.npz', max(3, int(np.ceil(1300*scale))))
    scene = 'indoor' if part == 'indoor' else 'outdoor'
    context = {1: {'scene': scene, 'outdoor': part == 'outdoor', 'position': _world(((3., -3., 1.6),))[0].tolist(),
                   'allowed_codes': set(CODES)}}
    return store, origin, context, planes


def _baseline_module(tmp_path):
    try:
        source = subprocess.check_output(['git', 'show', '4c0ba4c:e57cls/vote/shape_rules.py'], text=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git baseline 4c0ba4c is unavailable')
    path = tmp_path/'shape_rules_baseline.py'; path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('shape_rules_baseline', path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _digest(store):
    digest = hashlib.sha256()
    for key, code, confidence in store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key'):
        digest.update(bytes(key)); digest.update(np.asarray((code, confidence), np.float64).tobytes())
    return digest.hexdigest()


def run_shape_benchmark(module, tmp_path, part, scale=1., name='run', fixture=build_shape_fixture):
    store, origin, context, planes = fixture(tmp_path, part, scale, name)
    process, start, peak = psutil.Process(), psutil.Process().memory_info().rss, [0]
    def checkpoint(): peak[0] = max(peak[0], process.memory_info().rss)
    try:
        store.decide(1, .5, CODES); checkpoint()
        report = module.apply_shape_rules(store, BY_ID, origin, .02, _config(), active_codes=set(CODES),
                                          scan_context=context, planes_path=planes, checkpoint=checkpoint)
        checkpoint()
        return {'timings': report['timings'], 'digest': _digest(store), 'rss_delta': max(0, peak[0]-start), 'result': report}
    finally:
        store.close()


def test_shape_rules_fast_georeferenced_baseline_equivalence(tmp_path):
    old = run_shape_benchmark(_baseline_module(tmp_path), tmp_path, 'indoor', .003, 'old')
    new = run_shape_benchmark(__import__('e57cls.vote.shape_rules', fromlist=['apply_shape_rules']), tmp_path, 'indoor', .003, 'new')
    assert new['digest'] == old['digest']


def test_shape_rules_fast_mixed_outdoor_baseline_equivalence(tmp_path):
    old = run_shape_benchmark(_baseline_module(tmp_path), tmp_path, 'outdoor', .004, 'old')
    new = run_shape_benchmark(__import__('e57cls.vote.shape_rules', fromlist=['apply_shape_rules']), tmp_path, 'outdoor', .004, 'new')
    assert new['digest'] == old['digest']


def test_shape_rules_timings_keep_public_keys(tmp_path):
    assert TIMING_KEYS <= set(run_shape_benchmark(__import__('e57cls.vote.shape_rules', fromlist=['apply_shape_rules']), tmp_path, 'indoor', .002)['timings'])


# Speedup targets follow with the planned in-memory shape-rule pipeline; this benchmark guards equality and memory.
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1', reason='set NOMIVOX_RUN_SLOW=1')
def test_shape_rules_indoor_file_sqlite_benchmark(tmp_path):
    scale = float(os.environ.get('NOMIVOX_SHAPE_BENCH_SCALE', '.05'))
    fixture = lambda root, part, value, name: build_hslu_shape_fixture(root, part, value, BY_ID, CODES, name)
    old = run_shape_benchmark(_baseline_module(tmp_path), tmp_path, 'indoor', scale, 'old', fixture)
    new = run_shape_benchmark(__import__('e57cls.vote.shape_rules', fromlist=['apply_shape_rules']), tmp_path, 'indoor', scale, 'new', fixture)
    assert new['digest'] == old['digest']
    for key in sorted(set(old['timings']) | set(new['timings'])):
        print(f'{key}: old={old["timings"].get(key)} new={new["timings"].get(key)}')
    assert new['rss_delta'] <= old['rss_delta'] + 100*2**20


# Speedup targets follow with the planned in-memory shape-rule pipeline; this benchmark guards equality and memory.
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1', reason='set NOMIVOX_RUN_SLOW=1')
def test_shape_rules_outdoor_file_sqlite_benchmark(tmp_path):
    scale = float(os.environ.get('NOMIVOX_SHAPE_BENCH_SCALE', '.05'))
    fixture = lambda root, part, value, name: build_hslu_shape_fixture(root, part, value, BY_ID, CODES, name)
    old = run_shape_benchmark(_baseline_module(tmp_path), tmp_path, 'outdoor', scale, 'old', fixture)
    new = run_shape_benchmark(__import__('e57cls.vote.shape_rules', fromlist=['apply_shape_rules']), tmp_path, 'outdoor', scale, 'new', fixture)
    assert new['digest'] == old['digest']
    for key in sorted(set(old['timings']) | set(new['timings'])):
        print(f'{key}: old={old["timings"].get(key)} new={new["timings"].get(key)}')
    assert new['rss_delta'] <= old['rss_delta'] + 100*2**20
