"""Focused regressions for the shared building-outline scene signal."""

import numpy as np

from e57cls.pipeline.config import DEFAULTS
from e57cls.vote.engine import VoteStore
from e57cls.vote.scene_context import ensure_scene_context


GEO_ORIGIN = np.asarray((2_700_000., 1_200_000., 400.))
ROTATION_30 = np.asarray(((np.cos(np.deg2rad(30.)), -np.sin(np.deg2rad(30.)), 0.),
                          (np.sin(np.deg2rad(30.)), np.cos(np.deg2rad(30.)), 0.),
                          (0., 0., 1.)))


def _key(x, y, z):
    return np.asarray((x, y, z), dtype='<i8').tobytes()


def _ring(store, code=6, size=5, height=4):
    for x in range(size):
        for y in (0, size-1):
            for z in range(height):
                store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(x, y, z), code, 255))
    for y in range(1, size-1):
        for x in (0, size-1):
            for z in range(height):
                store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(x, y, z), code, 255))


def _fine_ring(store, code=6, size=41, height=14):
    for x in range(size):
        for y in (0, size-1):
            for z in range(height):
                store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(x, y, z), code, 255))
    for y in range(1, size-1):
        for x in (0, size-1):
            for z in range(height):
                store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(x, y, z), code, 255))


def _scan_position(local):
    return (GEO_ORIGIN+ROTATION_30@np.asarray(local, np.float64)).tolist()


def _observe(store, scan, *points):
    store.db.executemany('INSERT OR IGNORE INTO scan_observations VALUES (?,?)',
                         ((_key(*point), int(scan)) for point in points))


def _context(store, outline, outdoor=True, enabled=True, scans=None, voxel_size=1.):
    config = {**DEFAULTS, 'building_boundary_cell_size': 1., 'building_boundary_min_height': 2.,
              'scene_outline': enabled}
    if scans is None:
        scans = {1: {'scene': 'outdoor', 'outdoor': True, 'position': _scan_position((8., 2., 1.))}} if outdoor else {}
    return ensure_scene_context(store, {'building': 6, 'wall': 100, 'ceiling': 9}, scans, GEO_ORIGIN, voxel_size, config,
                                lambda: None, outline=outline)


def test_closed_outline_refines_inside_roof_and_outside_context(tmp_path):
    store = VoteStore(tmp_path/'outline.sqlite')
    try:
        _ring(store)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(2, 2, 3), 9, 255))
        for point in ((2, 2, 1), (2, 2, 4), (7, 2, 1)):
            store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*point), 1, 0))
        _observe(store, 1, (2, 2, 4))
        store.db.commit()
        outline = store.building_outline([6, 100], 1., 1., 2., 10_000)
        report = _context(store, outline)
        values = dict(store.db.execute('SELECT key,scene_context FROM geometry_context'))
        assert values[_key(2, 2, 1)] == 0
        assert values[_key(2, 2, 4)] == 1
        assert values[_key(7, 2, 1)] == 1
        assert report['scene_outline']['to_indoor'] >= 1
        # The roof keeps its already outdoor scanner context; only the point
        # outside the envelope is actively forced to outdoor.
        assert report['scene_outline']['to_outdoor'] >= 1
    finally:
        store.close()


def test_unclosed_outline_keeps_scanner_context_unchanged(tmp_path):
    store = VoteStore(tmp_path/'open.sqlite')
    try:
        for y in range(5):
            for z in range(4):
                store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(0, y, z), 6, 255))
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(2, 2, 1), 1, 0))
        store.db.commit()
        outline = store.building_outline([6], 1., 1., 2., 10_000)
        report = _context(store, outline)
        assert not report['scene_outline']['applied']
        assert store.db.execute('SELECT scene_context FROM geometry_context WHERE key=?',
                                (_key(2, 2, 1),)).fetchone()[0] == -1
    finally:
        store.close()


def test_outline_does_not_force_exterior_without_an_outdoor_scan(tmp_path):
    store = VoteStore(tmp_path/'no_outdoor.sqlite')
    try:
        _ring(store)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(8, 2, 1), 1, 0))
        store.db.commit()
        outline = store.building_outline([6], 1., 1., 2., 10_000)
        _context(store, outline, outdoor=False)
        assert store.db.execute('SELECT scene_context FROM geometry_context WHERE key=?',
                                (_key(8, 2, 1),)).fetchone()[0] == -1
    finally:
        store.close()


def test_scene_outline_01_interior_ceiling_never_forces_outdoor_without_outdoor_scan(tmp_path):
    store = VoteStore(tmp_path/'interior_only.sqlite')
    try:
        _ring(store)
        ceiling = (2, 2, 4)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*ceiling), 9, 255))
        _observe(store, 1, ceiling)
        store.db.commit()
        outline = store.building_outline([6, 100], 1., 1., 2., 10_000)
        report = _context(store, outline, scans={
            1: {'scene': 'indoor', 'outdoor': False, 'position': _scan_position((2., 2., 1.))},
        })
        assert store.db.execute('SELECT scene_context FROM geometry_context WHERE key=?',
                                (_key(*ceiling),)).fetchone()[0] == 0
        assert report['scene_outline']['to_outdoor'] == 0
    finally:
        store.close()


def test_scene_outline_02_thick_noisy_roof_keeps_outdoor_context(tmp_path):
    store = VoteStore(tmp_path/'thick_roof.sqlite')
    try:
        _fine_ring(store)
        roof = [(20, 20, level) for level in range(9, 13)]
        for point in roof:
            store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*point), 6, 255))
        _observe(store, 1, *roof)
        store.db.commit()
        outline = store.building_outline([6, 100], .1, 1., .2, 10_000)
        _context(store, outline, voxel_size=.1)
        values = dict(store.db.execute('SELECT key,scene_context FROM geometry_context'))
        assert all(values[_key(*point)] == 1 for point in roof)
    finally:
        store.close()


def test_scene_outline_03_open_courtyard_keeps_outdoor_context(tmp_path):
    store = VoteStore(tmp_path/'courtyard.sqlite')
    try:
        _ring(store)
        courtyard = ((2, 2, 0), (2, 2, 1))
        for point, code in zip(courtyard, (2, 5)):
            store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*point), code, 255))
        _observe(store, 1, *courtyard)
        store.db.commit()
        outline = store.building_outline([6, 100], 1., 1., 2., 10_000)
        _context(store, outline)
        values = dict(store.db.execute('SELECT key,scene_context FROM geometry_context'))
        assert all(values[_key(*point)] == 1 for point in courtyard)
    finally:
        store.close()


def test_scene_outline_04_local_cover_overrides_an_outdoor_window_context(tmp_path):
    store = VoteStore(tmp_path/'covered_window.sqlite')
    try:
        _ring(store)
        below_cover, ceiling = (2, 2, 1), (2, 2, 4)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*below_cover), 1, 0))
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*ceiling), 9, 255))
        _observe(store, 1, below_cover, ceiling)
        store.db.commit()
        outline = store.building_outline([6, 100], 1., 1., 2., 10_000)
        _context(store, outline)
        assert store.db.execute('SELECT scene_context FROM geometry_context WHERE key=?',
                                (_key(*below_cover),)).fetchone()[0] == 0
    finally:
        store.close()


def test_scene_outline_05_local_cover_gap_does_not_use_a_neighbour(tmp_path):
    store = VoteStore(tmp_path/'cover_gap.sqlite')
    try:
        _ring(store)
        uncovered, neighbouring_ceiling = (2, 2, 1), (1, 2, 4)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*uncovered), 1, 0))
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(*neighbouring_ceiling), 9, 255))
        _observe(store, 1, uncovered, neighbouring_ceiling)
        store.db.commit()
        outline = store.building_outline([6, 100], 1., 1., 2., 10_000)
        _context(store, outline)
        assert store.db.execute('SELECT scene_context FROM geometry_context WHERE key=?',
                                (_key(*uncovered),)).fetchone()[0] == 1
    finally:
        store.close()


def test_scene_outline_can_be_disabled_without_touching_decisions(tmp_path):
    store = VoteStore(tmp_path/'disabled.sqlite')
    try:
        _ring(store)
        store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (_key(2, 2, 1), 1, 0))
        store.db.commit()
        before = list(store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key'))
        _context(store, store.building_outline([6], 1., 1., 2., 10_000), enabled=False)
        assert list(store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key')) == before
    finally:
        store.close()
