"""Focused file-SQLite coverage for geometry-connected segment voting."""

import os
import time
import numpy as np
import pytest

from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline.reclassify import _postprocess
from e57cls.vote.geometry_rules import apply_geometry_rules
from e57cls.vote.engine import VoteStore
from e57cls.vote.engine import voxel_keys
import e57cls.vote.segments as segment_module
from e57cls.vote.segments import _accumulate_segment_votes, _apply_decisive_segments, _segment_cells, apply_segment_vote


def _key(index):
    return bytes(np.asarray(index, dtype='<i8').reshape(1, 3).view('V24')[0])


def _store(tmp_path, name, values):
    store = VoteStore(tmp_path/f'{name}.sqlite')
    for index, votes in values:
        key = _key(index)
        store.db.execute('INSERT INTO observations VALUES (?,1)', (key,))
        for code, weight in votes:
            store.db.execute('INSERT INTO votes VALUES (?,?,?,?)', (key, code, weight, weight))
    return store


def _config(**overrides):
    return {**DEFAULTS, 'geometry_tile_size': 1., 'geometry_halo': .2, 'geometry_knn_radius': .5,
            'segment_min_share': .6, **overrides}


def _manual_features(store, entries):
    """Install small deterministic feature tables for segmentation-only cases."""
    store.db.executescript('''
        CREATE TEMP TABLE geometry_cells(cx INTEGER,cy INTEGER,cz INTEGER,tx INTEGER,ty INTEGER,tz INTEGER,
            sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE geometry_features(cx INTEGER,cy INTEGER,cz INTEGER,x REAL,y REAL,z REAL,
            linearity REAL,planarity REAL,scatter REAL,verticality REAL,nx REAL,ny REAL,nz REAL,
            hag REAL,covered INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
    ''')
    for cell, planarity, normal, scatter in entries:
        x, y, z = cell
        store.db.execute('INSERT INTO geometry_cells VALUES (?,?,?,?,?,?,?,?,?,1)', (x, y, z, 0, 0, 0, x, y, z))
        store.db.execute('INSERT INTO geometry_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (x, y, z, (x+.5)*.1, (y+.5)*.1, (z+.5)*.1, 0., planarity, scatter, abs(normal[2]),
                          *normal, None, 0))


def _labels_for_features(tmp_path, name, entries):
    store = VoteStore(tmp_path/f'{name}.sqlite')
    _manual_features(store, entries)
    cells, labels, _, _, _ = _segment_cells(store, np.zeros(3), .02, _config(), lambda: None)
    return store, {tuple(cell): int(label) for cell, label in zip(cells, labels)}


def test_segment_vote_flips_weak_pillar_surface(tmp_path):
    values = []
    for x in range(30):
        for z in range(30):
            # A clear segment majority for column, but the last strip has a
            # locally weak wall decision.
            column = x < 24
            values.append(((x, 0, z), [(108, .75 if column else .45), (100, .25 if column else .55)]))
    store = _store(tmp_path, 'pillar', values)
    try:
        store.decide(.4, .5, [100, 108])
        report = apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, _config(), active_codes={100, 108})
        assert report['changed_voxels'] == 180
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=108').fetchone()[0] == 900
    finally:
        store.close()


def test_segment_vote_fills_unclassified_table(tmp_path):
    values = []
    for x in range(30):
        for y in range(12):
            furniture = (x+y) % 5 < 3
            votes = [(117, .7), (100, .3)] if furniture else [(117, .5), (100, .5)]
            values.append(((x, y, 38), votes))
    store = _store(tmp_path, 'table', values)
    try:
        store.decide(.4, .6, [117, 100])
        report = apply_segment_vote(store, {'furniture': 117, 'wall': 100}, np.zeros(3), .02, _config(), active_codes={117, 100})
        assert report['filled_unclassified'] == 144
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=117').fetchone()[0] == 360
    finally:
        store.close()


def test_segment_vote_keeps_protected_and_strong_minorities(tmp_path):
    values = []
    for x in range(30):
        for z in range(30):
            votes = [(100, .7), (108, .3)]
            if x == 0:
                votes = [(200, .95), (100, .05)]
            if x == 1:
                votes = [(300, .55), (100, .45)]
            if x == 2:
                votes = [(68, .55), (100, .45)]
            values.append(((x, 0, z), votes))
    store = _store(tmp_path, 'protected', values)
    try:
        store.decide(.4, .5, [100, 200, 300])
        apply_segment_vote(store, {'wall': 100, 'door': 200, 'stairs': 300, 'reflection': 68}, np.zeros(3), .02,
                           _config(segment_protected_classes=['door']), active_codes={68, 100, 200, 300})
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=200').fetchone()[0] == 30
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=300').fetchone()[0] == 30
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=68').fetchone()[0] == 30
    finally:
        store.close()


def test_segment_vote_off_leaves_decisions_byte_identical(tmp_path):
    values = [((x, 0, z), [(100, .45), (108, .55)]) for x in range(20) for z in range(20)]
    store = _store(tmp_path, 'off', values)
    try:
        store.decide(.4, .5, [100, 108])
        before = store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall()
        report = apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02,
                                    _config(segment_vote=False), active_codes={100, 108})
        assert report['enabled'] is False
        assert store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall() == before
    finally:
        store.close()


def test_segment_vote_keeps_right_angle_corner_separate(tmp_path):
    values = []
    # Horizontal floor and vertical wall share an edge, but not a segment.
    for x in range(120):
        for y in range(60):
            values.append(((x, y, 0), [(102, .8), (100, .2)]))
        for z in range(1, 130):
            values.append(((x, 0, z), [(100, .8), (102, .2)]))
    store = _store(tmp_path, 'corner', values)
    try:
        store.decide(.4, .5, [100, 102])
        report = apply_segment_vote(store, {'wall': 100, 'floor': 102}, np.zeros(3), .02, _config(), active_codes={100, 102})
        assert report['segments'] >= 2
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=102').fetchone()[0] == 7200
    finally:
        store.close()


def test_segment_vote_leaves_mixed_segment_unchanged(tmp_path):
    values = [((x, 0, z), [(100, .5), (108, .5)]) for x in range(30) for z in range(30)]
    store = _store(tmp_path, 'mixed', values)
    try:
        store.decide(.4, .6, [100, 108])
        before = store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall()
        report = apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, _config(), active_codes={100, 108})
        assert report['decisive_segments'] == 0
        assert store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall() == before
    finally:
        store.close()


def test_segment_vote_is_invariant_to_georeferenced_rotation(tmp_path):
    local = np.asarray([(x*.02, y*.02, .75) for x in range(30) for y in range(15)], np.float64)
    angle, tilt = np.deg2rad(30), np.deg2rad(5)
    rz = np.array(((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
    rx = np.array(((1, 0, 0), (0, np.cos(tilt), -np.sin(tilt)), (0, np.sin(tilt), np.cos(tilt))))
    worlds = [local, local@(rx@rz).T+np.asarray((2665953., 1207269., 440.))]
    counts, unique_counts = [], []
    for number, world in enumerate(worlds):
        origin = np.floor(world.min(axis=0)/.02)*.02-.04
        keys = np.unique(voxel_keys(world, origin, .02))
        # Rotation can quantize neighbouring source points into one voxel.
        # Generate exactly one deterministic vote record per retained key.
        values = [(np.frombuffer(bytes(key), '<i8'), [(117, .7), (100, .3)] if index % 5 < 3 else [(117, .5), (100, .5)])
                  for index, key in enumerate(keys)]
        store = _store(tmp_path, f'rotated_{number}', values)
        try:
            store.decide(.4, .6, [117, 100])
            apply_segment_vote(store, {'furniture': 117, 'wall': 100}, origin, .02, _config(), active_codes={117, 100})
            counts.append(store.db.execute('SELECT code,COUNT(*) FROM decisions GROUP BY code').fetchall())
            unique_counts.append(len(keys))
        finally:
            store.close()
    assert counts == [[(117, count)] for count in unique_counts]


def test_segment_vote_features_are_reused_by_geometry_rules(tmp_path, monkeypatch):
    import e57cls.vote.geometry_rules as geometry_rules
    calls, original = [], geometry_rules._make_features
    monkeypatch.setattr(geometry_rules, '_make_features', lambda *args: (calls.append(1), original(*args))[1])
    values = [((x, 0, z), [(108, .75 if x < 24 else .45), (100, .25 if x < 24 else .55)])
              for x in range(30) for z in range(30)]
    store = _store(tmp_path, 'reuse', values)
    try:
        store.decide(.4, .5, [100, 108])
        config = _config(geometry_rules=True, geometry_rules_outdoor_vegetation=False,
                         geometry_rules_ground_floor=False, scene_consistency=False)
        apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, config, active_codes={100, 108})
        apply_geometry_rules(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, config, tmp_path/'terrain.npz', active_codes={100, 108})
        assert len(calls) == 1
        assert store.db.execute('SELECT COUNT(*) FROM geometry_voxels WHERE code=108').fetchone()[0] == 900
    finally:
        store.close()


def test_reclassify_postprocess_reports_segment_vote_and_honours_override(tmp_path):
    values = [((x, 0, z), [(108, .75 if x < 24 else .45), (100, .25 if x < 24 else .55)])
              for x in range(30) for z in range(30)]
    store = _store(tmp_path, 'reclassify', values)
    try:
        config = _config(surface_tie_break=False, building_enclosure=False, geometry_rules=False,
                         stairs_geometry_check=False, scene_consistency=False, component_cleanup=False,
                         vegetation_base_to_ground=False, geometry_fill=False, _preset_classes=[], voxel_origin=[0., 0., 0.])
        stages, _, _ = _postprocess(store, config, {'wall': 100, 'column': 108}, [], tmp_path, {100, 108}, {})
        assert stages['segment_vote']['changed_voxels'] == 180
        stages, _, _ = _postprocess(store, {**config, 'segment_min_share': .99}, {'wall': 100, 'column': 108}, [], tmp_path, {100, 108}, {})
        assert stages['segment_vote']['changed_voxels'] == 0
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=100').fetchone()[0] == 180
    finally:
        store.close()


def test_segment_vote_regularizes_all_pillar_sides(tmp_path):
    values = []
    for x in range(20):
        for y in range(20):
            for z in range(130):
                if x not in (0, 19) and y not in (0, 19):
                    continue
                column = (x+y+z) % 5 != 0
                values.append(((x, y, z), [(108, .75 if column else .45), (100, .25 if column else .55)]))
    store = _store(tmp_path, 'pillar_volume', values)
    try:
        store.decide(.4, .5, [100, 108])
        apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, _config(), active_codes={100, 108})
        total = store.db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=108').fetchone()[0] >= .9*total
    finally:
        store.close()


def test_segment_vote_does_not_cross_parallel_table_and_floor(tmp_path):
    values = []
    for x in range(30):
        for y in range(15):
            values.extend([((x, y, 0), [(102, .8), (117, .2)]), ((x, y, 38), [(117, .7), (102, .3)])])
    store = _store(tmp_path, 'table_floor', values)
    try:
        store.decide(.4, .5, [102, 117])
        report = apply_segment_vote(store, {'floor': 102, 'furniture': 117}, np.zeros(3), .02, _config(), active_codes={102, 117})
        assert report['segments'] >= 2
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=102').fetchone()[0] == 450
    finally:
        store.close()


def test_planar_wall_does_not_absorb_adjacent_object_component(tmp_path):
    wall = [((x, 0, z), 1., (0., 1., 0.), 0.) for x in range(8) for z in range(8)]
    # A rough, directly neighbouring object has only object cells and normals
    # incompatible with the wall plane.
    object_cells = [((x, 1, z), .15, (1., 0., 0.), .45) for x in range(2, 7) for z in range(2, 7)]
    store, labels = _labels_for_features(tmp_path, 'wall_object', wall+object_cells)
    try:
        assert {labels[cell] for cell, *_ in wall}.isdisjoint({labels[cell] for cell, *_ in object_cells})
        assert store.db.execute("SELECT kind FROM segments WHERE segment=?", (labels[(3, 0, 3)],)).fetchone()[0] == 'planar'
        # Give the neighbouring object a weak furniture majority.  A bridged
        # wall segment would recolour it to wall; separate segments must not.
        store.db.execute('''CREATE TEMP TABLE geometry_voxels(key BLOB PRIMARY KEY,ix INTEGER,iy INTEGER,iz INTEGER,
            cx INTEGER,cy INTEGER,cz INTEGER,code INTEGER,confidence INTEGER) WITHOUT ROWID''')
        for index, (cell, _, _, _) in enumerate(wall+object_cells):
            key = _key((index, 0, 0))
            is_object = cell in {item[0] for item in object_cells}
            winner, other = (117, 100) if is_object else (100, 117)
            store.db.execute('INSERT INTO observations VALUES (?,1)', (key,))
            store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (key, winner, 140))
            store.db.execute('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)', (key, index, 0, 0, *cell, winner, 140))
            store.db.executemany('INSERT INTO votes VALUES (?,?,?,?)', ((key, winner, .55 if is_object else .8, .55 if is_object else .8),
                                                                         (key, other, .45 if is_object else .2, .45 if is_object else .2)))
        cells = np.asarray([item[0] for item in wall+object_cells], np.int64)
        ordered = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
        cells, ordered_labels = cells[ordered], np.asarray([labels[tuple(cell)] for cell in cells], np.int64)[ordered]
        low, shape = cells.min(axis=0), cells.max(axis=0)-cells.min(axis=0)+1
        _accumulate_segment_votes(store, cells, ordered_labels, low, shape, {100, 117}, 68, 1, lambda: None)
        _apply_decisive_segments(store, _config(), {'wall': 100, 'furniture': 117}, 68, lambda: None)
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=117').fetchone()[0] == len(object_cells)
    finally:
        store.close()


def test_tabletop_does_not_absorb_object_resting_on_it(tmp_path):
    table = [((x, y, 0), 1., (0., 0., 1.), 0.) for x in range(10) for y in range(6)]
    # The compact object touches the plate in cell space but its rough cells
    # cannot satisfy the normal-angle condition of the horizontal plane.
    object_cells = [((x, y, z), .2, (1., 0., 0.), .4) for x in range(3, 6) for y in range(2, 5) for z in range(1, 4)]
    store, labels = _labels_for_features(tmp_path, 'table_object', table+object_cells)
    try:
        assert {labels[cell] for cell, *_ in table}.isdisjoint({labels[cell] for cell, *_ in object_cells})
    finally:
        store.close()


def test_planar_segment_kind_includes_compatible_edge_cells(tmp_path):
    wall = [((x, 0, z), 1., (0., 1., 0.), 0.) for x in range(8) for z in range(5)]
    # This is the low-planarity edge cell that must attach once and therefore
    # preserve the wall segment's planar classification.
    edge = [((8, 0, 2), .2, (0., 1., 0.), .1)]
    store, labels = _labels_for_features(tmp_path, 'planar_edge', wall+edge)
    try:
        assert labels[(8, 0, 2)] == labels[(7, 0, 2)]
        assert store.db.execute("SELECT kind FROM segments WHERE segment=?", (labels[(8, 0, 2)],)).fetchone()[0] == 'planar'
    finally:
        store.close()


def test_segment_vote_respects_minimum_observations_for_votes_and_updates(tmp_path):
    values = []
    for x in range(30):
        for z in range(30):
            reliable = x < 15
            values.append(((x, 0, z), [(108, .8), (100, .2)] if reliable else [(100, .9), (108, .1)]))
    store = _store(tmp_path, 'minimum_observations', values)
    try:
        # The left half has enough observations and clearly supports column.
        # The right half has a stronger wall vote, but only one observation.
        for x in range(15):
            for z in range(30):
                store.db.execute('UPDATE observations SET count=2 WHERE key=?', (_key((x, 0, z)),))
        config = _config(min_observations=2)
        store.decide(2, .5, [100, 108])
        report = apply_segment_vote(store, {'wall': 100, 'column': 108}, np.zeros(3), .02, config,
                                    active_codes={100, 108})
        assert store.db.execute('SELECT winner FROM segments WHERE cells>=3 ORDER BY cells DESC LIMIT 1').fetchone()[0] == 108
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=108').fetchone()[0] == 450
        assert store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=1').fetchone()[0] == 450
        assert report['changed_voxels'] == 0
    finally:
        store.close()


@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1', reason='set NOMIVOX_RUN_SLOW=1 to run the segment benchmark')
def test_segment_vote_benchmark_file_sqlite(tmp_path, monkeypatch):
    """Exercise two million file-SQLite voxels; feature time is reported separately."""
    store = VoteStore(tmp_path/'segment_benchmark.sqlite')
    segment_count, cells_per_segment, voxels_per_cell = 100_000, 3, 7
    cell_count = segment_count*cells_per_segment
    try:
        store.db.executescript('''
            CREATE TEMP TABLE geometry_voxels(key BLOB PRIMARY KEY,ix INTEGER,iy INTEGER,iz INTEGER,
                cx INTEGER,cy INTEGER,cz INTEGER,code INTEGER,confidence INTEGER) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_cells(cx INTEGER,cy INTEGER,cz INTEGER,tx INTEGER,ty INTEGER,tz INTEGER,
                sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_features(cx INTEGER,cy INTEGER,cz INTEGER,x REAL,y REAL,z REAL,
                linearity REAL,planarity REAL,scatter REAL,verticality REAL,nx REAL,ny REAL,nz REAL,
                hag REAL,covered INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_meta(name TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID;
        ''')
        for start in range(0, cell_count, 10_000):
            stop = min(start+10_000, cell_count)
            ids = np.arange(start, stop, dtype=np.int64)
            cx = (ids//cells_per_segment)*4+(ids % cells_per_segment)
            store.db.executemany('INSERT INTO geometry_cells VALUES (?,?,?,0,0,0,?,?,?,?)',
                ((int(x), 0, 0, float(x*5), 0., 0., voxels_per_cell) for x in cx))
            store.db.executemany('INSERT INTO geometry_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                ((int(x), 0, 0, float(x*.1), .05, .05, 0., 1., 0., 1., 0., 0., 1., None, 0) for x in cx))
            observations, votes, geometry = [], [], []
            for cell in cx:
                for offset in range(voxels_per_cell):
                    index = (int(cell)*5+offset % 5, offset//5, 0)
                    key = _key(index)
                    observations.append((key, 1))
                    votes.extend((key, code, weight, weight) for code, weight in ((100, .45), (108, .4), (117, .15)))
                    geometry.append((key, *index, int(cell), 0, 0, 1, 0))
            store.db.executemany('INSERT INTO observations VALUES (?,?)', observations)
            store.db.executemany('INSERT INTO votes VALUES (?,?,?,?)', votes)
            store.db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)', geometry)
        started = time.monotonic()
        store.decide(.4, .4, [100, 108, 117])
        decide_s = time.monotonic()-started
        monkeypatch.setattr(segment_module, 'prepare_geometry_features', lambda *args, **kwargs: False)
        started = time.monotonic()
        report = apply_segment_vote(store, {'wall': 100, 'column': 108, 'furniture': 117}, np.zeros(3), .02,
                                    _config(), active_codes={100, 108, 117})
        segment_s = time.monotonic()-started
        print(f'decide={decide_s:.3f}s segment_without_features={segment_s:.3f}s features={report["timings"]["features_s"]:.3f}s')
        assert segment_s <= 1.5*decide_s
    finally:
        store.close()
