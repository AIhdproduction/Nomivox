import numpy as np

from e57cls.pipeline.config import DEFAULTS
from e57cls.vote.engine import VoteStore, voxel_keys
from e57cls.vote.scene_context import ensure_scene_context


def _votes(path, points):
    votes = VoteStore(path)
    keys = voxel_keys(points, np.zeros(3), 1.)
    for scan in (1, 2, 3):
        votes.add_scan(scan, str(scan), [(keys, np.ones((len(keys), 1), np.float32))], [2])
    votes.decide(1, .5, [2])
    return votes, keys


def test_nearest_scene_context_is_chunk_independent_and_reusable(tmp_path):
    votes, _ = _votes(tmp_path/'votes.sqlite', [[0., 0., 0.], [5., 0., 0.], [10., 0., 0.]])
    context = {1: {'scene': 'outdoor', 'outdoor': True, 'position': [0., 0., 0.], 'allowed_codes': {2}},
               2: {'scene': 'indoor', 'outdoor': False, 'position': [10., 0., 0.], 'allowed_codes': {2}},
               3: {'scene': None, 'outdoor': False, 'position': None, 'allowed_codes': {2}}}
    calls = []
    try:
        first = ensure_scene_context(votes, {'ground': 2}, context, np.zeros(3), 1., DEFAULTS,
                                     lambda: calls.append(1), chunk_rows=7)
        values = [row[0] for row in votes.db.execute('SELECT scene_context FROM geometry_context ORDER BY key')]
        assert values == [1, -1, 0]
        assert first['context_voxels'] == {'outdoor': 1, 'indoor': 1, 'ambiguous': 1, 'unknown': 0}
        assert calls
        again = ensure_scene_context(votes, {'ground': 2}, context, np.zeros(3), 1., DEFAULTS, lambda: None)
        assert again['reused']
        assert list(votes.db.execute('SELECT outdoor,indoor,allow_ground FROM geometry_context ORDER BY key')) == [(1, 1, 1)]*3
    finally:
        votes.close()


def test_scene_context_falls_back_to_unanimous_legacy_scene(tmp_path):
    votes, _ = _votes(tmp_path/'fallback.sqlite', [[0., 0., 0.]])
    try:
        report = ensure_scene_context(votes, {'ground': 2},
                                      {1: {'outdoor': True, 'allowed_codes': {2}},
                                       2: {'outdoor': True, 'allowed_codes': {2}},
                                       3: {'outdoor': False, 'allowed_codes': {2}}},
                                      np.zeros(3), 1., DEFAULTS, lambda: None)
        assert report['context_voxels']['ambiguous'] == 1
        assert votes.db.execute('SELECT nearest_scan,nearest_m FROM geometry_context').fetchone() == (None, None)
    finally:
        votes.close()


def test_unknown_scans_do_not_affect_distance_context_or_legacy_maxima(tmp_path):
    votes, _ = _votes(tmp_path/'unknown.sqlite', [[0., 0., 0.]])
    try:
        ensure_scene_context(votes, {'ground': 2, 'floor': 102},
            {1: {'scene': 'outdoor', 'outdoor': True, 'position': [0., 0., 0.], 'allowed_codes': {2}},
             2: {'scene': None, 'outdoor': False, 'position': None, 'allowed_codes': {102}},
             3: {'scene': None, 'outdoor': False, 'position': [50., 0., 0.], 'allowed_codes': {102}}},
            np.zeros(3), 1., DEFAULTS, lambda: None)
        assert votes.db.execute('''SELECT outdoor,indoor,allow_ground,allow_floor,nearest_scan,nearest_outdoor,
                                          scene_context FROM geometry_context''').fetchone() == (1, 1, 1, 1, 1, 1, 1)
    finally:
        votes.close()
