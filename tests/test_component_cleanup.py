import numpy as np

from e57cls.vote.engine import VoteStore, voxel_keys


def _feature_context(votes, keys, outdoor):
    votes.db.execute('CREATE TEMP TABLE geometry_voxels(key BLOB PRIMARY KEY,cx INTEGER,cy INTEGER,cz INTEGER) WITHOUT ROWID')
    votes.db.execute('CREATE TEMP TABLE geometry_context(key BLOB PRIMARY KEY,outdoor INTEGER,indoor INTEGER) WITHOUT ROWID')
    coordinates = np.frombuffer(b''.join(bytes(key) for key in keys), dtype='<i8').reshape(-1, 3)
    cells = np.floor(coordinates*.02/.10).astype(np.int64)
    votes.db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?)',
                         ((bytes(key), *map(int, cell)) for key, cell in zip(keys, cells)))
    votes.db.executemany('INSERT INTO geometry_context VALUES (?,?,?)',
                         ((bytes(key), int(outdoor), int(not outdoor)) for key in keys))
    votes.db.commit()


def test_small_components_are_removed_but_cable_span_is_kept(tmp_path):
    votes = VoteStore(tmp_path/'votes.sqlite')
    keys = voxel_keys([[0, 0, 0], [.02, 0, 0], [.04, 0, 0], [.06, 0, 0],
                       [.50, 0, 0]], np.zeros(3), .02)
    votes.add_scan(1, 'run', [(keys, np.ones((len(keys), 1), np.float32))], [112])
    votes.decide(1, .1, [112])
    result = votes.remove_small_components(.02, 12, [(112, 4, .05)])
    assert result['removed_voxels'] == 1
    assert votes.lookup(keys)[0].tolist() == [112, 112, 112, 112, 1]
    votes.close()


def test_sparse_voxels_survive_when_their_10cm_cells_are_connected(tmp_path):
    votes = VoteStore(tmp_path/'sparse-grid.sqlite')
    keys = voxel_keys([[0, 0, 0], [.12, 0, 0], [.22, 0, 0], [.32, 0, 0]], np.zeros(3), .02)
    votes.add_scan(1, 'run', [(keys, np.ones((len(keys), 1), np.float32))], [2])
    votes.decide(1, .1, [2])
    _feature_context(votes, keys, outdoor=True)
    result = votes.remove_small_components(.02, 12, component_min_cells=4, feature_cell_size=.10)
    assert result['removed_voxels'] == 0
    assert votes.lookup(keys)[0].tolist() == [2, 2, 2, 2]
    votes.close()


def test_thin_interior_raster_is_not_spared_by_coarse_cells(tmp_path):
    votes = VoteStore(tmp_path/'indoor-grid.sqlite')
    keys = voxel_keys([[0, 0, 0], [.12, 0, 0], [.22, 0, 0], [.32, 0, 0]], np.zeros(3), .02)
    votes.add_scan(1, 'run', [(keys, np.ones((len(keys), 1), np.float32))], [6])
    votes.decide(1, .1, [6])
    _feature_context(votes, keys, outdoor=False)
    result = votes.remove_small_components(.02, 12, component_min_cells=4, feature_cell_size=.10)
    assert result['removed_voxels'] == len(keys)
    assert set(votes.lookup(keys)[0]) == {1}
    votes.close()


def test_dense_interior_noise_is_not_spared_by_the_coarse_cells(tmp_path):
    votes = VoteStore(tmp_path/'dense-noise.sqlite')
    keys = voxel_keys([[index*.02, 0, 0] for index in range(11)], np.zeros(3), .02)
    votes.add_scan(1, 'run', [(keys, np.ones((len(keys), 1), np.float32))], [6])
    votes.decide(1, .1, [6])
    result = votes.remove_small_components(.02, 12, component_min_cells=2, feature_cell_size=.10,
                                           sparse_maximum_voxels_per_cell=1)
    assert result['removed_voxels'] == len(keys)
    assert set(votes.lookup(keys)[0]) == {1}
    votes.close()


def test_low_vegetation_bottom_band_follows_each_local_slope_cell(tmp_path):
    votes = VoteStore(tmp_path/'vegetation.sqlite')
    keys = voxel_keys([[0, 0, 0], [0, 0, .02], [0, 0, .10],
                       [.10, 0, .20], [.10, 0, .22], [.10, 0, .30]], np.zeros(3), .02)
    votes.add_scan(1, 'run', [(keys, np.ones((len(keys), 1), np.float32))], [3])
    votes.decide(1, .1, [3])
    report = votes.vegetation_bases_to_ground(3, 2, .02, .10, .04)
    assert report == {'moved_voxels': 4, 'cells': 2}
    assert votes.lookup(keys)[0].tolist() == [2, 2, 3, 2, 2, 3]
    votes.close()
