import numpy as np

from e57cls.vote.engine import VoteStore, voxel_keys


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
