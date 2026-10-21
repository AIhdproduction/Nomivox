import numpy as np

from e57cls.pipeline.config import DEFAULTS
from e57cls.vote.engine import VoteStore, voxel_keys
from e57cls.vote.scene_consistency import enforce_scene_consistency


def test_scene_consistency_uses_fallback_vote_then_unclassified(tmp_path):
    votes = VoteStore(tmp_path/'votes.sqlite')
    keys = voxel_keys([[0., 0., 0.], [1., 0., 0.]], np.zeros(3), 1.)
    try:
        votes.add_scan(1, 'outdoor', [(keys, np.array([[.7, .2], [.7, 0.]], np.float32))], [101, 100])
        votes.decide(1, .5, [101, 100])
        classes = [{'id': 'ceiling', 'scene': 'indoor'}, {'id': 'building', 'scene': 'outdoor'}]
        report = enforce_scene_consistency(
            votes, classes, {'ceiling': 101, 'building': 100},
            {1: {'scene': 'outdoor', 'outdoor': True, 'position': [0., 0., 0.], 'allowed_codes': {101, 100}}},
            np.zeros(3), 1., DEFAULTS, {101, 100}, lambda: None)
        codes, confidence = votes.lookup(keys)
        assert codes.tolist() == [100, 100]
        assert confidence.tolist()[0] > 0
        assert report['changed'] == {'101->100': 2}
    finally:
        votes.close()


def test_scene_consistency_leaves_ambiguous_and_can_be_disabled(tmp_path):
    votes = VoteStore(tmp_path/'votes.sqlite')
    keys = voxel_keys([[0., 0., 0.]], np.zeros(3), 1.)
    try:
        votes.add_scan(1, 'outdoor', [(keys, np.ones((1, 1), np.float32))], [101])
        votes.add_scan(2, 'indoor', [(keys, np.ones((1, 1), np.float32))], [101])
        votes.decide(1, .5, [101])
        classes = [{'id': 'ceiling', 'scene': 'indoor'}]
        report = enforce_scene_consistency(votes, classes, {'ceiling': 101},
            {1: {'scene': 'outdoor', 'outdoor': True, 'position': None, 'allowed_codes': {101}},
             2: {'scene': 'indoor', 'outdoor': False, 'position': None, 'allowed_codes': {101}}},
            np.zeros(3), 1., DEFAULTS, {101}, lambda: None)
        assert not report['changed']
        disabled = enforce_scene_consistency(votes, classes, {'ceiling': 101}, {}, np.zeros(3), 1.,
                                             {**DEFAULTS, 'scene_consistency': False}, {101}, lambda: None)
        assert not disabled['enabled']
    finally:
        votes.close()


def test_scene_consistency_uses_scene_specific_fallbacks_and_reports_changes(tmp_path):
    votes = VoteStore(tmp_path/'fallbacks.sqlite')
    keys = voxel_keys([[index, 0., 0.] for index in range(4)], np.zeros(3), 1.)
    codes = [101, 100, 102, 103, 104, 105, 106]
    try:
        # ceiling -> building, floor -> sidewalk, door -> building; no suitable vote -> class 1.
        weights = np.array([[.7, .2, 0., 0., 0., 0., 0.], [0., 0., .6, .3, 0., 0., 0.],
                            [0., 0., 0., 0., .8, .2, 0.], [0., 0., 0., 0., 0., 0., .8]], np.float32)
        votes.add_scan(1, 'outdoor', [(keys, weights)], codes)
        votes.decide(1, .5, codes)
        classes = [{'id': 'ceiling', 'scene': 'indoor'}, {'id': 'building', 'scene': 'outdoor'},
                   {'id': 'floor', 'scene': 'indoor'}, {'id': 'sidewalk', 'scene': 'outdoor'},
                   {'id': 'door', 'scene': 'indoor'}, {'id': 'window', 'scene': 'outdoor'}, {'id': 'lamp', 'scene': 'indoor'},
                   {'id': 'tree', 'scene': 'outdoor'}]
        report = enforce_scene_consistency(votes, classes,
            {'ceiling': 101, 'building': 100, 'floor': 102, 'sidewalk': 103, 'door': 104, 'window': 105, 'lamp': 106, 'tree': 5},
            {1: {'scene': 'outdoor', 'outdoor': True, 'position': [0., 0., 0.], 'allowed_codes': set(codes)}},
            np.zeros(3), 1., DEFAULTS, set(codes), lambda: None)
        assert votes.lookup(keys)[0].tolist() == [100, 103, 100, 1]
        assert report['changed'] == {'101->100': 1, '102->103': 1, '104->100': 1, '106->1': 1}
    finally:
        votes.close()


def test_scene_consistency_indoor_fallbacks_and_both_classes_are_preserved(tmp_path):
    votes = VoteStore(tmp_path/'indoor.sqlite')
    keys = voxel_keys([[index, 0., 0.] for index in range(6)], np.zeros(3), 1.)
    codes = [100, 101, 102, 103, 104, 105, 106]
    try:
        weights = np.array([[.8, .2, 0., 0., 0., 0., 0.], [0., 0., .8, .2, 0., 0., 0.],
                            [0., 0., 0., 0., .8, .2, 0.], [0., 0., 0., 0., .8, 0., 0.],
                            [0., 0., 0., 0., 0., 0., .8]], np.float32)
        votes.add_scan(1, 'indoor', [(keys[:5], weights)], codes)
        votes.add_scan(2, 'outdoor', [(keys[5:], np.array([[0., 0., 0., 0., 0., 0., .8]], np.float32))], codes)
        votes.decide(1, .5, codes)
        classes = [{'id': 'building', 'scene': 'outdoor'}, {'id': 'wall', 'scene': 'indoor'},
                   {'id': 'road', 'scene': 'outdoor'}, {'id': 'floor', 'scene': 'indoor'},
                   {'id': 'tree', 'scene': 'outdoor'}, {'id': 'furniture', 'scene': 'indoor'},
                   {'id': 'window', 'scene': 'both'}]
        enforce_scene_consistency(votes, classes,
            {'building': 100, 'wall': 101, 'road': 102, 'floor': 103, 'tree': 104, 'furniture': 105, 'window': 106},
            {1: {'scene': 'indoor', 'outdoor': False, 'position': [0., 0., 0.], 'allowed_codes': set(codes)},
             2: {'scene': 'outdoor', 'outdoor': True, 'position': [5., 0., 0.], 'allowed_codes': set(codes)}},
            np.zeros(3), 1., DEFAULTS, set(codes), lambda: None)
        assert votes.lookup(keys)[0].tolist() == [101, 103, 105, 1, 106, 106]
    finally:
        votes.close()
