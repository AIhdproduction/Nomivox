import numpy as np

from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline.run import evidence_map
from e57cls.vote.engine import VoteStore, voxel_keys


def _stairs_votes(path, points, target_code):
    votes = VoteStore(path)
    keys = voxel_keys(np.asarray(points, float), np.zeros(3), .02)
    weights = np.tile([.8, .6], (len(keys), 1)).astype(np.float32)
    votes.add_scan(1, 'stairs', [(keys, weights)], [105, target_code])
    votes.decide(1, .5, [105, target_code])
    return votes, keys


def test_default_stair_priority_excludes_ceiling_for_pixels_and_voxels(tmp_path):
    config = {**DEFAULTS, 'boundary_margin': 0}
    classes = [{'id': 'ceiling'}, {'id': 'floor'}, {'id': 'stairs'}]
    labels, _ = evidence_map([np.array([[.95, .0]], np.float32),
                              np.array([[.0, .95]], np.float32),
                              np.array([[.8, .8]], np.float32)], classes, config)
    assert labels.tolist() == [[0, 2]]

    votes = VoteStore(tmp_path/'priority.sqlite')
    keys = voxel_keys([[0, 0, 0], [1, 0, 0]], np.zeros(3), .02)
    votes.add_scan(1, 'priority', [(keys, np.array([[.8, .95, 0], [.8, 0, .95]], np.float32))], [105, 101, 102])
    try:
        votes.decide(1, .5, [105, 101, 102], priority={
            'code': 105, 'over': [102], 'confidence': config['stairs_min_confidence']})
        assert votes.lookup(keys)[0].tolist() == [101, 105]
    finally:
        votes.close()


def test_stair_priority_requires_a_minimum_vote_share(tmp_path):
    votes = VoteStore(tmp_path/'priority-share.sqlite')
    keys = voxel_keys([[0, 0, 0]], np.zeros(3), .02)
    votes.add_scan(1, 'priority', [(keys, np.array([[.26, .90]], np.float32))], [105, 102])
    try:
        votes.decide(1, .5, [105, 102], priority={
            'code': 105, 'over': [102], 'confidence': .25, 'min_share': .25})
        assert votes.lookup(keys)[0].tolist() == [102]
    finally:
        votes.close()


def test_planar_sloped_stairs_become_ceiling(tmp_path):
    points = [(x*.02+.001, y*.02+.001, x*.02+.001) for x in range(20) for y in range(10)]
    votes, keys = _stairs_votes(tmp_path/'slope.sqlite', points, 101)
    try:
        report = votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 101])
        assert report == {'checked_components': 1, 'rejected_components': 1, 'reclassified_voxels': {'101': len(keys)}}
        assert set(votes.lookup(keys)[0]) == {101}
    finally:
        votes.close()


def test_real_stair_geometry_remains_stairs(tmp_path):
    points = []
    for step in range(6):
        for x in range(step*14, (step+1)*14):
            for y in range(10):
                points.append((x*.02+.001, y*.02+.001, step*9*.02+.001))
        if step:
            for z in range((step-1)*9, step*9+1):
                for y in range(10):
                    points.append((step*14*.02+.001, y*.02+.001, z*.02+.001))
    votes, keys = _stairs_votes(tmp_path/'steps.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 102])
        assert report == {'checked_components': 1, 'rejected_components': 0, 'reclassified_voxels': {}}
        assert set(votes.lookup(keys)[0]) == {105}
    finally:
        votes.close()


def test_two_flights_with_a_landing_remain_stairs(tmp_path):
    points = []
    for step in range(6):
        for x in range(step*14, (step+1)*14):
            for y in range(10):
                points.append((x*.02+.001, y*.02+.001, step*9*.02+.001))
        if step:
            for z in range((step-1)*9, step*9+1):
                for y in range(10):
                    points.append((step*14*.02+.001, y*.02+.001, z*.02+.001))
    for x in range(84, 112):
        for y in range(10):
            points.append((x*.02+.001, y*.02+.001, 5*9*.02+.001))
    for step in range(1, 6):
        for x in range(112+(step-1)*14, 112+step*14):
            for y in range(10):
                points.append((x*.02+.001, y*.02+.001, (5+step)*9*.02+.001))
        for z in range((4+step)*9, (5+step)*9+1):
            for y in range(10):
                points.append(((112+(step-1)*14)*.02+.001, y*.02+.001, z*.02+.001))
    votes, keys = _stairs_votes(tmp_path/'two-flights.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 102])
        assert report['rejected_components'] == 0
        assert set(votes.lookup(keys)[0]) == {105}
    finally:
        votes.close()


def test_u_stair_with_landing_remains_stairs(tmp_path):
    levels = np.arange(11, dtype=float)*.18
    centres = np.array([(step*.28, 0.) for step in range(6)]
                       + [((10-step)*.28, .28) for step in range(6, 11)])
    assert VoteStore._stair_shape(levels, centres, 3, .10, .23, .20, .45)


def test_small_stair_component_is_restored_instead_of_being_skipped(tmp_path):
    points = [(x*.02+.001, y*.02+.001, .001) for x in range(4) for y in range(4)]
    votes, keys = _stairs_votes(tmp_path/'small.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 102])
        assert report['checked_components'] == 1
        assert report['rejected_components'] == 1
        assert report['reclassified_voxels'] == {'102': len(keys)}
        assert set(votes.lookup(keys)[0]) == {102}
    finally:
        votes.close()


def test_stair_level_at_nearby_floor_is_returned_to_floor(tmp_path):
    votes = VoteStore(tmp_path/'floor-ring.sqlite')
    stair_points = np.array([(x*.02+.001, y*.02+.001, .001) for x in range(15) for y in range(10)])
    floor_points = np.array([(x*.02+.001, .50, .001) for x in range(12)])
    stair_keys = voxel_keys(stair_points, np.zeros(3), .02)
    floor_keys = voxel_keys(floor_points, np.zeros(3), .02)
    votes.add_scan(1, 'stairs', [(stair_keys, np.tile([.8, .6], (len(stair_keys), 1)).astype(np.float32))], [105, 2])
    votes.add_scan(2, 'floor', [(floor_keys, np.ones((len(floor_keys), 1), np.float32))], [102])
    try:
        votes.decide(1, .5, [105, 2, 102])
        votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 2, 102],
                                   floor_code=102, floor_ring_radius=2., floor_level_tolerance=.03)
        assert set(votes.lookup(stair_keys)[0]) == {102}
    finally:
        votes.close()


def test_flat_stair_component_becomes_floor(tmp_path):
    points = [(x*.02+.001, y*.02+.001, .001) for x in range(10) for y in range(10)]
    votes, keys = _stairs_votes(tmp_path/'flat.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, np.zeros(3), .02, .03, .85, .25, 42, [105, 102])
        assert report['rejected_components'] == 1
        assert report['reclassified_voxels'] == {'102': len(keys)}
        assert set(votes.lookup(keys)[0]) == {102}
    finally:
        votes.close()
