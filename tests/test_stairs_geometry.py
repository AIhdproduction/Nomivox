import numpy as np

from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline.run import evidence_map
from e57cls.vote.engine import VoteStore, voxel_keys


GEO_ORIGIN = np.array([2665953., 1207269., 440.])
VOXEL_SIZE = .02


def _world_voxels(voxels, *, angle=23., noise=.007, seed=18):
    """Place dense local voxels at a rotated, georeferenced test location."""
    voxels = np.unique(np.asarray(voxels, np.int64), axis=0)
    local = (voxels.astype(float)+.5)*VOXEL_SIZE
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    local[:, :2] = local[:, :2] @ rotation.T
    return GEO_ORIGIN+local+np.random.default_rng(seed).normal(0, noise, local.shape)


def _surface_stair(steps, width_cells=60, tread_cells=14, rise_cells=(8, 9), *, missing=()):
    """Build tread and full-riser voxels, omitting selected tread levels."""
    heights = np.r_[0, np.cumsum(np.resize(np.asarray(rise_cells), steps-1))]
    voxels = []
    for step, height in enumerate(heights):
        start = step*tread_cells
        if step not in missing:
            voxels.extend((x, y, int(height)) for x in range(start, start+tread_cells)
                          for y in range(width_cells))
        if step:
            previous = int(heights[step-1])
            voxels.extend((start, y, z) for y in range(width_cells)
                          for z in range(previous, int(height)+1))
    return np.asarray(voxels, np.int64)


def _solid_stair(steps, width_cells=60, tread_cells=14, rise_cells=(8, 9)):
    """Build a dense staircase whose lower boundary rises diagonally."""
    heights = np.r_[0, np.cumsum(np.resize(np.asarray(rise_cells), steps-1))]
    voxels = []
    for x in range(steps*tread_cells):
        top = int(heights[x//tread_cells])
        bottom = int(np.floor(x*8.5/tread_cells))-14
        for y in range(width_cells):
            voxels.extend((x, y, z) for z in range(bottom, top+1))
    return np.asarray(voxels, np.int64)


def _add_l_flight(voxels, start, direction, first_height, steps, width_cells=60, tread_cells=14):
    direction = np.asarray(direction, np.int64)
    perpendicular = np.array((-direction[1], direction[0]), np.int64)
    heights = first_height+np.r_[0, np.cumsum(np.resize(np.array([8, 9]), steps-1))]
    for step, height in enumerate(heights):
        for along in range(step*tread_cells, (step+1)*tread_cells):
            for across in range(width_cells):
                xy = np.asarray(start)+direction*along+perpendicular*across
                voxels.append((int(xy[0]), int(xy[1]), int(height)))
        if step:
            for across in range(width_cells):
                xy = np.asarray(start)+direction*(step*tread_cells)+perpendicular*across
                for z in range(int(heights[step-1]), int(height)+1):
                    voxels.append((int(xy[0]), int(xy[1]), z))
    return int(heights[-1])


def _report_is_accepted(report):
    assert report['accepted_components'] >= 1
    assert report['accepted_voxels'] > 0


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
        assert report['checked_components'] == 1
        assert report['rejected_components'] == 1
        assert report['reclassified_voxels'] == {'101': len(keys)}
        assert report['kept_fragments'] == {'components': 0, 'voxels': 0}
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
        _report_is_accepted(report)
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
        _report_is_accepted(report)
        assert set(votes.lookup(keys)[0]) == {105}
    finally:
        votes.close()


def test_u_stair_with_landing_remains_stairs(tmp_path):
    levels = np.arange(11, dtype=float)*.18
    centres = np.array([(step*.28, 0.) for step in range(6)]
                       + [((10-step)*.28, .28) for step in range(6, 11)])
    assert VoteStore._stair_shape(levels, centres, 3, .10, .23, .20, .45)


def test_dense_rotated_straight_stair_with_risers_and_sloped_underside_is_kept(tmp_path):
    points = _world_voxels(_solid_stair(8), seed=1)
    votes, keys = _stairs_votes(tmp_path/'dense-straight.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        _report_is_accepted(report)
        assert np.count_nonzero(votes.lookup(keys)[0] == 105) >= .95*len(keys)
    finally:
        votes.close()


def test_dense_rotated_l_stair_with_landing_and_risers_is_kept(tmp_path):
    voxels = []
    landing_height = _add_l_flight(voxels, (0, 0), (1, 0), 0, 6)
    _add_l_flight(voxels, (84, 0), (0, 1), landing_height, 6)
    votes, keys = _stairs_votes(tmp_path/'l-stair.sqlite', _world_voxels(voxels, seed=2), 102)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        _report_is_accepted(report)
        assert np.count_nonzero(votes.lookup(keys)[0] == 105) >= .95*len(keys)
    finally:
        votes.close()


def test_dense_stair_with_one_occluded_middle_tread_is_kept(tmp_path):
    points = _world_voxels(_surface_stair(8, missing={3}), seed=3)
    votes, keys = _stairs_votes(tmp_path/'missing-tread.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        _report_is_accepted(report)
        assert np.count_nonzero(votes.lookup(keys)[0] == 105) >= .95*len(keys)
    finally:
        votes.close()


def test_small_edge_fragments_next_to_an_accepted_stair_are_kept(tmp_path):
    stair = _surface_stair(6)
    top = stair[:, 2].max()
    attached = np.array([(x, y, z) for x in range(stair[:, 0].max()+2, stair[:, 0].max()+5)
                         for y in range(28, 31) for z in range(top-1, top+2)], np.int64)
    isolated = attached+np.array((50, 0, 0), np.int64)
    points = _world_voxels(np.vstack((stair, attached, isolated)), noise=0)
    votes, keys = _stairs_votes(tmp_path/'fragments.sqlite', points, 102)
    attached_keys = voxel_keys(_world_voxels(attached, noise=0), np.zeros(3), VOXEL_SIZE)
    isolated_keys = voxel_keys(_world_voxels(isolated, noise=0), np.zeros(3), VOXEL_SIZE)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        assert report['kept_fragments']['components'] == 1
        assert report['kept_fragments']['voxels'] == len(np.unique(attached_keys))
        assert set(votes.lookup(attached_keys)[0]) == {105}
        assert set(votes.lookup(isolated_keys)[0]) == {102}
        assert report['reclassified_voxels']['102'] == len(isolated_keys)
    finally:
        votes.close()


def test_rotated_ramps_roof_floor_and_table_are_rejected(tmp_path):
    ramp = [(x, y, int(np.floor(x*np.tan(np.deg2rad(12))))) for x in range(80) for y in range(30)]
    roof = [(x+120, y, int(np.floor(x*np.tan(np.deg2rad(35))))) for x in range(50) for y in range(30)]
    floor = [(x, y+80, 0) for x in range(50) for y in range(50)]
    table = [(x+120, y+80, 37) for x in range(30) for y in range(30)]
    votes, keys = _stairs_votes(tmp_path/'negative-shapes.sqlite', _world_voxels(ramp+roof+floor+table, seed=4), 102)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        assert report['accepted_components'] == 0
        assert set(votes.lookup(keys)[0]) == {102}
    finally:
        votes.close()


def test_hslu_sized_dense_stair_regression_is_kept(tmp_path):
    # 2.04 m run, 3.60 m width and roughly one metre including the underside.
    points = _world_voxels(_solid_stair(6, width_cells=180, tread_cells=17), seed=5)
    votes, keys = _stairs_votes(tmp_path/'hslu-regression.sqlite', points, 102)
    try:
        report = votes.reject_planar_stairs(105, GEO_ORIGIN, VOXEL_SIZE, .03, .85, .25, 42, [105, 102])
        _report_is_accepted(report)
        assert np.count_nonzero(votes.lookup(keys)[0] == 105) >= .95*len(keys)
    finally:
        votes.close()


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
