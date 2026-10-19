import json
import os

import numpy as np
import pytest

from e57cls.geometry.fill import PlaneIndex, extract_planes
from e57cls.pipeline.config import DEFAULTS, PROJECT
from e57cls.pipeline.run import depth_gate_representatives
from e57cls.vote.engine import VoteStore, voxel_keys


def _votes(path, points, weights, codes=(100, 101, 102)):
    votes = VoteStore(path)
    keys = voxel_keys(np.asarray(points, float), np.zeros(3), .02)
    votes.add_scan(1, 'scan', [(keys, np.asarray(weights, np.float32))], codes)
    votes.decide(1, .5, codes)
    return votes, keys


def test_surface_tie_break_uses_local_surface_normal(tmp_path):
    sloped = [(x*.02, y*.02, x*.02) for x in range(3) for y in range(3)]
    weights = np.tile([.51, .49, 0], (len(sloped), 1))
    votes, keys = _votes(tmp_path/'slope.sqlite', sloped, weights)
    try:
        report = votes.resolve_surface_ties({'wall': 100, 'ceiling': 101, 'floor': 102}, np.zeros(3), .02, .55, .08)
        codes, _ = votes.lookup(keys)
        assert report['ceiling'] and set(codes) == {101}
    finally:
        votes.close()

    vertical = [(0, y*.02, z*.02) for y in range(3) for z in range(3)]
    votes, keys = _votes(tmp_path/'wall.sqlite', vertical, weights)
    try:
        votes.resolve_surface_ties({'wall': 100, 'ceiling': 101, 'floor': 102}, np.zeros(3), .02, .55, .08)
        assert set(votes.lookup(keys)[0]) == {100}
    finally:
        votes.close()

    horizontal = [(x*.02, y*.02, 0) for x in range(3) for y in range(3)]
    floor_weights = np.tile([.49, 0, .51], (len(horizontal), 1))
    votes, keys = _votes(tmp_path/'floor.sqlite', horizontal, floor_weights)
    try:
        votes.resolve_surface_ties({'wall': 100, 'ceiling': 101, 'floor': 102}, np.zeros(3), .02, .55, .08)
        assert set(votes.lookup(keys)[0]) == {102}
    finally:
        votes.close()


def test_depth_gate_removes_far_door_votes_and_keeps_small_components():
    labels = np.zeros((2, 125), np.int16)
    labels[0, :] = 1
    classes = [{'id': 'wall'}, {'id': 'door'}]
    pixels = np.array([(index, 0) for index in range(125)]*2)
    ranges = np.r_[np.full(50, 3.), np.full(200, 6.)]
    world = np.column_stack((np.r_[np.linspace(-.5, .5, 50), np.linspace(2, 4, 200)], np.zeros(250), np.zeros(250)))
    chosen = np.ones(250, np.int16)
    blocked, report = depth_gate_representatives(labels, pixels, ranges, world, chosen, classes, DEFAULTS)
    assert report['components'] == 1
    assert not blocked[:50].any() and blocked[50:].all()


def test_geometry_fill_closes_small_hole_without_growing_forever(tmp_path):
    points = [(x*.05, y*.05, 1.) for x in range(20) for y in range(20) if not (6 <= x < 14 and 6 <= y < 14)]
    votes, _ = _votes(tmp_path/'planes.sqlite', points, np.tile([0, 1, 0], (len(points), 1)))
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['ceiling'], 'geometry_fill_min_area': .1,
                  'geometry_fill_max_gap': .30, 'geometry_fill_tolerance': .02, 'seed': 42}
        target = tmp_path/'planes.npz'
        extract_planes(votes, {'ceiling': 101}, np.zeros(3), .02, config, target)
        planes = PlaneIndex(target)
        try:
            result = planes.fill(np.array([[.5, .5, 1.], [.5, .5, .9], [3., 3., 1.]]), .02)
            assert result.tolist() == [101, 0, 0]
        finally:
            planes.close()
    finally:
        votes.close()


def test_geometry_fill_closes_hole_in_thin_six_centimetre_wall(tmp_path):
    wall = _grid((0, 0, 0), 1, 2, spacing=.06, hole=(.18, .58, .18, .58))
    votes = _multi_class_votes(tmp_path/'thin-wall.sqlite', [(100, wall)])
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['wall'], 'geometry_fill_min_area': .1,
                  'geometry_fill_max_gap': .30, 'geometry_fill_tolerance': .03, 'seed': 42}
        target = tmp_path/'thin-wall.npz'
        extract_planes(votes, {'wall': 100}, np.zeros(3), .02, config, target)
        planes = PlaneIndex(target)
        try:
            assert planes.fill(np.array([[0., .36, .36], [0., 1.2, 1.2]]), .03).tolist() == [100, 0]
        finally:
            planes.close()
    finally:
        votes.close()


def test_quality_defaults_and_door_prompt():
    preset = json.loads((PROJECT/'presets'/'default.json').read_text(encoding='utf-8'))
    door = next(item for item in preset['classes'] if item['id'] == 'door')
    assert (DEFAULTS['min_winning_share'], DEFAULTS['mask_confidence'], DEFAULTS['boundary_margin']) == (.5, .5, 2)
    assert DEFAULTS['geometry_fill_max_planes'] == 400
    assert 'doorway' not in door['prompts']


def _grid(origin, first, second, spacing=.02, hole=None):
    values = np.arange(0, .8, spacing)
    result = []
    for a in values:
        for b in values:
            if hole and hole[0] <= a < hole[1] and hole[2] <= b < hole[3]:
                continue
            point = np.asarray(origin, float)
            point[first] += a
            point[second] += b
            result.append(point)
    return result


def _multi_class_votes(path, groups):
    votes = VoteStore(path)
    codes = (100, 101, 102)
    batches = []
    for code, points in groups:
        keys = voxel_keys(np.asarray(points), np.zeros(3), .02)
        weights = np.zeros((len(keys), len(codes)), np.float32)
        weights[:, codes.index(code)] = 1
        batches.append((keys, weights))
    votes.add_scan(1, 'scan', batches, codes)
    votes.decide(1, .5, codes)
    return votes


def test_geometry_fill_extracts_multiple_separate_surfaces(tmp_path):
    ceiling_a = _grid((0, 0, 2), 0, 1, hole=(.2, .6, .2, .6))
    ceiling_b = _grid((2, 0, 3), 0, 1, hole=(.2, .6, .2, .6))
    roof = [np.array((4+x, y, 2+x)) for x in np.arange(0, .8, .02) for y in np.arange(0, .8, .02)
            if not (.2 <= x < .6 and .2 <= y < .6)]
    walls = []
    for origin, first, second in [((0, 2, 0), 1, 2), ((2, 2, 0), 1, 2), ((4, 2, 0), 1, 2), ((6, 2, 0), 1, 2)]:
        walls.extend(_grid(origin, first, second, hole=(.2, .6, .2, .6)))
    votes = _multi_class_votes(tmp_path/'multi.sqlite', [(101, ceiling_a+ceiling_b+roof), (100, walls)])
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['ceiling', 'wall'], 'geometry_fill_min_area': .1,
                  'geometry_fill_max_gap': .30, 'geometry_fill_tolerance': .03, 'geometry_fill_max_planes': 20}
        target = tmp_path/'multi.npz'
        report = extract_planes(votes, {'wall': 100, 'ceiling': 101}, np.zeros(3), .02, config, target)
        assert report['planes']['ceiling'] >= 3 and report['planes']['wall'] >= 4
        planes = PlaneIndex(target)
        try:
            points = np.array([[.4, .4, 2], [2.4, .4, 3], [4.4, .4, 2.4], [0, 2.4, .4], [2, 2.4, .4], [4, 2.4, .4], [6, 2.4, .4]])
            assert planes.fill(points, .03).tolist() == [101, 101, 101, 100, 100, 100, 100]
        finally:
            planes.close()
    finally:
        votes.close()


def test_geometry_fill_keeps_gap_between_coplanar_pieces(tmp_path):
    first = _grid((0, 0, 0), 1, 2)
    second = _grid((0, 2.8, 0), 1, 2)
    votes = _multi_class_votes(tmp_path/'coplanar.sqlite', [(100, first+second)])
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['wall'], 'geometry_fill_min_area': .1,
                  'geometry_fill_max_gap': .30, 'geometry_fill_tolerance': .03, 'geometry_fill_max_planes': 10}
        target = tmp_path/'coplanar.npz'
        extract_planes(votes, {'wall': 100}, np.zeros(3), .02, config, target)
        planes = PlaneIndex(target)
        try:
            assert planes.fill(np.array([[0, .4, .4], [0, 1.8, .4], [0, 3.2, .4]]), .03).tolist() == [100, 0, 100]
        finally:
            planes.close()
    finally:
        votes.close()


def test_plane_grid_matches_brute_force_plane_selection(tmp_path):
    ceiling = _grid((0, 0, 2), 0, 1)
    wall = _grid((3, 0, 0), 1, 2)
    votes = _multi_class_votes(tmp_path/'grid.sqlite', [(101, ceiling), (100, wall)])
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['ceiling', 'wall'], 'geometry_fill_min_area': .1,
                  'geometry_fill_max_gap': .1, 'geometry_fill_tolerance': .03, 'geometry_fill_max_planes': 10}
        target = tmp_path/'grid.npz'
        extract_planes(votes, {'wall': 100, 'ceiling': 101}, np.zeros(3), .02, config, target)
        planes = PlaneIndex(target)
        try:
            points = np.array([[.2, .2, 2.], [.7, .4, 2.], [3., .3, .3], [6., 6., 6.]])
            fast = planes.fill(points, .03)
            cells = np.unique(np.floor(points/planes.grid_size).astype(np.int64), axis=0)
            planes.grid = {tuple(map(int, cell)): list(range(len(planes.planes))) for cell in cells}
            brute_force = planes.fill(points, .03)
            np.testing.assert_array_equal(fast, brute_force)
        finally:
            planes.close()
    finally:
        votes.close()


def test_geometry_fill_is_stable_after_small_vote_perturbation_and_ignores_islands(tmp_path):
    ceiling = _grid((0, 0, 2), 0, 1, spacing=.05, hole=(.65, .95, .65, .95))
    wall_a = _grid((0, 0, 0), 1, 2, spacing=.05)
    wall_b = _grid((2, 0, 0), 1, 2, spacing=.05)
    island = _grid((8, 8, 2), 0, 1, spacing=.05)[:10]
    config = {**DEFAULTS, 'geometry_fill_classes': ['ceiling', 'wall'], 'geometry_fill_min_area': .1,
              'geometry_fill_max_gap': .30, 'geometry_fill_tolerance': .03,
              'geometry_fill_min_component_voxels': 20, 'seed': 91}
    probes = np.array([(x, y, 2.) for x in np.arange(.68, .92, .05) for y in np.arange(.68, .92, .05)])
    counts = []
    for number, removed in enumerate((set(), set(range(0, len(ceiling), 200)))):
        points = [point for index, point in enumerate(ceiling) if index not in removed]
        votes = _multi_class_votes(tmp_path/f'stable-{number}.sqlite', [(101, points+island), (100, wall_a+wall_b)])
        try:
            target = tmp_path/f'stable-{number}.npz'
            extract_planes(votes, {'wall': 100, 'ceiling': 101}, np.zeros(3), .02, config, target)
            planes = PlaneIndex(target)
            try:
                counts.append(int(np.count_nonzero(planes.fill(probes, .03))))
                assert planes.fill(np.array([[8.1, 8.1, 2.]]), .03).tolist() == [0]
            finally:
                planes.close()
        finally:
            votes.close()
    assert abs(counts[1]-counts[0]) <= max(1, round(counts[0]*.02))


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get('NOMIVOX_RUN_SLOW'), reason='set NOMIVOX_RUN_SLOW=1 to run the million-voxel benchmark')
def test_geometry_fill_million_voxel_runtime(tmp_path):
    side = 1000
    indexes = np.arange(side*side)
    points = np.column_stack(((indexes % side)*.02, (indexes // side)*.02, np.full(side*side, 2.)))
    votes = _multi_class_votes(tmp_path/'million.sqlite', [(101, points)])
    try:
        config = {**DEFAULTS, 'geometry_fill_classes': ['ceiling'], 'geometry_fill_min_area': .5,
                  'geometry_fill_max_planes': 2}
        report = extract_planes(votes, {'ceiling': 101}, np.zeros(3), .02, config, tmp_path/'million.npz')
        assert report['planes']['ceiling'] == 1
    finally:
        votes.close()
