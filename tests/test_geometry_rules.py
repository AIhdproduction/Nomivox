import json

import numpy as np

from e57cls.output.classify import ClassifiedSource
from e57cls.pipeline.config import DEFAULTS
from e57cls.vote.engine import VoteStore, voxel_keys
from e57cls.vote.geometry_rules import TerrainIndex, _features, apply_geometry_rules


def _ground_votes(path):
    votes = VoteStore(path)
    points = np.array([[x*.1, y*.1, 0.] for x in range(9) for y in range(9)], np.float64)
    keys = voxel_keys(points, np.zeros(3), .02)
    weights = np.tile(np.array([[1., 0., 0.]], np.float32), (len(keys), 1))
    votes.add_scan(1, 'geometry', [(keys, weights)], [2, 102, 5])
    votes.decide(1, .5, [2, 102, 5])
    return votes, keys


def test_vote_family_rescues_split_ground_floor_evidence(tmp_path):
    votes = VoteStore(tmp_path/'family.sqlite')
    try:
        keys = voxel_keys([[0., 0., 0.]], np.zeros(3), .02)
        votes.add_scan(1, 'family', [(keys, np.array([[.31, .30, .39]], np.float32))], [2, 102, 6])
        votes.decide(1, .5, [2, 102, 6], vote_families=[[2, 102]])
        assert votes.lookup(keys)[0].tolist() == [2]
    finally:
        votes.close()


def test_geometry_rules_persist_a_thin_terrain_raster_deterministically(tmp_path):
    reports = []
    for index in range(2):
        votes, _ = _ground_votes(tmp_path/f'votes_{index}.sqlite')
        try:
            terrain = tmp_path/f'terrain_{index}.npz'
            reports.append(apply_geometry_rules(votes, {'ground': 2, 'floor': 102, 'tree': 5}, np.zeros(3), .02,
                                                {**DEFAULTS}, terrain))
            with np.load(terrain, allow_pickle=False) as artifact:
                assert len(artifact['xy']) > 0
                assert not artifact['covered'].any()
        finally:
            votes.close()
    assert reports[0]['terrain']['cells'] == reports[1]['terrain']['cells']


def test_terrain_fill_assigns_ground_with_its_own_confidence(tmp_path):
    terrain = tmp_path/'terrain.npz'
    np.savez_compressed(terrain, metadata=np.asarray(json.dumps({'cell_size': .5})),
                        xy=np.array([[0., 0.]]), z=np.array([0.]), covered=np.array([False]))

    class Votes:
        def lookup(self, keys):
            return np.ones(len(keys), np.uint8), np.zeros(len(keys), np.uint8)

    store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {'pose': {
        'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.},
        'translation': {'x': 0., 'y': 0., 'z': 0.}}}}})()
    with ClassifiedSource(tmp_path, [], Votes(), np.zeros(3), .02, terrain=terrain, terrain_fill=True) as source:
        _, codes, confidence, _ = source.block(store, 0, np.array([[.1, .1, .04]]), np.array([True]), 'original')
    assert codes.tolist() == [2]
    assert confidence.tolist() == [64]
    with ClassifiedSource(tmp_path, [], Votes(), np.zeros(3), .02, terrain=terrain, terrain_fill=True,
                          terrain_outdoor_scans=set()) as source:
        _, codes, confidence, _ = source.block(store, 0, np.array([[.1, .1, .04]]), np.array([True]), 'original')
    assert codes.tolist() == [1]
    assert confidence.tolist() == [0]


def test_facade_and_crown_features_are_distinguishable():
    facade = np.array([[0., y, z] for y in np.linspace(-.4, .4, 9) for z in np.linspace(0., .8, 9)])
    crown = np.array([[x, y, z] for x in (-.15, 0., .15) for y in (-.15, 0., .15) for z in (1.8, 2., 2.2)])
    facade_shape = _features(facade, 24, 1.)[:, 1:3].mean(axis=0)
    crown_shape = _features(crown, 24, 1.)[:, 1:3].mean(axis=0)
    assert facade_shape[0] > crown_shape[0]
    assert crown_shape[1] > facade_shape[1]


def test_trunk_crown_and_ceiling_scenes_have_stable_descriptors():
    trunk = np.array([[0., 0., z] for z in np.linspace(0., 1.2, 25)])
    ceiling = np.array([[x, y, 3.] for x in np.linspace(-.4, .4, 9) for y in np.linspace(-.4, .4, 9)])
    trunk_shape = _features(trunk, 24, 1.)
    ceiling_shape = _features(ceiling, 24, 1.)
    assert trunk_shape[:, 0].mean() > .8
    assert ceiling_shape[:, 1].mean() > .5


def test_geometry_tile_loops_checkpoint_for_bounded_work(tmp_path):
    votes, _ = _ground_votes(tmp_path/'bounded.sqlite')
    calls = []
    try:
        report = apply_geometry_rules(votes, {'ground': 2, 'floor': 102, 'tree': 5}, np.zeros(3), .02,
                                      {**DEFAULTS, 'geometry_tile_size': .3}, tmp_path/'bounded.npz',
                                      checkpoint=lambda: calls.append(1))
    finally:
        votes.close()
    assert report['terrain']['cells'] > 0
    assert len(calls) > 4


def test_geometry_rules_gate_outdoor_targets_for_indoor_scan(tmp_path):
    votes = VoteStore(tmp_path/'indoor.sqlite')
    try:
        grid = np.meshgrid(np.arange(7)*.1, np.arange(7)*.1, np.arange(7)*.1, indexing='ij')
        points = np.column_stack([axis.ravel() for axis in grid])
        keys = voxel_keys(points, np.zeros(3), .02)
        weights = np.tile(np.array([[.8, .2]], np.float32), (len(keys), 1))
        votes.add_scan(1, 'indoor', [(keys, weights)], [101, 100])
        votes.decide(1, .5, [101, 100])
        report = apply_geometry_rules(votes, {'ceiling': 101, 'wall': 100}, np.zeros(3), .02,
                                      {**DEFAULTS}, tmp_path/'terrain.npz',
                                      scan_context={1: {'outdoor': False, 'allowed_codes': {101, 100}}})
        codes, _ = votes.lookup(keys)
    finally:
        votes.close()
    assert set(codes.tolist()) <= {1, 100, 101}
    assert not report['fallbacks']


def test_geometry_rules_fall_back_when_an_outdoor_target_is_inactive(tmp_path):
    votes = VoteStore(tmp_path/'outdoor.sqlite')
    try:
        grid = np.meshgrid(np.arange(7)*.1, np.arange(7)*.1, np.arange(7)*.1, indexing='ij')
        points = np.column_stack([axis.ravel() for axis in grid])
        keys = voxel_keys(points, np.zeros(3), .02)
        weights = np.tile(np.array([[.8, .2]], np.float32), (len(keys), 1))
        votes.add_scan(1, 'outdoor', [(keys, weights)], [101, 100])
        votes.decide(1, .5, [101, 100])
        report = apply_geometry_rules(votes, {'ceiling': 101, 'wall': 100}, np.zeros(3), .02,
                                      {**DEFAULTS}, tmp_path/'terrain.npz',
                                      scan_context={1: {'outdoor': True, 'allowed_codes': {101, 100}}})
    finally:
        votes.close()
    assert report['fallbacks'].get('tree:alternative', 0) > 0


def test_indoor_slope_is_covered_and_keeps_floor(tmp_path):
    votes = VoteStore(tmp_path/'slope.sqlite')
    try:
        xx, yy = np.meshgrid(np.arange(9)*.1, np.arange(9)*.1, indexing='ij')
        floor = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        ceiling = np.column_stack((xx.ravel(), yy.ravel(), 2.+.8*xx.ravel()))
        points = np.vstack((floor, ceiling))
        keys = voxel_keys(points, np.zeros(3), .02)
        weights = np.zeros((len(keys), 2), np.float32)
        weights[:len(floor), 0] = .9
        weights[len(floor):, 1] = .9
        votes.add_scan(1, 'indoor', [(keys, weights)], [102, 101])
        votes.decide(1, .5, [102, 101])
        terrain = tmp_path/'terrain.npz'
        apply_geometry_rules(votes, {'floor': 102, 'ceiling': 101}, np.zeros(3), .02, {**DEFAULTS}, terrain,
                             scan_context={1: {'outdoor': False, 'allowed_codes': {102, 101}}})
        codes, _ = votes.lookup(keys[:len(floor)])
        with np.load(terrain, allow_pickle=False) as artifact:
            assert artifact['covered'].any()
    finally:
        votes.close()
    assert set(codes.tolist()) == {102}


def test_terrain_ignores_distant_indoor_features_without_a_neighbour(tmp_path):
    votes = VoteStore(tmp_path/'terrain-gap.sqlite')
    try:
        xx, yy = np.meshgrid(np.arange(9)*.1, np.arange(9)*.1, indexing='ij')
        ground = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        indoor = np.array([[.4, .4, 2.], [20., 0., 2.]])
        ground_keys = voxel_keys(ground, np.zeros(3), .02)
        indoor_keys = voxel_keys(indoor, np.zeros(3), .02)
        votes.add_scan(1, 'outdoor-ground', [(ground_keys, np.ones((len(ground_keys), 1), np.float32))], [102])
        votes.add_scan(2, 'indoor-ceiling', [(indoor_keys, np.ones((len(indoor_keys), 1), np.float32))], [101])
        votes.decide(1, .5, [102, 101])
        terrain = tmp_path/'terrain.npz'
        apply_geometry_rules(votes, {'floor': 102, 'ceiling': 101}, np.zeros(3), .02, {**DEFAULTS}, terrain,
                             scan_context={1: {'outdoor': True, 'allowed_codes': {102}},
                                           2: {'outdoor': False, 'allowed_codes': {101}}})
        with np.load(terrain, allow_pickle=False) as artifact:
            assert artifact['covered'].any()
    finally:
        votes.close()


def test_mixed_scan_context_applies_tree_rule_only_outdoors(tmp_path):
    votes = VoteStore(tmp_path/'mixed.sqlite')
    try:
        grid = np.meshgrid(np.arange(7)*.1, np.arange(7)*.1, np.arange(7)*.1, indexing='ij')
        outside = np.column_stack([axis.ravel() for axis in grid])
        inside = outside+np.array((3., 0., 0.))
        outside_keys = voxel_keys(outside, np.zeros(3), .02)
        inside_keys = voxel_keys(inside, np.zeros(3), .02)
        weights = np.ones((len(outside_keys), 1), np.float32)
        votes.add_scan(1, 'outdoor', [(outside_keys, weights)], [101])
        votes.add_scan(2, 'indoor', [(inside_keys, weights)], [101])
        votes.decide(1, .5, [101, 5])
        apply_geometry_rules(votes, {'ceiling': 101, 'tree': 5}, np.zeros(3), .02, {**DEFAULTS}, tmp_path/'terrain.npz',
                             scan_context={1: {'outdoor': True, 'allowed_codes': {101, 5}},
                                           2: {'outdoor': False, 'allowed_codes': {101}}})
        outside_codes, _ = votes.lookup(outside_keys)
        inside_codes, _ = votes.lookup(inside_keys)
    finally:
        votes.close()
    assert (outside_codes == 5).any()
    assert set(inside_codes.tolist()) == {101}


def test_covered_ground_floor_rule_respects_resolved_scene(tmp_path):
    xx, yy = np.meshgrid(np.arange(9)*.1, np.arange(9)*.1, indexing='ij')
    ground = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    canopy = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 3.)))

    def classify(path, scene, ground_code):
        votes = VoteStore(path)
        keys = voxel_keys(np.vstack((ground, canopy)), np.zeros(3), .02)
        weights = np.zeros((len(keys), 3), np.float32)
        weights[:len(ground), 0] = .9
        weights[len(ground):, 2] = .9
        codes = [ground_code, 102 if ground_code == 2 else 2, 100]
        votes.add_scan(1, scene, [(keys, weights)], codes)
        votes.decide(1, .5, codes)
        apply_geometry_rules(votes, {'ground': 2, 'floor': 102, 'building': 100}, np.zeros(3), .02,
                             {**DEFAULTS}, tmp_path/f'{scene}-{ground_code}.npz',
                             scan_context={1: {'scene': scene, 'outdoor': scene == 'outdoor',
                                               'position': [0., 0., 0.], 'allowed_codes': {2, 102, 100}}})
        values = votes.lookup(keys[:len(ground)])[0]
        votes.close()
        return values

    assert set(classify(tmp_path/'outdoor.sqlite', 'outdoor', 2).tolist()) == {2}
    assert set(classify(tmp_path/'indoor.sqlite', 'indoor', 2).tolist()) == {102}
    assert set(classify(tmp_path/'outdoor-floor.sqlite', 'outdoor', 102).tolist()) == {2}
