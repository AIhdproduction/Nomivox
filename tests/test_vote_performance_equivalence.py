"""Exact-regression coverage for the voting performance optimizations."""

from copy import deepcopy

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from e57cls.geometry.fill import CELL_SIZE, _cells_from_uv, _merge_coplanar_pieces, _split_cells
from e57cls.vote.engine import VoteStore, _touching_box_labels
from e57cls.vote.geometry_rules import _tree_seeds


def _reference_tree_seeds(rows, config):
    if len(rows) < 4:
        return []
    coordinates = np.asarray([row[:3] for row in rows], np.int64)
    count, labels = VoteStore._sparse_components(coordinates)
    seeds = []
    for component in range(count):
        points = coordinates[labels == component]
        if (points[:, 2].ptp()*config['geometry_cell_size']) < .8:
            continue
        centered = points-points.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        linearity = np.median(np.asarray([row[3] for row in np.asarray(rows, object)[labels == component]], np.float64))
        if abs(vectors[0, 2]) < .7 or linearity < .35:
            continue
        seeds.append((float(points[:, 0].mean()), float(points[:, 1].mean()), int(points[:, 2].max())))
    return seeds


class _RowsCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _TreeVotes:
    _sparse_components = staticmethod(VoteStore._sparse_components)

    def __init__(self, rows):
        self.rows = rows
        self.db = self

    def execute(self, *_):
        return _RowsCursor(self.rows)


def test_tree_seeds_match_reference_for_seeded_components_and_edge_cases():
    rng = np.random.default_rng(20261021)
    config = {'geometry_cell_size': .2}
    cases = [[], [(0, 0, index, .5) for index in range(5)]]
    rows = []
    for component, x in enumerate((0, 10, 20, 30)):
        for z in range(5):
            rows.append((x, component, z, float(rng.choice((.2, .4, .8)))))
    cases.append(rows)
    for rows in cases:
        expected = _reference_tree_seeds(rows, config)
        actual = _tree_seeds(_TreeVotes(rows), {'tree_trunk': 7}, config)
        assert actual == expected


def _reference_split_cells(cells):
    cells = np.unique(np.asarray(cells, np.int32), axis=0)
    if len(cells) < 2:
        return [cells]
    values = np.ascontiguousarray(cells, dtype='<i4').view('V8').reshape(-1)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    source, target = [], []
    for offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
        shifted = np.ascontiguousarray(cells+np.asarray(offset, np.int32), dtype='<i4').view('V8').reshape(-1)
        found = np.searchsorted(sorted_values, shifted)
        present = found < len(sorted_values)
        present[present] &= sorted_values[found[present]] == shifted[present]
        ids = np.flatnonzero(present)
        if len(ids):
            source.append(ids)
            target.append(order[found[ids]])
    if not source:
        return [cells[index:index+1] for index in range(len(cells))]
    graph = coo_matrix((np.ones(sum(map(len, source)), np.uint8),
                       (np.concatenate(source), np.concatenate(target))), shape=(len(cells), len(cells)))
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return [cells[labels == index] for index in range(count)]


def test_split_cells_matches_reference_order_for_seeded_and_edge_data():
    rng = np.random.default_rng(20261022)
    cases = [np.empty((0, 2), np.int32), np.array([[3, -2]], np.int32),
             np.array([[0, 0], [1, 1], [10, 0], [11, 0], [20, 0], [21, 1]], np.int32),
             rng.integers(-30, 31, size=(600, 2), dtype=np.int32)]
    for cells in cases:
        expected, actual = _reference_split_cells(cells), _split_cells(cells)
        assert len(actual) == len(expected)
        assert all(np.array_equal(item, reference) for item, reference in zip(actual, expected))


def _reference_merge_coplanar_pieces(pieces, tolerance):
    merged = []
    cosine = np.cos(np.deg2rad(3.))
    for piece in pieces:
        for current in merged:
            if abs(float(np.dot(piece['normal'], current['normal']))) < cosine:
                continue
            if abs(float(np.dot(piece['center']-current['center'], current['normal']))) >= tolerance:
                continue
            uv = piece['cells'].astype(np.float64)*CELL_SIZE
            world = uv@piece['basis'] + piece['normal']*np.dot(piece['center'], piece['normal'])
            projected = _cells_from_uv(world@current['basis'].T)
            combined = np.unique(np.vstack((current['cells'], projected)), axis=0)
            if len(_reference_split_cells(combined)) == 1:
                current['cells'] = combined
                current['area'] += piece['area']
                break
        else:
            merged.append(piece)
    return merged


def _piece(cells, normal=(0., 0., 1.), center=(0., 0., 0.), area=1.):
    return {'cells': np.asarray(cells, np.int32), 'normal': np.asarray(normal, np.float64),
            'center': np.asarray(center, np.float64), 'basis': np.eye(2, 3), 'area': area}


def test_merge_coplanar_pieces_matches_reference_and_first_match_order():
    rng = np.random.default_rng(20261024)
    random_pieces = [_piece(rng.integers(-50, 51, size=(rng.integers(1, 5), 2), dtype=np.int32),
                            center=rng.uniform(-1., 1., 3), area=float(index+1))
                     for index in range(40)]
    cases = [[_piece([[0, 0]])],
             [_piece([[0, 0], [1, 0]], area=2.), _piece([[2, 0]], area=1.),
              _piece([[10, 0]], area=3.), _piece([[11, 1]], area=4.),
              _piece([[0, 5]], normal=(0., 1., 0.), area=5.)]+random_pieces]
    for pieces in cases:
        expected = _reference_merge_coplanar_pieces(deepcopy(pieces), .02)
        actual = _merge_coplanar_pieces(deepcopy(pieces), .02)
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            assert item['area'] == reference['area']
            for key in ('cells', 'normal', 'center', 'basis'):
                np.testing.assert_array_equal(item[key], reference[key])


def _reference_touching_box_labels(low, high, maximum_gap):
    parents = np.arange(len(low), dtype=np.int64)

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for first in range(len(low)):
        for second in range(first):
            if np.maximum(low[first]-high[second], low[second]-high[first]).max() <= maximum_gap:
                parents[find(first)] = find(second)
    return np.asarray([find(index) for index in range(len(low))])


def test_touching_box_groups_match_nested_pair_reference_at_boundaries():
    rng = np.random.default_rng(20261023)
    cases = [(np.empty((0, 3)), np.empty((0, 3)), .2),
             (np.zeros((1, 3)), np.ones((1, 3)), .2),
             (np.array([[0., 0., 0.], [1.2, 0., 0.], [4., 0., 0.]]),
              np.array([[1., 1., 1.], [2., 1., 1.], [5., 1., 1.]]), .2)]
    low = rng.uniform(-20, 20, (200, 3))
    cases.append((low, low+rng.uniform(.01, 2., (200, 3)), .2))
    for low, high, gap in cases:
        expected = _reference_touching_box_labels(low, high, gap)
        actual = _touching_box_labels(low, high, gap)
        np.testing.assert_array_equal(actual[:, None] == actual, expected[:, None] == expected)
