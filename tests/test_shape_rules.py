"""Focused file-SQLite georeferenced scene tests for shape rules (stage 2a)."""

import os
import json
import threading
import time
from pathlib import Path
import numpy as np
import psutil
import pytest

from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline.reclassify import _postprocess
from e57cls.geometry.fill import extract_planes
from e57cls.vote.engine import VoteStore, voxel_keys
from e57cls.output.classify import ClassifiedSource
from e57cls.vote.scene_consistency import enforce_scene_consistency
from e57cls.vote.segments import apply_segment_vote
from e57cls.vote.geometry_rules import _tree_seeds, prepare_geometry_features
from e57cls.vote.scene_context import ensure_scene_context, refine_scene_context_with_outline
import e57cls.vote.shape_rules as shape_module
from e57cls.vote.shape_rules import OBJECT_RULE_NAMES, OUTDOOR_RULE_NAMES, RULE_NAMES, _code, apply_shape_rules


def test_shape_rules_default_is_enabled():
    assert DEFAULTS['shape_rules'] is True


def test_shape_rules_has_all_round_one_defaults():
    assert {'shape_min_vote_share', 'shape_object_cell', 'shape_level_cell', 'shape_cabinet_offset',
            'shape_clutter_max_size', 'shape_layer_sigma_factor', 'shape_layer_max_band',
            'shape_pipe_min_length', 'shape_ceiling_equipment_max_drop', 'tree_crown_fill'} <= set(DEFAULTS)


def test_shape_rules_class_alias_prefers_current_preset_name():
    assert _code({'construction_material': 76, 'clutter': 7}, 'construction_material', 'clutter') == 76


def test_shape_rules_class_alias_accepts_legacy_name():
    assert _code({'clutter': 76}, 'construction_material', 'clutter') == 76


# Georeferencing constants: 30° rotation around z + Swiss-scale translation
SHIFT = np.array([2665953.0, 1207269.0, 440.0], dtype=np.float64)
ANGLE = np.deg2rad(30.0)
ROT_Z = np.array([
    [np.cos(ANGLE), -np.sin(ANGLE), 0.0],
    [np.sin(ANGLE), np.cos(ANGLE), 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

BY_ID = {
    'wall': 100,
    'ceiling': 101,
    'floor': 102,
    'column': 108,
    'beam': 109,
    'steel_structure': 110,
    'pipe': 111,
    'cable': 112,
    'electrical': 113,
    'hvac': 114,
    'ceiling_equipment': 116,
    'furniture': 117,
    'appliances': 118,
    'sanitary_fixtures': 63,
    'railing': 107,
    'construction_material': 76,
    'stairs': 105,
    'door': 103,
    'window': 104,
    'reflection': 68,
    'building': 6,
    'ground': 2,
}
ALL_CODES = list(BY_ID.values())

OUTDOOR_BY_ID = {
    **BY_ID,
    'road': 11,
    'sidewalk': 73,
    'curb': 74,
    'outdoor_wall': 78,
    'utility_pole': 84,
    'traffic_sign': 81,
    'fence': 77,
    'vehicle': 91,
    'power_line': 14,
    'tree': 5,
    'tree_trunk': 95,
}


def _config(**overrides):
    return {
        **DEFAULTS,
        'geometry_tile_size': 2.0,
        'geometry_halo': 0.3,
        'geometry_knn_radius': 0.5,
        **overrides,
    }


def _write_benchmark_ransac_plane(path):
    """Write one valid, non-overlapping RANSAC wall for SQLite stress fixtures."""
    cells = np.asarray([(x, z) for x in range(41) for z in range(41)], np.int32)
    metadata = {
        'cell_size': .05,
        'planes': [{
            'class_id': 'wall', 'normal': [0., 1., 0.], 'center': [0., .5, 0.],
            'basis': [[1., 0., 0.], [0., 0., 1.]], 'area_m2': 4., 'cells': 'wall_cells',
            'bbox_min': [-.05, .49, -.05], 'bbox_max': [2.05, .51, 2.05],
        }],
    }
    np.savez(path, metadata=json.dumps(metadata), wall_cells=cells)
    return path


def _transform(local_pts):
    return np.asarray(local_pts, np.float64) @ ROT_Z.T + SHIFT


def _surface_grid(x_range, y_range, z_val):
    gx, gy = np.meshgrid(x_range, y_range)
    return np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, z_val)))


def _vertical_grid(x_range, y_val, z_range):
    gx, gz = np.meshgrid(x_range, z_range)
    return np.column_stack((gx.ravel(), np.full(gx.size, y_val), gz.ravel()))


def _vertical_grid_y(x_val, y_range, z_range):
    gy, gz = np.meshgrid(y_range, z_range)
    return np.column_stack((np.full(gy.size, x_val), gy.ravel(), gz.ravel()))


def _noisy_wall(room_x, amplitude=.019, holes=True):
    """Return a deterministic +/-2 cm wall with optional small SLAM-like holes."""
    xs = np.arange(0.0, room_x + .01, .02)
    zs = np.arange(.02, 2.59, .02)
    gx, gz = np.meshgrid(xs, zs)
    y = float(amplitude)*np.sin(gx*7. + gz*11.)
    points = np.column_stack((gx.ravel(), y.ravel(), gz.ravel()))
    if holes:
        keep = ((np.rint(gx.ravel()/.02).astype(np.int64)+3*np.rint(gz.ravel()/.02).astype(np.int64)) % 7) != 0
        points = points[keep]
    return points


def _box_surfaces(x0, x1, y0, y1, z0, z1, step=0.02):
    xs = np.arange(x0, x1 + step * 0.5, step)
    ys = np.arange(y0, y1 + step * 0.5, step)
    zs = np.arange(z0, z1 + step * 0.5, step)
    pts = [
        _vertical_grid(xs, y0, zs),
        _vertical_grid(xs, y1, zs),
    ]
    if len(ys) > 2:
        pts.append(_vertical_grid_y(x0, ys[1:-1], zs))
        pts.append(_vertical_grid_y(x1, ys[1:-1], zs))
    return np.vstack(pts)


def _cylinder_x(x0, x1, center_y, center_z, radius, step=.02):
    """Surface samples for a horizontal round pipe in the local scene frame."""
    xs = np.arange(x0, x1 + step*.5, step)
    angles = np.linspace(0., 2.*np.pi, max(12, int(np.ceil(2.*np.pi*radius/step))), endpoint=False)
    gx, ga = np.meshgrid(xs, angles)
    return np.column_stack((gx.ravel(),
                            center_y+radius*np.cos(ga.ravel()),
                            center_z+radius*np.sin(ga.ravel())))


def _cylinder_z(center_x, center_y, z0, z1, radius, step=.02):
    """Surface samples for a vertical cylinder in the local scene frame."""
    zs = np.arange(z0, z1 + step*.5, step)
    angles = np.linspace(0., 2.*np.pi, max(12, int(np.ceil(2.*np.pi*radius/step))), endpoint=False)
    gz, ga = np.meshgrid(zs, angles)
    return np.column_stack((center_x+radius*np.cos(ga.ravel()),
                            center_y+radius*np.sin(ga.ravel()), gz.ravel()))


def _crown_ring(center_x, center_y, z, radius, step=.04):
    angles = np.linspace(0., 2.*np.pi, max(16, int(np.ceil(2.*np.pi*radius/step))), endpoint=False)
    return np.column_stack((center_x+radius*np.cos(angles), center_y+radius*np.sin(angles),
                            np.full(len(angles), z)))


def _build_scene(tmp_path, name, object_parts, room_x=4.0, room_y=4.0, wall_y=0.0,
                 second_storey=False, outdoor=False, wall_points=None, wall_votes=None, ceiling_points=None):
    """Build a georeferenced synthetic room with floor, ceiling, wall, and object parts."""
    wall_points = (_vertical_grid(np.arange(0.0, room_x + 0.01, 0.02), wall_y,
                                  np.arange(0.02, 2.59, 0.02)) if wall_points is None else wall_points)
    ceiling_points = (_surface_grid(np.arange(0.0, room_x + 0.01, 0.02),
                                    np.arange(0.0, room_y + 0.01, 0.02), 2.6)
                      if ceiling_points is None else ceiling_points)
    wall_votes = [(100, 1.0)] if wall_votes is None else wall_votes
    room_parts = [
        # Floor (z=0, 102)
        (_surface_grid(np.arange(0.0, room_x + 0.01, 0.02),
                       np.arange(0.0, room_y + 0.01, 0.02), 0.0), [(102, 1.0)]),
        # Ceiling (z=2.6, 101)
        (ceiling_points, [(101, 1.0)]),
        # Wall (y=wall_y, 100)
        (wall_points, wall_votes),
    ]
    if second_storey:
        room_parts.extend([
            (_surface_grid(np.arange(0.0, room_x + 0.01, 0.02),
                           np.arange(0.0, room_y + 0.01, 0.02), 3.0), [(102, 1.0)]),
            (_surface_grid(np.arange(0.0, room_x + 0.01, 0.02),
                           np.arange(0.0, room_y + 0.01, 0.02), 5.6), [(101, 1.0)]),
            (_vertical_grid(np.arange(0.0, room_x + 0.01, 0.02), wall_y,
                            np.arange(2.62, 5.59, 0.02)), [(100, 1.0)]),
        ])

    all_parts = room_parts + list(object_parts)
    all_world = np.vstack([_transform(pts) for pts, _ in all_parts])
    origin = np.floor(all_world.min(axis=0) / 0.02) * 0.02 - 0.04

    # Build unique keys and associated weights
    unique_keys_map = {}
    object_keys_list = []

    for idx, (pts, vote_dist) in enumerate(all_parts):
        world = _transform(pts)
        keys = voxel_keys(world, origin, 0.02)
        unique_k = np.unique(keys)
        if idx >= len(room_parts):
            object_keys_list.append(unique_k)
        for k in unique_k:
            bkey = bytes(k)
            if bkey not in unique_keys_map:
                weights = np.zeros(len(ALL_CODES), dtype=np.float32)
                for code, weight in vote_dist:
                    weights[ALL_CODES.index(code)] = float(weight)
                unique_keys_map[bkey] = (k, weights)

    dense_keys = np.asarray([item[0] for item in unique_keys_map.values()], dtype='V24')
    dense_weights = np.asarray([item[1] for item in unique_keys_map.values()], dtype=np.float32)

    store = VoteStore(tmp_path / f'{name}.sqlite')
    store.add_scan(1, '1', [(dense_keys, dense_weights)], ALL_CODES)

    scanner_pos = _transform([room_x / 2.0, room_y / 2.0, 1.5]).tolist()
    scan_context = {
        1: {
            'scene': 'outdoor' if outdoor else 'indoor',
            'outdoor': bool(outdoor),
            'position': scanner_pos,
            'allowed_codes': set(ALL_CODES),
        }
    }
    return store, origin, scan_context, object_keys_list


def _run_pipeline(store, origin, config=None, scan_context=None, min_obs=1, min_share=0.5, ransac=True):
    cfg = _config(**(config or {}))
    store.decide(min_obs, min_share, ALL_CODES)
    seg_report = apply_segment_vote(store, BY_ID, origin, 0.02, cfg, active_codes=set(ALL_CODES))
    planes_path = None
    if ransac:
        planes_path = store.path.with_name(store.path.stem+'_planes.npz')
        extract_planes(store, BY_ID, origin, .02, cfg, planes_path)
    shape_report = apply_shape_rules(store, BY_ID, origin, 0.02, cfg, active_codes=set(ALL_CODES),
                                     scan_context=scan_context, planes_path=planes_path)
    return seg_report, shape_report


def _build_outdoor_scene(tmp_path, scene_name, parts, *, by_id=OUTDOOR_BY_ID, ground_points=None):
    """Build a rotated, georeferenced outdoor file-SQLite fixture.

    ``parts`` are ``(name, local_points, votes)`` records.  A caller may add
    sparse ground samples solely to provide a terrain datum without merging a
    full ground plane into the object being tested.
    """
    codes = list(dict.fromkeys(by_id.values()))
    materialized = []
    if ground_points is not None:
        materialized.append(('ground', np.asarray(ground_points, np.float64), [(by_id['ground'], 1.)]))
    materialized.extend((str(name), np.asarray(points, np.float64), list(votes)) for name, points, votes in parts)
    all_world = np.vstack([_transform(points) for _, points, _ in materialized])
    origin = np.floor(all_world.min(axis=0)/.02)*.02-.04
    records, object_keys = {}, {}
    for name, points, weights in materialized:
        keys = np.unique(voxel_keys(_transform(points), origin, .02))
        object_keys[name] = keys
        vector = np.zeros(len(codes), np.float32)
        for code, weight in weights:
            vector[codes.index(int(code))] = float(weight)
        for key in keys:
            records.setdefault(bytes(key), (key, vector.copy()))
    dense_keys = np.asarray([item[0] for item in records.values()], dtype='V24')
    dense_weights = np.asarray([item[1] for item in records.values()], dtype=np.float32)
    store = VoteStore(tmp_path / f'{scene_name}.sqlite')
    store.add_scan(1, '1', [(dense_keys, dense_weights)], codes)
    scan_context = {1: {'scene': 'outdoor', 'outdoor': True,
                        'position': _transform([-.5, -3., 2.]).tolist(), 'allowed_codes': set(codes)}}
    return store, origin, scan_context, object_keys, codes


def _run_outdoor_pipeline(store, origin, scan_context, codes, *, by_id=OUTDOOR_BY_ID, config=None,
                          min_share=.5, with_outline=True):
    """Run segment voting, actual RANSAC extraction, and shape rules for one outdoor fixture."""
    cfg = _config(**(config or {}))
    store.decide(1, min_share, codes)
    apply_segment_vote(store, by_id, origin, .02, cfg, active_codes=set(codes))
    planes_path = store.path.with_name(store.path.stem+'_planes.npz')
    extract_planes(store, by_id, origin, .02, cfg, planes_path)
    outline = None
    if with_outline:
        outline = store.building_outline([by_id[name] for name in ('building', 'wall') if name in by_id], .02,
                                         cfg['building_boundary_cell_size'], cfg['building_boundary_min_height'],
                                         cfg['building_boundary_max_cells'])
    report = apply_shape_rules(store, by_id, origin, .02, cfg, active_codes=set(codes),
                               scan_context=scan_context, planes_path=planes_path, outline=outline)
    assert planes_path.is_file()
    return report, outline


def _object_decision_shares(store, keys):
    blob_keys = [bytes(k) for k in keys]
    codes = []
    chunk_size = 5000
    cursor = store.db.cursor()
    for i in range(0, len(blob_keys), chunk_size):
        chunk = blob_keys[i:i + chunk_size]
        q = f"SELECT code FROM decisions WHERE key IN ({','.join(['?'] * len(chunk))})"
        codes.extend(row[0] for row in cursor.execute(q, chunk))
    if not codes:
        return {}
    unique, counts = np.unique(codes, return_counts=True)
    total = len(codes)
    return {int(u): count / total for u, count in zip(unique, counts)}


def _structure_layer_count(store, keys, kind=None):
    """Count tested layer keys in bounded SQLite parameter blocks."""
    blob_keys = [bytes(key) for key in keys]
    total = 0
    for start in range(0, len(blob_keys), 5000):
        chunk = blob_keys[start:start+5000]
        where = f"key IN ({','.join('?' for _ in chunk)})"
        if kind is not None:
            where = 'kind=? AND '+where
            params = (int(kind), *chunk)
        else:
            params = chunk
        total += store.db.execute(f'SELECT COUNT(DISTINCT key) FROM shape_structure_voxels WHERE {where}', params).fetchone()[0]
    return total


# --- Mandatory Test Cases 1 - 8 ---

def test_shape_rules_freestanding_column_becomes_column(tmp_path):
    """1. Freestanding column 0.4x0.4x2.6m, mixed votes -> column."""
    # Split column points in two halves: one with wall .6/col .4, one with col .6/wall .4
    pts = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    half = len(pts) // 2
    obj_parts = [
        (pts[:half], [(100, 0.6), (108, 0.4)]),
        (pts[half:], [(108, 0.6), (100, 0.4)]),
    ]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'col_scene', obj_parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        all_col_keys = np.concatenate(obj_keys)
        shares = _object_decision_shares(store, all_col_keys)
        assert report['rules']['column']['objects'] >= 1
        assert shares.get(108, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_pilaster_at_wall_becomes_wall(tmp_path):
    """2. Pilaster 0.4x0.3m against wall, full height, column votes -> wall."""
    # Pilaster touching wall at y=0.0
    pts = _box_surfaces(1.8, 2.2, 0.02, 0.32, 0.02, 2.58)
    obj_parts = [(pts, [(108, 1.0)])]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'pilaster_scene', obj_parts)
    try:
        _run_pipeline(store, origin, scan_context=ctx)
        shares = _object_decision_shares(store, obj_keys[0])
        assert shares.get(100, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_freestanding_large_block_not_column(tmp_path):
    """3. Freestanding block 1.6x1.6x2.6m, mixed votes -> column rule does NOT fire."""
    pts = _box_surfaces(2.2, 3.8, 2.2, 3.8, 0.02, 2.58)
    half = len(pts) // 2
    obj_parts = [
        (pts[:half], [(100, 0.6), (108, 0.4)]),
        (pts[half:], [(108, 0.6), (100, 0.4)]),
    ]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'block_scene', obj_parts, room_x=6.0, room_y=6.0)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        all_block_keys = np.concatenate(obj_keys)
        shares = _object_decision_shares(store, all_block_keys)
        assert report['rules']['column']['objects'] == 0
        assert shares.get(108, 0.0) < 0.90
    finally:
        store.close()


def test_shape_rules_ceiling_high_built_in_cabinet_becomes_furniture(tmp_path):
    """4. Full-height built-in cabinet front 2x2.6m at 0.6m in front of wall -> furniture."""
    xs = np.arange(1.0, 3.01, 0.02)
    zs = np.arange(0.02, 2.59, 0.02)
    front_pts = _vertical_grid(xs, 0.60, zs)
    obj_parts = [(front_pts, [(100, 0.55), (117, 0.45)])]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'cabinet_scene', obj_parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        shares = _object_decision_shares(store, obj_keys[0])
        assert report['rules']['cabinet']['objects'] >= 1
        assert shares.get(117, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_inclined_beam_becomes_beam_and_steel_protected(tmp_path):
    """5. Inclined beam (30°) 3x0.2x0.25m under ceiling -> beam; steel winner -> unchanged."""
    # Build 3m beam inclined 30° in XZ
    s = np.linspace(0.0, 3.0, 150)
    w = np.linspace(-0.1, 0.1, 10)
    t = np.linspace(-0.12, 0.12, 12)
    # Highest z point near 2.5m (just under 2.6m ceiling)
    cos30, sin30 = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    pts = []
    for si in s:
        for wi in w:
            for ti in (t[0], t[-1]):
                pts.append([0.5 + si * cos30 - ti * sin30, 2.0 + wi, 1.0 + si * sin30 + ti * cos30])
    pts = np.asarray(pts, np.float64)

    # Variant A: beam .4 / ceiling .6 -> flips to beam (109)
    store_a, origin_a, ctx_a, keys_a = _build_scene(tmp_path, 'beam_a', [(pts, [(109, 0.4), (101, 0.6)])])
    try:
        _, report_a = _run_pipeline(store_a, origin_a, scan_context=ctx_a)
        shares_a = _object_decision_shares(store_a, keys_a[0])
        assert report_a['rules']['beam']['objects'] >= 1
        assert shares_a.get(109, 0.0) >= 0.90
    finally:
        store_a.close()

    # Variant B: steel winner (110 .6 / beam .4) -> stays steel (110)
    store_b, origin_b, ctx_b, keys_b = _build_scene(tmp_path, 'beam_b', [(pts, [(110, 0.6), (109, 0.4)])])
    try:
        _, report_b = _run_pipeline(store_b, origin_b, scan_context=ctx_b)
        shares_b = _object_decision_shares(store_b, keys_b[0])
        assert report_b['rules']['beam']['objects'] == 0
        assert shares_b.get(110, 0.0) >= 0.90
    finally:
        store_b.close()


def test_shape_rules_carpet_on_floor_becomes_floor(tmp_path):
    """6. Carpet 2x1.5m, 2cm thick on floor, clutter .6/floor .4 -> floor."""
    xs = np.arange(1.0, 3.01, 0.02)
    ys = np.arange(1.0, 2.51, 0.02)
    pts = _surface_grid(xs, ys, 0.02)
    obj_parts = [(pts, [(76, 0.6), (102, 0.4)])]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'carpet_scene', obj_parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        shares = _object_decision_shares(store, obj_keys[0])
        assert report['rules']['carpet']['objects'] >= 1
        assert shares.get(102, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_wall_items(tmp_path):
    """7. Wall items: flush picture, protruding picture, radiator, wall cabinet, socket."""
    # 1. Flush picture (1 cm from wall, 0.6x0.4m at z=1.5m) with 76 votes -> wall (100)
    flush_pic = _vertical_grid(np.arange(1.7, 2.31, 0.02), 0.01, np.arange(1.3, 1.71, 0.02))
    # 2. Protruding picture (5 cm from wall) -> unclassified (1)
    prot_pic = _vertical_grid(np.arange(0.3, 0.91, 0.02), 0.05, np.arange(1.3, 1.61, 0.02))
    # 3. Radiator (10 cm from wall, hvac winner 114) -> unchanged (114)
    radiator = _vertical_grid(np.arange(1.0, 1.81, 0.02), 0.10, np.arange(0.2, 0.81, 0.02))
    # 4. Wall cabinet (depth 0.35m, height 0.7m at z=1.5m) -> furniture (117)
    cab_front = _vertical_grid(np.arange(2.5, 3.51, 0.02), 0.35, np.arange(1.4, 2.11, 0.02))
    cab_side = _vertical_grid_y(2.5, np.arange(0.02, 0.35, 0.02), np.arange(1.4, 2.11, 0.02))
    cabinet = np.vstack([cab_front, cab_side])
    # 5. Socket (electrical winner 113) -> unchanged (113)
    socket = _vertical_grid(np.arange(3.7, 3.81, 0.02), 0.02, np.arange(0.3, 0.41, 0.02))

    obj_parts = [
        (flush_pic, [(76, 1.0)]),
        (prot_pic, [(76, 1.0)]),
        (radiator, [(114, 0.9), (100, 0.1)]),
        (cabinet, [(117, 0.8), (76, 0.2)]),
        (socket, [(113, 0.9), (100, 0.1)]),
    ]
    store, origin, ctx, keys = _build_scene(tmp_path, 'wall_items', obj_parts)
    try:
        _run_pipeline(store, origin, scan_context=ctx)
        # 1. Flush picture -> wall (100) (expected per functional specification)
        shares_flush = _object_decision_shares(store, keys[0])
        assert shares_flush.get(100, 0.0) >= 0.90
        # 2. Protruding picture -> 1
        shares_prot = _object_decision_shares(store, keys[1])
        assert shares_prot.get(1, 0.0) >= 0.90
        # 3. Radiator -> 114
        shares_rad = _object_decision_shares(store, keys[2])
        assert shares_rad.get(114, 0.0) >= 0.90
        # 4. Wall cabinet -> 117
        shares_cab = _object_decision_shares(store, keys[3])
        assert shares_cab.get(117, 0.0) >= 0.90
        # 5. Socket -> 113
        shares_sock = _object_decision_shares(store, keys[4])
        assert shares_sock.get(113, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_voxel_layer_separates_flush_and_protruding_wall_items(tmp_path):
    """A 1 cm picture is structural layer; a 5 cm picture is a voxel object."""
    flush = _vertical_grid(np.arange(1.0, 1.41, 0.02), 0.01, np.arange(1.2, 1.61, 0.02))
    protruding = _vertical_grid(np.arange(2.4, 2.81, 0.02), 0.05, np.arange(1.2, 1.61, 0.02))
    store, origin, ctx, keys = _build_scene(
        tmp_path, 'voxel_layer_wall_items', [(flush, [(76, 1.0)]), (protruding, [(76, 1.0)])])
    try:
        _run_pipeline(store, origin, scan_context=ctx)
        flush_keys = [bytes(key) for key in keys[0]]
        protruding_keys = [bytes(key) for key in keys[1]]
        placeholders_flush = ','.join('?' for _ in flush_keys)
        placeholders_protruding = ','.join('?' for _ in protruding_keys)
        flush_layer = store.db.execute(
            f'SELECT COUNT(DISTINCT key) FROM shape_structure_voxels WHERE key IN ({placeholders_flush})', flush_keys).fetchone()[0]
        protruding_objects = store.db.execute(
            f'SELECT COUNT(*) FROM shape_object_voxels WHERE key IN ({placeholders_protruding})', protruding_keys).fetchone()[0]
        overlap = store.db.execute('''SELECT COUNT(*) FROM shape_object_voxels ov
                                      JOIN shape_structure_voxels sv USING(key)''').fetchone()[0]
        assert flush_layer == len(flush_keys)
        assert protruding_objects >= int(.90*len(protruding_keys))
        assert overlap == 0
    finally:
        store.close()


def test_shape_rules_furniture_vs_clutter(tmp_path):
    """8. Furniture vs clutter: 0.3m box on floor -> 76; 0.8m chair -> 117; cup on table -> cup 76, table 117."""
    # 0.3m box on floor with furniture votes
    box = _box_surfaces(0.5, 0.8, 1.5, 1.8, 0.02, 0.32)
    # 0.8m chair with construction_material votes
    chair = _box_surfaces(1.2, 1.6, 1.5, 1.9, 0.02, 0.82)
    # Table (1.2x0.8m at 0.75m)
    table_top = _surface_grid(np.arange(2.2, 3.41, 0.02), np.arange(1.5, 2.31, 0.02), 0.75)
    # Cup (0.1m) on table (at z=0.77..0.87m)
    cup = _box_surfaces(2.7, 2.8, 1.8, 1.9, 0.77, 0.87)

    obj_parts = [
        (box, [(117, 1.0)]),
        (chair, [(76, 1.0)]),
        (table_top, [(117, 1.0)]),
        (cup, [(117, 0.6), (76, 0.4)]),
    ]
    store, origin, ctx, keys = _build_scene(tmp_path, 'furn_clutter', obj_parts)
    try:
        _run_pipeline(store, origin, scan_context=ctx)
        # Box -> clutter (76)
        shares_box = _object_decision_shares(store, keys[0])
        assert shares_box.get(76, 0.0) >= 0.90
        # Chair -> furniture (117)
        shares_chair = _object_decision_shares(store, keys[1])
        assert shares_chair.get(117, 0.0) >= 0.90
        # Table top -> remains furniture (117)
        shares_table = _object_decision_shares(store, keys[2])
        assert shares_table.get(117, 0.0) >= 0.90
        # Cup -> clutter (76)
        shares_cup = _object_decision_shares(store, keys[3])
        assert shares_cup.get(76, 0.0) >= 0.90
    finally:
        store.close()


# --- Mandatory Test Cases 9 - 16 ---

def test_shape_rules_two_storeys_with_columns(tmp_path):
    """9. Two storeys (0/2.6m and 3/5.6m), each with a freestanding column -> both column."""
    col1 = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    col2 = _box_surfaces(1.8, 2.2, 1.8, 2.2, 3.02, 5.58)
    half1, half2 = len(col1) // 2, len(col2) // 2
    obj_parts = [
        (col1[:half1], [(100, 0.6), (108, 0.4)]),
        (col1[half1:], [(108, 0.6), (100, 0.4)]),
        (col2[:half2], [(100, 0.6), (108, 0.4)]),
        (col2[half2:], [(108, 0.6), (100, 0.4)]),
    ]
    store, origin, ctx, keys = _build_scene(tmp_path, 'two_storeys', obj_parts, second_storey=True)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        keys_col1 = np.concatenate(keys[:2])
        keys_col2 = np.concatenate(keys[2:])
        assert report['rules']['column']['objects'] >= 2
        assert _object_decision_shares(store, keys_col1).get(108, 0.0) >= 0.90
        assert _object_decision_shares(store, keys_col2).get(108, 0.0) >= 0.90
    finally:
        store.close()


def test_shape_rules_outdoor_context_leaves_scene_unchanged(tmp_path):
    """10. Same column scene with outdoor context -> no changes from shape rules."""
    pts = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    half = len(pts) // 2
    obj_parts = [
        (pts[:half], [(100, 0.6), (108, 0.4)]),
        (pts[half:], [(108, 0.6), (100, 0.4)]),
    ]
    store, origin, ctx, obj_keys = _build_scene(tmp_path, 'outdoor_scene', obj_parts, outdoor=True)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        all_col_keys = np.concatenate(obj_keys)
        shares = _object_decision_shares(store, all_col_keys)
        assert report['rules']['column']['voxels'] == 0
        assert shares.get(108, 0.0) < 0.90
    finally:
        store.close()


def test_shape_rules_protected_classes_and_min_observations(tmp_path):
    """11. Protected classes (door) and min_observations=2 -> unchanged."""
    # Keep the protected door voxels distinct from the room wall voxels.  At
    # y=0 both fixtures quantize to the same keys and the wall votes win while
    # constructing the scene, before either post-processing stage can run.
    door_pts = _vertical_grid(np.arange(1.5, 2.51, 0.02), 0.04, np.arange(0.02, 2.11, 0.02))
    col_pts = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    obj_parts = [
        (door_pts, [(103, 1.0)]),
        (col_pts, [(100, 0.6), (108, 0.4)]),
    ]
    store, origin, ctx, keys = _build_scene(tmp_path, 'protected_scene', obj_parts)
    try:
        # Establish the initial SAM decisions, then require two observations
        # for both post-processing stages. Every fixture voxel has count=1.
        store.decide(1, 0.5, ALL_CODES)
        cfg = _config(min_observations=2)
        apply_segment_vote(store, BY_ID, origin, 0.02, cfg, active_codes=set(ALL_CODES))
        planes_path = tmp_path/'protected_planes.npz'
        extract_planes(store, BY_ID, origin, .02, cfg, planes_path)
        apply_shape_rules(store, BY_ID, origin, 0.02, cfg, active_codes=set(ALL_CODES), scan_context=ctx,
                          planes_path=planes_path)
        # Door must be completely unchanged
        door_shares = _object_decision_shares(store, keys[0])
        assert door_shares.get(103, 0.0) == 1.0
        # Column has only count=1 so it must not be modified by shape rules
        col_shares = _object_decision_shares(store, keys[1])
        assert col_shares.get(108, 0.0) < 0.90
    finally:
        store.close()


def test_shape_rules_disabled_leaves_decisions_byte_identical(tmp_path):
    """12. shape_rules=False -> decisions byte-identical to state after segment voting."""
    pts = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    store, origin, ctx, _ = _build_scene(tmp_path, 'disabled_scene', [(pts, [(100, 0.6), (108, 0.4)])])
    try:
        store.decide(1, 0.5, ALL_CODES)
        apply_segment_vote(store, BY_ID, origin, 0.02, _config(), active_codes=set(ALL_CODES))
        before = store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall()
        report = apply_shape_rules(store, BY_ID, origin, 0.02, _config(shape_rules=False),
                                   active_codes=set(ALL_CODES), scan_context=ctx)
        assert report['enabled'] is False
        after = store.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key').fetchall()
        assert after == before
    finally:
        store.close()


def test_shape_rules_reclassify_postprocess_and_override(tmp_path):
    """13. reclassify _postprocess: shape_rules present, override shape_column_max_size honors change."""
    pts = _box_surfaces(1.8, 2.2, 1.8, 2.2, 0.02, 2.58)
    half = len(pts) // 2
    obj_parts = [
        (pts[:half], [(100, 0.6), (108, 0.4)]),
        (pts[half:], [(108, 0.6), (100, 0.4)]),
    ]
    store, origin, ctx, _ = _build_scene(tmp_path, 'reclass_scene', obj_parts)
    try:
        store.decide(1, 0.5, ALL_CODES)
        config = _config(surface_tie_break=False, building_enclosure=False, geometry_rules=False,
                         stairs_geometry_check=False, scene_consistency=False, component_cleanup=False,
                         vegetation_base_to_ground=False, geometry_fill=False, _preset_classes=[],
                         voxel_origin=origin.tolist())
        # With override shape_column_max_size=0.1, the 0.4m column is rejected
        override_config = {**config, 'shape_column_max_size': 0.1}
        stages, _, _ = _postprocess(store, override_config, BY_ID, ALL_CODES, tmp_path, set(ALL_CODES), ctx)
        assert 'shape_rules' in stages
        assert stages['shape_rules']['rules']['column']['voxels'] == 0
    finally:
        store.close()


def test_shape_rules_noisy_wall_layer_does_not_form_large_object(tmp_path):
    """A fragmented +/-2 cm wall remains a RANSAC structure layer, not a house-sized object."""
    wall = _noisy_wall(8.)
    store, origin, ctx, _ = _build_scene(tmp_path, 'noisy_wall', [], room_x=8., room_y=4., wall_points=wall)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        wall_keys = np.unique(voxel_keys(_transform(wall), origin, .02))
        layer_count = _structure_layer_count(store, wall_keys, kind=1)
        assert layer_count >= int(.80*len(wall_keys))
        assert store.db.execute('SELECT COUNT(*) FROM shape_objects WHERE l1>6.').fetchone()[0] == 0
        assert report['skipped_large_objects'] == {'objects': 0, 'voxels': 0}
        assert report['layer_band']['band_median_m'] > .03
    finally:
        store.close()


def test_shape_rules_large_object_is_counted_and_unchanged(tmp_path):
    """Objects longer than six metres are counted and deliberately left untouched."""
    object_points = _vertical_grid(np.arange(.5, 7.21, .02), 1.5, np.arange(.2, 1.01, .02))
    store, origin, ctx, keys = _build_scene(
        tmp_path, 'large_object', [(object_points, [(76, .6), (117, .4)])], room_x=8., room_y=4.)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['skipped_large_objects']['objects'] >= 1
        assert report['skipped_large_objects']['voxels'] >= len(keys[0])
        assert _object_decision_shares(store, keys[0]).get(76, 0.) >= .99
    finally:
        store.close()


def test_shape_rules_sloped_ceiling_uses_local_height_for_column(tmp_path):
    """A 35-degree roof plane supplies the local ceiling level for a freestanding column."""
    xs = np.arange(0., 4.01, .02)
    ys = np.arange(0., 4.01, .02)
    gx, gy = np.meshgrid(xs, ys)
    ceiling = np.column_stack((gx.ravel(), gy.ravel(), 2.6+np.tan(np.deg2rad(35.))*(gx.ravel()-2.)))
    column_points = _box_surfaces(1.8, 2.2, 1.8, 2.2, .02, 2.55)
    half = len(column_points)//2
    parts = [(column_points[:half], [(100, .6), (108, .4)]),
             (column_points[half:], [(108, .6), (100, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'sloped_ceiling', parts, ceiling_points=ceiling)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['levels']['ceiling_cells'] > 0
        assert report['rules']['column']['objects'] >= 1
        assert _object_decision_shares(store, np.concatenate(keys)).get(108, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_indoor_building_wall_becomes_structure_after_scene_consistency(tmp_path):
    """An indoor wall initially labelled building becomes wall before RANSAC structure matching."""
    store, origin, ctx, _ = _build_scene(tmp_path, 'building_wall', [], wall_votes=[(6, 1.)])
    try:
        cfg = _config()
        store.decide(1, .5, ALL_CODES)
        apply_segment_vote(store, BY_ID, origin, .02, cfg, active_codes=set(ALL_CODES))
        classes = [
            {'id': 'building', 'scene': 'outdoor'}, {'id': 'wall', 'scene': 'indoor'},
            {'id': 'floor', 'scene': 'indoor'}, {'id': 'ceiling', 'scene': 'indoor'},
        ]
        scene = enforce_scene_consistency(store, classes, BY_ID, ctx, origin, .02, cfg, set(ALL_CODES))
        assert scene['changed'].get('6->100', 0) > 0
        planes_path = tmp_path/'building_wall_planes.npz'
        extract_planes(store, BY_ID, origin, .02, cfg, planes_path)
        report = apply_shape_rules(store, BY_ID, origin, .02, cfg, active_codes=set(ALL_CODES),
                                   scan_context=ctx, planes_path=planes_path)
        assert report['structural_segments'] >= 3
        assert store.db.execute('SELECT COUNT(*) FROM shape_structure_voxels WHERE kind=1').fetchone()[0] > 0
    finally:
        store.close()


def test_shape_rules_narrow_wall_voted_pillar_uses_compact_orphan_guard(tmp_path):
    """A narrow freestanding pillar voted mostly wall remains an object and becomes column."""
    points = _box_surfaces(1.8, 2.2, 1.8, 2.2, .02, 2.58)
    half = len(points)//2
    parts = [(points[:half], [(100, .6), (108, .4)]), (points[half:], [(108, .6), (100, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'narrow_wall_pillar', parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        # The pillar touches the room floor and ceiling at its end faces; that
        # expected contact is a horizontal structure layer, not a narrow wall
        # plane.  The guard must keep its wall-voted side faces out of layer 1.
        layer_count = _structure_layer_count(store, np.concatenate(keys), kind=1)
        assert layer_count == 0
        assert report['rules']['column']['objects'] >= 1
        assert _object_decision_shares(store, np.concatenate(keys)).get(108, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_noisy_wall_band_keeps_2cm_picture_flush_and_6cm_picture_protruding(tmp_path):
    """The robust wall band absorbs a 2-cm picture but leaves a 6-cm picture as wall item."""
    wall = _noisy_wall(4., amplitude=.019, holes=False)
    flush_picture = _vertical_grid(np.arange(.5, 1.11, .02), .02, np.arange(1.2, 1.61, .02))
    protruding_picture = _vertical_grid(np.arange(2.4, 3.01, .02), .06, np.arange(1.2, 1.61, .02))
    parts = [(flush_picture, [(76, 1.)]), (protruding_picture, [(76, 1.)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'noisy_wall_pictures', parts, wall_points=wall)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['layer_band']['band_median_m'] >= .035
        assert _object_decision_shares(store, keys[0]).get(100, 0.) >= .90
        assert _object_decision_shares(store, keys[1]).get(1, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_segment_fallback_remains_available(tmp_path):
    """Without planes.npz the short segment-plane fallback still applies the column rule."""
    points = _box_surfaces(1.8, 2.2, 1.8, 2.2, .02, 2.58)
    half = len(points)//2
    parts = [(points[:half], [(100, .6), (108, .4)]), (points[half:], [(108, .6), (100, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'segment_fallback', parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx, ransac=False)
        assert report['rules']['column']['objects'] >= 1
        assert _object_decision_shares(store, np.concatenate(keys)).get(108, 0.) >= .90
    finally:
        store.close()


# --- Mandatory Stage 2b indoor-technology cases (all exercise the RANSAC path) ---

def test_shape_rules_round_pipe_under_ceiling_becomes_pipe(tmp_path):
    """A round HVAC-voted duct under the ceiling is pipe (111)."""
    points = _cylinder_x(.5, 3.5, 2., 2.30, .075)
    store, origin, ctx, keys = _build_scene(tmp_path, 'round_pipe', [(points, [(114, .6), (101, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['pipe']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(111, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_rectangular_duct_becomes_hvac(tmp_path):
    """A planar 0.4 x 0.3 m ventilation duct is HVAC (114)."""
    points = _box_surfaces(.3, 3.7, 1.3, 1.7, 2.05, 2.35)
    store, origin, ctx, keys = _build_scene(tmp_path, 'rectangular_duct', [(points, [(111, .6), (114, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['hvac_duct']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(114, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_sprinkler_line_and_head_become_pipe(tmp_path):
    """A sprinkler line and its head form one pipe object, not two classes."""
    line = _cylinder_x(.5, 3.5, 2., 2.36, .025)
    head = _box_surfaces(1.90, 2.10, 1.94, 2.06, 2.20, 2.40)
    parts = [(np.vstack((line, head)), [(111, .6), (116, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'sprinkler_pipe', parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['pipe']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(111, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_radiator_at_wall_becomes_hvac(tmp_path):
    """A wall-touching radiator plate with HVAC evidence becomes HVAC (114)."""
    points = _box_surfaces(.8, 1.8, .04, .14, .15, .75)
    store, origin, ctx, keys = _build_scene(tmp_path, 'radiator', [(points, [(76, .6), (114, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['radiator']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(114, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_ceiling_cable_tray_becomes_cable(tmp_path):
    """A thin flat cable tray at the ceiling is cable (112), not a beam or duct."""
    points = _box_surfaces(.5, 3.5, 1.5, 1.8, 2.53, 2.59)
    store, origin, ctx, keys = _build_scene(tmp_path, 'cable_tray', [(points, [(112, .6), (111, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['cable']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(112, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_ceiling_lamp_but_not_long_light_strip(tmp_path):
    """A short hanging lamp is equipment; a 2 m ceiling light strip is not."""
    lamp = np.vstack((
        _box_surfaces(1.8, 2.2, 1.8, 2.2, 2.25, 2.31),
        _box_surfaces(1.98, 2.02, 1.98, 2.02, 2.30, 2.58),
    ))
    strip = _box_surfaces(.5, 2.5, 1.2, 1.3, 2.53, 2.59)
    parts = [(lamp, [(101, .4), (116, .6)]), (strip, [(101, .6), (116, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'ceiling_equipment', parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['ceiling_equipment']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(116, 0.) >= .90
        assert _object_decision_shares(store, keys[1]).get(116, 0.) < .90
    finally:
        store.close()


def test_shape_rules_sanitary_cabinet_and_shower_glass_become_sanitary(tmp_path):
    """Sanitary evidence at 0.3 changes both a vanity and shower glass to 63."""
    vanity = _box_surfaces(.5, 1.3, 1.5, 2.0, .02, .80)
    shower_glass = _vertical_grid(np.arange(2.0, 3.21, .02), 2.2, np.arange(.02, 1.81, .02))
    parts = [(vanity, [(117, .6), (63, .4)]), (shower_glass, [(104, .6), (63, .4)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'sanitary', parts)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['sanitary']['objects'] >= 2
        assert _object_decision_shares(store, keys[0]).get(63, 0.) >= .90
        assert _object_decision_shares(store, keys[1]).get(63, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_glass_railing_with_handrail_becomes_railing(tmp_path):
    """A 3 m glass balustrade with its top at 1 m is railing (107)."""
    glass = _vertical_grid(np.arange(.5, 3.51, .02), 1.8, np.arange(.02, 1.01, .02))
    store, origin, ctx, keys = _build_scene(tmp_path, 'glass_railing', [(glass, [(107, .6), (104, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['railing']['objects'] >= 1
        assert _object_decision_shares(store, keys[0]).get(107, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_electrical_wall_items_remain_electrical(tmp_path):
    """Electrical-winning sockets and wall lamps stay protected from wall-item rules."""
    socket = _box_surfaces(.6, .72, .04, .08, .25, .37)
    wall_lamp = _box_surfaces(1.5, 1.8, .04, .12, 1.4, 1.7)
    store, origin, ctx, keys = _build_scene(tmp_path, 'electrical_wall_items', [(socket, [(113, 1.)]), (wall_lamp, [(113, 1.)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        assert report['rules']['wall_item']['objects'] == 0
        assert _object_decision_shares(store, keys[0]).get(113, 0.) >= .90
        assert _object_decision_shares(store, keys[1]).get(113, 0.) >= .90
    finally:
        store.close()


def test_shape_rules_suspended_ceiling_and_steel_beam_are_regressions(tmp_path):
    """A suspended ceiling remains ceiling and a steel beam remains steel."""
    steel_beam = _box_surfaces(.4, 2.4, 2.0, 2.2, 2.30, 2.55)
    store, origin, ctx, keys = _build_scene(tmp_path, 'ceiling_and_steel', [(steel_beam, [(110, .6), (109, .4)])])
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        ceiling_count = store.db.execute('SELECT COUNT(*) FROM decisions WHERE code=101').fetchone()[0]
        assert ceiling_count > 0
        assert report['rules']['beam']['objects'] == 0
        assert _object_decision_shares(store, keys[0]).get(110, 0.) >= .90
    finally:
        store.close()


@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1',
                    reason='set NOMIVOX_RUN_SLOW=1 to run the shape rules benchmark')
def test_shape_rules_benchmark_file_sqlite(tmp_path, monkeypatch):
    """17. HSLU-scale exterior benchmark stays below the per-rule decide budget."""
    store = VoteStore(tmp_path / 'shape_benchmark.sqlite')
    object_count = 50_000
    segments_per_object = 5
    segment_count = object_count * segments_per_object  # 250_000 segments
    cells_per_segment = 2
    cell_count = segment_count * cells_per_segment       # 500_000 cells
    voxels_per_cell = 11                                # 5.5M voxels
    affected_voxels = cell_count * 3                     # 1.5M exterior object candidates
    stem_count = 200

    try:
        store.db.executescript('''
            CREATE TEMP TABLE geometry_voxels(key BLOB PRIMARY KEY,ix INTEGER,iy INTEGER,iz INTEGER,
                cx INTEGER,cy INTEGER,cz INTEGER,code INTEGER,confidence INTEGER) WITHOUT ROWID;
            CREATE INDEX geometry_voxels_cells ON geometry_voxels(cx,cy,cz);
            CREATE TEMP TABLE geometry_cells(cx INTEGER,cy INTEGER,cz INTEGER,tx INTEGER,ty INTEGER,tz INTEGER,
                sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_features(cx INTEGER,cy INTEGER,cz INTEGER,x REAL,y REAL,z REAL,
                linearity REAL,planarity REAL,scatter REAL,verticality REAL,nx REAL,ny REAL,nz REAL,
                hag REAL,covered INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_meta(name TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID;
            CREATE TEMP TABLE segment_cells(cx INTEGER,cy INTEGER,cz INTEGER,segment INTEGER,
                PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE segments(segment INTEGER PRIMARY KEY,kind TEXT,cells INTEGER,voxels INTEGER,
                x REAL,y REAL,z REAL,nx REAL,ny REAL,nz REAL,min_x REAL,min_y REAL,min_z REAL,
                max_x REAL,max_y REAL,max_z REAL,winner INTEGER,share REAL,second INTEGER,second_share REAL) WITHOUT ROWID;
            CREATE TEMP TABLE segment_votes(segment INTEGER,class INTEGER,weight REAL,
                PRIMARY KEY(segment,class)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_context(
                key BLOB PRIMARY KEY, outdoor INTEGER NOT NULL, indoor INTEGER NOT NULL,
                allow_ground INTEGER NOT NULL, allow_floor INTEGER NOT NULL,
                allow_tree INTEGER NOT NULL, allow_low_vegetation INTEGER NOT NULL,
                nearest_scan INTEGER, nearest_m REAL, nearest_outdoor INTEGER, scene_context INTEGER
            ) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_context_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        ''')

        # Keep the synthetic insertion bounded; a 5M-voxel benchmark must
        # not retain one large Python tuple graph while filling SQLite.
        for start in range(0, cell_count, 2_000):
            stop = min(start + 2_000, cell_count)
            ids = np.arange(start, stop, dtype=np.int64)
            seg_ids = ids // cells_per_segment
            # Each object has five two-cell segments that form one 4 cm component.
            # Consecutive objects retain a two-cell gap so they remain separate.
            object_ids = seg_ids // segments_per_object
            cx = object_ids * (segments_per_object * cells_per_segment + 2) + (seg_ids % segments_per_object) * cells_per_segment + (ids % cells_per_segment)
            cz = np.zeros_like(cx)
            store.db.executemany('INSERT INTO geometry_cells VALUES (?,0,?,0,0,0,?,0.,?,?)',
                ((int(x), int(z), float(x * 5), float(z * 5), voxels_per_cell) for x, z in zip(cx, cz)))
            store.db.executemany('INSERT INTO geometry_features VALUES (?,0,?,?,0.05,?,0.,1.,0.,1.,0.,0.,1.,NULL,0)',
                ((int(x), int(z), float(x * 0.1), float(z * 0.1)) for x, z in zip(cx, cz)))
            store.db.executemany('INSERT INTO segment_cells VALUES (?,0,?,?)',
                ((int(x), int(z), int(s)) for x, z, s in zip(cx, cz, seg_ids)))

            observations, votes, geom_vox, geom_ctx = [], [], [], []
            for x, z in zip(cx, cz):
                for v in range(voxels_per_cell):
                    idx = (int(x) * 5 + v % 5, 0, int(z) * 5 + v // 5)
                    key = bytes(np.asarray(idx, dtype='<i8').reshape(1, 3).view('V24')[0])
                    observations.append((key, 1))
                    if v < 3:
                        # Compact, separated building candidates provide at
                        # least 50,000 exterior objects without a Python graph.
                        votes.append((key, 6, 1.0, 1.0))
                        geom_vox.append((key, *idx, int(x), 0, int(z), 6, 255))
                        geom_ctx.append((key, 1, 0, 0, 0, 0, 0, 1, 1.0, 1, 1))
                    elif v < 5:
                        paved_code = 11 if v == 3 else 73
                        votes.append((key, paved_code, 1.0, 1.0))
                        geom_vox.append((key, *idx, int(x), 0, int(z), paved_code, 255))
                        geom_ctx.append((key, 1, 0, 0, 0, 0, 0, 1, 1.0, 1, 1))
                    else:
                        # Protected exterior work keeps the file-SQLite store
                        # at HSLU scale without changing an outdoor target.
                        votes.append((key, 105, 1.0, 1.0))
                        geom_vox.append((key, *idx, int(x), 0, int(z), 105, 255))
                        geom_ctx.append((key, 1, 0, 0, 0, 0, 0, 1, 1.0, 1, 1))

            store.db.executemany('INSERT INTO observations VALUES (?,?)', observations)
            store.db.executemany('INSERT INTO votes VALUES (?,?,?,?)', votes)
            store.db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)', geom_vox)
            store.db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)', geom_ctx)

        # Two hundred independent, one-metre stems exercise the crown path.
        # They sit beyond the other fixture geometry and do not add a large
        # in-memory insertion graph.
        for start in range(0, stem_count, 25):
            stop = min(start+25, stem_count)
            stem_ids = np.arange(start, stop, dtype=np.int64)
            offsets = np.arange(50, dtype=np.int64)
            ix = (4_000_000+stem_ids[:, None]*100+offsets[None, :] % 2).reshape(-1)
            iy = (offsets[None, :] // 2).repeat(len(stem_ids), axis=0).reshape(-1)
            iz = (offsets[None, :]).repeat(len(stem_ids), axis=0).reshape(-1)
            indices = np.column_stack((ix, iy, iz)).astype('<i8', copy=False)
            keys = np.ascontiguousarray(indices).view('V24').reshape(-1)
            store.db.executemany('INSERT INTO observations VALUES (?,1)', ((bytes(key),) for key in keys))
            store.db.executemany('INSERT INTO votes VALUES (?,?,?,?)',
                                 ((bytes(key), 95, 1.0, 1.0) for key in keys))
            store.db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)',
                                 ((bytes(key), int(x), int(y), int(z), int(x), int(y), int(z), 95, 255)
                                  for key, (x, y, z) in zip(keys, indices)))
            store.db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                 ((bytes(key), 1, 0, 0, 0, 0, 0, 1, 1.0, 1, 1) for key in keys))

        # Insert segments and segment_votes
        store.db.executemany(
            'INSERT INTO segments VALUES (?, "planar", ?, ?, 1.0, 1.0, 1.0, 0., 1., 0., 0., 0., 0., 1., 1., 1., 117, 1.0, 1, 0.0)',
            ((int(s), cells_per_segment, cells_per_segment * voxels_per_cell) for s in range(segment_count)))
        store.db.executemany('INSERT INTO segment_votes VALUES (?, 117, 1.0)', ((int(s),) for s in range(segment_count)))
        store.db.commit()

        started = time.monotonic()
        store.decide(1, 0.4, [105, 6, 11, 73, 95, 5])
        decide_s = time.monotonic() - started

        # Exclude feature generation time from benchmark
        monkeypatch.setattr(shape_module, 'prepare_geometry_features', lambda *args, **kwargs: False)
        monkeypatch.setattr(shape_module, 'ensure_scene_context', lambda *args, **kwargs: {'duration_s': 0.})

        planes_path = _write_benchmark_ransac_plane(tmp_path / 'benchmark_planes.npz')
        process = psutil.Process()
        baseline = process.memory_info().rss
        peak = [baseline]
        finished = threading.Event()

        def sample_rss():
            while not finished.wait(.05):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        try:
            started = time.monotonic()
            report = apply_shape_rules(store, OUTDOOR_BY_ID, np.zeros(3), 0.02, _config(shape_object_cell=.1),
                                       active_codes={105, 6, 11, 73, 95, 5}, planes_path=planes_path)
            shape_s = time.monotonic() - started
        finally:
            finished.set()
            sampler.join()
            peak[0] = max(peak[0], process.memory_info().rss)

        timings = report['timings']
        memory_increase = peak[0]-baseline
        print(f'decide={decide_s:.3f}s shape_without_features={shape_s:.3f}s '
              f'outdoor={timings["outdoor_apply_s"]:.3f}s tree={timings["tree_apply_s"]:.3f}s '
              f'paved={timings["paved_apply_s"]:.3f}s peak_rss_increase={memory_increase / 1024**3:.3f} GiB')
        assert segment_count >= 100_000
        assert cell_count * voxels_per_cell >= 5_000_000
        assert affected_voxels >= 1_000_000
        assert stem_count >= 200
        assert store.db.execute('SELECT COUNT(*) FROM shape_outdoor_objects').fetchone()[0] >= object_count
        assert store.db.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE type = 'table' AND name = 'shape_object_voxels'"
        ).fetchone() is not None
        assert shape_s <= 600.
        assert timings['outdoor_apply_s'] <= .30*decide_s
        assert timings['tree_apply_s'] <= .30*decide_s
        assert timings['paved_apply_s'] <= .30*decide_s
        assert memory_increase <= 2*1024**3
    finally:
        store.close()


def test_shape_rules_long_wall_cabinet(tmp_path):
    """15. 10m long wall with a cabinet front 0.5m in front of wall at center -> furniture."""
    # Cabinet 2m wide at x in [4.0, 6.0], 0.5m in front of wall (y=0.50), z in [0.02, 2.58]
    xs = np.arange(4.0, 6.01, 0.02)
    zs = np.arange(0.02, 2.59, 0.02)
    cab_front = _vertical_grid(xs, 0.50, zs)
    obj_parts = [(cab_front, [(100, 0.55), (117, 0.45)])]
    store, origin, ctx, keys = _build_scene(tmp_path, 'long_wall_scene', obj_parts, room_x=10.0, room_y=4.0)
    try:
        _, report = _run_pipeline(store, origin, scan_context=ctx)
        shares = _object_decision_shares(store, keys[0])
        assert report['rules']['cabinet']['objects'] >= 1
        assert shares.get(117, 0.0) >= 0.90
    finally:
        store.close()


@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1',
                    reason='set NOMIVOX_RUN_SLOW=1 to run the 40k-object SQLite test')
def test_shape_rules_many_objects_do_not_exceed_sqlite_variable_limit(tmp_path, monkeypatch):
    """16. >= 40,000 objects in store do not hit 'too many SQL variables' error."""
    store = VoteStore(tmp_path / 'many_objects.sqlite')
    count = 40_001
    try:
        store.db.executescript('''
            CREATE TEMP TABLE geometry_voxels(key BLOB PRIMARY KEY,ix INTEGER,iy INTEGER,iz INTEGER,
                cx INTEGER,cy INTEGER,cz INTEGER,code INTEGER,confidence INTEGER) WITHOUT ROWID;
            CREATE INDEX geometry_voxels_cells ON geometry_voxels(cx,cy,cz);
            CREATE TEMP TABLE geometry_cells(cx INTEGER,cy INTEGER,cz INTEGER,tx INTEGER,ty INTEGER,tz INTEGER,
                sx REAL,sy REAL,sz REAL,n INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_features(cx INTEGER,cy INTEGER,cz INTEGER,x REAL,y REAL,z REAL,
                linearity REAL,planarity REAL,scatter REAL,verticality REAL,nx REAL,ny REAL,nz REAL,
                hag REAL,covered INTEGER,PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_meta(name TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID;
            CREATE TEMP TABLE segment_cells(cx INTEGER,cy INTEGER,cz INTEGER,segment INTEGER,
                PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
            CREATE TEMP TABLE segments(segment INTEGER PRIMARY KEY,kind TEXT,cells INTEGER,voxels INTEGER,
                x REAL,y REAL,z REAL,nx REAL,ny REAL,nz REAL,min_x REAL,min_y REAL,min_z REAL,
                max_x REAL,max_y REAL,max_z REAL,winner INTEGER,share REAL,second INTEGER,second_share REAL) WITHOUT ROWID;
            CREATE TEMP TABLE segment_votes(segment INTEGER,class INTEGER,weight REAL,
                PRIMARY KEY(segment,class)) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_context(
                key BLOB PRIMARY KEY, outdoor INTEGER NOT NULL, indoor INTEGER NOT NULL,
                allow_ground INTEGER NOT NULL, allow_floor INTEGER NOT NULL,
                allow_tree INTEGER NOT NULL, allow_low_vegetation INTEGER NOT NULL,
                nearest_scan INTEGER, nearest_m REAL, nearest_outdoor INTEGER, scene_context INTEGER
            ) WITHOUT ROWID;
            CREATE TEMP TABLE geometry_context_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        ''')

        # Insert 40,001 isolated cells / segments / objects
        for start in range(0, count, 10_000):
            stop = min(start + 10_000, count)
            ids = np.arange(start, stop, dtype=np.int64)
            # Space cells out by 3 in cx so no 26-connectivity merges them
            cx = ids * 3
            store.db.executemany('INSERT INTO geometry_cells VALUES (?,0,0,0,0,0,?,0.,0.,1)',
                ((int(x), float(x * 5)) for x in cx))
            store.db.executemany('INSERT INTO geometry_features VALUES (?,0,0,?,0.05,0.05,0.,1.,0.,1.,0.,0.,1.,NULL,0)',
                ((int(x), float(x * 0.1)) for x in cx))
            store.db.executemany('INSERT INTO segment_cells VALUES (?,0,0,?)',
                ((int(x), int(i)) for x, i in zip(cx, ids)))

            observations, votes, geom_vox, geom_ctx = [], [], [], []
            for x, i in zip(cx, ids):
                idx = (int(x) * 5, 0, 0)
                key = bytes(np.asarray(idx, dtype='<i8').reshape(1, 3).view('V24')[0])
                observations.append((key, 1))
                votes.append((key, 110, 1.0, 1.0))
                geom_vox.append((key, *idx, int(x), 0, 0, 110, 255))
                geom_ctx.append((key, 0, 1, 0, 1, 0, 0, 1, 1.0, 0, 0))

            store.db.executemany('INSERT INTO observations VALUES (?,?)', observations)
            store.db.executemany('INSERT INTO votes VALUES (?,?,?,?)', votes)
            store.db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)', geom_vox)
            store.db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)', geom_ctx)

        store.db.executemany(
            'INSERT INTO segments VALUES (?, "planar", 1, 1, 1.0, 1.0, 1.0, 0., 1., 0., 0., 0., 0., 1., 1., 1., 110, 1.0, 1, 0.0)',
            ((int(i),) for i in range(count)))
        store.db.executemany('INSERT INTO segment_votes VALUES (?, 110, 1.0)', ((int(i),) for i in range(count)))
        store.db.commit()

        store.decide(1, 0.5, [110])
        monkeypatch.setattr(shape_module, 'prepare_geometry_features', lambda *args, **kwargs: False)
        monkeypatch.setattr(shape_module, 'ensure_scene_context', lambda *args, **kwargs: {'duration_s': 0.})

        planes_path = _write_benchmark_ransac_plane(tmp_path / 'many_objects_planes.npz')
        report = apply_shape_rules(store, BY_ID, np.zeros(3), 0.02, _config(), active_codes={110},
                                   planes_path=planes_path)
        assert report['objects'] >= 40_000
        assert store.db.execute(
            "SELECT COUNT(*) FROM shape_object_voxels"
        ).fetchone()[0] >= 40_000
    finally:
        store.close()


# --- Mandatory stage 2c outdoor cases 2 - 15 ---

def test_stage_2c_rule_tables_are_consistent():
    """The report names and the two rule paths form one complete partition."""
    assert len(RULE_NAMES) == len(set(RULE_NAMES))
    assert set(OBJECT_RULE_NAMES).isdisjoint(OUTDOOR_RULE_NAMES)
    assert set(OBJECT_RULE_NAMES) | set(OUTDOOR_RULE_NAMES) == set(RULE_NAMES)


def test_stage_2c_02_facade_exterior_is_building_interior_is_wall(tmp_path):
    """2. A double-sided closed facade assigns its outer and inner skins differently."""
    xs = np.arange(0., 4.01, .02)
    ys = np.arange(0., 4.01, .02)
    zs = np.arange(.02, 3.01, .02)
    south_outer, south_inner = _vertical_grid(xs, 0., zs), _vertical_grid(xs, .25, zs)
    parts = [
        ('south_outer', south_outer, [(6, 1.)]),
        ('south_inner', south_inner, [(6, 1.)]),
        ('north', np.vstack((_vertical_grid(xs, 3.75, zs), _vertical_grid(xs, 4., zs))), [(6, 1.)]),
        ('west', np.vstack((_vertical_grid_y(0., ys, zs), _vertical_grid_y(.25, ys, zs))), [(6, 1.)]),
        ('east', np.vstack((_vertical_grid_y(3.75, ys, zs), _vertical_grid_y(4., ys, zs))), [(6, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_facade', parts)
    try:
        report, outline = _run_outdoor_pipeline(store, origin, context, codes, config={'geometry_fill_tolerance': .04})
        assert outline['has_enclosed']
        assert report['rules']['facade_side']['voxels'] > 0
        assert _object_decision_shares(store, keys['south_outer']).get(6, 0.) >= .90
        assert _object_decision_shares(store, keys['south_inner']).get(100, 0.) >= .90
    finally:
        store.close()


def test_facade_sides_skip_noisy_indoor_facade_without_an_outdoor_scan(tmp_path):
    """A noisy indoor-only facade must never manufacture building voxels."""
    xs = np.arange(0., 4.01, .02)
    ys = np.arange(0., 4.01, .02)
    zs = np.arange(.02, 3.01, .02)
    noisy_south = _noisy_wall(4., amplitude=.03, holes=False)
    parts = [
        ('south', noisy_south, [(100, 1.)]),
        ('north', _vertical_grid(xs, 4., zs), [(100, 1.)]),
        ('west', _vertical_grid_y(0., ys, zs), [(100, 1.)]),
        ('east', _vertical_grid_y(4., ys, zs), [(100, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'facade_indoor_only', parts)
    try:
        context[1].update(scene='indoor', outdoor=False)
        report, outline = _run_outdoor_pipeline(store, origin, context, codes, config={'geometry_fill_tolerance': .04})
        assert outline['has_enclosed']
        assert _object_decision_shares(store, keys['south']).get(6, 0.) == 0.
        assert not any(values['planes'] or values['voxels'] for values in report['facade_side'].values())
    finally:
        store.close()


def test_facade_sides_pair_noisy_parallel_layers_by_plane(tmp_path):
    """Two 25 cm noisy facade skins receive one stable class per RANSAC plane."""
    xs = np.arange(0., 4.01, .02)
    ys = np.arange(0., 4.01, .02)
    zs = np.arange(.02, 3.01, .02)
    south_outer = _noisy_wall(4., amplitude=.03, holes=False)
    south_inner = south_outer.copy()
    south_inner[:, 1] += .25
    parts = [
        ('south_outer', south_outer, [(100, 1.)]),
        ('south_inner', south_inner, [(100, 1.)]),
        ('north', np.vstack((_vertical_grid(xs, 3.75, zs), _vertical_grid(xs, 4., zs))), [(100, 1.)]),
        ('west', np.vstack((_vertical_grid_y(0., ys, zs), _vertical_grid_y(.25, ys, zs))), [(100, 1.)]),
        ('east', np.vstack((_vertical_grid_y(3.75, ys, zs), _vertical_grid_y(4., ys, zs))), [(100, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'facade_noisy_pair', parts)
    try:
        report, outline = _run_outdoor_pipeline(store, origin, context, codes, config={'geometry_fill_tolerance': .04})
        assert outline['has_enclosed']
        assert report['facade_side']['pair_outer']['planes'] >= 1
        assert report['facade_side']['pair_inner']['planes'] >= 1
        assert _object_decision_shares(store, keys['south_outer']).get(6, 0.) >= .90
        assert _object_decision_shares(store, keys['south_inner']).get(100, 0.) >= .90
    finally:
        store.close()


def test_facade_sides_single_noisy_layer_uses_context_majority(tmp_path):
    """Single noisy facade layers use context majorities instead of voxel sides."""
    xs = np.arange(0., 4.01, .02)
    ys = np.arange(0., 4.01, .02)
    zs = np.arange(.02, 3.01, .02)
    parts = [
        ('south', _noisy_wall(4., amplitude=.03, holes=False), [(100, 1.)]),
        ('north', _vertical_grid(xs, 4., zs), [(100, 1.)]),
        ('west', _vertical_grid_y(0., ys, zs), [(100, 1.)]),
        ('east', _vertical_grid_y(4., ys, zs), [(100, 1.)]),
    ]
    store_inside, origin_inside, context_inside, keys_inside, codes_inside = _build_outdoor_scene(
        tmp_path, 'facade_context_indoor', parts)
    try:
        context_inside[1].update(scene='indoor', outdoor=False)
        context_inside[2] = {'scene': 'outdoor', 'outdoor': True,
                             'position': _transform([-.5, -3., 2.]).tolist(), 'allowed_codes': set(codes_inside)}
        report_inside, outline_inside = _run_outdoor_pipeline(
            store_inside, origin_inside, context_inside, codes_inside, config={'geometry_fill_tolerance': .04})
        assert outline_inside['has_enclosed']
        assert report_inside['facade_side']['context_indoor']['planes'] >= 1
        assert _object_decision_shares(store_inside, keys_inside['south']).get(100, 0.) >= .90
    finally:
        store_inside.close()

    store_outside, origin_outside, context_outside, keys_outside, codes_outside = _build_outdoor_scene(
        tmp_path, 'facade_context_outdoor', parts)
    try:
        report_outside, outline_outside = _run_outdoor_pipeline(
            store_outside, origin_outside, context_outside, codes_outside, config={'geometry_fill_tolerance': .04})
        assert outline_outside['has_enclosed']
        assert report_outside['facade_side']['context_outdoor']['planes'] >= 1
        assert _object_decision_shares(store_outside, keys_outside['south']).get(6, 0.) >= .90
    finally:
        store_outside.close()


def test_stage_2c_03_curb_between_road_and_sidewalk(tmp_path):
    """3. A 12 cm raised strip between road and sidewalk becomes curb."""
    x_road = np.arange(-1.2, -.01, .02)
    x_curb = np.arange(0., .31, .02)
    x_walk = np.arange(.32, 1.61, .02)
    y = np.arange(-1., 1.01, .02)
    parts = [
        ('road', _surface_grid(x_road, y, 0.), [(11, 1.)]),
        ('curb', _surface_grid(x_curb, y, .06), [(2, 1.)]),
        ('sidewalk', _surface_grid(x_walk, y, .12), [(73, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_curb', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['curb']['voxels'] > 0
        assert _object_decision_shares(store, keys['curb']).get(74, 0.) >= .90
    finally:
        store.close()


def test_paved_sam_majorities_remain_unchanged(tmp_path):
    """Road and sidewalk retain their respective SAM majorities."""
    y = np.arange(-1., 1.01, .02)
    parts = [
        ('road', _surface_grid(np.arange(-2., 0., .02), y, 0.), [(11, .55), (73, .45)]),
        ('sidewalk', _surface_grid(np.arange(.02, 2.01, .02), y, 0.), [(11, .45), (73, .55)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'paved_sam_majorities', parts)
    try:
        _run_outdoor_pipeline(store, origin, context, codes)
        assert _object_decision_shares(store, keys['road']).get(11, 0.) >= .90
        assert _object_decision_shares(store, keys['sidewalk']).get(73, 0.) >= .90
    finally:
        store.close()


def test_road_without_curb_keeps_road_on_tied_or_stronger_road_votes(tmp_path):
    """A road-major paved component without a curb must remain road."""
    surface = _surface_grid(np.arange(0., 6.01, .02), np.arange(0., 4.01, .02), 0.)
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'road_without_curb', [('road', surface, [(11, .55), (73, .45)])])
    try:
        _run_outdoor_pipeline(store, origin, context, codes)
        assert _object_decision_shares(store, keys['road']).get(11, 0.) >= .90
    finally:
        store.close()


def test_long_paved_band_with_sidewalk_majority_remains_sidewalk(tmp_path):
    """A long paved band retains a sidewalk SAM majority."""
    band = _surface_grid(np.arange(0., 40.01, .10), np.arange(0., 6.01, .10), 0.)
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'road_shape_band', [('band', band, [(11, .45), (73, .55)])])
    try:
        _run_outdoor_pipeline(store, origin, context, codes)
        assert _object_decision_shares(store, keys['band']).get(73, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_05_garden_wall_and_facade_plinth(tmp_path):
    """5. A freestanding garden wall is outdoor_wall and a facade plinth is building."""
    xs = np.arange(0., 3.01, .02)
    ys = np.arange(0., 3.01, .02)
    zs = np.arange(.02, 3.01, .02)
    parts = [
        ('facade_south', _vertical_grid(xs, 0., zs), [(6, 1.)]),
        ('facade_north', _vertical_grid(xs, 3., zs), [(6, 1.)]),
        ('facade_west', _vertical_grid_y(0., ys, zs), [(6, 1.)]),
        ('facade_east', _vertical_grid_y(3., ys, zs), [(6, 1.)]),
        ('garden_wall', _box_surfaces(4., 6., .5, .62, 0., 1.5), [(100, .6), (78, .4)]),
        ('plinth', _box_surfaces(-.16, -.04, .8, 2.2, .02, .52), [(100, .6), (6, .4)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_walls', parts)
    try:
        report, outline = _run_outdoor_pipeline(store, origin, context, codes)
        assert outline['has_enclosed']
        assert report['rules']['outdoor_wall']['voxels'] > 0
        assert _object_decision_shares(store, keys['garden_wall']).get(78, 0.) >= .90
        assert _object_decision_shares(store, keys['plinth']).get(6, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_06_utility_pole_and_connected_sign(tmp_path):
    """6. A six metre pole becomes utility_pole and a sign plate relabels its mast too."""
    sign_pole = _cylinder_z(3., 0., 0., 5.4, .08)
    sign_panel = _vertical_grid(np.arange(2.6, 3.41, .02), 0., np.arange(4.35, 5.01, .02))
    parts = [
        ('pole', _cylinder_z(0., 0., 0., 6., .10), [(100, .6), (84, .4)]),
        ('sign_pole', sign_pole, [(100, .6), (84, .4)]),
        ('sign_panel', sign_panel, [(81, .8), (100, .2)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_sign', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['utility_pole']['voxels'] > 0
        assert report['rules']['traffic_sign']['voxels'] > 0
        assert _object_decision_shares(store, keys['pole']).get(84, 0.) >= .90
        sign_keys = np.concatenate((keys['sign_pole'], keys['sign_panel']))
        assert _object_decision_shares(store, sign_keys).get(81, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_07_long_outdoor_fence(tmp_path):
    """7. A 10 m by 1.2 m thin exterior panel becomes fence."""
    fence = _vertical_grid(np.arange(-5., 5.01, .02), 0., np.arange(0., 1.21, .02))
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'stage2c_fence', [('fence', fence, [(100, .6), (77, .4)])])
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['fence']['voxels'] > 0
        assert _object_decision_shares(store, keys['fence']).get(77, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_08_bicycle_vehicle_votes_relabel_full_object(tmp_path):
    """8. A compact bicycle proxy with vehicle evidence becomes vehicle as one object."""
    bicycle = _box_surfaces(-.9, .9, -.18, .18, .02, 1.15)
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'stage2c_bicycle', [('bicycle', bicycle, [(100, .6), (91, .4)])])
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['vehicle']['voxels'] > 0
        assert _object_decision_shares(store, keys['bicycle']).get(91, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_09_high_unsupported_power_line(tmp_path):
    """9. A thin unsupported line at six metres becomes power_line."""
    line = _cylinder_x(-3., 3., 0., 6., .05)
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'stage2c_power_line', [('line', line, [(100, .6), (14, .4)])],
        ground_points=np.asarray([[0., 0., 0.]]))
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['power_line']['voxels'] > 0
        assert _object_decision_shares(store, keys['line']).get(14, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_10_carport_building_and_steel_pergola_guard(tmp_path):
    """10. Building-evidence carport becomes building; a steel-winning pergola remains steel."""
    carport = _surface_grid(np.arange(0., 3.01, .02), np.arange(0., 3.01, .02), 2.5)
    pergola = _surface_grid(np.arange(4., 7.01, .02), np.arange(0., 3.01, .02), 2.5)
    parts = [
        ('carport', carport, [(100, .6), (6, .4)]),
        ('pergola', pergola, [(110, .7), (6, .3)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_carport', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes)
        assert report['rules']['outdoor_building']['voxels'] > 0
        assert _object_decision_shares(store, keys['carport']).get(6, 0.) >= .90
        assert _object_decision_shares(store, keys['pergola']).get(110, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_11_lollipop_crown_preserves_facade_and_pole(tmp_path):
    """11. A crown beside a facade wins over its labels but not structural facade or a pole."""
    facade = _vertical_grid(np.arange(-2., 2.01, .02), 0., np.arange(0., 4.01, .02))
    trunk = _cylinder_z(0., 1.9, 0., 2., .12)
    parts = [
        ('facade', facade, [(6, 1.)]),
        ('trunk', trunk, [(95, 1.)]),
        ('crown_low', _crown_ring(0., 1.9, 2.15, 1.4), [(100, .6), (6, .4)]),
        ('crown_high', _crown_ring(0., 1.9, 3.15, 1.4), [(100, .6), (6, .4)]),
        ('crown_pole', _cylinder_z(1.75, 1.9, 0., 6., .08), [(100, .6), (84, .4)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_lollipop_facade', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 3.})
        crown = np.concatenate((keys['crown_low'], keys['crown_high']))
        assert report['rules']['tree_crown']['voxels'] > 0
        assert _object_decision_shares(store, crown).get(5, 0.) >= .90
        assert _object_decision_shares(store, keys['facade']).get(6, 0.) >= .90
        assert _object_decision_shares(store, keys['crown_pole']).get(84, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_12_vote_only_tree_trunk_starts_lollipop_crown(tmp_path):
    """12. A 40 percent tree_trunk vote is sufficient to seed the crown."""
    parts = [
        ('trunk', _cylinder_z(0., 0., 0., 2., .12), [(100, .6), (95, .4)]),
        ('crown_low', _crown_ring(0., 0., 2.15, 1.5), [(100, .6), (6, .4)]),
        ('crown_high', _crown_ring(0., 0., 3.15, 1.5), [(100, .6), (6, .4)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_vote_trunk', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 3.})
        crown = np.concatenate((keys['crown_low'], keys['crown_high']))
        assert report.get('tree_seeds', 0) >= 1
        assert _object_decision_shares(store, crown).get(5, 0.) >= .90
    finally:
        store.close()


def test_tree_crown_fills_unclassified_voxels_inside_lollipop_volume(tmp_path):
    """e. Unclassified observed voxels in a measured crown become tree."""
    parts = [
        ('trunk', _cylinder_z(0., 0., 0., 2., .12), [(95, 1.)]),
        ('lower', _crown_ring(0., 0., 2.15, 1.4), [(6, 1.)]),
        ('upper', _crown_ring(0., 0., 3.15, 1.4), [(6, 1.)]),
        ('hollow', np.asarray([[.25, .15, 2.6]]), [(102, .4)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'tree_hollow_fill', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 3.})
        assert _object_decision_shares(store, keys['hollow']).get(5, 0.) >= .90
        assert Path(report['tree_crown']['artifact']).is_file()
    finally:
        store.close()


def test_tree_crown_fill_keeps_another_decision_code(tmp_path):
    """f. A non-tree decision inside a crown, such as a pole, remains intact."""
    parts = [
        ('trunk', _cylinder_z(0., 0., 0., 2., .12), [(95, 1.)]),
        ('lower', _crown_ring(0., 0., 2.15, 1.4), [(6, 1.)]),
        ('upper', _crown_ring(0., 0., 3.15, 1.4), [(6, 1.)]),
        ('pole_inside', np.asarray([[.2, .1, 2.6]]), [(84, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'tree_keep_pole', parts)
    try:
        _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 3.})
        assert _object_decision_shares(store, keys['pole_inside']).get(84, 0.) >= .90
    finally:
        store.close()


def test_tree_crown_export_fills_only_unknown_points(tmp_path):
    """g. Export fills only code-1 points inside a persisted crown volume."""
    crown_path = tmp_path/'tree_crowns.npz'
    np.savez(crown_path, x=np.asarray([SHIFT[0]]), y=np.asarray([SHIFT[1]]), radius=np.asarray([1.]),
             z_min=np.asarray([SHIFT[2]+2.]), z_max=np.asarray([SHIFT[2]+4.]))

    class Votes:
        def lookup(self, keys):
            return np.ones(len(keys), np.uint8), np.zeros(len(keys), np.uint8)

    store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
    points = np.asarray([[SHIFT[0]+.2, SHIFT[1]+.1, SHIFT[2]+2.5],
                         [SHIFT[0]+1.5, SHIFT[1], SHIFT[2]+2.5]])
    with ClassifiedSource(tmp_path, [], Votes(), np.zeros(3), .02, tree_crowns=crown_path, tree_fill=True) as source:
        _, codes, confidence, _ = source.block(store, 0, points, np.ones(len(points), bool), 'original')
        assert source.geometry_fill_counts['tree_fill'] == 1
    assert codes.tolist() == [5, 1]
    assert confidence.tolist() == [64, 0]


def test_tree_crown_fill_can_be_disabled(tmp_path):
    """h. Disabled crown filling leaves an otherwise identical export unchanged."""
    crown_path = tmp_path/'tree_crowns.npz'
    np.savez(crown_path, x=np.asarray([SHIFT[0]]), y=np.asarray([SHIFT[1]]), radius=np.asarray([1.]),
             z_min=np.asarray([SHIFT[2]+2.]), z_max=np.asarray([SHIFT[2]+4.]))

    class Votes:
        def lookup(self, keys):
            return np.ones(len(keys), np.uint8), np.zeros(len(keys), np.uint8)

    store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
    point = np.asarray([[SHIFT[0]+.2, SHIFT[1]+.1, SHIFT[2]+2.5]])
    with ClassifiedSource(tmp_path, [], Votes(), np.zeros(3), .02, tree_crowns=crown_path, tree_fill=False) as source:
        _, codes, confidence, _ = source.block(store, 0, point, np.ones(1, bool), 'original')
        assert source.geometry_fill_counts == {}
    assert codes.tolist() == [1]
    assert confidence.tolist() == [0]


def test_tree_crown_export_preserves_tree_trunk_in_crown_volume(tmp_path):
    """i. Export preserves code-95 tree_trunk points inside the crown volume."""
    crown_path = tmp_path/'tree_crowns.npz'
    np.savez(crown_path, x=np.asarray([SHIFT[0]]), y=np.asarray([SHIFT[1]]), radius=np.asarray([1.]),
             z_min=np.asarray([SHIFT[2]+2.]), z_max=np.asarray([SHIFT[2]+4.]))

    class Votes:
        def lookup(self, keys):
            codes = np.asarray([1, 95], dtype=np.uint8)
            conf = np.asarray([0, 255], dtype=np.uint8)
            return codes, conf

    store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
    points = np.asarray([[SHIFT[0]+.2, SHIFT[1]+.1, SHIFT[2]+2.5],
                         [SHIFT[0]+.1, SHIFT[1]+.1, SHIFT[2]+3.0]])
    with ClassifiedSource(tmp_path, [], Votes(), np.zeros(3), .02, tree_crowns=crown_path, tree_fill=True) as source:
        _, codes, confidence, _ = source.block(store, 0, points, np.ones(len(points), bool), 'original')
        assert source.geometry_fill_counts['tree_fill'] == 1
    assert codes.tolist() == [5, 95]
    assert confidence.tolist() == [64, 255]


def test_stage_2c_13_crown_radius_uses_larger_upper_slice(tmp_path):
    """13. The 3 m upper slice controls the crown; a 3.5 m point stays unchanged."""
    parts = [
        ('trunk', _cylinder_z(0., 0., 0., 2., .12), [(95, 1.)]),
        ('lower_slice', _crown_ring(0., 0., 2.15, 1.5), [(100, .6), (6, .4)]),
        ('upper_slice', _crown_ring(0., 0., 3.15, 3.0), [(100, .6), (6, .4)]),
        ('inside', np.asarray([[2.8, 0., 3.55]]), [(100, .6), (6, .4)]),
        ('outside', np.asarray([[3.5, 0., 4.1]]), [(100, 1.)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_radius', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 4.})
        radius = report['tree_crown']['radius_m'][0]
        assert 2.9 <= radius <= 3.1
        assert _object_decision_shares(store, keys['inside']).get(5, 0.) >= .90
        assert _object_decision_shares(store, keys['outside']).get(100, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_14_short_tree_seed_uses_metres_not_feature_cells(tmp_path):
    """14. A 0.5 m trunk is not a geometry-rule seed at 2 cm voxel resolution."""
    short_trunk = _cylinder_z(0., 0., 0., .5, .12)
    store, origin, context, _, codes = _build_outdoor_scene(
        tmp_path, 'stage2c_short_tree_seed', [('short_trunk', short_trunk, [(95, 1.)])])
    try:
        cfg = _config()
        store.decide(1, .5, codes)
        apply_segment_vote(store, OUTDOOR_BY_ID, origin, .02, cfg, active_codes=set(codes))
        extract_planes(store, OUTDOOR_BY_ID, origin, .02, cfg, store.path.with_name('stage2c_short_tree_seed_planes.npz'))
        prepare_geometry_features(store, origin, .02, cfg)
        ensure_scene_context(store, OUTDOOR_BY_ID, context, origin, .02, cfg, lambda: None)
        assert _tree_seeds(store, OUTDOOR_BY_ID, cfg, .02) == []
    finally:
        store.close()


def test_stage_2c_15_outdoor_stairs_remain_stairs(tmp_path):
    """15. Exterior stairs are immutable for all outdoor shape rules."""
    steps = np.vstack([
        _box_surfaces(-1., 1., -.5, -.15, .02, .30),
        _box_surfaces(-1., 1., -.14, .21, .02, .60),
        _box_surfaces(-1., 1., .22, .57, .02, .90),
    ])
    store, origin, context, keys, codes = _build_outdoor_scene(
        tmp_path, 'stage2c_outdoor_stairs', [('stairs', steps, [(105, 1.)])])
    try:
        _run_outdoor_pipeline(store, origin, context, codes)
        assert _object_decision_shares(store, keys['stairs']).get(105, 0.) >= .90
    finally:
        store.close()


def test_stage_2c_16_tree_trunk_extending_into_crown_remains_trunk(tmp_path):
    """16. A trunk extending 2 m into the crown keeps all trunk voxels at code 95."""
    trunk_base = _cylinder_z(0., 0., 0., 2., .12)
    trunk_crown = np.vstack([
        _cylinder_z(0., 0., 2.08, 2.68, .12),
        _cylinder_z(0., 0., 2.76, 3.36, .12),
        _cylinder_z(0., 0., 3.44, 4.00, .12),
    ])
    parts = [
        ('trunk_base', trunk_base, [(95, 1.)]),
        ('trunk_crown', trunk_crown, [(95, 1.)]),
        ('crown_low', _crown_ring(0., 0., 2.15, 1.4), [(100, .6), (6, .4)]),
        ('crown_high', _crown_ring(0., 0., 3.15, 1.4), [(100, .6), (6, .4)]),
    ]
    store, origin, context, keys, codes = _build_outdoor_scene(tmp_path, 'stage2c_trunk_in_crown', parts)
    try:
        report, _ = _run_outdoor_pipeline(store, origin, context, codes, config={'tree_crown_radius_max': 3.})
        crown = np.concatenate((keys['crown_low'], keys['crown_high']))
        assert report['rules']['tree_crown']['voxels'] > 0
        assert _object_decision_shares(store, crown).get(5, 0.) >= .90
        trunk_all = np.concatenate((keys['trunk_base'], keys['trunk_crown']))
        assert _object_decision_shares(store, trunk_all).get(95, 0.) >= .99
        assert _object_decision_shares(store, keys['trunk_crown']).get(95, 0.) >= .99
    finally:
        store.close()


@pytest.mark.skipif(os.environ.get('NOMIVOX_RUN_SLOW') != '1',
                    reason='set NOMIVOX_RUN_SLOW=1 to run the scene-outline benchmark')
def test_stage_2c_17_outline_refinement_benchmark_file_sqlite(tmp_path):
    """17. Refine a 2M-voxel outline context in at most half the decide time."""
    store = VoteStore(tmp_path / 'stage2c_outline_benchmark.sqlite')
    width, height, levels = 100, 100, 200
    count = width*height*levels
    try:
        store.db.executescript('''
            CREATE TEMP TABLE geometry_context(
                key BLOB PRIMARY KEY, outdoor INTEGER NOT NULL, indoor INTEGER NOT NULL,
                allow_ground INTEGER NOT NULL, allow_floor INTEGER NOT NULL,
                allow_tree INTEGER NOT NULL, allow_low_vegetation INTEGER NOT NULL,
                nearest_scan INTEGER, nearest_m REAL, nearest_outdoor INTEGER, scene_context INTEGER
            ) WITHOUT ROWID;
        ''')
        for start in range(0, count, 100_000):
            stop = min(start+100_000, count)
            values = np.arange(start, stop, dtype=np.int64)
            indices = np.column_stack((values % width, (values//width) % height, values//(width*height))).astype('<i8')
            keys = np.ascontiguousarray(indices).view('V24').reshape(-1)
            # Most rows begin with their correct result.  A deterministic
            # subset remains ambiguous so the timed pass also exercises
            # SQLite writes rather than only the read path.
            cells_x, cells_y = indices[:, 0]//10, indices[:, 1]//10
            inside = ((cells_x > 0) & (cells_x < 9) & (cells_y > 0) & (cells_y < 9))
            expected = np.where(inside & (indices[:, 2] < levels-1), 0, 1).astype(np.int8)
            context = expected.astype(np.int64)
            context[(values % 4096) == 0] = -1
            store.db.executemany('INSERT INTO observations VALUES (?,1)', ((bytes(key),) for key in keys))
            store.db.executemany('INSERT INTO votes VALUES (?,6,1.,1.)', ((bytes(key),) for key in keys))
            store.db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                 ((bytes(key), 1, 0, 0, 0, 0, 0, 1, 1., 1, int(value))
                                  for key, value in zip(keys, context)))
        store.db.commit()
        started = time.monotonic()
        store.decide(1, .5, [6])
        decide_s = time.monotonic()-started
        barrier = np.zeros((10, 10), bool)
        barrier[0, :], barrier[-1, :], barrier[:, 0], barrier[:, -1] = True, True, True, True
        inside = np.zeros((10, 10), bool)
        inside[1:-1, 1:-1] = True
        outline = {'applied': True, 'has_enclosed': True, 'origin': np.array([0, 0], np.int64),
                   'shape': (10, 10), 'cells': 10, 'cell_size_m': .2,
                   'barrier': barrier, 'inside': inside, 'outside': ~(barrier | inside)}
        started = time.monotonic()
        report = refine_scene_context_with_outline(store, outline, {'building': 6}, {1: {'scene': 'outdoor'}}, .02)
        refine_s = time.monotonic()-started
        print(f'decide={decide_s:.3f}s outline_refine={refine_s:.3f}s voxels={count} changed={report["changed"]}')
        assert count >= 2_000_000
        assert report['applied'] and report['changed'] > 0
        assert refine_s <= .5*decide_s
    finally:
        store.close()
