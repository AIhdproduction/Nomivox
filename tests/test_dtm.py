"""Tests for Triangulated Irregular Network Digital Terrain Model (TIN DTM)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import pytest

from e57cls.output.classify import ClassifiedSource
from e57cls.vote.dtm import TinDTM, apply_ground_dtm_band, build_tin_dtm
from e57cls.vote.engine import VoteStore, voxel_keys

# Georeferenced origin and 30-degree rotation around z
ORIGIN = np.array([2665953.0, 1207269.0, 440.0], dtype=np.float64)
ANGLE_DEG = 30.0
COS_30 = np.cos(np.radians(ANGLE_DEG))
SIN_30 = np.sin(np.radians(ANGLE_DEG))
VOXEL_SIZE = 0.02

BY_ID = {
    'ground': 2,
    'low_vegetation': 3,
    'road': 11,
    'sidewalk': 13,
    'curb': 14,
    'stairs': 15,
    'manhole': 17,
    'tree_trunk': 95,
    'building': 100,
    'floor': 102,
    'reflection': 68,
}


def _to_world(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Transform local rotated coordinates (u, v, w) to world coordinates."""
    x = ORIGIN[0] + u * COS_30 - v * SIN_30
    y = ORIGIN[1] + u * SIN_30 + v * COS_30
    z = ORIGIN[2] + w
    return np.column_stack((x, y, z))


def _to_local(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform world (x, y) back to local coordinates (u, v)."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    dx = pts[:, 0] - ORIGIN[0]
    dy = pts[:, 1] - ORIGIN[1]
    u = dx * COS_30 + dy * SIN_30
    v = -dx * SIN_30 + dy * COS_30
    return u, v


def _true_plane_height(xy: np.ndarray, slope_deg: float = 15.0) -> np.ndarray:
    """Compute exact height on plane inclined by slope_deg along local u axis."""
    u, _ = _to_local(xy)
    s = np.tan(np.radians(slope_deg))
    return ORIGIN[2] + u * s


def _store_with_decisions(path, entries: list[tuple[np.ndarray, int]]):
    """Create a VoteStore and populate the decisions table."""
    store = VoteStore(path)
    rows = []
    for world_pts, code in entries:
        keys = voxel_keys(world_pts, ORIGIN, VOXEL_SIZE)
        unique_keys = np.unique(keys)
        rows.extend((bytes(k), int(code), 255) for k in unique_keys)
    store.db.executemany("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)", rows)
    store.db.commit()
    return store


def _store_with_data(
    path,
    decisions_entries: list[tuple[np.ndarray, int, int]] | None = None,
    votes_entries: list[tuple[np.ndarray, list[tuple[int, float]]]] | None = None,
    obs_count: int = 1,
):
    """Create a VoteStore and populate decisions, votes, and observations."""
    store = VoteStore(path)
    if decisions_entries:
        rows = []
        for world_pts, code, conf in decisions_entries:
            keys = voxel_keys(world_pts, ORIGIN, VOXEL_SIZE)
            unique_keys = np.unique(keys)
            rows.extend((bytes(k), int(code), int(conf)) for k in unique_keys)
        store.db.executemany("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)", rows)
        if obs_count > 0:
            store.db.executemany(
                "INSERT OR REPLACE INTO observations VALUES (?, ?)",
                ((row[0], obs_count) for row in rows),
            )
    if votes_entries:
        v_rows = []
        for world_pts, class_weights in votes_entries:
            keys = voxel_keys(world_pts, ORIGIN, VOXEL_SIZE)
            unique_keys = np.unique(keys)
            for k in unique_keys:
                for cls, weight in class_weights:
                    v_rows.append((bytes(k), int(cls), float(weight), float(weight)))
        store.db.executemany(
            "INSERT OR REPLACE INTO votes VALUES (?, ?, ?, ?)",
            v_rows,
        )
    store.db.commit()
    return store


def test_dtm_inclined_plane_within_1cm(tmp_path):
    """(a) 15-degree inclined plane from ground voxels matches within 1 cm."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(
        np.arange(0.0, 6.0, 0.1),
        np.arange(0.0, 6.0, 0.1),
        indexing='ij',
    )
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    w_flat = u_flat * np.tan(np.radians(slope_deg))
    world_pts = _to_world(u_flat, v_flat, w_flat)

    db_path = tmp_path / "plane_votes.sqlite"
    store = _store_with_decisions(db_path, [(world_pts, BY_ID['ground'])])
    tin_path = tmp_path / "plane_dtm.npz"

    try:
        report = build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        assert report['nodes'] > 0
        assert report['triangles'] > 0

        with TinDTM(tin_path) as tin:
            test_u, test_v = np.meshgrid(
                np.arange(1.5, 4.5, 0.25),
                np.arange(1.5, 4.5, 0.25),
                indexing='ij',
            )
            test_u_flat = test_u.ravel()
            test_v_flat = test_v.ravel()
            test_world = _to_world(test_u_flat, test_v_flat, test_u_flat * np.tan(np.radians(slope_deg)))

            interpolated_h = tin.height(test_world[:, :2])
            expected_h = _true_plane_height(test_world[:, :2], slope_deg)

            assert np.all(np.isfinite(interpolated_h))
            max_deviation = np.max(np.abs(interpolated_h - expected_h))
            assert max_deviation < 0.01  # Within 1 cm
    finally:
        store.close()


def test_dtm_single_outliers_do_not_pull_down(tmp_path):
    """(b) Single outliers 1 m below ground do not pull DTM down (deviation < 3 cm)."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(
        np.arange(0.0, 6.0, 0.1),
        np.arange(0.0, 6.0, 0.1),
        indexing='ij',
    )
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    w_flat = u_flat * np.tan(np.radians(slope_deg))
    ground_pts = _to_world(u_flat, v_flat, w_flat)

    # Outlier 1: Reflection code 68, 1 m below ground at (u=3.0, v=3.0)
    outlier_refl_pts = _to_world(
        np.array([3.0]),
        np.array([3.0]),
        np.array([3.0 * np.tan(np.radians(slope_deg)) - 1.0]),
    )

    # Outlier 2: Ground code 2, 1 m below ground at (u=3.5, v=3.5)
    outlier_ground_pts = _to_world(
        np.array([3.5]),
        np.array([3.5]),
        np.array([3.5 * np.tan(np.radians(slope_deg)) - 1.0]),
    )

    db_path = tmp_path / "outlier_votes.sqlite"
    store = _store_with_decisions(
        db_path,
        [
            (ground_pts, BY_ID['ground']),
            (outlier_refl_pts, BY_ID['reflection']),
            (outlier_ground_pts, BY_ID['ground']),
        ],
    )
    tin_path = tmp_path / "outlier_dtm.npz"

    try:
        report = build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        assert report['nodes'] > 0

        with TinDTM(tin_path) as tin:
            test_locations = _to_world(
                np.array([3.0, 3.5]),
                np.array([3.0, 3.5]),
                np.array([0.0, 0.0]),
            )
            interpolated_h = tin.height(test_locations[:, :2])
            expected_h = _true_plane_height(test_locations[:, :2], slope_deg)

            assert np.all(np.isfinite(interpolated_h))
            deviations = np.abs(interpolated_h - expected_h)
            # Must not be pulled down by 1 m; stays within 3 cm of true ground
            assert np.all(deviations < 0.03)
    finally:
        store.close()


def test_dtm_large_gap_returns_nan(tmp_path):
    """(c) Gap larger than terrain_max_gap yields NaN in height query."""
    # Two separated ground patches: u in [0, 2] and u in [6, 8]
    # Gap in between is 4 m wide. With terrain_max_gap = 2.0 m, triangles cannot bridge it.
    u_p1, v_p1 = np.meshgrid(np.arange(0.0, 2.0, 0.1), np.arange(0.0, 4.0, 0.1), indexing='ij')
    u_p2, v_p2 = np.meshgrid(np.arange(6.0, 8.0, 0.1), np.arange(0.0, 4.0, 0.1), indexing='ij')

    pts_p1 = _to_world(u_p1.ravel(), v_p1.ravel(), np.zeros(u_p1.size))
    pts_p2 = _to_world(u_p2.ravel(), v_p2.ravel(), np.zeros(u_p2.size))

    db_path = tmp_path / "gap_votes.sqlite"
    store = _store_with_decisions(db_path, [(np.vstack((pts_p1, pts_p2)), BY_ID['ground'])])
    tin_path = tmp_path / "gap_dtm.npz"

    try:
        report = build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 2.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        assert report['triangles'] > 0

        with TinDTM(tin_path) as tin:
            # Query in the middle of the gap (u=4.0, v=2.0)
            gap_query = _to_world(np.array([4.0]), np.array([2.0]), np.array([0.0]))
            h_gap = tin.height(gap_query[:, :2])
            assert np.isnan(h_gap[0])

            # Query inside valid patch 1 (u=1.0, v=2.0)
            valid_query = _to_world(np.array([1.0]), np.array([2.0]), np.array([0.0]))
            h_valid = tin.height(valid_query[:, :2])
            assert np.isfinite(h_valid[0])
    finally:
        store.close()


def test_dtm_building_outline_excludes_indoor_voxels(tmp_path):
    """(d) Ground voxels inside an enclosed building outline are excluded from DTM."""
    # Create outdoor ground: u in [0, 10], v in [0, 10]
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 10.0, 0.2), np.arange(0.0, 10.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), np.zeros(u_grid.size))

    db_path = tmp_path / "outline_votes.sqlite"
    store = _store_with_decisions(db_path, [(ground_pts, BY_ID['ground'])])
    tin_path = tmp_path / "outline_dtm.npz"

    # Define an enclosed building outline covering local u in [3, 7], v in [3, 7]
    cell_size_m = 0.5
    outline_cells = max(1, round(cell_size_m / VOXEL_SIZE))

    # All voxels have indices: (pts - origin) / VOXEL_SIZE
    # outline raster index = voxelindex[:, :2] // outline['cells'] - outline['origin']
    ground_indices = np.floor((ground_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    min_grid_xy = ground_indices[:, :2] // outline_cells
    lo = min_grid_xy.min(axis=0) - 2
    hi = min_grid_xy.max(axis=0) + 2
    shape = tuple((hi - lo + 1).tolist())

    blocked = np.zeros(shape, bool)
    inside = np.zeros(shape, bool)
    outside = np.ones(shape, bool)

    # Mark local u in [3.0, 7.0] and v in [3.0, 7.0] as building interior
    building_u, building_v = np.meshgrid(np.arange(3.0, 7.0, 0.1), np.arange(3.0, 7.0, 0.1), indexing='ij')
    building_pts = _to_world(building_u.ravel(), building_v.ravel(), np.zeros(building_u.size))
    building_indices = np.floor((building_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    b_grid_xy = building_indices[:, :2] // outline_cells - lo

    # Interior cells
    inside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = True
    outside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = False

    outline = {
        'applied': True,
        'inside': inside,
        'outside': outside,
        'barrier': blocked,
        'origin': lo,
        'cells': outline_cells,
        'cell_size_m': cell_size_m,
        'shape': shape,
        'has_enclosed': True,
    }

    try:
        report = build_tin_dtm(
            store,
            BY_ID,
            outline=outline,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        assert report['nodes'] > 0

        with TinDTM(tin_path) as tin:
            # Verify no node in the DTM is inside the building outline
            dtm_indices = np.floor((tin.nodes - ORIGIN) / VOXEL_SIZE).astype(np.int64)
            dtm_grid_xy = dtm_indices[:, :2] // outline_cells - lo
            gx, gy = dtm_grid_xy[:, 0], dtm_grid_xy[:, 1]
            in_grid = (gx >= 0) & (gx < shape[0]) & (gy >= 0) & (gy < shape[1])

            nodes_inside = inside[gx[in_grid], gy[in_grid]]
            assert not np.any(nodes_inside)
    finally:
        store.close()


def test_dtm_bush_voxels_above_ground_become_low_vegetation(tmp_path):
    """(a) Bush voxels 0.3-1.2 m above ground with votes ground .6 / low_veg .4 -> 3."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(
        np.arange(0.0, 6.0, 0.1),
        np.arange(0.0, 6.0, 0.1),
        indexing='ij',
    )
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    w_flat = u_flat * np.tan(np.radians(slope_deg))
    ground_pts = _to_world(u_flat, v_flat, w_flat)

    # Bush points at (u=3.0, v=3.0) at heights 0.35m, 0.60m, 1.15m above terrain
    bush_heights = np.array([0.35, 0.60, 1.15])
    bush_u = np.full(len(bush_heights), 3.0)
    bush_v = np.full(len(bush_heights), 3.0)
    bush_w = bush_u * np.tan(np.radians(slope_deg)) + bush_heights
    bush_pts = _to_world(bush_u, bush_v, bush_w)

    db_path = tmp_path / "bush_votes.sqlite"
    tin_path = tmp_path / "bush_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (bush_pts, BY_ID['ground'], 153),
        ],
        votes_entries=[
            (bush_pts, [(BY_ID['ground'], 0.6), (BY_ID['low_vegetation'], 0.4)]),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            report = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        assert report['enabled'] is True
        bush_keys = voxel_keys(bush_pts, ORIGIN, VOXEL_SIZE)
        for k in bush_keys:
            row = store.db.execute("SELECT code, confidence FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row is not None
            assert row[0] == BY_ID['low_vegetation']  # 3
            assert row[1] == round(0.4 / 1.0 * 255)   # 102
    finally:
        store.close()


def test_dtm_indoor_voxels_become_floor_and_not_in_dtm(tmp_path):
    """(b) Ground voxels 4 m above terrain inside enclosed building with ground .7 / floor .3 -> 102."""
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 10.0, 0.2), np.arange(0.0, 10.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), np.zeros(u_grid.size))

    # Building outline for u in [3, 7], v in [3, 7]
    cell_size_m = 0.5
    outline_cells = max(1, round(cell_size_m / VOXEL_SIZE))
    ground_indices = np.floor((ground_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    min_grid_xy = ground_indices[:, :2] // outline_cells
    lo = min_grid_xy.min(axis=0) - 2
    hi = min_grid_xy.max(axis=0) + 2
    shape = tuple((hi - lo + 1).tolist())

    blocked = np.zeros(shape, bool)
    inside = np.zeros(shape, bool)
    outside = np.ones(shape, bool)

    building_u, building_v = np.meshgrid(np.arange(3.0, 7.0, 0.1), np.arange(3.0, 7.0, 0.1), indexing='ij')
    building_pts = _to_world(building_u.ravel(), building_v.ravel(), np.zeros(building_u.size))
    building_indices = np.floor((building_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    b_grid_xy = building_indices[:, :2] // outline_cells - lo

    inside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = True
    outside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = False

    outline = {
        'applied': True,
        'inside': inside,
        'outside': outside,
        'barrier': blocked,
        'origin': lo,
        'cells': outline_cells,
        'cell_size_m': cell_size_m,
        'shape': shape,
        'has_enclosed': True,
    }

    # Indoor ground voxels 4 m above ground at u=5.0, v=5.0
    indoor_pts = _to_world(np.array([5.0]), np.array([5.0]), np.array([4.0]))

    db_path = tmp_path / "indoor_votes.sqlite"
    tin_path = tmp_path / "indoor_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (indoor_pts, BY_ID['ground'], 178),
        ],
        votes_entries=[
            (indoor_pts, [(BY_ID['ground'], 0.7), (BY_ID['floor'], 0.3)]),
        ],
        obs_count=1,
    )
    try:
        report = build_tin_dtm(
            store,
            BY_ID,
            outline=outline,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        assert report['nodes'] > 0

        with TinDTM(tin_path) as tin:
            # Check indoor point is not in DTM nodes
            dtm_indices = np.floor((tin.nodes - ORIGIN) / VOXEL_SIZE).astype(np.int64)
            dtm_grid_xy = dtm_indices[:, :2] // outline_cells - lo
            gx, gy = dtm_grid_xy[:, 0], dtm_grid_xy[:, 1]
            in_grid = (gx >= 0) & (gx < shape[0]) & (gy >= 0) & (gy < shape[1])
            assert not np.any(inside[gx[in_grid], gy[in_grid]])

            # Apply DTM ground band
            band_rep = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=outline,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
            assert band_rep['ground_to'].get(BY_ID['floor'], 0) > 0

        indoor_keys = voxel_keys(indoor_pts, ORIGIN, VOXEL_SIZE)
        for k in indoor_keys:
            row = store.db.execute("SELECT code, confidence FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row is not None
            assert row[0] == BY_ID['floor']  # 102
            assert row[1] == round(0.3 / 1.0 * 255)
    finally:
        store.close()


def test_dtm_unclassified_within_3cm_outside_building_becomes_ground(tmp_path):
    """(c) Unclassified voxels within 3 cm over DTM outside building -> 2."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.1), np.arange(0.0, 6.0, 0.1), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Unclassified voxel 2 cm over DTM at (u=3.0, v=3.0)
    unclass_pts = _to_world(np.array([3.0]), np.array([3.0]), np.array([3.0 * np.tan(np.radians(slope_deg)) + 0.02]))

    db_path = tmp_path / "unclass_votes.sqlite"
    tin_path = tmp_path / "unclass_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (unclass_pts, 1, 0),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            report = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        assert report['to_ground'].get(1, 0) > 0
        unclass_keys = voxel_keys(unclass_pts, ORIGIN, VOXEL_SIZE)
        for k in unclass_keys:
            row = store.db.execute("SELECT code, confidence FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row is not None
            assert row[0] == BY_ID['ground']  # 2
    finally:
        store.close()


def test_dtm_lawn_and_road_remain_unchanged(tmp_path):
    """(d) Lawn (3) within 3 cm stays 3; road (11) within and 10 cm over DTM stays 11."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.1), np.arange(0.0, 6.0, 0.1), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Lawn 2 cm above DTM (u=2.0, v=2.0)
    lawn_pts = _to_world(np.array([2.0]), np.array([2.0]), np.array([2.0 * np.tan(np.radians(slope_deg)) + 0.02]))
    # Road 2 cm above DTM (u=3.0, v=3.0)
    road_pts_2cm = _to_world(np.array([3.0]), np.array([3.0]), np.array([3.0 * np.tan(np.radians(slope_deg)) + 0.02]))
    # Road 10 cm above DTM (u=4.0, v=4.0)
    road_pts_10cm = _to_world(np.array([4.0]), np.array([4.0]), np.array([4.0 * np.tan(np.radians(slope_deg)) + 0.10]))

    db_path = tmp_path / "exempt_votes.sqlite"
    tin_path = tmp_path / "exempt_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (lawn_pts, BY_ID['low_vegetation'], 200),
            (road_pts_2cm, BY_ID['road'], 220),
            (road_pts_10cm, BY_ID['road'], 220),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        for k in voxel_keys(lawn_pts, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == BY_ID['low_vegetation']  # 3
        for k in voxel_keys(road_pts_2cm, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == BY_ID['road']  # 11
        for k in voxel_keys(road_pts_10cm, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == BY_ID['road']  # 11
    finally:
        store.close()


def test_dtm_ground_without_second_vote_becomes_unclassified(tmp_path):
    """(e) Ground voxel without second vote 0.5 m above DTM -> 1."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.1), np.arange(0.0, 6.0, 0.1), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Outlier ground voxel 0.5 m over DTM at (u=3.0, v=3.0) with only ground vote
    outlier_pts = _to_world(np.array([3.0]), np.array([3.0]), np.array([3.0 * np.tan(np.radians(slope_deg)) + 0.50]))

    db_path = tmp_path / "no_second_votes.sqlite"
    tin_path = tmp_path / "no_second_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (outlier_pts, BY_ID['ground'], 255),
        ],
        votes_entries=[
            (outlier_pts, [(BY_ID['ground'], 1.0)]),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            report = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        assert report['ground_to'].get(1, 0) > 0
        for k in voxel_keys(outlier_pts, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code, confidence FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == 1
            assert row[1] == 0
    finally:
        store.close()


def test_dtm_no_coverage_gap_remains_unchanged(tmp_path):
    """(f) No DTM coverage (gap > terrain_max_gap) leaves voxels unchanged."""
    u_p1, v_p1 = np.meshgrid(np.arange(0.0, 2.0, 0.1), np.arange(0.0, 4.0, 0.1), indexing='ij')
    u_p2, v_p2 = np.meshgrid(np.arange(6.0, 8.0, 0.1), np.arange(0.0, 4.0, 0.1), indexing='ij')
    pts_p1 = _to_world(u_p1.ravel(), v_p1.ravel(), np.zeros(u_p1.size))
    pts_p2 = _to_world(u_p2.ravel(), v_p2.ravel(), np.zeros(u_p2.size))

    # Voxel in the middle of the gap at (u=4.0, v=2.0, w=0.5)
    gap_pts = _to_world(np.array([4.0]), np.array([2.0]), np.array([0.5]))

    db_path = tmp_path / "gap_cov_votes.sqlite"
    tin_path = tmp_path / "gap_cov_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (np.vstack((pts_p1, pts_p2)), BY_ID['ground'], 255),
            (gap_pts, BY_ID['ground'], 255),
        ],
        votes_entries=[
            (gap_pts, [(BY_ID['ground'], 0.6), (BY_ID['low_vegetation'], 0.4)]),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 2.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        for k in voxel_keys(gap_pts, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code, confidence FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == BY_ID['ground']  # Still 2
            assert row[1] == 255
    finally:
        store.close()


def test_dtm_tree_trunk_within_3cm_remains_tree_trunk(tmp_path):
    """(g) Tree trunk voxel within 3 cm over DTM remains code 95."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.1), np.arange(0.0, 6.0, 0.1), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Tree trunk 2 cm above DTM (u=3.0, v=3.0)
    trunk_pts = _to_world(np.array([3.0]), np.array([3.0]), np.array([3.0 * np.tan(np.radians(slope_deg)) + 0.02]))

    db_path = tmp_path / "trunk_votes.sqlite"
    tin_path = tmp_path / "trunk_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (trunk_pts, BY_ID['tree_trunk'], 210),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        with TinDTM(tin_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        for k in voxel_keys(trunk_pts, ORIGIN, VOXEL_SIZE):
            row = store.db.execute("SELECT code FROM decisions WHERE key=?", (bytes(k),)).fetchone()
            assert row[0] == BY_ID['tree_trunk']  # Remains 95
    finally:
        store.close()


def test_dtm_disabled_leaves_decisions_byte_identical(tmp_path):
    """(h) ground_dtm=False leaves decisions table byte-identical."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.1), np.arange(0.0, 6.0, 0.1), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    bush_pts = _to_world(np.array([3.0]), np.array([3.0]), np.array([3.0 * np.tan(np.radians(slope_deg)) + 0.60]))
    unclass_pts = _to_world(np.array([2.0]), np.array([2.0]), np.array([2.0 * np.tan(np.radians(slope_deg)) + 0.02]))

    db_path = tmp_path / "disabled_votes.sqlite"
    tin_path = tmp_path / "disabled_dtm.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (bush_pts, BY_ID['ground'], 150),
            (unclass_pts, 1, 0),
        ],
        votes_entries=[
            (bush_pts, [(BY_ID['ground'], 0.6), (BY_ID['low_vegetation'], 0.4)]),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=tin_path,
        )
        decisions_before = store.db.execute("SELECT key, code, confidence FROM decisions ORDER BY key").fetchall()

        with TinDTM(tin_path) as tin:
            report = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': False, 'ground_dtm_band': 0.03, 'min_observations': 1},
            )
        assert report['enabled'] is False

        decisions_after = store.db.execute("SELECT key, code, confidence FROM decisions ORDER BY key").fetchall()
        assert decisions_after == decisions_before
    finally:
        store.close()


def test_dtm_export_unclassified_outside_building_becomes_ground(tmp_path):
    """(i) Export: Point 2 cm over DTM with class 1 outside building -> 2."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.2), np.arange(0.0, 6.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Building outline for u in [3, 5], v in [3, 5]
    cell_size_m = 0.5
    outline_cells = max(1, round(cell_size_m / VOXEL_SIZE))
    ground_indices = np.floor((ground_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    min_grid_xy = ground_indices[:, :2] // outline_cells
    lo = min_grid_xy.min(axis=0) - 2
    hi = min_grid_xy.max(axis=0) + 2
    shape = tuple((hi - lo + 1).tolist())

    blocked = np.zeros(shape, bool)
    inside = np.zeros(shape, bool)
    outside = np.ones(shape, bool)

    b_u, b_v = np.meshgrid(np.arange(3.0, 5.0, 0.1), np.arange(3.0, 5.0, 0.1), indexing='ij')
    b_pts = _to_world(b_u.ravel(), b_v.ravel(), np.zeros(b_u.size))
    b_indices = np.floor((b_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    b_grid_xy = b_indices[:, :2] // outline_cells - lo
    inside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = True
    outside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = False

    outline = {
        'applied': True,
        'inside': inside,
        'outside': outside,
        'barrier': blocked,
        'origin': lo,
        'cells': outline_cells,
        'cell_size_m': cell_size_m,
        'shape': shape,
        'has_enclosed': True,
    }

    db_path = tmp_path / "export_i_votes.sqlite"
    dtm_path = tmp_path / "dtm.npz"
    alt_path = tmp_path / "ground_alternatives.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[(ground_pts, BY_ID['ground'], 255)],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=outline,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=dtm_path,
        )
        with TinDTM(dtm_path) as tin:
            rep = apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=outline,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
                alternatives_path=alt_path,
            )

        # Query point 2 cm over DTM outside building (u=1.5, v=1.5)
        u_pt, v_pt = 1.5, 1.5
        z_dtm = u_pt * np.tan(np.radians(slope_deg))
        pt_outside = _to_world(np.array([u_pt]), np.array([v_pt]), np.array([z_dtm + 0.02]))

        dummy_store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            ground_dtm=dtm_path,
            ground_dtm_alternatives=alt_path,
            ground_dtm_band=0.03,
        ) as cs:
            _, codes, confidence, _ = cs.block(dummy_store, 0, pt_outside, np.ones(1, bool), 'original')
            assert cs.geometry_fill_counts.get('ground_dtm_fill') == 1
        assert codes.tolist() == [BY_ID['ground']]
        assert confidence.tolist() == [64]
    finally:
        store.close()


def test_dtm_export_ground_above_band_becomes_alternative_or_unclassified(tmp_path):
    """(j) Point 5 cm over DTM with class 2 -> second class from mapping (e.g. 3); without entry -> 1."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.2), np.arange(0.0, 6.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Point with alternative at u=2.0, v=2.0, 5 cm above DTM
    u_a, v_a = 2.0, 2.0
    z_a = u_a * np.tan(np.radians(slope_deg)) + 0.05
    pt_with_alt = _to_world(np.array([u_a]), np.array([v_a]), np.array([z_a]))

    # Point without alternative at u=4.0, v=4.0, 5 cm above DTM
    u_b, v_b = 4.0, 4.0
    z_b = u_b * np.tan(np.radians(slope_deg)) + 0.05
    pt_without_alt = _to_world(np.array([u_b]), np.array([v_b]), np.array([z_b]))

    db_path = tmp_path / "export_j_votes.sqlite"
    dtm_path = tmp_path / "dtm.npz"
    alt_path = tmp_path / "ground_alternatives.npz"

    # pt_with_alt has observations count 0 so apply_ground_dtm_band does not demote it,
    # but its second-best vote (low_vegetation 3) is saved into alternatives.
    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (pt_with_alt, BY_ID['ground'], 200),
        ],
        votes_entries=[
            (pt_with_alt, [(BY_ID['ground'], 0.6), (BY_ID['low_vegetation'], 0.4)]),
        ],
        obs_count=1,
    )
    # Set obs count = 0 for pt_with_alt so decision-level demotion ignores it
    for k in voxel_keys(pt_with_alt, ORIGIN, VOXEL_SIZE):
        store.db.execute("UPDATE observations SET count = 0 WHERE key = ?", (bytes(k),))
    store.db.commit()

    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=dtm_path,
        )
        with TinDTM(dtm_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
                alternatives_path=alt_path,
            )

        # Now add pt_without_alt to decisions as ground (after alternatives was built)
        for k in voxel_keys(pt_without_alt, ORIGIN, VOXEL_SIZE):
            store.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)", (bytes(k), BY_ID['ground'], 200))
        store.db.commit()
        store._invalidate_lookup()

        points = np.vstack([pt_with_alt, pt_without_alt])
        dummy_store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()

        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            ground_dtm=dtm_path,
            ground_dtm_alternatives=alt_path,
            ground_dtm_band=0.03,
        ) as cs:
            _, codes, confidence, _ = cs.block(dummy_store, 0, points, np.ones(len(points), bool), 'original')
            assert cs.geometry_fill_counts.get('ground_dtm_to_alt') == 2

        assert codes.tolist() == [BY_ID['low_vegetation'], 1]  # 3, then 1
        assert confidence[1] == 0
    finally:
        store.close()


def test_dtm_export_unclassified_inside_building_remains_unclassified(tmp_path):
    """(k) Point with class 1 within 3 cm, but inside building outline -> remains 1."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 8.0, 0.2), np.arange(0.0, 8.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Building outline for u in [3, 6], v in [3, 6]
    cell_size_m = 0.5
    outline_cells = max(1, round(cell_size_m / VOXEL_SIZE))
    ground_indices = np.floor((ground_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    min_grid_xy = ground_indices[:, :2] // outline_cells
    lo = min_grid_xy.min(axis=0) - 2
    hi = min_grid_xy.max(axis=0) + 2
    shape = tuple((hi - lo + 1).tolist())

    blocked = np.zeros(shape, bool)
    inside = np.zeros(shape, bool)
    outside = np.ones(shape, bool)

    b_u, b_v = np.meshgrid(np.arange(3.0, 6.0, 0.1), np.arange(3.0, 6.0, 0.1), indexing='ij')
    b_pts = _to_world(b_u.ravel(), b_v.ravel(), np.zeros(b_u.size))
    b_indices = np.floor((b_pts - ORIGIN) / VOXEL_SIZE).astype(np.int64)
    b_grid_xy = b_indices[:, :2] // outline_cells - lo
    inside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = True
    outside[b_grid_xy[:, 0], b_grid_xy[:, 1]] = False

    outline = {
        'applied': True,
        'inside': inside,
        'outside': outside,
        'barrier': blocked,
        'origin': lo,
        'cells': outline_cells,
        'cell_size_m': cell_size_m,
        'shape': shape,
        'has_enclosed': True,
    }

    db_path = tmp_path / "export_k_votes.sqlite"
    dtm_path = tmp_path / "dtm.npz"
    alt_path = tmp_path / "ground_alternatives.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[(ground_pts, BY_ID['ground'], 255)],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=outline,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=dtm_path,
        )
        with TinDTM(dtm_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=outline,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
                alternatives_path=alt_path,
            )

        # Indoor unclassified point at u=4.5, v=4.5, 2 cm above DTM
        u_pt, v_pt = 4.5, 4.5
        z_dtm = u_pt * np.tan(np.radians(slope_deg))
        pt_indoor = _to_world(np.array([u_pt]), np.array([v_pt]), np.array([z_dtm + 0.02]))

        dummy_store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            ground_dtm=dtm_path,
            ground_dtm_alternatives=alt_path,
            ground_dtm_band=0.03,
        ) as cs:
            _, codes, confidence, _ = cs.block(dummy_store, 0, pt_indoor, np.ones(1, bool), 'original')
            assert cs.geometry_fill_counts.get('ground_dtm_fill', 0) == 0

        assert codes.tolist() == [1]  # Remains unclassified
        assert confidence.tolist() == [0]
    finally:
        store.close()


def test_dtm_export_tree_trunk_within_3cm_remains_tree_trunk(tmp_path):
    """(l) Point with class 95 within 3 cm -> remains 95."""
    slope_deg = 15.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.2), np.arange(0.0, 6.0, 0.2), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), u_grid.ravel() * np.tan(np.radians(slope_deg)))

    # Tree trunk point 1 cm above DTM at u=2.5, v=2.5
    u_pt, v_pt = 2.5, 2.5
    z_dtm = u_pt * np.tan(np.radians(slope_deg))
    pt_trunk = _to_world(np.array([u_pt]), np.array([v_pt]), np.array([z_dtm + 0.01]))

    db_path = tmp_path / "export_l_votes.sqlite"
    dtm_path = tmp_path / "dtm.npz"
    alt_path = tmp_path / "ground_alternatives.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[
            (ground_pts, BY_ID['ground'], 255),
            (pt_trunk, BY_ID['tree_trunk'], 240),
        ],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=dtm_path,
        )
        with TinDTM(dtm_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
                alternatives_path=alt_path,
            )

        dummy_store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()
        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            ground_dtm=dtm_path,
            ground_dtm_alternatives=alt_path,
            ground_dtm_band=0.03,
        ) as cs:
            _, codes, confidence, _ = cs.block(dummy_store, 0, pt_trunk, np.ones(1, bool), 'original')
            assert cs.geometry_fill_counts.get('ground_dtm_fill', 0) == 0
            assert cs.geometry_fill_counts.get('ground_dtm_to_alt', 0) == 0

        assert codes.tolist() == [BY_ID['tree_trunk']]  # 95
        assert confidence.tolist() == [240]
    finally:
        store.close()


def test_dtm_export_disabled_retains_legacy_terrain_fill(tmp_path):
    """(m) ground_dtm disabled -> export identical to previous (terrain_fill as before)."""
    slope_deg = 0.0
    u_grid, v_grid = np.meshgrid(np.arange(0.0, 6.0, 0.5), np.arange(0.0, 6.0, 0.5), indexing='ij')
    ground_pts = _to_world(u_grid.ravel(), v_grid.ravel(), np.zeros(u_grid.size))

    # Point 10 cm above terrain (within terrain_ground_tolerance 0.15 m, but outside DTM band 0.03 m)
    pt_10cm = _to_world(np.array([2.5]), np.array([2.5]), np.array([0.10]))

    terrain_path = tmp_path / "legacy_terrain.npz"
    np.savez(
        terrain_path,
        xy=ground_pts[:, :2],
        z=ground_pts[:, 2],
        covered=np.zeros(len(ground_pts), bool),
        metadata=json.dumps({'cell_size': 0.5, 'max_gap': 5.0}),
    )

    db_path = tmp_path / "export_m_votes.sqlite"
    dtm_path = tmp_path / "dtm.npz"
    alt_path = tmp_path / "ground_alternatives.npz"

    store = _store_with_data(
        db_path,
        decisions_entries=[(ground_pts, BY_ID['ground'], 255)],
        obs_count=1,
    )
    try:
        build_tin_dtm(
            store,
            BY_ID,
            outline=None,
            origin=ORIGIN,
            voxel_size=VOXEL_SIZE,
            config={'dtm_cell_size': 0.5, 'terrain_max_gap': 5.0, 'terrain_max_slope': 1.0},
            target_path=dtm_path,
        )
        with TinDTM(dtm_path) as tin:
            apply_ground_dtm_band(
                store,
                BY_ID,
                tin,
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': 0.03, 'min_observations': 1},
                alternatives_path=alt_path,
            )

        dummy_store = type('Store', (), {'meta': {'scan_index': 1, 'metadata': {}}})()

        # Case 1: ground_dtm=None, terrain_fill=True -> legacy terrain fill applies (10 cm <= 15 cm -> code 2)
        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            terrain=terrain_path,
            terrain_fill=True,
            terrain_ground_tolerance=0.15,
            ground_dtm=None,
        ) as cs_legacy:
            _, codes_legacy, conf_legacy, _ = cs_legacy.block(dummy_store, 0, pt_10cm, np.ones(1, bool), 'original')
            assert cs_legacy.geometry_fill_counts.get('terrain_fill') == 1
        assert codes_legacy.tolist() == [BY_ID['ground']]  # 2
        assert conf_legacy.tolist() == [64]

        # Case 2: ground_dtm active -> legacy terrain fill skipped, DTM band (0.03 m) rejects 10 cm -> remains 1
        with ClassifiedSource(
            tmp_path,
            [],
            store,
            ORIGIN,
            VOXEL_SIZE,
            terrain=terrain_path,
            terrain_fill=True,
            terrain_ground_tolerance=0.15,
            ground_dtm=dtm_path,
            ground_dtm_alternatives=alt_path,
            ground_dtm_band=0.03,
        ) as cs_dtm:
            _, codes_dtm, conf_dtm, _ = cs_dtm.block(dummy_store, 0, pt_10cm, np.ones(1, bool), 'original')
            assert 'terrain_fill' not in cs_dtm.geometry_fill_counts
            assert cs_dtm.geometry_fill_counts.get('ground_dtm_fill', 0) == 0
        assert codes_dtm.tolist() == [1]  # remains unclassified
        assert conf_dtm.tolist() == [0]
    finally:
        store.close()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get('NOMIVOX_RUN_SLOW') != '1',
    reason='set NOMIVOX_RUN_SLOW=1 to run the DTM memory benchmark',
)
def test_dtm_ground_band_memory_and_runtime_benchmark(tmp_path):
    """Exercise the DTM rule at HSLU-scale without retaining Python objects per voxel."""
    total_voxels = 5_000_000
    ground_voxels = 2_000_000
    insert_block_size = 100_000
    db_path = tmp_path / 'dtm_memory.sqlite'

    class FlatDtm:
        def height(self, xy):
            return np.full(len(xy), ORIGIN[2] + VOXEL_SIZE * .5, dtype=np.float64)

    def rss_peak_during(callback):
        process = psutil.Process()
        baseline = process.memory_info().rss
        peak = [baseline]
        finished = threading.Event()

        def sample():
            while not finished.wait(.02):
                peak[0] = max(peak[0], process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            started = time.perf_counter()
            callback()
            duration = time.perf_counter() - started
        finally:
            finished.set()
            sampler.join()
            peak[0] = max(peak[0], process.memory_info().rss)
        return duration, peak[0] - baseline

    store = VoteStore(db_path)
    try:
        for start in range(0, total_voxels, insert_block_size):
            stop = min(start + insert_block_size, total_voxels)
            count = stop - start
            indices = np.zeros((count, 3), dtype='<i8')
            indices[:, 0] = np.arange(start, stop, dtype=np.int64)
            ground = np.arange(start, stop, dtype=np.int64) < ground_voxels
            # Half of the ground winners are 20 cm above the flat DTM.
            indices[ground & ((indices[:, 0] & 1) == 1), 2] = 10
            indices[~ground, 2] = 100
            keys = np.ascontiguousarray(indices).view('V24').reshape(-1)
            winner_codes = np.where(ground, BY_ID['ground'], BY_ID['floor'])
            alternative_codes = np.where(ground, BY_ID['low_vegetation'], BY_ID['ground'])
            rows = (
                (bytes(key), int(winner), .60, .60)
                for key, winner in zip(keys, winner_codes)
            )
            store.db.executemany('INSERT INTO votes VALUES (?, ?, ?, ?)', rows)
            rows = (
                (bytes(key), int(alternative), .40, .40)
                for key, alternative in zip(keys, alternative_codes)
            )
            store.db.executemany('INSERT INTO votes VALUES (?, ?, ?, ?)', rows)
            store.db.executemany(
                'INSERT INTO observations VALUES (?, ?)',
                ((bytes(key), 1) for key in keys),
            )
            store.db.commit()

        decide_s, _ = rss_peak_during(store.decide)
        apply_s, memory_increase = rss_peak_during(
            lambda: apply_ground_dtm_band(
                store,
                BY_ID,
                FlatDtm(),
                outline=None,
                origin=ORIGIN,
                voxel_size=VOXEL_SIZE,
                config={'ground_dtm': True, 'ground_dtm_band': .03, 'min_observations': 1},
                alternatives_path=tmp_path / 'ground_alternatives.npz',
            )
        )
        print(
            'DTM memory benchmark: '
            f'decide={decide_s:.2f}s apply={apply_s:.2f}s '
            f'peak_rss_increase={memory_increase / 1024**3:.3f} GiB'
        )
        assert apply_s <= decide_s
        assert memory_increase <= 1024**3
    finally:
        store.close()
