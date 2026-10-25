"""Triangulated Irregular Network Digital Terrain Model (TIN DTM)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.qhull import QhullError

LOG = logging.getLogger(__name__)


def build_tin_dtm(
    votes: Any,
    by_id: dict[str, int],
    outline: dict[str, Any] | None,
    origin: Any,
    voxel_size: float,
    config: dict[str, Any] | None,
    target_path: str | Path,
    scan_context: Any = None,
    checkpoint: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Build a TIN DTM from ground family voxel decisions.

    Candidates are ground-family voxels outside building outlines, aggregated
    into 2-D cells by a robust 10th-percentile height, filtered by maximum
    slope, and triangulated using 2-D Delaunay.
    """
    started = time.perf_counter()
    config = config or {}
    origin = np.asarray(origin, dtype=np.float64)
    voxel_size = float(voxel_size)
    dtm_cell_size = float(config.get('dtm_cell_size', config.get('terrain_cell_size', 0.5)))
    terrain_max_slope = float(config.get('terrain_max_slope', 0.75))
    terrain_max_gap = float(config.get('terrain_max_gap', 5.0))

    ground_family = ('ground', 'road', 'sidewalk', 'curb', 'manhole')
    ground_codes = tuple(sorted({int(by_id[name]) for name in ground_family if name in by_id}))
    reflection_code = int(by_id.get('reflection', 68))
    candidate_codes = tuple(code for code in ground_codes if code != reflection_code)

    chunks = []
    if candidate_codes:
        placeholders = ','.join('?' for _ in candidate_codes)
        cursor = votes.db.execute(
            f"SELECT key FROM decisions WHERE code IN ({placeholders})",
            candidate_codes,
        )
        batch_size = 1_000_000
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            checkpoint()
            keys = b''.join(row[0] for row in rows)
            indices = np.frombuffer(keys, dtype='<i8').reshape(-1, 3)

            if outline and outline.get('applied'):
                inside_grid = outline.get('inside')
                barrier_grid = outline.get('barrier')
                lo = np.asarray(outline['origin'], dtype=np.int64)
                cells = int(outline['cells'])
                grid_xy = indices[:, :2] // cells
                raster_idx = grid_xy - lo
                gx = raster_idx[:, 0]
                gy = raster_idx[:, 1]
                shape = inside_grid.shape if inside_grid is not None else outline.get('shape')
                in_bounds = (gx >= 0) & (gx < shape[0]) & (gy >= 0) & (gy < shape[1])
                if in_bounds.any():
                    gx_valid = gx[in_bounds]
                    gy_valid = gy[in_bounds]
                    excluded_valid = np.zeros(len(gx_valid), dtype=bool)
                    if inside_grid is not None:
                        excluded_valid |= inside_grid[gx_valid, gy_valid]
                    if barrier_grid is not None:
                        excluded_valid |= barrier_grid[gx_valid, gy_valid]
                    keep_batch = np.ones(len(indices), dtype=bool)
                    keep_batch[in_bounds] = ~excluded_valid
                    indices = indices[keep_batch]

            if len(indices):
                chunks.append(indices)

    if chunks:
        all_indices = np.vstack(chunks)
        # The stacked array owns the data; keeping all row-block buffers alive
        # doubles the peak resident set on city-scale scans.
        del chunks
        world = origin + (all_indices.astype(np.float64) + 0.5) * voxel_size
    else:
        world = np.empty((0, 3), dtype=np.float64)

    if len(world):
        cell_coords = np.floor((world[:, :2] - origin[:2]) / dtm_cell_size).astype(np.int64)
        _, cell_inverse = np.unique(cell_coords, axis=0, return_inverse=True)
        cell_inverse = cell_inverse.reshape(-1)

        order = np.lexsort((world[:, 2], cell_inverse))
        sorted_cells = cell_inverse[order]
        starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
        counts = np.diff(np.r_[starts, len(sorted_cells)])
        q_offsets = np.floor(0.10 * (counts - 1)).astype(np.int64)
        picked_indices = order[starts + q_offsets]
        nodes = world[picked_indices]
    else:
        nodes = np.empty((0, 3), dtype=np.float64)

    if len(nodes) > 1:
        xy = nodes[:, :2]
        heights = nodes[:, 2]
        nearby_distance, nearby = cKDTree(xy).query(xy, k=min(5, len(xy)), workers=-1)
        slope = np.divide(
            np.abs(heights[:, None] - heights[nearby]),
            nearby_distance,
            out=np.full_like(nearby_distance, np.inf),
            where=nearby_distance > 1e-9,
        )
        keep = np.any(slope <= terrain_max_slope, axis=1)
        nodes = nodes[keep]

    triangles = np.empty((0, 3), dtype=np.int64)
    if len(nodes) >= 3:
        try:
            tri = Delaunay(nodes[:, :2])
            simplices = tri.simplices
            p = nodes[:, :2]
            e01 = np.hypot(p[simplices[:, 1], 0] - p[simplices[:, 0], 0], p[simplices[:, 1], 1] - p[simplices[:, 0], 1])
            e12 = np.hypot(p[simplices[:, 2], 0] - p[simplices[:, 1], 0], p[simplices[:, 2], 1] - p[simplices[:, 1], 1])
            e20 = np.hypot(p[simplices[:, 0], 0] - p[simplices[:, 2], 0], p[simplices[:, 0], 1] - p[simplices[:, 2], 1])
            max_edge = np.maximum(np.maximum(e01, e12), e20)
            valid_triangles = max_edge <= terrain_max_gap
            triangles = simplices[valid_triangles]
        except (QhullError, ValueError):
            triangles = np.empty((0, 3), dtype=np.int64)

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'schema_version': 1,
        'dtm_cell_size': dtm_cell_size,
        'terrain_max_gap': terrain_max_gap,
        'terrain_max_slope': terrain_max_slope,
        'origin': origin.tolist(),
        'voxel_size': voxel_size,
    }
    np.savez_compressed(
        target_path,
        metadata=np.asarray(json.dumps(metadata)),
        xyz=nodes,
        nodes=nodes,
        triangles=triangles,
        xy=nodes[:, :2] if len(nodes) else np.empty((0, 2), np.float64),
        z=nodes[:, 2] if len(nodes) else np.empty(0, np.float64),
    )

    return {
        'nodes': int(len(nodes)),
        'triangles': int(len(triangles)),
        'duration_s': float(round(time.perf_counter() - started, 3)),
        'artifact': str(target_path),
    }


def apply_ground_dtm_band(
    votes: Any,
    by_id: dict[str, int],
    dtm: Any,
    outline: dict[str, Any] | None,
    origin: Any,
    voxel_size: float,
    config: dict[str, Any] | None,
    active_codes: Any = None,
    checkpoint: Callable[[], None] = lambda: None,
    alternatives_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply height-relative decision rules using the TIN DTM ground band.

    Ground voxels located outside the configured vertical band above/below the
    DTM surface are reclassified to their second-best SAM class or 1. Unclassified
    or non-ground voxels within the band and outside building footprints become
    ground, preserving roads, sidewalks, low vegetation, and protected classes.
    """
    started = time.perf_counter()
    config = config or {}
    band = float(config.get('ground_dtm_band', 0.03))
    enabled = bool(config.get('ground_dtm', True))

    if not enabled or 'ground' not in by_id or dtm is None:
        return {
            'enabled': False,
            'band': band,
            'ground_to': {},
            'to_ground': {},
            'duration_s': 0.0,
            'ground_alternatives': None,
        }

    origin = np.asarray(origin, dtype=np.float64)
    voxel_size = float(voxel_size)
    min_obs = int(config.get('min_observations', 1))

    ground_code = int(by_id['ground'])
    reflection_code = int(by_id.get('reflection', 68))
    protected_names = config.get(
        'segment_protected_classes',
        ('stairs', 'door', 'window', 'electrical', 'traffic_sign', 'manhole'),
    )
    protected_codes = {int(by_id[name]) for name in protected_names if name in by_id}

    exempt_names = (
        'ground',
        'low_vegetation',
        'road',
        'sidewalk',
        'curb',
        'manhole',
        'water',
        'stairs',
        'tree_trunk',
        'reflection',
    )
    exempt_codes = {int(by_id[name]) for name in exempt_names if name in by_id}
    exempt_codes.update(protected_codes)
    exempt_codes.add(ground_code)
    exempt_codes.add(int(by_id.get('tree_trunk', 95)))
    exempt_codes.add(int(by_id.get('low_vegetation', 3)))
    exempt_codes.add(int(by_id.get('road', 11)))
    exempt_codes.add(int(by_id.get('sidewalk', 13)))
    exempt_codes.add(int(by_id.get('curb', 14)))
    exempt_codes.add(int(by_id.get('manhole', 17)))
    exempt_codes.add(int(by_id.get('water', 9)))
    exempt_codes.add(int(by_id.get('stairs', 15)))
    exempt_codes.add(reflection_code)
    exempt_codes_array = np.fromiter(exempt_codes, dtype=np.int64, count=len(exempt_codes))

    active_codes_array = (
        np.asarray(tuple(active_codes), dtype=np.int64)
        if active_codes is not None
        else None
    )

    has_observations = votes.db.execute("SELECT 1 FROM observations LIMIT 1").fetchone() is not None

    outline_applied = bool(outline and outline.get('applied'))
    inside_grid = outline.get('inside') if outline_applied else None
    barrier_grid = outline.get('barrier') if outline_applied else None
    lo = np.asarray(outline['origin'], dtype=np.int64) if outline_applied else None
    cells = int(outline['cells']) if outline_applied else 1
    outline_shape = (
        inside_grid.shape if inside_grid is not None else outline.get('shape')
    ) if outline_applied else None

    # Handle dtm provided as file path or TinDTM instance
    if isinstance(dtm, (str, Path)):
        dtm_instance = TinDTM(dtm)
        should_close_dtm = True
    else:
        dtm_instance = dtm
        should_close_dtm = False

    protected_codes_array = np.fromiter(
        protected_codes, dtype=np.int64, count=len(protected_codes),
    )
    # A few million keys remain bounded numeric working sets, while avoiding
    # hundreds of SQLite round trips on large scans.
    batch_size = max(1, int(config.get('ground_dtm_block_size', 2_500_000)))
    db = votes.db
    db.execute("DROP TABLE IF EXISTS temp.dtm_block_keys")
    db.execute("DROP TABLE IF EXISTS temp.dtm_block_updates")
    db.execute("CREATE TEMP TABLE dtm_block_keys(key BLOB PRIMARY KEY) WITHOUT ROWID")
    db.execute(
        "CREATE TEMP TABLE dtm_block_updates("
        "key BLOB PRIMARY KEY, code INTEGER NOT NULL, confidence INTEGER NOT NULL"
        ") WITHOUT ROWID"
    )

    ground_to: dict[int, int] = {}
    to_ground: dict[int, int] = {}

    def keys_from_rows(rows: list[tuple[Any, ...]]) -> np.ndarray:
        """Return a compact, owning array of 24-byte voxel keys for one block."""
        return np.frombuffer(b''.join(row[0] for row in rows), dtype='V24').copy()

    def add_counts(target: dict[int, int], values: np.ndarray) -> None:
        if len(values):
            classes, counts = np.unique(values, return_counts=True)
            for code, count in zip(classes, counts):
                target[int(code)] = target.get(int(code), 0) + int(count)

    def set_block_keys(keys: np.ndarray) -> None:
        db.execute("DELETE FROM temp.dtm_block_keys")
        if len(keys):
            db.executemany(
                "INSERT INTO temp.dtm_block_keys VALUES (?)",
                ((bytes(key),) for key in keys),
            )

    def second_best(keys: np.ndarray, with_confidence: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return the eligible SAM runner-up and optional confidence for one key block."""
        result_codes = np.ones(len(keys), dtype=np.int64)
        result_confidence = np.zeros(len(keys), dtype=np.int64)
        if not len(keys):
            return result_codes, result_confidence

        set_block_keys(keys)
        cursor = db.execute(
            "SELECT v.key, v.class, v.weight "
            "FROM votes AS v JOIN temp.dtm_block_keys AS b ON b.key = v.key "
            "ORDER BY v.key, v.weight DESC, v.class ASC"
        )
        key_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        try:
            while rows := cursor.fetchmany(batch_size):
                key_parts.append(keys_from_rows(rows))
                class_parts.append(np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows)))
                weight_parts.append(np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows)))
        finally:
            cursor.close()
        if not key_parts:
            return result_codes, result_confidence

        vote_keys = np.concatenate(key_parts)
        vote_classes = np.concatenate(class_parts)
        vote_weights = np.concatenate(weight_parts)
        unique_vote_keys, starts = np.unique(vote_keys, return_index=True)
        totals = np.add.reduceat(vote_weights, starts)

        eligible = (
            (vote_classes != ground_code)
            & (vote_classes != reflection_code)
            & ~np.isin(vote_classes, protected_codes_array)
            & (vote_weights > 0)
        )
        if active_codes_array is not None:
            eligible &= np.isin(vote_classes, active_codes_array)
        if not np.any(eligible):
            return result_codes, result_confidence

        # SQL order makes the first eligible vote per key the required
        # highest-weight class, with the smaller class ID winning ties.
        eligible_keys = vote_keys[eligible]
        eligible_classes = vote_classes[eligible]
        eligible_weights = vote_weights[eligible]
        best_keys, best_starts = np.unique(eligible_keys, return_index=True)
        best_classes = eligible_classes[best_starts]
        best_weights = eligible_weights[best_starts]

        key_to_best = np.searchsorted(best_keys, keys)
        has_best = key_to_best < len(best_keys)
        if np.any(has_best):
            positions = np.flatnonzero(has_best)
            matches = best_keys[key_to_best[positions]] == keys[positions]
            positions = positions[matches]
            if len(positions):
                result_codes[positions] = best_classes[key_to_best[positions]]
                if with_confidence:
                    key_to_total = np.searchsorted(unique_vote_keys, keys[positions])
                    total_weights = totals[key_to_total]
                    shares = np.divide(
                        best_weights[key_to_best[positions]], total_weights,
                        out=np.zeros(len(positions), dtype=np.float64),
                        where=total_weights > 0,
                    )
                    result_confidence[positions] = np.clip(
                        np.rint(shares * 255), 0, 255,
                    ).astype(np.int64)
        return result_codes, result_confidence

    def apply_updates(keys: np.ndarray, codes: np.ndarray, confidences: np.ndarray) -> None:
        """Apply one compact update block after all of its read cursors are closed."""
        if not len(keys):
            return
        db.execute("DELETE FROM temp.dtm_block_updates")
        db.executemany(
            "INSERT INTO temp.dtm_block_updates VALUES (?, ?, ?)",
            (
                (bytes(key), int(code), int(confidence))
                for key, code, confidence in zip(keys, codes, confidences)
            ),
        )
        db.execute(
            "UPDATE decisions AS d SET code = u.code, confidence = u.confidence "
            "FROM temp.dtm_block_updates AS u WHERE d.key = u.key"
        )
        db.commit()

    try:
        last_key: bytes | None = None
        while True:
            if has_observations:
                if last_key is None:
                    cursor = db.execute(
                        "SELECT d.key, d.code, d.confidence, COALESCE(o.count, 0) "
                        "FROM decisions AS d LEFT JOIN observations AS o ON o.key = d.key "
                        "ORDER BY d.key LIMIT ?",
                        (batch_size,),
                    )
                else:
                    cursor = db.execute(
                        "SELECT d.key, d.code, d.confidence, COALESCE(o.count, 0) "
                        "FROM decisions AS d LEFT JOIN observations AS o ON o.key = d.key "
                        "WHERE d.key > ? ORDER BY d.key LIMIT ?",
                        (last_key, batch_size),
                    )
            elif last_key is None:
                cursor = db.execute(
                    "SELECT d.key, d.code, d.confidence FROM decisions AS d ORDER BY d.key LIMIT ?",
                    (batch_size,),
                )
            else:
                cursor = db.execute(
                    "SELECT d.key, d.code, d.confidence FROM decisions AS d "
                    "WHERE d.key > ? ORDER BY d.key LIMIT ?",
                    (last_key, batch_size),
                )
            try:
                rows = cursor.fetchmany(batch_size)
            finally:
                cursor.close()
            if not rows:
                break
            checkpoint()
            last_key = rows[-1][0]

            keys = keys_from_rows(rows)
            codes = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
            confidences = np.fromiter((row[2] for row in rows), dtype=np.int64, count=len(rows))
            observations_ok = (
                np.fromiter((row[3] for row in rows), dtype=np.int64, count=len(rows)) >= min_obs
                if has_observations else np.ones(len(rows), dtype=bool)
            )
            indices = keys.view('<i8').reshape(-1, 3)
            world = origin + (indices.astype(np.float64) + 0.5) * voxel_size
            dtm_z = dtm_instance.height(world[:, :2])
            valid_dtm = np.isfinite(dtm_z)
            difference = np.abs(world[:, 2] - dtm_z)

            demote_mask = observations_ok & (codes == ground_code) & valid_dtm & (difference > band)
            promote_mask = observations_ok & (codes != ground_code) & valid_dtm & (difference <= band)
            if outline_applied and np.any(promote_mask):
                candidate_positions = np.flatnonzero(promote_mask)
                candidate_indices = indices[candidate_positions]
                raster_indices = candidate_indices[:, :2] // cells - lo
                gx, gy = raster_indices[:, 0], raster_indices[:, 1]
                in_bounds = (
                    (gx >= 0) & (gx < outline_shape[0])
                    & (gy >= 0) & (gy < outline_shape[1])
                )
                excluded = np.zeros(len(candidate_positions), dtype=bool)
                if np.any(in_bounds):
                    if inside_grid is not None:
                        excluded[in_bounds] |= inside_grid[gx[in_bounds], gy[in_bounds]]
                    if barrier_grid is not None:
                        excluded[in_bounds] |= barrier_grid[gx[in_bounds], gy[in_bounds]]
                promote_mask[candidate_positions[excluded]] = False
            promote_mask &= ~np.isin(codes, exempt_codes_array)

            update_mask = demote_mask | promote_mask
            if not np.any(update_mask):
                continue
            update_keys = keys[update_mask]
            update_codes = np.empty(len(update_keys), dtype=np.int64)
            update_confidences = np.empty(len(update_keys), dtype=np.int64)

            demote_positions = np.flatnonzero(demote_mask)
            if len(demote_positions):
                demote_codes, demote_confidences = second_best(keys[demote_positions], with_confidence=True)
                update_positions = np.searchsorted(np.flatnonzero(update_mask), demote_positions)
                update_codes[update_positions] = demote_codes
                update_confidences[update_positions] = demote_confidences
                add_counts(ground_to, demote_codes)

            promote_positions = np.flatnonzero(promote_mask)
            if len(promote_positions):
                update_positions = np.searchsorted(np.flatnonzero(update_mask), promote_positions)
                update_codes[update_positions] = ground_code
                update_confidences[update_positions] = np.where(
                    confidences[promote_positions] > 0,
                    confidences[promote_positions],
                    255,
                )
                add_counts(to_ground, codes[promote_positions])

            apply_updates(update_keys, update_codes, update_confidences)
    finally:
        if should_close_dtm:
            dtm_instance.close()

    votes._invalidate_lookup()

    # Build and save compact alternatives mapping and outline raster for export
    if alternatives_path is None:
        if hasattr(dtm_instance, 'path') and dtm_instance.path is not None:
            alternatives_path = Path(dtm_instance.path).parent / 'ground_alternatives.npz'
        else:
            alternatives_path = Path('ground_alternatives.npz')
    alternatives_path = Path(alternatives_path)
    alternatives_path.parent.mkdir(parents=True, exist_ok=True)

    alternative_count = int(db.execute(
        'SELECT COUNT(*) FROM decisions WHERE code=?', (ground_code,)
    ).fetchone()[0])
    temporary_paths: list[Path] = []
    if alternative_count:
        key_path = alternatives_path.with_name(alternatives_path.name + '.keys.tmp.npy')
        code_path = alternatives_path.with_name(alternatives_path.name + '.codes.tmp.npy')
        for path in (key_path, code_path):
            path.unlink(missing_ok=True)
        sorted_keys = np.lib.format.open_memmap(key_path, mode='w+', dtype='V24', shape=(alternative_count,))
        sorted_codes = np.lib.format.open_memmap(code_path, mode='w+', dtype=np.uint16, shape=(alternative_count,))
        temporary_paths = [key_path, code_path]
    else:
        sorted_keys = np.empty(0, dtype='V24')
        sorted_codes = np.empty(0, dtype=np.uint16)
    last_key = None
    alternative_start = 0
    while True:
        if last_key is None:
            cursor = db.execute(
                "SELECT key FROM decisions WHERE code = ? ORDER BY key LIMIT ?",
                (ground_code, batch_size),
            )
        else:
            cursor = db.execute(
                "SELECT key FROM decisions WHERE code = ? AND key > ? ORDER BY key LIMIT ?",
                (ground_code, last_key, batch_size),
            )
        try:
            rows = cursor.fetchmany(batch_size)
        finally:
            cursor.close()
        if not rows:
            break
        checkpoint()
        last_key = rows[-1][0]
        ground_keys = keys_from_rows(rows)
        alternative_codes, _ = second_best(ground_keys, with_confidence=False)
        alternative_stop = alternative_start+len(ground_keys)
        sorted_keys[alternative_start:alternative_stop] = ground_keys
        sorted_codes[alternative_start:alternative_stop] = alternative_codes.astype(np.uint16, copy=False)
        alternative_start = alternative_stop
    if alternative_start != alternative_count:
        raise RuntimeError('Ground alternative scan count changed during DTM processing')

    db.execute("DROP TABLE IF EXISTS temp.dtm_block_keys")
    db.execute("DROP TABLE IF EXISTS temp.dtm_block_updates")

    outline_dict = {}
    if outline_applied:
        shape = outline_shape
        inside_barrier = np.zeros(shape, dtype=bool)
        if inside_grid is not None:
            inside_barrier |= inside_grid
        if barrier_grid is not None:
            inside_barrier |= barrier_grid
        outline_dict = {
            'outline_applied': True,
            'outline_inside_barrier': inside_barrier,
            'outline_origin': lo,
            'outline_cells': cells,
            'outline_cell_size_m': float(outline.get('cell_size_m', cells * voxel_size)),
            'outline_voxel_size': float(voxel_size),
            'outline_shape': np.asarray(shape, dtype=np.int64),
        }
    else:
        outline_dict = {
            'outline_applied': False,
            'outline_inside_barrier': np.empty((0, 0), dtype=bool),
            'outline_origin': np.zeros(2, dtype=np.int64),
            'outline_cells': 1,
            'outline_cell_size_m': float(voxel_size),
            'outline_voxel_size': float(voxel_size),
            'outline_shape': np.zeros(2, dtype=np.int64),
        }

    try:
        np.savez_compressed(
            alternatives_path,
            keys=sorted_keys,
            codes=sorted_codes,
            origin=origin,
            voxel_size=voxel_size,
            **outline_dict,
        )
    finally:
        for values in (sorted_keys, sorted_codes):
            if isinstance(values, np.memmap):
                values.flush()
                values._mmap.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)

    return {
        'enabled': True,
        'band': band,
        'ground_to': ground_to,
        'to_ground': to_ground,
        'duration_s': float(round(time.perf_counter() - started, 3)),
        'ground_alternatives': str(alternatives_path),
        'alternatives': str(alternatives_path),
    }


class TinDTM:
    """Triangulated Irregular Network Digital Terrain Model."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._archive = np.load(self.path, allow_pickle=False)
        self.meta = json.loads(str(self._archive['metadata']))
        if 'xyz' in self._archive:
            self.nodes = np.asarray(self._archive['xyz'], dtype=np.float64)
        elif 'nodes' in self._archive:
            self.nodes = np.asarray(self._archive['nodes'], dtype=np.float64)
        elif 'xy' in self._archive and 'z' in self._archive:
            self.nodes = np.column_stack((self._archive['xy'], self._archive['z'])).astype(np.float64)
        else:
            self.nodes = np.empty((0, 3), dtype=np.float64)

        self.triangles = (
            np.asarray(self._archive['triangles'], dtype=np.int64)
            if 'triangles' in self._archive
            else np.empty((0, 3), dtype=np.int64)
        )
        self.max_gap = float(self.meta.get('terrain_max_gap', self.meta.get('max_gap', 5.0)))

        self.tri = None
        self.valid_simplex = None
        if len(self.nodes) >= 3:
            try:
                self.tri = Delaunay(self.nodes[:, :2])
                s = self.tri.simplices
                p = self.nodes[:, :2]
                e01 = np.hypot(p[s[:, 1], 0] - p[s[:, 0], 0], p[s[:, 1], 1] - p[s[:, 0], 1])
                e12 = np.hypot(p[s[:, 2], 0] - p[s[:, 1], 0], p[s[:, 2], 1] - p[s[:, 1], 1])
                e20 = np.hypot(p[s[:, 0], 0] - p[s[:, 2], 0], p[s[:, 0], 1] - p[s[:, 2], 1])
                max_edge = np.maximum(np.maximum(e01, e12), e20)
                self.valid_simplex = max_edge <= self.max_gap
            except (QhullError, ValueError):
                self.tri = None
                self.valid_simplex = None

    def height(self, xy: Any) -> np.ndarray:
        """Interpolate DTM elevation for query points, NaN outside valid triangles."""
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        count = len(pts)
        result = np.full(count, np.nan, dtype=np.float64)
        if count == 0 or self.tri is None or self.valid_simplex is None:
            return result

        q_xy = pts[:, :2]
        simplex_ids = self.tri.find_simplex(q_xy)

        in_hull = simplex_ids >= 0
        valid_query = np.zeros(count, dtype=bool)
        valid_query[in_hull] = self.valid_simplex[simplex_ids[in_hull]]

        if not valid_query.any():
            return result

        chosen_simplices = simplex_ids[valid_query]
        tris = self.tri.simplices[chosen_simplices]

        p1 = self.nodes[tris[:, 0]]
        p2 = self.nodes[tris[:, 1]]
        p3 = self.nodes[tris[:, 2]]

        qx = q_xy[valid_query, 0]
        qy = q_xy[valid_query, 1]

        x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]
        x2, y2, z2 = p2[:, 0], p2[:, 1], p2[:, 2]
        x3, y3, z3 = p3[:, 0], p3[:, 1], p3[:, 2]

        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        non_degen = np.abs(det) > 1e-12

        if non_degen.any():
            det_sub = det[non_degen]
            qx_sub = qx[non_degen]
            qy_sub = qy[non_degen]
            x1_s, y1_s, z1_s = x1[non_degen], y1[non_degen], z1[non_degen]
            x2_s, y2_s, z2_s = x2[non_degen], y2[non_degen], z2[non_degen]
            x3_s, y3_s, z3_s = x3[non_degen], y3[non_degen], z3[non_degen]

            w1 = ((y2_s - y3_s) * (qx_sub - x3_s) + (x3_s - x2_s) * (qy_sub - y3_s)) / det_sub
            w2 = ((y3_s - y1_s) * (qx_sub - x3_s) + (x1_s - x3_s) * (qy_sub - y3_s)) / det_sub
            w3 = 1.0 - w1 - w2
            interpolated = w1 * z1_s + w2 * z2_s + w3 * z3_s

            valid_indices = np.flatnonzero(valid_query)
            result[valid_indices[non_degen]] = interpolated

        return result

    def close(self) -> None:
        """Release underlying archive resources."""
        if hasattr(self, '_archive') and self._archive is not None:
            self._archive.close()
            self._archive = None

    def __enter__(self) -> TinDTM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
