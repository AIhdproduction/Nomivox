"""Compact, deterministic plane completion data used by both output writers."""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from e57cls.geometry.planes import fit_plane_robust, refine_plane


CELL_SIZE = .05
CELL_EPSILON = 1e-8
ISLAND_CELL_SIZE = .10


def _keys(cells):
    """Return sortable opaque keys for integer two-dimensional cells."""
    return np.ascontiguousarray(cells, dtype='<i4').view('V8').reshape(-1)


def _cells_from_uv(uv):
    """Quantize coordinates to centres of 5-cm occupancy cells.

    Voxel centres can shift a regularly sampled 5-cm surface by up to one
    centimetre.  Centre-based quantization prevents those samples from
    alternating across adjacent cell boundaries and fragmenting one patch.
    """
    return np.floor(np.asarray(uv)/CELL_SIZE+.5+CELL_EPSILON).astype(np.int32)


def _dilate(cells, radius):
    if radius <= 0 or not len(cells):
        return np.unique(cells, axis=0)
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = high-low+1
    # Dense dilation is faster for ordinary rooms.  For very large extents the
    # sparse fallback has a fixed, small neighbourhood and never allocates a
    # building-sized image.
    if int(np.prod(shape+2*radius, dtype=np.int64)) <= 20_000_000:
        image = np.zeros(shape+2*radius, bool)
        local = cells-low+radius
        image[local[:, 0], local[:, 1]] = True
        image = binary_dilation(image, iterations=radius)
        return np.argwhere(image)+low-radius
    offsets = np.array([(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1)], np.int32)
    expanded = np.concatenate([cells[start:start+20_000, None, :]+offsets for start in range(0, len(cells), 20_000)]).reshape(-1, 2)
    return np.unique(expanded, axis=0)


def _split_cells(cells):
    """Split occupied 2-D cells with eight-neighbour connectivity."""
    cells = np.unique(np.asarray(cells, np.int32), axis=0)
    if len(cells) < 2:
        return [cells]
    values = _keys(cells)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    source, target = [], []
    for offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
        wanted = _keys(cells+np.asarray(offset, np.int32))
        found = np.searchsorted(sorted_values, wanted)
        present = found < len(sorted_values)
        present[present] &= sorted_values[found[present]] == wanted[present]
        ids = np.flatnonzero(present)
        if len(ids):
            source.append(ids)
            target.append(order[found[ids]])
    if not source:
        return [cells[index:index+1] for index in range(len(cells))]
    graph = coo_matrix((np.ones(sum(map(len, source)), np.uint8),
                       (np.concatenate(source), np.concatenate(target))), shape=(len(cells), len(cells)))
    count, labels = connected_components(graph, directed=False, return_labels=True)
    order = np.argsort(labels, kind='stable')
    offsets = np.cumsum(np.bincount(labels, minlength=count))[:-1]
    return list(np.split(cells[order], offsets))


def _piece_bounds(cells, center, normal, basis, tolerance):
    low, high = (cells.min(axis=0)-.5)*CELL_SIZE, (cells.max(axis=0)+.5)*CELL_SIZE
    uv = np.array([[u, v] for u in (low[0], high[0]) for v in (low[1], high[1])])
    # ``basis`` spans the plane while the normal component fixes its position
    # in world space.  Dilated cells already include the in-plane gap margin.
    world = uv@basis + normal*np.dot(center, normal)
    return (world.min(axis=0)-tolerance).tolist(), (world.max(axis=0)+tolerance).tolist()


def _plane_points(votes, code, origin, voxel_size, checkpoint):
    rows = votes.db.execute('SELECT key FROM decisions WHERE code=? ORDER BY key', (int(code),))
    parts = []
    while batch := rows.fetchmany(250_000):
        checkpoint()
        indices = np.frombuffer(b''.join(row[0] for row in batch), dtype='<i8').reshape(-1, 3)
        parts.append(origin+(indices+.5)*voxel_size)
    return np.concatenate(parts) if parts else np.empty((0, 3), np.float64)


def _component_labels(cells):
    """Return sparse 26-neighbour component labels for unique integer cells."""
    cells = np.unique(np.asarray(cells, np.int64), axis=0)
    values = np.ascontiguousarray(cells, dtype='<i8').view([('x', '<i8'), ('y', '<i8'), ('z', '<i8')]).reshape(-1)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    source, target = [], []
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if ((dx, dy, dz) == (0, 0, 0) or (dx == 0 and dy < 0)
                        or (dx == 0 and dy == 0 and dz < 0)):
                    continue
                shifted = np.ascontiguousarray(cells+np.array((dx, dy, dz), np.int64), dtype='<i8').view(values.dtype).reshape(-1)
                found = np.searchsorted(sorted_values, shifted)
                present = found < len(sorted_values)
                present[present] &= sorted_values[found[present]] == shifted[present]
                found_ids = np.flatnonzero(present)
                if len(found_ids):
                    source.append(found_ids)
                    target.append(order[found[found_ids]])
    if source:
        graph = coo_matrix((np.ones(sum(map(len, source)), np.uint8),
                           (np.concatenate(source), np.concatenate(target))), shape=(len(cells), len(cells)))
    else:
        graph = coo_matrix((len(cells), len(cells)), dtype=np.uint8)
    return connected_components(graph, directed=False, return_labels=True)


def _without_small_islands(points, origin, voxel_size, minimum_voxels):
    """Discard small spatial components before they can seed a plane fit.

    Dense components keep the original voxel-level filtering.  Only a coarse
    component made entirely of sparse fragments is evaluated on 10-cm cells.
    This retains thin, regularly scanned surfaces without letting nearby noise
    attach to dense real surfaces and perturb their plane fit.
    """
    if minimum_voxels <= 1 or len(points) < minimum_voxels:
        return points
    fine_cells, fine_inverse = np.unique(
        np.rint((points-origin)/voxel_size-.5).astype(np.int64), axis=0, return_inverse=True)
    fine_count, fine_labels = _component_labels(fine_cells)
    fine_point_labels = fine_labels[fine_inverse]
    fine_sizes = np.bincount(fine_point_labels, minlength=fine_count)
    fine_keep = fine_sizes[fine_point_labels] >= minimum_voxels

    cell_size = max(ISLAND_CELL_SIZE, voxel_size)
    coarse_cells, coarse_inverse = np.unique(
        np.floor((points-origin)/cell_size).astype(np.int64), axis=0, return_inverse=True)
    coarse_count, coarse_labels = _component_labels(coarse_cells)
    coarse_point_labels = coarse_labels[coarse_inverse]
    coarse_sizes = np.bincount(coarse_point_labels, minlength=coarse_count)
    coarse_keep = coarse_sizes[coarse_point_labels] >= minimum_voxels
    coarse_has_dense = np.bincount(coarse_point_labels[fine_keep], minlength=coarse_count).astype(bool)
    keep = fine_keep | (coarse_keep & ~coarse_has_dense[coarse_point_labels])
    return points[keep]


def _merge_coplanar_pieces(pieces, tolerance):
    """Join adjacent occupancy patches when their supporting planes agree."""
    pieces = list(pieces)
    merged = []
    normals = np.empty((len(pieces), 3), np.float64)
    centers = np.empty((len(pieces), 3), np.float64)
    lows = np.empty((len(pieces), 2), np.int32)
    highs = np.empty((len(pieces), 2), np.int32)
    cosine = np.cos(np.deg2rad(3.))
    for piece in pieces:
        if merged:
            size = len(merged)
            current_normals = normals[:size]
            normal_values = np.abs(current_normals@piece['normal'])
            normal_matches = ~(normal_values < cosine-1e-12)
            deltas = piece['center']-centers[:size]
            offsets = np.abs(np.einsum('ij,ij->i', deltas, current_normals))
            rounding_margin = np.maximum(1e-10, np.finfo(np.float64).eps*64*np.sum(np.abs(deltas*current_normals), axis=1))
            candidate_indices = np.flatnonzero(normal_matches & ~(offsets >= tolerance+rounding_margin))
        else:
            candidate_indices = ()
        for index in candidate_indices:
            current = merged[index]
            if abs(float(np.dot(piece['normal'], current['normal']))) < cosine:
                continue
            if abs(float(np.dot(piece['center']-current['center'], current['normal']))) >= tolerance:
                continue
            uv = piece['cells'].astype(np.float64)*CELL_SIZE
            world = uv@piece['basis'] + piece['normal']*np.dot(piece['center'], piece['normal'])
            projected = _cells_from_uv(world@current['basis'].T)
            projected_low, projected_high = projected.min(axis=0), projected.max(axis=0)
            if np.maximum(projected_low-highs[index], lows[index]-projected_high).max() > 1:
                continue
            combined = np.unique(np.vstack((current['cells'], projected)), axis=0)
            if len(_split_cells(combined)) == 1:
                current['cells'] = combined
                current['area'] += piece['area']
                lows[index] = np.minimum(lows[index], projected_low)
                highs[index] = np.maximum(highs[index], projected_high)
                break
        else:
            index = len(merged)
            merged.append(piece)
            lows[index] = piece['cells'].min(axis=0)
            highs[index] = piece['cells'].max(axis=0)
            normals[index] = piece['normal']
            centers[index] = piece['center']
    return merged


def extract_planes(votes, by_id, origin, voxel_size, config, target: Path, checkpoint=lambda: None):
    """Extract supported plane pieces for every configured geometric class.

    A dominant plane is deliberately conservative: it fills small missing
    islands on a known surface but never extrapolates a semantic surface past
    its dilated occupied cells.  The working artifact stores only occupancy
    cells, not source voxels, and is therefore bounded by visible plane area.
    """
    origin = np.asarray(origin, np.float64)
    result, arrays, iterations = {'cell_size': CELL_SIZE, 'planes': []}, {}, {}
    for class_id in config['geometry_fill_classes']:
        if class_id not in by_id:
            continue
        remaining = _plane_points(votes, by_id[class_id], origin, voxel_size, checkpoint)
        remaining = _without_small_islands(remaining, origin, voxel_size,
                                           config.get('geometry_fill_min_component_voxels', 20))
        if len(remaining) < 12:
            continue
        iteration = 0
        tolerance = max(config['geometry_fill_tolerance'], voxel_size*1.5)
        pieces = []
        while len(remaining) >= 12 and iteration < config['geometry_fill_max_planes']:
            checkpoint()
            # RANSAC scores a reproducible bounded sample; support and the
            # occupancy mask are always evaluated on every remaining voxel.
            plane = fit_plane_robust(remaining, tolerance=tolerance,
                                    seed=config['seed']+int(by_id[class_id])*1009+iteration,
                                    trials=config.get('geometry_fill_ransac_trials', 256), min_fraction=.001,
                                    sample_limit=200_000,
                                    max_trials=config.get('geometry_fill_ransac_max_trials', 1024))
            if not plane:
                break
            normal = np.asarray(plane['normal'], np.float64)
            center, basis = np.asarray(plane['center']), np.asarray(plane['basis'])
            inlier_mask = np.abs((remaining-center)@normal) <= tolerance
            refined = refine_plane(remaining, inlier_mask, tolerance)
            if refined is None:
                break
            normal = np.asarray(refined['normal'], np.float64)
            center, basis = np.asarray(refined['center']), np.asarray(refined['basis'])
            inlier_mask = refined['inliers']
            inliers, remaining = remaining[inlier_mask], remaining[~inlier_mask]
            iteration += 1
            if len(inliers) < 12:
                break
            slope = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
            if ((class_id in {'floor', 'ground', 'road', 'sidewalk'} and slope >= 20)
                    or (class_id in {'wall', 'building'} and slope <= 70)):
                continue
            cells = np.unique(_cells_from_uv(inliers@basis.T), axis=0)
            for piece in _split_cells(cells):
                checkpoint()
                # The minimum area deliberately applies before dilation, so a
                # sparse accident cannot turn into a large artificial surface.
                area = len(piece)*CELL_SIZE**2
                # Small voxel islands have already been rejected before
                # RANSAC.  A quarter square metre is still a substantial
                # supported patch at the 5-cm occupancy resolution, while
                # retaining narrow, real wall fragments between large planes.
                minimum_area = min(config['geometry_fill_min_area'], .25)
                if area < minimum_area:
                    continue
                pieces.append({'center': center, 'normal': normal, 'basis': basis, 'cells': piece, 'area': float(area)})
        for piece in _merge_coplanar_pieces(pieces, tolerance):
            # A cell centre may be half a cell inside the visible boundary.
            # Keep the configured physical gap after this quantization error
            # instead of making identical planes depend on the cell phase.
            dilation_radius = int(np.ceil((config['geometry_fill_max_gap']+CELL_SIZE*.5)/CELL_SIZE))
            dilated = _dilate(piece['cells'], dilation_radius)
            key = f'cells_{len(result["planes"])}'
            arrays[key] = dilated
            lower, upper = _piece_bounds(dilated, piece['center'], piece['normal'], piece['basis'],
                                         config['geometry_fill_tolerance'])
            result['planes'].append({'class_id': class_id, 'code': int(by_id[class_id]),
                                     'center': piece['center'].tolist(), 'normal': piece['normal'].tolist(),
                                     'basis': piece['basis'].tolist(), 'cells': key, 'bbox_min': lower,
                                     'bbox_max': upper, 'area_m2': piece['area']})
        iterations[class_id] = iteration
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, metadata=np.array(json.dumps(result)), **arrays)
    return {'artifact': str(target), 'planes': {item: sum(p['class_id'] == item for p in result['planes'])
                                               for item in config['geometry_fill_classes']},
            'iterations': {item: iterations.get(item, 0) for item in config['geometry_fill_classes']}}


class PlaneIndex:
    """Read-only plane matcher; a point must be near a plane and inside its mask."""
    def __init__(self, path):
        self._archive = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self._archive['metadata']))
        self.planes = []
        self.grid_size = 1.
        self.grid = {}
        for item in self.meta['planes']:
            cells = self._archive[item['cells']]
            entry = (item, np.asarray(item['center']), np.asarray(item['normal']),
                     np.asarray(item['basis']), np.sort(_keys(cells)),
                     np.asarray(item.get('bbox_min', [-np.inf]*3)), np.asarray(item.get('bbox_max', [np.inf]*3)))
            index = len(self.planes)
            self.planes.append(entry)
            bbox_min, bbox_max = entry[-2:]
            if np.isfinite(bbox_min).all() and np.isfinite(bbox_max).all():
                low = np.floor(bbox_min/self.grid_size).astype(np.int64)
                high = np.floor(bbox_max/self.grid_size).astype(np.int64)
                for cell in product(*(range(int(a), int(b)+1) for a, b in zip(low, high))):
                    self.grid.setdefault(cell, []).append(index)

    def fill(self, points, tolerance, checkpoint=lambda: None):
        codes = np.zeros(len(points), np.uint8)
        best = np.full(len(points), np.inf)
        tied = np.zeros(len(points), bool)
        if not len(points):
            return codes
        cells = np.floor(points/self.grid_size).astype(np.int64)
        candidates = set()
        for cell in np.unique(cells, axis=0):
            candidates.update(self.grid.get(tuple(map(int, cell)), ()))
        for position, plane_index in enumerate(sorted(candidates)):
            if position % 32 == 0:
                checkpoint()
            item, center, normal, basis, occupied, bbox_min, bbox_max = self.planes[plane_index]
            in_box = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
            if not in_box.any():
                continue
            block_ids = np.flatnonzero(in_box)
            distance = np.abs((points[block_ids]-center)@normal)
            eligible = distance <= tolerance
            if not eligible.any():
                continue
            ids = block_ids[eligible]
            distance = distance[eligible]
            cells = _cells_from_uv(points[ids]@basis.T)
            key = _keys(cells)
            found = np.searchsorted(occupied, key)
            inside = found < len(occupied)
            inside[inside] &= occupied[found[inside]] == key[inside]
            ids = ids[inside]
            distances = distance[inside]
            closer = distances+0.005 < best[ids]
            same = np.abs(distances-best[ids]) < .005
            if np.any(closer):
                chosen = ids[closer]
                best[chosen], codes[chosen], tied[chosen] = distances[closer], item['code'], False
            conflict = same & (codes[ids] != item['code'])
            tied[ids[conflict]] = True
        codes[tied] = 0
        return codes

    def close(self):
        self._archive.close()
