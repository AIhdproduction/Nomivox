"""Compact, deterministic plane completion data used by both output writers."""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from e57cls.geometry.planes import fit_plane_batched, fit_plane_robust, refine_plane


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


def _split_cells(cells, connectivity=1):
    """Split occupancy cells using a bounded Chebyshev connectivity radius."""
    cells = np.unique(np.asarray(cells, np.int32), axis=0)
    if len(cells) < 2:
        return [cells]
    connectivity = max(1, int(connectivity))
    values = _keys(cells)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    source, target = [], []
    offsets = [(dx, dy) for dx in range(connectivity+1) for dy in range(-connectivity, connectivity+1)
               if (dx, dy) != (0, 0) and not (dx == 0 and dy < 0)]
    for offset in offsets:
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


def _plane_points(votes, code, origin, voxel_size, checkpoint, return_indices=False):
    """Read sorted decision voxels, optionally retaining their SQL indices.

    ``decisions.key`` already stores the exact integer voxel indices.  The
    prefilter can use them directly and therefore avoids a costly and less
    exact floating-point round-trip through world coordinates.
    """
    rows = votes.db.execute('SELECT key FROM decisions WHERE code=? ORDER BY key', (int(code),))
    parts = []
    while batch := rows.fetchmany(250_000):
        checkpoint()
        indices = np.frombuffer(b''.join(row[0] for row in batch), dtype='<i8').reshape(-1, 3)
        parts.append(indices)
    indices = np.concatenate(parts) if parts else np.empty((0, 3), np.int64)
    if return_indices:
        return indices
    return origin+(indices+.5)*voxel_size


def _packed_cell_keys(cells):
    """Pack non-negative local three-dimensional cell coordinates into int64.

    ``None`` signals that an unusually wide input needs the exact legacy
    structured-key fallback instead of risking an overflow or collision.
    """
    cells = np.asarray(cells, np.int64)
    low = cells.min(axis=0)
    local = cells-low
    spans = local.max(axis=0)
    bits = np.ceil(np.log2(spans.astype(np.float64)+1.)).astype(np.int64)
    bits[spans == 0] = 0
    if int(bits.sum()) > 62:
        return None
    shift_yz = int(bits[1]+bits[2])
    shift_z = int(bits[2])
    return (local[:, 0] << shift_yz) | (local[:, 1] << shift_z) | local[:, 2]


def _pack_cells_with_layout(cells, low, spans, bits):
    """Pack coordinates against a fixed component layout and flag outliers."""
    local = np.asarray(cells, np.int64)-low
    valid = np.all((local >= 0) & (local <= spans), axis=1)
    shift_yz = int(bits[1]+bits[2])
    shift_z = int(bits[2])
    keys = (local[:, 0] << shift_yz) | (local[:, 1] << shift_z) | local[:, 2]
    return keys, valid


def _unique_cells_packed(cells):
    """Return lexicographically ordered unique cells and an inverse cheaply."""
    cells = np.asarray(cells, np.int64)
    keys = _packed_cell_keys(cells)
    if keys is None:
        return np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    groups = np.cumsum(first, dtype=np.int64)-1
    inverse = np.empty(len(cells), np.int64)
    inverse[order] = groups
    return cells[order[first]], inverse


def _component_labels(cells):
    """Return sparse 26-neighbour component labels for unique integer cells."""
    cells = np.asarray(cells, np.int64)
    packed = _packed_cell_keys(cells)
    if packed is None:
        cells = np.unique(cells, axis=0)
        values = np.ascontiguousarray(cells, dtype='<i8').view([('x', '<i8'), ('y', '<i8'), ('z', '<i8')]).reshape(-1)
        order = np.argsort(values, kind='stable')
        sorted_values = values[order]
        packed_mode = False
    else:
        order = np.argsort(packed, kind='stable')
        packed = packed[order]
        unique = np.r_[True, packed[1:] != packed[:-1]]
        cells = cells[order[unique]]
        low = cells.min(axis=0)
        spans = cells.max(axis=0)-low
        bits = np.ceil(np.log2(spans.astype(np.float64)+1.)).astype(np.int64)
        bits[spans == 0] = 0
        values, _ = _pack_cells_with_layout(cells, low, spans, bits)
        order = np.arange(len(cells), dtype=np.int64)
        sorted_values = values
        packed_mode = True
    source, target = [], []
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if ((dx, dy, dz) == (0, 0, 0) or (dx == 0 and dy < 0)
                        or (dx == 0 and dy == 0 and dz < 0)):
                    continue
                shifted_cells = cells+np.array((dx, dy, dz), np.int64)
                if packed_mode:
                    shifted, valid = _pack_cells_with_layout(shifted_cells, low, spans, bits)
                else:
                    shifted = np.ascontiguousarray(shifted_cells, dtype='<i8').view(values.dtype).reshape(-1)
                    valid = np.ones(len(shifted), bool)
                found = np.searchsorted(sorted_values, shifted)
                present = valid & (found < len(sorted_values))
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


def _component_labels_legacy(cells):
    """Use the historical structured-key implementation for overflow inputs."""
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


def _without_small_islands(points, origin, voxel_size, minimum_voxels, voxel_indices=None, return_mask=False):
    """Discard small spatial components before they can seed a plane fit.

    Dense components keep the original voxel-level filtering.  Only a coarse
    component made entirely of sparse fragments is evaluated on 10-cm cells.
    This retains thin, regularly scanned surfaces without letting nearby noise
    attach to dense real surfaces and perturb their plane fit.
    """
    if minimum_voxels <= 1 or len(points) < minimum_voxels:
        return np.ones(len(points), bool) if return_mask else points
    indices = (np.asarray(voxel_indices, np.int64) if voxel_indices is not None
               else np.rint((points-origin)/voxel_size-.5).astype(np.int64))
    fine_cells, fine_inverse = _unique_cells_packed(indices)
    fine_count, fine_labels = _component_labels(fine_cells)
    fine_point_labels = fine_labels[fine_inverse]
    fine_sizes = np.bincount(fine_point_labels, minlength=fine_count)
    fine_keep = fine_sizes[fine_point_labels] >= minimum_voxels

    cell_size = max(ISLAND_CELL_SIZE, voxel_size)
    if voxel_indices is not None:
        # For voxel centres, floor((point-origin)/cell_size) is exactly the
        # corresponding integer-index expression; no world float arithmetic.
        coarse_cells = np.floor((indices.astype(np.float64)+.5)*voxel_size/cell_size).astype(np.int64)
        coarse_cells, coarse_inverse = _unique_cells_packed(coarse_cells)
    else:
        coarse_cells, coarse_inverse = _unique_cells_packed(
            np.floor((points-origin)/cell_size).astype(np.int64))
    coarse_count, coarse_labels = _component_labels(coarse_cells)
    coarse_point_labels = coarse_labels[coarse_inverse]
    coarse_sizes = np.bincount(coarse_point_labels, minlength=coarse_count)
    coarse_keep = coarse_sizes[coarse_point_labels] >= minimum_voxels
    coarse_has_dense = np.bincount(coarse_point_labels[fine_keep], minlength=coarse_count).astype(bool)
    keep = fine_keep | (coarse_keep & ~coarse_has_dense[coarse_point_labels])
    return keep if return_mask else points[keep]


def _basis_from_normal(normal):
    """Return a stable orthonormal plane basis for a unit normal."""
    normal = np.asarray(normal, np.float64)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return np.vstack((first, np.cross(normal, first)))


def _merged_plane_fit(indices, pieces, normals, areas):
    """Refit a merged plane from constant-size tile support moments."""
    if not all('fit_count' in pieces[index] for index in indices):
        center = np.average(np.asarray([pieces[index]['center'] for index in indices], np.float64),
                            axis=0, weights=areas[indices])
        normal = np.average(normals[indices], axis=0, weights=areas[indices])
        return center, normal/np.linalg.norm(normal)
    counts = np.asarray([pieces[index]['fit_count'] for index in indices], np.float64)
    centers = np.asarray([pieces[index]['fit_center'] for index in indices], np.float64)
    center = np.average(centers, axis=0, weights=counts)
    scatter = np.zeros((3, 3), np.float64)
    for index, count, local_center in zip(indices, counts, centers):
        delta = local_center-center
        scatter += pieces[index]['fit_scatter']+count*np.outer(delta, delta)
    _, vectors = np.linalg.eigh(scatter)
    normal = vectors[:, 0]
    reference = np.average(normals[indices], axis=0, weights=areas[indices])
    if normal@reference < 0:
        normal = -normal
    return center, normal


def _merge_coplanar_pieces(pieces, tolerance, connectivity=1):
    """Join adjacent occupancy patches without repeatedly growing cell arrays.

    Neighbour candidates come from a quantised normal/offset index.  A union-
    find pass then joins every compatible candidate exactly once; only the
    resulting connected groups allocate a combined occupancy array.
    """
    pieces = list(pieces)
    if len(pieces) < 2:
        return pieces
    count = len(pieces)
    normals = np.asarray([piece['normal'] for piece in pieces], np.float64)
    centers = np.asarray([piece['center'] for piece in pieces], np.float64)
    areas = np.asarray([piece['area'] for piece in pieces], np.float64)
    # A plane has no orientation.  Canonicalising its sign makes normal bins
    # stable across separately fitted tile pieces.
    dominant = np.argmax(np.abs(normals), axis=1)
    signs = np.sign(normals[np.arange(count), dominant])
    signs[signs == 0] = 1
    normals *= signs[:, None]
    # Plane offsets must be measured relative to the local piece cloud.  The
    # world origin can be Swiss-scale, where a tiny normal-fit difference
    # would otherwise turn into a hundred-metre cancellation error.
    offset_origin = np.mean(centers, axis=0)
    offsets = np.einsum('ij,ij->i', centers-offset_origin, normals)
    cosine = np.cos(np.deg2rad(3.))
    normal_step = .03
    offset_step = max(float(tolerance), CELL_SIZE)
    normal_bins = np.floor(normals/normal_step+.5).astype(np.int32)
    offset_bins = np.floor(offsets/offset_step+.5).astype(np.int64)
    parents = np.arange(count, dtype=np.int32)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    # At most 625 nearby normal/offset bins are checked.  This index is per
    # plane piece, never per voxel, and avoids the former O(pieces**2) scan.
    buckets = {}
    neighbour_offsets = np.asarray(np.meshgrid(*([[-2, -1, 0, 1, 2]]*4), indexing='ij')).reshape(4, -1).T
    for index, piece in enumerate(pieces):
        base = (*normal_bins[index].tolist(), int(offset_bins[index]))
        for delta in neighbour_offsets:
            nearby = buckets.get((base[0]+int(delta[0]), base[1]+int(delta[1]),
                                  base[2]+int(delta[2]), base[3]+int(delta[3])), ())
            for other in nearby:
                if abs(float(normals[index]@normals[other])) < cosine:
                    continue
                if abs(float((centers[index]-centers[other])@normals[other])) >= tolerance:
                    continue
                other_piece = pieces[other]
                # Project only four extent corners for the inexpensive
                # adjacency prefilter; exact connectedness is checked below.
                cells = piece['cells']
                low, high = cells.min(axis=0), cells.max(axis=0)
                corners = np.array([[u, v] for u in (low[0], high[0]) for v in (low[1], high[1])], np.float64)
                world = corners*CELL_SIZE@piece['basis'] + piece['normal']*np.dot(piece['center'], piece['normal'])
                projected = _cells_from_uv(world@other_piece['basis'].T)
                projected_low, projected_high = projected.min(axis=0), projected.max(axis=0)
                other_low, other_high = other_piece['cells'].min(axis=0), other_piece['cells'].max(axis=0)
                if np.maximum(projected_low-other_high, other_low-projected_high).max() <= connectivity:
                    union(index, other)
        buckets.setdefault(base, []).append(index)

    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for indices in groups.values():
        # Union-find deliberately discovers transitive neighbours.  A gently
        # curved facade can therefore form one candidate group even though
        # its outer tiles no longer agree with the group's average plane.
        # Split such a group at the first incompatible support piece before
        # generating its occupancy mask.  The four source extent corners are
        # a conservative, constant-size representation of every tile piece.
        pending = [indices]
        accepted = []
        while pending:
            candidate = pending.pop()
            candidate_center, candidate_normal = _merged_plane_fit(candidate, pieces, normals, areas)
            incompatible = []
            for index in candidate:
                piece = pieces[index]
                low, high = piece['cells'].min(axis=0), piece['cells'].max(axis=0)
                corners = np.array([[u, v] for u in (low, high) for v in (low, high)], np.float64)
                world = corners*CELL_SIZE@piece['basis'] + piece['normal']*np.dot(piece['center'], piece['normal'])
                if np.max(np.abs((world-candidate_center)@candidate_normal)) >= tolerance:
                    incompatible.append(index)
            if not incompatible or len(candidate) == 1:
                accepted.append(candidate)
            else:
                keep = [index for index in candidate if index not in set(incompatible)]
                if keep:
                    pending.append(keep)
                pending.extend([[index] for index in incompatible])
        for indices in accepted:
            # Constant-size support moments restore the global least-squares
            # position without retaining source points for every voxel.
            center, normal = _merged_plane_fit(indices, pieces, normals, areas)
            basis = _basis_from_normal(normal)
            projected_parts = []
            for index in indices:
                piece = pieces[index]
                uv = piece['cells'].astype(np.float64)*CELL_SIZE
                world = uv@piece['basis'] + piece['normal']*np.dot(piece['center'], piece['normal'])
                projected_parts.append(_cells_from_uv(world@basis.T))
            cells = np.unique(np.concatenate(projected_parts), axis=0)
            for connected in _split_cells(cells, connectivity):
                merged.append({'center': center, 'normal': normal, 'basis': basis,
                               'cells': connected, 'area': float(len(connected)*CELL_SIZE**2)})
    return merged


def _tile_groups(points, origin, tile_size):
    """Return sorted horizontal core-tile slices for world-space points."""
    tile_ids = np.floor((points[:, :2]-origin[:2])/tile_size).astype(np.int64)
    order = np.lexsort((tile_ids[:, 1], tile_ids[:, 0]))
    sorted_ids = tile_ids[order]
    start = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_ids, axis=0), axis=1))+1]
    stop = np.r_[start[1:], len(order)]
    keys = sorted_ids[start]
    return order, keys, start, stop


def _tile_seed(seed, code, tile_key, iteration):
    """Return a process-independent seed from class, tile and iteration."""
    x, y = (int(item) for item in tile_key)
    mixed = (x*73856093) ^ (y*19349663) ^ (int(iteration)*83492791)
    return int((int(seed)+int(code)*1009+mixed) & 0xffffffff)


def _extract_tile_pieces(points, core, class_id, code, origin, voxel_size, config, tolerance, tile_key, checkpoint):
    """Run the existing bounded RANSAC process for one tile's core and halo."""
    remaining = points
    core_remaining = core.copy()
    pieces, iteration = [], 0
    while len(remaining) >= 12 and core_remaining.any() and iteration < config['geometry_fill_max_planes']:
        checkpoint()
        plane = fit_plane_batched(remaining, tolerance=tolerance,
                                 seed=_tile_seed(config['seed'], code, tile_key, iteration),
                                 trials=config.get('geometry_fill_ransac_trials', 256), min_fraction=.001,
                                 score_sample=config.get('geometry_fill_ransac_score_sample', 25_000),
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
        inliers, core_inliers = remaining[inlier_mask], core_remaining[inlier_mask]
        remaining, core_remaining = remaining[~inlier_mask], core_remaining[~inlier_mask]
        iteration += 1
        # A fitted model with fewer than twelve inliers cannot form a valid
        # plane.  Continuing would consume one iteration per residual noise
        # sample until the tile limit, whereas a subsequent RANSAC run cannot
        # recover a supported plane after its best support was removed.
        if len(inliers) < 12 or not core_inliers.any():
            break
        slope = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
        if ((class_id in {'floor', 'ground', 'road', 'sidewalk'} and slope >= 20)
                or (class_id in {'wall', 'building'} and slope <= 70)):
            continue
        core_points = inliers[core_inliers]
        point_cells = _cells_from_uv(core_points@basis.T)
        cells = np.unique(point_cells, axis=0)
        # Voxel-centre reconstruction can move adjacent samples across two
        # 5-cm occupancy cells.  This bounded radius heals quantisation only;
        # the configured physical gap is still applied later by dilation.
        connectivity = max(1, int(np.ceil(voxel_size/CELL_SIZE))+1)
        for piece in _split_cells(cells, connectivity):
            area = len(piece)*CELL_SIZE**2
            if area >= min(config['geometry_fill_min_area'], .25):
                values = np.sort(_keys(piece))
                point_values = _keys(point_cells)
                found = np.searchsorted(values, point_values)
                owns = found < len(values)
                owns[owns] &= values[found[owns]] == point_values[owns]
                fit_points = core_points[owns]
                fit_center = fit_points.mean(axis=0)
                centered_fit_points = fit_points-fit_center
                pieces.append({'center': center, 'normal': normal, 'basis': basis,
                               'cells': piece, 'area': float(area),
                               'fit_count': int(len(fit_points)), 'fit_center': fit_center,
                               'fit_scatter': centered_fit_points.T@centered_fit_points})
    return pieces, iteration, iteration >= config['geometry_fill_max_planes']


def extract_planes(votes, by_id, origin, voxel_size, config, target: Path, checkpoint=lambda: None):
    """Extract supported plane pieces for every configured geometric class.

    A dominant plane is deliberately conservative: it fills small missing
    islands on a known surface but never extrapolates a semantic surface past
    its dilated occupied cells.  The working artifact stores only occupancy
    cells, not source voxels, and is therefore bounded by visible plane area.
    ``geometry_fill_max_planes`` is a per-core-tile limit, so large connected
    structures no longer consume a class-wide iteration budget.
    """
    origin = np.asarray(origin, np.float64)
    result, arrays, iterations, tiles, limit_hits = {'cell_size': CELL_SIZE, 'planes': []}, {}, {}, {}, {}
    tile_size = float(config.get('geometry_fill_tile_size', 16.))
    tile_overlap = float(config.get('geometry_fill_tile_overlap', .75))
    for class_id in config['geometry_fill_classes']:
        if class_id not in by_id:
            continue
        indices = _plane_points(votes, by_id[class_id], origin, voxel_size, checkpoint, return_indices=True)
        points = origin+(indices+.5)*voxel_size
        keep = _without_small_islands(points, origin, voxel_size,
                                      config.get('geometry_fill_min_component_voxels', 20),
                                      voxel_indices=indices, return_mask=True)
        points = points[keep]
        # Integer SQL keys were needed only by the component prefilter.  Drop
        # them before tiling so a large class does not retain a second 24-byte
        # coordinate array alongside its world-space points.
        del indices, keep
        if len(points) < 12:
            continue
        tolerance = max(config['geometry_fill_tolerance'], voxel_size*1.5)
        pieces = []
        order, keys, starts, stops = _tile_groups(points, origin, tile_size)
        # The dictionary is bounded by horizontal tiles (not source points).
        # Halo candidates are drawn from the at-most nine adjacent core tiles.
        tile_lookup = {tuple(map(int, key)): index for index, key in enumerate(keys)}
        class_iterations = 0
        class_limit_hits = 0
        for tile_index, key in enumerate(keys):
            checkpoint()
            core_indices = order[starts[tile_index]:stops[tile_index]]
            selected = []
            # With overlap smaller than a tile, every halo point is in this
            # 3x3 neighbourhood.  The final coordinate test keeps the stated
            # overlap exact even at non-aligned origins.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = tile_lookup.get((int(key[0])+dx, int(key[1])+dy))
                    if neighbour is not None:
                        selected.append(order[starts[neighbour]:stops[neighbour]])
            candidate_indices = np.concatenate(selected)
            lower = origin[:2]+np.asarray(key, np.float64)*tile_size-tile_overlap
            upper = lower+tile_size+2*tile_overlap
            candidate_indices = candidate_indices[np.all((points[candidate_indices, :2] >= lower)
                                                          & (points[candidate_indices, :2] < upper), axis=1)]
            local_points = points[candidate_indices]
            core = np.all(np.floor((local_points[:, :2]-origin[:2])/tile_size).astype(np.int64) == key, axis=1)
            tile_pieces, tile_iterations, hit_limit = _extract_tile_pieces(
                local_points, core, class_id, by_id[class_id], origin, voxel_size, config, tolerance, key, checkpoint)
            pieces.extend(tile_pieces)
            class_iterations += tile_iterations
            class_limit_hits += int(hit_limit)
        connectivity = max(1, int(np.ceil(voxel_size/CELL_SIZE))+1)
        for piece in _merge_coplanar_pieces(pieces, tolerance, connectivity):
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
        iterations[class_id] = class_iterations
        tiles[class_id] = len(keys)
        limit_hits[class_id] = class_limit_hits
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, metadata=np.array(json.dumps(result)), **arrays)
    return {'artifact': str(target), 'planes': {item: sum(p['class_id'] == item for p in result['planes'])
                                               for item in config['geometry_fill_classes']},
            'iterations': {item: iterations.get(item, 0) for item in config['geometry_fill_classes']},
            'tiles': {item: tiles.get(item, 0) for item in config['geometry_fill_classes']},
            'max_iterations_hit': {item: limit_hits.get(item, 0) for item in config['geometry_fill_classes']}}


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
