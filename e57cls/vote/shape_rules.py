"""Voxel-level indoor shape corrections built on segment-vote structure.

Segments find large structural planes cheaply.  The separation between a
structural surface and objects is done on the original voxel grid, so thin
carpets and flush wall items do not disappear inside 10 cm feature cells.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from e57cls.geometry.planes import refine_plane
from e57cls.vote.geometry_rules import prepare_geometry_features, refresh_geometry_decisions
from e57cls.vote.scene_context import ensure_scene_context
from e57cls.vote.segments import _linear_keys


RULE_NAMES = (
    'carpet', 'column', 'pilaster', 'cabinet', 'beam',
    'pipe', 'hvac_duct', 'radiator', 'cable', 'ceiling_equipment', 'sanitary', 'railing',
    'wall_item', 'furniture_clutter',
    'curb', 'outdoor_wall', 'utility_pole', 'traffic_sign', 'fence', 'vehicle',
    'power_line', 'outdoor_building', 'facade_side', 'tree_crown',
)

# Object selection below is deliberately limited to the indoor rules.  The
# outdoor rules operate on their own evidence and target tables later in this
# module, so they must not be looked up in an indoor source map.
OBJECT_RULE_NAMES = (
    'carpet', 'column', 'pilaster', 'cabinet', 'beam',
    'pipe', 'hvac_duct', 'radiator', 'cable', 'ceiling_equipment', 'sanitary', 'railing',
    'wall_item', 'furniture_clutter',
)
OUTDOOR_RULE_NAMES = tuple(name for name in RULE_NAMES if name not in OBJECT_RULE_NAMES)

if len(set(RULE_NAMES)) != len(RULE_NAMES) or set(OBJECT_RULE_NAMES) | set(OUTDOOR_RULE_NAMES) != set(RULE_NAMES):
    raise RuntimeError('shape rule names must form one unique indoor/outdoor partition')


def _code(by_id, *names):
    """Return the first available code; historic presets use a few aliases."""
    for name in names:
        if name in by_id:
            return int(by_id[name])
    return None


def _keys_from_rows(rows, key_index=0):
    """Return owning V24 keys from one SQLite result block."""
    return np.frombuffer(b''.join(row[key_index] for row in rows), dtype='V24').copy()


def _label_group_bounds(labels):
    """Return stable source indices and component boundaries for integer labels."""
    order = np.argsort(labels, kind='stable')
    ordered = labels[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1]
    return order, starts, np.r_[starts[1:], len(order)]


def _component_labels(cells):
    """Return deterministic 26-connected labels for sorted integer cells."""
    if not len(cells):
        return np.empty(0, np.int64)
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = high-low+1
    keys = _linear_keys(cells, low, shape)
    if not np.all(keys[1:] >= keys[:-1]):
        raise ValueError('Object grid cells must be sorted before component labelling')
    source, target = [], []
    for dx in range(2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if not (dx > 0 or (dx == 0 and dy > 0) or (dx == dy == 0 and dz > 0)):
                    continue
                shifted = cells+np.asarray((dx, dy, dz))
                # _linear_keys is collision-free only inside the enclosing
                # grid.  A neighbour outside a one-cell-wide axis otherwise
                # wraps onto a valid key and can join unrelated storeys.
                bounded = np.all((shifted >= low) & (shifted <= high), axis=1)
                candidate = _linear_keys(shifted, low, shape)
                found = np.searchsorted(keys, candidate)
                valid = bounded & (found < len(keys))
                valid[valid] &= keys[found[valid]] == candidate[valid]
                if valid.any():
                    source.append(np.flatnonzero(valid))
                    target.append(found[valid])
    if source:
        source, target = np.concatenate(source), np.concatenate(target)
        graph = coo_matrix((np.ones(2*len(source), np.uint8),
                            (np.r_[source, target], np.r_[target, source])), shape=(len(cells), len(cells)))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(cells), dtype=np.int64)
    unique, inverse = np.unique(labels, return_inverse=True)
    minimum = np.full(len(unique), np.iinfo(np.int64).max)
    np.minimum.at(minimum, inverse, keys)
    remap = np.empty(len(unique), np.int64)
    remap[np.argsort(minimum, kind='stable')] = np.arange(len(unique))
    return remap[inverse]


def _padded_levels(grid, values, cell):
    """Compact ragged z levels into a nearest-raster lookup."""
    if not len(values):
        return None
    unique, inverse = np.unique(grid, axis=0, return_inverse=True)
    order = np.lexsort((values, inverse))
    grouped = inverse[order]
    starts = np.r_[0, np.flatnonzero(grouped[1:] != grouped[:-1])+1]
    counts = np.diff(np.r_[starts, len(order)])
    matrix = np.full((len(unique), int(counts.max())), np.nan)
    ranks = np.arange(len(order))-np.repeat(starts, counts)
    matrix[grouped, ranks] = values[order]
    return unique, matrix, cKDTree((unique+.5)*cell)


def _level_at(points, level, search, floor):
    """Look up local floor or ceiling levels without a point loop."""
    result = np.full(len(points), np.nan)
    if level is None or not len(points):
        return result
    _, matrix, tree = level
    distance, selected = tree.query(points[:, :2], distance_upper_bound=float(search), workers=-1)
    valid = np.isfinite(distance) & (selected < len(matrix))
    if not valid.any():
        return result
    choices = matrix[selected[valid]]
    z = points[valid, 2, None]
    choices = np.where(choices <= z+.05, choices, np.nan) if floor else np.where(choices >= z, choices, np.nan)
    present = np.isfinite(choices).any(axis=1)
    rows = np.flatnonzero(valid)[present]
    result[rows] = np.nanmax(choices[present], axis=1) if floor else np.nanmin(choices[present], axis=1)
    return result


def _plane_level_at(points, planes, kind, search, floor):
    """Evaluate local floor/ceiling heights directly from RANSAC planes.

    Ceiling levels deliberately accept roof planes up to 60 degrees.  The
    projected point is checked against the plane's occupied mask, so a nearby
    roof plane cannot leak through a gable or beyond its visible extent.
    """
    result = np.full(len(points), np.nan)
    if not len(points):
        return result
    if kind == 2:
        allowed = (planes['kinds'] == 2) & (np.abs(planes['normal'][:, 2]) >= np.cos(np.deg2rad(10.)))
    else:
        allowed = (planes['kinds'] == 3) & (np.abs(planes['normal'][:, 2]) >= .5)
    if not allowed.any():
        return result
    tile_size = planes['tile_size']
    tiles = np.floor(points[:, :2]/tile_size).astype(np.int64)
    unique, inverse = np.unique(tiles, axis=0, return_inverse=True)
    global_ids = planes['global_ids']
    for group, tile in enumerate(unique):
        source = np.flatnonzero(inverse == group)
        # Include a small XY neighbourhood so local levels can be found just
        # outside a tile edge without reverting to a global plane scan.
        candidate_parts = []
        radius = int(np.ceil(float(search)/tile_size))
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                value = planes['level_tile_grid'].get((int(tile[0]+dx), int(tile[1]+dy)))
                if value is not None:
                    candidate_parts.append(value)
        if len(global_ids):
            candidate_parts.append(global_ids)
        if not candidate_parts:
            continue
        candidates = np.unique(np.concatenate(candidate_parts))
        candidates = candidates[allowed[candidates]]
        if not len(candidates):
            continue
        xy = points[source, :2]
        for plane in candidates:
            normal = planes['normal'][plane]
            z = planes['center'][plane, 2]-(xy-planes['center'][plane, :2])@normal[:2]/normal[2]
            projected = np.column_stack((xy, z))
            cells = _cells_from_uv(projected@planes['basis'][plane].T)
            keys, occupied = _keys(cells), planes['occupied'][plane]
            found = np.searchsorted(occupied, keys)
            inside = found < len(occupied)
            inside[inside] &= occupied[found[inside]] == keys[inside]
            if not inside.any():
                continue
            close = np.linalg.norm(xy-planes['center'][plane, :2], axis=1) <= float(search)+.5
            take = inside & close & ((z <= points[source, 2]+.05) if floor else (z >= points[source, 2]))
            if floor:
                result[source[take]] = np.fmax(result[source[take]], z[take])
            else:
                previous = result[source[take]]
                result[source[take]] = np.where(np.isfinite(previous), np.minimum(previous, z[take]), z[take])
    return result


def _load_cell_arrays(db, checkpoint, chunk_rows=100_000):
    """Load per-cell state used by the local floor/ceiling lookup."""
    count = int(db.execute('SELECT COUNT(*) FROM segment_cells').fetchone()[0])
    if not count:
        return None
    cells = np.empty((count, 3), np.int64)
    segments = np.empty(count, np.int64)
    points = np.empty((count, 3), np.float64)
    cursor = db.execute('''SELECT sc.cx,sc.cy,sc.cz,sc.segment,f.x,f.y,f.z
                           FROM segment_cells sc JOIN geometry_features f USING(cx,cy,cz)
                           ORDER BY sc.cx,sc.cy,sc.cz''')
    start = 0
    while rows := cursor.fetchmany(chunk_rows):
        checkpoint()
        stop = start+len(rows)
        batch = np.asarray(rows, np.float64)
        cells[start:stop] = batch[:, :3]
        segments[start:stop] = batch[:, 3].astype(np.int64)
        points[start:stop] = batch[:, 4:7]
        start = stop
    return cells, segments, points


def _segment_arrays(db):
    """Return dense segment metadata indexed by segment id."""
    rows = db.execute('''SELECT segment,cells,voxels,x,y,z,nx,ny,nz,min_x,min_y,min_z,max_x,max_y,max_z,winner,share
                         FROM segments ORDER BY segment''').fetchall()
    if not rows:
        return None
    maximum = int(rows[-1][0])+1
    values = np.zeros((maximum, 16), np.float64)
    ids = np.asarray([row[0] for row in rows], np.int64)
    values[ids] = np.asarray([row[1:] for row in rows], np.float64)
    normals = values[:, 5:8]
    length = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, length[:, None], out=np.zeros_like(normals), where=length[:, None] > 1e-12)
    return {'cells': values[:, 0].astype(np.int64), 'center': values[:, 2:5], 'normal': normals,
            'lower': values[:, 8:11], 'upper': values[:, 11:14], 'winner': values[:, 14].astype(np.int64),
            'share': values[:, 15]}


def _demote_parallel_wall_fronts(structural, winners, wall, segment_data, offset):
    """Keep a smaller wall-like front in front of a real wall as an object.

    Segment voting can legitimately give an inset cabinet front the wall code.
    Treating both parallel slabs as structure would make that cabinet invisible
    before voxel object formation starts.
    """
    if wall is None:
        return structural
    wall_ids = np.flatnonzero(structural & (winners == wall) &
                              (np.abs(segment_data['normal'][:, 2]) <= .2))
    if len(wall_ids) < 2:
        return structural
    tree = cKDTree(segment_data['center'][wall_ids])
    _, nearby = tree.query(segment_data['center'][wall_ids], k=min(16, len(wall_ids)), workers=-1)
    nearby = np.asarray(nearby).reshape(len(wall_ids), -1)
    candidates = wall_ids[nearby]
    normal = segment_data['normal'][candidates]
    delta = segment_data['center'][wall_ids, None, :]-segment_data['center'][candidates]
    distance = np.abs(np.einsum('nij,nij->ni', delta, normal))
    parallel = np.abs(np.einsum('nij,nj->ni', normal, segment_data['normal'][wall_ids])) >= np.cos(np.deg2rad(10.))
    first_lower, first_upper = segment_data['lower'][wall_ids, None, :2], segment_data['upper'][wall_ids, None, :2]
    second_lower, second_upper = segment_data['lower'][candidates, :2], segment_data['upper'][candidates, :2]
    overlap = np.all((first_lower <= second_upper) & (second_lower <= first_upper), axis=2)
    larger = segment_data['cells'][candidates] > segment_data['cells'][wall_ids, None]
    front = (parallel & overlap & larger & (distance >= float(offset[0])) & (distance <= float(offset[1])))
    result = structural.copy()
    result[wall_ids[np.any(front, axis=1)]] = False
    return result


def _create_tables(db):
    """Create TEMP state reusable by subsequent shape-rule rounds."""
    db.executescript('''
        DROP TABLE IF EXISTS temp.shape_object_segments;
        DROP TABLE IF EXISTS temp.shape_targets;
        DROP TABLE IF EXISTS temp.shape_structural_cells;
        DROP TABLE IF EXISTS temp.shape_structural_segments;
        DROP TABLE IF EXISTS temp.shape_structure_voxels;
        DROP TABLE IF EXISTS temp.shape_structure_grid;
        DROP TABLE IF EXISTS temp.shape_orphan_grid;
        DROP TABLE IF EXISTS temp.shape_compact_orphan_cells;
        DROP TABLE IF EXISTS temp.shape_object_candidates;
        DROP TABLE IF EXISTS temp.shape_object_grid;
        DROP TABLE IF EXISTS temp.shape_object_contacts;
        DROP TABLE IF EXISTS temp.shape_object_voxels;
        DROP TABLE IF EXISTS temp.shape_objects;
        DROP TABLE IF EXISTS temp.shape_facade_voxels;
        CREATE TEMP TABLE shape_structural_segments(segment INTEGER PRIMARY KEY,kind INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE shape_structure_voxels(key BLOB,segment INTEGER NOT NULL,kind INTEGER NOT NULL,distance REAL NOT NULL,
            PRIMARY KEY(key,segment)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_facade_voxels(key BLOB PRIMARY KEY) WITHOUT ROWID;
        CREATE TEMP TABLE shape_object_candidates(key BLOB PRIMARY KEY) WITHOUT ROWID;
        CREATE TEMP TABLE shape_object_grid(cx INTEGER,cy INTEGER,cz INTEGER,object INTEGER,
            PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_structure_grid(cx INTEGER,cy INTEGER,cz INTEGER,kind INTEGER,segment INTEGER,
            PRIMARY KEY(cx,cy,cz,kind,segment)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_orphan_grid(cx INTEGER,cy INTEGER,cz INTEGER,
            PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_compact_orphan_cells(cx INTEGER,cy INTEGER,cz INTEGER,
            PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_object_contacts(object INTEGER,kind INTEGER,segment INTEGER,
            PRIMARY KEY(object,kind,segment)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_object_voxels(key BLOB PRIMARY KEY,object INTEGER NOT NULL,
            gx INTEGER NOT NULL,gy INTEGER NOT NULL,gz INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE shape_objects(object INTEGER PRIMARY KEY,cells INTEGER,voxels INTEGER,x REAL,y REAL,z REAL,
            l1 REAL,l2 REAL,l3 REAL,axis_x REAL,axis_y REAL,axis_z REAL,min_z REAL,max_z REAL,winner INTEGER,share REAL,
            touches_wall INTEGER,touches_floor INTEGER,touches_ceiling INTEGER) WITHOUT ROWID;
    ''')


def _scanner_positions(scan_context):
    positions = []
    for item in (scan_context or {}).values():
        position = np.asarray(item.get('position'), np.float64) if item.get('position') is not None else None
        if position is not None and position.shape == (3,) and np.isfinite(position).all():
            positions.append(position)
    return np.asarray(positions, np.float64) if positions else np.empty((0, 3), np.float64)


def _plane_basis(normal):
    seed = np.array((0., 0., 1.)) if abs(normal[2]) < .9 else np.array((1., 0., 0.))
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    return np.vstack((first, np.cross(normal, first)))


def _keys(cells):
    """Return sortable opaque keys for integer two-dimensional cells."""
    return np.ascontiguousarray(cells, dtype='<i4').view('V8').reshape(-1)


def _cells_from_uv(uv):
    """Quantize coordinates to centres of 5-cm occupancy cells."""
    return np.floor(np.asarray(uv)/.05+.5+1e-8).astype(np.int32)


def _plane_tile_index(planes, margin, tile_size=2.):
    """Index finite plane bounds in coarse world-space tiles.

    The layer matcher is called for every voxel, whereas the number of planes
    is normally in the hundreds.  A two-metre raster keeps each matcher pass
    local without allocating a dense building-sized volume.
    """
    grid, level_grid, global_ids = {}, {}, []
    margin, tile_size = float(margin), float(tile_size)
    for index, (lower, upper) in enumerate(zip(planes['bbox_min'], planes['bbox_max'])):
        if not planes['valid'][index]:
            continue
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            global_ids.append(index)
            continue
        low = np.floor((lower-margin)/tile_size).astype(np.int64)
        high = np.floor((upper+margin)/tile_size).astype(np.int64)
        # A malformed plane must not turn the spatial index into a huge dense
        # dictionary.  It remains correct through the small global fallback.
        if np.prod(high-low+1, dtype=np.int64) > 100_000:
            global_ids.append(index)
            continue
        for x in range(int(low[0]), int(high[0])+1):
            for y in range(int(low[1]), int(high[1])+1):
                level_grid.setdefault((x, y), []).append(index)
                for z in range(int(low[2]), int(high[2])+1):
                    grid.setdefault((x, y, z), []).append(index)
    planes['tile_size'] = tile_size
    planes['tile_grid'] = {key: np.asarray(value, np.int64) for key, value in grid.items()}
    planes['level_tile_grid'] = {key: np.asarray(value, np.int64) for key, value in level_grid.items()}
    planes['global_ids'] = np.asarray(global_ids, np.int64)


def _plane_tile_groups(points, planes):
    """Yield point rows and the small set of planes in their spatial tile."""
    if not len(points):
        return
    tiles = np.floor(np.asarray(points)/planes['tile_size']).astype(np.int64)
    order = np.lexsort((tiles[:, 2], tiles[:, 1], tiles[:, 0]))
    sorted_tiles = tiles[order]
    starts = np.r_[0, np.flatnonzero(np.any(sorted_tiles[1:] != sorted_tiles[:-1], axis=1))+1, len(order)]
    globals_ = planes['global_ids']
    for first, last in zip(starts[:-1], starts[1:]):
        tile, source = sorted_tiles[first], order[first:last]
        candidates = planes['tile_grid'].get(tuple(map(int, tile)), np.empty(0, np.int64))
        if len(globals_):
            candidates = np.r_[candidates, globals_]
        if len(candidates):
            yield source, candidates


def _plane_hits(points, plane, planes, distance_limit):
    """Return source rows and signed distances inside one plane's visible mask."""
    lower = planes['bbox_min'][plane]-float(distance_limit)
    upper = planes['bbox_max'][plane]+float(distance_limit)
    in_box = np.all((points >= lower) & (points <= upper), axis=1)
    if not in_box.any():
        return np.empty(0, np.int64), np.empty(0, np.float64)
    source = np.flatnonzero(in_box)
    delta = points[source]-planes['center'][plane]
    distance = delta@planes['normal'][plane]
    close = np.abs(distance) <= float(distance_limit)
    if not close.any():
        return np.empty(0, np.int64), np.empty(0, np.float64)
    source, distance = source[close], distance[close]
    cells = _cells_from_uv(points[source]@planes['basis'][plane].T)
    keys, occupied = _keys(cells), planes['occupied'][plane]
    found = np.searchsorted(occupied, keys)
    visible = found < len(occupied)
    visible[visible] &= occupied[found[visible]] == keys[visible]
    return source[visible], distance[visible]


def _plane_voxel_cursor(db, planes, columns, margin):
    """Return voxels in the finite envelope of all active RANSAC planes.

    Plane matching still performs the exact masked test below.  This coarse
    feature-cell envelope only avoids loading unrelated raw voxels, which is
    decisive for a sparse plane beside a very large scan.
    """
    valid = planes['valid']
    lower, upper = planes['bbox_min'][valid], planes['bbox_max'][valid]
    if not len(lower) or not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        return db.execute(f'SELECT {columns} FROM geometry_voxels g')
    lower, upper = lower.min(axis=0)-float(margin), upper.max(axis=0)+float(margin)
    return db.execute(f'''SELECT {columns} FROM geometry_features f
                          CROSS JOIN geometry_voxels g ON (g.cx,g.cy,g.cz)=(f.cx,f.cy,f.cz)
                          WHERE f.x BETWEEN ? AND ? AND f.y BETWEEN ? AND ? AND f.z BETWEEN ? AND ?''',
                      (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1]),
                       float(lower[2]), float(upper[2])))


def _estimate_plane_bands(db, planes, origin, voxel_size, flush, sigma_factor, max_band, checkpoint):
    """Estimate robust per-plane noise from masked voxel distances in +/-10 cm."""
    _plane_tile_index(planes, .10)
    sample_limit, samples = 8192, [[] for _ in range(planes['count'])]
    sample_counts = np.zeros(planes['count'], np.int64)
    cursor = _plane_voxel_cursor(db, planes, 'g.ix,g.iy,g.iz', .25)
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        indices = np.asarray(rows, np.float64)
        points = origin+(indices+.5)*float(voxel_size)
        for source, candidates in _plane_tile_groups(points, planes):
            local_points = points[source]
            for plane in candidates:
                _, distance = _plane_hits(local_points, int(plane), planes, .10)
                remaining = sample_limit-sample_counts[plane]
                if remaining > 0 and len(distance):
                    values = distance[:remaining].copy()
                    samples[plane].append(values)
                    sample_counts[plane] += len(values)
    sigma = np.zeros(planes['count'], np.float64)
    for plane, values in enumerate(samples):
        if not values:
            continue
        distances = np.concatenate(values)
        median = np.median(distances)
        sigma[plane] = 1.4826*np.median(np.abs(distances-median))
    planes['sigma'] = sigma
    planes['band'] = np.minimum(float(max_band), np.maximum(float(flush), float(sigma_factor)*sigma))
    _plane_tile_index(planes, max(float(max_band), .10))


def _load_structural_planes_from_npz(planes_path, config, scan_context=None):
    """Load and filter RANSAC structural planes from planes.npz."""
    try:
        archive = np.load(planes_path, allow_pickle=False)
    except Exception:
        return None
    try:
        meta = json.loads(str(archive['metadata']))
    except Exception:
        archive.close()
        return None
    cell_size = float(meta.get('cell_size', .05))
    scanners = _scanner_positions(scan_context)
    structural_classes = {'wall', 'building', 'ceiling', 'floor', 'ground', 'road', 'sidewalk'}
    min_width = float(config['shape_structural_width'])
    min_area = float(config['shape_structural_area'])

    centers, normals, bases = [], [], []
    lowers, uppers, occupied_list = [], [], []
    kinds, bboxes_min, bboxes_max, areas = [], [], [], []
    class_ids = []

    for item in meta.get('planes', []):
        class_id = item.get('class_id')
        if class_id not in structural_classes:
            continue
        normal = np.asarray(item['normal'], np.float64)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal /= norm
        area = float(item.get('area_m2', 0.))
        cells_key = item.get('cells')
        if not cells_key or cells_key not in archive:
            continue
        cells = archive[cells_key]
        if not len(cells):
            continue

        center = np.asarray(item['center'], np.float64)
        basis = np.asarray(item['basis'], np.float64)
        low = (cells.min(axis=0)-.5)*cell_size
        high = (cells.max(axis=0)+.5)*cell_size

        abs_nz = abs(normal[2])
        if abs_nz <= .2:
            h = np.array([-normal[1], normal[0], 0.], np.float64)
            h_norm = np.linalg.norm(h)
            if h_norm > 1e-12:
                h /= h_norm
            uv_corners = np.array([[u, v] for u in (low[0], high[0]) for v in (low[1], high[1])], np.float64)
            world_corners = uv_corners@basis+normal*np.dot(center, normal)
            horiz_width = float((world_corners@h).max()-(world_corners@h).min())
            if horiz_width <= min_width:
                continue
        else:
            if area < min_area:
                continue

        # Ceiling planes can form a roof, but a surface steeper than 60
        # degrees is a wall according to the rule catalogue.
        if class_id == 'ceiling' and abs_nz < .5:
            continue

        if class_id in {'wall', 'building'}:
            kind = 1
        elif class_id in {'floor', 'ground', 'road', 'sidewalk'}:
            kind = 2
        elif class_id == 'ceiling':
            kind = 3
        else:
            continue

        if kind == 2 and normal[2] < 0:
            normal = -normal
        elif kind == 3 and normal[2] > 0:
            normal = -normal
        elif kind == 1:
            if len(scanners):
                nearest = scanners[np.argmin(np.sum((scanners-center)**2, axis=1))]
                if np.dot(nearest-center, normal) < 0:
                    normal = -normal
            elif normal[2] < 0 or (normal[2] == 0 and (normal[0] < 0 or (normal[0] == 0 and normal[1] < 0))):
                normal = -normal

        proj_center = np.dot(center, basis.T)
        lower = low-proj_center-.11
        upper = high-proj_center+.11

        bbox_min = np.asarray(item.get('bbox_min', [-np.inf, -np.inf, -np.inf]), np.float64)
        bbox_max = np.asarray(item.get('bbox_max', [np.inf, np.inf, np.inf]), np.float64)
        occ = np.sort(_keys(cells))

        centers.append(center)
        normals.append(normal)
        bases.append(basis)
        lowers.append(lower)
        uppers.append(upper)
        occupied_list.append(occ)
        areas.append(area)
        kinds.append(kind)
        bboxes_min.append(bbox_min)
        bboxes_max.append(bbox_max)
        class_ids.append(class_id)

    archive.close()
    count = len(centers)
    if not count:
        return None

    result = {
        'count': count,
        'center': np.asarray(centers, np.float64),
        'normal': np.asarray(normals, np.float64),
        'basis': np.asarray(bases, np.float64),
        'lower': np.asarray(lowers, np.float64),
        'upper': np.asarray(uppers, np.float64),
        'valid': np.ones(count, bool),
        'kinds': np.asarray(kinds, np.int8),
        'occupied': occupied_list,
        'area': np.asarray(areas, np.float64),
        'bbox_min': np.asarray(bboxes_min, np.float64),
        'bbox_max': np.asarray(bboxes_max, np.float64),
        'class_id': class_ids,
    }
    _demote_parallel_ransac_wall_fronts(result, config['shape_cabinet_offset'])
    return result


def _demote_parallel_ransac_wall_fronts(planes, offset):
    """Reject a smaller wall plane in front of a larger parallel wall.

    RANSAC sees a full-height cabinet front just as clearly as a room wall.
    The segment fallback already keeps this pair out of the structural layer;
    applying the equivalent test here keeps both paths semantically aligned.
    """
    wall_ids = np.flatnonzero((planes['kinds'] == 1) & planes['valid'])
    if len(wall_ids) < 2:
        return
    centers = planes['center'][wall_ids]
    normals = planes['normal'][wall_ids]
    local = np.einsum('ijk,ik->ij', centers[None, :, :]-centers[:, None, :], normals)
    distance = np.abs(local)
    parallel = np.abs(normals@normals.T) >= np.cos(np.deg2rad(10.))
    areas = np.asarray([len(planes['occupied'][index]) for index in wall_ids], np.float64)
    larger = areas[None, :] > areas[:, None]*1.10
    # The centre of the potential front must lie over the larger wall's
    # observed extent.  The 11-cm padding matches the plane-mask allowance.
    delta = centers[None, :, :]-centers[:, None, :]
    projected = np.einsum('ijk,ilk->ijl', delta, planes['basis'][wall_ids])
    lower, upper = planes['lower'][wall_ids], planes['upper'][wall_ids]
    within = np.all((projected >= lower[:, None, :]) & (projected <= upper[:, None, :]), axis=2)
    front = (parallel & larger & within & (distance >= float(offset[0])) & (distance <= float(offset[1])))
    planes['valid'][wall_ids[np.any(front, axis=1)]] = False


def _materialize_structure_layer_ransac(db, planes, voxel_size, origin, protected_codes, checkpoint):
    """Materialize the masked RANSAC structure layer through a tile index."""
    count = planes['count']
    db.executemany('INSERT INTO shape_structural_segments VALUES (?,?)',
                   ((int(i), int(planes['kinds'][i])) for i in np.flatnonzero(planes['valid'])))
    voxel_size = float(voxel_size)
    cursor = _plane_voxel_cursor(db, planes, 'g.key,g.ix,g.iy,g.iz,g.code', .25)
    pending_layer = []
    immutable = np.asarray(sorted(protected_codes), np.int64) if protected_codes else np.empty(0, np.int64)

    while rows := cursor.fetchmany(100_000):
        checkpoint()
        keys = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
        indices = np.asarray([row[1:4] for row in rows], np.float64)
        codes = np.asarray([row[4] for row in rows], np.int64)
        points = origin+(indices+.5)*voxel_size
        for source, candidates in _plane_tile_groups(points, planes):
            local_points = points[source]
            for p_idx in candidates:
                if not planes['valid'][p_idx]:
                    continue
                local_ids, distance = _plane_hits(local_points, int(p_idx), planes, planes['band'][p_idx])
                if not len(local_ids):
                    continue
                hit_ids = source[local_ids]
                if len(immutable):
                    valid_mask = ~np.isin(codes[hit_ids], immutable)
                    hit_ids, distance = hit_ids[valid_mask], distance[valid_mask]
                if len(hit_ids):
                    kind = int(planes['kinds'][p_idx])
                    pending_layer.extend(
                        (bytes(key), int(p_idx), kind, float(d))
                        for key, d in zip(keys[hit_ids], distance)
                    )
        if len(pending_layer) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_structure_voxels VALUES (?,?,?,?)', pending_layer)
            pending_layer.clear()

    if pending_layer:
        db.executemany('INSERT OR IGNORE INTO shape_structure_voxels VALUES (?,?,?,?)', pending_layer)
    db.commit()


def _fit_structural_planes(db, structural, kinds, segment_data, origin, voxel_size, flush, scan_context, checkpoint):
    """Refit structural segment planes from voxels and orient their room side."""
    count = len(structural)
    centers = np.zeros((count, 3), np.float64)
    normals = np.zeros((count, 3), np.float64)
    bases = np.zeros((count, 2, 3), np.float64)
    lower = np.zeros((count, 2), np.float64)
    upper = np.zeros((count, 2), np.float64)
    valid = np.zeros(count, bool)
    ids = np.flatnonzero(structural)
    if not len(ids):
        return {'center': centers, 'normal': normals, 'basis': bases, 'lower': lower, 'upper': upper, 'valid': valid,
                'band': np.full(count, float(flush), np.float64)}
    db.executemany('INSERT INTO shape_structural_segments VALUES (?,?)',
                   ((int(segment), int(kinds[segment])) for segment in ids))
    scanners = _scanner_positions(scan_context)
    tolerance = max(float(flush), float(voxel_size)*1.5)
    cursor = db.execute('''SELECT ss.segment,g.ix,g.iy,g.iz FROM shape_structural_segments ss
                           JOIN segment_cells sc ON sc.segment=ss.segment
                           JOIN geometry_voxels g USING(cx,cy,cz)
                           ORDER BY ss.segment,g.key''')
    carry = []

    def finish(rows):
        segment = int(rows[0][0])
        indices = np.asarray([row[1:] for row in rows], np.float64)
        points = origin+(indices+.5)*float(voxel_size)
        normal = segment_data['normal'][segment].copy()
        center = segment_data['center'][segment].copy()
        if len(points) >= 12 and np.linalg.norm(normal) > .5:
            inliers = np.abs((points-center)@normal) <= tolerance
            refined = refine_plane(points, inliers, tolerance) if inliers.sum() >= 12 else None
            if refined is not None:
                center, normal = refined['center'], refined['normal']
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            return
        normal /= norm
        kind = int(kinds[segment])
        if kind == 2 and normal[2] < 0:
            normal = -normal
        elif kind == 3 and normal[2] > 0:
            normal = -normal
        elif kind not in (2, 3):
            if len(scanners):
                nearest = scanners[np.argmin(np.sum((scanners-center)**2, axis=1))]
                if np.dot(nearest-center, normal) < 0:
                    normal = -normal
            elif normal[2] < 0 or (normal[2] == 0 and (normal[0] < 0 or (normal[0] == 0 and normal[1] < 0))):
                normal = -normal
        basis = _plane_basis(normal)
        projected = (points-center)@basis.T
        centers[segment], normals[segment], bases[segment] = center, normal, basis
        lower[segment] = projected.min(axis=0)-.11
        upper[segment] = projected.max(axis=0)+.11
        valid[segment] = True

    while rows := cursor.fetchmany(100_000):
        checkpoint()
        rows = carry+rows
        last, split = rows[-1][0], len(rows)
        while split and rows[split-1][0] == last:
            split -= 1
        if split:
            groups = np.asarray([row[0] for row in rows[:split]], np.int64)
            starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1])+1, split]
            for first, second in zip(starts[:-1], starts[1:]):
                finish(rows[int(first):int(second)])
        carry = rows[split:]
    if carry:
        finish(carry)
    return {'center': centers, 'normal': normals, 'basis': bases, 'lower': lower, 'upper': upper, 'valid': valid,
            'band': np.full(count, float(flush), np.float64)}


def _materialize_structure_layer(db, planes, kinds, voxel_size, flush, origin, checkpoint):
    """Classify structural cells and their 26-neighbour halo at voxel level."""
    db.executescript('''
        CREATE TEMP TABLE shape_structural_cells(cx INTEGER,cy INTEGER,cz INTEGER,segment INTEGER,
            PRIMARY KEY(cx,cy,cz,segment)) WITHOUT ROWID;
        WITH offsets(dx,dy,dz) AS (VALUES
            (-1,-1,-1),(-1,-1,0),(-1,-1,1),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,-1),(-1,1,0),(-1,1,1),
            (0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1),(0,1,-1),(0,1,0),(0,1,1),
            (1,-1,-1),(1,-1,0),(1,-1,1),(1,0,-1),(1,0,0),(1,0,1),(1,1,-1),(1,1,0),(1,1,1))
        INSERT OR IGNORE INTO shape_structural_cells
        SELECT gc.cx,gc.cy,gc.cz,ss.segment FROM shape_structural_segments ss
        JOIN segment_cells sc ON sc.segment=ss.segment CROSS JOIN offsets o
        JOIN geometry_cells gc ON gc.cx=sc.cx+o.dx AND gc.cy=sc.cy+o.dy AND gc.cz=sc.cz+o.dz;
    ''')
    cursor = db.execute('''SELECT sc.segment,g.key,g.ix,g.iy,g.iz FROM shape_structural_cells sc
                           JOIN geometry_voxels g USING(cx,cy,cz) ORDER BY sc.segment,g.key''')
    pending_layer = []
    centers, normals, bases = planes['center'], planes['normal'], planes['basis']
    lower, upper, valid = planes['lower'], planes['upper'], planes['valid']
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        segments = np.asarray([row[0] for row in rows], np.int64)
        keys = np.asarray([bytes(row[1]) for row in rows], dtype='V24')
        indices = np.asarray([row[2:] for row in rows], np.float64)
        points = origin+(indices+.5)*float(voxel_size)
        delta = points-centers[segments]
        distance = np.einsum('ij,ij->i', delta, normals[segments])
        projected = np.einsum('nij,nj->ni', bases[segments], delta)
        inside = valid[segments] & np.all((projected >= lower[segments]) & (projected <= upper[segments]), axis=1)
        layer = inside & (np.abs(distance) <= float(flush))
        if layer.any():
            pending_layer.extend((bytes(key), int(segment), int(kind), float(value))
                                 for key, segment, kind, value in zip(keys[layer], segments[layer], kinds[segments[layer]], distance[layer]))
        if len(pending_layer) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_structure_voxels VALUES (?,?,?,?)', pending_layer)
            pending_layer.clear()
    if pending_layer:
        db.executemany('INSERT OR IGNORE INTO shape_structure_voxels VALUES (?,?,?,?)', pending_layer)
    db.commit()


def _grid_coordinates(indices, origin, voxel_size, object_cell):
    return np.floor((origin+(indices.astype(np.float64)+.5)*float(voxel_size))/float(object_cell)).astype(np.int64)


def _materialize_object_candidates(db, structural_codes, origin, voxel_size, object_cell, max_size, checkpoint):
    """Keep generic objects and only compact orphaned structural components.

    A wall/floor/ceiling voxel outside every accepted RANSAC plane must not
    bridge furniture across an entire house.  It is admitted only when its
    *own* 20-cm connected component is compact.  The coarser guard absorbs
    SLAM holes while staying far cheaper than forming raw-voxel components.
    """
    codes = np.asarray(sorted({int(code) for code in structural_codes if code is not None}), np.int64)
    has_structure = db.execute('SELECT 1 FROM shape_structure_voxels LIMIT 1').fetchone() is not None
    if not len(codes):
        db.execute('''INSERT OR IGNORE INTO shape_object_candidates(key)
                      SELECT g.key FROM geometry_voxels g JOIN geometry_context gc USING(key)
                      WHERE gc.scene_context=0
                        {}'''.format('AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)'
                                     if has_structure else ''))
        db.commit()
        return
    placeholders = ','.join('?' for _ in codes)
    db.execute(f'''INSERT OR IGNORE INTO shape_object_candidates(key)
                   SELECT g.key FROM geometry_voxels g JOIN geometry_context gc USING(key)
                   WHERE gc.scene_context=0 AND g.code NOT IN ({placeholders})
                     {'AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)' if has_structure else ''}''',
               tuple(map(int, codes)))
    orphan_exists = db.execute(f'''SELECT 1 FROM geometry_voxels g JOIN geometry_context gc USING(key)
                                   WHERE gc.scene_context=0 AND g.code IN ({placeholders})
                                     {'AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)' if has_structure else ''}
                                   LIMIT 1''', tuple(map(int, codes))).fetchone() is not None
    if not orphan_exists:
        db.commit()
        return
    orphan_query = f'''SELECT g.cx,g.cy,g.cz,MIN(g.ix),MIN(g.iy),MIN(g.iz) FROM geometry_voxels g
                       JOIN geometry_context gc USING(key)
                       WHERE gc.scene_context=0 AND g.code IN ({placeholders})
                         AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)
                       GROUP BY g.cx,g.cy,g.cz ORDER BY g.cx,g.cy,g.cz'''
    guard_cell = max(.20, float(object_cell)*5.)
    cursor = db.execute(orphan_query, tuple(map(int, codes)))
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        grid = _grid_coordinates(np.asarray([row[3:] for row in rows], np.int64), origin, voxel_size, guard_cell)
        pending.extend(tuple(map(int, cell)) for cell in grid)
        if len(pending) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_orphan_grid VALUES (?,?,?)', pending)
            pending.clear()
    if pending:
        db.executemany('INSERT OR IGNORE INTO shape_orphan_grid VALUES (?,?,?)', pending)
    rows = db.execute('SELECT cx,cy,cz FROM shape_orphan_grid ORDER BY cx,cy,cz').fetchall()
    if not rows:
        db.commit()
        return
    cells = np.asarray(rows, np.int64)
    labels = _component_labels(cells)
    count = int(labels.max()+1)
    lower = np.full((count, 3), np.iinfo(np.int64).max)
    upper = np.full((count, 3), np.iinfo(np.int64).min)
    for axis in range(3):
        np.minimum.at(lower[:, axis], labels, cells[:, axis])
        np.maximum.at(upper[:, axis], labels, cells[:, axis])
    compact = np.max((upper-lower+1)*guard_cell, axis=1) <= float(max_size)+1e-9
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = high-low+1
    cell_keys = _linear_keys(cells, low, shape)
    cursor = db.execute(orphan_query, tuple(map(int, codes)))
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        feature_cells = np.asarray([row[:3] for row in rows], np.int64)
        grid = _grid_coordinates(np.asarray([row[3:] for row in rows], np.int64), origin, voxel_size, guard_cell)
        found = np.searchsorted(cell_keys, _linear_keys(grid, low, shape))
        keep = compact[labels[found]]
        pending.extend(tuple(map(int, cell)) for cell in feature_cells[keep])
        if len(pending) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_compact_orphan_cells VALUES (?,?,?)', pending)
            pending.clear()
    if pending:
        db.executemany('INSERT OR IGNORE INTO shape_compact_orphan_cells VALUES (?,?,?)', pending)
    db.execute(f'''INSERT OR IGNORE INTO shape_object_candidates(key)
                   SELECT g.key FROM geometry_voxels g JOIN shape_compact_orphan_cells c USING(cx,cy,cz)
                   JOIN geometry_context gc USING(key)
                   WHERE gc.scene_context=0 AND g.code IN ({placeholders})
                     {'AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)' if has_structure else ''}''',
               tuple(map(int, codes)))
    db.commit()


def _build_object_grid(db, origin, voxel_size, object_cell, checkpoint):
    """Map candidates to a coarser object raster and label its components.

    The raw voxel-to-bin mapping is materialized exactly once in SQLite and
    then reused for component labels and all later object operations.
    """
    db.execute('DROP TABLE IF EXISTS temp.shape_object_cells')
    db.execute('''CREATE TEMP TABLE shape_object_cells(cx INTEGER,cy INTEGER,cz INTEGER,
                  PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID''')
    grid_sql = ','.join(
        f'CAST(floor((?+(g.{axis}+.5)*?)/?) AS INTEGER)'
        for axis in ('ix', 'iy', 'iz')
    )
    grid_params = tuple(value for axis in range(3) for value in
                        (float(origin[axis]), float(voxel_size), float(object_cell)))
    db.execute(f'''INSERT INTO shape_object_voxels(key,object,gx,gy,gz)
                   SELECT c.key,-1,{grid_sql} FROM shape_object_candidates c
                   JOIN geometry_voxels g USING(key)''', grid_params)
    db.execute('''INSERT OR IGNORE INTO shape_object_cells
                  SELECT gx,gy,gz FROM shape_object_voxels''')
    rows = db.execute('SELECT cx,cy,cz FROM shape_object_cells ORDER BY cx,cy,cz').fetchall()
    if not rows:
        return None
    cells = np.asarray(rows, np.int64)
    labels = _component_labels(cells)
    db.executemany('INSERT INTO shape_object_grid VALUES (?,?,?,?)',
                   ((*map(int, cell), int(label)) for cell, label in zip(cells, labels)))
    db.execute('''UPDATE shape_object_voxels AS ov
                  SET object=(SELECT og.object FROM shape_object_grid og
                              WHERE og.cx=ov.gx AND og.cy=ov.gy AND og.cz=ov.gz)''')
    db.execute('CREATE INDEX shape_object_voxels_object ON shape_object_voxels(object,key)')
    db.commit()
    return {'count': int(labels.max()+1), 'grid_counts': np.bincount(labels), 'object_cell': float(object_cell)}


def _object_geometry(db, grid, origin, voxel_size, checkpoint):
    """Calculate voxel PCA from SQLite aggregate moments plus the object grid."""
    count = grid['count']
    counts, sums = np.zeros(count, np.int64), np.zeros((count, 3), np.float64)
    lower_index, upper_index = np.zeros((count, 3), np.int64), np.zeros((count, 3), np.int64)
    second_moment = np.zeros((count, 3, 3), np.float64)
    rows = db.execute('''SELECT ov.object,COUNT(*),
                         SUM(g.ix*1.0),SUM(g.iy*1.0),SUM(g.iz*1.0),
                         MIN(g.ix),MIN(g.iy),MIN(g.iz),MAX(g.ix),MAX(g.iy),MAX(g.iz),
                         SUM(g.ix*g.ix*1.0),SUM(g.ix*g.iy*1.0),SUM(g.ix*g.iz*1.0),
                         SUM(g.iy*g.iy*1.0),SUM(g.iy*g.iz*1.0),SUM(g.iz*g.iz*1.0)
                         FROM shape_object_voxels ov JOIN geometry_voxels g USING(key)
                         GROUP BY ov.object ORDER BY ov.object''').fetchall()
    for row in rows:
        object_id = int(row[0])
        counts[object_id] = int(row[1])
        sums[object_id] = row[2:5]
        lower_index[object_id], upper_index[object_id] = row[5:8], row[8:11]
        second_moment[object_id] = ((row[11], row[12], row[13]),
                                    (row[12], row[14], row[15]),
                                    (row[13], row[15], row[16]))
    center_index = sums/np.maximum(counts[:, None], 1)
    covariance = (second_moment/np.maximum(counts, 1)[:, None, None]-
                  np.einsum('ni,nj->nij', center_index, center_index))*float(voxel_size)**2
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, :, ::-1]
    center = origin+(center_index+.5)*float(voxel_size)
    lower = origin+(lower_index+.5)*float(voxel_size)
    upper = origin+(upper_index+.5)*float(voxel_size)
    projected_lower, projected_upper = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    cursor = db.execute('SELECT object,cx,cy,cz FROM shape_object_grid ORDER BY object,cx,cy,cz')
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        objects = np.asarray([row[0] for row in rows], np.int64)
        points = (np.asarray([row[1:] for row in rows], np.float64)+.5)*float(grid['object_cell'])
        local = points-center[objects]
        projected = np.einsum('ni,nij->nj', local, axes[objects])
        for axis in range(3):
            np.minimum.at(projected_lower[:, axis], objects, projected[:, axis])
            np.maximum.at(projected_upper[:, axis], objects, projected[:, axis])
    return {'count': count, 'cells': grid['grid_counts'], 'voxels': counts, 'center': center, 'lower': lower, 'upper': upper,
            'extent': projected_upper-projected_lower, 'axis': axes[:, :, 0], 'axes': axes}


def _materialize_contacts(db, origin, voxel_size, object_cell, count, checkpoint):
    """Record contacts through adjacent object-grid cells and structural layers."""
    cursor = db.execute('''SELECT sv.kind,sv.segment,g.ix,g.iy,g.iz FROM shape_structure_voxels sv
                           JOIN geometry_voxels g USING(key) ORDER BY sv.key,sv.segment''')
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        grid = _grid_coordinates(np.asarray([row[2:] for row in rows], np.int64), origin, voxel_size, object_cell)
        pending.extend((*map(int, cell), int(row[0]), int(row[1])) for cell, row in zip(grid, rows))
        if len(pending) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_structure_grid VALUES (?,?,?,?,?)', pending)
            pending.clear()
    if pending:
        db.executemany('INSERT OR IGNORE INTO shape_structure_grid VALUES (?,?,?,?,?)', pending)
    db.executescript('''
        WITH offsets(dx,dy,dz) AS (VALUES
            (-1,-1,-1),(-1,-1,0),(-1,-1,1),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,-1),(-1,1,0),( -1,1,1),
            (0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,0),(0,0,1),(0,1,-1),(0,1,0),(0,1,1),
            (1,-1,-1),(1,-1,0),(1,-1,1),(1,0,-1),(1,0,0),(1,0,1),(1,1,-1),(1,1,0),(1,1,1))
        INSERT OR IGNORE INTO shape_object_contacts
        SELECT og.object,sg.kind,sg.segment FROM shape_object_grid og CROSS JOIN offsets o
        JOIN shape_structure_grid sg ON sg.cx=og.cx+o.dx AND sg.cy=og.cy+o.dy AND sg.cz=og.cz+o.dz;
    ''')
    touch = {kind: np.zeros(count, bool) for kind in (1, 2, 3)}
    for kind, object_id in db.execute('SELECT kind,object FROM shape_object_contacts'):
        if int(kind) in touch:
            touch[int(kind)][int(object_id)] = True
    return touch


def _store_objects(db, objects, touches):
    db.executemany('''INSERT INTO shape_objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   ((int(index), int(objects['cells'][index]), int(objects['voxels'][index]),
                     *map(float, objects['center'][index]), *map(float, objects['extent'][index]),
                     *map(float, objects['axis'][index]), float(objects['lower'][index, 2]),
                     float(objects['upper'][index, 2]), 1, 0., int(touches[1][index]), int(touches[2][index]),
                     int(touches[3][index])) for index in range(objects['count'])))
    db.commit()


def _object_votes(db, count, codes, min_observations, checkpoint):
    """Sum voxel votes per object, keeping only class slots the rules need."""
    codes = np.asarray(sorted({int(code) for code in codes if code is not None}), np.int64)
    slots = {int(code): index for index, code in enumerate(codes)}
    selected, total = np.zeros((count, len(codes)), np.float64), np.zeros(count, np.float64)
    winner, winner_weight = np.full(count, 1, np.int64), np.zeros(count, np.float64)
    cursor = db.execute('''SELECT ov.object,v.class,SUM(v.weight) FROM shape_object_voxels ov
                           JOIN votes v USING(key) JOIN observations o USING(key)
                           WHERE o.count>=? GROUP BY ov.object,v.class ORDER BY ov.object,v.class''',
                       (int(min_observations),))
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        objects = np.asarray([row[0] for row in rows], np.int64)
        classes = np.asarray([row[1] for row in rows], np.int64)
        weights = np.asarray([row[2] for row in rows], np.float64)
        total += np.bincount(objects, weights=weights, minlength=count)
        order = np.lexsort((classes, -weights, objects))
        first = order[np.r_[True, objects[order][1:] != objects[order][:-1]]]
        obj, cls, amount = objects[first], classes[first], weights[first]
        better = (amount > winner_weight[obj]) | ((amount == winner_weight[obj]) & (cls < winner[obj]))
        winner[obj[better]], winner_weight[obj[better]] = cls[better], amount[better]
        positions = np.searchsorted(codes, classes)
        keep = positions < len(codes)
        keep[keep] &= codes[positions[keep]] == classes[keep]
        selected[objects[keep], positions[keep]] = weights[keep]
    return total, winner, np.divide(winner_weight, total, out=np.zeros(count), where=total > 0), selected, slots


def _object_mantle_planarity(db, objects, object_ids, planar_threshold, checkpoint):
    """Return the planar share of the non-end surface cells of selected objects.

    Geometry descriptors are computed on 10 cm cells.  Selecting only objects
    with pipe/HVAC evidence keeps this optional shape measurement out of the
    common object-only path and therefore out of the benchmark hot path.
    """
    count = objects['count']
    fraction = np.zeros(count, np.float64)
    object_ids = np.asarray(object_ids, np.int64)
    if not len(object_ids):
        return fraction
    db.executescript('''
        DROP TABLE IF EXISTS temp.shape_tech_objects;
        CREATE TEMP TABLE shape_tech_objects(object INTEGER PRIMARY KEY) WITHOUT ROWID;
    ''')
    db.executemany('INSERT INTO shape_tech_objects VALUES (?)', ((int(value),) for value in object_ids))
    planar, total = np.zeros(count, np.int64), np.zeros(count, np.int64)
    cursor = db.execute('''
        WITH object_cells AS (
            SELECT DISTINCT ov.object,g.cx,g.cy,g.cz
            FROM shape_object_voxels ov
            JOIN shape_tech_objects selected ON selected.object=ov.object
            JOIN geometry_voxels g USING(key)
        )
        SELECT object,f.x,f.y,f.z,f.planarity
        FROM object_cells JOIN geometry_features f USING(cx,cy,cz)
        ORDER BY object,cx,cy,cz
    ''')
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        ids = np.asarray([row[0] for row in rows], np.int64)
        points = np.asarray([row[1:4] for row in rows], np.float64)
        planarity = np.asarray([row[4] for row in rows], np.float64)
        axes = objects['axis'][ids]
        along = np.abs(np.einsum('ij,ij->i', points-objects['center'][ids], axes))
        # Excluding the terminal cells measures the pipe/duct mantle rather
        # than the planar end caps.  Very short sparse components simply use
        # their available cells, which avoids inventing a planar preference.
        mantle = along <= .45*objects['extent'][ids, 0]
        if not mantle.any():
            mantle = np.ones(len(ids), bool)
        ids, planarity = ids[mantle], planarity[mantle]
        total += np.bincount(ids, minlength=count)
        planar += np.bincount(ids, weights=planarity >= float(planar_threshold), minlength=count).astype(np.int64)
    return np.divide(planar, total, out=fraction, where=total > 0)


def _wall_metrics(db, objects, planes, walls, origin, voxel_size, checkpoint):
    """Return maximum signed wall protrusion and nearest parallel wall offset."""
    count = objects['count']
    protrusion = np.full(count, -np.inf)
    contact_band = np.full(count, np.inf)
    if db.execute('SELECT 1 FROM shape_object_contacts WHERE kind=1 LIMIT 1').fetchone() is not None:
        cursor = db.execute('''SELECT ov.object,c.segment,g.ix,g.iy,g.iz FROM shape_object_voxels ov
                               JOIN shape_object_contacts c ON c.object=ov.object AND c.kind=1
                               JOIN geometry_voxels g USING(key) ORDER BY ov.object,c.segment,g.key''')
        while rows := cursor.fetchmany(100_000):
            checkpoint()
            object_ids = np.asarray([row[0] for row in rows], np.int64)
            segments = np.asarray([row[1] for row in rows], np.int64)
            points = origin+(np.asarray([row[2:] for row in rows], np.float64)+.5)*float(voxel_size)
            distance = np.einsum('ij,ij->i', points-planes['center'][segments], planes['normal'][segments])
            np.maximum.at(protrusion, object_ids, np.maximum(distance, 0.))
            np.minimum.at(contact_band, object_ids, planes['band'][segments])
    protrusion[~np.isfinite(protrusion)] = np.inf
    wall_distance = np.full(count, np.inf)
    if not len(walls):
        return protrusion, wall_distance, contact_band
    tree = cKDTree(planes['center'][walls])
    _, nearest = tree.query(objects['center'], k=min(16, len(walls)), workers=-1)
    candidates = walls[np.asarray(nearest).reshape(count, -1)]
    normal = planes['normal'][candidates]
    parallel = np.abs(np.einsum('nij,nj->ni', normal, objects['axes'][:, :, 2])) >= np.cos(np.deg2rad(10.))
    delta = objects['center'][:, None, :]-planes['center'][candidates]
    distance = np.abs(np.einsum('nij,nij->ni', delta, normal))
    projected = np.einsum('nkij,nkj->nki', planes['basis'][candidates], delta)
    radius = .5*np.linalg.norm(objects['upper']-objects['lower'], axis=1)[:, None]
    overlap = np.all((projected >= planes['lower'][candidates]-radius[:, :, None]) &
                     (projected <= planes['upper'][candidates]+radius[:, :, None]), axis=2)
    distance[~(parallel & overlap)] = np.inf
    return protrusion, distance.min(axis=1), contact_band


def _bulk_update_decisions(db, keys, codes, confidence):
    """Write a vectorized decision block through one temporary-table update.

    The rule passes already resolve their precedence in NumPy.  Staging the
    resulting keys avoids one SQLite UPDATE and one Python tuple per voxel,
    which is material on city-scale outdoor scans.
    """
    keys = np.asarray(keys, dtype='V24')
    if not len(keys):
        return
    codes = np.asarray(codes, np.int64)
    confidence = np.asarray(confidence, np.int64)
    if len(keys) != len(codes) or len(keys) != len(confidence):
        raise ValueError('decision update arrays must have equal length')
    db.execute('''CREATE TEMP TABLE IF NOT EXISTS shape_decision_updates(
                    key BLOB PRIMARY KEY,code INTEGER NOT NULL,confidence INTEGER NOT NULL
                  ) WITHOUT ROWID''')
    db.execute('DELETE FROM shape_decision_updates')
    for start in range(0, len(keys), 100_000):
        stop = min(len(keys), start+100_000)
        db.executemany('INSERT INTO shape_decision_updates VALUES (?,?,?)',
                       ((bytes(key), int(code), int(value))
                        for key, code, value in zip(keys[start:stop], codes[start:stop], confidence[start:stop])))
    db.execute('''UPDATE decisions AS d SET code=u.code,confidence=u.confidence
                  FROM shape_decision_updates AS u WHERE d.key=u.key''')
    db.execute('DELETE FROM shape_decision_updates')


def _apply_layer_target(votes, target, kinds, source, active, protected, reflection, stairs, min_observations,
                        checkpoint, result, rule):
    """Apply carpet and flush-wall exceptions directly to structural layer voxels."""
    if target is None or int(target) not in active:
        return False
    db = votes.db
    immutable = np.asarray(sorted(protected | {int(code) for code in (reflection, stairs) if code is not None}), np.int64)
    source, active = np.asarray(sorted(source), np.int64), np.asarray(sorted({1, *active}), np.int64)
    eligible = np.setdiff1d(np.intersect1d(source, active), immutable, assume_unique=True)
    if not len(eligible):
        return False
    placeholders = ','.join('?' for _ in kinds)
    code_placeholders = ','.join('?' for _ in eligible)
    checkpoint()
    where = f'''sv.kind IN ({placeholders}) AND o.count>=? AND gc.scene_context=0
                AND d.code IN ({code_placeholders}) AND d.code<>?'''
    params = (*map(int, kinds), int(min_observations), *map(int, eligible), int(target))
    rows = db.execute(f'''SELECT d.code,COUNT(*),COUNT(DISTINCT sv.segment)
                          FROM decisions d JOIN shape_structure_voxels sv USING(key)
                          JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                          WHERE {where} GROUP BY d.code''', params).fetchall()
    changed = {f'{int(old)}->{int(target)}': int(amount) for old, amount, _ in rows}
    # There is one fixed target for this layer, so SQLite can apply the
    # set-based update directly. Object rules use the keyed temporary table
    # below because their per-voxel target and confidence may differ.
    if changed:
        db.execute(f'''UPDATE decisions SET code=?,confidence=? WHERE key IN (
                           SELECT d.key FROM decisions d JOIN shape_structure_voxels sv USING(key)
                           JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                           WHERE {where} GROUP BY d.key)''',
                   (int(target), 0 if int(target) == 1 else 255, *params))
    segments = int(sum(amount for _, _, amount in rows))
    checkpoint()
    result['rules'][rule]['voxels'] += sum(changed.values())
    result['rules'][rule]['objects'] += segments
    for label, amount in changed.items():
        result['changed_by_transition'][label] = result['changed_by_transition'].get(label, 0)+amount
    return bool(changed)


def _apply_object_targets(votes, target, rule, target_share, active, protected, reflection, stairs, source_by_rule,
                          min_observations, checkpoint, result, protected_rule_sources=None):
    """Apply one selected rule per voxel object in bounded decision blocks."""
    db = votes.db
    chosen = np.flatnonzero(target >= 0)
    if not len(chosen):
        return False
    db.executescript('''CREATE TEMP TABLE shape_targets(object INTEGER PRIMARY KEY,target INTEGER NOT NULL,rule TEXT NOT NULL,
                        share REAL NOT NULL) WITHOUT ROWID;''')
    db.executemany('INSERT INTO shape_targets VALUES (?,?,?,?)',
                   ((int(index), int(target[index]), str(rule[index]), float(target_share[index])) for index in chosen))
    if set(source_by_rule) != set(OBJECT_RULE_NAMES):
        missing = sorted(set(OBJECT_RULE_NAMES) - set(source_by_rule))
        extra = sorted(set(source_by_rule) - set(OBJECT_RULE_NAMES))
        raise RuntimeError(f'indoor shape rule source map mismatch: missing={missing}, extra={extra}')
    rule_index = {name: index for index, name in enumerate(OBJECT_RULE_NAMES)}
    sources = [np.asarray(sorted(source_by_rule[name]), np.int64) for name in OBJECT_RULE_NAMES]
    object_rules = np.bincount(np.asarray([rule_index[str(rule[index])] for index in chosen], np.int64),
                               minlength=len(OBJECT_RULE_NAMES))
    voxel_rules = np.zeros(len(OBJECT_RULE_NAMES), np.int64)
    protected = np.asarray(sorted(protected), np.int64)
    hard_immutable = np.asarray(sorted({int(code) for code in (reflection, stairs) if code is not None}), np.int64)
    protected_rule_sources = protected_rule_sources or {}
    active = np.asarray(sorted({1, *active}), np.int64)
    cursor = db.execute('''SELECT d.key,d.code,st.target,st.rule,st.share FROM decisions d
                           JOIN shape_object_voxels ov USING(key) JOIN shape_targets st ON st.object=ov.object
                           JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                           WHERE o.count>=? AND gc.scene_context=0 ORDER BY d.key''', (int(min_observations),))
    changed = {}
    update_keys, update_codes, update_conf = [], [], []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        keys = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
        current = np.asarray([row[1] for row in rows], np.int64)
        new = np.asarray([row[2] for row in rows], np.int64)
        ids = np.asarray([rule_index[row[3]] for row in rows], np.int64)
        shares = np.asarray([row[4] for row in rows], np.float64)
        allowed = (current != new) & np.isin(current, active) & ~np.isin(current, hard_immutable)
        for index, source in enumerate(sources):
            member = ids == index
            if not member.any():
                continue
            exceptions = np.asarray(sorted(protected_rule_sources.get(OBJECT_RULE_NAMES[index], ())), np.int64)
            mutable = ~np.isin(current[member], protected)
            if len(exceptions):
                mutable |= np.isin(current[member], exceptions)
            allowed[member] &= np.isin(current[member], source) & mutable
        take = np.flatnonzero(allowed)
        if len(take):
            confidence = np.where(new[take] == 1, 0, np.rint(shares[take]*255).astype(np.int64))
            update_keys.append(keys[take])
            update_codes.append(new[take])
            update_conf.append(confidence)
            voxel_rules += np.bincount(ids[take], minlength=len(OBJECT_RULE_NAMES))
            pairs, counts = np.unique(np.column_stack((current[take], new[take])), axis=0, return_counts=True)
            for pair, amount in zip(pairs, counts):
                label = f'{int(pair[0])}->{int(pair[1])}'
                changed[label] = changed.get(label, 0)+int(amount)
    cursor.close()
    if update_keys:
        _bulk_update_decisions(db, np.concatenate(update_keys), np.concatenate(update_codes),
                               np.concatenate(update_conf))
    for index, name in enumerate(OBJECT_RULE_NAMES):
        result['rules'][name]['objects'] += int(object_rules[index])
        result['rules'][name]['voxels'] += int(voxel_rules[index])
    for label, amount in changed.items():
        result['changed_by_transition'][label] = result['changed_by_transition'].get(label, 0)+amount
    return bool(changed)


def _apply_supported_small_items(votes, clutter, furniture, origin, voxel_size, object_cell, active, protected,
                                 reflection, stairs, min_observations, checkpoint, result):
    """Split small mixed-evidence items resting on a larger furniture surface.

    A cup and a table can touch in the 4 cm object raster.  Their mixed SAM
    evidence provides a deterministic voxel-level cut without splitting normal
    multi-surface furniture or columns.
    """
    if clutter is None or furniture is None:
        return False
    db = votes.db
    db.executescript('''
        DROP TABLE IF EXISTS temp.shape_mixed_grid;
        DROP TABLE IF EXISTS temp.shape_mixed_voxels;
        CREATE TEMP TABLE shape_mixed_grid(cx INTEGER,cy INTEGER,cz INTEGER,object INTEGER,
            PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE shape_mixed_voxels(key BLOB PRIMARY KEY,object INTEGER NOT NULL) WITHOUT ROWID;
    ''')
    query = '''SELECT ov.key,g.ix,g.iy,g.iz FROM shape_object_voxels ov
               JOIN votes v USING(key) JOIN geometry_voxels g USING(key)
               WHERE v.class IN (?,?) GROUP BY ov.key HAVING COUNT(DISTINCT v.class)=2 ORDER BY ov.key'''
    cursor = db.execute(query, (int(clutter), int(furniture)))
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        grid = _grid_coordinates(np.asarray([row[1:] for row in rows], np.int64), origin, voxel_size, object_cell)
        pending.extend((*map(int, cell), None) for cell in grid)
        if len(pending) >= 100_000:
            db.executemany('INSERT OR IGNORE INTO shape_mixed_grid VALUES (?,?,?,?)', pending)
            pending.clear()
    if pending:
        db.executemany('INSERT OR IGNORE INTO shape_mixed_grid VALUES (?,?,?,?)', pending)
    rows = db.execute('SELECT cx,cy,cz FROM shape_mixed_grid ORDER BY cx,cy,cz').fetchall()
    if not rows:
        return False
    cells = np.asarray(rows, np.int64)
    labels = _component_labels(cells)
    db.executemany('UPDATE shape_mixed_grid SET object=? WHERE cx=? AND cy=? AND cz=?',
                   ((int(label), *map(int, cell)) for label, cell in zip(labels, cells)))
    count = int(labels.max()+1)
    lower, upper = np.full((count, 3), np.iinfo(np.int64).max), np.full((count, 3), np.iinfo(np.int64).min)
    for axis in range(3):
        np.minimum.at(lower[:, axis], labels, cells[:, axis])
        np.maximum.at(upper[:, axis], labels, cells[:, axis])
    small = np.max((upper-lower+1)*float(object_cell), axis=1) < .5
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape, keys = high-low+1, _linear_keys(cells, low, high-low+1)
    cursor = db.execute(query, (int(clutter), int(furniture)))
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        voxel_keys = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
        grid = _grid_coordinates(np.asarray([row[1:] for row in rows], np.int64), origin, voxel_size, object_cell)
        mapped = labels[np.searchsorted(keys, _linear_keys(grid, low, shape))]
        pending.extend((bytes(key), int(label)) for key, label in zip(voxel_keys, mapped) if small[label])
        if len(pending) >= 100_000:
            db.executemany('INSERT INTO shape_mixed_voxels VALUES (?,?)', pending)
            pending.clear()
    if pending:
        db.executemany('INSERT INTO shape_mixed_voxels VALUES (?,?)', pending)
    immutable = np.asarray(sorted(protected | {int(code) for code in (reflection, stairs) if code is not None}), np.int64)
    allowed_source, active = np.asarray(sorted({int(clutter), int(furniture), 1}), np.int64), np.asarray(sorted({1, *active}), np.int64)
    cursor = db.execute('''SELECT d.key,d.code,m.object FROM decisions d JOIN shape_mixed_voxels m USING(key)
                           JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                           WHERE o.count>=? AND gc.scene_context=0 ORDER BY d.key''', (int(min_observations),))
    changed, objects = {}, set()
    update_keys = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        keys = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
        current = np.asarray([row[1] for row in rows], np.int64)
        take = ((current != int(clutter)) & np.isin(current, allowed_source) & np.isin(current, active)
                & ~np.isin(current, immutable))
        if take.any():
            update_keys.append(keys[take])
            old_codes, amounts = np.unique(current[take], return_counts=True)
            for old, amount in zip(old_codes, amounts):
                label = f'{int(old)}->{int(clutter)}'
                changed[label] = changed.get(label, 0)+int(amount)
            objects.update(int(rows[index][2]) for index in np.flatnonzero(take))
    cursor.close()
    if update_keys:
        all_keys = np.concatenate(update_keys)
        _bulk_update_decisions(db, all_keys, np.full(len(all_keys), int(clutter), np.int64),
                               np.zeros(len(all_keys), np.int64))
    result['rules']['furniture_clutter']['objects'] += len(objects)
    result['rules']['furniture_clutter']['voxels'] += sum(changed.values())
    for label, amount in changed.items():
        result['changed_by_transition'][label] = result['changed_by_transition'].get(label, 0)+amount
    return bool(changed)


def _outline_cells(indices, outline):
    """Map voxel indices to a shared outline raster without extrapolation."""
    result = np.zeros(len(indices), np.int8)
    if not outline or not outline.get('applied'):
        return result
    low = np.asarray(outline['origin'], np.int64)
    high = low+np.asarray(outline['shape'], np.int64)-1
    xy = np.asarray(indices, np.int64)[:, :2]//int(outline['cells'])
    valid = np.all((xy >= low) & (xy <= high), axis=1)
    grid = xy[valid]-low
    result[valid] = np.where(outline['barrier'][grid[:, 0], grid[:, 1]], 2,
                             np.where(outline['inside'][grid[:, 0], grid[:, 1]], 1, 0))
    return result


def _outline_facade_cells(indices, outline):
    """Return barrier cells that actually border the proven enclosed area."""
    result = np.zeros(len(indices), bool)
    if not outline or not outline.get('applied') or not outline.get('has_enclosed'):
        return result
    low = np.asarray(outline['origin'], np.int64)
    shape = np.asarray(outline['shape'], np.int64)
    xy = np.asarray(indices, np.int64)[:, :2]//int(outline['cells'])
    valid = np.all((xy >= low) & (xy < low+shape), axis=1)
    selected = np.flatnonzero(valid)
    grid = xy[selected]-low
    barrier = outline['barrier']
    inside = outline['inside']
    for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = grid+np.asarray(offset, np.int64)
        in_bounds = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
        touching = np.zeros(len(selected), bool)
        touching[in_bounds] = inside[neighbor[in_bounds, 0], neighbor[in_bounds, 1]]
        result[selected] |= barrier[grid[:, 0], grid[:, 1]] & touching
    return result


def _update_rule_result(result, rule, pairs, objects=0):
    amount = int(sum(pairs.values()))
    result['rules'][rule]['objects'] += int(objects)
    result['rules'][rule]['voxels'] += amount
    for label, count in pairs.items():
        result['changed_by_transition'][label] = result['changed_by_transition'].get(label, 0)+int(count)


def _apply_facade_sides(votes, planes, outline, by_id, origin, voxel_size, active, min_observations,
                        minimum_height, minimum_area, flush_max, scan_context, checkpoint, result):
    """Assign each RANSAC facade layer as a whole to building or wall."""
    report = result['facade_side']
    has_outdoor_scan = any(item.get('scene') == 'outdoor' or bool(item.get('outdoor'))
                           for item in (scan_context or {}).values())
    if not has_outdoor_scan:
        return False
    building, wall = _code(by_id, 'building'), _code(by_id, 'wall')
    if (building is None or wall is None or outline is None or not outline.get('has_enclosed')
            or planes is None or 'class_id' not in planes or not len(planes['center'])):
        return False
    relevant = np.asarray([(kind == 1 and name in {'wall', 'building'})
                           for kind, name in zip(planes['kinds'], planes['class_id'])], bool)
    if 'bbox_min' in planes and 'bbox_max' in planes:
        span = np.asarray(planes['bbox_max'], np.float64)[:, 2]-np.asarray(planes['bbox_min'], np.float64)[:, 2]
        relevant &= span >= float(minimum_height)
    if 'area' in planes:
        relevant &= np.asarray(planes['area'], np.float64) >= float(minimum_area)
    if not relevant.any():
        return False
    db = votes.db
    outward = np.zeros(len(planes['center']), np.int8)
    # A double facade can span one full outline cell.  Probe far enough to
    # reach proven inside/outside cells instead of stopping in its barrier.
    probe = max(float(outline['cell_size_m'])*3., .50)
    for plane in np.flatnonzero(relevant):
        normal, center = planes['normal'][plane], planes['center'][plane]
        samples = center[None, :]+np.asarray((-probe, probe))[:, None]*normal
        indices = np.floor((samples-np.asarray(origin, np.float64))/float(voxel_size)).astype(np.int64)
        membership = _outline_cells(indices, outline)
        # 0=outside, 1=inside, 2=barrier.  A valid pair gives the normal's
        # orientation; malformed or thick barrier samples are left untouched.
        if membership[0] == 1 and membership[1] == 0:
            outward[plane] = 1
        elif membership[0] == 0 and membership[1] == 1:
            outward[plane] = -1
    if not outward.any():
        return False
    # A laser scan commonly yields two parallel facade skins.  Decide the
    # complete RANSAC layer from the pair midpoint, never from noisy voxels.
    centers = np.asarray(planes['center'], np.float64)
    normals = np.asarray(planes['normal'], np.float64)
    outward_vectors = normals*outward[:, None]
    decision = np.zeros(len(centers), np.int8)
    candidates = relevant & (outward != 0)
    for plane in np.flatnonzero(candidates):
        alignment = outward_vectors@outward_vectors[plane] >= np.cos(np.deg2rad(10.))
        delta = centers-centers[plane]
        normal_delta = delta@outward_vectors[plane]
        lateral = np.linalg.norm(delta-normal_delta[:, None]*outward_vectors[plane], axis=1)
        paired = candidates & (np.arange(len(candidates)) != plane) & alignment
        paired &= (np.abs(normal_delta) >= .06) & (np.abs(normal_delta) <= .50) & (lateral <= .15)
        if not paired.any():
            continue
        partner = np.flatnonzero(paired)[np.argmin(np.abs(normal_delta[paired]))]
        pair_center = (centers[plane]+centers[partner])/2.
        side = float((centers[plane]-pair_center)@outward_vectors[plane])
        if side >= .03:
            decision[plane] = 1
        elif side <= -.03:
            decision[plane] = 2

    # An unpaired layer derives its complete label from the scanner context of
    # its layer voxels.  Unknown and mixed layers deliberately stay untouched.
    singles = candidates & (decision == 0)
    if singles.any():
        placeholders = ','.join('?' for _ in np.flatnonzero(singles))
        rows = db.execute(f'''WITH nearest AS (
                                  SELECT sv.segment,sv.key,
                                         ROW_NUMBER() OVER (PARTITION BY sv.key
                                                            ORDER BY ABS(sv.distance),sv.segment) AS rank
                                  FROM shape_structure_voxels sv WHERE sv.kind=1
                              )
                              SELECT n.segment,
                                     SUM(CASE WHEN gc.scene_context=1 THEN 1 ELSE 0 END),
                                     SUM(CASE WHEN gc.scene_context=0 THEN 1 ELSE 0 END)
                              FROM nearest n JOIN geometry_context gc USING(key)
                              JOIN observations o USING(key)
                              WHERE n.rank=1 AND o.count>=? AND n.segment IN ({placeholders})
                              GROUP BY n.segment''',
                          (int(min_observations), *(int(item) for item in np.flatnonzero(singles)))).fetchall()
        context_counts = {int(segment): (int(outdoor), int(indoor)) for segment, outdoor, indoor in rows}
        for plane in np.flatnonzero(singles):
            outdoor_count, indoor_count = context_counts.get(int(plane), (0, 0))
            known = outdoor_count+indoor_count
            if known and outdoor_count/known >= .70:
                decision[plane] = 3
            elif known and indoor_count/known >= .70:
                decision[plane] = 4
            else:
                decision[plane] = 5
    labels = ('pair_outer', 'pair_inner', 'context_outdoor', 'context_indoor', 'undecided')
    for index, label in enumerate(labels, start=1):
        report[label]['planes'] += int((decision == index).sum())

    transitions, touched = {}, set()
    cursor = db.execute('''WITH nearest AS (
                               SELECT sv.segment,sv.key,
                                      ROW_NUMBER() OVER (PARTITION BY sv.key
                                                         ORDER BY ABS(sv.distance),sv.segment) AS rank
                               FROM shape_structure_voxels sv WHERE sv.kind=1
                           )
                           SELECT n.segment,n.key,d.code
                           FROM nearest n
                           JOIN decisions d USING(key) JOIN observations o USING(key)
                           WHERE n.rank=1 AND o.count>=? ORDER BY n.segment,n.key''', (int(min_observations),))
    facade_keys = []
    update_keys, update_wanted = [], []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        segments = np.asarray([row[0] for row in rows], np.int64)
        valid = segments < len(decision)
        kinds = np.zeros(len(rows), np.int8)
        kinds[valid] = decision[segments[valid]]
        current = np.asarray([row[2] for row in rows], np.int64)
        layer = (kinds != 0) & np.isin(current, [1, int(building), int(wall)])
        if not layer.any():
            continue
        keys = np.asarray([bytes(row[1]) for row in rows], dtype='V24')
        facade_keys.append(keys[layer])
        for index, label in enumerate(labels, start=1):
            report[label]['voxels'] += int((layer & (kinds == index)).sum())
        decided = layer & (kinds != 5)
        wanted = np.where(np.isin(kinds, [1, 3]), int(building), int(wall))
        take = decided & (current != wanted)
        if take.any():
            update_keys.append(keys[take])
            update_wanted.append(wanted[take])
            pairs, amounts = np.unique(np.column_stack((current[take], wanted[take])), axis=0, return_counts=True)
            for pair, amount in zip(pairs, amounts):
                label = f'{int(pair[0])}->{int(pair[1])}'
                transitions[label] = transitions.get(label, 0)+int(amount)
            touched.update(int(segment) for segment in np.unique(segments[take]))
    cursor.close()
    if facade_keys:
        all_facade_keys = np.concatenate(facade_keys)
        for start in range(0, len(all_facade_keys), 100_000):
            stop = min(len(all_facade_keys), start+100_000)
            db.executemany('INSERT OR IGNORE INTO shape_facade_voxels VALUES (?)',
                           ((bytes(key),) for key in all_facade_keys[start:stop]))
    if update_keys:
        all_keys = np.concatenate(update_keys)
        all_wanted = np.concatenate(update_wanted)
        _bulk_update_decisions(db, all_keys, all_wanted, np.full(len(all_keys), 255, np.int64))
    if transitions:
        db.commit()
        votes._invalidate_lookup()
        _update_rule_result(result, 'facade_side', transitions, len(touched))
    return bool(transitions)


def _prepare_outdoor_objects(votes, by_id, origin, voxel_size, object_cell, outline, min_observations, checkpoint):
    """Materialize compact exterior components and their vote shares once.

    Large road and terrain sheets are intentionally excluded; their surface
    rules use a 2-D raster below.  This keeps the object pass bounded on large
    outdoor scans while retaining walls, roofs, signs, poles and vehicles.
    """
    db = votes.db
    target_names = ('building', 'outdoor_wall', 'utility_pole', 'traffic_sign', 'fence', 'vehicle', 'power_line')
    if not any(name in by_id for name in target_names):
        return None
    names = ('building', 'wall', 'steel_structure', 'utility_pole', 'traffic_sign', 'fence', 'vehicle',
             'power_line', 'tree_trunk')
    codes = {1} | {int(by_id[name]) for name in names if name in by_id}
    evidence = {int(by_id[name]) for name in ('building', 'wall', 'utility_pole', 'traffic_sign', 'fence',
                                               'vehicle', 'power_line') if name in by_id}
    if not codes and not evidence:
        return None
    db.executescript('''
        DROP TABLE IF EXISTS temp.shape_outdoor_voxels;
        DROP TABLE IF EXISTS temp.shape_outdoor_objects;
        DROP TABLE IF EXISTS temp.shape_outdoor_evidence;
        DROP TABLE IF EXISTS temp.shape_outdoor_structure;
        DROP TABLE IF EXISTS temp.shape_tree_grid;
        CREATE TEMP TABLE shape_outdoor_voxels(key BLOB PRIMARY KEY,object INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE shape_outdoor_objects(object INTEGER PRIMARY KEY,l1 REAL,l2 REAL,l3 REAL,
            min_z REAL,max_z REAL,x REAL,y REAL,z REAL,barrier INTEGER NOT NULL,structural INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE shape_outdoor_structure(key BLOB PRIMARY KEY) WITHOUT ROWID;
        INSERT INTO shape_outdoor_structure SELECT DISTINCT key FROM shape_structure_voxels;
    ''')
    code_sql = ','.join('?' for _ in sorted(codes))
    if evidence:
        evidence_sql = ','.join('?' for _ in sorted(evidence))
        db.execute('CREATE TEMP TABLE shape_outdoor_evidence(key BLOB PRIMARY KEY) WITHOUT ROWID')
        db.execute(f'''INSERT INTO shape_outdoor_evidence
                       SELECT DISTINCT key FROM votes WHERE class IN ({evidence_sql})''', tuple(sorted(evidence)))
    where = f'd.code IN ({code_sql})'
    evidence_join = ''
    if evidence:
        where += ' OR ev.key IS NOT NULL'
        evidence_join = 'LEFT JOIN shape_outdoor_evidence ev USING(key)'
    key_parts, cell_parts = [], []
    cursor = db.execute(f'''SELECT g.key,g.ix,g.iy,g.iz,d.code
                             FROM geometry_voxels g JOIN decisions d USING(key)
                             JOIN geometry_context gc USING(key) JOIN observations o USING(key)
                             {evidence_join}
                             WHERE gc.scene_context=1 AND o.count>=? AND ({where})
                             ORDER BY g.key''',
                        (int(min_observations), *sorted(codes)))
    while batch := cursor.fetchmany(100_000):
        checkpoint()
        keys = _keys_from_rows(batch)
        key_parts.append(keys)
        cell_parts.append(_grid_coordinates(keys.view('<i8').reshape(-1, 3), origin, voxel_size, object_cell))
    cursor.close()
    if not key_parts:
        return None
    keys = np.concatenate(key_parts)
    cells = np.concatenate(cell_parts)
    unique = np.unique(cells, axis=0)
    labels = _component_labels(unique)
    low, high = unique.min(axis=0), unique.max(axis=0)
    grid_keys = _linear_keys(unique, low, high-low+1)
    source_keys = _linear_keys(cells, low, high-low+1)
    objects = labels[np.searchsorted(grid_keys, source_keys)]
    db.executemany('INSERT INTO shape_outdoor_voxels VALUES (?,?)',
                   ((bytes(key), int(obj)) for key, obj in zip(keys, objects)))
    count = int(objects.max()+1)
    records = []

    def finish(rows):
        obj = int(rows[0][0])
        indices = np.asarray([row[1:4] for row in rows], np.int64)
        point = np.asarray(origin, np.float64)+(indices.astype(np.float64)+.5)*float(voxel_size)
        center = point.mean(axis=0)
        if len(point) >= 3:
            _, _, vectors = np.linalg.svd(point-center, full_matrices=False)
            projected = (point-center)@vectors.T
            extent = projected.max(axis=0)-projected.min(axis=0)
            extent = np.sort(extent)[::-1]
        else:
            extent = np.zeros(3)
        barriers = _outline_facade_cells(indices, outline)
        structural = np.fromiter((row[4] for row in rows), dtype=bool, count=len(rows))
        records.append((obj, *map(float, extent), float(point[:, 2].min()), float(point[:, 2].max()),
                        *map(float, center), int(barriers.any()), int(structural.any())))

    cursor = db.execute('''SELECT ov.object,g.ix,g.iy,g.iz,
                                  CASE WHEN sv.key IS NULL THEN 0 ELSE 1 END
                           FROM shape_outdoor_voxels ov JOIN geometry_voxels g USING(key)
                           LEFT JOIN shape_outdoor_structure sv USING(key)
                           ORDER BY ov.object,g.key''')
    carry = []
    while batch := cursor.fetchmany(100_000):
        checkpoint()
        rows = carry+batch
        last, split = rows[-1][0], len(rows)
        while split and rows[split-1][0] == last:
            split -= 1
        if split:
            start = 0
            for stop in range(1, split+1):
                if stop == split or rows[stop][0] != rows[start][0]:
                    finish(rows[start:stop])
                    start = stop
        carry = rows[split:]
    cursor.close()
    if carry:
        finish(carry)
    db.executemany('INSERT INTO shape_outdoor_objects VALUES (?,?,?,?,?,?,?,?,?,?,?)', records)
    db.execute('CREATE INDEX shape_outdoor_voxels_object ON shape_outdoor_voxels(object,key)')
    db.commit()
    return {'count': count, 'codes': codes}


def _outdoor_vote_shares(db, count, min_observations, checkpoint):
    totals = np.zeros(count, np.float64)
    shares = {}
    cursor = db.execute('''SELECT ov.object,v.class,SUM(v.weight) FROM shape_outdoor_voxels ov
                           JOIN votes v USING(key) JOIN observations o USING(key)
                           WHERE o.count>=? GROUP BY ov.object,v.class ORDER BY ov.object,v.class''',
                        (int(min_observations),))
    rows = []
    while batch := cursor.fetchmany(100_000):
        checkpoint()
        rows.extend(batch)
    if not rows:
        return totals, shares
    objects = np.asarray([row[0] for row in rows], np.int64)
    values = np.asarray([row[2] for row in rows], np.float64)
    totals += np.bincount(objects, weights=values, minlength=count)
    for code in np.unique(np.asarray([row[1] for row in rows], np.int64)):
        selected = np.asarray([row[1] == int(code) for row in rows])
        values_by_object = np.bincount(objects[selected], weights=values[selected], minlength=count)
        shares[int(code)] = np.divide(values_by_object, totals, out=np.zeros(count), where=totals > 0)
    return totals, shares


def _apply_outdoor_targets(votes, targets, rule_by_object, active, immutable, min_observations, checkpoint, result):
    if not targets:
        return False
    db = votes.db
    db.execute('DROP TABLE IF EXISTS temp.shape_outdoor_targets')
    rule_names = tuple(sorted(set(rule_by_object.values())))
    rule_ids = {name: index for index, name in enumerate(rule_names)}
    db.execute('CREATE TEMP TABLE shape_outdoor_targets(object INTEGER PRIMARY KEY,target INTEGER NOT NULL,rule INTEGER NOT NULL) WITHOUT ROWID')
    db.executemany('INSERT INTO shape_outdoor_targets VALUES (?,?,?)',
                   ((int(obj), int(code), int(rule_ids[rule_by_object[obj]])) for obj, code in targets.items()))
    transitions, object_changes = {}, {}
    cursor = db.execute('''SELECT ov.object,d.key,d.code,t.target,t.rule FROM shape_outdoor_targets t
                           JOIN shape_outdoor_voxels ov ON ov.object=t.object
                           JOIN decisions d USING(key) JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                           LEFT JOIN shape_facade_voxels fv ON fv.key=ov.key
                           WHERE gc.scene_context=1 AND o.count>=? AND fv.key IS NULL
                           ''', (int(min_observations),))
    immutable = np.asarray(sorted(immutable), np.int64)
    active = np.asarray(sorted({1, *active}), np.int64)
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        current = np.asarray([row[2] for row in rows], np.int64)
        wanted = np.asarray([row[3] for row in rows], np.int64)
        take = (current != wanted) & np.isin(current, active) & ~np.isin(current, immutable)
        if take.any():
            keys = _keys_from_rows(rows, 1)
            objects = np.asarray([row[0] for row in rows], np.int64)
            rules = np.asarray([row[4] for row in rows], np.int8)
            _bulk_update_decisions(db, keys[take], wanted[take], np.full(int(take.sum()), 255, np.int64))
            for rule_id in np.unique(rules[take]):
                selected = take & (rules == rule_id)
                pairs, amounts = np.unique(np.column_stack((current[selected], wanted[selected])), axis=0,
                                           return_counts=True)
                name = rule_names[int(rule_id)]
                values = transitions.setdefault(name, {})
                for pair, amount in zip(pairs, amounts):
                    label = f'{int(pair[0])}->{int(pair[1])}'
                    values[label] = values.get(label, 0)+int(amount)
                object_changes.setdefault(name, set()).update(map(int, np.unique(objects[selected])))
    if transitions:
        db.commit()
        votes._invalidate_lookup()
        for rule, pairs in transitions.items():
            _update_rule_result(result, rule, pairs, len(object_changes.get(rule, ())))
    return bool(transitions)


def _apply_key_target(votes, keys, target, rule, active, immutable, min_observations, checkpoint, result,
                      scene_contexts=(1,)):
    """Apply one exterior target to an explicit, bounded set of voxel keys."""
    if not len(keys) or target is None or int(target) not in active:
        return False
    db = votes.db
    db.execute('DROP TABLE IF EXISTS temp.shape_key_targets')
    db.execute('CREATE TEMP TABLE shape_key_targets(key BLOB PRIMARY KEY) WITHOUT ROWID')
    db.executemany('INSERT OR IGNORE INTO shape_key_targets VALUES (?)', ((bytes(key),) for key in keys))
    transitions = {}
    contexts = tuple(int(value) for value in scene_contexts)
    placeholders = ','.join('?' for _ in contexts)
    cursor = db.execute(f'''SELECT d.key,d.code FROM shape_key_targets t JOIN decisions d USING(key)
                            JOIN observations o USING(key) JOIN geometry_context gc USING(key)
                            WHERE gc.scene_context IN ({placeholders}) AND o.count>=? ORDER BY d.key''',
                        (*contexts, int(min_observations)))
    immutable, active = np.asarray(sorted(immutable), np.int64), np.asarray(sorted({1, *active}), np.int64)
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        current = np.asarray([row[1] for row in rows], np.int64)
        take = (current != int(target)) & np.isin(current, active) & ~np.isin(current, immutable)
        if take.any():
            row_keys = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
            _bulk_update_decisions(db, row_keys[take], np.full(int(take.sum()), int(target), np.int64),
                                   np.full(int(take.sum()), 255, np.int64))
            old_codes, amounts = np.unique(current[take], return_counts=True)
            for old, amount in zip(old_codes, amounts):
                label = f'{int(old)}->{int(target)}'
                transitions[label] = transitions.get(label, 0)+int(amount)
    if transitions:
        db.commit()
        votes._invalidate_lookup()
        _update_rule_result(result, rule, transitions, 0)
    return bool(transitions)


def _apply_curb_and_sidewalk(votes, by_id, voxel_size, active, immutable, min_observations, checkpoint, result):
    """Resolve curbs without changing road or sidewalk decisions."""
    road, sidewalk, ground, curb = (_code(by_id, name) for name in ('road', 'sidewalk', 'ground', 'curb'))
    if road is None or sidewalk is None or curb is None or int(curb) not in active:
        return False
    db = votes.db
    sources = [code for code in (road, sidewalk, ground, 1) if code is not None]
    clause = ','.join('?' for _ in sources)
    key_parts, index_parts, code_parts = [], [], []
    cursor = db.execute(f'''SELECT g.key,g.ix,g.iy,g.iz,d.code FROM geometry_voxels g JOIN decisions d USING(key)
                             JOIN geometry_context gc USING(key) JOIN observations o USING(key)
                             WHERE gc.scene_context=1 AND o.count>=? AND d.code IN ({clause}) ORDER BY g.key''',
                        (int(min_observations), *map(int, sources)))
    while batch := cursor.fetchmany(100_000):
        checkpoint()
        key_parts.append(_keys_from_rows(batch))
        index_parts.append(np.asarray([row[1:4] for row in batch], np.int64))
        code_parts.append(np.fromiter((row[4] for row in batch), dtype=np.int64, count=len(batch)))
    cursor.close()
    if not key_parts:
        return False
    keys = np.concatenate(key_parts)
    indices = np.concatenate(index_parts)
    codes = np.concatenate(code_parts)
    cell_size = .10
    xy = np.floor(indices[:, :2]*float(voxel_size)/cell_size).astype(np.int64)
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    heights = np.zeros(len(unique), np.float64)
    np.maximum.at(heights, inverse, (indices[:, 2]+.5)*float(voxel_size))
    road_weight, sidewalk_weight = np.zeros(len(unique), np.float64), np.zeros(len(unique), np.float64)
    low, high = unique.min(axis=0), unique.max(axis=0)
    shape = high-low+1

    def linear_xy(values):
        relative = values-low
        return (relative[:, 0]*shape[1]+relative[:, 1]).astype(np.int64)

    cells = linear_xy(unique)
    cursor = db.execute('''SELECT g.ix,g.iy,v.class,SUM(v.weight) FROM geometry_voxels g
                           JOIN geometry_context gc USING(key) JOIN observations o USING(key) JOIN votes v USING(key)
                           WHERE gc.scene_context=1 AND o.count>=? AND v.class IN (?,?)
                           GROUP BY g.key,v.class''',
                        (int(min_observations), int(road), int(sidewalk)))
    while vote_rows := cursor.fetchmany(100_000):
        checkpoint()
        vote_xy = np.floor(np.asarray([row[:2] for row in vote_rows], np.float64)*float(voxel_size)/cell_size).astype(np.int64)
        found = np.searchsorted(cells, linear_xy(vote_xy))
        valid = found < len(cells)
        valid[valid] &= cells[found[valid]] == linear_xy(vote_xy[valid])
        if not valid.any():
            continue
        classes = np.fromiter((row[2] for row in vote_rows), dtype=np.int64, count=len(vote_rows))
        weights = np.fromiter((row[3] for row in vote_rows), dtype=np.float64, count=len(vote_rows))
        np.add.at(road_weight, found[valid & (classes == int(road))], weights[valid & (classes == int(road))])
        np.add.at(sidewalk_weight, found[valid & (classes == int(sidewalk))], weights[valid & (classes == int(sidewalk))])
    cursor.close()
    road_cells, sidewalk_cells = road_weight > 0., sidewalk_weight > 0.
    curb_keys = np.empty(0, dtype='V24')
    if curb is not None and road_cells.any() and sidewalk_cells.any():
        road_tree, walk_tree = cKDTree((unique[road_cells]+.5)*cell_size), cKDTree((unique[sidewalk_cells]+.5)*cell_size)
        points = (unique+.5)*cell_size
        road_distance, road_index = road_tree.query(points, distance_upper_bound=.45)
        walk_distance, walk_index = walk_tree.query(points, distance_upper_bound=.45)
        road_ids, walk_ids = np.flatnonzero(road_cells), np.flatnonzero(sidewalk_cells)
        nearby = np.isfinite(road_distance) & np.isfinite(walk_distance)
        difference = np.zeros(len(unique), np.float64)
        difference[nearby] = np.abs(heights[road_ids[road_index[nearby]]]-heights[walk_ids[walk_index[nearby]]])
        # Prefer an unclassified/ground strip.  If semantic evidence labelled
        # the stone as road or sidewalk it still remains eligible as required.
        candidate = np.isin(codes, [int(ground) if ground is not None else -99, 1, int(road), int(sidewalk)])
        marked_cells = nearby & (difference >= .05) & (difference <= .20)
        marked = candidate & marked_cells[inverse]
        curb_keys = keys[marked]
    changed = False
    if len(curb_keys):
        changed |= _apply_key_target(votes, curb_keys, curb, 'curb', active, immutable, min_observations, checkpoint, result)
    return changed


def _apply_outdoor_objects(votes, by_id, origin, voxel_size, config, outline, active, immutable,
                           min_observations, checkpoint, result):
    target_codes = {_code(by_id, name) for name in
                    ('building', 'outdoor_wall', 'utility_pole', 'traffic_sign', 'fence', 'vehicle', 'power_line')}
    if not any(code is not None and int(code) in active for code in target_codes):
        return False
    prepared = _prepare_outdoor_objects(votes, by_id, origin, voxel_size, config['shape_object_cell'], outline,
                                        min_observations, checkpoint)
    if prepared is None:
        return False
    db, count = votes.db, prepared['count']
    _, shares = _outdoor_vote_shares(db, count, min_observations, checkpoint)
    rows = db.execute('SELECT object,l1,l2,l3,min_z,max_z,x,y,z,barrier,structural FROM shape_outdoor_objects ORDER BY object').fetchall()
    values = np.asarray([row[1:10] for row in rows], np.float64)
    l1, l2, l3, low, high = (values[:, index] for index in range(5))
    barrier = values[:, 8] > .5
    height = high-low
    base_z = float(db.execute('SELECT MIN(iz) FROM geometry_voxels').fetchone()[0] or 0)*float(voxel_size)
    def share(name):
        code = _code(by_id, name)
        return shares.get(code, np.zeros(count)) if code is not None else np.zeros(count)
    target, rule = {}, {}
    min_share = float(config['shape_min_vote_share'])
    steel = _code(by_id, 'steel_structure')
    steel_guard = share('steel_structure') >= min_share if steel is not None else np.zeros(count, bool)
    def choose(mask, name, label, overwrite=False):
        code = _code(by_id, name)
        if code is None or code not in active:
            return
        for obj in np.flatnonzero(mask & ((share(name) >= min_share) | (label == 'outdoor_building' and share('building') >= min_share))):
            if overwrite or int(obj) not in target:
                target[int(obj)], rule[int(obj)] = int(code), label
    # The order gives a sign plate precedence over its otherwise valid pole.
    choose((height >= 3.) & (np.maximum(l2, l3) <= .5), 'utility_pole', 'utility_pole')
    choose((l1 >= 2.) & (height >= .8) & (height <= 2.) & (np.minimum(l2, l3) <= .30), 'fence', 'fence')
    power_mask = (l1 >= 3.) & (l2 <= .30) & (low-base_z >= 4.)
    pole_like = (height >= 3.) & (np.maximum(l2, l3) <= .5)
    centers = values[:, 5:7]
    pole_ids = np.flatnonzero(pole_like)
    if len(pole_ids) and power_mask.any():
        pole_tree = cKDTree(centers[pole_ids])
        for obj, nearby in zip(np.flatnonzero(power_mask),
                               pole_tree.query_ball_point(centers[power_mask], .5)):
            nearby_ids = pole_ids[np.asarray(nearby, np.int64)]
            if len(nearby_ids) and np.any(low[nearby_ids] < low[obj]):
                power_mask[obj] = False
    choose(power_mask, 'power_line', 'power_line')
    choose((l1 <= 8.) & (height <= 3.5), 'vehicle', 'vehicle')
    outdoor_wall = _code(by_id, 'outdoor_wall')
    if outdoor_wall is not None and outdoor_wall in active:
        evidence = (share('outdoor_wall') >= min_share) | (share('wall') >= min_share)
        for obj in np.flatnonzero((height >= .3) & (height <= 3.) & (l3 >= .08) & ~barrier & evidence):
            if int(obj) not in target:
                target[int(obj)], rule[int(obj)] = int(outdoor_wall), 'outdoor_wall'
    building = _code(by_id, 'building')
    if building is not None and building in active:
        facade_evidence = (share('building') >= min_share) | (share('wall') >= min_share)
        for obj in np.flatnonzero((height >= .3) & (height <= 3.) & barrier & facade_evidence):
            target[int(obj)], rule[int(obj)] = int(building), 'outdoor_building'
    choose(~steel_guard, 'building', 'outdoor_building')
    choose(np.ones(count, bool), 'traffic_sign', 'traffic_sign', overwrite=True)
    return _apply_outdoor_targets(votes, target, rule, active, immutable, min_observations, checkpoint, result)


def _tree_stems(votes, by_id, origin, voxel_size, config, min_observations, checkpoint):
    """Find exterior stems from decisions or per-voxel trunk vote evidence."""
    trunk, tree = _code(by_id, 'tree_trunk'), _code(by_id, 'tree')
    if trunk is None:
        return []
    db = votes.db
    db.executescript('''
        DROP TABLE IF EXISTS temp.shape_tree_trunk_votes;
        CREATE TEMP TABLE shape_tree_trunk_votes(
            key BLOB PRIMARY KEY,trunk_weight REAL NOT NULL,total_weight REAL NOT NULL
        ) WITHOUT ROWID;
    ''')
    # Only keys carrying trunk evidence need a total-vote aggregate.  This
    # avoids grouping every exterior voxel through the votes table.
    db.execute('''INSERT INTO shape_tree_trunk_votes(key,trunk_weight,total_weight)
                  SELECT tv.key,tv.weight,SUM(v.weight)
                  FROM votes tv JOIN votes v USING(key)
                  WHERE tv.class=? GROUP BY tv.key''', (int(trunk),))
    has_trunk = (db.execute('SELECT 1 FROM decisions WHERE code=? LIMIT 1', (int(trunk),)).fetchone() is not None
                 or db.execute('SELECT 1 FROM shape_tree_trunk_votes LIMIT 1').fetchone() is not None)
    has_tree = tree is not None and db.execute('SELECT 1 FROM votes WHERE class=? LIMIT 1',
                                                (int(tree),)).fetchone() is not None
    if not has_trunk and not has_tree:
        return []
    index_parts, context_parts = [], []
    cursor = db.execute('''SELECT g.key,g.ix,g.iy,g.iz,d.code,gc.scene_context,
                                  tv.trunk_weight,tv.total_weight
                           FROM geometry_voxels g JOIN decisions d USING(key) JOIN geometry_context gc USING(key)
                           JOIN observations o USING(key) LEFT JOIN shape_tree_trunk_votes tv USING(key)
                           WHERE gc.scene_context IN (1,-1) AND o.count>=?
                             AND (d.code=? OR COALESCE(tv.trunk_weight,0.)>=.30*COALESCE(tv.total_weight,0.))
                           ORDER BY g.key''',
                        (int(min_observations), int(trunk)))
    while batch := cursor.fetchmany(100_000):
        checkpoint()
        index_parts.append(np.asarray([row[1:4] for row in batch], np.int64))
        context_parts.append(np.fromiter((row[5] for row in batch), dtype=np.int8, count=len(batch)))
    cursor.close()
    indices = np.concatenate(index_parts) if index_parts else np.empty((0, 3), np.int64)
    contexts = np.concatenate(context_parts) if context_parts else np.empty(0, np.int8)
    tree_evidence = np.empty((0, 3), np.float64)
    form_only = False
    if len(indices) < 4 and tree is not None:
        evidence_rows = db.execute('''SELECT g.ix,g.iy,g.iz FROM geometry_voxels g JOIN geometry_context gc USING(key)
                                      JOIN observations o USING(key) JOIN votes v USING(key)
                                      WHERE gc.scene_context IN (1,-1) AND o.count>=? AND v.class=?''',
                                   (int(min_observations), int(tree))).fetchall()
        if evidence_rows:
            tree_evidence = np.asarray(evidence_rows, np.float64)
            tree_tree = cKDTree(tree_evidence[:, :2])
            generic_rows = db.execute('''SELECT g.ix,g.iy,g.iz,gc.scene_context
                                         FROM geometry_voxels g JOIN decisions d USING(key) JOIN geometry_context gc USING(key)
                                         JOIN observations o USING(key)
                                         WHERE gc.scene_context IN (1,-1) AND o.count>=? AND d.code=1''',
                                      (int(min_observations),)).fetchall()
            if generic_rows:
                generic_indices = np.asarray([row[:3] for row in generic_rows], np.float64)
                distance, nearest = tree_tree.query(generic_indices[:, :2], distance_upper_bound=float(config['tree_crown_radius_max'])/float(voxel_size))
                above = tree_evidence[np.minimum(nearest, len(tree_evidence)-1), 2] > generic_indices[:, 2]
                take = np.isfinite(distance) & above
                indices = generic_indices[take].astype(np.int64)
                contexts = np.fromiter((row[3] for row in generic_rows), dtype=np.int8, count=len(generic_rows))[take]
                form_only = True
    if len(indices) < 4:
        return []
    cells = _grid_coordinates(indices, origin, voxel_size, config['shape_object_cell'])
    unique = np.unique(cells, axis=0)
    labels = _component_labels(unique)
    low, high = unique.min(axis=0), unique.max(axis=0)
    mapped = labels[np.searchsorted(_linear_keys(unique, low, high-low+1), _linear_keys(cells, low, high-low+1))]
    points = np.asarray(origin, np.float64)+(indices.astype(np.float64)+.5)*float(voxel_size)
    stems = []
    order, starts, stops = _label_group_bounds(mapped)
    for start, stop in zip(starts, stops):
        selected = order[start:stop]
        point = points[selected]
        if len(point) < 4 or np.ptp(point[:, 2]) < .8:
            continue
        centered = point-point.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        diameter = max(np.ptp(point[:, 0]), np.ptp(point[:, 1]))
        if abs(vectors[0, 2]) < .7 or not .1 <= diameter <= 1.5:
            continue
        if not (contexts[selected] == 1).any():
            # An ambiguous stem is allowed only when its observed component
            # has an outdoor anchor; otherwise indoor columns stay untouched.
            continue
        if form_only:
            if not len(tree_evidence):
                continue
            tree_tree = cKDTree(tree_evidence[:, :2])
            distance, nearest = tree_tree.query(indices[selected, :2].mean(axis=0), distance_upper_bound=float(config['tree_crown_radius_max'])/float(voxel_size))
            if not np.isfinite(distance) or tree_evidence[nearest, 2] <= indices[selected, 2].max():
                continue
        stems.append({'x': float(point[:, 0].mean()), 'y': float(point[:, 1].mean()),
                      'top': float(point[:, 2].max()), 'radius': float(diameter*.5)})
    return stems


def _apply_tree_crowns(votes, by_id, origin, voxel_size, config, active, immutable, min_observations,
                       checkpoint, result, artifact_path=None):
    """Apply Domenic's lollipop crown using two measured horizontal slices."""
    tree = _code(by_id, 'tree')
    if tree is None or tree not in active:
        return False
    stems = _tree_stems(votes, by_id, origin, voxel_size, config, min_observations, checkpoint)
    result['tree_seeds'] = len(stems)
    if not stems:
        if artifact_path is not None:
            Path(artifact_path).unlink(missing_ok=True)
        return False
    db = votes.db
    db.execute('DROP TABLE IF EXISTS temp.shape_tree_line_guard')
    db.execute('CREATE TEMP TABLE shape_tree_line_guard(key BLOB PRIMARY KEY) WITHOUT ROWID')
    exists = db.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name='shape_outdoor_objects'").fetchone()
    if exists:
        db.execute('''INSERT OR IGNORE INTO shape_tree_line_guard
                      SELECT ov.key FROM shape_outdoor_voxels ov JOIN shape_outdoor_objects o USING(object)
                      WHERE o.l1>=3. AND o.l2<=.30''')
    trunk = _code(by_id, 'tree_trunk')
    for name in ('utility_pole', 'power_line', 'tree_trunk'):
        code = _code(by_id, name)
        if code is not None:
            db.execute('INSERT OR IGNORE INTO shape_tree_line_guard SELECT key FROM decisions WHERE code=?', (int(code),))
    radius_max, radius_min = float(config['tree_crown_radius_max']), float(config['tree_crown_radius_min'])
    max_height = float(config['tree_crown_max_height'])
    tree_grid_size = max(1, int(np.ceil(radius_max/float(voxel_size))))
    db.executescript('DROP TABLE IF EXISTS temp.shape_tree_grid;')
    db.execute('''CREATE TEMP TABLE shape_tree_grid(
                   gx INTEGER NOT NULL,gy INTEGER NOT NULL,key BLOB NOT NULL,
                   PRIMARY KEY(gx,gy,key)
               ) WITHOUT ROWID''')
    # The tree grid turns one whole-store scan per stem into one sequential
    # materialization plus bounded cell lookups.  TEMP storage is file-backed.
    db.execute('''INSERT INTO shape_tree_grid
                  SELECT CAST(floor(g.ix*1.0/?) AS INTEGER),CAST(floor(g.iy*1.0/?) AS INTEGER),g.key
                  FROM geometry_voxels g JOIN geometry_context gc USING(key) JOIN observations o USING(key)
                  LEFT JOIN shape_tree_line_guard lg USING(key)
                  WHERE gc.scene_context IN (1,-1) AND o.count>=? AND lg.key IS NULL
                    AND NOT EXISTS (SELECT 1 FROM shape_structure_voxels sv WHERE sv.key=g.key)''',
               (tree_grid_size, tree_grid_size, int(min_observations)))
    db.commit()
    selected_key_parts, radii, crowns = [], [], []
    tree_immutable = set(immutable)
    if trunk is not None:
        tree_immutable.add(int(trunk))
    hard = np.asarray(sorted(tree_immutable), np.int64)
    for stem in stems:
        low_index = np.floor((np.asarray((stem['x']-radius_max, stem['y']-radius_max, stem['top']-.05))-np.asarray(origin))/float(voxel_size)).astype(np.int64)
        high_index = np.ceil((np.asarray((stem['x']+radius_max, stem['y']+radius_max, stem['top']+max_height))-np.asarray(origin))/float(voxel_size)).astype(np.int64)
        key_parts, index_parts, code_parts = [], [], []
        grid_low = np.floor_divide(low_index[:2], tree_grid_size)
        grid_high = np.floor_divide(high_index[:2], tree_grid_size)
        for grid_x in range(int(grid_low[0]), int(grid_high[0])+1):
            cursor = db.execute('''SELECT g.key,g.ix,g.iy,g.iz,d.code
                                   FROM shape_tree_grid tg JOIN geometry_voxels g USING(key) JOIN decisions d USING(key)
                                   WHERE tg.gx=? AND tg.gy BETWEEN ? AND ?
                                     AND g.ix BETWEEN ? AND ? AND g.iy BETWEEN ? AND ? AND g.iz BETWEEN ? AND ?''',
                                (grid_x, int(grid_low[1]), int(grid_high[1]),
                                 int(low_index[0]), int(high_index[0]), int(low_index[1]), int(high_index[1]),
                                 int(low_index[2]), int(high_index[2])))
            while batch := cursor.fetchmany(100_000):
                checkpoint()
                key_parts.append(_keys_from_rows(batch))
                index_parts.append(np.asarray([row[1:4] for row in batch], np.int64))
                code_parts.append(np.fromiter((row[4] for row in batch), dtype=np.int64, count=len(batch)))
            cursor.close()
        if not key_parts:
            radii.append(radius_min)
            continue
        keys = np.concatenate(key_parts)
        indices = np.concatenate(index_parts)
        codes = np.concatenate(code_parts)
        order = np.argsort(keys, kind='stable')
        keys, indices, codes = keys[order], indices[order], codes[order]
        points = np.asarray(origin, np.float64)+(indices.astype(np.float64)+.5)*float(voxel_size)
        distance = np.linalg.norm(points[:, :2]-np.asarray((stem['x'], stem['y'])), axis=1)
        relative = points[:, 2]-stem['top']
        slices = []
        for level in (0., 1.):
            take = (np.abs(relative-level) <= .25) & (distance <= radius_max)
            if take.any():
                slices.append(float(np.quantile(distance[take], .90)))
        radius = float(np.clip(max(slices, default=radius_min), radius_min, radius_max))
        radii.append(radius)
        in_crown = (relative > 0.) & (relative <= max_height) & (distance <= radius)
        covered = relative[in_crown]
        crowns.append((stem['x'], stem['y'], radius, stem['top'],
                       stem['top']+float(covered.max()) if len(covered) else stem['top']))
        # Existing crown observations retain the lollipop correction.  The
        # optional fill only extends it into otherwise unclassified voxels.
        take = in_crown & ~np.isin(codes, hard)
        if not bool(config.get('tree_crown_fill', True)):
            take &= codes != 1
        if trunk is not None:
            take &= codes != int(trunk)
        if take.any():
            selected_key_parts.append(keys[take])
    selected_keys = np.unique(np.concatenate(selected_key_parts)) if selected_key_parts else np.empty(0, dtype='V24')
    changed = _apply_key_target(votes, selected_keys, tree, 'tree_crown', active,
                                tree_immutable, min_observations, checkpoint, result, scene_contexts=(1, -1))
    if artifact_path is not None:
        artifact_path = Path(artifact_path)
        if crowns and bool(config.get('tree_crown_fill', True)):
            values = np.asarray(crowns, np.float64)
            np.savez(artifact_path, x=values[:, 0], y=values[:, 1], radius=values[:, 2],
                     z_min=values[:, 3], z_max=values[:, 4])
        else:
            artifact_path.unlink(missing_ok=True)
    result['tree_crown'] = {'stems': len(stems), 'radius_m': radii,
                            'selected_voxels': len(selected_keys),
                            'artifact': None if artifact_path is None else str(artifact_path)}
    return changed


def apply_shape_rules(votes, by_id, origin, voxel_size, config, checkpoint=lambda: None, active_codes=None,
                      scan_context=None, planes_path=None, outline=None):
    """Apply indoor shape rules after segment voting at voxel resolution."""
    started = time.monotonic()
    result = {'enabled': bool(config['shape_rules']), 'objects': 0, 'structural_segments': 0,
              'levels': {'floor_cells': 0, 'ceiling_cells': 0},
              'layer_band': {'sigma_median_m': 0., 'band_median_m': float(config['shape_wall_flush_max'])},
              'rules': {name: {'objects': 0, 'voxels': 0} for name in RULE_NAMES},
              'facade_side': {name: {'planes': 0, 'voxels': 0} for name in
                              ('pair_outer', 'pair_inner', 'context_outdoor', 'context_indoor', 'undecided')},
              'skipped_large_objects': {'objects': 0, 'voxels': 0},
              'changed_by_transition': {}, 'duration_s': 0.,
              'timings': {'load_s': 0., 'layer_sigma_s': 0., 'layer_s': 0., 'candidates_s': 0.,
                          'objects_s': 0., 'object_grid_s': 0., 'object_geometry_s': 0., 'contacts_s': 0.,
                          'object_store_s': 0., 'votes_s': 0., 'shape_metrics_s': 0., 'rules_s': 0., 'apply_s': 0.}}
    if not config['shape_rules']:
        return result
    db = votes.db
    origin = np.asarray(origin, np.float64)
    feature_started = time.monotonic()
    prepare_geometry_features(votes, origin, voxel_size, config, checkpoint)
    refresh_geometry_decisions(votes)
    context = ensure_scene_context(votes, by_id, scan_context, origin, voxel_size, config, checkpoint,
                                   outline=outline)
    result['scene_outline'] = context.get('scene_outline', {})
    result['timings']['features_s'] = time.monotonic()-feature_started
    load_started = time.monotonic()
    has_segments = db.execute("SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='segments'").fetchone() is not None
    structural_ransac = None
    if planes_path is not None and Path(planes_path).is_file():
        structural_ransac = _load_structural_planes_from_npz(planes_path, config, scan_context)
    if not has_segments and structural_ransac is None:
        result['duration_s'] = time.monotonic()-started
        return result
    loaded, segment_data = None, None
    if has_segments:
        loaded = _load_cell_arrays(db, checkpoint)
        segment_data = _segment_arrays(db)
    result['timings']['load_s'] = time.monotonic()-load_started
    wall, floor, ceiling = _code(by_id, 'wall'), _code(by_id, 'floor'), _code(by_id, 'ceiling')
    ground, building, road, sidewalk = (_code(by_id, name) for name in ('ground', 'building', 'road', 'sidewalk'))
    structural_codes = {code for code in (wall, floor, ceiling, ground, building, road, sidewalk) if code is not None}

    _create_tables(db)
    layer_started = time.monotonic()
    use_ransac = structural_ransac is not None
    if use_ransac:
        planes = structural_ransac
        result['structural_segments'] = int(planes['valid'].sum())
        protected_codes = {int(by_id[name]) for name in config['segment_protected_classes'] if name in by_id}
        trunk = _code(by_id, 'tree_trunk')
        protected_codes.update(code for code in (_code(by_id, 'reflection'), _code(by_id, 'stairs'), trunk) if code is not None)
        sigma_started = time.monotonic()
        _estimate_plane_bands(db, planes, origin, voxel_size, config['shape_wall_flush_max'],
                              config['shape_layer_sigma_factor'], config['shape_layer_max_band'], checkpoint)
        result['timings']['layer_sigma_s'] = time.monotonic()-sigma_started
        # The wall band is the quantity used by the flush/protrusion rule.
        # Reporting its robust median avoids clean floor and ceiling planes
        # hiding a noisy wall behind a global 3-cm median.
        wall_bands = (planes['kinds'] == 1) & planes['valid']
        selected_bands = wall_bands if wall_bands.any() else planes['valid']
        result['layer_band'] = {
            'sigma_median_m': float(np.median(planes['sigma'][selected_bands])),
            'band_median_m': float(np.median(planes['band'][selected_bands])),
            'band_max_m': float(planes['band'][selected_bands].max()),
            'wall_planes': int(wall_bands.sum()),
        }
        _materialize_structure_layer_ransac(db, planes, voxel_size, origin, protected_codes, checkpoint)
    else:
        if loaded is None or segment_data is None:
            result['duration_s'] = time.monotonic()-started
            return result
        winners, normals = segment_data['winner'], segment_data['normal']
        planar = np.linalg.norm(normals, axis=1) > .5
        horizontal = np.abs(normals[:, 2]) >= np.cos(np.deg2rad(10.))
        area = segment_data['cells']*float(config['geometry_cell_size'])**2
        width = np.linalg.norm((segment_data['upper']-segment_data['lower'])[:, :2], axis=1)
        structural_codes = {code for code in (wall, floor, ceiling, ground, building, road, sidewalk) if code is not None}
        structural = planar & np.isin(winners, list(structural_codes)) & (
            ((np.abs(normals[:, 2]) <= .2) & (width > float(config['shape_structural_width']))) |
            ((np.abs(normals[:, 2]) > .2) & (area >= float(config['shape_structural_area']))))
        structural = _demote_parallel_wall_fronts(structural, winners, wall, segment_data, config['shape_cabinet_offset'])
        kinds = np.zeros(len(structural), np.int8)
        if wall is not None:
            kinds[structural & (winners == wall)] = 1
        floor_codes = [code for code in (floor, ground) if code is not None]
        if floor_codes:
            kinds[structural & np.isin(winners, floor_codes)] = 2
        if ceiling is not None:
            kinds[structural & (winners == ceiling)] = 3
        result['structural_segments'] = int(structural.sum())
        planes = _fit_structural_planes(db, structural, kinds, segment_data, origin, voxel_size,
                                        float(config['shape_wall_flush_max']), scan_context, checkpoint)
        _materialize_structure_layer(db, planes, kinds, voxel_size, float(config['shape_wall_flush_max']), origin, checkpoint)
    result['timings']['layer_s'] = time.monotonic()-layer_started
    candidates_started = time.monotonic()
    _materialize_object_candidates(
        db, structural_codes, origin, voxel_size, config['shape_object_cell'], config['shape_object_max_size'], checkpoint)
    result['timings']['candidates_s'] = time.monotonic()-candidates_started
    object_started = time.monotonic()
    grid_started = time.monotonic()
    grid = _build_object_grid(db, origin, voxel_size, float(config['shape_object_cell']), checkpoint)
    result['timings']['object_grid_s'] = time.monotonic()-grid_started
    objects = None
    if grid is not None:
        geometry_started = time.monotonic()
        objects = _object_geometry(db, grid, origin, voxel_size, checkpoint)
        result['timings']['object_geometry_s'] = time.monotonic()-geometry_started
        contacts_started = time.monotonic()
        touches = _materialize_contacts(db, origin, voxel_size, float(config['shape_object_cell']), objects['count'], checkpoint)
        result['timings']['contacts_s'] = time.monotonic()-contacts_started
        store_started = time.monotonic()
        _store_objects(db, objects, touches)
        result['timings']['object_store_s'] = time.monotonic()-store_started
        result['objects'] = objects['count']
    result['timings']['objects_s'] = time.monotonic()-object_started
    active = {int(code) for code in (by_id.values() if active_codes is None else active_codes)}
    protected = {int(by_id[name]) for name in config['segment_protected_classes'] if name in by_id}
    reflection, stairs = _code(by_id, 'reflection'), _code(by_id, 'stairs')
    clutter = _code(by_id, 'construction_material', 'clutter', 'material')
    furniture = _code(by_id, 'furniture')
    source_layer = {code for code in (clutter, furniture, 1) if code is not None}
    changed = False
    apply_started = time.monotonic()
    trunk = _code(by_id, 'tree_trunk')
    immutable = protected | {int(code) for code in (reflection, stairs, trunk) if code is not None}
    result['timings']['facade_apply_s'] = result['timings']['object_apply_s'] = 0.
    result['timings']['layer_apply_s'] = result['timings']['paved_apply_s'] = 0.
    result['timings']['outdoor_apply_s'] = result['timings']['tree_apply_s'] = 0.
    if use_ransac:
        rule_started = time.monotonic()
        changed |= _apply_facade_sides(votes, planes, outline, by_id, origin, voxel_size, active,
                                       config['min_observations'], config['building_boundary_min_height'],
                                       config['shape_structural_area'], config['shape_wall_flush_max'], scan_context,
                                       checkpoint, result)
        result['timings']['facade_apply_s'] = time.monotonic()-rule_started
    indoor_rule_targets = {_code(by_id, name) for name in (
        'floor', 'wall', 'column', 'furniture', 'beam', 'pipe', 'hvac', 'cable',
        'ceiling_equipment', 'sanitary_fixtures', 'railing', 'construction_material')}
    level_free_targets = {_code(by_id, name) for name in
                          ('pipe', 'hvac', 'cable', 'ceiling_equipment', 'sanitary_fixtures', 'construction_material')}
    has_local_level_source = (not use_ransac or np.any(np.isin(planes['kinds'], (2, 3))))
    can_apply_indoor_objects = (any(code is not None and int(code) in active for code in indoor_rule_targets) and
                                (has_local_level_source or
                                 any(code is not None and int(code) in active for code in level_free_targets)))
    if objects is not None and can_apply_indoor_objects:
        object_apply_started = time.monotonic()
        level_started = time.monotonic()
        floors, ceilings = None, None
        if use_ransac:
            floor_count = int(sum(len(planes['occupied'][index]) for index in np.flatnonzero(planes['kinds'] == 2)))
            ceiling_count = int(sum(len(planes['occupied'][index]) for index in np.flatnonzero(planes['kinds'] == 3)))
        elif loaded is not None and segment_data is not None:
            _, cell_segments, cell_points = loaded
            winners, normals = segment_data['winner'], segment_data['normal']
            planar = np.linalg.norm(normals, axis=1) > .5
            horizontal = np.abs(normals[:, 2]) >= np.cos(np.deg2rad(10.))
            area = segment_data['cells']*float(config['geometry_cell_size'])**2
            eligible_level = planar & horizontal & (area >= float(config['shape_level_min_area']))
            floor_codes = [code for code in (floor, ground) if code is not None]
            floor_segments = eligible_level & np.isin(winners, floor_codes)
            ceiling_segments = eligible_level & (winners == ceiling if ceiling is not None else False)
            level_cell = float(config['shape_level_cell'])
            level_grid = np.floor(cell_points[:, :2]/level_cell).astype(np.int64)
            floors = _padded_levels(level_grid[floor_segments[cell_segments]], cell_points[floor_segments[cell_segments], 2], level_cell)
            ceilings = _padded_levels(level_grid[ceiling_segments[cell_segments]], cell_points[ceiling_segments[cell_segments], 2], level_cell)
            floor_count, ceiling_count = 0 if floors is None else int(len(floors[0])), 0 if ceilings is None else int(len(ceilings[0]))
        else:
            floor_count, ceiling_count = 0, 0
        result['levels'] = {'floor_cells': floor_count, 'ceiling_cells': ceiling_count}
        lower_point, upper_point = objects['lower'].copy(), objects['upper'].copy()
        lower_point[:, :2] = objects['center'][:, :2]; upper_point[:, :2] = objects['center'][:, :2]
        if use_ransac:
            local_floor = _plane_level_at(lower_point, planes, 2, config['shape_level_search'], True)
            local_ceiling = _plane_level_at(upper_point, planes, 3, config['shape_level_search'], False)
        else:
            local_floor = _level_at(lower_point, floors, config['shape_level_search'], True)
            local_ceiling = _level_at(upper_point, ceilings, config['shape_level_search'], False)
        above_floor = objects['lower'][:, 2]-local_floor
        room_height, top_gap = local_ceiling-local_floor, local_ceiling-objects['upper'][:, 2]
        result['timings']['levels_s'] = time.monotonic()-level_started
        steel, column = _code(by_id, 'steel_structure'), _code(by_id, 'column')
        beam = _code(by_id, 'beam')
        pipe, cable = _code(by_id, 'pipe'), _code(by_id, 'cable')
        electrical, hvac = _code(by_id, 'electrical'), _code(by_id, 'hvac')
        window, door = _code(by_id, 'window'), _code(by_id, 'door')
        sanitary, equipment, appliances = (_code(by_id, 'sanitary_fixtures', 'sanitary'), _code(by_id, 'ceiling_equipment'), _code(by_id, 'appliances'))
        railing = _code(by_id, 'railing')
        votes_started = time.monotonic()
        totals, object_winners, _, weights, slots = _object_votes(
            db, objects['count'], (wall, floor, ceiling, ground, building, column, furniture, beam, clutter, steel,
                                    pipe, cable, electrical, hvac, window, door, sanitary, equipment, appliances, railing),
            config['min_observations'], checkpoint)
        result['timings']['votes_s'] = time.monotonic()-votes_started
        def share(code):
            return weights[:, slots[code]]/np.maximum(totals, 1e-12) if code in slots else np.zeros(objects['count'])
        object_share = np.zeros(objects['count'])
        for code in np.unique(object_winners):
            if code in slots:
                object_share[object_winners == code] = share(int(code))[object_winners == code]
        if use_ransac:
            wall_ids = np.flatnonzero((planes['kinds'] == 1) & planes['valid'])
        else:
            wall_ids = np.flatnonzero(structural & (winners == wall if wall is not None else False) & planes['valid'])
        if np.isfinite(above_floor).any():
            protrusion, wall_distance, wall_band = _wall_metrics(db, objects, planes, wall_ids, origin, voxel_size, checkpoint)
        else:
            protrusion = np.full(objects['count'], np.inf)
            wall_distance, wall_band = np.full(objects['count'], np.inf), np.full(objects['count'], np.inf)
        rule_started = time.monotonic()
        count, min_share = objects['count'], float(config['shape_min_vote_share'])
        too_large = objects['extent'][:, 0] > float(config['shape_object_max_size'])
        result['skipped_large_objects'] = {'objects': int(too_large.sum()),
                                           'voxels': int(objects['voxels'][too_large].sum())}
        steel_guard = (steel is not None) & (object_winners == steel) & (share(steel) >= min_share)
        target, rule = np.full(count, -1, np.int64), np.full(count, '', dtype=object)
        vertical, world_span = np.abs(objects['axis'][:, 2]) >= .7, objects['upper']-objects['lower']
        height = world_span[:, 2]
        tech_codes = tuple(code for code in (pipe, cable, electrical, hvac, sanitary, equipment, railing) if code is not None)
        technical_evidence = sum((share(code) for code in tech_codes), np.zeros(count)) >= min_share
        column_base = vertical & (world_span[:, 0] < float(config['shape_column_max_size'])) & (world_span[:, 1] < float(config['shape_column_max_size'])) & np.isfinite(room_height) & (height >= float(config['shape_column_min_height_share'])*room_height)
        if column is not None:
            take = column_base & ~touches[1] & ~steel_guard & (share(column) >= min_share)
            target[take], rule[take] = column, 'column'
        if wall is not None:
            take = column_base & touches[1] & ~steel_guard
            target[take], rule[take] = wall, 'pilaster'
        cabinet = (vertical & np.isfinite(above_floor) & (above_floor <= .15) &
                   (wall_distance >= config['shape_cabinet_offset'][0]) & (wall_distance <= config['shape_cabinet_offset'][1]) &
                   ~technical_evidence)
        if furniture is not None:
            take = cabinet & (share(furniture) >= min_share) & (target < 0)
            target[take], rule[take] = furniture, 'cabinet'
        beam_mask = ((objects['extent'][:, 0] >= float(config['shape_beam_min_length'])) &
                     (objects['extent'][:, 1] <= float(config['shape_beam_max_width'])) &
                     (objects['extent'][:, 2] <= float(config['shape_beam_max_width'])) &
                     np.isfinite(top_gap) & (top_gap <= float(config['shape_beam_ceiling_gap'])) & ~steel_guard &
                     ~technical_evidence)
        if beam is not None:
            take = beam_mask & (share(beam) >= min_share) & (target < 0)
            target[take], rule[take] = beam, 'beam'
        excluded = np.isin(object_winners, [code for code in (electrical, hvac, window, door, sanitary, equipment, appliances) if code is not None])
        wall_item = (touches[1] & np.isfinite(above_floor) & (above_floor >= .1) & ~touches[2] &
                     np.isin(object_winners, [code for code in (clutter, furniture, wall, 1) if code is not None]) & ~excluded &
                     ~technical_evidence)
        # The layer band is already a robust uncertainty envelope.  Comparing
        # against max(3 cm, band), instead of adding both values, prevents
        # counting the same SLAM noise twice while retaining Domenic's 3-cm
        # minimum for clean planes.
        protruding = wall_item & (protrusion > np.maximum(float(config['shape_wall_flush_max']), wall_band))
        unknown = protruding & ((objects['extent'][:, 2] < float(config['shape_wall_cabinet_min_depth'])) | (height < .4))
        take = unknown & (target < 0)
        target[take], rule[take] = 1, 'wall_item'
        if furniture is not None:
            take = protruding & ~unknown & (target < 0)
            target[take], rule[take] = furniture, 'wall_item'
        if sanitary is not None:
            take = (share(sanitary) >= .30) & (target < 0) & ~steel_guard
            target[take], rule[take] = sanitary, 'sanitary'
        metrics_started = time.monotonic()
        surface_evidence = (share(pipe)+share(hvac) >= min_share)
        planar_mantle = _object_mantle_planarity(
            db, objects, np.flatnonzero(surface_evidence), config['shape_tech_planar_fraction'], checkpoint)
        result['timings']['shape_metrics_s'] = time.monotonic()-metrics_started
        cross_a, cross_b = objects['extent'][:, 1], objects['extent'][:, 2]
        round_pipe = ((objects['extent'][:, 0] >= float(config['shape_pipe_min_length'])) &
                      (cross_a >= float(config['shape_pipe_min_diameter'])) &
                      (cross_b >= float(config['shape_pipe_min_diameter'])) &
                      (cross_a <= float(config['shape_pipe_max_diameter'])) &
                      (cross_b <= float(config['shape_pipe_max_diameter'])) &
                      (cross_a <= 2.*np.maximum(cross_b, 1e-12)) &
                      (planar_mantle < .50))
        if pipe is not None:
            # A sprinkler head makes the aggregate cross-section locally
            # box-like.  Pipe plus ceiling-equipment evidence on one long
            # object is the catalogue's explicit sprinkler exception.
            sprinkler_pipe = ((objects['extent'][:, 0] >= float(config['shape_pipe_min_length'])) &
                              (share(pipe) >= min_share) & (share(equipment) >= min_share))
            take = ((round_pipe & (share(pipe)+share(hvac) >= min_share)) | sprinkler_pipe) & (target < 0) & ~steel_guard
            target[take], rule[take] = pipe, 'pipe'
        rectangular_duct = ((objects['extent'][:, 0] >= float(config['shape_duct_min_length'])) &
                            (cross_a >= float(config['shape_duct_min_cross_section'])) &
                            (cross_b >= float(config['shape_duct_min_cross_section'])) &
                            (planar_mantle >= .50))
        if hvac is not None:
            take = rectangular_duct & (share(hvac) >= min_share) & (target < 0) & ~steel_guard
            target[take], rule[take] = hvac, 'hvac_duct'
            radiator = (touches[1] & np.isfinite(above_floor) & (above_floor >= .05) & (above_floor <= .25) &
                        (protrusion >= .05) & (protrusion <= .30) & (height >= .30) &
                        (np.maximum(world_span[:, 0], world_span[:, 1]) >= float(config['shape_radiator_min_width'])))
            take = radiator & (share(hvac) >= min_share) & (target < 0) & ~steel_guard
            target[take], rule[take] = hvac, 'radiator'
        thin_cable = ((objects['extent'][:, 0] >= float(config['shape_pipe_min_length'])) &
                      (cross_a < float(config['shape_pipe_min_diameter'])))
        cable_tray = ((objects['extent'][:, 0] >= float(config['shape_duct_min_length'])) &
                      (cross_a >= .10) & (cross_a <= .60) & (cross_b < .15) & (touches[1] | touches[3]))
        if cable is not None:
            take = (thin_cable | cable_tray) & (share(cable)+share(electrical) >= min_share) & (target < 0) & ~steel_guard
            target[take], rule[take] = cable, 'cable'
        ceiling_drop = local_ceiling-objects['lower'][:, 2]
        if equipment is not None:
            take = (touches[3] & np.isfinite(ceiling_drop) &
                    (ceiling_drop <= float(config['shape_ceiling_equipment_max_drop'])) &
                    (objects['extent'][:, 0] < 1.5) & (share(equipment) >= min_share) &
                    (target < 0) & ~steel_guard)
            target[take], rule[take] = equipment, 'ceiling_equipment'
        if railing is not None:
            top_above_floor = objects['upper'][:, 2]-local_floor
            take = ((objects['extent'][:, 0] >= float(config['shape_railing_min_length'])) & ~vertical &
                    np.isfinite(top_above_floor) & (top_above_floor >= .80) & (top_above_floor <= 1.20) &
                    (share(railing) >= min_share) & (target < 0) & ~steel_guard)
            target[take], rule[take] = railing, 'railing'
        if clutter is not None and furniture is not None:
            mixed = (share(clutter)+share(furniture) >= .3) & (object_winners != appliances if appliances is not None else True)
            take = mixed & (target < 0) & ~steel_guard & ~technical_evidence & (objects['extent'][:, 0] < float(config['shape_clutter_max_size']))
            target[take], rule[take] = clutter, 'furniture_clutter'
            take = mixed & (target < 0) & ~steel_guard & ~technical_evidence
            target[take], rule[take] = furniture, 'furniture_clutter'
        target_share = np.zeros(count)
        for code in np.unique(target[target >= 0]):
            if code in slots:
                target_share[target == code] = share(int(code))[target == code]
        if pipe is not None:
            take = rule == 'pipe'
            target_share[take] = (share(pipe)+share(hvac))[take]
        if hvac is not None:
            take = rule == 'hvac_duct'
            target_share[take] = (share(hvac)+share(pipe))[take]
        if cable is not None:
            take = rule == 'cable'
            target_share[take] = (share(cable)+share(electrical))[take]
        inactive_target = (target >= 0) & (target != 1) & ~np.isin(target, np.asarray(sorted(active), np.int64))
        target[inactive_target] = -1
        rule[inactive_target] = ''
        regular = ~np.isin(rule, ('pilaster', 'wall_item', 'furniture_clutter', 'sanitary')) & (target >= 0) & (target != 1)
        target[regular & (target_share < min_share)] = -1
        target[too_large] = -1
        rule[too_large] = ''
        source_by_rule = {
            'carpet': source_layer,
            'column': {code for code in (wall, building, clutter, furniture, 1) if code is not None},
            'pilaster': {code for code in (column, 1) if code is not None},
            'cabinet': {code for code in (wall, 1) if code is not None},
            'beam': {code for code in (ceiling, wall, clutter, 1) if code is not None},
            # Sprinkler heads are commonly voted as ceiling equipment.  They
            # are part of the same G2 object as their pipe and are therefore
            # included here so the complete sprinkler is labelled pipe.
            'pipe': {code for code in (hvac, equipment, clutter, wall, ceiling, beam, 1) if code is not None},
            'hvac_duct': {code for code in (pipe, clutter, wall, ceiling, beam, 1) if code is not None},
            'radiator': {code for code in (clutter, furniture, wall, 1) if code is not None},
            'cable': {code for code in (clutter, pipe, electrical, wall, ceiling, 1) if code is not None},
            'ceiling_equipment': {code for code in (ceiling, clutter, furniture, 1) if code is not None},
            'sanitary': {code for code in (clutter, furniture, window, wall, 1) if code is not None},
            'railing': {code for code in (clutter, furniture, wall, window, 1) if code is not None},
            'wall_item': {code for code in (clutter, furniture, wall, 1) if code is not None},
            'furniture_clutter': {code for code in (clutter, furniture, 1) if code is not None},
        }
        result['timings']['rules_s'] = time.monotonic()-rule_started
        changed |= _apply_object_targets(votes, target, rule, target_share, active, protected, reflection, stairs,
                                         source_by_rule, config['min_observations'], checkpoint, result,
                                         {'sanitary': {window} if window is not None else set()})
        if clutter in slots and furniture in slots and np.any((weights[:, slots[clutter]] > 0) &
                                                               (weights[:, slots[furniture]] > 0)):
            changed |= _apply_supported_small_items(votes, clutter, furniture, origin, voxel_size,
                                                    float(config['shape_object_cell']), active, protected, reflection, stairs,
                                                    config['min_observations'], checkpoint, result)
        result['timings']['object_apply_s'] = time.monotonic()-object_apply_started
    if floor is not None:
        layer_apply_started = time.monotonic()
        changed |= _apply_layer_target(votes, floor, (2,), source_layer, active, protected, reflection, stairs,
                                       config['min_observations'], checkpoint, result, 'carpet')
        result['timings']['layer_apply_s'] += time.monotonic()-layer_apply_started
    if wall is not None:
        layer_apply_started = time.monotonic()
        wall_layer_source = source_layer | ({_code(by_id, 'column')} if _code(by_id, 'column') is not None else set())
        changed |= _apply_layer_target(votes, wall, (1,), wall_layer_source, active, protected, reflection, stairs,
                                       config['min_observations'], checkpoint, result, 'wall_item')
        result['timings']['layer_apply_s'] += time.monotonic()-layer_apply_started
    paved_apply_started = time.monotonic()
    changed |= _apply_curb_and_sidewalk(votes, by_id, voxel_size, active, immutable,
                                        config['min_observations'], checkpoint, result)
    result['timings']['paved_apply_s'] = time.monotonic()-paved_apply_started
    outdoor_apply_started = time.monotonic()
    changed |= _apply_outdoor_objects(votes, by_id, origin, voxel_size, config, outline, active, immutable,
                                      config['min_observations'], checkpoint, result)
    result['timings']['outdoor_apply_s'] = time.monotonic()-outdoor_apply_started
    crown_path = (Path(planes_path).with_name('tree_crowns.npz') if planes_path is not None
                  else Path(votes.path).with_name('tree_crowns.npz'))
    tree_apply_started = time.monotonic()
    changed |= _apply_tree_crowns(votes, by_id, origin, voxel_size, config, active, immutable,
                                  config['min_observations'], checkpoint, result, crown_path)
    result['timings']['tree_apply_s'] = time.monotonic()-tree_apply_started
    result['timings']['apply_s'] = time.monotonic()-apply_started
    if changed:
        refresh_geometry_decisions(votes)
        votes._invalidate_lookup()
    result['timings']['context_s'] = context['duration_s']
    result['duration_s'] = time.monotonic()-started
    return result
