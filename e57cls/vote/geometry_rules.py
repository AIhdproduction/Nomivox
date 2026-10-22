"""Bounded geometric plausibility rules and terrain evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from e57cls.vote.scene_context import ensure_scene_context


def _cell_key(indices, voxel_size, cell_size):
    return np.floor(np.asarray(indices, np.float64)*voxel_size/cell_size).astype(np.int64)


class TerrainIndex:
    """Read-only DTM and overhead-evidence lookup used during export."""

    def __init__(self, path):
        self._archive = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self._archive['metadata']))
        self.xy = np.asarray(self._archive['xy'], np.float64)
        self.z = np.asarray(self._archive['z'], np.float64)
        self.covered = np.asarray(self._archive['covered'], bool)
        self.tree = cKDTree(self.xy) if len(self.xy) else None

    def ground_mask(self, points, tolerance):
        result = np.zeros(len(points), bool)
        if not len(points) or self.tree is None:
            return result
        distance, ids = self.tree.query(np.asarray(points)[:, :2], k=1,
                                        distance_upper_bound=self.meta.get('max_gap', self.meta['cell_size']*.8), workers=-1)
        valid = np.isfinite(distance) & (ids < len(self.xy))
        result[valid] = (~self.covered[ids[valid]]
                         & (np.abs(np.asarray(points)[valid, 2]-self.z[ids[valid]]) <= tolerance))
        return result

    def close(self):
        self._archive.close()


def _features(points, k, radius):
    """Return local shape descriptors for the supplied feature-cell centres."""
    count = len(points)
    # The final three values retain the oriented local normal.  Keeping it
    # with the existing descriptors lets all geometric post-processing share
    # one bounded neighbourhood calculation.
    result = np.zeros((count, 7), np.float32)
    if not count:
        return result
    tree = cKDTree(points)
    distances, neighbours = tree.query(points, k=min(k, count), distance_upper_bound=radius, workers=-1)
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
        distances = distances[:, None]
    for index, (distance, ids) in enumerate(zip(distances, neighbours)):
        valid = np.isfinite(distance) & (ids < count)
        local = points[ids[valid]]
        if len(local) < 4:
            continue
        centered = local-local.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T@centered/max(1, len(local)-1))
        values = np.maximum(values[::-1], 0.)
        if values[0] <= 1e-12:
            continue
        normal = vectors[:, 0]
        # Plane normals have no inherent direction.  Use one deterministic
        # hemisphere so adjacent planar feature cells can be compared.
        if (normal[2] < 0 or (normal[2] == 0 and (normal[0] < 0
                or (normal[0] == 0 and normal[1] < 0)))):
            normal = -normal
        result[index, :4] = ((values[0]-values[1])/values[0],
                              (values[1]-values[2])/values[0],
                              values[2]/values.sum() if values.sum() else 0., abs(normal[2]))
        result[index, 4:] = normal
    return result


def _prepare_tables(votes, voxel_size, cell_size, tile_size, checkpoint):
    db = votes.db
    db.executescript("""
        DROP TABLE IF EXISTS temp.geometry_voxels;
        DROP TABLE IF EXISTS temp.geometry_cells;
        DROP TABLE IF EXISTS temp.geometry_features;
        DROP TABLE IF EXISTS temp.geometry_alternatives;
        CREATE TEMP TABLE geometry_voxels(
            key BLOB PRIMARY KEY, ix INTEGER, iy INTEGER, iz INTEGER,
            cx INTEGER, cy INTEGER, cz INTEGER, code INTEGER, confidence INTEGER
        ) WITHOUT ROWID;
        CREATE TEMP TABLE geometry_cells(
            cx INTEGER, cy INTEGER, cz INTEGER, tx INTEGER, ty INTEGER, tz INTEGER,
            sx REAL, sy REAL, sz REAL, n INTEGER, PRIMARY KEY(cx,cy,cz)
        ) WITHOUT ROWID;
        CREATE INDEX geometry_cells_tiles ON geometry_cells(tx,ty,tz);
        CREATE TEMP TABLE geometry_features(
            cx INTEGER, cy INTEGER, cz INTEGER, x REAL, y REAL, z REAL,
            linearity REAL, planarity REAL, scatter REAL, verticality REAL,
            nx REAL DEFAULT 0, ny REAL DEFAULT 0, nz REAL DEFAULT 0,
            hag REAL, covered INTEGER DEFAULT 0, PRIMARY KEY(cx,cy,cz)
        ) WITHOUT ROWID;
        CREATE TEMP TABLE geometry_meta(name TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID;
        CREATE INDEX geometry_voxels_cells ON geometry_voxels(cx,cy,cz);
    """)
    cursor = db.execute('SELECT key,code,confidence FROM decisions ORDER BY key')
    ratio = voxel_size/cell_size
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        keys = [bytes(row[0]) for row in rows]
        indices = np.frombuffer(b''.join(keys), dtype='<i8').reshape(-1, 3)
        cells = np.floor(indices*ratio).astype(np.int64)
        db.executemany('INSERT INTO geometry_voxels VALUES (?,?,?,?,?,?,?,?,?)',
                       ((key, *map(int, point), *map(int, cell), int(code), int(confidence))
                        for key, point, cell, (_, code, confidence) in zip(keys, indices, cells, rows)))
        tiles = np.floor(cells*cell_size/tile_size).astype(np.int64)
        db.executemany("""INSERT INTO geometry_cells VALUES (?,?,?,?,?,?,?,?,?,1)
            ON CONFLICT(cx,cy,cz) DO UPDATE SET sx=sx+excluded.sx,sy=sy+excluded.sy,sz=sz+excluded.sz,n=n+1""",
            ((int(cell[0]), int(cell[1]), int(cell[2]), int(tile[0]), int(tile[1]), int(tile[2]),
              float(point[0]), float(point[1]), float(point[2])) for point, cell, tile in zip(indices, cells, tiles)))
    db.commit()


def _make_features(votes, origin, voxel_size, config, checkpoint):
    db = votes.db
    cell_size, tile_size, halo = config['geometry_cell_size'], config['geometry_tile_size'], config['geometry_halo']
    tile_cells, halo_cells = max(1, round(tile_size/cell_size)), int(np.ceil(halo/cell_size))
    tiles = db.execute('SELECT DISTINCT tx,ty,tz FROM geometry_cells ORDER BY tx,ty,tz').fetchall()
    for tile_number, (tx, ty, tz) in enumerate(tiles):
        checkpoint()
        low = np.asarray((tx, ty, tz), np.int64)*tile_cells-halo_cells
        high = (np.asarray((tx, ty, tz), np.int64)+1)*tile_cells+halo_cells-1
        rows = db.execute('''SELECT cx,cy,cz,sx/n,sy/n,sz/n FROM geometry_cells
                           WHERE cx BETWEEN ? AND ? AND cy BETWEEN ? AND ? AND cz BETWEEN ? AND ?''',
                          (int(low[0]), int(high[0]), int(low[1]), int(high[1]), int(low[2]), int(high[2]))).fetchall()
        if not rows:
            continue
        coordinates = np.asarray([row[3:] for row in rows], np.float64)
        world = origin+(coordinates+.5)*voxel_size
        descriptors = _features(world, config['geometry_knn'], config['geometry_knn_radius'])
        core = [(row, point, values) for row, point, values in zip(rows, world, descriptors)
                if tuple(np.floor(np.asarray(row[:3])*cell_size/tile_size).astype(np.int64)) == (tx, ty, tz)]
        db.executemany('INSERT OR REPLACE INTO geometry_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)',
            ((int(row[0]), int(row[1]), int(row[2]), *map(float, point), *map(float, values), None)
             for row, point, values in core))
        if tile_number % 16 == 0:
            db.commit()
    db.commit()


def _geometry_signature(origin, voxel_size, config):
    """Return the feature parameters that must match for a TEMP-table reuse."""
    return json.dumps({'origin': np.asarray(origin, np.float64).tolist(), 'voxel_size': float(voxel_size),
                       'cell_size': float(config['geometry_cell_size']),
                       'tile_size': float(config['geometry_tile_size']), 'halo': float(config['geometry_halo']),
                       'knn': int(config['geometry_knn']), 'radius': float(config['geometry_knn_radius'])},
                      sort_keys=True, separators=(',', ':'))


def prepare_geometry_features(votes, origin, voxel_size, config, checkpoint=lambda: None):
    """Prepare feature tables once per compatible post-processing chain."""
    signature = _geometry_signature(origin, voxel_size, config)
    row = votes.db.execute("SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='geometry_meta'").fetchone()
    cached = (row is not None and votes.db.execute("SELECT value FROM geometry_meta WHERE name='signature'").fetchone())
    if cached and cached[0] == signature:
        return False
    _prepare_tables(votes, voxel_size, config['geometry_cell_size'], config['geometry_tile_size'], checkpoint)
    _make_features(votes, np.asarray(origin, np.float64), voxel_size, config, checkpoint)
    votes.db.execute("INSERT INTO geometry_meta VALUES ('signature',?)", (signature,))
    votes.db.commit()
    return True


def refresh_geometry_decisions(votes):
    """Reflect changed decisions in already prepared feature voxels in SQL."""
    exists = votes.db.execute("SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='geometry_voxels'").fetchone()
    if exists:
        votes.db.execute('''UPDATE geometry_voxels AS g SET code=d.code,confidence=d.confidence
                            FROM decisions AS d WHERE d.key=g.key''')
        votes.db.commit()


def _terrain(votes, origin, config, target, checkpoint, by_id):
    db = votes.db
    size = config['terrain_cell_size']
    rows = db.execute('''SELECT floor((x-?)/?) AS gx,floor((y-?)/?) AS gy,MIN(z)
                       FROM geometry_features WHERE planarity >= .35 AND verticality >= .80
                       GROUP BY gx,gy''', (float(origin[0]), size, float(origin[1]), size)).fetchall()
    if not rows:
        xy, heights, covered = np.empty((0, 2), np.float64), np.empty(0, np.float64), np.empty(0, bool)
    else:
        grid = np.asarray([row[:2] for row in rows], np.int64)
        heights = np.asarray([row[2] for row in rows], np.float64)
        xy = origin[:2]+(grid+.5)*size
        # A DTM gap is filled only from locally compatible terrain. This keeps
        # the nearest-cell lookup from bridging cliffs or separate storeys.
        if len(xy) > 1:
            nearby_distance, nearby = cKDTree(xy).query(xy, k=min(5, len(xy)), workers=-1)
            slope = np.divide(np.abs(heights[:, None]-heights[nearby]), nearby_distance,
                              out=np.full_like(nearby_distance, np.inf), where=nearby_distance > 1e-9)
            keep = np.any(slope <= config['terrain_max_slope'], axis=1)
            xy, heights = xy[keep], heights[keep]
            grid = grid[keep]
        if not len(xy):
            np.savez_compressed(target, metadata=np.asarray(json.dumps({'schema_version': 1, 'cell_size': size,
                'max_gap': config['terrain_max_gap'], 'origin': np.asarray(origin, np.float64).tolist()})),
                xy=np.empty((0, 2)), z=np.empty(0), covered=np.empty(0, bool))
            return {'artifact': str(target), 'cells': 0, 'covered_cells': 0}
        tree = cKDTree(xy)
        covered = np.zeros(len(xy), bool)
        feature_rows = db.execute('''SELECT f.cx,f.cy,f.cz,f.x,f.y,f.z,f.planarity,f.verticality,
                                    v.code,COALESCE(c.outdoor,0),COALESCE(c.indoor,0),COALESCE(c.scene_context,-1)
                                    FROM geometry_features f JOIN geometry_voxels v USING(cx,cy,cz)
                                    LEFT JOIN geometry_context c USING(key)''')
        updates = []
        ceiling, building, wall = (by_id.get(name) for name in ('ceiling', 'building', 'wall'))
        for batch in iter(lambda: feature_rows.fetchmany(100_000), []):
            checkpoint()
            points = np.asarray([row[3:6] for row in batch], np.float64)
            distance, nearest = tree.query(points[:, :2], k=1, distance_upper_bound=config['terrain_max_gap'], workers=-1)
            valid = np.isfinite(distance) & (nearest < len(heights))
            hag = np.full(len(batch), np.nan)
            hag[valid] = points[valid, 2]-heights[nearest[valid]]
            codes = np.asarray([row[8] for row in batch], np.int64)
            outdoor = np.asarray([row[9] for row in batch], bool)
            scene_context = np.asarray([row[11] for row in batch], np.int8)
            in_height = (hag >= config['terrain_cover_min_height']) & (hag <= config['terrain_cover_max_height'])
            ceiling_cover = codes == ceiling if ceiling is not None else np.zeros(len(batch), bool)
            slope_cover = np.isin(codes, [item for item in (building, wall) if item is not None]) & (np.asarray([row[6] for row in batch]) >= .35) & (np.asarray([row[7] for row in batch]) >= .5)
            cover = valid & in_height & (ceiling_cover | slope_cover)
            # Indoor-context cells are covered even when
            # no horizontal ceiling is visible (for example below a roof slope).
            cover |= valid & (scene_context == 0)
            covered[nearest[cover]] = True
            updates.extend((None if not np.isfinite(value) else float(value), 0, int(row[0]), int(row[1]), int(row[2]))
                           for row, value in zip(batch, hag))
            if len(updates) >= 100_000:
                db.executemany('UPDATE geometry_features SET hag=?,covered=? WHERE cx=? AND cy=? AND cz=?', updates)
                updates.clear()
        if updates:
            db.executemany('UPDATE geometry_features SET hag=?,covered=? WHERE cx=? AND cy=? AND cz=?', updates)
        # Every feature in a covered terrain cell inherits the XY coverage.
        feature_rows = db.execute('SELECT cx,cy,cz,x,y FROM geometry_features')
        updates = []
        for batch in iter(lambda: feature_rows.fetchmany(100_000), []):
            checkpoint()
            points = np.asarray([row[3:5] for row in batch], np.float64)
            distance, nearest = tree.query(points, k=1, distance_upper_bound=config['terrain_max_gap'], workers=-1)
            values = np.zeros(len(batch), np.int64)
            valid = np.isfinite(distance) & (nearest < len(covered))
            values[valid] = covered[nearest[valid]]
            updates.extend((int(value), int(row[0]), int(row[1]), int(row[2])) for row, value in zip(batch, values))
            if len(updates) >= 100_000:
                db.executemany('UPDATE geometry_features SET covered=? WHERE cx=? AND cy=? AND cz=?', updates)
                updates.clear()
        if updates:
            db.executemany('UPDATE geometry_features SET covered=? WHERE cx=? AND cy=? AND cz=?', updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'cell_size': size, 'max_gap': config['terrain_max_gap'],
                'origin': np.asarray(origin, np.float64).tolist()}
    np.savez_compressed(target, metadata=np.asarray(json.dumps(metadata)), xy=xy, z=heights, covered=covered)
    return {'artifact': str(target), 'cells': int(len(xy)), 'covered_cells': int(covered.sum())}


def _alternatives(votes, active_codes):
    """Keep only alternatives that the final run legend can represent."""
    placeholders = ','.join('?' for _ in active_codes)
    votes.db.executescript('DROP TABLE IF EXISTS temp.geometry_alternatives;\n'
                           'CREATE TEMP TABLE geometry_alternatives('
                           'key BLOB PRIMARY KEY, code INTEGER, confidence INTEGER) WITHOUT ROWID;')
    votes.db.execute(f'''INSERT INTO geometry_alternatives
        SELECT key,class,CAST(ROUND(255.*weight/total) AS INTEGER) FROM (
          SELECT v.key,v.class,v.weight,SUM(v.weight) OVER (PARTITION BY v.key) AS total,
                 ROW_NUMBER() OVER (PARTITION BY v.key ORDER BY v.weight DESC,v.class ASC) AS rank
          FROM votes v JOIN decisions d ON d.key=v.key
          WHERE v.class != d.code AND v.class IN ({placeholders})
        ) WHERE rank=1''', tuple(sorted(active_codes)))


def _tree_seeds(votes, by_id, config):
    if 'tree_trunk' not in by_id:
        return []
    rows = votes.db.execute('''SELECT v.ix,v.iy,v.iz,f.linearity FROM geometry_voxels v
        JOIN geometry_features f USING(cx,cy,cz) LEFT JOIN geometry_context c USING(key)
        WHERE v.code=? AND COALESCE(c.scene_context,-1)=1''', (by_id['tree_trunk'],)).fetchall()
    if len(rows) < 4:
        return []
    coordinates = np.asarray([row[:3] for row in rows], np.int64)
    linearities = np.asarray([row[3] for row in rows], np.float64)
    count, labels = votes._sparse_components(coordinates)
    order = np.argsort(labels, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    grouped_coordinates = coordinates[order]
    grouped_linearities = linearities[order]
    seeds = []
    for component in range(count):
        points = grouped_coordinates[offsets[component]:offsets[component+1]]
        if (points[:, 2].ptp()*config['geometry_cell_size']) < .8:
            continue
        centered = points-points.mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        linearity = np.median(grouped_linearities[offsets[component]:offsets[component+1]])
        if abs(vectors[0, 2]) < .7 or linearity < .35:
            continue
        seeds.append((float(points[:, 0].mean()), float(points[:, 1].mean()), int(points[:, 2].max())))
    return seeds


def apply_geometry_rules(votes, by_id, origin, voxel_size, config, terrain_target: Path, checkpoint=lambda: None,
                         active_codes=None, scan_context=None):
    """Apply geometric corrections and persist terrain/coverage evidence.

    Temporary SQLite tables keep the feature calculation bounded to an 8 m
    tile plus halo. Only the compact terrain raster survives the output stage.
    """
    result = {'enabled': bool(config['geometry_rules']), 'changed': {}, 'terrain': None, 'tree_seeds': 0,
              'fallbacks': {}}
    if not config['geometry_rules']:
        return result
    origin = np.asarray(origin, np.float64)
    prepare_geometry_features(votes, origin, voxel_size, config, checkpoint)
    refresh_geometry_decisions(votes)
    active_codes = {int(value) for value in (by_id.values() if active_codes is None else active_codes)}
    scene_report = ensure_scene_context(votes, by_id, scan_context, origin, voxel_size, config, checkpoint)
    result['terrain'] = _terrain(votes, origin, config, Path(terrain_target), checkpoint, by_id)
    _alternatives(votes, active_codes)
    db = votes.db
    code = lambda name: by_id.get(name)
    building_wall = {item for item in (code('building'), code('wall')) if item is not None}
    ground_like = {item for item in (code('ground'), code('road'), code('sidewalk')) if item is not None}
    crown_sources = {item for item in (code('building'), code('wall'), code('ceiling'), code('fence'), 1) if item is not None}
    changes = {}
    fallbacks = {}

    def target_or_fallback(name, alternative, allowed=True):
        target = code(name)
        if allowed and target in active_codes:
            return target
        alternative = int(alternative) if alternative is not None else None
        if alternative in active_codes:
            kind, selected = 'alternative', alternative
        else:
            kind, selected = 'unclassified', 1
        key = f'{name}:{kind}'
        fallbacks[key] = fallbacks.get(key, 0)+1
        return selected

    def alternative_or_unclassified(alternative):
        alternative = int(alternative) if alternative is not None else None
        return alternative if alternative in active_codes else 1

    cursor = db.execute('''SELECT v.key,v.code,v.confidence,v.ix,v.iy,v.iz,f.x,f.y,f.z,f.scatter,f.planarity,
                           f.linearity,f.verticality,f.hag,f.covered,a.code,a.confidence,
                           COALESCE(c.outdoor,0),COALESCE(c.indoor,1),COALESCE(c.allow_ground,0),
                           COALESCE(c.allow_floor,0),COALESCE(c.allow_tree,0),COALESCE(c.allow_low_vegetation,0)
                           ,COALESCE(c.scene_context,-1)
                           FROM geometry_voxels v JOIN geometry_features f USING(cx,cy,cz)
                           LEFT JOIN geometry_alternatives a USING(key) LEFT JOIN geometry_context c USING(key)''')
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        for row in rows:
            key, current, _, _, _, _, x, y, z, scatter, planarity, _, _, hag, covered, alternative, alt_conf, outdoor, _, allow_ground, allow_floor, allow_tree, allow_low_vegetation, scene_context = row
            so, si = scene_context == 1, scene_context == 0
            wanted = None
            confidence = 0
            if config['geometry_rules_outdoor_vegetation'] and so and current in building_wall and scatter > .25 and planarity < .4:
                wanted = (alternative_or_unclassified(alternative) if covered else
                          target_or_fallback('tree' if hag is not None and hag >= 1.5 else 'low_vegetation', alternative,
                                             allow_tree if hag is not None and hag >= 1.5 else allow_low_vegetation))
            elif config['geometry_rules_outdoor_vegetation'] and so and current == code('ceiling') and scatter > .25 and not covered:
                wanted = target_or_fallback('tree', alternative, allow_tree)
            elif config['geometry_rules_outdoor_vegetation'] and so and current in ground_like and hag is not None and hag > .4:
                wanted, confidence = alternative_or_unclassified(alternative), int(alt_conf or 0)
            elif config['geometry_rules_outdoor_vegetation'] and so and current == code('low_vegetation') and hag is not None and hag > 2.5 and scatter > .25:
                wanted = target_or_fallback('tree', alternative, allow_tree)
            elif config['geometry_rules_ground_floor'] and si and current == code('ground') and covered:
                wanted = target_or_fallback('floor', alternative, allow_floor)
            elif config['geometry_rules_ground_floor'] and so and current == code('floor') and hag is not None and abs(hag) <= .4:
                wanted = target_or_fallback('ground', alternative, allow_ground)
            if wanted is not None and wanted != current:
                changes[bytes(key)] = (int(wanted), confidence)
    seeds = _tree_seeds(votes, by_id, config)
    result['tree_seeds'] = len(seeds)
    if seeds:
        seed_xy = np.asarray([item[:2] for item in seeds], np.float64)*voxel_size
        seed_top = np.asarray([item[2] for item in seeds], np.float64)*voxel_size
        tree = cKDTree(seed_xy)
        cursor = db.execute('''SELECT v.key,v.code,v.ix,v.iy,v.iz,f.scatter,f.hag,
                              COALESCE(c.scene_context,-1),COALESCE(c.allow_tree,0) FROM geometry_voxels v
                             JOIN geometry_features f USING(cx,cy,cz) LEFT JOIN geometry_context c USING(key)''')
        for rows in iter(lambda: cursor.fetchmany(100_000), []):
            checkpoint()
            points = np.asarray([(row[2]*voxel_size, row[3]*voxel_size) for row in rows], np.float64)
            distance, nearest = tree.query(points, distance_upper_bound=config['tree_crown_grow_radius'])
            valid = np.isfinite(distance) & (nearest < len(seed_top))
            for row, d, seed, has_seed in zip(rows, distance, nearest, valid):
                if not has_seed or row[5] <= .25 or row[7] != 1:
                    continue
                relative_z = row[4]*voxel_size-seed_top[seed]
                in_cylinder = d <= config['tree_crown_radius'] and -1. <= relative_z <= config['tree_crown_max_height']
                growing = row[1] == 1 and row[6] is not None and row[6] > 2. and d <= config['tree_crown_grow_radius']
                if ((in_cylinder and row[1] in crown_sources) or growing) and config['geometry_rules_outdoor_vegetation'] and row[7] == 1:
                    changes[bytes(row[0])] = (target_or_fallback('tree', None, row[8]), 0)
    updates = [(new_code, new_conf, key) for key, (new_code, new_conf) in changes.items()]
    for start in range(0, len(updates), 50_000):
        checkpoint()
        db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?', updates[start:start+50_000])
    db.commit()
    votes._invalidate_lookup()
    if changes:
        values = np.asarray([value[0] for value in changes.values()], np.int64)
        result['changed'] = {str(int(item)): int(number) for item, number in zip(*np.unique(values, return_counts=True))}
    result['fallbacks'] = fallbacks
    result['scan_context'] = scene_report
    return result
