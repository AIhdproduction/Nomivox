"""Per-voxel scene evidence shared by geometry and class consistency rules."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np


def _outline_digest(outline, enabled):
    """Return a stable cache token for the optional building-envelope pass."""
    if not enabled:
        return 'disabled'
    if not outline or not outline.get('applied'):
        return 'unavailable:'+str((outline or {}).get('reason', 'none'))
    cached = outline.get('_scene_outline_digest')
    if cached:
        return str(cached)
    digest = hashlib.sha256()
    for key in ('origin', 'shape', 'cells', 'cell_size_m', 'barrier_cells', 'grid_cells', 'has_enclosed'):
        digest.update(repr(outline.get(key)).encode('utf-8'))
    for key in ('inside', 'outside', 'barrier'):
        digest.update(np.ascontiguousarray(outline[key], dtype=np.uint8).tobytes())
    value = digest.hexdigest()
    outline['_scene_outline_digest'] = value
    return value


def _context_hash(scan_context, margin, outline_token, outline_min_cover, outline_roof_band):
    records = []
    for scan, item in sorted((scan_context or {}).items(), key=lambda pair: int(pair[0])):
        position = item.get('position')
        if position is not None:
            position = [float(value) for value in position]
        records.append((int(scan), item.get('scene'), bool(item.get('outdoor')), position,
                        sorted(map(int, item.get('allowed_codes', ())))))
    return hashlib.sha256(json.dumps({'margin': float(margin), 'outline': outline_token,
                                     'outline_min_cover': float(outline_min_cover),
                                     'outline_roof_band': float(outline_roof_band), 'scans': records}, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _scene(item):
    value = item.get('scene')
    if value in ('outdoor', 'indoor'):
        return value
    return ('outdoor' if item.get('outdoor') else 'indoor') if 'scene' not in item else None


def _scan_lookup(scan_context, by_id):
    """Build compact vectorized scan metadata, including legacy MAX flags."""
    codes = np.asarray([int(by_id.get(name, -1)) for name in ('ground', 'floor', 'tree', 'low_vegetation')], np.int64)
    records = []
    for scan, item in (scan_context or {}).items():
        scene = _scene(item)
        position = np.asarray(item.get('position'), np.float64) if item.get('position') is not None else None
        valid_position = position is not None and position.shape == (3,) and np.isfinite(position).all()
        allowed = np.fromiter((int(code) for code in item.get('allowed_codes', ())), np.int64)
        records.append((int(scan), int(bool(item.get('outdoor'))),
                        1 if scene == 'outdoor' else 0 if scene == 'indoor' else -1,
                        *(position if valid_position else (np.nan, np.nan, np.nan)),
                        *np.isin(codes, allowed).astype(np.int8)))
    records.sort(key=lambda record: record[0])
    if not records:
        return None
    values = np.asarray(records, np.float64)
    return {'scan': values[:, 0].astype(np.int64), 'outdoor': values[:, 1].astype(np.int8),
            'known': values[:, 2].astype(np.int8), 'position': values[:, 3:6],
            'allowed': values[:, 6:10].astype(np.int8)}


def _insert_context_rows(db, keys, outdoor, indoor, allowed, nearest_scan, nearest_m, nearest_outdoor, scene_context):
    """Append one key-sorted NumPy-reduced chunk to the WITHOUT ROWID table."""
    db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ((bytes(key), int(out), int(inside), *(int(value) for value in flags),
                     None if scan < 0 else int(scan), None if not np.isfinite(distance) else float(distance),
                     None if nearest < 0 else int(nearest), None if scene < -1 else int(scene))
                    for key, out, inside, flags, scan, distance, nearest, scene in
                    zip(keys, outdoor, indoor, allowed, nearest_scan, nearest_m, nearest_outdoor, scene_context)))


def _reduce_rows(rows, lookup, origin, voxel_size, margin, db):
    """Reduce complete key groups from a cursor chunk and append their context."""
    if not rows:
        return np.zeros(4, np.int64)
    keys_all = np.asarray([bytes(row[0]) for row in rows], dtype='V24')
    scans_all = np.asarray([row[1] for row in rows], np.int64)
    slots = np.searchsorted(lookup['scan'], scans_all)
    present = slots < len(lookup['scan'])
    present[present] &= lookup['scan'][slots[present]] == scans_all[present]
    if not present.any():
        return np.zeros(4, np.int64)
    keys, scans, slots = keys_all[present], scans_all[present], slots[present]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    groups = np.repeat(np.arange(len(starts), dtype=np.int64), np.diff(np.r_[starts, len(keys)]))
    outdoor = np.maximum.reduceat(lookup['outdoor'][slots], starts)
    indoor = np.maximum.reduceat(1-lookup['outdoor'][slots], starts)
    allowed = np.column_stack([np.maximum.reduceat(lookup['allowed'][slots, column], starts)
                                for column in range(4)]).astype(np.int8)
    known = lookup['known'][slots]
    known_count = np.add.reduceat((known >= 0).astype(np.int8), starts)
    known_min = np.minimum.reduceat(np.where(known >= 0, known, 2), starts)
    known_max = np.maximum.reduceat(known, starts)
    scene = np.full(len(starts), -2, np.int8)
    unanimous = (known_count > 0) & (known_min == known_max)
    scene[unanimous] = known_max[unanimous]
    scene[(known_count > 0) & ~unanimous] = -1
    positions = lookup['position'][slots]
    with_position = (known >= 0) & np.isfinite(positions).all(axis=1)
    nearest_scan = np.full(len(starts), -1, np.int64)
    nearest_m = np.full(len(starts), np.inf, np.float64)
    nearest_outdoor = np.full(len(starts), -1, np.int8)
    if with_position.any():
        indices = keys.view('<i8').reshape(-1, 3)
        points = np.asarray(origin, np.float64)+(indices.astype(np.float64)+.5)*float(voxel_size)
        distance = np.full(len(keys), np.inf, np.float64)
        distance[with_position] = np.linalg.norm(points[with_position]-positions[with_position], axis=1)
        out_min, in_min = np.full(len(starts), np.inf), np.full(len(starts), np.inf)
        np.minimum.at(out_min, groups[(known == 1) & with_position], distance[(known == 1) & with_position])
        np.minimum.at(in_min, groups[(known == 0) & with_position], distance[(known == 0) & with_position])
        order = np.lexsort((scans, distance, groups))
        first = order[np.r_[True, groups[order][1:] != groups[order][:-1]]]
        nearest_scan, nearest_m, nearest_outdoor = scans[first], distance[first], known[first]
        positional = np.isfinite(nearest_m)
        scene[positional] = -1
        scene[positional & ((out_min < margin*in_min) | ~np.isfinite(in_min))] = 1
        scene[positional & ((in_min < margin*out_min) | ~np.isfinite(out_min))] = 0
    _insert_context_rows(db, keys[starts], outdoor, indoor, allowed, nearest_scan, nearest_m, nearest_outdoor, scene)
    return np.bincount(np.where(scene == 1, 0, np.where(scene == 0, 1, np.where(scene == -1, 2, 3))), minlength=4)


def _materialize_missing_context(db, checkpoint, chunk_rows):
    """Give every decision a context row so outline evidence applies to all classes."""
    cursor = db.execute('''SELECT d.key FROM decisions d LEFT JOIN geometry_context c USING(key)
                           WHERE c.key IS NULL ORDER BY d.key''')
    inserted = 0
    while rows := cursor.fetchmany(int(chunk_rows)):
        checkpoint()
        db.executemany('INSERT INTO geometry_context VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       ((bytes(row[0]), 0, 0, 0, 0, 0, 0, None, None, None, -1) for row in rows))
        inserted += len(rows)
    return inserted


def _context_counts(db):
    counts = dict(db.execute('''SELECT COALESCE(scene_context,-2),COUNT(*) FROM geometry_context
                                GROUP BY scene_context'''))
    return {'outdoor': int(counts.get(1, 0)), 'indoor': int(counts.get(0, 0)),
            'ambiguous': int(counts.get(-1, 0)), 'unknown': int(counts.get(-2, 0))}


def refine_scene_context_with_outline(votes, outline, by_id, scan_context, voxel_size, checkpoint=lambda: None,
                                      chunk_rows=100_000, min_cover=.5, roof_band=.4):
    """Apply a proven 2-D enclosure to the materialized scene context.

    The outline is deliberately a stronger signal only where flood-fill found
    a closed interior.  Its facade cells remain untouched, retaining the
    scanner evidence for noisy or double-sided surfaces.
    """
    report = {'enabled': True, 'applied': False, 'changed': 0, 'to_indoor': 0, 'to_outdoor': 0}
    if not outline or not outline.get('applied'):
        report['reason'] = (outline or {}).get('reason', 'no_outline')
        return report
    if not outline.get('has_enclosed'):
        report['reason'] = 'no_enclosed_area'
        return report
    db = votes.db
    cells = int(outline['cells'])
    low = np.asarray(outline['origin'], np.int64)
    shape = np.asarray(outline['shape'], np.int64)
    high = low+shape-1
    barrier, inside, outside = outline['barrier'], outline['inside'], outline['outside']

    # A cover is valid only in its own raster cell.  In particular, walls do
    # not cover a courtyard and a neighbouring ceiling cannot fill a gap.
    structural = [int(by_id[name]) for name in ('ceiling', 'building') if name in by_id]
    top = np.full(tuple(shape), np.iinfo(np.int64).min, np.int64)
    if structural:
        clause = ','.join('?' for _ in structural)
        rows = db.execute(f'SELECT key FROM decisions WHERE code IN ({clause})', structural).fetchall()
        if rows:
            coordinates = np.frombuffer(b''.join(bytes(row[0]) for row in rows), dtype='<i8').reshape(-1, 3)
            xy = coordinates[:, :2]//cells
            valid = np.all((xy >= low) & (xy <= high), axis=1)
            grid = xy[valid]-low
            np.maximum.at(top, (grid[:, 0], grid[:, 1]), coordinates[valid, 2])
    known = top != np.iinfo(np.int64).min
    if not known.any():
        report['reason'] = 'no_upper_structure'
        return report
    has_outdoor_scan = any(_scene(item) == 'outdoor' or bool(item.get('outdoor'))
                           for item in (scan_context or {}).values())

    cursor = db.execute('SELECT c.key,c.scene_context FROM geometry_context c ORDER BY c.key')
    updates = []
    while rows := cursor.fetchmany(int(chunk_rows)):
        checkpoint()
        keys = [bytes(row[0]) for row in rows]
        coordinates = np.frombuffer(b''.join(keys), dtype='<i8').reshape(-1, 3)
        xy = coordinates[:, :2]//cells
        in_grid = np.all((xy >= low) & (xy <= high), axis=1)
        target = np.asarray([row[1] if row[1] is not None else -1 for row in rows], np.int8)
        grid = xy[in_grid]-low
        on_barrier = np.zeros(len(rows), bool)
        on_barrier[in_grid] = barrier[grid[:, 0], grid[:, 1]]
        enclosed = np.zeros(len(rows), bool)
        enclosed[in_grid] = inside[grid[:, 0], grid[:, 1]]
        exposed = ~on_barrier & ~enclosed
        if has_outdoor_scan:
            target[exposed] = 1
        if enclosed.any():
            enclosed_ids = np.flatnonzero(enclosed)
            enclosed_grid = xy[enclosed_ids]-low
            caps = top[enclosed_grid[:, 0], enclosed_grid[:, 1]]
            local_cap = caps != np.iinfo(np.int64).min
            vertical_gap = (caps-coordinates[enclosed_ids, 2])*float(voxel_size)
            # Keep roofs, roof build-ups and everything above the local cap
            # in the scanner-derived context.  This pass may only establish
            # interior context beneath a sufficiently high, local cover.
            covered = local_cap & (vertical_gap >= float(min_cover))
            below_roof_band = vertical_gap > float(roof_band)
            indoors = enclosed_ids[covered & below_roof_band]
            target[indoors] = 0
        previous = np.asarray([row[1] if row[1] is not None else -1 for row in rows], np.int8)
        changed = target != previous
        if changed.any():
            updates.extend((int(target[index]), keys[index]) for index in np.flatnonzero(changed))
            report['to_indoor'] += int((target[changed] == 0).sum())
            report['to_outdoor'] += int((target[changed] == 1).sum())
        if len(updates) >= int(chunk_rows):
            db.executemany('UPDATE geometry_context SET scene_context=? WHERE key=?', updates)
            updates.clear()
    if updates:
        db.executemany('UPDATE geometry_context SET scene_context=? WHERE key=?', updates)
    db.commit()
    report['changed'] = int(report['to_indoor']+report['to_outdoor'])
    report['applied'] = True
    report['has_outdoor_scan'] = bool(has_outdoor_scan)
    return report


def ensure_scene_context(votes, by_id, scan_context, origin, voxel_size, config, checkpoint, chunk_rows=1_000_000,
                         outline=None):
    """Materialize scene evidence in one key-sorted scan-observation pass."""
    started = time.monotonic()
    db = votes.db
    margin = float(config.get('scene_context_margin', .8))
    context = scan_context or {}
    outline_enabled = bool(config.get('scene_outline', True))
    outline_min_cover = float(config.get('scene_outline_min_cover', .5))
    outline_roof_band = float(config.get('scene_outline_roof_band', .4))
    if outline_enabled and outline is None:
        outline = votes.building_outline([by_id[name] for name in ('building', 'wall') if name in by_id], voxel_size,
                                         config.get('building_boundary_cell_size', .25),
                                         config.get('building_boundary_min_height', 1.2),
                                         config.get('building_boundary_max_cells', 4_000_000))
    outline_token = _outline_digest(outline, outline_enabled)
    digest = _context_hash(context, margin, outline_token, outline_min_cover, outline_roof_band)
    exists = db.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name='geometry_context_meta'").fetchone()
    if exists:
        values = dict(db.execute('SELECT key,value FROM geometry_context_meta'))
        table = db.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name='geometry_context'").fetchone()
        if table and values.get('hash') == digest and 'counts' in values:
            return {'context_voxels': json.loads(values['counts']), 'duration_s': time.monotonic()-started,
                    'reused': True, 'available': bool(context),
                    'scene_outline': json.loads(values.get('scene_outline', '{}'))}
    db.executescript('''
        DROP TABLE IF EXISTS temp.geometry_context;
        DROP TABLE IF EXISTS temp.geometry_context_meta;
        CREATE TEMP TABLE geometry_context(
            key BLOB PRIMARY KEY, outdoor INTEGER NOT NULL, indoor INTEGER NOT NULL,
            allow_ground INTEGER NOT NULL, allow_floor INTEGER NOT NULL,
            allow_tree INTEGER NOT NULL, allow_low_vegetation INTEGER NOT NULL,
            nearest_scan INTEGER, nearest_m REAL, nearest_outdoor INTEGER, scene_context INTEGER
        ) WITHOUT ROWID;
        CREATE TEMP TABLE geometry_context_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
    ''')
    lookup = _scan_lookup(context, by_id)
    counts = np.zeros(4, np.int64)
    if lookup is not None:
        plan = db.execute('EXPLAIN QUERY PLAN SELECT key,scan FROM scan_observations ORDER BY key,scan').fetchall()
        if any('USE TEMP B-TREE' in str(row[-1]).upper() for row in plan):
            raise RuntimeError('scan_observations key order requires an unexpected SQLite sort')
        cursor = db.execute('SELECT key,scan FROM scan_observations ORDER BY key,scan')
        carry = []
        while rows := cursor.fetchmany(int(chunk_rows)):
            checkpoint()
            rows = carry+rows
            split = len(rows)
            while split and rows[split-1][0] == rows[-1][0]:
                split -= 1
            carry = rows[split:]
            counts += _reduce_rows(rows[:split], lookup, origin, voxel_size, margin, db)
        counts += _reduce_rows(carry, lookup, origin, voxel_size, margin, db)
    _materialize_missing_context(db, checkpoint, chunk_rows)
    outline_report = (refine_scene_context_with_outline(
        votes, outline, by_id, context, voxel_size, checkpoint, chunk_rows, outline_min_cover, outline_roof_band)
                      if outline_enabled else {'enabled': False, 'applied': False, 'changed': 0, 'to_indoor': 0, 'to_outdoor': 0})
    labels = _context_counts(db)
    db.executemany('INSERT INTO geometry_context_meta VALUES (?,?)',
                   (('hash', digest), ('counts', json.dumps(labels, separators=(',', ':'))),
                    ('scene_outline', json.dumps(outline_report, separators=(',', ':')))))
    db.commit()
    return {'context_voxels': labels, 'duration_s': time.monotonic()-started, 'reused': False,
            'available': lookup is not None, 'scene_outline': outline_report}
