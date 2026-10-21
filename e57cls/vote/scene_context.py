"""Per-voxel scene evidence shared by geometry and class consistency rules."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np


def _context_hash(scan_context, margin):
    records = []
    for scan, item in sorted((scan_context or {}).items(), key=lambda pair: int(pair[0])):
        position = item.get('position')
        if position is not None:
            position = [float(value) for value in position]
        records.append((int(scan), item.get('scene'), bool(item.get('outdoor')), position,
                        sorted(map(int, item.get('allowed_codes', ())))))
    return hashlib.sha256(json.dumps({'margin': float(margin), 'scans': records}, separators=(',', ':'),
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


def ensure_scene_context(votes, by_id, scan_context, origin, voxel_size, config, checkpoint, chunk_rows=1_000_000):
    """Materialize scene evidence in one key-sorted scan-observation pass."""
    started = time.monotonic()
    db = votes.db
    margin = float(config.get('scene_context_margin', .8))
    context = scan_context or {}
    digest = _context_hash(context, margin)
    exists = db.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name='geometry_context_meta'").fetchone()
    if exists:
        values = dict(db.execute('SELECT key,value FROM geometry_context_meta'))
        table = db.execute("SELECT name FROM sqlite_temp_master WHERE type='table' AND name='geometry_context'").fetchone()
        if table and values.get('hash') == digest and 'counts' in values:
            return {'context_voxels': json.loads(values['counts']), 'duration_s': time.monotonic()-started,
                    'reused': True, 'available': bool(context)}
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
    labels = {'outdoor': int(counts[0]), 'indoor': int(counts[1]), 'ambiguous': int(counts[2]), 'unknown': int(counts[3])}
    db.executemany('INSERT INTO geometry_context_meta VALUES (?,?)',
                   (('hash', digest), ('counts', json.dumps(labels, separators=(',', ':')))))
    db.commit()
    return {'context_voxels': labels, 'duration_s': time.monotonic()-started, 'reused': False,
            'available': lookup is not None}
