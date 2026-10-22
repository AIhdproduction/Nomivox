"""Geometry-connected segment voting for already decided voxel votes."""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from e57cls.vote.geometry_rules import prepare_geometry_features, refresh_geometry_decisions


def _linear_keys(cells, low, shape):
    """Return sortable, collision-free int64 keys for feature-cell coordinates."""
    relative = np.asarray(cells, np.int64)-low
    return ((relative[:, 0]*shape[1]+relative[:, 1])*shape[2]+relative[:, 2]).astype(np.int64)


def _segment_cells(votes, origin, voxel_size, config, checkpoint):
    """Create persistent TEMP segment tables from global feature-cell adjacency."""
    db = votes.db
    db.executescript('''
        DROP TABLE IF EXISTS temp.segment_cells;
        DROP TABLE IF EXISTS temp.segments;
        DROP TABLE IF EXISTS temp.segment_votes;
        CREATE TEMP TABLE segment_cells(cx INTEGER,cy INTEGER,cz INTEGER,segment INTEGER,
            PRIMARY KEY(cx,cy,cz)) WITHOUT ROWID;
        CREATE TEMP TABLE segments(segment INTEGER PRIMARY KEY,kind TEXT,cells INTEGER,voxels INTEGER,
            x REAL,y REAL,z REAL,nx REAL,ny REAL,nz REAL,min_x REAL,min_y REAL,min_z REAL,
            max_x REAL,max_y REAL,max_z REAL,winner INTEGER,share REAL,second INTEGER,second_share REAL) WITHOUT ROWID;
        CREATE TEMP TABLE segment_votes(segment INTEGER,class INTEGER,weight REAL,
            PRIMARY KEY(segment,class)) WITHOUT ROWID;
    ''')
    count = int(db.execute('SELECT COUNT(*) FROM geometry_cells').fetchone()[0])
    if not count:
        return (np.empty((0, 3), np.int64), np.empty(0, np.int64), np.zeros(3, np.int64),
                np.ones(3, np.int64), 0)
    cells = np.empty((count, 3), np.int64)
    voxels = np.empty(count, np.int64)
    points = np.empty((count, 3), np.float64)
    normals = np.empty((count, 3), np.float64)
    planarity = np.empty(count, np.float64)
    scatter = np.empty(count, np.float64)
    cursor = db.execute('''SELECT c.cx,c.cy,c.cz,c.n,f.x,f.y,f.z,f.nx,f.ny,f.nz,f.planarity,f.scatter
                           FROM geometry_cells c JOIN geometry_features f USING(cx,cy,cz)
                           ORDER BY c.cx,c.cy,c.cz''')
    start = 0
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        stop = start+len(rows)
        batch = np.asarray(rows, np.float64)
        cells[start:stop] = batch[:, :3]
        voxels[start:stop] = batch[:, 3]
        points[start:stop] = batch[:, 4:7]
        normals[start:stop] = batch[:, 7:10]
        planarity[start:stop] = batch[:, 10]
        scatter[start:stop] = batch[:, 11]
        start = stop
    planar = planarity >= config['segment_min_planarity']
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = high-low+1
    keys = _linear_keys(cells, low, shape)
    # Rows are lexicographically ordered, which is also the order of this key.
    assert np.all(keys[1:] >= keys[:-1])
    planar_source, planar_target = [], []
    object_source, object_target = [], []
    attach_object, attach_planar, attach_distance = [], [], []
    cosine = np.cos(np.deg2rad(float(config['segment_normal_angle'])))
    offsets = [(dx, dy, dz) for dx in range(2) for dy in range(-1, 2) for dz in range(-1, 2)
               if (dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0))]
    for offset in offsets:
        checkpoint()
        shift = np.asarray(offset, np.int64)
        neighbour = cells+shift
        inside = np.all((neighbour >= low) & (neighbour <= high), axis=1)
        candidate = _linear_keys(neighbour[inside], low, shape)
        found = np.searchsorted(keys, candidate)
        valid = (found < len(keys))
        valid[valid] &= keys[found[valid]] == candidate[valid]
        first = np.flatnonzero(inside)[valid]
        second = found[valid]
        if not len(first):
            continue
        first_planar, second_planar = planar[first], planar[second]
        first_norm = np.linalg.norm(normals[first], axis=1)
        second_norm = np.linalg.norm(normals[second], axis=1)
        both_object = ~first_planar & ~second_planar
        both_planar = first_planar & second_planar
        planar_edge = first_planar ^ second_planar
        geometric = both_planar | planar_edge
        compatible = np.zeros(len(first), bool)
        plane_distance = np.full(len(first), np.inf)
        if geometric.any():
            ids = np.flatnonzero(geometric)
            delta = points[second[ids]]-points[first[ids]]
            normal_ok = ((first_norm[ids] > .5) & (second_norm[ids] > .5)
                         & (np.einsum('ij,ij->i', normals[first[ids]], normals[second[ids]]) >= cosine))
            first_distance = np.abs(np.einsum('ij,ij->i', delta, normals[first[ids]]))
            second_distance = np.abs(np.einsum('ij,ij->i', delta, normals[second[ids]]))
            plane_ok = ((first_distance <= config['segment_plane_distance'])
                        & (second_distance <= config['segment_plane_distance']))
            compatible[ids] = normal_ok & plane_ok
            plane_distance[ids] = np.where(first_planar[ids], first_distance, second_distance)
        take = both_planar & compatible
        if take.any():
            planar_source.append(first[take])
            planar_target.append(second[take])
        take = planar_edge & compatible
        if take.any():
            objects = np.where(first_planar[take], second[take], first[take])
            planes = np.where(first_planar[take], first[take], second[take])
            attach_object.append(objects)
            attach_planar.append(planes)
            attach_distance.append(plane_distance[take])
        if both_object.any():
            object_source.append(first[both_object])
            object_target.append(second[both_object])

    # First form components only from planar-planar edges.  Every original
    # planar cell is represented, including isolated one-cell planes.
    if planar_source:
        source, target = np.concatenate(planar_source), np.concatenate(planar_target)
        graph = coo_matrix((np.ones(2*len(source), np.uint8),
                            (np.r_[source, target], np.r_[target, source])), shape=(len(cells), len(cells)))
        _, raw_planar = connected_components(graph, directed=False)
    else:
        raw_planar = np.arange(len(cells), dtype=np.int64)
    labels = np.full(len(cells), -1, np.int64)
    planar_raw = raw_planar[planar]
    raw_ids = np.unique(planar_raw)
    raw_minima = np.full(raw_planar.max()+1 if len(raw_planar) else 0, np.iinfo(np.int64).max)
    np.minimum.at(raw_minima, planar_raw, keys[planar])
    planar_order = raw_ids[np.argsort(raw_minima[raw_ids], kind='stable')]
    raw_to_segment = np.full(len(raw_minima), -1, np.int64)
    raw_to_segment[planar_order] = np.arange(len(planar_order))
    labels[planar] = raw_to_segment[planar_raw]

    # A non-planar cell can join exactly one compatible plane component.  A
    # lexicographic selection makes the closest plane and then the lowest
    # component ID deterministic; attached cells never enter object edges.
    attached = np.zeros(len(cells), bool)
    if attach_object:
        objects = np.concatenate(attach_object)
        components = labels[np.concatenate(attach_planar)]
        distances = np.concatenate(attach_distance)
        order = np.lexsort((components, distances, objects))
        ordered_objects = objects[order]
        first = np.r_[True, ordered_objects[1:] != ordered_objects[:-1]]
        selected = order[first]
        labels[objects[selected]] = components[selected]
        attached[objects[selected]] = True

    # Only cells which remain objects may form occupancy components.
    object_members = ~planar & ~attached
    if object_source:
        source, target = np.concatenate(object_source), np.concatenate(object_target)
        take = object_members[source] & object_members[target]
        source, target = source[take], target[take]
    else:
        source = target = np.empty(0, np.int64)
    if len(source):
        graph = coo_matrix((np.ones(2*len(source), np.uint8),
                            (np.r_[source, target], np.r_[target, source])), shape=(len(cells), len(cells)))
        _, raw_object = connected_components(graph, directed=False)
    else:
        raw_object = np.arange(len(cells), dtype=np.int64)
    object_raw = raw_object[object_members]
    object_ids = np.unique(object_raw)
    object_minima = np.full(raw_object.max()+1 if len(raw_object) else 0, np.iinfo(np.int64).max)
    np.minimum.at(object_minima, object_raw, keys[object_members])
    object_order = object_ids[np.argsort(object_minima[object_ids], kind='stable')]
    raw_to_object = np.full(len(object_minima), -1, np.int64)
    raw_to_object[object_order] = len(planar_order)+np.arange(len(object_order))
    labels[object_members] = raw_to_object[object_raw]

    # Segment IDs are globally deterministic, regardless of component kind.
    minima = np.full(labels.max()+1, np.iinfo(np.int64).max)
    np.minimum.at(minima, labels, keys)
    order = np.argsort(minima, kind='stable')
    remap = np.empty(len(order), np.int64)
    remap[order] = np.arange(len(order))
    labels = remap[labels]
    for start in range(0, len(cells), 100_000):
        block = np.column_stack((cells[start:start+100_000], labels[start:start+100_000]))
        db.executemany('INSERT INTO segment_cells VALUES (?,?,?,?)',
                       (tuple(map(int, row)) for row in block))
    counts = np.bincount(labels)
    voxel_counts = np.bincount(labels, weights=voxels).astype(np.int64)
    centers = np.column_stack([np.bincount(labels, weights=points[:, axis])/counts for axis in range(3)])
    lower = np.full((len(counts), 3), np.inf)
    upper = np.full((len(counts), 3), -np.inf)
    for axis in range(3):
        np.minimum.at(lower[:, axis], labels, points[:, axis])
        np.maximum.at(upper[:, axis], labels, points[:, axis])
    planar_counts = np.bincount(labels, weights=planar.astype(np.int8), minlength=len(counts))
    mean_normals = np.column_stack([
        np.divide(np.bincount(labels, weights=normals[:, axis]*planar, minlength=len(counts)), planar_counts,
                  out=np.zeros(len(counts)), where=planar_counts > 0) for axis in range(3)])
    lengths = np.linalg.norm(mean_normals, axis=1)
    mean_normals = np.divide(mean_normals, lengths[:, None], out=np.zeros_like(mean_normals), where=lengths[:, None] > 1e-12)
    kinds = np.where(planar_counts/counts >= .5, 'planar', 'object')
    for start in range(0, len(counts), 100_000):
        stop = min(start+100_000, len(counts))
        db.executemany('''INSERT INTO segments(segment,kind,cells,voxels,x,y,z,nx,ny,nz,min_x,min_y,min_z,max_x,max_y,max_z,
                                                 winner,share,second,second_share)
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,0,1,0)''',
                       ((int(index), str(kinds[index]), int(counts[index]), int(voxel_counts[index]), *map(float, centers[index]),
                         *map(float, mean_normals[index]), *map(float, lower[index]), *map(float, upper[index]))
                        for index in range(start, stop)))
    db.commit()
    return cells, labels, low, shape, len(counts)


def _accumulate_segment_votes(votes, cells, labels, low, shape, active_codes, reflection, min_observations, checkpoint):
    db = votes.db
    allowed = np.asarray(sorted(set(int(code) for code in active_codes)-{1, int(reflection)}), np.int64)
    if not len(allowed):
        return
    feature_keys = _linear_keys(cells, low, shape)
    ratio = None
    # geometry_voxels already holds exact integer cell coordinates.  The SQL
    # stream remains bounded and the sorted feature keys avoid a Python map.
    cursor = db.execute('''SELECT v.key,v.class,v.weight,g.cx,g.cy,g.cz FROM votes v
                           JOIN geometry_voxels g ON g.key=v.key
                           JOIN observations o ON o.key=v.key
                           WHERE o.count>=? ORDER BY v.key,v.class''', (int(min_observations),))
    pending = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        keys = np.asarray([row[3:6] for row in rows], np.int64)
        index = np.searchsorted(feature_keys, _linear_keys(keys, low, shape))
        valid = (index < len(feature_keys))
        valid[valid] &= feature_keys[index[valid]] == _linear_keys(keys[valid], low, shape)
        classes = np.asarray([row[1] for row in rows], np.int64)
        class_index = np.searchsorted(allowed, classes)
        valid &= (class_index < len(allowed))
        valid[valid] &= allowed[class_index[valid]] == classes[valid]
        if not valid.any():
            continue
        segment = labels[index[valid]]
        flat = segment*len(allowed)+class_index[valid]
        unique, inverse = np.unique(flat, return_inverse=True)
        weights = np.bincount(inverse, weights=np.asarray([row[2] for row in rows], np.float64)[valid])
        pending.extend((int(value//len(allowed)), int(allowed[value % len(allowed)]), float(weight))
                       for value, weight in zip(unique, weights))
        if len(pending) >= 50_000:
            db.executemany('''INSERT INTO segment_votes VALUES (?,?,?) ON CONFLICT(segment,class)
                              DO UPDATE SET weight=weight+excluded.weight''', pending)
            pending.clear()
    if pending:
        db.executemany('''INSERT INTO segment_votes VALUES (?,?,?) ON CONFLICT(segment,class)
                          DO UPDATE SET weight=weight+excluded.weight''', pending)
    db.executescript('''
        DROP TABLE IF EXISTS temp.segment_winners;
        CREATE TEMP TABLE segment_winners AS
        SELECT segment,
               MAX(CASE WHEN rank=1 THEN class END) AS winner,
               MAX(CASE WHEN rank=1 THEN weight END)/MAX(total) AS share,
               MAX(CASE WHEN rank=2 THEN class END) AS second,
               COALESCE(MAX(CASE WHEN rank=2 THEN weight END)/MAX(total),0) AS second_share
        FROM (
            SELECT segment,class,weight,SUM(weight) OVER (PARTITION BY segment) AS total,
                   ROW_NUMBER() OVER (PARTITION BY segment ORDER BY weight DESC,class ASC) AS rank
            FROM segment_votes
        ) GROUP BY segment;
        UPDATE segments AS s SET winner=w.winner,share=w.share,second=COALESCE(w.second,1),second_share=w.second_share
        FROM segment_winners AS w WHERE w.segment=s.segment;
        DROP TABLE segment_winners;
    ''')
    db.commit()


def _apply_decisive_segments(votes, config, by_id, reflection, checkpoint):
    db = votes.db
    protected = {int(by_id[name]) for name in config['segment_protected_classes'] if name in by_id}
    stairs = by_id.get('stairs')
    decisive = int(db.execute('''SELECT COUNT(*) FROM segments WHERE share>=? AND cells>=?''',
                              (config['segment_min_share'], config['segment_min_cells'])).fetchone()[0])
    unclassified_before = int(db.execute('SELECT COUNT(*) FROM decisions WHERE code=1').fetchone()[0])
    if not decisive:
        return decisive, 0, 0, 0, {}, unclassified_before, unclassified_before
    cursor = db.execute('''SELECT d.key,v.class,v.weight,d.code,s.winner,s.share
                           FROM decisions d JOIN geometry_voxels g ON g.key=d.key
                           JOIN segment_cells sc USING(cx,cy,cz) JOIN segments s ON s.segment=sc.segment
                           JOIN votes v ON v.key=d.key
                           JOIN observations o ON o.key=d.key
                           WHERE s.share>=? AND s.cells>=? AND o.count>=?
                           ORDER BY d.key,v.class''', (config['segment_min_share'], config['segment_min_cells'],
                                                       int(config['min_observations'])))
    pending, changed = [], {}
    filled = flipped = 0
    carry = []
    def finish(rows):
        nonlocal filled, flipped
        keys = np.frombuffer(b''.join(row[0] for row in rows), dtype='V24')
        classes = np.asarray([row[1] for row in rows], np.int64)
        weights = np.asarray([row[2] for row in rows], np.float64)
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
        groups = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rows)]))
        total = np.bincount(groups, weights=weights)
        current = np.asarray([rows[index][3] for index in starts], np.int64)
        winner = np.asarray([rows[index][4] for index in starts], np.int64)
        share = np.asarray([rows[index][5] for index in starts], np.float64)
        current_weight = np.bincount(groups[classes == current[groups]], weights=weights[classes == current[groups]], minlength=len(starts))
        winner_weight = np.bincount(groups[classes == winner[groups]], weights=weights[classes == winner[groups]], minlength=len(starts))
        current_share = np.divide(current_weight, total, out=np.zeros_like(total), where=total > 0)
        winner_share = np.divide(winner_weight, total, out=np.zeros_like(total), where=total > 0)
        allowed = (~np.isin(current, list(protected)+[reflection]) & (current != stairs if stairs is not None else True)
                   & (winner != reflection) & (winner != stairs if stairs is not None else True))
        fill = allowed & (current == 1) & (winner_share >= config['segment_fill_min_support'])
        flip = (allowed & (current != 1) & (current != winner) & (current_share < config['segment_weak_share'])
                & (winner_share >= config['segment_flip_min_support']))
        take = fill | flip
        if take.any():
            confidence = np.rint(share[take]*255).astype(np.int64)
            pending.extend((bytes(key), int(code), int(value)) for key, code, value in zip(keys[starts][take], winner[take], confidence))
            filled += int(fill.sum())
            flipped += int(flip.sum())
            for source, target in zip(current[take], winner[take]):
                label = f'{int(source)}->{int(target)}'
                changed[label] = changed.get(label, 0)+1
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        rows = carry+rows
        last, split = rows[-1][0], len(rows)
        while split and rows[split-1][0] == last:
            split -= 1
        if split:
            finish(rows[:split])
        carry = rows[split:]
        if len(pending) >= 50_000:
            db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?',
                           ((code, confidence, key) for key, code, confidence in pending))
            pending.clear()
    if carry:
        finish(carry)
    if pending:
        db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?',
                       ((code, confidence, key) for key, code, confidence in pending))
    db.commit()
    refresh_geometry_decisions(votes)
    votes._invalidate_lookup()
    unclassified_after = int(db.execute('SELECT COUNT(*) FROM decisions WHERE code=1').fetchone()[0])
    return decisive, filled+flipped, filled, flipped, changed, unclassified_before, unclassified_after


def apply_segment_vote(votes, by_id, origin, voxel_size, config, checkpoint=lambda: None, active_codes=None):
    """Pool SAM evidence over geometry-connected cells and update weak voxels."""
    started = time.monotonic()
    result = {'enabled': bool(config['segment_vote']), 'segments': 0, 'planar_segments': 0, 'object_segments': 0,
              'decisive_segments': 0, 'changed_voxels': 0, 'filled_unclassified': 0, 'flipped': 0,
              'changed_by_transition': {}, 'unclassified_before': 0, 'unclassified_after': 0, 'duration_s': 0.,
              'timings': {'features_s': 0., 'segmentation_s': 0., 'accumulate_s': 0., 'apply_s': 0.}}
    if not config['segment_vote']:
        return result
    features_started = time.monotonic()
    prepare_geometry_features(votes, np.asarray(origin, np.float64), voxel_size, config, checkpoint)
    features_s = time.monotonic()-features_started
    segmentation_started = time.monotonic()
    cells, labels, low, shape, count = _segment_cells(votes, origin, voxel_size, config, checkpoint)
    segmentation_s = time.monotonic()-segmentation_started
    active_codes = set(by_id.values() if active_codes is None else active_codes)
    reflection = int(by_id.get('reflection', 68))
    accumulate_started = time.monotonic()
    _accumulate_segment_votes(votes, cells, labels, low, shape, active_codes, reflection,
                              config['min_observations'], checkpoint)
    accumulate_s = time.monotonic()-accumulate_started
    apply_started = time.monotonic()
    decisive, changed, filled, flipped, transitions, before, after = _apply_decisive_segments(votes, config, by_id, reflection, checkpoint)
    apply_s = time.monotonic()-apply_started
    kinds = votes.db.execute("SELECT kind,COUNT(*) FROM segments GROUP BY kind").fetchall()
    result.update(segments=count, planar_segments=dict(kinds).get('planar', 0), object_segments=dict(kinds).get('object', 0),
                  decisive_segments=decisive, changed_voxels=changed, filled_unclassified=filled, flipped=flipped,
                  changed_by_transition=transitions, unclassified_before=before, unclassified_after=after,
                  duration_s=time.monotonic()-started,
                  timings={'features_s': features_s, 'segmentation_s': segmentation_s,
                           'accumulate_s': accumulate_s, 'apply_s': apply_s})
    return result
