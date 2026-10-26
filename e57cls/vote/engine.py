"""SQLite-backed spatial grouping with one observation per scan position."""

import json
import logging
from pathlib import Path
import sqlite3
import time

import numpy as np
import psutil
from scipy.ndimage import binary_propagation, generate_binary_structure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from e57cls.files import atomic_json, file_hash
from e57cls.geometry.cameras import scan_pose


LOG = logging.getLogger(__name__)


def connect(path):
    connection = sqlite3.connect(path)
    # Keep the working sets in SQLite's page cache without changing the schema
    # or transaction boundaries used by resumable vote stores.
    cache_mib = min(1024, max(512, int(psutil.virtual_memory().available // (1024**2) // 8)))
    connection.execute(f"PRAGMA cache_size={-cache_mib*1024}")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=268435456")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def voxel_keys(world, origin, size):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    values = np.floor((np.asarray(world, np.float64)-origin)/size)
    if not np.isfinite(values).all() or (np.abs(values) >= 2**62).any():
        raise ValueError("Voxel indices exceed the supported range")
    return np.ascontiguousarray(values, dtype="<i8").view("V24").reshape(-1)


def _coarse_component_spares(labels, sizes, cells, contexts, component_min_cells,
                             sparse_maximum_voxels_per_cell, sparse_components):
    """Return components exempt from the small-island threshold.

    All component reductions are done once over the complete voxel array. In
    particular, occupancies are sorted within each component before selecting
    their middle value(s), which is equivalent to ``np.median`` per component.
    """
    unique_cells, cell_inverse = np.unique(cells, axis=0, return_inverse=True)
    cell_inverse = np.asarray(cell_inverse).reshape(-1)
    cell_count, cell_labels = VoteStore._sparse_components(unique_cells)
    feature_sizes = np.bincount(cell_labels, minlength=cell_count)

    pairs, inverse = np.unique(np.column_stack((labels, cells)), axis=0, return_inverse=True)
    occupancies = np.bincount(inverse)
    pair_components = pairs[:, 0]
    occupancy_order = np.lexsort((occupancies, pair_components))
    sorted_components = pair_components[occupancy_order]
    starts = np.flatnonzero(np.r_[True, sorted_components[1:] != sorted_components[:-1]])
    lengths = np.diff(np.r_[starts, len(occupancy_order)])
    lower = occupancies[occupancy_order[starts+(lengths-1)//2]]
    upper = occupancies[occupancy_order[starts+lengths//2]]
    sparse = np.zeros(len(sizes), bool)
    sparse[sorted_components[starts]] = (lower.astype(np.float64)+upper) / 2 <= sparse_maximum_voxels_per_cell

    connected_cells = feature_sizes[cell_labels] >= component_min_cells
    connected = np.bincount(labels, weights=connected_cells[cell_inverse], minlength=len(sizes)) > 0
    outdoor_observed = np.zeros(len(sizes), bool)
    if contexts is not None:
        outdoor_observed = np.bincount(labels, weights=contexts[:, 0] != 0, minlength=len(sizes)) > 0
    return sparse_components & sparse & connected & outdoor_observed


def _touching_box_labels(low, high, maximum_gap):
    """Return connected labels for boxes meeting within ``maximum_gap``.

    Boxes are swept along x.  The remaining dimensions are compared in NumPy
    batches, preserving the inclusive boundary condition of ``_boxes_touch``.
    """
    count = len(low)
    if count < 2:
        return np.arange(count, dtype=np.int64)
    order = np.argsort(low[:, 0], kind='stable')
    active = np.empty(0, np.int64)
    source, target = [], []
    for first in order:
        active = active[high[active, 0] >= low[first, 0]-maximum_gap]
        if len(active):
            touching = np.maximum(low[first]-high[active], low[active]-high[first]).max(axis=1) <= maximum_gap
            second = active[touching]
            if len(second):
                source.append(np.full(len(second), first, np.int64))
                target.append(second)
        active = np.append(active, first)
    if not source:
        return np.arange(count, dtype=np.int64)
    graph = coo_matrix((np.ones(sum(map(len, source)), np.uint8),
                       (np.concatenate(source), np.concatenate(target))), shape=(count, count))
    return connected_components(graph, directed=False, return_labels=True)[1]


def representatives(store, directory: Path, origin, size, checkpoint=lambda: None):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/"representatives.json"
    if marker.is_file():
        meta = json.loads(marker.read_text())
        target = directory/"representatives.npy"
        if (meta["source_sha256"] == store.meta["source_sha256"] and meta["size"] == size
                and meta["origin"] == list(origin) and target.is_file()
                and meta.get("sha256") == file_hash(target)):
            return directory/"representatives.npy", meta["count"]
    db = connect(directory/"representatives.sqlite")
    try:
        db.execute("DROP TABLE IF EXISTS reps")
        db.execute("CREATE TABLE reps (key BLOB PRIMARY KEY, x REAL, y REAL, z REAL) WITHOUT ROWID")
        pose = scan_pose(store.meta["metadata"])
        for _, xyz, valid in store.chunks():
            checkpoint()
            world = pose.to_world(xyz[valid])
            keys = voxel_keys(world, origin, size)
            unique, ids = np.unique(keys, return_index=True)
            db.executemany("INSERT OR IGNORE INTO reps VALUES (?,?,?,?)",
                           ((bytes(k), *map(float, p)) for k, p in zip(unique, world[ids])))
            db.commit()
        count = db.execute("SELECT COUNT(*) FROM reps").fetchone()[0]
        dtype = np.dtype([("key", "V24"), ("xyz", "<f8", 3)])
        target = directory/"representatives.npy"
        temp = directory/"representatives.tmp.npy"
        out = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype, shape=(count,))
        cursor = db.execute("SELECT key,x,y,z FROM reps ORDER BY key")
        start = 0
        try:
            while rows := cursor.fetchmany(50_000):
                checkpoint()
                out["key"][start:start+len(rows)] = [np.void(row[0]) for row in rows]
                out["xyz"][start:start+len(rows)] = [row[1:] for row in rows]
                start += len(rows)
            out.flush()
        finally:
            out._mmap.close()
        temp.replace(target)
        atomic_json(marker, {"count": count, "source_sha256": store.meta["source_sha256"],
                             "size": size, "origin": list(origin), "sha256": file_hash(target)})
        return target, count
    finally:
        db.close()


class VoteStore:
    # Export may coalesce equal voxel keys before calling lookup.  Adapters
    # with order-dependent test or plugin behaviour can explicitly opt out.
    lookup_is_key_pure = True

    def __init__(self, path, station_tracking=False):
        self.path = Path(path)
        self.station_tracking = bool(station_tracking)
        self.db = connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS votes(key BLOB, class INTEGER, weight REAL, peak REAL, PRIMARY KEY(key,class)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS observations(key BLOB PRIMARY KEY, count INTEGER) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS scan_observations(key BLOB, scan INTEGER,
              PRIMARY KEY(key,scan)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS scans(scan INTEGER PRIMARY KEY, identity TEXT);
          CREATE TABLE IF NOT EXISTS decisions(key BLOB PRIMARY KEY, code INTEGER, confidence INTEGER) WITHOUT ROWID;
        """)
        if 'peak' not in {row[1] for row in self.db.execute('PRAGMA table_info(votes)')}:
            raise ValueError('Old vote schema requires a new pipeline cache identity')
        # Static runs deliberately retain their historic schema.  SLAM stores
        # station evidence separately, so a static resume database is never
        # migrated or modified merely by opening it.
        if self.station_tracking:
            self.db.executescript("""
              CREATE TABLE IF NOT EXISTS station_peaks(key BLOB, class INTEGER, station INTEGER, weight REAL,
                  PRIMARY KEY(key,class,station)) WITHOUT ROWID;
              CREATE TABLE IF NOT EXISTS station_decisions(key BLOB PRIMARY KEY, station INTEGER) WITHOUT ROWID;
            """)
        self.priority_overrides = 0
        self._lookup_keys = self._lookup_values = None
        self._lookup_paths = ()

    def _invalidate_lookup(self):
        for mapping in (self._lookup_keys, self._lookup_values):
            if isinstance(mapping, np.memmap):
                mapping.flush()
                mapping._mmap.close()
        for path in self._lookup_paths:
            path.unlink(missing_ok=True)
        self._lookup_keys = self._lookup_values = None
        self._lookup_paths = ()

    def _lookup_snapshot(self):
        if self._lookup_keys is None:
            count = int(self.db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0])
            required = count*(np.dtype('V24').itemsize+2)
            limit = min(2*1024**3, psutil.virtual_memory().available//4)
            if required > limit:
                paths = (Path(str(self.path)+'.lookup_keys.npy'), Path(str(self.path)+'.lookup_values.npy'))
                for path in paths:
                    path.unlink(missing_ok=True)
                self._lookup_keys = np.lib.format.open_memmap(paths[0], mode='w+', dtype='V24', shape=(count,))
                self._lookup_values = np.lib.format.open_memmap(paths[1], mode='w+', dtype=np.uint8, shape=(count, 2))
                self._lookup_paths = paths
            else:
                self._lookup_keys = np.empty(count, 'V24')
                self._lookup_values = np.empty((count, 2), np.uint8)
            cursor = self.db.execute('SELECT key,code,confidence FROM decisions ORDER BY key')
            start = 0
            while rows := cursor.fetchmany(1_000_000):
                stop = start+len(rows)
                self._lookup_keys[start:stop] = np.frombuffer(b''.join(row[0] for row in rows), dtype='V24')
                self._lookup_values[start:stop] = np.fromiter((value for row in rows for value in row[1:]),
                                                               dtype=np.uint8, count=2*len(rows)).reshape(-1, 2)
                start = stop
            if isinstance(self._lookup_keys, np.memmap):
                self._lookup_keys.flush()
                self._lookup_values.flush()
        return self._lookup_keys, self._lookup_values

    def close(self):
        self._invalidate_lookup()
        self.db.close()

    def completed(self, scan, identity):
        return self.db.execute("SELECT 1 FROM scans WHERE scan=? AND identity=?", (scan, identity)).fetchone() is not None

    def add_scan(self, scan, identity, batches, class_codes, station_id=None):
        if self.completed(scan, identity):
            return
        with self.db:
            if self.db.execute("SELECT 1 FROM scans WHERE scan=?", (scan,)).fetchone():
                raise ValueError("Changed voting identity requires a separate vote store")
            for keys, weights in batches:
                has_evidence = np.any(weights > 0, axis=1)
                self.db.executemany("INSERT INTO observations VALUES (?,1) ON CONFLICT(key) DO UPDATE SET count=count+1",
                                    ((bytes(k),) for k in keys[has_evidence]))
                self.db.executemany("INSERT OR IGNORE INTO scan_observations VALUES (?,?)",
                                    ((bytes(k), int(scan)) for k in keys[has_evidence]))
                rows, columns = np.nonzero(weights > 0)
                codes = np.asarray(class_codes, np.int64)[columns]
                values = weights[rows, columns]
                self.db.executemany("INSERT INTO votes VALUES (?,?,?,?) ON CONFLICT(key,class) DO UPDATE SET weight=weight+excluded.weight, peak=MAX(peak,excluded.peak)",
                                    ((bytes(key), int(code), float(weight), float(weight))
                                     for key, code, weight in zip(keys[rows], codes, values)))
                if self.station_tracking:
                    if station_id is None or not 0 < int(station_id) <= 2**32-1:
                        raise ValueError("SLAM vote requires a valid station ID")
                    self.db.executemany(
                        "INSERT INTO station_peaks VALUES (?,?,?,?) ON CONFLICT(key,class,station) DO UPDATE SET weight=MAX(weight,excluded.weight)",
                        ((bytes(key), int(code), int(station_id), float(weight))
                         for key, code, weight in zip(keys[rows], codes, values)))
            self.db.execute("INSERT INTO scans VALUES (?,?)", (scan, identity))
        self._invalidate_lookup()

    def rebuild_scan_observations(self, representatives_by_scan, checkpoint=lambda: None):
        """Restore per-scan voxel evidence from retained representative arrays.

        Older ``keep_all`` caches predate ``scan_observations`` but retain one
        representative array per static scan.  Rebuilding into a reclassify
        copy preserves the source cache and makes scene-aware postprocessing
        available to those runs as well.
        """
        if self.db.execute('SELECT 1 FROM scan_observations LIMIT 1').fetchone() is not None:
            return False
        inserted = 0
        with self.db:
            for scan, path in sorted(representatives_by_scan.items()):
                checkpoint()
                values = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    keys = values['key'] if values.dtype.names and 'key' in values.dtype.names else values
                    for start in range(0, len(keys), 100_000):
                        batch = keys[start:start+100_000]
                        self.db.executemany('INSERT OR IGNORE INTO scan_observations VALUES (?,?)',
                                            ((bytes(key), int(scan)) for key in batch))
                        inserted += len(batch)
                finally:
                    values._mmap.close()
        return bool(inserted)

    def decide(self, min_observations=1, min_share=.4, class_order=(), checkpoint=lambda: None, priority=None,
               vote_families=()):
        order = {int(code): i for i, code in enumerate(class_order)}
        self.db.execute("DELETE FROM decisions")
        self._invalidate_lookup()
        self.priority_overrides = 0
        cursor = self.db.execute("SELECT v.key,v.class,v.weight,o.count,v.peak FROM votes v JOIN observations o ON o.key=v.key ORDER BY v.key,v.class")
        pending = []
        carry = []

        def flush():
            if pending:
                checkpoint()
                self.db.executemany("INSERT INTO decisions VALUES (?,?,?)", pending)
                pending.clear()

        def finish(rows):
            keys = np.frombuffer(b''.join(row[0] for row in rows), dtype='V24')
            codes = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
            weights = np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows))
            observations = np.fromiter((row[3] for row in rows), dtype=np.int64, count=len(rows))
            peaks = np.fromiter((row[4] for row in rows), dtype=np.float64, count=len(rows))
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
            groups = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rows)]))
            positions = np.arange(len(rows))-starts[groups]
            lengths = np.bincount(groups)
            total = np.zeros(len(starts), np.float64)
            maximum = np.full(len(starts), -np.inf)
            winner = np.zeros(len(starts), np.int64)
            winner_order = np.full(len(starts), np.iinfo(np.int64).max)
            for position in range(int(lengths.max())):
                ids = np.flatnonzero(positions == position)
                grouped = groups[ids]
                values = weights[ids]
                total[grouped] += values
                candidate_order = np.fromiter((order.get(int(code), 1000+int(code)) for code in codes[ids]),
                                                dtype=np.int64, count=len(ids))
                take = (values > maximum[grouped]) | ((values == maximum[grouped]) & (candidate_order < winner_order[grouped]))
                if np.any(take):
                    selected = grouped[take]
                    maximum[selected] = values[take]
                    winner[selected] = codes[ids[take]]
                    winner_order[selected] = candidate_order[take]
            override = np.zeros(len(starts), bool)
            if priority:
                candidate = ((codes == priority['code']) & (peaks >= priority['confidence']))
                eligible = np.zeros(len(starts), bool)
                eligible[groups[candidate]] = True
                priority_weight = np.zeros(len(starts), np.float64)
                priority_rows = np.flatnonzero(candidate)
                priority_groups = groups[priority_rows]
                priority_weight[priority_groups] = weights[priority_rows]
                priority_share = np.divide(priority_weight, total, out=np.zeros_like(priority_weight), where=total > 0)
                override = (np.isin(winner, priority['over']) & eligible
                            & (observations[starts] >= min_observations)
                            & (priority_share >= priority.get('min_share', 0.)))
                if np.any(override):
                    # Every key/class pair is unique, hence at most one row per group.
                    winner[override] = priority['code']
                    maximum[override] = priority_weight[override]
                    self.priority_overrides += int(override.sum())
            share = maximum/total
            family_share = np.zeros(len(starts), np.float64)
            family_winner = winner.copy()
            # A family only rescues split evidence. Clear individual winners
            # remain untouched and ties retain the configured class order.
            for family in vote_families:
                members = np.asarray(tuple(family), np.int64)
                selected_rows = np.isin(codes, members)
                if not selected_rows.any():
                    continue
                sums = np.bincount(groups[selected_rows], weights=weights[selected_rows], minlength=len(starts))
                candidate_weight = np.full(len(starts), -np.inf)
                candidate_code = np.zeros(len(starts), np.int64)
                candidate_order = np.full(len(starts), np.iinfo(np.int64).max)
                for position in range(int(lengths.max())):
                    ids = np.flatnonzero((positions == position) & selected_rows)
                    grouped = groups[ids]
                    values = weights[ids]
                    choices = np.fromiter((order.get(int(code), 1000+int(code)) for code in codes[ids]),
                                            dtype=np.int64, count=len(ids))
                    take = (values > candidate_weight[grouped]) | ((values == candidate_weight[grouped])
                            & (choices < candidate_order[grouped]))
                    if take.any():
                        picked = grouped[take]
                        candidate_weight[picked] = values[take]
                        candidate_code[picked] = codes[ids[take]]
                        candidate_order[picked] = choices[take]
                shares = sums/total
                take = shares > family_share
                family_share[take], family_winner[take] = shares[take], candidate_code[take]
            rescued = (share < min_share) & (family_share >= min_share) & ~override
            winner[rescued] = family_winner[rescued]
            share[rescued] = family_share[rescued]
            decided = (observations[starts] >= min_observations) & ((share >= min_share) | override)
            confidence = np.fromiter((round(float(value)*255) for value in share), dtype=np.int64, count=len(share))
            confidence[~decided] = 0
            winner[~decided] = 1
            pending.extend((bytes(key), int(code), int(value)) for key, code, value in zip(keys[starts], winner, confidence))
            if len(pending) >= 50_000:
                flush()

        while rows := cursor.fetchmany(1_000_000):
            rows = carry+rows
            last = rows[-1][0]
            split = len(rows)
            while split and rows[split-1][0] == last:
                split -= 1
            if split:
                finish(rows[:split])
            carry = rows[split:]
        if carry:
            finish(carry)
        flush()
        self.db.commit()

    def finalize_station_ids(self):
        """Store the strongest station for each final SLAM voxel class."""
        if not self.station_tracking:
            return
        with self.db:
            self.db.execute("DELETE FROM station_decisions")
            self.db.execute("""
                INSERT INTO station_decisions(key,station)
                SELECT d.key, COALESCE((SELECT p.station FROM station_peaks p
                  WHERE p.key=d.key AND p.class=d.code
                  ORDER BY p.weight DESC, p.station ASC LIMIT 1), 0)
                FROM decisions d
            """)

    def lookup_station_ids(self, keys):
        """Return final winning station IDs, or zero for static/unobserved voxels."""
        keys = np.asarray(keys, 'V24')
        result = np.zeros(len(keys), np.uint32)
        if not self.station_tracking or not len(keys):
            return result
        unique, inverse = np.unique(keys, return_inverse=True)
        with self.db:
            self.db.execute("CREATE TEMP TABLE IF NOT EXISTS lookup_station_keys(key BLOB PRIMARY KEY) WITHOUT ROWID")
            self.db.execute("DELETE FROM lookup_station_keys")
            self.db.executemany("INSERT INTO lookup_station_keys VALUES (?)", ((bytes(key),) for key in unique))
            rows = self.db.execute("SELECT s.key,s.station FROM station_decisions s JOIN lookup_station_keys k ON k.key=s.key").fetchall()
        mapping = {row[0]: row[1] for row in rows}
        values = np.fromiter((mapping.get(bytes(key), 0) for key in unique), dtype=np.uint32, count=len(unique))
        return values[inverse]

    def resolve_surface_ties(self, by_id, origin, voxel_size, tie_share=.55, normal_radius=.06,
                             min_observations=1, scanner_heights=(), checkpoint=lambda: None):
        """Use local evidence geometry for uncertain wall/ceiling/floor votes.

        Normals use only the bounded neighbourhood of the candidate voxel.
        Evidence is read through the primary-key index in SQLite; no complete
        cloud-sized coordinate array or KD-tree is retained in RAM.
        """
        surface = {by_id[name]: name for name in ('wall', 'ceiling', 'floor') if name in by_id}
        result = {'wall': 0, 'ceiling': 0, 'floor': 0, 'candidates': 0, 'unchanged': 0}
        if len(surface) < 2:
            return result
        origin = np.asarray(origin, np.float64)
        radius = max(1, int(np.ceil(normal_radius/voxel_size)))
        offsets = np.array([(x, y, z) for x in range(-radius, radius+1) for y in range(-radius, radius+1)
                            for z in range(-radius, radius+1) if x*x+y*y+z*z <= radius*radius], np.int64)
        cursor = self.db.execute('SELECT v.key,v.class,v.weight,o.count FROM votes v JOIN observations o ON o.key=v.key ORDER BY v.key,v.class')
        updates, carry, processed = [], [], 0
        scanner_height = float(np.median(scanner_heights)) if len(scanner_heights) else float(origin[2])

        def normal_at(key):
            index = np.frombuffer(key, dtype='<i8', count=3)
            keys = np.ascontiguousarray(index+offsets, dtype='<i8').view('V24').reshape(-1)
            nearby = []
            for start in range(0, len(keys), 500):
                part = [bytes(item) for item in keys[start:start+500]]
                statement = 'SELECT key FROM observations WHERE key IN ('+','.join('?' for _ in part)+')'
                nearby.extend(row[0] for row in self.db.execute(statement, part))
            if len(nearby) < 6:
                return None
            points = origin+(np.frombuffer(b''.join(nearby), dtype='<i8').reshape(-1, 3)+.5)*voxel_size
            centered = points-points.mean(axis=0)
            return np.linalg.svd(centered, full_matrices=False)[2][-1]

        def process(group):
            total = sum(float(row[2]) for row in group)
            ordered = sorted(group, key=lambda row: (-float(row[2]), int(row[1])))
            if total and len(ordered) >= 2:
                first, second = ordered[:2]
                share = float(first[2])/total
                if (min_observations <= int(first[3]) and .5 <= share < tie_share
                        and int(first[1]) in surface and int(second[1]) in surface):
                    weights = {int(row[1]): float(row[2])/total for row in group}
                    result['candidates'] += 1
                    normal = normal_at(bytes(first[0]))
                    if normal is None:
                        result['unchanged'] += 1
                        return
                    slope = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
                    first_code, second_code = int(first[1]), int(second[1])
                    names = {surface[first_code], surface[second_code]}
                    selected = first_code
                    if slope > 60 and 'wall' in names:
                        selected = by_id['wall']
                    elif slope < 30:
                        horizontal = [name for name in ('floor', 'ceiling') if name in names]
                        if len(horizontal) == 1:
                            selected = by_id[horizontal[0]]
                        elif len(horizontal) == 2:
                            center_z = origin[2]+(np.frombuffer(first[0], dtype='<i8', count=3)[2]+.5)*voxel_size
                            selected = by_id['ceiling'] if center_z >= scanner_height else by_id['floor']
                    elif 'ceiling' in names:
                        selected = by_id['ceiling']
                    if selected != first_code:
                        updates.append((selected, int(round(weights[selected]*255)), bytes(first[0])))
                        result[surface[selected]] += 1

        while rows := cursor.fetchmany(50_000):
            rows = carry+rows
            last, split = rows[-1][0], len(rows)
            while split and rows[split-1][0] == last:
                split -= 1
            start = 0
            while start < split:
                stop = start+1
                while stop < split and rows[stop][0] == rows[start][0]:
                    stop += 1
                process(rows[start:stop])
                processed += 1
                if processed % 250 == 0:
                    checkpoint()
                if len(updates) >= 20_000:
                    self.db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?', updates)
                    updates.clear()
                start = stop
            carry = rows[split:]
        if carry:
            process(carry)
        if updates:
            self.db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?', updates)
        self.db.commit()
        self._invalidate_lookup()
        return result

    @staticmethod
    def _sparse_components(coordinates):
        """Return 26-neighbour labels without allocating a dense voxel grid."""
        key_dtype = np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])
        values = np.ascontiguousarray(coordinates).view(key_dtype).reshape(-1)
        order = np.argsort(values, kind='stable')
        sorted_values = values[order]
        edge_rows, edge_columns = [], []
        # The 13 forward offsets enumerate every undirected 26-neighbour edge
        # exactly once. Sparse components then match the old DFS.
        for dx in (0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if ((dx, dy, dz) == (0, 0, 0) or (dx == 0 and dy < 0)
                            or (dx == 0 and dy == 0 and dz < 0)):
                        continue
                    target = coordinates+np.array((dx, dy, dz), np.int64)
                    target_values = np.ascontiguousarray(target).view(key_dtype).reshape(-1)
                    found = np.searchsorted(sorted_values, target_values)
                    present = found < len(sorted_values)
                    present[present] &= sorted_values[found[present]] == target_values[present]
                    ids = np.flatnonzero(present)
                    if len(ids):
                        edge_rows.append(ids)
                        edge_columns.append(order[found[ids]])
        if edge_rows:
            source, target = np.concatenate(edge_rows), np.concatenate(edge_columns)
            graph = coo_matrix((np.ones(len(source), np.uint8), (source, target)),
                               shape=(len(coordinates), len(coordinates)))
        else:
            graph = coo_matrix((len(coordinates), len(coordinates)), dtype=np.uint8)
        return connected_components(graph, directed=False, return_labels=True)

    @staticmethod
    def _stair_levels(coordinates, voxel_size, level_tolerance):
        """Extract broad horizontal levels from a stair component's voxels."""
        horizontal = np.column_stack((np.floor(coordinates[:, :2]*voxel_size/.10).astype(np.int64), coordinates[:, 2]))
        horizontal = np.unique(horizontal, axis=0)
        heights, counts = np.unique(horizontal[:, 2], return_counts=True)
        if not len(heights):
            return np.empty(0), np.empty((0, 2))
        # A landing can cover far more 10 cm cells than each tread. Keep every
        # broad horizontal level and exclude only the narrow vertical risers.
        minimum_cells = 3
        heights, counts = heights[counts >= minimum_cells], counts[counts >= minimum_cells]
        if not len(heights):
            return np.empty(0), np.empty((0, 2))
        starts = np.r_[0, np.flatnonzero(np.diff(heights)*voxel_size > level_tolerance)+1, len(heights)]
        levels, centres = [], []
        for start, stop in zip(starts[:-1], starts[1:]):
            selected, weights = heights[start:stop], counts[start:stop]
            height = float(np.average(selected*voxel_size, weights=weights))
            points = coordinates[np.isin(coordinates[:, 2], selected), :2]*voxel_size
            levels.append(height)
            centres.append(points.mean(axis=0))
        return np.asarray(levels), np.asarray(centres)

    @staticmethod
    def _stair_shape(levels, centres, minimum_levels, rise_minimum, rise_maximum, tread_minimum, tread_maximum):
        """Accept one or more monotone flights separated by a landing."""
        if len(levels) < minimum_levels:
            return False
        rises = np.diff(levels)
        if not np.all((rises >= rise_minimum) & (rises <= rise_maximum)):
            return False
        vectors = np.diff(centres, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        flights, current = [], []
        for index, (vector, length) in enumerate(zip(vectors, lengths)):
            if not tread_minimum <= length <= tread_maximum:
                if current:
                    flights.append(current)
                    current = []
                continue
            direction = vector/length
            # A landing may turn the next flight by 90 or 180 degrees.  Split
            # at that turn and test each run in its own forward direction.
            if current and float(direction @ current[-1][1]) < .5:
                flights.append(current)
                current = []
            current.append((index, direction))
        if current:
            flights.append(current)
        return any(len(flight) >= minimum_levels-1 for flight in flights)

    @staticmethod
    def _boxes_touch(first, second, maximum_gap):
        """Return whether two axis-aligned component boxes meet within a gap."""
        return np.maximum(first[0]-second[1], second[0]-first[1]).max() <= maximum_gap

    def _restore_stairs(self, keys, indices, stairs_code, order, result, checkpoint):
        """Restore rejected stair voxels to their strongest permitted alternative."""
        for start in range(0, len(indices), 500):
            checkpoint()
            component_keys = [keys[index] for index in indices[start:start+500]]
            evidence = {key: [] for key in component_keys}
            statement = 'SELECT key,class,weight FROM votes WHERE key IN ('+','.join('?' for _ in component_keys)+')'
            for key, code, weight in self.db.execute(statement, component_keys):
                evidence[bytes(key)].append((int(code), float(weight)))
            updates = []
            for key in component_keys:
                votes = evidence[key]
                alternatives = [item for item in votes if item[0] != stairs_code]
                if not alternatives:
                    updates.append((1, 0, key))
                    result['reclassified_voxels']['1'] = result['reclassified_voxels'].get('1', 0)+1
                    continue
                code, weight = min(alternatives, key=lambda item: (-item[1], order.get(item[0], 1000+item[0])))
                total = sum(item[1] for item in votes)
                confidence = int(round(weight*255/total)) if total else 0
                updates.append((code, confidence, key))
                target = str(code)
                result['reclassified_voxels'][target] = result['reclassified_voxels'].get(target, 0)+1
            self.db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?', updates)

    def reject_planar_stairs(self, stairs_code, origin, voxel_size, plane_tolerance, planar_fraction,
                             minimum_rise, seed, class_order=(), checkpoint=lambda: None, *, minimum_voxels=50,
                             minimum_levels=3, level_tolerance=.03, rise_minimum=.10, rise_maximum=.23,
                             tread_minimum=.20, tread_maximum=.45, floor_code=None, floor_ring_radius=2.,
                             floor_level_tolerance=.03, underside_minimum_voxels=1000,
                             underside_cluster_size=3, underside_maximum_gap=.20):
        """Keep only components with a physically plausible stepped profile."""
        result = {'checked_components': 0, 'rejected_components': 0, 'reclassified_voxels': {}}
        rows = list(self.db.execute('SELECT key FROM decisions WHERE code=?', (int(stairs_code),)))
        if not rows:
            self._invalidate_lookup()
            return result
        keys = [bytes(row[0]) for row in rows]
        coordinates = np.frombuffer(b''.join(keys), dtype='<i8').reshape(-1, 3)
        count, labels = self._sparse_components(coordinates)
        order = {int(code): index for index, code in enumerate(class_order)}
        origin = np.asarray(origin, np.float64)
        component_order = np.argsort(labels, kind='stable')
        component_offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
        floor_tree = floor_heights = None
        if floor_code is not None:
            floor_rows = list(self.db.execute('SELECT key FROM decisions WHERE code=?', (int(floor_code),)))
            if floor_rows:
                floor_coordinates = np.frombuffer(b''.join(bytes(row[0]) for row in floor_rows), dtype='<i8').reshape(-1, 3)
                floor_points = origin+(floor_coordinates+.5)*voxel_size
                floor_tree, floor_heights = cKDTree(floor_points[:, :2]), floor_points[:, 2]

        component_data = []
        for component in range(count):
            indices = component_order[component_offsets[component]:component_offsets[component+1]]
            checkpoint()
            result['checked_components'] += 1
            component_coordinates = coordinates[indices]
            levels, centres = self._stair_levels(component_coordinates, voxel_size, level_tolerance)
            stepped = len(indices) >= minimum_voxels and self._stair_shape(
                levels, centres, minimum_levels, rise_minimum, rise_maximum, tread_minimum, tread_maximum)
            returned = np.empty(0, np.int64)
            if floor_tree is not None and len(levels):
                distance, nearby = floor_tree.query(centres, distance_upper_bound=floor_ring_radius, workers=-1)
                matching = np.zeros(len(levels), bool)
                valid_floor = np.isfinite(distance) & (nearby < len(floor_heights))
                matching[valid_floor] = np.abs(floor_heights[nearby[valid_floor]]-levels[valid_floor]) <= floor_level_tolerance
                if matching.any():
                    height_match = np.zeros(len(indices), bool)
                    for level in levels[matching]:
                        height_match |= np.abs((component_coordinates[:, 2]+.5)*voxel_size-level) <= level_tolerance
                    returned = indices[height_match]
                    self.db.executemany('UPDATE decisions SET code=?,confidence=confidence WHERE key=?',
                                        ((int(floor_code), keys[index]) for index in returned))
                    target = str(int(floor_code))
                    result['reclassified_voxels'][target] = result['reclassified_voxels'].get(target, 0)+len(returned)
            # Sparse scans often retain the inclined underside of each flight
            # but not enough tread tops for the level test.  A single slope is
            # deliberately not accepted: it is indistinguishable from a roof
            # or ceiling.  Several large, touching flights (or an underside
            # next to a verified stepped flight) are a staircase assembly.
            span = (component_coordinates.max(axis=0)-component_coordinates.min(axis=0)+1)*voxel_size
            underside = (len(indices) >= underside_minimum_voxels and span[2] >= minimum_rise
                         and np.linalg.norm(span[:2]) >= tread_minimum)
            component_data.append({'indices': indices, 'stepped': stepped, 'underside': underside,
                                   'box': np.stack((component_coordinates.min(axis=0)*voxel_size,
                                                    component_coordinates.max(axis=0)*voxel_size))})

        eligible = np.fromiter((data['stepped'] or data['underside'] for data in component_data), bool, count)
        eligible_components = np.flatnonzero(eligible)
        boxes = np.asarray([component_data[index]['box'] for index in eligible_components]).reshape(-1, 2, 3)
        group_labels = _touching_box_labels(boxes[:, 0], boxes[:, 1], underside_maximum_gap)
        groups = {}
        for component, group in zip(eligible_components, group_labels):
            groups.setdefault(int(group), []).append(int(component))
        accepted = set()
        for members in groups.values():
            has_steps = any(component_data[item]['stepped'] for item in members)
            enough_flights = sum(component_data[item]['underside'] for item in members) >= underside_cluster_size
            if has_steps or enough_flights:
                accepted.update(item for item in members if component_data[item]['stepped'] or component_data[item]['underside'])

        for component, data in enumerate(component_data):
            if component in accepted:
                continue
            result['rejected_components'] += 1
            indices = data['indices']
            # A component can contain a horizontal floor-level fragment even
            # when the full component is rejected.  Preserve that correction.
            component_coordinates = coordinates[indices]
            levels, centres = self._stair_levels(component_coordinates, voxel_size, level_tolerance)
            returned = np.empty(0, np.int64)
            if floor_tree is not None and len(levels):
                distance, nearby = floor_tree.query(centres, distance_upper_bound=floor_ring_radius, workers=-1)
                matching = np.zeros(len(levels), bool)
                valid_floor = np.isfinite(distance) & (nearby < len(floor_heights))
                matching[valid_floor] = np.abs(floor_heights[nearby[valid_floor]]-levels[valid_floor]) <= floor_level_tolerance
                if matching.any():
                    height_match = np.zeros(len(indices), bool)
                    for level in levels[matching]:
                        height_match |= np.abs((component_coordinates[:, 2]+.5)*voxel_size-level) <= level_tolerance
                    returned = indices[height_match]
            remaining = indices[~np.isin(indices, returned)]
            self._restore_stairs(keys, remaining, stairs_code, order, result, checkpoint)
        self.db.commit()
        self._invalidate_lookup()
        return result

    def remove_small_components(self, voxel_size, default_minimum, special_rules=(), checkpoint=lambda: None,
                                *, component_min_cells=1, feature_cell_size=.10,
                                sparse_maximum_voxels_per_cell=1, outdoor_codes=()):
        """Turn isolated semantic voxel islands back into class 1.

        Components use 26-neighbour connectivity so a thin diagonal cable is
        not broken into artificial fragments.  A special rule is
        ``(code, minimum_voxels, minimum_span_m)``; other classes use the
        default minimum and no span requirement.
        """
        rules = {int(code): (int(minimum), float(span)) for code, minimum, span in special_rules}
        outdoor_codes = {int(code) for code in outdoor_codes}
        removed, components = 0, 0
        codes = [row[0] for row in self.db.execute('SELECT DISTINCT code FROM decisions WHERE code != 1')]
        feature_cells_available = self.db.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='geometry_voxels'").fetchone() is not None
        for code in codes:
            class_started = time.perf_counter()
            if feature_cells_available and code not in rules:
                rows = list(self.db.execute('''SELECT d.key,g.cx,g.cy,g.cz,COALESCE(c.outdoor,0),COALESCE(c.indoor,0)
                                             FROM decisions d JOIN geometry_voxels g USING(key)
                                             LEFT JOIN geometry_context c USING(key) WHERE d.code=?''', (code,)))
                cells = np.asarray([row[1:] for row in rows], np.int64) if rows else np.empty((0, 3), np.int64)
                contexts = cells[:, 3:] if len(cells) else np.empty((0, 2), np.int64)
                cells = cells[:, :3]
            else:
                rows = list(self.db.execute('SELECT key FROM decisions WHERE code=?', (code,)))
                cells = None
                contexts = None
            if not rows:
                continue
            keys = [bytes(row[0]) for row in rows]
            coordinates = np.frombuffer(b''.join(keys), dtype='<i8').reshape(-1, 3)
            count, labels = self._sparse_components(coordinates)
            minimum, minimum_span = rules.get(code, (default_minimum, 0.))
            sizes = np.bincount(labels, minlength=count)
            low = np.full((count, 3), np.iinfo(np.int64).max, np.int64)
            high = np.full((count, 3), np.iinfo(np.int64).min, np.int64)
            np.minimum.at(low, labels, coordinates)
            np.maximum.at(high, labels, coordinates)
            too_small = ((sizes < minimum) | ((high-low).max(axis=1)*voxel_size < minimum_span))[labels]
            # Cable and electrical exceptions intentionally keep their historic
            # voxel/span rule.  A coarse-cell exception is only for genuinely
            # sparse observations made from an outdoor scan.  Dense interiors,
            # and every component without positive exterior evidence, keep the
            # historic voxel threshold even where 10 cm cells connect.
            if code not in rules and component_min_cells > 1:
                if cells is None:
                    cells = np.floor(coordinates*voxel_size/feature_cell_size).astype(np.int64)
                # Coarse cells may bridge 2 cm gaps in a thin distant raster.
                # Keep that shared connectivity, but measure density per voxel
                # component so dense interior islands cannot borrow a remote
                # sparse grid as an exemption.
                spare = _coarse_component_spares(
                    labels, sizes, cells, contexts, component_min_cells,
                    sparse_maximum_voxels_per_cell, sizes < minimum)
                too_small &= ~spare[labels]
            rejected = np.flatnonzero(too_small)
            components += count
            for start in range(0, len(rejected), 50000):
                checkpoint()
                self.db.executemany('UPDATE decisions SET code=1,confidence=0 WHERE key=?',
                                    ((keys[index],) for index in rejected[start:start+50000]))
            removed += len(rejected)
            LOG.info('Small-component cleanup: class=%s voxels=%d components=%d removed_voxels=%d duration_seconds=%.3f',
                     code, len(keys), count, len(rejected), time.perf_counter()-class_started)
        self.db.commit()
        self._invalidate_lookup()
        return {'components': components, 'removed_voxels': removed}

    def vegetation_bases_to_ground(self, vegetation_code, ground_code, voxel_size, cell_size, band_m, checkpoint=lambda: None):
        """Classify the bottom layer of low vegetation as ground.

        Lowest vegetation voxels are determined independently in small XY cells;
        this follows gentle terrain changes and does not flatten an entire bush
        to one global elevation.  Only the configured vertical band is moved.
        """
        rows = list(self.db.execute('SELECT key FROM decisions WHERE code=?', (vegetation_code,)))
        if not rows:
            return {'moved_voxels': 0, 'cells': 0}
        cells = max(1, round(cell_size/voxel_size))
        band = max(0, round(band_m/voxel_size))
        keys = [bytes(row[0]) for row in rows]
        positions = np.frombuffer(b''.join(keys), dtype='<i8').reshape(-1, 3)
        xy = positions[:, :2] // cells
        _, inverse = np.unique(xy, axis=0, return_inverse=True)
        floor = np.full(int(inverse.max())+1, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(floor, inverse, positions[:, 2])
        moved = [keys[index] for index in np.flatnonzero(positions[:, 2] <= floor[inverse]+band)]
        for start in range(0, len(moved), 50000):
            checkpoint()
            self.db.executemany('UPDATE decisions SET code=?,confidence=confidence WHERE key=?',
                                ((ground_code, key) for key in moved[start:start+50000]))
        self.db.commit()
        self._invalidate_lookup()
        return {'moved_voxels': len(moved), 'cells': len(floor)}

    def building_outline(self, barrier_codes, voxel_size, cell_size, minimum_height, maximum_cells):
        """Compute the 2-D building envelope raster (inside, outside, barrier, origin, cell size)."""
        barrier_tuple = tuple(sorted({int(code) for code in barrier_codes}))
        if not barrier_tuple:
            return {'applied': False, 'reason': 'no_barrier_classes'}
        clause = ','.join('?' for _ in barrier_tuple)
        rows = list(self.db.execute(f'SELECT key FROM decisions WHERE code IN ({clause})', barrier_tuple))
        if not rows:
            return {'applied': False, 'reason': 'no_building_or_wall_points'}
        building = np.frombuffer(b''.join(bytes(row[0]) for row in rows), dtype='<i8').reshape(-1, 3)
        cells = max(1, round(cell_size/voxel_size))
        xy = building[:, :2] // cells
        unique, inverse = np.unique(xy, axis=0, return_inverse=True)
        low = np.full(len(unique), np.iinfo(np.int64).max, np.int64)
        high = np.full(len(unique), np.iinfo(np.int64).min, np.int64)
        np.minimum.at(low, inverse, building[:, 2])
        np.maximum.at(high, inverse, building[:, 2])
        barrier = unique[(high-low+1)*voxel_size >= minimum_height]
        if not len(barrier):
            return {'applied': False, 'reason': 'no_vertical_facade_cells'}

        lo = barrier.min(axis=0)-1
        hi = barrier.max(axis=0)+1
        shape = tuple((hi-lo+1).tolist())
        if int(np.prod(shape, dtype=np.int64)) > maximum_cells:
            return {'applied': False, 'reason': 'building_extent_too_large', 'barrier_cells': int(len(barrier)),
                    'grid_cells': int(np.prod(shape, dtype=np.int64))}
        blocked = np.zeros(shape, bool)
        indices = barrier-lo
        blocked[indices[:, 0], indices[:, 1]] = True
        seed = np.zeros(shape, bool)
        seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = True, True, True, True
        outside = binary_propagation(seed & ~blocked, structure=generate_binary_structure(2, 1), mask=~blocked)
        inside = ~outside & ~blocked
        return {
            'applied': True,
            'inside': inside,
            'outside': outside,
            'barrier': blocked,
            'origin': lo,
            'cells': cells,
            'cell_size_m': cells*voxel_size,
            'barrier_cells': int(len(barrier)),
            'grid_cells': int(np.prod(shape, dtype=np.int64)),
            'shape': shape,
            'has_enclosed': bool(inside.any()),
        }

    def split_ground_floor_by_building(self, ground_code, floor_code, barrier_codes, voxel_size,
                                       cell_size, minimum_height, maximum_cells, checkpoint=lambda: None, outline=None):
        """Use observed, vertical building faces as a 2-D ground/floor barrier.

        Building and wall voxels form a barrier only where the same XY cell has
        a measured vertical extent.  Nothing is extrapolated between separate
        faces: the exterior is flood-filled solely through observed openings.
        Ground/floor labels are changed only when this produces an actual
        enclosed region, preventing an incomplete facade from rewriting a
        whole project.
        """
        outline = outline if outline is not None else self.building_outline(
            barrier_codes, voxel_size, cell_size, minimum_height, maximum_cells)
        if not outline.get('applied'):
            return {key: value for key, value in outline.items() if key not in {'inside', 'outside', 'barrier'}}

        target_rows = list(self.db.execute('SELECT key,code FROM decisions WHERE code IN (?,?)',
                                           (int(ground_code), int(floor_code))))
        if not target_rows:
            return {'applied': False, 'reason': 'no_ground_or_floor_points',
                    'barrier_cells': int(outline['barrier_cells'])}
        target_keys = [bytes(row[0]) for row in target_rows]
        target_codes = np.asarray([row[1] for row in target_rows], np.uint8)
        target = np.frombuffer(b''.join(target_keys), dtype='<i8').reshape(-1, 3)
        cells = int(outline['cells'])
        target_xy = target[:, :2] // cells
        lo = np.asarray(outline['origin'], np.int64)
        shape = np.asarray(outline['shape'], np.int64)
        high = lo+shape-1
        blocked, outside = outline['barrier'], outline['outside']
        in_grid = ((target_xy >= lo) & (target_xy <= high)).all(axis=1)
        grid_index = target_xy[in_grid]-lo
        inside = np.zeros(len(target), bool)
        inside[in_grid] = ~outside[grid_index[:, 0], grid_index[:, 1]] & ~blocked[grid_index[:, 0], grid_index[:, 1]]
        if not inside.any():
            return {'applied': False, 'reason': 'no_enclosed_floor_area', 'barrier_cells': int(outline['barrier_cells']),
                    'grid_cells': int(outline['grid_cells'])}

        # No nearest-neighbour or gap interpolation happens here.  A target is
        # inside only if an observed facade barrier blocks every 4-connected
        # path to the envelope edge; every other target is exterior.
        wanted = np.where(inside, int(floor_code), int(ground_code)).astype(np.uint8)
        changed = wanted != target_codes
        for start in range(0, int(changed.sum()), 50_000):
            checkpoint()
            ids = np.flatnonzero(changed)[start:start+50_000]
            self.db.executemany('UPDATE decisions SET code=?,confidence=confidence WHERE key=?',
                                ((int(wanted[index]), target_keys[index]) for index in ids))
        self.db.commit()
        self._invalidate_lookup()
        return {'applied': True, 'barrier_cells': int(outline['barrier_cells']), 'grid_cells': int(outline['grid_cells']),
                'inside_voxels': int(inside.sum()), 'ground_voxels': int((~inside).sum()),
                'changed_voxels': int(changed.sum()), 'cell_size_m': float(outline['cell_size_m'])}

    def lookup(self, keys):
        keys = np.ascontiguousarray(keys, dtype='V24').reshape(-1)
        known, values = self._lookup_snapshot()
        result = np.empty((len(keys), 2), np.uint8)
        result[:] = (1, 0)
        if len(known) and len(keys):
            found = np.searchsorted(known, keys)
            present = found < len(known)
            present[present] &= known[found[present]] == keys[present]
            result[present] = values[found[present]]
        return result[:, 0], result[:, 1]
