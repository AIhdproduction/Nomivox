"""SQLite-backed spatial grouping with one observation per scan position."""

import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.ndimage import binary_propagation, generate_binary_structure

from e57cls.files import atomic_json, file_hash
from e57cls.geometry.cameras import Pose


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def voxel_keys(world, origin, size):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    values = np.floor((np.asarray(world, np.float64)-origin)/size)
    if not np.isfinite(values).all() or (np.abs(values) >= 2**62).any():
        raise ValueError("Voxel indices exceed the supported range")
    return np.ascontiguousarray(values, dtype="<i8").view("V24").reshape(-1)


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
        pose = Pose.from_dict(store.meta["metadata"]["pose"])
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
    def __init__(self, path):
        self.db = connect(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS votes(key BLOB, class INTEGER, weight REAL, peak REAL, PRIMARY KEY(key,class)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS observations(key BLOB PRIMARY KEY, count INTEGER) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS scans(scan INTEGER PRIMARY KEY, identity TEXT);
          CREATE TABLE IF NOT EXISTS decisions(key BLOB PRIMARY KEY, code INTEGER, confidence INTEGER) WITHOUT ROWID;
        """)
        if 'peak' not in {row[1] for row in self.db.execute('PRAGMA table_info(votes)')}:
            raise ValueError('Old vote schema requires a new pipeline cache identity')
        self.priority_overrides = 0

    def close(self):
        self.db.close()

    def completed(self, scan, identity):
        return self.db.execute("SELECT 1 FROM scans WHERE scan=? AND identity=?", (scan, identity)).fetchone() is not None

    def add_scan(self, scan, identity, batches, class_codes):
        if self.completed(scan, identity):
            return
        with self.db:
            if self.db.execute("SELECT 1 FROM scans WHERE scan=?", (scan,)).fetchone():
                raise ValueError("Changed voting identity requires a separate vote store")
            for keys, weights in batches:
                has_evidence = np.any(weights > 0, axis=1)
                self.db.executemany("INSERT INTO observations VALUES (?,1) ON CONFLICT(key) DO UPDATE SET count=count+1",
                                    ((bytes(k),) for k in keys[has_evidence]))
                for column, code in enumerate(class_codes):
                    valid = weights[:, column] > 0
                    self.db.executemany("INSERT INTO votes VALUES (?,?,?,?) ON CONFLICT(key,class) DO UPDATE SET weight=weight+excluded.weight, peak=MAX(peak,excluded.peak)",
                                        ((bytes(k), int(code), float(w), float(w)) for k,w in zip(keys[valid], weights[valid, column])))
            self.db.execute("INSERT INTO scans VALUES (?,?)", (scan, identity))

    def decide(self, min_observations=1, min_share=.4, class_order=(), checkpoint=lambda: None, priority=None):
        order = {int(code): i for i, code in enumerate(class_order)}
        self.db.execute("DELETE FROM decisions")
        self.priority_overrides = 0
        cursor = self.db.execute("SELECT v.key,v.class,v.weight,o.count,v.peak FROM votes v JOIN observations o ON o.key=v.key ORDER BY v.key,v.class")
        pending = []
        last, evidence, observations = None, [], 0
        def finish(key, rows, count):
            total = sum(w for _, w, _ in rows)
            code, maximum, _ = min(rows, key=lambda item: (-item[1], order.get(item[0], 1000+item[0])))
            override = False
            if priority and code in priority['over'] and count >= min_observations:
                evidence = next((r for r in rows if r[0] == priority['code'] and r[2] >= priority['confidence']), None)
                if evidence:
                    code,maximum,_ = evidence
                    override = True
                    self.priority_overrides += 1
            share = maximum/total if total else 0
            decided = code if count >= min_observations and (share >= min_share or override) else 1
            return key, decided, round(share*255) if decided != 1 else 0
        for key, code, weight, count, peak in cursor:
            if key != last and last is not None:
                pending.append(finish(last, evidence, observations))
                evidence = []
                if len(pending) >= 50_000:
                    checkpoint()
                    self.db.executemany("INSERT INTO decisions VALUES (?,?,?)", pending)
                    pending.clear()
            last, observations = key, count
            evidence.append((code, weight, peak))
        if last is not None:
            pending.append(finish(last, evidence, observations))
        self.db.executemany("INSERT INTO decisions VALUES (?,?,?)", pending)
        self.db.commit()

    def remove_small_components(self, voxel_size, default_minimum, special_rules=(), checkpoint=lambda: None):
        """Turn isolated semantic voxel islands back into class 1.

        Components use 26-neighbour connectivity so a thin diagonal cable is
        not broken into artificial fragments.  A special rule is
        ``(code, minimum_voxels, minimum_span_m)``; other classes use the
        default minimum and no span requirement.
        """
        rules = {int(code): (int(minimum), float(span)) for code, minimum, span in special_rules}
        removed, components = 0, 0
        codes = [row[0] for row in self.db.execute('SELECT DISTINCT code FROM decisions WHERE code != 1')]
        for code in codes:
            rows = list(self.db.execute('SELECT key FROM decisions WHERE code=?', (code,)))
            if not rows:
                continue
            coordinates = [tuple(np.frombuffer(row[0], dtype='<i8', count=3)) for row in rows]
            positions = {position: index for index, position in enumerate(coordinates)}
            visited = np.zeros(len(coordinates), bool)
            minimum, minimum_span = rules.get(code, (default_minimum, 0.))
            rejected = []
            for start, point in enumerate(coordinates):
                if visited[start]:
                    continue
                stack, group = [start], []
                visited[start] = True
                while stack:
                    index = stack.pop()
                    group.append(index)
                    x, y, z = coordinates[index]
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                if dx == dy == dz == 0:
                                    continue
                                neighbour = positions.get((x+dx, y+dy, z+dz))
                                if neighbour is not None and not visited[neighbour]:
                                    visited[neighbour] = True
                                    stack.append(neighbour)
                components += 1
                group_positions = np.asarray([coordinates[index] for index in group], np.int64)
                span = float(np.ptp(group_positions, axis=0).max()*voxel_size)
                if len(group) < minimum or span < minimum_span:
                    rejected.extend(bytes(rows[index][0]) for index in group)
            for start in range(0, len(rejected), 50000):
                checkpoint()
                self.db.executemany('UPDATE decisions SET code=1,confidence=0 WHERE key=?',
                                    ((key,) for key in rejected[start:start+50000]))
            removed += len(rejected)
        self.db.commit()
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
        positions = [(bytes(row[0]), *np.frombuffer(row[0], dtype='<i8', count=3)) for row in rows]
        floor = {}
        for _, x, y, z in positions:
            key = (x//cells, y//cells)
            floor[key] = min(floor.get(key, z), z)
        moved = [key for key, x, y, z in positions if z <= floor[(x//cells, y//cells)]+band]
        for start in range(0, len(moved), 50000):
            checkpoint()
            self.db.executemany('UPDATE decisions SET code=?,confidence=confidence WHERE key=?',
                                ((ground_code, key) for key in moved[start:start+50000]))
        self.db.commit()
        return {'moved_voxels': len(moved), 'cells': len(floor)}

    def split_ground_floor_by_building(self, ground_code, floor_code, barrier_codes, voxel_size,
                                       cell_size, minimum_height, maximum_cells, checkpoint=lambda: None):
        """Use observed, vertical building faces as a 2-D ground/floor barrier.

        Building and wall voxels form a barrier only where the same XY cell has
        a measured vertical extent.  Nothing is extrapolated between separate
        faces: the exterior is flood-filled solely through observed openings.
        Ground/floor labels are changed only when this produces an actual
        enclosed region, preventing an incomplete facade from rewriting a
        whole project.
        """
        codes = tuple(sorted({int(code) for code in barrier_codes}))
        if not codes:
            return {'applied': False, 'reason': 'no_barrier_classes'}
        clause = ','.join('?' for _ in codes)
        rows = list(self.db.execute(f'SELECT key FROM decisions WHERE code IN ({clause})', codes))
        if not rows:
            return {'applied': False, 'reason': 'no_building_or_wall_points'}
        building = np.asarray([np.frombuffer(row[0], dtype='<i8', count=3) for row in rows], np.int64)
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

        target_rows = list(self.db.execute('SELECT key,code FROM decisions WHERE code IN (?,?)',
                                           (int(ground_code), int(floor_code))))
        if not target_rows:
            return {'applied': False, 'reason': 'no_ground_or_floor_points', 'barrier_cells': int(len(barrier))}
        target_keys = [bytes(row[0]) for row in target_rows]
        target_codes = np.asarray([row[1] for row in target_rows], np.uint8)
        target = np.asarray([np.frombuffer(key, dtype='<i8', count=3) for key in target_keys], np.int64)
        target_xy = target[:, :2] // cells

        # Work only around the actual building envelope.  Points outside are
        # unquestionably exterior, but are updated only after an enclosed
        # interior was proven below.
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

        in_grid = ((target_xy >= lo) & (target_xy <= hi)).all(axis=1)
        grid_index = target_xy[in_grid]-lo
        inside = np.zeros(len(target), bool)
        inside[in_grid] = ~outside[grid_index[:, 0], grid_index[:, 1]] & ~blocked[grid_index[:, 0], grid_index[:, 1]]
        if not inside.any():
            return {'applied': False, 'reason': 'no_enclosed_floor_area', 'barrier_cells': int(len(barrier)),
                    'grid_cells': int(np.prod(shape, dtype=np.int64))}

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
        return {'applied': True, 'barrier_cells': int(len(barrier)), 'grid_cells': int(np.prod(shape, dtype=np.int64)),
                'inside_voxels': int(inside.sum()), 'ground_voxels': int((~inside).sum()),
                'changed_voxels': int(changed.sum()), 'cell_size_m': cells*voxel_size}

    def lookup(self, keys):
        unique, inverse = np.unique(keys, return_inverse=True)
        rows = {}
        for start in range(0, len(unique), 500):
            part = [bytes(k) for k in unique[start:start+500]]
            statement = "SELECT key,code,confidence FROM decisions WHERE key IN ("+",".join("?" for _ in part)+")"
            rows.update((k, (code, confidence)) for k,code,confidence in self.db.execute(statement, part))
        values = np.array([rows.get(bytes(k), (1, 0)) for k in unique], np.uint8).reshape(-1, 2)
        result = values[inverse]
        return result[:, 0], result[:, 1]
