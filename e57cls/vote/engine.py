"""SQLite-backed spatial grouping with one observation per scan position."""

import json
from pathlib import Path
import sqlite3

import numpy as np

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
