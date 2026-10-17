"""Shared bounded classification and direct-neighbour transfer for writers."""

from collections import OrderedDict
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from e57cls.geometry.cameras import Pose
from e57cls.geometry.fill import PlaneIndex
from e57cls.vote.engine import voxel_keys


def classify_points(store, start, stop, xyz, valid, votes, origin, voxel_size,
                    skipped=(), mirror=None, world=None):
    """Classify the valid rows of one source block without transferring labels."""
    if world is None:
        world = Pose.from_dict(store.meta["metadata"]["pose"]).to_world(xyz[valid])
    codes, confidence = votes.lookup(voxel_keys(world, origin, voxel_size))
    if store.meta["scan_index"] in skipped:
        codes[:] = 1
        confidence[:] = 0
    states = np.zeros(len(world), np.uint8)
    if mirror:
        codes, states = mirror(store, start, stop, valid, world, codes)
    return world, codes, confidence, states


def output_chunks(store, density, capacity=500_000):
    if density == "original" and hasattr(store, "full_chunks"):
        yield from store.full_chunks(capacity)
    else:
        yield from store.chunks(capacity)


def selected_count(store, density):
    return store.meta["valid_count"] if density == "original" else getattr(store, "retained_count", store.meta["valid_count"])


class ClassPointIndex:
    """Disk-backed lookup with deterministic equal-distance handling."""
    def __init__(self, path, origin, block=.05):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE fragments(x INTEGER,y INTEGER,z INTEGER,points BLOB,codes BLOB)')
        self.db.execute('CREATE INDEX spatial ON fragments(x,y,z)')
        self.origin, self.block, self.cache = np.asarray(origin), block, OrderedDict()

    def add(self, points, codes):
        if not len(points): return
        bins = np.floor((points-self.origin)/self.block).astype(np.int64)
        cells, inverse = np.unique(bins, axis=0, return_inverse=True)
        order, offsets = np.argsort(inverse, kind='stable'), np.r_[0, np.cumsum(np.bincount(inverse))]
        rows = []
        for index, cell in enumerate(cells):
            ids = order[offsets[index]:offsets[index+1]]
            rows.append((*map(int, cell), np.ascontiguousarray(points[ids], '<f8').tobytes(), np.ascontiguousarray(codes[ids], np.uint8).tobytes()))
        self.db.executemany('INSERT INTO fragments VALUES(?,?,?,?,?)', rows)

    def nearest_codes(self, points, radius):
        result, best = np.zeros(len(points), np.uint8), np.full(len(points), np.inf)
        bins = np.floor((points-self.origin)/self.block).astype(np.int64)
        cells, inverse = np.unique(bins, axis=0, return_inverse=True)
        order, offsets, reach = np.argsort(inverse, kind='stable'), np.r_[0, np.cumsum(np.bincount(inverse))], int(np.ceil(radius/self.block))
        for number, cell in enumerate(cells):
            ids = order[offsets[number]:offsets[number+1]]; low, high = cell-reach, cell+reach
            query = ('SELECT rowid,points,codes FROM fragments WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ? ORDER BY x,y,z,rowid')
            for rowid, xyz, codes in self.db.execute(query, tuple(map(int, (low[0],high[0],low[1],high[1],low[2],high[2])))):
                if rowid not in self.cache:
                    self.cache[rowid] = (cKDTree(np.frombuffer(xyz, '<f8').reshape(-1, 3)), np.frombuffer(codes, np.uint8))
                    if len(self.cache) > 64: self.cache.popitem(last=False)
                tree, labels = self.cache[rowid]; self.cache.move_to_end(rowid)
                distances, local = tree.query(points[ids], distance_upper_bound=radius); take = distances < best[ids]
                if np.any(take):
                    selected = ids[take]; best[selected] = distances[take]; result[selected] = labels[local[take]]
        return result

    def close(self): self.cache.clear(); self.db.close()


class ClassifiedSource:
    """Reuse reduced direct predictions and transfer only from those predictions."""
    def __init__(self, directory, stores, votes, origin, voxel_size, skipped=(), mirror=None, radius=0., checkpoint=lambda: None,
                 geometry_planes=None, geometry_fill_tolerance=.02):
        self.directory, self.stores, self.votes, self.origin, self.voxel_size = Path(directory), stores, votes, np.asarray(origin), voxel_size
        self.skipped, self.mirror, self.radius, self.checkpoint = skipped, mirror, radius, checkpoint
        self.temporary = self.index = None; self.predictions = {}; self.class_transfer_count = 0
        self.geometry = PlaneIndex(geometry_planes) if geometry_planes else None
        self.geometry_fill_tolerance = geometry_fill_tolerance
        self.geometry_fill_counts = {}

    def __enter__(self):
        if not self.radius: return self
        self.temporary = tempfile.TemporaryDirectory(prefix='e57cls_class_neighbours_', dir=self.directory)
        self.index = ClassPointIndex(Path(self.temporary.name)/'classes.sqlite', self.origin)
        for store in self.stores:
            scan = store.meta['scan_index']
            maps = tuple(np.memmap(Path(self.temporary.name)/f'{scan}_{name}.bin', mode='w+', dtype=np.uint8, shape=store.meta['record_count']) for name in ('codes','confidence','states'))
            self.predictions[scan] = maps
            for start, xyz, valid in store.chunks(500_000):
                self.checkpoint()
                world, codes, confidence, states = classify_points(store, start, start+len(xyz), xyz, valid, self.votes, self.origin, self.voxel_size, self.skipped, self.mirror)
                direct = (codes != 1) & (codes != 68); self.index.add(world[direct], codes[direct])
                for mapping, values in zip(maps, (codes, confidence, states)):
                    view = mapping[start:start+len(xyz)]; view[valid] = values
            for mapping in maps: mapping.flush()
        self.index.db.commit(); return self

    def block(self, store, start, xyz, valid, density, world=None):
        stop = start+len(xyz)
        if self.index is not None and density == 'reduced':
            maps = self.predictions[store.meta['scan_index']]
            world = Pose.from_dict(store.meta['metadata']['pose']).to_world(xyz[valid]) if world is None else world
            codes, confidence, states = (np.array(mapping[start:stop][valid], copy=True) for mapping in maps)
        else:
            world, codes, confidence, states = classify_points(store, start, stop, xyz, valid, self.votes, self.origin, self.voxel_size, self.skipped, self.mirror, world)
        if self.index is not None:
            unknown = codes == 1; inherited = self.index.nearest_codes(world[unknown], self.radius); assigned = inherited != 0
            ids = np.flatnonzero(unknown)[assigned]; codes[ids], confidence[ids] = inherited[assigned], np.maximum(confidence[ids], 128)
            self.class_transfer_count += len(ids)
        if self.geometry is not None:
            unknown = codes == 1
            filled = self.geometry.fill(world[unknown], self.geometry_fill_tolerance)
            assigned = filled != 0
            ids = np.flatnonzero(unknown)[assigned]
            codes[ids], confidence[ids] = filled[assigned], 96
            for code, number in zip(*np.unique(filled[assigned], return_counts=True)):
                self.geometry_fill_counts[str(int(code))] = self.geometry_fill_counts.get(str(int(code)), 0)+int(number)
        return world, codes, confidence, states

    def __exit__(self, *args):
        if self.index: self.index.close()
        if self.geometry: self.geometry.close()
        for maps in self.predictions.values():
            for mapping in maps: mapping._mmap.close()
        if self.temporary: self.temporary.cleanup()
