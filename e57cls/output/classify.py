"""Shared bounded classification and direct-neighbour transfer for writers."""

from collections import OrderedDict
from itertools import product
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import psutil
from scipy.spatial import cKDTree

from e57cls.geometry.cameras import scan_pose
from e57cls.geometry.fill import PlaneIndex
from e57cls.vote.engine import voxel_keys
from e57cls.vote.geometry_rules import TerrainIndex


def classify_points(store, start, stop, xyz, valid, votes, origin, voxel_size,
                    skipped=(), mirror=None, world=None):
    """Classify the valid rows of one source block without transferring labels."""
    skipped_scan = store.meta["scan_index"] in skipped
    if world is None:
        world = scan_pose(store.meta["metadata"]).to_world(xyz[valid])
    if skipped_scan:
        return world, np.ones(len(world), np.uint8), np.zeros(len(world), np.uint8), np.zeros(len(world), np.uint8)
    codes, confidence = votes.lookup(voxel_keys(world, origin, voxel_size))
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
    """Bounded tiled direct-class index with deterministic equal-distance ties.

    Sources close to a tile edge are copied to its neighbouring tiles, so a
    query reads exactly one tile.  A source ordinal is kept with every copy;
    if multiple sources have exactly the same distance, the lowest ordinal
    wins.  This is deliberately independent of cKDTree's unspecified tie
    ordering.
    """
    record_dtype = np.dtype([('xyz', '<f8', (3,)), ('code', 'u1'), ('ordinal', '<i8')])

    class _CompatibilityDatabase:
        """Keep the historic ``.db.commit()`` call harmless for import users."""
        def commit(self):
            return None

    def __init__(self, path, origin, block=2., radius=None, max_tile_points=5_000_000, checkpoint=lambda: None):
        self.origin = np.asarray(origin, np.float64)
        self.block = float(block)
        # Stand-alone users of the historic class did not provide a radius.
        # Keep their small-cell queries correct; production passes its exact
        # transfer radius and therefore duplicates only the needed edge band.
        self.radius = self.block if radius is None else float(radius)
        self.max_tile_points = int(max_tile_points)
        self.directory = Path(str(path)+'.tiles')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = OrderedDict()
        self.db = self._CompatibilityDatabase()
        self._buffers, self._buffer_sizes, self._buffered_bytes, self._files = {}, {}, 0, {}
        self._pending, self._pending_bytes = [], 0
        self._file_spans = {}
        self._children, self._finalized, self._ordinal = {}, False, 0
        # Packed keys are relative to this root tile.  Keeping the grid itself
        # relative to ``origin`` preserves every tile boundary, while avoiding
        # a range limit for normal projected coordinates such as LV95 or UTM.
        self._anchor_root = None
        self._closed = False
        # Buffer enough data to make each append meaningful.  The global
        # ceiling prevents a sparse, large project from retaining every tile.
        self._buffer_limit = 8*1024**2
        available = int(psutil.virtual_memory().available)
        self._global_buffer_limit = min(512*1024**2, max(self._buffer_limit, available//8))
        # Keep a complete input block together.  This makes the normal
        # multi-block path sort each staged source exactly once, rather than
        # materializing the remainder of a previous block during the next
        # ``add`` call.  The limit is also bounded by the existing global
        # buffering ceiling.
        self._stage_limit = self._global_buffer_limit
        self._super_tile_width = max(1, int(round(16/self.block)))
        self._cache_bytes = 0
        self._cache_limit = max(1, int(psutil.virtual_memory().available)//4)
        self.checkpoint = checkpoint

    _key_bits = 21
    _key_offset = 1 << (_key_bits-1)
    _key_mask = (1 << _key_bits)-1
    _axis_offsets = np.asarray((1 << 42, 1 << 21, 1), np.int64)

    def _linear_keys(self, roots):
        """Pack root coordinates relative to the first source tile, if possible."""
        roots = np.asarray(roots, np.int64)
        if self._anchor_root is None:
            self._anchor_root = roots[0].copy()
        encoded = roots-self._anchor_root+self._key_offset
        if np.any((encoded < 0) | (encoded > self._key_mask)):
            return None
        return ((encoded[:, 0] << 42) | (encoded[:, 1] << 21) | encoded[:, 2]).astype(np.int64, copy=False)

    def _root_from_linear_key(self, key):
        key = int(key)
        relative = np.asarray(((key >> 42 & self._key_mask)-self._key_offset,
                               (key >> 21 & self._key_mask)-self._key_offset,
                               (key & self._key_mask)-self._key_offset), np.int64)
        return tuple((relative+self._anchor_root).tolist())

    def _roots_from_linear_keys(self, keys):
        """Decode the root key for every materialized group in one array pass."""
        keys = np.asarray(keys, np.int64)
        relative = np.empty((len(keys), 3), np.int64)
        relative[:, 0] = (keys >> 42 & self._key_mask)-self._key_offset
        relative[:, 1] = (keys >> 21 & self._key_mask)-self._key_offset
        relative[:, 2] = (keys & self._key_mask)-self._key_offset
        return relative+self._anchor_root

    @staticmethod
    def _name(key):
        root, path = key
        stem = '_'.join(map(str, root))
        return f'{stem}'+(f"__{'_'.join(map(str, path))}" if path else '')+'.bin'

    def _path(self, key):
        return self.directory/self._name(key)

    def _super_path(self, root):
        super_root = tuple(int(value//self._super_tile_width) for value in root)
        return self.directory/('super_'+'_'.join(map(str, super_root))+'.bin')

    def _append(self, key, records):
        if not len(records):
            return
        bucket = self._buffers.setdefault(key, [])
        values = np.asarray(records, self.record_dtype)
        bucket.append(values)
        size = int(values.nbytes)
        self._buffer_sizes[key] = self._buffer_sizes.get(key, 0)+size
        self._buffered_bytes += size
        if self._buffer_sizes[key] >= self._buffer_limit:
            self._flush(key)
        while self._buffered_bytes > self._global_buffer_limit:
            largest = max(self._buffer_sizes, key=self._buffer_sizes.__getitem__)
            self._flush(largest)

    def _stage(self, keys, records):
        """Retain packed tile keys until a single bulk grouping is worthwhile."""
        if not len(keys):
            return
        self._pending.append((keys, records))
        self._pending_bytes += int(keys.nbytes+records.nbytes)

    def _stage_roots(self, roots, records):
        """Stage root tiles, using an exact int64-triplet fallback when needed."""
        keys = self._linear_keys(roots)
        if keys is not None:
            self._stage(keys, records)
            return
        # A point more than roughly 2,097 km from the anchor cannot use the
        # compact 21-bit representation.  This path is deliberately slower,
        # but it retains the exact absolute root triples and never rejects it.
        roots = np.asarray(roots, np.int64)
        order = np.lexsort((roots[:, 2], roots[:, 1], roots[:, 0]))
        sorted_roots = roots[order]
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_roots, axis=0), axis=1))+1, len(order)]
        ordered_records = records[order]
        for first, last in zip(starts[:-1], starts[1:]):
            key = (tuple(sorted_roots[first].tolist()), ())
            self._append(key, ordered_records[first:last])

    def _materialize_pending(self):
        """Group staged records with the fastest non-stable integer-key sort."""
        if not self._pending:
            return
        key_parts, record_parts = zip(*self._pending)
        keys = np.concatenate(key_parts) if len(key_parts) > 1 else key_parts[0]
        records = np.concatenate(record_parts) if len(record_parts) > 1 else record_parts[0]
        self._pending, self._pending_bytes = [], 0
        # Equal keys have an explicit ordinal tie-break at lookup time, so
        # preserving their insertion order here is unnecessary and markedly
        # slower for the dense root tiles produced by large scans.
        order = np.argsort(keys)
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])+1, len(order)]
        ordered_records = records[order]
        roots = self._roots_from_linear_keys(sorted_keys[starts[:-1]])
        for root, first, last in zip(roots, starts[:-1], starts[1:]):
            key = (tuple(map(int, root)), ())
            self._append(key, ordered_records[first:last])

    def _flush(self, key):
        parts = self._buffers.pop(key, None)
        if not parts:
            return
        values = np.concatenate(parts) if len(parts) > 1 else parts[0]
        with self._path(key).open('ab') as handle:
            values.tofile(handle)
        self._files[key] = self._path(key)
        self._file_spans.pop(key, None)
        self._buffered_bytes -= self._buffer_sizes.pop(key)

    def _flush_supertiles(self):
        """Write sparse root tiles into a few append-only super-tile files."""
        grouped = {}
        for key in self._buffers:
            if key not in self._files:
                grouped.setdefault(self._super_path(key[0]), []).append(key)
        for path, keys in grouped.items():
            with path.open('ab') as handle:
                start = handle.tell()//self.record_dtype.itemsize
                for key in keys:
                    parts = self._buffers.pop(key)
                    values = np.concatenate(parts) if len(parts) > 1 else parts[0]
                    values.tofile(handle)
                    self._files[key] = path
                    self._file_spans[key] = (start, len(values))
                    start += len(values)
                    self._buffered_bytes -= self._buffer_sizes.pop(key)

    def _values(self, key):
        span = self._file_spans.get(key)
        if span is None:
            return np.fromfile(self._files[key], dtype=self.record_dtype)
        start, count = span
        with self._files[key].open('rb') as handle:
            handle.seek(start*self.record_dtype.itemsize)
            return np.fromfile(handle, dtype=self.record_dtype, count=count)

    def _tile_key(self, point, root):
        key = (tuple(map(int, root)), ())
        while key in self._children:
            level = len(key[1])
            low = self.origin+np.asarray(key[0])*self.block
            size = self.block/(2**level)
            bits = np.clip(np.floor((point-low)/(size/2)).astype(int), 0, 1)
            key = (key[0], key[1]+(int(bits[0])*4+int(bits[1])*2+int(bits[2]),))
        return key

    def add(self, points, codes, ordinals=None):
        """Append direct sources, duplicating only points near a root-tile edge."""
        if self._finalized:
            raise RuntimeError('ClassPointIndex cannot be extended after lookup')
        points, codes = np.asarray(points, np.float64), np.asarray(codes, np.uint8)
        if not len(points):
            return
        if ordinals is None:
            ordinals = np.arange(self._ordinal, self._ordinal+len(points), dtype=np.int64)
        else:
            ordinals = np.asarray(ordinals, np.int64)
        self._ordinal = max(self._ordinal, int(ordinals.max())+1)
        records = np.empty(len(points), self.record_dtype)
        records['xyz'], records['code'], records['ordinal'] = points, codes, ordinals
        roots = np.floor((points-self.origin)/self.block).astype(np.int64)
        if self._anchor_root is None:
            self._anchor_root = roots[0].copy()
        local = points-(self.origin+roots*self.block)
        edge = self.radius if self.radius else 0.
        # A neighbour combination is processed at once.  This keeps the edge
        # copies vectorized and only leaves a loop over the 27 possible tiles.
        axis_masks = ((local[:, axis] <= edge,
                       np.ones(len(points), bool),
                       self.block-local[:, axis] <= edge) for axis in range(3))
        masks = tuple(axis_masks)
        for directions in product(range(3), repeat=3):
            self.checkpoint()
            if directions == (1, 1, 1):
                # Every source belongs to its own root.  Keeping these views
                # avoids a full-size integer index and structured-record copy.
                self._stage_roots(roots, records)
                continue
            include = masks[0][directions[0]] & masks[1][directions[1]] & masks[2][directions[2]]
            ids = np.flatnonzero(include)
            if not len(ids):
                continue
            offset = np.asarray([direction-1 for direction in directions], np.int64)
            self._stage_roots(roots[ids]+offset, records[ids])
        if self._pending_bytes >= self._stage_limit:
            self._materialize_pending()

    def _split(self, key):
        values = self._values(key)
        if len(values) <= self.max_tile_points:
            return False
        level, root = len(key[1]), np.asarray(key[0], np.int64)
        low = self.origin+root*self.block
        size = self.block/(2**level)
        local = values['xyz']-low
        base = np.clip(np.floor(local/(size/2)).astype(np.int8), 0, 1)
        child_ids, child_keys = [], set()
        for shift in product((-1, 0, 1), repeat=3):
            self.checkpoint()
            include = np.ones(len(values), bool)
            for axis, direction in enumerate(shift):
                if direction < 0:
                    include &= base[:, axis] == 1
                    include &= np.abs(local[:, axis]-size/2) <= self.radius
                elif direction > 0:
                    include &= base[:, axis] == 0
                    include &= np.abs(local[:, axis]-size/2) <= self.radius
            if not include.any():
                continue
            child = base[include]+np.asarray(shift, np.int8)
            valid = np.all((child >= 0) & (child <= 1), axis=1)
            ids = np.flatnonzero(include)[valid]
            child = child[valid]
            for number in range(8):
                take = ids[(child[:, 0]*4+child[:, 1]*2+child[:, 2]) == number]
                if len(take):
                    child_key = (key[0], key[1]+(number,))
                    self._append(child_key, values[take])
                    child_ids.append(number)
                    child_keys.add(child_key)
        for child_key in tuple(self._buffers):
            self._flush(child_key)
        if len(set(child_ids)) <= 1:
            for child_key in child_keys:
                self._files.pop(child_key, self._path(child_key)).unlink(missing_ok=True)
            return False
        self._children[key] = True
        path = self._files[key]
        if key not in self._file_spans:
            path.unlink(missing_ok=True)
        self._file_spans.pop(key, None)
        del self._files[key]
        return True

    def _finalize(self):
        if self._finalized:
            return
        self._materialize_pending()
        self._flush_supertiles()
        for key in tuple(self._buffers):
            self._flush(key)
        pending = list(self._files)
        while pending:
            self.checkpoint()
            key = pending.pop()
            if key not in self._files:
                continue
            if self._split(key):
                pending.extend(child for child in self._files if child[0] == key[0] and len(child[1]) == len(key[1])+1)
        self._finalized = True

    def _tree(self, key):
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        values = self._values(key)
        cached = (cKDTree(values['xyz']), values['code'].copy(), values['ordinal'].copy(),
                  int(values['xyz'].nbytes+values['code'].nbytes+values['ordinal'].nbytes))
        self.cache[key] = cached
        self._cache_bytes += cached[3]
        while len(self.cache) > 1 and self._cache_bytes > self._cache_limit:
            _, old = self.cache.popitem(last=False)
            self._cache_bytes -= old[3]
        return cached

    def nearest_codes(self, points, radius, return_distances=False):
        self._finalize()
        points = np.asarray(points, np.float64)
        result = np.zeros(len(points), np.uint8)
        selected_distances = np.full(len(points), np.inf, np.float64)
        if not len(points):
            return (result, selected_distances) if return_distances else result
        roots = np.floor((points-self.origin)/self.block).astype(np.int64)
        tags = [self._tile_key(point, root) for point, root in zip(points, roots)]
        order = np.argsort(np.asarray([repr(tag) for tag in tags]), kind='stable')
        starts = np.r_[0, np.flatnonzero(np.asarray([tags[order[index]] != tags[order[index-1]] for index in range(1, len(order))]))+1, len(order)]
        for first, last in zip(starts[:-1], starts[1:]):
            ids, key = order[first:last], tags[order[first]]
            if key not in self._files:
                continue
            tree, labels, ordinals, _ = self._tree(key)
            count = min(2, len(labels))
            distances, local = tree.query(points[ids], k=count, distance_upper_bound=radius, workers=-1)
            if count == 1:
                distances, local = distances[:, None], local[:, None]
            valid = np.isfinite(distances[:, 0])
            if valid.any():
                result[ids[valid]] = labels[local[valid, 0]]
                selected_distances[ids[valid]] = distances[valid, 0]
            if count > 1:
                tied = valid & np.isfinite(distances[:, 1]) & np.isclose(distances[:, 0], distances[:, 1], rtol=0, atol=1e-12)
                for output_id, distance in zip(ids[tied], distances[tied, 0]):
                    candidates = np.asarray(tree.query_ball_point(points[output_id], distance+1e-12))
                    if len(candidates):
                        squared = ((tree.data[candidates]-points[output_id])**2).sum(axis=1)
                        exact = candidates[np.isclose(np.sqrt(squared), distance, rtol=0, atol=1e-12)]
                        chosen = exact[np.argmin(ordinals[exact])]
                        result[output_id] = labels[chosen]
        return (result, selected_distances) if return_distances else result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cache.clear()
        self._buffers.clear()
        self._buffer_sizes.clear()
        self._buffered_bytes = 0
        self._pending.clear()
        self._pending_bytes = 0
        self._files.clear()
        self._file_spans.clear()
        shutil.rmtree(self.directory, ignore_errors=True)


class ClassifiedSource:
    """Reuse reduced direct predictions and transfer only from those predictions."""
    def __init__(self, directory, stores, votes, origin, voxel_size, skipped=(), mirror=None, radius=0., checkpoint=lambda: None,
                 geometry_planes=None, geometry_fill_tolerance=.02, terrain=None, terrain_ground_tolerance=.15,
                 terrain_fill=False, progress=lambda *args: None, force_unclassified=None, terrain_outdoor_scans=None,
                 class_transfer_range_factor=0., class_transfer_radius_max=None, transfer_station_positions=None):
        self.directory, self.stores, self.votes, self.origin, self.voxel_size = Path(directory), stores, votes, np.asarray(origin), voxel_size
        self.skipped, self.mirror, self.radius, self.checkpoint = skipped, mirror, radius, checkpoint
        self.radius_max = float(radius if class_transfer_radius_max is None else class_transfer_radius_max)
        self.range_factor = float(class_transfer_range_factor)
        self.transfer_station_positions = {int(key): np.asarray(value, np.float64)
                                           for key, value in (transfer_station_positions or {}).items()}
        self.adaptive_transfer = bool(self.radius_max > self.radius and self.range_factor > 0.)
        self.temporary = self.index = None; self.predictions = {}; self.class_transfer_count = 0
        self.geometry = PlaneIndex(geometry_planes) if geometry_planes else None
        self.terrain = TerrainIndex(terrain) if terrain else None
        self.geometry_fill_tolerance = geometry_fill_tolerance
        self.terrain_ground_tolerance = terrain_ground_tolerance
        self.terrain_fill = terrain_fill
        self.terrain_outdoor_scans = None if terrain_outdoor_scans is None else set(terrain_outdoor_scans)
        self.geometry_fill_counts = {}
        self.progress = progress
        self.force_unclassified = force_unclassified
        self.timing_seconds = {'lookup_mirror': 0., 'transfer': 0., 'plane_fill': 0., 'terrain_fill': 0.}

    def __enter__(self):
        if not self.radius_max: return self
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix='e57cls_class_neighbours_', dir=self.directory)
            self.index = ClassPointIndex(Path(self.temporary.name)/'classes', self.origin, radius=self.radius_max, checkpoint=self.checkpoint)
            total = max(1, sum(selected_count(store, 'reduced') for store in self.stores))
            prepared = 0
            for store in self.stores:
                scan = store.meta['scan_index']
                maps = tuple(np.memmap(Path(self.temporary.name)/f'{scan}_{name}.bin', mode='w+', dtype=np.uint8, shape=store.meta['record_count']) for name in ('codes','confidence','states'))
                self.predictions[scan] = maps
                for start, xyz, valid in store.chunks(500_000):
                    self.checkpoint()
                    started = time.perf_counter()
                    world, codes, confidence, states = classify_points(store, start, start+len(xyz), xyz, valid, self.votes, self.origin, self.voxel_size, self.skipped, self.mirror)
                    self.timing_seconds['lookup_mirror'] += time.perf_counter()-started
                    direct = (codes != 1) & (codes != 68); self.index.add(world[direct], codes[direct])
                    for mapping, values in zip(maps, (codes, confidence, states)):
                        view = mapping[start:start+len(xyz)]; view[valid] = values
                    prepared += int(valid.sum())
                    self.progress('output_prepare', prepared/total, scan)
                for mapping in maps: mapping.flush()
            return self
        except Exception:
            self.close()
            raise

    def _transfer_radii(self, store, world):
        """Return per-point transfer radii from the owning scanner or station."""
        if not self.adaptive_transfer:
            return np.full(len(world), self.radius, np.float64)
        distances = np.full(len(world), np.nan, np.float64)
        if self.transfer_station_positions:
            stations = self.votes.lookup_station_ids(voxel_keys(world, self.origin, self.voxel_size))
            for station in np.unique(stations):
                positions = self.transfer_station_positions.get(int(station))
                if positions is None or not len(positions):
                    continue
                ids = np.flatnonzero(stations == station)
                distances[ids] = np.sqrt(((world[ids, None, :]-positions[None, :, :])**2).sum(axis=2).min(axis=1))
        else:
            position = scan_pose(store.meta['metadata']).translation
            distances = np.linalg.norm(world-position, axis=1)
        radii = np.full(len(world), self.radius, np.float64)
        known = np.isfinite(distances)
        radii[known] = np.clip(self.range_factor*distances[known], self.radius, self.radius_max)
        return radii

    def block(self, store, start, xyz, valid, density, world=None):
        stop = start+len(xyz)
        if store.meta['scan_index'] in self.skipped:
            world, codes, confidence, states = classify_points(store, start, stop, xyz, valid, self.votes, self.origin,
                                                                 self.voxel_size, self.skipped, self.mirror, world)
        elif self.index is not None and density == 'reduced':
            maps = self.predictions[store.meta['scan_index']]
            world = scan_pose(store.meta['metadata']).to_world(xyz[valid]) if world is None else world
            codes, confidence, states = (np.array(mapping[start:stop][valid], copy=True) for mapping in maps)
        else:
            started = time.perf_counter()
            world, codes, confidence, states = classify_points(store, start, stop, xyz, valid, self.votes, self.origin, self.voxel_size, self.skipped, self.mirror, world)
            self.timing_seconds['lookup_mirror'] += time.perf_counter()-started
        if self.index is not None and store.meta['scan_index'] not in self.skipped:
            started = time.perf_counter()
            unknown = codes == 1
            query_points = world[unknown]
            if self.adaptive_transfer:
                inherited, distances = self.index.nearest_codes(query_points, self.radius_max, return_distances=True)
                assigned = (inherited != 0) & (distances <= self._transfer_radii(store, query_points))
            else:
                inherited = self.index.nearest_codes(query_points, self.radius)
                assigned = inherited != 0
            ids = np.flatnonzero(unknown)[assigned]; codes[ids], confidence[ids] = inherited[assigned], np.maximum(confidence[ids], 128)
            self.class_transfer_count += len(ids)
            self.timing_seconds['transfer'] += time.perf_counter()-started
        if self.geometry is not None and store.meta['scan_index'] not in self.skipped:
            started = time.perf_counter()
            unknown = codes == 1
            filled = self.geometry.fill(world[unknown], self.geometry_fill_tolerance, self.checkpoint)
            assigned = filled != 0
            ids = np.flatnonzero(unknown)[assigned]
            codes[ids], confidence[ids] = filled[assigned], 96
            for code, number in zip(*np.unique(filled[assigned], return_counts=True)):
                self.geometry_fill_counts[str(int(code))] = self.geometry_fill_counts.get(str(int(code)), 0)+int(number)
            self.timing_seconds['plane_fill'] += time.perf_counter()-started
        if store.meta['scan_index'] not in self.skipped and self.terrain is not None and self.terrain_fill and (self.terrain_outdoor_scans is None
                                                               or store.meta['scan_index'] in self.terrain_outdoor_scans):
            started = time.perf_counter()
            unknown = codes == 1
            filled = self.terrain.ground_mask(world[unknown], self.terrain_ground_tolerance)
            ids = np.flatnonzero(unknown)[filled]
            codes[ids], confidence[ids] = 2, 64
            if len(ids):
                self.geometry_fill_counts['terrain_fill'] = self.geometry_fill_counts.get('terrain_fill', 0)+len(ids)
            self.timing_seconds['terrain_fill'] += time.perf_counter()-started
        if self.force_unclassified is not None:
            unseen = self.force_unclassified(store, start, valid)
            codes[unseen], confidence[unseen], states[unseen] = 1, 0, 0
        return world, codes, confidence, states

    def close(self):
        """Release all output-preparation resources; safe to call repeatedly."""
        index, self.index = self.index, None
        if index is not None:
            try:
                index.close()
            except Exception:
                pass
        geometry, self.geometry = self.geometry, None
        if geometry is not None:
            try:
                geometry.close()
            except Exception:
                pass
        terrain, self.terrain = self.terrain, None
        if terrain is not None:
            try:
                terrain.close()
            except Exception:
                pass
        predictions, self.predictions = self.predictions, {}
        for maps in predictions.values():
            for mapping in maps:
                try:
                    mapping.flush()
                except Exception:
                    pass
                try:
                    mapping._mmap.close()
                except Exception:
                    pass
        temporary, self.temporary = self.temporary, None
        if temporary is not None:
            temporary_path = Path(temporary.name)
            try:
                temporary.cleanup()
            except Exception:
                pass
            shutil.rmtree(temporary_path, ignore_errors=True)

    def __exit__(self, *args):
        self.close()
        return False
