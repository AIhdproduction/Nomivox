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
from e57cls.vote.dtm import TinDTM
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
    keys = voxel_keys(world, origin, voxel_size)
    # VoteStore's snapshot lookup is a pure key-to-value mapping.  TLS blocks
    # commonly contain many points in every voxel, so look up each key once
    # and scatter the result back to the original point order.  Third-party
    # lookup implementations retain their historic call/order semantics.
    if getattr(votes, 'lookup_is_key_pure', False) and len(keys) > 1:
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        unique_codes, unique_confidence = votes.lookup(unique_keys)
        codes, confidence = unique_codes[inverse], unique_confidence[inverse]
    else:
        codes, confidence = votes.lookup(keys)
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

    _tag_dtype = np.dtype([('root', '<i8'), ('path', '<u8')])
    _tree_node_bytes = 64
    _parallel_query_minimum = 20_000

    def __init__(self, path, origin, block=2., radius=None, max_tile_points=1_000_000, checkpoint=lambda: None,
                 cache_gb=4.0):
        self.origin = np.asarray(origin, np.float64)
        self.block = float(block)
        # Stand-alone users of the historic class did not provide a radius.
        # Keep their small-cell queries correct; production passes its exact
        # transfer radius and therefore duplicates only the needed edge band.
        self.radius = self.block if radius is None else float(radius)
        if not isinstance(max_tile_points, (int, np.integer)) or isinstance(max_tile_points, (bool, np.bool_)) or max_tile_points < 1:
            raise ValueError('max_tile_points must be a positive integer')
        if (not isinstance(cache_gb, (int, float, np.integer, np.floating)) or isinstance(cache_gb, (bool, np.bool_))
                or not np.isfinite(cache_gb) or cache_gb <= 0):
            raise ValueError('cache_gb must be greater than zero')
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
        self._cache_limit = max(1, int(float(cache_gb)*1024**3))
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._leaf_keys = {}
        self._numeric_leaf_keys = {}
        self._child_tags = np.empty(0, self._tag_dtype)
        self._children_by_root = {}
        self._max_tree_depth = 0
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

    @staticmethod
    def _path_code(path):
        """Encode a child path with a leading marker so prefixes stay distinct."""
        code = 1
        for child in path:
            code = (code << 3) | int(child)
        return code

    def _prepare_tile_lookup(self):
        """Build numeric child and leaf lookups after the tile tree is final."""
        self._leaf_keys = {}
        self._numeric_leaf_keys = {}
        self._children_by_root = {}
        numeric_children = []
        self._max_tree_depth = 0

        for key in self._files:
            root, path = key
            code = self._path_code(path)
            self._leaf_keys[(root, code)] = key
            packed = self._linear_keys(np.asarray(root, np.int64)[None, :])
            if packed is not None:
                self._numeric_leaf_keys[(int(packed[0]), code)] = key

        for key in self._children:
            root, path = key
            code = self._path_code(path)
            self._max_tree_depth = max(self._max_tree_depth, len(path)+1)
            self._children_by_root.setdefault(root, []).append(code)
            packed = self._linear_keys(np.asarray(root, np.int64)[None, :])
            if packed is not None:
                numeric_children.append((int(packed[0]), code))

        self._children_by_root = {root: np.sort(np.asarray(paths, np.uint64))
                                  for root, paths in self._children_by_root.items()}
        if numeric_children and self._max_tree_depth <= 20:
            self._child_tags = np.asarray(numeric_children, dtype=self._tag_dtype)
            self._child_tags.sort(order=('root', 'path'))
        else:
            self._child_tags = np.empty(0, self._tag_dtype)

    @staticmethod
    def _contains_sorted(values, probes):
        """Return membership of numeric probes in one sorted numeric array."""
        locations = np.searchsorted(values, probes)
        result = locations < len(values)
        result[result] &= values[locations[result]] == probes[result]
        return result

    def _numeric_child_mask(self, roots, paths):
        if not len(self._child_tags):
            return np.zeros(len(roots), bool)
        tags = np.empty(len(roots), self._tag_dtype)
        tags['root'] = roots
        tags['path'] = paths
        locations = np.searchsorted(self._child_tags, tags)
        result = locations < len(self._child_tags)
        result[result] &= self._child_tags[locations[result]] == tags[result]
        return result

    def _descend_paths(self, local, child_paths):
        """Follow one root's child tree for a whole point group in NumPy."""
        paths = np.ones(len(local), np.uint64)
        if child_paths is None or not len(child_paths):
            return paths
        levels = np.zeros(len(local), np.uint8)
        for _ in range(self._max_tree_depth):
            descend = self._contains_sorted(child_paths, paths)
            if not descend.any():
                break
            ids = np.flatnonzero(descend)
            size = self.block/np.exp2(levels[ids])
            bits = np.clip(np.floor(local[ids]/(size[:, None]/2)), 0, 1).astype(np.uint8)
            child = (bits[:, 0]*4+bits[:, 1]*2+bits[:, 2]).astype(np.uint64)
            paths[ids] = (paths[ids] << np.uint64(3)) | child
            levels[ids] += 1
        return paths

    def _tile_paths(self, points, roots, packed_roots):
        """Determine every leaf path with vector operations, not point callbacks."""
        if not self._children:
            return np.ones(len(points), np.uint64)
        root_low = self.origin+roots*self.block
        local = points-root_low
        if packed_roots is not None and len(self._child_tags):
            paths = np.ones(len(points), np.uint64)
            levels = np.zeros(len(points), np.uint8)
            for _ in range(self._max_tree_depth):
                descend = self._numeric_child_mask(packed_roots, paths)
                if not descend.any():
                    break
                ids = np.flatnonzero(descend)
                size = self.block/np.exp2(levels[ids])
                bits = np.clip(np.floor(local[ids]/(size[:, None]/2)), 0, 1).astype(np.uint8)
                child = (bits[:, 0]*4+bits[:, 1]*2+bits[:, 2]).astype(np.uint64)
                paths[ids] = (paths[ids] << np.uint64(3)) | child
                levels[ids] += 1
            return paths

        paths = np.ones(len(points), np.uint64)
        order = np.lexsort((roots[:, 2], roots[:, 1], roots[:, 0]))
        sorted_roots = roots[order]
        starts = np.r_[0, np.flatnonzero(np.any(sorted_roots[1:] != sorted_roots[:-1], axis=1))+1, len(order)]
        for first, last in zip(starts[:-1], starts[1:]):
            ids = order[first:last]
            root = tuple(map(int, sorted_roots[first]))
            paths[ids] = self._descend_paths(local[ids], self._children_by_root.get(root))
        return paths

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
        self._prepare_tile_lookup()
        self._finalized = True

    def cache_stats(self):
        return {'index_cache_hits': int(self._cache_hits), 'index_cache_misses': int(self._cache_misses),
                'index_cache_evictions': int(self._cache_evictions), 'index_cache_bytes': int(self._cache_bytes),
                'index_cache_limit_bytes': int(self._cache_limit), 'index_tile_max_depth': int(self._max_tree_depth)}

    def _tree(self, key):
        cached = self.cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self.cache.move_to_end(key)
            return cached
        self._cache_misses += 1
        values = self._values(key)
        xyz = np.ascontiguousarray(values['xyz'])
        tree = cKDTree(xyz, copy_data=False)
        labels, ordinals = values['code'].copy(), values['ordinal'].copy()
        cost = int(tree.data.nbytes+tree.indices.nbytes+labels.nbytes+ordinals.nbytes+
                   tree.size*self._tree_node_bytes)
        cached = (tree, labels, ordinals, cost)
        # A tile larger than the whole budget is still queryable, but retaining
        # it would violate the configured cache limit after this call returns.
        if cost > self._cache_limit:
            return cached
        self.cache[key] = cached
        self._cache_bytes += cost
        while self._cache_bytes > self._cache_limit:
            _, old = self.cache.popitem(last=False)
            self._cache_bytes -= old[3]
            self._cache_evictions += 1
        return cached

    def _tie_winners(self, tree, points, distances, ordinals):
        """Return deterministic lowest-ordinal winners for true equal-distance ties."""
        tied = (np.isfinite(distances[:, 0]) & np.isfinite(distances[:, 1]) &
                np.isclose(distances[:, 0], distances[:, 1], rtol=0, atol=1e-12))
        tie_ids = np.flatnonzero(tied)
        if not len(tie_ids):
            return tie_ids, np.empty(0, np.intp)
        workers = -1 if len(tie_ids) >= self._parallel_query_minimum else 1
        candidates = tree.query_ball_point(points[tie_ids], distances[tie_ids, 0]+1e-12, workers=workers)
        lengths = np.fromiter((len(item) for item in candidates), dtype=np.intp, count=len(candidates))
        if not lengths.any():
            return tie_ids[:0], np.empty(0, np.intp)
        flat = np.concatenate([np.asarray(item, np.intp) for item in candidates if len(item)])
        owners = np.repeat(np.arange(len(tie_ids), dtype=np.intp), lengths)
        deltas = tree.data[flat]-points[tie_ids[owners]]
        exact = np.isclose(np.sqrt(np.einsum('ij,ij->i', deltas, deltas)), distances[tie_ids[owners], 0],
                           rtol=0, atol=1e-12)
        if not exact.any():
            return tie_ids[:0], np.empty(0, np.intp)
        owners, flat = owners[exact], flat[exact]
        order = np.lexsort((ordinals[flat], owners))
        sorted_owners = owners[order]
        first = np.r_[True, sorted_owners[1:] != sorted_owners[:-1]]
        winners = order[first]
        return tie_ids[owners[winners]], flat[winners]

    def _query_group(self, points, ids, key, radius, result, selected_distances):
        if key is None or key not in self._files:
            return
        tree, labels, ordinals, _ = self._tree(key)
        count = min(2, len(labels))
        workers = -1 if len(ids) >= self._parallel_query_minimum else 1
        distances, local = tree.query(points[ids], k=count, distance_upper_bound=radius, workers=workers)
        if count == 1:
            distances, local = distances[:, None], local[:, None]
        valid = np.isfinite(distances[:, 0])
        if valid.any():
            result[ids[valid]] = labels[local[valid, 0]]
            selected_distances[ids[valid]] = distances[valid, 0]
        if count > 1:
            tied, chosen = self._tie_winners(tree, points[ids], distances, ordinals)
            if len(tied):
                result[ids[tied]] = labels[chosen]

    def nearest_codes(self, points, radius, return_distances=False):
        self._finalize()
        points = np.asarray(points, np.float64)
        result = np.zeros(len(points), np.uint8)
        selected_distances = np.full(len(points), np.inf, np.float64)
        if not len(points):
            return (result, selected_distances) if return_distances else result
        roots = np.floor((points-self.origin)/self.block).astype(np.int64)
        packed_roots = self._linear_keys(roots)
        paths = self._tile_paths(points, roots, packed_roots)
        if packed_roots is not None:
            order = np.lexsort((paths, packed_roots))
            sorted_roots, sorted_paths = packed_roots[order], paths[order]
            starts = np.r_[0, np.flatnonzero((sorted_roots[1:] != sorted_roots[:-1]) |
                                              (sorted_paths[1:] != sorted_paths[:-1]))+1, len(order)]
            for first, last in zip(starts[:-1], starts[1:]):
                ids = order[first:last]
                key = self._numeric_leaf_keys.get((int(sorted_roots[first]), int(sorted_paths[first])))
                self._query_group(points, ids, key, radius, result, selected_distances)
        else:
            order = np.lexsort((paths, roots[:, 2], roots[:, 1], roots[:, 0]))
            sorted_roots, sorted_paths = roots[order], paths[order]
            changed = ((sorted_paths[1:] != sorted_paths[:-1]) |
                       np.any(sorted_roots[1:] != sorted_roots[:-1], axis=1))
            starts = np.r_[0, np.flatnonzero(changed)+1, len(order)]
            for first, last in zip(starts[:-1], starts[1:]):
                ids = order[first:last]
                root = tuple(map(int, sorted_roots[first]))
                key = self._leaf_keys.get((root, int(sorted_paths[first])))
                self._query_group(points, ids, key, radius, result, selected_distances)
        return (result, selected_distances) if return_distances else result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cache.clear()
        self._cache_bytes = 0
        self._leaf_keys.clear()
        self._numeric_leaf_keys.clear()
        self._children_by_root.clear()
        self._child_tags = np.empty(0, self._tag_dtype)
        self._buffers.clear()
        self._buffer_sizes.clear()
        self._buffered_bytes = 0
        self._pending.clear()
        self._pending_bytes = 0
        self._files.clear()
        self._file_spans.clear()
        shutil.rmtree(self.directory, ignore_errors=True)


class TreeCrownIndex:
    """Compact lollipop crown volumes written by the shape-rule pass."""

    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            self.x = np.asarray(data['x'], np.float64)
            self.y = np.asarray(data['y'], np.float64)
            self.radius = np.asarray(data['radius'], np.float64)
            self.z_min = np.asarray(data['z_min'], np.float64)
            self.z_max = np.asarray(data['z_max'], np.float64)
        if not (len(self.x) == len(self.y) == len(self.radius) == len(self.z_min) == len(self.z_max)):
            raise ValueError('tree crown artifact arrays must have equal length')
        valid = (self.radius > 0.) & (self.z_max > self.z_min)
        self.x, self.y = self.x[valid], self.y[valid]
        self.radius, self.z_min, self.z_max = self.radius[valid], self.z_min[valid], self.z_max[valid]
        self._max_radius = float(self.radius.max()) if len(self.radius) else 0.
        self._tree = cKDTree(np.column_stack((self.x, self.y))) if len(self.radius) else None

    def mask(self, points):
        points = np.asarray(points, np.float64)
        result = np.zeros(len(points), bool)
        if self._tree is None or not len(points):
            return result
        for start in range(0, len(points), 100_000):
            stop = min(len(points), start+100_000)
            block = points[start:stop]
            candidates = self._tree.query_ball_point(block[:, :2], self._max_radius, workers=1)
            lengths = np.fromiter((len(item) for item in candidates), dtype=np.intp, count=len(candidates))
            if not lengths.any():
                continue
            crowns = np.concatenate([np.asarray(item, np.intp) for item in candidates if len(item)])
            rows = np.repeat(np.arange(len(block), dtype=np.intp), lengths)
            delta = block[rows, :2]-np.column_stack((self.x[crowns], self.y[crowns]))
            inside = ((block[rows, 2] >= self.z_min[crowns]) & (block[rows, 2] <= self.z_max[crowns]) &
                      (np.einsum('ij,ij->i', delta, delta) <= self.radius[crowns]**2))
            result[start+rows[inside]] = True
        return result


class GroundDtmIndex:
    """Read-only TIN DTM and ground alternative class lookup used during export."""

    def __init__(self, dtm_path, alternatives_path=None, band=0.03, origin=None, voxel_size=0.02):
        self.dtm = TinDTM(dtm_path) if dtm_path and Path(dtm_path).is_file() else None
        self.band = float(band)
        self.origin = np.asarray(origin, dtype=np.float64) if origin is not None else None
        self.voxel_size = float(voxel_size)

        if alternatives_path is None and dtm_path is not None:
            cand = Path(dtm_path).parent / 'ground_alternatives.npz'
            if cand.is_file():
                alternatives_path = cand

        self._alt_archive = None
        self.keys = np.empty(0, dtype='V24')
        self.codes = np.empty(0, dtype=np.uint16)
        self.outline_applied = False
        self.outline_inside_barrier = None
        self.outline_origin = None
        self.outline_cells = 1
        self.outline_shape = None

        if alternatives_path and Path(alternatives_path).is_file():
            self._alt_archive = np.load(alternatives_path, allow_pickle=False)
            if 'keys' in self._alt_archive:
                self.keys = np.asarray(self._alt_archive['keys'], dtype='V24')
            if 'codes' in self._alt_archive:
                self.codes = np.asarray(self._alt_archive['codes'], dtype=np.uint16)
            if 'outline_applied' in self._alt_archive:
                self.outline_applied = bool(self._alt_archive['outline_applied'])
            if self.outline_applied and 'outline_inside_barrier' in self._alt_archive:
                self.outline_inside_barrier = np.asarray(self._alt_archive['outline_inside_barrier'], dtype=bool)
                self.outline_origin = np.asarray(self._alt_archive['outline_origin'], dtype=np.int64)
                self.outline_cells = int(self._alt_archive['outline_cells'])
                self.outline_shape = tuple(np.asarray(self._alt_archive['outline_shape'], dtype=np.int64))
            if 'origin' in self._alt_archive and self.origin is None:
                self.origin = np.asarray(self._alt_archive['origin'], dtype=np.float64)
            if 'voxel_size' in self._alt_archive and not self.voxel_size:
                self.voxel_size = float(self._alt_archive['voxel_size'])

    def close(self):
        if self.dtm is not None:
            self.dtm.close()
            self.dtm = None
        if self._alt_archive is not None:
            self._alt_archive.close()
            self._alt_archive = None


class ClassifiedSource:
    """Reuse reduced direct predictions and transfer only from those predictions."""
    _transfer_record_dtype = np.dtype([('xyz', '<f8', (3,)), ('row', '<i8'), ('scan', '<i4')])
    _transfer_chunk_records = 250_000
    _transfer_buffer_bytes_limit = 224*1024**2
    def __init__(self, directory, stores, votes, origin, voxel_size, skipped=(), mirror=None, radius=0., checkpoint=lambda: None,
                  geometry_planes=None, geometry_fill_tolerance=.02, terrain=None, terrain_ground_tolerance=.15,
                  terrain_fill=False, progress=lambda *args: None, force_unclassified=None, terrain_outdoor_scans=None,
                  class_transfer_range_factor=0., class_transfer_radius_max=None, transfer_station_positions=None,
                  tree_crowns=None, tree_fill=False, ground_dtm=None, ground_dtm_alternatives=None, ground_dtm_band=.03,
                  export_index_cache_gb=4.0, export_index_max_tile_points=1_000_000):
        self.directory, self.stores, self.votes, self.origin, self.voxel_size = Path(directory), stores, votes, np.asarray(origin), voxel_size
        self.skipped, self.mirror, self.radius, self.checkpoint = skipped, mirror, radius, checkpoint
        self.radius_max = float(radius if class_transfer_radius_max is None else class_transfer_radius_max)
        self.range_factor = float(class_transfer_range_factor)
        self.transfer_station_positions = {int(key): np.asarray(value, np.float64)
                                           for key, value in (transfer_station_positions or {}).items()}
        self.adaptive_transfer = bool(self.radius_max > self.radius and self.range_factor > 0.)
        self.temporary = self.index = None; self.predictions = {}; self.class_transfer_count = 0
        self._stores_by_scan = {int(store.meta['scan_index']): store for store in stores}
        self._transfer_prepared = False
        self._transfer_query_count = 0
        self._transfer_buffers = {}
        self._transfer_buffer_bytes = 0
        self.transfer_bucket_stats = {'transfer_bucket_files': 0, 'transfer_bucket_open_calls': 0}
        self.export_index_cache_gb = float(export_index_cache_gb)
        self.export_index_max_tile_points = export_index_max_tile_points
        self.index_cache_stats = {'index_cache_hits': 0, 'index_cache_misses': 0, 'index_cache_evictions': 0,
                                  'index_cache_bytes': 0, 'index_cache_limit_bytes': 0, 'index_tile_max_depth': 0}
        self.geometry = PlaneIndex(geometry_planes) if geometry_planes else None
        self.terrain = TerrainIndex(terrain) if terrain else None
        self.geometry_fill_tolerance = geometry_fill_tolerance
        self.terrain_ground_tolerance = terrain_ground_tolerance
        self.terrain_fill = terrain_fill
        self.terrain_outdoor_scans = None if terrain_outdoor_scans is None else set(terrain_outdoor_scans)
        self.tree_crowns = TreeCrownIndex(tree_crowns) if tree_crowns and Path(tree_crowns).is_file() else None
        self.tree_fill = bool(tree_fill)
        if ground_dtm is not None:
            if isinstance(ground_dtm, GroundDtmIndex):
                self.ground_dtm = ground_dtm
            elif isinstance(ground_dtm, (list, tuple)) and len(ground_dtm) >= 2:
                self.ground_dtm = GroundDtmIndex(ground_dtm[0], ground_dtm[1], ground_dtm_band, self.origin, self.voxel_size)
            elif isinstance(ground_dtm, dict):
                dtm_p = ground_dtm.get('dtm') or ground_dtm.get('artifact') or ground_dtm.get('path')
                alt_p = ground_dtm.get('alternatives') or ground_dtm.get('ground_alternatives')
                b = float(ground_dtm.get('band', ground_dtm_band))
                self.ground_dtm = GroundDtmIndex(dtm_p, alt_p, b, self.origin, self.voxel_size)
            else:
                self.ground_dtm = GroundDtmIndex(ground_dtm, ground_dtm_alternatives, ground_dtm_band, self.origin, self.voxel_size)
        else:
            self.ground_dtm = None
        self.geometry_fill_counts = {}
        self.progress = progress
        self.force_unclassified = force_unclassified
        self.timing_seconds = {'lookup_mirror': 0., 'transfer': 0., 'plane_fill': 0., 'terrain_fill': 0., 'tree_fill': 0., 'ground_dtm': 0.}

    def __enter__(self):
        if not self.radius_max: return self
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix='e57cls_class_neighbours_', dir=self.directory)
            self.index = ClassPointIndex(Path(self.temporary.name)/'classes', self.origin, radius=self.radius_max,
                                         checkpoint=self.checkpoint, cache_gb=self.export_index_cache_gb,
                                         max_tile_points=self.export_index_max_tile_points)
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
            self._prepare_reduced_transfer(total)
            return self
        except Exception:
            self.close()
            raise

    def _transfer_bucket_path(self, root, path):
        """Return the on-disk bucket for one finalized index leaf."""
        root = tuple(int(value) for value in root)
        return Path(self.temporary.name)/f'queries_{root[0]}_{root[1]}_{root[2]}_{int(path)}.bin'

    def _flush_transfer_buffers(self):
        """Append buffered records leaf-wise, largest leaves first."""
        if not self._transfer_buffers:
            return
        buffers, self._transfer_buffers = self._transfer_buffers, {}
        self._transfer_buffer_bytes = 0
        for path, parts in sorted(buffers.items(), key=lambda item: -sum(part.nbytes for part in item[1])):
            with path.open('ab') as handle:
                self.transfer_bucket_stats['transfer_bucket_open_calls'] += 1
                for part in parts:
                    part.tofile(handle)

    def _buffer_transfer_records(self, records, roots, paths):
        """Keep leaf runs in RAM until the bounded append buffer is full."""
        if self._transfer_buffer_bytes and (self._transfer_buffer_bytes+records.nbytes > self._transfer_buffer_bytes_limit):
            self._flush_transfer_buffers()
        order = np.lexsort((paths, roots[:, 2], roots[:, 1], roots[:, 0]))
        sorted_roots, sorted_paths = roots[order], paths[order]
        changes = ((sorted_paths[1:] != sorted_paths[:-1]) |
                   np.any(sorted_roots[1:] != sorted_roots[:-1], axis=1))
        starts = np.r_[0, np.flatnonzero(changes)+1, len(order)]
        ordered = records[order]
        for first, last in zip(starts[:-1], starts[1:]):
            path = self._transfer_bucket_path(sorted_roots[first], sorted_paths[first])
            self._transfer_buffers.setdefault(path, []).append(ordered[first:last])
        self._transfer_buffer_bytes += records.nbytes

    def _stage_transfer_buckets(self, total):
        """Write unknown reduced predictions into bounded, spatially local buckets."""
        staged = 0
        for store in self.stores:
            scan = int(store.meta['scan_index'])
            if scan in self.skipped:
                continue
            codes = self.predictions[scan][0]
            pose = scan_pose(store.meta['metadata'])
            for start, xyz, valid in store.chunks(500_000):
                self.checkpoint()
                stop = start+len(xyz)
                valid_rows = np.flatnonzero(valid)
                unknown = np.asarray(codes[start:stop][valid], np.uint8) == 1
                if unknown.any():
                    rows = (start+valid_rows[unknown]).astype(np.int64, copy=False)
                    world = pose.to_world(xyz[valid][unknown])
                    roots = np.floor((world-self.origin)/self.index.block).astype(np.int64)
                    packed_roots = self.index._linear_keys(roots)
                    paths = self.index._tile_paths(world, roots, packed_roots)
                    records = np.empty(len(world), self._transfer_record_dtype)
                    records['xyz'], records['row'], records['scan'] = world, rows, scan
                    self._buffer_transfer_records(records, roots, paths)
                    self._transfer_query_count += len(records)
                staged += len(valid_rows)
                self.progress('output_transfer', staged/max(1, total), scan)
        self._flush_transfer_buffers()
        self.transfer_bucket_stats['transfer_bucket_files'] = len(list(Path(self.temporary.name).glob('queries_*.bin')))

    def _apply_transfer_records(self, records):
        """Query one spatial bucket and write its inherited codes to prediction maps."""
        query_radius = self.radius_max if self.adaptive_transfer else self.radius
        inherited, distances = self.index.nearest_codes(records['xyz'], query_radius, return_distances=True)
        assigned = inherited != 0
        scans = records['scan']
        if self.adaptive_transfer:
            radii = np.empty(len(records), np.float64)
            for scan in np.unique(scans):
                ids = np.flatnonzero(scans == scan)
                radii[ids] = self._transfer_radii(self._stores_by_scan[int(scan)], records['xyz'][ids])
            assigned &= distances <= radii
        if not assigned.any():
            return
        for scan in np.unique(scans[assigned]):
            ids = np.flatnonzero(assigned & (scans == scan))
            rows = records['row'][ids]
            codes, confidence, _ = self.predictions[int(scan)]
            codes[rows] = inherited[ids]
            confidence[rows] = np.maximum(confidence[rows], 128)
            self.class_transfer_count += len(ids)

    def _run_transfer_buckets(self):
        """Process one leaf at a time so the bounded index cache does not thrash."""
        completed = 0
        for path in sorted(Path(self.temporary.name).glob('queries_*.bin')):
            try:
                self.transfer_bucket_stats['transfer_bucket_open_calls'] += 1
                with path.open('rb') as handle:
                    while True:
                        self.checkpoint()
                        records = np.fromfile(handle, dtype=self._transfer_record_dtype,
                                              count=self._transfer_chunk_records)
                        if not len(records):
                            break
                        self._apply_transfer_records(records)
                        completed += len(records)
                        self.progress('output_transfer', completed/max(1, self._transfer_query_count),
                                      int(records['scan'][0]))
            finally:
                path.unlink(missing_ok=True)
        for maps in self.predictions.values():
            for mapping in maps:
                mapping.flush()

    def _prepare_reduced_transfer(self, total):
        """Finalize sources once, then transfer all reduced unknowns in leaf order."""
        started = time.perf_counter()
        self.index._finalize()
        self._stage_transfer_buckets(total)
        self._run_transfer_buckets()
        self.index_cache_stats = self.index.cache_stats()
        self.timing_seconds['transfer'] += time.perf_counter()-started
        self._transfer_prepared = True

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
        if (self.index is not None and store.meta['scan_index'] not in self.skipped and
                not (density == 'reduced' and self._transfer_prepared)):
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
            self.index_cache_stats = self.index.cache_stats()
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
        if store.meta['scan_index'] not in self.skipped and self.terrain is not None and self.terrain_fill and self.ground_dtm is None and (self.terrain_outdoor_scans is None
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
        if store.meta['scan_index'] not in self.skipped and self.tree_crowns is not None and self.tree_fill:
            started = time.perf_counter()
            unknown = codes == 1
            filled = self.tree_crowns.mask(world[unknown])
            ids = np.flatnonzero(unknown)[filled]
            codes[ids], confidence[ids] = 5, 64
            if len(ids):
                self.geometry_fill_counts['tree_fill'] = self.geometry_fill_counts.get('tree_fill', 0)+len(ids)
            self.timing_seconds['tree_fill'] += time.perf_counter()-started
        if store.meta['scan_index'] not in self.skipped and self.ground_dtm is not None and self.ground_dtm.dtm is not None:
            started = time.perf_counter()
            origin = self.origin
            voxel_size = self.voxel_size
            band = self.ground_dtm.band

            # 1. Ground points: check DTM height; if |z - DTM| > band -> second class or 1
            is_ground = (codes == 2)
            if np.any(is_ground):
                ground_indices = np.flatnonzero(is_ground)
                ground_pts = world[ground_indices]
                dtm_h = self.ground_dtm.dtm.height(ground_pts[:, :2])
                valid_dtm = np.isfinite(dtm_h)
                above_band = valid_dtm & (np.abs(ground_pts[:, 2] - dtm_h) > band)
                if np.any(above_band):
                    demote_sub_indices = np.flatnonzero(above_band)
                    demote_indices = ground_indices[demote_sub_indices]
                    demote_pts = ground_pts[demote_sub_indices]

                    q_keys = voxel_keys(demote_pts, origin, voxel_size)
                    if len(self.ground_dtm.keys):
                        found_idx = np.searchsorted(self.ground_dtm.keys, q_keys)
                        matched = (found_idx < len(self.ground_dtm.keys))
                        matched[matched] &= (self.ground_dtm.keys[found_idx[matched]] == q_keys[matched])
                        new_c = np.ones(len(q_keys), dtype=np.uint8)
                        if matched.any():
                            new_c[matched] = self.ground_dtm.codes[found_idx[matched]]
                    else:
                        new_c = np.ones(len(q_keys), dtype=np.uint8)

                    codes[demote_indices] = new_c
                    unclass_mask = (new_c == 1)
                    if np.any(unclass_mask):
                        confidence[demote_indices[unclass_mask]] = 0

                    self.geometry_fill_counts['ground_dtm_to_alt'] = (
                        self.geometry_fill_counts.get('ground_dtm_to_alt', 0) + len(demote_indices)
                    )

            # 2. Unclassified points (code 1): if outside building outline and |z - DTM| <= band -> 2
            is_unclass = (codes == 1)
            if np.any(is_unclass):
                unclass_indices = np.flatnonzero(is_unclass)
                unclass_pts = world[unclass_indices]
                dtm_h = self.ground_dtm.dtm.height(unclass_pts[:, :2])
                valid_dtm = np.isfinite(dtm_h)
                within_band = valid_dtm & (np.abs(unclass_pts[:, 2] - dtm_h) <= band)

                if np.any(within_band):
                    cand_sub_indices = np.flatnonzero(within_band)
                    cand_pts = unclass_pts[cand_sub_indices]

                    if self.ground_dtm.outline_applied and self.ground_dtm.outline_inside_barrier is not None:
                        pt_indices = np.floor((cand_pts - origin) / voxel_size).astype(np.int64)
                        grid_xy = pt_indices[:, :2] // self.ground_dtm.outline_cells
                        raster_idx = grid_xy - self.ground_dtm.outline_origin
                        gx = raster_idx[:, 0]
                        gy = raster_idx[:, 1]
                        shape = self.ground_dtm.outline_shape
                        in_grid = (gx >= 0) & (gx < shape[0]) & (gy >= 0) & (gy < shape[1])
                        excluded = np.zeros(len(cand_pts), bool)
                        if in_grid.any():
                            excluded[in_grid] = self.ground_dtm.outline_inside_barrier[gx[in_grid], gy[in_grid]]
                        outside = ~excluded
                    else:
                        outside = np.ones(len(cand_pts), bool)

                    if np.any(outside):
                        promote_indices = unclass_indices[cand_sub_indices[outside]]
                        codes[promote_indices] = 2
                        confidence[promote_indices] = 64
                        self.geometry_fill_counts['ground_dtm_fill'] = (
                            self.geometry_fill_counts.get('ground_dtm_fill', 0) + len(promote_indices)
                        )

            self.timing_seconds['ground_dtm'] += time.perf_counter() - started
        return world, codes, confidence, states

    def close(self):
        """Release all output-preparation resources; safe to call repeatedly."""
        self._transfer_buffers.clear()
        self._transfer_buffer_bytes = 0
        index, self.index = self.index, None
        if index is not None:
            try:
                self.index_cache_stats = index.cache_stats()
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
        ground_dtm, self.ground_dtm = self.ground_dtm, None
        if ground_dtm is not None:
            try:
                ground_dtm.close()
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
