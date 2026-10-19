"""Disk-backed tiled point index for bounded SLAM visibility queries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from e57cls.files import atomic_json
from e57cls.geometry.cameras import scan_pose


_RECORD = np.dtype([("xyz", "<f8", (3,)), ("point_id", "<i8"), ("spacing", "<f4"), ("normal", "<f4", (3,))])


def point_id(scan_index: int, record_index: int) -> int:
    """A stable, reversible point identifier for a data3D block and record row."""
    if not 0 <= record_index < (1 << 48):
        raise ValueError("Point record index is outside the supported range")
    return (int(scan_index) << 48) | int(record_index)


def unpack_point_id(value: int) -> tuple[int, int]:
    return int(value) >> 48, int(value) & ((1 << 48)-1)


def _tile_key(root: np.ndarray) -> str:
    return "_".join(str(int(value)) for value in root)


@dataclass(frozen=True)
class SpatialIndex:
    """Tile fragments on disk; queries materialise only intersecting tiles."""

    directory: Path
    meta: dict
    tile_fragments: dict[str, tuple[Path, ...]]

    @property
    def tile_size(self) -> float:
        return float(self.meta["tile_size"])

    @property
    def anchor(self) -> np.ndarray:
        return np.asarray(self.meta["anchor"], np.float64)

    def _roots_for_bounds(self, low, high):
        low = np.floor((np.asarray(low)-self.anchor)/self.tile_size).astype(np.int64)
        high = np.floor((np.asarray(high)-self.anchor)/self.tile_size).astype(np.int64)
        for x in range(int(low[0]), int(high[0])+1):
            for y in range(int(low[1]), int(high[1])+1):
                for z in range(int(low[2]), int(high[2])+1):
                    yield np.asarray((x, y, z), np.int64)

    def query_bounds(self, low, high, checkpoint=lambda: None) -> np.ndarray:
        """Return only points in a requested world-space box."""
        groups = []
        for root in self._roots_for_bounds(low, high):
            checkpoint()
            for path in self.tile_fragments.get(_tile_key(root), ()):
                checkpoint()
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                try:
                    keep = np.all((values["xyz"] >= low) & (values["xyz"] <= high), axis=1)
                    if np.any(keep):
                        groups.append(np.array(values[keep], copy=True))
                finally:
                    values._mmap.close()
        return np.concatenate(groups) if groups else np.empty(0, _RECORD)

    def query_range(self, position, maximum, checkpoint=lambda: None) -> np.ndarray:
        position = np.asarray(position, np.float64)
        values = self.query_bounds(position-maximum, position+maximum, checkpoint)
        if len(values):
            values = values[np.linalg.norm(values["xyz"]-position, axis=1) <= maximum]
        return values


def _tile_fragments(directory: Path) -> dict[str, tuple[Path, ...]]:
    """Read the immutable tile layout once, without probing missing roots."""
    tiles = Path(directory) / "tiles"
    result = {}
    try:
        with os.scandir(tiles) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                with os.scandir(entry.path) as fragments:
                    paths = tuple(sorted((Path(fragment.path) for fragment in fragments
                                          if fragment.is_file() and fragment.name.endswith(".npy")),
                                         key=lambda path: path.name))
                if paths:
                    result[entry.name] = paths
    except FileNotFoundError:
        pass
    return result


def load_spatial_index(directory: Path, source_hash: str | None = None) -> SpatialIndex:
    """Open a complete index without loading any point tile."""
    directory = Path(directory)
    meta = json.loads((directory/"meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or not meta.get("complete"):
        raise ValueError("Unsupported or incomplete SLAM spatial index")
    if source_hash is not None and meta.get("source_sha256") != source_hash:
        raise ValueError("SLAM spatial index belongs to another input")
    return SpatialIndex(directory, meta, _tile_fragments(directory))


def _spacing_and_normals(xyz: np.ndarray, voxel_size: float, checkpoint=lambda: None) -> tuple[np.ndarray, np.ndarray]:
    """One conservative local spacing estimate per voxel, without global RAM use."""
    if not len(xyz):
        return np.empty(0, np.float32), np.empty((0, 3), np.float32)
    roots = np.floor(xyz/voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(roots, axis=0, return_inverse=True, return_counts=True)
    # A cube-equivalent spacing is stable across chunk boundaries and deliberately
    # conservative, so sparse occluders close holes instead of exposing far walls.
    spacing = (voxel_size/np.cbrt(counts[inverse])).astype(np.float32)
    normals = np.full((len(xyz), 3), np.nan, np.float32)
    for group in range(len(counts)):
        checkpoint()
        selected = np.flatnonzero(inverse == group)
        if len(selected) < 3:
            continue
        centered = xyz[selected]-xyz[selected].mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        normal = vectors[-1]
        if np.isfinite(normal).all():
            normals[selected] = normal.astype(np.float32)
    return spacing, normals


def _write_fragment(directory: Path, root: np.ndarray, values: np.ndarray) -> None:
    target = directory / "tiles" / _tile_key(root)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f".{uuid4().hex}.tmp.npy"
    final = target / f"part_{uuid4().hex}.npy"
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(final)


def _build(directory: Path, batches, source_hash="arrays", tile_size=2.0, voxel_size=.10,
           ram_budget_bytes=64*1024**2, checkpoint=lambda: None) -> SpatialIndex:
    if not 0 < float(tile_size) < float("inf") or not 0 < float(voxel_size) < float("inf"):
        raise ValueError("Tile and voxel size must be positive")
    if not isinstance(ram_budget_bytes, int) or ram_budget_bytes < 64*1024:
        raise ValueError("RAM budget must be at least 64 KiB")
    marker = directory/"meta.json"
    if marker.is_file():
        try:
            existing = load_spatial_index(directory, source_hash)
            if existing.tile_size == float(tile_size) and existing.meta.get("voxel_size") == float(voxel_size):
                return existing
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    directory.mkdir(parents=True, exist_ok=True)
    anchor = None
    fragments = points = 0
    maximum_batch_bytes = 0
    for scan_index, record_start, xyz in batches:
        checkpoint()
        xyz = np.asarray(xyz, np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("Index batches must contain Nx3 coordinates")
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        original_offsets = np.flatnonzero(valid).astype(np.int64)+int(record_start)
        if not len(xyz):
            continue
        if anchor is None:
            anchor = xyz.min(axis=0)
        maximum_batch_bytes = max(maximum_batch_bytes, int(xyz.nbytes + original_offsets.nbytes))
        roots = np.floor((xyz-anchor)/tile_size).astype(np.int64)
        spacing, normals = _spacing_and_normals(xyz, voxel_size, checkpoint)
        for root in np.unique(roots, axis=0):
            checkpoint()
            take = np.all(roots == root, axis=1)
            rows = np.empty(int(take.sum()), _RECORD)
            rows["xyz"] = xyz[take]
            rows["point_id"] = [point_id(scan_index, offset) for offset in original_offsets[take]]
            rows["spacing"] = spacing[take]
            rows["normal"] = normals[take]
            _write_fragment(directory, root, rows)
            fragments += 1
            points += len(rows)
    meta = {"schema_version": 1, "complete": True, "source_sha256": source_hash,
            "tile_size": float(tile_size), "voxel_size": float(voxel_size),
            "anchor": (anchor if anchor is not None else np.zeros(3)).tolist(), "points": points,
            "fragments": fragments, "ram_budget_bytes": ram_budget_bytes,
            "maximum_batch_bytes": maximum_batch_bytes, "record_dtype": _RECORD.descr}
    atomic_json(directory/"meta.json", meta)
    return SpatialIndex(directory, meta, _tile_fragments(directory))


def build_spatial_index(stores, directory: Path, source_hash: str, tile_size=2.0, voxel_size=.10,
                        ram_budget_bytes=64*1024**2, checkpoint=lambda: None) -> SpatialIndex:
    """Build an on-disk index from schema-2 scan stores in budgeted chunks."""
    capacity = max(1, min(100_000, ram_budget_bytes//256))

    def batches():
        for store in stores:
            pose = scan_pose(store.meta["metadata"])
            for start, local, valid in store.full_chunks(capacity):
                checkpoint()
                # Keep record positions, not compacted ordinal positions.
                world = np.asarray(local, np.float64)
                world[~valid] = np.nan
                yield store.meta["scan_index"], start, pose.to_world(world)

    return _build(Path(directory), batches(), source_hash, tile_size, voxel_size, ram_budget_bytes, checkpoint)


def build_spatial_index_from_arrays(points, directory: Path, tile_size=2.0, voxel_size=.10,
                                  ram_budget_bytes=64*1024**2, checkpoint=lambda: None) -> SpatialIndex:
    """Build the same disk format from ``{scan_index: Nx3}`` arrays for tools and tests."""
    def batches():
        for scan_index, xyz in sorted(points.items()):
            # Split the helper input too, so its RAM accounting matches production.
            capacity = max(1, min(100_000, ram_budget_bytes//256))
            for start in range(0, len(xyz), capacity):
                checkpoint()
                yield int(scan_index), start, np.asarray(xyz[start:start+capacity], np.float64)
    return _build(Path(directory), batches(), "arrays", tile_size, voxel_size, ram_budget_bytes, checkpoint)
