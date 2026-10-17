"""Per-scan NPY storage with bounded mappings and atomic completion markers."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from e57cls.files import atomic_json, file_hash
from e57cls.io.reader import E57Reader, metadata, decode_raw


@dataclass(frozen=True)
class ScanStore:
    directory: Path
    meta: dict

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        if not 0 <= start <= stop <= self.meta["record_count"]:
            raise ValueError("Scan slice is outside the original record range")
        info = self.meta["arrays"][key]
        mapping = np.load(self.directory / info["path"], mmap_mode="r", allow_pickle=False)
        try:
            values = np.array(mapping[start:stop], copy=True)
        finally:
            mapping._mmap.close()
        if info.get("encoding") == "raw":
            return decode_raw(values, {"type": "ScaledInteger" if info.get("scale", 1) != 1 or info.get("offset", 0) != 0 else "Integer",
                                       "scale": info.get("scale", 1), "offset": info.get("offset", 0)})
        return values

    def read_raw(self, key: str, start: int, stop: int) -> np.ndarray:
        if not 0 <= start <= stop <= self.meta["record_count"]:
            raise ValueError("Scan slice is outside the original record range")
        info = self.meta["arrays"][key]
        mapping = np.load(self.directory / info["path"], mmap_mode="r", allow_pickle=False)
        try:
            return np.array(mapping[start:stop], copy=True)
        finally:
            mapping._mmap.close()

    def chunks(self, capacity: int = 100_000):
        if capacity < 1:
            raise ValueError("Capacity must be positive")
        for start in range(0, self.meta["record_count"], capacity):
            stop = min(start + capacity, self.meta["record_count"])
            yield start, self.read("xyz_local", start, stop), self.read("valid", start, stop)

    def full_chunks(self, capacity: int = 100_000):
        """Yield all valid-state information without a processing reduction."""
        yield from self.chunks(capacity)


@dataclass(frozen=True)
class ReducedStore:
    """A scan view whose ``chunks`` expose only deterministic retained points.

    Array reads and metadata intentionally remain those of the complete store.
    This keeps the scan-store cache independent of the chosen reduction and lets
    output code explicitly select ``full_chunks`` for original-density exports.
    """
    store: ScanStore
    keep_path: Path

    @property
    def directory(self):
        return self.store.directory

    @property
    def meta(self):
        return self.store.meta

    @property
    def retained_count(self):
        return json.loads(self.keep_path.with_suffix(".json").read_text(encoding="utf-8"))["retained_count"]

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        return self.store.read(key, start, stop)

    def read_raw(self, key: str, start: int, stop: int) -> np.ndarray:
        return self.store.read_raw(key, start, stop)

    def chunks(self, capacity: int = 100_000):
        keep = np.load(self.keep_path, mmap_mode="r", allow_pickle=False)
        try:
            for start, xyz, valid in self.store.chunks(capacity):
                yield start, xyz, valid & keep[start:start+len(valid)]
        finally:
            keep._mmap.close()

    def full_chunks(self, capacity: int = 100_000):
        yield from self.store.chunks(capacity)


def _keep_name(reduction: float, seed: int) -> str:
    return f"keep_{reduction:.8f}_{seed}.npy"


def reduction_mask(store: ScanStore, reduction: float, seed: int) -> Path | None:
    """Return a cached, chunk-size-independent retained-point mask.

    Selection is a seeded systematic sample across valid record order.  It
    retains exactly ``round(valid * (1-reduction))`` records while preserving a
    near-identical fraction in every part of a scan.
    """
    if reduction == 0:
        return None
    name = _keep_name(reduction, seed)
    path = store.directory/name
    marker = path.with_suffix(".json")
    if path.is_file() and marker.is_file():
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            if (info.get("reduction") == reduction and info.get("seed") == seed
                    and info.get("record_count") == store.meta["record_count"]
                    and info.get("retained_count") == round(store.meta["valid_count"]*(1-reduction))
                    and info.get("sha256") == file_hash(path)):
                return path
        except (OSError, ValueError):
            pass
    valid_count = store.meta["valid_count"]
    retained = round(valid_count*(1-reduction))
    temporary = path.with_name(path.stem+".partial.npy")
    mask = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.bool_,
                                     shape=(store.meta["record_count"],))
    mask[:] = False
    if retained and valid_count:
        # One point per equal-width stratum avoids density bands while the
        # seeded phase makes independent scans select different records.
        phase = np.random.default_rng(seed + store.meta["scan_index"]).random()
        ranks = np.floor((np.arange(retained)+phase)*valid_count/retained).astype(np.int64)
        rank_start = 0
        for start, _, valid in store.full_chunks():
            rank_stop = rank_start+int(valid.sum())
            left, right = np.searchsorted(ranks, [rank_start, rank_stop])
            if left != right:
                local_valid = np.flatnonzero(valid)
                mask[start+local_valid[ranks[left:right]-rank_start]] = True
            rank_start = rank_stop
    mask.flush()
    mask._mmap.close()
    temporary.replace(path)
    atomic_json(marker, {"reduction": reduction, "seed": seed,
                         "record_count": store.meta["record_count"], "retained_count": retained,
                         "sha256": file_hash(path)})
    return path


def reduced_store(store: ScanStore, reduction: float, seed: int) -> ScanStore | ReducedStore:
    """Return a reduced view, or the original object for the exact 0 % path."""
    path = reduction_mask(store, reduction, seed)
    return store if path is None else ReducedStore(store, path)


def load_scan(directory: Path, source_hash: str | None = None, verify_hashes: bool = True) -> ScanStore:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta["schema_version"] != 2 or not meta["complete"]:
        raise ValueError("Unsupported or incomplete scan store")
    if source_hash is not None and meta["source_sha256"] != source_hash:
        raise ValueError("Scan store input identity changed")
    for info in meta["arrays"].values():
        path = (directory / info["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid cache path")
        if verify_hashes and file_hash(path) != info["sha256"]:
            raise ValueError(f"Corrupt scan store: {path}")
        mapping = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if list(mapping.shape) != info["shape"] or str(mapping.dtype) != info["dtype"]:
                raise ValueError(f"Scan store array schema mismatch: {path}")
        finally:
            mapping._mmap.close()
    return ScanStore(directory, meta)


def _open_writer(path: Path, dtype, shape: tuple):
    """Open one sequential NPY writer and initialise its file-order hash."""
    mapping = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    data_bytes = int(np.prod(shape, dtype=np.int64))*np.dtype(dtype).itemsize
    with path.open("rb") as stream:
        digest = hashlib.sha256(stream.read(path.stat().st_size-data_bytes))
    return mapping, digest


def write_scan(reader: E57Reader, index: int, root: Path, source_hash: str,
               capacity: int, progress=None) -> ScanStore:
    directory = root / "scans" / f"scan_{index + 1:04d}"
    if (directory / "meta.json").is_file():
        try:
            return load_scan(directory, source_hash)
        except (OSError, ValueError, KeyError):
            # A new generation repairs corrupt/incompatible cache without trusting it.
            pass
    generation = directory / f"data_{uuid4().hex}"
    generation.mkdir(parents=True, exist_ok=False)
    scan_meta = metadata(reader.root["data3D"][index])
    count = scan_meta["points"]["record_count"]
    arrays, writers, chunks = {}, {}, []
    field_meta = scan_meta["points"]["fields"]
    valid_count = invalid_count = 0
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    started = time.monotonic()
    for chunk in reader.chunks(index, capacity, raw=True):
        cartesian = ("cartesianX", "cartesianY", "cartesianZ")
        raw_xyz = all(name in chunk.fields and field_meta[name]["type"] == "ScaledInteger" for name in cartesian)
        values = {"xyz_local": (np.column_stack([chunk.fields[name] for name in cartesian]) if raw_xyz else chunk.xyz),
                  "valid": chunk.valid}
        values.update({name: data for name, data in chunk.fields.items() if name not in cartesian})
        if not arrays:
            for number, (name, value) in enumerate(values.items()):
                # Numeric names prevent extension field names becoming filesystem paths.
                path = generation / f"array_{number:02d}.npy"
                shape = (count, *value.shape[1:])
                mapping, digest = _open_writer(path, value.dtype, shape)
                arrays[name] = {"path": path.relative_to(directory).as_posix(),
                                "dtype": str(value.dtype), "shape": list(shape)}
                if name == "xyz_local" and raw_xyz:
                    arrays[name].update(encoding="raw", scale=[field_meta[n].get("scale", 1.) for n in cartesian],
                                        offset=[field_meta[n].get("offset", 0.) for n in cartesian])
                elif name in field_meta and field_meta[name]["type"] in ("Integer", "ScaledInteger"):
                    arrays[name].update(encoding="raw", scale=field_meta[name].get("scale", 1.),
                                        offset=field_meta[name].get("offset", 0.))
                writers[name] = (mapping, digest)
        for name, value in values.items():
            mapping, digest = writers[name]
            mapping[chunk.start:chunk.start+len(value)] = value
            digest.update(np.ascontiguousarray(value).tobytes())
        nvalid = int(chunk.valid.sum())
        valid_count += nvalid
        invalid_count += len(chunk.valid) - nvalid
        if nvalid:
            np.minimum(low, chunk.xyz[chunk.valid].min(axis=0), out=low)
            np.maximum(high, chunk.xyz[chunk.valid].max(axis=0), out=high)
        chunks.append({"start": chunk.start, "count": len(chunk.valid), "valid": nvalid})
        if progress:
            progress(index + 1, chunk.start + len(chunk.valid), count)
    if valid_count + invalid_count != count:
        raise RuntimeError("Scan store record accounting failed")
    if not arrays:
        for number, (name, dtype, shape) in enumerate((
                ("xyz_local", np.float64, (0, 3)), ("valid", np.bool_, (0,)))):
            path = generation / f"array_{number:02d}.npy"
            mapping, digest = _open_writer(path, dtype, shape)
            mapping.flush()
            mapping._mmap.close()
            arrays[name] = {"path": path.relative_to(directory).as_posix(),
                            "dtype": str(np.dtype(dtype)), "shape": list(shape), "sha256": digest.hexdigest()}
    for name, (mapping, digest) in writers.items():
        mapping.flush()
        mapping._mmap.close()
        arrays[name]["sha256"] = digest.hexdigest()
    result = {"schema_version": 2, "complete": True, "source_sha256": source_hash,
              "scan_index": index + 1, "metadata": scan_meta, "record_count": count,
              "valid_count": valid_count, "invalid_count": invalid_count,
              "bounds_local": {"min": low.tolist(), "max": high.tolist()} if valid_count else None,
              "arrays": arrays, "chunks": chunks, "elapsed_seconds": time.monotonic() - started}
    atomic_json(directory / "meta.json", result)
    return ScanStore(directory, result)


def build_store(source: Path, cache: Path, capacity: int = 100_000, progress=None) -> dict:
    if not 1 <= capacity <= 1_000_000:
        raise ValueError("Chunk capacity must be between 1 and 1000000")
    source, cache = source.resolve(strict=True), cache.resolve()
    if cache == source or cache.is_file():
        raise ValueError("Cache must be a directory distinct from the input")
    before = source.stat()
    fingerprint = file_hash(source)
    root = cache / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    started = time.monotonic()
    with E57Reader(source) as reader:
        data = reader.vector("data3D")
        for i in range(data.childCount() if data else 0):
            store = write_scan(reader, i, root, fingerprint, capacity, progress)
            summary.append({"scan_index": i + 1, "path": str(store.directory),
                            **{k: store.meta[k] for k in ("record_count", "valid_count", "invalid_count")}})
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during scan store construction")
    result = {"schema_version": 2, "source": str(source), "source_sha256": fingerprint,
              "record_count": sum(s["record_count"] for s in summary),
              "valid_count": sum(s["valid_count"] for s in summary),
              "invalid_count": sum(s["invalid_count"] for s in summary),
              "scans": summary, "capacity": capacity, "elapsed_seconds": time.monotonic() - started,
              "coordinate_dtype": "float64", "input_stat_unchanged": True}
    atomic_json(root / "store.json", result)
    return result
