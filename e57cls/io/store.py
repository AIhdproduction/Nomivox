"""Per-scan NPY storage with bounded mappings and atomic completion markers."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from e57cls.files import atomic_json, file_hash
from e57cls.io.reader import E57Reader, metadata


@dataclass(frozen=True)
class ScanStore:
    directory: Path
    meta: dict

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        if not 0 <= start <= stop <= self.meta["record_count"]:
            raise ValueError("Scan slice is outside the original record range")
        mapping = np.load(self.directory / self.meta["arrays"][key]["path"], mmap_mode="r", allow_pickle=False)
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


def load_scan(directory: Path, source_hash: str | None = None, verify_hashes: bool = True) -> ScanStore:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta["schema_version"] != 1 or not meta["complete"]:
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


def _initialize(path: Path, dtype, shape: tuple) -> None:
    mapping = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    mapping.flush()
    mapping._mmap.close()


def _write_slice(path: Path, start: int, values: np.ndarray) -> None:
    mapping = np.load(path, mmap_mode="r+", allow_pickle=False)
    try:
        mapping[start:start + len(values)] = values
        mapping.flush()
    finally:
        mapping._mmap.close()


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
    arrays, chunks = {}, []
    valid_count = invalid_count = 0
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    started = time.monotonic()
    for chunk in reader.chunks(index, capacity):
        values = {"xyz_local": chunk.xyz, "valid": chunk.valid}
        values.update({name: data for name, data in chunk.fields.items()
                       if name not in ("cartesianX", "cartesianY", "cartesianZ")})
        if not arrays:
            for number, (name, value) in enumerate(values.items()):
                # Numeric names prevent extension field names becoming filesystem paths.
                path = generation / f"array_{number:02d}.npy"
                shape = (count, *value.shape[1:])
                _initialize(path, value.dtype, shape)
                arrays[name] = {"path": path.relative_to(directory).as_posix(),
                                "dtype": str(value.dtype), "shape": list(shape)}
        for name, value in values.items():
            _write_slice(directory / arrays[name]["path"], chunk.start, value)
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
            _initialize(path, dtype, shape)
            arrays[name] = {"path": path.relative_to(directory).as_posix(),
                            "dtype": str(np.dtype(dtype)), "shape": list(shape)}
    for info in arrays.values():
        info["sha256"] = file_hash(directory / info["path"])
    result = {"schema_version": 1, "complete": True, "source_sha256": source_hash,
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
    result = {"schema_version": 1, "source": str(source), "source_sha256": fingerprint,
              "record_count": sum(s["record_count"] for s in summary),
              "valid_count": sum(s["valid_count"] for s in summary),
              "invalid_count": sum(s["invalid_count"] for s in summary),
              "scans": summary, "capacity": capacity, "elapsed_seconds": time.monotonic() - started,
              "coordinate_dtype": "float64", "input_stat_unchanged": True}
    atomic_json(root / "store.json", result)
    return result
