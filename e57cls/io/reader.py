"""Foundation API access: bounded buffers, scaled values, no source writes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from pye57 import libe57 as lib


def metadata(node):
    if isinstance(node, lib.StructureNode):
        return {node[i].elementName(): metadata(node[i]) for i in range(node.childCount())}
    if isinstance(node, lib.VectorNode):
        return [metadata(node[i]) for i in range(node.childCount())]
    if isinstance(node, lib.CompressedVectorNode):
        proto = lib.StructureNode(node.prototype())
        fields = {}
        for i in range(proto.childCount()):
            field = proto[i]
            description = {"type": type(field).__name__.removesuffix("Node")}
            for attribute in ("minimum", "maximum", "scale", "offset"):
                if hasattr(field, attribute):
                    description[attribute] = getattr(field, attribute)()
            if hasattr(field, "precision"):
                description["precision"] = str(field.precision())
            fields[field.elementName()] = description
        return {"record_count": node.childCount(), "fields": fields}
    if isinstance(node, lib.BlobNode):
        return {"byte_count": node.byteCount()}
    if isinstance(node, lib.ScaledIntegerNode):
        return node.scaledValue()
    return node.value()


def coordinates(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    cartesian = ("cartesianX", "cartesianY", "cartesianZ")
    spherical = ("sphericalRange", "sphericalAzimuth", "sphericalElevation")
    if all(k in data for k in cartesian):
        xyz = np.column_stack([data[k] for k in cartesian]).astype(np.float64)
        state = data.get("cartesianInvalidState", np.zeros(len(xyz), np.uint8))
        valid = (state == 0) & np.isfinite(xyz).all(axis=1)
    elif all(k in data for k in spherical):
        radius, azimuth, elevation = (data[k] for k in spherical)
        xyz = np.column_stack((radius * np.cos(elevation) * np.cos(azimuth),
                               radius * np.cos(elevation) * np.sin(azimuth),
                               radius * np.sin(elevation)))
        state = data.get("sphericalInvalidState", np.zeros(len(xyz), np.uint8))
        valid = (state == 0) & (radius >= 0) & np.isfinite(xyz).all(axis=1)
    else:
        raise ValueError("No complete Cartesian or spherical coordinate fields")
    # Zero range is not discarded without an explicit invalid-state flag.
    return xyz, valid


def integer_dtype(description: dict) -> np.dtype:
    """Return the narrowest E57-compatible integer dtype for a prototype."""
    low, high = int(description.get("minimum", -2**63)), int(description.get("maximum", 2**63-1))
    for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return np.dtype(dtype)
    raise ValueError("E57 integer prototype is outside int64")


def decode_raw(values: np.ndarray, description: dict) -> np.ndarray:
    """Match libE57's scaled-integer conversion order in double precision."""
    if description.get("type") == "Integer":
        return values.astype(np.int64, copy=False)
    return (values.astype(np.float64, copy=False) * np.asarray(description.get("scale", 1.), dtype=np.float64)
            + np.asarray(description.get("offset", 0.), dtype=np.float64))


@dataclass
class PointChunk:
    start: int
    fields: dict[str, np.ndarray]
    xyz: np.ndarray
    valid: np.ndarray


class E57Reader:
    def __init__(self, path: Path):
        self.path = Path(path).resolve(strict=True)
        self.file = lib.ImageFile(str(self.path), "r")
        self.root = self.file.root()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def vector(self, name: str):
        return self.root[name] if self.root.isDefined(name) else None

    def chunks(self, scan_index: int, capacity: int = 100_000,
               limit: int | None = None, raw: bool = False) -> Iterator[PointChunk]:
        if capacity < 1 or (limit is not None and limit < 0):
            raise ValueError("Capacity must be positive and limit nonnegative")
        cv = self.root["data3D"][scan_index]["points"]
        count = cv.childCount() if limit is None else min(cv.childCount(), limit)
        if count == 0:
            return
        capacity = min(capacity, count)
        proto = lib.StructureNode(cv.prototype())
        descriptions = metadata(cv)["fields"]
        arrays, buffers = {}, lib.VectorSourceDestBuffer()
        for i in range(proto.childCount()):
            node = proto[i]
            name = node.elementName()
            if not isinstance(node, (lib.IntegerNode, lib.ScaledIntegerNode, lib.FloatNode)):
                raise ValueError(f"Unsupported nonnumeric point field: {name}")
            description = descriptions[name]
            raw_integer = raw and isinstance(node, (lib.IntegerNode, lib.ScaledIntegerNode))
            array = np.empty(capacity, integer_dtype(description) if raw_integer
                             else (np.int64 if isinstance(node, lib.IntegerNode) else np.float64))
            arrays[name] = array
            buffers.append(lib.SourceDestBuffer(self.file, name, array, capacity,
                                               not raw_integer, not raw_integer))
        reader = cv.reader(buffers)
        start = 0
        try:
            while start < count:
                received = reader.read()
                if received == 0:
                    raise ValueError(f"Unexpected end of scan {scan_index} at record {start}")
                n = min(received, count - start)
                # Owned arrays let consumers retain a chunk safely across reads.
                data = {key: value[:n].copy() for key, value in arrays.items()}
                coordinate_data = data
                if raw:
                    coordinate_data = {name: decode_raw(value, descriptions[name])
                                       if descriptions[name]["type"] in ("Integer", "ScaledInteger") else value
                                       for name, value in data.items()}
                xyz, valid = coordinates(coordinate_data)
                yield PointChunk(start, data, xyz, valid)
                start += n
        finally:
            reader.close()

    def blob(self, image_index: int, representation: str, name: str):
        return self.root["images2D"][image_index][representation][name]
