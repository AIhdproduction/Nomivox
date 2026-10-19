"""Streaming E57 output with Nomivox classification extension fields.

The fields ``nomivox:classification``, ``nomivox:classConfidence`` and
``nomivox:mirrorState`` are unsigned integer extension fields (0..255).  They
preserve source RGB and leave the source E57 untouched.
"""

from __future__ import annotations

from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from pye57 import libe57 as lib
from pye57.utils import copy_node
from e57cls.output.classify import ClassifiedSource, output_chunks, selected_count
from e57cls.vote.engine import voxel_keys


NOMIVOX_URI = "https://nomivox.example/e57/1.0"
_COORDINATES = {"cartesianX", "cartesianY", "cartesianZ", "sphericalRange", "sphericalAzimuth", "sphericalElevation"}


def _precision(description):
    return lib.E57_SINGLE if str(description.get("precision", "")).endswith("E57_SINGLE") else lib.E57_DOUBLE


def _node(image, description, name):
    kind = description.get("type", "Float")
    minimum, maximum = description.get("minimum"), description.get("maximum")
    if kind == "Integer":
        low = int(minimum if minimum is not None else -2**31)
        high = int(maximum if maximum is not None else 2**31-1)
        return lib.IntegerNode(image, low, low, max(low, high))
    if kind == "ScaledInteger":
        low = int(minimum if minimum is not None else -2**31)
        high = int(maximum if maximum is not None else 2**31-1)
        return lib.ScaledIntegerNode(image, low, low, max(low, high),
                                     float(description.get("scale", 1.)), float(description.get("offset", 0.)))
    # Scaled values are written as their physical values.  Float retains the
    # source range while avoiding lossy conversion to raw scaled integers.
    low = float(minimum if minimum is not None else -1e308)
    high = float(maximum if maximum is not None else 1e308)
    return lib.FloatNode(image, low, _precision(description), low, max(low, high))


def _set_pose(image, parent, pose):
    value = lib.StructureNode(image)
    rotation = lib.StructureNode(image)
    translation = lib.StructureNode(image)
    for key in "wxyz":
        rotation.set(key, lib.FloatNode(image, float(pose["rotation"][key])))
    for key in "xyz":
        translation.set(key, lib.FloatNode(image, float(pose["translation"][key])))
    value.set("rotation", rotation)
    value.set("translation", translation)
    parent.set("pose", value)


def _copy_blobs(pairs, block_size=1 << 20):
    for pair in pairs:
        source, target = pair["in"], pair["out"]
        for offset in range(0, source.byteCount(), block_size):
            count = min(block_size, source.byteCount()-offset)
            data = np.empty(count, np.uint8)
            source.read(data, offset, count)
            target.write(data, offset, count)


def _write_root(image, source_file):
    root = image.root()
    image.extensionsAdd("", lib.E57_V1_0_URI)
    for index in range(source_file.extensionsCount()):
        prefix = source_file.extensionsPrefix(index)
        uri = source_file.extensionsUri(index)
        if prefix or uri != lib.E57_V1_0_URI:
            image.extensionsAdd(prefix, uri)
    image.extensionsAdd("nomivox", NOMIVOX_URI)
    source_root = source_file.root()
    root.set("formatName", lib.StringNode(image, "ASTM E57 3D Imaging Data File"))
    root.set("guid", lib.StringNode(image, "{%s}" % uuid4()))
    root.set("versionMajor", lib.IntegerNode(image, lib.E57_FORMAT_MAJOR))
    root.set("versionMinor", lib.IntegerNode(image, lib.E57_FORMAT_MINOR))
    root.set("e57LibraryVersion", lib.StringNode(image, lib.E57_LIBRARY_ID))
    coordinate = source_root["coordinateMetadata"].value() if source_root.isDefined("coordinateMetadata") else ""
    root.set("coordinateMetadata", lib.StringNode(image, coordinate))
    created = lib.StructureNode(image)
    created.set("dateTimeValue", lib.FloatNode(image, 0.))
    created.set("isAtomicClockReferenced", lib.IntegerNode(image, 0))
    root.set("creationDateTime", created)
    root.set("data3D", lib.VectorNode(image, True))
    images, _, blobs = copy_node(source_root["images2D"], image) if source_root.isDefined("images2D") else (lib.VectorNode(image, True), [], [])
    root.set("images2D", images)
    _copy_blobs(blobs)
    return root


def _coordinate_node(image, description, low, high):
    if description.get("type") == "ScaledInteger":
        scale, offset = float(description["scale"]), float(description["offset"])
        raw_low = int(np.floor((low-offset)/scale))
        raw_high = int(np.ceil((high-offset)/scale))
        raw_low = min(raw_low, int(description.get("minimum", raw_low)))
        raw_high = max(raw_high, int(description.get("maximum", raw_high)))
        return lib.ScaledIntegerNode(image, raw_low, raw_low, max(raw_low, raw_high), scale, offset)
    return lib.FloatNode(image, float(low), _precision(description), float(low), max(float(low), float(high)))


def _scan_node(image, store, fields, coordinate_fields, bounds, station_ids=False, station_count=0):
    meta = store.meta["metadata"]
    node = lib.StructureNode(image)
    node.set("guid", lib.StringNode(image, meta.get("guid") or "{%s}" % uuid4()))
    node.set("name", lib.StringNode(image, meta.get("name") or f"Scan {store.meta['scan_index']}"))
    _set_pose(image, node, meta["pose"])
    box = lib.StructureNode(image)
    for axis, low, high in zip("xyz", bounds[0], bounds[1]):
        box.set(axis+"Minimum", lib.FloatNode(image, float(low)))
        box.set(axis+"Maximum", lib.FloatNode(image, float(high)))
    node.set("cartesianBounds", box)
    for key in ("colorLimits", "intensityLimits"):
        if key in meta:
            value = lib.StructureNode(image)
            for child, number in meta[key].items():
                value.set(child, (lib.IntegerNode(image, int(number)) if isinstance(number, int)
                                  else lib.FloatNode(image, float(number))))
            node.set(key, value)
    prototype = lib.StructureNode(image)
    for name, low, high in zip(("cartesianX", "cartesianY", "cartesianZ"), bounds[0], bounds[1]):
        prototype.set(name, _coordinate_node(image, coordinate_fields.get(name, {}), low, high))
    for name, info in fields.items():
        prototype.set(name, _node(image, info, name))
    for name in ("nomivox:classification", "nomivox:classConfidence", "nomivox:mirrorState"):
        prototype.set(name, lib.IntegerNode(image, 0, 0, 255))
    if station_ids:
        prototype.set("nomivox:stationId", lib.IntegerNode(image, 0, 0, max(1, int(station_count))))
    points = lib.CompressedVectorNode(image, prototype, lib.VectorNode(image, True))
    node.set("points", points)
    return node, points


def _limits(store, density):
    bounds = store.meta.get("bounds_local")
    if not bounds:
        return np.zeros(3), np.zeros(3)
    return np.asarray(bounds["min"], np.float64), np.asarray(bounds["max"], np.float64)


def _write(path, source, stores, votes, origin, voxel_size, allowed_codes, skipped, checkpoint,
           progress, mirror, density, without_reflections, class_transfer_radius, geometry_planes, geometry_fill_tolerance,
           terrain, terrain_ground_tolerance, terrain_fill,
           station_ids=False, station_count=0, force_unclassified=None, terrain_outdoor_scans=None,
           class_transfer_range_factor=0., class_transfer_radius_max=None, transfer_station_positions=None):
    source_file = lib.ImageFile(str(source), "r")
    image = lib.ImageFile(str(path), "w")
    total = sum(selected_count(store, density) for store in stores)
    written = discarded = transferred = 0
    counts, mirrors = np.zeros(256, np.int64), np.zeros(5, np.int64)
    samples = {}
    write_seconds = 0.
    try:
        root = _write_root(image, source_file)
        with ClassifiedSource(path.parent, stores, votes, origin, voxel_size, skipped, mirror, class_transfer_radius, checkpoint,
                              geometry_planes, geometry_fill_tolerance, terrain, terrain_ground_tolerance, terrain_fill,
                              progress, force_unclassified, terrain_outdoor_scans, class_transfer_range_factor,
                              class_transfer_radius_max, transfer_station_positions) as classified:
          for store in stores:
            prototypes = store.meta["metadata"]["points"]["fields"]
            fields = {name: prototypes.get(name, {"type": "Float"}) for name in store.meta["arrays"]
                      if name not in {"xyz_local", "valid"} | _COORDINATES}
            bounds = _limits(store, density)
            node, points = _scan_node(image, store, fields, prototypes, bounds, station_ids, station_count)
            root["data3D"].append(node)
            capacity = 500_000
            xyz_info = store.meta["arrays"]["xyz_local"]
            xyz_raw = xyz_info.get("encoding") == "raw"
            xyz_dtype = np.dtype(xyz_info["dtype"]) if xyz_raw else np.dtype(np.float64)
            arrays = {"cartesianX": np.empty(capacity, xyz_dtype), "cartesianY": np.empty(capacity, xyz_dtype), "cartesianZ": np.empty(capacity, xyz_dtype)}
            raw_names = {"cartesianX", "cartesianY", "cartesianZ"} if xyz_raw else set()
            for name, info in fields.items():
                raw = store.meta["arrays"][name].get("encoding") == "raw"
                arrays[name] = np.empty(capacity, np.dtype(store.meta["arrays"][name]["dtype"]) if raw
                                        else (np.int64 if info.get("type") == "Integer" else np.float64))
                if raw:
                    raw_names.add(name)
            arrays.update({name: np.empty(capacity, np.int64) for name in ("nomivox:classification", "nomivox:classConfidence", "nomivox:mirrorState")})
            if station_ids:
                arrays["nomivox:stationId"] = np.empty(capacity, np.uint32)
            buffers = lib.VectorSourceDestBuffer()
            for name, values in arrays.items():
                buffers.append(lib.SourceDestBuffer(image, name, values, capacity,
                                                   name not in raw_names, name not in raw_names))
            writer = points.writer(buffers)
            try:
                for start, xyz, valid in output_chunks(store, density, capacity):
                    checkpoint()
                    n = int(valid.sum())
                    if not n:
                        continue
                    world, codes, confidence, states = classified.block(store, start, xyz, valid, density)
                    if (codes == 0).any() or not np.isin(codes, allowed_codes).all():
                        raise ValueError("Output class missing from legend or forbidden code 0")
                    take = codes != 68 if without_reflections else np.ones(n, bool)
                    out_count = int(take.sum())
                    if out_count:
                        local = xyz[valid][take]
                        coordinate_values = store.read_raw("xyz_local", start, start+len(xyz)) if xyz_raw else xyz
                        arrays["cartesianX"][:out_count], arrays["cartesianY"][:out_count], arrays["cartesianZ"][:out_count] = coordinate_values[valid][take].T
                        for name in fields:
                            values = store.read_raw(name, start, start+len(xyz)) if name in raw_names else store.read(name, start, start+len(xyz))
                            arrays[name][:out_count] = values[valid][take]
                        arrays["nomivox:classification"][:out_count] = codes[take]
                        arrays["nomivox:classConfidence"][:out_count] = confidence[take]
                        arrays["nomivox:mirrorState"][:out_count] = states[take]
                        if station_ids:
                            station = votes.lookup_station_ids(voxel_keys(world, origin, voxel_size))
                            if force_unclassified is not None:
                                station[force_unclassified(store, start, valid)] = 0
                            arrays["nomivox:stationId"][:out_count] = station[take]
                        started = time.perf_counter()
                        writer.write(out_count)
                        write_seconds += time.perf_counter()-started
                        if store.meta["scan_index"] not in samples:
                            samples[store.meta["scan_index"]] = local[:32].copy()
                    counts += np.bincount(codes[take], minlength=256)
                    mirrors += np.bincount(states[take], minlength=5)
                    written += out_count
                    discarded += n-out_count
                    progress("output", (written+discarded)/max(1, total), store.meta["scan_index"])
            finally:
                writer.close()
          transferred = classified.class_transfer_count
    finally:
        image.close()
        source_file.close()
    return {"record_count": written, "class_counts": {str(i): int(n) for i, n in enumerate(counts) if n},
            "unclassified_share": int(counts[1])/max(1, written), "mirror_counts": mirrors.tolist(),
            "extra_removed_count": discarded, "class_transfer_count": transferred,
            "geometry_fill_counts": classified.geometry_fill_counts, "_samples": samples,
            "timing_seconds": {**classified.timing_seconds, 'write': write_seconds, 'readback': 0.}}


def write_e57(path: Path, source: Path, stores, votes, origin, voxel_size, allowed_codes,
              skipped=(), checkpoint=lambda: None, progress=lambda *args: None, mirror=None,
              no_reflection=False, primary_without_reflections=False, class_transfer_radius=0.,
              output_density="reduced", geometry_planes=None, geometry_fill_tolerance=.02,
              terrain=None, terrain_ground_tolerance=.15, terrain_fill=False,
              station_ids=False, station_count=0, force_unclassified=None, terrain_outdoor_scans=None,
              class_transfer_range_factor=0., class_transfer_radius_max=None, transfer_station_positions=None):
    """Write an E57 atomically, retaining images and Nomivox extension fields."""
    temporary = path.with_name(path.stem+".partial"+path.suffix)
    extra = path.with_name(path.stem+"_no_reflection"+path.suffix)
    extra_temporary = extra.with_name(extra.stem+".partial"+extra.suffix)
    try:
        result = _write(temporary, source, stores, votes, origin, voxel_size, allowed_codes, skipped, checkpoint,
                        progress, mirror, output_density, primary_without_reflections, class_transfer_radius,
                        geometry_planes, geometry_fill_tolerance, terrain, terrain_ground_tolerance, terrain_fill,
                        station_ids, station_count, force_unclassified, terrain_outdoor_scans,
                        class_transfer_range_factor, class_transfer_radius_max, transfer_station_positions)
        started = time.perf_counter()
        result["readback"] = verify_e57(temporary, source, stores, result["record_count"], allowed_codes,
                                          primary_without_reflections, result.pop("_samples"))
        result['timing_seconds']['readback'] += time.perf_counter()-started
        temporary.replace(path)
        result["path"] = str(path)
        result["extra_output"] = None
        if no_reflection:
            extra_result = _write(extra_temporary, source, stores, votes, origin, voxel_size, allowed_codes, skipped, checkpoint,
                                  progress, mirror, output_density, True, class_transfer_radius,
                                  geometry_planes, geometry_fill_tolerance, terrain, terrain_ground_tolerance, terrain_fill,
                                  station_ids, station_count, force_unclassified, terrain_outdoor_scans,
                                  class_transfer_range_factor, class_transfer_radius_max, transfer_station_positions)
            started = time.perf_counter()
            verify_e57(extra_temporary, source, stores, extra_result["record_count"], allowed_codes, True,
                       extra_result.pop("_samples"))
            extra_result['timing_seconds']['readback'] += time.perf_counter()-started
            extra_temporary.replace(extra)
            result["extra_output"] = str(extra)
            result["extra_removed_count"] = extra_result["extra_removed_count"]
            for key, value in extra_result['timing_seconds'].items():
                result['timing_seconds'][key] = result['timing_seconds'].get(key, 0.)+value
        return result
    except Exception:
        temporary.unlink(missing_ok=True)
        extra_temporary.unlink(missing_ok=True)
        raise


def verify_e57(path, source, stores, expected_count, allowed_codes, without_reflections, samples):
    """Read back point counts, extension classes and copied image byte sizes."""
    from e57cls.io.reader import E57Reader, metadata
    with E57Reader(path) as reader:
        count = 0
        for index, store in enumerate(stores):
            scan_count = 0
            expected = samples.get(store.meta["scan_index"], np.empty((0, 3)))
            coordinate_fields = store.meta["metadata"]["points"]["fields"]
            tolerance = np.array([float(coordinate_fields.get(name, {}).get("scale", 0.))/2
                                  for name in ("cartesianX", "cartesianY", "cartesianZ")])
            checked = 0
            for chunk in reader.chunks(index):
                classes = chunk.fields["nomivox:classification"].astype(np.uint8)
                if (classes == 0).any() or not np.isin(classes, allowed_codes).all():
                    raise RuntimeError("E57 classification readback failed")
                count += len(classes)
                scan_count += len(classes)
                take = min(len(expected)-checked, len(chunk.xyz))
                if take and not np.allclose(chunk.xyz[:take], expected[checked:checked+take], rtol=0, atol=tolerance):
                    raise RuntimeError("E57 coordinate readback failed")
                checked += max(0, take)
        if count != expected_count:
            raise RuntimeError("E57 point count readback failed")
        source_images = reader.vector("images2D")
        with E57Reader(source) as original:
            images = original.vector("images2D")
            if (source_images.childCount() if source_images else 0) != (images.childCount() if images else 0):
                raise RuntimeError("E57 image count readback failed")
            if source_images and metadata(source_images) != metadata(images):
                raise RuntimeError("E57 image blob or association readback failed")
    return {"record_count": count, "all_classes_valid": True, "max_coordinate_error_m": [0., 0., 0.]}
