"""LAS 1.4 streaming output with preserved RGB and explicit scan membership."""

from itertools import product
from pathlib import Path
import time

import laspy
from laspy.vlrs.known import WktCoordinateSystemVlr
import numpy as np

from e57cls.geometry.cameras import scan_pose
from e57cls.output.classify import ClassPointIndex as SharedClassPointIndex, ClassifiedSource, output_chunks, selected_count
from e57cls.vote.engine import voxel_keys


def make_header(stores, crs_wkt="", scale=.001, station_ids=False, station_count=0):
    if len(stores) > 65535:
        raise ValueError("LAS Point Source ID supports at most 65535 scans")
    rgb = any(all(k in s.meta["arrays"] for k in ("colorRed", "colorGreen", "colorBlue")) for s in stores)
    header = laspy.LasHeader(point_format=7 if rgb else 6, version="1.4")
    header.add_extra_dims([laspy.ExtraBytesParams(name="class_conf", type=np.uint8, description="Class winning share 0..255"),
                           laspy.ExtraBytesParams(name="mirror_state", type=np.uint8, description="Mirror evidence state 0..4")])
    if station_ids:
        maximum = max(1, int(station_count))
        station_type = np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
        header.add_extra_dim(laspy.ExtraBytesParams(name="station_id", type=station_type,
                                                     description="Strongest SLAM station"))
    corners = []
    for store in stores:
        bounds = store.meta["bounds_local"]
        if bounds:
            box = np.array(list(product(*zip(bounds["min"], bounds["max"]))))
            corners.append(scan_pose(store.meta["metadata"]).to_world(box))
    if corners:
        all_corners = np.concatenate(corners)
        low, high = all_corners.min(axis=0), all_corners.max(axis=0)
        header.offsets = (low+high)/2
        if ((high-low)/scale > 2**32-2).any():
            raise ValueError("Coordinate extent cannot be represented at the requested LAS precision")
    header.scales = np.full(3, scale)
    header.generating_software = "e57cls"
    if crs_wkt:
        header.vlrs.append(WktCoordinateSystemVlr(crs_wkt))
        header.global_encoding.wkt = True
    return header


def scaled_attribute(values, limits, low_key, high_key):
    if low_key not in limits or high_key not in limits or limits[high_key] <= limits[low_key]:
        raise ValueError(f"Missing or invalid attribute limits: {low_key}, {high_key}")
    values = np.nan_to_num(values, nan=limits[low_key])
    return np.rint(np.clip((values-limits[low_key])/(limits[high_key]-limits[low_key]), 0, 1)*65535).astype(np.uint16)


# Backwards-compatible public import; both writers use the shared implementation.
ClassPointIndex = SharedClassPointIndex


def write_cloud(path: Path, stores, votes, origin, voxel_size, crs_wkt, allowed_codes,
                skipped=(), checkpoint=lambda: None, progress=lambda *args: None,
                mirror=None, no_reflection=False, primary_without_reflections=False,
                class_transfer_radius=0., output_density="reduced", geometry_planes=None,
                geometry_fill_tolerance=.02, terrain=None, terrain_ground_tolerance=.15, terrain_fill=False,
                station_ids=False, station_count=0, force_unclassified=None, terrain_outdoor_scans=None,
                class_transfer_range_factor=0., class_transfer_radius_max=None, transfer_station_positions=None):
    header = make_header(stores, crs_wkt, station_ids=station_ids, station_count=station_count)
    temporary = path.with_name(path.stem+".partial"+path.suffix)
    extra = path.with_name(path.stem+"_no_reflection"+path.suffix)
    extra_temp = extra.with_name(extra.stem+".partial"+extra.suffix)
    counts, written, discarded, inherited_count = np.zeros(256, np.int64), 0, 0, 0
    mirrors = np.zeros(5, np.int64)
    warnings = []
    timing = {'write': 0., 'readback': 0.}
    sample_indices, sample_xyz, sample_scan = [], [], []
    rng = np.random.default_rng(42)
    total = sum(selected_count(store, output_density) for store in stores)
    wanted = np.sort(rng.choice(total, size=min(total, 100_000), replace=False)) if total else np.array([], int)
    laz_backend = (getattr(laspy.LazBackend, "LazrsParallel", laspy.LazBackend.Lazrs)
                   if path.suffix.lower() == ".laz" else None)
    writer_options = {"do_compress": path.suffix.lower()==".laz"}
    if laz_backend is not None:
        writer_options["laz_backend"] = laz_backend
    extra_writer = laspy.open(extra_temp, mode="w", header=header, **writer_options) if no_reflection else None
    try:
        with ClassifiedSource(path.parent, stores, votes, origin, voxel_size, skipped, mirror, class_transfer_radius, checkpoint,
                              geometry_planes, geometry_fill_tolerance, terrain, terrain_ground_tolerance, terrain_fill,
                              progress, force_unclassified, terrain_outdoor_scans, class_transfer_range_factor,
                              class_transfer_radius_max, transfer_station_positions) as classified, laspy.open(temporary, mode="w", header=header, **writer_options) as writer:
            for store in stores:
                scan = store.meta["scan_index"]
                meta = store.meta["metadata"]
                pose = scan_pose(meta)
                for start, xyz, valid in output_chunks(store, output_density):
                    checkpoint()
                    stop = start+len(xyz)
                    world = pose.to_world(xyz[valid])
                    n = len(world)
                    if not n:
                        continue
                    world, codes, confidence, states = classified.block(store, start, xyz, valid, output_density, world)
                    if not np.isin(codes, allowed_codes).all() or (codes == 0).any():
                        raise ValueError("Output class missing from legend or forbidden code 0")
                    points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
                    points.x, points.y, points.z = world.T
                    points.classification = codes
                    points.class_conf = confidence
                    points.mirror_state = states
                    points.point_source_id[:] = scan
                    if station_ids:
                        station = votes.lookup_station_ids(voxel_keys(world, origin, voxel_size))
                        if scan in skipped:
                            station[:] = 0
                        if force_unclassified is not None:
                            station[force_unclassified(store, start, valid)] = 0
                        points.station_id = station
                    # Unknown return structure is represented by zero, documented in the report.
                    if "intensity" in store.meta["arrays"]:
                        try:
                            intensity = store.read("intensity", start, stop)[valid]
                            points.intensity = scaled_attribute(intensity, meta.get("intensityLimits", {}), "intensityMinimum", "intensityMaximum")
                            if "isIntensityInvalid" in store.meta["arrays"]:
                                points.intensity[store.read("isIntensityInvalid", start, stop)[valid] != 0] = 0
                        except ValueError as error:
                            if str(error) not in warnings:
                                warnings.append(str(error))
                    if header.point_format.id == 7:
                        for color, field in (("Red", "red"), ("Green", "green"), ("Blue", "blue")):
                            key = "color"+color
                            if key in store.meta["arrays"]:
                                values = scaled_attribute(store.read(key, start, stop)[valid], meta.get("colorLimits", {}), key+"Minimum", key+"Maximum")
                                if "isColorInvalid" in store.meta["arrays"]:
                                    values[store.read("isColorInvalid", start, stop)[valid] != 0] = 0
                                setattr(points, field, values)
                    # A reflection-only run produces a clean cloud as its main
                    # artifact.  The standard classify-and-filter workflow keeps
                    # its complete cloud and may additionally write this view.
                    if primary_without_reflections:
                        started = time.perf_counter()
                        writer.write_points(points[codes != 68])
                        timing['write'] += time.perf_counter()-started
                        discarded += int((codes == 68).sum())
                    else:
                        started = time.perf_counter()
                        writer.write_points(points)
                        timing['write'] += time.perf_counter()-started
                    if extra_writer:
                        started = time.perf_counter()
                        extra_writer.write_points(points[codes != 68])
                        timing['write'] += time.perf_counter()-started
                        if not primary_without_reflections:
                            discarded += int((codes == 68).sum())
                    left, right = np.searchsorted(wanted, [written, written+n])
                    ids = wanted[left:right]-written
                    sample_indices.extend(wanted[left:right].tolist())
                    sample_xyz.extend(world[ids].tolist())
                    sample_scan.extend([scan]*len(ids))
                    written += n
                    counts += np.bincount(codes, minlength=256)
                    mirrors += np.bincount(states, minlength=5)
                    progress("output", written/max(1,total), scan)
        inherited_count = classified.class_transfer_count
        if extra_writer:
            extra_writer.close()
            extra_writer = None
        if written != total:
            raise RuntimeError("LAS output count does not equal valid E57 count")
        expected_output_count = written-discarded if primary_without_reflections else written
        if primary_without_reflections:
            # Sampling coordinates by original output ordinal is not meaningful
            # after reflection points are omitted; still verify the whole file's
            # classes and point count.
            sample_indices, sample_xyz, sample_scan = [], [], []
        started = time.perf_counter()
        verification = verify_readback(temporary, expected_output_count, sample_indices, sample_xyz, sample_scan, allowed_codes,
                                       laz_backend=laz_backend)
        timing['readback'] += time.perf_counter()-started
        temporary.replace(path)
        if no_reflection:
            extra_temp.replace(extra)
        output_counts = counts.copy()
        if primary_without_reflections:
            output_counts[68] = 0
        return {"path": str(path), "record_count": expected_output_count,
                "class_counts": {str(i): int(n) for i,n in enumerate(output_counts) if n},
                "unclassified_share": int(output_counts[1])/max(1,expected_output_count), "mirror_counts": mirrors.tolist(),
                "extra_output": str(extra) if no_reflection else None, "extra_removed_count": discarded,
                "class_transfer_count": inherited_count,
                "geometry_fill_counts": classified.geometry_fill_counts,
                "readback": verification, "warnings": warnings, "las_scale_m": header.scales.tolist(),
                "timing_seconds": {**classified.timing_seconds, **timing}}
    finally:
        if extra_writer:
            extra_writer.close()


def verify_readback(path, expected_count, indices, xyz, scans, allowed_codes, laz_backend=None):
    indices, xyz, scans = np.asarray(indices), np.asarray(xyz), np.asarray(scans)
    max_error = np.zeros(3)
    allowed = np.zeros(256, bool)
    allowed_codes = np.asarray(allowed_codes)
    valid_codes = allowed_codes[(allowed_codes >= 0) & (allowed_codes < 256)]
    allowed[valid_codes.astype(np.uint8)] = True
    offset = 0
    if laz_backend is None and Path(path).suffix.lower() == '.laz':
        laz_backend = getattr(laspy.LazBackend, 'LazrsParallel', laspy.LazBackend.Lazrs)
    options = {"laz_backend": laz_backend} if laz_backend is not None else {}
    with laspy.open(path, **options) as reader:
        if reader.header.point_count != expected_count or reader.header.point_format.id not in (6, 7):
            raise RuntimeError("LAS header readback failed")
        for block in reader.chunk_iterator(500_000):
            classification = np.asarray(block.classification)
            if not allowed[classification].all() or (classification == 0).any():
                raise RuntimeError("LAS classification readback failed")
            left, right = np.searchsorted(indices, [offset, offset+len(block)])
            local = indices[left:right]-offset
            if len(local):
                actual = np.column_stack((block.x, block.y, block.z))[local]
                max_error = np.maximum(max_error, np.abs(actual-xyz[left:right]).max(axis=0))
                if not np.array_equal(np.asarray(block.point_source_id)[local], scans[left:right]):
                    raise RuntimeError("LAS scan membership readback failed")
            offset += len(block)
        if np.any(max_error > reader.header.scales/2 + 1e-9):
            raise RuntimeError("LAS coordinates exceed half a scale unit")
    return {"sample_count": len(indices), "max_coordinate_error_m": max_error.tolist(), "all_classes_valid": True,
            "point_source_ids_match": True, "point_format": reader.header.point_format.id}
