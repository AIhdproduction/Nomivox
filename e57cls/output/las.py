"""LAS 1.4 streaming output with preserved RGB and explicit scan membership."""

from itertools import product
from pathlib import Path

import laspy
from laspy.vlrs.known import WktCoordinateSystemVlr
import numpy as np

from e57cls.geometry.cameras import Pose
from e57cls.vote.engine import voxel_keys


def make_header(stores, crs_wkt="", scale=.001):
    if len(stores) > 65535:
        raise ValueError("LAS Point Source ID supports at most 65535 scans")
    rgb = any(all(k in s.meta["arrays"] for k in ("colorRed", "colorGreen", "colorBlue")) for s in stores)
    header = laspy.LasHeader(point_format=7 if rgb else 6, version="1.4")
    header.add_extra_dims([laspy.ExtraBytesParams(name="class_conf", type=np.uint8, description="Class winning share 0..255"),
                           laspy.ExtraBytesParams(name="mirror_state", type=np.uint8, description="Mirror evidence state 0..4")])
    corners = []
    for store in stores:
        bounds = store.meta["bounds_local"]
        if bounds:
            box = np.array(list(product(*zip(bounds["min"], bounds["max"]))))
            corners.append(Pose.from_dict(store.meta["metadata"]["pose"]).to_world(box))
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


def write_cloud(path: Path, stores, votes, origin, voxel_size, crs_wkt, allowed_codes,
                skipped=(), checkpoint=lambda: None, progress=lambda *args: None,
                mirror=None, no_reflection=False):
    header = make_header(stores, crs_wkt)
    temporary = path.with_name(path.stem+".partial"+path.suffix)
    extra = path.with_name(path.stem+"_no_reflection"+path.suffix)
    extra_temp = extra.with_name(extra.stem+".partial"+extra.suffix)
    counts, written, discarded = np.zeros(256, np.int64), 0, 0
    mirrors = np.zeros(5, np.int64)
    warnings = []
    sample_indices, sample_xyz, sample_scan = [], [], []
    rng = np.random.default_rng(42)
    total = sum(s.meta["valid_count"] for s in stores)
    wanted = np.sort(rng.choice(total, size=min(total, 100_000), replace=False)) if total else np.array([], int)
    extra_writer = laspy.open(extra_temp, mode="w", header=header, do_compress=path.suffix==".laz") if no_reflection else None
    try:
        with laspy.open(temporary, mode="w", header=header, do_compress=path.suffix==".laz") as writer:
            for store in stores:
                scan = store.meta["scan_index"]
                meta = store.meta["metadata"]
                pose = Pose.from_dict(meta["pose"])
                for start, xyz, valid in store.chunks():
                    checkpoint()
                    stop = start+len(xyz)
                    world = pose.to_world(xyz[valid])
                    n = len(world)
                    if not n:
                        continue
                    keys = voxel_keys(world, origin, voxel_size)
                    codes, confidence = votes.lookup(keys)
                    if scan in skipped:
                        codes[:] = 1
                        confidence[:] = 0
                    states = np.zeros(n, np.uint8)
                    if mirror:
                        codes, states = mirror(store, start, stop, valid, world, codes)
                    if not np.isin(codes, allowed_codes).all() or (codes == 0).any():
                        raise ValueError("Output class missing from legend or forbidden code 0")
                    points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
                    points.x, points.y, points.z = world.T
                    points.classification = codes
                    points.class_conf = confidence
                    points.mirror_state = states
                    points.point_source_id[:] = scan
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
                    writer.write_points(points)
                    if extra_writer:
                        extra_writer.write_points(points[codes != 68])
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
        if extra_writer:
            extra_writer.close()
            extra_writer = None
        if written != total:
            raise RuntimeError("LAS output count does not equal valid E57 count")
        verification = verify_readback(temporary, written, sample_indices, sample_xyz, sample_scan, allowed_codes)
        temporary.replace(path)
        if no_reflection:
            extra_temp.replace(extra)
        return {"path": str(path), "record_count": written, "class_counts": {str(i): int(n) for i,n in enumerate(counts) if n},
                "unclassified_share": int(counts[1])/max(1,written), "mirror_counts": mirrors.tolist(),
                "extra_output": str(extra) if no_reflection else None, "extra_removed_count": discarded,
                "readback": verification, "warnings": warnings, "las_scale_m": header.scales.tolist()}
    finally:
        if extra_writer:
            extra_writer.close()


def verify_readback(path, expected_count, indices, xyz, scans, allowed_codes):
    indices, xyz, scans = np.asarray(indices), np.asarray(xyz), np.asarray(scans)
    max_error = np.zeros(3)
    offset = 0
    with laspy.open(path) as reader:
        if reader.header.point_count != expected_count or reader.header.point_format.id not in (6, 7):
            raise RuntimeError("LAS header readback failed")
        for block in reader.chunk_iterator(100_000):
            if not np.isin(np.asarray(block.classification), allowed_codes).all() or (np.asarray(block.classification) == 0).any():
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
