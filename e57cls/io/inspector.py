"""Bounded inspection of E57 metadata, records and encoded image headers."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import math
import logging
import platform
import time

import numpy as np
from PIL import ImageFile

from e57cls.files import atomic_json, atomic_text, new_output
from e57cls.geometry.cameras import scan_pose
from e57cls.io.reader import E57Reader, metadata
from e57cls.io.input_mode import detect_input_mode

REPRESENTATIONS = ("sphericalRepresentation", "pinholeRepresentation",
                   "cylindricalRepresentation", "visualReferenceRepresentation")
INSPECTION_SCHEMA = 2
LOG = logging.getLogger(__name__)


def pose_status(pose: dict | None) -> str:
    if not pose:
        return "missing"
    try:
        q = np.array([pose["rotation"][k] for k in ("w", "x", "y", "z")], np.float64)
        t = np.array([pose["translation"][k] for k in ("x", "y", "z")], np.float64)
    except (KeyError, TypeError, ValueError):
        return "invalid"
    return "valid" if np.isfinite(t).all() and np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-6 else "invalid"


def image_header(blob, limit: int = 1024 * 1024) -> dict:
    parser = ImageFile.Parser()
    read = 0
    while read < min(blob.byteCount(), limit):
        count = min(4096, blob.byteCount() - read, limit - read)
        buffer = np.empty(count, np.uint8)
        blob.read(buffer, read, count)
        parser.feed(buffer.tobytes())
        read += count
        if parser.image:
            result = {"format": parser.image.format, "width": parser.image.width,
                      "height": parser.image.height, "mode": parser.image.mode,
                      "header_bytes_read": read, "byte_count": blob.byteCount()}
            parser.image.close()
            return result
    raise ValueError(f"Image header unavailable within {limit} bytes")


def image_decodable(blob) -> None:
    """Fully decode an embedded image without retaining its compressed bytes."""
    parser = ImageFile.Parser()
    offset = 0
    while offset < blob.byteCount():
        count = min(64 * 1024, blob.byteCount()-offset)
        buffer = np.empty(count, np.uint8)
        blob.read(buffer, offset, count)
        parser.feed(buffer.tobytes())
        offset += count
    image = parser.close()
    try:
        image.load()
    finally:
        image.close()


def sample_summary(chunk) -> dict:
    xyz = chunk.xyz[chunk.valid]
    result = {"records_read": len(chunk.xyz), "valid_records": int(chunk.valid.sum()),
              "invalid_records": int((~chunk.valid).sum()), "coverage": "first_contiguous_records",
              "whole_scan_invalid_count": None, "density_estimate": None}
    if not len(xyz):
        return result
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    volume = float(np.prod(high - low))
    result.update({"bounds_local": {"min": low.tolist(), "max": high.tolist()},
                   "max_absolute_coordinate": float(np.abs(xyz).max()),
                   "median_range_m": float(np.median(np.linalg.norm(xyz, axis=1))),
                   "zero_range_records": int((np.linalg.norm(xyz, axis=1) == 0).sum()),
                   "density_estimate": {"points_per_bbox_m3": len(xyz) / volume if volume > 0 else None,
                                        "method": "valid sample count / sample bounding box volume",
                                        "limitation": "Contiguous sample; not surface density or a whole-scan estimate"}})
    result["field_ranges"] = {}
    for name, values in chunk.fields.items():
        finite = values[np.isfinite(values)]
        result["field_ranges"][name] = {"min": float(finite.min()) if len(finite) else None,
                                       "max": float(finite.max()) if len(finite) else None}
    return result


def inspect_file(source: Path, sample_points: int = 100_000, config: dict | None = None) -> dict:
    if not 0 <= sample_points <= 1_000_000:
        raise ValueError("Sample size must be between 0 and 1000000")
    source = source.resolve(strict=True)
    before = source.stat()
    started = time.monotonic()
    with E57Reader(source) as reader:
        root = reader.root
        file_meta = {root[i].elementName(): metadata(root[i]) for i in range(root.childCount())
                     if root[i].elementName() not in ("data3D", "images2D")}
        scans, images, issues = [], [], []
        data3d = reader.vector("data3D")
        for index in range(data3d.childCount() if data3d else 0):
            item = metadata(data3d[index])
            points = item.pop("points", {"record_count": 0, "fields": {}})
            scan = {"index": index + 1, "metadata": item, **points,
                    "pose_status": pose_status(item.get("pose")), "image_indices": [],
                    "unit": "metre", "sample": None, "classification_eligible_metadata": False}
            if sample_points == 0:
                # The GUI preview needs metadata only.  A non-null marker keeps
                # eligibility checks equivalent to an empty point sample.
                scan["sample"] = {"records_read": 0}
            else:
                try:
                    generator = reader.chunks(index, sample_points, sample_points)
                    try:
                        chunk = next(generator, None)
                        scan["sample"] = sample_summary(chunk) if chunk is not None else {"records_read": 0}
                    finally:
                        generator.close()
                except (RuntimeError, ValueError) as error:
                    issues.append({"code": "point_sample_failed", "scan": index + 1, "detail": str(error)})
            scans.append(scan)
        guid_map = {}
        for scan in scans:
            guid_map.setdefault(scan["metadata"].get("guid"), []).append(scan)
        image_nodes = reader.vector("images2D")
        for index in range(image_nodes.childCount() if image_nodes else 0):
            item = metadata(image_nodes[index])
            reps = [name for name in REPRESENTATIONS if name in item]
            image = {"index": index + 1, "metadata": item, "representations": reps,
                     "pose_status": pose_status(item.get("pose")), "blob_headers": {},
                     "association_status": "uncertain", "scan_index": None,
                     "classification_eligible_slam": False}
            for rep in reps:
                image["blob_headers"][rep] = {}
                for name in ("jpegImage", "pngImage", "imageMask"):
                    if name in item[rep]:
                        try:
                            header = image_header(reader.blob(index, rep, name))
                            header["matches_metadata"] = (header["width"] == item[rep].get("imageWidth")
                                                          and header["height"] == item[rep].get("imageHeight"))
                            image["blob_headers"][rep][name] = header
                        except (RuntimeError, ValueError, OSError) as error:
                            issues.append({"code": "image_header_failed", "image": index + 1, "detail": str(error)})
            matches = guid_map.get(item.get("associatedData3DGuid"), []) if item.get("associatedData3DGuid") else []
            if len(matches) == 1:
                scan = matches[0]
                image.update(association_status="explicit_guid", scan_index=scan["index"])
                scan["image_indices"].append(index + 1)
                if pose_status(item.get("pose")) == "valid" and scan["pose_status"] == "valid":
                    a = scan_pose(scan["metadata"]).translation
                    b = item["pose"]["translation"]
                    image["camera_scanner_offset_m"] = math.sqrt(sum((a[index] - b[key]) ** 2 for index, key in enumerate("xyz")))
                    usable = []
                    required = {"sphericalRepresentation": ("pixelWidth", "pixelHeight"),
                                "pinholeRepresentation": ("pixelWidth", "pixelHeight", "focalLength", "principalPointX", "principalPointY"),
                                "cylindricalRepresentation": ("pixelWidth", "pixelHeight", "radius", "principalPointY")}
                    for rep in reps:
                        params = item[rep]
                        headers = image["blob_headers"][rep]
                        if rep in required and all(k in params for k in required[rep]):
                            if all(params.get(k, 0) > 0 for k in ("imageWidth", "imageHeight", "pixelWidth", "pixelHeight")):
                                usable.extend(h for n, h in headers.items() if n != "imageMask" and h["matches_metadata"])
                    scan["classification_eligible_metadata"] |= bool(usable) and scan["sample"] is not None
            else:
                issues.append({"code": "image_association_uncertain", "image": index + 1})
            required = {"sphericalRepresentation": ("pixelWidth", "pixelHeight"),
                        "pinholeRepresentation": ("pixelWidth", "pixelHeight", "focalLength", "principalPointX", "principalPointY"),
                        "cylindricalRepresentation": ("pixelWidth", "pixelHeight", "radius", "principalPointY")}
            if image["pose_status"] == "valid":
                for rep in reps:
                    params, headers = item[rep], image["blob_headers"][rep]
                    if rep in required and all(key in params for key in required[rep]):
                        dimensions = all(params.get(key, 0) > 0 for key in ("imageWidth", "imageHeight", "pixelWidth", "pixelHeight"))
                        image["classification_eligible_slam"] |= dimensions and any(
                            header["matches_metadata"] for name, header in headers.items() if name != "imageMask")
            if not image["classification_eligible_slam"]:
                issues.append({"code": "image_not_classifiable_slam", "image": index + 1})
            images.append(image)
        for scan in scans:
            if not scan["classification_eligible_metadata"]:
                issues.append({"code": "scan_not_classifiable", "scan": scan["index"]})
        detected = detect_input_mode({"scans": scans, "images": images}, config or {})
        if detected["mode"] == "slam":
            for image in images:
                if not image["classification_eligible_slam"]:
                    continue
                try:
                    representation = next(rep for rep in image["representations"]
                                          if rep != "visualReferenceRepresentation"
                                          and any(name in image["metadata"][rep] for name in ("jpegImage", "pngImage")))
                    name = "jpegImage" if "jpegImage" in image["metadata"][representation] else "pngImage"
                    image_decodable(reader.blob(image["index"]-1, representation, name))
                    image["decode_status"] = "valid"
                except (RuntimeError, ValueError, OSError, StopIteration) as error:
                    image["classification_eligible_slam"] = False
                    image["decode_status"] = "invalid"
                    issues.append({"code": "image_decode_failed", "image": image["index"], "detail": str(error)})
                    LOG.warning("Skipping SLAM image %d because it cannot be decoded: %s", image["index"], error)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during inspection")
    report = {"schema_version": INSPECTION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(source), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
            "versions": {"python": platform.python_version(), **{k: version(k) for k in ("pye57", "numpy", "Pillow")}},
            "metadata": file_meta, "scan_count": len(scans), "image_count": len(images),
            "record_count": sum(s["record_count"] for s in scans),
            "data3d_layout": "separate_sections" if len(scans) > 1 else "single_section_provenance_unclear",
            "scans": scans, "images": images, "issues": issues,
            "sample_limit_per_scan": sample_points, "elapsed_seconds": time.monotonic() - started,
            "input_stat_unchanged": True, "projection_alignment_verified": False}
    report["input_mode"] = detected
    return report


def markdown(report: dict) -> str:
    lines = ["# E57 inspection", "", f"Source: `{report['source']['path']}`", "",
             f"Scans: {report['scan_count']} | Images: {report['image_count']} | Records: {report['record_count']:,}", "",
             "This bounded inspection does not establish full-file integrity, total invalid counts or image alignment.", "",
             "## Scans", "", "| Index | Name | Records | Images | Pose | Sample invalid |", "|---|---|---:|---|---|---:|"]
    for scan in report["scans"]:
        lines.append(f"| {scan['index']} | {scan['metadata'].get('name', '').replace('|', '/')} | {scan['record_count']:,} | {scan['image_indices']} | {scan['pose_status']} | {(scan['sample'] or {}).get('invalid_records', 'unknown')} |")
    lines.extend(["", "## Images", "", "| Index | Representation | Dimensions | Association | Offset (m) |", "|---|---|---|---|---:|"])
    for image in report["images"]:
        for rep in image["representations"]:
            params = image["metadata"][rep]
            lines.append(f"| {image['index']} | {rep} | {params.get('imageWidth')} x {params.get('imageHeight')} | {image['association_status']} | {image.get('camera_scanner_offset_m', 'unknown')} |")
    input_mode = report.get("input_mode")
    if input_mode:
        lines.extend(["", "## Input mode", "", f"Selected: {input_mode['mode']} (requested: {input_mode['requested']})", ""])
        lines.extend(f"- Rule {rule['id']}: {'hit' if rule['hit'] else 'no hit'}; {rule['detail']}" for rule in input_mode['rules'])
    lines.extend(["", "## Coordinate metadata", "", "```text", report["metadata"].get("coordinateMetadata", "Absent"), "```", "", "## Issues", ""])
    lines.extend(f"- {issue}" for issue in report["issues"])
    if not report["issues"]:
        lines.append("No metadata eligibility errors. Projection and visual review are still required.")
    lines.extend(["", "Full poses, field prototypes and scales, bounds, sample statistics and image headers are in inspect_report.json.", ""])
    return "\n".join(lines)


def write_inspection(source: Path, output: Path, sample_points: int) -> dict:
    output = new_output(output, source)
    report = inspect_file(source, sample_points)
    atomic_json(output / "inspect_report.json", report)
    atomic_text(output / "inspect_report.md", markdown(report))
    return report
