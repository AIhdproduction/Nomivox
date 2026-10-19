"""Independent probes and reviewable, camera-aware depth/image diagnostics."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from e57cls.files import atomic_json, atomic_text, file_hash, new_output
from e57cls.geometry.cameras import Camera, Pose, plan_views, scan_pose, scan_to_camera
from e57cls.geometry.images import extract_blob, open_image, render_view
from e57cls.geometry.reference import project_point
from e57cls.geometry.visibility import DepthBuffer
from e57cls.io.reader import E57Reader
from e57cls.io.store import load_scan


def edge_alignment(image, depth):
    gray = image.astype(np.float64).mean(axis=2)/255
    image_edge = np.zeros(gray.shape)
    image_edge[:, 1:] = np.maximum(image_edge[:, 1:], np.abs(np.diff(gray, axis=1)))
    image_edge[1:, :] = np.maximum(image_edge[1:, :], np.abs(np.diff(gray, axis=0)))
    depth_edge = np.zeros(gray.shape, bool)
    for axis in (0, 1):
        a = depth[:-1, :] if axis == 0 else depth[:, :-1]
        b = depth[1:, :] if axis == 0 else depth[:, 1:]
        finite = np.isfinite(a) & np.isfinite(b)
        difference = np.zeros(a.shape)
        np.subtract(a, b, out=difference, where=finite)
        edge = finite & (np.abs(difference) > np.maximum(.05, .02*np.minimum(a, b)))
        if axis == 0:
            depth_edge[1:, :] |= edge
        else:
            depth_edge[:, 1:] |= edge
    candidates = int(depth_edge.sum())
    edges = image_edge > .08
    near = edges.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near[max(0,dy):near.shape[0]+min(0,dy), max(0,dx):near.shape[1]+min(0,dx)] |= edges[max(0,-dy):near.shape[0]-max(0,dy), max(0,-dx):near.shape[1]-max(0,dx)]
    return {"score": float(near[depth_edge].mean()) if candidates else None,
            "depth_edge_pixels": candidates, "image_edge_fraction": float(near.mean()),
            "definition": "fraction of depth discontinuities within one pixel of an image gradient > 0.08",
            "thresholds_unvalidated": True}, depth_edge


def diagnose(source: Path, inspection: Path, store_report: Path, output: Path,
             cache: Path, view_size: int = 768, diagnostic_width: int = 1600) -> dict:
    if not 64 <= view_size <= 1512 or not 128 <= diagnostic_width <= 4096:
        raise ValueError("View size must be 64..1512; diagnostic width must be 128..4096")
    output = new_output(output, source)
    report = json.loads(inspection.read_text(encoding="utf-8"))
    stores = json.loads(store_report.read_text(encoding="utf-8"))
    fingerprint = file_hash(source)
    stat = source.stat()
    if fingerprint != stores["source_sha256"] or (stat.st_size, stat.st_mtime_ns) != (report["source"]["bytes"], report["source"]["mtime_ns"]):
        raise ValueError("Source does not match the inspection and scan store")
    image_cache = cache.resolve() / fingerprint / "images"
    started = time.monotonic()
    results = []
    with E57Reader(source) as reader:
        for item in report["images"]:
            index = item["index"]
            print(f"Geometry image {index}/{len(report['images'])}", flush=True)
            scan_index = item["scan_index"]
            if scan_index is None:
                results.append({"image_index": index, "passed": False, "reason": "association_uncertain"})
                continue
            try:
                scan = report["scans"][scan_index-1]
                scan_transform = scan_pose(scan["metadata"])
                camera_pose = Pose.from_dict(item["metadata"]["pose"])
                kind = next(r for r in item["representations"] if r != "visualReferenceRepresentation")
                params = item["metadata"][kind]
                camera = Camera(kind, params)
                store = load_scan(Path(stores["scans"][scan_index-1]["path"]), fingerprint)
                width = min(diagnostic_width, camera.width)
                height = max(1, round(width*camera.height/camera.width))
                buffer = DepthBuffer(width, height, radius=1, wrap=camera.wraps)
                samples = []
                inside = behind = total = 0
                for _, xyz, valid in store.chunks():
                    xyz = xyz[valid]
                    transformed = scan_to_camera(xyz, scan_transform, camera_pose)
                    uv, depth, projected = camera.project(transformed)
                    buffer.add(uv*np.array([width/camera.width, height/camera.height]), depth, projected)
                    total += len(xyz)
                    inside += int(projected.sum())
                    behind += int((depth <= 0).sum())
                    if len(xyz):
                        samples.append(xyz[np.linspace(0, len(xyz)-1, min(8, len(xyz)), dtype=int)])
                candidates = np.concatenate(samples) if samples else np.empty((0, 3))
                if not len(candidates):
                    raise ValueError("No valid points available for projection probes")
                points = candidates[np.linspace(0, len(candidates)-1, min(200, len(candidates)), dtype=int)]
                cam_points = scan_to_camera(points, scan_transform, camera_pose)
                production, probe_depth, _ = camera.project(cam_points)
                reference_scan_pose = scan["metadata"].get("pose") or {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                                                                         "translation": {"x": 0., "y": 0., "z": 0.}}
                reference = np.array([project_point(p, reference_scan_pose, item["metadata"]["pose"], kind, params) for p in points])
                delta = np.abs(production-reference)
                if camera.wraps:
                    delta[:, 0] = np.minimum(delta[:, 0], camera.width-delta[:, 0])
                errors = np.linalg.norm(delta, axis=1)
                threshold = .5 if kind != "pinholeRepresentation" else 1.
                passed = bool(np.isfinite(errors).all() and errors.max() <= threshold)
                destination = output / f"image_{index:04d}"
                destination.mkdir()
                local_cache = image_cache / f"image_{index:04d}"
                name = "jpegImage" if "jpegImage" in params else "pngImage"
                image_path = local_cache / ("image.jpg" if name == "jpegImage" else "image.png")
                extract_blob(reader.blob(index-1, kind, name), image_path)
                image = open_image(image_path, "RGB")
                mask = None
                if "imageMask" in params:
                    mask_path = local_cache / "mask.png"
                    extract_blob(reader.blob(index-1, kind, "imageMask"), mask_path)
                    mask = open_image(mask_path, "L")
                if image.shape[:2] != (camera.height, camera.width) or (mask is not None and mask.shape != image.shape[:2]):
                    raise ValueError("Decoded image or mask dimensions differ from camera metadata")
                preview = np.array(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))
                score, depth_edges = edge_alignment(preview, buffer.depth)
                overlay = preview.copy()
                overlay[depth_edges] = [0, 255, 255]
                Image.fromarray(preview).save(destination / "panorama.jpg", quality=90)
                Image.fromarray(overlay).save(destination / "depth_edges.jpg", quality=95)
                finite = np.isfinite(buffer.depth)
                depth_rgb = np.zeros_like(preview)
                if finite.any():
                    values = buffer.depth[finite]
                    lo, hi = np.percentile(values, [2, 98])
                    fraction = np.clip((values-lo)/max(hi-lo, 1e-8), 0, 1)
                    depth_rgb[finite] = np.column_stack((255*(1-fraction), 100*np.ones(len(fraction)), 255*fraction)).astype(np.uint8)
                blend = preview.copy()
                blend[finite] = (.6*preview[finite]+.4*depth_rgb[finite]).astype(np.uint8)
                Image.fromarray(blend).save(destination / "depth_overlay.jpg", quality=95)
                if mask is not None:
                    Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST).save(destination / "image_mask.png")
                views = plan_views(camera, cap=view_size)
                atomic_json(destination / "view_plan.json", [asdict(v) for v in views])
                rendered_names = []
                for view in ([views[i] for i in (0, 8, 16)] if len(views) >= 17 else views[:3]):
                    rendered, valid_view = render_view(image, mask, camera, view)
                    Image.fromarray(rendered).save(destination / f"{view.id}.jpg", quality=95)
                    Image.fromarray(valid_view.astype(np.uint8)*255).save(destination / f"{view.id}_valid.png")
                    rendered_names.append(f"{view.id}.jpg")
                atomic_json(local_cache / "image_meta.json", {"metadata": item, "image_sha256": file_hash(image_path)})
                results.append({"image_index": index, "scan_index": scan_index, "representation": kind,
                                "passed": passed, "probe_count": len(points), "max_error_px": float(errors.max()),
                                "probe_threshold_px": threshold, "projected_fraction": inside/max(1, total),
                                "points_behind_camera": behind, "median_probe_depth_m": float(np.median(probe_depth)),
                                "alignment": score, "alignment_warning": score["score"] is None or score["score"] < .2,
                                "diagnostics": str(destination), "views": rendered_names,
                                "image_mask_valid_fraction": float((mask > 0).mean()) if mask is not None else None,
                                "owner_review": "pending"})
                del image, mask, buffer
            except (ValueError, KeyError, RuntimeError, OSError, StopIteration) as error:
                results.append({"image_index": index, "scan_index": scan_index, "passed": False, "reason": str(error)})
    result = {"schema_version": 1, "source_sha256": fingerprint, "images": results,
              "elapsed_seconds": time.monotonic()-started,
              "all_probes_passed": bool(results) and all(r["passed"] for r in results),
              "owner_review": "pending", "sam_calls": 0}
    atomic_json(output / "geometry_report.json", result)
    write_review(output, result)
    return result


def write_review(output: Path, result: dict):
    import html
    strings = json.loads((Path(__file__).resolve().parents[1] / "gui/strings_de.json").read_text(encoding="utf-8"))
    blocks = []
    for item in result["images"]:
        title = strings["geometry_image"].format(index=item["image_index"])
        if "diagnostics" not in item:
            blocks.append(f"<section><h2>{title}</h2><p>{strings['geometry_failed']}</p></section>")
            continue
        relative = f"image_{item['image_index']:04d}"
        figures = [("panorama.jpg", "geometry_original"), ("depth_edges.jpg", "geometry_edges"),
                   ("depth_overlay.jpg", "geometry_depth"), ("image_mask.png", "geometry_mask")]
        figures.extend((name, "geometry_view") for name in item["views"])
        figures_html = "".join(f'<figure><figcaption>{html.escape(strings[label])}</figcaption><img loading="lazy" src="{relative}/{name}"></figure>' for name, label in figures if (output/relative/name).is_file())
        metrics = strings["geometry_metrics"].format(error=item["max_error_px"], inside=item["projected_fraction"]*100)
        blocks.append(f"<section><h2>{title}</h2><p>{metrics}</p>{figures_html}</section>")
    page = '<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+strings["geometry_title"]+'</title><style>body{font:16px system-ui;margin:32px auto;max-width:1400px;background:#18202a;color:#e7edf5}section{border-top:1px solid #526172;padding:24px 0}img{max-width:100%;height:auto}figure{margin:20px 0}figcaption{margin-bottom:8px;color:#a5cfff}h1{font-size:30px}p{line-height:1.6}</style><h1>'+strings["geometry_title"]+'</h1><p>'+strings["geometry_review_hint"]+'</p>'+"".join(blocks)+'</html>'
    atomic_text(output / "geometry_review.html", page)
