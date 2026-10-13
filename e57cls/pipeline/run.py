"""E57 -> panorama views -> SAM -> independent scan votes -> original LAS."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from e57cls.files import atomic_json, atomic_text, file_hash
from e57cls.geometry.cameras import Camera, Pose, plan_views, scan_to_camera
from e57cls.geometry.images import extract_blob, open_image, render_view
from e57cls.geometry.visibility import DepthBuffer
from e57cls.io.inspector import inspect_file, markdown
from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan
from e57cls.output.las import write_cloud
from e57cls.output.reports import legend_rows, write_legends, write_report
from e57cls.output.diagnostics import save_overlay
from e57cls.pipeline.control import Cancelled, Control
from e57cls.seg.backend import PromptClass, SAM3Backend
from e57cls.seg.cache import MaskCache, identity_hash, segment_view
from e57cls.vote.engine import VoteStore, representatives
from e57cls.scene.presets import validate_preset
from e57cls.scene.detect import detect
from e57cls.mirror.pipeline import build as build_mirrors

LOG = logging.getLogger(__name__)


def select_classes(preset, mode):
    if mode == 'unsure':
        mode = 'indoor'
    return [c for c in preset["classes"] if c.get("enabled", True)
            and (mode not in ("indoor", "outdoor") or c["scene"] in (mode, "both"))]


def image_assets(reader, item, cache):
    kind = next(r for r in item["representations"] if r != "visualReferenceRepresentation")
    params = item["metadata"][kind]
    camera = Camera(kind, params)
    directory = cache/f"image_{item['index']:04d}"
    image_name = "jpegImage" if "jpegImage" in params else "pngImage"
    path = directory/("image.jpg" if image_name == "jpegImage" else "image.png")
    if not path.is_file():
        extract_blob(reader.blob(item["index"]-1, kind, image_name), path)
    image = open_image(path, "RGB")
    mask = None
    mask_hash = None
    if "imageMask" in params:
        mask_path = directory/"mask.png"
        if not mask_path.is_file():
            extract_blob(reader.blob(item["index"]-1, kind, "imageMask"), mask_path)
        mask = open_image(mask_path, "L")
        mask_hash = file_hash(mask_path)
    if image.shape[:2] != (camera.height, camera.width):
        raise ValueError("Image dimensions differ from camera metadata")
    return camera, image, mask, file_hash(path), mask_hash


def camera_depth(store, scan_pose, camera_pose, camera, width, control):
    width = min(camera.width, width)
    height = max(1, round(camera.height*width/camera.width))
    buffer = DepthBuffer(width, height, radius=1, wrap=camera.wraps)
    for _, xyz, valid in store.chunks():
        control.checkpoint()
        xyz = scan_to_camera(xyz[valid], scan_pose, camera_pose)
        uv, depth, valid_projection = camera.project(xyz)
        buffer.add(uv*np.array([width/camera.width, height/camera.height]), depth, valid_projection)
    return buffer


def evidence_map(scores, classes, config):
    # Pixel conflicts are resolved by score, then preset order. Raw scores remain in cache.
    stacked = np.stack(scores).astype(np.float32)
    labels = stacked.argmax(axis=0).astype(np.int16)
    confidence = stacked.max(axis=0)
    ids = {c['id']: i for i,c in enumerate(classes)}
    if config['stairs_priority'] and 'stairs' in ids:
        stair = ids['stairs']
        competitors = [ids[k] for k in ('floor','ceiling','ground') if k in ids]
        preferred = np.isin(labels,competitors) & (stacked[stair] >= config['stairs_min_confidence'])
        labels[preferred] = stair
        confidence[preferred] = stacked[stair][preferred]
    thresholds = np.array([c.get("min_mask_conf") if c.get("min_mask_conf") is not None
                           else config["mask_confidence"] for c in classes])
    accepted = (confidence >= thresholds[labels]) & (confidence > 0)
    margin = config["boundary_margin"]
    if margin:
        compare = np.where(accepted, labels, -1)
        size = 2*margin+1
        accepted &= minimum_filter(compare, size=size, mode="constant", cval=-1) == labels
        accepted &= maximum_filter(compare, size=size, mode="constant", cval=-1) == labels
    labels[~accepted] = -1
    return labels, confidence


def classify_scan(reader, store, inspection, config, work, image_cache, mask_cache,
                  model_hash, factory, control, classes, all_classes, identity, mirrors=None):
    scan = store.meta["scan_index"]
    scan_meta = inspection["scans"][scan-1]
    origin = np.array(config["voxel_origin"], np.float64)
    rep_path, count = representatives(store, work/f"scan_{scan:04d}", origin, config["voxel_size"], control.checkpoint)
    reps = np.load(rep_path, mmap_mode="r", allow_pickle=False)
    weights_path = work/f"scan_{scan:04d}"/"weights.npy"
    weights = np.lib.format.open_memmap(weights_path, mode="w+", dtype=np.float32, shape=(count, len(all_classes)))
    weights[:] = 0
    by_id = {c["id"]: i for i,c in enumerate(all_classes)}
    global_columns = np.array([by_id[c["id"]] for c in classes])
    prompt_classes = [PromptClass(c["id"], tuple(c["prompts"]), c.get("min_instance_conf") if c.get("min_instance_conf") is not None else config["instance_confidence"]) for c in classes]
    scan_pose = Pose.from_dict(store.meta["metadata"]["pose"])
    stats = {"hits": 0, "misses": 0, "views": 0, "representatives": count}
    rep_states = None
    try:
        if mirrors:
            rep_states = np.lib.format.open_memmap(work/f'scan_{scan:04d}'/'mirror_rep.npy', mode='w+', dtype=np.uint8, shape=(count,))
            for start in range(0,count,50000):
                control.checkpoint()
                rep_states[start:start+50000] = mirrors.states(reps['xyz'][start:start+50000],scan,scan_pose.translation,scan_meta['image_indices'])
        images = [inspection["images"][i-1] for i in scan_meta["image_indices"]]
        usable = [i for i in images if i["pose_status"] == "valid" and any(r != "visualReferenceRepresentation" for r in i["representations"])]
        if not usable:
            raise ValueError("No associated projectable image with camera pose")
        for item in usable:
            camera, image, mask, image_hash, mask_hash = image_assets(reader, item, image_cache)
            camera_pose = Pose.from_dict(item["metadata"]["pose"])
            depth_buffer = camera_depth(store, scan_pose, camera_pose, camera, config["depth_width"], control)
            views = plan_views(camera, config["profile"], config["view_size"], config["view_fov"])
            planned = views if views else [None]
            for view_index, view in enumerate(planned):
                control.checkpoint()
                if view:
                    rendered, valid = render_view(image, mask, camera, view)
                    view_info = asdict(view)
                else:
                    rendered, valid = image, (mask > 0) if mask is not None else np.ones(image.shape[:2], bool)
                    view_info = {"kind": camera.kind, "params": camera.params}
                view_info.update(mask_sha256=mask_hash, torch_version=version("torch"))
                scores, cached = segment_view(rendered, valid, prompt_classes, factory, mask_cache, image_hash,
                                               view_info, model_hash, config["image_scale"], config["tile_size"],
                                               config["tile_overlap"], control.checkpoint)
                stats["hits"] += cached["hits"]
                stats["misses"] += cached["misses"]
                stats["views"] += 1
                labels, confidence = evidence_map(scores, classes, config)
                if config['diagnostics'] == 'all' or (config['diagnostics'] == 'sample' and view_index == 0):
                    save_overlay(config['output'],scan,item['index'],view_index,rendered,labels,classes)
                h, w = labels.shape
                for start in range(0, count, 50_000):
                    control.checkpoint()
                    world = reps["xyz"][start:start+50_000]
                    camera_points = camera_pose.from_world(world)
                    camera_uv, camera_range, camera_valid = camera.project(camera_points)
                    visible = depth_buffer.visible(camera_uv*np.array([depth_buffer.width/camera.width, depth_buffer.height/camera.height]),
                                                   camera_range, camera_valid, config["visibility_abs"], config["visibility_rel"])
                    if mirrors:
                        states = rep_states[start:start+50_000]
                        # An uncertain SAM mask must not suppress ordinary semantic evidence.
                        visible &= states != 2
                    uv, _, projected = view.project(camera_points) if view else (camera_uv, camera_range, camera_valid)
                    size = np.array([view.size, view.size] if view else [camera.width, camera.height])
                    ids = np.flatnonzero(visible & projected)
                    pixels = np.floor(uv[ids]*np.array([w,h])/size).astype(int)
                    pixels[:, 0] = pixels[:, 0].clip(0, w-1)
                    pixels[:, 1] = pixels[:, 1].clip(0, h-1)
                    chosen = labels[pixels[:, 1], pixels[:, 0]]
                    accepted = chosen >= 0
                    columns = global_columns[chosen[accepted]]
                    weights[start+ids[accepted], columns] = np.maximum(weights[start+ids[accepted], columns],
                        confidence[pixels[accepted, 1], pixels[accepted, 0]])
                weights.flush()
                control.event("progress", phase="segmentation", scan=scan, scan_count=inspection["scan_count"],
                              image=item["index"], image_count=inspection["image_count"], view=view_index+1,
                              view_count=len(planned), fraction=(scan-1+(view_index+1)/len(planned))/inspection["scan_count"], cache=cached)
                LOG.info("Scan %d image %d view %d/%d complete; cache %s", scan, item["index"], view_index+1, len(planned), cached)
            del image, mask, depth_buffer
        atomic_json(work/f"scan_{scan:04d}"/"evidence.json", {"identity": identity, **stats})
    finally:
        if rep_states is not None:
            rep_states._mmap.close()
        weights._mmap.close()
        reps._mmap.close()
    return rep_path, weights_path, stats


def execute(config, backend_factory=None, listen=True):
    source, output, cache = (Path(config[k]) for k in ("source", "output", "cache"))
    if output == source or cache == source:
        raise ValueError("Output/cache cannot be the input file")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for variable, name in (("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("SQLITE_TMPDIR", "temp")):
        directory = cache/name
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    control = Control(output, listen)
    handler = logging.FileHandler(output/"log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.captureWarnings(True)
    started = time.monotonic()
    timings, warnings, scan_results, skipped = {}, [], [], []
    backend = None
    votes = None
    mirrors = None
    def factory():
        nonlocal backend
        if backend is None:
            import random
            random.seed(config['seed'])
            np.random.seed(config['seed'])
            if not backend_factory:
                import torch
                torch.manual_seed(config['seed'])
            backend = backend_factory() if backend_factory else SAM3Backend(Path(config["model"]), config["gpu_device"], config["prompt_batch"], config["gpu_memory_fraction"])
        return backend
    try:
        control.event("state", state="running")
        control.event("phase_started", phase="inspection")
        LOG.info("Run started for %s", source)
        inspection = inspect_file(source)
        atomic_json(output/"inspect_report.json", inspection)
        atomic_text(output/"inspect_report.md", markdown(inspection))
        store_report = build_store(source, cache/"scan_store", progress=lambda s,n,t: control.checkpoint())
        atomic_json(output/"store_report.json", store_report)
        stores = [load_scan(Path(item["path"]), store_report["source_sha256"], verify_hashes=False) for item in store_report["scans"]]
        if not stores:
            raise ValueError("E57 contains no scans")
        preset = json.loads(Path(config["preset"]).read_text(encoding="utf-8"))
        errors, preset_warnings = validate_preset(preset)
        if errors:
            raise ValueError("Invalid preset: "+"; ".join(errors))
        warnings.extend({'code': 'preset_warning', 'detail': w} for w in preset_warnings)
        all_classes = [c for c in preset["classes"] if c.get("enabled", True)]
        if not all_classes:
            raise ValueError("No active classes")
        config = {**config, "voxel_origin": [stores[0].meta["metadata"]["pose"]["translation"][k] for k in "xyz"]}
        model_hash = file_hash(Path(config["model"]))
        input_hash = store_report["source_sha256"]
        identity = identity_hash({"pipeline_revision": "wall-frame-v3", "source": input_hash, "model": model_hash, "preset": preset,
                                  "config": {k:v for k,v in config.items() if k not in ("output", "cache")}})
        manifest = {**config, "pipeline_revision": "wall-frame-v3", "identity": identity, "input_sha256": input_hash, "model_sha256": model_hash,
                    "resolved_preset": preset, "versions": {n: version(n) for n in ("numpy","pye57","Pillow","torch","torchvision","sam3","laspy")}}
        previous = output/"run_config.json"
        if previous.exists() and json.loads(previous.read_text(encoding="utf-8")).get("identity") != identity:
            raise ValueError("This output directory belongs to a different run; choose a new folder")
        atomic_json(previous, manifest)
        rows = legend_rows(preset)
        write_legends(output, rows)
        atomic_json(output/"scan_index.json", {str(s.meta["scan_index"]): {"guid": s.meta["metadata"].get("guid"), "name": s.meta["metadata"].get("name")} for s in stores})
        timings["inspection"] = time.monotonic()-started
        control.event("phase_finished", phase="inspection")
        work = cache/"votes"/identity
        work.mkdir(parents=True, exist_ok=True)
        votes = VoteStore(work/"votes.sqlite")
        mask_cache = MaskCache(cache/"seg")
        image_cache = cache/"geometry"/input_hash/"images"
        mapping = preset.get("code_map", {})
        codes = [int(mapping.get(str(c["code"]), c["code"])) for c in all_classes]
        with E57Reader(source) as reader:
            if config['mirror_analysis']:
                control.event('phase_started',phase='mirror')
                mirror_start = time.monotonic()
                mirrors = build_mirrors(reader,stores,inspection,config,work,image_cache,mask_cache,model_hash,factory,control,preset,image_assets)
                atomic_json(output/'mirror_registry.json',[p.serialize() for p in mirrors.registry.planes])
                timings['mirror'] = time.monotonic()-mirror_start
                warnings.append({'code':'mirror_limits'})
                for item in mirrors.report['images']:
                    if any(not region['accepted'] for region in item['regions']):
                        warnings.append({'code':'reflection_plane_missing','scan':item['scan'],
                                         'image':item['image'],'mask_kind':item['kind']})
                control.event('phase_finished',phase='mirror')
            control.event("phase_started", phase="segmentation")
            seg_start = time.monotonic()
            for store in stores:
                control.checkpoint()
                scan = store.meta["scan_index"]
                mode = config["scene_overrides"].get(str(scan), config["scene_mode"])
                classes = select_classes(preset, mode)
                if votes.completed(scan, identity):
                    scan_results.append({"scan": scan, "scene": mode, "resumed": True})
                    continue
                try:
                    scene = {'scene': mode, 'manual': True}
                    if mode == 'both':
                        control.event('phase_started', phase='scene', scan=scan)
                        scene = detect(reader, store, inspection, config, image_cache, mask_cache, model_hash,
                                       factory, control, preset, image_assets, camera_depth)
                        mode = scene['scene']
                        classes = select_classes(preset, mode)
                        atomic_json(work/f'scene_{scan:04d}.json', scene)
                        control.event('scene', scan=scan, **scene)
                        control.event('phase_finished', phase='scene', scan=scan)
                        if mode in ('unsure', 'mixed'):
                            warnings.append({'code': 'scene_unsure', 'scan': scan})
                    if not classes:
                        raise ValueError('No enabled classes for this scene')
                    rep_path, weights_path, stats = classify_scan(reader, store, inspection, config, work, image_cache,
                        mask_cache, model_hash, factory, control, classes, all_classes, identity, mirrors)
                    def batches():
                        reps = np.load(rep_path, mmap_mode="r", allow_pickle=False)
                        weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
                        try:
                            for start in range(0, len(reps), 50_000):
                                control.checkpoint()
                                yield reps["key"][start:start+50_000], weights[start:start+50_000]
                        finally:
                            reps._mmap.close()
                            weights._mmap.close()
                    votes.add_scan(scan, identity, batches(), codes)
                    scan_results.append({"scan": scan, "scene": mode, "scene_detection": scene, **stats})
                except Cancelled:
                    raise
                except Exception as error:
                    if config["strict"]:
                        raise
                    LOG.exception("Scan %d failed; retaining original valid points as class 1", scan)
                    skipped.append(scan)
                    warning = {"code": "scan_failed", "scan": scan, "detail": str(error)}
                    warnings.append(warning)
                    control.event("warning", **warning)
                    scan_results.append({"scan": scan, "scene": mode, "error": str(error)})
        timings["segmentation"] = time.monotonic()-seg_start
        if len(skipped) == len(stores):
            raise RuntimeError("No scan could be classified; see per-scan errors in log.txt")
        control.event("phase_finished", phase="segmentation")
        control.event("phase_started", phase="voting")
        vote_start = time.monotonic()
        by_id = {c['id']: codes[i] for i,c in enumerate(all_classes)}
        priority = {'code': by_id['stairs'], 'over': [by_id[k] for k in ('floor','ceiling','ground') if k in by_id],
                    'confidence': config['stairs_min_confidence']} if config['stairs_priority'] and 'stairs' in by_id else None
        votes.decide(config["min_observations"], config["min_winning_share"], codes, control.checkpoint, priority)
        timings["voting"] = time.monotonic()-vote_start
        control.event("phase_finished", phase="voting")
        control.event("phase_started", phase="output")
        output_start = time.monotonic()
        path = output/f"{source.stem}_classified.{config['format']}"
        if path.exists():
            raise FileExistsError("Completed output exists; choose a new run directory")
        result = write_cloud(path, stores, votes, np.array(config["voxel_origin"]), config["voxel_size"],
                             inspection["metadata"].get("coordinateMetadata", ""), [r["code"] for r in rows], skipped,
                             control.checkpoint, lambda phase,fraction,scan: control.event("progress", phase=phase, fraction=fraction, scan=scan),
                             mirror=mirrors.output if mirrors else None, no_reflection=config['extra_no_reflection'])
        timings["output"] = time.monotonic()-output_start
        transferred = {"cartesianX","cartesianY","cartesianZ","colorRed","colorGreen","colorBlue","intensity"}
        omitted = {str(s.meta["scan_index"]): sorted(set(s.meta["metadata"]["points"]["fields"])-transferred) for s in stores}
        if any(omitted.values()):
            warnings.append({"code": "attributes_omitted"})
        report = {"schema_version": 1, "input": inspection["source"], "input_sha256": input_hash,
                  "output": result, "scans": scan_results, "skipped_scans": skipped, "warnings": warnings,
                  "invalid_counts": {str(s.meta["scan_index"]): s.meta["invalid_count"] for s in stores},
                  "non_transferred_attributes": omitted, "timing_seconds": timings,
                  "total_seconds": time.monotonic()-started, "sam_calls": backend.calls if backend else 0,
                  "gpu_reductions": backend.reductions if backend else [], "seed": config["seed"],
                  "sampled_peak_rss_bytes": control.peak_rss, "sampled_peak_handles": control.peak_handles,
                  "stairs_priority_overrides": votes.priority_overrides,
                  "reflection_analysis": mirrors.report if mirrors else None,
                  "time_policy": "GPS time is zero when time base is not verified",
                  "return_policy": "Return number/count are zero when unavailable",
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_report(output, report, rows)
        control.event("phase_finished", phase="output")
        control.event("finished", output=str(path), report=str(output/"report.html"))
        return report
    except Cancelled:
        LOG.info("Run cancelled at a safe work-unit boundary; cache remains reusable")
        return {"cancelled": True}
    except Exception:
        LOG.exception("Run failed")
        control.event("error", code="run_failed")
        raise
    finally:
        if mirrors:
            mirrors.close()
        if votes:
            votes.close()
        control.close()
        logging.getLogger().removeHandler(handler)
        handler.close()
