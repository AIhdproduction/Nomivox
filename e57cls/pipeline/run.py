"""E57 -> panorama views -> SAM -> independent scan votes -> original LAS."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import label, maximum_filter, minimum_filter

from e57cls.files import atomic_json, atomic_text, file_hash
from e57cls.geometry.cameras import Camera, Pose, plan_views, scan_to_camera
from e57cls.geometry.images import extract_blob, open_image, render_view
from e57cls.geometry.visibility import DepthBuffer
from e57cls.geometry.fill import extract_planes
from e57cls.io.inspector import inspect_file, markdown
from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan, reduced_store
from e57cls.output.las import write_cloud
from e57cls.output.e57 import write_e57
from e57cls.output.classify import selected_count
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


def cleanup_run_cache(cache: Path, source_hash: str, identity: str, mask_cache: MaskCache, mode: str) -> dict:
    """Remove only cache artifacts owned by the completed run."""
    result = {"mode": mode, "freed_bytes": 0, "removed_paths_count": 0}
    if mode == "keep_all":
        return result
    cache = cache.resolve()
    targets = [cache/"scan_store"/source_hash, cache/"votes"/identity, cache/"geometry"/source_hash]
    if mode == "all":
        targets.extend(mask_cache.used_paths)
    failures = []
    for target in targets:
        try:
            target = Path(target).resolve()
            if not target.is_relative_to(cache) or target == cache:
                raise ValueError("refusing to remove path outside cache")
            if not target.exists():
                continue
            if target.is_dir():
                size = sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
                shutil.rmtree(target)
            else:
                size = target.stat().st_size
                target.unlink()
            result["freed_bytes"] += size
            result["removed_paths_count"] += 1
        except Exception as error:
            failures.append({"path": str(target), "detail": str(error)})
    if failures:
        result["incomplete"] = failures
    return result


class InputNotClassifiable(ValueError):
    """Input lacks the image/camera metadata required before expensive processing."""


class UnclassifiedVotes:
    """VoteStore-compatible source for reflection-only exports.

    Reflection filtering deliberately does not invent semantic classes.  Points
    that are retained therefore remain LAS class 1 and the mirror stage can
    still mark proven virtual points as class 68 before they are removed.
    """
    priority_overrides = 0

    @staticmethod
    def lookup(keys):
        return np.ones(len(keys), np.uint8), np.zeros(len(keys), np.uint8)

    @staticmethod
    def close():
        return None


def select_classes(preset, mode, building_enclosure=False):
    if mode == 'unsure':
        mode = 'indoor'
    selected = [c for c in preset["classes"] if c.get("enabled", True)
                and (mode not in ("indoor", "outdoor") or c["scene"] in (mode, "both"))]
    if building_enclosure:
        # Both ground labels and facade evidence must be available from every
        # scanner position.  The final spatial split happens after voting.
        selected_ids = {c['id'] for c in selected}
        selected.extend(c for c in preset['classes'] if c.get('enabled', True)
                        and c['id'] in {'ground', 'floor', 'building', 'wall'} and c['id'] not in selected_ids)
    return selected


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
        competitors = [ids[key] for key in config['stairs_priority_over'] if key in ids]
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


def depth_gate_representatives(labels, pixels, camera_range, world, chosen, classes, config):
    """Return representatives which must not cast a depth-gated class vote.

    ``pixels`` and ``chosen`` describe only visible/projectable representatives.
    A rejected representative deliberately gets no fallback label from this
    view: semantic masks have exactly one label per pixel.
    """
    blocked = np.zeros(len(chosen), bool)
    report = {'components': 0, 'rejected_representatives': 0}
    targets = {index for index, item in enumerate(classes) if item['id'] in config['depth_gate_classes']}
    if not targets or not len(chosen):
        return blocked, report
    structure = np.ones((3, 3), np.uint8)
    for target in sorted(targets):
        components, count = label(labels == target, structure=structure)
        for component in range(1, count+1):
            ids = np.flatnonzero((chosen == target) & (components[pixels[:, 1], pixels[:, 0]] == component))
            if len(ids) < 10:
                continue
            report['components'] += 1
            reference = float(np.percentile(camera_range[ids], 10))
            keep = ids[camera_range[ids] <= reference+config['door_depth_limit']]
            reject = np.setdiff1d(ids, keep, assume_unique=True)
            if len(keep):
                core = keep[camera_range[keep] <= reference+.15]
                core = core if len(core) else keep
                center = np.median(world[core], axis=0)
                horizontal = np.linalg.norm(world[keep, :2]-center[:2], axis=1) <= config['door_max_width']/2
                vertical = np.abs(world[keep, 2]-center[2]) <= config['door_max_height']/2
                reject = np.unique(np.r_[reject, keep[~(horizontal & vertical)]])
            blocked[reject] = True
            report['rejected_representatives'] += int(len(reject))
    return blocked, report


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
    stats = {"hits": 0, "misses": 0, "views": 0, "representatives": count,
             "depth_gate": {'components': 0, 'rejected_representatives': 0, 'seconds': 0.}}
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
                # Build a small per-view gate map before accumulating votes.
                # The geometry is evaluated only for labels which can be gated
                # (normally ``door``), keeping other semantic classes fast.
                blocked = np.zeros(count, bool)
                target_indices = {index for index, item in enumerate(classes) if item['id'] in config['depth_gate_classes']}
                if target_indices:
                    gate_started = time.monotonic()
                    gate_ids, gate_pixels, gate_ranges, gate_world, gate_labels = [], [], [], [], []
                    for start in range(0, count, 50_000):
                        control.checkpoint()
                        world = reps['xyz'][start:start+50_000]
                        camera_points = camera_pose.from_world(world)
                        camera_uv, camera_range, camera_valid = camera.project(camera_points)
                        visible = depth_buffer.visible(camera_uv*np.array([depth_buffer.width/camera.width, depth_buffer.height/camera.height]),
                                                       camera_range, camera_valid, config['visibility_abs'], config['visibility_rel'])
                        if mirrors:
                            visible &= rep_states[start:start+50_000] != 2
                        uv, _, projected = view.project(camera_points) if view else (camera_uv, camera_range, camera_valid)
                        size = np.array([view.size, view.size] if view else [camera.width, camera.height])
                        ids = np.flatnonzero(visible & projected)
                        if not len(ids):
                            continue
                        pixels = np.floor(uv[ids]*np.array([w, h])/size).astype(int)
                        pixels[:, 0] = pixels[:, 0].clip(0, w-1)
                        pixels[:, 1] = pixels[:, 1].clip(0, h-1)
                        chosen = labels[pixels[:, 1], pixels[:, 0]]
                        take = np.isin(chosen, list(target_indices))
                        if take.any():
                            gate_ids.append(start+ids[take]); gate_pixels.append(pixels[take]); gate_ranges.append(camera_range[ids[take]])
                            gate_world.append(world[ids[take]]); gate_labels.append(chosen[take])
                    if gate_ids:
                        local, gate_report = depth_gate_representatives(labels, np.concatenate(gate_pixels), np.concatenate(gate_ranges),
                                                                         np.concatenate(gate_world), np.concatenate(gate_labels), classes, config)
                        blocked[np.concatenate(gate_ids)[local]] = True
                        for key, value in gate_report.items():
                            stats['depth_gate'][key] += value
                    stats['depth_gate']['seconds'] += time.monotonic()-gate_started
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
                    accepted = (chosen >= 0) & ~blocked[start+ids]
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
    cleanup_report = None
    vegetation_report = None
    building_report = None
    stairs_geometry_report = None
    surface_tie_report = None
    geometry_fill_report = None
    geometry_planes = None
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
        if not any(scan['classification_eligible_metadata'] for scan in inspection['scans']):
            strings = json.loads((Path(__file__).resolve().parents[1]/'gui/strings_de.json').read_text(encoding='utf-8'))
            reason = 'input_no_images' if inspection['image_count'] == 0 else 'input_no_usable_images'
            raise InputNotClassifiable(strings[reason])
        store_report = build_store(source, cache/"scan_store", progress=lambda s,n,t: control.checkpoint())
        atomic_json(output/"store_report.json", store_report)
        full_stores = [load_scan(Path(item["path"]), store_report["source_sha256"], verify_hashes=False) for item in store_report["scans"]]
        stores = [reduced_store(store, config["point_reduction"], config["seed"]) for store in full_stores]
        if not stores:
            raise ValueError("E57 contains no scans")
        preset = json.loads(Path(config["preset"]).read_text(encoding="utf-8"))
        semantic_run = config['processing_mode'] == 'classify_and_filter'
        errors, preset_warnings = validate_preset(preset, require_active_classes=semantic_run)
        if errors:
            raise ValueError("Invalid preset: "+"; ".join(errors))
        warnings.extend({'code': 'preset_warning', 'detail': w} for w in preset_warnings)
        all_classes = [c for c in preset["classes"] if c.get("enabled", True)]
        if semantic_run and not all_classes:
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
        rows = legend_rows(preset if semantic_run else {**preset, 'classes': []})
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
                    if item.get('failed'):
                        warnings.append({'code':'mirror_image_failed','scan':item['scan'],
                                         'image':item['image'],'mask_kind':item['kind'],'detail':item['error']})
                    if any(not region['accepted'] for region in item['regions']):
                        warnings.append({'code':'reflection_plane_missing','scan':item['scan'],
                                         'image':item['image'],'mask_kind':item['kind']})
                control.event('phase_finished',phase='mirror')
            if semantic_run:
                control.event("phase_started", phase="segmentation")
                seg_start = time.monotonic()
                for store in stores:
                    control.checkpoint()
                    scan = store.meta["scan_index"]
                    mode = config["scene_overrides"].get(str(scan), config["scene_mode"])
                    classes = select_classes(preset, mode, config['building_enclosure'])
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
                            classes = select_classes(preset, mode, config['building_enclosure'])
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
                        # The committed SQLite vote rows are the resume boundary.
                        # Representatives remain useful for diagnostics, but these
                        # large per-scan intermediates are no longer needed.
                        weights_path.unlink(missing_ok=True)
                        (work/f"scan_{scan:04d}"/"mirror_rep.npy").unlink(missing_ok=True)
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
                priority = {'code': by_id['stairs'], 'over': [by_id[key] for key in config['stairs_priority_over'] if key in by_id],
                            'confidence': config['stairs_min_confidence']} if config['stairs_priority'] and 'stairs' in by_id else None
                votes.decide(config["min_observations"], config["min_winning_share"], codes, control.checkpoint, priority)
                if config['surface_tie_break']:
                    tie_started = time.monotonic()
                    surface_tie_report = votes.resolve_surface_ties(
                        by_id, np.array(config['voxel_origin']), config['voxel_size'], config['surface_tie_share'],
                        config['surface_normal_radius'], config['min_observations'],
                        [store.meta['metadata']['pose']['translation']['z'] for store in stores], control.checkpoint)
                    timings['surface_ties'] = time.monotonic()-tie_started
                if config['building_enclosure'] and {'ground', 'floor'}.issubset(by_id):
                    barrier_codes = [by_id[key] for key in ('building', 'wall') if key in by_id]
                    building_report = votes.split_ground_floor_by_building(
                        by_id['ground'], by_id['floor'], barrier_codes, config['voxel_size'],
                        config['building_boundary_cell_size'], config['building_boundary_min_height'],
                        config['building_boundary_max_cells'], control.checkpoint)
                if config['stairs_geometry_check'] and 'stairs' in by_id:
                    stairs_geometry_report = votes.reject_planar_stairs(
                        by_id['stairs'], np.array(config['voxel_origin']), config['voxel_size'],
                        config['stairs_plane_tolerance'], config['stairs_planar_fraction'], config['stairs_min_rise'],
                        config['seed'], codes, control.checkpoint)
                if config['component_cleanup']:
                    cleanup_rules = []
                    if 'cable' in by_id:
                        cleanup_rules.append((by_id['cable'], config['cable_component_min_voxels'],
                                              config['cable_component_min_span']))
                    if 'electrical' in by_id:
                        cleanup_rules.append((by_id['electrical'], config['electrical_component_min_voxels'],
                                              config['electrical_component_min_span']))
                    cleanup_report = votes.remove_small_components(config['voxel_size'], config['component_min_voxels'],
                                                                     cleanup_rules, control.checkpoint)
                if config['vegetation_base_to_ground'] and 'low_vegetation' in by_id and 'ground' in by_id:
                    vegetation_report = votes.vegetation_bases_to_ground(
                        by_id['low_vegetation'], by_id['ground'], config['voxel_size'],
                        config['vegetation_ground_cell_size'], config['vegetation_ground_band'], control.checkpoint)
                if config['geometry_fill']:
                    fill_started = time.monotonic()
                    geometry_planes = work/'planes.npz'
                    geometry_fill_report = extract_planes(votes, by_id, np.array(config['voxel_origin']), config['voxel_size'],
                                                         config, geometry_planes, control.checkpoint)
                    timings['geometry_fill_planes'] = time.monotonic()-fill_started
                timings["voting"] = time.monotonic()-vote_start
                control.event("phase_finished", phase="voting")
            else:
                # The mask/plane stage above has already done the only required
                # SAM work.  Retained points are intentionally unclassified.
                votes.close()
                votes = UnclassifiedVotes()
                scan_results = [{"scan": s.meta["scan_index"], "scene": "filter_only", "mode": "filter_only"} for s in stores]
        control.event("phase_started", phase="output")
        output_start = time.monotonic()
        suffix = 'reflections_filtered' if not semantic_run else 'classified'
        path = output/f"{source.stem}_{suffix}.{config['format']}"
        if path.exists():
            raise FileExistsError("Completed output exists; choose a new run directory")
        writer = write_e57 if config["format"] == "e57" else write_cloud
        output_args = (path, source, stores, votes, np.array(config["voxel_origin"]), config["voxel_size"], [r["code"] for r in rows], skipped,
                       control.checkpoint, lambda phase,fraction,scan: control.event("progress", phase=phase, fraction=fraction, scan=scan)) if writer is write_e57 else (
                       path, stores, votes, np.array(config["voxel_origin"]), config["voxel_size"], inspection["metadata"].get("coordinateMetadata", ""),
                       [r["code"] for r in rows], skipped, control.checkpoint,
                       lambda phase,fraction,scan: control.event("progress", phase=phase, fraction=fraction, scan=scan))
        result = writer(*output_args, mirror=mirrors.output if mirrors else None,
                        no_reflection=config['extra_no_reflection'], primary_without_reflections=not semantic_run,
                        class_transfer_radius=config['class_transfer_radius'] if semantic_run else 0.,
                        output_density=config['output_density'],
                        geometry_planes=geometry_planes if semantic_run and config['geometry_fill'] else None,
                        geometry_fill_tolerance=config['geometry_fill_tolerance'])
        timings["output"] = time.monotonic()-output_start
        transferred = {"cartesianX","cartesianY","cartesianZ","colorRed","colorGreen","colorBlue","intensity"}
        omitted = {str(s.meta["scan_index"]): sorted(set(s.meta["metadata"]["points"]["fields"])-transferred) for s in stores}
        if any(omitted.values()):
            warnings.append({"code": "attributes_omitted"})
        result.update(point_reduction=config["point_reduction"], output_density=config["output_density"],
                      computed_points=sum(selected_count(store, "reduced") for store in stores),
                      output_points=result["record_count"], format=config["format"])
        report = {"schema_version": 1, "input": inspection["source"], "input_sha256": input_hash,
                  "output": result, "scans": scan_results, "skipped_scans": skipped, "warnings": warnings,
                  "invalid_counts": {str(s.meta["scan_index"]): s.meta["invalid_count"] for s in stores},
                  "non_transferred_attributes": omitted, "timing_seconds": timings,
                  "total_seconds": time.monotonic()-started, "sam_calls": backend.calls if backend else 0,
                  "gpu_reductions": backend.reductions if backend else [], "seed": config["seed"],
                  "sampled_peak_rss_bytes": control.peak_rss, "sampled_peak_handles": control.peak_handles,
                  "stairs_priority_overrides": votes.priority_overrides,
                  "stairs_geometry": stairs_geometry_report,
                  "surface_ties": surface_tie_report,
                  "component_cleanup": cleanup_report,
                  "vegetation_base_to_ground": vegetation_report, "building_enclosure": building_report,
                  "depth_gate": {str(item['scan']): item.get('depth_gate', {}) for item in scan_results if 'scan' in item},
                  "geometry_fill": {**(geometry_fill_report or {'planes': {}}),
                                    'filled_points': result.get('geometry_fill_counts', {}),
                                    'confidence': 96},
                  "mask_cache_behavior": "Mask identities include the prompts of one class. Removing doorway invalidates only door masks; other class masks remain reusable.",
                  "reflection_analysis": mirrors.report if mirrors else None,
                  "time_policy": "GPS time is zero when time base is not verified",
                  "return_policy": "Return number/count are zero when unavailable",
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_report(output, report, rows)
        # Output writers have completed their read-back verification and the
        # report is durable before run-owned cache is removed.  Closing first
        # matters on Windows, where memmaps and SQLite keep file handles open.
        if mirrors:
            mirrors.close()
            mirrors = None
        if votes:
            votes.close()
            votes = None
        cleanup = cleanup_run_cache(cache, input_hash, identity, mask_cache, config['cache_cleanup'])
        report['cache_cleanup'] = {key: value for key, value in cleanup.items() if key != 'incomplete'}
        if cleanup.get('incomplete'):
            warnings.append({'code': 'cache_cleanup_incomplete', 'detail': cleanup['incomplete']})
            report['warnings'] = warnings
        write_report(output, report, rows)
        control.event('cache_cleanup', **report['cache_cleanup'])
        control.event("phase_finished", phase="output")
        control.event("finished", output=str(path), report=str(output/"report.html"))
        return report
    except Cancelled:
        LOG.info("Run cancelled at a safe work-unit boundary; cache remains reusable")
        return {"cancelled": True}
    except InputNotClassifiable as error:
        LOG.warning("Input rejected before classification: %s", error)
        control.event("error", code="input_rejected", detail=str(error))
        raise
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
