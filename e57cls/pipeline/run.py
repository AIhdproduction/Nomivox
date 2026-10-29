"""E57 -> panorama views -> SAM -> independent scan votes -> original LAS."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import threading
import time

import numpy as np
import psutil
from scipy.ndimage import label, maximum_filter, minimum_filter

from e57cls.files import atomic_json, atomic_text, file_hash
from e57cls.geometry.cameras import Camera, Pose, plan_views, scan_pose, scan_to_camera
from e57cls.geometry.images import extract_blob, open_image, render_view
from e57cls.geometry.visibility import DepthBuffer
from e57cls.geometry.fill import extract_planes
from e57cls.vote.geometry_rules import apply_geometry_rules
from e57cls.vote.segments import apply_segment_vote
from e57cls.vote.shape_rules import apply_shape_rules
from e57cls.vote.dtm import TinDTM, apply_ground_dtm_band, build_tin_dtm
from e57cls.vote.scene_consistency import enforce_scene_consistency
from e57cls.io.inspector import inspect_file, markdown
from e57cls.io.input_mode import detect_input_mode, pseudo_setup_points_check
from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan, reduced_store
from e57cls.output.las import write_cloud
from e57cls.output.e57 import write_e57
from e57cls.output.classify import selected_count
from e57cls.output.reports import legend_rows, write_legends, write_report
from e57cls.output.diagnostics import save_overlay
from e57cls.pipeline.control import Cancelled, Control
from e57cls.seg.backend import PromptClass, SAM3Backend
from e57cls.seg.cache import MaskCache, identity_hash, resize_mask_nearest, segment_view
from e57cls.vote.engine import VoteStore, representatives, voxel_keys
from e57cls.slam.cameras import build_camera_list, iter_camera_views
from e57cls.slam.cache import VisibilityCache
from e57cls.slam.index import build_spatial_index, unpack_point_id
from e57cls.slam.stations import cluster_stations
from e57cls.scene.presets import validate_preset
from e57cls.scene.detect import detect
from e57cls.scene.resolve import resolve_scan_scene, resolve_neighbours
from e57cls.mirror.pipeline import (MirrorAnalysis, build as build_mirrors, finalize_states,
                                    fit_scan_surfaces, reflection_prompt_classes)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamProducerMessage:
    """One complete message from the shared-SAM producer to its consumer."""

    kind: str
    scan: int | None = None
    payload: object = None


def cleanup_run_cache(cache: Path, source_hash: str, identity: str, mask_cache: MaskCache, mode: str) -> dict:
    """Remove only cache artifacts owned by the completed run."""
    result = {"mode": mode, "freed_bytes": 0, "removed_paths_count": 0}
    if mode == "keep_all":
        return result
    cache = cache.resolve()
    targets = [cache/"scan_store"/source_hash, cache/"votes"/identity, cache/"geometry"/source_hash,
               cache/"slam_visibility"/source_hash, cache/"slam_index"/source_hash]
    if mode == "all":
        targets.extend(mask_cache.used_paths_snapshot())
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


def geometry_scan_context(preset, config, scan_scenes, positions=None, class_ids=None):
    """Return conservative scene and selected-class evidence for geometry rules."""
    mapping = preset.get('code_map', {})
    result = {}
    for scan, scene in scan_scenes.items():
        # Only an explicit outdoor decision enables exterior plausibility rules.
        mode = scene if scene in ('indoor', 'outdoor') else 'indoor'
        ids = (class_ids or {}).get(scan) if isinstance(class_ids, dict) else None
        selected = ([item for item in preset['classes'] if item['id'] in ids] if ids is not None
                    else select_classes(preset, mode, config['building_enclosure']))
        result[int(scan)] = {'outdoor': mode == 'outdoor',
                             'scene': scene if scene in ('indoor', 'outdoor') else None,
                             'position': (np.asarray((positions or {}).get(scan), dtype=float).tolist()
                                          if isinstance(positions, dict) and (positions or {}).get(scan) is not None else None),
                             'allowed_codes': {int(mapping.get(str(item['code']), item['code'])) for item in selected}}
    return result


def _scene_record(scan, resolution, classes):
    """Return the durable scene record shared by normal and shared-SAM paths."""
    return {'scan': scan, 'mode': resolution['scene'], 'scene': resolution,
            'class_ids': [item['id'] for item in classes]}


def _emit_scene_event(control, scan, resolution):
    """Emit a scene event with the current scan taking precedence in its payload."""
    control.event('scene', **{**resolution, 'scan': int(scan)})


def _load_scene_record(work, scan):
    """Load both current shared-SAM and legacy detector scene records."""
    data = json.loads((work/f'scene_{scan:04d}.json').read_text(encoding='utf-8'))
    if 'mode' in data and isinstance(data.get('scene'), dict):
        return data
    if isinstance(data, dict) and 'scene' in data:
        resolution = resolve_scan_scene(data, {}, {**{'scene_unresolved_classes': 'all', 'scene_name_hints': False},
                                                    **data.get('thresholds', {})})
        return _scene_record(scan, resolution, [])
    raise ValueError('Invalid scene record')


def finalize_scene_resolution(scan_results, stores, config, warnings, control):
    """Resolve pending scan scenes once all static scan decisions are available."""
    by_scan = {item['scan']: item for item in scan_results if 'scan' in item}
    metadata = {store.meta['scan_index']: store.meta['metadata'] for store in stores}
    for result in scan_results:
        if result.get('scan') in metadata and not result.get('scene_resolution'):
            result['scene_resolution'] = resolve_scan_scene(None, metadata[result['scan']], config)
            result['scene_detection'] = result['scene_resolution']
    records = [dict(record, scan=item['scan']) if 'scan' not in record else record for item, record in
               ((item, item.get('scene_resolution') or item.get('scene_detection')) for item in scan_results)
               if record]
    positions = {store.meta['scan_index']: scan_pose(store.meta['metadata']).translation.tolist() for store in stores}
    resolved, neighbour_warnings = resolve_neighbours(records, positions, config)
    for resolution in resolved:
        result = by_scan.get(resolution.get('scan'))
        if result is not None:
            changed = result.get('scene') != resolution['scene']
            result['scene'] = resolution['scene']
            result['scene_resolution'] = resolution
            result['scene_detection'] = resolution
            if changed:
                _emit_scene_event(control, resolution['scan'], resolution)
    for result in scan_results:
        resolution = result.get('scene_resolution')
        if not resolution:
            continue
        hint = resolution.get('name_hint')
        if hint and hint.get('ambiguous'):
            warnings.append({'code': 'scene_name_ambiguous', 'scan': result['scan'], 'detail': hint})
        if resolution.get('conflict'):
            warnings.append({'code': 'scene_name_conflict', 'scan': result['scan'],
                             'detail': {'name': hint, 'data': resolution.get('data_scene')}})
    warnings.extend(neighbour_warnings)
    for result in scan_results:
        if result.get('scene') == 'unsure':
            warnings.append({'code': 'scene_unsure', 'scan': result['scan']})
    work = config.get('_scene_work')
    if work:
        atomic_json(Path(work)/'scene_resolution.json', {str(item['scan']): item.get('scene_resolution')
                                                          for item in scan_results if 'scene_resolution' in item})
    return scan_results


def scene_resolution_report(scan_results):
    """Summarize final scan decisions without duplicating per-scan records."""
    counts = {'scene': {}, 'scene_detail': {}, 'source': {}}
    conflicts, unresolved = [], []
    for item in scan_results:
        resolution = item.get('scene_resolution') or item.get('scene_detection') or {}
        for key in counts:
            value = resolution.get(key) if key != 'scene' else item.get('scene', resolution.get('scene'))
            if value:
                counts[key][value] = counts[key].get(value, 0)+1
        if resolution.get('conflict'):
            conflicts.append({'scan': item.get('scan'), 'name_hint': resolution.get('name_hint'),
                              'data_scene': resolution.get('data_scene')})
        if item.get('scene', resolution.get('scene')) not in ('indoor', 'outdoor'):
            unresolved.append(item.get('scan'))
    return {**counts, 'conflicts': conflicts, 'unresolved': unresolved}


def scene_audit(scan_results, scan_class_counts, preset, mapping):
    """Measure output classes that disagree with a resolved scan scene."""
    code_scenes = {}
    for item in preset.get('classes', []):
        code = int(mapping.get(str(item['code']), item['code']))
        code_scenes.setdefault(code, set()).add(item.get('scene', 'both'))
    scans, totals = [], {'outdoor': {'points': 0, 'wrong': 0}, 'indoor': {'points': 0, 'wrong': 0}}
    for item in scan_results:
        scene = item.get('scene')
        if scene not in totals:
            continue
        values = (scan_class_counts or {}).get(str(item.get('scan')), (scan_class_counts or {}).get(item.get('scan'), {}))
        total = wrong = 0
        for code, count in (values or {}).items():
            code, count = int(code), int(count)
            if code in (1, 68) or 'both' in code_scenes.get(code, {'both'}):
                continue
            total += count
            if ((scene == 'outdoor' and 'indoor' in code_scenes.get(code, set())) or
                    (scene == 'indoor' and 'outdoor' in code_scenes.get(code, set()))):
                wrong += count
        totals[scene]['points'] += total
        totals[scene]['wrong'] += wrong
        scans.append({'scan': item.get('scan'), 'scene': scene, 'points': total, 'wrong': wrong,
                      'wrong_scene_share': wrong/max(1, total)})
    for value in totals.values():
        value['wrong_scene_share'] = value['wrong']/max(1, value['points'])
    return {'scans': scans, 'totals': totals}


def prompt_classes(classes, config):
    return [PromptClass(c['id'], tuple(c['prompts']),
                        c.get('min_instance_conf') if c.get('min_instance_conf') is not None
                        else config['instance_confidence'])
            for c in classes]


def prepare_shared_sam(reader, stores, inspection, config, work, image_cache, mask_cache,
                       model_hash, factory, control, preset, assets, depth_builder,
                       progress_stores=None, image_numbers=None, total_images=None):
    """Populate one shared SAM cache pass for semantic and reflection prompts.

    The small JSON files make scene selection durable at the same boundary as
    the masks.  A resumed run deliberately visits all views again: cache hits
    are cheap while a missing entry from an interrupted pass is regenerated.
    """
    records = {}
    _, _, mirror_phrases = reflection_prompt_classes(config, preset)
    images = [(store, item) for store in stores
              for index in inspection['scans'][store.meta['scan_index']-1]['image_indices']
              for item in [inspection['images'][index-1]]
              if item['pose_status'] == 'valid'
              and any(r != 'visualReferenceRepresentation' for r in item['representations'])]
    image_count = total_images if total_images is not None else len(images)
    image_numbers = image_numbers or {item['index']: number for number, (_, item) in enumerate(images, 1)}
    progress_stores = progress_stores or stores
    for store in stores:
        control.checkpoint()
        scan = store.meta['scan_index']
        path = work/f'scene_{scan:04d}.json'
        manual = config['scene_overrides'].get(str(scan), config['scene_mode'])
        record = {'scan': scan, 'mode': manual, 'status': 'failed', 'failed_images': {}}
        try:
            detection = {'scene': manual, 'manual': True}
            if manual == 'both':
                control.event('phase_started', phase='scene', scan=scan)
                try:
                    detection = detect(reader, store, inspection, config, image_cache, mask_cache, model_hash,
                                       factory, control, preset, assets, depth_builder)
                finally:
                    control.event('phase_finished', phase='scene', scan=scan)
            resolution = resolve_scan_scene(detection, store.meta['metadata'], config,
                                            None if manual == 'both' else manual)
            classes = select_classes(preset, resolution['class_mode'], config['building_enclosure'])
            if not classes:
                raise ValueError('No enabled classes for this scene')
            record.update(_scene_record(scan, resolution, classes))
            _emit_scene_event(control, scan, resolution)
            prompts = prompt_classes(classes, config)+mirror_phrases
            scan_images = [item for candidate, item in images if candidate is store]
            for item in scan_images:
                image = mask = None
                try:
                    camera, image, mask, image_hash, mask_hash = assets(reader, item, image_cache)
                    views = plan_views(camera, config['profile'], config['view_size'], config['view_fov']) or [None]
                    for view_index, view in enumerate(views):
                        control.checkpoint()
                        rendered, valid = (render_view(image, mask, camera, view) if view else
                                           (image, mask > 0 if mask is not None else np.ones(image.shape[:2], bool)))
                        view_info = asdict(view) if view else {'kind': camera.kind, 'params': camera.params}
                        view_info.update(mask_sha256=mask_hash, torch_version=version('torch'))
                        cached_scores, cached = segment_view(
                            rendered, valid, prompts, factory, mask_cache, image_hash, view_info, model_hash,
                            config['image_scale'], config['tile_size'], config['tile_overlap'], control.checkpoint)
                        del cached_scores
                        number = image_numbers[item['index']]
                        control.event('progress', phase='sam', scan=scan, scan_count=len(progress_stores),
                                      image=item['index'], image_number=number, image_count=image_count,
                                      view=view_index+1, view_count=len(views),
                                      fraction=(number-1+(view_index+1)/len(views))/max(1, image_count), cache=cached)
                except Cancelled:
                    raise
                except Exception as error:
                    if config['strict']:
                        raise
                    record['failed_images'][str(item['index'])] = str(error)
                    break
                finally:
                    del image, mask
            if record['failed_images']:
                record['error'] = next(iter(record['failed_images'].values()))
            else:
                record['status'] = 'complete'
        except Cancelled:
            raise
        except Exception as error:
            if config['strict']:
                raise
            record['error'] = str(error)
        atomic_json(path, record)
        records[scan] = record
    return records


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


class ImageProjectionCache:
    """Camera-space data shared by every planned view of one source image."""

    _names = ("camera_points", "camera_uv", "camera_range", "camera_valid", "visible")

    def __init__(self, directory, image_index, count):
        # The five arrays use 50 bytes per representative. Keep a generous
        # margin for segmentation, depth buffers and the representative mmap.
        required = count*(3*np.dtype(np.float64).itemsize + 2*np.dtype(np.float64).itemsize
                          + np.dtype(np.float64).itemsize + 2*np.dtype(np.bool_).itemsize)
        self.paths = []
        self.memmap = count > 2_000_000 or required > psutil.virtual_memory().available//4
        specifications = {
            "camera_points": (np.float64, (count, 3)),
            "camera_uv": (np.float64, (count, 2)),
            "camera_range": (np.float64, (count,)),
            "camera_valid": (np.bool_, (count,)),
            "visible": (np.bool_, (count,)),
        }
        self.arrays = {}
        for name, (dtype, shape) in specifications.items():
            if self.memmap:
                path = directory/f"projection_{image_index}_{name}.npy"
                path.unlink(missing_ok=True)
                self.paths.append(path)
                self.arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            else:
                self.arrays[name] = np.empty(shape, dtype=dtype)

    def populate(self, reps, camera_pose, camera, depth_buffer, config, control, rep_states):
        scale = np.array([depth_buffer.width/camera.width, depth_buffer.height/camera.height])
        for start in range(0, len(reps), 50_000):
            control.checkpoint()
            stop = start+50_000
            world = reps["xyz"][start:stop]
            camera_points = camera_pose.from_world(world)
            camera_uv, camera_range, camera_valid = camera.project(camera_points)
            visible = depth_buffer.visible(camera_uv*scale, camera_range, camera_valid,
                                           config["visibility_abs"], config["visibility_rel"])
            if rep_states is not None:
                visible &= rep_states[start:stop] != 2
            self.arrays["camera_points"][start:stop] = camera_points
            self.arrays["camera_uv"][start:stop] = camera_uv
            self.arrays["camera_range"][start:stop] = camera_range
            self.arrays["camera_valid"][start:stop] = camera_valid
            self.arrays["visible"][start:stop] = visible
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array.flush()

    def close(self):
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array.flush()
                array._mmap.close()
        self.arrays.clear()
        for path in self.paths:
            path.unlink(missing_ok=True)
        self.paths.clear()


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
    prompts = prompt_classes(classes, config)
    scan_transform = scan_pose(store.meta["metadata"])
    stats = {"hits": 0, "misses": 0, "views": 0, "representatives": count,
             "depth_gate": {'components': 0, 'rejected_representatives': 0, 'seconds': 0.}}
    rep_states = None
    try:
        if mirrors:
            rep_states = np.lib.format.open_memmap(work/f'scan_{scan:04d}'/'mirror_rep.npy', mode='w+', dtype=np.uint8, shape=(count,))
            for start in range(0,count,50000):
                control.checkpoint()
                rep_states[start:start+50000] = mirrors.states(reps['xyz'][start:start+50000],scan,scan_transform.translation,scan_meta['image_indices'])
        images = [inspection["images"][i-1] for i in scan_meta["image_indices"]]
        usable = [i for i in images if i["pose_status"] == "valid" and any(r != "visualReferenceRepresentation" for r in i["representations"])]
        if not usable:
            raise ValueError("No associated projectable image with camera pose")
        for item in usable:
            camera, image, mask, image_hash, mask_hash = image_assets(reader, item, image_cache)
            camera_pose = Pose.from_dict(item["metadata"]["pose"])
            depth_buffer = camera_depth(store, scan_transform, camera_pose, camera, config["depth_width"], control)
            projection = ImageProjectionCache(work/f"scan_{scan:04d}", item["index"], count)
            camera_points = camera_uv = camera_range = camera_valid = visible = uv = projected = None
            try:
                projection.populate(reps, camera_pose, camera, depth_buffer, config, control, rep_states)
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
                    scores, cached = segment_view(rendered, valid, prompts, factory, mask_cache, image_hash,
                                                   view_info, model_hash, config["image_scale"], config["tile_size"],
                                                   config["tile_overlap"], control.checkpoint,
                                                   cache_only=config.get('_shared_sam', False))
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
                            stop = start+50_000
                            world = reps['xyz'][start:stop]
                            camera_points = projection.arrays['camera_points'][start:stop]
                            camera_uv = projection.arrays['camera_uv'][start:stop]
                            camera_range = projection.arrays['camera_range'][start:stop]
                            camera_valid = projection.arrays['camera_valid'][start:stop]
                            visible = projection.arrays['visible'][start:stop]
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
                        stop = start+50_000
                        camera_points = projection.arrays['camera_points'][start:stop]
                        camera_uv = projection.arrays['camera_uv'][start:stop]
                        camera_range = projection.arrays['camera_range'][start:stop]
                        camera_valid = projection.arrays['camera_valid'][start:stop]
                        visible = projection.arrays['visible'][start:stop]
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
            finally:
                camera_points = camera_uv = camera_range = camera_valid = visible = uv = projected = None
                projection.close()
                del image, mask, depth_buffer
        atomic_json(work/f"scan_{scan:04d}"/"evidence.json", {"identity": identity, **stats})
    finally:
        if rep_states is not None:
            rep_states._mmap.close()
        weights._mmap.close()
        reps._mmap.close()
    return rep_path, weights_path, stats


def _slam_point_values(stores, point_ids, fields=None):
    """Read world coordinates (and optionally explicit point times) by source row."""
    point_ids = np.asarray(point_ids, np.int64)
    world = np.empty((len(point_ids), 3), np.float64)
    times = np.full(len(point_ids), np.nan, np.float64)
    by_scan = {store.meta['scan_index']: store for store in stores}
    scans = point_ids >> 48
    rows = point_ids & ((1 << 48)-1)
    for scan in np.unique(scans):
        ids = np.flatnonzero(scans == scan)
        store = by_scan.get(int(scan))
        if store is None:
            raise ValueError('Visibility index refers to an unknown scan')
        ordered = ids[np.argsort(rows[ids], kind='stable')]
        sorted_rows = rows[ordered]
        pose = scan_pose(store.meta['metadata'])
        world[ordered] = pose.to_world(store.read_rows('xyz_local', sorted_rows))
        field = fields.get(int(scan)) if isinstance(fields, dict) else fields
        if field and field in store.meta['arrays']:
            values = np.asarray(store.read_rows(field, sorted_rows), np.float64)
            times[ordered] = np.where(np.isfinite(values), values, np.nan)
    return world, times


def _slam_time_field(stores):
    names = ('timeStamp', 'timestamp', 'time', 'gpsTime', 'gps_time')
    matches = []
    for store in stores:
        matches.append(next((name for name in names if name in store.meta['arrays']), None))
    return matches if all(matches) else None


def slam_observation_weight(distance, incidence_angle, radial_position, config):
    """Combine SAM confidence with distance, surface incidence and image-centre quality.

    The configurable factors favour closer points, rays more perpendicular to
    a surface normal, and pixels near the image centre.  They intentionally
    remain multiplicative so a weak geometric factor cannot increase SAM
    evidence.
    """
    distance = np.asarray(distance, np.float64)
    incidence_angle = np.asarray(incidence_angle, np.float64)
    radial_position = np.asarray(radial_position, np.float64)
    distance_factor = float(config['slam_weight_distance_scale']) / (distance + float(config['slam_weight_distance_scale']))
    incidence_factor = np.clip(np.cos(incidence_angle), 0., 1.) ** float(config['slam_weight_incidence_power'])
    radial_factor = 1. / (1.+np.maximum(radial_position, 0.)) ** float(config['slam_weight_radial_power'])
    return (distance_factor*incidence_factor*radial_factor).astype(np.float32)


def _slam_view_image(reader, item, camera, view, image_cache):
    camera_model, image, mask, image_hash, mask_hash = image_assets(reader, item, image_cache)
    if isinstance(view, Camera):
        return image, (mask > 0 if mask is not None else np.ones(image.shape[:2], bool)), image_hash, mask_hash, {
            'id': f'image_{item["index"]}_native', 'kind': camera_model.kind, 'params': camera_model.params}
    rendered, valid = render_view(image, mask, camera_model, view)
    return rendered, valid, image_hash, mask_hash, {**asdict(view), 'image_index': item['index'], 'mask_sha256': mask_hash}


def _slam_batches(values, columns):
    """Yield dense class vectors after per-station, per-voxel maxima."""
    if not values:
        return
    dtype = np.dtype([('key', 'V24'), ('column', '<i4'), ('weight', '<f4')])
    rows = np.empty(len(values), dtype=dtype)
    for index, ((key, column), weight) in enumerate(values.items()):
        rows[index] = (np.void(key), column, weight)
    rows.sort(order=('key', 'column'))
    for start in range(0, len(rows), 50_000):
        block = rows[start:start+50_000]
        keys = block['key']
        weights = np.zeros((len(block), columns), np.float32)
        weights[np.arange(len(block)), block['column']] = block['weight']
        yield keys, weights


def _slam_scene(config, preset, cameras, image_by_index, reader, index, stores, visibility_cache,
                mask_cache, image_cache, model_hash, factory, control):
    """Choose one dataset-wide SLAM scene without any scanner-return heuristic.

    SLAM cameras form one trajectory, therefore a single dataset scene is less
    surprising than changing the visibility range between neighbouring stations.
    Explicit scene_mode continues to take precedence.
    """
    requested = config['scene_mode']
    if requested in ('indoor', 'outdoor'):
        return requested, {'decision': 'manual', 'scope': 'dataset', 'sky_fraction': None,
                           'overhead_view_fraction': None, 'range_m': None}
    overrides = [(str(key), value) for key, value in config.get('scene_overrides', {}).items()
                 if value in ('indoor', 'outdoor')]
    if overrides:
        key, scene = sorted(overrides)[0]
        return scene, {'decision': 'override', 'scope': 'dataset', 'override_key': key,
                       'sky_fraction': None, 'overhead_view_fraction': None, 'range_m': None}
    sky = next((item for item in preset.get('auxiliary', []) if item.get('id') == 'sky'), None)
    provisional = {**config, 'slam_max_range': config.get('slam_max_range') or
                   max(config['slam_max_range_indoor'], config['slam_max_range_outdoor'])}
    sky_pixels = sky_total = overhead_views = evaluated = 0
    image_count = len(cameras)
    for camera in cameras:
        control.checkpoint()
        view = camera.views(config['profile'], config['view_size'], config['view_fov'])[-1]
        item = image_by_index[camera.image_index]
        rendered, valid, image_hash, _, info = _slam_view_image(reader, item, camera.camera, view, image_cache)
        candidates = _slam_candidate_loader(index, camera.pose, provisional, control.checkpoint)
        # The scene pass has one view per camera.  Pass the loader separately
        # so cache hits keep its large candidate set entirely unloaded.
        visibility, _ = visibility_cache.visibility_for_view(index, camera.pose, view, provisional,
            intrinsics=camera.camera.params, view=info, checkpoint=control.checkpoint,
            candidates_loader=candidates)
        evaluated += 1
        if hasattr(control, 'event'):
            control.event('progress', phase='slam_scene', fraction=evaluated/max(1, image_count),
                          image=camera.image_index, image_count=image_count)
        if sky is not None:
            scores, _ = segment_view(rendered, valid, [PromptClass('sky', tuple(sky['prompts']))], factory,
                mask_cache, image_hash, {**info, 'slam_scene': True}, model_hash, config['image_scale'],
                config['tile_size'], config['tile_overlap'], control.checkpoint)
            # SAM scores are stored at ``image_scale``.  Count sky evidence in
            # that coordinate system so the score and validity masks align.
            score_valid = resize_mask_nearest(valid, scores[0].shape)
            sky_pixels += int(((scores[0] >= config['scene_sky_confidence']) & score_valid).sum())
            sky_total += int(score_valid.sum())
        # Only z-buffer-visible points contribute to the overhead decision.
        if len(visibility.point_indices):
            world, _ = _slam_point_values(stores, visibility.point_indices)
            overhead_views += int(np.any((world[:, 2]-camera.pose.translation[2]) > .5))
    sky_fraction = sky_pixels/max(1, sky_total)
    overhead_fraction = overhead_views/max(1, evaluated)
    indoor = overhead_fraction >= .20
    # Positive sky evidence is required for outdoor classes.  If neither signal
    # is decisive, retain the project's conservative indoor default.
    scene = 'indoor' if indoor else ('outdoor' if sky_fraction >= config['scene_sky_outdoor'] else 'indoor')
    return scene, {'decision': 'automatic', 'scope': 'dataset', 'sky_fraction': float(sky_fraction),
                   'overhead_view_fraction': float(overhead_fraction), 'views_evaluated': evaluated,
                   'method': 'sky masks plus visible points above cameras; no upper-hemisphere return rule'}


def _slam_candidate_loader(index, pose, config, checkpoint):
    """Load one camera's range candidates only when a visibility cache misses."""
    loaded = None
    maximum = float(config.get('slam_max_range') or config.get('slam_max_range_indoor', 20.))

    def load():
        nonlocal loaded
        if loaded is None:
            values = index.query_range(pose.translation, maximum, checkpoint)
            loaded = values, pose.from_world(values['xyz'])
        return loaded
    return load


def _slam_observed_path(work, scan):
    """Keep run-owned visibility masks separate from reusable geometry caches."""
    return work/'observed'/f'observed_{int(scan)}.npy'


def _open_slam_observed(work, stores):
    """Open durable per-scan observed masks, creating only missing or invalid files."""
    observed = {}
    for store in stores:
        scan = store.meta['scan_index']
        shape = (store.meta['record_count'],)
        path = _slam_observed_path(work, scan)
        values = None
        if path.exists():
            try:
                values = np.lib.format.open_memmap(path, mode='r+')
                if values.dtype != np.dtype(bool) or values.shape != shape:
                    values._mmap.close()
                    values = None
            except (OSError, ValueError):
                values = None
        if values is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            values = np.lib.format.open_memmap(path, mode='w+', dtype=bool, shape=shape)
            values[...] = False
            values.flush()
        observed[scan] = values
    return observed


def _slam_station_state_path(work, station_id):
    return work/'stations'/f'station_{int(station_id):04d}.json'


def _load_slam_station_state(work, station_id):
    """Return a completed station's durable transfer metadata, if valid."""
    try:
        state = json.loads(_slam_station_state_path(work, station_id).read_text(encoding='utf-8'))
        if (state.get('version') != 1 or int(state.get('station')) != int(station_id)
                or not isinstance(state.get('views'), list)
                or not isinstance(state.get('observed_counts'), dict)):
            return None
        return state
    except (OSError, ValueError, TypeError):
        return None


def _flush_slam_observed(observed):
    """Make all visibility bits durable before a station is marked completed."""
    counts = {}
    for scan, values in observed.items():
        values.flush()
        counts[str(scan)] = int(np.count_nonzero(values))
    return counts


def _slam_observed_matches_state(observed, state):
    """Detect deleted or truncated masks even when transfer metadata still exists."""
    expected = state.get('observed_counts', {})
    return all(str(scan) in expected and int(np.count_nonzero(values)) >= int(expected[str(scan)])
               for scan, values in observed.items())


def _save_slam_station_state(work, station, views, observed, *, camera_count, voxel_class_observations):
    """Persist report values before VoteStore makes the station resumable."""
    atomic_json(_slam_station_state_path(work, station.id), {
        'version': 1,
        'station': station.id,
        'views': views,
        'usable_views': sum(item['used'] for item in views),
        'camera_count': camera_count,
        'voxel_class_observations': voxel_class_observations,
        'observed_counts': _flush_slam_observed(observed),
    })


def _mark_slam_observed(observed, point_indices):
    for point in point_indices:
        scan, row = unpack_point_id(point)
        observed[scan][row] = True


def _reconstruct_slam_station_visibility(station, cameras, image_by_index, reader, image_cache, index,
                                         visibility_cache, config, control, observed):
    """Restore legacy completed-station visibility without invoking SAM."""
    views = []
    for camera_index in station.camera_indices:
        camera = cameras[camera_index]
        item = image_by_index[camera.image_index]
        candidates = _slam_candidate_loader(index, camera.pose, config, control.checkpoint)
        for _, view in iter_camera_views([camera], config['profile'], config['view_size'], config['view_fov']):
            control.checkpoint()
            rendered, _, _, _, view_info = _slam_view_image(reader, item, camera.camera, view, image_cache)
            visibility, visibility_hit = visibility_cache.visibility_for_view(
                index, camera.pose, view, config, intrinsics=camera.camera.params, view=view_info,
                checkpoint=control.checkpoint, image=rendered, diagnostics_directory=None,
                candidates_loader=candidates)
            if len(visibility.point_indices):
                _mark_slam_observed(observed, visibility.point_indices)
                views.append({'image': camera.image_index, 'station': station.id, 'used': True,
                              'visible_points': int(len(visibility.point_indices)),
                              'visibility_cache': bool(visibility_hit)})
            else:
                views.append({'image': camera.image_index, 'station': station.id, 'used': False,
                              'reason': 'no_visible_points'})
    return views


def _slam_voxel_origin(cameras, stores):
    """Choose a nearby world-space origin without assuming a scanner origin."""
    if cameras:
        return cameras[0].pose.translation.tolist()
    for store in stores:
        bounds = store.meta.get('bounds_local')
        if bounds and all(axis in bounds for axis in ('min', 'max')):
            return np.asarray(bounds['min'], np.float64).tolist()
    return scan_pose(stores[0].meta['metadata']).translation.tolist()


def execute_slam(config, source, output, cache, inspection, store_report, stores, factory, control, started):
    """Run the mobile path without changing the static TLS processing path."""
    preset = json.loads(Path(config['preset']).read_text(encoding='utf-8'))
    semantic_run = config['processing_mode'] == 'classify_and_filter'
    errors, preset_warnings = validate_preset(preset, require_active_classes=semantic_run)
    if errors:
        raise ValueError('Invalid preset: '+'; '.join(errors))
    if not semantic_run:
        raise ValueError('SLAM mode requires semantic classification')
    all_classes = [item for item in preset['classes'] if item.get('enabled', True)]
    if not all_classes:
        raise ValueError('No active classes')
    cameras = build_camera_list(inspection)
    if not cameras:
        raise InputNotClassifiable('No decodable SLAM camera with a valid pose and camera model')
    stations, membership, ordering = cluster_stations(cameras, config['slam_station_spacing'])
    planned_views = sum(len(camera.views(config['profile'], config['view_size'], config['view_fov']))
                        for camera in cameras)
    control.event('slam_plan', cameras=len(cameras), views_total=planned_views, stations=len(stations))
    reflection_masks_requested = bool(config['mirror_analysis'] or config['glass_analysis'])
    config = {**config, 'voxel_origin': _slam_voxel_origin(cameras, stores),
              '_slam': True, '_shared_sam': False, 'mirror_analysis': False}
    model_hash, input_hash = file_hash(Path(config['model'])), store_report['source_sha256']
    identity = identity_hash({'pipeline_revision': 'slam-v1', 'source': input_hash, 'model': model_hash,
                              'preset': preset, 'config': {key: value for key, value in config.items() if key not in ('output', 'cache')}})
    manifest = {**config, 'pipeline_revision': 'slam-v1', 'identity': identity, 'input_sha256': input_hash,
                'model_sha256': model_hash, 'resolved_preset': preset,
                'versions': {name: version(name) for name in ('numpy', 'pye57', 'Pillow', 'torch', 'torchvision', 'sam3', 'laspy')}}
    previous = output/'run_config.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8')).get('identity') != identity:
        raise ValueError('This output directory belongs to a different run; choose a new folder')
    atomic_json(previous, manifest)
    rows = legend_rows(preset)
    write_legends(output, rows)
    atomic_json(output/'scan_index.json', {str(store.meta['scan_index']): {'guid': store.meta['metadata'].get('guid'),
                'name': store.meta['metadata'].get('name')} for store in stores})
    work = cache/'votes'/identity
    work.mkdir(parents=True, exist_ok=True)
    votes = VoteStore(work/'votes.sqlite', station_tracking=True)
    mask_cache, image_cache = MaskCache(cache/'seg'), cache/'geometry'/input_hash/'images'
    control.event('phase_started', phase='slam_index')
    index = build_spatial_index(stores, cache/'slam_index'/input_hash, input_hash, checkpoint=control.checkpoint)
    control.event('progress', phase='slam_index', fraction=1.)
    control.event('phase_finished', phase='slam_index')
    visibility_cache = VisibilityCache(cache, input_hash)
    code_map = preset.get('code_map', {})
    codes = [int(code_map.get(str(item['code']), item['code'])) for item in all_classes]
    by_id = {item['id']: codes[position] for position, item in enumerate(all_classes)}
    time_fields = _slam_time_field(stores)
    time_possible = bool(config['slam_time_filter'] and time_fields and all(camera.timestamp is not None for camera in cameras))
    time_field_by_scan = {store.meta['scan_index']: field for store, field in zip(stores, time_fields or [None]*len(stores))}
    LOG.info('SLAM timestamp filtering requested=%s possible=%s', config['slam_time_filter'], time_possible)
    input_warnings = [item if isinstance(item, dict)
                      else {'code': 'input_mode_warning', 'detail': str(item)}
                      for item in inspection['input_mode'].get('warnings', [])]
    warnings = [*input_warnings,
                *({'code': 'preset_warning', 'detail': item} for item in preset_warnings)]
    warnings.append({'code': 'slam_reflection_disabled', 'detail': 'Reflection class 68 is disabled for mobile SLAM input.'})
    LOG.info('SLAM class 68 skipped because mobile inputs have no reliable scan origin')
    if reflection_masks_requested:
        warnings.append({'code': 'slam_reflection_masks_exclude_labels',
                         'detail': 'Mirror and glass masks exclude ordinary semantic labels; class 68 remains disabled.'})
    if config['slam_time_filter'] and not time_possible:
        warnings.append({'code': 'slam_time_filter_unavailable', 'detail': 'Point and image timestamps are both required.'})
    image_by_index = {item['index']: item for item in inspection['images']}
    timings, station_results = {}, []
    view_results = []
    observed = _open_slam_observed(work, stores)
    with E57Reader(source) as reader:
        control.event('phase_started', phase='slam_scene')
        scene, scene_info = _slam_scene(config, preset, cameras, image_by_index, reader, index, stores,
            visibility_cache, mask_cache, image_cache, model_hash, factory, control)
        control.event('progress', phase='slam_scene', fraction=1., image_count=len(cameras))
        control.event('phase_finished', phase='slam_scene')
        effective_range = config['slam_max_range'] or config[f'slam_max_range_{scene}']
        config = {**config, 'slam_max_range': effective_range}
        scene_info['range_m'] = effective_range
        control.event('slam_plan', cameras=len(cameras), views_total=planned_views,
                      stations=len(stations), scene=scene)
        selected = select_classes(preset, scene, config['building_enclosure'])
        selected_ids = {item['id'] for item in selected}
        active_columns = {index for index, item in enumerate(all_classes) if item['id'] in selected_ids}
        if not active_columns:
            raise ValueError('No enabled classes for this SLAM scene mode')
        reflection_prompts = []
        if reflection_masks_requested:
            _, reflection_prompts, _ = reflection_prompt_classes({**config, 'glass_analysis': config['glass_analysis']}, preset)
        view_counts = {camera_index: len(camera.views(config['profile'], config['view_size'], config['view_fov']))
                       for camera_index, camera in enumerate(cameras)}
        total_views = sum(view_counts[camera_index] for station in stations for camera_index in station.camera_indices)
        completed_views = 0
        control.event('phase_started', phase='slam_segmentation')
        for station in stations:
            control.checkpoint()
            if votes.completed(station.id, identity):
                state = _load_slam_station_state(work, station.id)
                if state is None or not _slam_observed_matches_state(observed, state):
                    restored_views = _reconstruct_slam_station_visibility(
                        station, cameras, image_by_index, reader, image_cache, index, visibility_cache,
                        config, control, observed)
                    _save_slam_station_state(work, station, restored_views, observed,
                                             camera_count=len(station.camera_indices), voxel_class_observations=0)
                    state = _load_slam_station_state(work, station.id)
                view_results.extend(state['views'])
                station_results.append({'station': station.id, 'resumed': True, 'views': state['usable_views'],
                                        'camera_count': state['camera_count'],
                                        'voxel_class_observations': state['voxel_class_observations']})
                completed_views += sum(view_counts[camera_index] for camera_index in station.camera_indices)
                continue
            station_values = {}
            station_views = []
            for camera_index in station.camera_indices:
                camera = cameras[camera_index]
                item = image_by_index[camera.image_index]
                candidates = _slam_candidate_loader(index, camera.pose, config, control.checkpoint)
                for view_number, (_, view) in enumerate(iter_camera_views([camera], config['profile'], config['view_size'], config['view_fov']), 1):
                    control.checkpoint()
                    rendered, valid_image, image_hash, mask_hash, view_info = _slam_view_image(reader, item, camera.camera, view, image_cache)
                    prompts = prompt_classes([item for item in all_classes if item['id'] in selected_ids], config)
                    selected_indices = [index for index, item in enumerate(all_classes) if item['id'] in selected_ids]
                    scores, cache_stats = segment_view(rendered, valid_image, prompts+reflection_prompts, factory, mask_cache, image_hash,
                        view_info, model_hash, config['image_scale'], config['tile_size'], config['tile_overlap'], control.checkpoint)
                    label_config = {**config, 'boundary_margin': config['boundary_margin']+config['slam_boundary_margin_extra']}
                    labels, confidence = evidence_map(scores[:len(prompts)], [item for item in all_classes if item['id'] in selected_ids], label_config)
                    visibility, visibility_hit = visibility_cache.visibility_for_view(index, camera.pose, view, config,
                        intrinsics=camera.camera.params, view=view_info, checkpoint=control.checkpoint,
                        progress=lambda phase, hit: control.event('progress', phase=phase, station=station.id, image=camera.image_index, cache=hit),
                        image=rendered, diagnostics_directory=output/'slam_diagnostics', candidates_loader=candidates)
                    if not len(visibility.point_indices):
                        station_views.append({'image': camera.image_index, 'station': station.id, 'used': False,
                                              'reason': 'no_visible_points'})
                        completed_views += 1
                        control.event('progress', phase='slam_label_transfer', fraction=completed_views/max(1, total_views),
                                      station=station.id, station_count=len(stations), image=camera.image_index,
                                      image_count=len(cameras), view=view_number, view_count=view_counts[camera_index],
                                      cache=cache_stats, visible_points=0)
                        control.event('progress', phase='slam_segmentation', fraction=completed_views/max(1, total_views),
                                      station=station.id, station_count=len(stations), image=camera.image_index,
                                      image_count=len(cameras), view=view_number, view_count=view_counts[camera_index])
                        continue
                    _mark_slam_observed(observed, visibility.point_indices)
                    # Point IDs retain original block/record provenance; load only the visible source rows.
                    world, point_times = _slam_point_values(stores, visibility.point_indices,
                                                            time_field_by_scan if time_possible else None)
                    pixels = np.floor(visibility.pixels*np.asarray((labels.shape[1]/rendered.shape[1], labels.shape[0]/rendered.shape[0]))).astype(np.int64)
                    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < labels.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < labels.shape[0]))
                    local_labels = np.full(len(pixels), -1, np.int16)
                    local_confidence = np.zeros(len(pixels), np.float32)
                    local_labels[inside] = labels[pixels[inside, 1], pixels[inside, 0]]
                    local_confidence[inside] = confidence[pixels[inside, 1], pixels[inside, 0]]
                    if reflection_prompts:
                        reflection = np.maximum.reduce(scores[len(prompts):]) >= config['mirror_mask_confidence']
                        inside &= ~reflection[pixels[:, 1].clip(0, reflection.shape[0]-1), pixels[:, 0].clip(0, reflection.shape[1]-1)]
                    if time_possible:
                        inside &= np.isfinite(point_times) & (np.abs(point_times-camera.timestamp) <= config['slam_time_window'])
                    weights = local_confidence*slam_observation_weight(visibility.distance, visibility.incidence_angle,
                                                                          visibility.radial_position, config)
                    keys = voxel_keys(world, np.asarray(config['voxel_origin']), config['voxel_size'])
                    for local_class, global_class in enumerate(selected_indices):
                        take = inside & (local_labels == local_class) & (weights > 0)
                        for key, weight in zip(keys[take], weights[take]):
                            marker = (bytes(key), global_class)
                            station_values[marker] = max(station_values.get(marker, 0.), float(weight))
                    station_views.append({'image': camera.image_index, 'station': station.id, 'used': True,
                                          'visible_points': int(len(visibility.point_indices)),
                                          'visibility_cache': bool(visibility_hit)})
                    completed_views += 1
                    control.event('progress', phase='slam_label_transfer', fraction=completed_views/max(1, total_views),
                                  station=station.id, station_count=len(stations), image=camera.image_index,
                                  image_count=len(cameras), view=view_number, view_count=view_counts[camera_index],
                                  cache=cache_stats, visible_points=len(visibility.point_indices))
                    control.event('progress', phase='slam_segmentation', fraction=completed_views/max(1, total_views),
                                  station=station.id, station_count=len(stations), image=camera.image_index,
                                  image_count=len(cameras), view=view_number, view_count=view_counts[camera_index])
                del candidates
            _save_slam_station_state(work, station, station_views, observed,
                                     camera_count=len(station.camera_indices), voxel_class_observations=len(station_values))
            votes.add_scan(station.id, identity, _slam_batches(station_values, len(all_classes)), codes, station_id=station.id)
            view_results.extend(station_views)
            station_results.append({'station': station.id, 'views': sum(item['used'] for item in station_views),
                                    'camera_count': len(station.camera_indices), 'voxel_class_observations': len(station_values)})
    control.event('phase_finished', phase='slam_segmentation')
    control.event('phase_started', phase='voting')
    priority = {'code': by_id['stairs'], 'over': [by_id[key] for key in config['stairs_priority_over'] if key in by_id],
                'confidence': config['stairs_min_confidence'], 'min_share': config['stairs_priority_min_share']} if config['stairs_priority'] and 'stairs' in by_id else None
    families = [[by_id[name] for name in family if name in by_id] for family in config['vote_families']]
    votes.decide(config['min_observations'], config['min_winning_share'], codes, control.checkpoint, priority, families)
    surface_ties = building = stairs = components = vegetation = geometry_rules = segment_vote = shape_rules = scene_consistency = geometry_fill = None
    outline = None
    geometry_planes = None
    terrain = None
    if config['surface_tie_break']:
        surface_ties = votes.resolve_surface_ties(by_id, np.asarray(config['voxel_origin']), config['voxel_size'],
            config['surface_tie_share'], config['surface_normal_radius'], config['min_observations'],
            [float(station.position[2]) for station in stations], control.checkpoint)
    if config.get('scene_outline', True) or config['building_enclosure']:
        outline = votes.building_outline([by_id[key] for key in ('building', 'wall') if key in by_id], config['voxel_size'],
                                         config['building_boundary_cell_size'], config['building_boundary_min_height'],
                                         config['building_boundary_max_cells'])
    if config['building_enclosure'] and {'ground', 'floor'}.issubset(by_id):
        building = votes.split_ground_floor_by_building(by_id['ground'], by_id['floor'],
            [by_id[key] for key in ('building', 'wall') if key in by_id], config['voxel_size'],
            config['building_boundary_cell_size'], config['building_boundary_min_height'],
            config['building_boundary_max_cells'], control.checkpoint, outline=outline)
    segment_started = time.monotonic()
    segment_vote = apply_segment_vote(votes, by_id, np.asarray(config['voxel_origin']), config['voxel_size'], config,
                                      control.checkpoint, active_codes={row['code'] for row in rows})
    timings['segment_vote'] = time.monotonic()-segment_started
    rule_context = geometry_scan_context(
        preset, config, {station.id: scene for station in stations},
        positions={station.id: station.position for station in stations})
    if config['geometry_rules']:
        terrain = work/'terrain.npz'
        geometry_rules = apply_geometry_rules(votes, by_id, np.asarray(config['voxel_origin']), config['voxel_size'],
                                              config, terrain, control.checkpoint,
                                              active_codes={row['code'] for row in rows}, scan_context=rule_context, outline=outline)
    if config['stairs_geometry_check'] and 'stairs' in by_id:
        stairs = votes.reject_planar_stairs(by_id['stairs'], np.asarray(config['voxel_origin']), config['voxel_size'],
            config['stairs_plane_tolerance'], config['stairs_planar_fraction'], config['stairs_min_rise'],
            config['seed'], codes, control.checkpoint, minimum_voxels=config['stairs_min_voxels'],
            minimum_levels=config['stairs_min_levels'], level_tolerance=config['stairs_level_tolerance'],
            rise_minimum=config['stairs_step_rise_min'], rise_maximum=config['stairs_step_rise_max'],
            tread_minimum=config['stairs_step_tread_min'], tread_maximum=config['stairs_step_tread_max'],
            floor_code=by_id.get('floor'), floor_ring_radius=config['stairs_floor_ring_radius'],
            floor_level_tolerance=config['stairs_floor_level_tolerance'],
            underside_minimum_voxels=config['stairs_underside_min_voxels'],
            underside_cluster_size=config['stairs_underside_cluster_size'],
            underside_maximum_gap=config['stairs_underside_max_gap'])
    scene_started = time.monotonic()
    scene_consistency = enforce_scene_consistency(votes, all_classes, by_id, rule_context,
                                                  np.asarray(config['voxel_origin']), config['voxel_size'], config,
                                                  {row['code'] for row in rows}, control.checkpoint, outline=outline)
    timings['scene_context'] = ((geometry_rules or {}).get('scan_context', {}).get('duration_s', 0.)
                                if geometry_rules else scene_consistency['duration_s'])
    timings['scene_consistency'] = time.monotonic()-scene_started
    if config['geometry_fill'] or config['shape_rules']:
        fill_started = time.monotonic()
        geometry_planes = work/'planes.npz'
        extracted_planes = extract_planes(votes, by_id, np.asarray(config['voxel_origin']), config['voxel_size'],
                                          config, geometry_planes, control.checkpoint)
        timings['geometry_fill_planes'] = time.monotonic()-fill_started
        if config['geometry_fill']:
            geometry_fill = extracted_planes
    shape_started = time.monotonic()
    shape_rules = apply_shape_rules(votes, by_id, np.asarray(config['voxel_origin']), config['voxel_size'], config,
                                    control.checkpoint, active_codes={row['code'] for row in rows}, scan_context=rule_context,
                                    planes_path=geometry_planes, outline=outline)
    timings['shape_rules'] = time.monotonic()-shape_started
    ground_dtm = None
    if config['ground_dtm']:
        dtm_started = time.monotonic()
        if outline is None:
            barrier_codes = [by_id[key] for key in ('building', 'wall') if key in by_id]
            outline = votes.building_outline(barrier_codes, config['voxel_size'],
                                             config['building_boundary_cell_size'], config['building_boundary_min_height'],
                                             config['building_boundary_max_cells'])
        dtm_path = work/'dtm.npz'
        dtm_build_report = build_tin_dtm(
            votes, by_id, outline, np.asarray(config['voxel_origin']), config['voxel_size'],
            config, target_path=dtm_path, scan_context=rule_context, checkpoint=control.checkpoint
        )
        with TinDTM(dtm_path) as tin:
            band_report = apply_ground_dtm_band(
                votes, by_id, tin, outline, np.asarray(config['voxel_origin']), config['voxel_size'],
                config, active_codes={row['code'] for row in rows}, checkpoint=control.checkpoint
            )
        timings['ground_dtm'] = time.monotonic()-dtm_started
        ground_dtm = {
            **band_report,
            'dtm': dtm_build_report,
            'artifact': str(dtm_path),
            'path': str(dtm_path),
        }
    if config['component_cleanup']:
        rules = []
        if 'cable' in by_id:
            rules.append((by_id['cable'], config['cable_component_min_voxels'], config['cable_component_min_span']))
        if 'electrical' in by_id:
            rules.append((by_id['electrical'], config['electrical_component_min_voxels'], config['electrical_component_min_span']))
        components = votes.remove_small_components(config['voxel_size'], config['component_min_voxels'], rules, control.checkpoint,
                                                   component_min_cells=config['component_min_cells'],
                                                   feature_cell_size=config['geometry_cell_size'],
                                                   sparse_maximum_voxels_per_cell=config['component_sparse_max_voxels_per_cell'],
                                                   outdoor_codes=[by_id[name] for name in ('building', 'ground', 'low_vegetation', 'tree', 'road', 'sidewalk') if name in by_id])
    if config['vegetation_base_to_ground'] and 'low_vegetation' in by_id and 'ground' in by_id:
        vegetation = votes.vegetation_bases_to_ground(by_id['low_vegetation'], by_id['ground'], config['voxel_size'],
            config['vegetation_ground_cell_size'], config['vegetation_ground_band'], control.checkpoint)
    # Origin-based fills and direct-neighbour transfer are disabled.  The writer
    # additionally restores class 1 for every point never marked visible.
    votes.finalize_station_ids()
    control.event('phase_finished', phase='voting')
    control.event('phase_started', phase='output')
    suffix, path = 'classified', output/f'{source.stem}_classified.{config["format"]}'
    if path.exists():
        raise FileExistsError('Completed output exists; choose a new run directory')
    writer = write_e57 if config['format'] == 'e57' else write_cloud
    progress = lambda phase, fraction, scan: control.event('progress', phase=phase, fraction=fraction, scan=scan)
    never_visible = 0
    for store in stores:
        values = observed[store.meta['scan_index']]
        values.flush()
        for start, xyz, valid in store.chunks():
            never_visible += int((~values[start:start+len(xyz)] & valid).sum())
    total_valid = sum(store.meta['valid_count'] for store in stores)
    force_unclassified = lambda store, start, valid: ~np.asarray(observed[store.meta['scan_index']][start:start+len(valid)])[valid]
    transfer_station_positions = {station.id: np.asarray([cameras[index].pose.translation for index in station.camera_indices], np.float64)
                                  for station in stations}
    export_ground_dtm = work/'dtm.npz' if config['ground_dtm'] and (work/'dtm.npz').is_file() else None
    export_ground_alt = work/'ground_alternatives.npz' if config['ground_dtm'] and (work/'ground_alternatives.npz').is_file() else None
    if writer is write_e57:
        result = writer(path, source, stores, votes, np.asarray(config['voxel_origin']), config['voxel_size'], [row['code'] for row in rows], (),
                        control.checkpoint, progress, output_density=config['output_density'], class_transfer_radius=config['class_transfer_radius'],
                        class_transfer_range_factor=config['class_transfer_range_factor'],
                        class_transfer_radius_max=config['class_transfer_radius_max'], transfer_station_positions=transfer_station_positions,
                        geometry_planes=geometry_planes if config['geometry_fill'] else None, terrain=terrain, terrain_ground_tolerance=config['terrain_ground_tolerance'],
                        terrain_fill=config['terrain_fill'] and any(row['code'] == 2 for row in rows),
                        tree_crowns=work/'tree_crowns.npz' if config['tree_crown_fill'] else None,
                        tree_fill=config['tree_crown_fill'],
                        station_ids=True, station_count=len(stations), force_unclassified=force_unclassified,
                        terrain_outdoor_scans={store.meta['scan_index'] for store in stores} if scene == 'outdoor' else set(),
                        ground_dtm=export_ground_dtm, ground_dtm_alternatives=export_ground_alt,
                        ground_dtm_band=config['ground_dtm_band'], export_index_cache_gb=config['export_index_cache_gb'],
                        export_index_max_tile_points=config['export_index_max_tile_points'])
    else:
        result = writer(path, stores, votes, np.asarray(config['voxel_origin']), config['voxel_size'],
                        inspection['metadata'].get('coordinateMetadata', ''), [row['code'] for row in rows], (), control.checkpoint, progress,
                        output_density=config['output_density'], class_transfer_radius=config['class_transfer_radius'],
                        class_transfer_range_factor=config['class_transfer_range_factor'],
                        class_transfer_radius_max=config['class_transfer_radius_max'], transfer_station_positions=transfer_station_positions,
                        geometry_planes=geometry_planes if config['geometry_fill'] else None, terrain=terrain,
                        terrain_ground_tolerance=config['terrain_ground_tolerance'],
                        terrain_fill=config['terrain_fill'] and any(row['code'] == 2 for row in rows),
                        tree_crowns=work/'tree_crowns.npz' if config['tree_crown_fill'] else None,
                        tree_fill=config['tree_crown_fill'],
                        station_ids=True, station_count=len(stations), force_unclassified=force_unclassified,
                        terrain_outdoor_scans={store.meta['scan_index'] for store in stores} if scene == 'outdoor' else set(),
                        ground_dtm=export_ground_dtm, ground_dtm_alternatives=export_ground_alt,
                        ground_dtm_band=config['ground_dtm_band'], export_index_cache_gb=config['export_index_cache_gb'],
                        export_index_max_tile_points=config['export_index_max_tile_points'])
    control.event('phase_finished', phase='output')
    result.update(point_reduction=config['point_reduction'], output_density=config['output_density'],
                  computed_points=sum(selected_count(store, 'reduced') for store in stores), output_points=result['record_count'], format=config['format'])
    report = {'schema_version': 1, 'input': inspection['source'], 'input_sha256': input_hash, 'input_mode': inspection['input_mode'],
              'output': result, 'scans': [{'scan': item['station'], 'scene': scene} for item in station_results],
              'stations': station_results, 'virtual_stations': len(stations), 'station_ordering': ordering,
              'views_used': sum(item['used'] for item in view_results),
              'views_skipped': [item for item in view_results if not item['used']],
              'views': view_results, 'never_visible_share': never_visible/max(1, total_valid),
              'class_1_share': result['class_counts'].get('1', 0)/max(1, result['record_count']), 'warnings': warnings,
              'slam': {'station_spacing_m': config['slam_station_spacing'], 'visibility_cache_hits': visibility_cache.hits,
                       'visibility_recomputations': visibility_cache.recomputations, 'time_filter_requested': config['slam_time_filter'],
                       'time_filter_possible': time_possible, 'time_window_s': config['slam_time_window'],
                       'scene': scene_info, 'visibility_parameters': {key: config[key] for key in
                           ('slam_max_range', 'slam_max_range_indoor', 'slam_max_range_outdoor', 'slam_depth_resolution',
                            'slam_splat_max_px', 'slam_depth_tol_abs', 'slam_depth_tol_rel')},
                       'median_visible_points_per_view': float(np.median([item['visible_points'] for item in view_results if item['used']])) if any(item['used'] for item in view_results) else 0.,
                       'weighting': {'distance_scale_m': config['slam_weight_distance_scale'],
                                     'incidence_power': config['slam_weight_incidence_power'], 'radial_power': config['slam_weight_radial_power'],
                                     'description': 'SAM confidence multiplied by distance, incidence and image-centre factors.'},
                       'never_visible_policy': 'No direct-neighbour or plane-fill propagation is used in SLAM mode.'},
              'sam_calls': 0,
              'stairs_priority_overrides': votes.priority_overrides, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'surface_ties': surface_ties, 'building_enclosure': building, 'stairs_geometry': stairs,
              'component_cleanup': components, 'vegetation_base_to_ground': vegetation, 'geometry_rules': geometry_rules,
              'segment_vote': segment_vote, 'shape_rules': shape_rules, 'ground_dtm': ground_dtm,
              'scene_consistency': scene_consistency,
              'scene_outline': (scene_consistency or {}).get('scene_outline', (geometry_rules or {}).get('scan_context', {}).get('scene_outline')),
              'scene_resolution': {'scene': {scene: len(station_results)},
                                   'scene_detail': {scene: len(station_results)}, 'source': {'data': len(station_results)},
                                   'conflicts': [], 'unresolved': []},
              'scene_audit': scene_audit([{'scan': store.meta['scan_index'], 'scene': scene} for store in stores],
                                         result.get('scan_class_counts', {}), preset, code_map),
              'total_seconds': time.monotonic()-started}
    write_report(output, report, rows)
    votes.close()
    for values in observed.values():
        values.flush()
        values._mmap.close()
    observed.clear()
    del values
    cleanup = cleanup_run_cache(cache, input_hash, identity, mask_cache, config['cache_cleanup'])
    report['cache_cleanup'] = {key: value for key, value in cleanup.items() if key != 'incomplete'}
    write_report(output, report, rows)
    control.event('finished', output=str(path), report=str(output/'report.html'))
    return report


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
    geometry_rules_report = None
    segment_vote_report = None
    shape_rules_report = None
    ground_dtm_report = None
    scene_consistency_report = None
    scene_outline_report = None
    terrain = None
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
        inspection = inspect_file(source, config=config)
        atomic_json(output/"inspect_report.json", inspection)
        atomic_text(output/"inspect_report.md", markdown(inspection))
        input_mode = inspection['input_mode']
        control.event("input_mode", **input_mode)
        for scan in inspection['scans']:
            if scan.get('pose_status') == 'invalid':
                scan_pose(scan['metadata'])
        missing_pose_scans = [scan for scan in inspection['scans'] if scan.get('pose_status') == 'missing']
        if input_mode['mode'] == 'static' and missing_pose_scans and len(missing_pose_scans) == len(inspection['scans']):
            raise InputNotClassifiable('Datenblock ohne Scannerposition – bitte „Kinematisch (SLAM)“ wählen oder mit Pose exportieren.')
        eligible = (any(image.get('classification_eligible_slam') for image in inspection['images'])
                    if input_mode['mode'] == 'slam'
                    else any(scan['classification_eligible_metadata'] for scan in inspection['scans']))
        if not eligible:
            strings = json.loads((Path(__file__).resolve().parents[1]/'gui/strings_de.json').read_text(encoding='utf-8'))
            reason = ('input_no_images' if inspection['image_count'] == 0 else
                      ('input_no_usable_slam_images' if input_mode['mode'] == 'slam' else 'input_no_usable_images'))
            raise InputNotClassifiable(strings[reason])
        store_report = build_store(source, cache/"scan_store", progress=lambda s,n,t: control.checkpoint())
        atomic_json(output/"store_report.json", store_report)
        full_stores = [load_scan(Path(item["path"]), store_report["source_sha256"], verify_hashes=False) for item in store_report["scans"]]
        stores = [reduced_store(store, config["point_reduction"], config["seed"]) for store in full_stores]
        if not stores:
            raise ValueError("E57 contains no scans")
        input_mode = detect_input_mode(inspection, config,
                                       points_check=lambda _: pseudo_setup_points_check(full_stores, config=config,
                                                                                         checkpoint=control.checkpoint))
        inspection['input_mode'] = input_mode
        atomic_json(output/"inspect_report.json", inspection)
        atomic_text(output/"inspect_report.md", markdown(inspection))
        control.event("input_mode", **input_mode)
        if input_mode['mode'] == 'slam':
            control.event("phase_finished", phase="inspection")
            return execute_slam(config, source, output, cache, inspection, store_report, stores, factory, control, started)
        preset = json.loads(Path(config["preset"]).read_text(encoding="utf-8"))
        semantic_run = config['processing_mode'] == 'classify_and_filter'
        shared_sam = semantic_run and config['mirror_analysis']
        # This runtime-only flag keeps the mirror consumer pinned to the same
        # view identities prepared below.  It is part of the run identity so a
        # legacy fast-mirror run can never be resumed as a shared-SAM run.
        config = {**config, '_shared_sam': shared_sam}
        errors, preset_warnings = validate_preset(preset, require_active_classes=semantic_run)
        if errors:
            raise ValueError("Invalid preset: "+"; ".join(errors))
        warnings.extend({'code': 'preset_warning', 'detail': w} for w in preset_warnings)
        all_classes = [c for c in preset["classes"] if c.get("enabled", True)]
        if semantic_run and not all_classes:
            raise ValueError("No active classes")
        missing_origins = [store.meta['scan_index'] for store in stores if store.meta['metadata'].get('pose') is None]
        if len(missing_origins) == len(stores):
            raise InputNotClassifiable('Datenblock ohne Scannerposition – bitte „Kinematisch (SLAM)“ wählen oder mit Pose exportieren.')
        classifiable_stores = [store for store in stores if store.meta['scan_index'] not in missing_origins]
        if missing_origins:
            skipped.extend(missing_origins)
            warning = {'code': 'scan_missing_pose_unclassified', 'scans': missing_origins,
                       'detail': 'Datenblöcke ohne Scannerposition wurden unverändert und unklassifiziert ausgegeben.'}
            warnings.append(warning)
            scan_results.extend({'scan': scan, 'unclassified': True, 'reason': 'missing_scanner_pose'}
                                for scan in missing_origins)
            LOG.warning('%s', warning['detail'])
        for store in classifiable_stores:
            scan_pose(store.meta['metadata'])
        config = {**config, "voxel_origin": scan_pose(classifiable_stores[0].meta["metadata"]).translation.tolist()}
        model_hash = file_hash(Path(config["model"]))
        input_hash = store_report["source_sha256"]
        identity = identity_hash({"pipeline_revision": "shared-sam-v1", "source": input_hash, "model": model_hash, "preset": preset,
                                  "config": {k:v for k,v in config.items() if k not in ("output", "cache")}})
        manifest = {**config, "pipeline_revision": "shared-sam-v1", "identity": identity, "input_sha256": input_hash, "model_sha256": model_hash,
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
        config = {**config, '_scene_work': str(work)}
        votes = VoteStore(work/"votes.sqlite")
        mask_cache = MaskCache(cache/"seg")
        image_cache = cache/"geometry"/input_hash/"images"
        mapping = preset.get("code_map", {})
        codes = [int(mapping.get(str(c["code"]), c["code"])) for c in all_classes]
        with E57Reader(source) as reader:
            sam_records = {}
            def append_mirror_warnings():
                warnings.append({'code':'mirror_limits'})
                for item in mirrors.report['images']:
                    if item.get('failed'):
                        warnings.append({'code':'mirror_image_failed','scan':item['scan'],
                                         'image':item['image'],'mask_kind':item['kind'],'detail':item['error']})
                    if any(not region['accepted'] for region in item['regions']):
                        warnings.append({'code':'reflection_plane_missing','scan':item['scan'],
                                         'image':item['image'],'mask_kind':item['kind']})

            def classify_ready_scan(store, record=None):
                """Classify one scan after its cache and local mirror surfaces exist."""
                nonlocal skipped
                control.checkpoint()
                scan = store.meta["scan_index"]
                manual = config["scene_overrides"].get(str(scan), config["scene_mode"])
                if record is None and (shared_sam or votes.completed(scan, identity)):
                    try:
                        record = _load_scene_record(work, scan)
                    except (OSError, ValueError) as error:
                        if shared_sam:
                            raise RuntimeError('Shared SAM preparation record is missing') from error
                resolution = record.get('scene') if record else None
                mode = resolution['scene'] if resolution else manual
                classes = (select_classes(preset, resolution['class_mode'], config['building_enclosure'])
                           if resolution else select_classes(preset, mode, config['building_enclosure']))
                if votes.completed(scan, identity):
                    if record is None:
                        resolution = resolve_scan_scene({'scene': manual, 'metrics': {}}, store.meta['metadata'], config,
                                                        None if manual == 'both' else manual)
                        mode = resolution['scene']
                        classes = select_classes(preset, resolution['class_mode'], config['building_enclosure'])
                        record = _scene_record(scan, resolution, classes)
                    class_ids = record.get('class_ids') or [item['id'] for item in classes]
                    scan_results.append({"scan": scan, "scene": mode, "scene_detection": resolution,
                                         "scene_resolution": resolution, "class_ids": class_ids,
                                         "resumed": True})
                    return
                try:
                    if shared_sam:
                        if record.get('status') != 'complete':
                            raise RuntimeError(record.get('error', 'Shared SAM preparation failed'))
                        resolution = record['scene']
                        mode = resolution['scene']
                        wanted = set(record['class_ids'])
                        classes = [item for item in all_classes if item['id'] in wanted]
                    else:
                        detection = {'scene': manual, 'manual': True}
                        if manual == 'both':
                            control.event('phase_started', phase='scene', scan=scan)
                            try:
                                detection = detect(reader, store, inspection, config, image_cache, mask_cache, model_hash,
                                                   factory, control, preset, image_assets, camera_depth)
                            finally:
                                control.event('phase_finished', phase='scene', scan=scan)
                        resolution = resolve_scan_scene(detection, store.meta['metadata'], config,
                                                        None if manual == 'both' else manual)
                        mode = resolution['scene']
                        classes = select_classes(preset, resolution['class_mode'], config['building_enclosure'])
                        record = _scene_record(scan, resolution, classes)
                        atomic_json(work/f'scene_{scan:04d}.json', record)
                        _emit_scene_event(control, scan, resolution)
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
                    weights_path.unlink(missing_ok=True)
                    (work/f"scan_{scan:04d}"/"mirror_rep.npy").unlink(missing_ok=True)
                    scan_results.append({"scan": scan, "scene": mode, "scene_detection": resolution,
                                         "scene_resolution": resolution, "class_ids": record['class_ids'], **stats})
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

            parallel_pipeline = shared_sam and config['pipeline_parallel']
            if parallel_pipeline:
                control.event('phase_started', phase='mirror')
                control.event("phase_started", phase="segmentation")
                sam_started = time.monotonic()
                mirror_started = time.monotonic()
                segmentation_started = time.monotonic()
                mirrors = MirrorAnalysis(work/'mirrors', config)
                prepared_images = [inspection['images'][index-1] for store in classifiable_stores
                                   for index in inspection['scans'][store.meta['scan_index']-1]['image_indices']
                                   if inspection['images'][index-1]['pose_status'] == 'valid'
                                   and any(kind != 'visualReferenceRepresentation' for kind in inspection['images'][index-1]['representations'])]
                image_numbers = {item['index']: number for number, item in enumerate(prepared_images, 1)}
                ready = queue.Queue(maxsize=config['pipeline_prefetch_scans'])
                producer_stop = threading.Event()
                producer_timing = {}

                class ProducerControl:
                    def checkpoint(self):
                        if producer_stop.is_set():
                            raise Cancelled()
                        control.checkpoint()

                    def event(self, kind, **payload):
                        control.event(kind, **payload)

                producer_control = ProducerControl()

                def publish(message, check_cancel=True):
                    while not producer_stop.is_set():
                        if check_cancel:
                            producer_control.checkpoint()
                        try:
                            ready.put(message, timeout=.1)
                            return True
                        except queue.Full:
                            continue
                    return False

                def produce_sam():
                    scan = None
                    try:
                        control.event('phase_started', phase='sam')
                        with E57Reader(source) as producer_reader:
                            for producer_store in classifiable_stores:
                                producer_control.checkpoint()
                                scan = producer_store.meta['scan_index']
                                record = prepare_shared_sam(
                                    producer_reader, [producer_store], inspection, config, work, image_cache, mask_cache,
                                    model_hash, factory, producer_control, preset, image_assets, camera_depth,
                                    progress_stores=classifiable_stores, image_numbers=image_numbers, total_images=len(prepared_images))[scan]
                                kind = 'scan' if record.get('status') == 'complete' else 'scan_failed'
                                if not publish(SamProducerMessage(kind, scan, record)):
                                    return
                    except Cancelled:
                        # A backend may raise ``Cancelled`` directly without
                        # setting Control.cancelled.  Forward that case so the
                        # consumer cannot wait on an empty queue.
                        if not control.cancelled.is_set():
                            publish(SamProducerMessage('cancelled'), check_cancel=False)
                        return
                    except BaseException as error:
                        publish(SamProducerMessage('error', scan, error), check_cancel=False)
                    finally:
                        producer_timing['seconds'] = time.monotonic()-sam_started
                        if not producer_stop.is_set():
                            control.event('phase_finished', phase='sam')
                            publish(SamProducerMessage('done'), check_cancel=False)

                producer = threading.Thread(target=produce_sam, name='sam-producer', daemon=True)
                producer.start()
                try:
                    for expected_store in classifiable_stores:
                        while True:
                            control.checkpoint()
                            try:
                                message = ready.get(timeout=.1)
                                break
                            except queue.Empty:
                                continue
                        kind, scan, record = message.kind, message.scan, message.payload
                        if kind == 'cancelled':
                            raise Cancelled()
                        if kind == 'error':
                            raise record
                        if kind not in ('scan', 'scan_failed') or scan != expected_store.meta['scan_index']:
                            raise RuntimeError('SAM producer stopped before all scans were prepared')
                        sam_records[scan] = record
                        prepared_failures = {int(index): detail for index, detail in record.get('failed_images', {}).items()}
                        fit_scan_surfaces(reader, expected_store, inspection, config, mirrors, image_cache, mask_cache,
                                          model_hash, factory, control, preset, image_assets, prepared_failures, image_numbers)
                        classify_ready_scan(expected_store, record)
                    while True:
                        try:
                            message = ready.get(timeout=.1)
                        except queue.Empty:
                            control.checkpoint()
                            continue
                        kind, scan, record = message.kind, message.scan, message.payload
                        if kind == 'cancelled':
                            raise Cancelled()
                        if kind == 'error':
                            raise record
                        if kind == 'done':
                            break
                finally:
                    producer_stop.set()
                    producer.join()
                timings['sam'] = producer_timing.get('seconds', time.monotonic()-sam_started)
                finalize_states(mirrors, classifiable_stores, inspection, config, control)
                atomic_json(output/'mirror_registry.json',[p.serialize() for p in mirrors.registry.planes])
                timings['mirror'] = time.monotonic()-mirror_started
                append_mirror_warnings()
                control.event('phase_finished', phase='mirror')
                timings['segmentation'] = time.monotonic()-segmentation_started
                if all(store.meta['scan_index'] in skipped for store in classifiable_stores):
                    raise RuntimeError("No scan could be classified; see per-scan errors in log.txt")
                control.event("phase_finished", phase="segmentation")
            else:
                if shared_sam:
                    control.event('phase_started', phase='sam')
                    sam_start = time.monotonic()
                    sam_records = prepare_shared_sam(reader, classifiable_stores, inspection, config, work, image_cache, mask_cache,
                                                     model_hash, factory, control, preset, image_assets, camera_depth)
                    timings['sam'] = time.monotonic()-sam_start
                    control.event('phase_finished', phase='sam')
                if config['mirror_analysis']:
                    control.event('phase_started',phase='mirror')
                    mirror_start = time.monotonic()
                    prepared_failures = {int(index): detail for record in sam_records.values()
                                         for index, detail in record.get('failed_images', {}).items()}
                    mirrors = build_mirrors(reader,classifiable_stores,inspection,config,work,image_cache,mask_cache,model_hash,factory,control,preset,image_assets,
                                            prepared_failures)
                    atomic_json(output/'mirror_registry.json',[p.serialize() for p in mirrors.registry.planes])
                    timings['mirror'] = time.monotonic()-mirror_start
                    append_mirror_warnings()
                    control.event('phase_finished',phase='mirror')
                if semantic_run:
                    control.event("phase_started", phase="segmentation")
                    seg_start = time.monotonic()
                    for store in classifiable_stores:
                        classify_ready_scan(store, sam_records.get(store.meta['scan_index']))
                    timings["segmentation"] = time.monotonic()-seg_start
                    if all(store.meta['scan_index'] in skipped for store in classifiable_stores):
                        raise RuntimeError("No scan could be classified; see per-scan errors in log.txt")
                    control.event("phase_finished", phase="segmentation")
            if semantic_run:
                finalize_scene_resolution(scan_results, classifiable_stores, config, warnings, control)
                control.event("phase_started", phase="voting")
                vote_start = time.monotonic()
                by_id = {c['id']: codes[i] for i,c in enumerate(all_classes)}
                priority = {'code': by_id['stairs'], 'over': [by_id[key] for key in config['stairs_priority_over'] if key in by_id],
                            'confidence': config['stairs_min_confidence'], 'min_share': config['stairs_priority_min_share']} if config['stairs_priority'] and 'stairs' in by_id else None
                families = [[by_id[name] for name in family if name in by_id] for family in config['vote_families']]
                votes.decide(config["min_observations"], config["min_winning_share"], codes, control.checkpoint, priority, families)
                if config['surface_tie_break']:
                    tie_started = time.monotonic()
                    surface_tie_report = votes.resolve_surface_ties(
                        by_id, np.array(config['voxel_origin']), config['voxel_size'], config['surface_tie_share'],
                        config['surface_normal_radius'], config['min_observations'],
                        [scan_pose(store.meta['metadata']).translation[2] for store in classifiable_stores], control.checkpoint)
                    timings['surface_ties'] = time.monotonic()-tie_started
                outline = None
                if config.get('scene_outline', True) or config['building_enclosure']:
                    barrier_codes = [by_id[key] for key in ('building', 'wall') if key in by_id]
                    outline = votes.building_outline(barrier_codes, config['voxel_size'],
                                                     config['building_boundary_cell_size'], config['building_boundary_min_height'],
                                                     config['building_boundary_max_cells'])
                if config['building_enclosure'] and {'ground', 'floor'}.issubset(by_id):
                    barrier_codes = [by_id[key] for key in ('building', 'wall') if key in by_id]
                    building_report = votes.split_ground_floor_by_building(
                        by_id['ground'], by_id['floor'], barrier_codes, config['voxel_size'],
                        config['building_boundary_cell_size'], config['building_boundary_min_height'],
                        config['building_boundary_max_cells'], control.checkpoint, outline=outline)
                segment_started = time.monotonic()
                segment_vote_report = apply_segment_vote(
                    votes, by_id, np.array(config['voxel_origin']), config['voxel_size'], config, control.checkpoint,
                    active_codes={row['code'] for row in rows})
                timings['segment_vote'] = time.monotonic()-segment_started
                rule_context = geometry_scan_context(
                    preset, config, {item['scan']: item.get('scene') for item in scan_results if 'scan' in item},
                    positions={store.meta['scan_index']: scan_pose(store.meta['metadata']).translation.tolist()
                               for store in classifiable_stores},
                    class_ids={item['scan']: item.get('class_ids', []) for item in scan_results if 'scan' in item})
                if config['geometry_rules']:
                    geometry_started = time.monotonic()
                    terrain = work/'terrain.npz'
                    geometry_rules_report = apply_geometry_rules(votes, by_id, np.array(config['voxel_origin']),
                        config['voxel_size'], config, terrain, control.checkpoint,
                        active_codes={row['code'] for row in rows}, scan_context=rule_context, outline=outline)
                    timings['geometry_rules'] = time.monotonic()-geometry_started
                if config['stairs_geometry_check'] and 'stairs' in by_id:
                    stairs_geometry_report = votes.reject_planar_stairs(
                        by_id['stairs'], np.array(config['voxel_origin']), config['voxel_size'],
                        config['stairs_plane_tolerance'], config['stairs_planar_fraction'], config['stairs_min_rise'],
                        config['seed'], codes, control.checkpoint, minimum_voxels=config['stairs_min_voxels'],
                        minimum_levels=config['stairs_min_levels'], level_tolerance=config['stairs_level_tolerance'],
                        rise_minimum=config['stairs_step_rise_min'], rise_maximum=config['stairs_step_rise_max'],
                        tread_minimum=config['stairs_step_tread_min'], tread_maximum=config['stairs_step_tread_max'],
                        floor_code=by_id.get('floor'), floor_ring_radius=config['stairs_floor_ring_radius'],
                        floor_level_tolerance=config['stairs_floor_level_tolerance'],
                        underside_minimum_voxels=config['stairs_underside_min_voxels'],
                        underside_cluster_size=config['stairs_underside_cluster_size'],
                        underside_maximum_gap=config['stairs_underside_max_gap'])
                scene_started = time.monotonic()
                scene_consistency_report = enforce_scene_consistency(
                    votes, all_classes, by_id, rule_context, np.array(config['voxel_origin']), config['voxel_size'], config,
                    {row['code'] for row in rows}, control.checkpoint, outline=outline)
                scene_outline_report = scene_consistency_report.get('scene_outline')
                timings['scene_context'] = ((geometry_rules_report or {}).get('scan_context', {}).get('duration_s', 0.)
                                            if geometry_rules_report else scene_consistency_report['duration_s'])
                timings['scene_consistency'] = time.monotonic()-scene_started
                if config['geometry_fill'] or config['shape_rules']:
                    fill_started = time.monotonic()
                    geometry_planes = work/'planes.npz'
                    extracted_planes = extract_planes(votes, by_id, np.array(config['voxel_origin']), config['voxel_size'],
                                                      config, geometry_planes, control.checkpoint)
                    timings['geometry_fill_planes'] = time.monotonic()-fill_started
                    if config['geometry_fill']:
                        geometry_fill_report = extracted_planes
                shape_started = time.monotonic()
                shape_rules_report = apply_shape_rules(
                    votes, by_id, np.array(config['voxel_origin']), config['voxel_size'], config, control.checkpoint,
                    active_codes={row['code'] for row in rows}, scan_context=rule_context, planes_path=geometry_planes,
                    outline=outline)
                timings['shape_rules'] = time.monotonic()-shape_started
                ground_dtm_report = None
                if config['ground_dtm']:
                    dtm_started = time.monotonic()
                    if outline is None:
                        barrier_codes = [by_id[key] for key in ('building', 'wall') if key in by_id]
                        outline = votes.building_outline(barrier_codes, config['voxel_size'],
                                                         config['building_boundary_cell_size'], config['building_boundary_min_height'],
                                                         config['building_boundary_max_cells'])
                    dtm_path = work/'dtm.npz'
                    dtm_build_report = build_tin_dtm(
                        votes, by_id, outline, np.array(config['voxel_origin']), config['voxel_size'],
                        config, target_path=dtm_path, scan_context=rule_context, checkpoint=control.checkpoint
                    )
                    with TinDTM(dtm_path) as tin:
                        band_report = apply_ground_dtm_band(
                            votes, by_id, tin, outline, np.array(config['voxel_origin']), config['voxel_size'],
                            config, active_codes={row['code'] for row in rows}, checkpoint=control.checkpoint
                        )
                    timings['ground_dtm'] = time.monotonic()-dtm_started
                    ground_dtm_report = {
                        **band_report,
                        'dtm': dtm_build_report,
                        'artifact': str(dtm_path),
                        'path': str(dtm_path),
                    }
                if config['component_cleanup']:
                    cleanup_rules = []
                    if 'cable' in by_id:
                        cleanup_rules.append((by_id['cable'], config['cable_component_min_voxels'],
                                              config['cable_component_min_span']))
                    if 'electrical' in by_id:
                        cleanup_rules.append((by_id['electrical'], config['electrical_component_min_voxels'],
                                              config['electrical_component_min_span']))
                    cleanup_report = votes.remove_small_components(config['voxel_size'], config['component_min_voxels'],
                                                                     cleanup_rules, control.checkpoint,
                                                                     component_min_cells=config['component_min_cells'],
                                                                     feature_cell_size=config['geometry_cell_size'],
                                                                     sparse_maximum_voxels_per_cell=config['component_sparse_max_voxels_per_cell'],
                                                                     outdoor_codes=[by_id[name] for name in ('building', 'ground', 'low_vegetation', 'tree', 'road', 'sidewalk') if name in by_id])
                if config['vegetation_base_to_ground'] and 'low_vegetation' in by_id and 'ground' in by_id:
                    vegetation_report = votes.vegetation_bases_to_ground(
                        by_id['low_vegetation'], by_id['ground'], config['voxel_size'],
                        config['vegetation_ground_cell_size'], config['vegetation_ground_band'], control.checkpoint)
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
        export_ground_dtm = work/'dtm.npz' if semantic_run and config['ground_dtm'] and (work/'dtm.npz').is_file() else None
        export_ground_alt = work/'ground_alternatives.npz' if semantic_run and config['ground_dtm'] and (work/'ground_alternatives.npz').is_file() else None
        result = writer(*output_args, mirror=mirrors.output if mirrors else None,
                        no_reflection=config['extra_no_reflection'], primary_without_reflections=not semantic_run,
                        class_transfer_radius=config['class_transfer_radius'] if semantic_run else 0.,
                        class_transfer_range_factor=config['class_transfer_range_factor'] if semantic_run else 0.,
                        class_transfer_radius_max=config['class_transfer_radius_max'] if semantic_run else 0.,
                        output_density=config['output_density'],
                        geometry_planes=geometry_planes if semantic_run and config['geometry_fill'] else None,
                        geometry_fill_tolerance=config['geometry_fill_tolerance'], terrain=terrain if semantic_run else None,
                        terrain_ground_tolerance=config['terrain_ground_tolerance'],
                        terrain_fill=config['terrain_fill'] and any(row['code'] == 2 for row in rows),
                        tree_crowns=work/'tree_crowns.npz' if semantic_run and config['tree_crown_fill'] else None,
                        tree_fill=semantic_run and config['tree_crown_fill'],
                        terrain_outdoor_scans={item['scan'] for item in scan_results if item.get('scene') == 'outdoor'},
                        ground_dtm=export_ground_dtm,
                        ground_dtm_alternatives=export_ground_alt,
                        ground_dtm_band=config['ground_dtm_band'], export_index_cache_gb=config['export_index_cache_gb'],
                        export_index_max_tile_points=config['export_index_max_tile_points'])
        timings["output"] = time.monotonic()-output_start
        transferred = {"cartesianX","cartesianY","cartesianZ","colorRed","colorGreen","colorBlue","intensity"}
        omitted = {str(s.meta["scan_index"]): sorted(set(s.meta["metadata"]["points"]["fields"])-transferred) for s in stores}
        if any(omitted.values()):
            warnings.append({"code": "attributes_omitted"})
        result.update(point_reduction=config["point_reduction"], output_density=config["output_density"],
                      computed_points=sum(selected_count(store, "reduced") for store in stores),
                      output_points=result["record_count"], format=config["format"])
        report = {"schema_version": 1, "input": inspection["source"], "input_sha256": input_hash,
                  "input_mode": input_mode,
                  "output": result, "scans": scan_results, "skipped_scans": skipped, "warnings": warnings,
                  "invalid_counts": {str(s.meta["scan_index"]): s.meta["invalid_count"] for s in stores},
                  "non_transferred_attributes": omitted, "timing_seconds": timings,
                  "total_seconds": time.monotonic()-started, "sam_calls": backend.calls if backend else 0,
                  "gpu_reductions": backend.reductions if backend else [], "seed": config["seed"],
                  "sampled_peak_rss_bytes": control.peak_rss, "sampled_peak_handles": control.peak_handles,
                  "stairs_priority_overrides": votes.priority_overrides,
                  "stairs_geometry": stairs_geometry_report,
                  "surface_ties": surface_tie_report,
                  "segment_vote": segment_vote_report,
                  "shape_rules": shape_rules_report,
                  "ground_dtm": ground_dtm_report,
                  "component_cleanup": cleanup_report,
                  "vegetation_base_to_ground": vegetation_report, "building_enclosure": building_report,
                  "depth_gate": {str(item['scan']): item.get('depth_gate', {}) for item in scan_results if 'scan' in item},
                  "geometry_fill": {**(geometry_fill_report or {'planes': {}}),
                                    'filled_points': result.get('geometry_fill_counts', {}),
                                    'confidence': 96},
                  "geometry_rules": geometry_rules_report,
                  "scene_consistency": scene_consistency_report,
                  "scene_outline": scene_outline_report,
                  "scene_resolution": scene_resolution_report(scan_results),
                  "scene_audit": scene_audit(scan_results, result.get('scan_class_counts', {}), preset, mapping),
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
