"""Require positive open-sky evidence before enabling outdoor scene classes."""
from dataclasses import asdict
from importlib.metadata import version
import numpy as np
from e57cls.geometry.cameras import Pose, View, scan_pose
from e57cls.geometry.images import render_view
from e57cls.geometry.planes import fit_plane
from e57cls.seg.backend import PromptClass
from e57cls.seg.cache import segment_view


def decide(metrics, config):
    sky, ceiling = metrics['sky_fraction'], metrics['ceiling_fraction']
    open_sky = metrics.get('open_sky_fraction', 0.)
    supported_ceiling = (ceiling >= config['scene_ceiling_supported']
                         and metrics.get('upper_return_fraction',0) >= config['scene_upper_returns_indoor'])
    indoor = ceiling >= config['scene_ceiling_indoor'] or supported_ceiling or metrics['overhead_plane']
    outdoor = (sky >= config['scene_sky_outdoor'] and open_sky >= config['scene_open_sky_outdoor']
               and ceiling < config['scene_ceiling_outdoor_max'] and not indoor)
    if metrics['upper_coverage'] < .6:
        return 'unsure'
    if indoor and open_sky <= config['scene_open_sky_indoor_max']:
        return 'indoor'
    if open_sky >= config['scene_open_sky_outdoor'] and ceiling >= config['scene_ceiling_outdoor_max']:
        return 'mixed'
    return 'outdoor' if outdoor else 'indoor' if indoor else 'unsure'


def detect(reader, store, inspection, config, image_cache, mask_cache, model_hash, factory, control, preset, assets, depth_builder):
    scan = store.meta['scan_index']
    scan_transform = scan_pose(store.meta['metadata'])
    # Equal-area upper-hemisphere rays in world axes, 8192 samples.
    azimuth, z = np.meshgrid(np.linspace(-np.pi, np.pi, 128, endpoint=False), (np.arange(64)+.5)/64)
    rays = np.column_stack((np.sqrt(1-z.ravel()**2)*np.cos(azimuth.ravel()),
                            np.sqrt(1-z.ravel()**2)*np.sin(azimuth.ravel()), z.ravel()))
    merged = np.zeros((2, len(rays)), np.float32)
    covered, returns = np.zeros(len(rays), bool), np.zeros(len(rays), bool)
    sky = next(c for c in preset['auxiliary'] if c['id'] == 'sky')
    ceiling = next((c for c in preset['classes'] if c['id'] == 'ceiling'), {'prompts': ['ceiling']})
    classes = [PromptClass('sky', tuple(sky['prompts'])), PromptClass('ceiling', tuple(ceiling['prompts']))]
    thresholds = np.array([config['scene_sky_confidence'],config['scene_ceiling_confidence']])
    for index in inspection['scans'][scan-1]['image_indices']:
        item = inspection['images'][index-1]
        if item['pose_status'] != 'valid' or not any(r != 'visualReferenceRepresentation' for r in item['representations']):
            continue
        camera, image, mask, image_hash, mask_hash = assets(reader, item, image_cache)
        pose = Pose.from_dict(item['metadata']['pose'])
        camera_rays = rays@pose.rotation
        uv, _, in_image = camera.project(camera_rays)
        depth = depth_builder(store, scan_transform, pose, camera, 512, control)
        pixel = np.floor(uv*np.array([depth.width/camera.width, depth.height/camera.height])).astype(int)
        ids = np.flatnonzero(in_image)
        pixel[:, 0] = pixel[:, 0].clip(0, depth.width-1)
        pixel[:, 1] = pixel[:, 1].clip(0, depth.height-1)
        returns[ids] |= np.isfinite(depth.depth[pixel[ids, 1], pixel[ids, 0]])
        views = [View(f'scene_{i}', yaw, 45, 95, 768) for i,yaw in enumerate((0, 90, 180, 270))]
        views.append(View('scene_zenith', 0, 90, 95, 768))
        if camera.kind == 'pinholeRepresentation':
            views = [None]
        for view in views:
            control.checkpoint()
            rendered, valid = render_view(image, mask, camera, view) if view else (image, mask > 0 if mask is not None else np.ones(image.shape[:2], bool))
            info = asdict(view) if view else {'kind': camera.kind, 'params': camera.params}
            info.update(mask_sha256=mask_hash, torch_version=version('torch'))
            scores, _ = segment_view(rendered, valid, classes, factory, mask_cache, image_hash, info, model_hash,
                                     config['image_scale'], config['tile_size'], config['tile_overlap'], control.checkpoint)
            coords, _, good = view.project(camera_rays) if view else camera.project(camera_rays)
            h, w = scores[0].shape
            size = [view.size, view.size] if view else [camera.width, camera.height]
            pixels = np.floor(coords*np.array([w,h])/size).astype(int)
            ids = np.flatnonzero(good & in_image)
            pixels[:, 0] = pixels[:, 0].clip(0, w-1)
            pixels[:, 1] = pixels[:, 1].clip(0, h-1)
            native = np.floor(uv[ids]).astype(int)
            if mask is not None:
                ids = ids[mask[native[:,1].clip(0,camera.height-1), native[:,0].clip(0,camera.width-1)] > 0]
            covered[ids] = True
            for c in range(2):
                merged[c, ids] = np.maximum(merged[c, ids], scores[c][pixels[ids,1],pixels[ids,0]])
        del image, mask, depth
    # Deterministic bounded reservoir spread over the entire original scan.
    overhead = []
    stride = max(1, store.meta['record_count']//20000)
    for start, xyz, valid in store.chunks():
        control.checkpoint()
        selected = valid & ((np.arange(start, start+len(xyz)) % stride) == 0)
        world = scan_transform.to_world(xyz[selected])-scan_transform.translation
        overhead.append(world[(world[:,2] > .5) & (world[:,2] <= config['scene_overhead_range'])])
    plane = fit_plane(np.concatenate(overhead) if overhead else np.empty((0,3)), config['scene_plane_tolerance'],
                      config['seed'], min_fraction=config['scene_plane_fraction'], horizontal=True)
    sky_pixels = (merged[0] >= thresholds[0]) & covered
    metrics = {'sky_fraction': float(np.mean(sky_pixels)),
               'open_sky_fraction': float(np.mean(sky_pixels & ~returns)),
               'ceiling_fraction': float(np.mean((merged[1] >= thresholds[1]) & covered)),
               'upper_return_fraction': float(returns.mean()), 'upper_coverage': float(covered.mean()),
               'overhead_plane': plane is not None,
               'overhead_plane_inliers': plane['inlier_fraction'] if plane else 0.}
    return {'scene': decide(metrics, config), 'metrics': metrics,
            'thresholds': {k:v for k,v in config.items() if k.startswith('scene_') and k not in ('scene_mode','scene_overrides')}}
