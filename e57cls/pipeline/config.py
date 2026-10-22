"""Resolved pipeline parameters and project-local defaults."""

from datetime import datetime
import json
import re
from pathlib import Path

from e57cls.seg.cache import SCORE_FLOOR

PROJECT = Path(__file__).resolve().parents[2]
DEFAULTS = {"schema_version": 1, "processing_mode": "classify_and_filter", "input_mode": "auto", "scene_mode": "both", "profile": "balanced", "image_scale": .90,
            "tile_size": 1008, "tile_overlap": .3, "prompt_batch": "auto", "instance_confidence": .4,
            "mask_confidence": .5, "boundary_margin": 2, "voxel_size": .02, "min_observations": 1,
            "min_winning_share": .50, "surface_tie_break": True, "surface_tie_share": .55,
            "vote_families": [["ground", "floor"]],
            "surface_normal_radius": .06, "view_size": 1512, "view_fov": 95, "visibility_abs": .03,
            "visibility_rel": .01, "depth_width": 4096, "gpu_device": "cuda:0", "gpu_memory_fraction": .85,
            "mirror_analysis": True, "glass_analysis": True, "reflection_policy": "masked_depth",
            "extra_no_reflection": False, "format": "laz", "output_density": "reduced",
            "point_reduction": .40, "strict": False, "pipeline_parallel": True,
            "pipeline_prefetch_scans": 1,
            "cache_cleanup": "all",
            "slam_max_range_indoor": 20., "slam_max_range_outdoor": 40., "slam_max_range": None,
            "slam_depth_resolution": 1024, "slam_splat_max_px": 6,
            "slam_depth_tol_abs": .02, "slam_depth_tol_rel": .01, "slam_diagnostics": False,
            "slam_station_spacing": 1.0, "slam_boundary_margin_extra": 1,
            "slam_weight_distance_scale": 5.0, "slam_weight_incidence_power": 1.0,
            "slam_weight_radial_power": 1.0, "slam_time_filter": False,
            "slam_time_window": 10.0,
            "seed": 42, "scene_overrides": {}, "diagnostics": "sample",
            "component_cleanup": True, "component_min_voxels": 12, "component_min_cells": 4,
            "component_sparse_max_voxels_per_cell": 1,
            "cable_component_min_voxels": 8, "cable_component_min_span": .20,
            "electrical_component_min_voxels": 8, "electrical_component_min_span": .06,
            "class_transfer_radius": .01, "class_transfer_range_factor": .003,
            "class_transfer_radius_max": .05,
            "depth_gate_classes": ["door"], "door_depth_limit": 1.0,
            "door_max_width": 1.4, "door_max_height": 2.6,
            "geometry_fill": True, "geometry_fill_classes": ["ceiling", "wall", "floor", "ground", "road", "sidewalk"],
            "geometry_fill_min_area": .5, "geometry_fill_max_gap": .35, "geometry_fill_tolerance": .02,
            "geometry_fill_max_planes": 400, "geometry_fill_min_component_voxels": 20,
            "geometry_fill_ransac_trials": 256, "geometry_fill_ransac_max_trials": 1024,
            "geometry_rules": True, "geometry_cell_size": .10, "geometry_tile_size": 8.,
            "segment_vote": True, "segment_min_planarity": .5, "segment_normal_angle": 15.,
            "segment_plane_distance": .03, "segment_min_cells": 3, "segment_min_share": .6,
            "segment_weak_share": .6, "segment_fill_min_support": 0., "segment_flip_min_support": .25,
            "segment_protected_classes": ["stairs", "door", "window", "electrical", "traffic_sign", "manhole"],
            "geometry_rules_outdoor_vegetation": True, "geometry_rules_ground_floor": True,
            "scene_context_margin": .8, "scene_consistency": True,
            "scene_consistency_fallbacks": {"floor": ["ground", "sidewalk", "road"], "ceiling": ["building"],
                "wall": ["building"], "door": ["building"], "railing": ["fence"], "ground": ["floor"],
                "road": ["floor"], "sidewalk": ["floor"], "curb": ["floor"], "manhole": ["floor"],
                "building": ["wall"], "fence": ["railing"]},
            "geometry_halo": 1.5, "geometry_knn": 24, "geometry_knn_radius": 1.,
            "terrain_cell_size": .5, "terrain_max_gap": 5., "terrain_max_slope": .75,
            "terrain_ground_tolerance": .15, "terrain_fill": True,
            "terrain_cover_min_height": 1.8, "terrain_cover_max_height": 6.,
            "tree_crown_radius": 4., "tree_crown_max_height": 25.,
            "tree_crown_grow_radius": 6.,
            "vegetation_base_to_ground": True, "vegetation_ground_cell_size": .10,
            "vegetation_ground_band": .04,
            "building_enclosure": True, "building_boundary_cell_size": .25,
            "building_boundary_min_height": 1.2, "building_boundary_max_cells": 4_000_000,
            "scene_sky_outdoor": .15, "scene_ceiling_outdoor_max": .05, "scene_ceiling_indoor": .3,
            "scene_sky_indoor_max": .02, "scene_overhead_range": 8., "scene_plane_tolerance": .02,
            "scene_plane_fraction": .5, "scene_ceiling_supported": .15, "scene_upper_returns_indoor": .75,
            "scene_open_sky_outdoor": .10, "scene_open_sky_indoor_max": .05,
            "scene_sky_confidence": .65, "scene_ceiling_confidence": .5,
            "scene_openness_elevation_min": 0., "scene_openness_elevation_max": 25., "scene_openness_sectors": 72,
            "scene_openness_samples": 2_000_000, "scene_openness_range": 15., "scene_openness_cell_open_share": .5,
            "scene_openness_sector_cells": .8, "scene_openness_covered_outdoor": .5,
            "scene_openness_outdoor_min": .35, "scene_openness_indoor_max": .15,
            "scene_name_hints": True, "scene_name_hint_mode": "prefer", "scene_unresolved_classes": "all",
            "scene_name_fields": ["rlms:scanposGuid", "guid", "name", "description"],
            "scene_name_outdoor_patterns": ["au(?:ss|ß)en", "outdoor", "outside", "exterior", "fassade", "facade", "innenhof", "umgebung"],
            "scene_name_indoor_patterns": ["innen(?!hof)", "indoor", "inside", "interior", "keller", "geschoss", "etage", "treppenhaus", "(?<![a-z])\\d{0,2}\\.?\\s?(?:ug|kg|eg|og|dg|zg|sg)\\s?\\d{0,3}(?![a-z])"],
            "scene_neighbour_consensus": True, "scene_neighbour_radius": 15., "scene_neighbour_height": 1.5,
            "scene_neighbour_min": 1,
            "stairs_priority": True, "stairs_min_confidence": .65, "stairs_priority_min_share": .25,
            "stairs_priority_over": ["floor", "ground"], "stairs_geometry_check": True,
            "stairs_plane_tolerance": .03, "stairs_planar_fraction": .85, "stairs_min_rise": .25,
            "stairs_min_voxels": 50, "stairs_min_levels": 3, "stairs_level_tolerance": .03,
            "stairs_step_rise_min": .10, "stairs_step_rise_max": .23,
            "stairs_step_tread_min": .20, "stairs_step_tread_max": .45,
            "stairs_floor_ring_radius": 2., "stairs_floor_level_tolerance": .03,
            "stairs_underside_min_voxels": 1000, "stairs_underside_cluster_size": 3,
            "stairs_underside_max_gap": .20,
            # Reflection masks keep the complete angular coverage of a panorama,
            # but do not need the much more expensive classification resolution.
            # A 1008 px view at 75 % scale fits in one SAM tile instead of the
            # four tiles used by the normal 1512 px semantic-classification view.
            "mirror_view_size": 1008, "mirror_image_scale": .75,
            "mirror_mask_width": 2048, "mirror_mask_confidence": .45, "reflection_prompt_agreement": 2,
            "mirror_fit_tolerance": .02, "mirror_plane_tolerance": .03,
            "mirror_confirmation_radius": .05, "mirror_merge_angle": 3., "mirror_merge_distance": .03,
            "mirror_max_range": 30., "mirror_min_pixels": 40, "mirror_frame_band": 10, "mirror_frame_band_max": 30,
            "reflection_neighbor_radius": .05,
            "mirror_shape_regularize": True, "mirror_shape_kinds": ["mirror_mask", "glass_mask"],
            "mirror_shape_min_iou": .75, "mirror_shape_max_growth": 1.35, "mirror_edge_search": .10}


def resolve_config(config: dict, base: Path = PROJECT):
    value = {**DEFAULTS, **config}
    source = Path(value["source"])
    source = (base/source).resolve() if not source.is_absolute() else source.resolve()
    if not source.is_file() or source.suffix.lower() != ".e57":
        raise ValueError(f"E57 input not found: {source}")
    value["source"] = str(source)
    model = PROJECT/"models/sam.pt"
    if not model.exists():
        model = PROJECT/"models/sam3.pt"
    defaults = {"cache": PROJECT/".cache", "model": model, "preset": PROJECT/"presets/default.json",
                "output": source.parent/"klassifiziert"/f"run_{datetime.now():%Y%m%d_%H%M%S}"}
    for key, default in defaults.items():
        path = Path(value.get(key, default))
        value[key] = str((base/path).resolve() if not path.is_absolute() else path.resolve())
    if value["format"] not in ("las", "laz", "e57") or value["scene_mode"] not in ("indoor", "outdoor", "both"):
        raise ValueError("Invalid output format or scene mode")
    if value["output_density"] not in ("reduced", "original"):
        raise ValueError("Invalid output density")
    if value["cache_cleanup"] not in ("all", "keep_masks", "keep_all"):
        raise ValueError("Invalid cache cleanup mode")
    if (not isinstance(value["point_reduction"], (int, float)) or isinstance(value["point_reduction"], bool)
            or not 0 <= value["point_reduction"] <= .95):
        raise ValueError("Invalid point reduction")
    if value["processing_mode"] not in ("classify_and_filter", "filter_only"):
        raise ValueError("Invalid processing mode")
    if value["input_mode"] not in ("auto", "static", "slam"):
        raise ValueError("Invalid input mode")
    if value["slam_max_range"] is not None and (not isinstance(value["slam_max_range"], (int, float))
                                                 or isinstance(value["slam_max_range"], bool)
                                                 or not 0 < value["slam_max_range"] < float("inf")):
        raise ValueError("Invalid SLAM maximum range")
    for key in ("slam_max_range_indoor", "slam_max_range_outdoor"):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 < value[key] < float("inf")):
            raise ValueError(f"Invalid {key}")
    for key in ("slam_depth_tol_abs", "slam_depth_tol_rel"):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 <= value[key] < float("inf")):
            raise ValueError(f"Invalid {key}")
    if not 64 <= value["slam_depth_resolution"] <= 8192:
        raise ValueError("Invalid SLAM depth-map resolution")
    if not 1 <= value["slam_splat_max_px"] <= 64:
        raise ValueError("Invalid SLAM splat maximum")
    if not isinstance(value["slam_diagnostics"], bool):
        raise ValueError("Invalid SLAM diagnostics setting")
    if (not isinstance(value["slam_station_spacing"], (int, float)) or isinstance(value["slam_station_spacing"], bool)
            or not 0 < value["slam_station_spacing"] < float("inf")):
        raise ValueError("Invalid SLAM station spacing")
    if (not isinstance(value["slam_boundary_margin_extra"], int) or isinstance(value["slam_boundary_margin_extra"], bool)
            or not 0 <= value["slam_boundary_margin_extra"] <= 16):
        raise ValueError("Invalid SLAM boundary margin")
    for key in ("slam_weight_distance_scale", "slam_time_window"):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 < value[key] < float("inf")):
            raise ValueError(f"Invalid {key}")
    for key in ("slam_weight_incidence_power", "slam_weight_radial_power"):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 <= value[key] < float("inf")):
            raise ValueError(f"Invalid {key}")
    if not isinstance(value["slam_time_filter"], bool):
        raise ValueError("Invalid SLAM time filter")
    if not isinstance(value['pipeline_parallel'], bool):
        raise ValueError('Invalid pipeline parallel setting')
    if (not isinstance(value['pipeline_prefetch_scans'], int) or isinstance(value['pipeline_prefetch_scans'], bool)
            or not 1 <= value['pipeline_prefetch_scans'] <= 3):
        raise ValueError('Invalid pipeline prefetch scans')
    # A reflection-only run has no useful result without this analysis.  Keep this
    # invariant for command-line configurations as well as the GUI.
    if value["processing_mode"] == "filter_only":
        value["mirror_analysis"] = True
    for key in ("image_scale", "mirror_image_scale", "gpu_memory_fraction"):
        if not 0 < value[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("instance_confidence", "mask_confidence", "min_winning_share", "surface_tie_share",
                "mirror_shape_min_iou"):
        if not 0 <= value[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("instance_confidence", "mask_confidence"):
        if value[key] < SCORE_FLOOR:
            raise ValueError(f"{key} must be at least {SCORE_FLOOR}")
    if not 0 <= value["tile_overlap"] < 1 or not .001 <= value["voxel_size"] <= 1:
        raise ValueError("Invalid overlap or voxel size")
    if not 1 <= value["min_observations"] <= 65535 or not 0 <= value["boundary_margin"] <= 16:
        raise ValueError("Invalid observations or boundary margin")
    if (not 64 <= value["view_size"] <= 1512 or not 336 <= value["mirror_view_size"] <= 1512
            or not 336 <= value["tile_size"] <= 1512):
        raise ValueError("Invalid image work-unit size")
    if value['profile'] not in ('fast','balanced','high') or value['diagnostics'] not in ('off','sample','all'):
        raise ValueError('Invalid profile or diagnostics mode')
    if value['prompt_batch'] != 'auto' and (not isinstance(value['prompt_batch'],int) or isinstance(value['prompt_batch'],bool)
                                            or not 1 <= value['prompt_batch'] <= 64):
        raise ValueError('Invalid prompt batch')
    if not 256 <= value['depth_width'] <= 8192 or not 0 < value['view_fov'] < 180:
        raise ValueError('Invalid projection resolution or field of view')
    if not isinstance(value['scene_overrides'],dict) or any(v not in ('both','indoor','outdoor') for v in value['scene_overrides'].values()):
        raise ValueError('Invalid scene override')
    if value['scene_name_hint_mode'] not in ('prefer', 'tiebreak', 'off') or value['scene_unresolved_classes'] not in ('all', 'indoor'):
        raise ValueError('Invalid scene resolution mode')
    if not isinstance(value['scene_name_hints'], bool) or not isinstance(value['scene_neighbour_consensus'], bool):
        raise ValueError('Invalid scene resolution setting')
    if (not isinstance(value['scene_name_fields'], list) or not all(isinstance(item, str) for item in value['scene_name_fields'])
            or any(not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key])
                   for key in ('scene_name_outdoor_patterns', 'scene_name_indoor_patterns'))):
        raise ValueError('Invalid scene name patterns')
    try:
        for key in ('scene_name_outdoor_patterns', 'scene_name_indoor_patterns'):
            for pattern in value[key]:
                re.compile(pattern)
    except re.error as error:
        raise ValueError('Invalid scene name pattern') from error
    for key in ('scene_sky_outdoor','scene_ceiling_outdoor_max','scene_ceiling_indoor',
                'scene_sky_indoor_max','scene_plane_fraction','mirror_mask_confidence','scene_ceiling_supported',
                'scene_upper_returns_indoor','scene_open_sky_outdoor','scene_open_sky_indoor_max',
                'scene_sky_confidence','scene_ceiling_confidence','stairs_min_confidence', 'stairs_priority_min_share'):
        if not 0 <= value[key] <= 1:
            raise ValueError(f'Invalid {key}')
    for key in ('scene_openness_cell_open_share', 'scene_openness_sector_cells', 'scene_openness_covered_outdoor',
                'scene_openness_outdoor_min', 'scene_openness_indoor_max'):
        if not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or not 0 <= value[key] <= 1:
            raise ValueError(f'Invalid {key}')
    if (not 0 <= value['scene_openness_elevation_min'] < value['scene_openness_elevation_max'] <= 90
            or not isinstance(value['scene_openness_sectors'], int) or value['scene_openness_sectors'] < 1
            or not isinstance(value['scene_openness_samples'], int) or value['scene_openness_samples'] < 1
            or not isinstance(value['scene_neighbour_min'], int) or value['scene_neighbour_min'] < 1):
        raise ValueError('Invalid scene sampling setting')
    for key in ('scene_openness_range', 'scene_neighbour_radius', 'scene_neighbour_height'):
        if not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    for key in ('mirror_mask_confidence', 'scene_sky_confidence', 'scene_ceiling_confidence', 'stairs_min_confidence'):
        if value[key] < SCORE_FLOOR:
            raise ValueError(f'{key} must be at least {SCORE_FLOOR}')
    for key in ('scene_overhead_range','scene_plane_tolerance','mirror_fit_tolerance',
                'mirror_plane_tolerance','mirror_confirmation_radius','mirror_merge_angle',
                'mirror_merge_distance','mirror_max_range','mirror_min_pixels','mirror_mask_width','mirror_frame_band','mirror_frame_band_max',
                'mirror_shape_max_growth','mirror_edge_search'):
        if not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    for key in ('stairs_plane_tolerance', 'stairs_min_rise', 'stairs_level_tolerance',
                'stairs_step_rise_min', 'stairs_step_rise_max', 'stairs_step_tread_min',
                'stairs_step_tread_max', 'stairs_floor_ring_radius', 'stairs_floor_level_tolerance',
                'stairs_underside_max_gap'):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 < value[key] < float('inf')):
            raise ValueError(f'Invalid {key}')
    if (not isinstance(value['stairs_planar_fraction'], (int, float)) or isinstance(value['stairs_planar_fraction'], bool)
            or not 0 <= value['stairs_planar_fraction'] <= 1):
        raise ValueError('Invalid stairs planar fraction')
    for key in ('stairs_min_voxels', 'stairs_min_levels', 'stairs_underside_min_voxels',
                'stairs_underside_cluster_size'):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1:
            raise ValueError(f'Invalid {key}')
    if value['stairs_min_levels'] < 3:
        raise ValueError('Invalid stairs minimum levels')
    if value['stairs_step_rise_min'] > value['stairs_step_rise_max']:
        raise ValueError('Invalid stairs rise range')
    if value['stairs_step_tread_min'] > value['stairs_step_tread_max']:
        raise ValueError('Invalid stairs tread range')
    if not isinstance(value['stairs_priority_over'], list) or not all(isinstance(item, str) for item in value['stairs_priority_over']):
        raise ValueError('Invalid stairs priority classes')
    if (not isinstance(value['vote_families'], list) or any(not isinstance(family, list)
            or not family or not all(isinstance(item, str) for item in family) for family in value['vote_families'])):
        raise ValueError('Invalid vote families')
    if (not isinstance(value['component_cleanup'], bool) or not isinstance(value['surface_tie_break'], bool)
            or not isinstance(value['geometry_fill'], bool) or not isinstance(value['stairs_geometry_check'], bool)
            or not isinstance(value['mirror_shape_regularize'], bool) or not isinstance(value['geometry_rules'], bool)
            or not isinstance(value['segment_vote'], bool)
            or not isinstance(value['geometry_rules_outdoor_vegetation'], bool)
            or not isinstance(value['geometry_rules_ground_floor'], bool) or not isinstance(value['terrain_fill'], bool)
            or not isinstance(value['scene_consistency'], bool)):
        raise ValueError('Invalid component cleanup setting')
    for key in ('component_min_voxels', 'component_min_cells', 'component_sparse_max_voxels_per_cell', 'cable_component_min_voxels',
                'electrical_component_min_voxels', 'geometry_fill_max_planes', 'geometry_fill_min_component_voxels',
                'geometry_fill_ransac_trials', 'geometry_fill_ransac_max_trials', 'segment_min_cells'):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1:
            raise ValueError(f'Invalid {key}')
    for key in ('cable_component_min_span','electrical_component_min_span','class_transfer_radius', 'class_transfer_radius_max', 'reflection_neighbor_radius',
                'surface_normal_radius','door_depth_limit','door_max_width','door_max_height','geometry_fill_min_area',
                'geometry_fill_max_gap','geometry_fill_tolerance', 'geometry_cell_size', 'geometry_tile_size',
                'segment_plane_distance',
                'geometry_halo', 'geometry_knn_radius', 'terrain_cell_size', 'terrain_max_gap', 'terrain_max_slope',
                'terrain_ground_tolerance', 'terrain_cover_min_height', 'terrain_cover_max_height',
                'tree_crown_radius', 'tree_crown_max_height', 'tree_crown_grow_radius'):
        if not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    for key in ('segment_min_planarity', 'segment_min_share', 'segment_weak_share',
                'segment_fill_min_support', 'segment_flip_min_support'):
        if (not isinstance(value[key], (int, float)) or isinstance(value[key], bool)
                or not 0 <= value[key] <= 1):
            raise ValueError(f'Invalid {key}')
    if (not isinstance(value['segment_normal_angle'], (int, float)) or isinstance(value['segment_normal_angle'], bool)
            or not 0 <= value['segment_normal_angle'] <= 180):
        raise ValueError('Invalid segment normal angle')
    if (not isinstance(value['segment_protected_classes'], list)
            or not all(isinstance(item, str) for item in value['segment_protected_classes'])):
        raise ValueError('Invalid segment protected classes')
    if (not isinstance(value['scene_context_margin'], (int, float)) or isinstance(value['scene_context_margin'], bool)
            or not 0 < value['scene_context_margin'] <= 1):
        raise ValueError('Invalid scene context margin')
    if (not isinstance(value['scene_consistency_fallbacks'], dict)
            or any(not isinstance(name, str) or not isinstance(items, list)
                   or not all(isinstance(item, str) for item in items)
                   for name, items in value['scene_consistency_fallbacks'].items())):
        raise ValueError('Invalid scene consistency fallbacks')
    if (not isinstance(value['class_transfer_range_factor'], (int, float))
            or isinstance(value['class_transfer_range_factor'], bool)
            or not 0 <= value['class_transfer_range_factor'] < float('inf')):
        raise ValueError('Invalid class_transfer_range_factor')
    if value['class_transfer_radius_max'] < value['class_transfer_radius']:
        raise ValueError('Invalid class transfer radius range')
    if value['geometry_fill_ransac_max_trials'] < value['geometry_fill_ransac_trials']:
        raise ValueError('Invalid geometry fill RANSAC trial range')
    if (not isinstance(value['geometry_knn'], int) or isinstance(value['geometry_knn'], bool)
            or not 3 <= value['geometry_knn'] <= 128):
        raise ValueError('Invalid geometry neighbour count')
    if value['terrain_cover_max_height'] <= value['terrain_cover_min_height']:
        raise ValueError('Invalid terrain cover range')
    if (not isinstance(value['depth_gate_classes'], list) or not all(isinstance(item, str) for item in value['depth_gate_classes'])
            or not isinstance(value['geometry_fill_classes'], list) or not all(isinstance(item, str) for item in value['geometry_fill_classes'])):
        raise ValueError('Invalid geometry class list')
    if not isinstance(value['vegetation_base_to_ground'], bool):
        raise ValueError('Invalid vegetation ground setting')
    for key in ('vegetation_ground_cell_size','vegetation_ground_band'):
        if not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    if not isinstance(value['building_enclosure'], bool):
        raise ValueError('Invalid building enclosure setting')
    for key in ('building_boundary_cell_size','building_boundary_min_height'):
        if not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    if (not isinstance(value['building_boundary_max_cells'], int) or isinstance(value['building_boundary_max_cells'], bool)
            or value['building_boundary_max_cells'] < 1):
        raise ValueError('Invalid building boundary maximum')
    if value['reflection_policy'] != 'masked_depth':
        raise ValueError('Unsupported reflection policy')
    if not isinstance(value['reflection_prompt_agreement'],int) or not 1 <= value['reflection_prompt_agreement'] <= 8:
        raise ValueError('Invalid reflection prompt agreement')
    if (not isinstance(value['mirror_shape_kinds'], list)
            or not all(isinstance(item, str) for item in value['mirror_shape_kinds'])):
        raise ValueError('Invalid mirror shape kinds')
    return value


def load_config(path: Path):
    return resolve_config(json.loads(path.read_text(encoding="utf-8")), path.resolve().parent)
