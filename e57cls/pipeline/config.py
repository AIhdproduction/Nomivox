"""Resolved pipeline parameters and project-local defaults."""

from datetime import datetime
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULTS = {"schema_version": 1, "scene_mode": "both", "profile": "balanced", "image_scale": .75,
            "tile_size": 1008, "tile_overlap": .2, "prompt_batch": 8, "instance_confidence": .3,
            "mask_confidence": .5, "boundary_margin": 2, "voxel_size": .02, "min_observations": 1,
            "min_winning_share": .4, "view_size": 1512, "view_fov": 95, "visibility_abs": .03,
            "visibility_rel": .01, "depth_width": 4096, "gpu_device": "cuda:0", "gpu_memory_fraction": .85,
            "mirror_analysis": True, "glass_analysis": True, "reflection_policy": "masked_depth",
            "extra_no_reflection": False, "format": "las", "strict": False,
            "seed": 42, "scene_overrides": {}, "diagnostics": "sample", "neighborhood_cleanup": False,
            "scene_sky_outdoor": .15, "scene_ceiling_outdoor_max": .05, "scene_ceiling_indoor": .3,
            "scene_sky_indoor_max": .02, "scene_overhead_range": 8., "scene_plane_tolerance": .02,
            "scene_plane_fraction": .5, "scene_ceiling_supported": .15, "scene_upper_returns_indoor": .75,
            "scene_open_sky_outdoor": .10, "scene_open_sky_indoor_max": .05,
            "scene_sky_confidence": .65, "scene_ceiling_confidence": .5,
            "stairs_priority": True, "stairs_min_confidence": .65,
            "mirror_mask_width": 2048, "mirror_mask_confidence": .45, "reflection_prompt_agreement": 2,
            "mirror_fit_tolerance": .02, "mirror_plane_tolerance": .03,
            "mirror_confirmation_radius": .05, "mirror_merge_angle": 3., "mirror_merge_distance": .03,
            "mirror_max_range": 30., "mirror_min_pixels": 40, "mirror_frame_band": 10, "mirror_frame_band_max": 30}


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
    if value["format"] not in ("las", "laz") or value["scene_mode"] not in ("indoor", "outdoor", "both"):
        raise ValueError("Invalid output format or scene mode")
    for key in ("image_scale", "gpu_memory_fraction"):
        if not 0 < value[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("instance_confidence", "mask_confidence", "min_winning_share"):
        if not 0 <= value[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not 0 <= value["tile_overlap"] < 1 or not .001 <= value["voxel_size"] <= 1:
        raise ValueError("Invalid overlap or voxel size")
    if not 1 <= value["min_observations"] <= 65535 or not 0 <= value["boundary_margin"] <= 16:
        raise ValueError("Invalid observations or boundary margin")
    if not 64 <= value["view_size"] <= 1512 or not 336 <= value["tile_size"] <= 1512:
        raise ValueError("Invalid image work-unit size")
    if value['profile'] not in ('fast','balanced','high') or value['diagnostics'] not in ('off','sample','all'):
        raise ValueError('Invalid profile or diagnostics mode')
    if not isinstance(value['prompt_batch'],int) or not 1 <= value['prompt_batch'] <= 64:
        raise ValueError('Invalid prompt batch')
    if not 256 <= value['depth_width'] <= 8192 or not 0 < value['view_fov'] < 180:
        raise ValueError('Invalid projection resolution or field of view')
    if not isinstance(value['scene_overrides'],dict) or any(v not in ('both','indoor','outdoor') for v in value['scene_overrides'].values()):
        raise ValueError('Invalid scene override')
    for key in ('scene_sky_outdoor','scene_ceiling_outdoor_max','scene_ceiling_indoor',
                'scene_sky_indoor_max','scene_plane_fraction','mirror_mask_confidence','scene_ceiling_supported',
                'scene_upper_returns_indoor','scene_open_sky_outdoor','scene_open_sky_indoor_max',
                'scene_sky_confidence','scene_ceiling_confidence','stairs_min_confidence'):
        if not 0 <= value[key] <= 1:
            raise ValueError(f'Invalid {key}')
    for key in ('scene_overhead_range','scene_plane_tolerance','mirror_fit_tolerance',
                'mirror_plane_tolerance','mirror_confirmation_radius','mirror_merge_angle',
                'mirror_merge_distance','mirror_max_range','mirror_min_pixels','mirror_mask_width','mirror_frame_band','mirror_frame_band_max'):
        if not 0 < value[key] < float('inf'):
            raise ValueError(f'Invalid {key}')
    if value['neighborhood_cleanup']:
        raise ValueError('Neighborhood cleanup is not enabled in this release')
    if value['reflection_policy'] != 'masked_depth':
        raise ValueError('Unsupported reflection policy')
    if not isinstance(value['reflection_prompt_agreement'],int) or not 1 <= value['reflection_prompt_agreement'] <= 8:
        raise ValueError('Invalid reflection prompt agreement')
    return value


def load_config(path: Path):
    return resolve_config(json.loads(path.read_text(encoding="utf-8")), path.resolve().parent)
