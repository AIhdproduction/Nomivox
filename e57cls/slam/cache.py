"""Run-owned, prompt-independent persistence for per-view visibility."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from e57cls.files import atomic_json
from e57cls.seg.cache import identity_hash
from .visibility import ViewVisibility


_PARAMETERS = ("slam_max_range", "slam_max_range_indoor", "slam_max_range_outdoor", "slam_depth_resolution",
               "slam_splat_max_px", "slam_depth_tol_abs", "slam_depth_tol_rel")


def visibility_key(source_hash, pose, intrinsics, view, config):
    """Hash exactly the inputs that can alter a geometric visibility result."""
    pose_value = {"rotation": np.asarray(pose.rotation).round(12).tolist(),
                  "translation": np.asarray(pose.translation).round(12).tolist()}
    values = {key: config.get(key) for key in _PARAMETERS}
    return identity_hash({"version": 2, "source": source_hash, "pose": pose_value,
                          "intrinsics": intrinsics, "view": view, "visibility": values})


class VisibilityCache:
    def __init__(self, directory: Path, source_hash: str):
        self.source_hash = str(source_hash)
        self.directory = Path(directory)/"slam_visibility"/source_hash
        self.directory.mkdir(parents=True, exist_ok=True)
        self.recomputations = 0
        self.hits = 0

    def _path(self, key):
        return self.directory/f"{key}.npz"

    def load(self, key):
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as values:
                result = ViewVisibility(values["point_indices"], values["pixels"], values["distance"],
                                        values["incidence_angle"], values["radial_position"], values["depth_map"],
                                        int(values["width"]), int(values["height"]))
            self.hits += 1
            return result
        except (OSError, ValueError, KeyError):
            return None

    def save(self, key, result: ViewVisibility):
        path = self._path(key)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
        np.savez_compressed(temporary, point_indices=result.point_indices, pixels=result.pixels,
                            distance=result.distance, incidence_angle=result.incidence_angle,
                            radial_position=result.radial_position, depth_map=result.depth_map,
                            width=np.int32(result.width), height=np.int32(result.height))
        temporary.replace(path)

    def get_or_compute(self, key, builder):
        result = self.load(key)
        if result is not None:
            return result, True
        result = builder()
        self.save(key, result)
        self.recomputations += 1
        return result, False

    def visibility_for_view(self, index, pose, projector, config, *, intrinsics, view, checkpoint=lambda: None,
                            progress=None, image=None, diagnostics_directory=None):
        """Fetch or render one view and optionally produce its diagnostic pair."""
        key = visibility_key(self.source_hash, pose, intrinsics, view, config)
        from .visibility import visibility_for_view
        result, cached = self.get_or_compute(
            key, lambda: visibility_for_view(index, pose, projector, config, checkpoint))
        checkpoint()
        if progress is not None:
            progress("slam_visibility", cached)
        if config.get("slam_diagnostics") and image is not None and diagnostics_directory is not None:
            from .diagnostics import write_view_diagnostics
            write_view_diagnostics(diagnostics_directory, str(view.get("id", key)), image, result)
        return result, cached
