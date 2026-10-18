"""Torch depth maps and per-view point visibility."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import minimum_filter

from e57cls.geometry.cameras import Camera, Pose, View


@dataclass(frozen=True)
class ViewVisibility:
    point_indices: np.ndarray
    pixels: np.ndarray
    distance: np.ndarray
    incidence_angle: np.ndarray
    radial_position: np.ndarray
    depth_map: np.ndarray
    width: int
    height: int

    @staticmethod
    def empty(width, height):
        return ViewVisibility(np.empty(0, np.int64), np.empty((0, 2), np.float32),
                              np.empty(0, np.float32), np.empty(0, np.float32),
                              np.empty(0, np.float32), np.full((height, width), np.inf, np.float32), width, height)


def _device(requested):
    import torch
    if requested and str(requested).startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _dimensions(projector, resolution):
    if isinstance(projector, View):
        source_width = source_height = projector.size
    else:
        source_width, source_height = projector.width, projector.height
    scale = min(1., float(resolution)/max(source_width, source_height))
    return source_width, source_height, max(1, round(source_width*scale)), max(1, round(source_height*scale))


def _project(world, pose, projector):
    camera_xyz = pose.from_world(world)
    return projector.project(camera_xyz), camera_xyz


def _torch_depth(uv, depth, valid, spacing, width, height, focal, splat_max_px, device, checkpoint,
                 splat_batch_entries=8_000_000):
    """Render an amin z-buffer with square, bounded-size Torch splats.

    ``splat_max_px`` is the maximum footprint edge length in pixels, not a
    radius.  Even edge lengths are deterministically shifted one half-pixel
    towards smaller pixel coordinates.  Expansion stays on the selected Torch
    device and is processed in portions to bound the temporary allocation.
    """
    import torch
    if int(splat_batch_entries) < 1:
        raise ValueError("Splat batch entry budget must be positive")
    target = torch.full((height*width,), float("inf"), dtype=torch.float64, device=device)
    if not np.any(valid):
        return target.reshape(height, width).cpu().numpy().astype(np.float32)

    # A batch cannot contain more entries than this even if every point uses
    # the configured maximum footprint.  Keep this intentionally below the
    # 32-million-entry upper limit to leave headroom for Torch temporaries.
    points_per_batch = max(1, int(splat_batch_entries)//(int(splat_max_px)**2))
    for start in range(0, len(depth), points_per_batch):
        checkpoint()
        stop = min(len(depth), start+points_per_batch)
        batch_valid = valid[start:stop]
        if not np.any(batch_valid):
            continue
        values = depth[start:stop][batch_valid]
        footprint = np.clip(np.ceil(spacing[start:stop][batch_valid]/np.maximum(values, .3)*focal),
                            1, splat_max_px).astype(np.int64)
        batch_uv = torch.as_tensor(uv[start:stop][batch_valid], dtype=torch.float64, device=device)
        batch_depth = torch.as_tensor(values, dtype=torch.float64, device=device)
        batch_footprint = torch.as_tensor(footprint, dtype=torch.int64, device=device)
        base = torch.floor(batch_uv).to(torch.int64)
        for edge in range(1, int(splat_max_px)+1):
            selected = batch_footprint == edge
            if not bool(torch.any(selected)):
                continue
            checkpoint()
            offset_start = -(edge//2)
            offsets = torch.arange(offset_start, offset_start+edge, dtype=torch.int64, device=device)
            offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
            x = base[selected, 0, None] + offset_x.reshape(1, -1)
            y = base[selected, 1, None] + offset_y.reshape(1, -1)
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            flat = (y*width+x)[inside]
            source = batch_depth[selected, None].expand(-1, edge*edge)[inside]
            target.scatter_reduce_(0, flat, source, reduce="amin", include_self=True)
    return target.reshape(height, width).cpu().numpy().astype(np.float32)


def _render(uv, depth, valid, spacing, source_width, source_height, width, height, focal, splat_max_px, device,
            checkpoint, splat_batch_entries):
    scaled = uv*np.asarray((width/source_width, height/source_height))
    depth_map = _torch_depth(scaled, depth, valid, spacing, width, height, focal*width/source_width,
                             splat_max_px, device, checkpoint, splat_batch_entries)
    return depth_map, scaled


def _conservative_visibility_depth(depth_map):
    """Close one-pixel bounded-splat gaps for the visibility comparison only."""
    # A bounded splat can leave a one-pixel raster gap when a close, sparse
    # foreground surface reaches its configured footprint limit.  Close that
    # quantisation gap conservatively so a distant point cannot receive the
    # foreground pixel's semantic label through it.
    return minimum_filter(depth_map, size=3, mode="constant", cval=np.inf)


def _footprint_focal(projector, source_width):
    """Return horizontal image samples per radian for spacing-based splats."""
    if isinstance(projector, View):
        return projector.focal
    if isinstance(projector, Camera):
        if projector.kind in ("sphericalRepresentation", "cylindricalRepresentation"):
            return source_width/(2*math.pi)
        return projector.params.get("focalLength", source_width/2)
    return projector.focal


def visibility_for_points(world, point_indices, pose: Pose, projector: View | Camera, *, spacing=None,
                          normals=None, max_range=20., depth_resolution=1024, splat_max_px=6,
                          tolerance_abs=.02, tolerance_rel=.01, device="cuda:0", checkpoint=lambda: None,
                          splat_batch_entries=8_000_000, close_depth_gaps=True) -> ViewVisibility:
    """Calculate visible points with 1-to-``splat_max_px``-pixel square splats."""
    if (not 0 < max_range or not 64 <= int(depth_resolution) or not 1 <= int(splat_max_px) <= 64
            or int(splat_batch_entries) < 1):
        raise ValueError("Invalid SLAM depth-map settings")
    world = np.asarray(world, np.float64)
    point_indices = np.asarray(point_indices, np.int64)
    if world.ndim != 2 or world.shape != (len(point_indices), 3):
        raise ValueError("World points and point indices must have matching Nx3/N shapes")
    absolute = np.asarray(tolerance_abs, np.float64)
    relative = np.asarray(tolerance_rel, np.float64)
    if absolute.ndim > 1 or relative.ndim > 1 or (absolute.ndim == 1 and absolute.shape != (len(world),)) \
            or (relative.ndim == 1 and relative.shape != (len(world),)) \
            or np.any(absolute < 0) or np.any(relative < 0):
        raise ValueError("Visibility tolerances must be nonnegative")
    source_width, source_height, width, height = _dimensions(projector, depth_resolution)
    if not len(world):
        return ViewVisibility.empty(width, height)
    (uv, depth, projected), camera_xyz = _project(world, pose, projector)
    valid = projected & (depth >= .3) & (depth <= max_range)
    spacing = np.asarray(spacing if spacing is not None else np.full(len(world), .05), np.float64)
    if spacing.shape != (len(world),):
        raise ValueError("Point spacing must have one value per point")
    focal = _footprint_focal(projector, source_width)
    depth_map, scaled = _render(uv, depth, valid, spacing, source_width, source_height, width, height, focal,
                                int(splat_max_px), _device(device), checkpoint, int(splat_batch_entries))
    pixel = np.zeros((len(world), 2), np.int64)
    finite_uv = np.isfinite(scaled).all(axis=1)
    pixel[finite_uv] = np.floor(scaled[finite_uv]).astype(np.int64)
    inside = valid & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    reference = np.full(len(world), np.inf, np.float32)
    visibility_depth = _conservative_visibility_depth(depth_map) if close_depth_gaps else depth_map
    reference[inside] = visibility_depth[pixel[inside, 1], pixel[inside, 0]]
    accepted = inside & np.isfinite(reference) & (np.abs(depth-reference) <= absolute+relative*depth)
    if normals is None:
        incidence = np.zeros(len(world), np.float32)
    else:
        normal = np.asarray(normals, np.float64)
        rays = camera_xyz/np.maximum(np.linalg.norm(camera_xyz, axis=1, keepdims=True), 1e-12)
        cosine = np.abs(np.einsum("ij,ij->i", normal, rays))
        incidence = np.arccos(np.clip(cosine, 0, 1)).astype(np.float32)
        incidence[~np.isfinite(incidence)] = 0
    centre = np.asarray((source_width/2, source_height/2))
    radial = np.linalg.norm((uv-centre)/np.maximum(centre, 1), axis=1).astype(np.float32)
    sam_pixels = uv.astype(np.float32)
    return ViewVisibility(point_indices[accepted], sam_pixels[accepted], depth[accepted].astype(np.float32),
                          incidence[accepted], radial[accepted], depth_map, width, height)


def visibility_for_view(index, pose: Pose, projector: View | Camera, config: dict, checkpoint=lambda: None) -> ViewVisibility:
    """Load only range candidates from a spatial index, then render one view."""
    maximum = float(config.get("slam_max_range") or config.get("slam_max_range_indoor", 20.))
    candidates = index.query_range(pose.translation, maximum, checkpoint)
    checkpoint()
    return visibility_for_points(candidates["xyz"], candidates["point_id"], pose, projector,
                                 spacing=candidates["spacing"], normals=candidates["normal"], max_range=maximum,
                                 depth_resolution=config.get("slam_depth_resolution", 1024),
                                 splat_max_px=config.get("slam_splat_max_px", 6),
                                 tolerance_abs=config.get("slam_depth_tol_abs", .02),
                                 tolerance_rel=config.get("slam_depth_tol_rel", .01),
                                 device=config.get("gpu_device", "cuda:0"), checkpoint=checkpoint,
                                 splat_batch_entries=config.get("slam_splat_batch_entries", 8_000_000))


class _SphericalProjector:
    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self.focal = width/(2*math.pi)

    def project(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        depth = np.linalg.norm(xyz, axis=1)
        uv = np.column_stack((self.width/2-np.arctan2(xyz[:, 1], xyz[:, 0])/(2*np.pi/self.width),
                              self.height/2-np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))/(np.pi/self.height)))
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv, depth, valid


def origin_depth_width(point_count, maximum=1024, pixels_per_sample=7) -> int:
    """Choose an origin-map width proportional to the sampled point count.

    A spherical map has roughly ``width**2 / 2`` pixels.  Seven pixels per
    sample preserve the depth slope of an ordinary static scan (including
    room corners) while still lowering the 1024-pixel map for small samples.
    The 64-pixel floor is required by the shared renderer; ``maximum`` keeps
    the rule bounded for unusually large samples.
    """
    if int(point_count) < 0 or int(maximum) < 64 or int(pixels_per_sample) < 1:
        raise ValueError("Invalid origin depth-map settings")
    target = max(1, int(math.ceil(point_count*int(pixels_per_sample))))
    return min(int(maximum), max(64, int(math.ceil(math.sqrt(2*target)))))


def _origin_spacing(values):
    """Estimate a bounded local sample spacing for the one-off origin test."""
    from scipy.spatial import cKDTree

    if len(values) < 2:
        return np.full(len(values), .005, np.float64)
    distance, _ = cKDTree(values).query(values, k=2, workers=-1)
    # The floor retains a finite splat for duplicate samples.  The 25 cm cap
    # prevents a single sparse gap from accepting an unrelated background.
    return np.clip(distance[:, 1], .005, .25)


def visibility_from_origin(points, *, origin=(0., 0., 0.), width=1024, tolerance_abs=.02,
                           tolerance_rel=.01, splat_max_px=1, device="cuda:0") -> np.ndarray:
    """Return the visible-point mask for the rule-d spherical origin test.

    ``width`` is an upper bound, not a fixed resolution: the depth map follows
    the sample size with seven pixels per sample.  A locally estimated
    nearest-neighbour spacing adds a matching absolute tolerance; its 25 cm
    cap avoids accepting an unrelated background across sparse gaps.  The
    independent one-pixel footprint does not inherit ``slam_splat_max_px``,
    which remains exclusive to per-view visibility.
    """
    values = np.asarray(points, np.float64)
    depth_width = origin_depth_width(len(values), width)
    finite = np.isfinite(values).all(axis=1)
    spacing = np.full(len(values), .005, np.float64)
    if np.any(finite):
        spacing[finite] = _origin_spacing(values[finite])
    absolute = np.asarray(tolerance_abs, np.float64)+spacing
    result = visibility_for_points(values, np.arange(len(values)),
                                   Pose(np.eye(3), np.asarray(origin, np.float64)),
                                   _SphericalProjector(depth_width, max(1, depth_width//2)),
                                   spacing=spacing, max_range=float("inf"), depth_resolution=depth_width,
                                   splat_max_px=splat_max_px, tolerance_abs=absolute,
                                   tolerance_rel=tolerance_rel, device=device, close_depth_gaps=False)
    accepted = np.zeros(len(values), bool)
    accepted[result.point_indices] = True
    return accepted
