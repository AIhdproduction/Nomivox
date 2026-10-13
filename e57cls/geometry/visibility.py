"""Depth splats and visibility with explicit depth conventions."""

import numpy as np


class DepthBuffer:
    def __init__(self, width: int, height: int, radius: int = 1, wrap: bool = False):
        if min(width, height) < 1 or not 0 <= radius <= 8:
            raise ValueError("Invalid depth-buffer dimensions or splat radius")
        self.depth = np.full((height, width), np.inf, np.float64)
        self.width, self.height, self.radius, self.wrap = width, height, radius, wrap

    def add(self, uv, depth, valid):
        keep = valid & np.isfinite(depth) & (depth > 0) & np.isfinite(uv).all(axis=1)
        pixels = np.floor(uv[keep]).astype(np.int64)
        values = depth[keep]
        for dy in range(-self.radius, self.radius+1):
            for dx in range(-self.radius, self.radius+1):
                x, y = pixels[:, 0]+dx, pixels[:, 1]+dy
                if self.wrap:
                    x = x % self.width
                inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                np.minimum.at(self.depth, (y[inside], x[inside]), values[inside])

    def visible(self, uv, depth, valid, absolute: float = .03, relative: float = .01):
        if min(absolute, relative) < 0:
            raise ValueError("Visibility tolerances must be nonnegative")
        accepted = np.zeros(len(depth), bool)
        keep = valid & np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
        ids = np.flatnonzero(keep)
        xy = np.floor(uv[keep]).astype(np.int64)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < self.width) & (xy[:, 1] >= 0) & (xy[:, 1] < self.height)
        ids, xy = ids[inside], xy[inside]
        reference = self.depth[xy[:, 1], xy[:, 0]]
        accepted[ids] = np.isfinite(reference) & (np.abs(depth[ids]-reference) <= np.maximum(absolute, relative*depth[ids]))
        return accepted
