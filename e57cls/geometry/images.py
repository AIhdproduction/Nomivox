"""Bounded blob extraction and strip-wise panorama resampling."""

import os
from pathlib import Path
import warnings

import numpy as np
from PIL import Image


def extract_blob(blob, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            for offset in range(0, blob.byteCount(), 1024*1024):
                count = min(1024*1024, blob.byteCount()-offset)
                data = np.empty(count, np.uint8)
                blob.read(data, offset, count)
                stream.write(data.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def open_image(path: Path, mode: str, max_pixels: int = 150_000_000):
    # Header check precedes decoding; keep an explicit bound for large TLS panoramas.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.width*image.height > max_pixels:
                raise ValueError(f"Image exceeds configured pixel budget: {image.size}")
            return np.array(image.convert(mode))


def bilinear(image, uv, wrap):
    h, w = image.shape[:2]
    positions = np.asarray(uv, np.float64) - .5
    xy = np.floor(positions).astype(np.int64)
    fraction = positions - xy
    x0, y0 = xy[:, 0], xy[:, 1]
    x1, y1 = x0+1, y0+1
    if wrap:
        x0, x1 = x0 % w, x1 % w
    else:
        x0, x1 = x0.clip(0, w-1), x1.clip(0, w-1)
    y0, y1 = y0.clip(0, h-1), y1.clip(0, h-1)
    fx, fy = fraction[:, 0], fraction[:, 1]
    if image.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((1-fx)*(1-fy)*image[y0, x0] + fx*(1-fy)*image[y0, x1]
            + (1-fx)*fy*image[y1, x0] + fx*fy*image[y1, x1])


def render_view(image, mask, camera, view):
    rendered = np.zeros((view.size, view.size, 3), np.uint8)
    valid_image = np.zeros((view.size, view.size), bool)
    for start in range(0, view.size, 64):
        stop = min(start+64, view.size)
        x, y = np.meshgrid(np.arange(view.size)+.5, np.arange(start, stop)+.5)
        rays = view.rays(np.column_stack((x.ravel(), y.ravel())))
        uv, _, valid = camera.project(rays)
        safe_uv = np.nan_to_num(uv)
        if mask is not None:
            # All four interpolation neighbors must be valid to avoid mask bleed.
            valid &= bilinear((mask > 0).astype(np.float32), safe_uv, camera.wraps) >= 1-1e-6
        values = bilinear(image, safe_uv, camera.wraps)
        values[~valid] = 0
        rendered[start:stop] = np.rint(values).clip(0, 255).astype(np.uint8).reshape(stop-start, view.size, 3)
        valid_image[start:stop] = valid.reshape(stop-start, view.size)
    return rendered, valid_image
