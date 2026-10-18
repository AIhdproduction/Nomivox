"""Optional visual checks for SLAM depth maps and visible points."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_view_diagnostics(directory: Path, view_id: str, image, visibility) -> tuple[Path, Path]:
    """Write a normalized depth map and a visible-point overlay for one view."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(visibility.depth_map, np.float32)
    finite = depth[np.isfinite(depth)]
    scaled = np.zeros(depth.shape, np.uint8)
    if len(finite):
        low, high = np.percentile(finite, (1, 99))
        scaled[np.isfinite(depth)] = np.clip(255*(depth[np.isfinite(depth)]-low)/max(high-low, 1e-6), 0, 255).astype(np.uint8)
    depth_path = directory/f"{view_id}_depth.png"
    Image.fromarray(scaled).save(depth_path)
    overlay = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for x, y in np.asarray(visibility.pixels):
        draw.ellipse((float(x)-1, float(y)-1, float(x)+1, float(y)+1), fill=(255, 64, 32))
    overlay_path = directory/f"{view_id}_visible.png"
    overlay.save(overlay_path)
    return depth_path, overlay_path
