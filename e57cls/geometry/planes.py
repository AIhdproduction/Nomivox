"""Seeded bounded RANSAC and least-squares plane refinement."""
import numpy as np


def fit_plane(points, tolerance=.02, seed=42, trials=200, min_fraction=.5, horizontal=False, sample_limit=20_000):
    points = np.asarray(points, np.float64)
    if len(points) < 12:
        return None
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), sample_limit), replace=False)]
    origin = sample.mean(axis=0)
    centered = sample-origin
    best = np.zeros(len(sample), bool)
    for _ in range(trials):
        a, b, c = centered[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if horizontal and abs(normal[2]) < np.cos(np.deg2rad(15)):
            continue
        inliers = np.abs((centered-a)@normal) <= tolerance
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < min_fraction or best.sum() < 12:
        return None
    center = sample[best].mean(axis=0)
    _, singular, vh = np.linalg.svd(sample[best]-center, full_matrices=False)
    if singular[1] < tolerance:
        return None
    normal = vh[-1]
    residual = np.abs((sample[best]-center)@normal)
    if horizontal and abs(normal[2]) < np.cos(np.deg2rad(15)):
        return None
    return {'center': center, 'normal': normal, 'basis': vh[:2], 'inlier_fraction': float(best.mean()),
            'residual_p95': float(np.quantile(residual, .95)), 'points': sample[best]}
