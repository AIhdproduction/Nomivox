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


def refine_plane(points, inliers, tolerance, horizontal=False):
    """Fit a PCA plane and re-evaluate its support until it settles."""
    points = np.asarray(points, np.float64)
    inliers = np.asarray(inliers, bool)
    for _ in range(3):
        if inliers.sum() < 12:
            return None
        center = points[inliers].mean(axis=0)
        _, singular, vh = np.linalg.svd(points[inliers]-center, full_matrices=False)
        if singular[1] < tolerance:
            return None
        normal = vh[-1]
        if horizontal and abs(normal[2]) < np.cos(np.deg2rad(15)):
            return None
        refined = np.abs((points-center)@normal) <= tolerance
        if np.array_equal(refined, inliers):
            break
        inliers = refined
    if inliers.sum() < 12:
        return None
    center = points[inliers].mean(axis=0)
    _, singular, vh = np.linalg.svd(points[inliers]-center, full_matrices=False)
    if singular[1] < tolerance:
        return None
    normal = vh[-1]
    if horizontal and abs(normal[2]) < np.cos(np.deg2rad(15)):
        return None
    residual = np.abs((points[inliers]-center)@normal)
    return {'center': center, 'normal': normal, 'basis': vh[:2], 'inliers': inliers,
            'residual_p95': float(np.quantile(residual, .95))}


def fit_plane_robust(points, tolerance=.02, seed=42, trials=256, min_fraction=.5, horizontal=False,
                     sample_limit=20_000, max_trials=1024, confidence=.999):
    """Fit a deterministic, adaptively sampled RANSAC plane for geometry fill.

    This deliberately remains separate from :func:`fit_plane`: reflection and
    scene-analysis callers depend on the historic bounded sampling behaviour.
    """
    points = np.asarray(points, np.float64)
    if len(points) < 12:
        return None
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), sample_limit), replace=False)]
    origin = sample.mean(axis=0)
    centered = sample-origin
    best = np.zeros(len(sample), bool)
    needed = int(trials)
    for attempt in range(int(max_trials)):
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
            probability = min(max(float(best.mean()), 1e-12), 1.-1e-12)
            denominator = np.log1p(-probability**3)
            if denominator < 0:
                needed = min(int(max_trials), max(int(trials), int(np.ceil(np.log1p(-confidence)/denominator))))
        if attempt+1 >= needed:
            break
    if best.mean() < min_fraction or best.sum() < 12:
        return None
    result = refine_plane(sample, best, tolerance, horizontal)
    if result is None:
        return None
    result['inlier_fraction'] = float(result['inliers'].mean())
    result['trials'] = attempt+1
    return result
