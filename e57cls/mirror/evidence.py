"""Project-wide finite mirror planes and exact, bounded direct-geometry evidence."""
from collections import OrderedDict
from itertools import product
from pathlib import Path
import sqlite3
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from e57cls.geometry.planes import fit_plane


class Plane:
    def __init__(self, center, normal, basis, outline, frames, quality=False):
        self.center, self.normal, self.basis = map(lambda x: np.asarray(x, np.float64), (center, normal, basis))
        self.outline, self.frames = np.asarray(outline, np.float64), np.asarray(frames, np.float64)
        self.quality = quality
        self.hull = ConvexHull(self.outline)

    def contains(self, xyz, tolerance=1e-8):
        uv = (xyz-self.center)@self.basis.T
        return (uv@self.hull.equations[:,:2].T+self.hull.equations[:,2] <= tolerance).all(axis=1)

    def serialize(self):
        return {k: getattr(self,k).tolist() for k in ('center','normal','basis','outline','frames')} | {'quality': self.quality}


class Registry:
    def __init__(self, planes=()):
        self.planes = list(planes)

    def add(self, plane, angle=3., distance=.03):
        for i, other in enumerate(self.planes):
            if abs(other.normal@plane.normal) < np.cos(np.deg2rad(angle)) or abs((other.center-plane.center)@plane.normal) > distance:
                continue
            other_vertices = other.center+other.outline@other.basis
            new_vertices = plane.center+plane.outline@plane.basis
            # Separating-axis polygon intersection, including crossing thin rectangles.
            a = (other_vertices-other.center)@other.basis.T
            b = (new_vertices-other.center)@other.basis.T
            axes = np.vstack((ConvexHull(a).equations[:,:2], ConvexHull(b).equations[:,:2]))
            if any((a@axis).max() < (b@axis).min() or (b@axis).max() < (a@axis).min() for axis in axes):
                continue
            frames = np.vstack((other.frames, plane.frames))
            frames = frames[::max(1, len(frames)//20000)]
            fit = fit_plane(frames, tolerance=distance, min_fraction=.8)
            if fit is None:
                continue
            outline = (np.vstack((other_vertices, new_vertices))-fit['center'])@fit['basis'].T
            self.planes[i] = Plane(fit['center'], fit['normal'], fit['basis'], outline,
                                   fit['points'], other.quality or plane.quality)
            return
        self.planes.append(plane)

    def rays(self, origin, points, tolerance=.03):
        selected = np.full(len(points), -1, np.int32)
        states = np.zeros(len(points), np.uint8)
        nearest = np.full(len(points), np.inf)
        directions = points-origin
        lengths = np.linalg.norm(directions, axis=1)
        for i, plane in enumerate(self.planes):
            denominator = directions@plane.normal
            fraction = np.divide((plane.center-origin)@plane.normal, denominator,
                                 out=np.full(len(points), np.inf), where=np.abs(denominator)>1e-10)
            finite = np.isfinite(fraction) & (fraction > 0) & (fraction <= 1+tolerance/np.maximum(lengths,1e-10))
            ids = np.flatnonzero(finite)
            hit = origin+fraction[ids,None]*directions[ids]
            inside = plane.contains(hit)
            ids = ids[inside]
            distance = (1-fraction[ids])*lengths[ids]
            take = (distance >= -tolerance) & (fraction[ids] < nearest[ids])
            ids, distance = ids[take], distance[take]
            selected[ids] = i
            nearest[ids] = fraction[ids]
            states[ids] = np.where(distance <= tolerance, 1, 4)
        return selected, states


class DirectIndex:
    """Disk fragments of all direct points; each KD tree contains <=100000 points."""
    def __init__(self, path: Path, origin, block=1.):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.execute('CREATE TABLE IF NOT EXISTS fragments(x INTEGER,y INTEGER,z INTEGER,scan INTEGER,points BLOB)')
        self.db.execute('CREATE INDEX IF NOT EXISTS spatial ON fragments(x,y,z)')
        self.origin, self.block = np.asarray(origin), block
        self.cache = OrderedDict()

    def add(self, points, scan, commit=True):
        bins = np.floor((points-self.origin)/self.block).astype(np.int64)
        unique, inverse = np.unique(bins, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        offsets = np.r_[0, np.cumsum(np.bincount(inverse))]
        self.db.executemany('INSERT INTO fragments VALUES(?,?,?,?,?)',
                           ((*map(int,cell), scan, np.ascontiguousarray(points[order[offsets[i]:offsets[i+1]]], dtype='<f8').tobytes()) for i,cell in enumerate(unique)))
        # The multiscan reflection check can index millions of temporary points.
        # Its caller may keep one transaction open; queries on this connection
        # still see those rows, while avoiding one disk sync per input chunk.
        if commit:
            self.db.commit()

    def nearby(self, points, radius, exclude_scan=None):
        result = np.zeros(len(points), bool)
        bins = np.floor((points-self.origin)/self.block).astype(np.int64)
        unique, inverse = np.unique(bins, axis=0, return_inverse=True)
        reach = int(np.ceil(radius/self.block))
        for i,cell in enumerate(unique):
            ids = np.flatnonzero(inverse == i)
            for delta in product(range(-reach,reach+1), repeat=3):
                key = tuple(map(int, cell+delta))
                for rowid, scan, blob in self.db.execute('SELECT rowid,scan,points FROM fragments WHERE x=? AND y=? AND z=?', key):
                    if scan == exclude_scan:
                        continue
                    if rowid not in self.cache:
                        self.cache[rowid] = cKDTree(np.frombuffer(blob, '<f8').reshape(-1,3))
                        if len(self.cache) > 24:
                            self.cache.popitem(last=False)
                    tree = self.cache[rowid]
                    self.cache.move_to_end(rowid)
                    pending = ids[~result[ids]]
                    if not len(pending):
                        break
                    distances, _ = tree.query(points[pending], distance_upper_bound=radius)
                    result[pending] |= np.isfinite(distances)
        return result

    def close(self):
        self.cache.clear()
        self.db.close()


def label_points(registry, direct, origin, points, scan, tolerance=.03, radius=.05, unknown_mask=None):
    plane_ids, states = registry.rays(origin, points, tolerance)
    if unknown_mask is not None:
        states[unknown_mask & (states == 0)] = 4
    candidates = np.flatnonzero((states == 4) & (plane_ids >= 0))
    confirmed = direct.nearby(points[candidates], radius, exclude_scan=scan)
    states[candidates[confirmed]] = 3
    for index, plane in enumerate(registry.planes):
        ids = candidates[~confirmed & (plane_ids[candidates] == index)]
        if not len(ids):
            continue
        if plane.quality:
            reflected = np.ones(len(ids), bool)
        else:
            mirrored = points[ids]-2*((points[ids]-plane.center)@plane.normal)[:,None]*plane.normal
            reflected = direct.nearby(mirrored, radius)
        states[ids[reflected]] = 2
    return states
