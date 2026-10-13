"""Camera equations from the E57 committee's coordinate-system guide."""

from dataclasses import dataclass
import math

import numpy as np

from e57cls.io.inspector import pose_status


@dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_dict(cls, value: dict):
        if pose_status(value) != "valid":
            raise ValueError("A finite unit-quaternion pose is required")
        w, x, y, z = (value["rotation"][k] for k in ("w", "x", "y", "z"))
        matrix = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                           [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                           [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]], np.float64)
        return cls(matrix, np.array([value["translation"][k] for k in "xyz"], np.float64))

    def to_world(self, local):
        return np.asarray(local, np.float64) @ self.rotation.T + self.translation

    def from_world(self, world):
        return (np.asarray(world, np.float64) - self.translation) @ self.rotation


def scan_to_camera(local, scan: Pose, camera: Pose):
    # Subtract the large origins first; never cast georeferenced coordinates to float32.
    return np.asarray(local, np.float64) @ (scan.rotation.T @ camera.rotation) + (scan.translation - camera.translation) @ camera.rotation


@dataclass(frozen=True)
class Camera:
    kind: str
    params: dict

    def __post_init__(self):
        requirements = {
            "sphericalRepresentation": ("pixelWidth", "pixelHeight"),
            "pinholeRepresentation": ("pixelWidth", "pixelHeight", "focalLength", "principalPointX", "principalPointY"),
            "cylindricalRepresentation": ("pixelWidth", "pixelHeight", "radius", "principalPointY")}
        if self.kind not in requirements:
            raise ValueError(f"No projectable camera model: {self.kind}")
        keys = ("imageWidth", "imageHeight", *requirements[self.kind])
        if any(k not in self.params or not math.isfinite(self.params[k]) for k in keys):
            raise ValueError("Missing or nonfinite projection parameters")
        positive = [k for k in keys if not k.startswith("principal")]
        if any(self.params[k] <= 0 for k in positive):
            raise ValueError("Camera dimensions and sampling parameters must be positive")

    @property
    def width(self):
        return int(self.params["imageWidth"])

    @property
    def height(self):
        return int(self.params["imageHeight"])

    @property
    def wraps(self):
        return self.kind != "pinholeRepresentation" and abs(self.width * self.params["pixelWidth"] - 2*np.pi) < 1e-6

    def project(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        x, y, z = xyz.T
        p = self.params
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind == "sphericalRepresentation":
                depth = np.linalg.norm(xyz, axis=1)
                u = self.width/2 - np.arctan2(y, x) / p["pixelWidth"]
                v = self.height/2 - np.arctan2(z, np.hypot(x, y)) / p["pixelHeight"]
            elif self.kind == "cylindricalRepresentation":
                depth = np.hypot(x, y)
                u = self.width/2 - np.arctan2(y, x) / p["pixelWidth"]
                v = p["principalPointY"] - z * p["radius"] / (depth*p["pixelHeight"])
            else:
                depth = -z
                u = p["principalPointX"] - x/z*p["focalLength"]/p["pixelWidth"]
                v = p["principalPointY"] - y/z*p["focalLength"]/p["pixelHeight"]
        if self.wraps:
            u = u % self.width
        uv = np.column_stack((u, v))
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return uv, depth, valid

    def rays(self, uv):
        uv = np.asarray(uv, np.float64)
        p = self.params
        if self.kind == "pinholeRepresentation":
            rays = np.column_stack(((uv[:, 0] - p["principalPointX"])*p["pixelWidth"]/p["focalLength"],
                                    (uv[:, 1] - p["principalPointY"])*p["pixelHeight"]/p["focalLength"],
                                    -np.ones(len(uv))))
        else:
            theta = (self.width/2 - uv[:, 0])*p["pixelWidth"]
            if self.kind == "sphericalRepresentation":
                phi = (self.height/2 - uv[:, 1])*p["pixelHeight"]
                rays = np.column_stack((np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)))
            else:
                rays = np.column_stack((np.cos(theta), np.sin(theta),
                                        (p["principalPointY"] - uv[:, 1])*p["pixelHeight"]/p["radius"]))
        return rays / np.linalg.norm(rays, axis=1)[:, None]


@dataclass(frozen=True)
class View:
    id: str
    yaw: float
    pitch: float
    fov_degrees: float = 95
    size: int = 1512

    def __post_init__(self):
        if not 0 < self.fov_degrees < 180 or self.size < 1:
            raise ValueError("Invalid perspective view size or field of view")

    @property
    def basis(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        forward = np.array([math.cos(p)*math.cos(y), math.cos(p)*math.sin(y), math.sin(p)])
        right = np.array([math.sin(y), -math.cos(y), 0.])
        up = np.cross(right, forward)
        return np.column_stack((right, -up, forward))

    @property
    def focal(self):
        return self.size / (2*math.tan(math.radians(self.fov_degrees)/2))

    def project(self, camera_xyz):
        xyz = np.asarray(camera_xyz, np.float64) @ self.basis
        depth = xyz[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.size/2 + self.focal*xyz[:, :2]/depth[:, None]
        valid = np.isfinite(uv).all(axis=1) & (depth > 0)
        valid &= ((uv >= 0) & (uv < self.size)).all(axis=1)
        return uv, depth, valid

    def rays(self, uv):
        rays = np.column_stack(((np.asarray(uv)-self.size/2)/self.focal, np.ones(len(uv))))
        rays = rays @ self.basis.T
        return rays / np.linalg.norm(rays, axis=1)[:, None]


def plan_views(camera: Camera, profile: str = "balanced", cap: int = 1512, fov: float = 95) -> list[View]:
    if camera.kind == "pinholeRepresentation":
        return []
    if profile not in ("fast", "balanced", "high"):
        raise ValueError("Unknown view profile")
    size = min(cap, max(1, round(2*math.tan(math.radians(fov)/2)/min(camera.params["pixelWidth"], camera.params["pixelHeight"]))))
    rings = [(0, 6), (50, 3)] if profile == "fast" else [(0, 8), (50, 4), (-50, 4), (90, 1)]
    return [View(f"view_{i:02d}", yaw, pitch, fov, size)
            for i, (yaw, pitch) in enumerate((360*j/n, pitch) for pitch, n in rings for j in range(n))]
