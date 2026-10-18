"""Camera records and perspective views derived from E57 ``images2D`` metadata."""

from dataclasses import dataclass
from typing import Iterator

from e57cls.geometry.cameras import Camera, Pose, View, plan_views


@dataclass(frozen=True)
class SlamCamera:
    """One usable E57 image camera.  No pose or timestamp is inferred."""

    image_index: int
    pose: Pose
    camera: Camera
    timestamp: float | None = None

    def views(self, profile="balanced", size=1512, fov=95) -> list[View | Camera]:
        """Return perspective views for panoramas and the native view for pinholes."""
        return plan_views(self.camera, profile, size, fov) or [self.camera]


def _timestamp(value):
    """Find an explicitly supplied E57-style timestamp without guessing its unit."""
    if not isinstance(value, dict):
        return None
    for key in ("dateTimeValue", "timestamp", "time"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    for key in ("acquisitionDateTime", "dateTime"):
        found = _timestamp(value.get(key))
        if found is not None:
            return found
    return None


def build_camera_list(inspection: dict) -> list[SlamCamera]:
    """Create cameras from valid, decodable SLAM-eligible inspection records."""
    result = []
    for item in inspection.get("images", []):
        if item.get("pose_status") != "valid" or not item.get("classification_eligible_slam"):
            continue
        metadata = item.get("metadata", {})
        try:
            representation = next(
                key for key in item.get("representations", ())
                if key != "visualReferenceRepresentation" and key in metadata
                and any(blob in metadata[key] for blob in ("jpegImage", "pngImage"))
            )
            result.append(SlamCamera(int(item["index"]), Pose.from_dict(metadata["pose"]),
                                     Camera(representation, metadata[representation]), _timestamp(metadata)))
        except (KeyError, StopIteration, TypeError, ValueError):
            # The inspector records the reason and SLAM is specified to skip it.
            continue
    return result


def iter_camera_views(cameras: list[SlamCamera], profile="balanced", size=1512, fov=95) -> Iterator[tuple[SlamCamera, View | Camera]]:
    for camera in cameras:
        for view in camera.views(profile, size, fov):
            yield camera, view
