"""Deterministic virtual stations for mobile camera trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VirtualStation:
    """A contiguous trajectory segment used as one independent observation."""

    id: int
    camera_indices: tuple[int, ...]
    position: np.ndarray


def ordered_camera_indices(cameras) -> list[int]:
    """Order cameras by explicit time, otherwise build a deterministic NN chain."""
    if not cameras:
        return []
    if all(camera.timestamp is not None for camera in cameras):
        return sorted(range(len(cameras)), key=lambda index: (cameras[index].timestamp, cameras[index].image_index))
    positions = np.asarray([camera.pose.translation for camera in cameras], np.float64)
    start = min(range(len(cameras)), key=lambda index: (*positions[index].tolist(), cameras[index].image_index))
    result, remaining = [start], set(range(len(cameras))) - {start}
    while remaining:
        previous = result[-1]
        chosen = min(remaining, key=lambda index: (float(np.sum((positions[index]-positions[previous])**2)),
                                                    cameras[index].image_index))
        result.append(chosen)
        remaining.remove(chosen)
    return result


def cluster_stations(cameras, spacing: float = 1.0) -> tuple[list[VirtualStation], dict[int, int], str]:
    """Cluster ordered camera positions into one-metre trajectory stations.

    A station starts once its next camera is at least ``spacing`` metres from
    the first camera of the current station.  This keeps the result stable and
    ensures that every view is assigned to exactly one independent station.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("SLAM station spacing must be finite and positive")
    order = ordered_camera_indices(cameras)
    ordering = "timestamp" if cameras and all(camera.timestamp is not None for camera in cameras) else "nearest_neighbour"
    stations, membership = [], {}
    current = []
    anchor = None
    for index in order:
        position = np.asarray(cameras[index].pose.translation, np.float64)
        if current and np.linalg.norm(position-anchor) >= spacing:
            station_id = len(stations)+1
            stations.append(VirtualStation(station_id, tuple(current), np.mean(
                [cameras[item].pose.translation for item in current], axis=0)))
            for item in current:
                membership[item] = station_id
            current, anchor = [], None
        if not current:
            anchor = position
        current.append(index)
    if current:
        station_id = len(stations)+1
        stations.append(VirtualStation(station_id, tuple(current), np.mean(
            [cameras[item].pose.translation for item in current], axis=0)))
        for item in current:
            membership[item] = station_id
    return stations, membership, ordering
