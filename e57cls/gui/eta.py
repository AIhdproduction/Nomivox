"""Scan-based runtime estimates for the GUI.

The estimator deliberately has no Qt dependency so its event accounting can be
tested independently from the window.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from e57cls.files import atomic_json


STAGES = ("mirror_masks", "mirror_evidence", "segmentation")
DEFAULT_PROFILE = {
    "mirror_masks": {"avg_seconds": 40.0, "max_seconds": 70.0},
    "mirror_evidence": {"avg_seconds": 103.0, "max_seconds": 511.0},
    "segmentation": {"avg_seconds": 275.0, "max_seconds": 472.0},
}
PRIOR_WEIGHT = 3


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def load_profile(path):
    """Return validated persisted timings merged over the built-in defaults."""
    profile = deepcopy(DEFAULT_PROFILE)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return profile
    if not isinstance(payload, dict):
        return profile
    for stage in STAGES:
        value = payload.get(stage)
        if not isinstance(value, dict):
            continue
        units = value.get("units")
        average = value.get("avg_seconds")
        maximum = value.get("max_seconds")
        if (isinstance(units, int) and units >= 3 and _number(average) and _number(maximum)
                and maximum >= average):
            profile[stage] = {"avg_seconds": float(average), "max_seconds": float(maximum), "units": units}
    return profile


class RunEstimator:
    """Estimate remaining work from completed scan-sized units.

    Paused intervals are removed when matching ``state`` events are supplied.
    Without those events, elapsed time is necessarily treated as active time.
    """
    def __init__(self, applies=None, unit_counts=None, profile=None):
        applies = applies or {}
        unit_counts = unit_counts or {}
        self.applies = {stage: bool(applies.get(stage, False)) for stage in STAGES}
        self.unit_counts = {stage: unit_counts.get(stage) for stage in STAGES}
        self.unit_indices = {stage: None for stage in STAGES}
        self.prior_profile = deepcopy(DEFAULT_PROFILE)
        for stage in STAGES:
            value = (profile or {}).get(stage, {})
            if _number(value.get("avg_seconds")) and _number(value.get("max_seconds")):
                self.prior_profile[stage] = dict(value)
        self.measurements = {stage: [] for stage in STAGES}
        self.stage_started = {stage: None for stage in STAGES}
        self.stage_completed = set()
        self.current_stage = None
        self.current_unit = None
        self.current_unit_started = None
        self.last_completed_index = {stage: None for stage in STAGES}
        self.last_progress_at = {stage: None for stage in STAGES}
        self.phase_started_at = {}
        self.phase_fraction = {}
        self.paused_since = None
        self.paused_seconds = 0.0

    @property
    def counts_known(self):
        return all(not self.applies[stage] or isinstance(self.unit_counts[stage], int)
                   and self.unit_counts[stage] >= 0 for stage in STAGES)

    def set_unit_count(self, stage, count):
        if stage in STAGES and isinstance(count, int) and count >= 0:
            self.unit_counts[stage] = count

    def set_unit_indices(self, stage, indices):
        """Use known scan indices instead of inferred completed-unit counts."""
        if stage not in STAGES:
            return
        try:
            values = sorted({int(index) for index in indices})
        except (TypeError, ValueError):
            return
        if any(index < 1 for index in values):
            return
        self.unit_indices[stage] = values
        self.unit_counts[stage] = len(values)

    def _active_time(self, timestamp):
        timestamp = float(timestamp)
        paused = self.paused_seconds
        if self.paused_since is not None:
            paused += max(0.0, timestamp-self.paused_since)
        return max(0.0, timestamp-paused)

    def state(self, state, timestamp):
        timestamp = float(timestamp)
        if state == "paused" and self.paused_since is None:
            self.paused_since = timestamp
        elif state == "running" and self.paused_since is not None:
            self.paused_seconds += max(0.0, timestamp-self.paused_since)
            self.paused_since = None

    def phase_started(self, phase, timestamp):
        now = self._active_time(timestamp)
        self.phase_started_at[phase] = now
        if phase == "mirror":
            self._start_stage("mirror_masks", now)
        elif phase == "segmentation":
            self._start_stage("segmentation", now)
        elif phase in ("voting", "output"):
            if self.current_stage in STAGES:
                self._finish_stage(self.current_stage, now)
            self.current_stage = phase

    def phase_finished(self, phase, timestamp):
        now = self._active_time(timestamp)
        if phase == "mirror":
            self._finish_stage("mirror_masks", now)
            self._finish_stage("mirror_evidence", now)
        elif phase == "segmentation":
            self._finish_stage("segmentation", now)

    def progress(self, phase, timestamp, **event):
        now = self._active_time(timestamp)
        if phase in ("voting", "output"):
            self.phase_fraction[phase] = max(0.0, min(1.0, float(event.get("fraction", 0.0))))
            return
        if phase not in STAGES:
            return
        if phase == "mirror_masks" and isinstance(event.get("image_count"), int):
            self.set_unit_count(phase, event["image_count"])
        if phase == "mirror_evidence" and isinstance(event.get("scan_count"), int):
            self.set_unit_count(phase, event["scan_count"])
            if self.unit_indices[phase] is None:
                self.set_unit_indices(phase, range(1, event["scan_count"]+1))
        self._start_stage(phase, now)
        unit = event.get("image") if phase == "mirror_masks" else event.get("scan")
        if phase == "mirror_evidence":
            self._complete_current(now)
            if isinstance(unit, int):
                self.last_completed_index[phase] = unit
            self.last_progress_at[phase] = now
            return
        if unit != self.current_unit:
            if self.current_unit is not None:
                self._complete_current(now)
            self.current_unit = unit
        self.last_progress_at[phase] = now

    def _start_stage(self, stage, now):
        if not self.applies.get(stage) or stage in self.stage_completed:
            return
        if self.current_stage and self.current_stage != stage:
            previous = self.current_stage
            boundary = self.last_progress_at[previous]
            now = boundary if boundary is not None else now
            self._finish_stage(previous, now)
        if self.stage_started[stage] is None:
            self.stage_started[stage] = now
        self.current_stage = stage
        if self.current_unit_started is None:
            self.current_unit_started = now
            self.current_unit = None

    def _complete_current(self, now):
        if self.current_stage is None or self.current_unit_started is None:
            return
        stage = self.current_stage
        count = self.unit_counts.get(stage)
        if count is None or len(self.measurements[stage]) < count:
            self.measurements[stage].append(max(0.0, now-self.current_unit_started))
        if isinstance(self.current_unit, int):
            self.last_completed_index[stage] = self.current_unit
        self.current_unit_started = now
        self.current_unit = None

    def _finish_stage(self, stage, now):
        if not self.applies.get(stage) or stage in self.stage_completed:
            return
        if self.current_stage == stage and self.current_unit_started is not None:
            self._complete_current(now)
        self.stage_completed.add(stage)
        if self.current_stage == stage:
            self.current_stage = None
            self.current_unit = None
            self.current_unit_started = None

    def per_unit(self, stage):
        prior = self.prior_profile[stage]
        values = self.measurements[stage]
        if not values:
            return float(prior["avg_seconds"]), float(prior["max_seconds"])
        average = (float(prior["avg_seconds"])*PRIOR_WEIGHT+sum(values))/(PRIOR_WEIGHT+len(values))
        observed_maximum = max(values)
        maximum = max(observed_maximum, average,
                      (float(prior["max_seconds"])*PRIOR_WEIGHT+observed_maximum*len(values)) /
                      (PRIOR_WEIGHT+len(values)))
        return average, maximum

    def _stage_remaining(self, stage, now, value):
        count = self.unit_counts[stage]
        if stage in self.stage_completed or count == 0:
            return 0.0
        if stage != self.current_stage:
            return count*value
        indices = self.unit_indices[stage]
        if indices is not None:
            if isinstance(self.current_unit, int):
                future = [index for index in indices if index > self.current_unit]
                started = now if self.current_unit_started is None else self.current_unit_started
                elapsed = max(0.0, now-started)
                return len(future)*value+max(0.0, value-elapsed)
            completed_index = self.last_completed_index[stage]
            if completed_index is None:
                return len(indices)*value
            future = [index for index in indices if index > completed_index]
            if not future:
                return 0.0
            started = now if self.current_unit_started is None else self.current_unit_started
            elapsed = max(0.0, now-started)
            return max(0, len(future)-1)*value+max(0.0, value-elapsed)
        completed = len(self.measurements[stage])
        started = now if self.current_unit_started is None else self.current_unit_started
        elapsed = max(0.0, now-started)
        full_units = max(0, count-completed-1)
        return full_units*value+max(0.0, value-elapsed)

    def remaining(self, timestamp):
        """Return ``(total_estimate, total_max, stage_estimate, stage_max)``."""
        if not self.counts_known:
            return None
        now = self._active_time(timestamp)
        stage_estimate = stage_maximum = 0.0
        total_estimate = total_maximum = 0.0
        for stage in STAGES:
            if not self.applies[stage]:
                continue
            average, maximum = self.per_unit(stage)
            estimate = self._stage_remaining(stage, now, average)
            upper = self._stage_remaining(stage, now, maximum)
            total_estimate += estimate
            total_maximum += upper
            if stage == self.current_stage:
                stage_estimate, stage_maximum = estimate, upper

        segmentation_count = self.unit_counts["segmentation"] if self.applies["segmentation"] else 0
        if self.current_stage in ("voting", "output"):
            fraction = self.phase_fraction.get(self.current_stage, 0.0)
            started = self.phase_started_at.get(self.current_stage)
            if started is not None and 0.0 < fraction < 1.0:
                phase_remaining = max(0.0, (now-started)*(1.0-fraction)/fraction)
                return phase_remaining, max(phase_remaining, 0.0), phase_remaining, max(phase_remaining, 0.0)
        if segmentation_count:
            predicted = segmentation_count*self.per_unit("segmentation")[0]
            tail = min(900.0, max(120.0, .12*predicted))
        else:
            tail = 60.0
        total_estimate += tail
        total_maximum += 2.0*tail
        if self.current_stage not in STAGES:
            stage_estimate, stage_maximum = tail, 2.0*tail
        return total_estimate, total_maximum, stage_estimate, stage_maximum


def save_profile(path, estimator):
    """Persist completed-run measurements, retaining prior values with little data."""
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {}
    for stage in STAGES:
        values = estimator.measurements[stage]
        prior = estimator.prior_profile[stage]
        if len(values) >= 3:
            average, maximum, units = sum(values)/len(values), max(values), len(values)
        else:
            average = float(prior["avg_seconds"])
            maximum = float(prior["max_seconds"])
            units = int(prior.get("units", 0))
        payload[stage] = {"avg_seconds": average, "max_seconds": maximum,
                          "units": units, "saved_at": saved_at}
    atomic_json(Path(path), payload)
