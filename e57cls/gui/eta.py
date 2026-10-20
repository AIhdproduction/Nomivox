"""Runtime estimates for the GUI.

The estimator deliberately has no Qt dependency so its event accounting can be
tested independently from the window.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from e57cls.files import atomic_json


STAGES = ("sam", "mirror_masks", "mirror_evidence", "segmentation", "slam_scene", "slam_segmentation")
# Progress events define the persisted timing unit for every stage.  Changing a
# stage from scan/image timing to view timing must invalidate older profiles.
PROFILE_UNITS = {
    "sam": "view",
    "mirror_masks": "view",
    "mirror_evidence": "scan",
    "segmentation": "view",
    "slam_scene": "image",
    "slam_segmentation": "view",
}
# These stages changed from scan/image timing to per-view timing.  Legacy
# profiles without a unit are incompatible only for these changed stages.
VIEW_PROFILE_STAGES = frozenset(("sam", "mirror_masks", "segmentation", "slam_segmentation"))
DEFAULT_PROFILE = {
    # Per-view values draw on HSLU static, Test_PW_2 accelerated and Test_PW
    # SLAM runs measured on 2026-10-17 through 2026-10-19.
    "sam": {"avg_seconds": 18.0, "max_seconds": 45.0, "unit": "view"},
    "mirror_masks": {"avg_seconds": 5.0, "max_seconds": 15.0, "unit": "view"},
    "mirror_evidence": {"avg_seconds": 10.0, "max_seconds": 80.0, "unit": "scan"},
    "segmentation": {"avg_seconds": 15.0, "max_seconds": 45.0, "unit": "view"},
    "slam_scene": {"avg_seconds": 2.0, "max_seconds": 10.0, "unit": "image"},
    "slam_segmentation": {"avg_seconds": 30.0, "max_seconds": 60.0, "unit": "view"},
}
PRIOR_WEIGHT = 3
SLAM_FIXED_PHASE_SECONDS = {"slam_index": 20.0, "voting": 90.0, "output": 90.0}


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
        unit = value.get("unit")
        average = value.get("avg_seconds")
        maximum = value.get("max_seconds")
        minimum_units = 1 if stage in ("slam_scene", "slam_segmentation") else 3
        if (isinstance(units, int) and units >= minimum_units and _number(average) and _number(maximum)
                and maximum >= average
                # Profiles before unit versioning remain valid for stages
                # whose event unit never changed.  Explicit mismatches fail.
                and (unit == PROFILE_UNITS[stage]
                     or ("unit" not in value and stage not in VIEW_PROFILE_STAGES))
                # Reject stale per-view timings and extreme outliers.  This
                # prevents a former per-scan value from dominating a new run.
                and (unit != "view" or (average <= 10*DEFAULT_PROFILE[stage]["avg_seconds"]
                                         and maximum <= 10*DEFAULT_PROFILE[stage]["max_seconds"]))):
            profile[stage] = {"avg_seconds": float(average), "max_seconds": float(maximum),
                              "units": units, "unit": PROFILE_UNITS[stage]}
    return profile


class RunEstimator:
    """Estimate remaining work from completed units.

    Paused intervals are removed when matching ``state`` events are supplied.
    Without those events, elapsed time is necessarily treated as active time.
    """
    def __init__(self, applies=None, unit_counts=None, profile=None, parallel=False):
        applies = applies or {}
        unit_counts = unit_counts or {}
        self.applies = {stage: bool(applies.get(stage, False)) for stage in STAGES}
        self.parallel = bool(parallel)
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
        self.current_phase = None
        self.paused_since = None
        self.paused_seconds = 0.0
        self.slam_plan_received = False
        self.slam_mode_detected = False
        self.slam_seen_units = set()

    @property
    def is_slam(self):
        return ((self.slam_mode_detected or self.slam_plan_received)
                and (self.applies["slam_scene"] or self.applies["slam_segmentation"]))

    def set_slam_mode(self, enabled=True, semantic=True):
        """Enable SLAM accounting after the inspected input mode becomes known."""
        if enabled:
            self.slam_mode_detected = True
            self.applies["slam_scene"] = True
            self.applies["slam_segmentation"] = bool(semantic)
            self.applies["segmentation"] = False

    def set_slam_plan(self, cameras=None, views_total=None, stations=None):
        """Apply the exact mobile-work plan supplied by the pipeline."""
        self.slam_plan_received = True
        if isinstance(cameras, int) and cameras >= 0:
            self.set_unit_count("slam_scene", cameras)
        if isinstance(views_total, int) and views_total >= 0:
            self.set_unit_count("slam_segmentation", views_total)

    def _profile_is_learned(self, stage):
        return int(self.prior_profile[stage].get("units", 0)) >= 1

    def _slam_measurement_count(self):
        return len(self.measurements["slam_scene"])+len(self.measurements["slam_segmentation"])

    def _slam_seen_count(self):
        return len(self.slam_seen_units)

    @property
    def counts_known(self):
        # ``sam`` is introduced only by the shared pipeline.  Older callers
        # can keep their persisted timing profile without supplying its count.
        return all(not self.applies[stage] or stage in ('sam', 'slam_scene', 'slam_segmentation') or isinstance(self.unit_counts[stage], int)
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
        self.current_phase = phase
        if phase == "sam":
            self._start_stage("sam", now)
        elif phase == "mirror":
            self._start_stage("mirror_masks", now)
        elif phase == "segmentation":
            self._start_stage("segmentation", now)
        elif phase == "slam_scene":
            self._start_stage("slam_scene", now)
        elif phase == "slam_segmentation":
            self._start_stage("slam_segmentation", now)
        elif phase in ("voting", "output"):
            if self.current_stage in STAGES:
                self._finish_stage(self.current_stage, now)
            self.current_stage = phase

    def phase_finished(self, phase, timestamp):
        now = self._active_time(timestamp)
        if phase == "sam":
            self._finish_stage("sam", now)
        elif phase == "mirror":
            self._finish_stage("mirror_masks", now)
            self._finish_stage("mirror_evidence", now)
        elif phase == "segmentation":
            self._finish_stage("segmentation", now)
        elif phase == "slam_scene":
            self._finish_stage("slam_scene", now)
        elif phase == "slam_segmentation":
            self._finish_stage("slam_segmentation", now)

    def progress(self, phase, timestamp, **event):
        now = self._active_time(timestamp)
        if phase == 'output_prepare':
            phase = 'output'
            event = {**event, 'fraction': .1*float(event.get('fraction', 0.0))}
        if phase in ("voting", "output"):
            self.phase_fraction[phase] = max(0.0, min(1.0, float(event.get("fraction", 0.0))))
            return
        if phase not in STAGES:
            return
        view_unit = (isinstance(event.get("image"), int) and isinstance(event.get("view"), int)
                     and event["view"] >= 1)
        if phase == "sam" and not view_unit and isinstance(event.get("scan_count"), int):
            self.set_unit_count(phase, event["scan_count"])
            if self.unit_indices[phase] is None:
                self.set_unit_indices(phase, range(1, event["scan_count"]+1))
        if phase == "mirror_masks" and not view_unit and isinstance(event.get("image_count"), int):
            self.set_unit_count(phase, event["image_count"])
        if phase == "slam_scene" and isinstance(event.get("image_count"), int):
            self.set_unit_count(phase, event["image_count"])
        if phase == "mirror_evidence" and isinstance(event.get("scan_count"), int):
            self.set_unit_count(phase, event["scan_count"])
            if self.unit_indices[phase] is None:
                self.set_unit_indices(phase, range(1, event["scan_count"]+1))
        self._start_stage(phase, now)
        unit = self._event_unit(phase, event, view_unit)
        if phase in ("slam_scene", "slam_segmentation") and unit is not None:
            self.slam_seen_units.add((phase, unit))
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

    @staticmethod
    def _event_unit(phase, event, view_unit):
        if view_unit:
            # A camera can contribute the same view to multiple SLAM stations.
            # The station therefore belongs to the identity of a SLAM view.
            if phase == "slam_segmentation" and isinstance(event.get("station"), int):
                return event["station"], event["image"], event["view"]
            return event["image"], event["view"]
        if phase in ("mirror_masks", "slam_scene"):
            return event.get("image")
        return event.get("scan")

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
        if (stage in ("slam_scene", "slam_segmentation") and not values
                and not self._profile_is_learned(stage) and self._slam_measurement_count() >= 2):
            measured = self.measurements["slam_scene"]+self.measurements["slam_segmentation"]
            average = sum(measured)/len(measured)
            return average, max(measured)
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
        # Built-in SLAM values are progress weights only.  Do not display an
        # invented ETA until this run has measured a cadence or a prior run did.
        if (self.is_slam and self._slam_seen_count() < 2
                and not any(self._profile_is_learned(stage) for stage in ("slam_scene", "slam_segmentation"))):
            return None
        now = self._active_time(timestamp)
        stage_estimate = stage_maximum = 0.0
        total_estimate = total_maximum = 0.0
        for stage in STAGES:
            if not self.applies[stage] or self.unit_counts[stage] is None:
                continue
            average, maximum = self.per_unit(stage)
            estimate = self._stage_remaining(stage, now, average)
            upper = self._stage_remaining(stage, now, maximum)
            total_estimate += estimate
            total_maximum += upper
            if stage == self.current_stage:
                stage_estimate, stage_maximum = estimate, upper

        segmentation_count = (self.unit_counts["segmentation"] if self.applies["segmentation"]
                              else self.unit_counts["slam_segmentation"] if self.applies["slam_segmentation"] else 0)
        if self.current_stage in ("voting", "output"):
            fraction = self.phase_fraction.get(self.current_stage, 0.0)
            started = self.phase_started_at.get(self.current_stage)
            if started is not None and 0.0 < fraction < 1.0:
                phase_remaining = max(0.0, (now-started)*(1.0-fraction)/fraction)
                output_remaining = SLAM_FIXED_PHASE_SECONDS["output"] if self.is_slam and self.current_stage == "voting" else 0.0
                return (phase_remaining+output_remaining, max(phase_remaining, 0.0)+output_remaining,
                        phase_remaining, max(phase_remaining, 0.0))
        if self.is_slam:
            # Indexing, voting and writing have no per-view unit.  Their
            # bounded estimates still belong to the total SLAM remainder.
            tail = (SLAM_FIXED_PHASE_SECONDS["voting"]+SLAM_FIXED_PHASE_SECONDS["output"]
                    if self.current_phase in ("slam_scene", "slam_segmentation")
                    else sum(SLAM_FIXED_PHASE_SECONDS.values()))
            total_estimate += tail
            total_maximum += 2.0*tail
            if self.current_stage not in STAGES:
                stage_estimate, stage_maximum = tail, 2.0*tail
            return total_estimate, total_maximum, stage_estimate, stage_maximum
        if segmentation_count:
            stage = "segmentation" if self.applies["segmentation"] else "slam_segmentation"
            predicted = segmentation_count*self.per_unit(stage)[0]
            tail = min(900.0, max(120.0, .12*predicted))
        else:
            tail = 60.0
        total_estimate += tail
        total_maximum += 2.0*tail
        if self.current_stage not in STAGES:
            stage_estimate, stage_maximum = tail, 2.0*tail
        return total_estimate, total_maximum, stage_estimate, stage_maximum

    def slam_overall_fraction(self, phase, fraction):
        """Return duration-weighted SLAM progress for the global progress bar."""
        if not self.is_slam or not self.slam_plan_received:
            return None
        scene_count = self.unit_counts["slam_scene"]
        views_count = self.unit_counts["slam_segmentation"]
        if not isinstance(scene_count, int) or not isinstance(views_count, int):
            return None
        durations = {
            "slam_index": SLAM_FIXED_PHASE_SECONDS["slam_index"],
            "slam_scene": scene_count*self.per_unit("slam_scene")[0],
            "slam_segmentation": views_count*self.per_unit("slam_segmentation")[0],
            "voting": SLAM_FIXED_PHASE_SECONDS["voting"],
            "output": SLAM_FIXED_PHASE_SECONDS["output"],
        }
        order = ("slam_index", "slam_scene", "slam_segmentation", "voting", "output")
        if phase not in order:
            return None
        total = sum(durations.values())
        complete = sum(durations[item] for item in order[:order.index(phase)])
        return max(0.0, min(1.0, (complete+durations[phase]*max(0.0, min(1.0, fraction)))/max(1.0, total)))


def save_profile(path, estimator):
    """Persist completed-run measurements, retaining prior values with little data."""
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {}
    for stage in STAGES:
        values = estimator.measurements[stage]
        prior = estimator.prior_profile[stage]
        if len(values) >= (1 if stage in ("slam_scene", "slam_segmentation") else 3):
            average, maximum, units = sum(values)/len(values), max(values), len(values)
        else:
            average = float(prior["avg_seconds"])
            maximum = float(prior["max_seconds"])
            units = int(prior.get("units", 0))
        payload[stage] = {"avg_seconds": average, "max_seconds": maximum,
                          "units": units, "unit": PROFILE_UNITS[stage], "saved_at": saved_at}
    atomic_json(Path(path), payload)
