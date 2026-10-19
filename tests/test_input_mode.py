import json

import numpy as np
import pytest

from e57cls.io.input_mode import detect_input_mode, pseudo_setup_points_check
from e57cls.io.inspector import inspect_file
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.run import execute
from slam_fixtures import write_pseudo_setup_e57, write_slam_e57


class _MemoryStore:
    def __init__(self, points):
        self.points = points
        self.meta = {"scan_index": 1, "valid_count": len(points), "metadata": {"pose": _identity_pose()}}

    def full_chunks(self):
        yield 0, self.points, np.ones(len(self.points), bool)


def _identity_pose():
    return {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
            "translation": {"x": 0., "y": 0., "z": 0.}}


def _room_with_box():
    values = np.linspace(-6., 6., 55)
    a, b = np.meshgrid(values, values)
    walls = [np.column_stack((np.full(a.size, -6.), a.ravel(), b.ravel())),
             np.column_stack((np.full(a.size, 6.), a.ravel(), b.ravel())),
             np.column_stack((a.ravel(), np.full(a.size, -6.), b.ravel())),
             np.column_stack((a.ravel(), np.full(a.size, 6.), b.ravel())),
             np.column_stack((a.ravel(), b.ravel(), np.full(a.size, -2.))),
             np.column_stack((a.ravel(), b.ravel(), np.full(a.size, 4.)))]
    side, top = np.meshgrid(np.linspace(-.9, .9, 32), np.linspace(-1., 1., 32))
    box = np.column_stack((np.full(side.size, 1.5), side.ravel(), top.ravel()+.5))
    return np.vstack([*walls, box])


def test_slam_metadata_detection_and_manual_override(tmp_path):
    source = write_slam_e57(tmp_path / "slam.e57")
    report = inspect_file(source, config={"input_mode": "auto"})
    assert report["input_mode"]["mode"] == "slam"
    assert next(item for item in report["input_mode"]["rules"] if item["id"] == "a")["hit"]
    forced = detect_input_mode(report, {"input_mode": "static"})
    assert forced["mode"] == "static"


def test_unassociated_slam_input_dispatches_to_slam_pipeline(tmp_path, monkeypatch):
    source = write_slam_e57(tmp_path / "unassociated.e57")
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture")
    config = resolve_config({"source": str(source), "output": str(tmp_path / "out"), "cache": str(tmp_path / "cache"),
                             "model": str(model), "input_mode": "auto"})
    monkeypatch.setattr("e57cls.pipeline.run.execute_slam", lambda *args: {"slam": True})
    assert execute(config, listen=False) == {"slam": True}
    report = json.loads((tmp_path / "out" / "inspect_report.json").read_text(encoding="utf-8"))
    assert report["input_mode"]["mode"] == "slam"
    assert all(image["scan_index"] is None for image in report["images"])


def test_mixed_evidence_prefers_slam_and_records_warning():
    pose = lambda x: {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                      "translation": {"x": x, "y": 0., "z": 0.}}
    inspection = {"scans": [{"index": 1, "record_count": 10, "metadata": {"pose": pose(0)}}],
                  "images": [{"pose_status": "valid", "scan_index": 1, "association_status": "explicit_guid", "metadata": {"pose": pose(0)}},
                             {"pose_status": "valid", "scan_index": None, "association_status": "uncertain", "metadata": {"pose": pose(2)}}]}
    result = detect_input_mode(inspection, {"input_mode": "auto"})
    assert result["mode"] == "slam"
    assert result["mixed"]
    assert result["warnings"]


def test_static_room_with_box_stays_static_after_origin_depth_check():
    check = pseudo_setup_points_check([_MemoryStore(_room_with_box())], config={"gpu_device": "cpu"})
    assert check["value"][0]["visible_share"] >= .90
    inspection = {"scans": [{"index": 1, "record_count": 1, "metadata": {"pose": _identity_pose()}}], "images": []}
    result = detect_input_mode(inspection, {"input_mode": "auto"}, points_check=lambda _: check)
    assert result["mode"] == "static"


def test_missing_scan_pose_is_slam_evidence_and_skips_origin_depth_check():
    store = _MemoryStore(_room_with_box())
    store.meta["metadata"] = {}
    check = pseudo_setup_points_check([store], config={"gpu_device": "cpu"})
    assert check["value"] == "nicht anwendbar"
    inspection = {"scans": [{"index": 1, "record_count": 1, "metadata": {}}], "images": []}
    result = detect_input_mode(inspection, {"input_mode": "auto"}, points_check=lambda _: check)
    assert result["mode"] == "slam"
    assert next(item for item in result["rules"] if item["id"] == "e")["hit"]
    assert next(item for item in result["rules"] if item["id"] == "d")["value"] == "nicht anwendbar"


def test_missing_scan_pose_counts_as_identity_for_rule_c():
    pose = lambda x: {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                      "translation": {"x": x, "y": 0., "z": 0.}}
    inspection = {"scans": [{"index": 1, "record_count": 1, "metadata": {"cartesianBounds": {
        "cartesianXMinimum": -50., "cartesianXMaximum": 50., "cartesianYMinimum": -1.,
        "cartesianYMaximum": 1., "cartesianZMinimum": -1., "cartesianZMaximum": 1.}}}],
                  "images": [{"pose_status": "valid", "scan_index": 1, "association_status": "explicit_guid", "metadata": {"pose": pose(-2)}},
                             {"pose_status": "valid", "scan_index": 1, "association_status": "explicit_guid", "metadata": {"pose": pose(2)}}]}
    result = detect_input_mode(inspection, {"input_mode": "auto"})
    assert next(item for item in result["rules"] if item["id"] == "c")["hit"]


def test_forced_static_without_scan_pose_reports_actionable_message(tmp_path):
    source = write_slam_e57(tmp_path / "without-pose.e57", associated=True, camera_count=1, without_pose=True)
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture")
    config = resolve_config({"source": str(source), "output": str(tmp_path / "out"), "cache": str(tmp_path / "cache"),
                             "model": str(model), "input_mode": "static"})
    with pytest.raises(ValueError, match="Datenblock ohne Scannerposition"):
        execute(config, listen=False)


def test_pseudo_setup_dispatches_to_slam_pipeline_after_store_build(tmp_path, monkeypatch):
    source = write_pseudo_setup_e57(tmp_path / "pseudo.e57")
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture")
    config = resolve_config({"source": str(source), "output": str(tmp_path / "out"), "cache": str(tmp_path / "cache"),
                             "model": str(model), "input_mode": "auto"})
    monkeypatch.setattr("e57cls.pipeline.run.execute_slam", lambda *args: {"slam": True})
    assert execute(config, listen=False) == {"slam": True}
    inspection = json.loads((tmp_path / "out" / "inspect_report.json").read_text(encoding="utf-8"))
    assert inspection["input_mode"]["mode"] == "slam"
    assert next(item for item in inspection["input_mode"]["rules"] if item["id"] == "d")["hit"]
