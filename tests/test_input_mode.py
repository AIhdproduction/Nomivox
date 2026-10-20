import json
import math

import numpy as np
import pytest

from e57cls.io.input_mode import detect_input_mode, pseudo_setup_points_check
from e57cls.io.inspector import inspect_file
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.run import execute
from slam_fixtures import write_pseudo_setup_e57, write_slam_e57


class _MemoryStore:
    def __init__(self, points, index=1, pose=None):
        self.points = points
        self.meta = {"scan_index": index, "valid_count": len(points),
                     "metadata": {} if pose is False else {"pose": _identity_pose() if pose is None else pose}}

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


def _window_scan():
    y, z = np.meshgrid(np.linspace(-2., 2., 120), np.linspace(-1., 1., 80))
    return np.column_stack((np.full(y.size, 10.), y.ravel(), z.ravel()))


def _static_scan(index, with_pose=True):
    return {"index": index, "record_count": 1,
            "metadata": {"pose": _identity_pose()} if with_pose else {}}


def _image(index, scan_index, associated=True):
    return {"index": index, "pose_status": "valid", "scan_index": scan_index if associated else None,
            "association_status": "explicit_guid" if associated else "uncertain",
            "metadata": {"pose": _identity_pose()}}


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


def test_window_scan_uses_angular_depth_map():
    check = pseudo_setup_points_check([_MemoryStore(_window_scan())], config={"gpu_device": "cpu"})
    assert check["value"][0]["visible_share"] >= .60


def test_one_dense_window_scan_does_not_switch_many_static_blocks_to_slam():
    stores = [_MemoryStore(_room_with_box(), index=index) for index in range(1, 11)]
    stores.append(_MemoryStore(_window_scan(), index=11))
    check = pseudo_setup_points_check(stores, config={"gpu_device": "cpu"})
    inspection = {"scans": [_static_scan(index) for index in range(1, 12)], "images": []}
    result = detect_input_mode(inspection, {"input_mode": "auto"}, points_check=lambda _: check)
    assert result["mode"] == "static"


def test_single_metadata_outliers_keep_static_mode_and_record_warnings():
    scans = [_static_scan(index, with_pose=index != 10) for index in range(1, 11)]
    images = [_image(index, index, associated=index != 9) for index in range(1, 11)]
    result = detect_input_mode({"scans": scans, "images": images}, {"input_mode": "auto"})
    assert result["mode"] == "static"
    assert result["warnings"]
    assert not next(rule for rule in result["rules"] if rule["id"] == "b")["hit"]
    assert not next(rule for rule in result["rules"] if rule["id"] == "e")["hit"]


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


def test_origin_depth_check_with_georeferenced_origin():
    pose = {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
            "translation": {"x": 2665953., "y": 1207269., "z": 440.}}
    check = pseudo_setup_points_check([_MemoryStore(_room_with_box(), pose=pose)], config={"gpu_device": "cpu"})
    assert check["value"][0]["visible_share"] >= .90
    check_window = pseudo_setup_points_check([_MemoryStore(_window_scan(), pose=pose)], config={"gpu_device": "cpu"})
    assert check_window["value"][0]["visible_share"] >= .60


def test_origin_depth_check_with_rotated_georeferenced_origin():
    pose = {"rotation": {"w": math.cos(math.radians(15)), "x": 0., "y": 0., "z": math.sin(math.radians(15))},
            "translation": {"x": 2665953., "y": 1207269., "z": 440.}}
    check = pseudo_setup_points_check([_MemoryStore(_room_with_box(), pose=pose)], config={"gpu_device": "cpu"})
    assert check["value"][0]["visible_share"] >= .90
    check_window = pseudo_setup_points_check([_MemoryStore(_window_scan(), pose=pose)], config={"gpu_device": "cpu"})
    assert check_window["value"][0]["visible_share"] >= .60


def test_origin_depth_check_skips_blocks_without_origin():
    stores = [_MemoryStore(_room_with_box(), index=index) for index in range(1, 4)]
    stores.append(_MemoryStore(_room_with_box(), index=4, pose=False))
    check = pseudo_setup_points_check(stores, config={"gpu_device": "cpu"})
    assert len(check["value"]) == 3
    assert [item["scan"] for item in check["value"]] == [1, 2, 3]
    assert check["without_origin"] == [4]
    assert check["hit"] is False


def test_skipped_scan_output_keeps_scan_pose():
    from e57cls.output.classify import classify_points

    store = type("S", (), {"meta": {"scan_index": 7, "metadata": {
        "pose": {"rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                 "translation": {"x": 100., "y": 200., "z": 10.}}}}})()
    xyz = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    valid = np.ones(3, bool)

    class _FailingVotes:
        def lookup(self, *args, **kwargs):
            raise AssertionError("lookup should not be called for skipped scans")

    world, codes, confidence, states = classify_points(
        store, 0, 3, xyz, valid, _FailingVotes(), np.zeros(3), .1, skipped={7}
    )
    assert np.allclose(world, xyz + [100., 200., 10.])
    assert (codes == 1).all()
    assert (confidence == 0).all()
