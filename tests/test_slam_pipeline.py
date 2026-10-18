import json
from pathlib import Path
import sqlite3

import laspy
import numpy as np
import pytest

from e57cls.io.reader import E57Reader
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.run import execute
from slam_fixtures import _COLORS, room_labels, write_slam_e57


class GeometryMaskBackend:
    """Return deterministic class masks from the labelled fixture panoramas."""

    def __init__(self, all_wall=False):
        self.all_wall = all_wall
        self.calls = 0
        self.reductions = []

    def predict_tile(self, image, classes):
        self.calls += 1
        pixels = np.asarray(image, np.int16)
        result = np.zeros((len(classes), image.height, image.width), np.float32)
        if self.all_wall:
            result[[item.id for item in classes].index("wall")] = .95
            return result
        names = ("wall", "floor", "ceiling", "box")
        colors = np.asarray([_COLORS[name] for name in names], np.int16)
        nearest = np.square(pixels[..., None, :]-colors).sum(axis=3).argmin(axis=2)
        for index, item in enumerate(classes):
            name = "box" if item.id == "construction_material" else item.id
            if name in names:
                result[index, nearest == names.index(name)] = .95
        return result


def _preset(path):
    source = json.loads(Path("presets/default.json").read_text(encoding="utf-8"))
    wanted = {"wall", "floor", "ceiling", "construction_material"}
    source["classes"] = [item for item in source["classes"] if item["id"] in wanted]
    source["scene_resolved_pairs"] = []
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _config(tmp_path, source, *, output, cache, **overrides):
    model = tmp_path / "model.pt"
    model.write_bytes(b"synthetic model")
    return resolve_config({"source": str(source), "output": str(output), "cache": str(cache),
                           "model": str(model), "preset": str(_preset(tmp_path / "preset.json")),
                           "input_mode": "slam", "profile": "fast", "scene_mode": "indoor",
                           "view_size": 128, "view_fov": 95, "image_scale": 1., "tile_size": 336,
                           "tile_overlap": 0., "voxel_size": .02, "point_reduction": 0,
                           "format": "las", "cache_cleanup": "keep_all", "gpu_device": "cpu",
                           "slam_depth_resolution": 128, "slam_station_spacing": 3.,
                           "boundary_margin": 0, "component_cleanup": False, "surface_tie_break": False,
                           "building_enclosure": False, "stairs_geometry_check": False,
                           "vegetation_base_to_ground": False, "mirror_analysis": False, **overrides})


def _classify(tmp_path, source, *, name="run", **overrides):
    config = _config(tmp_path, source, output=tmp_path / name, cache=tmp_path / f"{name}-cache", **overrides)
    report = execute(config, lambda: GeometryMaskBackend(), listen=False)
    return report, laspy.read(next((tmp_path / name).glob("*.las")))


def _expected_labels(points):
    codes = {"wall": 100, "ceiling": 101, "floor": 102, "box": 76}
    return np.asarray([codes[label] for label in room_labels(points)], np.uint8)


def test_slam_pipeline_transfers_visible_labels_and_writes_station_id(tmp_path):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=6, labelled_images=True, pinhole=True)
    report, cloud = _classify(tmp_path, source)
    station = np.asarray(cloud.station_id)
    actual = np.asarray(cloud.classification)
    expected = _expected_labels(np.column_stack((cloud.x, cloud.y, cloud.z)))

    assert "station_id" in cloud.point_format.extra_dimension_names
    assert station.max() >= 1
    assert not np.any(actual == 68)
    for code in (100, 101, 102, 76):
        observed = (expected == code) & (station >= 1)
        assert observed.any()
        assert (actual[observed] == code).mean() >= .90
    assert any(item["code"] == "slam_reflection_disabled" for item in report["warnings"])


def test_slam_occlusion_prevents_hidden_wall_votes(tmp_path):
    source = write_slam_e57(tmp_path / "occlusion.e57", camera_count=1, labelled_images=True, pinhole=True)
    config = _config(tmp_path, source, output=tmp_path / "out", cache=tmp_path / "cache",
                     slam_station_spacing=10.)
    cloud = laspy.read(next(Path(execute(config, lambda: GeometryMaskBackend(all_wall=True), listen=False)["output"]["path"]).parent.glob("*.las")))
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    station = np.asarray(cloud.station_id)
    behind_box = (np.isclose(xyz[:, 0], 6.)) & (np.abs(xyz[:, 1]) <= .75) & (xyz[:, 2] >= -.4) & (xyz[:, 2] <= 1.4)
    visible_wall = (np.isclose(xyz[:, 0], 6.) & (xyz[:, 1] > 1.)
                    & (xyz[:, 2] >= -.4) & (xyz[:, 2] <= 1.4))
    assert behind_box.any() and visible_wall.any()
    assert not station[behind_box].any()
    assert station[visible_wall].max() >= 1


def test_slam_time_filter_requires_both_timestamps_and_filters_points(tmp_path):
    source = write_slam_e57(tmp_path / "without-time.e57", camera_count=3, labelled_images=True, pinhole=True)
    baseline, base_cloud = _classify(tmp_path, source, name="baseline")
    unavailable, unavailable_cloud = _classify(tmp_path, source, name="unavailable", slam_time_filter=True)
    assert any(item["code"] == "slam_time_filter_unavailable" for item in unavailable["warnings"])
    np.testing.assert_array_equal(unavailable_cloud.classification, base_cloud.classification)
    np.testing.assert_array_equal(unavailable_cloud.station_id, base_cloud.station_id)

    timed = write_slam_e57(tmp_path / "with-time.e57", camera_count=3, timestamps=True, labelled_images=True, pinhole=True)
    _, cloud = _classify(tmp_path, timed, name="timed", slam_time_filter=True, slam_time_window=1.)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    ceiling = np.isclose(xyz[:, 2], 4.)
    assert ceiling.any()
    assert np.all(np.asarray(cloud.station_id)[ceiling] == 0)
    assert np.all(np.asarray(cloud.classification)[ceiling] == 1)


def test_slam_e57_station_id_matches_las_and_static_output_has_no_station_field(tmp_path):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=3, labelled_images=True, pinhole=True)
    _, las = _classify(tmp_path, source, name="las")
    config = _config(tmp_path, source, output=tmp_path / "e57", cache=tmp_path / "e57-cache", format="e57")
    report = execute(config, lambda: GeometryMaskBackend(), listen=False)
    with E57Reader(Path(report["output"]["path"])) as reader:
        station = np.concatenate([chunk.fields["nomivox:stationId"] for chunk in reader.chunks(0)])
    np.testing.assert_array_equal(station, np.asarray(las.station_id))

    static_source = write_slam_e57(tmp_path / "static.e57", associated=True, camera_count=1,
                                   labelled_images=True, pinhole=True)
    static = _config(tmp_path, static_source, output=tmp_path / "static", cache=tmp_path / "static-cache",
                     input_mode="static", format="las")
    execute(static, lambda: GeometryMaskBackend(), listen=False)
    static_las = laspy.read(next((tmp_path / "static").glob("*.las")))
    assert "station_id" not in static_las.point_format.extra_dimension_names
    identity = json.loads((tmp_path / "static" / "run_config.json").read_text(encoding="utf-8"))["identity"]
    with sqlite3.connect(tmp_path / "static-cache" / "votes" / identity / "votes.sqlite") as database:
        names = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {"station_peaks", "station_decisions"} & names

    second = _config(tmp_path, static_source, output=tmp_path / "static-second", cache=tmp_path / "static-second-cache",
                     input_mode="static", format="las")
    execute(second, lambda: GeometryMaskBackend(), listen=False)
    assert next((tmp_path / "static-second").glob("*.las")).read_bytes() == next((tmp_path / "static").glob("*.las")).read_bytes()
