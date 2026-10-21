import json
from pathlib import Path
import sqlite3
import shutil

import laspy
import numpy as np
import pytest

from e57cls.io.reader import E57Reader
from e57cls.pipeline.config import resolve_config
from e57cls.geometry.cameras import Camera, Pose, scan_pose
from e57cls.io.store import ScanStore
from e57cls.pipeline import run as run_module
from e57cls.pipeline.run import _slam_scene, execute
from e57cls.seg.cache import resize_mask_nearest
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


def _slam_value_store(path, scan_index, xyz, times, translation):
    path.mkdir()
    np.save(path/'xyz.npy', xyz)
    np.save(path/'time.npy', times)
    pose = {'rotation': {'w': 1., 'x': 0., 'y': 0., 'z': 0.},
            'translation': dict(zip('xyz', translation))}
    return ScanStore(path, {'record_count': len(xyz), 'valid_count': len(xyz), 'scan_index': scan_index,
                            'arrays': {'xyz_local': {'path': 'xyz.npy'}, 'timeStamp': {'path': 'time.npy'}},
                            'metadata': {'pose': pose, 'points': {'fields': {}}}})


def _old_slam_point_values(stores, point_ids, fields=None):
    point_ids = np.asarray(point_ids, np.int64)
    world = np.empty((len(point_ids), 3), np.float64)
    times = np.full(len(point_ids), np.nan, np.float64)
    by_scan = {store.meta['scan_index']: store for store in stores}
    scans = point_ids >> 48
    rows = point_ids & ((1 << 48)-1)
    for scan in np.unique(scans):
        ids = np.flatnonzero(scans == scan)
        store = by_scan[int(scan)]
        ordered = ids[np.argsort(rows[ids], kind='stable')]
        sorted_rows = rows[ordered]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_rows) != 1)+1, len(ordered)]
        pose = scan_pose(store.meta['metadata'])
        for first, last in zip(starts[:-1], starts[1:]):
            target = ordered[first:last]
            low, high = int(rows[target[0]]), int(rows[target[-1]])+1
            world[target] = pose.to_world(store.read('xyz_local', low, high)[rows[target]-low])
            field = fields.get(int(scan)) if isinstance(fields, dict) else fields
            if field and field in store.meta['arrays']:
                values = np.asarray(store.read(field, low, high)[rows[target]-low], np.float64)
                times[target] = np.where(np.isfinite(values), values, np.nan)
    return world, times


def test_slam_point_values_matches_run_reference_for_scattered_rows_and_times(tmp_path):
    rng = np.random.default_rng(907)
    stores = []
    for scan, translation in ((1, (10., -2., .5)), (2, (-3., 4., 1.))):
        xyz = rng.normal(size=(500, 3)).astype(np.float64)
        times = rng.normal(size=500).astype(np.float64)
        times[::31] = np.nan
        stores.append(_slam_value_store(tmp_path/f'scan_{scan}', scan, xyz, times, translation))
    point_ids = np.concatenate([(np.int64(scan) << 48) | rng.integers(0, 500, 800, dtype=np.int64)
                                for scan in (1, 2)])
    rng.shuffle(point_ids)
    fields = {1: 'timeStamp', 2: 'timeStamp'}
    expected = _old_slam_point_values(stores, point_ids, fields)
    actual = run_module._slam_point_values(stores, point_ids, fields)
    np.testing.assert_array_equal(actual[0].view(np.uint8), expected[0].view(np.uint8))
    np.testing.assert_equal(actual[1], expected[1])


@pytest.mark.parametrize("pinhole", [True, False])
@pytest.mark.parametrize("image_scale", [1., .9])
def test_slam_pipeline_transfers_visible_labels_and_writes_station_id(tmp_path, pinhole, image_scale):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=6, labelled_images=True, pinhole=pinhole)
    report, cloud = _classify(tmp_path, source, image_scale=image_scale, view_size=127)
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


def test_slam_pipeline_handles_indoor_features_beyond_terrain_gap(tmp_path):
    source = write_slam_e57(tmp_path / "terrain-gap.e57", camera_count=1, labelled_images=True, pinhole=True)
    report, cloud = _classify(tmp_path, source, terrain_max_gap=.1)

    assert report["output"]["record_count"] == len(cloud.x)
    assert "station_id" in cloud.point_format.extra_dimension_names


def test_slam_scene_context_uses_virtual_station_positions(tmp_path, monkeypatch):
    source = write_slam_e57(tmp_path/'stations.e57', camera_count=3, labelled_images=True, pinhole=True)
    captured = []
    original = run_module.geometry_scan_context
    def record(preset, config, scenes, positions=None, class_ids=None):
        captured.append((dict(scenes), dict(positions or {})))
        return original(preset, config, scenes, positions, class_ids)
    monkeypatch.setattr(run_module, 'geometry_scan_context', record)
    report, _ = _classify(tmp_path, source, name='station-context', slam_station_spacing=.1)
    assert captured
    scenes, positions = captured[-1]
    assert set(scenes) == set(positions)
    assert all(len(position) == 3 for position in positions.values())
    assert report['scene_consistency'] is not None


def test_slam_scene_counts_sky_in_score_coordinates(monkeypatch):
    rendered = np.zeros((5, 7, 3), np.uint8)
    valid = np.zeros((5, 7), bool)
    valid[:3, :4] = True
    scores = np.zeros((4, 6), np.float16)
    scores[:2] = 1.

    class CameraRecord:
        camera = Camera("pinholeRepresentation", {"imageWidth": 7, "imageHeight": 5,
                 "pixelWidth": 1., "pixelHeight": 1., "focalLength": 4.,
                 "principalPointX": 3.5, "principalPointY": 2.5})
        pose = Pose(np.eye(3), np.zeros(3))
        image_index = 1

        def views(self, *args):
            return [self.camera]

    camera = CameraRecord()

    monkeypatch.setattr(run_module, "_slam_view_image",
                        lambda *args: (rendered, valid, "image", None, {"id": "native"}))
    monkeypatch.setattr(run_module, "segment_view", lambda *args: ([scores], {"hits": 0, "misses": 1}))

    class VisibilityCache:
        def visibility_for_view(self, *args, **kwargs):
            return type("Visibility", (), {"point_indices": np.empty(0, np.int64)})(), False

    config = {"scene_mode": "auto", "scene_overrides": {}, "slam_max_range": 20.,
              "slam_max_range_indoor": 20., "slam_max_range_outdoor": 40., "profile": "fast",
              "view_size": 127, "view_fov": 95., "image_scale": .9, "tile_size": 336,
              "tile_overlap": 0., "scene_sky_confidence": .5, "scene_sky_outdoor": .2}
    preset = {"auxiliary": [{"id": "sky", "prompts": ["sky"]}]}
    control = type("Control", (), {"checkpoint": lambda self: None})()

    scene, info = _slam_scene(config, preset, [camera], {1: {"index": 1}}, None, None, [], VisibilityCache(),
                              None, None, "model", lambda: None, control)

    score_valid = resize_mask_nearest(valid, scores.shape)
    expected = ((scores >= .5) & score_valid).sum() / score_valid.sum()
    assert scene == "outdoor"
    assert info["sky_fraction"] == expected


def test_slam_pipeline_without_scan_pose_uses_world_coordinates(tmp_path):
    source = write_slam_e57(tmp_path / "room-without-pose.e57", camera_count=6, labelled_images=True,
                            pinhole=True, without_pose=True)
    report, cloud = _classify(tmp_path, source)
    identity_source = write_slam_e57(tmp_path / "room-with-identity-pose.e57", camera_count=6,
                                     labelled_images=True, pinhole=True)
    _, identity_cloud = _classify(tmp_path, identity_source, name="identity")
    inspection = json.loads((tmp_path / "run" / "inspect_report.json").read_text(encoding="utf-8"))
    assert inspection["input_mode"]["mode"] == "slam"
    assert next(item for item in inspection["input_mode"]["rules"] if item["id"] == "e")["hit"]
    expected = _expected_labels(np.column_stack((cloud.x, cloud.y, cloud.z)))
    observed = (expected == 100) & (np.asarray(cloud.station_id) >= 1)
    assert observed.any()
    assert (np.asarray(cloud.classification)[observed] == 100).mean() >= .90
    assert np.array_equal(np.asarray(cloud.classification), np.asarray(identity_cloud.classification))
    assert report["output"]["record_count"] == len(cloud.x)


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


def _slam_output(path):
    return laspy.read(next(path.glob("*.las")))


def _assert_same_slam_output(expected, actual):
    np.testing.assert_array_equal(actual.classification, expected.classification)
    np.testing.assert_array_equal(actual.station_id, expected.station_id)


def _assert_same_slam_statistics(expected, actual):
    assert actual["views_used"] == expected["views_used"]
    assert actual["never_visible_share"] == expected["never_visible_share"]
    assert actual["slam"]["median_visible_points_per_view"] == expected["slam"]["median_visible_points_per_view"]


def test_slam_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=6, labelled_images=True, pinhole=True)
    full_config = _config(tmp_path, source, output=tmp_path / "full", cache=tmp_path / "full-cache")
    full_report = execute(full_config, lambda: GeometryMaskBackend(), listen=False)
    full_cloud = _slam_output(tmp_path / "full")

    resume_config = _config(tmp_path, source, output=tmp_path / "resume", cache=tmp_path / "resume-cache")
    original_add_scan = run_module.VoteStore.add_scan
    completed = 0

    def cancel_after_first_station(store, *args, **kwargs):
        nonlocal completed
        result = original_add_scan(store, *args, **kwargs)
        completed += 1
        if completed == 1:
            raise run_module.Cancelled()
        return result

    monkeypatch.setattr(run_module.VoteStore, "add_scan", cancel_after_first_station)
    assert execute(resume_config, lambda: GeometryMaskBackend(), listen=False) == {"cancelled": True}
    monkeypatch.setattr(run_module.VoteStore, "add_scan", original_add_scan)

    resumed_report = execute(resume_config, lambda: GeometryMaskBackend(), listen=False)
    resumed_cloud = _slam_output(tmp_path / "resume")
    _assert_same_slam_output(full_cloud, resumed_cloud)
    _assert_same_slam_statistics(full_report, resumed_report)
    assert resumed_report["class_1_share"] == full_report["class_1_share"]
    assert full_report["class_1_share"] < 1.0
    observed = np.asarray(resumed_cloud.station_id) >= 1
    assert observed.any()
    assert (np.asarray(resumed_cloud.classification)[observed] == 1).mean() <= .10


def test_slam_resume_reconstructs_missing_observed_without_sam(tmp_path):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=6, labelled_images=True, pinhole=True)
    config = _config(tmp_path, source, output=tmp_path / "first", cache=tmp_path / "cache")
    baseline_report = execute(config, lambda: GeometryMaskBackend(), listen=False)
    baseline_cloud = _slam_output(tmp_path / "first")
    identity = json.loads((tmp_path / "first" / "run_config.json").read_text(encoding="utf-8"))["identity"]
    shutil.rmtree(tmp_path / "cache" / "votes" / identity / "observed")

    resumed_config = _config(tmp_path, source, output=tmp_path / "reconstructed", cache=tmp_path / "cache")
    backend = GeometryMaskBackend()
    reconstructed_report = execute(resumed_config, lambda: backend, listen=False)
    _assert_same_slam_output(baseline_cloud, _slam_output(tmp_path / "reconstructed"))
    _assert_same_slam_statistics(baseline_report, reconstructed_report)
    assert backend.calls == 0


def test_slam_completed_cache_reuses_observed_and_masks_without_sam(tmp_path):
    source = write_slam_e57(tmp_path / "room.e57", camera_count=6, labelled_images=True, pinhole=True)
    config = _config(tmp_path, source, output=tmp_path / "first", cache=tmp_path / "cache")
    first_report = execute(config, lambda: GeometryMaskBackend(), listen=False)
    first_cloud = _slam_output(tmp_path / "first")

    repeated_config = _config(tmp_path, source, output=tmp_path / "repeat", cache=tmp_path / "cache")
    backend = GeometryMaskBackend()
    repeated_report = execute(repeated_config, lambda: backend, listen=False)
    _assert_same_slam_output(first_cloud, _slam_output(tmp_path / "repeat"))
    _assert_same_slam_statistics(first_report, repeated_report)
    assert backend.calls == 0
