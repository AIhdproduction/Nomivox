import json

import laspy
import numpy as np

from e57cls.output.quality import analyze_run


def test_quality_reports_known_synthetic_las_shares(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    header = laspy.LasHeader(point_format=6, version='1.4')
    points = laspy.ScaleAwarePointRecord.zeros(4, header=header)
    points.x, points.y, points.z = [0, 1, 2, 3], [0]*4, [0, 0, 0, 0]
    points.classification = np.array([2, 1, 2, 1], np.uint8)
    points.point_source_id = np.array([1, 1, 1, 1], np.uint16)
    with laspy.open(run / 'result.las', mode='w', header=header) as writer:
        writer.write_points(points)
    (run / 'report.json').write_text(json.dumps({'scans': [{'scan': 1, 'scene': 'outdoor'}]}), encoding='utf-8')
    (run / 'inspect_report.json').write_text(json.dumps({'scans': [{'index': 1, 'metadata': {'pose': {'translation': {'x': 0, 'y': 0, 'z': 1.5}}}}]}), encoding='utf-8')
    (run / 'scan_index.json').write_text(json.dumps({'1': {'name': 'synthetic'}}), encoding='utf-8')
    (run / 'legend.json').write_text('[]', encoding='utf-8')

    result = analyze_run(run, step=1)

    band = result['distance_unclassified']['outdoor']['all_points']['0-5m']
    assert band == {'sampled_points': 4, 'unclassified_points': 2, 'unclassified_share': .5}
    assert (run / 'quality' / 'quality.json').is_file()
    assert (run / 'quality' / 'outdoor_overview.png').is_file()


def test_quality_counts_misclassified_outdoor_tree_but_not_indoor_ceiling(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    rng = np.random.default_rng(42)
    indoor_ground = np.array([(x, y, 0.) for x in range(-3, 4) for y in range(-3, 4)])
    outdoor_ground = np.array([(x, y, 0.) for x in range(7, 14) for y in range(-3, 4)])
    indoor_ceiling = rng.normal((0., 0., 4.), (.25, .25, .25), size=(160, 3))
    outdoor_tree = rng.normal((10., 0., 4.), (.25, .25, .25), size=(160, 3))
    xyz = np.vstack((indoor_ground, outdoor_ground, indoor_ceiling, outdoor_tree))
    codes = np.concatenate((np.full(len(indoor_ground), 2), np.full(len(outdoor_ground), 2),
                            np.full(len(indoor_ceiling), 101), np.full(len(outdoor_tree), 6))).astype(np.uint8)
    scans = np.concatenate((np.full(len(indoor_ground), 1), np.full(len(outdoor_ground), 2),
                            np.full(len(indoor_ceiling), 1), np.full(len(outdoor_tree), 2))).astype(np.uint16)
    header = laspy.LasHeader(point_format=6, version='1.4')
    points = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
    points.x, points.y, points.z = xyz.T
    points.classification, points.point_source_id = codes, scans
    with laspy.open(run / 'result.las', mode='w', header=header) as writer:
        writer.write_points(points)
    (run / 'report.json').write_text(json.dumps({'scans': [
        {'scan': 1, 'scene': 'mixed'}, {'scan': 2, 'scene': 'outdoor'}]}), encoding='utf-8')
    (run / 'inspect_report.json').write_text(json.dumps({'scans': [
        {'index': 1, 'metadata': {'pose': {'translation': {'x': 0, 'y': 0, 'z': 1.5}}}},
        {'index': 2, 'metadata': {'pose': {'translation': {'x': 10, 'y': 0, 'z': 1.5}}}}]}), encoding='utf-8')
    (run / 'scan_index.json').write_text(json.dumps({'1': {}, '2': {}}), encoding='utf-8')
    (run / 'legend.json').write_text('[]', encoding='utf-8')

    result = analyze_run(run, step=1, indoor_scans=[1], outdoor_scans=[2])
    metric = result['vegetation_like_outside_buildings']
    groups = metric['by_scan_group']

    assert metric['class_counts'][6] > 0
    assert 101 not in metric['class_counts']
    assert groups['outdoor_scans']['class_counts'][6] == metric['class_counts'][6]
    assert groups['indoor_scans']['sampled_points'] == 0
    assert metric['footprint_indoor_scan_ids'] == [1]
