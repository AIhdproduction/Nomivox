import numpy as np
import laspy
from pathlib import Path

from test_inspector import source
from e57cls.io.store import build_store, load_scan
from e57cls.vote.engine import VoteStore, voxel_keys
from e57cls.output.las import write_cloud


def test_votes_require_independent_scans_and_resolve_ties(tmp_path):
    votes = VoteStore(tmp_path/'votes.sqlite')
    keys = voxel_keys([[0, 0, 0], [1, 0, 0]], np.zeros(3), .02)
    batch = [(keys, np.array([[.8, .8], [.2, .9]]))]
    votes.add_scan(1, 'same', batch, [100, 101])
    votes.add_scan(1, 'same', batch, [100, 101])
    votes.decide(2, .4, [100, 101])
    assert votes.lookup(keys)[0].tolist() == [1, 1]
    votes.add_scan(2, 'same', batch, [100, 101])
    votes.decide(2, .4, [100, 101])
    assert votes.lookup(keys)[0].tolist() == [100, 101]
    votes.decide(2, .6, [100, 101])
    assert votes.lookup(keys)[0].tolist() == [1, 101]
    votes.close()


def test_las_preserves_points_rgb_extended_codes_and_scan_ids(source, tmp_path):
    report = build_store(source, tmp_path/'cache')
    stores = [load_scan(Path(item['path']), report['source_sha256']) for item in report['scans']]
    class Votes:
        def lookup(self, keys):
            return np.resize(np.array([68, 152], np.uint8), len(keys)), np.full(len(keys), 200, np.uint8)
    result = write_cloud(tmp_path/'test.las', stores, Votes(), np.zeros(3), .02, '', [1, 68, 152], no_reflection=True)
    cloud = laspy.read(tmp_path/'test.las')
    assert len(cloud) == 10
    assert set(cloud.classification) == {68, 152}
    assert np.all(cloud.blue == 200*257)
    assert np.all(cloud.point_source_id == 1)
    np.testing.assert_allclose(cloud.x, 2700000+np.delete(np.arange(11), 2), atol=.0005)
    assert np.all(cloud.class_conf == 200)
    assert len(laspy.read(tmp_path/'test_no_reflection.las')) == 5
    assert result['readback']['all_classes_valid']
    per_scan = np.zeros(256, np.int64)
    for counts in result['scan_class_counts'].values():
        for code, count in counts.items():
            per_scan[int(code)] += count
    assert {str(code): int(count) for code, count in enumerate(per_scan) if count} == result['class_counts']
