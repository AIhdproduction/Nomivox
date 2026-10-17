import numpy as np
import laspy
from pathlib import Path

from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan
from e57cls.output.e57 import write_e57
from e57cls.output.las import write_cloud
from test_inspector import source
from test_pipeline import small_project


def test_e57_roundtrip_preserves_rgb_pose_and_nomivox_classification(source, tmp_path):
    report = build_store(source, tmp_path/'cache')
    stores = [load_scan(Path(item['path']), report['source_sha256']) for item in report['scans']]
    class Votes:
        def lookup(self, keys):
            return np.full(len(keys), 152, np.uint8), np.full(len(keys), 200, np.uint8)
    result = write_e57(tmp_path/'classified.e57', source, stores, Votes(), np.zeros(3), .02, [1, 152])
    assert result['record_count'] == 10
    with E57Reader(tmp_path/'classified.e57') as reader:
        chunk = next(reader.chunks(0, 32))
        assert np.all(chunk.fields['nomivox:classification'] == 152)
        assert np.all(chunk.fields['colorBlue'] == 200)
        assert reader.root['data3D'][0]['pose']['translation']['x'].value() == 2700000.


def test_e57_roundtrip_copies_image_blobs_and_association(tmp_path):
    source_path = tmp_path/'with_image.e57'
    small_project(source_path)
    report = build_store(source_path, tmp_path/'cache')
    stores = [load_scan(Path(item['path']), report['source_sha256']) for item in report['scans']]
    class Votes:
        def lookup(self, keys):
            return np.full(len(keys), 1, np.uint8), np.zeros(len(keys), np.uint8)
    write_e57(tmp_path/'classified.e57', source_path, stores, Votes(), np.zeros(3), .02, [1])
    with E57Reader(source_path) as before, E57Reader(tmp_path/'classified.e57') as after:
        old = before.root['images2D'][0]
        new = after.root['images2D'][0]
        assert new['associatedData3DGuid'].value() == old['associatedData3DGuid'].value()
        assert new['sphericalRepresentation']['pngImage'].byteCount() == old['sphericalRepresentation']['pngImage'].byteCount()


def test_e57_and_las_share_transfer_and_mirror_results(source, tmp_path):
    report = build_store(source, tmp_path/'cache')
    stores = [load_scan(Path(item['path']), report['source_sha256']) for item in report['scans']]
    class Votes:
        def lookup(self, keys):
            return np.resize(np.array([2, 1], np.uint8), len(keys)), np.resize(np.array([20, 0], np.uint8), len(keys))
    def mirror(store, start, stop, valid, world, codes):
        return codes, np.full(len(codes), 4, np.uint8)
    las = write_cloud(tmp_path/'cloud.las', stores, Votes(), np.zeros(3), .02, '', [1, 2], mirror=mirror, class_transfer_radius=.01)
    e57 = write_e57(tmp_path/'cloud.e57', source, stores, Votes(), np.zeros(3), .02, [1, 2], mirror=mirror, class_transfer_radius=.01)
    cloud = laspy.read(tmp_path/'cloud.las')
    with E57Reader(tmp_path/'cloud.e57') as reader:
        chunk = next(reader.chunks(0, 32))
    np.testing.assert_array_equal(cloud.classification, chunk.fields['nomivox:classification'])
    np.testing.assert_array_equal(cloud.class_conf, chunk.fields['nomivox:classConfidence'])
    np.testing.assert_array_equal(cloud.mirror_state, chunk.fields['nomivox:mirrorState'])
    assert las['class_transfer_count'] == e57['class_transfer_count']
