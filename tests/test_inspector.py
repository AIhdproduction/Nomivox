import io
import json

import numpy as np
from PIL import Image
import pye57
import pytest

from e57cls.files import file_hash
from e57cls.io.inspector import image_header, inspect_file, pose_status, write_inspection
from e57cls.io.reader import E57Reader, coordinates


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "fixture.e57"
    data = {"cartesianX": np.arange(11, dtype=np.float64),
            "cartesianY": np.ones(11), "cartesianZ": np.full(11, 2.),
            "cartesianInvalidState": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8),
            "colorRed": np.arange(11, dtype=np.uint8),
            "colorGreen": np.zeros(11, np.uint8), "colorBlue": np.full(11, 200, np.uint8)}
    writer = pye57.E57(str(path), mode="w")
    writer.write_scan_raw(data, name="fixture", translation=np.array([2700000., 1200000., 400.]))
    writer.close()
    return path


def test_bounded_inspection_preserves_input_and_reports_absent_images(source, tmp_path):
    before = file_hash(source)
    report = write_inspection(source, tmp_path / "out", 4)
    assert file_hash(source) == before
    assert report["record_count"] == 11
    assert report["image_count"] == 0
    sample = report["scans"][0]["sample"]
    assert sample["records_read"] == 4
    assert sample["invalid_records"] == 1
    assert sample["whole_scan_invalid_count"] is None
    assert report["issues"][0]["code"] == "scan_not_classifiable"
    assert not report["scans"][0]["classification_eligible_metadata"]
    assert json.loads((tmp_path / "out/inspect_report.json").read_text())["record_count"] == 11
    with pytest.raises(FileExistsError):
        write_inspection(source, tmp_path / "out", 4)


def test_chunk_boundaries_and_owned_buffers(source):
    with E57Reader(source) as reader:
        chunks = list(reader.chunks(0, capacity=4))
    assert [c.start for c in chunks] == [0, 4, 8]
    assert [len(c.xyz) for c in chunks] == [4, 4, 3]
    np.testing.assert_array_equal(np.concatenate([c.xyz[:, 0] for c in chunks]), np.arange(11))


def test_nonfinite_and_partial_invalid_coordinates():
    _, valid = coordinates({"cartesianX": np.array([0, np.nan, 1, 2]),
                            "cartesianY": np.zeros(4), "cartesianZ": np.zeros(4),
                            "cartesianInvalidState": np.array([0, 0, 1, 2])})
    assert valid.tolist() == [True, False, False, False]
    xyz, valid = coordinates({"sphericalRange": np.array([2., -1.]),
                              "sphericalAzimuth": np.array([np.pi/2, 0]),
                              "sphericalElevation": np.zeros(2)})
    np.testing.assert_allclose(xyz[0], [0, 2, 0], atol=1e-15)
    assert valid.tolist() == [True, False]


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_image_headers_without_full_decode(format):
    stream = io.BytesIO()
    Image.new("RGB", (32, 16)).save(stream, format=format)
    content = stream.getvalue()
    class Blob:
        def byteCount(self):
            return len(content)
        def read(self, target, offset, count):
            target[:] = np.frombuffer(content[offset:offset+count], np.uint8)
    assert image_header(Blob())["width"] == 32
    assert image_header(Blob())["format"] == format
    with pytest.raises(ValueError):
        image_header(Blob(), limit=2)


def test_pose_validation():
    assert pose_status(None) == "missing"
    assert pose_status({"rotation": {}}) == "invalid"
    assert pose_status({"rotation": dict(w=-1, x=0, y=0, z=0),
                        "translation": dict(x=2700000, y=1200000, z=400)}) == "valid"


def test_metadata_only_inspection_skips_point_sampling(source):
    report = inspect_file(source, 0)
    assert report["sample_limit_per_scan"] == 0
    assert report["scans"][0]["sample"] == {"records_read": 0}
    assert report["input_mode"]["mode"] == "static"
