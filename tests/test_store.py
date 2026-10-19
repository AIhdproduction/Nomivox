import numpy as np
import pytest
import pye57
from pye57 import libe57 as lib
from pathlib import Path

from e57cls.files import file_hash
from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan, write_scan
from test_inspector import source


def _write_scaled_fixture(path, spherical=False):
    """Write a tiny E57 whose prototype exercises raw integer storage."""
    count = 7
    writer = pye57.E57(str(path), 'w')
    image = writer.image_file
    scan = lib.StructureNode(image)
    scan.set('guid', lib.StringNode(image, '{scaled-fixture}'))
    scan.set('name', lib.StringNode(image, 'scaled fixture'))
    pose = lib.StructureNode(image)
    scan.set('pose', pose)
    rotation = lib.StructureNode(image)
    translation = lib.StructureNode(image)
    pose.set('rotation', rotation)
    pose.set('translation', translation)
    for name, value in {'w': 1., 'x': 0., 'y': 0., 'z': 0.}.items():
        rotation.set(name, lib.FloatNode(image, value))
    for name in 'xyz':
        translation.set(name, lib.FloatNode(image, 0.))
    prototype = lib.StructureNode(image)
    arrays = {}
    if spherical:
        coordinates = {'sphericalRange': np.linspace(1., 2., count),
                       'sphericalAzimuth': np.linspace(-.5, .5, count),
                       'sphericalElevation': np.linspace(-.2, .2, count)}
        for name, values in coordinates.items():
            prototype.set(name, lib.FloatNode(image, 0., lib.E57_DOUBLE, -10., 10.))
            arrays[name] = values.astype(np.float64)
    else:
        raw = np.array([0, 1, 42, 65535, 100000, 423998, 17], np.int64)
        for offset, name in enumerate(('cartesianX', 'cartesianY', 'cartesianZ')):
            prototype.set(name, lib.ScaledIntegerNode(image, 0, 0, 424000, .00025, -67.))
            arrays[name] = (raw + offset).astype(np.uint32)
    prototype.set('intensity', lib.ScaledIntegerNode(image, 0, 0, 3000, .01, 0.))
    prototype.set('amplitude', lib.ScaledIntegerNode(image, 0, 0, 3000, .01, 0.))
    for name in ('colorRed', 'colorGreen', 'colorBlue'):
        prototype.set(name, lib.IntegerNode(image, 0, 0, 255))
    prototype.set('rowIndex', lib.IntegerNode(image, 0, 0, 1729))
    prototype.set('timeStamp', lib.FloatNode(image, 0., lib.E57_DOUBLE, -1., 10.))
    arrays.update(intensity=np.arange(count, dtype=np.uint16)*300,
                  amplitude=np.arange(count, dtype=np.uint16)*200,
                  colorRed=np.arange(count, dtype=np.uint8),
                  colorGreen=np.arange(count, dtype=np.uint8)+10,
                  colorBlue=np.arange(count, dtype=np.uint8)+20,
                  rowIndex=np.arange(count, dtype=np.uint16)*200,
                  timeStamp=np.linspace(.1, .7, count, dtype=np.float64))
    codecs = lib.VectorNode(image, True)
    points = lib.CompressedVectorNode(image, prototype, codecs)
    scan.set('points', points)
    writer.data3d.append(scan)
    buffers = lib.VectorSourceDestBuffer()
    for name, values in arrays.items():
        buffers.append(lib.SourceDestBuffer(image, name, values, count, False, False))
    point_writer = points.writer(buffers)
    point_writer.write(count)
    point_writer.close()
    writer.close()


def test_scaled_integer_store_roundtrip_hash_and_schema_rebuild(tmp_path):
    source = tmp_path / 'scaled.e57'
    _write_scaled_fixture(source)
    report = build_store(source, tmp_path / 'cache', 3)
    directory = Path(report['scans'][0]['path'])
    store = load_scan(directory, verify_hashes=True)
    with E57Reader(source) as reader:
        reference = next(reader.chunks(0, 20))
    expected_values = {name: values for name, values in reference.fields.items()
                       if name not in ('cartesianX', 'cartesianY', 'cartesianZ')}
    expected_values['xyz_local'] = reference.xyz
    for name, expected in expected_values.items():
        actual = store.read(name, 0, len(reference.valid))
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    assert store.read_raw('xyz_local', 0, 7).dtype == np.uint32
    assert store.read_raw('intensity', 0, 7).dtype == np.uint16
    assert store.read_raw('colorRed', 0, 7).dtype == np.uint8
    assert store.read_raw('rowIndex', 0, 7).dtype == np.uint16
    assert store.read_raw('timeStamp', 0, 7).dtype == np.float64
    for info in store.meta['arrays'].values():
        assert file_hash(store.directory / info['path']) == info['sha256']
    meta = store.directory / 'meta.json'
    data = __import__('json').loads(meta.read_text(encoding='utf-8'))
    data['schema_version'] = 1
    meta.write_text(__import__('json').dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Unsupported'):
        load_scan(store.directory)
    rebuilt = build_store(source, tmp_path / 'cache', 3)
    assert load_scan(Path(rebuilt['scans'][0]['path'])).meta['schema_version'] == 2


def test_spherical_source_keeps_float64_coordinates(tmp_path):
    source = tmp_path / 'spherical.e57'
    _write_scaled_fixture(source, spherical=True)
    store = load_scan(Path(build_store(source, tmp_path / 'cache', 3)['scans'][0]['path']))
    assert store.read_raw('xyz_local', 0, 7).dtype == np.float64
    assert 'encoding' not in store.meta['arrays']['xyz_local']


@pytest.mark.parametrize('spherical', [False, True])
def test_read_rows_matches_individual_read_slices_for_all_store_encodings(tmp_path, spherical):
    source = tmp_path / 'rows.e57'
    _write_scaled_fixture(source, spherical=spherical)
    store = load_scan(Path(build_store(source, tmp_path / 'cache', 3)['scans'][0]['path']))
    rows = np.array([6, 2, 3, 0, 6, 1, 4], np.int64)
    for key in store.meta['arrays']:
        expected = np.concatenate([store.read(key, int(row), int(row)+1) for row in rows])
        actual = store.read_rows(key, rows)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(store.read_rows(key, np.array([], np.int64)), store.read(key, 0, 0))


def test_full_accounting_preservation_and_resume(source, tmp_path):
    report = build_store(source, tmp_path / "cache", 4)
    assert (report["record_count"], report["valid_count"], report["invalid_count"]) == (11, 10, 1)
    from pathlib import Path
    directory = Path(report["scans"][0]["path"])
    store = load_scan(directory)
    assert store.read("xyz_local", 0, 11).dtype == np.float64
    np.testing.assert_array_equal(store.read("colorRed", 0, 11), np.arange(11))
    with E57Reader(source) as reader:
        reference = next(reader.chunks(0, 11))
    np.testing.assert_array_equal(store.read("xyz_local", 0, 11), reference.xyz)
    meta_before = file_hash(directory / "meta.json")
    resumed = build_store(source, tmp_path / "cache", 3)
    assert resumed["valid_count"] == 10
    assert file_hash(directory / "meta.json") == meta_before


def test_interrupted_scan_never_publishes_completion_and_can_retry(source, tmp_path):
    def interrupt(*args):
        raise InterruptedError("Simulated worker interruption")
    with E57Reader(source) as reader:
        with pytest.raises(InterruptedError):
            write_scan(reader, 0, tmp_path / "cache", file_hash(source), 3, interrupt)
    assert not (tmp_path / "cache/scans/scan_0001/meta.json").exists()
    with E57Reader(source) as reader:
        store = write_scan(reader, 0, tmp_path / "cache", file_hash(source), 3)
    assert store.meta["valid_count"] == 10
    info = store.meta["arrays"]["xyz_local"]
    with (store.directory / info["path"]).open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError, match="Corrupt"):
        load_scan(store.directory)
