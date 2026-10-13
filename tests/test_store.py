import numpy as np
import pytest

from e57cls.files import file_hash
from e57cls.io.reader import E57Reader
from e57cls.io.store import build_store, load_scan, write_scan
from test_inspector import source


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
