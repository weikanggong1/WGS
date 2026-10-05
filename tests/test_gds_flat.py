"""Optional SDK adapter boundaries; these are units, not performance data.

SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from staar_phewas import gds_flat


def test_loader_only_falls_back_for_absent_adapter(monkeypatch):
    def absent(name):
        raise ModuleNotFoundError("adapter absent", name=name)
    monkeypatch.setattr(gds_flat.importlib, "import_module", absent)
    assert gds_flat.load_flat_reader() is None

    def missing_dependency(name):
        raise ModuleNotFoundError("SDK absent", name="pygds")
    monkeypatch.setattr(gds_flat.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="SDK absent"):
        gds_flat.load_flat_reader()

    def broken_adapter(name):
        raise RuntimeError("SDK capsule incompatible")
    monkeypatch.setattr(gds_flat.importlib, "import_module", broken_adapter)
    with pytest.raises(RuntimeError, match="SDK capsule incompatible"):
        gds_flat.load_flat_reader()


def test_metadata_hashes_the_actual_binary(tmp_path):
    binary = tmp_path / "adapter.so"
    binary.write_bytes(b"unit fixture, not a compiled module")
    metadata = gds_flat.flat_reader_metadata(SimpleNamespace(__file__=str(binary)))
    assert metadata == {"reader_backend": "pygds_flat_sdk",
                        "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    assert gds_flat.flat_reader_metadata() == {
        "reader_backend": "pygds_generic", "native_binary_sha256": None}


def test_optional_native_path_boundaries(tmp_path):
    pygds = pytest.importorskip("pygds")
    native = pytest.importorskip("staar_gds_flat")
    assert not hasattr(native, "read_flat")
    file = pygds.gdsfile()
    file.create(str(tmp_path / "bits.gds"))
    expected = np.arange(30, dtype=np.uint8).reshape(5, 3, 2) % 4
    try:
        file.root().add("bits", expected, storage="bit2", compress="ZIP_RA")
        file.root().add("integer", expected, storage="int32")
    finally:
        file.close()
    file.open(str(tmp_path / "bits.gds"), readonly=True)
    file_id = file.fileid
    try:
        for offset, count in ((0, 30), (1, 17), (3, 1), (29, 1), (30, 0)):
            actual = native.read_flat_path(file_id, "bits", offset, count)
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous
            np.testing.assert_array_equal(actual, expected.reshape(-1)[offset:offset+count])
        for selection in (np.ones(6, dtype=bool), np.zeros(6, dtype=bool),
                          np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8),
                          np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)):
            for first_row, rows in ((0, 5), (1, 3), (4, 1), (5, 0)):
                actual = native.read_selected_rows_path(file_id, "bits", first_row*6,
                                                        rows, 6, selection)
                wanted = expected[first_row:first_row+rows].reshape(rows, 6)[:, selection.astype(bool)]
                assert actual.dtype == np.uint8 and actual.flags.c_contiguous
                np.testing.assert_array_equal(actual, wanted.reshape(-1))
        for offset, count in ((-1, 1), (0, -1), (31, 0), (29, 2), (2**63-1, 1)):
            with pytest.raises((ValueError, IndexError)):
                native.read_flat_path(file_id, "bits", offset, count)
        with pytest.raises(OverflowError):
            native.read_flat_path(file_id, "bits", 0, 2**63)
        for path in ("integer", "/"):
            with pytest.raises(TypeError):
                native.read_flat_path(file_id, path, 0, 1)
        with pytest.raises(ValueError):
            native.read_flat_path(file_id, "bits", 0, 1, "float64")
        with pytest.raises(RuntimeError):
            native.read_flat_path(file_id, "missing", 0, 1)
        valid = np.ones(6, dtype=bool)
        for offset, rows, width in ((-1, 1, 6), (0, -1, 6), (0, 1, 0),
                                    (1, 1, 6), (24, 2, 6), (0, 1, 7), (0, 2**63-1, 6)):
            with pytest.raises((ValueError, IndexError)):
                native.read_selected_rows_path(file_id, "bits", offset, rows, width, valid)
        for invalid in (np.ones(3, dtype=bool), np.ones((2, 3), dtype=bool),
                        np.ones(6, dtype=np.int32), np.array([1, 1, 2, 1, 1, 1], dtype=np.uint8),
                        np.ones(12, dtype=bool)[::2]):
            with pytest.raises((TypeError, ValueError, BufferError)):
                native.read_selected_rows_path(file_id, "bits", 0, 1, 6, invalid)
    finally:
        file.close()
    with pytest.raises(RuntimeError):
        native.read_flat_path(file_id, "bits", 0, 1)
    with pytest.raises(RuntimeError):
        native.read_selected_rows_path(file_id, "bits", 0, 1, 6, np.ones(6, dtype=bool))
