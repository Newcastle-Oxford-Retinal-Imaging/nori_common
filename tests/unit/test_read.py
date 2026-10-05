"""Tests for the nori_common read API."""

import numpy as np
import pytest
from astropy.io import fits

from nori_common.exceptions import ExtensionNotFound, FitsError
from nori_common.fits import (
    Compression,
    Extension,
    FitsFile,
    Precision,
    WritePolicy,
    write_fits,
    write_fits_multi,
)

F32 = WritePolicy(precision=Precision.FLOAT32)
GZIP = WritePolicy(compression=Compression.GZIP)


def _data(shape=(10, 100)):
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


@pytest.mark.parametrize("compression", [Compression.NONE, Compression.GZIP])
def test_array_roundtrip(tmp_path, compression):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(
        path,
        data,
        policy=WritePolicy(precision=Precision.FLOAT64, compression=compression),
    )

    with FitsFile.open(path) as fits_file:
        np.testing.assert_array_equal(fits_file.array(), data)


def test_array_returns_independent_copy_by_default(tmp_path):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(precision=Precision.FLOAT64))

    with FitsFile.open(path) as fits_file:
        result = fits_file.array()
        result[:] = -1

    with FitsFile.open(path) as fits_file:
        np.testing.assert_array_equal(fits_file.array(), data)


def test_array_can_avoid_copy(tmp_path):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(precision=Precision.FLOAT64))

    with FitsFile.open(path, memmap=True) as fits_file:
        np.testing.assert_array_equal(fits_file.array(copy=False), data)


def test_header_returns_independent_mapping(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data(), header={"TESTKEY": 42})

    with FitsFile.open(path) as fits_file:
        header = fits_file.header()
        assert header["TESTKEY"] == 42
        header["MUTATED"] = 1

    with FitsFile.open(path) as fits_file:
        assert "MUTATED" not in fits_file.header()


def test_extensions_lists_names(tmp_path):
    path = tmp_path / "out.fits"
    write_fits_multi(
        path,
        [Extension(np.zeros((2, 2)), "PRIMARY"), Extension(np.zeros((3, 3)), "SECOND")],
        policy=WritePolicy(),
    )

    with FitsFile.open(path) as fits_file:
        assert fits_file.extensions() == ["PRIMARY", "SECOND"]


def test_default_selects_data_hdu_when_primary_is_blank(tmp_path):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(compression=Compression.GZIP))

    with FitsFile.open(path) as fits_file:
        np.testing.assert_array_equal(fits_file.array(), data)


def test_primary_name_selects_data_hdu_when_compressed(tmp_path):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(compression=Compression.GZIP))

    with FitsFile.open(path) as fits_file:
        np.testing.assert_array_equal(fits_file.array("PRIMARY"), data)


def test_named_selection_is_case_insensitive(tmp_path):
    path = tmp_path / "out.fits"
    fit_params = np.zeros((2, 3), dtype=np.float64)
    write_fits_multi(
        path,
        [Extension(np.zeros((2, 2)), "PRIMARY"), Extension(fit_params, "FIT_PARAMS")],
        policy=WritePolicy(),
    )

    with FitsFile.open(path) as fits_file:
        np.testing.assert_array_equal(fits_file.array("fit_params"), fit_params)


def test_missing_extension_raises(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data())

    with FitsFile.open(path) as fits_file, pytest.raises(ExtensionNotFound):
        fits_file.array("NOPE")


def test_no_data_bearing_hdu_raises(tmp_path):
    path = tmp_path / "blank.fits"
    fits.PrimaryHDU().writeto(path)

    with FitsFile.open(path) as fits_file, pytest.raises(FitsError):
        fits_file.array()


def test_as_hdu_returns_astropy_hdu(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data())

    with FitsFile.open(path) as fits_file:
        assert isinstance(fits_file.as_hdu(), fits.PrimaryHDU)


def test_close_is_idempotent_and_releases_file(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data())

    fits_file = FitsFile.open(path, memmap=True)
    fits_file.close()
    fits_file.close()
    with pytest.raises((ValueError, OSError)):
        fits_file.array()
