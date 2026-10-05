"""Tests for the nori_common write API."""

import logging

import numpy as np
import pytest
from astropy.io import fits

from nori_common.fits import (
    Compression,
    Extension,
    Precision,
    WritePolicy,
    write_fits,
    write_fits_multi,
)


def _data(shape=(10, 100)):
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


def _first_data(hdu_list):
    return next(h for h in hdu_list if h.data is not None)


def _ztiles(hdu_list, hdu):
    bintable = hdu._bintable.header
    return [bintable[f"ZTILE{j}"] for j in range(1, bintable["ZNAXIS"] + 1)]


def test_write_policy_defaults():
    policy = WritePolicy()
    assert policy.precision is Precision.FLOAT32
    assert policy.compression is Compression.NONE
    assert policy.gzip_level == 6


@pytest.mark.parametrize("level", [0, 10, -1])
def test_write_policy_rejects_out_of_range_gzip_level(level):
    with pytest.raises(ValueError, match="gzip_level"):
        WritePolicy(gzip_level=level)


def test_extension_defaults():
    data = np.zeros((2, 2))
    ext = Extension(data)
    assert ext.data is data
    assert ext.name is None
    assert ext.header is None


@pytest.mark.parametrize(
    "precision,itemsize",
    [(Precision.FLOAT64, 8), (Precision.FLOAT32, 4)],
)
def test_write_fits_precision(tmp_path, precision, itemsize):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(precision=precision))

    with fits.open(path) as hdu_list:
        written = _first_data(hdu_list).data
        assert written.shape == data.shape
        assert written.dtype.kind == "f"
        assert written.dtype.itemsize == itemsize
        np.testing.assert_allclose(written, data)


def test_write_fits_default_policy_is_float32_uncompressed(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data())

    with fits.open(path) as hdu_list:
        assert len(hdu_list) == 1
        assert hdu_list[0].data is not None
        assert hdu_list[0].data.dtype.itemsize == 4


def test_write_fits_uncompressed_uses_primary(tmp_path):
    path = tmp_path / "out.fits"
    data = _data()
    write_fits(path, data, policy=WritePolicy(compression=Compression.NONE))

    with fits.open(path) as hdu_list:
        assert len(hdu_list) == 1
        assert hdu_list.index_of(_first_data(hdu_list)) == 0


def test_write_fits_gzip_prepends_blank_primary(tmp_path):
    path = tmp_path / "out.fits"
    write_fits(path, _data(), policy=WritePolicy(compression=Compression.GZIP))

    with fits.open(path) as hdu_list:
        assert hdu_list[0].data is None
        assert hdu_list.index_of(_first_data(hdu_list)) >= 1


def test_write_fits_gzip_is_smaller(tmp_path):
    data = _data((100, 100))
    compressed = tmp_path / "c.fits"
    uncompressed = tmp_path / "u.fits"
    write_fits(compressed, data, policy=WritePolicy(compression=Compression.GZIP))
    write_fits(uncompressed, data, policy=WritePolicy(compression=Compression.NONE))
    assert compressed.stat().st_size < uncompressed.stat().st_size


@pytest.mark.parametrize("level", [1, 6, 9])
def test_write_fits_gzip_roundtrip_bit_exact(tmp_path, level):
    path = tmp_path / "out.fits"
    data = _data()
    data[5, 5] = np.nan
    write_fits(
        path,
        data,
        policy=WritePolicy(
            precision=Precision.FLOAT64,
            compression=Compression.GZIP,
            gzip_level=level,
        ),
    )

    with fits.open(path) as hdu_list:
        written = _first_data(hdu_list).data
        assert written.dtype == data.dtype
        np.testing.assert_array_equal(written, data)


def test_write_fits_gzip_tiles_per_image(tmp_path):
    path = tmp_path / "out.fits"
    data = np.random.default_rng(0).random((8, 100, 80, 2))
    write_fits(
        path,
        data,
        policy=WritePolicy(precision=Precision.FLOAT32, compression=Compression.GZIP),
    )

    with fits.open(path) as hdu_list:
        # numpy (8, 100, 80, 2) -> FITS ZNAXIS reversed; leading frame axis
        # must tile with size 1 so each image is its own tile.
        assert _ztiles(hdu_list, _first_data(hdu_list)) == [2, 80, 100, 1]


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((2, 3), None),
        ((100, 80), None),
        ((8, 100, 80), (1, 100, 80)),
        ((8, 100, 2), (1, 100, 2)),
        ((8, 100, 80, 2), (1, 100, 80, 2)),
        ((2, 8, 100, 80, 2), (1, 1, 100, 80, 2)),
    ],
)
def test_tile_shape_for(shape, expected):
    from nori_common.fits import _tile_shape_for

    assert _tile_shape_for(np.empty(shape)) == expected


def test_write_fits_header_roundtrip(tmp_path):
    path = tmp_path / "out.fits"
    header = {"TESTKEY": 42, "Non_flyback_start": 10}
    write_fits(path, _data(), header=header)

    with fits.open(path) as hdu_list:
        assert hdu_list[0].header["TESTKEY"] == 42
        assert hdu_list[0].header["Non_flyback_start"] == 10


def test_write_fits_logs_compression(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "out.fits"
    write_fits(
        path,
        _data(),
        policy=WritePolicy(compression=Compression.GZIP, gzip_level=6),
    )
    assert "lossless" in caplog.text
    assert "level=6" in caplog.text


def test_write_fits_multi_uncompressed_layout(tmp_path):
    path = tmp_path / "out.fits"
    first = np.arange(6, dtype=np.float64).reshape(2, 3)
    second = np.arange(12, dtype=np.float64).reshape(3, 4)
    write_fits_multi(
        path,
        [Extension(first, "PRIMARY"), Extension(second, "SECOND")],
        policy=WritePolicy(),
    )

    with fits.open(path) as hdu_list:
        assert len(hdu_list) == 2
        np.testing.assert_allclose(hdu_list[0].data, first)
        np.testing.assert_allclose(hdu_list[1].data, second)
        assert hdu_list[1].name == "SECOND"


def test_write_fits_multi_gzip_bit_exact(tmp_path):
    path = tmp_path / "out.fits"
    frames = np.random.default_rng(1).random((2, 16, 16)) * 100
    fit_params = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    line_starts = np.array([0.0, 1e-3, 2e-3], dtype=np.float64)
    write_fits_multi(
        path,
        [
            Extension(frames, "PRIMARY"),
            Extension(fit_params, "FIT_PARAMS"),
            Extension(line_starts, "LINE_STARTS"),
        ],
        policy=WritePolicy(precision=Precision.FLOAT32, compression=Compression.GZIP),
    )

    with fits.open(path) as hdu_list:
        assert hdu_list[0].data is None
        frames_hdu = next(
            h for h in hdu_list if h.name == "PRIMARY" and h.data is not None
        )
        np.testing.assert_array_equal(frames_hdu.data, frames.astype(np.float32))
        np.testing.assert_array_equal(
            next(h for h in hdu_list if h.name == "FIT_PARAMS").data,
            fit_params.astype(np.float32),
        )
        np.testing.assert_array_equal(
            next(h for h in hdu_list if h.name == "LINE_STARTS").data,
            line_starts.astype(np.float32),
        )


def test_write_fits_multi_gzip_tiles_per_image(tmp_path):
    path = tmp_path / "out.fits"
    frames = np.random.default_rng(2).random((2, 8, 100, 80, 2))
    write_fits_multi(
        path,
        [
            Extension(frames, "PRIMARY"),
            Extension(np.zeros((2, 5)), "FIT_PARAMS"),
        ],
        policy=WritePolicy(precision=Precision.FLOAT32, compression=Compression.GZIP),
    )

    with fits.open(path) as hdu_list:
        frames_hdu = next(
            h for h in hdu_list if h.name == "PRIMARY" and h.data is not None
        )
        assert _ztiles(hdu_list, frames_hdu) == [2, 80, 100, 1, 1]
