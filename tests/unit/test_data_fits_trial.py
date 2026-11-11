"""Contract tests for the FITS trial adapter."""

from pathlib import Path

import numpy
import pytest
from astropy.io import fits

from nori_common.data import (
    CapabilityNotAvailable,
    FitsTrialAdapter,
    FrameRate,
    Frames,
    Provenance,
    ScanGeometry,
    SchemaError,
)


def _frames():
    return numpy.arange(1 * 2 * 5 * 4 * 2, dtype=numpy.float32).reshape((1, 2, 5, 4, 2))


def _header():
    header = fits.Header()
    header["Non_flyback_start"] = 1
    header["Non_flyback_end"] = 3
    header["SOURCE_TYPE"] = "measured"
    header["CREATOR"] = "unit-test"
    header["CREATOR_VERSION"] = "1.0"
    header["CUSTOM_KEY"] = "kept"
    return header


def _write_current(tmp_path: Path) -> Path:
    """Current-style compressed file with a named PRIMARY extension."""
    path = tmp_path / "current.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(_frames(), header=_header(), name="PRIMARY"),
        ]
    )
    hdul.writeto(path)
    return path


def _write_legacy(tmp_path: Path) -> Path:
    """Legacy-style uncompressed primary with unnamed secondary HDUs."""
    path = tmp_path / "legacy.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(_frames(), header=_header()),
            fits.ImageHDU(numpy.zeros((2, 5), dtype=numpy.float32)),
            fits.ImageHDU(numpy.zeros((10,), dtype=numpy.float32)),
        ]
    )
    hdul.writeto(path)
    return path


@pytest.mark.parametrize("writer", [_write_current, _write_legacy])
def test_fits_trial_contract(tmp_path, writer):
    path = writer(tmp_path)

    trial = FitsTrialAdapter(path, frame_rate=123.0)

    assert trial.supports(Frames)
    assert trial.supports(ScanGeometry)
    assert trial.supports(FrameRate)
    assert trial.supports(Provenance)

    numpy.testing.assert_array_equal(trial.require(Frames).frames, _frames())
    assert trial.require(ScanGeometry).non_flyback_rows == (1, 3)
    assert trial.require(FrameRate).frame_rate_hz == 123.0
    assert trial.provenance.source_type == "measured"
    assert trial.provenance.creator == "unit-test"
    assert trial.provenance.creator_version == "1.0"


def test_fits_trial_preserves_extra_header_cards(tmp_path):
    path = _write_current(tmp_path)
    trial = FitsTrialAdapter(path, frame_rate=30.0)

    assert trial.extras.get_extra("fits_header.CUSTOM_KEY") == "kept"
    assert trial.extras.get_extra("fits_header.CREATOR") is None


def test_fits_trial_requires_non_flyback_cards(tmp_path):
    path = tmp_path / "missing.fits"
    header = fits.Header()
    fits.PrimaryHDU(_frames(), header=header).writeto(path)

    with pytest.raises(SchemaError):
        FitsTrialAdapter(path, frame_rate=30.0)


def test_fits_trial_without_frame_rate_is_capability_absence(tmp_path):
    path = _write_current(tmp_path)
    trial = FitsTrialAdapter(path)

    assert not trial.supports(FrameRate)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(FrameRate)


def test_fits_trial_reads_frame_rate_from_header(tmp_path):
    path = tmp_path / "header_rate.fits"
    header = _header()
    header["AOSLO_frame_rate"] = 456.0
    fits.PrimaryHDU(_frames(), header=header).writeto(path)

    trial = FitsTrialAdapter(path)

    assert trial.require(FrameRate).frame_rate_hz == 456.0


def test_fits_trial_rejects_wrong_rank(tmp_path):
    path = tmp_path / "wrong_rank.fits"
    header = _header()
    fits.PrimaryHDU(numpy.zeros((2, 5), dtype=numpy.float32), header=header).writeto(
        path
    )

    trial = FitsTrialAdapter(path, frame_rate=30.0)
    with pytest.raises(SchemaError):
        _ = trial.frames
