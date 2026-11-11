"""Contract tests for trial timing, geometry, pixel scale, and photodiode data."""

from pathlib import Path

import numpy
import pytest
from astropy.io import fits

from nori_common.data import (
    CapabilityNotAvailable,
    FrameShape,
    PhotodiodeTransitions,
    PixelScale,
    ScanGeometry,
    Timing,
    read_trial,
)


def _frames():
    return numpy.zeros((1, 2, 5, 4, 2), dtype=numpy.float32)


def _header():
    header = fits.Header()
    header["Non_flyback_start"] = 1
    header["Non_flyback_end"] = 3
    header["n_frames"] = 2
    header["n_pixels_vertical"] = 5
    header["n_pixels_horizontal_int"] = 8
    header["scanner_offset_x"] = 1.5
    header["scanner_offset_y"] = -2.5
    header["n_samples_original"] = 22
    header["time_interval"] = 1.0e-07
    header["Pixel_size_x_arcmin"] = 0.11
    header["Pixel_size_y_arcmin"] = 0.12
    header["PD_value_001"] = 10.0
    header["PD_value_002"] = 20.0
    header["PD_transition_001"] = 0.001
    header["PD_transition_002"] = 0.002
    return header


def _write_named(tmp_path: Path) -> Path:
    path = tmp_path / "named.fits"
    fit_parameters = numpy.zeros((2, 5), dtype=numpy.float32)
    line_starts = numpy.arange(11, dtype=numpy.float32)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(_frames(), header=_header()),
            fits.ImageHDU(fit_parameters, name="FIT_PARAMS"),
            fits.ImageHDU(line_starts, name="LINE_STARTS"),
        ]
    )
    hdul.writeto(path)
    return path


def _write_legacy(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.fits"
    fit_parameters = numpy.zeros((2, 5), dtype=numpy.float32)
    line_starts = numpy.arange(11, dtype=numpy.float32)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(_frames(), header=_header()),
            fits.ImageHDU(fit_parameters),
            fits.ImageHDU(line_starts),
        ]
    )
    hdul.writeto(path)
    return path


@pytest.mark.parametrize("writer", [_write_named, _write_legacy])
def test_trial_timing_and_geometry(tmp_path, writer):
    path = writer(tmp_path)

    trial = read_trial(path, frame_rate=30.0, pixel_size_arcmin=(0.11, 0.12))

    assert trial.supports(FrameShape)
    assert trial.supports(ScanGeometry)
    assert trial.supports(Timing)
    assert trial.supports(PixelScale)
    assert trial.supports(PhotodiodeTransitions)

    assert trial.require(FrameShape).n_frames == 2
    assert trial.require(FrameShape).frame_shape == (5, 4)
    assert trial.require(ScanGeometry).non_flyback_rows == (1, 3)
    assert trial.require(ScanGeometry).scanner_offset_pixels == (-2.5, 1.5)

    timing = trial.require(Timing)
    assert timing.n_samples_original == 22
    assert timing.time_interval_s == pytest.approx(1.0e-07)
    assert timing.fit_parameters.shape == (2, 5)
    assert timing.line_starts.shape == (11,)

    assert trial.pixel_size_arcmin_x == pytest.approx(0.11)
    assert trial.pixel_size_arcmin_y == pytest.approx(0.12)

    photodiode = trial.require(PhotodiodeTransitions)
    assert photodiode.values == {1: 10.0, 2: 20.0}
    assert photodiode.transitions == {1: 0.001, 2: 0.002}


def test_trial_timing_absent_when_cards_missing(tmp_path):
    path = tmp_path / "no_timing.fits"
    header = _header()
    del header["n_samples_original"]
    fits.PrimaryHDU(_frames(), header=header).writeto(path)

    trial = read_trial(path)

    assert not trial.supports(Timing)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(Timing)


def test_trial_scanner_offsets_from_header(tmp_path):
    path = _write_named(tmp_path)
    trial = read_trial(path)

    assert trial.require(ScanGeometry).scanner_offset_pixels == (-2.5, 1.5)


def test_trial_pixel_scale_from_header(tmp_path):
    path = _write_named(tmp_path)
    trial = read_trial(path)

    assert trial.require(PixelScale).pixel_size_arcmin_x == pytest.approx(0.11)
    assert trial.require(PixelScale).pixel_size_arcmin_y == pytest.approx(0.12)


def test_trial_photodiode_absent(tmp_path):
    path = tmp_path / "no_pd.fits"
    header = _header()
    for key in list(header.keys()):
        if key.upper().startswith("PD_"):
            del header[key]
    fits.PrimaryHDU(_frames(), header=header).writeto(path)

    trial = read_trial(path)

    assert not trial.supports(PhotodiodeTransitions)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(PhotodiodeTransitions)
