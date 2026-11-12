"""Contract tests for simple FITS array adapters."""

import numpy
import pytest
from astropy.io import fits

from nori_common.data import (
    CapabilityNotAvailable,
    SchemaError,
    read_coarse_motion,
    read_eye_motion,
    read_eye_motion_per_row,
    read_frame_positions,
    read_metrics,
    read_montage_motion,
)


def _write(tmp_path, name, data):
    path = tmp_path / name
    fits.PrimaryHDU(numpy.asarray(data)).writeto(path)
    return path


def test_eye_motion_adapter(tmp_path):
    data = numpy.arange(2 * 3 * 4 * 2, dtype=numpy.float32).reshape(2, 3, 4, 2)
    path = _write(tmp_path, "eye_motion.fits", data)

    adapter = read_eye_motion(path)

    numpy.testing.assert_array_equal(adapter.positions, data)


def test_eye_motion_adapter_rejects_wrong_trailing_axis(tmp_path):
    path = _write(tmp_path, "bad_eye_motion.fits", numpy.zeros((2, 3, 4, 3)))

    with pytest.raises(SchemaError):
        _ = read_eye_motion(path).positions


def test_eye_motion_per_row_adapter(tmp_path):
    data = numpy.arange(2 * 3 * 2, dtype=numpy.float32).reshape(2, 3, 2)
    path = _write(tmp_path, "eye_motion_row.fits", data)

    adapter = read_eye_motion_per_row(path)

    numpy.testing.assert_array_equal(adapter.positions, data)


def test_metrics_adapter(tmp_path):
    data = numpy.arange(2 * 4, dtype=numpy.float32).reshape(2, 4)
    path = _write(tmp_path, "metrics.fits", data)

    adapter = read_metrics(path)

    numpy.testing.assert_array_equal(adapter.values, data)
    numpy.testing.assert_array_equal(adapter.ncc, data[:, 0])
    numpy.testing.assert_array_equal(adapter.blink, data[:, 1])
    numpy.testing.assert_array_equal(adapter.std_dev, data[:, 2])


def test_metrics_adapter_std_dev_absent(tmp_path):
    path = _write(tmp_path, "metrics_two.fits", numpy.zeros((2, 2)))

    adapter = read_metrics(path)

    with pytest.raises(CapabilityNotAvailable):
        _ = adapter.std_dev


def test_frame_positions_adapter(tmp_path):
    data = numpy.arange(2 * 8, dtype=numpy.float32).reshape(2, 8)
    path = _write(tmp_path, "frame_positions.fits", data)

    numpy.testing.assert_array_equal(read_frame_positions(path).positions, data)


def test_frame_positions_adapter_rejects_short_rows(tmp_path):
    path = _write(tmp_path, "bad_positions.fits", numpy.zeros((2, 7)))

    with pytest.raises(SchemaError):
        _ = read_frame_positions(path).positions


def test_coarse_motion_adapter(tmp_path):
    data = numpy.arange(2 * 2, dtype=numpy.float32).reshape(2, 2)
    path = _write(tmp_path, "coarse.fits", data)

    numpy.testing.assert_array_equal(read_coarse_motion(path).positions, data)


def test_montage_motion_adapter(tmp_path):
    data = numpy.arange(3 * 4 * 2, dtype=numpy.float32).reshape(3, 4, 2)
    path = _write(tmp_path, "montage_motion.fits", data)

    numpy.testing.assert_array_equal(read_montage_motion(path).motion, data)


def test_montage_motion_adapter_rejects_wrong_trailing_axis(tmp_path):
    path = _write(tmp_path, "bad_montage.fits", numpy.zeros((3, 4, 3)))

    with pytest.raises(SchemaError):
        _ = read_montage_motion(path).motion
