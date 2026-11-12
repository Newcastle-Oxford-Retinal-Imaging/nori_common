"""Tests for array-aware contract validation."""

import numpy
import pytest

from nori_common.contracts import (
    FrameMetadata,
    RawAcquisitionMetadata,
    SchemaError,
    TrialMetadata,
)
from nori_common.validation import (
    validate_frame_arrays,
    validate_raw_data,
    validate_trial_arrays,
)


def _frame_metadata():
    return FrameMetadata(
        n_frames=2,
        n_samples_original=20,
        time_interval=1.0e-07,
        n_pixels_vertical=5,
        n_pixels_horizontal_int=8,
        scanner_offset_x=0.0,
        scanner_offset_y=0.0,
        Non_flyback_start=0,
        Non_flyback_end=4,
    )


def _frames():
    return numpy.zeros((1, 2, 5, 4, 2), dtype=numpy.float32)


def test_validate_frame_arrays_accepts_consistent_data():
    validate_frame_arrays(_frame_metadata(), _frames())


def test_validate_frame_arrays_rejects_frame_count_mismatch():
    with pytest.raises(SchemaError):
        validate_frame_arrays(_frame_metadata(), numpy.zeros((1, 3, 5, 4, 2)))


def test_validate_trial_arrays_accepts_consistent_data():
    metadata = TrialMetadata(frame=_frame_metadata())
    fit_parameters = numpy.zeros((2, 5), dtype=numpy.float32)
    line_starts = numpy.arange(11, dtype=numpy.float32)

    validate_trial_arrays(metadata, _frames(), fit_parameters, line_starts)


def test_validate_trial_arrays_rejects_short_line_starts():
    metadata = TrialMetadata(frame=_frame_metadata())
    fit_parameters = numpy.zeros((2, 5), dtype=numpy.float32)
    line_starts = numpy.arange(5, dtype=numpy.float32)

    with pytest.raises(SchemaError):
        validate_trial_arrays(metadata, _frames(), fit_parameters, line_starts)


def test_validate_raw_data_accepts_consistent_data():
    metadata = RawAcquisitionMetadata(
        channel_list=(0, 1, 2),
        slow_channel="A",
        fast_channel="B",
        confocal_channel="C",
        n_pixels_horizontal_float=2.0,
        n_pixels_vertical=4,
        time_interval=1.0e-07,
        frame_rate_hz=300.0,
    )
    validate_raw_data(metadata, numpy.zeros((3, 16), dtype=numpy.int16))


def test_validate_raw_data_rejects_channel_count_mismatch():
    metadata = RawAcquisitionMetadata(
        channel_list=(0, 1, 2),
        slow_channel="A",
        fast_channel="B",
        confocal_channel="C",
        n_pixels_horizontal_float=2.0,
        n_pixels_vertical=4,
        time_interval=1.0e-07,
        frame_rate_hz=300.0,
    )
    with pytest.raises(SchemaError):
        validate_raw_data(metadata, numpy.zeros((2, 16), dtype=numpy.int16))
