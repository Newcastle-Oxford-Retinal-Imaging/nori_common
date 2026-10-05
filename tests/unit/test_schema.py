"""Tests for the norifmt schema contract."""

import numpy as np
import pytest

from nori_common.exceptions import IncompatibleFormatError
from nori_common.fits import Compression, Precision, WritePolicy, write_fits_multi
from nori_common.fits.schema import (
    CARD_N_FRAMES,
    CARD_N_PIXELS_HORIZONTAL_INT,
    CARD_N_PIXELS_VERTICAL,
    CARD_N_SAMPLES_ORIGINAL,
    CARD_NON_FLYBACK_END,
    CARD_NON_FLYBACK_START,
    CARD_SCANNER_OFFSET_X,
    CARD_SCANNER_OFFSET_Y,
    CARD_TIME_INTERVAL,
    EXT_FIT_PARAMS,
    EXT_FRAMES,
    EXT_LINE_STARTS,
    FORMAT_VERSION,
    FrameMetadata,
    TrialArrays,
)

F64_GZIP = WritePolicy(precision=Precision.FLOAT64, compression=Compression.GZIP)


def _metadata():
    return FrameMetadata(
        n_frames=3,
        n_samples_original=3 * 4 * 5,
        time_interval=1.0 / 30_000,
        n_pixels_vertical=4,
        n_pixels_horizontal_int=5,
        scanner_offset_x=1.25,
        scanner_offset_y=-0.5,
        non_flyback_start=10,
        non_flyback_end=90,
    )


def _trial():
    metadata = _metadata()
    return TrialArrays(
        frames=np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5),
        fit_parameters=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        line_starts=np.array([0.0, 1e-3, 2e-3], dtype=np.float64),
        metadata=metadata,
    )


def test_extension_name_constants():
    assert EXT_FRAMES == "PRIMARY"
    assert EXT_FIT_PARAMS == "FIT_PARAMS"
    assert EXT_LINE_STARTS == "LINE_STARTS"
    assert FORMAT_VERSION == 1


def test_card_name_constants():
    assert CARD_N_FRAMES == "n_frames"
    assert CARD_N_SAMPLES_ORIGINAL == "n_samples_original"
    assert CARD_TIME_INTERVAL == "time_interval"
    assert CARD_N_PIXELS_VERTICAL == "n_pixels_vertical"
    assert CARD_N_PIXELS_HORIZONTAL_INT == "n_pixels_horizontal_int"
    assert CARD_SCANNER_OFFSET_X == "scanner_offset_x"
    assert CARD_SCANNER_OFFSET_Y == "scanner_offset_y"
    assert CARD_NON_FLYBACK_START == "Non_flyback_start"
    assert CARD_NON_FLYBACK_END == "Non_flyback_end"


def test_frame_metadata_header_roundtrip():
    metadata = _metadata()
    header = metadata.to_header()
    assert header[CARD_N_FRAMES] == 3
    assert header[CARD_TIME_INTERVAL] == pytest.approx(1.0 / 30_000)
    assert FrameMetadata.from_header(header) == metadata


def test_frame_metadata_missing_card_raises():
    header = _metadata().to_header()
    del header[CARD_N_FRAMES]
    with pytest.raises(IncompatibleFormatError, match=CARD_N_FRAMES):
        FrameMetadata.from_header(header)


def test_trial_to_extensions_layout():
    extensions = _trial().to_extensions()
    assert [extension.name for extension in extensions] == [
        EXT_FRAMES,
        EXT_FIT_PARAMS,
        EXT_LINE_STARTS,
    ]
    assert CARD_N_FRAMES in extensions[0].header


def test_trial_roundtrip_via_file(tmp_path):
    path = tmp_path / "trial.fits"
    trial = _trial()
    write_fits_multi(path, trial.to_extensions(), policy=F64_GZIP)

    loaded = TrialArrays.from_file(path)
    np.testing.assert_array_equal(loaded.frames, trial.frames)
    np.testing.assert_array_equal(loaded.fit_parameters, trial.fit_parameters)
    np.testing.assert_array_equal(loaded.line_starts, trial.line_starts)
    assert loaded.metadata.n_frames == trial.metadata.n_frames
    assert loaded.metadata.n_samples_original == trial.metadata.n_samples_original
    assert loaded.metadata.n_pixels_vertical == trial.metadata.n_pixels_vertical
    assert (
        loaded.metadata.n_pixels_horizontal_int
        == trial.metadata.n_pixels_horizontal_int
    )
    assert loaded.metadata.non_flyback_start == trial.metadata.non_flyback_start
    assert loaded.metadata.non_flyback_end == trial.metadata.non_flyback_end
    # FITS header floats round-trip as text at ~15 significant digits.
    assert loaded.metadata.time_interval == pytest.approx(trial.metadata.time_interval)
    assert loaded.metadata.scanner_offset_x == pytest.approx(
        trial.metadata.scanner_offset_x
    )
    assert loaded.metadata.scanner_offset_y == pytest.approx(
        trial.metadata.scanner_offset_y
    )


def test_fit_parameters_row_convention_is_slow_then_fast(tmp_path):
    path = tmp_path / "trial.fits"
    write_fits_multi(path, _trial().to_extensions(), policy=F64_GZIP)

    loaded = TrialArrays.from_file(path)
    slow_params = loaded.fit_parameters[0]
    fast_params = loaded.fit_parameters[1]
    np.testing.assert_array_equal(slow_params, [0.1, 0.2])
    np.testing.assert_array_equal(fast_params, [0.3, 0.4])
