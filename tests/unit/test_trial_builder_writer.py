"""Tests for trial payload builder and schema-aware writer."""

import numpy
import pytest

from nori_common.adapters import read_trial
from nori_common.builders import build_trial_payload
from nori_common.contracts import (
    ArtifactProvenance,
    FrameShape,
    PhotodiodeTransitionsData,
    PixelScaleMetadata,
    SchemaError,
    Timing,
)
from nori_common.fits import WritePolicy
from nori_common.manifest import read_manifest, read_sidecar
from nori_common.writers import write_trial


def _payload():
    frames = numpy.zeros((2, 3, 5, 4, 2), dtype=numpy.float32)
    fit_parameters = numpy.zeros((2, 5), dtype=numpy.float32)
    line_starts = numpy.arange(16, dtype=numpy.float32)
    return build_trial_payload(
        frames,
        fit_parameters,
        line_starts,
        n_samples_original=3 * 5 * 8,
        time_interval=1.0e-07,
        scanner_offset=(1.5, -2.5),
        non_flyback_rows=(1, 4),
        provenance=ArtifactProvenance(source_type="simulated", creator="ERICA"),
        pixel_scale=PixelScaleMetadata(
            pixel_size_arcmin_x=0.11,
            pixel_size_arcmin_y=0.12,
        ),
        photodiode=PhotodiodeTransitionsData(
            values={1: 10.0},
            transitions={1: 0.001},
        ),
        channel_roles={0: "confocal", 1: "photodiode"},
    )


def test_builder_derives_frame_metadata():
    payload = _payload()
    frame = payload.metadata.frame

    assert frame.n_frames == 3
    assert frame.n_pixels_vertical == 5
    assert frame.n_pixels_horizontal_int == 8
    assert frame.scanner_offset_x == -2.5
    assert frame.scanner_offset_y == 1.5
    assert payload.metadata.channel_roles == {0: "confocal", 1: "photodiode"}


def test_builder_rejects_short_line_starts():
    with pytest.raises(SchemaError):
        build_trial_payload(
            numpy.zeros((1, 2, 5, 4, 2)),
            numpy.zeros((2, 5)),
            numpy.arange(3),
            n_samples_original=20,
            time_interval=1.0e-07,
            scanner_offset=(0.0, 0.0),
            non_flyback_rows=(0, 4),
        )


def test_writer_roundtrip(tmp_path):
    payload = _payload()
    fits_path = tmp_path / "trial.fits"
    sidecar_path = tmp_path / "trial.yaml"

    write_trial(
        fits_path,
        payload,
        policy=WritePolicy(),
        sidecar_path=sidecar_path,
    )

    manifest = read_manifest(fits_path)
    assert manifest.schema_name == "trial"
    assert read_sidecar(sidecar_path) == manifest

    trial = read_trial(fits_path)
    assert trial.require(FrameShape).n_frames == 3
    assert trial.require(FrameShape).frame_shape == (5, 4)
    assert trial.require(Timing).fit_parameters.shape == (2, 5)
    assert trial.require(Timing).line_starts.shape == (16,)
