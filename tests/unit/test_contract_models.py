"""Tests for Pydantic contract models."""

import pytest
from pydantic import ValidationError

from nori_common.contracts import (
    ArrayDescriptor,
    ArtifactProvenance,
    AxisDescriptor,
    FrameMetadata,
    Manifest,
    PhotodiodeTransitionsData,
    PixelScaleMetadata,
    RawAcquisitionMetadata,
    TrialMetadata,
)


def _frames_descriptor(shape=(1, 2, 5, 4, 2)):
    names = ("channel", "frame", "y", "x", "direction")
    labels = (("confocal",), None, None, None, ("forward", "reverse"))
    return ArrayDescriptor(
        name="PRIMARY",
        role="frames",
        shape=shape,
        axes=tuple(
            AxisDescriptor(name=name, size=size, labels=label)
            for name, size, label in zip(names, shape, labels, strict=True)
        ),
    )


def _frame_metadata(**overrides):
    values = {
        "n_frames": 2,
        "n_samples_original": 20,
        "time_interval": 1.0e-07,
        "n_pixels_vertical": 5,
        "n_pixels_horizontal_int": 8,
        "scanner_offset_x": 1.0,
        "scanner_offset_y": -0.5,
        "Non_flyback_start": 1,
        "Non_flyback_end": 3,
    }
    values.update(overrides)
    return FrameMetadata(**values)


def test_frame_metadata_alias_roundtrip():
    frame = _frame_metadata()

    header = frame.to_header()

    assert header["Non_flyback_start"] == 1
    assert header["Non_flyback_end"] == 3
    assert FrameMetadata.from_header(header) == frame


def test_frame_metadata_rejects_invalid_bounds():
    with pytest.raises(ValidationError):
        _frame_metadata(Non_flyback_start=4, Non_flyback_end=3)


def test_trial_metadata_preserves_extra_fields():
    trial = TrialMetadata(
        frame=_frame_metadata(),
        provenance=ArtifactProvenance(
            source_type="simulated",
            creator="ERICA",
            custom_field="kept",
        ),
        arrays=(_frames_descriptor(),),
        pixel_scale=PixelScaleMetadata(
            pixel_size_arcmin_x=0.1,
            pixel_size_arcmin_y=0.2,
        ),
        photodiode=PhotodiodeTransitionsData(values={1: 1.0}),
    )

    assert trial.schema_name == "trial"
    assert trial.provenance.custom_field == "kept"
    assert trial.arrays[0].role == "frames"


def test_raw_acquisition_metadata_validation():
    raw = RawAcquisitionMetadata(
        channel_list=(0, 1, 2),
        slow_channel="A",
        fast_channel="B",
        confocal_channel="C",
        n_pixels_horizontal_float=2.0,
        n_pixels_vertical=4,
        time_interval=1.0e-07,
        down_sample_ratio=1,
        frame_rate_hz=300.0,
    )

    assert raw.schema_name == "raw"
    assert raw.channel_list == (0, 1, 2)


def test_raw_acquisition_metadata_rejects_duplicate_channels():
    with pytest.raises(ValidationError):
        RawAcquisitionMetadata(
            channel_list=(0, 0, 2),
            slow_channel="A",
            fast_channel="B",
            confocal_channel="C",
            n_pixels_horizontal_float=2.0,
            n_pixels_vertical=4,
            time_interval=1.0e-07,
            frame_rate_hz=300.0,
        )


def test_manifest_contract():
    manifest = Manifest(
        schema_name="trial",
        arrays=(_frames_descriptor(),),
        metadata={"n_frames": 2},
        provenance=ArtifactProvenance(creator="nori"),
    )

    assert manifest.schema_name == "trial"
    assert manifest.arrays[0].role == "frames"
    assert manifest.arrays[0].shape == (1, 2, 5, 4, 2)
    assert manifest.arrays[0].axes[0].labels == ("confocal",)


def test_axis_descriptor_rejects_label_count_mismatch():
    with pytest.raises(ValidationError):
        AxisDescriptor(name="channel", size=2, labels=("confocal",))


def test_array_descriptor_rejects_shape_axis_mismatch():
    with pytest.raises(ValidationError):
        ArrayDescriptor(
            name="PRIMARY",
            role="frames",
            shape=(2, 3),
            axes=(
                AxisDescriptor(name="frame", size=3),
                AxisDescriptor(name="row", size=2),
            ),
        )
