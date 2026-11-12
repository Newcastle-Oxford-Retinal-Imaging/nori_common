"""Tests for single-array payload builder and writer."""

import numpy
import pytest

from nori_common.adapters import read_eye_motion_per_row
from nori_common.builders import build_array_payload
from nori_common.contracts import ArtifactProvenance, SchemaError
from nori_common.fits import FitsFile, WritePolicy
from nori_common.manifest import (
    MANIFEST_CARD,
    SCHEMA_CARD,
    SCHEMA_VERSION_CARD,
    read_manifest,
    read_sidecar,
)
from nori_common.writers import write_array_artifact


def test_array_writer_roundtrip(tmp_path):
    data = numpy.arange(12, dtype=numpy.float32).reshape(2, 3, 2)
    payload = build_array_payload(
        data,
        schema_name="eye_motion_per_row",
        role="per_row_positions",
        axes=("frame", "row", "axis"),
        metadata={"frame_count": 2},
        provenance=ArtifactProvenance(source_type="measured", creator="nori"),
        header={"frame_rate": 30.0},
    )
    fits_path = tmp_path / "eye_motion.fits"
    sidecar_path = tmp_path / "eye_motion.yaml"

    write_array_artifact(
        fits_path,
        payload,
        policy=WritePolicy(),
        sidecar_path=sidecar_path,
    )

    manifest = read_manifest(fits_path)
    assert manifest.schema_name == "eye_motion_per_row"
    assert manifest.arrays[0].role == "per_row_positions"
    assert manifest.arrays[0].shape == (2, 3, 2)
    assert [axis.name for axis in manifest.arrays[0].axes] == [
        "frame",
        "row",
        "axis",
    ]
    assert manifest.arrays[0].axes[-1].labels == ("y", "x")
    assert read_sidecar(sidecar_path) == manifest

    with FitsFile.open(fits_path) as fits_file:
        numpy.testing.assert_array_equal(fits_file.array(), data)
        header = fits_file.header()
    assert header[SCHEMA_CARD] == "eye_motion_per_row"
    assert header[SCHEMA_VERSION_CARD] == "1.0"
    assert header[MANIFEST_CARD] == 1
    assert header["frame_rate"] == 30.0

    adapter = read_eye_motion_per_row(fits_path)
    assert adapter.manifest is not None
    assert adapter.manifest.arrays[0].shape == (2, 3, 2)
    numpy.testing.assert_array_equal(adapter.positions, data)


def test_metrics_axes_use_column_names():
    payload = build_array_payload(
        numpy.zeros((3, 4), dtype=numpy.float32),
        schema_name="metrics",
        role="metrics",
        metadata={"column_names": ("ncc", "blink", "std_dev", "mean")},
    )

    descriptor = payload.manifest.arrays[0]
    assert descriptor.shape == (3, 4)
    assert descriptor.axes[1].labels == ("ncc", "blink", "std_dev", "mean")


def test_trace_axes_use_cols():
    payload = build_array_payload(
        numpy.zeros((9, 20), dtype=numpy.float32),
        schema_name="trace",
        role="eye_motion_trace",
        metadata={"cols": "t,y,x,fn,fb,pd,bl,tk,sd"},
    )

    descriptor = payload.manifest.arrays[0]
    assert descriptor.axes[0].labels == (
        "t",
        "y",
        "x",
        "fn",
        "fb",
        "pd",
        "bl",
        "tk",
        "sd",
    )


def test_frame_positions_axes_use_bound_labels():
    payload = build_array_payload(
        numpy.zeros((3, 8), dtype=numpy.float32),
        schema_name="frame_positions",
        role="frame_positions",
    )

    descriptor = payload.manifest.arrays[0]
    assert descriptor.axes[1].labels == (
        "start_y",
        "stop_y",
        "start_x",
        "stop_x",
        "y_of_s",
        "y_of_s2",
        "x_of_s",
        "x_of_s2",
    )


def test_unknown_product_without_axes_is_rejected():
    with pytest.raises(SchemaError):
        build_array_payload(
            numpy.zeros((2, 3), dtype=numpy.float32),
            schema_name="unknown_product",
            role="unknown_role",
        )
