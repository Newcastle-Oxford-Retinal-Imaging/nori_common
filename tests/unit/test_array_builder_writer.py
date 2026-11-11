"""Tests for single-array payload builder and writer."""

import numpy

from nori_common.builders import build_array_payload
from nori_common.contracts import ArtifactProvenance
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
        schema_name="eye_motion",
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
    assert manifest.schema_name == "eye_motion"
    assert manifest.arrays[0].role == "per_row_positions"
    assert read_sidecar(sidecar_path) == manifest

    with FitsFile.open(fits_path) as fits_file:
        numpy.testing.assert_array_equal(fits_file.array(), data)
        header = fits_file.header()
    assert header[SCHEMA_CARD] == "eye_motion"
    assert header[SCHEMA_VERSION_CARD] == "1.0"
    assert header[MANIFEST_CARD] == 1
    assert header["frame_rate"] == 30.0
