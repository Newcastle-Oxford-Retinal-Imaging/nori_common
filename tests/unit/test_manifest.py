"""Tests for manifest and sidecar mechanics."""

import numpy
from astropy.io import fits

from nori_common.contracts import (
    ArrayDescriptor,
    ArtifactProvenance,
    Manifest,
    PixelScaleMetadata,
)
from nori_common.fits import Extension, FitsFile, WritePolicy, write_fits_multi
from nori_common.manifest import (
    MANIFEST_CARD,
    MANIFEST_EXTNAME,
    SCHEMA_CARD,
    SCHEMA_VERSION_CARD,
    apply_manifest_markers,
    manifest_extension,
    manifest_from_bytes,
    manifest_to_bytes,
    read_manifest,
    read_sidecar,
    write_sidecar,
)


def _manifest():
    return Manifest(
        schema_name="trial",
        schema_version="1.0",
        arrays=(ArrayDescriptor(name="PRIMARY", role="frames"),),
        metadata={"n_frames": 2},
        provenance=ArtifactProvenance(
            source_type="simulated",
            creator="ERICA",
        ),
        pixel_scale=PixelScaleMetadata(
            pixel_size_arcmin_x=0.1,
            pixel_size_arcmin_y=0.2,
        ),
    )


def test_manifest_byte_roundtrip():
    manifest = _manifest()

    encoded = manifest_to_bytes(manifest)
    decoded = manifest_from_bytes(encoded)

    assert decoded == manifest


def test_manifest_fits_roundtrip(tmp_path):
    path = tmp_path / "trial.fits"
    manifest = _manifest()
    header = fits.Header()
    apply_manifest_markers(header, manifest)

    write_fits_multi(
        path,
        [
            Extension(numpy.zeros((1, 2, 3, 4, 2)), "PRIMARY", header=header),
            manifest_extension(manifest),
        ],
        policy=WritePolicy(),
    )

    assert read_manifest(path) == manifest

    with FitsFile.open(path) as fits_file:
        header_out = fits_file.header()
    assert header_out[SCHEMA_CARD] == "trial"
    assert header_out[SCHEMA_VERSION_CARD] == "1.0"
    assert header_out[MANIFEST_CARD] == 1


def test_manifest_extension_name():
    extension = manifest_extension(_manifest())

    assert extension.name == MANIFEST_EXTNAME
    assert extension.data.dtype == numpy.uint8


def test_sidecar_roundtrip(tmp_path):
    path = tmp_path / "trial.yaml"
    manifest = _manifest()

    write_sidecar(path, manifest)

    assert read_sidecar(path) == manifest
