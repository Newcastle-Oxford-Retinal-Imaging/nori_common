"""Manifest and sidecar mechanics for NORI artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from nori_common.contracts.manifest import Manifest
from nori_common.fits import Extension, FitsFile

MANIFEST_EXTNAME = "MANIFEST"
SCHEMA_CARD = "NORIFMT_SCHEMA"
SCHEMA_VERSION_CARD = "NORIFMT_SCHEMA_VERSION"
MANIFEST_CARD = "NORIFMT_MANIFEST"


def manifest_to_bytes(manifest: Manifest) -> np.ndarray:
    """Encode a manifest as a uint8 JSON byte array."""
    payload = manifest.model_dump_json()
    return np.frombuffer(payload.encode("utf-8"), dtype=np.uint8).copy()


def manifest_from_bytes(data: np.ndarray) -> Manifest:
    """Decode a manifest from a uint8 JSON byte array."""
    payload = np.asarray(data, dtype=np.uint8).tobytes().decode("utf-8")
    return Manifest.model_validate(json.loads(payload))


def manifest_extension(manifest: Manifest) -> Extension:
    """Return a FITS extension carrying the manifest."""
    return Extension(manifest_to_bytes(manifest), MANIFEST_EXTNAME)


def read_manifest(path: str | Path) -> Manifest:
    """Read a manifest from a FITS file's MANIFEST extension."""
    with FitsFile.open(path) as fits_file:
        data = fits_file.array(MANIFEST_EXTNAME)
    return manifest_from_bytes(data)


def apply_manifest_markers(header: fits.Header, manifest: Manifest) -> None:
    """Write schema markers into a FITS header."""
    header[f"HIERARCH {SCHEMA_CARD}"] = manifest.schema_name
    header[f"HIERARCH {SCHEMA_VERSION_CARD}"] = manifest.schema_version
    header[f"HIERARCH {MANIFEST_CARD}"] = 1


def write_sidecar(path: str | Path, manifest: Manifest) -> None:
    """Write a human-readable YAML sidecar projection of a manifest."""
    payload = manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
    with open(path, "w", encoding="utf-8") as sidecar_file:
        yaml.safe_dump(payload, sidecar_file, sort_keys=False)


def read_sidecar(path: str | Path) -> Manifest:
    """Read a manifest projection from a YAML sidecar."""
    with open(path, "r", encoding="utf-8") as sidecar_file:
        payload = yaml.safe_load(sidecar_file)
    if not isinstance(payload, dict):
        raise ValueError("sidecar must contain a mapping")
    return Manifest.model_validate(payload)


__all__ = [
    "MANIFEST_CARD",
    "MANIFEST_EXTNAME",
    "SCHEMA_CARD",
    "SCHEMA_VERSION_CARD",
    "apply_manifest_markers",
    "manifest_extension",
    "manifest_from_bytes",
    "manifest_to_bytes",
    "read_manifest",
    "read_sidecar",
    "write_sidecar",
]
