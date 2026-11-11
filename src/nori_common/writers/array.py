"""Schema-aware writer for single-array artifacts."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from nori_common.builders.array import ArrayPayload
from nori_common.fits import Extension, WritePolicy, write_fits_multi
from nori_common.manifest import (
    apply_manifest_markers,
    manifest_extension,
    write_sidecar,
)


def _set_header_card(header: fits.Header, key: str, value: object) -> None:
    if key.isupper() and len(key) <= 8:
        header[key] = value
    else:
        header[f"HIERARCH {key}"] = value


def write_array_artifact(
    path: str | Path,
    payload: ArrayPayload,
    *,
    policy: WritePolicy | None = None,
    sidecar_path: str | Path | None = None,
) -> None:
    """Write a single-array artifact with manifest, markers, and sidecar."""
    header = fits.Header()
    for key, value in payload.header.items():
        _set_header_card(header, str(key), value)
    apply_manifest_markers(header, payload.manifest)

    write_fits_multi(
        Path(path),
        [
            Extension(payload.data, "PRIMARY", header),
            manifest_extension(payload.manifest),
        ],
        policy=policy,
    )

    if sidecar_path is not None:
        write_sidecar(sidecar_path, payload.manifest)
