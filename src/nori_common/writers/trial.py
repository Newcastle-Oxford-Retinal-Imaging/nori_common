"""Schema-aware writers for processed trials."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from nori_common.builders.trial import TrialPayload
from nori_common.contracts import Manifest
from nori_common.fits import Extension, WritePolicy, write_fits_multi
from nori_common.manifest import (
    apply_manifest_markers,
    manifest_extension,
    write_sidecar,
)
from nori_common.validation import validate_trial_arrays


def _set_header_card(header: fits.Header, key: str, value: object) -> None:
    if key.isupper() and len(key) <= 8:
        header[key] = value
    else:
        header[f"HIERARCH {key}"] = value


def _trial_manifest(payload: TrialPayload) -> Manifest:
    metadata = payload.metadata
    metadata_dict: dict[str, object] = metadata.frame.to_header()
    if metadata.channel_roles:
        metadata_dict["channel_roles"] = {
            str(index): role for index, role in metadata.channel_roles.items()
        }
    if metadata.photodiode is not None:
        metadata_dict["photodiode"] = metadata.photodiode.model_dump(mode="json")
    return Manifest(
        schema_name="trial",
        schema_version=metadata.schema_version,
        arrays=metadata.arrays,
        metadata=metadata_dict,
        capabilities={
            "timing": True,
            "photodiode": metadata.photodiode is not None,
        },
        provenance=metadata.provenance,
        pixel_scale=metadata.pixel_scale,
    )


def write_trial(
    path: str | Path,
    payload: TrialPayload,
    *,
    policy: WritePolicy | None = None,
    sidecar_path: str | Path | None = None,
) -> None:
    """Validate and write a processed trial with metadata and manifest."""
    validate_trial_arrays(
        payload.metadata,
        payload.frames,
        payload.fit_parameters,
        payload.line_starts,
    )

    manifest = _trial_manifest(payload)
    header = fits.Header()
    for key, value in payload.metadata.frame.to_header().items():
        _set_header_card(header, key, value)
    for index, role in sorted(payload.metadata.channel_roles.items()):
        _set_header_card(header, f"Channel_{index:03d}", role)
    if payload.metadata.photodiode is not None:
        for index, value in sorted(payload.metadata.photodiode.values.items()):
            _set_header_card(header, f"PD_value_{index:03d}", value)
        for index, value in sorted(payload.metadata.photodiode.transitions.items()):
            _set_header_card(header, f"PD_transition_{index:03d}", value)
    apply_manifest_markers(header, manifest)

    write_fits_multi(
        Path(path),
        [
            Extension(payload.frames, "PRIMARY", header),
            Extension(payload.fit_parameters, "FIT_PARAMS"),
            Extension(payload.line_starts, "LINE_STARTS"),
            manifest_extension(manifest),
        ],
        policy=policy,
    )

    if sidecar_path is not None:
        write_sidecar(sidecar_path, manifest)
