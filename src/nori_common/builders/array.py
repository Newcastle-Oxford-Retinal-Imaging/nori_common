"""Builder for validated single-array artifact payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from nori_common.builders.axes import array_spec, build_axis_descriptors
from nori_common.contracts import (
    ArrayDescriptor,
    ArtifactProvenance,
    Manifest,
    PixelScaleMetadata,
)
from nori_common.contracts.errors import SchemaError


@dataclass(frozen=True)
class ArrayPayload:
    """Array data and manifest for a single-array artifact."""

    data: np.ndarray
    manifest: Manifest
    header: Mapping[str, object] = field(default_factory=dict)


def build_array_payload(
    data: np.ndarray,
    *,
    schema_name: str,
    role: str,
    schema_version: str = "1.0",
    axes: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
    provenance: ArtifactProvenance | None = None,
    capabilities: Mapping[str, bool] | None = None,
    pixel_scale: PixelScaleMetadata | None = None,
    header: Mapping[str, object] | None = None,
    array_name: str = "PRIMARY",
) -> ArrayPayload:
    """Build a validated single-array payload and manifest."""
    data = np.asarray(data)
    if data.ndim == 0:
        raise SchemaError("data", schema_name)

    metadata_dict = dict(metadata or {})
    axis_descriptors = build_axis_descriptors(
        data,
        schema_name=schema_name,
        role=role,
        metadata=metadata_dict,
        axes_names=axes,
    )
    spec = array_spec(schema_name, role)

    manifest = Manifest(
        schema_name=schema_name,
        schema_version=schema_version,
        arrays=(
            ArrayDescriptor(
                name=array_name,
                role=role,
                dtype=str(data.dtype),
                shape=tuple(int(size) for size in data.shape),
                axes=axis_descriptors,
                unit=spec.unit if spec is not None else None,
            ),
        ),
        metadata=metadata_dict,
        capabilities=dict(capabilities or {}),
        provenance=provenance or ArtifactProvenance(),
        pixel_scale=pixel_scale,
    )
    return ArrayPayload(data=data, manifest=manifest, header=dict(header or {}))
