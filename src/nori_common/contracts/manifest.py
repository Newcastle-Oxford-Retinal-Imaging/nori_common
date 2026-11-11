"""Manifest contract models."""

from __future__ import annotations

from typing import Any

from nori_common.contracts.model_base import ContractBase
from nori_common.contracts.provenance import ArtifactProvenance


class ArrayDescriptor(ContractBase):
    """Description of one array in an artifact."""

    name: str
    role: str
    dtype: str | None = None
    axes: tuple[str, ...] = ()
    unit: str | None = None
    required: bool = True


class Manifest(ContractBase):
    """Schema and array manifest for a NORI artifact."""

    manifest_version: str = "1.0"
    schema_name: str
    schema_version: str = "1.0"
    arrays: tuple[ArrayDescriptor, ...] = ()
    metadata: dict[str, Any] = {}
    capabilities: dict[str, bool] = {}
    provenance: ArtifactProvenance = ArtifactProvenance()
