"""Manifest contract models."""

from __future__ import annotations

from typing import Any

from pydantic import field_validator, model_validator

from nori_common.contracts.model_base import ContractBase
from nori_common.contracts.pixel_scale import PixelScaleMetadata
from nori_common.contracts.provenance import ArtifactProvenance


class CoordinateSpec(ContractBase):
    """Linear coordinate description for one axis."""

    start: float = 0.0
    step: float = 1.0
    unit: str | None = None


class AxisDescriptor(ContractBase):
    """Description of one array axis."""

    name: str
    size: int
    labels: tuple[str, ...] | None = None
    unit: str | None = None
    coordinates: CoordinateSpec | None = None

    @field_validator("size")
    @classmethod
    def _positive_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("axis size must be positive")
        return value

    @model_validator(mode="after")
    def _labels_match_size(self) -> AxisDescriptor:
        if self.labels is None:
            return self
        if len(self.labels) != self.size:
            raise ValueError(
                f"axis {self.name!r} has {self.size} elements but "
                f"{len(self.labels)} labels"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"axis {self.name!r} labels must be unique")
        return self


class ArrayDescriptor(ContractBase):
    """Description of one array in an artifact."""

    name: str
    role: str
    dtype: str | None = None
    shape: tuple[int, ...]
    axes: tuple[AxisDescriptor, ...]
    unit: str | None = None
    required: bool = True

    @field_validator("shape")
    @classmethod
    def _valid_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("shape must not be empty")
        if any(dimension <= 0 for dimension in value):
            raise ValueError("shape dimensions must be positive")
        return value

    @model_validator(mode="after")
    def _axes_match_shape(self) -> ArrayDescriptor:
        if len(self.axes) != len(self.shape):
            raise ValueError(
                f"array {self.name!r} has {len(self.shape)} dimensions but "
                f"{len(self.axes)} axes"
            )
        for axis, size in zip(self.axes, self.shape, strict=True):
            if axis.size != size:
                raise ValueError(
                    f"axis {axis.name!r} size {axis.size} does not match "
                    f"array shape dimension {size}"
                )
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError(f"array {self.name!r} axis names must be unique")
        return self


class Manifest(ContractBase):
    """Schema and array manifest for a NORI artifact."""

    manifest_version: str = "1.0"
    schema_name: str
    schema_version: str = "1.0"
    arrays: tuple[ArrayDescriptor, ...] = ()
    metadata: dict[str, Any] = {}
    capabilities: dict[str, bool] = {}
    provenance: ArtifactProvenance = ArtifactProvenance()
    pixel_scale: PixelScaleMetadata | None = None
