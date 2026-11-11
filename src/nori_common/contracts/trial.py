"""Trial contract models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from nori_common.contracts.manifest import ArrayDescriptor
from nori_common.contracts.model_base import ContractBase
from nori_common.contracts.photodiode import PhotodiodeTransitionsData
from nori_common.contracts.pixel_scale import PixelScaleMetadata
from nori_common.contracts.provenance import ArtifactProvenance


class FrameMetadata(ContractBase):
    """Required frame/timing metadata for a processed trial."""

    n_frames: int
    n_samples_original: int
    time_interval: float
    n_pixels_vertical: int
    n_pixels_horizontal_int: int
    scanner_offset_x: float
    scanner_offset_y: float
    non_flyback_start: int = Field(alias="Non_flyback_start")
    non_flyback_end: int = Field(alias="Non_flyback_end")

    @field_validator("n_pixels_vertical", "n_pixels_horizontal_int")
    @classmethod
    def _positive_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("pixel dimensions must be positive")
        return value

    @field_validator("time_interval")
    @classmethod
    def _positive_time_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("time_interval must be positive")
        return value

    @model_validator(mode="after")
    def _valid_non_flyback_range(self) -> FrameMetadata:
        if self.non_flyback_start < 0:
            raise ValueError("non_flyback_start must be non-negative")
        if self.non_flyback_end < self.non_flyback_start:
            raise ValueError("non_flyback_end must be >= non_flyback_start")
        if self.non_flyback_end >= self.n_pixels_vertical:
            raise ValueError("non_flyback_end must be < n_pixels_vertical")
        return self

    def to_header(self) -> dict[str, object]:
        """Return metadata using FITS card aliases."""
        return self.model_dump(by_alias=True, exclude_none=True)

    @classmethod
    def from_header(cls, header: dict[str, object]) -> FrameMetadata:
        """Build metadata from a FITS-like header mapping."""
        return cls.model_validate(header)


class TrialMetadata(ContractBase):
    """Metadata contract for a processed trial artifact."""

    schema_name: Literal["trial"] = "trial"
    schema_version: str = "1.0"
    provenance: ArtifactProvenance = ArtifactProvenance()
    frame: FrameMetadata
    arrays: tuple[ArrayDescriptor, ...] = ()
    pixel_scale: PixelScaleMetadata | None = None
    photodiode: PhotodiodeTransitionsData | None = None

    @model_validator(mode="after")
    def _unique_array_names(self) -> TrialMetadata:
        names = [array.name for array in self.arrays]
        if len(names) != len(set(names)):
            raise ValueError("array names must be unique")
        return self
