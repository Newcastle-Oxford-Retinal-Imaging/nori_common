"""Raw acquisition contract models."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from nori_common.contracts.model_base import ContractBase
from nori_common.contracts.pixel_scale import PixelScaleMetadata
from nori_common.contracts.provenance import ArtifactProvenance


class RawAcquisitionMetadata(ContractBase):
    """Metadata contract for a raw acquisition artifact."""

    schema_name: Literal["raw"] = "raw"
    schema_version: str = "1.0"
    metadata_version: int = 2

    provenance: ArtifactProvenance = ArtifactProvenance()
    system_id: str | None = None
    profile_id: str | None = None
    wiring_id: str | None = None

    channel_list: tuple[int, ...]
    slow_channel: str
    fast_channel: str
    confocal_channel: str
    trigger_channel: str | None = None

    n_pixels_horizontal_float: float
    n_pixels_vertical: int
    n_lines_flyback: int | None = None
    galvo_direction: str | None = None

    time_interval: float
    down_sample_ratio: int = 1
    frame_rate_hz: float

    pixel_scale: PixelScaleMetadata | None = None

    @field_validator("channel_list")
    @classmethod
    def _valid_channel_list(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("channel_list must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("channel_list must not contain duplicates")
        return value

    @field_validator("n_pixels_vertical")
    @classmethod
    def _positive_vertical(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_pixels_vertical must be positive")
        return value

    @field_validator("n_pixels_horizontal_float")
    @classmethod
    def _positive_horizontal(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("n_pixels_horizontal_float must be positive")
        return value

    @field_validator("time_interval")
    @classmethod
    def _positive_time_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("time_interval must be positive")
        return value

    @field_validator("down_sample_ratio")
    @classmethod
    def _valid_down_sample_ratio(cls, value: int) -> int:
        if value < 1:
            raise ValueError("down_sample_ratio must be >= 1")
        return value

    @field_validator("frame_rate_hz")
    @classmethod
    def _positive_frame_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("frame_rate_hz must be positive")
        return value
