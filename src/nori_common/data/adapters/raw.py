"""Raw acquisition adapters and capability objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from nori_common.data import protocols
from nori_common.data.base import ArtifactBase
from nori_common.data.errors import CapabilityNotAvailable, SchemaError
from nori_common.data.extras import Extras, PixelScaleData, ProvenanceData


def _required(metadata: Mapping[str, Any], key: str) -> Any:
    if key not in metadata:
        raise SchemaError(key, "raw acquisition")
    return metadata[key]


def _letter_to_index(letter: str, channel_list: list[int]) -> int:
    scope_index = ord(str(letter).upper()) - ord("A")
    try:
        return list(channel_list).index(scope_index)
    except ValueError as exc:
        raise SchemaError("channel_list", "raw acquisition") from exc


def _channel_list(metadata: Mapping[str, Any]) -> list[int]:
    values = _required(metadata, "channel_list")
    return [int(value) for value in values]


def _channel_roles(
    metadata: Mapping[str, Any],
    channel_list: list[int],
) -> dict[int, str]:
    role_by_label = {
        str(_required(metadata, "slow_channel")).upper(): "slow_scan",
        str(_required(metadata, "fast_channel")).upper(): "fast_scan",
    }
    confocal = metadata.get("confocal_channel") or metadata.get("confocal_channel1")
    if confocal is None:
        raise SchemaError("confocal_channel", "raw acquisition")
    role_by_label[str(confocal).upper()] = "confocal"

    photodiode = metadata.get("photodiode_channel") or metadata.get("trigger_channel")
    if photodiode is not None:
        role_by_label[str(photodiode).upper()] = "photodiode"

    roles: dict[int, str] = {}
    for row, scope_index in enumerate(channel_list):
        label = chr(ord("A") + int(scope_index))
        roles[row] = role_by_label.get(label, "extra")
    return roles


def _frame_rate(metadata: Mapping[str, Any], n_pixels_vertical: int) -> float:
    value = metadata.get("AOSLO_frame_rate")
    if value is not None:
        return float(value)

    frequency = metadata.get("resonant_frequency_pico")
    if frequency is None:
        frequency = metadata.get("resonant_frequency")
    if frequency is None:
        raise SchemaError("AOSLO_frame_rate", "raw acquisition")
    return float(frequency) / float(n_pixels_vertical)


def _pixel_scale(metadata: Mapping[str, Any]) -> PixelScaleData | None:
    x = metadata.get("Pixel_size_x_arcmin")
    y = metadata.get("Pixel_size_y_arcmin")
    if x is None or y is None:
        return None
    return PixelScaleData(pixel_size_arcmin_x=float(x), pixel_size_arcmin_y=float(y))


class RawAcquisitionData(ArtifactBase):
    """Concrete raw acquisition capability object."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        channel_roles: Mapping[int, str],
        slow_channel_index: int,
        fast_channel_index: int,
        confocal_channel_index: int,
        photodiode_channel_index: int | None,
        n_pixels_horizontal_float: float,
        n_pixels_vertical: int,
        n_lines_flyback: int | None,
        galvo_direction: str | None,
        sample_interval_s: float,
        down_sample_ratio: int,
        frame_rate_hz: float,
        source_metadata: Mapping[str, Any],
        metadata_version: int | None = None,
        system_id: str | None = None,
        profile_id: str | None = None,
        wiring_id: str | None = None,
        source_type: str = "measured",
        creator: str = "unknown",
        pixel_scale: PixelScaleData | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self._channel_roles = dict(channel_roles)
        self._slow_channel_index = int(slow_channel_index)
        self._fast_channel_index = int(fast_channel_index)
        self._confocal_channel_index = int(confocal_channel_index)
        self._photodiode_channel_index = (
            int(photodiode_channel_index)
            if photodiode_channel_index is not None
            else None
        )
        self._n_pixels_horizontal_float = float(n_pixels_horizontal_float)
        self._n_pixels_vertical = int(n_pixels_vertical)
        self._n_lines_flyback = (
            int(n_lines_flyback) if n_lines_flyback is not None else None
        )
        self._galvo_direction = galvo_direction
        self._sample_interval_s = float(sample_interval_s)
        self._down_sample_ratio = int(down_sample_ratio)
        self._frame_rate_hz = float(frame_rate_hz)
        self._source_metadata = dict(source_metadata)
        self._metadata_version = metadata_version
        self._system_id = system_id
        self._profile_id = profile_id
        self._wiring_id = wiring_id
        self._source_type = source_type
        self._creator = creator
        self._pixel_scale = pixel_scale
        self._extras = Extras(
            {f"acq.{key}": value for key, value in source_metadata.items()}
        )

    def _provided_capabilities(self) -> frozenset[type]:
        capabilities: set[type] = {
            protocols.RawData,
            protocols.RawChannelRoles,
            protocols.RawScanGeometry,
            protocols.AcquisitionTiming,
            protocols.AcquisitionProvenance,
        }
        if self._pixel_scale is not None:
            capabilities.add(protocols.PixelScale)
        return frozenset(capabilities)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def channel_roles(self) -> dict[int, str]:
        return dict(self._channel_roles)

    @property
    def slow_channel_index(self) -> int:
        return self._slow_channel_index

    @property
    def fast_channel_index(self) -> int:
        return self._fast_channel_index

    @property
    def confocal_channel_index(self) -> int:
        return self._confocal_channel_index

    @property
    def photodiode_channel_index(self) -> int | None:
        return self._photodiode_channel_index

    @property
    def n_pixels_horizontal_float(self) -> float:
        return self._n_pixels_horizontal_float

    @property
    def n_pixels_vertical(self) -> int:
        return self._n_pixels_vertical

    @property
    def n_lines_flyback(self) -> int | None:
        return self._n_lines_flyback

    @property
    def galvo_direction(self) -> str | None:
        return self._galvo_direction

    @property
    def sample_interval_s(self) -> float:
        return self._sample_interval_s

    @property
    def down_sample_ratio(self) -> int:
        return self._down_sample_ratio

    @property
    def frame_rate_hz(self) -> float:
        return self._frame_rate_hz

    @property
    def n_samples_total(self) -> int:
        return int(self._data.shape[1])

    @property
    def metadata_version(self) -> int | None:
        return self._metadata_version

    @property
    def system_id(self) -> str | None:
        return self._system_id

    @property
    def profile_id(self) -> str | None:
        return self._profile_id

    @property
    def wiring_id(self) -> str | None:
        return self._wiring_id

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def creator(self) -> str:
        return self._creator

    @property
    def pixel_size_arcmin_x(self) -> float:
        if self._pixel_scale is None:
            raise CapabilityNotAvailable(self, protocols.PixelScale)
        return self._pixel_scale.pixel_size_arcmin_x

    @property
    def pixel_size_arcmin_y(self) -> float:
        if self._pixel_scale is None:
            raise CapabilityNotAvailable(self, protocols.PixelScale)
        return self._pixel_scale.pixel_size_arcmin_y

    @property
    def provenance(self) -> ProvenanceData:
        return ProvenanceData(
            source_type=self._source_type,
            creator=self._creator,
        )

    @property
    def extras(self) -> Extras:
        return self._extras

    @property
    def source_metadata(self) -> dict[str, Any]:
        """Raw metadata mapping for legacy writer compatibility."""
        return dict(self._source_metadata)


def from_metadata(data: np.ndarray, metadata: Mapping[str, Any]) -> RawAcquisitionData:
    """Build a raw acquisition from data and resolved metadata."""
    if data.ndim != 2:
        raise SchemaError("data", "raw acquisition")

    channel_list = _channel_list(metadata)
    if len(channel_list) != data.shape[0]:
        raise SchemaError("channel_list", "raw acquisition")

    slow_label = str(_required(metadata, "slow_channel")).upper()
    fast_label = str(_required(metadata, "fast_channel")).upper()
    confocal_label = metadata.get("confocal_channel") or metadata.get(
        "confocal_channel1"
    )
    if confocal_label is None:
        raise SchemaError("confocal_channel", "raw acquisition")
    confocal_label = str(confocal_label).upper()

    photodiode_label = metadata.get("photodiode_channel") or metadata.get(
        "trigger_channel"
    )
    photodiode_index = (
        _letter_to_index(str(photodiode_label), channel_list)
        if photodiode_label is not None
        else None
    )

    n_pixels_vertical = int(_required(metadata, "n_pixels_vertical"))
    n_pixels_horizontal = metadata.get("n_pixels_horizontal_float")
    if n_pixels_horizontal is None:
        n_pixels_horizontal = metadata.get("n_pixels_horizontal")
    if n_pixels_horizontal is None:
        raise SchemaError("n_pixels_horizontal_float", "raw acquisition")

    version = metadata.get("metadata_version")
    version_int = int(version) if version is not None else None
    wiring = metadata.get("wiring") or {}

    return RawAcquisitionData(
        data,
        channel_roles=_channel_roles(metadata, channel_list),
        slow_channel_index=_letter_to_index(slow_label, channel_list),
        fast_channel_index=_letter_to_index(fast_label, channel_list),
        confocal_channel_index=_letter_to_index(confocal_label, channel_list),
        photodiode_channel_index=photodiode_index,
        n_pixels_horizontal_float=float(n_pixels_horizontal),
        n_pixels_vertical=n_pixels_vertical,
        n_lines_flyback=metadata.get("n_lines_flyback"),
        galvo_direction=metadata.get("galvo_direction"),
        sample_interval_s=float(_required(metadata, "time_interval")),
        down_sample_ratio=int(metadata.get("down_sample_ratio", 1)),
        frame_rate_hz=_frame_rate(metadata, n_pixels_vertical),
        source_metadata=metadata,
        metadata_version=version_int,
        system_id=metadata.get("system") or wiring.get("system"),
        profile_id=metadata.get("profile"),
        wiring_id=metadata.get("wiring_name") or wiring.get("name"),
        source_type=str(metadata.get("source_type", "measured")),
        creator=str(metadata.get("creator", "unknown")),
        pixel_scale=_pixel_scale(metadata),
    )


def read_raw_acquisition(
    data: np.ndarray, metadata: Mapping[str, Any]
) -> RawAcquisitionData:
    """Build a raw acquisition from data and resolved metadata.

    ``metadata_version`` 2 is currently treated as the only explicitly
    versioned acquisition metadata layout; absent version means legacy flat
    metadata. Unknown versions fail loudly.
    """
    version = metadata.get("metadata_version")
    if version is not None and int(version) != 2:
        raise SchemaError("metadata_version", "raw acquisition")
    return from_metadata(data, metadata)
