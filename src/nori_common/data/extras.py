"""Concrete metadata helpers for the data access layer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from nori_common.data.errors import MetadataNotFound


@dataclass(frozen=True)
class ProvenanceData:
    """Concrete implementation of the Provenance protocol."""

    source_type: str = "unknown"
    creator: str = "unknown"
    creator_version: str | None = None
    parent_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PixelScaleData:
    """Concrete implementation of the PixelScale protocol."""

    pixel_size_arcmin_x: float
    pixel_size_arcmin_y: float


class Extras(Mapping[str, object]):
    """Read-only, namespaced extra metadata mapping."""

    def __init__(self, data: Mapping[str, object] | None = None) -> None:
        self._data: dict[str, object] = dict(data or {})

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_extra(self, key: str, default: object = None) -> object:
        """Return an extra metadata value, or ``default`` if absent."""
        return self._data.get(key, default)

    def require_extra(self, key: str) -> object:
        """Return an extra metadata value, raising if absent."""
        try:
            return self._data[key]
        except KeyError as exc:
            raise MetadataNotFound(key) from exc

    @property
    def extras(self) -> dict[str, object]:
        """Return an independent copy of the extras mapping."""
        return dict(self._data)
