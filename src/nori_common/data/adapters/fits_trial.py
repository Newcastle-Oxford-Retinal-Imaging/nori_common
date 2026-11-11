"""FITS adapter for processed trial data.

This adapter handles both current processed FITS files and legacy processed FITS
files for the capabilities needed by motion estimation:

- frames from the first data-bearing HDU,
- non-flyback row bounds from the header,
- an optional frame rate from an explicit value or the header.

The analysis-facing protocol does not expose FITS card names.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from nori_common.data import protocols
from nori_common.data.base import ArtifactBase
from nori_common.data.errors import CapabilityNotAvailable, SchemaError
from nori_common.data.extras import Extras, ProvenanceData
from nori_common.fits import FitsFile
from nori_common.fits.schema import (
    CARD_NON_FLYBACK_END,
    CARD_NON_FLYBACK_START,
)

_STRUCTURAL_CARDS = {
    "SIMPLE",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "NAXIS4",
    "NAXIS5",
    "EXTEND",
    "XTENSION",
    "PCOUNT",
    "GCOUNT",
    "EXTNAME",
}

_FRAME_RATE_CARDS = (
    "AOSLO_frame_rate",
    "AOSLO_FRAME_RATE",
    "FRAMERATE",
    "FRAME_RATE",
    "frame_rate",
)

_KNOWN_CARDS = {
    CARD_NON_FLYBACK_START.casefold(),
    CARD_NON_FLYBACK_END.casefold(),
    "source_type",
    "creator",
    "creator_version",
    "aoslo_frame_rate",
    "aoslo_frame_rate",
    "framerate",
    "frame_rate",
}


def _lookup(header: Mapping[str, object], key: str) -> object | None:
    """Case-insensitive header lookup for a plain mapping."""
    folded = key.casefold()
    for card, value in header.items():
        if card.casefold() == folded:
            return value
    return None


class FitsTrialAdapter(ArtifactBase):
    """Trial adapter over the current and legacy processed FITS layouts."""

    def __init__(
        self,
        path: str | Path,
        *,
        frame_rate: float | None = None,
    ) -> None:
        self._path = Path(path)
        with FitsFile.open(self._path) as fits_file:
            self._header: dict[str, object] = fits_file.header()
        self._frames: np.ndarray | None = None
        self._extras = Extras(self._build_extras())

        start = _lookup(self._header, CARD_NON_FLYBACK_START)
        end = _lookup(self._header, CARD_NON_FLYBACK_END)
        if start is None or end is None:
            missing = CARD_NON_FLYBACK_START if start is None else CARD_NON_FLYBACK_END
            raise SchemaError(missing, "trial")

        self._non_flyback_rows = (int(start), int(end))
        self._frame_rate_hz = frame_rate
        if self._frame_rate_hz is None:
            for card in _FRAME_RATE_CARDS:
                value = _lookup(self._header, card)
                if value is not None:
                    self._frame_rate_hz = float(value)
                    break

    def _build_extras(self) -> dict[str, object]:
        structural = {card.casefold() for card in _STRUCTURAL_CARDS}
        extras: dict[str, object] = {}
        for key, value in self._header.items():
            folded = key.casefold()
            if folded in structural or folded in _KNOWN_CARDS:
                continue
            extras[f"fits_header.{key}"] = value
        return extras

    def _provided_capabilities(self) -> frozenset[type]:
        capabilities: set[type] = {
            protocols.Provenance,
            protocols.Frames,
            protocols.ScanGeometry,
        }
        if self._frame_rate_hz is not None:
            capabilities.add(protocols.FrameRate)
        return frozenset(capabilities)

    @property
    def frames(self) -> np.ndarray:
        if self._frames is None:
            with FitsFile.open(self._path) as fits_file:
                frames = np.asarray(fits_file.array())
            if frames.ndim != 5:
                raise SchemaError("frames", "trial")
            self._frames = frames
        return self._frames

    @property
    def non_flyback_rows(self) -> tuple[int, int]:
        return self._non_flyback_rows

    @property
    def frame_rate_hz(self) -> float:
        if self._frame_rate_hz is None:
            raise CapabilityNotAvailable(self, protocols.FrameRate)
        return self._frame_rate_hz

    @property
    def provenance(self) -> ProvenanceData:
        source_type = _lookup(self._header, "SOURCE_TYPE") or "unknown"
        creator = _lookup(self._header, "CREATOR") or "unknown"
        creator_version = _lookup(self._header, "CREATOR_VERSION")
        return ProvenanceData(
            source_type=str(source_type),
            creator=str(creator),
            creator_version=(
                str(creator_version) if creator_version is not None else None
            ),
        )

    @property
    def extras(self) -> Extras:
        return self._extras
