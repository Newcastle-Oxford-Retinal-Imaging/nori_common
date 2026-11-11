"""In-memory reference implementation of the trial capabilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from nori_common.data import protocols
from nori_common.data.base import ArtifactBase
from nori_common.data.errors import CapabilityNotAvailable
from nori_common.data.extras import Extras, PixelScaleData, ProvenanceData


class InMemoryTrial(ArtifactBase):
    """A trial backed entirely by in-memory data.

    This is the reference implementation used by contract tests and, later, by
    in-memory pipeline composition. It provides only the capabilities it was
    given.
    """

    def __init__(
        self,
        frames: np.ndarray,
        non_flyback_rows: tuple[int, int],
        *,
        frame_rate_hz: float | None = None,
        provenance: ProvenanceData | None = None,
        extras: Mapping[str, object] | None = None,
        timing: object | None = None,
        pixel_size_arcmin: tuple[float, float] | None = None,
        channel_roles: Mapping[int, str] | None = None,
        scanner_offset_pixels: tuple[float, float] | None = None,
    ) -> None:
        self._frames = np.asarray(frames)
        self._non_flyback_rows = (int(non_flyback_rows[0]), int(non_flyback_rows[1]))
        self._frame_rate_hz = (
            float(frame_rate_hz) if frame_rate_hz is not None else None
        )
        self._provenance = provenance or ProvenanceData()
        self._extras = Extras(extras)
        self._timing = timing
        self._pixel_scale = (
            PixelScaleData(
                pixel_size_arcmin_x=float(pixel_size_arcmin[0]),
                pixel_size_arcmin_y=float(pixel_size_arcmin[1]),
            )
            if pixel_size_arcmin is not None
            else None
        )
        self._channel_roles = dict(channel_roles) if channel_roles is not None else None
        self._scanner_offset_pixels = (
            (float(scanner_offset_pixels[0]), float(scanner_offset_pixels[1]))
            if scanner_offset_pixels is not None
            else None
        )

    def _provided_capabilities(self) -> frozenset[type]:
        capabilities: set[type] = {
            protocols.Provenance,
            protocols.Frames,
            protocols.FrameShape,
            protocols.ScanGeometry,
        }
        if self._frame_rate_hz is not None:
            capabilities.add(protocols.FrameRate)
        if self._timing is not None:
            capabilities.add(protocols.Timing)
        if self._pixel_scale is not None:
            capabilities.add(protocols.PixelScale)
        if self._channel_roles is not None:
            capabilities.add(protocols.ChannelRoles)
        return frozenset(capabilities)

    @property
    def frames(self) -> np.ndarray:
        return self._frames

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[1])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self._frames.shape[2]), int(self._frames.shape[3])

    @property
    def non_flyback_rows(self) -> tuple[int, int]:
        return self._non_flyback_rows

    @property
    def scanner_offset_pixels(self) -> tuple[float, float]:
        if self._scanner_offset_pixels is None:
            raise CapabilityNotAvailable(self, protocols.ScanGeometry)
        return self._scanner_offset_pixels

    @property
    def frame_rate_hz(self) -> float:
        if self._frame_rate_hz is None:
            raise CapabilityNotAvailable(self, protocols.FrameRate)
        return self._frame_rate_hz

    @property
    def provenance(self) -> ProvenanceData:
        return self._provenance

    @property
    def extras(self) -> Extras:
        return self._extras

    @property
    def timing(self) -> object:
        if self._timing is None:
            raise CapabilityNotAvailable(self, protocols.Timing)
        return self._timing

    @property
    def pixel_scale(self) -> object:
        if self._pixel_scale is None:
            raise CapabilityNotAvailable(self, protocols.PixelScale)
        return self._pixel_scale

    @property
    def channel_roles(self) -> dict[int, str]:
        if self._channel_roles is None:
            raise CapabilityNotAvailable(self, protocols.ChannelRoles)
        return dict(self._channel_roles)
