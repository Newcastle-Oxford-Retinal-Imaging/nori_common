"""Analysis-facing data access protocols.

These protocols describe what consumers need in analysis terms, not FITS terms.
Adapters implement them for specific containers and layouts.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np

T = TypeVar("T")


class Provenance(Protocol):
    """Where an artifact came from and what produced it."""

    @property
    def source_type(self) -> str:
        """'measured', 'simulated', or 'derived'."""

    @property
    def creator(self) -> str:
        """Free-form creator name."""

    @property
    def creator_version(self) -> str | None:
        """Free-form creator version, if known."""

    @property
    def parent_ids(self) -> tuple[str, ...]:
        """Optional lineage: identifiers of parent artifacts."""


class Frames(Protocol):
    """A sequence of image frames."""

    @property
    def frames(self) -> np.ndarray:
        """Frame data, shape (channels, frames, y, x, direction)."""


class FrameShape(Protocol):
    """Frame geometry without loading pixel data."""

    @property
    def n_frames(self) -> int:
        """Number of frames."""

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(y, x) frame shape."""


class ScanGeometry(Protocol):
    """Raster scan geometry for a trial."""

    @property
    def non_flyback_rows(self) -> tuple[int, int]:
        """Inclusive (first, last) valid scan rows."""

    @property
    def scanner_offset_pixels(self) -> tuple[float, float]:
        """Scanner offset in pixels, ordered (y, x)."""


class FrameRate(Protocol):
    """Frame rate associated with a trial."""

    @property
    def frame_rate_hz(self) -> float:
        """Frame rate in hertz."""


class PhotodiodeTransitions(Protocol):
    """Optional photodiode transition metadata."""

    @property
    def values(self) -> dict[int, float]:
        """Photodiode transition values, keyed by transition number."""

    @property
    def transitions(self) -> dict[int, float]:
        """Photodiode transition times, keyed by transition number."""


class EyeMotionPerPixel(Protocol):
    """Per-pixel eye motion for a trial."""

    @property
    def positions(self) -> np.ndarray:
        """Positions, shape (frames, y, x, 2)."""


class EyeMotionPerRow(Protocol):
    """Per-row eye motion for a trial."""

    @property
    def positions(self) -> np.ndarray:
        """Positions, shape (frames, rows, 2)."""


class Metrics(Protocol):
    """Frame-quality metrics for a trial."""

    @property
    def values(self) -> np.ndarray:
        """Metric matrix, shape (frames, n_metrics)."""

    @property
    def ncc(self) -> np.ndarray:
        """Normalised cross-correlation per frame."""

    @property
    def blink(self) -> np.ndarray:
        """Blink metric per frame."""

    @property
    def std_dev(self) -> np.ndarray:
        """Standard deviation per frame."""


class FramePositions(Protocol):
    """Frame-position bounds for per-pixel eye motion."""

    @property
    def positions(self) -> np.ndarray:
        """Frame positions, shape (frames, 8)."""


class CoarseMotion(Protocol):
    """Coarse per-frame motion."""

    @property
    def positions(self) -> np.ndarray:
        """Coarse positions, shape (frames, 2)."""


class MontageMotion(Protocol):
    """Montage motion field."""

    @property
    def motion(self) -> np.ndarray:
        """Montage motion, shape (y, x, 2)."""


class Timing(Protocol):
    """Timing and desinusoid metadata (optional capability)."""

    @property
    def n_frames(self) -> int: ...

    @property
    def n_samples_original(self) -> int: ...

    @property
    def time_interval_s(self) -> float: ...

    @property
    def fit_parameters(self) -> np.ndarray: ...

    @property
    def line_starts(self) -> np.ndarray: ...


class PixelScale(Protocol):
    """Pixel scale in arcminutes (optional capability)."""

    @property
    def pixel_size_arcmin_x(self) -> float: ...

    @property
    def pixel_size_arcmin_y(self) -> float: ...


class ChannelRoles(Protocol):
    """Mapping from channel row index to analysis role (optional)."""

    @property
    def channel_roles(self) -> dict[int, str]: ...


class MetadataExtras(Protocol):
    """Read-only access to unmodelled metadata."""

    def get_extra(self, key: str, default: object = None) -> object: ...

    def require_extra(self, key: str) -> object: ...

    @property
    def extras(self) -> dict[str, object]: ...


class Artifact(Protocol):
    """Base capability query interface for all data artifacts."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    def require(self, capability: type[T]) -> T: ...


#: Capability protocols known to the data access layer.
CAPABILITY_TYPES: tuple[type, ...] = (
    Provenance,
    Frames,
    FrameShape,
    ScanGeometry,
    FrameRate,
    Timing,
    PixelScale,
    ChannelRoles,
    EyeMotionPerPixel,
    EyeMotionPerRow,
    Metrics,
    FramePositions,
    CoarseMotion,
    MontageMotion,
    PhotodiodeTransitions,
)
