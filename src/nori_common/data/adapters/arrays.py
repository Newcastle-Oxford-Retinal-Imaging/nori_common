"""FITS adapters for simple analysis arrays.

These adapters expose typed capabilities over arrays stored in the first
data-bearing HDU of a FITS file. They validate rank and trailing-axis shape so
analysis code does not have to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nori_common.data import protocols
from nori_common.data.errors import CapabilityNotAvailable, SchemaError
from nori_common.fits import FitsFile


class FitsArrayAdapter:
    """Base adapter for a single analysis array in a FITS file."""

    product = "array"
    expected_ndim: int | None = None

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: np.ndarray | None = None

    def _validate(self, data: np.ndarray) -> None:
        if self.expected_ndim is not None and data.ndim != self.expected_ndim:
            raise SchemaError("array", self.product)

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            with FitsFile.open(self._path) as fits_file:
                data = np.asarray(fits_file.array())
            self._validate(data)
            self._data = data
        return self._data


class FitsEyeMotionAdapter(FitsArrayAdapter):
    """Per-pixel eye motion, shape (frames, y, x, 2)."""

    product = "eye_motion_per_pixel"
    expected_ndim = 4

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[-1] != 2:
            raise SchemaError("positions", self.product)

    @property
    def positions(self) -> np.ndarray:
        return self.data


class FitsEyeMotionPerRowAdapter(FitsArrayAdapter):
    """Per-row eye motion, shape (frames, rows, 2)."""

    product = "eye_motion_per_row"
    expected_ndim = 3

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[-1] != 2:
            raise SchemaError("positions", self.product)

    @property
    def positions(self) -> np.ndarray:
        return self.data


class FitsMetricsAdapter(FitsArrayAdapter):
    """Frame metrics, shape (frames, n_metrics) with at least two metrics."""

    product = "metrics"
    expected_ndim = 2

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[1] < 2:
            raise SchemaError("values", self.product)

    @property
    def values(self) -> np.ndarray:
        return self.data

    @property
    def ncc(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def blink(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def std_dev(self) -> np.ndarray:
        if self.data.shape[1] < 3:
            raise CapabilityNotAvailable(self, protocols.Metrics)
        return self.data[:, 2]


class FitsFramePositionsAdapter(FitsArrayAdapter):
    """Frame-position bounds, shape (frames, 8)."""

    product = "frame_positions"
    expected_ndim = 2

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[1] < 8:
            raise SchemaError("positions", self.product)

    @property
    def positions(self) -> np.ndarray:
        return self.data


class FitsCoarseMotionAdapter(FitsArrayAdapter):
    """Coarse per-frame motion, shape (frames, 2)."""

    product = "coarse_motion"
    expected_ndim = 2

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[1] < 2:
            raise SchemaError("positions", self.product)

    @property
    def positions(self) -> np.ndarray:
        return self.data


class FitsMontageMotionAdapter(FitsArrayAdapter):
    """Montage motion field, shape (y, x, 2)."""

    product = "montage_motion"
    expected_ndim = 3

    def _validate(self, data: np.ndarray) -> None:
        super()._validate(data)
        if data.shape[-1] != 2:
            raise SchemaError("motion", self.product)

    @property
    def motion(self) -> np.ndarray:
        return self.data


def read_eye_motion(path: str | Path) -> FitsEyeMotionAdapter:
    """Open per-pixel eye motion for a trial."""
    return FitsEyeMotionAdapter(path)


def read_eye_motion_per_row(path: str | Path) -> FitsEyeMotionPerRowAdapter:
    """Open per-row eye motion for a trial."""
    return FitsEyeMotionPerRowAdapter(path)


def read_metrics(path: str | Path) -> FitsMetricsAdapter:
    """Open metrics for a trial."""
    return FitsMetricsAdapter(path)


def read_frame_positions(path: str | Path) -> FitsFramePositionsAdapter:
    """Open frame positions for a trial."""
    return FitsFramePositionsAdapter(path)


def read_coarse_motion(path: str | Path) -> FitsCoarseMotionAdapter:
    """Open coarse motion for a trial."""
    return FitsCoarseMotionAdapter(path)


def read_montage_motion(path: str | Path) -> FitsMontageMotionAdapter:
    """Open montage motion for a participant or session."""
    return FitsMontageMotionAdapter(path)
