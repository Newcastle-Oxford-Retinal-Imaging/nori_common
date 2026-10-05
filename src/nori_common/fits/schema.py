"""The ``norifmt`` on-disk contract shared with downstream consumers.

Constants name the logical FITS keywords; astropy emits long or non-standard
keywords as ``HIERARCH`` cards automatically, so writers should assign the
plain keyword (for example ``header["n_frames"]``) and let astropy handle the
card form.

Header values are stored as FITS card text: floating-point metadata round-trips
to roughly 15 significant digits, and integer metadata round-trips exactly.

The array conventions are part of the contract:

* ``frames`` is indexed ``(frame, y, x, ...)`` with the same axis order the
  processing pipeline produces.
* ``fit_parameters`` row 0 holds the slow-axis parameters and row 1 the
  fast-axis parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nori_common.exceptions import IncompatibleFormatError
from nori_common.fits import Extension, FitsFile

__all__ = [
    "CARD_CHANNEL_PREFIX",
    "CARD_COLS",
    "CARD_FRAME_RATE",
    "CARD_NON_FLYBACK_END",
    "CARD_NON_FLYBACK_START",
    "CARD_N_FRAMES",
    "CARD_N_PIXELS_HORIZONTAL_INT",
    "CARD_N_PIXELS_VERTICAL",
    "CARD_N_SAMPLES_ORIGINAL",
    "CARD_PD_TRANSITION_PREFIX",
    "CARD_PD_VALUE_PREFIX",
    "CARD_PIXEL_ARCMIN_X",
    "CARD_PIXEL_ARCMIN_Y",
    "CARD_SCANNER_OFFSET_X",
    "CARD_SCANNER_OFFSET_Y",
    "CARD_TIME_INTERVAL",
    "CARD_UNITS",
    "EXT_FIT_PARAMS",
    "EXT_FRAMES",
    "EXT_LINE_STARTS",
    "FORMAT_VERSION",
    "FrameMetadata",
    "TrialArrays",
]

FORMAT_VERSION = 1

#: Extension names.
EXT_FRAMES = "PRIMARY"
EXT_FIT_PARAMS = "FIT_PARAMS"
EXT_LINE_STARTS = "LINE_STARTS"

#: Processed-trial card names.
CARD_N_FRAMES = "n_frames"
CARD_N_SAMPLES_ORIGINAL = "n_samples_original"
CARD_TIME_INTERVAL = "time_interval"
CARD_N_PIXELS_VERTICAL = "n_pixels_vertical"
CARD_N_PIXELS_HORIZONTAL_INT = "n_pixels_horizontal_int"
CARD_SCANNER_OFFSET_X = "scanner_offset_x"
CARD_SCANNER_OFFSET_Y = "scanner_offset_y"
CARD_NON_FLYBACK_START = "Non_flyback_start"
CARD_NON_FLYBACK_END = "Non_flyback_end"
CARD_CHANNEL_PREFIX = "Channel_"
CARD_PD_TRANSITION_PREFIX = "PD_transition_"
CARD_PD_VALUE_PREFIX = "PD_value_"

#: Trace card names.
CARD_FRAME_RATE = "frame_rate"
CARD_UNITS = "units"
CARD_COLS = "cols"
CARD_PIXEL_ARCMIN_X = "pixel_arcmin_x"
CARD_PIXEL_ARCMIN_Y = "pixel_arcmin_y"


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Typed view of the required processed-trial header cards."""

    n_frames: int
    n_samples_original: int
    time_interval: float
    n_pixels_vertical: int
    n_pixels_horizontal_int: int
    scanner_offset_x: float
    scanner_offset_y: float
    non_flyback_start: int
    non_flyback_end: int

    @classmethod
    def from_header(cls, header: Mapping[str, Any]) -> FrameMetadata:
        """Build metadata from a header, failing on a missing required card."""
        try:
            return cls(
                n_frames=int(header[CARD_N_FRAMES]),
                n_samples_original=int(header[CARD_N_SAMPLES_ORIGINAL]),
                time_interval=float(header[CARD_TIME_INTERVAL]),
                n_pixels_vertical=int(header[CARD_N_PIXELS_VERTICAL]),
                n_pixels_horizontal_int=int(header[CARD_N_PIXELS_HORIZONTAL_INT]),
                scanner_offset_x=float(header[CARD_SCANNER_OFFSET_X]),
                scanner_offset_y=float(header[CARD_SCANNER_OFFSET_Y]),
                non_flyback_start=int(header[CARD_NON_FLYBACK_START]),
                non_flyback_end=int(header[CARD_NON_FLYBACK_END]),
            )
        except KeyError as exc:
            raise IncompatibleFormatError(
                f"missing required header card {exc}"
            ) from exc

    def to_header(self) -> dict[str, Any]:
        """Return the metadata as a plain mapping of FITS keywords."""
        return {
            CARD_N_FRAMES: self.n_frames,
            CARD_N_SAMPLES_ORIGINAL: self.n_samples_original,
            CARD_TIME_INTERVAL: self.time_interval,
            CARD_N_PIXELS_VERTICAL: self.n_pixels_vertical,
            CARD_N_PIXELS_HORIZONTAL_INT: self.n_pixels_horizontal_int,
            CARD_SCANNER_OFFSET_X: self.scanner_offset_x,
            CARD_SCANNER_OFFSET_Y: self.scanner_offset_y,
            CARD_NON_FLYBACK_START: self.non_flyback_start,
            CARD_NON_FLYBACK_END: self.non_flyback_end,
        }


@dataclass(frozen=True, slots=True)
class TrialArrays:
    """A processed trial: frames, desinusoiding parameters and line starts."""

    frames: np.ndarray
    fit_parameters: np.ndarray
    line_starts: np.ndarray
    metadata: FrameMetadata

    @classmethod
    def from_file(cls, path: str | Path) -> TrialArrays:
        """Read a processed trial from a norifmt FITS file."""
        with FitsFile.open(path) as fits_file:
            frames = fits_file.array(EXT_FRAMES)
            fit_parameters = fits_file.array(EXT_FIT_PARAMS)
            line_starts = fits_file.array(EXT_LINE_STARTS)
            metadata = FrameMetadata.from_header(fits_file.header(EXT_FRAMES))
        return cls(
            frames=frames,
            fit_parameters=fit_parameters,
            line_starts=line_starts,
            metadata=metadata,
        )

    def to_extensions(self) -> list[Extension]:
        """Return the trial as FITS extensions ready for ``write_fits_multi``."""
        return [
            Extension(self.frames, EXT_FRAMES, self.metadata.to_header()),
            Extension(self.fit_parameters, EXT_FIT_PARAMS),
            Extension(self.line_starts, EXT_LINE_STARTS),
        ]
