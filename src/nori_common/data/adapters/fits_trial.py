"""FITS adapters for processed trial data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.io import fits

from nori_common.data import protocols
from nori_common.data.base import ArtifactBase
from nori_common.data.errors import CapabilityNotAvailable, SchemaError
from nori_common.data.extras import Extras, PixelScaleData, ProvenanceData
from nori_common.exceptions import ExtensionNotFound
from nori_common.fits import FitsFile
from nori_common.fits.schema import (
    CARD_N_FRAMES,
    CARD_N_PIXELS_HORIZONTAL_INT,
    CARD_N_PIXELS_VERTICAL,
    CARD_N_SAMPLES_ORIGINAL,
    CARD_NON_FLYBACK_END,
    CARD_NON_FLYBACK_START,
    CARD_PD_TRANSITION_PREFIX,
    CARD_PD_VALUE_PREFIX,
    CARD_SCANNER_OFFSET_X,
    CARD_SCANNER_OFFSET_Y,
    CARD_TIME_INTERVAL,
    EXT_FIT_PARAMS,
    EXT_LINE_STARTS,
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

_PIXEL_SIZE_CARDS = (
    ("Pixel_size_x_arcmin", "PIXARCMX"),
    ("Pixel_size_y_arcmin", "PIXARCMY"),
)

_KNOWN_CARDS = {
    CARD_NON_FLYBACK_START.casefold(),
    CARD_NON_FLYBACK_END.casefold(),
    CARD_N_FRAMES.casefold(),
    CARD_N_PIXELS_VERTICAL.casefold(),
    CARD_N_PIXELS_HORIZONTAL_INT.casefold(),
    CARD_N_SAMPLES_ORIGINAL.casefold(),
    CARD_SCANNER_OFFSET_X.casefold(),
    CARD_SCANNER_OFFSET_Y.casefold(),
    CARD_TIME_INTERVAL.casefold(),
    "source_type",
    "creator",
    "creator_version",
    "aoslo_frame_rate",
    "framerate",
    "frame_rate",
    "pixel_size_x_arcmin",
    "pixel_size_y_arcmin",
    "pixarcmx",
    "pixarcmy",
}

_PD_PREFIXES = (
    CARD_PD_VALUE_PREFIX.casefold(),
    CARD_PD_TRANSITION_PREFIX.casefold(),
)


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
        pixel_size_arcmin: tuple[float, float] | None = None,
    ) -> None:
        self._path = Path(path)
        with FitsFile.open(self._path) as fits_file:
            self._header: dict[str, object] = fits_file.header()
        self._frames: np.ndarray | None = None
        self._timing_arrays: tuple[np.ndarray, np.ndarray] | None = None
        self._extras = Extras(self._build_extras())

        start = _lookup(self._header, CARD_NON_FLYBACK_START)
        end = _lookup(self._header, CARD_NON_FLYBACK_END)
        if start is None or end is None:
            missing = CARD_NON_FLYBACK_START if start is None else CARD_NON_FLYBACK_END
            raise SchemaError(missing, "trial")

        self._non_flyback_rows = (int(start), int(end))

        scanner_offset_y = _lookup(self._header, CARD_SCANNER_OFFSET_Y)
        scanner_offset_x = _lookup(self._header, CARD_SCANNER_OFFSET_X)
        self._scanner_offset_pixels: tuple[float, float] | None = None
        if scanner_offset_y is not None and scanner_offset_x is not None:
            self._scanner_offset_pixels = (
                float(scanner_offset_y),
                float(scanner_offset_x),
            )

        self._frame_rate_hz = frame_rate
        if self._frame_rate_hz is None:
            for card in _FRAME_RATE_CARDS:
                value = _lookup(self._header, card)
                if value is not None:
                    self._frame_rate_hz = float(value)
                    break

        self._n_frames = _lookup(self._header, CARD_N_FRAMES)
        n_vertical = _lookup(self._header, CARD_N_PIXELS_VERTICAL)
        n_horizontal_int = _lookup(self._header, CARD_N_PIXELS_HORIZONTAL_INT)
        self._frame_shape: tuple[int, int] | None = None
        if n_vertical is not None and n_horizontal_int is not None:
            self._frame_shape = (
                int(n_vertical),
                int(n_horizontal_int) // 2,
            )

        self._n_samples_original = _lookup(self._header, CARD_N_SAMPLES_ORIGINAL)
        self._time_interval = _lookup(self._header, CARD_TIME_INTERVAL)
        self._timing_available = (
            self._n_samples_original is not None
            and self._time_interval is not None
            and self._timing_arrays_available()
        )

        self._pixel_scale = (
            PixelScaleData(
                pixel_size_arcmin_x=float(pixel_size_arcmin[0]),
                pixel_size_arcmin_y=float(pixel_size_arcmin[1]),
            )
            if pixel_size_arcmin is not None
            else self._pixel_scale_from_header()
        )

        self._pd_values, self._pd_transitions = self._read_photodiode()

    def _timing_arrays_available(self) -> bool:
        with FitsFile.open(self._path) as fits_file:
            names = {name.upper() for name in fits_file.extensions()}
        return EXT_FIT_PARAMS in names and EXT_LINE_STARTS in names

    def _load_timing_arrays(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._timing_arrays is not None:
            return self._timing_arrays
        with FitsFile.open(self._path) as fits_file:
            try:
                fit_parameters = np.asarray(fits_file.array(EXT_FIT_PARAMS))
                line_starts = np.asarray(fits_file.array(EXT_LINE_STARTS))
            except ExtensionNotFound:
                return None
        self._timing_arrays = (fit_parameters, line_starts)
        return self._timing_arrays

    def _pixel_scale_from_header(self) -> PixelScaleData | None:
        x = None
        y = None
        for key in _PIXEL_SIZE_CARDS[0]:
            value = _lookup(self._header, key)
            if value is not None:
                x = float(value)
                break
        for key in _PIXEL_SIZE_CARDS[1]:
            value = _lookup(self._header, key)
            if value is not None:
                y = float(value)
                break
        if x is None or y is None:
            return None
        return PixelScaleData(pixel_size_arcmin_x=x, pixel_size_arcmin_y=y)

    def _read_photodiode(self) -> tuple[dict[int, float], dict[int, float]]:
        values: dict[int, float] = {}
        transitions: dict[int, float] = {}
        for key, value in self._header.items():
            folded = key.casefold()
            if folded.startswith(CARD_PD_VALUE_PREFIX.casefold()):
                values[int(folded[len(CARD_PD_VALUE_PREFIX) :])] = float(value)
            elif folded.startswith(CARD_PD_TRANSITION_PREFIX.casefold()):
                transitions[int(folded[len(CARD_PD_TRANSITION_PREFIX) :])] = float(
                    value
                )
        return values, transitions

    def _build_extras(self) -> dict[str, object]:
        structural = {card.casefold() for card in _STRUCTURAL_CARDS}
        extras: dict[str, object] = {}
        for key, value in self._header.items():
            folded = key.casefold()
            if folded in structural or folded in _KNOWN_CARDS:
                continue
            if any(folded.startswith(prefix) for prefix in _PD_PREFIXES):
                continue
            extras[f"fits_header.{key}"] = value
        return extras

    def _provided_capabilities(self) -> frozenset[type]:
        capabilities: set[type] = {
            protocols.Provenance,
            protocols.Frames,
            protocols.ScanGeometry,
        }
        if self._frame_shape is not None and self._n_frames is not None:
            capabilities.add(protocols.FrameShape)
        if self._frame_rate_hz is not None:
            capabilities.add(protocols.FrameRate)
        if self._timing_available:
            capabilities.add(protocols.Timing)
        if self._pixel_scale is not None:
            capabilities.add(protocols.PixelScale)
        if self._pd_values or self._pd_transitions:
            capabilities.add(protocols.PhotodiodeTransitions)
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
    def scanner_offset_pixels(self) -> tuple[float, float]:
        if self._scanner_offset_pixels is None:
            raise SchemaError("scanner_offset_x/scanner_offset_y", "trial")
        return self._scanner_offset_pixels

    @property
    def n_frames(self) -> int:
        if self._n_frames is None:
            raise CapabilityNotAvailable(self, protocols.FrameShape)
        return int(self._n_frames)

    @property
    def frame_shape(self) -> tuple[int, int]:
        if self._frame_shape is None:
            raise CapabilityNotAvailable(self, protocols.FrameShape)
        return self._frame_shape

    @property
    def frame_rate_hz(self) -> float:
        if self._frame_rate_hz is None:
            raise CapabilityNotAvailable(self, protocols.FrameRate)
        return self._frame_rate_hz

    @property
    def n_samples_original(self) -> int:
        if not self._timing_available:
            raise CapabilityNotAvailable(self, protocols.Timing)
        return int(self._n_samples_original)

    @property
    def time_interval_s(self) -> float:
        if not self._timing_available:
            raise CapabilityNotAvailable(self, protocols.Timing)
        return float(self._time_interval)

    @property
    def fit_parameters(self) -> np.ndarray:
        arrays = self._load_timing_arrays()
        if arrays is None:
            raise CapabilityNotAvailable(self, protocols.Timing)
        return arrays[0]

    @property
    def line_starts(self) -> np.ndarray:
        arrays = self._load_timing_arrays()
        if arrays is None:
            raise CapabilityNotAvailable(self, protocols.Timing)
        return arrays[1]

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
    def values(self) -> dict[int, float]:
        if not self._pd_values:
            raise CapabilityNotAvailable(self, protocols.PhotodiodeTransitions)
        return dict(self._pd_values)

    @property
    def transitions(self) -> dict[int, float]:
        if not self._pd_transitions:
            raise CapabilityNotAvailable(self, protocols.PhotodiodeTransitions)
        return dict(self._pd_transitions)

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


class LegacyFitsTrialAdapter(FitsTrialAdapter):
    """Trial adapter for legacy processed files with unnamed timing HDUs.

    The legacy layout stores the primary frames HDU, followed by unnamed
    ``FIT_PARAMS`` and ``LINE_STARTS`` HDUs.
    """

    def _timing_arrays_available(self) -> bool:
        with fits.open(self._path) as hdul:
            data_hdus = [hdu for hdu in hdul if hdu.size]
        return len(data_hdus) >= 3

    def _load_timing_arrays(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._timing_arrays is not None:
            return self._timing_arrays
        with fits.open(self._path) as hdul:
            data_hdus = [hdu for hdu in hdul if hdu.size]
            if len(data_hdus) < 3:
                return None
            arrays = (
                np.asarray(data_hdus[1].data),
                np.asarray(data_hdus[2].data),
            )
        self._timing_arrays = arrays
        return self._timing_arrays
