"""Lossless FITS read/write with an Astropy-free public boundary.

Compression is always lossless: ``gzip`` tile-compresses the data without
quantization (bit-exact for floating-point data) and ``none`` writes
uncompressed FITS.  There is deliberately no lossy codec.

Headers cross the boundary as plain mappings rather than
``astropy.io.fits.Header`` objects.  The single escape hatch to raw Astropy
objects is :meth:`FitsFile.as_hdu`.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

import numpy as np
from astropy.io import fits

from nori_common._gzip import GZIP_KEY, gzip_level
from nori_common.exceptions import ExtensionNotFound, FitsError

logger = logging.getLogger(__name__)

__all__ = [
    "Compression",
    "Extension",
    "FitsFile",
    "Precision",
    "WritePolicy",
    "write_fits",
    "write_fits_multi",
]


class Precision(str, Enum):
    """Floating-point precision for FITS output."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class Compression(str, Enum):
    """FITS tile-compression algorithm.

    Both options are lossless; a lossy codec is deliberately not offered.
    """

    NONE = "none"
    GZIP = "gzip"


@dataclass(frozen=True, slots=True)
class WritePolicy:
    """Precision and (lossless) compression settings for a FITS write."""

    precision: Precision = Precision.FLOAT32
    compression: Compression = Compression.NONE
    gzip_level: int = 6

    def __post_init__(self) -> None:
        if not 1 <= self.gzip_level <= 9:
            raise ValueError(
                f"gzip_level must be between 1 and 9, got {self.gzip_level}"
            )


@dataclass(frozen=True, slots=True)
class Extension:
    """One array destined for a FITS extension.

    ``name`` sets the ``EXTNAME``; ``header`` is an optional plain mapping of
    FITS keywords.
    """

    data: np.ndarray
    name: str | None = None
    header: Mapping[str, Any] | None = None


class FitsFile:
    """A read-only view over a FITS file.

    Selection is by extension name, or by the first data-bearing HDU when no
    name is given.  The latter handles tile-compressed files, which prepend a
    blank primary and keep the real data in an extension.  There is no
    positional index: callers name what they want, which fails loudly rather
    than silently returning the wrong HDU.
    """

    def __init__(self, hdu_list: fits.HDUList) -> None:
        self._hdu_list = hdu_list

    @classmethod
    def open(cls, path: str | Path, *, memmap: bool = False) -> FitsFile:
        """Open ``path`` lazily; close it via :meth:`close` or a ``with`` block."""
        return cls(fits.open(path, memmap=memmap))

    def _select(self, name: str | None) -> fits.hdu.base.BaseHDU:
        if name is not None:
            wanted = name.upper()
            for hdu in self._hdu_list:
                if hdu.name.upper() == wanted and hdu.data is not None:
                    return hdu
            if wanted == "PRIMARY":
                # Tile-compressed files keep the primary payload in an
                # extension under a blank primary, so "PRIMARY" means the
                # primary payload, not the literal blank HDU.
                for hdu in self._hdu_list:
                    if hdu.data is not None:
                        return hdu
            raise ExtensionNotFound(name)

        for hdu in self._hdu_list:
            if hdu.data is not None:
                return hdu
        raise FitsError(f"no data-bearing HDU in {self._hdu_list.filename()!r}")

    def array(self, name: str | None = None, *, copy: bool = True) -> np.ndarray:
        """Return the data of the named extension (or the primary payload).

        The result is an independent in-memory copy unless ``copy=False``, in
        which case it may alias the file and is only valid while open.
        """
        data = self._select(name).data
        if data is None:
            raise FitsError(f"extension {name!r} carries no data")
        return np.array(data) if copy else data

    def header(self, name: str | None = None) -> dict[str, Any]:
        """Return an independent plain-dict copy of an extension header."""
        return dict(self._select(name).header)

    def extensions(self) -> list[str]:
        """Return the extension names in file order."""
        return [hdu.name for hdu in self._hdu_list]

    def as_hdu(self, name: str | None = None) -> fits.hdu.base.BaseHDU:
        """Escape hatch: return the underlying astropy HDU.

        Not a stability boundary -- the return type tracks the installed
        Astropy version.
        """
        return self._select(name)

    def close(self) -> None:
        self._hdu_list.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _cast(data: np.ndarray, precision: Precision) -> np.ndarray:
    if precision is Precision.FLOAT32:
        return np.asarray(data, dtype=np.float32)
    return np.asarray(data)


def _tile_shape_for(data: np.ndarray) -> tuple[int, ...] | None:
    """Return a per-image tile shape for ``data`` when tile-compressing.

    Astropy defaults to compressing an entire array as a single tile when no
    ``tile_shape`` is given, which is pathologically slow and memory-heavy for
    large multi-frame arrays.  Treat each array as a stack of identically
    shaped images (the trailing three dimensions) and compress each image as
    its own tile.

    Arrays of two dimensions or fewer are left to astropy's default since a
    single image is already an appropriate tile.  A three-dimensional array is
    a stack of ``(H, W)`` images; four or more dimensions are a stack of
    ``(H, W, D)`` images, one tile per combination of leading indices.
    """
    if data.ndim <= 2:
        return None
    if data.ndim == 3:
        return (1,) + tuple(data.shape[-2:])
    return (1,) * (data.ndim - 3) + tuple(data.shape[-3:])


def _as_header(header: Mapping[str, Any] | None) -> fits.Header | None:
    """Convert a plain mapping to an astropy ``Header`` (astropy requires one)."""
    if header is None or isinstance(header, fits.Header):
        return header
    result = fits.Header()
    for key, value in header.items():
        result[key] = value
    return result


def _compressed_hdu(
    data: np.ndarray,
    header: Mapping[str, Any] | None,
    name: str | None,
    tile_shape: tuple[int, ...] | None,
) -> fits.CompImageHDU:
    """Build a losslessly GZIP-compressed ``CompImageHDU`` for ``data``."""
    return fits.CompImageHDU(
        data,
        header=_as_header(header),
        name=name,
        compression_type=GZIP_KEY,
        tile_shape=tile_shape,
        quantize_level=0,
    )


def _compression_context(policy: WritePolicy):
    if policy.compression is Compression.GZIP:
        return gzip_level(policy.gzip_level)
    return contextlib.nullcontext()


def _compression_label(policy: WritePolicy) -> str:
    if policy.compression is Compression.GZIP:
        return f"{GZIP_KEY}, lossless, level={policy.gzip_level}"
    return Compression.NONE.value


def _log_write(path: Path, count: int, policy: WritePolicy, start: float) -> None:
    logger.info(
        "Wrote %s (%d HDUs, precision=%s, compression=%s, %.2f MiB) in %.2fs",
        path,
        count,
        policy.precision.value,
        _compression_label(policy),
        path.stat().st_size / (1024 * 1024),
        time.perf_counter() - start,
    )


def write_fits(
    path: Path,
    data: np.ndarray,
    *,
    policy: WritePolicy | None = None,
    name: str | None = None,
    header: Mapping[str, Any] | None = None,
    tile_shape: tuple[int, ...] | None = None,
) -> None:
    """Write a single array to a FITS file.

    When the policy compression is ``gzip`` the data is tile-compressed
    losslessly (bit-exact for floating-point data); with ``none`` it is written
    uncompressed.  ``name`` sets the ``EXTNAME`` on the compressed data HDU and
    is ignored for uncompressed writes, where the data lives in the primary
    HDU.
    """
    policy = policy if policy is not None else WritePolicy()
    start = time.perf_counter()
    array = _cast(data, policy.precision)

    with _compression_context(policy):
        if policy.compression is Compression.GZIP:
            if tile_shape is None:
                tile_shape = _tile_shape_for(array)
            hdu = _compressed_hdu(array, header, name, tile_shape)
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
        else:
            fits.PrimaryHDU(array, header=_as_header(header)).writeto(
                path, overwrite=True
            )

    _log_write(path, 1, policy, start)


def write_fits_multi(
    path: Path,
    extensions: Sequence[Extension],
    *,
    policy: WritePolicy | None = None,
    tile_shapes: Sequence[tuple[int, ...] | None] | None = None,
) -> None:
    """Write multiple arrays to a multi-HDU FITS file.

    When the policy compression is ``gzip`` each array is tile-compressed
    losslessly (bit-exact for floating-point data); with ``none`` the arrays
    are written uncompressed.  When compressed, a blank primary is prepended so
    the data HDUs are extensions; otherwise the first entry is the primary.
    """
    policy = policy if policy is not None else WritePolicy()
    extensions = list(extensions)
    if tile_shapes is None:
        tile_shapes = [None] * len(extensions)
    start = time.perf_counter()

    with _compression_context(policy):
        if policy.compression is Compression.GZIP:
            # Tile compression requires the compressed images to live in
            # extension HDUs (>=1), so a blank primary is prepended.  Astropy
            # defaults a primary's EXTNAME to "PRIMARY", which collides with
            # the frames HDU name; readers must therefore select the first one
            # holding data.
            hdu_list = fits.HDUList([fits.PrimaryHDU()])
            for extension, tile_shape in zip(extensions, tile_shapes):
                array = _cast(extension.data, policy.precision)
                if tile_shape is None:
                    tile_shape = _tile_shape_for(array)
                hdu_list.append(
                    _compressed_hdu(array, extension.header, extension.name, tile_shape)
                )
        else:
            hdu_list = fits.HDUList()
            for idx, extension in enumerate(extensions):
                array = _cast(extension.data, policy.precision)
                if idx == 0:
                    hdu = fits.PrimaryHDU(array, header=_as_header(extension.header))
                    if extension.name:
                        hdu.header["EXTNAME"] = extension.name
                else:
                    hdu = fits.ImageHDU(
                        array,
                        header=_as_header(extension.header),
                        name=extension.name,
                    )
                hdu_list.append(hdu)

        hdu_list.writeto(path, overwrite=True)

    _log_write(path, len(extensions), policy, start)
