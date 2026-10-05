"""A zlib-level-aware GZIP codec for astropy tile compression.

astropy's stock ``GZIP_1`` codec always encodes at zlib level 9, which is
pathologically slow on float64 data (DEFLATE's level-9 match search is roughly
15x slower than level 6 on 8-byte floating-point tiles) for negligible size
gain.  This module substitutes a subclass that encodes at a chosen level.

The substitution mutates astropy's process-global ``ALGORITHMS`` registry, so
it is confined to the :func:`gzip_level` context manager and the original codec
is always restored.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from astropy.io.fits.hdu.compressed import _codecs as _astropy_codecs
    from astropy.io.fits.hdu.compressed import _tiled_compression as _astropy_tiled

    _AVAILABLE = True
except ImportError:  # pragma: no cover - astropy internals drift
    _astropy_codecs = None
    _astropy_tiled = None
    _AVAILABLE = False

GZIP_KEY = "GZIP_1"


def _codec_for(level: int) -> type:
    """Build a ``GZIP_1`` codec class that encodes at ``level``.

    astropy instantiates codecs with no constructor arguments, so the level is
    captured in the closure rather than read from shared state.
    """

    class _GzipLevelCodec(_astropy_codecs.Gzip1):
        def encode(self, buf):
            dbytes = _astropy_codecs._as_big_endian_array(buf).tobytes()
            return gzip.compress(dbytes, compresslevel=level)

    return _GzipLevelCodec


@contextmanager
def gzip_level(level: int) -> Iterator[None]:
    """Temporarily install a level-``level`` GZIP_1 codec for astropy writes.

    The on-disk stream remains a standard ``GZIP_1`` stream (the inherited
    ``decode`` is level-agnostic), so files stay lossless and readable by any
    FITS tool.  The original codec is restored on exit, even if the body
    raises.
    """
    if not _AVAILABLE:
        logger.info(
            "astropy compressed-codec internals unavailable; GZIP writes "
            "will use astropy's default zlib level"
        )
        yield
        return

    original = _astropy_tiled.ALGORITHMS[GZIP_KEY]
    _astropy_tiled.ALGORITHMS[GZIP_KEY] = _codec_for(level)
    try:
        yield
    finally:
        _astropy_tiled.ALGORITHMS[GZIP_KEY] = original
