"""Shared FITS I/O and norifmt schema utilities for AOSLO retinal imaging."""

from nori_common.exceptions import (
    ExtensionNotFound,
    FitsError,
    IncompatibleFormatError,
)
from nori_common.fits import (
    Compression,
    Extension,
    FitsFile,
    Precision,
    WritePolicy,
    write_fits,
    write_fits_multi,
)

__all__ = [
    "Compression",
    "Extension",
    "ExtensionNotFound",
    "FitsError",
    "FitsFile",
    "IncompatibleFormatError",
    "Precision",
    "WritePolicy",
    "write_fits",
    "write_fits_multi",
]
