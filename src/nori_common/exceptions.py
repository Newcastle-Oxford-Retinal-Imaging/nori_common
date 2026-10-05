"""Exceptions raised by :mod:`nori_common`."""

from __future__ import annotations


class FitsError(Exception):
    """Base class for all nori_common FITS errors."""


class ExtensionNotFound(FitsError, KeyError):
    """A requested FITS extension name is not present in the file."""


class IncompatibleFormatError(FitsError):
    """A file does not match the expected norifmt layout or metadata."""
