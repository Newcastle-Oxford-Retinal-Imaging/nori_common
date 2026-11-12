"""Reader dispatch for processed trial data."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from nori_common.adapters.fits_trial import FitsTrialAdapter
from nori_common.adapters.legacy_timing import LegacyFitsTrialAdapter
from nori_common.contracts.errors import SchemaError
from nori_common.exceptions import ExtensionNotFound
from nori_common.fits import FitsFile
from nori_common.fits.schema import EXT_FIT_PARAMS, EXT_LINE_STARTS
from nori_common.manifest import read_manifest


def _has_named_timing(path: Path) -> bool:
    with FitsFile.open(path) as fits_file:
        names = {name.upper() for name in fits_file.extensions()}
    return EXT_FIT_PARAMS in names and EXT_LINE_STARTS in names


def _has_legacy_timing(path: Path) -> bool:
    with fits.open(path) as hdul:
        data_hdus = [hdu for hdu in hdul if hdu.size]
    return len(data_hdus) >= 3


def read_trial(
    path: str | Path,
    *,
    frame_rate: float | None = None,
    pixel_size_arcmin: tuple[float, float] | None = None,
) -> FitsTrialAdapter:
    """Open a processed trial, choosing the current or legacy layout adapter.

    Parameters
    ----------
    path : str | Path
        Path to the processed FITS file.
    frame_rate : float, optional
        Frame rate in hertz, if already known.
    pixel_size_arcmin : tuple[float, float], optional
        Pixel size as ``(x, y)`` in arcminutes, if already known.

    Returns
    -------
    FitsTrialAdapter
        Trial capability object.
    """
    trial_path = Path(path)
    adapter: type[FitsTrialAdapter]
    try:
        manifest = read_manifest(trial_path)
    except ExtensionNotFound:
        manifest = None
    if manifest is not None:
        if manifest.schema_name != "trial":
            raise SchemaError("schema_name", "trial")
        adapter = FitsTrialAdapter
    elif _has_named_timing(trial_path):
        adapter = FitsTrialAdapter
    elif _has_legacy_timing(trial_path):
        adapter = LegacyFitsTrialAdapter
    else:
        adapter = FitsTrialAdapter

    return adapter(
        trial_path,
        frame_rate=frame_rate,
        pixel_size_arcmin=pixel_size_arcmin,
    )
