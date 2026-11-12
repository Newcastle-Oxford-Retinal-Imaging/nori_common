"""Legacy processed-trial timing adapter.

The legacy layout stores the primary frames HDU, followed by unnamed
``FIT_PARAMS`` and ``LINE_STARTS`` HDUs. This adapter keeps positional access
isolated from the current named-extension adapter.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from nori_common.adapters.fits_trial import FitsTrialAdapter


class LegacyFitsTrialAdapter(FitsTrialAdapter):
    """Trial adapter for legacy processed files with unnamed timing HDUs."""

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
