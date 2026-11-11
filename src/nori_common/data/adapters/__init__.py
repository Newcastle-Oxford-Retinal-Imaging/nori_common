"""Data access adapters."""

from nori_common.data.adapters.fits_trial import FitsTrialAdapter
from nori_common.data.adapters.in_memory import InMemoryTrial

__all__ = ["FitsTrialAdapter", "InMemoryTrial"]
