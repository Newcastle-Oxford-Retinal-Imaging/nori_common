"""Data access adapters."""

from nori_common.data.adapters.arrays import (
    FitsArrayAdapter,
    FitsCoarseMotionAdapter,
    FitsEyeMotionAdapter,
    FitsFramePositionsAdapter,
    FitsMetricsAdapter,
    FitsMontageMotionAdapter,
    read_coarse_motion,
    read_eye_motion,
    read_frame_positions,
    read_metrics,
    read_montage_motion,
)
from nori_common.data.adapters.fits_trial import FitsTrialAdapter
from nori_common.data.adapters.in_memory import InMemoryTrial

__all__ = [
    "FitsArrayAdapter",
    "FitsCoarseMotionAdapter",
    "FitsEyeMotionAdapter",
    "FitsFramePositionsAdapter",
    "FitsMetricsAdapter",
    "FitsMontageMotionAdapter",
    "FitsTrialAdapter",
    "InMemoryTrial",
    "read_coarse_motion",
    "read_eye_motion",
    "read_frame_positions",
    "read_metrics",
    "read_montage_motion",
]
