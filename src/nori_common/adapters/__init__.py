"""Data access adapters."""

from nori_common.adapters.arrays import (
    FitsArrayAdapter,
    FitsCoarseMotionAdapter,
    FitsEyeMotionAdapter,
    FitsEyeMotionPerRowAdapter,
    FitsFramePositionsAdapter,
    FitsMetricsAdapter,
    FitsMontageMotionAdapter,
    read_coarse_motion,
    read_eye_motion,
    read_eye_motion_per_row,
    read_frame_positions,
    read_metrics,
    read_montage_motion,
)
from nori_common.adapters.fits_trial import FitsTrialAdapter
from nori_common.adapters.in_memory import InMemoryTrial
from nori_common.adapters.legacy_timing import LegacyFitsTrialAdapter
from nori_common.adapters.raw import (
    RawAcquisitionData,
    from_metadata,
    read_raw_acquisition,
)
from nori_common.adapters.readers import read_trial

__all__ = [
    "FitsArrayAdapter",
    "FitsCoarseMotionAdapter",
    "FitsEyeMotionAdapter",
    "FitsEyeMotionPerRowAdapter",
    "FitsFramePositionsAdapter",
    "FitsMetricsAdapter",
    "FitsMontageMotionAdapter",
    "FitsTrialAdapter",
    "InMemoryTrial",
    "LegacyFitsTrialAdapter",
    "RawAcquisitionData",
    "from_metadata",
    "read_coarse_motion",
    "read_eye_motion",
    "read_eye_motion_per_row",
    "read_frame_positions",
    "read_metrics",
    "read_montage_motion",
    "read_raw_acquisition",
    "read_trial",
]
