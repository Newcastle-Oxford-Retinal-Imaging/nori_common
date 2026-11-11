"""Data access adapters."""

from nori_common.data.adapters.arrays import (
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
from nori_common.data.adapters.fits_trial import (
    FitsTrialAdapter,
    LegacyFitsTrialAdapter,
)
from nori_common.data.adapters.in_memory import InMemoryTrial
from nori_common.data.adapters.raw import (
    RawAcquisitionData,
    from_metadata,
    read_raw_acquisition,
)

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
]
