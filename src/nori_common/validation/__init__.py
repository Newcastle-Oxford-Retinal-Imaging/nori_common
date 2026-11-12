"""Validation helpers that require array data."""

from nori_common.validation.array_consistency import (
    validate_array_descriptor,
    validate_frame_arrays,
    validate_raw_data,
    validate_trial_arrays,
)

__all__ = [
    "validate_array_descriptor",
    "validate_frame_arrays",
    "validate_raw_data",
    "validate_trial_arrays",
]
