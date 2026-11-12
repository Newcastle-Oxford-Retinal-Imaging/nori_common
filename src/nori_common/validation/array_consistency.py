"""Array-aware consistency validation for metadata contracts."""

from __future__ import annotations

import numpy as np

from nori_common.contracts.errors import SchemaError
from nori_common.contracts.manifest import ArrayDescriptor
from nori_common.contracts.raw import RawAcquisitionMetadata
from nori_common.contracts.trial import FrameMetadata, TrialMetadata


def validate_array_descriptor(
    descriptor: ArrayDescriptor,
    data: np.ndarray,
) -> None:
    """Validate one array descriptor against its data."""
    if tuple(int(size) for size in data.shape) != descriptor.shape:
        raise SchemaError("shape", descriptor.name)
    if len(descriptor.axes) != data.ndim:
        raise SchemaError("axes", descriptor.name)
    for axis, size in zip(descriptor.axes, data.shape, strict=True):
        if axis.size != int(size):
            raise SchemaError(axis.name, descriptor.name)


def validate_frame_arrays(metadata: FrameMetadata, frames: np.ndarray) -> None:
    """Validate frame metadata against a frames array."""
    if frames.ndim != 5:
        raise SchemaError("frames", "trial")
    if frames.shape[1] != metadata.n_frames:
        raise SchemaError("n_frames", "trial")
    if frames.shape[2] != metadata.n_pixels_vertical:
        raise SchemaError("n_pixels_vertical", "trial")
    if frames.shape[3] != metadata.n_pixels_horizontal_int // 2:
        raise SchemaError("n_pixels_horizontal_int", "trial")
    if frames.shape[4] != 2:
        raise SchemaError("scan_directions", "trial")
    if metadata.non_flyback_end >= frames.shape[2]:
        raise SchemaError("Non_flyback_end", "trial")


def validate_trial_arrays(
    metadata: TrialMetadata,
    frames: np.ndarray,
    fit_parameters: np.ndarray,
    line_starts: np.ndarray,
) -> None:
    """Validate a trial contract against its array data."""
    validate_frame_arrays(metadata.frame, frames)

    if fit_parameters.ndim != 2 or fit_parameters.shape[0] != 2:
        raise SchemaError("FIT_PARAMS", "trial")
    if fit_parameters.shape[1] < 5:
        raise SchemaError("FIT_PARAMS", "trial")

    required_line_starts = (
        metadata.frame.n_frames * metadata.frame.n_pixels_vertical + 1
    )
    if line_starts.ndim != 1 or line_starts.shape[0] < required_line_starts:
        raise SchemaError("LINE_STARTS", "trial")

    if metadata.arrays:
        descriptors = {descriptor.name: descriptor for descriptor in metadata.arrays}
        for name, data in (
            ("PRIMARY", frames),
            ("FIT_PARAMS", fit_parameters),
            ("LINE_STARTS", line_starts),
        ):
            descriptor = descriptors.get(name)
            if descriptor is not None:
                validate_array_descriptor(descriptor, data)


def validate_raw_data(metadata: RawAcquisitionMetadata, data: np.ndarray) -> None:
    """Validate raw acquisition metadata against its data array."""
    if data.ndim != 2:
        raise SchemaError("data", "raw acquisition")
    if data.shape[0] != len(metadata.channel_list):
        raise SchemaError("channel_list", "raw acquisition")
    if data.shape[1] == 0:
        raise SchemaError("data", "raw acquisition")
