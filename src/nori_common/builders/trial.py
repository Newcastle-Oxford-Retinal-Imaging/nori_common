"""Builder for validated trial payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from nori_common.builders.axes import build_axis_descriptors
from nori_common.contracts import (
    ArrayDescriptor,
    ArtifactProvenance,
    FrameMetadata,
    PhotodiodeTransitionsData,
    PixelScaleMetadata,
    TrialMetadata,
)
from nori_common.contracts.errors import SchemaError
from nori_common.validation import validate_trial_arrays


@dataclass(frozen=True)
class TrialPayload:
    """Arrays and validated metadata for a processed trial."""

    frames: np.ndarray
    fit_parameters: np.ndarray
    line_starts: np.ndarray
    metadata: TrialMetadata


def build_trial_payload(
    frames: np.ndarray,
    fit_parameters: np.ndarray,
    line_starts: np.ndarray,
    *,
    n_samples_original: int,
    time_interval: float,
    scanner_offset: tuple[float, float],
    non_flyback_rows: tuple[int, int],
    provenance: ArtifactProvenance | None = None,
    pixel_scale: PixelScaleMetadata | None = None,
    photodiode: PhotodiodeTransitionsData | None = None,
    channel_roles: Mapping[int, str] | None = None,
    schema_version: str = "1.0",
) -> TrialPayload:
    """Build and validate a processed-trial payload.

    Frame and array descriptors are derived from the actual arrays; callers do
    not supply n_frames, n_pixels_*, or n_pixels_horizontal_int manually.
    """
    frames = np.asarray(frames)
    fit_parameters = np.asarray(fit_parameters)
    line_starts = np.asarray(line_starts)

    if frames.ndim != 5:
        raise SchemaError("frames", "trial")
    if frames.shape[4] != 2:
        raise SchemaError("scan_directions", "trial")

    n_frames = int(frames.shape[1])
    n_pixels_vertical = int(frames.shape[2])
    n_pixels_horizontal = int(frames.shape[3])
    n_pixels_horizontal_int = 2 * n_pixels_horizontal

    frame_metadata = FrameMetadata(
        n_frames=n_frames,
        n_samples_original=n_samples_original,
        time_interval=time_interval,
        n_pixels_vertical=n_pixels_vertical,
        n_pixels_horizontal_int=n_pixels_horizontal_int,
        scanner_offset_x=scanner_offset[1],
        scanner_offset_y=scanner_offset[0],
        Non_flyback_start=non_flyback_rows[0],
        Non_flyback_end=non_flyback_rows[1],
    )

    channel_roles_dict = dict(channel_roles or {})
    frames_axes = build_axis_descriptors(
        frames,
        schema_name="trial",
        role="frames",
        metadata={"channel_roles": channel_roles_dict},
    )
    fit_parameter_axes = build_axis_descriptors(
        fit_parameters,
        schema_name="trial",
        role="desinusoid_parameters",
    )
    line_start_axes = build_axis_descriptors(
        line_starts,
        schema_name="trial",
        role="line_start_samples",
    )

    arrays = (
        ArrayDescriptor(
            name="PRIMARY",
            role="frames",
            dtype=str(frames.dtype),
            shape=tuple(int(size) for size in frames.shape),
            axes=frames_axes,
            unit="a.u.",
        ),
        ArrayDescriptor(
            name="FIT_PARAMS",
            role="desinusoid_parameters",
            dtype=str(fit_parameters.dtype),
            shape=tuple(int(size) for size in fit_parameters.shape),
            axes=fit_parameter_axes,
        ),
        ArrayDescriptor(
            name="LINE_STARTS",
            role="line_start_samples",
            dtype=str(line_starts.dtype),
            shape=tuple(int(size) for size in line_starts.shape),
            axes=line_start_axes,
            unit="sample",
        ),
    )

    metadata = TrialMetadata(
        schema_version=schema_version,
        provenance=provenance or ArtifactProvenance(),
        frame=frame_metadata,
        arrays=arrays,
        channel_roles=channel_roles_dict,
        pixel_scale=pixel_scale,
        photodiode=photodiode,
    )

    validate_trial_arrays(metadata, frames, fit_parameters, line_starts)

    return TrialPayload(
        frames=frames,
        fit_parameters=fit_parameters,
        line_starts=line_starts,
        metadata=metadata,
    )
