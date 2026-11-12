"""Fixed axis recipes for known NORI products."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisRule:
    """Declarative recipe for one array axis."""

    name: str
    labels: tuple[str, ...] | None = None
    label_source: str | None = None
    unit: str | None = None
    coordinate_source: str | None = None


@dataclass(frozen=True)
class ArraySpec:
    """Fixed axis recipe for one product array."""

    name: str
    role: str
    axes: tuple[AxisRule, ...]
    unit: str | None = None


_BOUND_LABELS = (
    "start_y",
    "stop_y",
    "start_x",
    "stop_x",
    "y_of_s",
    "y_of_s2",
    "x_of_s",
    "x_of_s2",
)

_TRACE_FIELD_LABELS = ("t", "y", "x", "fn", "fb", "pd", "bl", "tk")

PRODUCT_SPECS: dict[tuple[str, str], ArraySpec] = {
    ("trial", "frames"): ArraySpec(
        name="PRIMARY",
        role="frames",
        unit="a.u.",
        axes=(
            AxisRule("channel", label_source="channel_roles"),
            AxisRule("frame", unit="frame"),
            AxisRule("y", unit="pixel"),
            AxisRule("x", unit="pixel"),
            AxisRule("direction", labels=("forward", "reverse")),
        ),
    ),
    ("trial", "desinusoid_parameters"): ArraySpec(
        name="FIT_PARAMS",
        role="desinusoid_parameters",
        axes=(
            AxisRule("axis", labels=("slow", "fast")),
            AxisRule("coefficient"),
        ),
    ),
    ("trial", "line_start_samples"): ArraySpec(
        name="LINE_STARTS",
        role="line_start_samples",
        unit="sample",
        axes=(AxisRule("line", unit="sample"),),
    ),
    ("eye_motion_per_pixel", "per_pixel_positions"): ArraySpec(
        name="PRIMARY",
        role="per_pixel_positions",
        unit="pixel",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("y", unit="pixel"),
            AxisRule("x", unit="pixel"),
            AxisRule("axis", labels=("y", "x")),
        ),
    ),
    ("metrics", "metrics"): ArraySpec(
        name="PRIMARY",
        role="metrics",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("metric", label_source="column_names"),
        ),
    ),
    ("frame_positions", "frame_positions"): ArraySpec(
        name="PRIMARY",
        role="frame_positions",
        unit="pixel",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("bound", labels=_BOUND_LABELS, unit="pixel"),
        ),
    ),
    ("coarse_motion", "coarse_motion"): ArraySpec(
        name="PRIMARY",
        role="coarse_motion",
        unit="pixel",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("axis", labels=("y", "x")),
        ),
    ),
    ("corrected_frames", "corrected_frames"): ArraySpec(
        name="PRIMARY",
        role="corrected_frames",
        unit="a.u.",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("y", unit="pixel"),
            AxisRule("x", unit="pixel"),
        ),
    ),
    ("montage_motion", "motion_field"): ArraySpec(
        name="PRIMARY",
        role="motion_field",
        unit="pixel",
        axes=(
            AxisRule("y", unit="pixel"),
            AxisRule("x", unit="pixel"),
            AxisRule("axis", labels=("y", "x")),
        ),
    ),
    ("eye_motion_per_row", "per_row_positions"): ArraySpec(
        name="PRIMARY",
        role="per_row_positions",
        unit="pixel",
        axes=(
            AxisRule("frame", unit="frame"),
            AxisRule("row", unit="pixel"),
            AxisRule("axis", labels=("y", "x")),
        ),
    ),
    ("trace", "eye_motion_trace"): ArraySpec(
        name="PRIMARY",
        role="eye_motion_trace",
        axes=(
            AxisRule("field", label_source="cols"),
            AxisRule("sample", unit="sample"),
        ),
    ),
    ("raw", "raw_samples"): ArraySpec(
        name="PRIMARY",
        role="raw_samples",
        axes=(
            AxisRule("channel", label_source="channel_roles"),
            AxisRule(
                "sample",
                unit="sample",
                coordinate_source="time_interval",
            ),
        ),
    ),
}

# Explicit, readable aliases for tests and validation code.
BOUND_LABELS = _BOUND_LABELS
TRACE_FIELD_LABELS = _TRACE_FIELD_LABELS
