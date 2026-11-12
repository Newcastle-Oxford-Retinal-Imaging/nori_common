"""Resolve fixed axis recipes into concrete axis descriptors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from nori_common.contracts import AxisDescriptor, CoordinateSpec
from nori_common.contracts.axis_specs import PRODUCT_SPECS, ArraySpec, AxisRule
from nori_common.contracts.errors import SchemaError


def array_spec(schema_name: str, role: str) -> ArraySpec | None:
    """Return the fixed recipe for a product array, if one exists."""
    return PRODUCT_SPECS.get((schema_name, role))


def _labels_from_source(
    source: str,
    size: int,
    metadata: Mapping[str, object],
    schema_name: str,
) -> tuple[str, ...]:
    if source == "channel_roles":
        raw = metadata.get("channel_roles")
        if raw is None:
            raise SchemaError("channel_roles", schema_name)
        if isinstance(raw, Mapping):
            labels = tuple(
                str(value)
                for _, value in sorted(raw.items(), key=lambda item: int(item[0]))
            )
        else:
            labels = tuple(str(value) for value in raw)
    elif source == "column_names":
        raw = metadata.get("column_names")
        if raw is None:
            raise SchemaError("column_names", schema_name)
        labels = (
            tuple(part.strip() for part in str(raw).split(","))
            if isinstance(raw, str)
            else tuple(str(value) for value in raw)
        )
    elif source == "cols":
        raw = metadata.get("cols")
        if raw is None:
            raise SchemaError("cols", schema_name)
        labels = (
            tuple(part.strip() for part in str(raw).split(","))
            if isinstance(raw, str)
            else tuple(str(value) for value in raw)
        )
    else:
        raise SchemaError("label_source", schema_name)

    if len(labels) != size:
        raise SchemaError(source, schema_name)
    return labels


def _labels_for_rule(
    rule: AxisRule,
    size: int,
    metadata: Mapping[str, object],
    schema_name: str,
) -> tuple[str, ...] | None:
    if rule.labels is not None:
        if len(rule.labels) != size:
            raise SchemaError(rule.name, schema_name)
        return rule.labels
    if rule.label_source is None:
        return None
    return _labels_from_source(rule.label_source, size, metadata, schema_name)


def _coordinates_for_rule(
    rule: AxisRule,
    metadata: Mapping[str, object],
) -> CoordinateSpec | None:
    if rule.coordinate_source == "time_interval":
        value = metadata.get("time_interval")
        if value is None:
            return None
        return CoordinateSpec(start=0.0, step=float(value), unit="s")
    return None


def build_axis_descriptors(
    data: np.ndarray,
    *,
    schema_name: str,
    role: str,
    metadata: Mapping[str, object] | None = None,
    axes_names: Sequence[str] = (),
) -> tuple[AxisDescriptor, ...]:
    """Build axis descriptors for an array.

    Known products use their fixed recipe. Unknown products can still provide
    explicit axis names; those axes are emitted without labels.
    """
    data = np.asarray(data)
    metadata = metadata or {}
    spec = array_spec(schema_name, role)

    if spec is None:
        if not axes_names:
            raise SchemaError("schema_name", schema_name)
        if len(axes_names) != data.ndim:
            raise SchemaError("axes", schema_name)
        return tuple(
            AxisDescriptor(name=str(name), size=int(size))
            for name, size in zip(axes_names, data.shape, strict=True)
        )

    spec_names = tuple(axis.name for axis in spec.axes)
    if axes_names and tuple(axes_names) != spec_names:
        raise SchemaError("axes", schema_name)
    if len(spec.axes) != data.ndim:
        raise SchemaError("shape", schema_name)

    return tuple(
        AxisDescriptor(
            name=rule.name,
            size=int(data.shape[index]),
            labels=_labels_for_rule(
                rule,
                int(data.shape[index]),
                metadata,
                schema_name,
            ),
            unit=rule.unit,
            coordinates=_coordinates_for_rule(rule, metadata),
        )
        for index, rule in enumerate(spec.axes)
    )
