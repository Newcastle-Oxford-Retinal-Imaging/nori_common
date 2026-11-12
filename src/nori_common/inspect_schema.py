"""Inspect NORI FITS artifacts and schema manifests.

The command is intentionally tolerant: it must be usable on current
schema-aware files, older schema manifests, legacy processed trials, raw
acquisition FITS + YAML, and unknown FITS files.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from nori_common.contracts.axis_specs import BOUND_LABELS, TRACE_FIELD_LABELS
from nori_common.contracts.manifest import ArrayDescriptor, Manifest
from nori_common.manifest import MANIFEST_EXTNAME, SCHEMA_CARD, SCHEMA_VERSION_CARD

__all__ = [
    "ArrayReport",
    "AxisReport",
    "InspectionResult",
    "Issue",
    "inspect_path",
    "main",
]


@dataclass(frozen=True)
class Issue:
    """A diagnostic recorded while inspecting a file."""

    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class AxisReport:
    """A readable view of one array axis."""

    name: str
    size: int
    labels: tuple[str, ...] | None = None
    unit: str | None = None
    coordinates: dict[str, Any] | None = None
    source: str = "manifest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "labels": list(self.labels) if self.labels is not None else None,
            "unit": self.unit,
            "coordinates": self.coordinates,
            "source": self.source,
        }


@dataclass
class ArrayReport:
    """A readable view of one array in a FITS artifact."""

    name: str
    role: str | None
    dtype: str | None
    shape: tuple[int, ...]
    axes: list[AxisReport] = field(default_factory=list)
    source: str = "manifest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": [axis.to_dict() for axis in self.axes],
            "source": self.source,
        }


@dataclass
class InspectionResult:
    """The result of inspecting one file."""

    path: Path
    kind: str
    schema_name: str | None = None
    schema_version: str | None = None
    manifest_status: str = "absent"
    sidecar_status: str = "absent"
    arrays: list[ArrayReport] = field(default_factory=list)
    metadata_keys: list[str] = field(default_factory=list)
    provenance: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "manifest_status": self.manifest_status,
            "sidecar_status": self.sidecar_status,
            "arrays": [array.to_dict() for array in self.arrays],
            "metadata_keys": self.metadata_keys,
            "provenance": self.provenance,
            "details": _jsonable(self.details),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _jsonable(value: Any) -> Any:
    """Convert arbitrary detail values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _lookup(header: Mapping[str, Any], key: str) -> Any | None:
    """Case-insensitive header lookup."""
    folded = key.casefold()
    for card, value in header.items():
        if card.casefold() == folded:
            return value
    return None


def _header_int(header: Mapping[str, Any], key: str) -> int | None:
    value = _lookup(header, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _header_float(header: Mapping[str, Any], key: str) -> float | None:
    value = _lookup(header, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _data_hdus(hdul: fits.HDUList) -> list[fits.hdu.base.BaseHDU]:
    return [hdu for hdu in hdul if hdu.data is not None]


def _hdu_name(hdu: fits.hdu.base.BaseHDU, index: int) -> str:
    name = (hdu.name or "").strip()
    if name:
        return name
    return f"HDU {index}"


def _find_manifest_hdu(
    hdul: fits.HDUList,
) -> fits.hdu.base.BaseHDU | None:
    for hdu in hdul:
        if (hdu.name or "").upper() == MANIFEST_EXTNAME and hdu.data is not None:
            return hdu
    return None


def _decode_manifest(hdu: fits.hdu.base.BaseHDU) -> dict[str, Any]:
    data = np.asarray(hdu.data, dtype=np.uint8)
    payload = data.tobytes().decode("utf-8")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _load_manifest(
    hdul: fits.HDUList,
    path: Path,
) -> tuple[Manifest | None, dict[str, Any] | None, str, list[Issue]]:
    hdu = _find_manifest_hdu(hdul)
    if hdu is None:
        return None, None, "absent", []

    try:
        payload = _decode_manifest(hdu)
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            None,
            "invalid",
            [
                Issue(
                    "error",
                    "manifest.decode",
                    f"could not decode {MANIFEST_EXTNAME} extension: {exc}",
                )
            ],
        )

    try:
        manifest = Manifest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            payload,
            "invalid",
            [
                Issue(
                    "warning",
                    "manifest.invalid",
                    f"manifest does not match current schema; showing raw fields: {exc}",
                )
            ],
        )

    return manifest, payload, "present", []


def _axis_from_descriptor(axis: Any) -> AxisReport:
    coordinates = None
    if getattr(axis, "coordinates", None) is not None:
        coordinates = axis.coordinates.model_dump(mode="json")
    return AxisReport(
        name=axis.name,
        size=axis.size,
        labels=axis.labels,
        unit=axis.unit,
        coordinates=coordinates,
        source="manifest",
    )


def _array_from_descriptor(
    descriptor: ArrayDescriptor,
    hdu: fits.hdu.base.BaseHDU | None,
    issues: list[Issue],
) -> ArrayReport:
    shape = descriptor.shape
    if hdu is None:
        issues.append(
            Issue(
                "error",
                "array.missing",
                f"array {descriptor.name!r} has no data-bearing HDU",
            )
        )
    else:
        actual = tuple(int(size) for size in hdu.data.shape)
        if actual != descriptor.shape:
            issues.append(
                Issue(
                    "error",
                    "array.shape",
                    f"array {descriptor.name!r} shape {actual} does not match "
                    f"manifest shape {descriptor.shape}",
                )
            )
    return ArrayReport(
        name=descriptor.name,
        role=descriptor.role,
        dtype=descriptor.dtype,
        shape=shape,
        axes=[_axis_from_descriptor(axis) for axis in descriptor.axes],
        source="manifest",
    )


def _hdus_by_name(
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
) -> dict[str, fits.hdu.base.BaseHDU]:
    return {(hdu.name or "").upper(): hdu for hdu in data_hdus}


def _arrays_from_manifest(
    manifest: Manifest,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    issues: list[Issue],
) -> list[ArrayReport]:
    by_name = _hdus_by_name(data_hdus)
    if not manifest.arrays:
        issues.append(
            Issue("warning", "manifest.arrays", "manifest contains no arrays")
        )
        return _generic_arrays(data_hdus)
    return [
        _array_from_descriptor(descriptor, by_name.get(descriptor.name.upper()), issues)
        for descriptor in manifest.arrays
    ]


def _legacy_axis(
    name: str,
    size: int,
    *,
    labels: tuple[str, ...] | None = None,
    unit: str | None = None,
) -> AxisReport:
    return AxisReport(
        name=name,
        size=size,
        labels=labels,
        unit=unit,
        source="legacy manifest",
    )


def _arrays_from_legacy_manifest(
    payload: Mapping[str, Any],
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    issues: list[Issue],
) -> list[ArrayReport]:
    by_name = _hdus_by_name(data_hdus)
    raw_arrays = payload.get("arrays")
    if not isinstance(raw_arrays, list):
        issues.append(
            Issue("warning", "manifest.arrays", "legacy manifest contains no arrays")
        )
        return _generic_arrays(data_hdus)

    reports: list[ArrayReport] = []
    for item in raw_arrays:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "PRIMARY"))
        hdu = by_name.get(name.upper())
        data_shape = (
            tuple(int(size) for size in hdu.data.shape)
            if hdu is not None and hdu.data is not None
            else ()
        )
        raw_shape = item.get("shape")
        shape = (
            tuple(int(size) for size in raw_shape)
            if isinstance(raw_shape, (list, tuple))
            else data_shape
        )
        if not shape and hdu is not None and hdu.data is not None:
            shape = data_shape
            issues.append(
                Issue(
                    "warning",
                    "manifest.shape",
                    f"manifest shape missing for {name!r}; inferred from FITS",
                )
            )
        if data_shape and shape and data_shape != shape:
            issues.append(
                Issue(
                    "error",
                    "array.shape",
                    f"array {name!r} shape {data_shape} does not match "
                    f"manifest shape {shape}",
                )
            )

        raw_axes = item.get("axes", [])
        axes: list[AxisReport] = []
        if isinstance(raw_axes, list):
            for index, axis in enumerate(raw_axes):
                size = shape[index] if index < len(shape) else 0
                if isinstance(axis, Mapping):
                    axes.append(
                        AxisReport(
                            name=str(axis.get("name", f"axis_{index}")),
                            size=int(axis.get("size", size) or size),
                            labels=(
                                tuple(axis["labels"])
                                if isinstance(axis.get("labels"), list)
                                else None
                            ),
                            unit=axis.get("unit"),
                            source="legacy manifest",
                        )
                    )
                else:
                    axes.append(_legacy_axis(str(axis), size))
        if any(axis.source == "legacy manifest" for axis in axes):
            issues.append(
                Issue(
                    "warning",
                    "manifest.axes",
                    f"axis labels unavailable for legacy manifest array {name!r}",
                )
            )

        reports.append(
            ArrayReport(
                name=name,
                role=item.get("role"),
                dtype=item.get("dtype"),
                shape=shape,
                axes=axes,
                source="legacy manifest",
            )
        )

    return reports or _generic_arrays(data_hdus)


def _generic_axes(shape: Sequence[int]) -> list[AxisReport]:
    return [
        AxisReport(
            name=f"axis_{index}",
            size=int(size),
            source="fits header",
        )
        for index, size in enumerate(shape)
    ]


def _generic_arrays(
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
) -> list[ArrayReport]:
    reports: list[ArrayReport] = []
    for index, hdu in enumerate(data_hdus):
        shape = tuple(int(size) for size in hdu.data.shape)
        reports.append(
            ArrayReport(
                name=_hdu_name(hdu, index),
                role=None,
                dtype=str(hdu.data.dtype),
                shape=shape,
                axes=_generic_axes(shape),
                source="fits header",
            )
        )
    return reports


def _trial_axes(shape: Sequence[int]) -> list[AxisReport]:
    if len(shape) != 5:
        return _generic_axes(shape)
    return [
        AxisReport("channel", int(shape[0]), source="inferred"),
        AxisReport("frame", int(shape[1]), unit="frame", source="inferred"),
        AxisReport("y", int(shape[2]), unit="pixel", source="inferred"),
        AxisReport("x", int(shape[3]), unit="pixel", source="inferred"),
        AxisReport(
            "direction",
            int(shape[4]),
            labels=("forward", "reverse"),
            source="inferred",
        ),
    ]


def _timing_axes(shape: Sequence[int], *, lines: bool = False) -> list[AxisReport]:
    if lines:
        return [AxisReport("line", int(shape[0]), unit="sample", source="inferred")]
    if len(shape) != 2:
        return _generic_axes(shape)
    return [
        AxisReport(
            "axis",
            int(shape[0]),
            labels=("slow", "fast"),
            source="inferred",
        ),
        AxisReport("coefficient", int(shape[1]), source="inferred"),
    ]


def _legacy_trial_arrays(
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    issues: list[Issue],
) -> list[ArrayReport]:
    first = data_hdus[0]
    shape = tuple(int(size) for size in first.data.shape)
    reports = [
        ArrayReport(
            name="PRIMARY",
            role="frames",
            dtype=str(first.data.dtype),
            shape=shape,
            axes=_trial_axes(shape),
            source="inferred",
        )
    ]

    named = _hdus_by_name(data_hdus)
    fit_params = named.get("FIT_PARAMS")
    line_starts = named.get("LINE_STARTS")
    if fit_params is None or line_starts is None:
        if len(data_hdus) >= 3:
            fit_params = data_hdus[1]
            line_starts = data_hdus[2]
        else:
            fit_params = None
            line_starts = None

    if fit_params is not None:
        fit_shape = tuple(int(size) for size in fit_params.data.shape)
        reports.append(
            ArrayReport(
                name="FIT_PARAMS",
                role="desinusoid_parameters",
                dtype=str(fit_params.data.dtype),
                shape=fit_shape,
                axes=_timing_axes(fit_shape),
                source="inferred",
            )
        )
        if fit_shape[:1] != (2,) or len(fit_shape) < 2 or fit_shape[1] < 5:
            issues.append(
                Issue(
                    "error",
                    "FIT_PARAMS.shape",
                    f"FIT_PARAMS shape {fit_shape} is not 2 x >=5",
                )
            )

    if line_starts is not None:
        line_shape = tuple(int(size) for size in line_starts.data.shape)
        reports.append(
            ArrayReport(
                name="LINE_STARTS",
                role="line_start_samples",
                dtype=str(line_starts.data.dtype),
                shape=line_shape,
                axes=_timing_axes(line_shape, lines=True),
                source="inferred",
            )
        )

    return reports


def _looks_like_legacy_trial(
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    header: Mapping[str, Any],
) -> bool:
    if not data_hdus:
        return False
    named = {(hdu.name or "").upper() for hdu in data_hdus}
    if "FIT_PARAMS" in named and "LINE_STARTS" in named:
        return True
    return (
        _lookup(header, "n_frames") is not None
        and _lookup(header, "n_pixels_vertical") is not None
        and _lookup(header, "n_pixels_horizontal_int") is not None
    )


def _header_trial_details(header: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "n_frames",
        "n_samples_original",
        "time_interval",
        "n_pixels_vertical",
        "n_pixels_horizontal_int",
        "scanner_offset_x",
        "scanner_offset_y",
        "Non_flyback_start",
        "Non_flyback_end",
    )
    details = {key: _lookup(header, key) for key in keys}
    channels = {
        key: value
        for key, value in header.items()
        if key.casefold().startswith("channel_")
    }
    if channels:
        details["channels"] = channels
    return {key: value for key, value in details.items() if value is not None}


def _find_yaml(path: Path) -> Path | None:
    for suffix in (".yaml", ".yml"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _load_sidecar_manifest(
    path: Path,
) -> tuple[Manifest | None, dict[str, Any] | None]:
    sidecar = _find_yaml(path)
    if sidecar is None:
        return None, None
    payload = _load_yaml(sidecar)
    if payload is None or not (
        "manifest_version" in payload
        or ("schema_name" in payload and "arrays" in payload)
    ):
        return None, None
    try:
        return Manifest.model_validate(payload), payload
    except Exception:  # noqa: BLE001
        return None, payload


def _sidecar_status(
    path: Path,
    manifest: Manifest | None,
    issues: list[Issue],
) -> str:
    sidecar = _find_yaml(path)
    if sidecar is None:
        return "absent"
    payload = _load_yaml(sidecar)
    if payload is None:
        issues.append(
            Issue(
                "warning",
                "sidecar.invalid",
                f"{sidecar.name} is not a valid YAML mapping",
            )
        )
        return "invalid"
    if manifest is None:
        return "present"
    try:
        sidecar_manifest = Manifest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            Issue(
                "warning",
                "sidecar.invalid",
                f"{sidecar.name} does not match the current manifest schema: {exc}",
            )
        )
        return "invalid"
    if sidecar_manifest != manifest:
        issues.append(
            Issue(
                "warning",
                "sidecar.mismatch",
                f"{sidecar.name} does not match the MANIFEST extension",
            )
        )
        return "mismatch"
    return "present"


def _inspect_schema(
    path: Path,
    hdul: fits.HDUList,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    header: Mapping[str, Any],
    issues: list[Issue],
    *,
    sidecar_manifest: Manifest | None = None,
    sidecar_payload: dict[str, Any] | None = None,
) -> InspectionResult:
    manifest, payload, status, manifest_issues = _load_manifest(hdul, path)
    issues.extend(manifest_issues)
    if manifest is None and payload is None and sidecar_payload is not None:
        manifest = sidecar_manifest
        payload = sidecar_payload
        status = "sidecar" if sidecar_manifest is not None else "sidecar_invalid"
        issues.append(
            Issue(
                "warning",
                "manifest.sidecar",
                "schema declared by the YAML sidecar; no embedded MANIFEST extension",
            )
        )

    schema_name = manifest.schema_name if manifest is not None else None
    schema_version = manifest.schema_version if manifest is not None else None
    if payload is not None:
        schema_name = str(payload.get("schema_name") or schema_name or "") or None
        schema_version = (
            str(payload.get("schema_version") or schema_version or "") or None
        )

    marker_schema = _lookup(header, SCHEMA_CARD)
    marker_version = _lookup(header, SCHEMA_VERSION_CARD)
    if marker_schema is None:
        issues.append(
            Issue("warning", "marker.schema", f"{SCHEMA_CARD} marker is missing")
        )
    elif schema_name and str(marker_schema) != schema_name:
        issues.append(
            Issue(
                "error",
                "marker.schema",
                f"{SCHEMA_CARD}={marker_schema!r} does not match manifest "
                f"schema_name={schema_name!r}",
            )
        )
    if marker_version is None:
        issues.append(
            Issue(
                "warning",
                "marker.version",
                f"{SCHEMA_VERSION_CARD} marker is missing",
            )
        )
    elif schema_version and str(marker_version) != schema_version:
        issues.append(
            Issue(
                "error",
                "marker.version",
                f"{SCHEMA_VERSION_CARD}={marker_version!r} does not match "
                f"manifest schema_version={schema_version!r}",
            )
        )

    if manifest is not None:
        arrays = _arrays_from_manifest(manifest, data_hdus, issues)
        metadata_keys = sorted(str(key) for key in manifest.metadata)
        provenance = manifest.provenance.model_dump(mode="json")
        details = {"capabilities": manifest.capabilities}
    elif payload is not None:
        arrays = _arrays_from_legacy_manifest(payload, data_hdus, issues)
        raw_metadata = payload.get("metadata")
        metadata_keys = (
            sorted(str(key) for key in raw_metadata)
            if isinstance(raw_metadata, Mapping)
            else []
        )
        raw_provenance = payload.get("provenance")
        provenance = (
            dict(raw_provenance) if isinstance(raw_provenance, Mapping) else None
        )
        details = {"manifest_payload": payload}
    else:
        arrays = _generic_arrays(data_hdus)
        metadata_keys = []
        provenance = None
        details = {}

    if schema_name == "trace" and arrays:
        raw_metadata = (
            manifest.metadata
            if manifest is not None
            else (payload.get("metadata") if payload is not None else None)
        )
        cols = raw_metadata.get("cols") if isinstance(raw_metadata, Mapping) else None
        field_axis = next(
            (axis for axis in arrays[0].axes if axis.name == "field"),
            None,
        )
        if cols is not None and field_axis is not None:
            col_count = len(str(cols).split(","))
            if col_count != field_axis.size:
                issues.append(
                    Issue(
                        "error",
                        "trace.cols",
                        f"cols has {col_count} names but field axis has "
                        f"{field_axis.size} entries",
                    )
                )

    sidecar_status = _sidecar_status(path, manifest, issues)
    if status == "absent":
        issues.append(
            Issue(
                "warning",
                "manifest.absent",
                f"no {MANIFEST_EXTNAME} extension",
            )
        )

    return InspectionResult(
        path=path,
        kind="schema",
        schema_name=schema_name,
        schema_version=schema_version,
        manifest_status=status,
        sidecar_status=sidecar_status,
        arrays=arrays,
        metadata_keys=metadata_keys,
        provenance=provenance,
        details=details,
        issues=issues,
    )


def _inspect_legacy_trial(
    path: Path,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    header: Mapping[str, Any],
    issues: list[Issue],
) -> InspectionResult:
    issues.append(
        Issue(
            "warning",
            "manifest.absent",
            "no MANIFEST extension; treating as a legacy processed trial",
        )
    )
    issues.append(
        Issue(
            "warning",
            "axes.inferred",
            "trial axes are inferred from the legacy layout",
        )
    )
    arrays = _legacy_trial_arrays(data_hdus, issues)
    details = _header_trial_details(header)
    details["layout"] = (
        "named timing HDUs"
        if {"FIT_PARAMS", "LINE_STARTS"}
        <= {(hdu.name or "").upper() for hdu in data_hdus}
        else "legacy positional timing HDUs"
    )

    if arrays and arrays[0].shape:
        shape = arrays[0].shape
        n_frames = _header_int(header, "n_frames")
        n_vertical = _header_int(header, "n_pixels_vertical")
        n_horizontal_int = _header_int(header, "n_pixels_horizontal_int")
        if n_frames is not None and len(shape) > 1 and shape[1] != n_frames:
            issues.append(
                Issue(
                    "error",
                    "n_frames",
                    f"n_frames={n_frames} does not match frames shape {shape}",
                )
            )
        if n_vertical is not None and len(shape) > 2 and shape[2] != n_vertical:
            issues.append(
                Issue(
                    "error",
                    "n_pixels_vertical",
                    f"n_pixels_vertical={n_vertical} does not match frames shape {shape}",
                )
            )
        if (
            n_horizontal_int is not None
            and len(shape) > 3
            and shape[3] != n_horizontal_int // 2
        ):
            issues.append(
                Issue(
                    "error",
                    "n_pixels_horizontal_int",
                    f"n_pixels_horizontal_int={n_horizontal_int} does not match "
                    f"frames shape {shape}",
                )
            )

    sidecar = _find_yaml(path)
    return InspectionResult(
        path=path,
        kind="legacy_trial",
        manifest_status="absent",
        sidecar_status="present" if sidecar is not None else "absent",
        arrays=arrays,
        details=details,
        issues=issues,
    )


_LEGACY_DERIVED_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_EYEMOTION_PER_PIXEL_COARSE", "coarse_motion"),
    ("_EYEMOTION_PER_PIXEL_FRAME_POS", "frame_positions"),
    ("_EYEMOTION_PER_PIXEL", "eye_motion_per_pixel"),
    ("_EYEMOTION_PER_ROW", "eye_motion_per_row"),
    ("_METRICS", "metrics"),
    ("_TRACE", "trace"),
    ("MONTAGE_MOTION", "montage_motion"),
    ("_CORRECTED", "corrected_frames"),
)


def _legacy_derived_kind(path: Path) -> str | None:
    name = path.name.upper()
    for suffix, kind in _LEGACY_DERIVED_SUFFIXES:
        if suffix in name:
            return kind
    return None


def _legacy_derived_axes(
    kind: str,
    shape: Sequence[int],
    header: Mapping[str, Any],
    issues: list[Issue],
) -> tuple[list[AxisReport], str | None]:
    if kind == "eye_motion_per_pixel":
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport("y", int(shape[1]), unit="pixel", source="inferred"),
            AxisReport("x", int(shape[2]), unit="pixel", source="inferred"),
            AxisReport("axis", int(shape[3]), labels=("y", "x"), source="inferred"),
        ]
        return axes, "per_pixel_positions"
    if kind == "eye_motion_per_row":
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport("row", int(shape[1]), unit="pixel", source="inferred"),
            AxisReport("axis", int(shape[2]), labels=("y", "x"), source="inferred"),
        ]
        return axes, "per_row_positions"
    if kind == "coarse_motion":
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport("axis", int(shape[1]), labels=("y", "x"), source="inferred"),
        ]
        return axes, "coarse_motion"
    if kind == "frame_positions":
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport(
                "bound",
                int(shape[1]),
                labels=BOUND_LABELS,
                unit="pixel",
                source="inferred",
            ),
        ]
        if shape[1] != len(BOUND_LABELS):
            issues.append(
                Issue(
                    "error",
                    "frame_positions.bound",
                    f"frame_positions has {shape[1]} bounds; expected "
                    f"{len(BOUND_LABELS)}",
                )
            )
        return axes, "frame_positions"
    if kind == "metrics":
        labels = ("ncc", "blink", "std_dev", "mean") if shape[1] == 4 else None
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport("metric", int(shape[1]), labels=labels, source="inferred"),
        ]
        return axes, "metrics"
    if kind == "montage_motion":
        axes = [
            AxisReport("y", int(shape[0]), unit="pixel", source="inferred"),
            AxisReport("x", int(shape[1]), unit="pixel", source="inferred"),
            AxisReport("axis", int(shape[2]), labels=("y", "x"), source="inferred"),
        ]
        return axes, "motion_field"
    if kind == "corrected_frames":
        axes = [
            AxisReport("frame", int(shape[0]), unit="frame", source="inferred"),
            AxisReport("y", int(shape[1]), unit="pixel", source="inferred"),
            AxisReport("x", int(shape[2]), unit="pixel", source="inferred"),
        ]
        return axes, "corrected_frames"
    if kind == "trace":
        raw_cols = _lookup(header, "cols")
        if isinstance(raw_cols, str):
            labels = tuple(part.strip() for part in raw_cols.split(","))
        elif shape[0] == len(TRACE_FIELD_LABELS):
            labels = TRACE_FIELD_LABELS
        else:
            labels = None
        axes = [
            AxisReport("field", int(shape[0]), labels=labels, source="inferred"),
            AxisReport("sample", int(shape[1]), unit="sample", source="inferred"),
        ]
        if labels is not None and len(labels) != shape[0]:
            issues.append(
                Issue(
                    "error",
                    "trace.cols",
                    f"cols/field labels have {len(labels)} entries but field axis "
                    f"has {shape[0]} entries",
                )
            )
        return axes, "eye_motion_trace"
    return _generic_axes(shape), None


def _inspect_legacy_derived(
    path: Path,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    header: Mapping[str, Any],
    kind: str,
    issues: list[Issue],
) -> InspectionResult:
    first = data_hdus[0]
    shape = tuple(int(size) for size in first.data.shape)
    issues.append(
        Issue(
            "warning",
            "manifest.absent",
            "no MANIFEST extension; treating as a legacy derived array",
        )
    )
    issues.append(
        Issue(
            "warning",
            "axes.inferred",
            f"{kind} axes are inferred from the filename suffix and array shape",
        )
    )
    axes, role = _legacy_derived_axes(kind, shape, header, issues)
    sidecar = _find_yaml(path)
    return InspectionResult(
        path=path,
        kind="legacy_derived",
        schema_name=kind,
        manifest_status="absent",
        sidecar_status="present" if sidecar is not None else "absent",
        arrays=[
            ArrayReport(
                name=_hdu_name(first, 0),
                role=role,
                dtype=str(first.data.dtype),
                shape=shape,
                axes=axes,
                source="inferred",
            )
        ],
        details={"inferred_from": "filename suffix"},
        issues=issues,
    )


def _letter_to_index(letter: str, channel_list: Sequence[int]) -> int | None:
    try:
        scope_index = ord(str(letter).upper()) - ord("A")
        return list(channel_list).index(scope_index)
    except (TypeError, ValueError):
        return None


def _raw_channel_roles(
    metadata: Mapping[str, Any],
    channel_list: Sequence[int],
) -> dict[int, str]:
    role_by_label: dict[str, str] = {}
    for key, role in (
        ("slow_channel", "slow_scan"),
        ("fast_channel", "fast_scan"),
        ("confocal_channel", "confocal"),
        ("trigger_channel", "photodiode"),
        ("photodiode_channel", "photodiode"),
    ):
        value = metadata.get(key)
        if value is not None:
            role_by_label[str(value).upper()] = role
    roles: dict[int, str] = {}
    for row, scope_index in enumerate(channel_list):
        label = chr(ord("A") + int(scope_index))
        roles[row] = role_by_label.get(label, "extra")
    return roles


def _raw_frame_rate(
    metadata: Mapping[str, Any],
    n_pixels_vertical: int | None,
) -> float | None:
    value = metadata.get("AOSLO_frame_rate")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    frequency = metadata.get("resonant_frequency_pico")
    if frequency is None:
        frequency = metadata.get("resonant_frequency")
    if frequency is not None and n_pixels_vertical:
        try:
            return float(frequency) / float(n_pixels_vertical)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _inspect_raw(
    path: Path,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    header: Mapping[str, Any],
    issues: list[Issue],
) -> InspectionResult:
    yaml_path = _find_yaml(path)
    metadata = _load_yaml(yaml_path) if yaml_path is not None else None
    metadata = metadata or {}
    first = data_hdus[0]
    shape = tuple(int(size) for size in first.data.shape)

    raw_issues = list(issues)
    raw_issues.append(
        Issue(
            "warning",
            "manifest.absent",
            "raw acquisition is not yet schema-managed; no MANIFEST extension",
        )
    )
    version = metadata.get("metadata_version")
    try:
        version_int = int(version) if version is not None else None
    except (TypeError, ValueError):
        version_int = None
        raw_issues.append(
            Issue(
                "warning",
                "metadata.version",
                f"metadata_version {version!r} is not an integer",
            )
        )
    if version is None:
        raw_issues.append(
            Issue(
                "warning",
                "metadata.legacy",
                "acquisition YAML has no metadata_version; treating as legacy flat metadata",
            )
        )
    elif version_int == 1:
        raw_issues.append(
            Issue(
                "warning",
                "metadata.legacy",
                "acquisition YAML uses legacy metadata_version 1",
            )
        )

    channel_list = metadata.get("channel_list")
    channel_list = (
        [int(value) for value in channel_list] if isinstance(channel_list, list) else []
    )
    channel_roles = _raw_channel_roles(metadata, channel_list)
    n_vertical = _header_int(metadata, "n_pixels_vertical")
    frame_rate = _raw_frame_rate(metadata, n_vertical)

    axes = [
        AxisReport(
            "channel",
            shape[0] if shape else 0,
            labels=tuple(channel_roles[row] for row in sorted(channel_roles)),
            source="acquisition YAML" if metadata else "fits header",
        )
    ]
    if len(shape) > 1:
        axes.append(
            AxisReport(
                "sample",
                shape[1],
                unit="sample",
                coordinates=(
                    {
                        "start": 0.0,
                        "step": float(metadata["time_interval"]),
                        "unit": "s",
                    }
                    if metadata.get("time_interval") is not None
                    else None
                ),
                source="acquisition YAML" if metadata else "fits header",
            )
        )

    if channel_list and len(channel_list) != shape[0]:
        raw_issues.append(
            Issue(
                "error",
                "channel_list",
                f"channel_list has {len(channel_list)} entries but data has "
                f"{shape[0]} channels",
            )
        )

    details: dict[str, Any] = {
        "metadata_path": str(yaml_path) if yaml_path is not None else None,
        "metadata_version": version,
        "channel_list": channel_list,
        "channel_roles": channel_roles,
        "slow_channel": metadata.get("slow_channel"),
        "fast_channel": metadata.get("fast_channel"),
        "confocal_channel": metadata.get("confocal_channel"),
        "trigger_channel": metadata.get("trigger_channel"),
        "n_pixels_horizontal_float": metadata.get("n_pixels_horizontal_float"),
        "n_pixels_vertical": n_vertical,
        "n_lines_flyback": metadata.get("n_lines_flyback"),
        "galvo_direction": metadata.get("galvo_direction"),
        "time_interval": metadata.get("time_interval"),
        "down_sample_ratio": metadata.get("down_sample_ratio"),
        "frame_rate_hz": frame_rate,
        "pixel_scale_x_arcmin": metadata.get("Pixel_size_x_arcmin"),
        "pixel_scale_y_arcmin": metadata.get("Pixel_size_y_arcmin"),
    }

    return InspectionResult(
        path=path,
        kind="raw_acquisition",
        schema_name="raw",
        manifest_status="absent",
        sidecar_status="present" if yaml_path is not None else "absent",
        arrays=[
            ArrayReport(
                name=_hdu_name(first, 0),
                role="raw_samples",
                dtype=str(first.data.dtype),
                shape=shape,
                axes=axes,
                source="acquisition YAML" if metadata else "fits header",
            )
        ],
        metadata_keys=sorted(str(key) for key in metadata),
        details={
            key: value
            for key, value in details.items()
            if value is not None or key == "metadata_version"
        },
        issues=raw_issues,
    )


def _inspect_unknown(
    path: Path,
    data_hdus: Sequence[fits.hdu.base.BaseHDU],
    issues: list[Issue],
) -> InspectionResult:
    issues.append(
        Issue(
            "warning",
            "schema.unknown",
            "no schema marker, manifest, legacy trial layout or acquisition YAML detected",
        )
    )
    return InspectionResult(
        path=path,
        kind="unknown",
        manifest_status="absent",
        sidecar_status="present" if _find_yaml(path) is not None else "absent",
        arrays=_generic_arrays(data_hdus),
        details={
            "hdus": [
                {
                    "name": _hdu_name(hdu, index),
                    "shape": list(hdu.data.shape),
                    "dtype": str(hdu.data.dtype),
                }
                for index, hdu in enumerate(data_hdus)
            ]
        },
        issues=issues,
    )


def inspect_path(path: str | Path) -> InspectionResult:
    """Inspect one FITS file without raising for malformed or legacy data."""
    fits_path = Path(path)
    issues: list[Issue] = []
    try:
        with fits.open(fits_path) as hdul:
            data_hdus = _data_hdus(hdul)
            if not data_hdus:
                return InspectionResult(
                    path=fits_path,
                    kind="unknown",
                    issues=[
                        Issue(
                            "error",
                            "fits.data",
                            "no data-bearing HDU found",
                        )
                    ],
                )
            header = dict(data_hdus[0].header)
            has_manifest = _find_manifest_hdu(hdul) is not None
            has_schema_marker = _lookup(header, SCHEMA_CARD) is not None
            if has_manifest or has_schema_marker:
                return _inspect_schema(fits_path, hdul, data_hdus, header, issues)

            sidecar_manifest, sidecar_payload = _load_sidecar_manifest(fits_path)
            if sidecar_payload is not None:
                return _inspect_schema(
                    fits_path,
                    hdul,
                    data_hdus,
                    header,
                    issues,
                    sidecar_manifest=sidecar_manifest,
                    sidecar_payload=sidecar_payload,
                )

            if _looks_like_legacy_trial(data_hdus, header):
                return _inspect_legacy_trial(fits_path, data_hdus, header, issues)

            yaml_path = _find_yaml(fits_path)
            yaml_payload = _load_yaml(yaml_path) if yaml_path is not None else None
            if yaml_payload is not None and "channel_list" in yaml_payload:
                return _inspect_raw(fits_path, data_hdus, header, issues)

            legacy_kind = _legacy_derived_kind(fits_path)
            if legacy_kind is not None:
                return _inspect_legacy_derived(
                    fits_path,
                    data_hdus,
                    header,
                    legacy_kind,
                    issues,
                )

            if (
                len(data_hdus) == 1
                and data_hdus[0].data.ndim == 2
                and fits_path.stem.lower().startswith("data_stream")
            ):
                return _inspect_raw(fits_path, data_hdus, header, issues)

            return _inspect_unknown(fits_path, data_hdus, issues)
    except Exception as exc:  # noqa: BLE001
        return InspectionResult(
            path=fits_path,
            kind="error",
            issues=[
                Issue(
                    "error",
                    "fits.open",
                    f"could not inspect file: {exc}",
                )
            ],
        )


def _format_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(_jsonable(value), sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(_jsonable(value))
    return str(value)


def _result_status(result: InspectionResult) -> str:
    if result.has_errors:
        return "ERROR"
    if result.has_warnings:
        return "WARN"
    return "OK"


def _render_text(results: Sequence[InspectionResult], *, summary: bool = False) -> str:
    lines: list[str] = []
    for result in results:
        if summary:
            schema = result.schema_name or result.kind
            version = f" {result.schema_version}" if result.schema_version else ""
            lines.append(f"{_result_status(result):5} {result.path}  {schema}{version}")
            continue

        lines.append(f"File: {result.path}")
        lines.append(f"Kind: {result.kind}")
        schema = result.schema_name or "legacy/unknown"
        if result.schema_version:
            schema += f" {result.schema_version}"
        lines.append(f"Schema: {schema}")
        lines.append(f"Manifest: {result.manifest_status}")
        lines.append(f"Sidecar: {result.sidecar_status}")

        if result.details:
            lines.append("Details:")
            for key, value in result.details.items():
                if key == "manifest_payload" and isinstance(value, Mapping):
                    continue
                lines.append(f"  {key}: {_format_value(value)}")

        if result.arrays:
            lines.append("Arrays:")
            for array in result.arrays:
                lines.append(
                    f"  {array.name}  role={array.role or '-'} "
                    f"dtype={array.dtype or '-'} shape={array.shape} "
                    f"source={array.source}"
                )
                for axis in array.axes:
                    labels = (
                        f"  labels={list(axis.labels)}"
                        if axis.labels is not None
                        else ""
                    )
                    unit = f"  unit={axis.unit}" if axis.unit else ""
                    lines.append(f"    {axis.name}: {axis.size}{unit}{labels}")

        if result.metadata_keys:
            lines.append(f"Metadata keys: {', '.join(result.metadata_keys)}")
        if result.provenance:
            lines.append(f"Provenance: {_format_value(result.provenance)}")
        if result.issues:
            lines.append("Issues:")
            for issue in result.issues:
                lines.append(
                    f"  {issue.severity.upper():7} {issue.code}: {issue.message}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def _render_html(results: Sequence[InspectionResult]) -> str:
    parts = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        "<title>NORI schema inspection</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#17202a}",
        "h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}",
        "table{border-collapse:collapse;margin:.5rem 0 1rem;width:100%}",
        "th,td{border:1px solid #cbd5e1;padding:.35rem .5rem;text-align:left}",
        "th{background:#f1f5f9}",
        ".ERROR{color:#b42318;font-weight:600}.WARN{color:#b54708;font-weight:600}",
        ".OK{color:#027a48;font-weight:600}",
        "code,pre{background:#f8fafc;padding:.15rem .25rem}",
        "</style></head><body>",
        "<h1>NORI schema inspection</h1>",
    ]
    for result in results:
        parts.append(f"<h2>{html.escape(str(result.path))}</h2>")
        parts.append(
            "<p>"
            f"<strong>Kind:</strong> {html.escape(result.kind)}<br>"
            f"<strong>Schema:</strong> "
            f"{html.escape(result.schema_name or 'legacy/unknown')} "
            f"{html.escape(result.schema_version or '')}<br>"
            f"<strong>Manifest:</strong> {html.escape(result.manifest_status)}<br>"
            f"<strong>Sidecar:</strong> {html.escape(result.sidecar_status)}"
            "</p>"
        )
        if result.arrays:
            parts.append(
                "<table><thead><tr><th>Array</th><th>Role</th><th>Dtype</th>"
                "<th>Shape</th><th>Source</th></tr></thead><tbody>"
            )
            for array in result.arrays:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(array.name)}</td>"
                    f"<td>{html.escape(array.role or '-')}</td>"
                    f"<td>{html.escape(array.dtype or '-')}</td>"
                    f"<td>{html.escape(str(array.shape))}</td>"
                    f"<td>{html.escape(array.source)}</td>"
                    "</tr>"
                )
                if array.axes:
                    parts.append(
                        '<tr><td colspan="5"><strong>Axes:</strong> '
                        + "<br>".join(
                            f"{html.escape(axis.name)}: {axis.size}"
                            + (
                                f" labels={html.escape(str(list(axis.labels)))}"
                                if axis.labels is not None
                                else ""
                            )
                            + (f" unit={html.escape(axis.unit)}" if axis.unit else "")
                            for axis in array.axes
                        )
                        + "</td></tr>"
                    )
            parts.append("</tbody></table>")
        if result.metadata_keys:
            parts.append(
                "<p><strong>Metadata keys:</strong> "
                + html.escape(", ".join(result.metadata_keys))
                + "</p>"
            )
        if result.provenance:
            parts.append(
                "<pre>"
                + html.escape(
                    json.dumps(_jsonable(result.provenance), indent=2, sort_keys=True)
                )
                + "</pre>"
            )
        if result.issues:
            parts.append(
                "<table><thead><tr><th>Severity</th><th>Code</th>"
                "<th>Message</th></tr></thead><tbody>"
            )
            for issue in result.issues:
                parts.append(
                    f'<tr class="{html.escape(issue.severity.upper())}">'
                    f"<td>{html.escape(issue.severity.upper())}</td>"
                    f"<td>{html.escape(issue.code)}</td>"
                    f"<td>{html.escape(issue.message)}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _expand_paths(paths: Sequence[str]) -> list[Path] | None:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.fits")))
            files.extend(sorted(path.rglob("*.fit")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"nori-schema-inspect: path not found: {value}", file=sys.stderr)
            return None

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``nori-schema-inspect``."""
    parser = argparse.ArgumentParser(
        prog="nori-schema-inspect",
        description="Inspect NORI FITS schemas, manifests and legacy layouts.",
    )
    parser.add_argument("paths", nargs="+", help="FITS file or directory")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="JSON output")
    output.add_argument("--html", action="store_true", help="HTML output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on warnings such as legacy or missing-manifest files",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="one line per file (text output only)",
    )
    args = parser.parse_args(argv)

    files = _expand_paths(args.paths)
    if files is None:
        return 2
    if not files:
        print("nori-schema-inspect: no FITS files found", file=sys.stderr)
        return 2

    results = [inspect_path(path) for path in files]
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    elif args.html:
        print(_render_html(results), end="")
    else:
        print(_render_text(results, summary=args.summary), end="")

    has_errors = any(result.has_errors for result in results)
    has_warnings = any(result.has_warnings for result in results)
    if has_errors or (args.strict and has_warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
