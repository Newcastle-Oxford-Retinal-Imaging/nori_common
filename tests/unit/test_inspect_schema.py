"""Tests for the nori-schema-inspect command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from nori_common.builders import build_array_payload, build_trial_payload
from nori_common.contracts import ArtifactProvenance, Manifest
from nori_common.fits import WritePolicy
from nori_common.inspect_schema import inspect_path, main
from nori_common.manifest import write_sidecar
from nori_common.writers import write_array_artifact, write_trial


def _trial_payload():
    frames = np.zeros((2, 3, 5, 4, 2), dtype=np.float32)
    fit_parameters = np.zeros((2, 5), dtype=np.float32)
    line_starts = np.arange(16, dtype=np.float32)
    return build_trial_payload(
        frames,
        fit_parameters,
        line_starts,
        n_samples_original=120,
        time_interval=1.0e-07,
        scanner_offset=(1.0, 2.0),
        non_flyback_rows=(1, 4),
        provenance=ArtifactProvenance(source_type="measured", creator="test"),
        channel_roles={0: "confocal", 1: "photodiode"},
    )


def _write_trial(path: Path) -> None:
    write_trial(
        path,
        _trial_payload(),
        policy=WritePolicy(),
        sidecar_path=path.with_suffix(".yaml"),
    )


def _write_array(path: Path, *, schema_name: str, role: str, data, **kwargs) -> None:
    payload = build_array_payload(
        data,
        schema_name=schema_name,
        role=role,
        provenance=ArtifactProvenance(source_type="measured", creator="test"),
        **kwargs,
    )
    write_array_artifact(
        path,
        payload,
        policy=WritePolicy(),
        sidecar_path=path.with_suffix(".yaml"),
    )


def _legacy_header() -> fits.Header:
    header = fits.Header()
    header["n_frames"] = 3
    header["n_samples_original"] = 120
    header["time_interval"] = 1.0e-07
    header["n_pixels_vertical"] = 5
    header["n_pixels_horizontal_int"] = 8
    header["scanner_offset_x"] = 2.0
    header["scanner_offset_y"] = 1.0
    header["Non_flyback_start"] = 1
    header["Non_flyback_end"] = 4
    return header


def _write_legacy_trial(path: Path, *, named: bool) -> None:
    frames = np.zeros((2, 3, 5, 4, 2), dtype=np.float32)
    fit_parameters = np.zeros((2, 5), dtype=np.float32)
    line_starts = np.arange(16, dtype=np.float32)
    if named:
        hdus = [
            fits.PrimaryHDU(frames, header=_legacy_header()),
            fits.ImageHDU(fit_parameters, name="FIT_PARAMS"),
            fits.ImageHDU(line_starts, name="LINE_STARTS"),
        ]
    else:
        hdus = [
            fits.PrimaryHDU(frames, header=_legacy_header()),
            fits.ImageHDU(fit_parameters),
            fits.ImageHDU(line_starts),
        ]
    fits.HDUList(hdus).writeto(path)


def _write_raw(path: Path, *, metadata_version: int | None) -> None:
    data = np.zeros((4, 16), dtype=np.int16)
    fits.PrimaryHDU(data).writeto(path)
    metadata = {
        "channel_list": [0, 1, 4, 7],
        "slow_channel": "A",
        "fast_channel": "B",
        "confocal_channel": "E",
        "trigger_channel": "H",
        "n_pixels_horizontal_float": 2.0,
        "n_pixels_vertical": 4,
        "n_lines_flyback": 1,
        "galvo_direction": "rising",
        "time_interval": 1.0e-07,
        "down_sample_ratio": 1,
        "AOSLO_frame_rate": 30.0,
        "Pixel_size_x_arcmin": 0.1,
        "Pixel_size_y_arcmin": 0.2,
    }
    if metadata_version is not None:
        metadata["metadata_version"] = metadata_version
    with path.with_suffix(".yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream)


def _write_old_manifest(path: Path, *, trace: bool = False) -> None:
    data = (
        np.zeros((9, 10), dtype=np.float32)
        if trace
        else np.zeros((2, 3, 5, 4, 2), dtype=np.float32)
    )
    payload = {
        "manifest_version": "1.0",
        "schema_name": "trace" if trace else "trial",
        "schema_version": "1.0",
        "arrays": [
            {
                "name": "PRIMARY",
                "role": "eye_motion_trace" if trace else "frames",
                "dtype": "float32",
                "axes": ["field", "sample"]
                if trace
                else [
                    "channel",
                    "frame",
                    "y",
                    "x",
                    "direction",
                ],
            }
        ],
        "metadata": ({"cols": "t,y,x,fn,fb,pd,bl,tk"} if trace else {"n_frames": 3}),
        "provenance": {"source_type": "measured", "creator": "test"},
    }
    header = fits.Header()
    header["NORIFMT_SCHEMA"] = payload["schema_name"]
    header["NORIFMT_SCHEMA_VERSION"] = "1.0"
    header["NORIFMT_MANIFEST"] = 1
    manifest_data = np.frombuffer(json.dumps(payload).encode("utf-8"), dtype=np.uint8)
    fits.HDUList(
        [
            fits.PrimaryHDU(data, header=header),
            fits.ImageHDU(manifest_data, name="MANIFEST"),
        ]
    ).writeto(path)


def test_inspects_current_trial(tmp_path):
    path = tmp_path / "trial.fits"
    _write_trial(path)

    result = inspect_path(path)

    assert result.kind == "schema"
    assert result.schema_name == "trial"
    assert result.schema_version == "1.0"
    assert result.manifest_status == "present"
    assert result.sidecar_status == "present"
    assert result.arrays[0].shape == (2, 3, 5, 4, 2)
    assert result.arrays[0].axes[0].labels == ("confocal", "photodiode")
    assert result.arrays[0].axes[4].labels == ("forward", "reverse")
    assert "n_frames" in result.metadata_keys
    assert not result.has_errors


def test_inspects_current_array(tmp_path):
    path = tmp_path / "eye_motion.fits"
    _write_array(
        path,
        schema_name="eye_motion_per_pixel",
        role="per_pixel_positions",
        data=np.zeros((3, 4, 5, 2), dtype=np.float32),
        metadata={"frame_count": 3, "source_channel": "confocal"},
    )

    result = inspect_path(path)

    assert result.kind == "schema"
    assert result.schema_name == "eye_motion_per_pixel"
    assert result.arrays[0].shape == (3, 4, 5, 2)
    assert result.arrays[0].axes[-1].labels == ("y", "x")
    assert "frame_count" in result.metadata_keys
    assert not result.has_errors


@pytest.mark.parametrize("named", [True, False])
def test_inspects_legacy_trial(tmp_path, named):
    path = tmp_path / "legacy.fits"
    _write_legacy_trial(path, named=named)

    result = inspect_path(path)

    assert result.kind == "legacy_trial"
    assert result.manifest_status == "absent"
    assert result.arrays[0].role == "frames"
    assert result.arrays[0].axes[4].labels == ("forward", "reverse")
    assert result.details["layout"] == (
        "named timing HDUs" if named else "legacy positional timing HDUs"
    )
    assert result.has_warnings
    assert not result.has_errors


@pytest.mark.parametrize("metadata_version", [2, 1, None])
def test_inspects_raw_acquisition(tmp_path, metadata_version):
    path = tmp_path / "data_stream_001.fits"
    _write_raw(path, metadata_version=metadata_version)

    result = inspect_path(path)

    assert result.kind == "raw_acquisition"
    assert result.schema_name == "raw"
    assert result.manifest_status == "absent"
    assert result.sidecar_status == "present"
    assert result.arrays[0].shape == (4, 16)
    assert result.arrays[0].axes[0].labels == (
        "slow_scan",
        "fast_scan",
        "confocal",
        "photodiode",
    )
    assert result.details["metadata_version"] == metadata_version
    if metadata_version != 2:
        assert any(issue.code == "metadata.legacy" for issue in result.issues)


def test_inspects_raw_without_yaml(tmp_path):
    path = tmp_path / "data_stream_002.fits"
    fits.PrimaryHDU(np.zeros((4, 16), dtype=np.int16)).writeto(path)

    result = inspect_path(path)

    assert result.kind == "raw_acquisition"
    assert result.sidecar_status == "absent"
    assert result.arrays[0].shape == (4, 16)
    assert any(issue.code == "manifest.absent" for issue in result.issues)


def test_inspects_old_manifest_with_string_axes(tmp_path):
    path = tmp_path / "old_manifest.fits"
    _write_old_manifest(path)

    result = inspect_path(path)

    assert result.kind == "schema"
    assert result.manifest_status == "invalid"
    assert result.arrays[0].shape == (2, 3, 5, 4, 2)
    assert [axis.name for axis in result.arrays[0].axes] == [
        "channel",
        "frame",
        "y",
        "x",
        "direction",
    ]
    assert any(issue.code == "manifest.invalid" for issue in result.issues)


def test_detects_trace_cols_mismatch_in_old_manifest(tmp_path):
    path = tmp_path / "old_trace.fits"
    _write_old_manifest(path, trace=True)

    result = inspect_path(path)

    assert result.kind == "schema"
    assert any(issue.code == "trace.cols" for issue in result.issues)
    assert result.has_errors


def test_sidecar_mismatch_is_reported(tmp_path):
    path = tmp_path / "trial.fits"
    _write_trial(path)
    sidecar = path.with_suffix(".yaml")
    payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9"
    sidecar.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = inspect_path(path)

    assert result.sidecar_status == "mismatch"
    assert any(issue.code == "sidecar.mismatch" for issue in result.issues)


def test_uses_sidecar_manifest_when_embedded_manifest_is_absent(tmp_path):
    path = tmp_path / "trial.fits"
    payload = _trial_payload()
    header = fits.Header(payload.metadata.frame.to_header())
    fits.HDUList(
        [
            fits.PrimaryHDU(payload.frames, header=header),
            fits.ImageHDU(payload.fit_parameters, name="FIT_PARAMS"),
            fits.ImageHDU(payload.line_starts, name="LINE_STARTS"),
        ]
    ).writeto(path)
    manifest = Manifest(
        schema_name="trial",
        arrays=payload.metadata.arrays,
        metadata={"n_frames": 3},
        provenance=payload.metadata.provenance,
    )
    write_sidecar(path.with_suffix(".yaml"), manifest)

    result = inspect_path(path)

    assert result.kind == "schema"
    assert result.schema_name == "trial"
    assert result.manifest_status == "sidecar"
    assert result.sidecar_status == "present"
    assert result.arrays[0].axes[0].labels == ("confocal", "photodiode")
    assert any(issue.code == "manifest.sidecar" for issue in result.issues)
    assert not result.has_errors


def test_infers_legacy_derived_array_from_filename(tmp_path):
    path = tmp_path / "trial_EXTRACTED_EYEMOTION_PER_PIXEL.fits"
    fits.PrimaryHDU(np.zeros((3, 4, 5, 2), dtype=np.float32)).writeto(path)

    result = inspect_path(path)

    assert result.kind == "legacy_derived"
    assert result.schema_name == "eye_motion_per_pixel"
    assert result.arrays[0].shape == (3, 4, 5, 2)
    assert result.arrays[0].axes[-1].labels == ("y", "x")
    assert result.has_warnings
    assert not result.has_errors


def test_infers_legacy_trace_and_reports_cols_mismatch(tmp_path):
    path = tmp_path / "trial_EXTRACTED_TRACE.fits"
    header = fits.Header()
    header["cols"] = "t,y,x,fn,fb,pd,bl,tk"
    fits.PrimaryHDU(np.zeros((9, 20), dtype=np.float32), header=header).writeto(path)

    result = inspect_path(path)

    assert result.kind == "legacy_derived"
    assert result.schema_name == "trace"
    assert any(issue.code == "trace.cols" for issue in result.issues)
    assert result.has_errors


def test_unknown_fits(tmp_path):
    path = tmp_path / "unknown.fits"
    fits.PrimaryHDU(np.zeros(10)).writeto(path)

    result = inspect_path(path)

    assert result.kind == "unknown"
    assert result.has_warnings
    assert result.arrays[0].shape == (10,)


def test_directory_traversal_and_json(tmp_path, capsys):
    first = tmp_path / "one.fits"
    second = tmp_path / "nested" / "two.fits"
    second.parent.mkdir()
    _write_trial(first)
    _write_array(
        second,
        schema_name="metrics",
        role="metrics",
        data=np.zeros((3, 4), dtype=np.float32),
        metadata={"column_names": ("ncc", "blink", "std_dev", "mean")},
    )

    code = main([str(tmp_path), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert {item["kind"] for item in payload} == {"schema"}


def test_strict_turns_legacy_warning_into_failure(tmp_path):
    path = tmp_path / "legacy.fits"
    _write_legacy_trial(path, named=True)

    assert main([str(path), "--summary"]) == 0
    assert main([str(path), "--strict", "--summary"]) == 1


def test_html_output(tmp_path, capsys):
    path = tmp_path / "trial.fits"
    _write_trial(path)

    assert main([str(path), "--html"]) == 0
    output = capsys.readouterr().out
    assert "<html>" in output
    assert "eye_motion_per_pixel" not in output
    assert "PRIMARY" in output


def test_missing_path_returns_usage_error(capsys):
    code = main(["/definitely/not/a/real/path"])

    assert code == 2
    assert "path not found" in capsys.readouterr().err
