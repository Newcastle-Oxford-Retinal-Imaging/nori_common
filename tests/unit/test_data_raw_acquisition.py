"""Contract tests for raw acquisition capabilities."""

import numpy
import pytest

from nori_common.data import (
    AcquisitionProvenance,
    AcquisitionTiming,
    PixelScale,
    RawChannelRoles,
    RawData,
    RawScanGeometry,
    SchemaError,
    read_raw_acquisition,
)
from nori_common.adapters import RawAcquisitionData


def _data():
    return numpy.arange(4 * 16, dtype=numpy.int16).reshape(4, 16)


def _flat_metadata():
    return {
        "channel_list": [0, 1, 2, 3],
        "slow_channel": "A",
        "fast_channel": "B",
        "confocal_channel": "C",
        "photodiode_channel": "D",
        "n_pixels_vertical": 4,
        "n_pixels_horizontal_float": 2.0,
        "n_lines_flyback": 1,
        "galvo_direction": "rising",
        "time_interval": 1.0e-07,
        "down_sample_ratio": 2,
        "resonant_frequency_pico": 1200.0,
        "Pixel_size_x_arcmin": 0.11,
        "Pixel_size_y_arcmin": 0.12,
        "Experiment_name": "example",
    }


def test_flat_raw_acquisition_contract():
    raw = read_raw_acquisition(_data(), _flat_metadata())

    assert isinstance(raw, RawAcquisitionData)
    assert raw.supports(RawData)
    assert raw.supports(RawChannelRoles)
    assert raw.supports(RawScanGeometry)
    assert raw.supports(AcquisitionTiming)
    assert raw.supports(AcquisitionProvenance)
    assert raw.supports(PixelScale)

    numpy.testing.assert_array_equal(raw.require(RawData).data, _data())
    assert raw.require(RawChannelRoles).channel_roles == {
        0: "slow_scan",
        1: "fast_scan",
        2: "confocal",
        3: "photodiode",
    }
    assert raw.require(RawChannelRoles).slow_channel_index == 0
    assert raw.require(RawChannelRoles).fast_channel_index == 1
    assert raw.require(RawChannelRoles).confocal_channel_index == 2
    assert raw.require(RawChannelRoles).photodiode_channel_index == 3

    geometry = raw.require(RawScanGeometry)
    assert geometry.n_pixels_vertical == 4
    assert geometry.n_pixels_horizontal_float == 2.0
    assert geometry.n_lines_flyback == 1
    assert geometry.galvo_direction == "rising"

    timing = raw.require(AcquisitionTiming)
    assert timing.sample_interval_s == pytest.approx(1.0e-07)
    assert timing.down_sample_ratio == 2
    assert timing.frame_rate_hz == pytest.approx(300.0)
    assert timing.n_samples_total == 16

    assert raw.require(PixelScale).pixel_size_arcmin_x == pytest.approx(0.11)
    assert raw.require(PixelScale).pixel_size_arcmin_y == pytest.approx(0.12)
    assert raw.require(AcquisitionProvenance).metadata_version is None
    assert raw.source_metadata["Experiment_name"] == "example"
    assert raw.extras.get_extra("acq.Experiment_name") == "example"


def test_metadata_v2_raw_acquisition_contract():
    metadata = _flat_metadata()
    metadata.update(
        {
            "metadata_version": 2,
            "system": "OXF",
            "profile": "single_pd",
            "wiring_name": "single_pd",
            "channel_list": [0, 1, 4, 7],
            "confocal_channel": "E",
            "trigger_channel": "H",
            "photodiode_channel": None,
            "AOSLO_frame_rate": 30.0,
            "wiring": {
                "name": "single_pd",
                "slow_channel": "A",
                "fast_channel": "B",
                "confocal_channel": "E",
                "trigger_channel": "H",
            },
        }
    )

    raw = read_raw_acquisition(_data(), metadata)

    assert raw.require(AcquisitionProvenance).metadata_version == 2
    assert raw.require(AcquisitionProvenance).system_id == "OXF"
    assert raw.require(AcquisitionProvenance).profile_id == "single_pd"
    assert raw.require(AcquisitionProvenance).wiring_id == "single_pd"
    assert raw.require(AcquisitionTiming).frame_rate_hz == pytest.approx(30.0)

    roles = raw.require(RawChannelRoles)
    assert roles.channel_roles == {
        0: "slow_scan",
        1: "fast_scan",
        2: "confocal",
        3: "photodiode",
    }
    assert roles.confocal_channel_index == 2
    assert roles.photodiode_channel_index == 3


def test_unknown_metadata_version_raises():
    metadata = _flat_metadata()
    metadata["metadata_version"] = 3

    with pytest.raises(SchemaError):
        read_raw_acquisition(_data(), metadata)


def test_missing_channel_raises():
    metadata = _flat_metadata()
    del metadata["fast_channel"]

    with pytest.raises(SchemaError):
        read_raw_acquisition(_data(), metadata)


def test_channel_list_must_match_data_rows():
    metadata = _flat_metadata()
    metadata["channel_list"] = [0, 1, 2]

    with pytest.raises(SchemaError):
        read_raw_acquisition(_data(), metadata)
