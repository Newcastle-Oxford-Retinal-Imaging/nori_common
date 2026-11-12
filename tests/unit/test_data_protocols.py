"""Contract tests for in-memory data access capabilities."""

import numpy
import pytest

from nori_common.data import (
    CapabilityNotAvailable,
    Extras,
    FrameRate,
    Frames,
    FrameShape,
    InMemoryTrial,
    MetadataNotFound,
    PixelScale,
    Provenance,
    ProvenanceData,
    ScanGeometry,
    Timing,
)


def _trial(**kwargs):
    frames = numpy.arange(1 * 2 * 5 * 4 * 2, dtype=numpy.float32).reshape(
        (1, 2, 5, 4, 2)
    )
    defaults = dict(
        frames=frames,
        non_flyback_rows=(1, 3),
        frame_rate_hz=30.0,
        provenance=ProvenanceData(
            source_type="simulated",
            creator="ERICA",
            creator_version="test",
        ),
        extras={"acq.example": 42},
    )
    defaults.update(kwargs)
    return InMemoryTrial(**defaults)


def test_in_memory_provides_required_capabilities():
    trial = _trial()

    assert trial.supports(Frames)
    assert trial.supports(FrameShape)
    assert trial.supports(ScanGeometry)
    assert trial.supports(FrameRate)
    assert trial.supports(Provenance)
    assert trial.capabilities == frozenset(
        {"Frames", "FrameShape", "ScanGeometry", "FrameRate", "Provenance"}
    )


def test_require_returns_self_for_provided_capability():
    trial = _trial()

    assert trial.require(Frames) is trial
    assert trial.require(ScanGeometry) is trial
    assert trial.require(FrameRate) is trial
    assert trial.require(Provenance) is trial


def test_frames_and_geometry_values():
    trial = _trial()

    assert trial.frames.shape == (1, 2, 5, 4, 2)
    assert trial.non_flyback_rows == (1, 3)
    assert trial.frame_rate_hz == 30.0


def test_frame_shape_capability():
    trial = _trial()

    assert trial.require(FrameShape) is trial
    assert trial.n_frames == 2
    assert trial.frame_shape == (5, 4)


def test_optional_capability_can_be_absent():
    trial = _trial(frame_rate_hz=None)

    assert not trial.supports(FrameRate)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(FrameRate)
    with pytest.raises(CapabilityNotAvailable):
        trial.frame_rate_hz


def test_unimplemented_capability_raises():
    trial = _trial()

    assert not trial.supports(Timing)
    assert not trial.supports(PixelScale)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(Timing)
    with pytest.raises(CapabilityNotAvailable):
        trial.require(PixelScale)


def test_extras_access():
    trial = _trial(extras={"acq.example": 42})

    assert trial.extras.get_extra("acq.example") == 42
    assert trial.extras.get_extra("acq.missing", "fallback") == "fallback"
    assert trial.extras.require_extra("acq.example") == 42
    with pytest.raises(MetadataNotFound):
        trial.extras.require_extra("acq.missing")


def test_extras_is_read_only_mapping():
    extras = Extras({"acq.example": 42})

    assert extras.extras == {"acq.example": 42}
    assert list(extras) == ["acq.example"]
    assert len(extras) == 1


def test_provenance_values():
    trial = _trial()

    assert trial.provenance.source_type == "simulated"
    assert trial.provenance.creator == "ERICA"
    assert trial.provenance.creator_version == "test"
    assert trial.provenance.parent_ids == ()
