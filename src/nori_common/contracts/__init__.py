"""Shared product contracts: protocols, value objects, and metadata models."""

from nori_common.contracts.base import ArtifactBase
from nori_common.contracts.errors import (
    CapabilityNotAvailable,
    DataAccessError,
    MetadataNotFound,
    SchemaError,
)
from nori_common.contracts.manifest import (
    ArrayDescriptor,
    AxisDescriptor,
    CoordinateSpec,
    Manifest,
)
from nori_common.contracts.photodiode import PhotodiodeTransitionsData
from nori_common.contracts.pixel_scale import PixelScaleMetadata
from nori_common.contracts.provenance import ArtifactProvenance
from nori_common.contracts.protocols import (
    AcquisitionProvenance,
    AcquisitionTiming,
    Artifact,
    ChannelRoles,
    CoarseMotion,
    EyeMotionPerPixel,
    EyeMotionPerRow,
    FramePositions,
    Frames,
    FrameRate,
    FrameShape,
    MetadataExtras,
    Metrics,
    MontageMotion,
    PhotodiodeTransitions,
    PixelScale,
    Provenance,
    RawChannelRoles,
    RawData,
    RawScanGeometry,
    ScanGeometry,
    Timing,
)
from nori_common.contracts.raw import RawAcquisitionMetadata
from nori_common.contracts.trial import FrameMetadata, TrialMetadata
from nori_common.contracts.values import Extras, PixelScaleData, ProvenanceData

__all__ = [
    "AcquisitionProvenance",
    "AcquisitionTiming",
    "ArrayDescriptor",
    "Artifact",
    "ArtifactBase",
    "ArtifactProvenance",
    "AxisDescriptor",
    "CoordinateSpec",
    "CapabilityNotAvailable",
    "ChannelRoles",
    "CoarseMotion",
    "DataAccessError",
    "Extras",
    "EyeMotionPerPixel",
    "EyeMotionPerRow",
    "FrameMetadata",
    "FramePositions",
    "FrameRate",
    "Frames",
    "FrameShape",
    "Manifest",
    "MetadataExtras",
    "MetadataNotFound",
    "Metrics",
    "MontageMotion",
    "PhotodiodeTransitions",
    "PhotodiodeTransitionsData",
    "PixelScale",
    "PixelScaleData",
    "PixelScaleMetadata",
    "Provenance",
    "ProvenanceData",
    "RawAcquisitionMetadata",
    "RawChannelRoles",
    "RawData",
    "RawScanGeometry",
    "ScanGeometry",
    "SchemaError",
    "Timing",
    "TrialMetadata",
]
