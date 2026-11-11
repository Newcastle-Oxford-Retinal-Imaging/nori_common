"""Analysis-facing data access protocols, adapters, and capability types."""

from nori_common.data.adapters import FitsTrialAdapter, InMemoryTrial
from nori_common.data.base import ArtifactBase
from nori_common.data.errors import (
    CapabilityNotAvailable,
    DataAccessError,
    MetadataNotFound,
    SchemaError,
)
from nori_common.data.extras import Extras, ProvenanceData
from nori_common.data.protocols import (
    Artifact,
    ChannelRoles,
    Frames,
    FrameRate,
    MetadataExtras,
    PixelScale,
    Provenance,
    ScanGeometry,
    Timing,
)

__all__ = [
    "Artifact",
    "ArtifactBase",
    "CapabilityNotAvailable",
    "ChannelRoles",
    "DataAccessError",
    "Extras",
    "FitsTrialAdapter",
    "FrameRate",
    "Frames",
    "InMemoryTrial",
    "MetadataExtras",
    "MetadataNotFound",
    "PixelScale",
    "Provenance",
    "ProvenanceData",
    "ScanGeometry",
    "SchemaError",
    "Timing",
]
