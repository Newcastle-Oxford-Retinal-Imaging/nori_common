"""Provenance contract models."""

from __future__ import annotations

from nori_common.contracts.model_base import ContractBase


class ArtifactProvenance(ContractBase):
    """Where an artifact came from and what produced it."""

    source_type: str = "unknown"
    creator: str = "unknown"
    creator_version: str | None = None
    parent_ids: tuple[str, ...] = ()
