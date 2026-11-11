"""Base configuration for Pydantic contract models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContractBase(BaseModel):
    """Base model for artifact metadata contracts.

    Unknown fields are preserved so acquisition-specific and future metadata
    can round-trip through the schema layer.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        populate_by_name=True,
    )
