"""Photodiode transition contract models."""

from __future__ import annotations

from nori_common.contracts.model_base import ContractBase


class PhotodiodeTransitionsData(ContractBase):
    """Photodiode transition values and times."""

    values: dict[int, float] = {}
    transitions: dict[int, float] = {}
