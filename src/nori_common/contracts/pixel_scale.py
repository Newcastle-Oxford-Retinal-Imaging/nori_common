"""Pixel-scale contract models."""

from __future__ import annotations

from nori_common.contracts.model_base import ContractBase


class PixelScaleMetadata(ContractBase):
    """Pixel scale in arcminutes."""

    pixel_size_arcmin_x: float
    pixel_size_arcmin_y: float
