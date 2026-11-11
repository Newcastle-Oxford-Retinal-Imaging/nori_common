"""Base class for concrete artifacts implementing the capability pattern."""

from __future__ import annotations

from typing import TypeVar

from nori_common.contracts.errors import CapabilityNotAvailable

T = TypeVar("T")


class ArtifactBase:
    """Mixin providing capability queries and ``require`` semantics.

    Concrete adapters declare the capabilities they provide by overriding
    :meth:`_provided_capabilities`. Optional capabilities are simply omitted,
    and ``require`` raises :class:`CapabilityNotAvailable` for them.
    """

    def _provided_capabilities(self) -> frozenset[type]:
        return frozenset()

    def supports(self, capability: type) -> bool:
        """Return whether this artifact provides ``capability``."""
        return capability in self._provided_capabilities()

    @property
    def capabilities(self) -> frozenset[str]:
        """Names of the capabilities this artifact provides."""
        return frozenset(cap.__name__ for cap in self._provided_capabilities())

    def require(self, capability: type[T]) -> T:
        """Return ``self`` as ``capability``, or raise if not provided."""
        if capability not in self._provided_capabilities():
            raise CapabilityNotAvailable(self, capability)
        return self  # type: ignore[return-value]
