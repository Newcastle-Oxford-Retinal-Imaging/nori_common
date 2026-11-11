"""Exceptions for the data access layer."""

from __future__ import annotations


class DataAccessError(Exception):
    """Base class for data access errors."""


class CapabilityNotAvailable(DataAccessError):
    """A requested optional capability is not provided by an artifact."""

    def __init__(self, artifact: object, capability: type) -> None:
        name = getattr(capability, "__name__", str(capability))
        super().__init__(
            f"{type(artifact).__name__} does not provide capability {name}"
        )
        self.artifact = artifact
        self.capability = capability


class MetadataNotFound(DataAccessError):
    """A required extra metadata key is missing."""

    def __init__(self, key: str) -> None:
        super().__init__(f"extra metadata key not found: {key}")
        self.key = key


class SchemaError(DataAccessError):
    """A required field is missing or malformed."""

    def __init__(self, field: str, product: str = "data") -> None:
        super().__init__(f"invalid or missing field {field!r} for {product}")
        self.field = field
        self.product = product
