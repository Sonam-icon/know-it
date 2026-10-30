"""Application-level errors independent of HTTP concerns."""


class ApplicationError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ApplicationError):
    """Required configuration is absent or inconsistent."""


class DocumentLoadError(ApplicationError):
    """A source could not be loaded or safely decoded."""


class DocumentTooLargeError(DocumentLoadError):
    """A source exceeds its configured size limit."""


class UnsupportedDocumentError(DocumentLoadError):
    """A source format is not accepted by the ingestion pipeline."""


class UnsafeUrlError(DocumentLoadError):
    """A URL points to a network location that must not be fetched."""


class EmptyDocumentError(DocumentLoadError):
    """A source did not contain indexable text."""


class IndexConfigurationError(ApplicationError):
    """The stored vector index is incompatible with current configuration."""
