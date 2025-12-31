"""Custom exceptions for VoiceMatch API."""


class VoiceMatchError(Exception):
    """Base exception for VoiceMatch errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ImageAnalysisError(VoiceMatchError):
    """Error during image analysis."""
    pass


class VoiceNotFoundError(VoiceMatchError):
    """Requested voice actor not found."""
    pass


class DatabaseError(VoiceMatchError):
    """Database operation failed."""
    pass


class ConfigurationError(VoiceMatchError):
    """Configuration is invalid or missing."""
    pass


class UnsupportedFormatError(VoiceMatchError):
    """Unsupported file format."""
    pass


class APIConnectionError(VoiceMatchError):
    """Failed to connect to external API."""
    pass
