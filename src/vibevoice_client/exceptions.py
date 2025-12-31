"""Custom exceptions for VibeVoice client."""


class VibeVoiceError(Exception):
    """Base exception for VibeVoice errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConnectionError(VibeVoiceError):
    """Failed to connect to VibeVoice endpoint."""
    pass


class GenerationError(VibeVoiceError):
    """Error during audio generation."""
    pass


class ValidationError(VibeVoiceError):
    """Invalid input parameters."""
    pass


class TimeoutError(VibeVoiceError):
    """Operation timed out."""
    pass


class AudioProcessingError(VibeVoiceError):
    """Error processing audio data."""
    pass
