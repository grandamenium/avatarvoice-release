"""Streaming support for VibeVoice TTS generation."""

import asyncio
from typing import AsyncGenerator, Callable
import io

from .models import StreamingChunk, StreamingState, StreamingStatus
from .exceptions import GenerationError, TimeoutError


class StreamingHandler:
    """Handles streaming audio generation."""

    def __init__(
        self,
        buffer_size: int = 4096,
        timeout_seconds: float = 60.0,
    ):
        """Initialize streaming handler.

        Args:
            buffer_size: Size of audio chunks in bytes.
            timeout_seconds: Timeout for streaming operations.
        """
        self.buffer_size = buffer_size
        self.timeout_seconds = timeout_seconds
        self._state = StreamingState()
        self._chunks: list[StreamingChunk] = []
        self._buffer = io.BytesIO()
        self._callbacks: list[Callable[[StreamingChunk], None]] = []

    @property
    def state(self) -> StreamingState:
        """Get current streaming state."""
        return self._state

    @property
    def chunks(self) -> list[StreamingChunk]:
        """Get all received chunks."""
        return self._chunks.copy()

    def add_callback(self, callback: Callable[[StreamingChunk], None]) -> None:
        """Add a callback to be called when new chunks arrive.

        Args:
            callback: Function to call with each new chunk.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[StreamingChunk], None]) -> None:
        """Remove a callback.

        Args:
            callback: Callback to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def reset(self) -> None:
        """Reset streaming state."""
        self._state = StreamingState()
        self._chunks = []
        self._buffer = io.BytesIO()

    def process_chunk(self, audio_data: bytes, is_final: bool = False) -> StreamingChunk:
        """Process a chunk of audio data.

        Args:
            audio_data: Raw audio bytes.
            is_final: Whether this is the final chunk.

        Returns:
            StreamingChunk object.
        """
        chunk = StreamingChunk(
            audio_bytes=audio_data,
            chunk_index=len(self._chunks),
            is_final=is_final,
        )

        self._chunks.append(chunk)
        self._buffer.write(audio_data)
        self._state.chunks_received = len(self._chunks)
        self._state.status = StreamingStatus.COMPLETE if is_final else StreamingStatus.STREAMING

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(chunk)
            except Exception:
                pass  # Don't let callback errors break streaming

        return chunk

    def get_complete_audio(self) -> bytes:
        """Get all audio data concatenated.

        Returns:
            Complete audio bytes.
        """
        return self._buffer.getvalue()

    async def stream_from_source(
        self,
        source: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream audio chunks from an async generator.

        Args:
            source: Async generator yielding audio bytes.

        Yields:
            StreamingChunk objects.

        Raises:
            TimeoutError: If streaming times out.
            GenerationError: If streaming fails.
        """
        self._state.status = StreamingStatus.PROCESSING

        try:
            async for audio_data in source:
                chunk = self.process_chunk(audio_data, is_final=False)
                yield chunk

            # Mark final chunk
            if self._chunks:
                self._chunks[-1].is_final = True
                self._state.status = StreamingStatus.COMPLETE

        except asyncio.TimeoutError as e:
            self._state.status = StreamingStatus.ERROR
            self._state.error_message = "Streaming timed out"
            raise TimeoutError("Streaming timed out") from e
        except Exception as e:
            self._state.status = StreamingStatus.ERROR
            self._state.error_message = str(e)
            raise GenerationError(f"Streaming failed: {e}") from e


class AudioBuffer:
    """Thread-safe buffer for streaming audio."""

    def __init__(self, max_size: int = 10 * 1024 * 1024):  # 10MB default
        """Initialize audio buffer.

        Args:
            max_size: Maximum buffer size in bytes.
        """
        self.max_size = max_size
        self._buffer = io.BytesIO()
        self._lock = asyncio.Lock()
        self._closed = False

    async def write(self, data: bytes) -> int:
        """Write data to buffer.

        Args:
            data: Bytes to write.

        Returns:
            Number of bytes written.

        Raises:
            GenerationError: If buffer is full or closed.
        """
        if self._closed:
            raise GenerationError("Buffer is closed")

        async with self._lock:
            current_size = self._buffer.tell()
            if current_size + len(data) > self.max_size:
                raise GenerationError("Audio buffer full")

            return self._buffer.write(data)

    async def read(self, size: int = -1) -> bytes:
        """Read data from buffer.

        Args:
            size: Number of bytes to read (-1 for all).

        Returns:
            Read bytes.
        """
        async with self._lock:
            if size == -1:
                return self._buffer.getvalue()
            return self._buffer.read(size)

    async def getvalue(self) -> bytes:
        """Get all buffer contents.

        Returns:
            Complete buffer contents.
        """
        async with self._lock:
            return self._buffer.getvalue()

    async def close(self) -> None:
        """Close the buffer."""
        self._closed = True

    def __len__(self) -> int:
        """Get current buffer size."""
        return self._buffer.tell()
