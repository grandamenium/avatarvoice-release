"""VibeVoice TTS Client - Main client implementation."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

import httpx

from .exceptions import (
    ConnectionError,
    GenerationError,
    TimeoutError,
    ValidationError,
)
from .models import (
    GenerationConfig,
    GenerationResult,
    StreamingChunk,
    VoiceReference,
)
from .streaming import StreamingHandler

# Optional gradio_client for proper Gradio 5 support
try:
    from gradio_client import Client as GradioClient
    HAS_GRADIO_CLIENT = True
except ImportError:
    HAS_GRADIO_CLIENT = False


class VibeVoiceClient:
    """Client for VibeVoice TTS Gradio API.

    Provides a clean interface for text-to-speech generation using
    the VibeVoice model through its Gradio endpoint.

    Updated for VibeVoice Podcast WebUI (Gradio 5).
    """

    DEFAULT_ENDPOINT = "http://localhost:7860"
    # Gradio 5 API path format
    API_PATH = "/gradio_api/call/generate_podcast_wrapper"
    RESULT_PATH = "/gradio_api/call/generate_podcast_wrapper/{event_id}"

    # Available speakers in VibeVoice
    SPEAKERS = [
        "en-Alice_woman",
        "en-Carter_man",
        "en-Frank_man",
        "en-Mary_woman_bgm",
        "en-Maya_woman",
        "in-Samuel_man",
        "zh-Anchen_man_bgm",
        "zh-Bowen_man",
        "zh-Xinran_woman",
    ]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize VibeVoice client.

        Args:
            endpoint: Gradio endpoint URL. Uses default if not provided.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
        """
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._streaming_handler = StreamingHandler()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "VibeVoiceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def health_check(self) -> bool:
        """Check if the VibeVoice endpoint is healthy.

        Returns:
            True if endpoint is healthy, False otherwise.
        """
        try:
            response = await self.client.get("/")
            return response.status_code == 200
        except Exception:
            return False

    def _validate_config(self, config: GenerationConfig) -> None:
        """Validate generation config.

        Args:
            config: Config to validate.

        Raises:
            ValidationError: If config is invalid.
        """
        if not config.text or not config.text.strip():
            raise ValidationError("Text cannot be empty")

        if len(config.text) > 5000:
            raise ValidationError("Text exceeds maximum length of 5000 characters")

    # Mapping from gender to VibeVoice speakers
    MALE_SPEAKERS = ["en-Carter_man", "en-Frank_man", "in-Samuel_man", "zh-Bowen_man"]
    FEMALE_SPEAKERS = ["en-Alice_woman", "en-Maya_woman", "en-Mary_woman_bgm", "zh-Xinran_woman"]

    def _get_speaker(self, config: GenerationConfig) -> str:
        """Get speaker ID from config or voice reference metadata.

        Maps CREMA-D actor gender to VibeVoice built-in speakers.

        Args:
            config: Generation config.

        Returns:
            Speaker ID string.
        """
        # Check if voice_reference has a speaker_id (direct mapping)
        if config.voice_reference and config.voice_reference.speaker_id:
            speaker = config.voice_reference.speaker_id
            if speaker in self.SPEAKERS:
                return speaker

        # Try to infer gender from audio_path filename (CREMA-D format: XXXX_STATEMENT_EMOTION_LEVEL.wav)
        if config.voice_reference and config.voice_reference.audio_path:
            # Use actor gender from our database if available
            # For now, use a simple heuristic based on the path
            # CREMA-D has odd actor IDs for female, even for male (not exactly, but approximate)
            path_name = config.voice_reference.audio_path.stem
            try:
                actor_id = int(path_name.split("_")[0])
                # Look up gender from database would be ideal, for now use built-in defaults
            except (ValueError, IndexError):
                pass

        # Default to female speaker (more common in TTS demos)
        return "en-Alice_woman"

    def _prepare_request_data(
        self,
        config: GenerationConfig,
    ) -> dict:
        """Prepare request data for Gradio API.

        Args:
            config: Generation config.

        Returns:
            Request data dictionary for VibeVoice Podcast API.
        """
        # Get speaker for this request
        speaker = self._get_speaker(config)

        # Build Gradio 5 API request for /generate_podcast_wrapper
        # Parameters: num_speakers, script, speaker1, speaker2, speaker3, speaker4, cfg_scale
        data = {
            "data": [
                1,  # num_speakers - use 1 for simple TTS
                config.text,  # script - the text to speak
                speaker,  # Speaker 1
                "en-Carter_man",  # Speaker 2 (unused with num_speakers=1)
                "en-Frank_man",  # Speaker 3 (unused)
                "en-Maya_woman",  # Speaker 4 (unused)
                config.cfg_scale,  # CFG scale
            ]
        }

        return data

    async def generate(
        self,
        config: GenerationConfig,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Generate speech from text using VibeVoice Podcast API.

        Uses gradio_client for proper Gradio 5 streaming support.

        Args:
            config: Generation configuration.
            output_path: Optional path to save audio file.

        Returns:
            GenerationResult with audio data.

        Raises:
            ValidationError: If config is invalid.
            ConnectionError: If connection fails.
            GenerationError: If generation fails.
            TimeoutError: If request times out.
        """
        self._validate_config(config)

        if HAS_GRADIO_CLIENT:
            return await self._generate_with_gradio_client(config, output_path)
        else:
            return await self._generate_with_httpx(config, output_path)

    async def _generate_with_gradio_client(
        self,
        config: GenerationConfig,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Generate using gradio_client (preferred method)."""

        # If we have a custom voice reference with audio_path, use the new direct endpoint
        if config.voice_reference and config.voice_reference.audio_path:
            audio_path = config.voice_reference.audio_path
            if audio_path.exists():
                print(f"Using direct audio reference: {audio_path}")

                # Run in thread pool since gradio_client is sync
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._sync_generate_with_audio_reference,
                    str(audio_path),
                    config.text,
                    config.cfg_scale,
                )

                return self._process_gradio_result(result, config, output_path)

        # For built-in voices, use the standard endpoint
        speaker = self._get_speaker(config)

        try:
            # Run in thread pool since gradio_client is sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._sync_gradio_generate,
                config.text,
                speaker,
                config.cfg_scale,
            )

            return self._process_gradio_result(result, config, output_path)

        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e

    def _process_gradio_result(
        self,
        result: tuple,
        config: GenerationConfig,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Process result from Gradio API call.

        Args:
            result: Tuple from gradio_client.predict()
            config: Generation config
            output_path: Optional path to save audio

        Returns:
            GenerationResult with audio data
        """
        # Result format: (streaming_audio, complete_audio, log, ...)
        audio_path = None
        audio_bytes = None

        # Check index 1 for complete audio (can be dict, tuple, or string path)
        if len(result) > 1:
            complete_audio = result[1]

            if isinstance(complete_audio, dict):
                # Gradio file dict format
                audio_file = complete_audio.get("value") or complete_audio.get("path")
                if audio_file and Path(audio_file).exists():
                    audio_path = Path(audio_file)
                    audio_bytes = audio_path.read_bytes()
            elif isinstance(complete_audio, tuple) and len(complete_audio) == 2:
                # (sample_rate, numpy_array) format
                import numpy as np
                sample_rate, audio_data = complete_audio
                # Convert to WAV bytes
                import io
                import wave
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    if isinstance(audio_data, np.ndarray):
                        wav_file.writeframes(audio_data.tobytes())
                    else:
                        wav_file.writeframes(audio_data)
                audio_bytes = buffer.getvalue()
            elif isinstance(complete_audio, str) and Path(complete_audio).exists():
                # Direct file path
                audio_path = Path(complete_audio)
                audio_bytes = audio_path.read_bytes()

        if not audio_bytes:
            raise GenerationError("No audio data in response")

        gen_result = GenerationResult(
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            sample_rate=config.sample_rate,
            config_used=config,
            metadata={"log": result[2] if len(result) > 2 else ""},
        )

        # Save to output path if specified
        if output_path and audio_bytes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            gen_result.audio_path = output_path

        return gen_result

    def _sync_generate_with_audio_reference(
        self,
        audio_path: str,
        text: str,
        cfg_scale: float,
    ) -> tuple:
        """Generate using direct audio reference endpoint (bypasses enum validation).

        Args:
            audio_path: Path to reference audio file
            text: Text to synthesize
            cfg_scale: CFG guidance scale

        Returns:
            Result tuple from Gradio API
        """
        from gradio_client import handle_file

        client = GradioClient(self.endpoint)
        return client.predict(
            audio_file=handle_file(audio_path),
            script=text,
            cfg_scale=cfg_scale,
            api_name="/generate_with_audio_reference",
        )

    def _sync_gradio_generate(self, text: str, speaker: str, cfg_scale: float):
        """Synchronous generation call.

        Uses raw HTTP for custom voices (to bypass gradio_client validation),
        or gradio_client for built-in voices.
        """

        # For custom voices, use raw HTTP to bypass validation
        if speaker.startswith("custom-"):
            return self._sync_raw_generate(text, speaker, cfg_scale)

        # For built-in voices, use gradio_client
        client = GradioClient(self.endpoint)
        return client.predict(
            num_speakers=1,
            script=text,
            param_2=speaker,
            param_3="en-Carter_man",
            param_4="en-Frank_man",
            param_5="en-Maya_woman",
            param_6=cfg_scale,
            api_name="/generate_podcast_wrapper",
        )

    def _sync_raw_generate(self, text: str, speaker: str, cfg_scale: float):
        """Generate using raw HTTP (bypasses gradio_client validation)."""
        import json as json_module

        data = {
            "data": [
                1,  # num_speakers
                text,  # script
                speaker,  # Speaker 1 (can be custom)
                "en-Carter_man",
                "en-Frank_man",
                "en-Maya_woman",
                cfg_scale,
            ]
        }

        with httpx.Client(timeout=180.0, base_url=self.endpoint) as client:
            # Initiate generation
            r = client.post("/gradio_api/call/generate_podcast_wrapper", json=data)
            if r.status_code != 200:
                raise GenerationError(f"Init failed: {r.status_code}")

            event_id = r.json().get("event_id")
            if not event_id:
                raise GenerationError(f"No event_id: {r.json()}")

            # Stream results
            final_result = None
            with client.stream(
                "GET", f"/gradio_api/call/generate_podcast_wrapper/{event_id}"
            ) as stream:
                for line in stream.iter_lines():
                    if line.startswith("data:"):
                        try:
                            d = json_module.loads(line[5:].strip())
                            if isinstance(d, list):
                                final_result = d
                        except json_module.JSONDecodeError:
                            continue

            if not final_result:
                raise GenerationError("No result from stream")

            # Return in same format as gradio_client
            return tuple(final_result)

    def upload_custom_voice(self, audio_path: Path, voice_name: str) -> Optional[str]:
        """Upload a custom voice to VibeVoice.

        Args:
            audio_path: Path to audio file.
            voice_name: Name to assign to the voice.

        Returns:
            The speaker ID to use (e.g., 'custom-voice_name') or None if failed.
        """
        if not HAS_GRADIO_CLIENT:
            return None

        try:
            from gradio_client import handle_file
            client = GradioClient(self.endpoint)
            result = client.predict(
                audio_file=handle_file(str(audio_path)),
                voice_name=voice_name,
                api_name="/handle_custom_voice_upload",
            )
            # Check if upload was successful
            status = result[0] if result else ""
            if "success" in status.lower() or "uploaded" in status.lower():
                # Return the custom speaker ID (VibeVoice prefixes with 'custom-')
                return f"custom-{voice_name}"
            return None
        except Exception as e:
            print(f"Voice upload failed: {e}")
            return None

    def get_available_speakers(self) -> list:
        """Get list of available speakers including custom uploads."""
        if not HAS_GRADIO_CLIENT:
            return self.SPEAKERS.copy()

        try:
            client = GradioClient(self.endpoint)
            # Trigger visibility update to get current speaker list
            result = client.predict(
                num_speakers=1,
                api_name="/update_speaker_visibility",
            )
            # Result contains dropdown choices for each speaker slot
            if result and isinstance(result[0], str):
                # The dropdown values are returned
                return [result[i] for i in range(min(4, len(result))) if result[i]]
        except:
            pass
        return self.SPEAKERS.copy()

    async def _generate_with_httpx(
        self,
        config: GenerationConfig,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Fallback generation using httpx (may not work for streaming)."""
        request_data = self._prepare_request_data(config)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Step 1: Initiate generation
                response = await self.client.post(
                    self.API_PATH,
                    json=request_data,
                )

                if response.status_code != 200:
                    raise GenerationError(
                        f"Generation failed with status {response.status_code}",
                        details={"response": response.text},
                    )

                init_data = response.json()
                event_id = init_data.get("event_id")

                if not event_id:
                    raise GenerationError("No event_id in response", details=init_data)

                # Step 2: Poll for results (Gradio 5 uses SSE streaming)
                result_url = f"{self.API_PATH}/{event_id}"
                result_data = await self._poll_for_result(result_url)

                return self._process_response(result_data, config, output_path)

            except httpx.ConnectError as e:
                last_error = ConnectionError(f"Failed to connect to endpoint: {e}")
            except httpx.TimeoutException as e:
                last_error = TimeoutError(f"Request timed out: {e}")
            except GenerationError:
                raise
            except Exception as e:
                last_error = GenerationError(f"Unexpected error: {e}")

            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        raise last_error or GenerationError("Generation failed after retries")

    async def _poll_for_result(self, result_url: str) -> dict:
        """Poll for generation result using SSE streaming.

        Args:
            result_url: URL to poll for results.

        Returns:
            Result data dictionary.
        """
        import json as json_module

        async with self.client.stream("GET", result_url) as response:
            if response.status_code != 200:
                raise GenerationError(f"Result fetch failed: {response.status_code}")

            result_data = None
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            data = json_module.loads(data_str)
                            # Look for the final result with audio
                            if isinstance(data, list) and len(data) > 0:
                                result_data = {"data": data}
                        except json_module.JSONDecodeError:
                            continue

            if result_data is None:
                raise GenerationError("No valid result received from stream")

            return result_data

    def _process_response(
        self,
        response_data: dict,
        config: GenerationConfig,
        output_path: Optional[Path] = None,
    ) -> GenerationResult:
        """Process Gradio API response from VibeVoice Podcast.

        Response format: [streaming_audio, complete_audio, log_text, html]

        Args:
            response_data: Raw API response.
            config: Original generation config.
            output_path: Optional path to save audio.

        Returns:
            GenerationResult object.
        """
        import base64

        # Extract audio from response
        data = response_data.get("data", [])

        if not data:
            raise GenerationError("No data in response")

        result = GenerationResult(
            sample_rate=config.sample_rate,
            config_used=config,
            metadata={"raw_response": response_data},
        )

        # Find the audio data - look for dict with 'url' or 'path' key
        audio_data = None
        for idx, item in enumerate(data):
            if isinstance(item, dict) and ("url" in item or "path" in item):
                audio_data = item
                break

        if audio_data is None:
            raise GenerationError("No audio data in response")

        # Process the audio data based on its format
        if isinstance(audio_data, str):
            # Could be a URL, file path, or base64
            if audio_data.startswith("data:"):
                # Base64 encoded audio
                header, encoded = audio_data.split(",", 1)
                result.audio_bytes = base64.b64decode(encoded)
            elif audio_data.startswith(("http://", "https://")):
                # URL - need to download
                result.metadata["audio_url"] = audio_data
                # We'll fetch this synchronously for now
                import httpx as httpx_sync
                with httpx_sync.Client() as client:
                    audio_response = client.get(audio_data)
                    if audio_response.status_code == 200:
                        result.audio_bytes = audio_response.content
            elif Path(audio_data).exists():
                # Local file path
                result.audio_path = Path(audio_data)
                result.audio_bytes = result.audio_path.read_bytes()
        elif isinstance(audio_data, dict):
            # Gradio 5 file object format
            if "url" in audio_data and audio_data["url"]:
                # Download from URL
                audio_url = audio_data["url"]
                # Handle relative URLs
                if audio_url.startswith("/"):
                    audio_url = f"{self.endpoint}{audio_url}"
                result.metadata["audio_url"] = audio_url
                import httpx as httpx_sync
                with httpx_sync.Client() as client:
                    audio_response = client.get(audio_url)
                    if audio_response.status_code == 200:
                        result.audio_bytes = audio_response.content
            elif "path" in audio_data and audio_data["path"]:
                local_path = Path(audio_data["path"])
                if local_path.exists():
                    result.audio_path = local_path
                    result.audio_bytes = local_path.read_bytes()
            elif "data" in audio_data:
                result.audio_bytes = base64.b64decode(audio_data["data"])

        # Save to output path if specified
        if output_path and result.audio_bytes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(result.audio_bytes)
            result.audio_path = output_path

        # Calculate duration if possible
        if result.audio_bytes:
            # Rough estimate based on file size (assuming 16-bit mono)
            result.duration_seconds = len(result.audio_bytes) / (config.sample_rate * 2)

        return result

    async def generate_stream(
        self,
        config: GenerationConfig,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Generate speech with streaming output.

        Args:
            config: Generation configuration (stream will be enabled).

        Yields:
            StreamingChunk objects with audio data.

        Raises:
            ValidationError: If config is invalid.
            ConnectionError: If connection fails.
            GenerationError: If generation fails.
        """
        self._validate_config(config)

        # Enable streaming in config
        stream_config = config.model_copy()
        stream_config.stream = True

        request_data = self._prepare_request_data(stream_config)

        self._streaming_handler.reset()

        try:
            async with self.client.stream(
                "POST",
                self.API_PATH,
                json=request_data,
            ) as response:
                if response.status_code != 200:
                    raise GenerationError(
                        f"Streaming failed with status {response.status_code}"
                    )

                chunk_index = 0
                async for chunk in response.aiter_bytes():
                    streaming_chunk = self._streaming_handler.process_chunk(
                        chunk, is_final=False
                    )
                    chunk_index += 1
                    yield streaming_chunk

                # Mark last chunk as final
                if self._streaming_handler.chunks:
                    self._streaming_handler.chunks[-1].is_final = True

        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect: {e}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Streaming timed out: {e}") from e

    async def generate_simple(
        self,
        text: str,
        voice_reference: Optional[Union[str, Path, bytes]] = None,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> GenerationResult:
        """Simple interface for quick generation.

        Args:
            text: Text to synthesize.
            voice_reference: Optional voice reference (path, URL, or bytes).
            output_path: Optional path to save audio.
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with audio data.
        """
        # Build voice reference
        voice_ref = None
        if voice_reference:
            if isinstance(voice_reference, bytes):
                voice_ref = VoiceReference(audio_bytes=voice_reference)
            elif isinstance(voice_reference, Path):
                voice_ref = VoiceReference(audio_path=voice_reference)
            elif isinstance(voice_reference, str):
                if voice_reference.startswith(("http://", "https://")):
                    voice_ref = VoiceReference(audio_url=voice_reference)
                else:
                    voice_ref = VoiceReference(audio_path=Path(voice_reference))

        config = GenerationConfig(
            text=text,
            voice_reference=voice_ref,
            **kwargs,
        )

        return await self.generate(config, output_path)

    def set_endpoint(self, endpoint: str) -> None:
        """Update the endpoint URL.

        Args:
            endpoint: New endpoint URL.
        """
        self.endpoint = endpoint
        # Reset client to use new endpoint
        if self._client is not None:
            asyncio.create_task(self.close())
            self._client = None
