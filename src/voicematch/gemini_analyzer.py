"""Gemini Vision API integration for avatar analysis."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import google.generativeai as genai


class GeminiError(Exception):
    """Raised when Gemini API operations fail."""

    pass


# Valid emotions that map to CREMA-D dataset
VALID_EMOTIONS = ("anger", "disgust", "fear", "happy", "neutral", "sad", "ambiguous")


@dataclass
class AvatarAnalysis:
    """Structured analysis result from Gemini."""

    estimated_age: int
    age_range: tuple[int, int]  # (min, max) confidence interval
    gender: Literal["male", "female", "ambiguous"]
    race: Literal["african_american", "asian", "caucasian", "hispanic", "mixed", "ambiguous"]
    ethnicity: Optional[str]
    emotion: Literal["anger", "disgust", "fear", "happy", "neutral", "sad", "ambiguous"] = "neutral"
    confidence: float = 0.5  # 0.0 to 1.0
    raw_response: dict = None  # type: ignore[assignment]  # Original Gemini response

    def __post_init__(self):
        if self.raw_response is None:
            self.raw_response = {}


class GeminiAnalyzer:
    """Analyzes avatar images/videos using Gemini Vision API."""

    ANALYSIS_PROMPT = '''Analyze this image of a person (likely an AI avatar or photograph).

Return a JSON object with the following fields:
{
    "estimated_age": <integer, best estimate>,
    "age_range_min": <integer, minimum likely age>,
    "age_range_max": <integer, maximum likely age>,
    "gender": <"male" | "female" | "ambiguous">,
    "race": <"african_american" | "asian" | "caucasian" | "hispanic" | "mixed" | "ambiguous">,
    "ethnicity_notes": <string or null, any additional ethnicity observations>,
    "emotion": <"anger" | "disgust" | "fear" | "happy" | "neutral" | "sad" | "ambiguous">,
    "confidence": <float 0-1, how confident you are in this analysis>
}

Important:
- Base analysis on visible characteristics only
- If unsure, use "ambiguous" and lower confidence
- Age should be biological/apparent age, not chronological
- Emotion should reflect the apparent emotional state of the person in the image
- Return ONLY the JSON object, no other text
'''

    # Retry settings for rate limiting
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 2  # seconds

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize the Gemini analyzer.

        Args:
            api_key: Google Gemini API key.
            model: Model to use for analysis (default: gemini-2.0-flash).
        """
        self.api_key = api_key
        self.model_name = model
        self._model: Optional[genai.GenerativeModel] = None

        # Configure the API
        genai.configure(api_key=api_key)

    def _call_with_retry(self, func, *args, **kwargs):
        """Call a function with exponential backoff retry on rate limit errors.

        Args:
            func: Function to call.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result from func.

        Raises:
            GeminiError: If all retries are exhausted.
        """
        last_error = None
        delay = self.INITIAL_RETRY_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error (429)
                if "429" in str(e) or "resource exhausted" in error_str or "rate limit" in error_str:
                    last_error = e
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    continue
                # Not a rate limit error, raise immediately
                raise

        raise GeminiError(f"Rate limit exceeded after {self.MAX_RETRIES} retries: {last_error}")

    def _get_model(self) -> genai.GenerativeModel:
        """Get or create the Gemini model instance."""
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def analyze_image(self, image_path: Path) -> AvatarAnalysis:
        """Analyze a single image file.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            AvatarAnalysis with demographic information.

        Raises:
            GeminiError: If analysis fails.
        """
        if not image_path.exists():
            raise GeminiError(f"Image file not found: {image_path}")

        try:
            # Read the image
            image_data = image_path.read_bytes()

            # Determine MIME type from extension
            extension = image_path.suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(extension, "image/jpeg")

            # Create the image part
            image_part = {
                "mime_type": mime_type,
                "data": image_data,
            }

            # Generate content with retry on rate limit
            model = self._get_model()
            response = self._call_with_retry(
                model.generate_content, [self.ANALYSIS_PROMPT, image_part]
            )

            return self._parse_response(response.text)

        except GeminiError:
            raise
        except Exception as e:
            raise GeminiError(f"Failed to analyze image: {e}") from e

    def analyze_video(
        self, video_path: Path, frame_time_sec: float = 1.0
    ) -> AvatarAnalysis:
        """Extract a frame from video and analyze it.

        Args:
            video_path: Path to the video file.
            frame_time_sec: Time in seconds to extract frame from.

        Returns:
            AvatarAnalysis with demographic information.

        Raises:
            GeminiError: If analysis fails.
        """
        if not video_path.exists():
            raise GeminiError(f"Video file not found: {video_path}")

        try:
            # Try to extract frame using ffmpeg via pydub/moviepy
            # For MVP, we'll upload the video directly to Gemini
            # which can handle video analysis

            video_data = video_path.read_bytes()

            extension = video_path.suffix.lower()
            mime_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
            }
            mime_type = mime_types.get(extension, "video/mp4")

            video_part = {
                "mime_type": mime_type,
                "data": video_data,
            }

            model = self._get_model()

            # Modify prompt to specify frame analysis
            video_prompt = f"""Look at the person shown in this video at approximately {frame_time_sec} seconds.

{self.ANALYSIS_PROMPT}"""

            # Generate content with retry on rate limit
            response = self._call_with_retry(
                model.generate_content, [video_prompt, video_part]
            )

            return self._parse_response(response.text)

        except GeminiError:
            raise
        except Exception as e:
            raise GeminiError(f"Failed to analyze video: {e}") from e

    def _parse_response(self, response_text: str) -> AvatarAnalysis:
        """Parse Gemini response into AvatarAnalysis.

        Args:
            response_text: Raw text response from Gemini.

        Returns:
            Parsed AvatarAnalysis object.

        Raises:
            GeminiError: If response cannot be parsed.
        """
        try:
            # Try to extract JSON from the response
            # Sometimes Gemini wraps JSON in markdown code blocks
            text = response_text.strip()

            # Remove markdown code block if present
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            data = json.loads(text)

            # Validate and extract fields
            estimated_age = int(data.get("estimated_age", 30))
            age_min = int(data.get("age_range_min", estimated_age - 5))
            age_max = int(data.get("age_range_max", estimated_age + 5))

            gender = data.get("gender", "ambiguous")
            if gender not in ("male", "female", "ambiguous"):
                gender = "ambiguous"

            race = data.get("race", "ambiguous")
            valid_races = (
                "african_american",
                "asian",
                "caucasian",
                "hispanic",
                "mixed",
                "ambiguous",
            )
            if race not in valid_races:
                race = "ambiguous"

            ethnicity = data.get("ethnicity_notes")
            confidence = float(data.get("confidence", 0.5))

            # Extract and validate emotion
            emotion = data.get("emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "ambiguous"

            return AvatarAnalysis(
                estimated_age=estimated_age,
                age_range=(age_min, age_max),
                gender=gender,
                race=race,
                ethnicity=ethnicity,
                emotion=emotion,
                confidence=confidence,
                raw_response=data,
            )

        except json.JSONDecodeError as e:
            raise GeminiError(f"Failed to parse Gemini response as JSON: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise GeminiError(f"Failed to parse Gemini response fields: {e}") from e
