"""Unit tests for Gemini Vision API integration."""

import json

import pytest
from unittest.mock import Mock, patch

from voicematch.gemini_analyzer import (
    GeminiAnalyzer,
    AvatarAnalysis,
    GeminiError,
)


class TestGeminiAnalyzer:
    """Tests for GeminiAnalyzer class."""

    def test_analyze_image_returns_valid_analysis(self, temp_dir, sample_audio_file):
        """Test that analyzing an image returns a valid AvatarAnalysis."""
        # Create a mock image file (we'll use the audio file path structure)
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "estimated_age": 35,
                "age_range_min": 30,
                "age_range_max": 40,
                "gender": "male",
                "race": "caucasian",
                "ethnicity_notes": None,
                "confidence": 0.85,
            }
        )

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            result = analyzer.analyze_image(mock_image)

        assert isinstance(result, AvatarAnalysis)
        assert result.estimated_age == 35
        assert result.gender == "male"
        assert result.race == "caucasian"
        assert result.confidence == 0.85

    def test_analyze_image_handles_api_error(self, temp_dir):
        """Test that API errors are handled gracefully."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.side_effect = Exception("API Error")
            mock_get_model.return_value = mock_model

            with pytest.raises(GeminiError, match="Failed to analyze"):
                analyzer.analyze_image(mock_image)

    def test_analyze_image_handles_invalid_image(self, temp_dir):
        """Test that missing/invalid image raises error."""
        analyzer = GeminiAnalyzer(api_key="test_key")

        non_existent = temp_dir / "does_not_exist.png"

        with pytest.raises(GeminiError, match="not found"):
            analyzer.analyze_image(non_existent)

    def test_validates_response_format(self, temp_dir):
        """Test that malformed API responses are handled."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        # Return invalid JSON
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            with pytest.raises(GeminiError, match="parse"):
                analyzer.analyze_image(mock_image)

    def test_handles_ambiguous_demographics(self, temp_dir):
        """Test handling of ambiguous demographic values."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "estimated_age": 40,
                "age_range_min": 35,
                "age_range_max": 50,
                "gender": "ambiguous",
                "race": "mixed",
                "ethnicity_notes": "Could not determine with confidence",
                "confidence": 0.4,
            }
        )

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            result = analyzer.analyze_image(mock_image)

        assert result.gender == "ambiguous"
        assert result.race == "mixed"
        assert result.confidence == 0.4


class TestAvatarAnalysis:
    """Tests for AvatarAnalysis dataclass."""

    def test_avatar_analysis_creation(self):
        """Test creating an AvatarAnalysis instance."""
        analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            confidence=0.85,
            raw_response={"test": "data"},
        )

        assert analysis.estimated_age == 35
        assert analysis.age_range == (30, 40)
        assert analysis.gender == "male"
        assert analysis.race == "caucasian"
        assert analysis.ethnicity is None
        assert analysis.confidence == 0.85

    def test_avatar_analysis_age_range_tuple(self):
        """Test that age_range is a tuple of (min, max)."""
        analysis = AvatarAnalysis(
            estimated_age=45,
            age_range=(40, 50),
            gender="female",
            race="asian",
            ethnicity="East Asian",
            confidence=0.9,
            raw_response={},
        )

        min_age, max_age = analysis.age_range
        assert min_age == 40
        assert max_age == 50


class TestVideoAnalysis:
    """Tests for video analysis functionality."""

    def test_analyze_video_extracts_frame(self, temp_dir):
        """Test that video analysis extracts and analyzes a frame."""
        # This would require ffmpeg/opencv to properly test
        # For now, we test that the function exists and accepts video paths
        analyzer = GeminiAnalyzer(api_key="test_key")

        # The analyze_video method should exist
        assert hasattr(analyzer, "analyze_video")
        assert callable(analyzer.analyze_video)


class TestEmotionDetection:
    """Tests for emotion detection in avatar analysis."""

    def test_avatar_analysis_includes_emotion_field(self):
        """Test that AvatarAnalysis has emotion field."""
        analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",
            confidence=0.85,
            raw_response={},
        )

        assert hasattr(analysis, "emotion")
        assert analysis.emotion == "happy"

    def test_emotion_field_accepts_valid_values(self):
        """Test that emotion field accepts all valid CREMA-D emotions."""
        valid_emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "ambiguous"]

        for emotion in valid_emotions:
            analysis = AvatarAnalysis(
                estimated_age=30,
                age_range=(25, 35),
                gender="female",
                race="asian",
                ethnicity=None,
                emotion=emotion,
                confidence=0.8,
                raw_response={},
            )
            assert analysis.emotion == emotion

    def test_analyze_image_returns_emotion(self, temp_dir):
        """Test that image analysis includes emotion detection."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "estimated_age": 35,
                "age_range_min": 30,
                "age_range_max": 40,
                "gender": "male",
                "race": "caucasian",
                "ethnicity_notes": None,
                "emotion": "happy",
                "confidence": 0.85,
            }
        )

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            result = analyzer.analyze_image(mock_image)

        assert result.emotion == "happy"

    def test_analyze_image_defaults_to_neutral_emotion(self, temp_dir):
        """Test that missing emotion in response defaults to neutral."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        # Response without emotion field
        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "estimated_age": 35,
                "age_range_min": 30,
                "age_range_max": 40,
                "gender": "male",
                "race": "caucasian",
                "ethnicity_notes": None,
                "confidence": 0.85,
            }
        )

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            result = analyzer.analyze_image(mock_image)

        assert result.emotion == "neutral"

    def test_analyze_image_normalizes_invalid_emotion(self, temp_dir):
        """Test that invalid emotion values are normalized to ambiguous."""
        mock_image = temp_dir / "test_avatar.png"
        mock_image.write_bytes(b"fake image data")

        analyzer = GeminiAnalyzer(api_key="test_key")

        mock_response = Mock()
        mock_response.text = json.dumps(
            {
                "estimated_age": 35,
                "age_range_min": 30,
                "age_range_max": 40,
                "gender": "male",
                "race": "caucasian",
                "ethnicity_notes": None,
                "emotion": "excited",  # Not a valid CREMA-D emotion
                "confidence": 0.85,
            }
        )

        with patch.object(analyzer, "_get_model") as mock_get_model:
            mock_model = Mock()
            mock_model.generate_content.return_value = mock_response
            mock_get_model.return_value = mock_model

            result = analyzer.analyze_image(mock_image)

        assert result.emotion == "ambiguous"

    def test_gemini_prompt_includes_emotion_request(self):
        """Test that the Gemini prompt asks for emotion analysis."""
        analyzer = GeminiAnalyzer(api_key="test_key")

        # Check that the prompt mentions emotion
        assert "emotion" in analyzer.ANALYSIS_PROMPT.lower()
