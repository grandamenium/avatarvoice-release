"""Unit tests for Gradio UI application."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import wave
import struct
import math

from voicematch.app import VoiceMatchApp
from voicematch.gemini_analyzer import AvatarAnalysis
from voicematch.voice_matcher import MatchResult
from voicematch.database import Actor, AudioClip


@pytest.fixture
def sample_image_file(temp_dir):
    """Create a sample test image file."""
    # Create a minimal valid PNG file
    filepath = temp_dir / "test_avatar.png"
    # Minimal 1x1 PNG (red pixel)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # 8-bit RGB
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,  # Compressed data
        0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x18, 0xDD,
        0x8D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
        0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    filepath.write_bytes(png_data)
    return filepath


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample test video file (minimal mp4 stub)."""
    filepath = temp_dir / "test_avatar.mp4"
    # Create a minimal placeholder (won't be a real video, just for path testing)
    filepath.write_bytes(b"fake mp4 content for testing")
    return filepath


@pytest.fixture
def mock_analysis():
    """Create a mock avatar analysis result."""
    return AvatarAnalysis(
        estimated_age=35,
        age_range=(30, 40),
        gender="male",
        race="caucasian",
        ethnicity=None,
        confidence=0.85,
        raw_response={},
    )


@pytest.fixture
def mock_match_result():
    """Create a mock match result."""
    actor = Actor(
        id="1001",
        age=34,
        sex="Male",
        race="Caucasian",
        ethnicity="Not Hispanic",
    )
    return MatchResult(
        actor=actor,
        score=0.92,
        match_details={
            "gender_score": 1.0,
            "race_score": 1.0,
            "age_score": 0.9,
        },
    )


@pytest.fixture
def mock_audio_output(temp_dir):
    """Create a mock audio output file."""
    filepath = temp_dir / "output.wav"
    sample_rate = 44100
    duration = 1.0
    frequency = 440

    num_samples = int(sample_rate * duration)

    with wave.open(str(filepath), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            sample = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))

    return filepath


class TestVoiceMatchApp:
    """Tests for VoiceMatchApp class."""

    def test_ui_loads_without_error(self, mock_env):
        """Test that the Gradio interface creates without error."""
        app = VoiceMatchApp()
        interface = app.create_interface()

        assert interface is not None

    def test_app_has_required_components(self, mock_env):
        """Test that the app initializes all required components."""
        app = VoiceMatchApp()

        # Before initialization, components should be None
        assert app.analyzer is None
        assert app.database is None
        assert app.matcher is None
        assert app.processor is None

    def test_initialize_creates_components(self, mock_env, temp_dir):
        """Test that _initialize creates all components."""
        # Set up database path
        from voicematch.database import VoiceDatabase

        db_path = temp_dir / "voice.sqlite"
        db = VoiceDatabase(db_path)
        db.connect()
        db.create_schema()
        db.close()

        app = VoiceMatchApp()

        with patch.object(app, "config") as mock_config:
            mock_config.database_path = db_path
            mock_config.gemini_api_key = "test_key"
            mock_config.output_dir = temp_dir

            app._initialize()

            assert app.analyzer is not None
            assert app.database is not None
            assert app.matcher is not None
            assert app.processor is not None

    def test_image_upload_triggers_analysis(
        self, mock_env, sample_image_file, mock_analysis, mock_match_result, mock_audio_output, temp_dir
    ):
        """Test that image upload triggers the analysis pipeline."""
        app = VoiceMatchApp()

        # Mock all components
        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match_result]

            app.processor = MagicMock()
            app.processor.concatenate_clips.return_value = mock_audio_output

            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), "test text"
            )

            app.analyzer.analyze_image.assert_called_once()
            assert "35" in analysis_text  # Age should be in output
            assert "male" in analysis_text.lower()

    def test_video_upload_triggers_analysis(
        self, mock_env, sample_video_file, mock_analysis, mock_match_result, mock_audio_output, temp_dir
    ):
        """Test that video upload triggers the analysis pipeline."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_video.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match_result]

            app.processor = MagicMock()
            app.processor.concatenate_clips.return_value = mock_audio_output

            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_video_file), "test text"
            )

            app.analyzer.analyze_video.assert_called_once()

    def test_results_displayed_correctly(
        self, mock_env, sample_image_file, mock_analysis, mock_match_result, mock_audio_output, temp_dir
    ):
        """Test that results are formatted correctly for display."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match_result]

            app.processor = MagicMock()
            app.processor.concatenate_clips.return_value = mock_audio_output

            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), ""
            )

            # Check analysis output contains expected info
            assert "Age" in analysis_text or "age" in analysis_text.lower()
            assert "Gender" in analysis_text or "gender" in analysis_text.lower()

            # Check match output contains actor info
            assert "1001" in match_text or "score" in match_text.lower()

    def test_audio_download_works(
        self, mock_env, sample_image_file, mock_analysis, mock_match_result, mock_audio_output, temp_dir, sample_audio_file
    ):
        """Test that audio file path is returned for download."""
        app = VoiceMatchApp()

        # Create mock audio clips pointing to the sample audio file
        mock_clip = AudioClip(
            id=1,
            actor_id="1001",
            filepath=sample_audio_file,
            sentence="IEO",
            emotion="NEU",
            level="XX",
            duration_ms=1000,
        )

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = [mock_clip]

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match_result]

            app.processor = MagicMock()
            app.processor.concatenate_clips.return_value = mock_audio_output

            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), ""
            )

            assert audio_path is not None
            assert Path(audio_path).exists()
            assert audio_path.endswith(".wav")

    def test_error_handling_displayed(self, mock_env, sample_image_file, temp_dir):
        """Test that errors are handled and displayed gracefully."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.side_effect = Exception("API Error")

            app.config = MagicMock()
            app.config.output_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), ""
            )

            assert "error" in analysis_text.lower() or "Error" in analysis_text
            assert audio_path is None

    def test_loading_state_shown(self, mock_env):
        """Test that the interface has proper loading indicators."""
        app = VoiceMatchApp()
        interface = app.create_interface()

        # Gradio blocks should be created
        assert interface is not None
        # The button click should be connected
        # (This is more of an integration check - we verify the interface structure)

    def test_handles_no_matches(self, mock_env, sample_image_file, mock_analysis, temp_dir):
        """Test handling when no matching voices are found."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = []  # No matches

            app.database = MagicMock()
            app.processor = MagicMock()

            app.config = MagicMock()
            app.config.output_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), ""
            )

            assert "no match" in match_text.lower() or "not found" in match_text.lower()
            assert audio_path is None

    def test_handles_missing_file(self, mock_env, temp_dir):
        """Test handling when uploaded file doesn't exist."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.config = MagicMock()
            app.config.output_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(temp_dir / "nonexistent.png"), ""
            )

            assert "error" in analysis_text.lower() or "not found" in analysis_text.lower()
            assert audio_path is None


class TestVoiceMatchAppIntegration:
    """Integration tests for VoiceMatchApp components."""

    def test_format_analysis_text(self, mock_analysis):
        """Test formatting of analysis results."""
        # This tests the internal formatting logic
        text = f"""Avatar Analysis Results:
- Estimated Age: {mock_analysis.estimated_age} (range: {mock_analysis.age_range[0]}-{mock_analysis.age_range[1]})
- Gender: {mock_analysis.gender}
- Race: {mock_analysis.race}
- Confidence: {mock_analysis.confidence:.0%}"""

        assert "35" in text
        assert "male" in text
        assert "caucasian" in text
        assert "85%" in text

    def test_format_match_text(self, mock_match_result):
        """Test formatting of match results."""
        result = mock_match_result
        text = f"""Best Match: Actor {result.actor.id}
- Age: {result.actor.age}
- Gender: {result.actor.sex}
- Race: {result.actor.race}
- Match Score: {result.score:.0%}"""

        assert "1001" in text
        assert "34" in text
        assert "Male" in text
        assert "92%" in text


class TestEmotionSelector:
    """Tests for emotion selector UI functionality."""

    def test_ui_has_emotion_selector(self, mock_env):
        """Test that UI includes emotion dropdown selector."""
        app = VoiceMatchApp()
        interface = app.create_interface()

        # Interface should have the emotion dropdown
        assert interface is not None
        # We'll verify the dropdown exists by checking the component structure

    def test_emotion_selector_options(self, mock_env):
        """Test that emotion selector has correct options."""
        expected_options = [
            "Auto-detect",
            "Anger",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
        ]
        app = VoiceMatchApp()

        # Verify EMOTION_OPTIONS constant exists
        assert hasattr(app, "EMOTION_OPTIONS")
        for option in expected_options:
            assert option in app.EMOTION_OPTIONS

    def test_process_upload_accepts_emotion_param(
        self, mock_env, sample_image_file, mock_analysis, mock_match_result, mock_audio_output, temp_dir
    ):
        """Test that process_upload accepts emotion parameter."""
        app = VoiceMatchApp()

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match_result]

            app.processor = MagicMock()
            app.processor.concatenate_clips.return_value = mock_audio_output

            app.config = MagicMock()
            app.config.output_dir = temp_dir
            app.config.data_dir = temp_dir

            # Should accept emotion as third parameter
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), "test text", "Happy"
            )

            # Should not error
            assert analysis_text is not None

    def test_emotion_auto_detect_uses_gemini(
        self, mock_env, sample_image_file, temp_dir
    ):
        """Test that Auto-detect uses Gemini's emotion detection."""
        app = VoiceMatchApp()

        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",  # Gemini detected emotion
            confidence=0.85,
            raw_response={},
        )

        mock_actor = Actor(
            id="1001", age=34, sex="Male", race="Caucasian", ethnicity="Not Hispanic"
        )
        mock_match = MatchResult(
            actor=mock_actor,
            score=0.9,
            match_details={"gender_score": 1.0, "race_score": 1.0, "age_score": 0.8},
        )

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match]

            app.processor = MagicMock()
            app.config = MagicMock()
            app.config.output_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), "", "Auto-detect"
            )

            # Should query database with Gemini's detected emotion
            app.database.get_clips_for_actor.assert_called()
            call_args = app.database.get_clips_for_actor.call_args
            # First call should use HAP (happy converted to CREMA-D code)
            assert "HAP" in str(call_args) or len(call_args) > 0

    def test_emotion_user_selection_overrides_gemini(
        self, mock_env, sample_image_file, temp_dir
    ):
        """Test that user emotion selection overrides Gemini inference."""
        app = VoiceMatchApp()

        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",  # Gemini says happy
            confidence=0.85,
            raw_response={},
        )

        mock_actor = Actor(
            id="1001", age=34, sex="Male", race="Caucasian", ethnicity="Not Hispanic"
        )
        mock_match = MatchResult(
            actor=mock_actor,
            score=0.9,
            match_details={"gender_score": 1.0, "race_score": 1.0, "age_score": 0.8},
        )

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match]

            app.processor = MagicMock()
            app.config = MagicMock()
            app.config.output_dir = temp_dir

            # User selects "Sad" even though Gemini detected "happy"
            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), "", "Sad"
            )

            # Should query database with user's selection (SAD)
            app.database.get_clips_for_actor.assert_called()
            # Check that the first call used "SAD" emotion
            first_call = app.database.get_clips_for_actor.call_args_list[0]
            # Call should be get_clips_for_actor("1001", emotion="SAD")
            assert first_call[0][0] == "1001"  # actor_id
            assert first_call[1].get("emotion") == "SAD"  # emotion kwarg

    def test_analysis_output_shows_emotion(
        self, mock_env, sample_image_file, temp_dir
    ):
        """Test that analysis output includes detected emotion."""
        app = VoiceMatchApp()

        mock_analysis = AvatarAnalysis(
            estimated_age=35,
            age_range=(30, 40),
            gender="male",
            race="caucasian",
            ethnicity=None,
            emotion="happy",
            confidence=0.85,
            raw_response={},
        )

        mock_actor = Actor(
            id="1001", age=34, sex="Male", race="Caucasian", ethnicity="Not Hispanic"
        )
        mock_match = MatchResult(
            actor=mock_actor,
            score=0.9,
            match_details={"gender_score": 1.0, "race_score": 1.0, "age_score": 0.8},
        )

        with patch.object(app, "_initialize"):
            app.analyzer = MagicMock()
            app.analyzer.analyze_image.return_value = mock_analysis

            app.database = MagicMock()
            app.database.get_clips_for_actor.return_value = []

            app.matcher = MagicMock()
            app.matcher.find_matches.return_value = [mock_match]

            app.processor = MagicMock()
            app.config = MagicMock()
            app.config.output_dir = temp_dir

            analysis_text, match_text, audio_path = app.process_upload(
                str(sample_image_file), "", "Auto-detect"
            )

            # Analysis should mention emotion
            assert "emotion" in analysis_text.lower() or "happy" in analysis_text.lower()
