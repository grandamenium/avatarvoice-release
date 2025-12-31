"""Unit tests for the PromptOptimizer module."""

import pytest
import re
from unittest.mock import AsyncMock, MagicMock, patch


def extract_words(text: str) -> list:
    """Extract words from text, ignoring punctuation and case."""
    return re.findall(r"\b\w+\b", text.lower())


class TestPromptOptimizer:
    """Tests for the PromptOptimizer class."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        """Set a mock API key."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_12345")

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.generativeai module."""
        with patch("voicematch.prompt_optimizer.genai") as mock:
            yield mock

    def test_init_with_api_key(self, mock_genai):
        """Test initialization with explicit API key."""
        from voicematch.prompt_optimizer import PromptOptimizer

        optimizer = PromptOptimizer(api_key="explicit_key")

        mock_genai.configure.assert_called_once_with(api_key="explicit_key")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash-exp")

    def test_init_with_env_api_key(self, mock_genai, mock_api_key):
        """Test initialization with API key from environment."""
        from voicematch.prompt_optimizer import PromptOptimizer

        optimizer = PromptOptimizer()

        mock_genai.configure.assert_called_once_with(api_key="test_api_key_12345")

    def test_init_without_api_key_raises(self, monkeypatch, mock_genai):
        """Test initialization without API key raises ValueError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from voicematch.prompt_optimizer import PromptOptimizer

        with pytest.raises(ValueError, match="Gemini API key required"):
            PromptOptimizer()

    def test_system_prompt_exists(self, mock_genai, mock_api_key):
        """Test that system prompt is defined."""
        from voicematch.prompt_optimizer import PromptOptimizer

        assert PromptOptimizer.SYSTEM_PROMPT is not None
        assert len(PromptOptimizer.SYSTEM_PROMPT) > 100
        assert "VibeVoice" in PromptOptimizer.SYSTEM_PROMPT
        assert "punctuation" in PromptOptimizer.SYSTEM_PROMPT.lower()

    @pytest.mark.asyncio
    async def test_optimize_returns_modified_text(self, mock_genai, mock_api_key):
        """Test that optimize returns modified text (not identical)."""
        from voicematch.prompt_optimizer import PromptOptimizer

        # Set up mock response
        mock_response = MagicMock()
        mock_response.text = "Hello! How are you doing today?"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        original_text = "Hello how are you doing today"
        result = await optimizer.optimize(original_text, "happy")

        assert result != original_text
        assert result == "Hello! How are you doing today?"

    @pytest.mark.asyncio
    async def test_optimize_preserves_word_order(self, mock_genai, mock_api_key):
        """Test that word order is preserved in optimization."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "the quick brown fox jumps"
        optimized_text = "The quick brown fox jumps!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        # Extract words (ignoring punctuation)
        original_words = extract_words(original_text)
        result_words = extract_words(result)

        assert original_words == result_words

    @pytest.mark.asyncio
    async def test_optimize_empty_text_returns_empty(self, mock_genai, mock_api_key):
        """Test that empty text returns empty."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")

        result = await optimizer.optimize("", "happy")
        assert result == ""

        result = await optimizer.optimize("   ", "happy")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_optimize_with_different_emotions(self, mock_genai, mock_api_key):
        """Test optimization with different emotion values."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Test text!"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")

        emotions = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS", "happy", "angry"]
        for emotion in emotions:
            result = await optimizer.optimize("Test text", emotion)
            assert result is not None

    @pytest.mark.asyncio
    async def test_optimize_passes_correct_prompt(self, mock_genai, mock_api_key):
        """Test that the correct prompt is passed to the model."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Optimized text"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        await optimizer.optimize("Hello world", "happy")

        # Check that generate_content_async was called with system prompt and user prompt
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert len(call_args) == 2
        assert PromptOptimizer.SYSTEM_PROMPT in call_args
        assert "Emotion: happy" in call_args[1]
        assert "Hello world" in call_args[1]

    def test_optimize_sync(self, mock_genai, mock_api_key):
        """Test synchronous optimization method."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello! How are you?"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = optimizer.optimize_sync("Hello how are you", "happy")

        assert result == "Hello! How are you?"
        mock_model.generate_content.assert_called_once()

    def test_optimize_sync_empty_text(self, mock_genai, mock_api_key):
        """Test sync optimization with empty text."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")

        result = optimizer.optimize_sync("", "happy")
        assert result == ""

        # generate_content should not be called for empty text
        mock_model.generate_content.assert_not_called()


class TestPromptOptimizerEdgeCases:
    """Edge case tests for the PromptOptimizer class."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        """Set a mock API key."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key_12345")

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.generativeai module."""
        with patch("voicematch.prompt_optimizer.genai") as mock:
            yield mock

    # --- Empty and Whitespace Input Tests ---

    @pytest.mark.asyncio
    async def test_empty_string_input(self, mock_genai, mock_api_key):
        """Test that empty string returns empty string without API call."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("", "happy")

        assert result == ""
        mock_model.generate_content_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_input_single_space(self, mock_genai, mock_api_key):
        """Test whitespace-only input with single space."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(" ", "happy")

        assert result == " "
        mock_model.generate_content_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_input_multiple_spaces(self, mock_genai, mock_api_key):
        """Test whitespace-only input with multiple spaces."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("     ", "sad")

        assert result == "     "
        mock_model.generate_content_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_input_tabs_newlines(self, mock_genai, mock_api_key):
        """Test whitespace-only input with tabs and newlines."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("\t\n  \t", "neutral")

        assert result == "\t\n  \t"
        mock_model.generate_content_async.assert_not_called()

    def test_sync_empty_string_input(self, mock_genai, mock_api_key):
        """Test sync version with empty string."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = optimizer.optimize_sync("", "angry")

        assert result == ""
        mock_model.generate_content.assert_not_called()

    def test_sync_whitespace_only_input(self, mock_genai, mock_api_key):
        """Test sync version with whitespace-only input."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = optimizer.optimize_sync("   \t\n   ", "fearful")

        assert result == "   \t\n   "
        mock_model.generate_content.assert_not_called()

    # --- Very Long Text Tests ---

    @pytest.mark.asyncio
    async def test_very_long_text_1000_chars(self, mock_genai, mock_api_key):
        """Test optimization with 1000+ character input."""
        from voicematch.prompt_optimizer import PromptOptimizer

        long_text = "word " * 200  # 1000 chars
        optimized_long_text = "Word, " * 199 + "word!"

        mock_response = MagicMock()
        mock_response.text = optimized_long_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(long_text.strip(), "happy")

        assert result is not None
        assert len(result) > 0
        mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_very_long_text_preserves_words(self, mock_genai, mock_api_key):
        """Test that very long text preserves all words."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_words = ["hello", "world", "this", "is", "a", "test"] * 50
        original_text = " ".join(original_words)
        optimized_text = "Hello! World, this is a test. " * 49 + "Hello! World, this is a test!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        original_word_list = extract_words(original_text)
        result_word_list = extract_words(result)

        assert len(original_word_list) == len(result_word_list)

    # --- Special Characters Tests ---

    @pytest.mark.asyncio
    async def test_text_with_special_characters(self, mock_genai, mock_api_key):
        """Test text with special characters (!@#$%^&*())."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello world"
        optimized_text = "Hello! World!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_with_at_symbol_and_hash(self, mock_genai, mock_api_key):
        """Test text containing @ and # symbols."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "check this out @user and #topic"
        optimized_text = "Check this out @user and #topic!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_ampersand_and_percent(self, mock_genai, mock_api_key):
        """Test text containing & and % symbols."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "you and me 100 percent"
        optimized_text = "You and me, 100 percent!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    # --- Emoji Tests ---

    @pytest.mark.asyncio
    async def test_text_with_emojis(self, mock_genai, mock_api_key):
        """Test text containing emojis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "I am so happy today"
        optimized_text = "I am so happy today!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_with_multiple_emojis(self, mock_genai, mock_api_key):
        """Test text with multiple different emojis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "feeling great and awesome"
        optimized_text = "Feeling great and awesome!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "HAP")

        assert result is not None

    # --- Unicode Characters Tests ---

    @pytest.mark.asyncio
    async def test_text_with_unicode_characters(self, mock_genai, mock_api_key):
        """Test text with unicode characters (accented letters)."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "cafe resume naive"
        optimized_text = "Cafe, resume, naive!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_chinese_characters(self, mock_genai, mock_api_key):
        """Test text with Chinese characters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello friend"
        optimized_text = "Hello, friend!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_arabic_characters(self, mock_genai, mock_api_key):
        """Test text with Arabic characters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "peace be with you"
        optimized_text = "Peace be with you!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        assert result is not None

    # --- CREMA-D Emotion Code Tests ---

    @pytest.mark.asyncio
    async def test_emotion_code_neu(self, mock_genai, mock_api_key):
        """Test with NEU (neutral) emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello, how are you."
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "NEU")

        assert result is not None
        # Verify emotion is passed correctly
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: NEU" in call_args[1]

    @pytest.mark.asyncio
    async def test_emotion_code_hap(self, mock_genai, mock_api_key):
        """Test with HAP (happy) emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello! How are you!"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "HAP")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: HAP" in call_args[1]

    @pytest.mark.asyncio
    async def test_emotion_code_sad(self, mock_genai, mock_api_key):
        """Test with SAD emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello... how are you..."
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "SAD")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: SAD" in call_args[1]

    @pytest.mark.asyncio
    async def test_emotion_code_ang(self, mock_genai, mock_api_key):
        """Test with ANG (anger) emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "HELLO! HOW ARE YOU!!"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "ANG")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: ANG" in call_args[1]

    @pytest.mark.asyncio
    async def test_emotion_code_fea(self, mock_genai, mock_api_key):
        """Test with FEA (fear) emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello... how are you?"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "FEA")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: FEA" in call_args[1]

    @pytest.mark.asyncio
    async def test_emotion_code_dis(self, mock_genai, mock_api_key):
        """Test with DIS (disgust) emotion code."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello. How are you."
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "DIS")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: DIS" in call_args[1]

    # --- Mixed Case Emotion Tests ---

    @pytest.mark.asyncio
    async def test_mixed_case_emotion_happy(self, mock_genai, mock_api_key):
        """Test with mixed case emotion: HaPpY."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello! How are you!"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "HaPpY")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: HaPpY" in call_args[1]

    @pytest.mark.asyncio
    async def test_mixed_case_emotion_angry(self, mock_genai, mock_api_key):
        """Test with mixed case emotion: AnGrY."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "HELLO! HOW ARE YOU!!"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "AnGrY")

        assert result is not None
        call_args = mock_model.generate_content_async.call_args[0][0]
        assert "Emotion: AnGrY" in call_args[1]

    @pytest.mark.asyncio
    async def test_lowercase_emotion_sad(self, mock_genai, mock_api_key):
        """Test with lowercase emotion: sad."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello... how are you..."
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "sad")

        assert result is not None

    @pytest.mark.asyncio
    async def test_uppercase_emotion_fearful(self, mock_genai, mock_api_key):
        """Test with uppercase emotion: FEARFUL."""
        from voicematch.prompt_optimizer import PromptOptimizer

        mock_response = MagicMock()
        mock_response.text = "Hello? How are you?"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize("hello how are you", "FEARFUL")

        assert result is not None

    # --- Already Optimized Text Tests ---

    @pytest.mark.asyncio
    async def test_already_optimized_text_with_punctuation(self, mock_genai, mock_api_key):
        """Test text that already has proper punctuation."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "Hello! How are you doing today?"
        # Model might return same or slightly different
        optimized_text = "Hello! How are you doing today?"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        # Word count should remain the same
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    @pytest.mark.asyncio
    async def test_already_optimized_angry_text(self, mock_genai, mock_api_key):
        """Test text that already has capitalization for anger."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "I CAN'T BELIEVE THIS!!"
        optimized_text = "I CAN'T BELIEVE THIS!!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "angry")

        assert result is not None

    # --- Existing Punctuation Tests ---

    @pytest.mark.asyncio
    async def test_text_with_existing_periods(self, mock_genai, mock_api_key):
        """Test text that already has periods."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "Hello. This is a test. How are you."
        optimized_text = "Hello! This is a test! How are you!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert original_words == result_words

    @pytest.mark.asyncio
    async def test_text_with_existing_commas(self, mock_genai, mock_api_key):
        """Test text that already has commas."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello, world, how, are, you"
        optimized_text = "Hello, world, how are you!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_existing_ellipses(self, mock_genai, mock_api_key):
        """Test text that already has ellipses."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "I don't know... maybe... perhaps"
        optimized_text = "I don't know... maybe... perhaps..."

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "sad")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_question_marks(self, mock_genai, mock_api_key):
        """Test text that already has question marks."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "What is this? Why did you do that?"
        optimized_text = "What is this?! Why did you do that?!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "surprised")

        assert result is not None

    # --- Single Word Input Tests ---

    @pytest.mark.asyncio
    async def test_single_word_input(self, mock_genai, mock_api_key):
        """Test single word input."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello"
        optimized_text = "Hello!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    @pytest.mark.asyncio
    async def test_single_word_angry(self, mock_genai, mock_api_key):
        """Test single word with anger emotion."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "stop"
        optimized_text = "STOP!!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "angry")

        assert result is not None
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    # --- Numbers Only Input Tests ---

    @pytest.mark.asyncio
    async def test_numbers_only_input(self, mock_genai, mock_api_key):
        """Test numbers-only input."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "123 456 789"
        optimized_text = "123, 456, 789!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    @pytest.mark.asyncio
    async def test_mixed_numbers_and_words(self, mock_genai, mock_api_key):
        """Test mixed numbers and words input."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "I have 5 apples and 3 oranges"
        optimized_text = "I have 5 apples and 3 oranges!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert original_words == result_words

    # --- Anger Emotion Uses Capitals Test ---

    @pytest.mark.asyncio
    async def test_anger_emotion_uses_capitals(self, mock_genai, mock_api_key):
        """Test that anger emotion can use capital letters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "i hate this so much"
        optimized_text = "I HATE this so much!!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "angry")

        # Check that HATE is capitalized (mock returns it)
        assert "HATE" in result
        # Verify word count is preserved
        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    @pytest.mark.asyncio
    async def test_ang_code_uses_capitals(self, mock_genai, mock_api_key):
        """Test that ANG emotion code can use capital letters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "this is ridiculous"
        optimized_text = "This is RIDICULOUS!!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "ANG")

        assert "RIDICULOUS" in result

    # --- Non-Anger Emotions Do NOT Use Capitals Tests ---

    @pytest.mark.asyncio
    async def test_happy_emotion_no_capitals_for_emphasis(self, mock_genai, mock_api_key):
        """Test that happy emotion uses punctuation only, not capitals for emphasis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "i am so happy today"
        # Happy should use punctuation, not caps
        optimized_text = "I am so happy today!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        # Words should not be ALL CAPS (except first letter capitalization)
        words = result.split()
        for word in words:
            # Remove punctuation for check
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word:
                # Should not be all uppercase (except single letters like "I")
                if len(clean_word) > 1:
                    assert not clean_word.isupper(), f"Word '{word}' should not be all caps for happy emotion"

    @pytest.mark.asyncio
    async def test_sad_emotion_no_capitals_for_emphasis(self, mock_genai, mock_api_key):
        """Test that sad emotion uses punctuation only, not capitals for emphasis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "i feel so sad today"
        optimized_text = "I feel so sad today..."

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "sad")

        words = result.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 1:
                assert not clean_word.isupper(), f"Word '{word}' should not be all caps for sad emotion"

    @pytest.mark.asyncio
    async def test_fearful_emotion_no_capitals_for_emphasis(self, mock_genai, mock_api_key):
        """Test that fearful emotion uses punctuation only, not capitals for emphasis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "i am scared of the dark"
        optimized_text = "I am scared of the dark..."

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "fearful")

        words = result.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 1:
                assert not clean_word.isupper(), f"Word '{word}' should not be all caps for fearful emotion"

    @pytest.mark.asyncio
    async def test_disgusted_emotion_no_capitals_for_emphasis(self, mock_genai, mock_api_key):
        """Test that disgusted emotion uses punctuation only, not capitals for emphasis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "this food is gross"
        optimized_text = "This food is gross."

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "disgusted")

        words = result.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 1:
                assert not clean_word.isupper(), f"Word '{word}' should not be all caps for disgusted emotion"

    @pytest.mark.asyncio
    async def test_neutral_emotion_no_capitals_for_emphasis(self, mock_genai, mock_api_key):
        """Test that neutral emotion uses punctuation only, not capitals for emphasis."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "the weather is nice today"
        optimized_text = "The weather is nice today."

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        words = result.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and len(clean_word) > 1:
                assert not clean_word.isupper(), f"Word '{word}' should not be all caps for neutral emotion"

    # --- Word Count Preservation Tests ---

    @pytest.mark.asyncio
    async def test_word_count_preserved_simple(self, mock_genai, mock_api_key):
        """Test that word count is preserved in simple case."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello world this is a test"
        optimized_text = "Hello, world! This is a test!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words), (
            f"Word count mismatch: original={len(original_words)}, result={len(result_words)}"
        )

    @pytest.mark.asyncio
    async def test_word_count_preserved_with_contractions(self, mock_genai, mock_api_key):
        """Test that word count is preserved with contractions."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "I can't believe you don't know"
        optimized_text = "I can't believe you don't know!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "surprised")

        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    @pytest.mark.asyncio
    async def test_words_never_added(self, mock_genai, mock_api_key):
        """Test that no words are added to the output."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "the cat sat"
        optimized_text = "The cat sat!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        original_words = extract_words(original_text)
        result_words = extract_words(result)

        # No words should be added
        for word in result_words:
            assert word in original_words, f"Word '{word}' was added but shouldn't be"

    @pytest.mark.asyncio
    async def test_words_never_removed(self, mock_genai, mock_api_key):
        """Test that no words are removed from the output."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "the big brown dog jumps"
        optimized_text = "The big brown dog jumps!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        original_words = extract_words(original_text)
        result_words = extract_words(result)

        # No words should be removed
        for word in original_words:
            assert word in result_words, f"Word '{word}' was removed but shouldn't be"

    def test_sync_word_count_preserved(self, mock_genai, mock_api_key):
        """Test that sync version preserves word count."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello there friend"
        optimized_text = "Hello there, friend!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = optimizer.optimize_sync(original_text, "happy")

        original_words = extract_words(original_text)
        result_words = extract_words(result)
        assert len(original_words) == len(result_words)

    # --- Additional Edge Cases ---

    @pytest.mark.asyncio
    async def test_text_with_newlines(self, mock_genai, mock_api_key):
        """Test text with newline characters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello\nworld\nhow are you"
        optimized_text = "Hello!\nWorld!\nHow are you!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None

    @pytest.mark.asyncio
    async def test_text_with_tabs(self, mock_genai, mock_api_key):
        """Test text with tab characters."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello\tworld\ttest"
        optimized_text = "Hello\tworld\ttest!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "neutral")

        assert result is not None

    @pytest.mark.asyncio
    async def test_response_with_leading_trailing_whitespace(self, mock_genai, mock_api_key):
        """Test that response whitespace is stripped."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello world"
        # Mock response has whitespace that should be stripped
        optimized_text = "   Hello, world!   "

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        # Result should have whitespace stripped
        assert result == "Hello, world!"
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    @pytest.mark.asyncio
    async def test_text_with_multiple_spaces_between_words(self, mock_genai, mock_api_key):
        """Test text with multiple spaces between words."""
        from voicematch.prompt_optimizer import PromptOptimizer

        original_text = "hello    world    test"
        optimized_text = "Hello    world    test!"

        mock_response = MagicMock()
        mock_response.text = optimized_text
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        optimizer = PromptOptimizer(api_key="test_key")
        result = await optimizer.optimize(original_text, "happy")

        assert result is not None


class TestPromptOptimizerIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.requires_api
    @pytest.mark.asyncio
    async def test_real_optimization(self):
        """Test with real Gemini API (requires GEMINI_API_KEY)."""
        import os
        from voicematch.prompt_optimizer import PromptOptimizer

        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        optimizer = PromptOptimizer()
        original = "hello how are you doing today"
        result = await optimizer.optimize(original, "happy")

        # Result should be different (optimized)
        assert result != original
        # But should contain the same words
        original_words = set(extract_words(original))
        result_words = set(extract_words(result))
        assert original_words == result_words
