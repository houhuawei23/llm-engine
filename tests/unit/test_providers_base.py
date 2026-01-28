"""Unit tests for base provider module."""

import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.exceptions import LLMProviderError
from llm_engine.providers.base import BaseLLMProvider


class ConcreteProvider(BaseLLMProvider):
    """Concrete implementation for testing."""

    def _get_env_api_key(self):
        return os.getenv("TEST_API_KEY")

    def _get_default_base_url(self):
        return "https://api.test.com/v1"

    async def generate(self, prompt: str, system_prompt=None):
        return f"Response to: {prompt}"


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider class."""

    def test_initialization(self, llm_config):
        """Test provider initialization."""
        provider = ConcreteProvider(llm_config)
        assert provider.config == llm_config
        assert provider.base_url == llm_config.base_url or "https://api.test.com/v1"

    def test_get_api_key_from_config(self, llm_config):
        """Test getting API key from config."""
        provider = ConcreteProvider(llm_config)
        assert provider.api_key == "test-api-key"

    def test_get_api_key_from_env(self, llm_config):
        """Test getting API key from environment variable."""
        os.environ["TEST_API_KEY"] = "env-api-key"
        try:
            provider = ConcreteProvider(llm_config)
            # Environment variable should take priority
            assert provider.api_key == "env-api-key"
        finally:
            os.environ.pop("TEST_API_KEY", None)

    def test_get_api_key_priority(self, llm_config):
        """Test API key priority (env > config)."""
        os.environ["TEST_API_KEY"] = "env-key"
        try:
            provider = ConcreteProvider(llm_config)
            assert provider.api_key == "env-key"
        finally:
            os.environ.pop("TEST_API_KEY", None)

    def test_base_url_from_config(self, llm_config):
        """Test base URL from config."""
        llm_config.base_url = "https://custom.api.com/v1"
        provider = ConcreteProvider(llm_config)
        assert provider.base_url == "https://custom.api.com/v1"

    def test_base_url_default(self, llm_config):
        """Test default base URL."""
        llm_config.base_url = None
        provider = ConcreteProvider(llm_config)
        assert provider.base_url == "https://api.test.com/v1"

    @pytest.mark.asyncio
    async def test_generate_stream_default_implementation(self, llm_config):
        """Test default generate_stream implementation."""
        provider = ConcreteProvider(llm_config)
        chunks = []
        async for chunk, tokens in provider.generate_stream("test prompt"):
            chunks.append((chunk, tokens))
        assert len(chunks) == 1
        assert "test prompt" in chunks[0][0]
        assert chunks[0][1] > 0

    def test_estimate_tokens_english(self, llm_config):
        """Test token estimation for English text."""
        provider = ConcreteProvider(llm_config)
        text = "This is a test sentence with English words."
        tokens = provider._estimate_tokens(text)
        assert tokens > 0
        # English text: ~4 chars per token
        assert tokens <= len(text) / 2  # Should be reasonable

    def test_estimate_tokens_chinese(self, llm_config):
        """Test token estimation for Chinese text."""
        provider = ConcreteProvider(llm_config)
        text = "这是一个中文测试句子"
        tokens = provider._estimate_tokens(text)
        assert tokens > 0
        # Chinese text: ~1.5 chars per token
        assert tokens <= len(text)  # Should be reasonable

    def test_estimate_tokens_mixed(self, llm_config):
        """Test token estimation for mixed language text."""
        provider = ConcreteProvider(llm_config)
        text = "This is English and 这是中文"
        tokens = provider._estimate_tokens(text)
        assert tokens > 0

    def test_estimate_tokens_minimum(self, llm_config):
        """Test token estimation returns at least 1."""
        provider = ConcreteProvider(llm_config)
        tokens = provider._estimate_tokens("")
        assert tokens == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, llm_config):
        """Test generate_with_retry with successful generation."""
        provider = ConcreteProvider(llm_config)
        result = await provider.generate_with_retry("test prompt")
        assert "test prompt" in result

    @pytest.mark.asyncio
    async def test_generate_with_retry_timeout_retry(self, llm_config):
        """Test generate_with_retry with timeout retry."""
        provider = ConcreteProvider(llm_config)
        provider.config.max_retries = 2

        call_count = 0

        async def mock_generate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("Timeout")
            return "Success"

        provider.generate = mock_generate
        result = await provider.generate_with_retry("test")
        assert result == "Success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_retry_max_retries_exceeded(self, llm_config):
        """Test generate_with_retry exceeds max retries."""
        provider = ConcreteProvider(llm_config)
        provider.config.max_retries = 2

        async def mock_generate(prompt, system_prompt=None):
            raise asyncio.TimeoutError("Timeout")

        provider.generate = mock_generate
        with pytest.raises(asyncio.TimeoutError):
            await provider.generate_with_retry("test")

    @pytest.mark.asyncio
    async def test_generate_with_retry_api_error_no_retry(self, llm_config):
        """Test generate_with_retry doesn't retry on API errors."""
        provider = ConcreteProvider(llm_config)
        provider.config.max_retries = 3

        async def mock_generate(prompt, system_prompt=None):
            raise Exception("API error: Invalid API key")

        provider.generate = mock_generate
        with pytest.raises(Exception, match="API error"):
            await provider.generate_with_retry("test")

    def test_call_with_prompt(self, llm_config):
        """Test call method with prompt."""
        provider = ConcreteProvider(llm_config)
        result = provider.call(prompt="test prompt")
        assert isinstance(result, str)
        assert "test prompt" in result

    def test_call_with_messages(self, llm_config):
        """Test call method with messages."""
        provider = ConcreteProvider(llm_config)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "test prompt"},
        ]
        result = provider.call(messages=messages)
        assert isinstance(result, str)
        assert "test prompt" in result

    def test_call_without_prompt_or_messages(self, llm_config):
        """Test call method without prompt or messages."""
        provider = ConcreteProvider(llm_config)
        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            provider.call()

    def test_call_with_only_system_message(self, llm_config):
        """Test call method with only system message."""
        provider = ConcreteProvider(llm_config)
        messages = [{"role": "system", "content": "System message"}]
        # Current code logic: only system message without user message raises error
        # because user_prompt will be None (prompt is None, no user_msgs)
        with pytest.raises(ValueError, match="No user prompt found"):
            provider.call(messages=messages)

    def test_call_streaming(self, llm_config):
        """Test call method with streaming."""
        provider = ConcreteProvider(llm_config)
        generator = provider.call(prompt="test", stream=True)
        assert hasattr(generator, "__iter__")
        chunks = list(generator)
        assert len(chunks) > 0

    def test_call_with_temperature_override(self, llm_config):
        """Test call method with temperature override."""
        provider = ConcreteProvider(llm_config)
        original_temp = provider.config.temperature
        provider.call(prompt="test", temperature=0.5)
        # Temperature override shouldn't change config
        assert provider.config.temperature == original_temp

    def test_call_with_model_override(self, llm_config):
        """Test call method with model override."""
        provider = ConcreteProvider(llm_config)
        original_model = provider.config.model_name
        provider.call(prompt="test", model="other-model")
        # Model override shouldn't change config
        assert provider.config.model_name == original_model

    def test_sync_stream_generate(self, llm_config):
        """Test _sync_stream_generate method."""
        provider = ConcreteProvider(llm_config)
        loop = asyncio.new_event_loop()
        try:
            generator = provider._sync_stream_generate("test", None, 0.7, "model", loop)
            chunks = list(generator)
            assert len(chunks) > 0
        finally:
            loop.close()

    @patch("llm_engine.providers.base.get_model_info")
    def test_load_token_per_character_config(self, mock_get_model_info, llm_config):
        """Test loading token_per_character config."""
        mock_get_model_info.return_value = {
            "token_per_character": {
                "english": 0.25,
                "chinese": 0.5,
            }
        }
        provider = ConcreteProvider(llm_config)
        assert provider._token_per_char_config is not None
        assert provider._token_per_char_config["english"] == 0.25
        assert provider._token_per_char_config["chinese"] == 0.5

    @patch("llm_engine.providers.base.get_model_info")
    def test_load_token_per_character_config_not_found(self, mock_get_model_info, llm_config):
        """Test loading token_per_character config when not found."""
        mock_get_model_info.return_value = None
        provider = ConcreteProvider(llm_config)
        # Should use default heuristic
        tokens = provider._estimate_tokens("test")
        assert tokens > 0

    @patch("llm_engine.providers.base.get_model_info")
    def test_estimate_tokens_with_config(self, mock_get_model_info, llm_config):
        """Test token estimation with config."""
        mock_get_model_info.return_value = {
            "token_per_character": {
                "english": 0.3,
                "chinese": 0.6,
            }
        }
        provider = ConcreteProvider(llm_config)
        text = "test"  # 4 English chars
        tokens = provider._estimate_tokens(text)
        # Should use config: 4 * 0.3 = 1.2 -> 1
        assert tokens >= 1
