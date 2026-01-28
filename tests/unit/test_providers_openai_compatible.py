"""Unit tests for OpenAI compatible provider module."""

import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest
from openai import APIError, AuthenticationError, RateLimitError

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.exceptions import LLMProviderError
from llm_engine.engine import OpenAIProvider, DeepSeekProvider, CustomProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider


class TestOpenAICompatibleProvider:
    """Tests for OpenAICompatibleProvider base class."""

    def test_initialization(self, llm_config):
        """Test provider initialization."""
        provider = OpenAIProvider(llm_config)
        assert provider.config == llm_config
        assert provider.provider_name == "OpenAI"

    def test_build_messages_with_system_prompt(self, llm_config):
        """Test building messages with system prompt."""
        provider = OpenAIProvider(llm_config)
        messages = provider._build_messages("user prompt", "system prompt")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user prompt"

    def test_build_messages_without_system_prompt(self, llm_config):
        """Test building messages without system prompt."""
        provider = OpenAIProvider(llm_config)
        messages = provider._build_messages("user prompt")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "user prompt"

    def test_build_payload(self, llm_config):
        """Test building request payload."""
        provider = OpenAIProvider(llm_config)
        messages = [{"role": "user", "content": "test"}]
        payload = provider._build_payload(messages, stream=False)
        assert payload["model"] == llm_config.model_name
        assert payload["messages"] == messages
        assert payload["temperature"] == llm_config.temperature
        assert payload["max_tokens"] == llm_config.max_tokens
        assert payload["top_p"] == llm_config.top_p
        # stream=False means stream key is not included in payload
        assert "stream" not in payload or payload.get("stream") is False

    def test_build_payload_streaming(self, llm_config):
        """Test building request payload for streaming."""
        provider = OpenAIProvider(llm_config)
        messages = [{"role": "user", "content": "test"}]
        payload = provider._build_payload(messages, stream=True)
        assert payload["stream"] is True

    def test_get_litellm_model_name(self, llm_config):
        """Test getting LiteLLM model name."""
        provider = OpenAIProvider(llm_config)
        model_name = provider._get_litellm_model_name()
        assert model_name == f"openai/{llm_config.model_name}"

    def test_client_property(self, llm_config):
        """Test client property."""
        provider = OpenAIProvider(llm_config)
        client = provider.client
        assert client is not None
        # Second call should return same instance
        assert provider.client is client

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_generate_success(self, mock_acompletion, llm_config):
        """Test successful generation."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = OpenAIProvider(llm_config)
        result = await provider.generate("test prompt")
        assert result == "Test response"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_generate_no_api_key(self, mock_acompletion, llm_config):
        """Test generation without API key."""
        llm_config.api_key = None
        provider = OpenAIProvider(llm_config)
        with pytest.raises(ValueError, match="API key not set"):
            await provider.generate("test prompt")

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_generate_empty_response(self, mock_acompletion, llm_config):
        """Test generation with empty response."""
        mock_response = Mock()
        mock_response.choices = []
        mock_acompletion.return_value = mock_response

        provider = OpenAIProvider(llm_config)
        with pytest.raises(LLMProviderError, match="no choices field"):
            await provider.generate("test prompt")

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_generate_stream_success(self, mock_acompletion, llm_config):
        """Test successful streaming generation."""
        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="Test"))]),
                Mock(choices=[Mock(delta=Mock(content=" response"))]),
            ]
            for chunk in chunks:
                yield chunk

        mock_acompletion.return_value = mock_stream()

        provider = OpenAIProvider(llm_config)
        chunks = []
        async for chunk, tokens in provider.generate_stream("test prompt"):
            chunks.append((chunk, tokens))
        assert len(chunks) == 2
        assert chunks[0][0] == "Test"
        assert chunks[1][0] == " response"

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_generate_stream_timeout(self, mock_acompletion, llm_config):
        """Test streaming generation timeout."""
        mock_acompletion.side_effect = asyncio.TimeoutError("Timeout")

        provider = OpenAIProvider(llm_config)
        with pytest.raises(Exception, match="timeout"):
            async for _ in provider.generate_stream("test prompt"):
                pass

    def test_call_complete_response(self, llm_config):
        """Test call method for complete response."""
        provider = OpenAIProvider(llm_config)
        provider._complete_response = Mock(return_value="Test response")
        result = provider.call(prompt="test")
        assert result == "Test response"

    def test_call_streaming_response(self, llm_config):
        """Test call method for streaming response."""
        provider = OpenAIProvider(llm_config)
        provider._stream_response = Mock(return_value=iter(["chunk1", "chunk2"]))
        result = provider.call(prompt="test", stream=True)
        chunks = list(result)
        assert chunks == ["chunk1", "chunk2"]

    def test_call_with_messages(self, llm_config):
        """Test call method with messages."""
        provider = OpenAIProvider(llm_config)
        provider._complete_response = Mock(return_value="Test response")
        messages = [{"role": "user", "content": "test"}]
        result = provider.call(messages=messages)
        assert result == "Test response"

    def test_complete_response(self, llm_config):
        """Test _complete_response method."""
        provider = OpenAIProvider(llm_config)
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "  Test response  "
        mock_response.choices = [mock_choice]
        provider.client.chat.completions.create = Mock(return_value=mock_response)

        result = provider._complete_response({"model": "gpt-4", "messages": []})
        assert result == "Test response"

    def test_stream_response(self, llm_config):
        """Test _stream_response method."""
        provider = OpenAIProvider(llm_config)
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="chunk1"))]),
            Mock(choices=[Mock(delta=Mock(content="chunk2"))]),
        ]
        provider.client.chat.completions.create = Mock(return_value=mock_chunks)

        chunks = list(provider._stream_response({"model": "gpt-4", "messages": []}))
        assert chunks == ["chunk1", "chunk2"]

    def test_stream_response_error(self, llm_config):
        """Test _stream_response with error."""
        provider = OpenAIProvider(llm_config)
        provider.client.chat.completions.create = Mock(side_effect=Exception("Stream error"))

        with pytest.raises(RuntimeError, match="Stream failed"):
            list(provider._stream_response({"model": "gpt-4", "messages": []}))

    def test_call_authentication_error(self, llm_config):
        """Test call method with authentication error."""
        provider = OpenAIProvider(llm_config)
        # Create a mock response object for AuthenticationError
        mock_response = Mock()
        mock_response.status_code = 401
        auth_error = AuthenticationError("Invalid key", response=mock_response, body={})
        provider.client.chat.completions.create = Mock(side_effect=auth_error)

        with pytest.raises(RuntimeError, match="authentication failed"):
            provider.call(prompt="test")

    def test_call_rate_limit_error(self, llm_config):
        """Test call method with rate limit error."""
        provider = OpenAIProvider(llm_config)
        # Create a mock response object for RateLimitError
        mock_response = Mock()
        mock_response.status_code = 429
        rate_error = RateLimitError("Rate limit", response=mock_response, body={})
        provider.client.chat.completions.create = Mock(side_effect=rate_error)

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            provider.call(prompt="test")

    def test_call_api_error(self, llm_config):
        """Test call method with API error."""
        provider = OpenAIProvider(llm_config)
        # APIError from openai library requires body parameter
        api_error = APIError("API error", request=None, body={})
        provider.client.chat.completions.create = Mock(side_effect=api_error)

        with pytest.raises(RuntimeError, match="API error"):
            provider.call(prompt="test")


class TestOpenAIProvider:
    """Tests for OpenAIProvider class."""

    def test_get_env_api_key(self, llm_config_openai):
        """Test getting OpenAI API key from environment."""
        import os

        os.environ["OPENAI_API_KEY"] = "env-openai-key"
        try:
            provider = OpenAIProvider(llm_config_openai)
            # Environment variable should take priority
            assert provider.api_key == "env-openai-key"
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_get_default_base_url(self, llm_config_openai):
        """Test getting default base URL."""
        llm_config_openai.base_url = None
        provider = OpenAIProvider(llm_config_openai)
        assert provider.base_url == "https://api.openai.com/v1"

    def test_get_provider_name(self, llm_config_openai):
        """Test getting provider name."""
        provider = OpenAIProvider(llm_config_openai)
        assert provider.provider_name == "OpenAI"

    def test_get_litellm_model_name(self, llm_config_openai):
        """Test getting LiteLLM model name."""
        provider = OpenAIProvider(llm_config_openai)
        assert provider._get_litellm_model_name() == "openai/gpt-4"

    def test_add_provider_specific_params(self, llm_config_openai):
        """Test adding provider-specific parameters."""
        provider = OpenAIProvider(llm_config_openai)
        payload = {}
        provider._add_provider_specific_params(payload)
        assert "presence_penalty" in payload
        assert "frequency_penalty" in payload


class TestDeepSeekProvider:
    """Tests for DeepSeekProvider class."""

    def test_get_env_api_key(self, llm_config):
        """Test getting DeepSeek API key from environment."""
        import os

        os.environ["DEEPSEEK_API_KEY"] = "env-deepseek-key"
        try:
            provider = DeepSeekProvider(llm_config)
            assert provider.api_key == "env-deepseek-key"
        finally:
            os.environ.pop("DEEPSEEK_API_KEY", None)

    def test_get_default_base_url(self, llm_config):
        """Test getting default base URL."""
        llm_config.base_url = None
        provider = DeepSeekProvider(llm_config)
        assert provider.base_url == "https://api.deepseek.com/v1"

    def test_get_provider_name(self, llm_config):
        """Test getting provider name."""
        provider = DeepSeekProvider(llm_config)
        assert provider.provider_name == "DeepSeek"

    def test_get_litellm_model_name(self, llm_config):
        """Test getting LiteLLM model name."""
        provider = DeepSeekProvider(llm_config)
        assert provider._get_litellm_model_name() == "deepseek/deepseek-chat"

    @patch("llm_engine.engine.get_model_info")
    def test_load_json_output_config(self, mock_get_model_info, llm_config):
        """Test loading JSON output configuration."""
        mock_get_model_info.return_value = {
            "functions": {
                "json_output": True,
            }
        }
        provider = DeepSeekProvider(llm_config)
        assert provider._json_output_enabled is True

    @patch("llm_engine.engine.get_model_info")
    def test_add_provider_specific_params_with_json(self, mock_get_model_info, llm_config):
        """Test adding provider-specific params with JSON output enabled."""
        mock_get_model_info.return_value = {
            "functions": {
                "json_output": True,
            }
        }
        provider = DeepSeekProvider(llm_config)
        payload = {
            "messages": [
                {"role": "user", "content": "Return json format"}
            ]
        }
        provider._add_provider_specific_params(payload)
        assert "response_format" in payload
        assert payload["response_format"] == {"type": "json_object"}

    @patch("llm_engine.engine.get_model_info")
    def test_add_provider_specific_params_without_json_keyword(self, mock_get_model_info, llm_config):
        """Test adding provider-specific params without JSON keyword."""
        mock_get_model_info.return_value = {
            "functions": {
                "json_output": True,
            }
        }
        provider = DeepSeekProvider(llm_config)
        payload = {
            "messages": [
                {"role": "user", "content": "Regular prompt"}
            ]
        }
        provider._add_provider_specific_params(payload)
        # Should not add response_format if no "json" keyword
        assert "response_format" not in payload


class TestCustomProvider:
    """Tests for CustomProvider class."""

    def test_get_env_api_key(self):
        """Test getting custom API key from environment."""
        import os

        os.environ["CUSTOM_API_KEY"] = "env-custom-key"
        try:
            config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
            provider = CustomProvider(config)
            assert provider.api_key == "env-custom-key"
        finally:
            os.environ.pop("CUSTOM_API_KEY", None)

    def test_get_default_base_url(self):
        """Test getting default base URL."""
        import os

        os.environ["CUSTOM_API_BASE_URL"] = "https://custom.api.com/v1"
        try:
            config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
            config.base_url = None
            provider = CustomProvider(config)
            assert provider.base_url == "https://custom.api.com/v1"
        finally:
            os.environ.pop("CUSTOM_API_BASE_URL", None)

    def test_get_provider_name(self):
        """Test getting provider name."""
        config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
        provider = CustomProvider(config)
        assert provider.provider_name == "Custom API"

    def test_get_litellm_model_name(self):
        """Test getting LiteLLM model name."""
        config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
        provider = CustomProvider(config)
        assert provider._get_litellm_model_name() == "openai/custom-model"
