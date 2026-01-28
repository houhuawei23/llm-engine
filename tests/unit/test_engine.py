"""Unit tests for engine module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.engine import (
    LLMEngine,
    OpenAIProvider,
    DeepSeekProvider,
    OllamaProvider,
    CustomProvider,
)


class TestLLMEngine:
    """Tests for LLMEngine class."""

    def test_initialization_openai(self, llm_config_openai):
        """Test engine initialization with OpenAI provider."""
        engine = LLMEngine(llm_config_openai)
        assert engine.config == llm_config_openai
        assert isinstance(engine.provider, OpenAIProvider)

    def test_initialization_deepseek(self, llm_config):
        """Test engine initialization with DeepSeek provider."""
        engine = LLMEngine(llm_config)
        assert engine.config == llm_config
        assert isinstance(engine.provider, DeepSeekProvider)

    def test_initialization_ollama(self, llm_config_ollama):
        """Test engine initialization with Ollama provider."""
        engine = LLMEngine(llm_config_ollama)
        assert engine.config == llm_config_ollama
        assert isinstance(engine.provider, OllamaProvider)

    def test_initialization_custom(self):
        """Test engine initialization with Custom provider."""
        config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
        engine = LLMEngine(config)
        assert isinstance(engine.provider, CustomProvider)

    def test_initialization_unsupported_provider(self):
        """Test engine initialization with unsupported provider."""
        # Create a config with an unsupported provider by directly setting it
        config = LLMConfig()
        config.provider = "unsupported"  # type: ignore
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMEngine(config)

    @pytest.mark.asyncio
    async def test_generate(self, llm_config):
        """Test generate method."""
        engine = LLMEngine(llm_config)
        engine.provider.generate_with_retry = AsyncMock(return_value="Test response")
        result = await engine.generate("test prompt")
        assert result == "Test response"
        engine.provider.generate_with_retry.assert_called_once_with("test prompt", None)

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm_config):
        """Test generate method with system prompt."""
        engine = LLMEngine(llm_config)
        engine.provider.generate_with_retry = AsyncMock(return_value="Test response")
        result = await engine.generate("test prompt", system_prompt="system")
        assert result == "Test response"
        engine.provider.generate_with_retry.assert_called_once_with("test prompt", "system")

    @pytest.mark.asyncio
    async def test_stream_generate(self, llm_config):
        """Test stream_generate method."""
        engine = LLMEngine(llm_config)

        async def mock_stream(prompt, system_prompt=None):
            yield ("chunk1", 10)
            yield ("chunk2", 20)

        engine.provider.generate_stream = mock_stream
        chunks = []
        async for chunk, tokens in engine.stream_generate("test prompt"):
            chunks.append((chunk, tokens))
        assert len(chunks) == 2
        assert chunks[0] == ("chunk1", 10)
        assert chunks[1] == ("chunk2", 20)


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_get_env_api_key(self, llm_config_ollama):
        """Test getting API key (should be None for Ollama)."""
        provider = OllamaProvider(llm_config_ollama)
        assert provider.api_key is None

    def test_get_default_base_url(self, llm_config_ollama):
        """Test getting default base URL."""
        import os

        os.environ["OLLAMA_BASE_URL"] = "http://custom:11434"
        try:
            llm_config_ollama.base_url = None
            provider = OllamaProvider(llm_config_ollama)
            assert provider.base_url == "http://custom:11434"
        finally:
            os.environ.pop("OLLAMA_BASE_URL", None)

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_generate_success(self, mock_acompletion, llm_config_ollama):
        """Test successful generation."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = OllamaProvider(llm_config_ollama)
        result = await provider.generate("test prompt")
        assert result == "Test response"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_generate_with_system_prompt(self, mock_acompletion, llm_config_ollama):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = OllamaProvider(llm_config_ollama)
        result = await provider.generate("test prompt", system_prompt="system")
        assert result == "Test response"
        # Check that system prompt was included
        call_args = mock_acompletion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_generate_stream_success(self, mock_acompletion, llm_config_ollama):
        """Test successful streaming generation."""
        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="Test"))]),
                Mock(choices=[Mock(delta=Mock(content=" response"))]),
            ]
            for chunk in chunks:
                yield chunk

        mock_acompletion.return_value = mock_stream()

        provider = OllamaProvider(llm_config_ollama)
        chunks = []
        async for chunk, tokens in provider.generate_stream("test prompt"):
            chunks.append((chunk, tokens))
        assert len(chunks) == 2

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_generate_empty_response(self, mock_acompletion, llm_config_ollama):
        """Test generation with empty response."""
        mock_response = Mock()
        mock_response.choices = []
        mock_acompletion.return_value = mock_response

        provider = OllamaProvider(llm_config_ollama)
        from llm_engine.exceptions import LLMProviderError

        with pytest.raises(LLMProviderError, match="no choices field"):
            await provider.generate("test prompt")
