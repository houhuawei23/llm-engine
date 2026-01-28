"""Integration tests for LLM providers."""

import os
import pytest
from unittest.mock import patch, AsyncMock, Mock

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.engine import (
    OpenAIProvider,
    DeepSeekProvider,
    OllamaProvider,
    CustomProvider,
)
from llm_engine.config_loader import create_llm_config_from_provider


class TestProviderIntegration:
    """Integration tests for providers."""

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_openai_provider_full_flow(self, mock_acompletion, llm_config_openai):
        """Test OpenAI provider full flow."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "OpenAI response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = OpenAIProvider(llm_config_openai)
        result = await provider.generate("test prompt", system_prompt="system")
        assert result == "OpenAI response"

        # Verify call parameters
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4"
        assert len(call_kwargs["messages"]) == 2

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_deepseek_provider_full_flow(self, mock_acompletion, llm_config):
        """Test DeepSeek provider full flow."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "DeepSeek response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = DeepSeekProvider(llm_config)
        result = await provider.generate("test prompt")
        assert result == "DeepSeek response"

        # Verify call parameters
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "deepseek/deepseek-chat"

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_deepseek_provider_with_json_output(self, mock_acompletion, llm_config):
        """Test DeepSeek provider with JSON output enabled."""
        with patch("llm_engine.engine.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "functions": {
                    "json_output": True,
                }
            }

            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = '{"key": "value"}'
            mock_response.choices = [mock_choice]
            mock_acompletion.return_value = mock_response

            provider = DeepSeekProvider(llm_config)
            result = await provider.generate("Return json format")
            assert result == '{"key": "value"}'

            # Verify response_format was added
            call_kwargs = mock_acompletion.call_args.kwargs
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_ollama_provider_full_flow(self, mock_acompletion, llm_config_ollama):
        """Test Ollama provider full flow."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Ollama response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        provider = OllamaProvider(llm_config_ollama)
        result = await provider.generate("test prompt", system_prompt="system")
        assert result == "Ollama response"

        # Verify call parameters
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "ollama/llama2"
        assert call_kwargs["api_base"] == "http://localhost:11434"

    @patch("llm_engine.providers.openai_compatible.OpenAICompatibleProvider._complete_response")
    def test_provider_synchronous_call(self, mock_complete_response, llm_config):
        """Test provider synchronous call method."""
        mock_complete_response.return_value = "Sync response"
        provider = DeepSeekProvider(llm_config)

        result = provider.call(prompt="test prompt")
        assert result == "Sync response"

    @patch("llm_engine.providers.openai_compatible.OpenAICompatibleProvider._complete_response")
    def test_provider_synchronous_call_with_messages(self, mock_complete_response, llm_config):
        """Test provider synchronous call with messages."""
        mock_complete_response.return_value = "Sync response"
        provider = DeepSeekProvider(llm_config)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "test prompt"},
        ]
        result = provider.call(messages=messages)
        assert result == "Sync response"

    @patch("llm_engine.providers.openai_compatible.OpenAICompatibleProvider._stream_response")
    def test_provider_synchronous_streaming(self, mock_stream_response, llm_config):
        """Test provider synchronous streaming."""
        mock_stream_response.return_value = iter(["chunk1", "chunk2"])
        provider = DeepSeekProvider(llm_config)

        generator = provider.call(prompt="test", stream=True)
        chunks = list(generator)
        assert len(chunks) == 2
        assert chunks[0] == "chunk1"
        assert chunks[1] == "chunk2"

    @pytest.mark.asyncio
    async def test_provider_retry_logic(self, llm_config):
        """Test provider retry logic."""
        provider = DeepSeekProvider(llm_config)
        provider.config.max_retries = 2

        call_count = 0

        async def mock_generate(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "Success after retry"

        provider.generate = mock_generate
        result = await provider.generate_with_retry("test")
        assert result == "Success after retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_provider_token_estimation(self, llm_config):
        """Test provider token estimation."""
        provider = DeepSeekProvider(llm_config)

        # Test English text
        tokens_en = provider._estimate_tokens("This is a test sentence.")
        assert tokens_en > 0

        # Test Chinese text
        tokens_cn = provider._estimate_tokens("这是一个测试句子")
        assert tokens_cn > 0

        # Test mixed text
        tokens_mixed = provider._estimate_tokens("This is English and 这是中文")
        assert tokens_mixed > 0

    def test_provider_config_from_file(self, sample_providers_config_file):
        """Test creating provider config from file."""
        config = create_llm_config_from_provider("deepseek", config_path=sample_providers_config_file)
        provider = DeepSeekProvider(config)
        assert provider.config.provider == LLMProvider.DEEPSEEK
        assert provider.config.model_name == "deepseek-chat"

    def test_custom_provider_full_flow(self):
        """Test Custom provider full flow."""
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model_name="custom-model",
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
        )
        provider = CustomProvider(config)
        assert provider.config.provider == LLMProvider.CUSTOM
        assert provider.base_url == "https://custom.api.com/v1"


class TestProviderErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_provider_timeout_error(self, mock_acompletion, llm_config):
        """Test provider timeout error handling."""
        import asyncio

        mock_acompletion.side_effect = asyncio.TimeoutError("Request timeout")

        provider = DeepSeekProvider(llm_config)
        with pytest.raises(Exception, match="timeout"):
            await provider.generate("test prompt")

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_provider_api_error(self, mock_acompletion, llm_config):
        """Test provider API error handling."""
        mock_acompletion.side_effect = Exception("API error: Invalid API key")

        provider = DeepSeekProvider(llm_config)
        from llm_engine.exceptions import LLMProviderError

        with pytest.raises(LLMProviderError, match="API call failed"):
            await provider.generate("test prompt")

    def test_provider_missing_api_key(self):
        """Test provider with missing API key."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key=None,
        )
        provider = OpenAIProvider(config)
        # Should not raise error until actual call
        assert provider.api_key is None
