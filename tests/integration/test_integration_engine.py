"""Integration tests for LLM Engine."""

import pytest
from unittest.mock import patch, AsyncMock, Mock

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.engine import LLMEngine
from llm_engine.factory import create_provider_from_config, create_provider_adapter


class TestLLMEngineIntegration:
    """Integration tests for LLMEngine."""

    @pytest.mark.asyncio
    async def test_engine_generate_flow(self, llm_config):
        """Test engine generate flow."""
        engine = LLMEngine(llm_config)
        engine.provider.generate_with_retry = AsyncMock(return_value="Engine response")

        result = await engine.generate("test prompt", system_prompt="system")
        assert result == "Engine response"
        engine.provider.generate_with_retry.assert_called_once_with("test prompt", "system")

    @pytest.mark.asyncio
    async def test_engine_stream_generate_flow(self, llm_config):
        """Test engine stream generate flow."""
        engine = LLMEngine(llm_config)

        async def mock_stream(prompt, system_prompt=None):
            yield ("chunk1", 10)
            yield ("chunk2", 20)
            yield ("chunk3", 30)

        engine.provider.generate_stream = mock_stream
        chunks = []
        async for chunk, tokens in engine.stream_generate("test prompt"):
            chunks.append((chunk, tokens))

        assert len(chunks) == 3
        assert chunks[0] == ("chunk1", 10)
        assert chunks[1] == ("chunk2", 20)
        assert chunks[2] == ("chunk3", 30)

    def test_engine_provider_selection(self):
        """Test engine provider selection."""
        # Test OpenAI
        config_openai = LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4")
        engine_openai = LLMEngine(config_openai)
        assert engine_openai.provider.__class__.__name__ == "OpenAIProvider"

        # Test DeepSeek
        config_deepseek = LLMConfig(provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat")
        engine_deepseek = LLMEngine(config_deepseek)
        assert engine_deepseek.provider.__class__.__name__ == "DeepSeekProvider"

        # Test Ollama
        config_ollama = LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama2")
        engine_ollama = LLMEngine(config_ollama)
        assert engine_ollama.provider.__class__.__name__ == "OllamaProvider"

        # Test Custom
        config_custom = LLMConfig(provider=LLMProvider.CUSTOM, model_name="custom-model")
        engine_custom = LLMEngine(config_custom)
        assert engine_custom.provider.__class__.__name__ == "CustomProvider"


class TestFactoryIntegration:
    """Integration tests for factory functions."""

    def test_create_provider_from_config_dict(self):
        """Test creating provider from config dictionary."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.provider == LLMProvider.DEEPSEEK
        assert provider.config.model_name == "deepseek-chat"

    def test_create_provider_adapter_from_config(self):
        """Test creating provider adapter from config."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat", "deepseek-coder"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        adapter = create_provider_adapter(config_dict, default_model="deepseek-coder")
        assert adapter.name == "deepseek"
        assert adapter.default_model == "deepseek-coder"
        assert adapter.available_models == ["deepseek-chat", "deepseek-coder"]

    def test_adapter_call_integration(self):
        """Test adapter call integration."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        adapter = create_provider_adapter(config_dict)
        adapter.provider.call = Mock(return_value="Adapter response")

        result = adapter.call(prompt="test")
        assert result == "Adapter response"

    def test_adapter_test_connection_integration(self):
        """Test adapter test connection integration."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        adapter = create_provider_adapter(config_dict)
        adapter.provider.call = Mock(return_value="Connection test response")

        success, message, latency = adapter.test_connection()
        assert success is True
        assert "Response:" in message
        assert latency >= 0


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_end_to_end_generate(self, mock_acompletion, llm_config):
        """Test end-to-end generate flow."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "End-to-end response"
        mock_response.choices = [mock_choice]
        mock_acompletion.return_value = mock_response

        # Create engine
        engine = LLMEngine(llm_config)
        result = await engine.generate("test prompt", system_prompt="system")

        assert result == "End-to-end response"
        mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_engine.providers.openai_compatible.litellm.acompletion")
    async def test_end_to_end_stream(self, mock_acompletion, llm_config):
        """Test end-to-end streaming flow."""
        async def mock_stream():
            chunks = [
                Mock(choices=[Mock(delta=Mock(content="Stream"))]),
                Mock(choices=[Mock(delta=Mock(content=" response"))]),
                Mock(choices=[Mock(delta=Mock(content=" chunks"))]),
            ]
            for chunk in chunks:
                yield chunk

        mock_acompletion.return_value = mock_stream()

        # Create engine
        engine = LLMEngine(llm_config)
        chunks = []
        async for chunk, tokens in engine.stream_generate("test prompt"):
            chunks.append((chunk, tokens))

        assert len(chunks) == 3
        assert chunks[0][0] == "Stream"
        assert chunks[1][0] == " response"
        assert chunks[2][0] == " chunks"

    @patch("llm_engine.providers.openai_compatible.OpenAICompatibleProvider._complete_response")
    def test_end_to_end_synchronous_call(self, mock_complete_response, llm_config):
        """Test end-to-end synchronous call flow."""
        mock_complete_response.return_value = ("Sync response", None)
        engine = LLMEngine(llm_config)

        result = engine.provider.call(prompt="test prompt")
        assert result == "Sync response"

    def test_end_to_end_config_loading(self, sample_providers_config_file):
        """Test end-to-end config loading flow."""
        from llm_engine.config_loader import create_llm_config_from_provider

        config = create_llm_config_from_provider("deepseek", config_path=sample_providers_config_file)
        engine = LLMEngine(config)

        assert engine.config.provider == LLMProvider.DEEPSEEK
        assert engine.config.model_name == "deepseek-chat"
        assert isinstance(engine.provider, type(engine.provider))
