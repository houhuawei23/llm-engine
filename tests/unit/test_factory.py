"""Unit tests for factory module."""

import pytest
from unittest.mock import Mock, patch

from llm_engine.config import LLMProvider
from llm_engine.exceptions import LLMProviderError
from llm_engine.factory import (
    create_provider_from_config,
    create_provider_adapter,
    ProviderAdapter,
)
from llm_engine.providers.base import BaseLLMProvider


class TestCreateProviderFromConfig:
    """Tests for create_provider_from_config function."""

    def test_create_deepseek_provider_from_dict(self, llm_config):
        """Test creating DeepSeek provider from dictionary."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-deepseek-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.provider == LLMProvider.DEEPSEEK
        assert provider.config.model_name == "deepseek-chat"
        assert provider.config.api_key == "test-deepseek-key"

    def test_create_openai_provider_from_dict(self):
        """Test creating OpenAI provider from dictionary."""
        config_dict = {
            "api_provider": "openai",
            "api_key": "test-openai-key",
            "api_base": "https://api.openai.com/v1",
            "models": ["gpt-4"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.provider == LLMProvider.OPENAI
        assert provider.config.model_name == "gpt-4"

    def test_create_custom_provider_from_dict(self):
        """Test creating Custom provider from dictionary."""
        config_dict = {
            "api_provider": "qwen",
            "api_key": "test-qwen-key",
            "api_base": "https://api.qwen.com/v1",
            "models": ["qwen-turbo"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.provider == LLMProvider.CUSTOM
        assert provider.config.model_name == "qwen-turbo"

    def test_create_provider_from_object(self):
        """Test creating provider from object with attributes."""
        config_obj = Mock()
        config_obj.api_provider = "deepseek"
        config_obj.api_key = "test-key"
        config_obj.api_base = "https://api.deepseek.com/v1"
        config_obj.models = ["deepseek-chat"]
        config_obj.api_temperature = 0.7
        config_obj.api_top_p = None
        config_obj.max_tokens = None
        config_obj.timeout = 60.0

        provider = create_provider_from_config(config_obj)
        assert provider.config.provider == LLMProvider.DEEPSEEK

    def test_create_provider_with_default_model(self):
        """Test creating provider with default_model parameter."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat", "deepseek-coder"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict, default_model="deepseek-coder")
        assert provider.config.model_name == "deepseek-coder"

    def test_create_provider_uses_first_model(self):
        """Test creating provider uses first model if default_model not specified."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat", "deepseek-coder"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.model_name == "deepseek-chat"

    def test_create_provider_missing_api_key(self):
        """Test creating provider with missing API key."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        with pytest.raises(ValueError, match="API key is required"):
            create_provider_from_config(config_dict)

    def test_create_provider_placeholder_api_key(self):
        """Test creating provider with placeholder API key."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "your-api-key-here",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        with pytest.raises(ValueError, match="API key is required"):
            create_provider_from_config(config_dict)

    def test_create_provider_missing_api_base(self):
        """Test creating provider with missing API base."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        with pytest.raises(ValueError, match="API base URL is required"):
            create_provider_from_config(config_dict)

    def test_create_provider_no_models(self):
        """Test creating provider with no models."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": [],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        with pytest.raises(ValueError, match="No default model specified"):
            create_provider_from_config(config_dict)

    def test_create_provider_with_top_p(self):
        """Test creating provider with top_p parameter."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "api_top_p": 0.9,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.top_p == 0.9

    def test_create_provider_with_max_tokens(self):
        """Test creating provider with max_tokens parameter."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60.0,
        }
        provider = create_provider_from_config(config_dict)
        assert provider.config.max_tokens == 4000


class TestCreateProviderAdapter:
    """Tests for create_provider_adapter function."""

    def test_create_adapter_from_dict(self):
        """Test creating adapter from dictionary."""
        config_dict = {
            "api_provider": "deepseek",
            "api_key": "test-key",
            "api_base": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat"],
            "api_temperature": 0.7,
            "timeout": 60.0,
        }
        adapter = create_provider_adapter(config_dict)
        assert isinstance(adapter, ProviderAdapter)
        assert adapter.name == "deepseek"
        assert adapter.default_model == "deepseek-chat"

    def test_create_adapter_from_object(self):
        """Test creating adapter from object."""
        config_obj = Mock()
        config_obj.api_provider = "deepseek"
        config_obj.api_key = "test-key"
        config_obj.api_base = "https://api.deepseek.com/v1"
        config_obj.models = ["deepseek-chat"]
        config_obj.api_temperature = 0.7
        config_obj.api_top_p = None
        config_obj.max_tokens = None
        config_obj.timeout = 60.0

        adapter = create_provider_adapter(config_obj)
        assert isinstance(adapter, ProviderAdapter)
        assert adapter.name == "deepseek"


class TestProviderAdapter:
    """Tests for ProviderAdapter class."""

    def test_adapter_initialization(self, llm_config):
        """Test adapter initialization."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)
        assert adapter.provider == provider
        assert adapter.config == config_dict

    def test_adapter_name_property(self, llm_config):
        """Test adapter name property."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)
        assert adapter.name == "deepseek"

    def test_adapter_default_model_property(self, llm_config):
        """Test adapter default_model property."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat", "deepseek-coder"],
        }
        adapter = ProviderAdapter(provider, config_dict, default_model="deepseek-coder")
        assert adapter.default_model == "deepseek-coder"

    def test_adapter_available_models_property(self, llm_config):
        """Test adapter available_models property."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat", "deepseek-coder"],
        }
        adapter = ProviderAdapter(provider, config_dict)
        assert adapter.available_models == ["deepseek-chat", "deepseek-coder"]

    def test_adapter_call_method(self, llm_config):
        """Test adapter call method."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)

        # Mock provider.call method
        provider.call = Mock(return_value="Test response")
        result = adapter.call(prompt="Test prompt")
        assert result == "Test response"
        provider.call.assert_called_once()

    def test_adapter_call_with_temperature_override(self, llm_config):
        """Test adapter call with temperature override."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)

        original_temp = provider.config.temperature
        adapter.call(prompt="Test", temperature=0.5)
        assert provider.config.temperature == 0.5
        # Restore
        provider.config.temperature = original_temp

    def test_adapter_call_with_model_override(self, llm_config):
        """Test adapter call with model override."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)

        original_model = provider.config.model_name
        adapter.call(prompt="Test", model="deepseek-coder")
        assert provider.config.model_name == "deepseek-coder"
        # Restore
        provider.config.model_name = original_model

    def test_adapter_test_connection_success(self, llm_config):
        """Test adapter test_connection method with success."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)

        # Mock provider.call to return success
        provider.call = Mock(return_value="Test response")
        success, message, latency = adapter.test_connection()
        assert success is True
        assert "Response:" in message
        assert latency >= 0

    def test_adapter_test_connection_failure(self, llm_config):
        """Test adapter test_connection method with failure."""
        from llm_engine.engine import DeepSeekProvider

        provider = DeepSeekProvider(llm_config)
        config_dict = {
            "api_provider": "deepseek",
            "models": ["deepseek-chat"],
        }
        adapter = ProviderAdapter(provider, config_dict)

        # Mock provider.call to raise exception
        provider.call = Mock(side_effect=Exception("Connection failed"))
        success, message, latency = adapter.test_connection()
        assert success is False
        assert "Connection failed" in message
        assert latency >= 0
