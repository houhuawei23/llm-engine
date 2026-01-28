"""Unit tests for config module."""

import pytest
from pydantic import ValidationError

from llm_engine.config import LLMConfig, LLMProvider


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.DEEPSEEK == "deepseek"
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.CUSTOM == "custom"
        assert LLMProvider.KIMI == "kimi"
        assert LLMProvider.ANTHROPIC == "anthropic"

    def test_provider_string_representation(self):
        """Test provider string representation."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.DEEPSEEK.value == "deepseek"


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.DEEPSEEK
        assert config.model_name == "deepseek-chat"
        assert config.api_key is None
        assert config.base_url is None
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.top_p == 1.0
        assert config.presence_penalty == 0.0
        assert config.frequency_penalty == 0.0
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.api_keys == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.test.com/v1",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            timeout=30,
            max_retries=5,
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.test.com/v1"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.presence_penalty == 0.1
        assert config.frequency_penalty == 0.2
        assert config.timeout == 30
        assert config.max_retries == 5

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

        # Invalid temperature (too low)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1

        config = LLMConfig(max_tokens=10000)
        assert config.max_tokens == 10000

        # Invalid max_tokens (too low)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)

    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid top_p
        config = LLMConfig(top_p=0.0)
        assert config.top_p == 0.0

        config = LLMConfig(top_p=1.0)
        assert config.top_p == 1.0

        # Invalid top_p (too low)
        with pytest.raises(ValidationError):
            LLMConfig(top_p=-0.1)

        # Invalid top_p (too high)
        with pytest.raises(ValidationError):
            LLMConfig(top_p=1.1)

    def test_presence_penalty_validation(self):
        """Test presence_penalty validation."""
        # Valid presence_penalty
        config = LLMConfig(presence_penalty=-2.0)
        assert config.presence_penalty == -2.0

        config = LLMConfig(presence_penalty=2.0)
        assert config.presence_penalty == 2.0

        # Invalid presence_penalty (too low)
        with pytest.raises(ValidationError):
            LLMConfig(presence_penalty=-2.1)

        # Invalid presence_penalty (too high)
        with pytest.raises(ValidationError):
            LLMConfig(presence_penalty=2.1)

    def test_frequency_penalty_validation(self):
        """Test frequency_penalty validation."""
        # Valid frequency_penalty
        config = LLMConfig(frequency_penalty=-2.0)
        assert config.frequency_penalty == -2.0

        config = LLMConfig(frequency_penalty=2.0)
        assert config.frequency_penalty == 2.0

        # Invalid frequency_penalty (too low)
        with pytest.raises(ValidationError):
            LLMConfig(frequency_penalty=-2.1)

        # Invalid frequency_penalty (too high)
        with pytest.raises(ValidationError):
            LLMConfig(frequency_penalty=2.1)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeout
        config = LLMConfig(timeout=1)
        assert config.timeout == 1

        config = LLMConfig(timeout=300)
        assert config.timeout == 300

        # Invalid timeout (too low)
        with pytest.raises(ValidationError):
            LLMConfig(timeout=0)

    def test_max_retries_validation(self):
        """Test max_retries validation."""
        # Valid max_retries
        config = LLMConfig(max_retries=0)
        assert config.max_retries == 0

        config = LLMConfig(max_retries=10)
        assert config.max_retries == 10

        # Invalid max_retries (negative)
        with pytest.raises(ValidationError):
            LLMConfig(max_retries=-1)

    def test_api_key_env_var_reference(self):
        """Test API key with environment variable reference."""
        # Should not parse env var reference in validator
        config = LLMConfig(api_key="${TEST_API_KEY}")
        assert config.api_key == "${TEST_API_KEY}"

    def test_get_api_key_with_api_keys(self):
        """Test get_api_key prioritizes api_keys list."""
        config = LLMConfig(
            api_key="old-key",
            api_keys=["first-key", "second-key"],
        )
        assert config.get_api_key() == "first-key"

    def test_get_api_key_without_api_keys(self):
        """Test get_api_key falls back to api_key."""
        config = LLMConfig(api_key="test-key")
        assert config.get_api_key() == "test-key"

    def test_get_api_key_none(self):
        """Test get_api_key returns None when no keys."""
        config = LLMConfig()
        assert config.get_api_key() is None

    def test_api_keys_default_factory(self):
        """Test api_keys default factory."""
        config1 = LLMConfig()
        config2 = LLMConfig()
        assert config1.api_keys == []
        assert config2.api_keys == []
        # Should be different lists
        assert config1.api_keys is not config2.api_keys
