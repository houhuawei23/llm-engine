"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock

import pytest
import yaml

from llm_engine.config import LLMConfig, LLMProvider


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_providers_config_dict():
    """Sample providers.yml configuration dictionary."""
    return {
        "providers": {
            "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "test-deepseek-key",
                "default_model": "deepseek-chat",
                "models": [
                    {
                        "name": "deepseek-chat",
                        "token_per_character": {
                            "english": 0.3,
                            "chinese": 0.6,
                        },
                        "functions": {
                            "json_output": True,
                        },
                    }
                ],
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "test-openai-key",
                "default_model": "gpt-4",
                "models": [
                    {"name": "gpt-4"},
                    {"name": "gpt-3.5-turbo"},
                ],
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "api_key": None,
                "default_model": "llama2",
                "models": [
                    {"name": "llama2"},
                    {"name": "mistral"},
                ],
            },
        }
    }


@pytest.fixture
def sample_providers_config_file(temp_dir, sample_providers_config_dict):
    """Create sample providers.yml config file."""
    config_path = temp_dir / "providers.yml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_providers_config_dict, f)
    return config_path


@pytest.fixture
def llm_config():
    """Sample LLMConfig."""
    return LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="test-api-key",
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
        max_tokens=2000,
    )


@pytest.fixture
def llm_config_openai():
    """Sample LLMConfig for OpenAI."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-openai-key",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
        max_tokens=2000,
    )


@pytest.fixture
def llm_config_ollama():
    """Sample LLMConfig for Ollama."""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama2",
        api_key=None,
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2000,
    )


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "Test response"
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_litellm_stream_response():
    """Mock LiteLLM streaming response."""
    async def mock_stream():
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Test"))]),
            Mock(choices=[Mock(delta=Mock(content=" response"))]),
            Mock(choices=[Mock(delta=Mock(content=" chunk"))]),
        ]
        for chunk in chunks:
            yield chunk

    return mock_stream()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "Test response"
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_openai_stream():
    """Mock OpenAI streaming response."""
    def mock_stream():
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Test"))]),
            Mock(choices=[Mock(delta=Mock(content=" response"))]),
            Mock(choices=[Mock(delta=Mock(content=" chunk"))]),
        ]
        for chunk in chunks:
            yield chunk

    return mock_stream()


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment variables before each test."""
    # Store original values
    original_env = {}
    env_vars = [
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "CUSTOM_API_KEY",
        "OLLAMA_BASE_URL",
        "CUSTOM_API_BASE_URL",
    ]
    for var in env_vars:
        original_env[var] = os.environ.get(var)

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
