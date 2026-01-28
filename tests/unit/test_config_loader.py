"""Unit tests for config_loader module."""

import os
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.config_loader import (
    resolve_env_vars,
    load_providers_config,
    create_llm_config_from_provider,
    get_model_info,
)


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolve_simple_string(self):
        """Test resolving simple string without env vars."""
        result = resolve_env_vars("simple string")
        assert result == "simple string"

    def test_resolve_env_var_in_string(self):
        """Test resolving environment variable in string."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            result = resolve_env_vars("prefix_${TEST_VAR}_suffix")
            assert result == "prefix_test_value_suffix"
        finally:
            os.environ.pop("TEST_VAR", None)

    def test_resolve_multiple_env_vars(self):
        """Test resolving multiple environment variables."""
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"
        try:
            result = resolve_env_vars("${VAR1}_and_${VAR2}")
            assert result == "value1_and_value2"
        finally:
            os.environ.pop("VAR1", None)
            os.environ.pop("VAR2", None)

    def test_resolve_missing_env_var(self):
        """Test resolving missing environment variable."""
        # Test that missing env var is handled gracefully
        # The function should return the original string with env var reference
        result = resolve_env_vars("prefix_${MISSING_VAR}_suffix")
        assert result == "prefix_${MISSING_VAR}_suffix"
        # Warning is logged to stderr via loguru, which is hard to capture in pytest
        # So we just verify the function doesn't crash and returns original string

    def test_resolve_dict(self):
        """Test resolving environment variables in dictionary."""
        os.environ["TEST_KEY"] = "test_value"
        try:
            data = {"key": "${TEST_KEY}", "other": "value"}
            result = resolve_env_vars(data)
            assert result == {"key": "test_value", "other": "value"}
        finally:
            os.environ.pop("TEST_KEY", None)

    def test_resolve_list(self):
        """Test resolving environment variables in list."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            data = ["${TEST_VAR}", "other"]
            result = resolve_env_vars(data)
            assert result == ["test_value", "other"]
        finally:
            os.environ.pop("TEST_VAR", None)

    def test_resolve_nested_structure(self):
        """Test resolving environment variables in nested structure."""
        os.environ["NESTED_VAR"] = "nested_value"
        try:
            data = {
                "key1": "${NESTED_VAR}",
                "key2": {
                    "nested": "${NESTED_VAR}",
                    "list": ["${NESTED_VAR}", "other"],
                },
            }
            result = resolve_env_vars(data)
            assert result == {
                "key1": "nested_value",
                "key2": {
                    "nested": "nested_value",
                    "list": ["nested_value", "other"],
                },
            }
        finally:
            os.environ.pop("NESTED_VAR", None)

    def test_resolve_non_string_value(self):
        """Test resolving non-string value."""
        result = resolve_env_vars(123)
        assert result == 123

        result = resolve_env_vars(None)
        assert result is None


class TestLoadProvidersConfig:
    """Tests for load_providers_config function."""

    def test_load_config_from_file(self, sample_providers_config_file):
        """Test loading config from file."""
        config = load_providers_config(sample_providers_config_file)
        assert "providers" in config
        assert "deepseek" in config["providers"]
        assert "openai" in config["providers"]

    def test_load_config_default_locations(self, temp_dir, sample_providers_config_dict):
        """Test loading config from default locations."""
        # Create providers.yml in current directory
        config_path = temp_dir / "providers.yml"
        import yaml

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(sample_providers_config_dict, f)

        with patch("llm_engine.config_loader.Path") as mock_path:
            # Mock Path to return our temp file
            mock_path.return_value.exists.return_value = False
            mock_path.side_effect = lambda p: config_path if str(p) == str(config_path) else Path(p)

            # Test with explicit path
            config = load_providers_config(config_path)
            assert "providers" in config

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        non_existent = Path("/non/existent/path/providers.yml")
        with pytest.raises(FileNotFoundError):
            load_providers_config(non_existent)

    def test_load_config_empty_file(self, temp_dir):
        """Test loading empty config file."""
        config_path = temp_dir / "providers.yml"
        config_path.write_text("")
        config = load_providers_config(config_path)
        assert config == {}

    def test_load_config_with_env_vars(self, temp_dir):
        """Test loading config with environment variables."""
        os.environ["TEST_API_KEY"] = "env_key_value"
        try:
            config_content = """
providers:
  test:
    api_key: ${TEST_API_KEY}
"""
            config_path = temp_dir / "providers.yml"
            config_path.write_text(config_content)
            config = load_providers_config(config_path)
            assert config["providers"]["test"]["api_key"] == "env_key_value"
        finally:
            os.environ.pop("TEST_API_KEY", None)


class TestCreateLLMConfigFromProvider:
    """Tests for create_llm_config_from_provider function."""

    def test_create_config_from_provider(self, sample_providers_config_file):
        """Test creating config from provider."""
        config = create_llm_config_from_provider("deepseek", config_path=sample_providers_config_file)
        assert config.provider == LLMProvider.DEEPSEEK
        assert config.model_name == "deepseek-chat"
        assert config.api_key == "test-deepseek-key"
        assert config.base_url == "https://api.deepseek.com/v1"

    def test_create_config_with_model_name(self, sample_providers_config_file):
        """Test creating config with specific model name."""
        config = create_llm_config_from_provider(
            "openai", model_name="gpt-3.5-turbo", config_path=sample_providers_config_file
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-3.5-turbo"

    def test_create_config_with_overrides(self, sample_providers_config_file):
        """Test creating config with overrides."""
        config = create_llm_config_from_provider(
            "deepseek",
            config_path=sample_providers_config_file,
            temperature=0.5,
            max_tokens=1000,
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_create_config_provider_not_found(self, sample_providers_config_file):
        """Test creating config with non-existent provider."""
        with pytest.raises(ValueError, match="Provider 'nonexistent' not found"):
            create_llm_config_from_provider("nonexistent", config_path=sample_providers_config_file)

    def test_create_config_no_model_specified(self, temp_dir):
        """Test creating config when no model is specified."""
        config_content = """
providers:
  test:
    api_key: test-key
    base_url: https://api.test.com/v1
"""
        config_path = temp_dir / "providers.yml"
        config_path.write_text(config_content)
        with pytest.raises(ValueError, match="No model specified"):
            create_llm_config_from_provider("test", config_path=config_path)

    def test_create_config_uses_first_model(self, sample_providers_config_file):
        """Test creating config uses first model if default_model not specified."""
        # Modify config to remove default_model
        import yaml

        config_dict = yaml.safe_load(sample_providers_config_file.read_text())
        del config_dict["providers"]["openai"]["default_model"]

        config_path = sample_providers_config_file.parent / "modified_providers.yml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

        config = create_llm_config_from_provider("openai", config_path=config_path)
        assert config.model_name == "gpt-4"  # First model in list


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_get_model_info_dict_format(self, sample_providers_config_file):
        """Test getting model info in dict format."""
        info = get_model_info("deepseek", "deepseek-chat", config_path=sample_providers_config_file)
        assert info is not None
        assert "token_per_character" in info
        assert "functions" in info

    def test_get_model_info_string_format(self, sample_providers_config_file):
        """Test getting model info in string format."""
        info = get_model_info("openai", "gpt-4", config_path=sample_providers_config_file)
        # When model is a string in config, it returns empty dict
        # But if it's a dict with name, it returns the dict
        assert isinstance(info, dict)

    def test_get_model_info_not_found(self, sample_providers_config_file):
        """Test getting model info for non-existent model."""
        info = get_model_info("deepseek", "nonexistent-model", config_path=sample_providers_config_file)
        assert info is None

    def test_get_model_info_provider_not_found(self, sample_providers_config_file):
        """Test getting model info for non-existent provider."""
        info = get_model_info("nonexistent", "model", config_path=sample_providers_config_file)
        assert info is None
