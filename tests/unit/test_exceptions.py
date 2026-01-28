"""Unit tests for exceptions module."""

import pytest

from llm_engine.exceptions import LLMProviderError, LLMConfigError


class TestLLMProviderError:
    """Tests for LLMProviderError exception."""

    def test_exception_creation(self):
        """Test creating LLMProviderError."""
        error = LLMProviderError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_exception_with_message(self):
        """Test LLMProviderError with custom message."""
        error = LLMProviderError("API call failed")
        assert str(error) == "API call failed"

    def test_exception_inheritance(self):
        """Test LLMProviderError inheritance."""
        error = LLMProviderError("Test")
        assert isinstance(error, Exception)


class TestLLMConfigError:
    """Tests for LLMConfigError exception."""

    def test_exception_creation(self):
        """Test creating LLMConfigError."""
        error = LLMConfigError("Config error")
        assert str(error) == "Config error"
        assert isinstance(error, Exception)

    def test_exception_with_message(self):
        """Test LLMConfigError with custom message."""
        error = LLMConfigError("Invalid configuration")
        assert str(error) == "Invalid configuration"

    def test_exception_inheritance(self):
        """Test LLMConfigError inheritance."""
        error = LLMConfigError("Test")
        assert isinstance(error, Exception)
