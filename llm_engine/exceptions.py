"""
Exceptions for LLM Engine.
"""


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMConfigError(Exception):
    """Exception for LLM configuration errors."""

    pass
