"""
LLM Engine - Unified LLM API provider and engine library.

Provides a unified interface for calling multiple LLM APIs with support for
both synchronous and asynchronous operations.
"""

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.config_loader import (
    create_llm_config_from_provider,
    get_model_info,
    load_providers_config,
)
from llm_engine.engine import (
    CustomProvider,
    DeepSeekProvider,
    LLMEngine,
    OllamaProvider,
    OpenAIProvider,
)
from llm_engine.exceptions import LLMConfigError, LLMProviderError
from llm_engine.factory import ProviderAdapter, create_provider_adapter, create_provider_from_config
from llm_engine.providers.base import BaseLLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

__version__ = "0.1.3"

__all__ = [
    "LLMConfig",
    "LLMProvider",
    "LLMEngine",
    "BaseLLMProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "OllamaProvider",
    "CustomProvider",
    "create_llm_config_from_provider",
    "create_provider_from_config",
    "create_provider_adapter",
    "ProviderAdapter",
    "get_model_info",
    "load_providers_config",
    "LLMProviderError",
    "LLMConfigError",
]
