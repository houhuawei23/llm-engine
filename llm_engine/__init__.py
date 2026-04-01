"""
LLM Engine - Unified LLM API provider and engine library.

Provides a unified interface for calling multiple LLM APIs with support for
both synchronous and asynchronous operations.
"""

# Thread-pool batch execution (optional import surface)
from llm_engine.concurrent import (
    ThreadPoolRetryRunner,
    exponential_backoff_seconds,
    is_transient_error,
    run_thread_pool_with_retries,
)
from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.config_loader import (
    create_llm_config_from_provider,
    get_model_info,
    load_providers_config,
)
from llm_engine.engine import (
    AnthropicProvider,
    CustomProvider,
    DeepSeekProvider,
    LLMEngine,
    OllamaProvider,
    OpenAIProvider,
)
from llm_engine.exceptions import LLMConfigError, LLMProviderError
from llm_engine.factory import (
    ProviderAdapter,
    create_provider_adapter,
    create_provider_from_config,
)
from llm_engine.providers.base import BaseLLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

__version__ = "0.1.4"

__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "CustomProvider",
    "DeepSeekProvider",
    "LLMConfig",
    "LLMConfigError",
    "LLMEngine",
    "LLMProvider",
    "LLMProviderError",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ProviderAdapter",
    "ThreadPoolRetryRunner",
    "create_llm_config_from_provider",
    "create_provider_adapter",
    "create_provider_from_config",
    "exponential_backoff_seconds",
    "get_model_info",
    "is_transient_error",
    "load_providers_config",
    "run_thread_pool_with_retries",
]
