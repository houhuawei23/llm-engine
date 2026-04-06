"""
LLM Engine - Unified LLM API provider and engine library.

Provides a unified interface for calling multiple LLM APIs with support for
both synchronous and asynchronous operations.
"""

# Thread-pool batch execution (optional import surface)
from llm_engine.caching import (
    CacheConfig,
    CacheEntry,
    CachingMiddleware,
    DiskCacheBackend,
    LLMCache,
    MemoryCacheBackend,
    RedisCacheBackend,
    SemanticCache,
    SimpleEmbedder,
)
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
from llm_engine.middleware import (
    ContentFilterMiddleware,
    HeaderInjectionMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    MiddlewareError,
    RequestContext,
    Response,
    RetryMiddleware,
    TimingMiddleware,
)
from llm_engine.observability import (
    CostEntry,
    CostTracker,
    FileCostStorage,
    InMemoryCostStorage,
    InMemoryMetricsCollector,
    ObservabilityMiddleware,
    PricingProvider,
    PrometheusMetricsCollector,
    RequestMetrics,
)
from llm_engine.performance import (
    ConcurrencyMiddleware,
    ConnectionPool,
    PerformanceMiddleware,
    ProviderRateLimits,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitingMiddleware,
    RateLimitManager,
)
from llm_engine.providers.base import BaseLLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

__version__ = "0.2.0"

__all__ = [
    # Core classes
    "AnthropicProvider",
    "BaseLLMProvider",
    # Caching
    "CacheConfig",
    "CacheEntry",
    "CachingMiddleware",
    "ConcurrencyMiddleware",
    # Performance
    "ConnectionPool",
    # Middleware
    "ContentFilterMiddleware",
    # Observability
    "CostEntry",
    "CostTracker",
    "CustomProvider",
    "DeepSeekProvider",
    "DiskCacheBackend",
    "FileCostStorage",
    "HeaderInjectionMiddleware",
    "InMemoryCostStorage",
    "InMemoryMetricsCollector",
    "LLMCache",
    "LLMConfig",
    "LLMConfigError",
    "LLMEngine",
    "LLMProvider",
    "LLMProviderError",
    "LoggingMiddleware",
    "MemoryCacheBackend",
    "Middleware",
    "MiddlewareChain",
    "MiddlewareError",
    "ObservabilityMiddleware",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "PerformanceMiddleware",
    "PricingProvider",
    "PrometheusMetricsCollector",
    "ProviderAdapter",
    "ProviderRateLimits",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitManager",
    "RateLimitingMiddleware",
    "RedisCacheBackend",
    "RequestContext",
    "RequestMetrics",
    "Response",
    "RetryMiddleware",
    "SemanticCache",
    "SimpleEmbedder",
    "ThreadPoolRetryRunner",
    "TimingMiddleware",
    # Functions
    "create_llm_config_from_provider",
    "create_provider_adapter",
    "create_provider_from_config",
    "exponential_backoff_seconds",
    "get_model_info",
    "is_transient_error",
    "load_providers_config",
    "run_thread_pool_with_retries",
]
