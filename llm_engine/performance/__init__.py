"""
Performance utilities for LLM Engine.

Provides rate limiting, connection pooling, and concurrency control.

Example:
    >>> from llm_engine import LLMEngine
    >>> from llm_engine.performance import PerformanceMiddleware
    >>>
    >>> mw = PerformanceMiddleware(
    ...     rate_limiting=True,
    ...     max_concurrent=20,
    ... )
    >>> engine = LLMEngine(config, middleware=[mw])

"""

from llm_engine.performance.connection_pool import ConnectionPool, PooledClient
from llm_engine.performance.middleware import (
    ConcurrencyMiddleware,
    PerformanceMiddleware,
    RateLimitingMiddleware,
)
from llm_engine.performance.rate_limiting import (
    ProviderRateLimits,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitManager,
    RateLimitStrategy,
    TokenBucketRateLimiter,
)

__all__ = [
    "ConcurrencyMiddleware",
    # Connection pooling
    "ConnectionPool",
    "PerformanceMiddleware",
    "PooledClient",
    "ProviderRateLimits",
    # Rate limiting
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitManager",
    "RateLimitStrategy",
    # Middleware
    "RateLimitingMiddleware",
    "TokenBucketRateLimiter",
]
