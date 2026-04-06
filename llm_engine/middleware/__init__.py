"""
Middleware system for LLM Engine.

Provides pluggable request/response processing for cross-cutting concerns
like logging, caching, metrics, and authentication.

Example:
    >>> from llm_engine import LLMEngine
    >>> from llm_engine.middleware import (
    ...     MiddlewareChain,
    ...     LoggingMiddleware,
    ...     TimingMiddleware,
    ... )
    >>>
    >>> chain = MiddlewareChain([
    ...     LoggingMiddleware(),
    ...     TimingMiddleware(),
    ... ])
    >>>
    >>> engine = LLMEngine(config, middleware=chain)

"""

from llm_engine.middleware.base import (
    Middleware,
    MiddlewareError,
    RequestContext,
    Response,
)
from llm_engine.middleware.builtin import (
    ContentFilterMiddleware,
    HeaderInjectionMiddleware,
    LoggingMiddleware,
    RetryMiddleware,
    TimingMiddleware,
)
from llm_engine.middleware.chain import (
    ConditionalMiddleware,
    MiddlewareChain,
)

__all__ = [
    "ConditionalMiddleware",
    "ContentFilterMiddleware",
    "HeaderInjectionMiddleware",
    # Built-in middleware
    "LoggingMiddleware",
    # Base classes
    "Middleware",
    # Chain
    "MiddlewareChain",
    "MiddlewareError",
    "RequestContext",
    "Response",
    "RetryMiddleware",
    "TimingMiddleware",
]
