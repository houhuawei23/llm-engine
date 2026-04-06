"""
Performance middleware for LLM Engine.

Integrates rate limiting and connection pooling with the middleware framework.
"""

import asyncio
from typing import Optional

from llm_engine.middleware.base import Middleware, RequestContext, Response
from llm_engine.performance.rate_limiting import (
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitManager,
)


class RateLimitingMiddleware(Middleware):
    """Middleware for rate limiting LLM requests.

    Prevents exceeding provider rate limits with automatic waiting
    or error raising.
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        provider_configs: Optional[dict] = None,
    ):
        """Initialize rate limiting middleware."""
        self._manager = RateLimitManager(
            default_config=default_config,
            provider_configs=provider_configs,
        )

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Acquire rate limit before request."""
        acquired = await self._manager.acquire(
            provider=context.provider,
            model=context.model,
        )

        if not acquired:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {context.provider}/{context.model}",
                provider=context.provider,
            )

        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Pass through - no action on response."""
        return response

    def get_status(self, provider: str, model: Optional[str] = None) -> dict:
        """Get rate limit status for provider."""
        return self._manager.get_status(provider, model)


class ConcurrencyMiddleware(Middleware):
    """Middleware for limiting concurrent requests."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize concurrency middleware."""
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Acquire semaphore before request."""
        await self._semaphore.acquire()
        context.metadata["_concurrency_semaphore"] = self._semaphore
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Release semaphore after request."""
        sem = context.metadata.get("_concurrency_semaphore")
        if sem:
            sem.release()
        return response


class PerformanceMiddleware(Middleware):
    """Combined performance middleware with rate limiting and concurrency."""

    def __init__(
        self,
        rate_limiting: bool = True,
        max_concurrent: int = 10,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize performance middleware."""
        self._rate_limiting = rate_limiting
        self._concurrency = ConcurrencyMiddleware(max_concurrent)

        if rate_limiting:
            self._rate_limiter = RateLimitingMiddleware(
                default_config=rate_limit_config
            )
        else:
            self._rate_limiter = None

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Apply rate limiting and concurrency control."""
        if self._rate_limiter:
            context = await self._rate_limiter.process_request(context)
        context = await self._concurrency.process_request(context)
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Release concurrency semaphore."""
        response = await self._concurrency.process_response(context, response)
        if self._rate_limiter:
            response = await self._rate_limiter.process_response(context, response)
        return response

    def get_rate_limit_status(self, provider: str, model: Optional[str] = None) -> Optional[dict]:
        """Get rate limit status if enabled."""
        if self._rate_limiter:
            return self._rate_limiter.get_status(provider, model)
        return None
