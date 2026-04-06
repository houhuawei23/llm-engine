"""
Caching middleware for LLM Engine.

Integrates the caching system with the middleware framework.
"""

from typing import Optional

from llm_engine.caching.cache import CacheConfig, LLMCache
from llm_engine.middleware.base import Middleware, RequestContext, Response


class CachingMiddleware(Middleware):
    """Middleware that adds caching to LLM requests.

    Checks cache before making requests and stores responses in cache.

    Example:
        >>> from llm_engine import LLMEngine
        >>> from llm_engine.middleware import MiddlewareChain
        >>> from llm_engine.caching import CachingMiddleware, CacheConfig
        >>>
        >>> config = CacheConfig(
        ...     enable_semantic=True,
        ...     semantic_threshold=0.9,
        ...     ttl=3600,
        ... )
        >>> cache_mw = CachingMiddleware(config)
        >>>
        >>> engine = LLMEngine(
        ...     llm_config,
        ...     middleware=[cache_mw]
        ... )

    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize caching middleware.

        Args:
            config: Cache configuration
        """
        self._cache = LLMCache(config)
        self._stats = {"hits": 0, "misses": 0, "semantic_hits": 0}

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Check cache for existing response.

        If found, store in context metadata for short-circuiting.
        """
        # Skip cache for streaming requests
        if context.stream:
            context.metadata["cache_skip"] = True
            return context

        entry = await self._cache.get(
            provider=context.provider,
            model=context.model,
            messages=context.messages,
            temperature=context.temperature,
            max_tokens=context.max_tokens,
            top_p=context.top_p,
        )

        if entry:
            # Cache hit - store response for process_response to return
            context.metadata["cache_hit"] = True
            context.metadata["cached_response"] = entry

            # Track semantic vs exact hit
            if entry.metadata.get("semantic_match"):
                self._stats["semantic_hits"] += 1
            else:
                self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Return cached response or cache new response.

        If cache hit in process_request, return cached response.
        Otherwise, store the new response in cache.
        """
        # If cache was skipped, just pass through
        if context.metadata.get("cache_skip"):
            return response

        # If cache hit, return cached response
        if context.metadata.get("cache_hit"):
            cached = context.metadata["cached_response"]

            # Create new response from cached data
            return Response(
                content=cached.content,
                reasoning=cached.metadata.get("reasoning"),
                usage=cached.metadata.get("usage", {}),
                metadata={
                    **response.metadata,
                    "cache_hit": True,
                    "cached_at": cached.created_at.isoformat(),
                },
                latency_ms=response.latency_ms,
            )

        # Cache miss - store the new response
        await self._cache.set(
            provider=context.provider,
            model=context.model,
            messages=context.messages,
            content=response.content,
            usage=response.usage,
            reasoning=response.reasoning,
            temperature=context.temperature,
            max_tokens=context.max_tokens,
            top_p=context.top_p,
        )

        # Mark as cache miss in metadata
        response.metadata["cache_hit"] = False

        return response

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with hit/miss statistics
        """
        total = self._stats["hits"] + self._stats["semantic_hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] + self._stats["semantic_hits"]) / total * 100
            if total > 0
            else 0
        )

        return {
            **self._stats,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            **self._cache.get_stats(),
        }

    async def clear(self) -> None:
        """Clear the cache."""
        await self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "semantic_hits": 0}
