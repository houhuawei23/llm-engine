"""
Caching system for LLM Engine.

Provides two-tier caching (exact + semantic similarity) with pluggable backends.

Example:
    >>> from llm_engine import LLMEngine
    >>> from llm_engine.caching import CachingMiddleware, CacheConfig
    >>>
    >>> config = CacheConfig(
    ...     enable_semantic=True,
    ...     semantic_threshold=0.9,
    ...     ttl=3600,
    ... )
    >>>
    >>> engine = LLMEngine(
    ...     llm_config,
    ...     middleware=[CachingMiddleware(config)]
    ... )

"""

from llm_engine.caching.backends import (
    CacheBackend,
    CacheEntry,
    DiskCacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
)
from llm_engine.caching.cache import CacheConfig, LLMCache
from llm_engine.caching.middleware import CachingMiddleware
from llm_engine.caching.semantic import (
    OpenAIEmbedder,
    SemanticCache,
    SimpleEmbedder,
    cosine_similarity,
)

__all__ = [
    # Backends
    "CacheBackend",
    "CacheConfig",
    "CacheEntry",
    # Middleware
    "CachingMiddleware",
    "DiskCacheBackend",
    # Main cache
    "LLMCache",
    "MemoryCacheBackend",
    "OpenAIEmbedder",
    "RedisCacheBackend",
    # Semantic
    "SemanticCache",
    "SimpleEmbedder",
    "cosine_similarity",
]
