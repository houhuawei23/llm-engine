"""
Main caching implementation for LLM Engine.

Provides two-tier caching: exact match and semantic similarity.
Integrates with the middleware system.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from llm_engine.caching.backends import CacheBackend, CacheEntry, MemoryCacheBackend
from llm_engine.caching.semantic import SemanticCache, SimpleEmbedder


@dataclass
class CacheConfig:
    """Configuration for LLM caching.

    Attributes:
        backend: Cache backend to use (defaults to memory)
        enable_semantic: Whether to enable semantic similarity caching
        semantic_threshold: Similarity threshold for semantic hits (0-1)
        ttl: Default time-to-live in seconds
        exclude_providers: Providers to exclude from caching
        exclude_models: Models to exclude from caching
        embedder: Custom embedder function for semantic caching
    """

    backend: Optional[CacheBackend] = None
    enable_semantic: bool = False
    semantic_threshold: float = 0.95
    ttl: Optional[int] = None
    exclude_providers: List[str] = field(default_factory=list)
    exclude_models: List[str] = field(default_factory=list)
    embedder: Optional[Callable[[str], List[float]]] = None

    def __post_init__(self):
        """Initialize default backend if not provided."""
        if self.backend is None:
            self.backend = MemoryCacheBackend()


class LLMCache:
    """Two-tier LLM response cache.

    First tier: Exact match based on (provider, model, messages, params)
    Second tier: Semantic similarity (optional)

    Example:
        >>> from llm_engine.caching import LLMCache, CacheConfig
        >>> config = CacheConfig(
        ...     enable_semantic=True,
        ...     semantic_threshold=0.9,
        ...     ttl=3600,
        ... )
        >>> cache = LLMCache(config)
        >>>
        >>> # Check cache
        >>> entry = await cache.get("openai", "gpt-4", messages, temperature=0.7)
        >>> if entry:
        ...     print(f"Cache hit: {entry.content}")
        >>> else:
        ...     # Call LLM and cache result
        ...     response = await call_llm(...)
        ...     await cache.set(messages, response, provider="openai", model="gpt-4")

    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize LLM cache.

        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._backend = self._config.backend or MemoryCacheBackend()
        self._semantic: Optional[SemanticCache] = None

        if self._config.enable_semantic:
            self._semantic = SemanticCache(
                backend=self._backend,
                embedder=self._config.embedder or SimpleEmbedder(),
                similarity_threshold=self._config.semantic_threshold,
                embedding_key_prefix="semantic:",
            )

    def _generate_key(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        **params
    ) -> str:
        """Generate cache key from request parameters.

        The key includes provider, model, messages, and relevant parameters
        that affect the response (temperature, max_tokens, etc.).
        """
        # Build key components
        key_data = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "presence_penalty": params.get("presence_penalty"),
            "frequency_penalty": params.get("frequency_penalty"),
        }

        # Remove None values
        key_data = {k: v for k, v in key_data.items() if v is not None}

        # Generate hash
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _should_cache(self, provider: str, model: str) -> bool:
        """Check if caching is enabled for this provider/model."""
        if provider in self._config.exclude_providers:
            return False
        return model not in self._config.exclude_models

    async def get(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        **params
    ) -> Optional[CacheEntry]:
        """Get cached response.

        First tries exact match, then semantic similarity if enabled.

        Args:
            provider: LLM provider name
            model: Model name
            messages: List of message dictionaries
            **params: Additional parameters that affect generation

        Returns:
            Cached entry if found, None otherwise
        """
        if not self._should_cache(provider, model):
            return None

        # Try exact match first
        key = self._generate_key(provider, model, messages, **params)
        entry = await self._backend.get(key)

        if entry:
            return entry

        # Try semantic match if enabled and we have a single user message
        if self._semantic and len(messages) == 1:
            prompt = messages[0].get("content", "")
            if prompt:
                entry = await self._semantic.get(prompt)
                if entry:
                    return entry

        return None

    async def set(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        content: str,
        usage: Optional[Dict[str, int]] = None,
        reasoning: Optional[str] = None,
        **params
    ) -> None:
        """Cache a response.

        Args:
            provider: LLM provider name
            model: Model name
            messages: List of message dictionaries
            content: Response content
            usage: Token usage information
            reasoning: Optional reasoning content
            **params: Additional parameters that affect generation
        """
        if not self._should_cache(provider, model):
            return

        entry = CacheEntry(
            content=content,
            metadata={
                "provider": provider,
                "model": model,
                "usage": usage or {},
                "reasoning": reasoning,
                **params,
            },
            created_at=datetime.utcnow(),
            ttl=self._config.ttl,
        )

        # Store exact match
        key = self._generate_key(provider, model, messages, **params)
        await self._backend.set(key, entry, self._config.ttl)

        # Store semantic match if enabled and single message
        if self._semantic and len(messages) == 1:
            prompt = messages[0].get("content", "")
            if prompt:
                await self._semantic.set(prompt, entry, self._config.ttl)

    async def delete(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        **params
    ) -> bool:
        """Delete a cached entry.

        Args:
            provider: LLM provider name
            model: Model name
            messages: List of message dictionaries
            **params: Additional parameters

        Returns:
            True if deleted, False if not found
        """
        key = self._generate_key(provider, model, messages, **params)
        result = await self._backend.delete(key)

        # Also delete from semantic cache if applicable
        if self._semantic and len(messages) == 1:
            prompt = messages[0].get("content", "")
            if prompt:
                await self._semantic.delete(prompt)

        return result

    async def clear(self) -> None:
        """Clear all cached entries."""
        await self._backend.clear()

    async def close(self) -> None:
        """Close the cache and release resources."""
        await self._backend.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "backend_type": type(self._backend).__name__,
            "semantic_enabled": self._semantic is not None,
            "excluded_providers": self._config.exclude_providers,
            "excluded_models": self._config.exclude_models,
        }

        if isinstance(self._backend, MemoryCacheBackend):
            stats["memory_entries"] = len(self._backend)

        return stats
