"""
Rate limiting for LLM Engine.

Provides token bucket rate limiting with provider-aware defaults.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute
        tokens_per_minute: Maximum tokens per minute
        burst_size: Maximum burst size for token bucket
        strategy: Rate limiting strategy
        wait_on_limit: Whether to wait or raise when limit reached
    """
    requests_per_minute: int = 60
    tokens_per_minute: Optional[int] = None
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    wait_on_limit: bool = True
    max_wait_seconds: float = 60.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter.

    Allows bursts up to burst_size, then throttles to
    steady rate of requests_per_minute.

    Example:
        >>> limiter = TokenBucketRateLimiter(
        ...     requests_per_minute=100,
        ...     burst_size=20
        ... )
        >>> await limiter.acquire()  # Wait if needed
        >>> # Make request
        >>> limiter.release()  # Optional: release token

    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        wait_on_limit: bool = True,
        max_wait_seconds: float = 60.0,
    ):
        """Initialize token bucket rate limiter.

        Args:
            requests_per_minute: Target request rate
            burst_size: Maximum burst (defaults to requests_per_minute // 6)
            wait_on_limit: Whether to wait or raise exception
            max_wait_seconds: Maximum time to wait
        """
        self._rate = requests_per_minute / 60.0  # Convert to per second
        self._capacity = burst_size or max(1, requests_per_minute // 6)
        self._tokens = float(self._capacity)
        self._last_update = time.monotonic()
        self._wait_on_limit = wait_on_limit
        self._max_wait = max_wait_seconds
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rejected

        Raises:
            RateLimitExceeded: If wait_on_limit is False and limit reached
        """
        async with self._lock:
            # Calculate tokens to add based on time elapsed
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate
            )
            self._last_update = now

            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not self._wait_on_limit:
                return False

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._rate

            if wait_time > self._max_wait:
                return False

        # Wait outside the lock to allow concurrent waiting
        await asyncio.sleep(wait_time)

        # Recurse to try again
        return await self.acquire(tokens)

    def get_status(self) -> dict:
        """Get current rate limiter status.

        Returns:
            Dictionary with current token count and rate
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        current_tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate
        )
        return {
            "tokens_available": current_tokens,
            "capacity": self._capacity,
            "rate_per_second": self._rate,
            "utilization": 1 - (current_tokens / self._capacity),
        }


class ProviderRateLimits:
    """Provider-specific rate limit configurations.

    Contains default rate limits for major LLM providers.
    These are conservative defaults; actual limits may vary
    based on your account tier.
    """

    DEFAULT_LIMITS: Dict[str, RateLimitConfig] = {
        "openai": RateLimitConfig(
            requests_per_minute=500,
            tokens_per_minute=150000,
            burst_size=50,
        ),
        "deepseek": RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=100000,
            burst_size=20,
        ),
        "anthropic": RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=40000,
            burst_size=100,
        ),
        "ollama": RateLimitConfig(
            requests_per_minute=10000,  # Local, essentially unlimited
            burst_size=1000,
        ),
    }

    @classmethod
    def get_config(cls, provider: str) -> RateLimitConfig:
        """Get rate limit config for provider.

        Args:
            provider: Provider name

        Returns:
            Rate limit configuration
        """
        return cls.DEFAULT_LIMITS.get(
            provider.lower(),
            RateLimitConfig(requests_per_minute=60)  # Conservative default
        )


class RateLimitManager:
    """Manages rate limiters for multiple providers/models.

    Example:
        >>> manager = RateLimitManager()
        >>>
        >>> # Get limiter for specific provider
        >>> limiter = manager.get_limiter("openai", "gpt-4")
        >>> await limiter.acquire()
        >>>
        >>> # Or use context manager
        >>> async with manager.limit("openai", "gpt-4"):
        ...     # Request is rate limited
        ...     pass

    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        provider_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ):
        """Initialize rate limit manager.

        Args:
            default_config: Default config for unknown providers
            provider_configs: Override provider configurations
        """
        self._default_config = default_config or RateLimitConfig()
        self._provider_configs = provider_configs or {}
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}

    def _get_key(self, provider: str, model: Optional[str] = None) -> str:
        """Get cache key for provider/model."""
        if model:
            return f"{provider}:{model}"
        return provider

    def get_limiter(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> TokenBucketRateLimiter:
        """Get or create rate limiter for provider.

        Args:
            provider: Provider name
            model: Model name (optional)

        Returns:
            TokenBucketRateLimiter instance
        """
        key = self._get_key(provider, model)

        if key not in self._limiters:
            # Get config for this provider
            if provider in self._provider_configs:
                config = self._provider_configs[provider]
            elif provider in ProviderRateLimits.DEFAULT_LIMITS:
                config = ProviderRateLimits.get_config(provider)
            else:
                config = self._default_config

            self._limiters[key] = TokenBucketRateLimiter(
                requests_per_minute=config.requests_per_minute,
                burst_size=config.burst_size,
                wait_on_limit=config.wait_on_limit,
                max_wait_seconds=config.max_wait_seconds,
            )

        return self._limiters[key]

    async def acquire(
        self,
        provider: str,
        model: Optional[str] = None,
        tokens: int = 1,
    ) -> bool:
        """Acquire rate limit for provider.

        Args:
            provider: Provider name
            model: Model name
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rejected
        """
        limiter = self.get_limiter(provider, model)
        return await limiter.acquire(tokens)

    def limit(self, provider: str, model: Optional[str] = None):
        """Context manager for rate limiting.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Async context manager
        """
        return RateLimitContext(self, provider, model)

    def get_status(self, provider: str, model: Optional[str] = None) -> dict:
        """Get rate limit status for provider.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Status dictionary
        """
        limiter = self.get_limiter(provider, model)
        return {
            "provider": provider,
            "model": model,
            **limiter.get_status(),
        }


class RateLimitContext:
    """Async context manager for rate limiting."""

    def __init__(
        self,
        manager: RateLimitManager,
        provider: str,
        model: Optional[str] = None,
    ):
        """Initialize context manager.

        Args:
            manager: RateLimitManager instance
            provider: Provider name
            model: Model name
        """
        self._manager = manager
        self._provider = provider
        self._model = model

    async def __aenter__(self):
        """Enter context - acquire rate limit."""
        acquired = await self._manager.acquire(self._provider, self._model)
        if not acquired:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {self._provider}"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - nothing to do."""


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
