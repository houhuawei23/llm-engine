"""Tests for performance utilities."""

import pytest
import asyncio

from llm_engine.performance import (
    RateLimitConfig,
    TokenBucketRateLimiter,
    RateLimitManager,
    ProviderRateLimits,
    RateLimitExceeded,
)


class TestRateLimitConfig:
    """Test RateLimitConfig."""

    def test_default_creation(self):
        """Test default configuration."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.burst_size == 10
        assert config.wait_on_limit is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=20,
            wait_on_limit=False,
        )
        assert config.requests_per_minute == 100
        assert config.burst_size == 20
        assert config.wait_on_limit is False


class TestTokenBucketRateLimiter:
    """Test TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            burst_size=10,
            wait_on_limit=False,
        )

        # Should succeed (within burst)
        for _ in range(10):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_reject_when_empty(self):
        """Test rejection when bucket empty."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            burst_size=1,
            wait_on_limit=False,
        )

        # First acquire succeeds
        assert await limiter.acquire() is True

        # Second should fail (wait_on_limit=False)
        assert await limiter.acquire() is False

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self):
        """Test waiting for tokens."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=6000,  # 100/sec for fast test
            burst_size=1,
            wait_on_limit=True,
            max_wait_seconds=1.0,
        )

        # First acquire
        assert await limiter.acquire() is True

        # Second should wait and succeed
        start = asyncio.get_event_loop().time()
        result = await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert result is True
        assert elapsed > 0.01  # Should have waited a bit

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status reporting."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            burst_size=10,
        )

        status = limiter.get_status()
        assert "tokens_available" in status
        assert "capacity" in status
        assert status["capacity"] == 10

    @pytest.mark.asyncio
    async def test_max_wait_timeout(self):
        """Test that max_wait is respected."""
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,  # 1 per second
            burst_size=1,
            wait_on_limit=True,
            max_wait_seconds=0.01,  # Very short wait
        )

        # Use up the token
        await limiter.acquire()

        # Should fail due to max wait
        result = await limiter.acquire()
        assert result is False


class TestProviderRateLimits:
    """Test ProviderRateLimits."""

    def test_get_config_openai(self):
        """Test OpenAI config."""
        config = ProviderRateLimits.get_config("openai")
        assert config.requests_per_minute == 500
        assert config.tokens_per_minute == 150000

    def test_get_config_deepseek(self):
        """Test DeepSeek config."""
        config = ProviderRateLimits.get_config("deepseek")
        assert config.requests_per_minute == 100

    def test_get_config_unknown(self):
        """Test unknown provider returns default."""
        config = ProviderRateLimits.get_config("unknown_provider")
        assert config.requests_per_minute == 60  # Default

    def test_case_insensitive(self):
        """Test case insensitive lookup."""
        config1 = ProviderRateLimits.get_config("OpenAI")
        config2 = ProviderRateLimits.get_config("openai")
        assert config1.requests_per_minute == config2.requests_per_minute


class TestRateLimitManager:
    """Test RateLimitManager."""

    @pytest.fixture
    def manager(self):
        return RateLimitManager()

    @pytest.mark.asyncio
    async def test_get_limiter_creates_new(self, manager):
        """Test that get_limiter creates new limiter."""
        limiter = manager.get_limiter("openai", "gpt-4")
        assert limiter is not None

    @pytest.mark.asyncio
    async def test_get_limiter_caches(self, manager):
        """Test that limiters are cached."""
        limiter1 = manager.get_limiter("openai", "gpt-4")
        limiter2 = manager.get_limiter("openai", "gpt-4")
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_acquire(self, manager):
        """Test acquire method."""
        result = await manager.acquire("openai", "gpt-4")
        assert result is True

    @pytest.mark.asyncio
    async def test_context_manager(self, manager):
        """Test async context manager."""
        async with manager.limit("openai", "gpt-4"):
            pass  # Should succeed

    @pytest.mark.asyncio
    async def test_context_manager_raises_on_limit(self):
        """Test that context manager raises when limit exceeded."""
        manager = RateLimitManager(
            default_config=RateLimitConfig(
                requests_per_minute=60,
                burst_size=1,
                wait_on_limit=False,
            )
        )

        # Use up the token
        await manager.acquire("test")

        # Context manager should raise
        with pytest.raises(RateLimitExceeded):
            async with manager.limit("test"):
                pass

    def test_get_status(self, manager):
        """Test getting status."""
        # First create a limiter
        manager.get_limiter("openai", "gpt-4")

        status = manager.get_status("openai", "gpt-4")
        assert status["provider"] == "openai"
        assert status["model"] == "gpt-4"
        assert "tokens_available" in status


class TestRateLimitExceeded:
    """Test RateLimitExceeded exception."""

    def test_exception_message(self):
        """Test exception message."""
        exc = RateLimitExceeded("Rate limit hit", provider="openai")
        assert str(exc) == "Rate limit hit"
        assert exc.provider == "openai"

    def test_exception_no_provider(self):
        """Test exception without provider."""
        exc = RateLimitExceeded("Rate limit hit")
        assert exc.provider is None
