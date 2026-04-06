"""Tests for caching system."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from llm_engine.caching import (
    CacheConfig,
    CacheEntry,
    DiskCacheBackend,
    LLMCache,
    MemoryCacheBackend,
    SimpleEmbedder,
    cosine_similarity,
)
from llm_engine.caching.semantic import SemanticCache


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(
            content="Hello world",
            metadata={"model": "gpt-4"},
            created_at=datetime.utcnow(),
        )
        assert entry.content == "Hello world"
        assert entry.metadata["model"] == "gpt-4"
        assert entry.ttl is None
        assert not entry.is_expired

    def test_expired_entry(self):
        """Test expired cache entry detection."""
        old_time = datetime.utcnow() - timedelta(seconds=100)
        entry = CacheEntry(
            content="Hello",
            metadata={},
            created_at=old_time,
            ttl=50,  # 50 second TTL
        )
        assert entry.is_expired

    def test_not_expired(self):
        """Test non-expired entry."""
        entry = CacheEntry(
            content="Hello",
            metadata={},
            created_at=datetime.utcnow(),
            ttl=3600,  # 1 hour TTL
        )
        assert not entry.is_expired

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        original = CacheEntry(
            content="Test content",
            metadata={"key": "value"},
            created_at=datetime.utcnow(),
            ttl=3600,
        )

        data = original.to_dict()
        restored = CacheEntry.from_dict(data)

        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.ttl == original.ttl


class TestMemoryCacheBackend:
    """Test MemoryCacheBackend."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        backend = MemoryCacheBackend()

        entry = CacheEntry(
            content="Hello",
            metadata={},
            created_at=datetime.utcnow(),
        )

        await backend.set("key1", entry)
        retrieved = await backend.get("key1")

        assert retrieved is not None
        assert retrieved.content == "Hello"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        backend = MemoryCacheBackend()
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_entry_removed(self):
        """Test that expired entries are removed on get."""
        backend = MemoryCacheBackend()

        old_time = datetime.utcnow() - timedelta(seconds=100)
        entry = CacheEntry(
            content="Hello",
            metadata={},
            created_at=old_time,
            ttl=50,
        )

        await backend.set("key1", entry)
        result = await backend.get("key1")

        assert result is None
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        backend = MemoryCacheBackend()

        entry = CacheEntry(content="Hello", metadata={}, created_at=datetime.utcnow())
        await backend.set("key1", entry)

        result = await backend.delete("key1")
        assert result is True
        assert len(backend) == 0

        # Delete non-existent
        result = await backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear operation."""
        backend = MemoryCacheBackend()

        entry = CacheEntry(content="Hello", metadata={}, created_at=datetime.utcnow())
        await backend.set("key1", entry)
        await backend.set("key2", entry)

        await backend.clear()

        assert len(backend) == 0
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        backend = MemoryCacheBackend(max_size=2)

        entry = CacheEntry(content="Hello", metadata={}, created_at=datetime.utcnow())

        await backend.set("key1", entry)
        await backend.set("key2", entry)

        # Access key1 to make it more recent
        await backend.get("key1")

        # Add third entry - should evict key2
        await backend.set("key3", entry)

        assert await backend.get("key1") is not None
        assert await backend.get("key2") is None
        assert await backend.get("key3") is not None

    @pytest.mark.asyncio
    async def test_ttl_override(self):
        """Test that TTL parameter overrides entry TTL."""
        backend = MemoryCacheBackend()

        entry = CacheEntry(
            content="Hello",
            metadata={},
            created_at=datetime.utcnow(),
            ttl=3600,  # 1 hour
        )

        await backend.set("key1", entry, ttl=1)  # Override to 1 second
        await asyncio.sleep(1.1)

        result = await backend.get("key1")
        assert result is None


class TestDiskCacheBackend:
    """Test DiskCacheBackend."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, tmp_path):
        """Test basic disk cache operations."""
        cache_dir = tmp_path / "cache"
        backend = DiskCacheBackend(cache_dir)

        entry = CacheEntry(
            content="Hello from disk",
            metadata={"model": "gpt-4"},
            created_at=datetime.utcnow(),
        )

        await backend.set("key1", entry)
        retrieved = await backend.get("key1")

        assert retrieved is not None
        assert retrieved.content == "Hello from disk"

        await backend.close()

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        cache_dir = tmp_path / "cache"

        # First instance
        backend1 = DiskCacheBackend(cache_dir)
        entry = CacheEntry(content="Persistent", metadata={}, created_at=datetime.utcnow())
        await backend1.set("key1", entry)
        await backend1.close()

        # Second instance - should see the same data
        backend2 = DiskCacheBackend(cache_dir)
        retrieved = await backend2.get("key1")

        assert retrieved is not None
        assert retrieved.content == "Persistent"
        await backend2.close()

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path):
        """Test clear removes all entries."""
        cache_dir = tmp_path / "cache"
        backend = DiskCacheBackend(cache_dir)

        entry = CacheEntry(content="Hello", metadata={}, created_at=datetime.utcnow())
        await backend.set("key1", entry)
        await backend.set("key2", entry)

        await backend.clear()

        assert await backend.get("key1") is None
        assert await backend.get("key2") is None
        await backend.close()


class TestLLMCache:
    """Test LLMCache two-tier caching."""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match caching."""
        config = CacheConfig()
        cache = LLMCache(config)

        messages = [{"role": "user", "content": "Hello"}]

        # Store
        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="Hi there!",
            usage={"prompt_tokens": 1, "completion_tokens": 2},
        )

        # Retrieve
        entry = await cache.get(
            provider="openai",
            model="gpt-4",
            messages=messages,
        )

        assert entry is not None
        assert entry.content == "Hi there!"
        assert entry.metadata["usage"]["completion_tokens"] == 2

    @pytest.mark.asyncio
    async def test_different_params_different_keys(self):
        """Test that different parameters create different cache entries."""
        config = CacheConfig()
        cache = LLMCache(config)

        messages = [{"role": "user", "content": "Hello"}]

        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="Response at temp 0.7",
            temperature=0.7,
        )

        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="Response at temp 0.9",
            temperature=0.9,
        )

        # Get with temp 0.7
        entry1 = await cache.get(
            provider="openai",
            model="gpt-4",
            messages=messages,
            temperature=0.7,
        )
        assert entry1.content == "Response at temp 0.7"

        # Get with temp 0.9
        entry2 = await cache.get(
            provider="openai",
            model="gpt-4",
            messages=messages,
            temperature=0.9,
        )
        assert entry2.content == "Response at temp 0.9"

    @pytest.mark.asyncio
    async def test_excluded_providers(self):
        """Test that excluded providers are not cached."""
        config = CacheConfig(exclude_providers=["openai"])
        cache = LLMCache(config)

        messages = [{"role": "user", "content": "Hello"}]

        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="Should not cache",
        )

        entry = await cache.get(
            provider="openai",
            model="gpt-4",
            messages=messages,
        )

        assert entry is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig()
        cache = LLMCache(config)

        stats = cache.get_stats()

        assert stats["backend_type"] == "MemoryCacheBackend"
        assert stats["semantic_enabled"] is False
        assert stats["memory_entries"] == 0

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test cache deletion."""
        config = CacheConfig()
        cache = LLMCache(config)

        messages = [{"role": "user", "content": "Hello"}]

        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="To be deleted",
        )

        result = await cache.delete(
            provider="openai",
            model="gpt-4",
            messages=messages,
        )

        assert result is True
        assert await cache.get("openai", "gpt-4", messages) is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test cache clear."""
        config = CacheConfig()
        cache = LLMCache(config)

        messages = [{"role": "user", "content": "Hello"}]

        await cache.set(
            provider="openai",
            model="gpt-4",
            messages=messages,
            content="To be cleared",
        )

        await cache.clear()

        assert await cache.get("openai", "gpt-4", messages) is None


class TestSimpleEmbedder:
    """Test SimpleEmbedder."""

    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test that embeddings are generated."""
        embedder = SimpleEmbedder(dimensions=64)

        embedding = await embedder("Hello world")

        assert len(embedding) == 64
        # Should be normalized (magnitude ~1)
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert 0.99 < magnitude < 1.01 or magnitude == 0

    @pytest.mark.asyncio
    async def test_consistency(self):
        """Test that same text produces same embedding."""
        embedder = SimpleEmbedder(dimensions=64)

        embedding1 = await embedder("Hello world")
        embedding2 = await embedder("Hello world")

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_different_texts(self):
        """Test that different texts produce different embeddings."""
        embedder = SimpleEmbedder(dimensions=64)

        embedding1 = await embedder("Hello world")
        embedding2 = await embedder("Goodbye world")

        assert embedding1 != embedding2


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1."""
        vec = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-10)

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    def test_different_dimensions_raises(self):
        """Test that different dimensions raise error."""
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# Need to import asyncio for sleep in test
import asyncio