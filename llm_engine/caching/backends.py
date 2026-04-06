"""
Cache backend implementations for LLM Engine.

Provides pluggable cache backends: memory, disk, and Redis.
"""

import contextlib
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union


@dataclass
class CacheEntry:
    """A cached response entry.

    Attributes:
        content: The cached response content
        metadata: Additional metadata (model, tokens, etc.)
        created_at: When the entry was cached
        ttl: Time-to-live in seconds
    """

    content: str
    metadata: dict
    created_at: datetime
    ttl: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Deserialize from dictionary."""
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            ttl=data.get("ttl"),
        )


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key.

        Args:
            key: Cache key

        Returns:
            Cache entry if found and not expired, None otherwise
        """

    @abstractmethod
    async def set(
        self, key: str, entry: CacheEntry, ttl: Optional[int] = None
    ) -> None:
        """Set a cache entry.

        Args:
            key: Cache key
            entry: Cache entry to store
            ttl: Time-to-live in seconds (overrides entry.ttl)
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""

    @abstractmethod
    async def close(self) -> None:
        """Close the cache backend and release resources."""


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend.

    Simple dictionary-based cache. Data is lost when process exits.

    Example:
        >>> backend = MemoryCacheBackend()
        >>> await backend.set("key", entry, ttl=3600)
        >>> cached = await backend.get("key")

    """

    def __init__(self, max_size: int = 1000):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries (LRU eviction)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._access_order: list[str] = []

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from memory cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if entry.is_expired:
            await self.delete(key)
            return None

        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return entry

    async def set(
        self, key: str, entry: CacheEntry, ttl: Optional[int] = None
    ) -> None:
        """Set entry in memory cache."""
        if ttl is not None:
            entry.ttl = ttl

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = entry
        if key not in self._access_order:
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete entry from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    async def clear(self) -> None:
        """Clear memory cache."""
        self._cache.clear()
        self._access_order.clear()

    async def close(self) -> None:
        """Close memory cache."""
        await self.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend using JSON files.

    Persists cache across process restarts.

    Example:
        >>> backend = DiskCacheBackend(Path("/tmp/llm_cache"))
        >>> await backend.set("key", entry, ttl=3600)

    """

    def __init__(self, cache_dir: Union[str, Path], max_size: int = 10000):
        """Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of entries
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._metadata_file = self._cache_dir / "_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {"entries": {}, "size": 0}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f)
        except OSError:
            pass

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{key_hash}.json"

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk cache."""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            entry = CacheEntry.from_dict(data)

            if entry.is_expired:
                await self.delete(key)
                return None

            # Update access time in metadata
            if key in self._metadata["entries"]:
                self._metadata["entries"][key]["last_accessed"] = datetime.utcnow().isoformat()
                self._save_metadata()

            return entry

        except (OSError, json.JSONDecodeError, KeyError):
            return None

    async def set(
        self, key: str, entry: CacheEntry, ttl: Optional[int] = None
    ) -> None:
        """Set entry in disk cache."""
        if ttl is not None:
            entry.ttl = ttl

        # Check if we need to evict
        current_entries = list(self._metadata["entries"].keys())
        if len(current_entries) >= self._max_size and key not in current_entries:
            # Evict oldest by creation time
            oldest_key = min(
                current_entries,
                key=lambda k: self._metadata["entries"][k].get("created", ""),
            )
            await self.delete(oldest_key)

        cache_file = self._get_cache_file(key)

        try:
            with open(cache_file, "w") as f:
                json.dump(entry.to_dict(), f)

            self._metadata["entries"][key] = {
                "created": entry.created_at.isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
            }
            self._save_metadata()

        except OSError:
            pass

    async def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        cache_file = self._get_cache_file(key)

        try:
            if cache_file.exists():
                cache_file.unlink()

            if key in self._metadata["entries"]:
                del self._metadata["entries"][key]
                self._save_metadata()

            return True
        except OSError:
            return False

    async def clear(self) -> None:
        """Clear disk cache."""
        for cache_file in self._cache_dir.glob("*.json"):
            with contextlib.suppress(OSError):
                cache_file.unlink()

        self._metadata = {"entries": {}, "size": 0}
        self._save_metadata()

    async def close(self) -> None:
        """Close disk cache."""
        self._save_metadata()


class RedisCacheBackend(CacheBackend):
    """Redis-based cache backend.

    Requires redis package: pip install redis

    Example:
        >>> backend = RedisCacheBackend("redis://localhost:6379")
        >>> await backend.set("key", entry, ttl=3600)

    """

    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "llm_engine:"):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._redis = None

    def _get_key(self, key: str) -> str:
        """Get prefixed Redis key."""
        return f"{self._key_prefix}{key}"

    def _get_redis(self):
        """Lazy-load Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self._redis_url)
            except ImportError:
                raise ImportError(
                    "Redis backend requires 'redis' package. "
                    "Install with: pip install redis"
                )
        return self._redis

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis cache."""
        try:
            redis = self._get_redis()
            data = await redis.get(self._get_key(key))

            if data is None:
                return None

            entry_dict = pickle.loads(data)
            entry = CacheEntry.from_dict(entry_dict)

            if entry.is_expired:
                await self.delete(key)
                return None

            return entry

        except Exception:
            return None

    async def set(
        self, key: str, entry: CacheEntry, ttl: Optional[int] = None
    ) -> None:
        """Set entry in Redis cache."""
        try:
            redis = self._get_redis()
            data = pickle.dumps(entry.to_dict())
            effective_ttl = ttl or entry.ttl

            if effective_ttl:
                await redis.setex(self._get_key(key), effective_ttl, data)
            else:
                await redis.set(self._get_key(key), data)

        except Exception:
            pass

    async def delete(self, key: str) -> bool:
        """Delete entry from Redis cache."""
        try:
            redis = self._get_redis()
            result = await redis.delete(self._get_key(key))
            return result > 0
        except Exception:
            return False

    async def clear(self) -> None:
        """Clear all entries with the key prefix."""
        try:
            redis = self._get_redis()
            keys = await redis.keys(f"{self._key_prefix}*")
            if keys:
                await redis.delete(*keys)
        except Exception:
            pass

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
