"""
Semantic similarity caching for LLM Engine.

Enables caching based on embedding similarity rather than exact key matching.
Uses vector similarity to find semantically equivalent prompts.
"""

import hashlib
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from llm_engine.caching.backends import CacheBackend, CacheEntry


class Embedder(Protocol):
    """Protocol for text embedding functions."""

    async def __call__(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same dimension: {len(a)} vs {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


@dataclass
class SemanticCacheEntry:
    """A semantic cache entry with embedding.

    Attributes:
        prompt_hash: Hash of the original prompt
        embedding: Vector embedding of the prompt
        response: The cached response entry
    """

    prompt_hash: str
    embedding: List[float]
    response: CacheEntry

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "prompt_hash": self.prompt_hash,
            "embedding": self.embedding,
            "response": self.response.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticCacheEntry":
        """Deserialize from dictionary."""
        return cls(
            prompt_hash=data["prompt_hash"],
            embedding=data["embedding"],
            response=CacheEntry.from_dict(data["response"]),
        )


class SimpleEmbedder:
    """Simple embedder using character n-grams.

    This is a lightweight fallback that doesn't require external APIs.
    Good for basic semantic similarity, but not as accurate as model-based embeddings.

    Example:
        >>> embedder = SimpleEmbedder(dimensions=128)
        >>> embedding = await embedder("Hello world")

    """

    def __init__(self, dimensions: int = 128, n_gram_size: int = 3):
        """Initialize simple embedder.

        Args:
            dimensions: Embedding dimensionality
            n_gram_size: Character n-gram size
        """
        self.dimensions = dimensions
        self.n_gram_size = n_gram_size

    async def __call__(self, text: str) -> List[float]:
        """Generate simple n-gram based embedding."""
        text = text.lower().strip()

        # Initialize vector
        vector = [0.0] * self.dimensions

        # Generate character n-grams
        for i in range(len(text) - self.n_gram_size + 1):
            ngram = text[i : i + self.n_gram_size]
            # Hash n-gram to dimension index
            idx = hash(ngram) % self.dimensions
            # Simple weight based on position
            weight = 1.0 / (i + 1)
            vector[idx] += weight

        # Normalize
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector


class SemanticCache:
    """Semantic cache using embedding similarity.

    Finds cache hits based on semantic similarity rather than exact matching.

    Example:
        >>> from llm_engine.caching.backends import MemoryCacheBackend
        >>> backend = MemoryCacheBackend()
        >>> embedder = SimpleEmbedder()
        >>>
        >>> cache = SemanticCache(backend, embedder, similarity_threshold=0.9)
        >>>
        >>> # Store a response
        >>> await cache.set("What is Python?", "Python is a programming language...")
        >>>
        >>> # Retrieve with semantically similar query
        >>> result = await cache.get("Tell me about Python programming")
        >>> # May match if similarity > threshold

    """

    def __init__(
        self,
        backend: CacheBackend,
        embedder: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.95,
        embedding_key_prefix: str = "semantic:",
    ):
        """Initialize semantic cache.

        Args:
            backend: Cache backend for storage
            embedder: Embedding function (defaults to SimpleEmbedder)
            similarity_threshold: Minimum similarity for cache hit (0-1)
            embedding_key_prefix: Prefix for cache keys
        """
        self._backend = backend
        self._embedder = embedder or SimpleEmbedder()
        self._similarity_threshold = similarity_threshold
        self._embedding_key_prefix = embedding_key_prefix

    def _get_key(self, prompt_hash: str) -> str:
        """Get prefixed cache key."""
        return f"{self._embedding_key_prefix}{prompt_hash}"

    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    async def get(self, prompt: str) -> Optional[CacheEntry]:
        """Get cached response for semantically similar prompt.

        Args:
            prompt: The prompt to look up

        Returns:
            Cached entry if similar prompt found, None otherwise
        """
        # First try exact match
        prompt_hash = self._hash_prompt(prompt)
        exact_key = self._get_key(prompt_hash)

        exact_entry = await self._backend.get(exact_key)
        if exact_entry:
            return exact_entry

        # Try semantic match
        query_embedding = await self._embedder(prompt)

        # Get all semantic cache entries
        # Note: This is inefficient for large caches; production use should use
        # vector databases like Chroma, Pinecone, or FAISS
        all_keys = await self._get_all_semantic_keys()

        best_match: Optional[SemanticCacheEntry] = None
        best_similarity = 0.0

        for key in all_keys:
            entry_data = await self._backend.get(key)
            if not entry_data:
                continue

            try:
                semantic_entry = SemanticCacheEntry.from_dict(entry_data.metadata)
            except (KeyError, TypeError):
                continue

            if semantic_entry.response.is_expired:
                await self._backend.delete(key)
                continue

            similarity = cosine_similarity(query_embedding, semantic_entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = semantic_entry

        if best_match and best_similarity >= self._similarity_threshold:
            return best_match.response

        return None

    async def set(
        self, prompt: str, response: CacheEntry, ttl: Optional[int] = None
    ) -> None:
        """Cache a response with its embedding.

        Args:
            prompt: The prompt text
            response: The response to cache
            ttl: Time-to-live in seconds
        """
        prompt_hash = self._hash_prompt(prompt)
        embedding = await self._embedder(prompt)

        semantic_entry = SemanticCacheEntry(
            prompt_hash=prompt_hash,
            embedding=embedding,
            response=response,
        )

        # Store as regular cache entry with semantic metadata
        cache_entry = CacheEntry(
            content="",  # Content stored in semantic entry
            metadata=semantic_entry.to_dict(),
            created_at=response.created_at,
            ttl=ttl or response.ttl,
        )

        key = self._get_key(prompt_hash)
        await self._backend.set(key, cache_entry, ttl)

    async def delete(self, prompt: str) -> bool:
        """Delete cached entry for prompt.

        Args:
            prompt: The prompt to delete

        Returns:
            True if deleted, False if not found
        """
        prompt_hash = self._hash_prompt(prompt)
        key = self._get_key(prompt_hash)
        return await self._backend.delete(key)

    async def clear(self) -> None:
        """Clear all semantic cache entries."""
        keys = await self._get_all_semantic_keys()
        for key in keys:
            await self._backend.delete(key)

    async def _get_all_semantic_keys(self) -> List[str]:
        """Get all semantic cache keys.

        Note: This is a naive implementation. Production use should use
        proper indexing or a vector database.
        """
        # This is backend-specific; for now, we'll assume we can't enumerate
        # Return empty list to fall back to exact matching only
        # Subclasses can override this for specific backends
        return []

    async def close(self) -> None:
        """Close the semantic cache."""
        await self._backend.close()


class OpenAIEmbedder:
    """Embedder using OpenAI's embedding API.

    Requires openai package.

    Example:
        >>> embedder = OpenAIEmbedder(api_key="sk-...")
        >>> embedding = await embedder("Hello world")

    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            base_url: Optional custom base URL
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                kwargs = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "OpenAIEmbedder requires 'openai' package. "
                    "Install with: pip install openai"
                )
        return self._client

    async def __call__(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        client = self._get_client()
        response = await client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding
