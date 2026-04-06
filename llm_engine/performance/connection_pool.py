"""
Connection pooling for LLM Engine.

Provides HTTP connection pooling for improved performance.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx


class ConnectionPool:
    """HTTP connection pool for LLM API requests.

    Manages a shared httpx.AsyncClient with proper connection limits
    and keep-alive settings.

    Example:
        >>> pool = ConnectionPool(
        ...     max_connections=100,
        ...     max_keepalive=20,
        ... )
        >>>
        >>> async with pool.get_client() as client:
        ...     response = await client.post(...)

    """

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive: int = 20,
        timeout: float = 60.0,
        http2: bool = False,
    ):
        """Initialize connection pool.

        Args:
            max_connections: Maximum concurrent connections
            max_keepalive: Maximum keepalive connections
            timeout: Default timeout in seconds
            http2: Enable HTTP/2 support
        """
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        )
        self._timeout = httpx.Timeout(timeout)
        self._http2 = http2
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_or_create_client(self) -> httpx.AsyncClient:
        """Get or create the shared client."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        limits=self._limits,
                        timeout=self._timeout,
                        http2=self._http2,
                    )
        return self._client

    @asynccontextmanager
    async def get_client(self):
        """Get a client from the pool.

        Yields:
            httpx.AsyncClient instance
        """
        client = await self._get_or_create_client()
        try:
            yield client
        finally:
            pass  # Client is reused, don't close

    async def close(self) -> None:
        """Close the connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_limits(self) -> Dict[str, Any]:
        """Get current connection limits.

        Returns:
            Dictionary with connection limits
        """
        return {
            "max_connections": self._limits.max_connections,
            "max_keepalive": self._limits.max_keepalive_connections,
            "timeout": str(self._timeout),
            "http2": self._http2,
        }


class PooledClient:
    """Wrapper for using connection pool with existing code.

    Provides a drop-in replacement for creating new httpx clients.

    Example:
        >>> pooled = PooledClient()
        >>>
        >>> # Instead of: async with httpx.AsyncClient() as client:
        >>> async with pooled as client:
        ...     response = await client.post(...)

    """

    _instance: Optional["PooledClient"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for shared pool."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive: int = 20,
        timeout: float = 60.0,
    ):
        """Initialize pooled client (only first call has effect)."""
        if not hasattr(self, "_initialized"):
            self._pool = ConnectionPool(
                max_connections=max_connections,
                max_keepalive=max_keepalive,
                timeout=timeout,
            )
            self._initialized = True

    async def __aenter__(self):
        """Enter async context."""
        self._client_ctx = self._pool.get_client()
        return await self._client_ctx.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        return await self._client_ctx.__aexit__(exc_type, exc_val, exc_tb)

    @classmethod
    async def close_global(cls) -> None:
        """Close the global pooled client."""
        if cls._instance:
            await cls._instance._pool.close()
            cls._instance = None
