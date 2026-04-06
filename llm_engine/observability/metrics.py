"""
Metrics collection for LLM Engine.

Tracks request latency, token usage, costs, and cache performance.
Integrates with Prometheus and other monitoring systems.
"""

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class RequestMetrics:
    """Metrics for a single LLM request.

    Attributes:
        provider: LLM provider name
        model: Model name used
        latency_ms: Request latency in milliseconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total token count (auto-calculated)
        cost_usd: Estimated cost in USD
        cache_hit: Whether this was a cache hit
        error: Error message if request failed
        timestamp: When the request was made
        metadata: Additional request metadata
    """
    provider: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MetricsCollector(Protocol):
    """Protocol for metrics collectors."""

    def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to the collector.

        Args:
            metrics: Request metrics to emit
        """
        ...

    def close(self) -> None:
        """Close the collector and release resources."""
        ...


class InMemoryMetricsCollector:
    """In-memory metrics collector for testing and simple use cases.

    Stores metrics in memory with optional retention limit.

    Example:
        >>> collector = InMemoryMetricsCollector(max_entries=1000)
        >>> collector.emit(RequestMetrics(...))
        >>> stats = collector.get_statistics()
    """

    def __init__(self, max_entries: int = 10000):
        """Initialize in-memory collector.

        Args:
            max_entries: Maximum number of entries to retain
        """
        self._metrics: List[RequestMetrics] = []
        self._max_entries = max_entries

    def emit(self, metrics: RequestMetrics) -> None:
        """Store metrics in memory."""
        self._metrics.append(metrics)

        # Enforce retention limit
        if len(self._metrics) > self._max_entries:
            self._metrics = self._metrics[-self._max_entries:]

    def get_metrics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[RequestMetrics]:
        """Get filtered metrics.

        Args:
            provider: Filter by provider
            model: Filter by model
            since: Filter by timestamp

        Returns:
            List of matching metrics
        """
        result = self._metrics

        if provider:
            result = [m for m in result if m.provider == provider]
        if model:
            result = [m for m in result if m.model == model]
        if since:
            result = [m for m in result if m.timestamp >= since]

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate aggregate statistics.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self._metrics:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }

        latencies = [m.latency_ms for m in self._metrics]
        tokens = [m.input_tokens + m.output_tokens for m in self._metrics]
        costs = [m.cost_usd for m in self._metrics]
        cache_hits = sum(1 for m in self._metrics if m.cache_hit)
        errors = sum(1 for m in self._metrics if m.error)

        return {
            "total_requests": len(self._metrics),
            "total_tokens": sum(tokens),
            "total_cost_usd": sum(costs),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": self._percentile(latencies, 0.5),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "avg_tokens_per_request": sum(tokens) / len(tokens),
            "cache_hit_rate": cache_hits / len(self._metrics) * 100,
            "error_rate": errors / len(self._metrics) * 100,
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._metrics.clear()

    def close(self) -> None:
        """Close the collector."""


class PrometheusMetricsCollector:
    """Prometheus metrics collector.

    Requires prometheus-client package.

    Example:
        >>> collector = PrometheusMetricsCollector(
        ...     port=8000,
        ...     prefix="llm_engine"
        ... )
        >>> collector.emit(RequestMetrics(...))
    """

    def __init__(
        self,
        port: int = 8000,
        prefix: str = "llm_engine",
        start_server: bool = True,
    ):
        """Initialize Prometheus collector.

        Args:
            port: Port for Prometheus metrics endpoint
            prefix: Prefix for metric names
            start_server: Whether to start HTTP server
        """
        try:
            from prometheus_client import Counter, Histogram, start_http_server
        except ImportError:
            raise ImportError(
                "Prometheus collector requires 'prometheus-client' package. "
                "Install with: pip install prometheus-client"
            )

        self._prefix = prefix
        self._port = port

        # Define metrics
        self._request_count = Counter(
            f"{prefix}_requests_total",
            "Total LLM requests",
            ["provider", "model", "cache_hit", "error"]
        )

        self._latency = Histogram(
            f"{prefix}_request_latency_seconds",
            "Request latency in seconds",
            ["provider", "model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        self._tokens = Counter(
            f"{prefix}_tokens_total",
            "Total tokens processed",
            ["provider", "model", "token_type"]
        )

        self._cost = Counter(
            f"{prefix}_cost_usd_total",
            "Total cost in USD",
            ["provider", "model"]
        )

        if start_server:
            start_http_server(port)

    def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to Prometheus."""
        provider = metrics.provider
        model = metrics.model
        cache_hit = "true" if metrics.cache_hit else "false"
        error = "true" if metrics.error else "false"

        # Request count
        self._request_count.labels(
            provider=provider,
            model=model,
            cache_hit=cache_hit,
            error=error
        ).inc()

        # Latency (convert ms to seconds)
        self._latency.labels(
            provider=provider,
            model=model
        ).observe(metrics.latency_ms / 1000.0)

        # Tokens
        if metrics.input_tokens > 0:
            self._tokens.labels(
                provider=provider,
                model=model,
                token_type="input"
            ).inc(metrics.input_tokens)

        if metrics.output_tokens > 0:
            self._tokens.labels(
                provider=provider,
                model=model,
                token_type="output"
            ).inc(metrics.output_tokens)

        # Cost
        if metrics.cost_usd > 0:
            self._cost.labels(
                provider=provider,
                model=model
            ).inc(metrics.cost_usd)

    def close(self) -> None:
        """Close the collector."""


class MultiMetricsCollector:
    """Collector that sends metrics to multiple backends.

    Example:
        >>> collector = MultiMetricsCollector([
        ...     InMemoryMetricsCollector(),
        ...     PrometheusMetricsCollector(),
        ... ])
    """

    def __init__(self, collectors: List[MetricsCollector]):
        """Initialize multi-collector.

        Args:
            collectors: List of collectors to emit to
        """
        self._collectors = collectors

    def emit(self, metrics: RequestMetrics) -> None:
        """Emit metrics to all collectors."""
        for collector in self._collectors:
            try:
                collector.emit(metrics)
            except Exception:
                # Don't let one collector failure affect others
                pass

    def close(self) -> None:
        """Close all collectors."""
        for collector in self._collectors:
            with contextlib.suppress(Exception):
                collector.close()
