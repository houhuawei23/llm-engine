"""
Observability middleware for LLM Engine.

Integrates metrics collection and cost tracking with the middleware framework.
"""

from typing import Optional

from llm_engine.middleware.base import Middleware, RequestContext, Response
from llm_engine.observability.cost_tracking import CostTracker
from llm_engine.observability.metrics import (
    InMemoryMetricsCollector,
    MetricsCollector,
    RequestMetrics,
)


class ObservabilityMiddleware(Middleware):
    """Middleware for collecting metrics and costs.

    Tracks request latency, token usage, and costs for all LLM requests.

    Example:
        >>> from llm_engine import LLMEngine
        >>> from llm_engine.observability import ObservabilityMiddleware
        >>>
        >>> mw = ObservabilityMiddleware()
        >>> engine = LLMEngine(config, middleware=[mw])
        >>>
        >>> # Later, get statistics
        >>> stats = mw.get_statistics()
        >>> print(f"Avg latency: {stats['avg_latency_ms']}ms")
        >>> print(f"Total cost: ${stats['total_cost_usd']}")

    """

    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        """Initialize observability middleware.

        Args:
            collector: Metrics collector (defaults to in-memory)
            cost_tracker: Cost tracker (creates default if not provided)
        """
        self._collector = collector or InMemoryMetricsCollector()
        self._cost_tracker = cost_tracker or CostTracker()

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Record request start time."""
        from time import time

        context.metadata["_observability_start_time"] = time() * 1000  # ms
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Record metrics for the request."""
        from time import time

        # Calculate latency
        start_time = context.metadata.get("_observability_start_time")
        latency_ms = time() * 1000 - start_time if start_time else response.latency_ms or 0.0

        # Create metrics
        metrics = RequestMetrics(
            provider=context.provider,
            model=context.model,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cost_usd=0.0,  # Will be calculated below
            cache_hit=response.metadata.get("cache_hit", False),
            error=response.metadata.get("error"),
            metadata={
                "temperature": context.temperature,
                "max_tokens": context.max_tokens,
            },
        )

        # Calculate cost
        cost_entry = self._cost_tracker.record_request(
            provider=context.provider,
            model=context.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        metrics.cost_usd = cost_entry.cost_usd

        # Add cost to response metadata
        response.metadata["cost_usd"] = cost_entry.cost_usd

        # Emit metrics
        self._collector.emit(metrics)

        return response

    def get_statistics(self) -> dict:
        """Get aggregate statistics.

        Returns:
            Dictionary with statistics from metrics and cost tracking
        """
        from llm_engine.observability.metrics import InMemoryMetricsCollector

        stats = {
            "cost_tracking": self._cost_tracker.get_budget_status(),
            "spend_breakdown": self._cost_tracker.get_spend_breakdown("daily"),
        }

        if isinstance(self._collector, InMemoryMetricsCollector):
            stats.update(self._collector.get_statistics())

        return stats

    def get_cost_tracker(self) -> CostTracker:
        """Get the cost tracker instance."""
        return self._cost_tracker

    def close(self) -> None:
        """Close the middleware and release resources."""
        self._collector.close()
        self._cost_tracker.close()
