"""
Observability system for LLM Engine.

Provides metrics collection, cost tracking, and monitoring integrations.

Example:
    >>> from llm_engine import LLMEngine
    >>> from llm_engine.observability import ObservabilityMiddleware
    >>>
    >>> mw = ObservabilityMiddleware()
    >>> engine = LLMEngine(config, middleware=[mw])
    >>>
    >>> # Get statistics later
    >>> stats = mw.get_statistics()
    >>> print(f"Total cost: ${stats['total_cost_usd']:.4f}")

"""

from llm_engine.observability.cost_tracking import (
    CostEntry,
    CostStorage,
    CostTracker,
    FileCostStorage,
    InMemoryCostStorage,
    PricingProvider,
)
from llm_engine.observability.metrics import (
    InMemoryMetricsCollector,
    MetricsCollector,
    MetricType,
    MultiMetricsCollector,
    PrometheusMetricsCollector,
    RequestMetrics,
)
from llm_engine.observability.middleware import ObservabilityMiddleware

__all__ = [
    # Cost Tracking
    "CostEntry",
    "CostStorage",
    "CostTracker",
    "FileCostStorage",
    "InMemoryCostStorage",
    "InMemoryMetricsCollector",
    "MetricType",
    "MetricsCollector",
    "MultiMetricsCollector",
    # Middleware
    "ObservabilityMiddleware",
    "PricingProvider",
    "PrometheusMetricsCollector",
    # Metrics
    "RequestMetrics",
]
