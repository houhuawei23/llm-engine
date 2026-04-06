"""
Cost tracking and budget management for LLM Engine.

Tracks API costs across providers and models with budget alerts.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class CostEntry:
    """A single cost entry.

    Attributes:
        provider: LLM provider name
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count
        cost_usd: Cost in USD
        timestamp: When the request was made
    """
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEntry":
        """Deserialize from dictionary."""
        return cls(
            provider=data["provider"],
            model=data["model"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost_usd=data["cost_usd"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class CostStorage(Protocol):
    """Protocol for cost storage backends."""

    def store(self, entry: CostEntry) -> None:
        """Store a cost entry."""
        ...

    def get_entries(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CostEntry]:
        """Get filtered cost entries."""
        ...

    def close(self) -> None:
        """Close the storage."""
        ...


class InMemoryCostStorage:
    """In-memory cost storage."""

    def __init__(self, max_entries: int = 10000):
        """Initialize storage.

        Args:
            max_entries: Maximum entries to retain
        """
        self._entries: List[CostEntry] = []
        self._max_entries = max_entries

    def store(self, entry: CostEntry) -> None:
        """Store a cost entry."""
        self._entries.append(entry)

        # Enforce retention
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def get_entries(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CostEntry]:
        """Get filtered entries."""
        result = self._entries

        if since:
            result = [e for e in result if e.timestamp >= since]
        if until:
            result = [e for e in result if e.timestamp <= until]
        if provider:
            result = [e for e in result if e.provider == provider]
        if model:
            result = [e for e in result if e.model == model]

        return result

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def close(self) -> None:
        """Close storage."""


class FileCostStorage:
    """File-based cost storage with JSON lines format."""

    def __init__(self, file_path: Path):
        """Initialize file storage.

        Args:
            file_path: Path to storage file
        """
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def store(self, entry: CostEntry) -> None:
        """Append entry to file."""
        with open(self._file_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_entries(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CostEntry]:
        """Read and filter entries from file."""
        entries = []

        if not self._file_path.exists():
            return entries

        with open(self._file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = CostEntry.from_dict(data)

                    # Apply filters
                    if since and entry.timestamp < since:
                        continue
                    if until and entry.timestamp > until:
                        continue
                    if provider and entry.provider != provider:
                        continue
                    if model and entry.model != model:
                        continue

                    entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue

        return entries

    def close(self) -> None:
        """Close storage."""


class PricingProvider:
    """Provider for model pricing information.

    Contains pricing data for various LLM providers.
    Prices are in USD per 1M tokens.
    """

    # Pricing data: (provider, model_pattern) -> (input_price, output_price)
    # Prices in USD per 1M tokens
    DEFAULT_PRICING: Dict[tuple, tuple] = {
        ("openai", "gpt-4"): (30.0, 60.0),
        ("openai", "gpt-4-turbo"): (10.0, 30.0),
        ("openai", "gpt-3.5-turbo"): (0.5, 1.5),
        ("deepseek", "deepseek-chat"): (0.14, 0.28),  # ~$2/1M CNY
        ("deepseek", "deepseek-reasoner"): (0.14, 0.28),
        ("anthropic", "claude-3-opus"): (15.0, 75.0),
        ("anthropic", "claude-3-sonnet"): (3.0, 15.0),
        ("anthropic", "claude-3-haiku"): (0.25, 1.25),
        ("ollama", "*"): (0.0, 0.0),  # Local model, no cost
    }

    def __init__(self, custom_pricing: Optional[Dict[tuple, tuple]] = None):
        """Initialize pricing provider.

        Args:
            custom_pricing: Override or add pricing data
        """
        self._pricing = dict(self.DEFAULT_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def get_price(self, provider: str, model: str) -> tuple[float, float]:
        """Get pricing for a model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Tuple of (input_price, output_price) per 1M tokens
        """
        # Try exact match first
        key = (provider.lower(), model.lower())
        if key in self._pricing:
            return self._pricing[key]

        # Try provider wildcard
        key = (provider.lower(), "*")
        if key in self._pricing:
            return self._pricing[key]

        # Try model prefix match
        for (p, m), prices in self._pricing.items():
            if p == provider.lower() and model.lower().startswith(m.rstrip("-")):
                return prices

        # Default fallback
        return (0.0, 0.0)

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a request.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        input_price, output_price = self.get_price(provider, model)

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost


class CostTracker:
    """Tracks LLM API costs with budget management.

    Example:
        >>> tracker = CostTracker(
        ...     budget_usd=100.0,
        ...     alert_threshold=0.8,
        ... )
        >>>
        >>> tracker.record_request(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ... )
        >>>
        >>> spend = tracker.get_current_spend()  # Returns total USD spent
        >>> if spend > tracker.budget_usd * tracker.alert_threshold:
        ...     print("Budget alert!")
    """

    def __init__(
        self,
        storage: Optional[CostStorage] = None,
        pricing: Optional[PricingProvider] = None,
        budget_usd: Optional[float] = None,
        alert_threshold: float = 0.8,
    ):
        """Initialize cost tracker.

        Args:
            storage: Storage backend for cost entries
            pricing: Pricing provider
            budget_usd: Budget limit in USD
            alert_threshold: Alert threshold (0-1)
        """
        self._storage = storage or InMemoryCostStorage()
        self._pricing = pricing or PricingProvider()
        self.budget_usd = budget_usd
        self.alert_threshold = alert_threshold
        self._alerts_triggered: List[str] = []

    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Optional[float] = None,
    ) -> CostEntry:
        """Record a request's cost.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Optional pre-calculated cost

        Returns:
            Cost entry that was recorded
        """
        if cost_usd is None:
            cost_usd = self._pricing.calculate_cost(
                provider, model, input_tokens, output_tokens
            )

        entry = CostEntry(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

        self._storage.store(entry)

        # Check budget alert
        if self.budget_usd and self.alert_threshold:
            current_spend = self.get_current_spend()
            threshold_amount = self.budget_usd * self.alert_threshold

            if current_spend >= threshold_amount:
                alert_key = f"{self.alert_threshold}:{datetime.utcnow().strftime('%Y-%m-%d')}"
                if alert_key not in self._alerts_triggered:
                    self._alerts_triggered.append(alert_key)
                    # Could emit alert here

        return entry

    def get_current_spend(
        self,
        period: str = "daily",
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> float:
        """Get current spend for a period.

        Args:
            period: "daily", "weekly", "monthly", or "all"
            provider: Filter by provider
            model: Filter by model

        Returns:
            Total cost in USD
        """
        now = datetime.utcnow()

        if period == "daily":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            since = now - timedelta(days=7)
        elif period == "monthly":
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            since = None

        entries = self._storage.get_entries(
            since=since,
            provider=provider,
            model=model,
        )

        return sum(e.cost_usd for e in entries)

    def get_spend_breakdown(
        self,
        period: str = "daily",
    ) -> Dict[str, Any]:
        """Get detailed spend breakdown.

        Args:
            period: Time period for breakdown

        Returns:
            Dictionary with spend breakdown by provider and model
        """
        now = datetime.utcnow()

        if period == "daily":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            since = now - timedelta(days=7)
        elif period == "monthly":
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            since = None

        entries = self._storage.get_entries(since=since)

        # Aggregate by provider
        by_provider: Dict[str, Dict[str, Any]] = {}

        for entry in entries:
            if entry.provider not in by_provider:
                by_provider[entry.provider] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "request_count": 0,
                    "models": {},
                }

            p = by_provider[entry.provider]
            p["total_cost"] += entry.cost_usd
            p["total_tokens"] += entry.input_tokens + entry.output_tokens
            p["request_count"] += 1

            if entry.model not in p["models"]:
                p["models"][entry.model] = {
                    "cost": 0.0,
                    "tokens": 0,
                    "requests": 0,
                }

            m = p["models"][entry.model]
            m["cost"] += entry.cost_usd
            m["tokens"] += entry.input_tokens + entry.output_tokens
            m["requests"] += 1

        return {
            "period": period,
            "total_cost": sum(p["total_cost"] for p in by_provider.values()),
            "total_requests": sum(p["request_count"] for p in by_provider.values()),
            "by_provider": by_provider,
        }

    def estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        expected_output_tokens: int = 500,
    ) -> float:
        """Estimate cost for a request.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Expected input tokens
            expected_output_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        return self._pricing.calculate_cost(
            provider, model, input_tokens, expected_output_tokens
        )

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status.

        Returns:
            Budget status information
        """
        if not self.budget_usd:
            return {"has_budget": False}

        current_spend = self.get_current_spend(period="daily")
        remaining = self.budget_usd - current_spend
        percent_used = (current_spend / self.budget_usd) * 100

        return {
            "has_budget": True,
            "budget_usd": self.budget_usd,
            "spent_usd": current_spend,
            "remaining_usd": remaining,
            "percent_used": percent_used,
            "alert_threshold": self.alert_threshold,
            "alert_triggered": percent_used >= self.alert_threshold * 100,
        }

    def close(self) -> None:
        """Close the tracker and release resources."""
        self._storage.close()
