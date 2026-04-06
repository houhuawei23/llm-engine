"""Tests for observability system."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from llm_engine.observability import (
    CostEntry,
    CostTracker,
    FileCostStorage,
    InMemoryCostStorage,
    InMemoryMetricsCollector,
    PricingProvider,
    RequestMetrics,
)


class TestRequestMetrics:
    """Test RequestMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metrics = RequestMetrics(
            provider="openai",
            model="gpt-4",
            latency_ms=100.5,
            input_tokens=100,
            output_tokens=50,
        )
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.latency_ms == 100.5
        assert metrics.total_tokens == 150

    def test_to_dict(self):
        """Test serialization."""
        metrics = RequestMetrics(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.05,
        )

        data = metrics.to_dict()
        assert data["provider"] == "openai"
        assert data["cost_usd"] == 0.05
        assert "timestamp" in data


class TestInMemoryMetricsCollector:
    """Test InMemoryMetricsCollector."""

    @pytest.fixture
    def collector(self):
        return InMemoryMetricsCollector()

    def test_emit_and_retrieve(self, collector):
        """Test emitting and retrieving metrics."""
        metrics = RequestMetrics(
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            input_tokens=100,
            output_tokens=50,
        )

        collector.emit(metrics)

        retrieved = collector.get_metrics()
        assert len(retrieved) == 1
        assert retrieved[0].provider == "openai"

    def test_filter_by_provider(self, collector):
        """Test filtering by provider."""
        collector.emit(RequestMetrics("openai", "gpt-4", 100.0))
        collector.emit(RequestMetrics("deepseek", "deepseek-chat", 200.0))

        openai_metrics = collector.get_metrics(provider="openai")
        assert len(openai_metrics) == 1
        assert openai_metrics[0].provider == "openai"

    def test_filter_by_model(self, collector):
        """Test filtering by model."""
        collector.emit(RequestMetrics("openai", "gpt-4", 100.0))
        collector.emit(RequestMetrics("openai", "gpt-3.5", 50.0))

        gpt4_metrics = collector.get_metrics(model="gpt-4")
        assert len(gpt4_metrics) == 1
        assert gpt4_metrics[0].model == "gpt-4"

    def test_filter_by_time(self, collector):
        """Test filtering by timestamp."""
        old_time = datetime.utcnow() - timedelta(hours=2)
        new_time = datetime.utcnow()

        old_metrics = RequestMetrics("openai", "gpt-4", 100.0)
        old_metrics.timestamp = old_time

        new_metrics = RequestMetrics("openai", "gpt-4", 100.0)
        new_metrics.timestamp = new_time

        collector.emit(old_metrics)
        collector.emit(new_metrics)

        recent = collector.get_metrics(since=datetime.utcnow() - timedelta(minutes=30))
        assert len(recent) == 1

    def test_statistics(self, collector):
        """Test statistics calculation."""
        # Add metrics with varying latencies
        for i in range(10):
            collector.emit(RequestMetrics(
                provider="openai",
                model="gpt-4",
                latency_ms=float(i * 10),  # 0, 10, 20, ..., 90
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01 * i,
            ))

        stats = collector.get_statistics()

        assert stats["total_requests"] == 10
        assert stats["total_tokens"] == 1500
        assert stats["avg_latency_ms"] == 45.0
        assert stats["p50_latency_ms"] == 50.0

    def test_empty_statistics(self, collector):
        """Test statistics with no data."""
        stats = collector.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_retention_limit(self):
        """Test that old entries are dropped."""
        collector = InMemoryMetricsCollector(max_entries=5)

        for i in range(10):
            collector.emit(RequestMetrics("openai", "gpt-4", float(i)))

        assert len(collector.get_metrics()) == 5


class TestPricingProvider:
    """Test PricingProvider."""

    @pytest.fixture
    def pricing(self):
        return PricingProvider()

    def test_get_price_exact_match(self, pricing):
        """Test exact price match."""
        input_price, output_price = pricing.get_price("openai", "gpt-4")
        assert input_price == 30.0
        assert output_price == 60.0

    def test_get_price_case_insensitive(self, pricing):
        """Test case insensitive matching."""
        input_price, _ = pricing.get_price("OpenAI", "GPT-4")
        assert input_price == 30.0

    def test_get_price_unknown_model(self, pricing):
        """Test unknown model returns zero."""
        input_price, output_price = pricing.get_price("unknown", "unknown")
        assert input_price == 0.0
        assert output_price == 0.0

    def test_calculate_cost(self, pricing):
        """Test cost calculation."""
        cost = pricing.calculate_cost("openai", "gpt-4", 1000000, 1000000)
        # Input: $30, Output: $60 = $90 total
        assert cost == 90.0

    def test_calculate_cost_small_tokens(self, pricing):
        """Test cost calculation with small token counts."""
        cost = pricing.calculate_cost("openai", "gpt-4", 1000, 500)
        # Input: (1000/1M) * 30 = 0.03
        # Output: (500/1M) * 60 = 0.03
        # Total: 0.06
        assert cost == pytest.approx(0.06, rel=1e-5)

    def test_custom_pricing(self):
        """Test custom pricing overrides."""
        custom = {("custom", "model"): (1.0, 2.0)}
        pricing = PricingProvider(custom_pricing=custom)

        input_price, output_price = pricing.get_price("custom", "model")
        assert input_price == 1.0
        assert output_price == 2.0


class TestCostEntry:
    """Test CostEntry dataclass."""

    def test_creation(self):
        """Test basic creation."""
        entry = CostEntry(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.05,
        )
        assert entry.provider == "openai"
        assert entry.input_tokens == 100
        assert entry.cost_usd == 0.05

    def test_serialization(self):
        """Test serialization roundtrip."""
        original = CostEntry(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.05,
        )

        data = original.to_dict()
        restored = CostEntry.from_dict(data)

        assert restored.provider == original.provider
        assert restored.cost_usd == original.cost_usd


class TestInMemoryCostStorage:
    """Test InMemoryCostStorage."""

    @pytest.fixture
    def storage(self):
        return InMemoryCostStorage()

    def test_store_and_retrieve(self, storage):
        """Test storing and retrieving entries."""
        entry = CostEntry("openai", "gpt-4", 100, 50, 0.05)
        storage.store(entry)

        entries = storage.get_entries()
        assert len(entries) == 1
        assert entries[0].provider == "openai"

    def test_filter_by_provider(self, storage):
        """Test filtering by provider."""
        storage.store(CostEntry("openai", "gpt-4", 100, 50, 0.05))
        storage.store(CostEntry("deepseek", "deepseek-chat", 100, 50, 0.01))

        openai_entries = storage.get_entries(provider="openai")
        assert len(openai_entries) == 1

    def test_filter_by_time(self, storage):
        """Test filtering by time range."""
        old_entry = CostEntry("openai", "gpt-4", 100, 50, 0.05)
        old_entry.timestamp = datetime.utcnow() - timedelta(hours=2)

        new_entry = CostEntry("openai", "gpt-4", 100, 50, 0.05)

        storage.store(old_entry)
        storage.store(new_entry)

        recent = storage.get_entries(since=datetime.utcnow() - timedelta(minutes=30))
        assert len(recent) == 1

    def test_clear(self, storage):
        """Test clearing entries."""
        storage.store(CostEntry("openai", "gpt-4", 100, 50, 0.05))
        storage.clear()

        assert len(storage.get_entries()) == 0


class TestFileCostStorage:
    """Test FileCostStorage."""

    @pytest.fixture
    def storage(self, tmp_path):
        return FileCostStorage(tmp_path / "costs.jsonl")

    def test_store_and_retrieve(self, storage):
        """Test storing and retrieving entries."""
        entry = CostEntry("openai", "gpt-4", 100, 50, 0.05)
        storage.store(entry)

        entries = storage.get_entries()
        assert len(entries) == 1
        assert entries[0].provider == "openai"

    def test_persistence(self, tmp_path):
        """Test that data persists across instances."""
        file_path = tmp_path / "costs.jsonl"

        # First instance
        storage1 = FileCostStorage(file_path)
        storage1.store(CostEntry("openai", "gpt-4", 100, 50, 0.05))

        # Second instance
        storage2 = FileCostStorage(file_path)
        entries = storage2.get_entries()

        assert len(entries) == 1


class TestCostTracker:
    """Test CostTracker."""

    @pytest.fixture
    def tracker(self):
        return CostTracker()

    def test_record_request(self, tracker):
        """Test recording a request."""
        entry = tracker.record_request("openai", "gpt-4", 1000000, 1000000)

        assert entry.provider == "openai"
        assert entry.input_tokens == 1000000
        assert entry.cost_usd > 0

    def test_get_current_spend(self, tracker):
        """Test getting current spend."""
        tracker.record_request("openai", "gpt-4", 1000000, 1000000)
        tracker.record_request("openai", "gpt-4", 1000000, 1000000)

        spend = tracker.get_current_spend(period="all")
        assert spend > 0

    def test_get_spend_breakdown(self, tracker):
        """Test spend breakdown."""
        tracker.record_request("openai", "gpt-4", 1000, 500)
        tracker.record_request("deepseek", "deepseek-chat", 1000, 500)

        breakdown = tracker.get_spend_breakdown(period="all")

        assert breakdown["total_cost"] > 0
        assert "by_provider" in breakdown
        assert "openai" in breakdown["by_provider"]
        assert "deepseek" in breakdown["by_provider"]

    def test_estimate_cost(self, tracker):
        """Test cost estimation."""
        estimated = tracker.estimate_cost("openai", "gpt-4", 1000000, 1000000)

        # Should match actual calculation
        actual = tracker.record_request("openai", "gpt-4", 1000000, 1000000)
        assert estimated == actual.cost_usd

    def test_budget_status_no_budget(self, tracker):
        """Test budget status when no budget set."""
        status = tracker.get_budget_status()

        assert status["has_budget"] is False

    def test_budget_status_with_budget(self):
        """Test budget status with budget set."""
        tracker = CostTracker(budget_usd=100.0, alert_threshold=0.8)

        # Record some spending
        tracker.record_request("openai", "gpt-4", 1000000, 1000000)

        status = tracker.get_budget_status()

        assert status["has_budget"] is True
        assert status["budget_usd"] == 100.0
        assert status["spent_usd"] > 0
        assert status["remaining_usd"] < 100.0
        assert 0 < status["percent_used"] < 100

    def test_excluded_from_budget(self):
        """Test that local providers have zero cost."""
        tracker = CostTracker()

        entry = tracker.record_request("ollama", "llama2", 1000, 500)

        assert entry.cost_usd == 0.0

    def test_get_current_spend_by_period(self, tracker):
        """Test getting spend for different periods."""
        # All these should work without error
        tracker.get_current_spend(period="daily")
        tracker.get_current_spend(period="weekly")
        tracker.get_current_spend(period="monthly")
        tracker.get_current_spend(period="all")
