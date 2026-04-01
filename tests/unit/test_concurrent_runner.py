"""Tests for llm_engine.concurrent.runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import patch

import pytest

from llm_engine.concurrent.runner import (
    ThreadPoolRetryRunner,
    exponential_backoff_seconds,
    is_transient_error,
    run_thread_pool_with_retries,
)


def test_is_transient_error_default_keywords() -> None:
    assert is_transient_error("connection reset by peer") is True
    assert is_transient_error("HTTP 429 too many requests") is True
    assert is_transient_error("invalid api key") is False


def test_is_transient_error_custom_keywords() -> None:
    assert is_transient_error("foo bar", keywords=("foo",)) is True
    assert is_transient_error("baz", keywords=("foo",)) is False


def test_exponential_backoff_seconds() -> None:
    assert exponential_backoff_seconds(1, initial=1.0, maximum=100.0) == 1.0
    assert exponential_backoff_seconds(2, initial=1.0, maximum=100.0) == 2.0
    assert exponential_backoff_seconds(3, initial=1.0, maximum=100.0) == 4.0
    assert exponential_backoff_seconds(0, initial=1.0, maximum=100.0) == 0.0
    assert exponential_backoff_seconds(5, initial=1.0, maximum=3.0) == 3.0


@dataclass
class _FakeResult:
    task_id: int
    failed: bool
    error: str
    retry_count: int


def test_runner_success_no_retry() -> None:
    def worker(task: int, rc: int) -> _FakeResult:
        return _FakeResult(task_id=task, failed=False, error="", retry_count=rc)

    out = run_thread_pool_with_retries(
        [1, 2, 3],
        worker,
        max_workers=2,
        max_retries=3,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.failed,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert [r.task_id for r in out] == [1, 2, 3]
    assert all(not r.failed for r in out)


def test_runner_transient_fail_then_succeed() -> None:
    attempts: Dict[int, int] = {}

    def worker(task: int, rc: int) -> _FakeResult:
        attempts[task] = attempts.get(task, 0) + 1
        if attempts[task] < 2:
            return _FakeResult(
                task_id=task,
                failed=True,
                error="connection timeout",
                retry_count=rc,
            )
        return _FakeResult(task_id=task, failed=False, error="", retry_count=rc)

    with patch("llm_engine.concurrent.runner.time.sleep"):
        out = run_thread_pool_with_retries(
            [10],
            worker,
            max_workers=1,
            max_retries=3,
            retry_delay=0.01,
            retry_delay_max=0.1,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.task_id,
        )
    assert len(out) == 1
    assert out[0].failed is False
    assert attempts[10] == 2


def test_runner_stops_after_max_retries() -> None:
    def worker(task: int, rc: int) -> _FakeResult:
        return _FakeResult(
            task_id=task,
            failed=True,
            error="503 service unavailable",
            retry_count=rc,
        )

    with patch("llm_engine.concurrent.runner.time.sleep"):
        out = run_thread_pool_with_retries(
            [1],
            worker,
            max_workers=1,
            max_retries=2,
            retry_delay=0.01,
            retry_delay_max=0.1,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            order_key=lambda r: r.task_id,
        )
    assert len(out) == 1
    assert out[0].failed is True
    assert out[0].retry_count == 2


def test_runner_non_retryable_fails_immediately() -> None:
    calls = {"n": 0}

    def worker(task: int, rc: int) -> _FakeResult:
        calls["n"] += 1
        return _FakeResult(
            task_id=task,
            failed=True,
            error="invalid api key",
            retry_count=rc,
        )

    out = run_thread_pool_with_retries(
        [1],
        worker,
        max_workers=1,
        max_retries=3,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.failed,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        order_key=lambda r: r.task_id,
    )
    assert calls["n"] == 1
    assert out[0].error == "invalid api key"


def test_runner_on_retry_scheduled() -> None:
    attempts: Dict[int, int] = {}

    def worker(task: int, rc: int) -> _FakeResult:
        attempts[task] = attempts.get(task, 0) + 1
        if attempts[task] < 2:
            return _FakeResult(
                task_id=task,
                failed=True,
                error="timeout",
                retry_count=rc,
            )
        return _FakeResult(task_id=task, failed=False, error="", retry_count=rc)

    scheduled: List[tuple] = []

    def on_retry(task: int, result: _FakeResult) -> None:
        scheduled.append((task, result.retry_count))

    with patch("llm_engine.concurrent.runner.time.sleep"):
        run_thread_pool_with_retries(
            [7],
            worker,
            max_workers=1,
            max_retries=3,
            retry_delay=0.01,
            retry_delay_max=0.1,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
            on_retry_scheduled=on_retry,
            order_key=lambda r: r.task_id,
        )
    assert scheduled == [(7, 0)]


def test_runner_on_worker_exception() -> None:
    def worker(task: int, rc: int) -> _FakeResult:
        raise RuntimeError("boom")

    def on_exc(task: int, exc: BaseException) -> _FakeResult:
        return _FakeResult(task_id=task, failed=True, error=str(exc), retry_count=0)

    out = run_thread_pool_with_retries(
        [1],
        worker,
        max_workers=1,
        max_retries=0,
        retry_delay=0.01,
        retry_delay_max=0.1,
        is_failed=lambda r: r.failed,
        error_message=lambda r: r.error,
        retry_count_from_result=lambda r: r.retry_count,
        on_worker_exception=on_exc,
        order_key=lambda r: r.task_id,
    )
    assert len(out) == 1
    assert "boom" in out[0].error


def test_thread_pool_retry_runner_class() -> None:
    runner = ThreadPoolRetryRunner(
        max_workers=2,
        max_retries=1,
        retry_delay=0.01,
        retry_delay_max=0.1,
    )

    def w(t: int, rc: int) -> _FakeResult:
        return _FakeResult(task_id=t, failed=False, error="", retry_count=rc)

    r = runner.run(
        [1, 2],
        w,
        is_failed=lambda x: x.failed,
        error_message=lambda x: x.error,
        retry_count_from_result=lambda x: x.retry_count,
        order_key=lambda x: x.task_id,
    )
    assert [x.task_id for x in r] == [1, 2]


def test_worker_exception_without_handler_raises() -> None:
    def worker(task: int, rc: int) -> _FakeResult:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        run_thread_pool_with_retries(
            [1],
            worker,
            max_workers=1,
            max_retries=0,
            retry_delay=0.01,
            retry_delay_max=0.1,
            is_failed=lambda r: r.failed,
            error_message=lambda r: r.error,
            retry_count_from_result=lambda r: r.retry_count,
        )
