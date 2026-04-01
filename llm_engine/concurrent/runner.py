"""
ThreadPoolExecutor + as_completed loop with retry queue and exponential backoff.

Mirrors the control flow previously duplicated in ask_llm BatchProcessor /
GlobalBatchProcessor (no Rich, no domain types).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

TTask = TypeVar("TTask")
TResult = TypeVar("TResult")

# Same keyword list as ask_llm batch._is_retryable_error (kept in sync intentionally).
DEFAULT_TRANSIENT_ERROR_KEYWORDS: Tuple[str, ...] = (
    "timeout",
    "connection",
    "network",
    "rate limit",
    "429",
    "503",
    "502",
    "500",
)


def is_transient_error(
    error_message: str,
    keywords: Optional[Sequence[str]] = None,
) -> bool:
    """
    Return True if *error_message* looks like a transient / retryable failure.

    Args:
        error_message: Typically ``str(exc)`` or API error body.
        keywords: If None, uses :data:`DEFAULT_TRANSIENT_ERROR_KEYWORDS`.
    """
    if not error_message:
        return False
    kws = tuple(keywords) if keywords is not None else DEFAULT_TRANSIENT_ERROR_KEYWORDS
    lower = error_message.lower()
    return any(kw in lower for kw in kws)


def exponential_backoff_seconds(
    attempt_1_based: int,
    *,
    initial: float,
    maximum: float,
) -> float:
    """
    Delay before retry *attempt_1_based* (1 = first retry after initial failure).

    ``min(initial * 2 ** (n - 1), maximum)`` — matches ask_llm batch backoff.
    """
    if attempt_1_based < 1:
        return 0.0
    raw = initial * (2 ** (attempt_1_based - 1))
    return min(raw, maximum)


def run_thread_pool_with_retries(
    tasks: Sequence[TTask],
    worker: Callable[[TTask, int], TResult],
    *,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
    retry_delay_max: float,
    is_failed: Callable[[TResult], bool],
    error_message: Callable[[TResult], str],
    retry_count_from_result: Callable[[TResult], int],
    is_retryable_error: Callable[[str], bool] = is_transient_error,
    on_worker_exception: Optional[Callable[[TTask, BaseException], TResult]] = None,
    on_retry_scheduled: Optional[Callable[[TTask, TResult], None]] = None,
    order_key: Callable[[TResult], Any] = lambda r: getattr(r, "task_id", 0),
) -> List[TResult]:
    """
    Run ``worker(task, retry_count)`` for each task in a thread pool.

    Failed results that pass ``is_failed`` and ``is_retryable_error(error_message(r))``
    and ``retry_count_from_result(r) < max_retries`` are re-submitted with
    ``retry_count`` incremented by one, after exponential backoff.

    Futures that raise call ``on_worker_exception`` if set; otherwise the exception
    propagates.

    Results are sorted by ``order_key`` before return (stable task order).
    """
    runner = ThreadPoolRetryRunner(
        max_workers=max_workers,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_delay_max=retry_delay_max,
    )
    return runner.run(
        tasks,
        worker,
        is_failed=is_failed,
        error_message=error_message,
        retry_count_from_result=retry_count_from_result,
        is_retryable_error=is_retryable_error,
        on_worker_exception=on_worker_exception,
        on_retry_scheduled=on_retry_scheduled,
        order_key=order_key,
    )


class ThreadPoolRetryRunner(Generic[TTask, TResult]):
    """Configurable runner; prefer :func:`run_thread_pool_with_retries` for one-shots."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_retries: int,
        retry_delay: float,
        retry_delay_max: float,
    ) -> None:
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_delay_max = retry_delay_max

    def run(
        self,
        tasks: Sequence[TTask],
        worker: Callable[[TTask, int], TResult],
        *,
        is_failed: Callable[[TResult], bool],
        error_message: Callable[[TResult], str],
        retry_count_from_result: Callable[[TResult], int],
        is_retryable_error: Callable[[str], bool] = is_transient_error,
        on_worker_exception: Optional[Callable[[TTask, BaseException], TResult]] = None,
        on_retry_scheduled: Optional[Callable[[TTask, TResult], None]] = None,
        order_key: Callable[[TResult], Any] = lambda r: getattr(r, "task_id", 0),
    ) -> List[TResult]:
        results: List[TResult] = []
        future_to_task: Dict[Any, TTask] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for t in tasks:
                fut = executor.submit(worker, t, 0)
                future_to_task[fut] = t

            retry_queue: List[Tuple[TTask, int]] = []

            while future_to_task or retry_queue:
                if future_to_task:
                    for future in as_completed(future_to_task):
                        task = future_to_task.pop(future)
                        try:
                            result = future.result()
                        except BaseException as exc:
                            if on_worker_exception is None:
                                raise
                            result = on_worker_exception(task, exc)

                        if is_failed(result) and (
                            retry_count_from_result(result) < self.max_retries
                            and is_retryable_error(error_message(result) or "")
                        ):
                            if on_retry_scheduled is not None:
                                on_retry_scheduled(task, result)
                            retry_queue.append(
                                (task, retry_count_from_result(result) + 1),
                            )
                        else:
                            results.append(result)

                if retry_queue:
                    first_retry = retry_queue[0][1]
                    delay = exponential_backoff_seconds(
                        first_retry,
                        initial=self.retry_delay,
                        maximum=self.retry_delay_max,
                    )
                    time.sleep(delay)
                    for task, retry_count in retry_queue:
                        fut = executor.submit(worker, task, retry_count)
                        future_to_task[fut] = task
                    retry_queue.clear()

        results.sort(key=order_key)
        return results
