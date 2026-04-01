"""
Thread-pool execution with transient-error retries and exponential backoff.

UI-agnostic; suitable for wrapping synchronous LLM client calls from worker threads.
"""

from llm_engine.concurrent.runner import (
    DEFAULT_TRANSIENT_ERROR_KEYWORDS,
    ThreadPoolRetryRunner,
    exponential_backoff_seconds,
    is_transient_error,
    run_thread_pool_with_retries,
)

__all__ = [
    "DEFAULT_TRANSIENT_ERROR_KEYWORDS",
    "ThreadPoolRetryRunner",
    "exponential_backoff_seconds",
    "is_transient_error",
    "run_thread_pool_with_retries",
]
