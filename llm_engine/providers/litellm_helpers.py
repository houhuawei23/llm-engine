"""
LiteLLM integration helpers.

Centralizes ``acompletion`` keyword construction and maps LiteLLM / provider
exceptions to :class:`llm_engine.exceptions.LLMProviderError` for consistent
error handling across providers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from llm_engine.exceptions import LLMProviderError

try:
    from litellm.exceptions import (
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        ContentPolicyViolationError,
        ContextWindowExceededError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
        ServiceUnavailableError,
    )
    from litellm.exceptions import (
        Timeout as LitellmTimeout,
    )
except ImportError:  # pragma: no cover - defensive
    APIConnectionError = AuthenticationError = BadRequestError = None  # type: ignore
    ContentPolicyViolationError = ContextWindowExceededError = None  # type: ignore
    InternalServerError = NotFoundError = PermissionDeniedError = None  # type: ignore
    RateLimitError = ServiceUnavailableError = LitellmTimeout = None  # type: ignore


def build_acompletion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    payload: dict[str, Any],
    api_key: str | None,
    api_base: str | None,
    timeout: int | None,
    stream: bool,
) -> dict[str, Any]:
    """
    Build kwargs for :func:`litellm.acompletion` from engine payload.

    Keeps one code path for OpenAI-compatible and Ollama (via LiteLLM).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "top_p": payload.get("top_p"),
        "stream": stream,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    for key in ("presence_penalty", "frequency_penalty", "response_format"):
        if key in payload:
            kwargs[key] = payload[key]
    # Avoid nested retries: application-level ``generate_with_retry`` handles retries.
    kwargs.setdefault("num_retries", 0)
    return kwargs


def _provider_error(cause: BaseException, message: str) -> LLMProviderError:
    err = LLMProviderError(message)
    err.__cause__ = cause
    return err


def map_litellm_exception(exc: BaseException, provider_name: str) -> BaseException:
    """
    Map LiteLLM / HTTP-layer errors to :class:`LLMProviderError` or timeout errors.

    Passes through :class:`LLMProviderError` unchanged. Wraps :class:`asyncio.TimeoutError`
    so callers get a stable message containing \"timed out\".
    """
    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, asyncio.TimeoutError):
        return _provider_error(exc, f"{provider_name} request timed out: {exc}")

    if LitellmTimeout is not None and isinstance(exc, LitellmTimeout):
        return _provider_error(exc, f"{provider_name} request timed out (LiteLLM): {exc}")

    if AuthenticationError is not None and isinstance(exc, AuthenticationError):
        logger.error(f"{provider_name} authentication failed: {exc}")
        return _provider_error(exc, f"{provider_name} authentication failed: {exc}")

    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return _provider_error(exc, f"{provider_name} rate limit: {exc}")

    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return _provider_error(exc, f"{provider_name} connection error: {exc}")

    if ContextWindowExceededError is not None and isinstance(exc, ContextWindowExceededError):
        return _provider_error(exc, f"{provider_name} context window exceeded: {exc}")

    if ContentPolicyViolationError is not None and isinstance(exc, ContentPolicyViolationError):
        return _provider_error(exc, f"{provider_name} content policy: {exc}")

    if BadRequestError is not None and isinstance(exc, BadRequestError):
        return _provider_error(exc, f"{provider_name} bad request: {exc}")

    if NotFoundError is not None and isinstance(exc, NotFoundError):
        return _provider_error(exc, f"{provider_name} not found: {exc}")

    if PermissionDeniedError is not None and isinstance(exc, PermissionDeniedError):
        return _provider_error(exc, f"{provider_name} permission denied: {exc}")

    if ServiceUnavailableError is not None and isinstance(exc, ServiceUnavailableError):
        return _provider_error(exc, f"{provider_name} service unavailable: {exc}")

    if InternalServerError is not None and isinstance(exc, InternalServerError):
        return _provider_error(exc, f"{provider_name} server error: {exc}")

    err = str(exc).lower()
    if "timeout" in err or "timed out" in err:
        return _provider_error(exc, f"{provider_name} request timed out: {exc}")

    return _provider_error(exc, f"{provider_name} API call failed: {exc}")
