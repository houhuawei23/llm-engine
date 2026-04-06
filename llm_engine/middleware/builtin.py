"""
Built-in middleware implementations for common use cases.

Provides ready-to-use middleware for logging, timing, retries, and more.
"""

import time
from typing import Callable, Optional

from loguru import logger

from llm_engine.middleware.base import Middleware, RequestContext, Response


class LoggingMiddleware(Middleware):
    """Logs requests and responses with configurable detail level.

    Example:
        >>> # Basic logging
        >>> mw = LoggingMiddleware()
        >>>
        >>> # Debug level with full content
        >>> mw = LoggingMiddleware(level="DEBUG", log_content=True)
        >>>
        >>> # Custom logger
        >>> mw = LoggingMiddleware(logger=my_logger, max_content_length=1000)

    """

    def __init__(
        self,
        level: str = "INFO",
        log_content: bool = False,
        max_content_length: int = 500,
        logger_instance: Optional[Callable] = None,
    ):
        """Initialize logging middleware.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_content: Whether to log full message content
            max_content_length: Truncate content longer than this
            logger_instance: Custom logger function (defaults to loguru)
        """
        self.level = level.upper()
        self.log_content = log_content
        self.max_content_length = max_content_length
        self._logger = logger_instance or logger

    def _truncate(self, text: str) -> str:
        """Truncate text if too long."""
        if len(text) <= self.max_content_length:
            return text
        return text[: self.max_content_length] + "..."

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Log the request."""
        messages_summary = f"{len(context.messages)} messages"

        if self.log_content and context.messages:
            last_msg = context.messages[-1].get("content", "")
            content_preview = f" | Last: {self._truncate(last_msg)}"
        else:
            content_preview = ""

        getattr(self._logger, self.level.lower())(
            f"[LLM Request] {context.provider}/{context.model} | "
            f"{messages_summary}{content_preview} | "
            f"temp={context.temperature} max_tokens={context.max_tokens}"
        )
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Log the response."""
        content_preview = (
            f" | Content: {self._truncate(response.content)}"
            if self.log_content
            else f" | {len(response.content)} chars"
        )

        latency_info = (
            f" | Latency: {response.latency_ms:.1f}ms"
            if response.latency_ms
            else ""
        )

        token_info = (
            f" | Tokens: {response.input_tokens}→{response.output_tokens}"
            if response.usage
            else ""
        )

        getattr(self._logger, self.level.lower())(
            f"[LLM Response] {context.provider}/{context.model}{content_preview}"
            f"{token_info}{latency_info}"
        )
        return response


class TimingMiddleware(Middleware):
    """Measures and records request latency.

    Adds latency information to response metadata.

    Example:
        >>> mw = TimingMiddleware()
        >>> # Response will have latency_ms set
        >>> response = await chain.execute(context, provider_call)
        >>> print(f"Took {response.latency_ms}ms")

    """

    def __init__(self, metric_name: str = "llm_request_duration_ms"):
        """Initialize timing middleware.

        Args:
            metric_name: Name for the latency metric in metadata
        """
        self.metric_name = metric_name
        self._request_times: dict = {}  # Store start times by id

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Record start time."""
        self._request_times[id(context)] = time.time()
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Calculate and add latency."""
        start_time = self._request_times.pop(id(context), None)
        if start_time:
            latency_ms = (time.time() - start_time) * 1000
            response.latency_ms = latency_ms
            response.metadata[self.metric_name] = latency_ms
        return response


class RetryMiddleware(Middleware):
    """Retry failed requests with exponential backoff.

    Note: This is in addition to provider-level retries. Use sparingly.

    Example:
        >>> mw = RetryMiddleware(max_retries=3, base_delay=1.0)
        >>> # Will retry on transient errors up to 3 times

    """

    TRANSIENT_ERRORS = (
        "timeout",
        "connection",
        "rate limit",
        "429",
        "503",
        "502",
        "500",
    )

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_on: Optional[tuple] = None,
    ):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            retry_on: Tuple of error substrings to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on or self.TRANSIENT_ERRORS

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if error is transient/retryable."""
        error_str = str(error).lower()
        return any(err in error_str for err in self.retry_on)

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        import random

        delay = self.base_delay * (2 ** attempt)
        # Add jitter
        delay = delay * (0.5 + random.random())
        return min(delay, self.max_delay)

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Pass through - no action on request."""
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Pass through - retries handled at execution level."""
        return response


class ContentFilterMiddleware(Middleware):
    """Filter or transform request/response content.

    Useful for PII removal, content moderation, or format normalization.

    Example:
        >>> # Remove PII patterns
        >>> mw = ContentFilterMiddleware(
        ...     request_filter=remove_pii,
        ...     response_filter=normalize_whitespace
        ... )

    """

    def __init__(
        self,
        request_filter: Optional[Callable[[str], str]] = None,
        response_filter: Optional[Callable[[str], str]] = None,
    ):
        """Initialize content filter middleware.

        Args:
            request_filter: Function to filter request content
            response_filter: Function to filter response content
        """
        self.request_filter = request_filter
        self.response_filter = response_filter

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Apply request filter to user messages."""
        if self.request_filter:
            context.messages = [
                {
                    **msg,
                    "content": self.request_filter(msg.get("content", ""))
                    if msg.get("role") == "user"
                    else msg.get("content", ""),
                }
                for msg in context.messages
            ]
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Apply response filter to content."""
        if self.response_filter:
            response.content = self.response_filter(response.content)
            if response.reasoning:
                response.reasoning = self.response_filter(response.reasoning)
        return response


class HeaderInjectionMiddleware(Middleware):
    """Inject custom headers into provider requests.

    Useful for request tracking, authentication, or routing.

    Example:
        >>> mw = HeaderInjectionMiddleware({
        ...     "X-Request-ID": lambda ctx: str(uuid.uuid4()),
        ...     "X-User-ID": lambda ctx: ctx.metadata.get("user_id", "anonymous"),
        ... })

    """

    def __init__(
        self,
        headers: dict[str, Callable[[RequestContext], str]],
    ):
        """Initialize header injection middleware.

        Args:
            headers: Dict of header names to functions that return header values
        """
        self.headers = headers

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Inject headers into context metadata."""
        if "headers" not in context.metadata:
            context.metadata["headers"] = {}

        for name, value_fn in self.headers.items():
            context.metadata["headers"][name] = value_fn(context)

        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Pass through - no action on response."""
        return response
