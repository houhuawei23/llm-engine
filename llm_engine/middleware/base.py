"""
Base middleware interfaces for LLM Engine.

Middleware provides a pluggable way to process requests and responses,
enabling cross-cutting concerns like logging, caching, metrics, and authentication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RequestContext:
    """Context object passed through middleware chain.

    Contains all information about the request being processed.

    Attributes:
        provider: The LLM provider name (e.g., "openai", "deepseek")
        model: The model name being used
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        stream: Whether this is a streaming request
        metadata: Additional metadata for middleware use
        start_time: When the request started
    """

    provider: str
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)

    def copy(self) -> "RequestContext":
        """Create a deep copy of the context."""
        return RequestContext(
            provider=self.provider,
            model=self.model,
            messages=self.messages.copy(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=self.stream,
            metadata=self.metadata.copy(),
            start_time=self.start_time,
        )


@dataclass
class Response:
    """Response object passed through middleware chain.

    Attributes:
        content: The generated text content
        reasoning: Optional reasoning content (e.g., from DeepSeek reasoner)
        usage: Token usage information
        finish_reason: Why the generation finished
        metadata: Additional metadata from the provider
        latency_ms: Request latency in milliseconds
    """

    content: str
    reasoning: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", self.input_tokens + self.output_tokens)


class Middleware(ABC):
    """Abstract base class for LLM Engine middleware.

    Middleware can modify requests before they are sent to the provider
    and process responses before they are returned to the caller.

    Example:
        >>> class LoggingMiddleware(Middleware):
        ...     async def process_request(self, context):
        ...         print(f"Request to {context.provider}")
        ...         return context
        ...
        ...     async def process_response(self, context, response):
        ...         print(f"Response: {len(response.content)} chars")
        ...         return response
    """

    @abstractmethod
    async def process_request(self, context: RequestContext) -> RequestContext:
        """Process request before sending to provider.

        Args:
            context: The request context

        Returns:
            The possibly modified context
        """

    @abstractmethod
    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Process response from provider.

        Args:
            context: The original request context
            response: The provider response

        Returns:
            The possibly modified response
        """


class MiddlewareError(Exception):
    """Error raised by middleware processing."""

    def __init__(self, message: str, middleware_name: Optional[str] = None):
        super().__init__(message)
        self.middleware_name = middleware_name
