"""
Middleware chain execution for LLM Engine.

Provides sequential and conditional middleware execution.
"""

from typing import Callable, Coroutine, List, Optional, TypeVar

from llm_engine.middleware.base import (
    Middleware,
    MiddlewareError,
    RequestContext,
    Response,
)

T = TypeVar("T")


class MiddlewareChain:
    """Chain of middleware executed sequentially.

    Each middleware's process_request is called in order,
    then process_response is called in reverse order.

    Example:
        >>> chain = MiddlewareChain([
        ...     LoggingMiddleware(),
        ...     TimingMiddleware(),
        ... ])
        >>> context = RequestContext(provider="openai", model="gpt-4", messages=[...])
        >>> response = await chain.execute(context, provider_call)
    """

    def __init__(self, middleware: Optional[List[Middleware]] = None):
        """Initialize middleware chain.

        Args:
            middleware: List of middleware instances to execute
        """
        self._middleware: List[Middleware] = list(middleware) if middleware else []

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add middleware to the chain.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining
        """
        self._middleware.append(middleware)
        return self

    def insert(self, index: int, middleware: Middleware) -> "MiddlewareChain":
        """Insert middleware at specific position.

        Args:
            index: Position to insert at
            middleware: Middleware instance to insert

        Returns:
            Self for method chaining
        """
        self._middleware.insert(index, middleware)
        return self

    def remove(self, middleware_type: type) -> bool:
        """Remove first middleware of given type.

        Args:
            middleware_type: Type of middleware to remove

        Returns:
            True if removed, False if not found
        """
        for i, m in enumerate(self._middleware):
            if isinstance(m, middleware_type):
                self._middleware.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all middleware from chain."""
        self._middleware.clear()

    @property
    def middleware(self) -> List[Middleware]:
        """Get list of middleware in chain."""
        return self._middleware.copy()

    async def execute(
        self,
        context: RequestContext,
        provider_call: Callable[[RequestContext], Coroutine[None, None, Response]],
    ) -> Response:
        """Execute middleware chain around provider call.

        Process request through all middleware in order,
        call provider, then process response through middleware in reverse.

        Args:
            context: Initial request context
            provider_call: Async function that calls the provider

        Returns:
            Final response after all middleware processing

        Raises:
            MiddlewareError: If middleware processing fails
        """
        if not self._middleware:
            # No middleware, just call provider
            return await provider_call(context)

        # Process requests through middleware (forward)
        current_context = context
        for _i, mw in enumerate(self._middleware):
            try:
                current_context = await mw.process_request(current_context)
                if current_context is None:
                    raise MiddlewareError(
                        f"Middleware {type(mw).__name__} returned None from process_request",
                        middleware_name=type(mw).__name__,
                    )
            except Exception as e:
                if isinstance(e, MiddlewareError):
                    raise
                raise MiddlewareError(
                    f"Middleware {type(mw).__name__} failed during process_request: {e}",
                    middleware_name=type(mw).__name__,
                ) from e

        # Call provider with final context
        try:
            response = await provider_call(current_context)
        except Exception:
            # Provider errors pass through middleware for potential handling
            raise

        # Process responses through middleware (reverse)
        current_response = response
        for mw in reversed(self._middleware):
            try:
                current_response = await mw.process_response(
                    current_context, current_response
                )
                if current_response is None:
                    raise MiddlewareError(
                        f"Middleware {type(mw).__name__} returned None from process_response",
                        middleware_name=type(mw).__name__,
                    )
            except Exception as e:
                if isinstance(e, MiddlewareError):
                    raise
                raise MiddlewareError(
                    f"Middleware {type(mw).__name__} failed during process_response: {e}",
                    middleware_name=type(mw).__name__,
                ) from e

        return current_response


class ConditionalMiddleware(Middleware):
    """Middleware that only applies when condition is met.

    Example:
        >>> # Only log requests to specific providers
        >>> ConditionalMiddleware(
        ...     LoggingMiddleware(),
        ...     condition=lambda ctx: ctx.provider == "openai"
        ... )
    """

    def __init__(
        self,
        middleware: Middleware,
        condition: Callable[[RequestContext], bool],
    ):
        """Initialize conditional middleware.

        Args:
            middleware: Middleware to conditionally execute
            condition: Function that returns True if middleware should run
        """
        self._middleware = middleware
        self._condition = condition

    async def process_request(self, context: RequestContext) -> RequestContext:
        """Process request if condition is met."""
        if self._condition(context):
            return await self._middleware.process_request(context)
        return context

    async def process_response(
        self, context: RequestContext, response: Response
    ) -> Response:
        """Process response if condition was met for request."""
        if self._condition(context):
            return await self._middleware.process_response(context, response)
        return response
