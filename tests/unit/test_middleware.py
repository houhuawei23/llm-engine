"""Tests for middleware framework."""

import pytest
from unittest.mock import AsyncMock, Mock

from llm_engine.middleware import (
    ContentFilterMiddleware,
    HeaderInjectionMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    MiddlewareError,
    RequestContext,
    Response,
    TimingMiddleware,
)


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_basic_creation(self):
        """Test creating a basic request context."""
        ctx = RequestContext(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert ctx.provider == "openai"
        assert ctx.model == "gpt-4"
        assert ctx.temperature == 0.7
        assert ctx.top_p == 1.0
        assert ctx.stream is False

    def test_copy(self):
        """Test that copy creates independent copy."""
        ctx = RequestContext(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"key": "value"},
        )
        copied = ctx.copy()

        assert copied.provider == ctx.provider
        assert copied.messages == ctx.messages
        assert copied.metadata == ctx.metadata

        # Modifications should not affect original
        copied.messages.append({"role": "assistant", "content": "Hi"})
        copied.metadata["new"] = "data"

        assert len(ctx.messages) == 1
        assert "new" not in ctx.metadata


class TestResponse:
    """Test Response dataclass."""

    def test_basic_creation(self):
        """Test creating a basic response."""
        resp = Response(content="Hello world")
        assert resp.content == "Hello world"
        assert resp.reasoning is None
        assert resp.latency_ms is None

    def test_with_usage(self):
        """Test response with token usage."""
        resp = Response(
            content="Hello",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.total_tokens == 15

    def test_token_defaults(self):
        """Test token defaults when usage is empty."""
        resp = Response(content="Hello")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.total_tokens == 0


class CustomTestMiddleware(Middleware):
    """Test middleware for unit testing."""

    def __init__(self, name="test"):
        self.name = name
        self.request_calls = []
        self.response_calls = []

    async def process_request(self, context):
        self.request_calls.append(context)
        context.metadata[f"{self.name}_request"] = True
        return context

    async def process_response(self, context, response):
        self.response_calls.append((context, response))
        response.metadata[f"{self.name}_response"] = True
        return response


class TestMiddlewareChain:
    """Test MiddlewareChain execution."""

    @pytest.mark.asyncio
    async def test_empty_chain(self):
        """Test chain with no middleware passes through."""
        chain = MiddlewareChain()
        ctx = RequestContext(provider="test", model="model", messages=[])

        async def provider_call(c):
            return Response(content="response")

        response = await chain.execute(ctx, provider_call)
        assert response.content == "response"

    @pytest.mark.asyncio
    async def test_single_middleware(self):
        """Test chain with single middleware."""
        mw = CustomTestMiddleware("single")
        chain = MiddlewareChain([mw])

        ctx = RequestContext(provider="test", model="model", messages=[])

        async def provider_call(c):
            return Response(content="response")

        response = await chain.execute(ctx, provider_call)

        assert len(mw.request_calls) == 1
        assert len(mw.response_calls) == 1
        assert response.metadata["single_response"] is True

    @pytest.mark.asyncio
    async def test_multiple_middleware(self):
        """Test chain with multiple middleware executes in order."""
        mw1 = CustomTestMiddleware("first")
        mw2 = CustomTestMiddleware("second")
        chain = MiddlewareChain([mw1, mw2])

        ctx = RequestContext(provider="test", model="model", messages=[])

        async def provider_call(c):
            return Response(content="response")

        response = await chain.execute(ctx, provider_call)

        # Both middleware should have been called
        assert len(mw1.request_calls) == 1
        assert len(mw2.request_calls) == 1
        assert len(mw1.response_calls) == 1
        assert len(mw2.response_calls) == 1

        # Request order: first, then second
        # Note: context is modified in place through the chain
        # mw1 sees context after its own modification
        assert mw1.request_calls[0].metadata.get("first_request") is True

        # Response order: second, first (reverse)
        assert response.metadata.get("second_response") is True
        assert response.metadata.get("first_response") is True

    @pytest.mark.asyncio
    async def test_middleware_modifies_request(self):
        """Test that middleware can modify request context."""
        class ModifyMiddleware(Middleware):
            async def process_request(self, context):
                context.temperature = 0.5
                return context

            async def process_response(self, context, response):
                return response

        chain = MiddlewareChain([ModifyMiddleware()])
        ctx = RequestContext(provider="test", model="model", messages=[], temperature=0.9)

        received_ctx = None

        async def provider_call(c):
            nonlocal received_ctx
            received_ctx = c
            return Response(content="response")

        await chain.execute(ctx, provider_call)

        assert received_ctx.temperature == 0.5

    @pytest.mark.asyncio
    async def test_middleware_error_in_request(self):
        """Test error handling in request processing."""
        class ErrorMiddleware(Middleware):
            async def process_request(self, context):
                raise ValueError("Request error")

            async def process_response(self, context, response):
                return response

        chain = MiddlewareChain([ErrorMiddleware()])
        ctx = RequestContext(provider="test", model="model", messages=[])

        async def provider_call(c):
            return Response(content="response")

        with pytest.raises(MiddlewareError) as exc_info:
            await chain.execute(ctx, provider_call)

        assert "ErrorMiddleware" in str(exc_info.value)
        assert "request" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_middleware_error_in_response(self):
        """Test error handling in response processing."""
        class ErrorMiddleware(Middleware):
            async def process_request(self, context):
                return context

            async def process_response(self, context, response):
                raise ValueError("Response error")

        chain = MiddlewareChain([ErrorMiddleware()])
        ctx = RequestContext(provider="test", model="model", messages=[])

        async def provider_call(c):
            return Response(content="response")

        with pytest.raises(MiddlewareError) as exc_info:
            await chain.execute(ctx, provider_call)

        assert "ErrorMiddleware" in str(exc_info.value)
        assert "response" in str(exc_info.value).lower()

    def test_chain_add(self):
        """Test adding middleware to chain."""
        chain = MiddlewareChain()
        mw = CustomTestMiddleware()

        result = chain.add(mw)

        assert result is chain  # Returns self for chaining
        assert len(chain.middleware) == 1

    def test_chain_insert(self):
        """Test inserting middleware at specific position."""
        mw1 = CustomTestMiddleware("first")
        mw2 = CustomTestMiddleware("second")
        chain = MiddlewareChain([mw1])

        chain.insert(0, mw2)

        middleware = chain.middleware
        assert middleware[0] is mw2
        assert middleware[1] is mw1

    def test_chain_remove(self):
        """Test removing middleware by type."""
        mw = CustomTestMiddleware()
        chain = MiddlewareChain([mw])

        result = chain.remove(CustomTestMiddleware)

        assert result is True
        assert len(chain.middleware) == 0

    def test_chain_remove_not_found(self):
        """Test removing non-existent middleware type."""
        chain = MiddlewareChain([CustomTestMiddleware()])

        result = chain.remove(LoggingMiddleware)

        assert result is False
        assert len(chain.middleware) == 1

    def test_chain_clear(self):
        """Test clearing all middleware."""
        chain = MiddlewareChain([CustomTestMiddleware(), CustomTestMiddleware()])

        chain.clear()

        assert len(chain.middleware) == 0


class TestLoggingMiddleware:
    """Test LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_request(self, caplog):
        """Test that request is logged."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        mw = LoggingMiddleware(level="INFO", log_content=False, logger_instance=mock_logger)

        ctx = RequestContext(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello world"}],
        )

        result = await mw.process_request(ctx)
        assert result is ctx

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "openai/gpt-4" in log_message
        assert "1 messages" in log_message

    @pytest.mark.asyncio
    async def test_logs_response(self, caplog):
        """Test that response is logged."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        mw = LoggingMiddleware(level="INFO", log_content=True, logger_instance=mock_logger)

        ctx = RequestContext(provider="openai", model="gpt-4", messages=[])
        response = Response(
            content="Test response",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=100.5,
        )

        result = await mw.process_response(ctx, response)
        assert result is response

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Test response" in log_message
        assert "10→5" in log_message
        assert "100.5ms" in log_message

    @pytest.mark.asyncio
    async def test_truncate_long_content(self, caplog):
        """Test that long content is truncated."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        mw = LoggingMiddleware(level="INFO", log_content=True, max_content_length=10, logger_instance=mock_logger)

        ctx = RequestContext(provider="test", model="model", messages=[])
        response = Response(content="This is a very long response")

        await mw.process_response(ctx, response)

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "This is a ..." in log_message
        assert "very long response" not in log_message


class TestTimingMiddleware:
    """Test TimingMiddleware."""

    @pytest.mark.asyncio
    async def test_adds_latency_to_response(self):
        """Test that latency is added to response."""
        import asyncio

        mw = TimingMiddleware()
        ctx = RequestContext(provider="test", model="model", messages=[])

        # Process request
        await mw.process_request(ctx)
        await asyncio.sleep(0.01)  # Small delay

        # Process response
        response = Response(content="test")
        result = await mw.process_response(ctx, response)

        assert result.latency_ms is not None
        assert result.latency_ms >= 10  # At least 10ms
        assert result.metadata["llm_request_duration_ms"] == result.latency_ms

    @pytest.mark.asyncio
    async def test_custom_metric_name(self):
        """Test custom metric name."""
        mw = TimingMiddleware(metric_name="custom_latency")
        ctx = RequestContext(provider="test", model="model", messages=[])

        await mw.process_request(ctx)
        response = Response(content="test")
        result = await mw.process_response(ctx, response)

        assert "custom_latency" in result.metadata


class TestContentFilterMiddleware:
    """Test ContentFilterMiddleware."""

    @pytest.mark.asyncio
    async def test_filter_request_content(self):
        """Test filtering user message content."""

        def remove_hello(text):
            return text.replace("Hello", "Hi")

        mw = ContentFilterMiddleware(request_filter=remove_hello)
        ctx = RequestContext(
            provider="test",
            model="model",
            messages=[
                {"role": "system", "content": "Hello system"},
                {"role": "user", "content": "Hello user"},
            ],
        )

        result = await mw.process_request(ctx)

        # User message should be filtered
        assert result.messages[1]["content"] == "Hi user"
        # System message should not be filtered
        assert result.messages[0]["content"] == "Hello system"

    @pytest.mark.asyncio
    async def test_filter_response_content(self):
        """Test filtering response content."""

        def to_upper(text):
            return text.upper()

        mw = ContentFilterMiddleware(response_filter=to_upper)
        ctx = RequestContext(provider="test", model="model", messages=[])
        response = Response(content="hello world", reasoning="some reasoning")

        result = await mw.process_response(ctx, response)

        assert result.content == "HELLO WORLD"
        assert result.reasoning == "SOME REASONING"


class TestHeaderInjectionMiddleware:
    """Test HeaderInjectionMiddleware."""

    @pytest.mark.asyncio
    async def test_inject_headers(self):
        """Test header injection into context."""
        mw = HeaderInjectionMiddleware(
            headers={
                "X-Request-ID": lambda ctx: "12345",
                "X-User": lambda ctx: ctx.metadata.get("user", "anon"),
            }
        )

        ctx = RequestContext(
            provider="test", model="model", messages=[], metadata={"user": "alice"}
        )

        result = await mw.process_request(ctx)

        assert result.metadata["headers"]["X-Request-ID"] == "12345"
        assert result.metadata["headers"]["X-User"] == "alice"

    @pytest.mark.asyncio
    async def test_preserves_existing_headers(self):
        """Test that existing headers are preserved."""
        mw = HeaderInjectionMiddleware(headers={"X-New": lambda ctx: "value"})

        ctx = RequestContext(
            provider="test",
            model="model",
            messages=[],
            metadata={"headers": {"X-Existing": "old"}},
        )

        result = await mw.process_request(ctx)

        assert result.metadata["headers"]["X-Existing"] == "old"
        assert result.metadata["headers"]["X-New"] == "value"
