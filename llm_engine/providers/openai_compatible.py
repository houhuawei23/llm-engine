"""
OpenAI Compatible API provider base class.

Contains common logic for OpenAI-compatible APIs, inherited by OpenAIProvider,
DeepSeekProvider, CustomProvider, etc.
"""

import asyncio
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import litellm
from loguru import logger
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

from llm_engine.config import LLMConfig
from llm_engine.exceptions import LLMProviderError
from llm_engine.providers.base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI Compatible API provider base class.

    Provides common implementation for OpenAI-compatible APIs, including:
    - Message building
    - Request payload building
    - Response handling
    - Streaming response parsing
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI compatible provider.

        Args:
            config: LLM configuration object
        """
        super().__init__(config)
        self.provider_name = self._get_provider_name()
        self._client: Optional[OpenAI] = None

    def _get_provider_name(self) -> str:
        """
        Get provider name (for error messages).

        Returns:
            Provider name
        """
        return "OpenAI Compatible API"

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build message list.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Message list
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Dict:
        """
        Build request payload.

        Args:
            messages: Message list
            stream: Whether to enable streaming output

        Returns:
            Request payload dictionary
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if stream:
            payload["stream"] = True

        # Add provider-specific parameters
        self._add_provider_specific_params(payload)

        return payload

    def _add_provider_specific_params(self, payload: Dict) -> None:
        """
        Add provider-specific parameters (subclasses can override).

        Args:
            payload: Request payload dictionary
        """
        # Default implementation: don't add any specific parameters
        # Subclasses can override this method to add specific parameters

    def _get_litellm_model_name(self) -> str:
        """
        Get LiteLLM model name.

        Returns LiteLLM format model name based on provider type.

        Returns:
            LiteLLM model name
        """
        # Default returns original model name, subclasses can override
        return self.config.model_name

    def _get_litellm_api_key(self) -> Optional[str]:
        """
        Get LiteLLM API key.

        Returns:
            API key
        """
        return self.api_key

    def _get_litellm_api_base(self) -> Optional[str]:
        """
        Get LiteLLM API Base URL.

        Returns:
            API Base URL
        """
        return self.base_url

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client (for synchronous calls)."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    # Async methods (primary interface)
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text (async).

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Generated text

        Raises:
            ValueError: If API key not set
            Exception: If API call fails
        """
        if not self.api_key:
            raise ValueError(f"{self.provider_name} API key not set")

        messages = self._build_messages(prompt, system_prompt)
        payload = self._build_payload(messages, stream=False)

        # Build LiteLLM call parameters
        litellm_kwargs = {
            "model": self._get_litellm_model_name(),
            "messages": messages,
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "top_p": payload.get("top_p"),
            "api_key": self._get_litellm_api_key(),
            "timeout": self.config.timeout,
        }

        # Add API base URL (if provided)
        api_base = self._get_litellm_api_base()
        if api_base:
            litellm_kwargs["api_base"] = api_base

        # Add provider-specific parameters
        if "presence_penalty" in payload:
            litellm_kwargs["presence_penalty"] = payload["presence_penalty"]
        if "frequency_penalty" in payload:
            litellm_kwargs["frequency_penalty"] = payload["frequency_penalty"]
        if "response_format" in payload:
            litellm_kwargs["response_format"] = payload["response_format"]

        try:
            response = await litellm.acompletion(**litellm_kwargs)
            if not response or not response.choices or len(response.choices) == 0:
                raise LLMProviderError(
                    f"{self.provider_name} API returned error: no choices field in response"
                )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Catch all exceptions and log detailed information
            error_msg = str(e)
            if "timeout" in error_msg.lower() or isinstance(e, asyncio.TimeoutError):
                raise Exception(f"Request timeout (timeout: {self.config.timeout}s): {e}") from e
            logger.exception(f"{self.provider_name} API call failed: {e}")
            raise LLMProviderError(f"{self.provider_name} API call failed: {error_msg}") from e

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, int]]:
        """
        Stream generate text.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Yields:
            (text chunk, accumulated token count) tuple

        Raises:
            ValueError: If API key not set
            Exception: If API call fails
        """
        if not self.api_key:
            raise ValueError(f"{self.provider_name} API key not set")

        messages = self._build_messages(prompt, system_prompt)
        payload = self._build_payload(messages, stream=True)

        # Build LiteLLM call parameters
        litellm_kwargs = {
            "model": self._get_litellm_model_name(),
            "messages": messages,
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "top_p": payload.get("top_p"),
            "stream": True,
            "api_key": self._get_litellm_api_key(),
            "timeout": self.config.timeout,
        }

        # Add API base URL (if provided)
        api_base = self._get_litellm_api_base()
        if api_base:
            litellm_kwargs["api_base"] = api_base

        # Add provider-specific parameters
        if "presence_penalty" in payload:
            litellm_kwargs["presence_penalty"] = payload["presence_penalty"]
        if "frequency_penalty" in payload:
            litellm_kwargs["frequency_penalty"] = payload["frequency_penalty"]
        if "response_format" in payload:
            litellm_kwargs["response_format"] = payload["response_format"]

        accumulated_tokens = 0

        try:
            # LiteLLM uses acompletion with stream=True for async streaming calls
            # 使用 asyncio.wait_for 确保超时机制可靠
            # 给 litellm 的超时留一些缓冲时间（+5秒）
            timeout_seconds = self.config.timeout + 5 if self.config.timeout else 65
            response_stream = await asyncio.wait_for(
                litellm.acompletion(**litellm_kwargs),
                timeout=timeout_seconds
            )
            # 迭代响应流，添加超时检查
            iteration_start_time = time.time()
            async for chunk in response_stream:
                # 检查是否超时（在每次迭代时检查）
                current_time = time.time()
                elapsed = current_time - iteration_start_time
                if elapsed > timeout_seconds:
                    raise asyncio.TimeoutError(
                        f"Stream iteration timeout after {elapsed:.1f}s (timeout: {timeout_seconds}s)"
                    )
                if not chunk or not chunk.choices or len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content = delta.content
                    # Estimate new token count
                    new_tokens = self._estimate_tokens(content)
                    accumulated_tokens += new_tokens
                    yield (content, accumulated_tokens)
        except Exception as e:
            # Catch all exceptions and log detailed information
            error_msg = str(e)
            if "timeout" in error_msg.lower() or isinstance(e, asyncio.TimeoutError):
                raise Exception(f"Request timeout (timeout: {self.config.timeout}s): {e}") from e
            logger.exception(f"{self.provider_name} API call failed: {e}")
            raise LLMProviderError(f"{self.provider_name} API call failed: {error_msg}") from e

    # Synchronous methods (for compatibility with ask_llm)
    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """
        Call OpenAI-compatible API (synchronous).

        Uses OpenAI SDK for synchronous calls.

        Args:
            prompt: Single prompt text
            messages: List of message dictionaries
            temperature: Sampling temperature
            model: Model name override
            stream: Whether to stream response
            **kwargs: Additional parameters (max_tokens, top_p, return_reasoning, etc.)

        Returns:
            Response string, or ``(content, reasoning)`` when ``return_reasoning=True`` (non-stream),
            or a streaming generator when ``stream=True``. With ``stream=True`` and
            ``return_reasoning=True``, the generator yields ``(content_delta, reasoning_delta)``
            pairs; otherwise it yields content string chunks only.

        Raises:
            ValueError: If neither prompt nor messages provided
            RuntimeError: If API call fails
        """
        return_reasoning = bool(kwargs.pop("return_reasoning", False))

        # Validate inputs
        if messages:
            api_messages = messages
        elif prompt:
            api_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        model_name = model or self.config.model_name
        temp = temperature if temperature is not None else self.config.temperature

        # Build API parameters
        params: Dict = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temp,
            "stream": stream,
        }

        # Add optional parameters
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens

        # Apply any additional kwargs
        for key in ["max_tokens", "top_p", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                params[key] = kwargs[key]

        # Add provider-specific parameters
        payload = self._build_payload(api_messages, stream=stream)
        if "presence_penalty" in payload:
            params["presence_penalty"] = payload["presence_penalty"]
        if "frequency_penalty" in payload:
            params["frequency_penalty"] = payload["frequency_penalty"]
        if "response_format" in payload:
            params["response_format"] = payload["response_format"]

        # Build detailed parameter string for logging
        param_details = [
            f"provider={self.provider_name}",
            f"model={model_name}",
            f"temperature={temp}",
        ]
        if params.get("max_tokens") is not None:
            param_details.append(f"max_tokens={params['max_tokens']}")
        if params.get("top_p") is not None:
            param_details.append(f"top_p={params['top_p']}")
        if params.get("presence_penalty") is not None:
            param_details.append(f"presence_penalty={params['presence_penalty']}")
        if params.get("frequency_penalty") is not None:
            param_details.append(f"frequency_penalty={params['frequency_penalty']}")
        param_details.append(f"stream={stream}")

        logger.debug(f"Calling API with {', '.join(param_details)}")

        try:
            if stream:
                return self._stream_response(params, return_reasoning=return_reasoning)
            else:
                content, reasoning = self._complete_response(params)
                if return_reasoning:
                    return (content, reasoning)
                return content

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError("API authentication failed. Please check your API key.") from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RuntimeError("API rate limit exceeded. Please try again later.") from e
        except APIError as e:
            mt = params.get("max_tokens")
            logger.error(
                f"API error: {e}  (model={model_name!r}, max_tokens={mt})"
            )
            raise RuntimeError(
                f"API error: {e}  (model={model_name!r}, max_tokens={mt})"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"API call failed: {e}") from e

    def _complete_response(self, params: Dict) -> Tuple[str, Optional[str]]:
        """
        Get complete (non-streaming) response.

        Args:
            params: API parameters

        Returns:
            ``(content, reasoning)`` — reasoning is set for models that
            return ``reasoning_content`` (e.g. DeepSeek reasoner).
        """
        # Remove stream parameter for non-streaming call
        params = {k: v for k, v in params.items() if k != "stream"}

        response = self.client.chat.completions.create(**params)
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        reasoning: Optional[str] = None
        rc = getattr(msg, "reasoning_content", None)
        if isinstance(rc, str) and rc.strip():
            reasoning = rc.strip()
        else:
            try:
                dumped = msg.model_dump() if hasattr(msg, "model_dump") else {}
                if isinstance(dumped, dict):
                    rc2 = dumped.get("reasoning_content")
                    if isinstance(rc2, str) and rc2.strip():
                        reasoning = rc2.strip()
            except Exception:
                pass

        logger.debug(
            f"Received response: {len(content)} chars content"
            + (f", {len(reasoning)} chars reasoning" if reasoning else "")
        )
        return content, reasoning

    def _stream_response(self, params: Dict, *, return_reasoning: bool = False):
        """
        Stream response from API.

        Args:
            params: API parameters
            return_reasoning: If True, yield ``(content_delta, reasoning_delta)`` for
                models that stream ``reasoning_content`` (e.g. DeepSeek reasoner).
                If False, yield content-only strings (backward compatible).

        Yields:
            Text chunks, or ``(content_delta, reasoning_delta)`` when *return_reasoning*
            is True.
        """
        try:
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    if return_reasoning:
                        c = delta.content or ""
                        r = self._delta_reasoning_content(delta)
                        if c or r:
                            yield (c, r)
                    elif delta.content:
                        yield delta.content
        except GeneratorExit:
            # Handle generator cleanup
            logger.debug("Stream generator closed")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Stream failed: {e}") from e

    @staticmethod
    def _delta_reasoning_content(delta) -> str:
        """Extract reasoning stream piece from a chat completion delta (DeepSeek, etc.)."""
        rc = getattr(delta, "reasoning_content", None)
        if isinstance(rc, str) and rc:
            return rc
        try:
            if hasattr(delta, "model_dump"):
                dumped = delta.model_dump()
                if isinstance(dumped, dict):
                    rc2 = dumped.get("reasoning_content")
                    if isinstance(rc2, str) and rc2:
                        return rc2
        except Exception:
            pass
        return ""
