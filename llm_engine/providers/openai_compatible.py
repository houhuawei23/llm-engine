"""
OpenAI Compatible API provider base class.

Contains common logic for OpenAI-compatible APIs, inherited by OpenAIProvider,
DeepSeekProvider, CustomProvider, etc.
"""

import asyncio
from typing import AsyncIterator, Dict, List, Optional, Tuple

import litellm
from loguru import logger
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

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
                raise Exception(
                    f"Request timeout (timeout: {self.config.timeout}s): {e}"
                ) from e
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
            response_stream = await litellm.acompletion(**litellm_kwargs)
            async for chunk in response_stream:
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
                raise Exception(
                    f"Request timeout (timeout: {self.config.timeout}s): {e}"
                ) from e
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
        **kwargs
    ) -> str:
        """
        Call OpenAI-compatible API (synchronous).

        Uses OpenAI SDK for synchronous calls.

        Args:
            prompt: Single prompt text
            messages: List of message dictionaries
            temperature: Sampling temperature
            model: Model name override
            stream: Whether to stream response
            **kwargs: Additional parameters (max_tokens, top_p, etc.)

        Returns:
            Response string or streaming generator

        Raises:
            ValueError: If neither prompt nor messages provided
            RuntimeError: If API call fails
        """
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

        logger.debug(
            f"Calling {self.provider_name} API with model={model_name}, "
            f"temperature={temp}, stream={stream}"
        )

        try:
            if stream:
                return self._stream_response(params)
            else:
                return self._complete_response(params)

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError(
                f"API authentication failed. Please check your API key."
            ) from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RuntimeError(
                f"API rate limit exceeded. Please try again later."
            ) from e
        except APIError as e:
            logger.error(f"API error: {e}")
            raise RuntimeError(f"API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"API call failed: {e}") from e

    def _complete_response(self, params: Dict) -> str:
        """
        Get complete (non-streaming) response.

        Args:
            params: API parameters

        Returns:
            Response text
        """
        # Remove stream parameter for non-streaming call
        params = {k: v for k, v in params.items() if k != "stream"}

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        logger.debug(f"Received response: {len(content)} characters")
        return content.strip()

    def _stream_response(
        self,
        params: Dict
    ):
        """
        Stream response from API.

        Args:
            params: API parameters

        Yields:
            Text chunks from the streaming response
        """
        try:
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except GeneratorExit:
            # Handle generator cleanup
            logger.debug("Stream generator closed")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise RuntimeError(f"Stream failed: {e}") from e
