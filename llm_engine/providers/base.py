"""
Base provider class for LLM API providers.

Supports both synchronous and asynchronous operations.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import AsyncIterator, Generator, List, Optional, Tuple, Union

from loguru import logger

from llm_engine.config import LLMConfig
from llm_engine.config_loader import get_model_info


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All concrete LLM providers should inherit from this class and implement abstract methods.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM provider.

        Args:
            config: LLM configuration object
        """
        self.config = config
        self.api_key = self._get_api_key()
        self.base_url = config.base_url or self._get_default_base_url()
        # Load token_per_character configuration
        self._token_per_char_config = self._load_token_per_character_config()

    def _get_api_key(self) -> Optional[str]:
        """
        Get API key.

        Prioritizes environment variable, then uses key from config.

        Returns:
            API key
        """
        # Get from environment variable
        env_key = self._get_env_api_key()
        if env_key:
            return env_key

        # Get from config
        return self.config.get_api_key()

    @abstractmethod
    def _get_env_api_key(self) -> Optional[str]:
        """
        Get API key from environment variable (implemented by subclasses).

        Returns:
            API key
        """

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """
        Get default API base URL (implemented by subclasses).

        Returns:
            Base URL
        """

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text (async).

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Generated text
        """

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, int]]:
        """
        Stream generate text (default implementation: fallback to non-streaming).

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Yields:
            (text chunk, accumulated token count) tuple
        """
        # Default implementation: call non-streaming method and return complete result
        result = await self.generate(prompt, system_prompt)
        # Simple token estimation
        estimated_tokens = self._estimate_tokens(result)
        yield (result, estimated_tokens)

    def _load_token_per_character_config(self) -> Optional[dict]:
        """
        Load token_per_character configuration from providers.yml.

        Returns:
            Dictionary with english and chinese keys, or None if not found
        """
        try:
            # Get provider name from config.provider enum
            provider_name = self.config.provider.value.lower()
            model_name = self.config.model_name
            
            # Load model info from providers.yml
            model_info = get_model_info(provider_name, model_name)
            if model_info:
                token_config = model_info.get("token_per_character", {})
                if token_config:
                    return {
                        "english": token_config.get("english", 0.3),
                        "chinese": token_config.get("chinese", 0.6),
                    }
        except Exception as e:
            logger.debug(f"Failed to load token_per_character config: {e}")
        return None

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Priority:
        - If config exists: use token_per_character values from config
        - If config doesn't exist: use default heuristic
          (Chinese ~1.5 chars/token, other ~4 chars/token)

        Args:
            text: Text content

        Returns:
            Estimated token count
        """
        # Count Chinese characters (CJK unified ideographs)
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # Count other characters
        other_chars = len(text) - chinese_chars

        # If config exists, use config values
        if self._token_per_char_config:
            chinese_token_ratio = self._token_per_char_config.get("chinese", 0.6)
            english_token_ratio = self._token_per_char_config.get("english", 0.3)
            # token_per_character represents tokens per character
            # e.g., 0.6 means 1 Chinese char ≈ 0.6 token, i.e., 1 token ≈ 1.67 chars
            estimated = int(chinese_chars * chinese_token_ratio + other_chars * english_token_ratio)
        else:
            # Default heuristic: Chinese ~1.5 chars/token, other ~4 chars/token
            estimated = int(chinese_chars / 1.5 + other_chars / 4)

        return max(estimated, 1)  # At least return 1

    async def generate_with_retry(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text with retry logic.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Generated text

        Raises:
            Exception: If all retries fail
        """
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.generate(prompt, system_prompt)
            except (asyncio.TimeoutError, ConnectionError) as e:
                # Network errors and timeouts should retry
                last_error = e
                if attempt < self.config.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    error_msg = str(e)
                    if isinstance(e, asyncio.TimeoutError):
                        error_msg = f"Timeout error: {error_msg}"
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {error_msg}"
                    )
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries failed: {e}")
            except Exception as e:
                # Other errors (API errors, auth errors, etc.) don't retry, raise directly
                error_msg = str(e)
                if "API error" in error_msg.lower() or "key" in error_msg.lower() or "auth" in error_msg.lower():
                    logger.error(f"API error (no retry): {error_msg}")
                    raise
                # Other unknown errors also retry
                last_error = e
                if attempt < self.config.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {error_msg}"
                    )
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries failed: {e}")

        raise last_error or Exception("Generation failed")

    # Synchronous methods (for compatibility with ask_llm)
    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Call LLM API (synchronous).

        This is a compatibility method for synchronous code. It wraps async methods.

        Args:
            prompt: Single prompt text (alternative to messages)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides config)
            model: Model name (overrides config)
            stream: If True, return a generator for streaming responses
            **kwargs: Additional API parameters

        Returns:
            Response string, or generator if streaming

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

        # Extract system prompt if present
        system_prompt = None
        user_prompt = prompt
        if messages:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if system_msgs:
                system_prompt = system_msgs[-1].get("content")
            if user_msgs:
                user_prompt = user_msgs[-1].get("content")
            elif not system_msgs:
                # If only system message, use it as prompt
                user_prompt = system_msgs[0].get("content") if system_msgs else prompt

        if not user_prompt:
            raise ValueError("No user prompt found in messages")

        # Override config if needed
        temp = temperature if temperature is not None else self.config.temperature
        model_name = model or self.config.model_name

        # Create a new event loop or use existing one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if stream:
            # Return generator for streaming
            return self._sync_stream_generate(user_prompt, system_prompt, temp, model_name, loop)
        else:
            # Return complete response
            return loop.run_until_complete(
                self.generate(user_prompt, system_prompt)
            )

    def _sync_stream_generate(
        self, prompt: str, system_prompt: Optional[str], temperature: float, model: str, loop
    ) -> Generator[str, None, None]:
        """
        Synchronous wrapper for streaming generation.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature
            model: Model name
            loop: Event loop

        Yields:
            Text chunks
        """
        async def _async_gen():
            async for chunk, _ in self.generate_stream(prompt, system_prompt):
                yield chunk

        gen = _async_gen()
        while True:
            try:
                chunk = loop.run_until_complete(gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
