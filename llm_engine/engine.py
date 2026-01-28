"""
LLM Engine unified interface.

Automatically selects appropriate provider based on configuration and provides unified calling interface.
"""

import os
from typing import AsyncIterator, Optional, Tuple

from loguru import logger

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.config_loader import get_model_info
from llm_engine.exceptions import LLMProviderError
from llm_engine.providers.base import BaseLLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider"""

    def _get_env_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment variable"""
        return os.getenv("OPENAI_API_KEY")

    def _get_default_base_url(self) -> str:
        """Get OpenAI default API URL"""
        return "https://api.openai.com/v1"

    def _get_provider_name(self) -> str:
        """Get provider name"""
        return "OpenAI"

    def _get_litellm_model_name(self) -> str:
        """Get LiteLLM model name"""
        # LiteLLM format: openai/model_name
        return f"openai/{self.config.model_name}"

    def _add_provider_specific_params(self, payload: dict) -> None:
        """
        Add OpenAI-specific parameters

        Args:
            payload: Request payload dictionary
        """
        # OpenAI supports presence_penalty and frequency_penalty
        if hasattr(self.config, "presence_penalty"):
            payload["presence_penalty"] = self.config.presence_penalty
        if hasattr(self.config, "frequency_penalty"):
            payload["frequency_penalty"] = self.config.frequency_penalty


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider (OpenAI compatible)"""

    def __init__(self, config: LLMConfig):
        """
        Initialize DeepSeek provider

        Args:
            config: LLM configuration object
        """
        super().__init__(config)
        # Load JSON output configuration
        self._json_output_enabled = self._load_json_output_config()

    def _load_json_output_config(self) -> bool:
        """
        Load functions.json_output configuration from providers.yml

        Returns:
            True if configured as true, otherwise False
        """
        try:
            model_info = get_model_info("deepseek", self.config.model_name)
            if model_info:
                functions = model_info.get("functions", {})
                return functions.get("json_output", False)
        except Exception as e:
            logger.debug(f"Failed to load json_output config: {e}")
        return False

    def _get_env_api_key(self) -> Optional[str]:
        """Get DeepSeek API key from environment variable"""
        return os.getenv("DEEPSEEK_API_KEY")

    def _get_default_base_url(self) -> str:
        """Get DeepSeek default API URL"""
        return "https://api.deepseek.com/v1"

    def _get_provider_name(self) -> str:
        """Get provider name"""
        return "DeepSeek"

    def _get_litellm_model_name(self) -> str:
        """Get LiteLLM model name"""
        # LiteLLM format: deepseek/model_name
        return f"deepseek/{self.config.model_name}"

    def _add_provider_specific_params(self, payload: dict) -> None:
        """
        Add DeepSeek-specific parameters

        Args:
            payload: Request payload dictionary
        """
        # Only add response_format if:
        # 1. JSON output is enabled in config (for ai-anki-cards)
        # 2. AND the prompt/messages contain "json" keyword (required by DeepSeek API)
        if self._json_output_enabled:
            # Check if any message contains "json" keyword (case-insensitive)
            messages = payload.get("messages", [])
            has_json_keyword = False
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str) and "json" in content.lower():
                    has_json_keyword = True
                    break

            # Only add response_format if json keyword is found
            # This ensures we only use JSON mode when explicitly requested (e.g., in ai-anki-cards)
            # and satisfies DeepSeek API requirement that prompt must contain "json"
            if has_json_keyword:
                payload["response_format"] = {"type": "json_object"}


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider"""

    def _get_env_api_key(self) -> Optional[str]:
        """Ollama doesn't need API key"""
        return None

    def _get_default_base_url(self) -> str:
        """Get Ollama default API URL"""
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama API (via LiteLLM)"""
        import litellm

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # LiteLLM format: ollama/model_name
        model_name = f"ollama/{self.config.model_name}"

        try:
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                api_base=self.base_url,
                timeout=self.config.timeout,
            )
            if not response or not response.choices or len(response.choices) == 0:
                raise LLMProviderError("Ollama API returned error: no choices field in response")
            return response.choices[0].message.content or ""
        except Exception as e:
            error_msg = str(e)
            logger.exception(f"Ollama API call failed: {e}")
            raise LLMProviderError(f"Ollama API call failed: {error_msg}") from e

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, int]]:
        """Stream generate text using Ollama API (via LiteLLM)"""
        import litellm

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # LiteLLM format: ollama/model_name
        model_name = f"ollama/{self.config.model_name}"

        accumulated_tokens = 0

        try:
            # LiteLLM uses acompletion with stream=True for async streaming calls
            response_stream = await litellm.acompletion(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
                api_base=self.base_url,
                timeout=self.config.timeout,
            )
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
            error_msg = str(e)
            logger.exception(f"Ollama API call failed: {e}")
            raise LLMProviderError(f"Ollama API call failed: {error_msg}") from e


class CustomProvider(OpenAICompatibleProvider):
    """Custom OpenAI-compatible API provider"""

    def _get_env_api_key(self) -> Optional[str]:
        """Get custom API key from environment variable"""
        return os.getenv("CUSTOM_API_KEY")

    def _get_default_base_url(self) -> str:
        """Get custom API URL"""
        return os.getenv("CUSTOM_API_BASE_URL", "https://api.example.com/v1")

    def _get_provider_name(self) -> str:
        """Get provider name"""
        return "Custom API"

    def _get_litellm_model_name(self) -> str:
        """Get LiteLLM model name"""
        # For custom API, use openai/ prefix to indicate OpenAI-compatible format
        return f"openai/{self.config.model_name}"


class LLMEngine:
    """
    LLM Engine unified interface

    Automatically selects appropriate provider based on configuration and provides unified calling interface.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM engine

        Args:
            config: LLM configuration object
        """
        self.config = config
        self.provider = self._create_provider()

    def _create_provider(self) -> BaseLLMProvider:
        """
        Create LLM provider instance

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_map = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.DEEPSEEK: DeepSeekProvider,
            LLMProvider.OLLAMA: OllamaProvider,
            LLMProvider.CUSTOM: CustomProvider,
        }

        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        return provider_class(self.config)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Generated text
        """
        return await self.provider.generate_with_retry(prompt, system_prompt)

    async def stream_generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, int]]:
        """
        Stream generate text

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Yields:
            (text chunk, accumulated token count) tuple
        """
        async for chunk in self.provider.generate_stream(prompt, system_prompt):
            yield chunk
