"""
LLM Engine unified interface.

Automatically selects appropriate provider based on configuration and provides unified calling interface.
"""

import os
from typing import AsyncIterator, Optional, Tuple

from loguru import logger
from openai import OpenAI

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.config_loader import get_model_info
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


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic API provider (Claude models)

    Note: For Kimi Code API, this provider uses OpenAI-compatible format
    since Kimi Code API is compatible with OpenAI's chat completions API.
    """

    def _get_env_api_key(self) -> Optional[str]:
        """Get Anthropic API key from environment variable"""
        # For Kimi Code API, use KIMI_CODE_API_KEY
        base_url = self.config.base_url or self._get_default_base_url()
        if base_url and "kimi.com" in base_url:
            return os.getenv("KIMI_CODE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("ANTHROPIC_API_KEY")

    def _is_kimi_code(self) -> bool:
        """Check if this is Kimi Code API endpoint."""
        return "kimi.com" in (self.config.base_url or "")

    @property
    def client(self):
        """Get or create OpenAI client with custom headers for Kimi Code API."""
        if self._client is None:
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "timeout": self.config.timeout,
            }
            # Add User-Agent header for Kimi Code API
            if self._is_kimi_code():
                client_kwargs["default_headers"] = {"User-Agent": "claude-code/1.0.0"}
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _get_default_base_url(self) -> str:
        """Get Anthropic default API URL"""
        return "https://api.anthropic.com/v1"

    def _get_provider_name(self) -> str:
        """Get provider name"""
        return "Anthropic"

    def _get_litellm_model_name(self) -> str:
        """Get LiteLLM model name"""
        # For Kimi Code API, use openai/ prefix (OpenAI-compatible)
        if "kimi.com" in self.base_url:
            return f"openai/{self.config.model_name}"
        # LiteLLM format: anthropic/model_name
        return f"anthropic/{self.config.model_name}"

    def _get_litellm_api_base(self) -> Optional[str]:
        """Get LiteLLM API Base URL."""
        return self.base_url


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


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama local model provider (via LiteLLM ``ollama/`` routing)."""

    def _get_env_api_key(self) -> Optional[str]:
        return None

    def _requires_api_key(self) -> bool:
        return False

    def _get_default_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def _get_provider_name(self) -> str:
        return "Ollama"

    def _get_litellm_model_name(self) -> str:
        return f"ollama/{self.config.model_name}"

    def _get_litellm_api_key(self) -> Optional[str]:
        return None

    @property
    def client(self) -> OpenAI:
        """Sync OpenAI SDK client; Ollama ignores auth so a placeholder key is used."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "ollama",
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client


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
            LLMProvider.ANTHROPIC: AnthropicProvider,
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
