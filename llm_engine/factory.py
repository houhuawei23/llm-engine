"""
Factory functions for creating LLM providers from various configuration formats.

Provides utilities to create providers from different configuration sources,
making it easy to integrate llm_engine with other applications.
"""

from typing import Any, Dict, List, Optional, Protocol, Union

from llm_engine.config import LLMConfig, LLMProvider
from llm_engine.engine import CustomProvider, DeepSeekProvider, OpenAIProvider
from llm_engine.providers.base import BaseLLMProvider


class ProviderConfigProtocol(Protocol):
    """Protocol for provider configuration objects."""

    api_provider: str
    api_key: str
    api_base: str
    models: List[str]
    api_temperature: float
    api_top_p: Optional[float]
    max_tokens: Optional[int]
    timeout: float


def create_provider_from_config(
    config: ProviderConfigProtocol | Dict[str, Any],
    default_model: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Create an LLM provider from a configuration object or dictionary.

    This function provides a unified way to create providers from various
    configuration formats, making it easy to integrate llm_engine with
    other applications like ask_llm.

    Args:
        config: Provider configuration object (with attributes) or dictionary
        default_model: Default model name (if None, uses first model from models list)

    Returns:
        LLM provider instance (DeepSeekProvider, OpenAIProvider, or CustomProvider)

    Raises:
        ValueError: If configuration is invalid or provider is unsupported
        ImportError: If required dependencies are missing

    Example:
        >>> from llm_engine.factory import create_provider_from_config
        >>>
        >>> # From a configuration object
        >>> provider = create_provider_from_config(config_obj, default_model="deepseek-chat")
        >>>
        >>> # From a dictionary
        >>> config_dict = {
        ...     "api_provider": "deepseek",
        ...     "api_key": "sk-...",
        ...     "api_base": "https://api.deepseek.com/v1",
        ...     "models": ["deepseek-chat"],
        ...     "api_temperature": 0.7,
        ...     "timeout": 60.0,
        ... }
        >>> provider = create_provider_from_config(config_dict)
    """
    # Extract configuration values
    if isinstance(config, dict):
        api_provider = config.get("api_provider", "").lower()
        api_key = config.get("api_key", "")
        api_base = config.get("api_base", "")
        models = config.get("models", [])
        api_temperature = config.get("api_temperature", 0.7)
        api_top_p = config.get("api_top_p")
        max_tokens = config.get("max_tokens")
        timeout = config.get("timeout", 60.0)
    else:
        # Assume it's an object with attributes
        api_provider = getattr(config, "api_provider", "").lower()
        api_key = getattr(config, "api_key", "")
        api_base = getattr(config, "api_base", "")
        models = getattr(config, "models", [])
        api_temperature = getattr(config, "api_temperature", 0.7)
        api_top_p = getattr(config, "api_top_p", None)
        max_tokens = getattr(config, "max_tokens", None)
        timeout = getattr(config, "timeout", 60.0)

    # Validate required fields
    if not api_key or api_key in ("your-api-key-here", "placeholder", ""):
        raise ValueError("API key is required and cannot be a placeholder")

    if not api_base:
        raise ValueError("API base URL is required")

    # Determine default model
    model_name = default_model or (models[0] if models else None)
    if not model_name:
        raise ValueError(f"No default model specified and provider '{api_provider}' has no models")

    # Map provider name to LLMProvider enum
    provider_map: Dict[str, LLMProvider] = {
        "deepseek": LLMProvider.DEEPSEEK,
        "openai": LLMProvider.OPENAI,
        "qwen": LLMProvider.CUSTOM,  # Qwen uses custom provider
    }
    provider_enum = provider_map.get(api_provider, LLMProvider.CUSTOM)

    # Create LLMConfig
    llm_config = LLMConfig(
        provider=provider_enum,
        model_name=model_name,
        api_key=api_key,
        base_url=api_base,
        temperature=api_temperature,
        max_tokens=max_tokens or 2000,
        top_p=api_top_p or 1.0,
        timeout=int(timeout),
    )

    # Create and return the appropriate provider
    if api_provider == "deepseek":
        return DeepSeekProvider(llm_config)
    elif api_provider == "openai":
        return OpenAIProvider(llm_config)
    else:
        # Use CustomProvider for other providers (e.g., qwen)
        return CustomProvider(llm_config)


def create_provider_adapter(
    config: ProviderConfigProtocol | Dict[str, Any],
    default_model: Optional[str] = None,
) -> "ProviderAdapter":
    """
    Create a provider adapter with a simplified interface.

    This adapter wraps the llm_engine provider and provides additional
    convenience methods and properties for easier integration.

    Args:
        config: Provider configuration object or dictionary
        default_model: Default model name

    Returns:
        ProviderAdapter instance with enhanced interface

    Example:
        >>> from llm_engine.factory import create_provider_adapter
        >>> adapter = create_provider_adapter(config_obj)
        >>> response = adapter.call(prompt="Hello")
        >>> print(adapter.name, adapter.default_model)
    """
    provider = create_provider_from_config(config, default_model=default_model)
    return ProviderAdapter(provider, config, default_model)


class ProviderAdapter:
    """
    Adapter wrapper for llm_engine providers with enhanced interface.

    Provides a simplified interface compatible with various application
    frameworks while maintaining full access to the underlying provider.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: ProviderConfigProtocol | Dict[str, Any],
        default_model: Optional[str] = None,
    ):
        """
        Initialize the adapter.

        Args:
            provider: The underlying llm_engine provider instance
            config: Original configuration object or dictionary
            default_model: Default model name
        """
        self._provider = provider
        self._config = config

        # Extract provider name and models
        if isinstance(config, dict):
            self._provider_name = config.get("api_provider", "unknown")
            self._models = config.get("models", [])
        else:
            self._provider_name = getattr(config, "api_provider", "unknown")
            self._models = getattr(config, "models", [])

        self._default_model = default_model or (
            self._models[0] if self._models else provider.config.model_name
        )

    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Call the LLM API.

        Args:
            prompt: Single prompt text (alternative to messages)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides config)
            model: Model name (overrides config)
            stream: If True, return a generator for streaming responses
            **kwargs: Additional API parameters

        Returns:
            Response string, or generator if streaming
        """
        # Update model if override provided
        if model:
            self._provider.config.model_name = model
        if temperature is not None:
            self._provider.config.temperature = temperature
        # Update max_tokens if provided
        if "max_tokens" in kwargs:
            self._provider.config.max_tokens = kwargs["max_tokens"]

        return self._provider.call(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            model=model,
            stream=stream,
            **kwargs,
        )

    def test_connection(self, test_message: str = "Hello") -> tuple[bool, str, float]:
        """
        Test API connection.

        Args:
            test_message: Test message to send

        Returns:
            Tuple of (success, message, latency_seconds)
        """
        import time

        start = time.time()
        try:
            response = self.call(prompt=test_message, max_tokens=10, temperature=0.0)
            latency = time.time() - start
            return True, f"Response: {response[:50]}...", latency
        except Exception as e:
            latency = time.time() - start
            return False, str(e), latency

    @property
    def name(self) -> str:
        """Get provider name."""
        return self._provider_name

    @property
    def default_model(self) -> str:
        """Get default model name."""
        return self._default_model

    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        return self._models if self._models else [self._default_model]

    @property
    def config(self):
        """Get original configuration object."""
        return self._config

    @property
    def provider(self) -> BaseLLMProvider:
        """Get underlying llm_engine provider instance."""
        return self._provider
