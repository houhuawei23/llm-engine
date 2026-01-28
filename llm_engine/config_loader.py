"""
Configuration loader for LLM Engine.

Loads configuration from providers.yml file with environment variable resolution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger

from llm_engine.config import LLMConfig, LLMProvider


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references.

    Supports ${VAR_NAME} format environment variable references.

    Args:
        value: Value that may contain environment variable references

    Returns:
        Resolved value
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} format
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        if matches:
            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value:
                    value = value.replace(f"${{{var_name}}}", env_value)
                else:
                    logger.warning(f"Environment variable {var_name} not set")

        return value
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    else:
        return value


def load_providers_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load providers configuration from YAML file.

    Args:
        config_path: Path to providers.yml file. If None, searches default locations.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    if config_path is None:
        # Search default locations
        default_paths = [
            Path("providers.yml"),
            Path(__file__).parent.parent / "providers.yml",
            Path.home() / ".config" / "llm-engine" / "providers.yml",
        ]

        for path in default_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError(
                f"providers.yml not found. Searched: {[str(p) for p in default_paths]}"
            )

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        return {}

    # Resolve environment variable references
    config_dict = resolve_env_vars(config_dict)

    return config_dict


def create_llm_config_from_provider(
    provider_name: str,
    model_name: Optional[str] = None,
    config_path: Optional[Path] = None,
    **overrides: Any,
) -> LLMConfig:
    """
    Create LLMConfig from provider configuration.

    Args:
        provider_name: Provider name (e.g., "deepseek", "openai")
        model_name: Model name. If None, uses default_model from config.
        config_path: Path to providers.yml. If None, searches default locations.
        **overrides: Additional configuration overrides

    Returns:
        LLMConfig instance

    Raises:
        ValueError: If provider not found or configuration invalid
    """
    config_dict = load_providers_config(config_path)
    providers = config_dict.get("providers", {})

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Provider '{provider_name}' not found. Available providers: {available}")

    provider_config = providers[provider_name]

    # Map provider name to LLMProvider enum
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "deepseek": LLMProvider.DEEPSEEK,
        "ollama": LLMProvider.OLLAMA,
        "kimi": LLMProvider.KIMI,
        "custom": LLMProvider.CUSTOM,
    }
    provider_enum = provider_map.get(provider_name.lower(), LLMProvider.CUSTOM)

    # Get model name
    if model_name is None:
        model_name = provider_config.get("default_model")
        if not model_name and provider_config.get("models"):
            # Use first model if default_model not specified
            first_model = provider_config["models"][0]
            if isinstance(first_model, dict):
                model_name = first_model.get("name", first_model)
            else:
                model_name = first_model

    if not model_name:
        raise ValueError(f"No model specified for provider '{provider_name}'")

    # Build LLMConfig
    llm_config_dict = {
        "provider": provider_enum,
        "model_name": model_name,
        "api_key": provider_config.get("api_key"),
        "base_url": provider_config.get("base_url"),
        **overrides,
    }

    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            llm_config_dict[key] = value

    return LLMConfig(**llm_config_dict)


def get_model_info(
    provider_name: str, model_name: str, config_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Get model-specific information from configuration.

    Args:
        provider_name: Provider name
        model_name: Model name
        config_path: Path to providers.yml

    Returns:
        Model information dictionary, or None if not found
    """
    config_dict = load_providers_config(config_path)
    providers = config_dict.get("providers", {})

    if provider_name not in providers:
        return None

    provider_config = providers[provider_name]
    models = provider_config.get("models", [])

    for model in models:
        if isinstance(model, dict):
            if model.get("name") == model_name:
                return model
        elif model == model_name:
            # Simple string match
            return {}

    return None
