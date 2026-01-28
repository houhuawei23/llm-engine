"""
Configuration models for LLM Engine.

Defines data structures for LLM configuration, provider settings, and model information.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """LLM provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    KIMI = "kimi"


class LLMConfig(BaseModel):
    """
    LLM configuration model.

    Contains provider, API key, request parameters, and other configuration.
    """

    provider: LLMProvider = Field(default=LLMProvider.DEEPSEEK, description="LLM provider")
    model_name: str = Field(default="deepseek-chat", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(default=2000, ge=1, description="Maximum tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty parameter"
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty parameter"
    )
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry count")
    api_keys: List[str] = Field(default_factory=list, description="Multiple API keys for rotation")

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate API key.

        Args:
            v: API key value

        Returns:
            Validated API key
        """
        if v and v.startswith("${") and v.endswith("}"):
            # Environment variable reference, don't parse here
            return v
        return v

    def get_api_key(self) -> Optional[str]:
        """
        Get effective API key.

        Prioritizes first key from api_keys list, otherwise uses api_key.

        Returns:
            API key string
        """
        if self.api_keys:
            return self.api_keys[0]
        return self.api_key
