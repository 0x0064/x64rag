from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from baml_py import ClientRegistry

from x64rag.common.errors import ConfigurationError

_MAX_RETRIES_LIMIT = 5

_CLIENT_DEFAULT = "Default"
_CLIENT_FALLBACK = "Fallback"
_CLIENT_ROUTER = "Router"


@dataclass
class LanguageModelClientConfig:
    provider: str
    model: str
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0


@dataclass
class LanguageModelConfig:
    client: LanguageModelClientConfig
    fallback: LanguageModelClientConfig | None = None
    max_retries: int = 3
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    boundary_api_key: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(f"Invalid strategy {self.strategy!r}, must be 'primary_only' or 'fallback'")
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES_LIMIT:
            raise ConfigurationError(f"max_retries must be 0-{_MAX_RETRIES_LIMIT}, got {self.max_retries}")
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback client")


def _retry_policy_name(max_retries: int) -> str | None:
    if max_retries == 0:
        return None
    return f"Retry{max_retries}"


def _build_client_options(client_config: LanguageModelClientConfig) -> dict:
    options = {
        "model": client_config.model,
        "temperature": client_config.temperature,
        "max_tokens": client_config.max_tokens,
    }
    if client_config.api_key:
        options["api_key"] = client_config.api_key
    return options


def build_registry(config: LanguageModelConfig) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(config.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=config.client.provider,
        options=_build_client_options(config.client),
        retry_policy=policy,
    )

    if config.strategy == "fallback" and config.fallback:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=config.fallback.provider,
            options=_build_client_options(config.fallback),
            retry_policy=policy,
        )
        registry.add_llm_client(
            _CLIENT_ROUTER,
            provider="fallback",
            options={"strategy": [_CLIENT_DEFAULT, _CLIENT_FALLBACK]},
        )
        registry.set_primary(_CLIENT_ROUTER)
    else:
        registry.set_primary(_CLIENT_DEFAULT)

    if config.boundary_api_key:
        os.environ["BOUNDARY_API_KEY"] = config.boundary_api_key

    return registry
