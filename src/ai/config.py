# src/ai/config.py
from __future__ import annotations

import math
import os

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit
from dotenv import load_dotenv


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class AIProvider(StrEnum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    base_url: str
    api_key_env: str


PROVIDERS = {
    AIProvider.OPENROUTER: ProviderProfile(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    AIProvider.OLLAMA: ProviderProfile(
        base_url="https://ollama.com/v1",
        api_key_env="OLLAMA_API_KEY",
    ),
    AIProvider.OPENAI: ProviderProfile(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
}


def _read_positive_float(
    name: str,
    default: float,
) -> float:
    raw = os.environ.get(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number.") from None

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be a finite positive number."
        )

    return value


def _read_int_between(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.environ.get(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer."
        ) from None

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between "
            f"{minimum} and {maximum}."
        )

    return value


@dataclass(frozen=True, slots=True)
class AIConfig:
    provider: AIProvider
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 120.0
    max_retries: int = 2
    retry_base_seconds: float = 1.0
    retry_max_wait_seconds: float = 90.0
    app_url: str | None = None
    app_name: str | None = None

    @property
    def client_headers(self) -> dict[str, str]:
        if self.provider is not AIProvider.OPENROUTER:
            return {}

        headers = {}
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = DEFAULT_ENV_FILE,
    ) -> Self:
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        provider_name = os.environ.get("AI_PROVIDER", "").strip().lower()

        try:
            provider = AIProvider(provider_name)
        except ValueError:
            supported = ", ".join(item.value for item in AIProvider)
            if not provider_name:
                raise ValueError(
                    "Missing required AI configuration: AI_PROVIDER"
                ) from None
            raise ValueError(
                f"Unsupported AI_PROVIDER {provider_name!r}; "
                f"expected one of: {supported}."
            ) from None

        profile = PROVIDERS[provider]
        values = {
            "provider": provider,
            "model": os.environ.get("AI_MODEL", "").strip(),
            "api_key": os.environ.get(profile.api_key_env, "").strip(),
            "base_url": os.environ.get(
                "AI_BASE_URL",
                profile.base_url,
            ).strip(),
            "app_url": os.environ.get("AI_APP_URL", "").strip() or None,
            "app_name": os.environ.get("AI_APP_NAME", "").strip() or None,
            "timeout": _read_positive_float(
                "AI_TIMEOUT_SECONDS",
                120.0
            ),
            "max_retries": _read_int_between(
                "AI_MAX_RETRIES",
                2,
                minimum=0,
                maximum=2,
            ),
            "retry_base_seconds": _read_positive_float(
                "AI_RETRY_BASE_SECONDS",
                1.0,
            ),
            "retry_max_wait_seconds": _read_positive_float(
                "AI_RETRY_MAX_WAIT_SECONDS",
                90.0,
            ),
        }

        missing = [
            name
            for name, value in (
                ("AI_MODEL", values["model"]),
                (profile.api_key_env, values["api_key"]),
            )
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing required AI configuration: "
                + ", ".join(missing)
            )

        return cls(**values)

    def audit_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_wait_seconds": self.retry_max_wait_seconds,
        }

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)

        if not isinstance(self.provider, AIProvider):
            raise ValueError("AI provider must be an AIProvider value.")

        if not self.model.strip():
            raise ValueError("AI model must not be empty.")

        if not self.api_key.strip():
            raise ValueError("AI API key must not be empty.")

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid AI endpoint configuration.")

        if not 0 <= self.max_retries <= 2:
            raise ValueError(
                "AI retries must be between zero and two."
            )

        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError(
                "AI timeout must be a finite positive number."
            )

        if (
            not math.isfinite(self.retry_base_seconds)
            or self.retry_base_seconds <= 0
        ):
            raise ValueError(
                "AI retry base seconds must be a finite "
                "positive number."
            )

        if (
            not math.isfinite(self.retry_max_wait_seconds)
            or self.retry_max_wait_seconds <= 0
        ):
            raise ValueError(
                "AI retry maximum wait must be a finite "
                "positive number."
            )
