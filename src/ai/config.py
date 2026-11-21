# src/ai/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit
from enum import StrEnum

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


@dataclass(frozen=True, slots=True)
class AIConfig:
    provider: AIProvider
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 120.0
    max_retries: int = 2
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

        values = {
            "model": os.environ.get("OLLAMA_MODEL", "").strip(),
            "api_key": os.environ.get("OLLAMA_API_KEY", "").strip(),
            "base_url": os.environ.get(
                "OLLAMA_BASE_URL",
                "https://ollama.com/v1",
            ).strip(),
        }

        missing = [
            name
            for name, value in (
                ("OLLAMA_MODEL", values["model"]),
                ("OLLAMA_API_KEY", values["api_key"]),
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
            "schema_version": 1,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)

        if not self.model.strip():
            raise ValueError("AI model must not be empty.")

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid AI endpoint configuration.")

        if self.timeout <= 0:
            raise ValueError("AI timeout must be positive.")

        if not 0 <= self.max_retries <= 2:
            raise ValueError(
                "AI retries must be between zero and two."
            )
        