# src/ai/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from dotenv import load_dotenv


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True, slots=True)
class AIConfig:
    model: str
    base_url: str = "https://ollama.com/v1"
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0

    # Application-level retries, not OpenAI SDK retries.
    max_retries: int = 2

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
        