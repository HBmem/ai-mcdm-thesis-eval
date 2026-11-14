from __future__ import annotations

import os

from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlsplit

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

@dataclass(frozen=True)
class AIConfig:
    model: str
    base_url: str = "https://ollama.com/v1"
    api_key: str = field(default="", repr=False)
    timeout: float = 120
    max_retries: int = 2

    @classmethod
    def from_env(cls, env_file: str | Path | None = DEFAULT_ENV_FILE):
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        missing = [key for key in ("OLLAMA_API_KEY", "OLLAMA_MODEL") if not os.environ.get(key, "").strip()]
        if missing:
            raise ValueError("Missing required AI configuration: " + ", ".join(missing))
        return cls(model=os.environ["OLLAMA_MODEL"], api_key=os.environ["OLLAMA_API_KEY"],
                    base_url=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1"))

    def safe(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            # "prompt_version": PROMPT_VERSION,
            "schema_version": 1
        }

    def client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if not self.model.strip() or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid AI model or endpoint configuration.")
        if self.timeout <= 0 or not 0 <= self.max_retries <= 2:
            raise ValueError("AI timeout must be positive and retries must be between zero and two.")