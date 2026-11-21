# src/ai/component.py
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from openai import OpenAI

from src.ai.backend import (
    OpenAICompatibleBackend,
    StructuredAIBackend,
)
from src.ai.config import AIConfig
from src.ai.preflight import PreflightRecord, check_model_access


@dataclass(frozen=True)
class AIComponent:
    config: AIConfig
    backend: StructuredAIBackend
    preflight: PreflightRecord


def create_client(config: AIConfig) -> OpenAI:
    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        # Retries will be managed and audited by the application.
        max_retries=0,
        default_headers=config.client_headers or None,
    )


@contextmanager
def build_ai_component(
    config: AIConfig,
) -> Generator[AIComponent]:
    client = create_client(config)

    try:
        preflight = check_model_access(config, client)

        backend = OpenAICompatibleBackend(
            config=config,
            client=client,
        )

        yield AIComponent(
            config=config,
            backend=backend,
            preflight=preflight,
        )
    finally:
        client.close()
