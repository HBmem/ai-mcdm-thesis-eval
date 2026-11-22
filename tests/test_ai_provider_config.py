from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from src.ai.backend import OpenAICompatibleBackend
from src.ai.component import create_client
from src.ai.config import AIConfig, AIProvider
from src.common.privacy import sanitize


class Answer(BaseModel):
    answer: str


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="returned-model",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"answer":"ok"}',
                        refusal=None,
                    ),
                )
            ],
        )


def config(provider: AIProvider) -> AIConfig:
    return AIConfig(
        provider=provider,
        model="provider/model",
        base_url="https://example.test/v1",
        api_key="test-secret",
    )


class AIConfigTests(unittest.TestCase):
    def test_openrouter_environment_resolution_and_audit(self):
        environment = {
            "AI_PROVIDER": " OpenRouter ",
            "AI_MODEL": "author/model",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "AI_APP_URL": "https://example.test/app",
            "AI_APP_NAME": "Evaluation app",
        }
        with patch.dict(os.environ, environment, clear=True):
            resolved = AIConfig.from_env(env_file=None)

        self.assertIs(resolved.provider, AIProvider.OPENROUTER)
        self.assertEqual(resolved.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(
            resolved.client_headers,
            {
                "HTTP-Referer": "https://example.test/app",
                "X-Title": "Evaluation app",
            },
        )
        self.assertEqual(resolved.audit_metadata()["schema_version"], 2)
        self.assertEqual(resolved.audit_metadata()["provider"], "openrouter")
        self.assertNotIn(resolved.api_key, repr(resolved))
        self.assertNotIn(resolved.api_key, str(resolved.audit_metadata()))

    def test_provider_defaults_and_environment_precedence(self):
        for provider, key_name, endpoint in (
            ("ollama", "OLLAMA_API_KEY", "https://ollama.com/v1"),
            ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        ):
            with self.subTest(provider=provider), patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": provider,
                    "AI_MODEL": "model",
                    key_name: "key",
                },
                clear=True,
            ):
                resolved = AIConfig.from_env(env_file=None)
                self.assertEqual(resolved.base_url, endpoint)
                self.assertEqual(resolved.client_headers, {})

        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "AI_PROVIDER=openrouter\n"
                "AI_MODEL=file-model\n"
                "OPENROUTER_API_KEY=file-key\n"
            )
            with patch.dict(
                os.environ,
                {"AI_MODEL": "environment-model"},
                clear=True,
            ):
                resolved = AIConfig.from_env(env_file)
        self.assertEqual(resolved.model, "environment-model")
        self.assertEqual(resolved.api_key, "file-key")

    def test_missing_and_invalid_environment_values(self):
        cases = (
            ({}, "AI_PROVIDER"),
            ({"AI_PROVIDER": "unknown"}, "Unsupported AI_PROVIDER"),
            ({"AI_PROVIDER": "openrouter"}, "AI_MODEL, OPENROUTER_API_KEY"),
            (
                {
                    "AI_PROVIDER": "openrouter",
                    "AI_MODEL": "model",
                },
                "OPENROUTER_API_KEY",
            ),
        )
        for environment, message in cases:
            with self.subTest(environment=environment), patch.dict(
                os.environ, environment, clear=True
            ), self.assertRaisesRegex(ValueError, message):
                AIConfig.from_env(env_file=None)

    def test_endpoint_and_direct_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            AIConfig(
                provider=AIProvider.OPENROUTER,
                model="model",
                base_url="https://example.test/v1",
                api_key=" ",
            )
        with self.assertRaisesRegex(ValueError, "endpoint"):
            AIConfig(
                provider=AIProvider.OPENROUTER,
                model="model",
                base_url="https://name:password@example.test/v1",
                api_key="key",
            )

    def test_openrouter_key_is_redacted_from_exported_text(self):
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "openrouter-secret"},
            clear=True,
        ):
            self.assertEqual(
                sanitize("Bearer openrouter-secret"),
                "Bearer [REDACTED]",
            )


class ProviderRequestTests(unittest.TestCase):
    def test_client_receives_openrouter_attribution_headers(self):
        openrouter = config(AIProvider.OPENROUTER)
        openrouter = replace(
            openrouter,
            app_url="https://example.test/app",
            app_name="Evaluation app",
        )
        with patch("src.ai.component.OpenAI") as sdk:
            create_client(openrouter)
        self.assertEqual(
            sdk.call_args.kwargs["default_headers"],
            {
                "HTTP-Referer": "https://example.test/app",
                "X-Title": "Evaluation app",
            },
        )
        self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)

    def test_only_openrouter_requires_structured_parameter_support(self):
        messages = ({"role": "user", "content": "Reply with JSON."},)
        for provider in AIProvider:
            with self.subTest(provider=provider):
                client = FakeClient()
                backend = OpenAICompatibleBackend(
                    config=config(provider),
                    client=client,
                )
                result = backend.generate_structured(
                    messages=messages,
                    response_model=Answer,
                )
                self.assertEqual(result.value.answer, "ok")
                if provider is AIProvider.OPENROUTER:
                    self.assertEqual(
                        client.requests[0]["extra_body"],
                        {"provider": {"require_parameters": True}},
                    )
                else:
                    self.assertNotIn("extra_body", client.requests[0])


if __name__ == "__main__":
    unittest.main()
