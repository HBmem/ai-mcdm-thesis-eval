from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Literal, TypedDict, TypeVar, Protocol

from openai import OpenAI, APITimeoutError, APIConnectionError, APIError, APIStatusError
from pydantic import BaseModel, Field, AwareDatetime, ValidationError, JsonValue

from src.ai.config import AIConfig, AIProvider
from src.ai.models.analysis import Record


DIAGNOSTIC_RESPONSE_HEADERS = frozenset({
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "x-request-id",
    "request-id",
    "cf-ray",
})


ResponseT = TypeVar("ResponseT", bound=BaseModel)
BackendFailurePhase = Literal[
    "provider",
    "response",
    "parsing",
    "schema",
]


class ProviderErrorDetails(Record):
    status_code: int = Field(ge=100, le=599)
    body: JsonValue | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    limit_source: str | None = None
    request_id: str | None = None


def _metadata_headers(
    body: object,
) -> dict[str, str]:
    if not isinstance(body, dict):
        return {}

    error_data = body.get("error", body)
    if not isinstance(error_data, dict):
        return {}

    metadata = error_data.get("metadata")
    if not isinstance(metadata, dict):
        return {}

    return _diagnostic_headers(
        metadata.get("headers")
    )


def _diagnostic_headers(
    headers: object,
) -> dict[str, str]:
    if headers is None or not hasattr(headers, "items"):
        return {}

    return {
        str(name).lower(): str(value)
        for name, value in headers.items()
        if str(name).lower() in DIAGNOSTIC_RESPONSE_HEADERS
    }


def _limit_source(body: object) -> str | None:
    if not isinstance(body, dict):
        return None

    error_data = body.get("error", body)
    if not isinstance(error_data, dict):
        return None

    metadata = error_data.get("metadata")
    if not isinstance(metadata, dict):
        return None

    value = metadata.get("limit_source")
    return value if isinstance(value, str) else None


def provider_error_details(
    error: APIStatusError,
) -> ProviderErrorDetails:
    body = error.body

    headers = {
        **_metadata_headers(body),
        **_diagnostic_headers(
            getattr(error.response, "headers", None)
        ),
    }

    return ProviderErrorDetails(
        status_code=error.status_code,
        body=body,
        headers=headers,
        limit_source=_limit_source(body),
        request_id=(
            getattr(error, "request_id", None)
            or headers.get("x-request-id")
            or headers.get("request-id")
        ),
    )


class BackendCallError(RuntimeError):
    def __init__(
        self,
        *,
        phase: BackendFailurePhase,
        code: str,
        message: str,
        retryable: bool,
        raw_text: str | None = None,
        receipt: BackendReceipt | None = None,
        provider_details: ProviderErrorDetails | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.retryable = retryable
        self.raw_text = raw_text
        self.receipt = receipt
        self.provider_details = provider_details


class AIMessage(TypedDict):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class BackendReceipt(Record):
    response_id: str | None = None
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    received_at: AwareDatetime
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    
@dataclass(frozen=True, slots=True)
class StructuredGeneration(Generic[ResponseT]):
    value: ResponseT
    raw_text: str
    receipt: BackendReceipt


class StructuredAIBackend(Protocol):
    def generate_structured(
        self,
        *,
        messages: tuple[AIMessage, ...],
        response_model: type[ResponseT],
    ) -> StructuredGeneration[ResponseT]:
        ...


@dataclass
class OpenAICompatibleBackend:
    config: AIConfig
    client: OpenAI

    def generate_structured(
        self,
        *,
        messages: tuple[AIMessage, ...],
        response_model: type[ResponseT],
    ) -> StructuredGeneration[ResponseT]:
        request = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "response_format": _response_format(response_model),
            "stream": False,
            "timeout": self.config.timeout,
        }

        if self.config.provider is AIProvider.OPENROUTER:
            request["extra_body"] = {
                "provider": {"require_parameters": True},
            }

        # Provider invocation
        try:
            response = self.client.chat.completions.create(**request)
        except APITimeoutError:
            raise BackendCallError(
                phase="provider",
                code="request_timeout",
                message="The structured AI request timed out.",
                retryable=True,
            ) from None

        except APIConnectionError:
            raise BackendCallError(
                phase="provider",
                code="connection_error",
                message=(
                    "The structured AI request could not reach "
                    "the provider."
                ),
                retryable=True,
            ) from None

        except APIStatusError as error:
            body = error.body
            status_code = error.status_code

            retryable = (
                status_code in {408, 409, 429}
                or status_code >= 500
            )

            response_headers = {
                **_metadata_headers(body),
                **_diagnostic_headers(
                    getattr(error.response, "headers", None)
                ),
            }

            provider_details = ProviderErrorDetails(
                status_code=status_code,
                body=body,
                headers=response_headers,
                limit_source=_limit_source(body),
                request_id=(
                    getattr(error, "request_id", None)
                    or response_headers.get("x-request-id")
                    or response_headers.get("request-id")
                ),
            )
            details = provider_error_details(error)

            raise BackendCallError(
                phase="provider",
                code=f"provider_status_{status_code}",
                message=(
                    "The AI provider rejected the structured request "
                    f"with status {status_code}."
                ),
                retryable=retryable,
                provider_details=details,
            ) from None

        except APIError:
            raise BackendCallError(
                phase="provider",
                code="provider_error",
                message=(
                    "The AI provider could not complete the "
                    "structured request."
                ),
                retryable=False,
            ) from None

        # Validate that the provider returned at least one choice.
        try:
            choices = response.choices
        except AttributeError:
            choices = None

        if not choices:
            raise BackendCallError(
                phase="response",
                code="missing_choices",
                message=(
                    "The provider response did not contain "
                    "a completion choice."
                ),
                retryable=True,
            )
        
        choice = choices[0]
        message = getattr(choice, "message", None)

        if message is None:
            raise BackendCallError(
                phase="response",
                code="missing_message",
                message=(
                    "The provider response did not contain "
                    "a completion message."
                ),
                retryable=True,
            )

        # Build the receipt as soon as a choice is available. This lets parsing/schema failures retain provider metadata.
        usage = getattr(response, "usage", None)
        finish_reason = getattr(choice, "finish_reason", None)

        receipt = BackendReceipt(
            response_id=getattr(response, "id", None),
            requested_model=self.config.model,
            returned_model=getattr(response, "model", None),
            received_at=datetime.now(timezone.utc),
            finish_reason=finish_reason,
            input_tokens=(
                getattr(usage, "prompt_tokens", None)
                if usage is not None
                else None
            ),
            output_tokens=(
                getattr(usage, "completion_tokens", None)
                if usage is not None
                else None
            ),
        )

        if getattr(message, "refusal", None):
            raise BackendCallError(
                phase="response",
                code="model_refusal",
                message="The model refused the structured request.",
                retryable=False,
                receipt=receipt,
            )

        # Only "stop" indicates a complete structured response.
        if finish_reason != "stop":
            retryable = finish_reason == "length"

            raise BackendCallError(
                phase="response",
                code="incomplete_response",
                message=(
                    "The model did not complete the structured "
                    "response."
                ),
                retryable=retryable,
                receipt=receipt,
            )

        raw_text = getattr(message, "content", None)

        if not isinstance(raw_text, str):
            raise BackendCallError(
                phase="response",
                code="missing_content",
                message=(
                    "The provider response did not contain "
                    "textual JSON content."
                ),
                retryable=True,
                receipt=receipt,
            )

        if not raw_text.strip():
            raise BackendCallError(
                phase="response",
                code="empty_content",
                message=(
                    "The provider returned an empty structured "
                    "response."
                ),
                retryable=True,
                raw_text=raw_text,
                receipt=receipt,
            )

        # First distinguish syntactically invalid JSON from valid JSON that does not satisfy the requested schema.
        try:
            json.loads(raw_text)
        except json.JSONDecodeError:
            raise BackendCallError(
                phase="parsing",
                code="invalid_json",
                message="The provider returned invalid JSON.",
                retryable=True,
                raw_text=raw_text,
                receipt=receipt,
            ) from None

        # Validate from the original JSON string. In strict Pydantic models, JSON arrays can correctly become tuple fields here.
        try:
            value = response_model.model_validate_json(raw_text)
        except ValidationError:
            raise BackendCallError(
                phase="schema",
                code="schema_validation_failed",
                message=(
                    "The provider response did not match "
                    "the requested schema."
                ),
                retryable=True,
                raw_text=raw_text,
                receipt=receipt,
            ) from None

        return StructuredGeneration(
            value=value,
            raw_text=raw_text,
            receipt=receipt,
        )

def _response_format(
    response_model: type[ResponseT],
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": response_model.model_json_schema(),
        },
    }

