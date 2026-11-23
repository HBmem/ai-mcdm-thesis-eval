"""One bounded startup request verifies current key/model access, before a batch exists."""
import time
import json

from typing import Literal
from pydantic import Field
from uuid import uuid4
from openai import APIConnectionError, APITimeoutError, APIStatusError, APIError, OpenAI

from src.ai.backend import ProviderErrorDetails, provider_error_details
from src.ai.comprehension.models import RetryDecision
from src.ai.retry import choose_retry_delay
from src.common.privacy import sanitize
from src.common.time import timestamp
from src.ai.config import AIConfig
from src.ai.models.analysis import Record
from src.ai.progress import PreflightProgress, PreflightProgressCallback

PREFLIGHT_MESSAGES = [{"role": "user", "content": "Connectivity check. Reply briefly with OK."}]
EXIT_PAYMENT_REQUIRED = 3


class PreflightAttemptFailure(Record):
    attempt: int = Field(ge=1)
    started_at: str
    completed_at: str
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    provider_details: ProviderErrorDetails | None = None
    retry: RetryDecision | None = None


class PreflightError(ValueError):
    def __init__(
        self,
        message: str,
        exit_code: int,
        *,
        attempts: tuple[
            PreflightAttemptFailure, ...
        ] = (),
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.attempts = attempts

class PreflightRecord(Record):
    schema_version: Literal[1] = 1
    stage: Literal["startup_check"] = "startup_check"
    status: Literal["passed"] = "passed"
    call_id: str
    batch_id: str | None = None
    started_at: str
    completed_at: str
    duration_seconds: float = Field(ge=0)
    messages: list[dict[str, str]]
    input: dict
    configuration: dict
    raw_output: str | None
    response: dict
    receipt: dict
    redacted: bool = False
    failed_attempts: tuple[PreflightAttemptFailure, ...] = ()


def _preflight_exit_code(
    status_code: int,
) -> int:
    if status_code == 402:
        return EXIT_PAYMENT_REQUIRED

    if status_code in {
        400,
        401,
        403,
        404,
        405,
        422,
    }:
        return 2

    return 1


def check_model_access(
    config: AIConfig,
    client: OpenAI,
    on_progress: PreflightProgressCallback | None = None,
) -> PreflightRecord:
    def emit(event: PreflightProgress) -> None:
        if on_progress is not None:
            on_progress(event)

    request = {
        "model": config.model,
        "messages": [dict(m) for m in PREFLIGHT_MESSAGES],
        "max_tokens": 8,
        "stream": False,
        "timeout": 30
    }

    if sanitize(request, (config.api_key,)) != request:
        raise PreflightError("AI configuration contains sensitive values in request metadata.", 2)

    failures: list[PreflightAttemptFailure] = []
    preflight_started_at = timestamp()
    tick = time.perf_counter()

    for attempt in range(1, config.max_retries + 2):
        attempt_started_at = timestamp()

        emit(PreflightProgress(
            attempt=attempt,
            max_attempts=config.max_retries + 1,
            state="request_started",
        ))

        try:
            response = client.chat.completions.create(
                **request
            )
            break
        except APITimeoutError:
            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=None,
                    base_seconds=config.retry_base_seconds,
                    max_wait_seconds=config.retry_max_wait_seconds,
                )
                if attempt <= config.max_retries
                else None
            )
            failures.append(
                PreflightAttemptFailure(
                    attempt=attempt,
                    started_at=attempt_started_at,
                    completed_at=timestamp(),
                    code="timeout",
                    message="AI access verification timed out; key/model validity was not established.",
                    provider_details=None,
                    retry=retry,
                )
            )
            if retry is None:
                raise PreflightError(
                    "AI access verification timed out; key/model validity was not established.",
                    1,
                    attempts=tuple(failures),
                ) from None

            emit(PreflightProgress(
                attempt=attempt,
                max_attempts=config.max_retries + 1,
                state="retry_wait",
                delay_seconds=retry.delay_seconds,
                code="timeout",
            ))
            time.sleep(retry.delay_seconds)
            continue
        except APIConnectionError:
            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=None,
                    base_seconds=config.retry_base_seconds,
                    max_wait_seconds=config.retry_max_wait_seconds,
                )
                if attempt <= config.max_retries
                else None
            )
            failures.append(
                PreflightAttemptFailure(
                    attempt=attempt,
                    started_at=attempt_started_at,
                    completed_at=timestamp(),
                    code="connection_error",
                    message="Cannot connect to the configured AI endpoint; key/model validity was not established.",
                    provider_details=None,
                    retry=retry,
                )
            )
            if retry is None:
                raise PreflightError(
                    "Cannot connect to the configured AI endpoint; key/model validity was not established.",
                    1,
                    attempts=tuple(failures),
                ) from None

            emit(PreflightProgress(
                attempt=attempt,
                max_attempts=config.max_retries + 1,
                state="retry_wait",
                delay_seconds=retry.delay_seconds,
                code="connection_error",
            ))
            time.sleep(retry.delay_seconds)
            continue
        except APIStatusError as error:
            details = provider_error_details(error)

            retryable = (
                error.status_code in {408, 409, 429}
                or error.status_code >= 500
            )

            can_retry = (
                retryable
                and attempt <= config.max_retries
            )

            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=details,
                    base_seconds=config.retry_base_seconds,
                    max_wait_seconds=(
                        config.retry_max_wait_seconds
                    ),
                )
                if can_retry
                else None
            )

            failures.append(
                PreflightAttemptFailure(
                    attempt=attempt,
                    started_at=attempt_started_at,
                    completed_at=timestamp(),
                    code=(
                        f"provider_status_"
                        f"{error.status_code}"
                    ),
                    message=(
                        "The provider rejected the "
                        "startup request."
                    ),
                    provider_details=details,
                    retry=retry,
                )
            )

            if retry is None:
                raise PreflightError(
                    "AI access verification was "
                    "unsuccessful.",
                    _preflight_exit_code(error.status_code),
                    attempts=tuple(failures),
                ) from None
            
            emit(PreflightProgress(
                attempt=attempt,
                max_attempts=config.max_retries + 1,
                state="retry_wait",
                delay_seconds=retry.delay_seconds,
                code=f"provider_status_{error.status_code}",
                limit_source=details.limit_source,
            ))

            time.sleep(retry.delay_seconds)
            continue
        except APIError:
            raise PreflightError("AI access verification returned an invalid provider response.", 1) from None

    try:
        if not response.choices:
            raise ValueError("Missing completion choices.")
        choice = response.choices[0]
        message = choice.message
        if choice.finish_reason not in {"stop", "length"} or getattr(message, "refusal", None):
            raise ValueError("Refused or unsupported completion.")
        content = message.content
        if content is not None and not isinstance(content, str):
            raise ValueError("Invalid completion content.")
        if choice.finish_reason == "stop" and (content is None or not content.strip()):
            raise ValueError("Empty completed response.")
        usage = getattr(response, "usage", None)
        receipt = {
            "response_id": getattr(response, "id", None),
            "request_model": config.model,
            "returned_model": getattr(response, "model", None),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None)
        }
        record = PreflightRecord(
            call_id=uuid4().hex,
            started_at=preflight_started_at,
            completed_at=timestamp(),
            duration_seconds=time.perf_counter()-tick,
            messages=request["messages"],
            input=request,
            configuration={**config.audit_metadata(), "timeout": 30, "max_tokens": 8},
            raw_output=content, response={
                "id": receipt["response_id"],
                "model": receipt["returned_model"],
                "choices": [{"finish_reason": choice.finish_reason, "message": {"content": content, "refusal": getattr(message, "refusal", None)}}]
            },
            receipt=receipt,
            failed_attempts=tuple(failures)
        )
    except (AttributeError, IndexError, TypeError, ValueError):
            raise PreflightError("AI access verification returned a malformed, empty or refused completion.", 1) from None
    original = record.model_dump(mode="json")
    clean = sanitize(original, (config.api_key,))
    clean["redacted"] = clean != original
    return PreflightRecord.model_validate_json(
        json.dumps(
            clean,
            allow_nan=False,
        )
    )
