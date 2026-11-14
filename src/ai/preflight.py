"""One bounded startup request verifies current key/model access, before a batch exists."""
import time

from typing import Literal
from pydantic import Field
from uuid import uuid4
from openai import APIConnectionError, APITimeoutError, APIStatusError, APIError

from src.common.privacy import sanitize
from src.ai.config import AIConfig
from src.ai.models.analysis import Record

PREFLIGHT_MESSAGES = [{"role": "user", "content": "Connectivity check. Reply briefly with OK."}]
EXIT_PAYMENT_REQUIRED = 3

class PreflightError(ValueError):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code

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

def timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def check_model_access(config: AIConfig, client):
    started_at, tick = timestamp(), time.perf_counter()
    request = {
        "model": config.model,
        "messages": [dict(m) for m in PREFLIGHT_MESSAGES],
        "max_tokens": 8,
        "stream": False,
        "timeout": 30
    }

    if sanitize(request, (config.api_key,)) != request:
        raise PreflightError("AI configuration contains sensitive values in request metadata.", 2)

    try:
        response = client.chat.completions.create(**request)
    except APITimeoutError:
        raise PreflightError("AI access verification timed out; key/model validity was not established.", 1) from None
    except APIConnectionError:
        raise PreflightError("Cannot connect to the configured AI endpoint; key/model validity was not established.", 1) from None
    except APIStatusError as error:
        if error.status_code in {401, 403}:
            message, code = "AI authentication or permission check failed. Check OLLAMA_API_KEY and model access.", 2
        elif error.status_code in {400, 404, 405, 422}:
            message, code = "The endpoint rejected the configured model or request. Check OLLAMA_MODEL and OLLAMA_BASE_URL.", 2
        elif error.status_code == 402:
            message, code = "The AI provider requires payment or available credits to process this request. Check account billing, credits, or plan.", EXIT_PAYMENT_REQUIRED
        else:
            message, code = "AI access verification was unsuccessful; the provider may be rate-limited or unavailable.", 1
        raise PreflightError(message, code) from None
    except APIError:
        raise PreflightError("AI access verification returned an invalid provider response.", 1) from None

    print(response)

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
            all_id=uuid4().hex,
            started_at=started_at,
            completed_at=timestamp(),
            duration_seconds=time.perf_counter()-tick,
            messages=request["messages"],
            input=request,
            configuration={**config.safe(), "timeout": 30, "max_retries": 0, "max_tokens": 8},
            raw_output=content, response={
                "id": receipt["response_id"],
                "model": receipt["returned_model"],
                "choices": [{"finish_reason": choice.finish_reason, "message": {"content": content, "refusal": getattr(message, "refusal", None)}}]
            },
            receipt=receipt
        )
    except (AttributeError, IndexError, TypeError, ValueError):
            raise PreflightError("AI access verification returned a malformed, empty or refused completion.", 1) from None
    original = record.model_dump(mode="json")
    clean = sanitize(original, (config.api_key,))
    clean["redacted"] = clean != original
    return PreflightRecord.model_validate(clean)