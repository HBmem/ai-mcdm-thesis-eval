from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.ai.advisory.models import (
    PolicyDecisionAdvisoryInput,
    PolicyDecisionAdvisoryResponse,
)
from src.ai.advisory.progress import (
    AdvisoryCallProgress,
    AdvisoryProgressCallback,
)
from src.ai.advisory.prompts import (
    POLICY_DECISION_ADVISORY_PROMPT_VERSION,
    build_policy_decision_advisory_messages,
)
from src.ai.advisory.records import (
    AdvisoryCallError,
    AdvisoryCallRecord,
    AdvisoryRequestMetadata,
    PolicyDecisionAdvisoryArtifact,
    validate_advisory_request_binding,
)
from src.ai.advisory.validation import (
    validate_policy_decision_advisory_response,
)
from src.ai.backend import BackendCallError, StructuredAIBackend
from src.ai.reporting.context import validate_reporting_available_evidence
from src.ai.retry import choose_retry_delay


@dataclass(frozen=True, slots=True)
class AdvisoryOutcome:
    calls: tuple[AdvisoryCallRecord, ...]

    @property
    def succeeded(self) -> bool:
        return self.calls[-1].status == "completed"

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError(
                "An advisory outcome requires at least one call record."
            )

        first_request = self.calls[0].request
        if any(call.request != first_request for call in self.calls):
            raise ValueError("An advisory outcome cannot mix requests.")

        expected_attempts = list(range(1, len(self.calls) + 1))
        actual_attempts = [call.attempt for call in self.calls]
        if actual_attempts != expected_attempts:
            raise ValueError("Advisory call attempts must be consecutive.")

        for call in self.calls[:-1]:
            if call.status != "failed" or call.retry is None:
                raise ValueError(
                    "Every nonfinal advisory call must be a failed "
                    "attempt with a scheduled retry."
                )

        if self.calls[-1].retry is not None:
            raise ValueError(
                "The final advisory call cannot have a pending retry."
            )


def run_policy_decision_advisory(
    *,
    backend: StructuredAIBackend,
    request: AdvisoryRequestMetadata,
    input_snapshot: PolicyDecisionAdvisoryInput,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_wait_seconds: float,
    on_progress: AdvisoryProgressCallback | None = None,
) -> AdvisoryOutcome:
    def emit(event: AdvisoryCallProgress) -> None:
        if on_progress is not None:
            on_progress(event)

    if max_retries < 0:
        raise ValueError("Retries must be >= 0.")

    if (
        not math.isfinite(retry_base_seconds)
        or retry_base_seconds <= 0
    ):
        raise ValueError("Retry base seconds must be positive and finite.")

    if (
        not math.isfinite(retry_max_wait_seconds)
        or retry_max_wait_seconds <= 0
    ):
        raise ValueError("Retry maximum wait must be positive and finite.")

    validate_advisory_request_binding(request, input_snapshot)

    if request.prompt_version != POLICY_DECISION_ADVISORY_PROMPT_VERSION:
        raise ValueError(
            "Request prompt version does not match "
            "the active advisory prompt."
        )

    input_issues = validate_reporting_available_evidence(
        input_snapshot.evidence_snapshot
    )
    if input_issues:
        now = datetime.now(timezone.utc)
        call = AdvisoryCallRecord(
            call_id=uuid4().hex,
            request=request,
            attempt=1,
            input_snapshot=input_snapshot,
            messages=(),
            status="failed",
            started_at=now,
            completed_at=now,
            error=AdvisoryCallError(
                phase="input",
                code="invalid_advisory_evidence",
                message=(
                    "Advisory input contains unresolvable "
                    "evidence references."
                ),
                issues=input_issues,
            ),
        )
        return AdvisoryOutcome(calls=(call,))

    messages = build_policy_decision_advisory_messages(input_snapshot)
    backend_messages = tuple(
        {"role": message.role, "content": message.content}
        for message in messages
    )

    calls: list[AdvisoryCallRecord] = []

    for attempt in range(1, max_retries + 2):
        emit(AdvisoryCallProgress(
            stage=request.stage,
            attempt=attempt,
            max_attempts=max_retries + 1,
            state="request_started",
        ))
        started_at = datetime.now(timezone.utc)

        try:
            generation = backend.generate_structured(
                messages=backend_messages,
                response_model=PolicyDecisionAdvisoryResponse,
            )
        except BackendCallError as error:
            can_retry = error.retryable and attempt <= max_retries
            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=error.provider_details,
                    base_seconds=retry_base_seconds,
                    max_wait_seconds=retry_max_wait_seconds,
                )
                if can_retry
                else None
            )

            calls.append(AdvisoryCallRecord(
                call_id=uuid4().hex,
                request=request,
                attempt=attempt,
                input_snapshot=input_snapshot,
                messages=messages,
                status="failed",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                receipt=error.receipt,
                raw_response=error.raw_text,
                parsed_response=None,
                error=AdvisoryCallError(
                    phase=error.phase,
                    code=error.code,
                    message=str(error),
                    provider_details=error.provider_details,
                ),
                retry=retry,
            ))

            if retry is None:
                return AdvisoryOutcome(calls=tuple(calls))

            emit(AdvisoryCallProgress(
                stage=request.stage,
                attempt=attempt,
                max_attempts=max_retries + 1,
                state="retry_wait",
                delay_seconds=retry.delay_seconds,
                code=error.code,
                limit_source=(
                    error.provider_details.limit_source
                    if error.provider_details is not None
                    else None
                ),
            ))
            time.sleep(retry.delay_seconds)
            continue

        issues = validate_policy_decision_advisory_response(
            advisory_input=input_snapshot,
            response=generation.value,
        )
        if issues:
            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=None,
                    base_seconds=retry_base_seconds,
                    max_wait_seconds=retry_max_wait_seconds,
                )
                if attempt <= max_retries
                else None
            )

            calls.append(AdvisoryCallRecord(
                call_id=uuid4().hex,
                request=request,
                attempt=attempt,
                input_snapshot=input_snapshot,
                messages=messages,
                status="failed",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                receipt=generation.receipt,
                raw_response=generation.raw_text,
                parsed_response=None,
                error=AdvisoryCallError(
                    phase="validation",
                    code="invalid_advisory_response",
                    message=(
                        "The structured advisory response failed "
                        "application validation."
                    ),
                    issues=issues,
                ),
                retry=retry,
            ))

            if retry is None:
                return AdvisoryOutcome(calls=tuple(calls))

            emit(AdvisoryCallProgress(
                stage=request.stage,
                attempt=attempt,
                max_attempts=max_retries + 1,
                state="retry_wait",
                delay_seconds=retry.delay_seconds,
                code="invalid_advisory_response",
            ))
            time.sleep(retry.delay_seconds)
            continue

        calls.append(AdvisoryCallRecord(
            call_id=uuid4().hex,
            request=request,
            attempt=attempt,
            input_snapshot=input_snapshot,
            messages=messages,
            status="completed",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            receipt=generation.receipt,
            raw_response=generation.raw_text,
            parsed_response=generation.value,
        ))
        return AdvisoryOutcome(calls=tuple(calls))

    raise RuntimeError("Unreachable advisory retry state.")


def build_policy_decision_advisory_artifact(
    *,
    outcome: AdvisoryOutcome,
) -> PolicyDecisionAdvisoryArtifact:
    final_call = outcome.calls[-1]

    if final_call.status != "completed":
        raise ValueError(
            "An advisory artifact requires a completed final call."
        )

    if final_call.parsed_response is None or final_call.receipt is None:
        raise ValueError(
            "A completed advisory call must contain a response and receipt."
        )

    artifact_id = uuid5(
        NAMESPACE_URL,
        f"policy-decision-advisory-artifact:{final_call.call_id}",
    ).hex

    return PolicyDecisionAdvisoryArtifact(
        artifact_id=artifact_id,
        source_call_id=final_call.call_id,
        request=final_call.request,
        input_snapshot=final_call.input_snapshot,
        response=final_call.parsed_response,
        receipt=final_call.receipt,
    )
