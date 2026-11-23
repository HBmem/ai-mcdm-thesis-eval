from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

from src.ai.backend import (
    BackendCallError,
    StructuredAIBackend,
    StructuredGeneration,
)
from src.ai.reporting.context import (
    validate_reporting_available_evidence,
)
from src.ai.reporting.models import (
    DataExplanationReport,
    DataExplanationReportingInput,
)
from src.ai.reporting.progress import (
    ReportingCallProgress,
    ReportingProgressCallback,
)
from src.ai.reporting.prompts import (
    DATA_EXPLANATION_REPORTING_PROMPT_VERSION,
    build_data_explanation_reporting_messages,
)
from src.ai.reporting.records import (
    DataExplanationReportArtifact,
    ReportingCallError,
    ReportingCallRecord,
    ReportingRequestMetadata,
)
from src.ai.reporting.serialization import (
    hash_reporting_input,
)
from src.ai.reporting.validation import (
    validate_report_response,
)
from src.ai.retry import choose_retry_delay


def make_reporting_artifact_id(
    request_id: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        (
            "data-explanation-report-artifact:"
            f"{request_id}"
        ),
    ).hex


@dataclass(frozen=True, slots=True)
class ReportingOutcome:
    generation: (
        StructuredGeneration[DataExplanationReport]
        | None
    )
    calls: tuple[ReportingCallRecord, ...]

    @property
    def succeeded(self) -> bool:
        return self.generation is not None

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError(
                "A reporting outcome requires at least "
                "one call record."
            )

        first_request = self.calls[0].request

        if any(
            call.request != first_request
            for call in self.calls
        ):
            raise ValueError(
                "A reporting outcome cannot mix requests."
            )

        expected_attempts = list(
            range(1, len(self.calls) + 1)
        )
        actual_attempts = [
            call.attempt
            for call in self.calls
        ]

        if actual_attempts != expected_attempts:
            raise ValueError(
                "Reporting call attempts must be consecutive."
            )

        for call in self.calls[:-1]:
            if (
                call.status != "failed"
                or call.retry is None
            ):
                raise ValueError(
                    "Every nonfinal reporting call must be a "
                    "failed attempt with a scheduled retry."
                )

        final_call = self.calls[-1]

        if self.generation is None:
            if final_call.retry is not None:
                raise ValueError(
                    "An unsuccessful outcome cannot end while "
                    "another retry is scheduled."
                )

            if final_call.status != "failed":
                raise ValueError(
                    "An unsuccessful outcome must end "
                    "with a failed call."
                )

            return

        if final_call.status != "completed":
            raise ValueError(
                "A successful outcome must end with a "
                "completed call."
            )

        if (
            final_call.parsed_response
            != self.generation.value
        ):
            raise ValueError(
                "The completed call and generation must "
                "contain the same report."
            )

        if final_call.receipt != self.generation.receipt:
            raise ValueError(
                "The completed call and generation must "
                "contain the same receipt."
            )

        if (
            final_call.raw_response
            != self.generation.raw_text
        ):
            raise ValueError(
                "The completed call and generation must "
                "contain the same raw response."
            )

        if final_call.retry is not None:
            raise ValueError(
                "An unsuccessful outcome cannot end while "
                "another retry is scheduled."
            )


def run_data_explanation_reporting(
    *,
    backend: StructuredAIBackend,
    request: ReportingRequestMetadata,
    input_snapshot: DataExplanationReportingInput,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_wait_seconds: float,
    on_progress: ReportingProgressCallback | None = None,
) -> ReportingOutcome:
    def emit(event: ReportingCallProgress) -> None:
        if on_progress is not None:
            on_progress(event)

    if max_retries < 0:
        raise ValueError("Retries must be >= 0.")

    if retry_base_seconds <= 0:
        raise ValueError(
            "Retry base seconds must be positive."
        )

    if retry_max_wait_seconds <= 0:
        raise ValueError(
            "Retry maximum wait must be positive."
        )

    if (
        request.prompt_version
        != DATA_EXPLANATION_REPORTING_PROMPT_VERSION
    ):
        raise ValueError(
            "Request prompt version does not match "
            "the active reporting prompt."
        )

    if request.input_hash != hash_reporting_input(
        input_snapshot
    ):
        raise ValueError(
            "Request input hash does not match the "
            "reporting snapshot."
        )

    input_issues = (
        validate_reporting_available_evidence(
            input_snapshot
        )
    )

    if input_issues:
        now = datetime.now(timezone.utc)

        call = ReportingCallRecord(
            call_id=uuid4().hex,
            request=request,
            attempt=1,
            input_snapshot=input_snapshot,
            messages=(),
            status="failed",
            started_at=now,
            completed_at=now,
            error=ReportingCallError(
                phase="input",
                code="invalid_reporting_evidence",
                message=(
                    "Reporting input contains unresolvable "
                    "evidence references."
                ),
                issues=input_issues,
            ),
        )

        return ReportingOutcome(
            generation=None,
            calls=(call,),
        )

    calls: list[ReportingCallRecord] = []

    messages = (
        build_data_explanation_reporting_messages(
            input_snapshot
        )
    )

    backend_messages = tuple(
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    )

    for attempt in range(1, max_retries + 2):
        emit(ReportingCallProgress(
            stage=request.stage,
            attempt=attempt,
            max_attempts=max_retries + 1,
            state="request_started",
        ))

        started_at = datetime.now(timezone.utc)

        try:
            generation = backend.generate_structured(
                messages=backend_messages,
                response_model=DataExplanationReport,
            )
        except BackendCallError as error:
            can_retry = (
                error.retryable
                and attempt <= max_retries
            )

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

            calls.append(
                ReportingCallRecord(
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
                    error=ReportingCallError(
                        phase=error.phase,
                        code=error.code,
                        message=str(error),
                        provider_details=error.provider_details,
                    ),
                    retry=retry,
                )
            )

            if retry is None:
                return ReportingOutcome(
                    generation=None,
                    calls=tuple(calls),
                )

            emit(ReportingCallProgress(
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

        validation = validate_report_response(
            reporting_input=input_snapshot,
            report=generation.value,
        )

        if not validation.valid:
            can_retry = attempt <= max_retries

            retry = (
                choose_retry_delay(
                    attempt=attempt,
                    provider_details=None,
                    base_seconds=retry_base_seconds,
                    max_wait_seconds=retry_max_wait_seconds,
                )
                if can_retry
                else None
            )

            calls.append(
                ReportingCallRecord(
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
                    error=ReportingCallError(
                        phase="validation",
                        code="invalid_report_response",
                        message=(
                            "The structured report failed "
                            "application validation."
                        ),
                        issues=validation.issues,
                    ),
                    retry=retry,
                )
            )

            if retry is None:
                return ReportingOutcome(
                    generation=None,
                    calls=tuple(calls),
                )

            emit(
                ReportingCallProgress(
                    stage=request.stage,
                    attempt=attempt,
                    max_attempts=max_retries + 1,
                    state="retry_wait",
                    delay_seconds=retry.delay_seconds,
                    code="invalid_report_response",
                    limit_source=None,
                )
            )

            time.sleep(retry.delay_seconds)
            continue

        calls.append(
            ReportingCallRecord(
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
                error=None,
            )
        )

        return ReportingOutcome(
            generation=generation,
            calls=tuple(calls),
        )

    raise RuntimeError("Unreachable Reporting retry state.")


def build_data_explanation_report_artifact(
    *,
    request: ReportingRequestMetadata,
    input_snapshot: DataExplanationReportingInput,
    outcome: ReportingOutcome,
) -> DataExplanationReportArtifact:
    if not outcome.succeeded:
        raise ValueError(
            "Cannot build an artifact from a failed "
            "reporting outcome."
        )

    generation = outcome.generation

    if generation is None:
        raise RuntimeError(
            "Successful outcome is missing its generation."
        )

    final_call = outcome.calls[-1]

    if final_call.status != "completed":
        raise ValueError(
            "An artifact requires a completed final call."
        )

    if final_call.request != request:
        raise ValueError(
            "Final call request does not match the "
            "artifact request."
        )

    if final_call.input_snapshot != input_snapshot:
        raise ValueError(
            "Final call input does not match the "
            "artifact input."
        )

    if final_call.parsed_response != generation.value:
        raise ValueError(
            "Final call response does not match the "
            "stage generation."
        )

    if final_call.receipt != generation.receipt:
        raise ValueError(
            "Final call receipt does not match the "
            "stage generation."
        )

    if final_call.raw_response != generation.raw_text:
        raise ValueError(
            "Final call raw response does not match "
            "the reporting generation."
        )

    return DataExplanationReportArtifact(
        artifact_id=make_reporting_artifact_id(
            request.request_id
        ),
        source_call_id=final_call.call_id,
        request=request,
        input_snapshot=input_snapshot,
        report=generation.value,
        receipt=generation.receipt,
        required_disclosures=(
            input_snapshot.required_disclosures
        ),
    )
