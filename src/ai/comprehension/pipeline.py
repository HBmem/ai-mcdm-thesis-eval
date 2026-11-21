from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, TypeVar
from uuid import uuid4, uuid5, NAMESPACE_URL

from src.ai.backend import BackendCallError, StructuredAIBackend, StructuredGeneration
from src.ai.comprehension.models import DCLCallError, DCLCallRecord, DCLRequestMetadata, DCLStageInput, GroupProfileArtifact, GroupProfileInput, GroupProfileResponse, PromptMessage, StoredInterpretation
from src.ai.comprehension.prompts import GROUP_PROFILE_PROMPT_VERSION, build_group_profile_messages
from src.ai.comprehension.serialization import hash_input_snapshot
from src.ai.comprehension.validation import materialize_validated_findings, validate_available_evidence, validate_response_evidence
from src.ai.models.analysis import Record


ResponseT = TypeVar("ResponseT", bound=Record)


def make_artifact_id(request_id: str) -> str:
    return uuid5(
        NAMESPACE_URL,
        f"dcl-artifact:{request_id}",
    ).hex


@dataclass(frozen=True, slots=True)
class DCLStageOutcome(Generic[ResponseT]):
    generation: StructuredGeneration[ResponseT] | None
    calls: tuple[DCLCallRecord, ...]
    findings: tuple[StoredInterpretation, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.generation is not None

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("A stage outcome requires at least one call record.")

        final_call = self.calls[-1]

        # Failed outcome
        if self.generation is None:
            if final_call.status != "failed":
                raise ValueError("An unsuccessful outcome must end with a failed call.")

            if self.findings:
                raise ValueError("An unsuccessful outcome cannot expose findings.")

            return

        # Successful outcome
        if final_call.status != "completed":
            raise ValueError("A successful outcome must end with a completed call.")

        if final_call.parsed_response != self.generation.value:
            raise ValueError("The completed call and stage generation must contain the same response.")

        if final_call.receipt != self.generation.receipt:
            raise ValueError("The completed call and stage generation must contain the same receipt.")

        if final_call.raw_response != self.generation.raw_text:
            raise ValueError("The completed call and stage generation must contain the same raw response.")

        parsed_response = final_call.parsed_response

        if parsed_response is None:
            raise ValueError("A successful outcome requires a parsed response.")

        # Revalidate the citations against the exact input snapshot used by the successful provider call.
        reference_validation = validate_response_evidence(
            final_call.input_snapshot,
            parsed_response,
        )

        if not reference_validation.valid:
            raise ValueError("A successful outcome cannot contain findings with invalid evidence references.")

        # Reconstruct the canonical findings from the response. This verifies IDs, paths, drafts, validation status, completeness, and deterministic ordering.
        expected_findings = materialize_validated_findings(
            request_id=final_call.request.request_id,
            validation=reference_validation,
        )

        if not expected_findings:
            raise ValueError("A successful DCL outcome requires findings.")

        if self.findings != expected_findings:
            raise ValueError("The outcome findings must exactly match the reference-validated response findings.")


def run_dcl_stage(
    *,
    backend: StructuredAIBackend,
    request: DCLRequestMetadata,
    input_snapshot: DCLStageInput,
    messages: tuple[PromptMessage, ...],
    response_model: type[ResponseT],
    max_retries: int,
) -> DCLStageOutcome[ResponseT]:
    if max_retries < 0:
        raise ValueError("Retries must be >= 0.")

    input_issues = validate_available_evidence(
        input_snapshot
    )

    if input_issues:
        now = datetime.now(timezone.utc)

        call = DCLCallRecord(
            call_id=uuid4().hex,
            request=request,
            attempt=1,
            input_snapshot=input_snapshot,

            # No messages were sent to the provider.
            messages=(),

            status="failed",
            started_at=now,
            completed_at=now,
            receipt=None,
            raw_response=None,
            parsed_response=None,
            error=DCLCallError(
                phase="input",
                code="invalid_available_evidence",
                message=(
                    "The DCL input contains invalid "
                    "evidence references."
                ),
                issues=input_issues,
            ),
        )

        return DCLStageOutcome(
            generation=None,
            calls=(call,),
            findings=(),
        )
    
    calls: list[DCLCallRecord] = []

    backend_messages = tuple(
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    )

    for attempt in range(1, max_retries + 2):
        started_at = datetime.now(timezone.utc)

        try:
            generation = backend.generate_structured(
                messages=backend_messages,
                response_model=response_model,
            )
        except BackendCallError as error:
            calls.append(
                DCLCallRecord(
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
                    error=DCLCallError(
                        phase=error.phase,
                        code=error.code,
                        message=str(error)
                    )
                )
            )

            if not error.retryable or attempt > max_retries:
                return DCLStageOutcome(
                    generation=None,
                    calls=tuple(calls),
                    findings=(),
                )

            continue

        reference_validation = validate_response_evidence(
            input_snapshot,
            generation.value,
        )

        if not reference_validation.valid:
            calls.append(
                DCLCallRecord(
                    call_id=uuid4().hex,
                    request=request,
                    attempt=attempt,
                    input_snapshot=input_snapshot,
                    messages=messages,
                    status="failed",
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),

                    # A provider response was received.
                    receipt=generation.receipt,
                    raw_response=generation.raw_text,

                    # The structured response failed post-schema validation.
                    parsed_response=None,

                    error=DCLCallError(
                        phase="references",
                        code="invalid_evidence_references",
                        message=(
                            "The structured response contains "
                            "invalid evidence references."
                        ),
                        issues=reference_validation.issues,
                    ),
                )
            )

            if attempt > max_retries:
                return DCLStageOutcome(
                    generation=None,
                    calls=tuple(calls),
                    findings=(),
                )

            continue

        findings = materialize_validated_findings(
            request_id=request.request_id,
            validation=reference_validation,
        )

        calls.append(
            DCLCallRecord(
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

        return DCLStageOutcome(
            generation=generation,
            calls=tuple(calls),
            findings=findings,
        )

    raise RuntimeError("Unreachable DCL retry state.")


def run_group_profile_stage(
    *,
    backend: StructuredAIBackend,
    request: DCLRequestMetadata,
    input_snapshot: GroupProfileInput,
    max_retries: int,
) -> DCLStageOutcome[GroupProfileResponse]:
    if request.stage != "stakeholder_group_profile":
        raise ValueError("Group-profile execution requires group-profile request metadata.")

    if request.group_id != input_snapshot.group.id:
        raise ValueError("Request and snapshot must identify the same group.")

    if request.prompt_version != GROUP_PROFILE_PROMPT_VERSION:
        raise ValueError("Request prompt version does not match the active group-profile prompt.")

    actual_hash = hash_input_snapshot(input_snapshot)

    if request.input_hash != actual_hash:
        raise ValueError("Request input hash does not match the supplied input snapshot.")

    messages = build_group_profile_messages(input_snapshot)

    return run_dcl_stage(
        backend=backend,
        request=request,
        input_snapshot=input_snapshot,
        messages=messages,
        response_model=GroupProfileResponse,
        max_retries=max_retries,
    )


def build_group_profile_artifact(
    *,
    request: DCLRequestMetadata,
    input_snapshot: GroupProfileInput,
    outcome: DCLStageOutcome[GroupProfileResponse],
) -> GroupProfileArtifact:
    """Build an artifact from a successfully validated stage outcome."""

    if not outcome.succeeded:
        raise ValueError(
            "Cannot build an artifact from a failed stage outcome."
        )

    generation = outcome.generation

    # Helps type checkers understand the succeeded invariant.
    if generation is None:
        raise RuntimeError(
            "Successful outcome is missing its generation."
        )

    if request.stage != "stakeholder_group_profile":
        raise ValueError(
            "Group-profile artifacts require group-profile metadata."
        )

    if request.group_id != input_snapshot.group.id:
        raise ValueError(
            "Request and input must identify the same group."
        )

    final_call = outcome.calls[-1]

    if final_call.status != "completed":
        raise ValueError(
            "An artifact requires a completed final call."
        )

    if final_call.parsed_response != generation.value:
        raise ValueError(
            "Final call response does not match the stage generation."
        )

    if final_call.receipt != generation.receipt:
        raise ValueError(
            "Final call receipt does not match the stage generation."
        )

    return GroupProfileArtifact(
        artifact_id=make_artifact_id(
            request.request_id
        ),
        request=request,
        input_snapshot=input_snapshot,
        response=generation.value,
        receipt=generation.receipt,
        findings=outcome.findings,
        required_limitations=(
            input_snapshot.required_limitations
        ),
    )