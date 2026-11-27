from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from src.ai.advisory.models import PolicyDecisionAdvisoryInput, PolicyDecisionAdvisoryResponse
from src.ai.advisory.serialization import hash_advisory_input
from src.ai.advisory.validation import validate_policy_decision_advisory_response
from src.ai.backend import BackendReceipt, ProviderErrorDetails
from src.ai.comprehension.models import ContentReview, PromptMessage, RetryDecision, ValidationIssue
from src.ai.models.analysis import Record


class AdvisoryRequestMetadata(Record):
    schema_version: Literal["advisory-request-v1"] = "advisory-request-v1"
    stage: Literal["policy_decision_advisory"] = "policy_decision_advisory"

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    ai_repetition: int = Field(ge=1)
    source_report_artifact_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def validate_advisory_request_binding(
    request: AdvisoryRequestMetadata,
    snapshot: PolicyDecisionAdvisoryInput,
) -> None:
    evidence_snapshot = snapshot.evidence_snapshot

    if request.run_id != evidence_snapshot.run_id:
        raise ValueError(
            "Advisory request and input must identify "
            "the same numerical run."
        )

    if request.ai_repetition != evidence_snapshot.ai_repetition:
        raise ValueError(
            "Advisory request and input must identify "
            "the same AI repetition."
        )

    if request.source_report_artifact_id != snapshot.source_report_artifact_id:
        raise ValueError(
            "Advisory request and input must identify "
            "the same source report."
        )

    if request.input_hash != hash_advisory_input(snapshot):
        raise ValueError(
            "Advisory request input hash does not match "
            "the supplied advisory snapshot."
        )


class AdvisoryCallError(Record):
    phase: Literal[
        "input", "serialization", "provider", "response",
        "parsing", "schema", "validation",
    ]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    issues: tuple[ValidationIssue, ...] = ()
    provider_details: ProviderErrorDetails | None = None


class AdvisoryCallRecord(Record):
    schema_version: Literal["advisory-call-v1"] = "advisory-call-v1"
    call_id: str = Field(min_length=1)
    request: AdvisoryRequestMetadata
    attempt: int = Field(ge=1)
    input_snapshot: PolicyDecisionAdvisoryInput
    messages: tuple[PromptMessage, ...] = ()
    status: Literal["completed", "failed"]
    started_at: AwareDatetime
    completed_at: AwareDatetime

    receipt: BackendReceipt | None = None
    raw_response: str | None = None
    parsed_response: PolicyDecisionAdvisoryResponse | None = None
    error: AdvisoryCallError | None = None
    retry: RetryDecision | None = None
    redacted: bool = False

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        validate_advisory_request_binding(
            self.request,
            self.input_snapshot,
        )

        if self.completed_at < self.started_at:
            raise ValueError("Call completion cannot precede its start.")

        if self.status == "failed":
            if self.error is None:
                raise ValueError("A failed advisory call requires an explicit error.")

            if self.parsed_response is not None:
                raise ValueError("A failed advisory call cannot retain a parsed response.")

            if self.error.phase == "validation":
                if (
                    self.receipt is None
                    or self.raw_response is None
                    or not self.raw_response.strip()
                ):
                    raise ValueError(
                        "An advisory validation failure must preserve "
                        "the receipt and nonblank raw response."
                    )

                if not self.error.issues or any(
                    not issue.response_path for issue in self.error.issues
                ):
                    raise ValueError(
                        "An advisory validation failure must preserve "
                        "issues with their response paths."
                    )

            return self

        if (
            not self.messages
            or self.receipt is None
            or self.raw_response is None
            or not self.raw_response.strip()
            or self.parsed_response is None
            or self.error is not None
        ):
            raise ValueError(
                "A completed advisory call requires messages, receipt, "
                "nonblank raw response, and parsed response, with no error."
            )

        if self.retry is not None:
            raise ValueError("A completed advisory call cannot schedule a retry.")

        issues = validate_policy_decision_advisory_response(
            advisory_input=self.input_snapshot,
            response=self.parsed_response,
        )
        if issues:
            details = ", ".join(
                f"{issue.code}@{issue.response_path}" for issue in issues
            )
            raise ValueError(
                f"A completed advisory call must pass validation: {details}"
            )

        return self


class PolicyDecisionAdvisoryArtifact(Record):
    schema_version: Literal["advisory-artifact-v1"] = "advisory-artifact-v1"
    artifact_id: str = Field(min_length=1)
    source_call_id: str = Field(min_length=1)
    request: AdvisoryRequestMetadata
    input_snapshot: PolicyDecisionAdvisoryInput
    response: PolicyDecisionAdvisoryResponse
    receipt: BackendReceipt
    content_review: ContentReview = Field(default_factory=ContentReview)

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        validate_advisory_request_binding(
            self.request,
            self.input_snapshot,
        )

        issues = validate_policy_decision_advisory_response(
            advisory_input=self.input_snapshot,
            response=self.response,
        )
        if issues:
            details = ", ".join(
                f"{issue.code}@{issue.response_path}" for issue in issues
            )
            raise ValueError(
                f"Advisory artifact response failed validation: {details}"
            )

        return self
