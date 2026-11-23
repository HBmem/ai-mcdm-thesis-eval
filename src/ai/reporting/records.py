from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    model_validator,
)

from src.ai.backend import (
    BackendReceipt,
    ProviderErrorDetails,
)
from src.ai.comprehension.models import (
    ContentReview,
    PromptMessage,
    RetryDecision,
    ValidationIssue,
)
from src.ai.models.analysis import Record
from src.ai.reporting.models import (
    DataExplanationReport,
    DataExplanationReportingInput,
)
from src.ai.reporting.serialization import (
    hash_reporting_input,
)
from src.ai.reporting.validation import (
    validate_report_response,
)


ReportingFailurePhase = Literal[
    "input",
    "serialization",
    "provider",
    "response",
    "parsing",
    "schema",
    "validation",
    "content",
]


class ReportingRequestMetadata(Record):
    schema_version: Literal["reporting-request-v1"] = "reporting-request-v1"

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    ai_repetition: int = Field(ge=1)

    stage: Literal[
        "data_explanation_reporting"
    ] = "data_explanation_reporting"

    prompt_version: str = Field(min_length=1)
    input_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated"


def _validate_request_binding(
    request: ReportingRequestMetadata,
    input_snapshot: DataExplanationReportingInput,
) -> None:
    if request.run_id != input_snapshot.run_id:
        raise ValueError(
            "Reporting request and input must identify "
            "the same numerical run."
        )

    if (
        request.ai_repetition
        != input_snapshot.ai_repetition
    ):
        raise ValueError(
            "Reporting request and input must identify "
            "the same AI repetition."
        )

    if request.input_origin != input_snapshot.input_origin:
        raise ValueError(
            "Reporting request and input must have the "
            "same input origin."
        )

    expected_hash = hash_reporting_input(
        input_snapshot
    )

    if request.input_hash != expected_hash:
        raise ValueError(
            "Reporting request input hash does not match "
            "the supplied reporting snapshot."
        )


class ReportingCallError(Record):
    phase: ReportingFailurePhase
    code: str = Field(min_length=1)

    message: str = Field(
        min_length=1,
        description=(
            "Sanitized explanation without credentials "
            "or unrestricted provider headers."
        ),
    )

    issues: tuple[ValidationIssue, ...] = ()

    provider_details: (
        ProviderErrorDetails | None
    ) = None


class ReportingCallRecord(Record):
    schema_version: Literal["reporting-call-v1"] = "reporting-call-v1"

    call_id: str = Field(min_length=1)
    request: ReportingRequestMetadata
    attempt: int = Field(ge=1)

    input_snapshot: DataExplanationReportingInput
    messages: tuple[PromptMessage, ...] = ()

    status: Literal["completed", "failed"]

    started_at: AwareDatetime
    completed_at: AwareDatetime

    receipt: BackendReceipt | None = None
    raw_response: str | None = None
    parsed_response: DataExplanationReport | None = None

    error: ReportingCallError | None = None
    retry: RetryDecision | None = None
    redacted: bool = False

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        _validate_request_binding(
            self.request,
            self.input_snapshot,
        )

        if self.completed_at < self.started_at:
            raise ValueError(
                "Call completion cannot precede its start."
            )

        if self.status == "failed":
            if self.error is None:
                raise ValueError(
                    "A failed reporting call requires "
                    "an explicit error."
                )

            if self.parsed_response is not None:
                raise ValueError(
                    "A failed reporting call cannot retain "
                    "a parsed report as an accepted response."
                )

            return self

        if (
            not self.messages
            or self.receipt is None
            or self.raw_response is None
            or self.parsed_response is None
            or self.error is not None
        ):
            raise ValueError(
                "A completed reporting call requires "
                "messages, receipt, raw response, parsed "
                "response, and no error."
            )

        if not self.raw_response.strip():
            raise ValueError(
                "A completed call cannot have an empty "
                "raw response."
            )

        if self.retry is not None:
            raise ValueError(
                "A completed call cannot schedule a retry."
            )

        validation = validate_report_response(
            reporting_input=self.input_snapshot,
            report=self.parsed_response,
        )

        if not validation.valid:
            raise ValueError(
                "A completed reporting call must contain "
                "a valid report response."
            )

        return self


class DataExplanationReportArtifact(Record):
    schema_version: Literal["data-explanation-report-artifact-v1"] = "data-explanation-report-artifact-v1"

    artifact_id: str = Field(min_length=1)
    source_call_id: str = Field(min_length=1)

    request: ReportingRequestMetadata
    input_snapshot: DataExplanationReportingInput
    report: DataExplanationReport
    receipt: BackendReceipt

    validation_status: Literal[
        "validated"
    ] = "validated"

    required_disclosures: tuple[str, ...]
    content_review: ContentReview = Field(
        default_factory=ContentReview
    )

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        _validate_request_binding(
            self.request,
            self.input_snapshot,
        )

        if (
            self.required_disclosures
            != self.input_snapshot.required_disclosures
        ):
            raise ValueError(
                "The report artifact must preserve required "
                "disclosures unchanged."
            )

        validation = validate_report_response(
            reporting_input=self.input_snapshot,
            report=self.report,
        )

        if not validation.valid:
            raise ValueError(
                "A report artifact cannot contain an "
                "invalid report response."
            )

        return self