from typing import Literal, Self
from pydantic import Field, model_validator

from src.ai.comprehension.models import (
    ContentReview,
    DCLStage,
    DiagnosticContext,
    EvidenceReference,
    GroupContext,
    InterpretationDraft,
    ScenarioContext,
    SessionCalculationView,
    Uncertainty,
    ProcessingContext,
)
from src.ai.models.analysis import Record
from src.mcdm.models.domain import Alternative, Criterion


ReportClaimType = Literal[
    "computed_result",
    "descriptive_comparison",
    "contextual_interpretation",
    "evidence_gap",
]


class ReportingFindingContext(Record):
    finding_id: str = Field(min_length=1)
    stage: DCLStage
    group_id: str | None = None
    draft: InterpretationDraft
    reference_status: Literal["validated"] = "validated"
    content_review: ContentReview

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.stage == "stakeholder_group_profile":
            if self.group_id is None:
                raise ValueError(
                    "A group-profile finding requires "
                    "a group ID."
                )
        elif self.group_id is not None:
            raise ValueError(
                "Session-level findings cannot claim "
                "a group ID."
            )

        if self.content_review.status == "rejected":
            raise ValueError(
                "Rejected DCL findings cannot enter "
                "reporting context."
            )

        return self


class DataExplanationReportingInput(Record):
    schema_version: Literal[
        "reporting-input-v1"
    ] = "reporting-input-v1"

    run_id: str = Field(min_length=1)
    ai_repetition: int = Field(ge=1)

    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated"

    scenario: ScenarioContext

    criteria: tuple[
        Criterion, ...
    ] = Field(min_length=1)

    alternatives: tuple[
        Alternative, ...
    ] = Field(min_length=1)

    groups: tuple[
        GroupContext, ...
    ] = Field(min_length=1)

    processing: ProcessingContext

    session_calculation: (
        SessionCalculationView | None
    )

    diagnostics: tuple[
        DiagnosticContext, ...
    ] = ()

    comprehension_findings: tuple[
        ReportingFindingContext, ...
    ] = Field(min_length=1)

    available_evidence: tuple[
        EvidenceReference, ...
    ] = Field(min_length=1)

    required_disclosures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_input_identifiers(self) -> Self:
        def require_unique(
            label: str,
            identifiers: list[object],
        ) -> None:
            if len(identifiers) != len(
                set(identifiers)
            ):
                raise ValueError(
                    f"{label} identifiers must be unique."
                )

        require_unique(
            "Finding",
            [
                finding.finding_id
                for finding
                in self.comprehension_findings
            ],
        )

        require_unique(
            "Diagnostic",
            [
                diagnostic.diagnostic_id
                for diagnostic in self.diagnostics
            ],
        )

        require_unique(
            "Criterion",
            [
                criterion.id
                for criterion in self.criteria
            ],
        )

        require_unique(
            "Alternative",
            [
                alternative.id
                for alternative
                in self.alternatives
            ],
        )

        require_unique(
            "Group",
            [
                group.group.id
                for group in self.groups
            ],
        )

        evidence_identities = [
            reference.identity()
            for reference
            in self.available_evidence
        ]

        if len(evidence_identities) != len(
            set(evidence_identities)
        ):
            raise ValueError(
                "Available evidence references "
                "must be unique."
            )

        for finding in self.comprehension_findings:
            if (
                finding.content_review.status
                == "rejected"
            ):
                raise ValueError(
                    "Rejected findings cannot enter "
                    "reporting input."
                )

        group_ids = {
            group.group.id
            for group in self.groups
        }

        for finding in self.comprehension_findings:
            if (
                finding.group_id is not None
                and finding.group_id not in group_ids
            ):
                raise ValueError(
                    "A reporting finding references an "
                    "unknown stakeholder group."
                )
        known_entity_ids = set(group_ids)

        if self.session_calculation is not None:
            known_entity_ids.add(
                self.session_calculation.entity_id
            )

        for diagnostic in self.diagnostics:
            unknown = (
                set(diagnostic.entity_ids)
                - known_entity_ids
            )

            if unknown:
                raise ValueError(
                    "A reporting diagnostic references "
                    "an entity outside the reporting input."
                )

        available_evidence = {
            reference.identity()
            for reference in self.available_evidence
        }

        for finding in self.comprehension_findings:
            unavailable = {
                reference.identity()
                for reference in finding.draft.evidence
            } - available_evidence

            if unavailable:
                raise ValueError(
                    "A reporting finding cites evidence that is "
                    "not available in the reporting input."
                )

        return self


class ReportClaim(Record):
    claim_type: ReportClaimType
    statement: str = Field(min_length=1)

    evidence: tuple[
        EvidenceReference, ...
    ]

    source_finding_ids: tuple[str, ...] = ()

    assumptions: tuple[str, ...] = ()
    uncertainties: tuple[Uncertainty, ...] = ()
    caveats: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        evidence_ids = [
            item.identity()
            for item in self.evidence
        ]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(
                "A report claim cannot repeat evidence."
            )

        if len(self.source_finding_ids) != len(
            set(self.source_finding_ids)
        ):
            raise ValueError(
                "A report claim cannot repeat source findings."
            )

        if (
            self.claim_type != "evidence_gap"
            and not self.evidence
        ):
            raise ValueError(
                f"{self.claim_type} claims require evidence."
            )

        return self


class ReportSection(Record):
    heading: str = Field(min_length=1)
    claims: tuple[ReportClaim, ...] = Field(min_length=1)


class DataExplanationReport(Record):
    schema_version: Literal["reporting-response-v1"] = "reporting-response-v1"

    executive_summary: ReportSection
    decision_context: ReportSection
    aggregate_results: ReportSection
    stakeholder_patterns: ReportSection
    methodology: ReportSection
    limitations: ReportSection

    agreement: ReportSection | None = None
    influence: ReportSection | None = None
    robustness: ReportSection | None = None
