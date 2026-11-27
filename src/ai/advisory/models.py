from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.ai.comprehension.models import ContentReview, EvidenceReference
from src.ai.models.analysis import Record
from src.ai.reporting.models import DataExplanationReport, DataExplanationReportingInput


ImprovementCategory = Literal[
    "objective_clarification",
    "criterion_definition",
    "criterion_coverage",
    "alternative_description",
    "data_quality",
    "preference_consistency",
    "stakeholder_representation",
    "aggregation_assumptions",
    "constraint_coverage",
    "sensitivity_testing",
    "supporting_evidence",
]


class EvaluatedConfiguration(Record):
    status: Literal["evaluated"]
    configuration: str = Field(min_length=1)
    observed_consequence: str = Field(min_length=1)
    diagnostic_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    removed_criterion_id: str = Field(min_length=1)
    observed_winner_ids: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


class ProposedConfiguration(Record):
    status: Literal["proposed"]
    configuration: str = Field(min_length=1)
    question_to_evaluate: str = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    evaluation_needed: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


ConditionalAlternative = Annotated[
    EvaluatedConfiguration | ProposedConfiguration,
    Field(discriminator="status"),
]


class ImprovementProposal(Record):
    category: ImprovementCategory
    issue: str = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    proposed_modification: str = Field(min_length=1)
    expected_analytical_benefit: str = Field(min_length=1)
    tradeoffs_or_limitations: tuple[str, ...] = Field(min_length=1)

    further_evidence_needed: tuple[str, ...]
    approvals_needed: tuple[str, ...]

    @model_validator(mode="after")
    def validate_proposal(self) -> "ImprovementProposal":
        if len(self.evidence) != len({ref.identity() for ref in self.evidence}):
            raise ValueError("evidence references must have unique identities")

        prose_fields = (
            self.issue,
            self.proposed_modification,
            self.expected_analytical_benefit,
        )
        if any(not value.strip() for value in prose_fields):
            raise ValueError("prose fields must not be blank")

        text_tuples = (
            self.tradeoffs_or_limitations,
            self.further_evidence_needed,
            self.approvals_needed,
        )
        if any(not entry.strip() for values in text_tuples for entry in values):
            raise ValueError("text tuple entries must not be blank")

        return self


class PolicyDecisionAdvisoryResponse(Record):
    schema_version: Literal["policy-decision-advisory-response-v1"] = ("policy-decision-advisory-response-v1")
    improvements: tuple[ImprovementProposal, ...]
    conditional_alternatives: tuple[ConditionalAlternative, ...]


class PolicyDecisionAdvisoryInput(Record):
    schema_version: Literal["policy-decision-advisory-input-v1"] = ("policy-decision-advisory-input-v1")
    source_report_artifact_id: str = Field(min_length=1)
    evidence_snapshot: DataExplanationReportingInput
    explanatory_report: DataExplanationReport
    report_content_review: ContentReview
