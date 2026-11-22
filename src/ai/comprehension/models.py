"""Contracts for a DCL that interprets selected, existing MCDM objects.

Data flow:
    numerical objects -> stage input -> model response -> validated artifact

There is no fact catalog here. ``available_evidence_ids`` contains aliases such as
``group:X:criterion_weights``; the future context boundary resolves those aliases
against fields in the exact request snapshot. It must never duplicate values into
EvidenceFact records or ask the model to generate numerical checks.

Ownership matters:
* Inputs, IDs, hashes, receipts, required limitations and review state belong to
  the application. The model returns only a stage response.
* A response's ``assessment`` is the model's proposal. Reference validation does
  not prove that prose is supported by the cited object.
* Stage handler classes belong in pipeline.py, not in these serializable records.
  Handlers should select input, render prompts, invoke a shared provider adapter,
  validate references/content and construct artifacts using these contracts.

Record freezes model attributes, not every nested MCDM object. The boundary must
copy/serialize inputs and hash the exact saved snapshot before model execution.
Privacy filtering remains a boundary responsibility; types cannot identify all
sensitive prose. This module defines contracts; it does not implement a resolver
or an AI pipeline.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from src.ai.models.analysis import Record
from src.ai.backend import BackendReceipt
from src.mcdm.models.computation import (
    EntityCalculation, EntityMetadata, RankingResult, WeightResult,
)
from src.mcdm.models.domain import Alternative, Criterion, StakeholderGroup
from src.mcdm.models.methods import (
    AggregationMethodConfig, ConsistencyMethodConfig,
    RankingMethodConfig, WeightingMethodConfig,
)


PROCESSING_EVIDENCE_ID = "mcdm_processing"

StageName = Literal[
    "session_understanding", "stakeholder_group_profile", "session_profile",
]
ModelAssessment = Literal["supported", "conditional", "insufficient_evidence"]
ContentStatus = Literal["not_assessed", "accepted", "needs_review", "rejected"]
EvidenceObjectType = Literal[
    "scenario",
    "criterion",
    "alternative",
    "group",
    "calculation",
    "diagnostic",
    "objective",
    "assumption",
    "constraint",
    "processing",
]


class EvidenceReference(Record):
    """A path to a field within an object in the saved input snapshot."""

    object_type: EvidenceObjectType

    object_id: str = Field(
        min_length=1,
        description="Stable identifier of the referenced source object.",
    )

    field_path: tuple[str | int, ...] = Field(
        min_length=1,
        description="Path from the referenced object to the cited field.",
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        for part in self.field_path:
            if isinstance(part, str) and not part.strip():
                raise ValueError(
                    "Evidence path strings must not be empty."
                )

            if isinstance(part, int) and part < 0:
                raise ValueError(
                    "Evidence path indexes must not be negative."
                )

        return self

    def identity(self) -> tuple[object, ...]:
        """Stable identity used for duplicate/reference checks."""

        return (
            self.object_type,
            self.object_id,
            self.field_path,
        )

# Application-owned context. Reuse numerical/domain types rather than importing
# Experiment or DecisionSession, which would bring unrelated settings or people.
class ScenarioContext(Record):
    """The authored decision frame, without participant identities or submissions.

    Construct from resolved session/scenario records. None means information was
    not recorded; an explicitly recorded empty tuple means no items were listed.
    Do not infer objectives or constraints from criterion names.
    """

    scenario_id: str = Field(min_length=1, description="Original scenario identifier.")
    title: str = Field(min_length=1, description="Authored scenario/session title.")
    description: str = Field(description="Authored scope and setting, treated as source data.")
    decision_question: str = Field(min_length=1, description="The decision this evaluation can inform.")
    context: dict[str, JsonValue] | None = Field(default=None, description="Additional authored scenario context, if recorded.")
    objectives: tuple[ScenarioObjective, ...] | None = Field(default=None, description="Explicit objectives, if recorded.")
    assumptions: tuple[ScenarioAssumption, ...] | None = Field(default=None, description="Recorded assumptions, without AI additions.")
    constraints: tuple[ScenarioConstraint, ...] | None = Field(default=None, description="Documented constraints; None does not establish feasibility.")

    @model_validator(mode="after")
    def validate_collection_ids(self) -> Self:
        for label, collection in (
            ("objective", self.objectives),
            ("assumption", self.assumptions),
            ("constraint", self.constraints),
        ):
            if collection is None:
                continue

            ids = [item.id for item in collection]
            if any(not item_id.strip() for item_id in ids):
                raise ValueError(f"{label} identifiers must be nonempty.")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} identifiers must be unique.")

        return self


class ProcessingContext(Record):
    """Only numerical method/configuration objects relevant to interpretation.

    These are the same MCDM configuration types used by ExperimentProcess. This
    selection keeps the AI layer usable outside the experiment runner and avoids
    importing experimental expectations, provider settings or credentials.
    """

    weighting_method: WeightingMethodConfig = Field(description="Selected weighting method and parameters.")
    ranking_method: RankingMethodConfig = Field(description="Selected ranking method and parameters.")
    aggregation_method: AggregationMethodConfig = Field(description="Within/between-group aggregation, power and empty-group policies.")
    consistency_method: ConsistencyMethodConfig = Field(description="Consistency threshold and handling of exceedances.")


class CalculationView(Record):
    """Selected fields of EntityCalculation, with no recalculation or matrices.

    WeightResult contains original preference weights; metadata contains applied
    ranking weights and their source. Keep both when criteria are restricted or
    explicit weights are applied. Stage records distinguish failed/not-run data
    from a valid zero or a completed calculation. None is never proof of stability.
    Pairwise matrices should be a deliberate future input extension if needed.
    """

    entity_id: str = Field(min_length=1, description="Original numerical entity identifier.")
    scope: Literal["group", "session"] = Field(description="Participant results are deliberately excluded from this view.")
    weighting: WeightResult | None = Field(description="Original weights and consistency diagnostics, or unavailable.")
    ranking: RankingResult | None = Field(description="Recorded ranking and scores, or unavailable; ordinal ranks alone do not establish tie policy.")
    metadata: EntityMetadata = Field(description="Participation, powers, applied weights, reuse provenance and stage availability.")

    @classmethod
    def from_calculation(cls, calculation: EntityCalculation) -> Self:
        """Copy the selected source fields; never mutate the numerical result.

        The subclass's scope literal rejects a group/session mismatch. This is a
        projection, not an evidence reconstruction or mathematical verification.
        """
        return cls(
            entity_id=calculation.entity_id, scope=calculation.scope,
            weighting=deepcopy(calculation.weighting), ranking=deepcopy(calculation.ranking),
            metadata=deepcopy(calculation.metadata),
        )


class GroupCalculationView(CalculationView):
    """A group's numerical output, excluding participant identities and matrices."""

    scope: Literal["group"] = "group"


class SessionCalculationView(CalculationView):
    """Aggregate session output; never substitute group weights for session weights."""

    scope: Literal["session"] = "session"


class GroupContext(Record):
    """A configured group and its optional numerical result for session synthesis.

    Retain groups without a result rather than silently excluding them from the
    description. None means no result is supplied; it does not establish a zero
    participant count. Known counts should be recorded in numerical/app metadata.
    """

    group: StakeholderGroup = Field(description="Resolved group definition and configured voting power.")
    calculation: GroupCalculationView | None = Field(description="Matching group calculation, if supplied.")

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if self.calculation is not None and self.calculation.entity_id != self.group.id:
            raise ValueError("Group definition and calculation must identify the same group.")
        return self


class ScenarioObjective(Record):
    """One explicitly recorded objective in the authored decision scenario."""

    id: str = Field(min_length=1, description="Original objective identifier.")
    description: str = Field(min_length=1, description="Recorded objective; do not infer it from criterion names.")
    priority: str | None = Field(default=None, description="Recorded priority, if specified.")
    related_criteria: tuple[str, ...] = Field(default=(), description="Identifiers of criteria explicitly related to this objective.")


class ScenarioAssumption(Record):
    """An assumption captured in the source scenario, not added by interpretation."""

    id: str = Field(min_length=1, description="Original assumption identifier.")
    statement: str = Field(min_length=1, description="Recorded assumption statement.")
    status: str | None = Field(default=None, description="Recorded status, if available.")
    uncertainty: str | None = Field(default=None, description="Recorded uncertainty associated with the assumption, if available.")


class ScenarioConstraint(Record):
    """A documented scenario constraint and its recorded verification details."""

    id: str = Field(min_length=1, description="Original constraint identifier.")
    description: str = Field(min_length=1, description="Recorded constraint description.")
    type: str | None = Field(default=None, description="Recorded constraint category, if specified.")
    parameter: str | None = Field(default=None, description="Parameter governed by the constraint, if specified.")
    operator: str | None = Field(default=None, description="Recorded comparison or relation operator, if specified.")
    value: JsonValue = Field(default=None, description="Recorded constraint value; None means no value was supplied.")
    unit: str | None = Field(default=None, description="Unit associated with the recorded value, if specified.")
    verification_status: str | None = Field(default=None, description="Recorded verification status; not independently verified here.")
    scope: str | None = Field(default=None, description="Recorded scope to which the constraint applies, if specified.")


class DiagnosticContext(Record):
    """Optional, already-computed diagnostic output, supplied as one analysis block.

    This is an extension point for actual agreement, influence or sensitivity
    results, not a registry of individual facts. Keep native numeric structures
    in result. Do not populate results with an AI-generated robustness score.
    A future diagnostic-specific MCDM result type can replace JsonValue when its
    computation contract is established; no new diagnostic algorithm is implied.
    """

    diagnostic_id: str = Field(min_length=1, description="Application-owned identifier for this analysis result.")
    method: str = Field(min_length=1, description="Actual diagnostic method/configuration identifier.")
    entity_ids: tuple[str, ...] = Field(description="Groups/session covered, without participant identifiers.")
    status: Literal["completed", "not_run", "not_evaluable", "failed"]
    result: JsonValue = Field(description="Existing numerical output; null if no result is available.")
    limitations: tuple[str, ...] = Field(default=(), description="Recorded coverage, undefined metrics and tested-condition limits.")

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.status != "completed" and self.result is not None:
            raise ValueError("Unavailable diagnostics must not contain a completed result.")
        if self.status == "completed" and self.result is None:
            raise ValueError("Completed diagnostics require a supplied result.")
        return self


# Model-owned findings. The model neither repeats source values in a check object
# nor assigns application validation status, finding UUIDs or provider receipts.
class Uncertainty(Record):
    """One explicit source of uncertainty; never a generic confidence percentage."""

    kind: Literal[
        "missing_information", "limited_participation", "measurement_uncertainty",
        "preference_uncertainty", "model_sensitivity", "conflicting_evidence",
        "interpretive_ambiguity",
    ]
    detail: str = Field(min_length=1, description="What is uncertain and how it limits this interpretation.")


class InterpretationDraft(Record):
    """A proposed explanation and citations, awaiting application content review.

    Evidence IDs address existing object fields in this request snapshot, e.g.
    group:X:criterion_weights. Their origin is assigned by the application, not
    repeated as a model-authored 'authoritative' label. A resolving citation alone
    does not prove that 'highest', a numerical value or a causal claim is correct.
    qualification has been replaced with explicit assumption/uncertainty/caveat
    fields to avoid two overlapping places for the same limitation.
    """

    statement: str = Field(min_length=1, description="Concise evidence-grounded explanation; do not invent results or motivations.")
    assessment: ModelAssessment = Field(description="Model-proposed support level, not the application's verdict.")
    evidence: tuple[EvidenceReference, ...] = Field(default=(), description="Only aliases supplied in this request; never prior AI prose as independent evidence.")
    assumptions: tuple[str, ...] = Field(description="Additional conditions adopted by this interpretation, not established facts; empty if none.")
    uncertainties: tuple[Uncertainty, ...] = Field(description="Specific unresolved information or analytical uncertainty; empty if none.")
    caveats: tuple[str, ...] = Field(description="Restrictions on interpretation or generalization; empty only when none apply.")

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        keys = [
            reference.identity()
            for reference in self.evidence
        ]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "An interpretation must not repeat evidence."
            )

        if (
            self.assessment in {"supported", "conditional"}
            and not self.evidence
        ):
            raise ValueError(
                f"{self.assessment} interpretations require evidence."
            )

        return self


class PriorInterpretation(Record):
    """A small application-selected view of an earlier finding for a later stage.

    Carry the unchanged draft and its original references/qualifications. The
    boundary must make those original references resolvable in the new snapshot.
    not_assessed or rejected content must never become accepted merely through
    reuse. Exclude rejected findings when assembling downstream requests.
    """

    finding_id: str = Field(min_length=1, description="Application-owned identity of the original finding.")
    source_request_id: str = Field(min_length=1, description="Request that produced the original draft.")
    source_run_id: str = Field(min_length=1, description="Numerical run to which the original finding applies.")
    entity_ids: tuple[str, ...] = Field(description="Application-assigned group/session scope.")
    draft: InterpretationDraft
    content_status: ContentStatus = Field(description="Existing review status; reuse does not upgrade authority.")


class DCLInput(Record):
    """Shared, application-owned input fields; a thin selection of existing objects.

    A handler must serialize/hash/save the exact request and validate aliases
    against it. This contract checks shape and unique identifiers, not semantic
    correctness of source data. It contains no credentials or participant roster.
    """

    schema_version: Literal["dcl-input-v1"] = "dcl-input-v1"
    stage: StageName
    scenario: ScenarioContext
    criteria: tuple[Criterion, ...] = Field(min_length=1, description="Relevant original and active criterion definitions, units and directions; ranking_criterion_ids identifies the applied subset.")
    alternatives: tuple[Alternative, ...] = Field(min_length=1, description="Authored options with existing criterion-performance values.")
    processing: ProcessingContext
    available_evidence: tuple[EvidenceReference, ...] = Field(min_length=1, description="References to fields resolvable against this exact input snapshot; no source values are copied here.")
    required_limitations: tuple[str, ...] = Field(default=(), description="Application-owned disclosures stored independently of the model's wording.")
    prior_interpretations: tuple[PriorInterpretation, ...] = Field(default=(), description="Only task-relevant earlier findings, retaining their original evidence.")

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        for label, values in (
            ("criterion", [c.id for c in self.criteria]),
            ("alternative", [a.id for a in self.alternatives]),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(
                    f"{label} identifiers must be nonempty."
                )

            if len(values) != len(set(values)):
                raise ValueError(
                    f"{label} identifiers must be unique."
                )
        evidence_keys = [
            reference.identity()
            for reference in self.available_evidence
        ]

        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError(
                "Available evidence references must be unique."
            )
        
        return self


class SessionUnderstandingInput(DCLInput):
    """Stage 1 input: authored setting and entities, not earlier AI conclusions.

    Numerical traces are unnecessary for basic scenario comprehension. Use the
    original definitions; report missing objectives/constraints explicitly.
    """

    stage: Literal["session_understanding"] = "session_understanding"
    groups: tuple[StakeholderGroup, ...] = Field(min_length=1, description="Configured constituencies, without participant identities.")

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if len({g.id for g in self.groups}) != len(self.groups):
            raise ValueError("Configured group identifiers must be unique.")
        if self.prior_interpretations:
            raise ValueError("Session understanding starts from source inputs, not prior AI findings.")
        return self


class GroupProfileInput(DCLInput):
    """Stage 2 input: one group's definition and selected numerical result.

    One group per request isolates scope and limits response size. Prior session
    understanding is optional context, never a substitute for original evidence.
    Only include diagnostics that were actually computed and cover this group.
    """

    stage: Literal["stakeholder_group_profile"] = "stakeholder_group_profile"
    group: StakeholderGroup
    calculation: GroupCalculationView | None = Field(description="Matching original numerical result, or explicitly unavailable.")
    diagnostics: tuple[DiagnosticContext, ...] = Field(default=(), description="Optional supplied analyses; absence establishes no agreement or robustness.")

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if self.calculation is not None and self.calculation.entity_id != self.group.id:
            raise ValueError("Group input and calculation must identify the same group.")
        return self


class SessionProfileInput(DCLInput):
    """Stage 3 input: session result and group data for cross-group synthesis.

    Include relevant earlier findings through prior_interpretations, not entire
    previous request/receipt histories. Keep raw supporting objects available.
    Configured voting power, effective power and measured influence are distinct.
    """

    stage: Literal["session_profile"] = "session_profile"
    groups: tuple[GroupContext, ...] = Field(min_length=1)
    calculation: SessionCalculationView | None = Field(description="Aggregate numerical result; None requires an evidence-gap explanation.")
    diagnostics: tuple[DiagnosticContext, ...] = Field(default=(), description="Actual cross-group, agreement, influence or robustness analyses.")

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if len({g.group.id for g in self.groups}) != len(self.groups):
            raise ValueError("Session profile group identifiers must be unique.")
        return self


# Only these small responses are generated by the model. Stage/request identity,
# source values, hashes, warning retention and review state remain outside them.
class SessionUnderstandingResponse(Record):
    """Interpret the source decision setting without constructing a fact catalog.

    Citation coverage of criteria/alternatives/groups is a stage-handler check;
    schema validity alone cannot establish that every entity was considered.
    """

    decision_frame: InterpretationDraft
    objectives: tuple[InterpretationDraft, ...] = Field(min_length=1)
    alternatives: tuple[InterpretationDraft, ...] = Field(min_length=1)
    criteria: tuple[InterpretationDraft, ...] = Field(min_length=1)
    stakeholder_structure: tuple[InterpretationDraft, ...] = Field(min_length=1)
    relationships: tuple[InterpretationDraft, ...] = Field(
        min_length=1,
        description=(
            "Recorded or explicitly conditional relationships. "
            "When none can be established, include an insufficient-evidence finding."
        )
    )
    constraints: tuple[InterpretationDraft, ...] = Field(min_length=1)
    authored_assumptions: tuple[InterpretationDraft, ...] = Field(min_length=1)
    methodological_assumptions: tuple[InterpretationDraft, ...] = Field(min_length=1)
    missing_or_ambiguous_information: tuple[InterpretationDraft, ...] = Field(min_length=1)
    evidence_limitations: tuple[InterpretationDraft, ...] = Field(min_length=1)


class GroupProfileResponse(Record):
    """Interpret one group's modeled preferences, without repeating math/checks.

    priorities/ranking/representation are required topics, even when their
    interpretation is that evidence is unavailable. diagnostics and limitations
    may be empty; do not force the model to invent an unperformed analysis.
    """

    priorities: InterpretationDraft = Field(description="Highest/lowest importance and ties; priorities do not establish motivations.")
    ranking: InterpretationDraft = Field(description="Recorded ordering/scores and ties, or unavailable ranking evidence.")
    representation: InterpretationDraft = Field(description="Counts and configured/effective power, with representativeness limits.")
    diagnostics: tuple[InterpretationDraft, ...] = Field(description="Only interpretations of supplied agreement, variation, influence or robustness results.")
    limitations: tuple[InterpretationDraft, ...] = Field(description="Additional evidence-linked gaps beyond application-required disclosures.")


class SessionProfileResponse(Record):
    """Synthesize aggregate results and cross-group findings.

    Optional analytical topics use empty arrays when evidence is unavailable;
    explain material omissions in limitations. Matching group rankings cannot
    establish individual unanimity or population consensus. No advisory/policy
    recommendation is authorized by this comprehension artifact.
    """

    priorities: InterpretationDraft
    ranking: InterpretationDraft
    group_comparisons: tuple[InterpretationDraft, ...] = Field(min_length=1)
    agreement: tuple[InterpretationDraft, ...]
    influence: tuple[InterpretationDraft, ...]
    robustness: tuple[InterpretationDraft, ...]
    limitations: tuple[InterpretationDraft, ...] = Field(min_length=1)


DCLStageInput = Annotated[
    SessionUnderstandingInput | GroupProfileInput | SessionProfileInput,
    Field(discriminator="stage"),
]
DCLStageResponse = SessionUnderstandingResponse | GroupProfileResponse | SessionProfileResponse


# Application-owned provenance, audit and review contracts. Never ask the model
# to reproduce these fields in its response schema.
class DCLRequestMetadata(Record):
    """Bind a request to its numerical run, AI repetition and exact source snapshot.

    input_hash replaces the old fact-package evidence_hash. The context boundary
    must define canonical JSON serialization and fingerprint the complete input.
    reported origin permits controlled experimental result interventions without
    relabeling a supplied/modified result as independently verified truth.
    """

    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    ai_repetition: int = Field(ge=1)
    stage: StageName
    group_id: str | None = Field(default=None, description="Required only for the single-group stage.")
    prompt_version: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_origin: Literal["calculated", "reported"] = "calculated"

    @model_validator(mode="after")
    def validate_stage_scope(self) -> Self:
        if self.stage == "stakeholder_group_profile":
            if self.group_id is None or not self.group_id.strip():
                raise ValueError("A group-profile request requires its group ID.")
        elif self.group_id is not None:
            raise ValueError("Session-level requests must not claim a single-group scope.")
        return self


class ValidationIssue(Record):
    """One application-detected problem, without leaking raw sensitive payloads."""

    code: str = Field(min_length=1, description="Stable programmatic reason, e.g. unknown_evidence_id.")
    message: str = Field(min_length=1, description="Sanitized human-readable explanation.")
    response_path: tuple[str | int, ...] = Field(default=(), description="Location of the affected response field, if known.")


class ContentReview(Record):
    """An application/human review outcome, independent of model self-assessment.

    An evaluator's judgment is not independent mathematical authority. Preserve
    the reviewer/method and reasons; never promote a finding merely because its
    citations exist. Runtime review/approval workflows are not implemented here.
    """

    status: ContentStatus = "not_assessed"
    reviewer: str | None = Field(default=None, description="Authorized reviewer or named evaluation method.")
    reviewed_at: AwareDatetime | None = None
    issues: tuple[ValidationIssue, ...] = ()


class StoredInterpretation(Record):
    """An unchanged model draft with application identity and validation/review state.

    response_path identifies its slot in the original stage response. Stable IDs
    permit later references without assigning authority to repeated AI prose.
    An invalid/unassessed finding is retained for audit, not accepted downstream.
    """

    finding_id: str = Field(min_length=1)
    response_path: tuple[str | int, ...] = Field(min_length=1)
    draft: InterpretationDraft
    reference_status: Literal["not_checked", "validated", "invalid"] = "not_checked"
    content_review: ContentReview = Field(default_factory=ContentReview)


class PromptMessage(Record):
    """A captured textual message in the exact order sent to the provider."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(description="Exact sent text, subject to the application's privacy policy.")


class DCLCallError(Record):
    """A failure belonging to one attempt; it must remain associated with that call."""

    phase: Literal["input", "serialization", "provider", "response", "parsing", "schema", "references", "content"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1, description="Sanitized explanation; never include credentials/provider headers.")
    issues: tuple[ValidationIssue, ...] = ()


class DCLCallRecord(Record):
    """Application audit for one attempted call, including unsuccessful attempts.

    Input-only failures can have no messages/receipt; malformed JSON can have raw
    response but no parsed response. Save exact input/messages and raw output for
    inspection. Redaction is a boundary responsibility; do not serialize a client.
    Retry policy and provider invocation belong to the future runtime, not here.
    """

    call_id: str = Field(min_length=1)
    request: DCLRequestMetadata
    attempt: int = Field(ge=1)
    input_snapshot: DCLStageInput
    messages: tuple[PromptMessage, ...] = ()
    status: Literal["completed", "failed"]
    started_at: AwareDatetime
    completed_at: AwareDatetime
    receipt: BackendReceipt | None = None
    raw_response: str | None = None
    parsed_response: DCLStageResponse | None = None
    error: DCLCallError | None = None
    redacted: bool = False

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        if self.request.stage != self.input_snapshot.stage:
            raise ValueError("Call metadata and input must identify the same stage.")
        if isinstance(self.input_snapshot, GroupProfileInput) and self.request.group_id != self.input_snapshot.group.id:
            raise ValueError("Call metadata and input must identify the same group.")
        response_types = {
            "session_understanding": SessionUnderstandingResponse,
            "stakeholder_group_profile": GroupProfileResponse,
            "session_profile": SessionProfileResponse,
        }
        if self.parsed_response is not None and not isinstance(self.parsed_response, response_types[self.request.stage]):
            raise ValueError("Parsed response does not match the requested stage.")
        if self.completed_at < self.started_at:
            raise ValueError("Call completion cannot precede its start.")
        if self.status == "failed" and self.error is None:
            raise ValueError("A failed call requires an explicit error.")
        if self.status == "completed":
            if (
                self.receipt is None
                or self.raw_response is None
                or self.parsed_response is None
                or self.error is not None
            ):
                raise ValueError(
                    "A completed call requires receipt, raw response, "
                    "parsed response, and no error."
                )

        if self.status == "failed" and self.parsed_response is not None:
            raise ValueError(
                "A failed call cannot contain a parsed response."
            )
        return self


InputT = TypeVar("InputT", bound=DCLInput)
ResponseT = TypeVar("ResponseT", bound=Record)


class DCLArtifact(Record, Generic[InputT, ResponseT]):
    """Typed, application-owned envelope for one stage's output and source snapshot.

    A stored artifact is not automatically approved interpretation. Keep the raw
    response in the associated DCLCallRecord, preserve drafts unchanged, and store
    application-required limitations independently of model wording. Inputs are
    retained for reproducibility, not resent wholesale to subsequent stages.
    """

    schema_version: Literal["dcl-artifact-v1"] = "dcl-artifact-v1"
    artifact_id: str = Field(min_length=1)
    request: DCLRequestMetadata
    input_snapshot: InputT
    response: ResponseT
    receipt: BackendReceipt
    findings: tuple[StoredInterpretation, ...] = Field(description="Application-extracted drafts and per-finding review records.")
    required_limitations: tuple[str, ...] = Field(description="Exact input disclosures preserved independently of generated output.")
    content_review: ContentReview = Field(default_factory=ContentReview)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.request.stage != self.input_snapshot.stage:
            raise ValueError("Artifact metadata and input must identify the same stage.")
        if isinstance(self.input_snapshot, GroupProfileInput) and self.request.group_id != self.input_snapshot.group.id:
            raise ValueError("Artifact request and input must identify the same group.")
        if self.required_limitations != self.input_snapshot.required_limitations:
            raise ValueError("Required input limitations must be preserved unchanged.")
        if len({f.finding_id for f in self.findings}) != len(self.findings):
            raise ValueError("Stored finding identifiers must be unique.")
        seen_paths = set()
        for finding in self.findings:
            if finding.response_path in seen_paths:
                raise ValueError("Each stored finding must identify a distinct response slot.")
            seen_paths.add(finding.response_path)
            current = self.response
            for part in finding.response_path:
                if isinstance(part, str) and isinstance(current, Record) and part in type(current).model_fields:
                    current = getattr(current, part)
                elif isinstance(part, int) and isinstance(current, tuple) and 0 <= part < len(current):
                    current = current[part]
                else:
                    raise ValueError("Stored finding path does not identify a response field.")
            if not isinstance(current, InterpretationDraft) or current != finding.draft:
                raise ValueError("A stored draft must match its original response slot unchanged.")
            if finding.reference_status == "validated":
                available = {
                    reference.identity()
                    for reference in self.input_snapshot.available_evidence
                }

                cited = {
                    reference.identity()
                    for reference in finding.draft.evidence
                }

                if not cited <= available:
                    raise ValueError("A validated finding cannot cite evidence absent from its input.")
        return self


class SessionUnderstandingArtifact(DCLArtifact[SessionUnderstandingInput, SessionUnderstandingResponse]):
    """Stored stage 1 result with its typed input, original drafts and review state."""


class GroupProfileArtifact(DCLArtifact[GroupProfileInput, GroupProfileResponse]):
    """Stored result of one group-profile call, not an entire group's AI history."""


class SessionProfileArtifact(DCLArtifact[SessionProfileInput, SessionProfileResponse]):
    """Stored synthesis with original mathematical inputs and qualified interpretations."""


class DCLRunState(Record):
    """Available DCL artifacts for one numerical run and one AI repetition.

    None/empty means a stage has not produced an artifact; a failed attempt is
    recorded in DCLCallRecord rather than invented as a successful profile. A
    handler must decide content readiness before downstream reuse. This state
    deliberately contains no numerical recalculation, provider client or secrets.
    """

    run_id: str = Field(min_length=1)
    ai_repetition: int = Field(ge=1)
    session_understanding: SessionUnderstandingArtifact | None = None
    group_profiles: tuple[GroupProfileArtifact, ...] = ()
    session_profile: SessionProfileArtifact | None = None

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        artifacts = [a for a in (self.session_understanding, *self.group_profiles, self.session_profile) if a is not None]
        if any(a.request.run_id != self.run_id or a.request.ai_repetition != self.ai_repetition for a in artifacts):
            raise ValueError("DCL state cannot mix numerical runs or AI repetitions.")
        group_ids = [a.request.group_id for a in self.group_profiles]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("DCL state must contain at most one current artifact per group.")
        return self
