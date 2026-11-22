from __future__ import annotations

from pydantic import TypeAdapter

from src.ai.comprehension.models import (
    PROCESSING_EVIDENCE_ID,
    DCLRunState,
    DiagnosticContext,
    EvidenceReference,
    GroupCalculationView,
    GroupContext,
    GroupProfileArtifact,
    GroupProfileInput,
    PriorInterpretation,
    ProcessingContext,
    ScenarioAssumption,
    ScenarioConstraint,
    ScenarioContext,
    ScenarioObjective,
    SessionCalculationView,
    SessionProfileInput,
    SessionUnderstandingArtifact,
    SessionUnderstandingInput,
)
from src.experiments.results import RunResult
from src.mcdm.models.domain import (
    Alternative,
    Criterion,
    StakeholderGroup,
)
from src.mcdm.models.methods import (
    AggregationMethodConfig,
    ConsistencyMethodConfig,
    RankingMethodConfig,
    WeightingMethodConfig,
)


_WEIGHTING_CONFIG_ADAPTER = TypeAdapter(
    WeightingMethodConfig
)

_RANKING_CONFIG_ADAPTER = TypeAdapter(
    RankingMethodConfig
)

_AGGREGATION_CONFIG_ADAPTER = TypeAdapter(
    AggregationMethodConfig
)

_CONSISTENCY_CONFIG_ADAPTER = TypeAdapter(
    ConsistencyMethodConfig
)

_SCENARIO_OBJECTIVES_ADAPTER = TypeAdapter(
    tuple[ScenarioObjective, ...]
)

_SCENARIO_ASSUMPTIONS_ADAPTER = TypeAdapter(
    tuple[ScenarioAssumption, ...]
)

_SCENARIO_CONSTRAINTS_ADAPTER = TypeAdapter(
    tuple[ScenarioConstraint, ...]
)


def require_completed_numerical_result(
    result: RunResult,
) -> None:
    if (result.execution.get("numerical_status") != "completed"):
        raise ValueError(
            "DCL input construction requires a completed "
            "numerical result."
        )

    if not isinstance(result.inputs, dict):
        raise ValueError(
            "DCL input construction requires resolved inputs."
        )


def _resolved_objects(
    result: RunResult,
) -> tuple[
    ScenarioContext,
    tuple[Criterion, ...],
    tuple[Alternative, ...],
    tuple[StakeholderGroup, ...],
    ProcessingContext,
]:
    inputs = result.inputs
    session = inputs.get("session")
    processing = inputs.get("processing")

    if not isinstance(session, dict):
        raise ValueError(
            "Resolved inputs are missing the session snapshot."
        )

    if not isinstance(processing, dict):
        raise ValueError(
            "Resolved inputs are missing processing configuration."
        )

    scenario = ScenarioContext(
        scenario_id=inputs["scenario_id"],
        title=session["title"],
        description=inputs["scenario_description"],
        decision_question=inputs[
            "scenario_policy_question"
        ],
        context=inputs.get("scenario_context"),
        objectives=_optional_scenario_collection(
            _SCENARIO_OBJECTIVES_ADAPTER,
            inputs.get("scenario_objectives"),
        ),
        assumptions=_optional_scenario_collection(
            _SCENARIO_ASSUMPTIONS_ADAPTER,
            inputs.get("scenario_assumptions"),
        ),
        constraints=_optional_scenario_collection(
            _SCENARIO_CONSTRAINTS_ADAPTER,
            inputs.get("scenario_constraints"),
        ),
    )

    criteria = tuple(
        Criterion.model_validate(value)
        for value in session["criteria"]
    )

    alternatives = tuple(
        Alternative.model_validate(value)
        for value in session["alternatives"]
    )

    groups = tuple(
        StakeholderGroup.model_validate(value)
        for value in session["stakeholder_groups"]
    )

    processing_context = ProcessingContext(
        weighting_method=(
            _WEIGHTING_CONFIG_ADAPTER.validate_python(
                processing["weighting_method"]
            )
        ),
        ranking_method=(
            _RANKING_CONFIG_ADAPTER.validate_python(
                processing["ranking_method"]
            )
        ),
        aggregation_method=(
            _AGGREGATION_CONFIG_ADAPTER.validate_python(
                processing["aggregation_method"]
            )
        ),
        consistency_method=(
            _CONSISTENCY_CONFIG_ADAPTER.validate_python(
                processing["consistency_method"]
            )
        ),
    )

    return (
        scenario,
        criteria,
        alternatives,
        groups,
        processing_context,
    )


def _artifact_entity_ids(
    artifact: (
        SessionUnderstandingArtifact
        | GroupProfileArtifact
    ),
) -> tuple[str, ...]:
    if isinstance(
        artifact,
        SessionUnderstandingArtifact,
    ):
        return (
            artifact.input_snapshot
            .scenario.scenario_id,
        )

    group_id = artifact.request.group_id

    if group_id is None:
        raise ValueError(
            "A group-profile artifact requires "
            "a group identifier."
        )

    return (group_id,)


def _optional_scenario_collection(
    adapter: TypeAdapter,
    value: object,
):
    if value is None:
        return None

    return adapter.validate_python(
        value,
        strict=False,
    )


def _session_understanding_evidence(
    *,
    scenario: ScenarioContext,
    criteria: tuple[Criterion, ...],
    alternatives: tuple[Alternative, ...],
    groups: tuple[StakeholderGroup, ...],
) -> tuple[EvidenceReference, ...]:
    references = [
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("title",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("description",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("decision_question",),
        ),

        # These collection-level paths support statements about missing or explicitly empty sections.
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("context",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("objectives",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("assumptions",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("constraints",),
        ),
    ]

    if scenario.context is not None:
        for key in sorted(scenario.context):
            references.append(
                EvidenceReference(
                    object_type="scenario",
                    object_id=scenario.scenario_id,
                    field_path=("context", key),
                )
            )

    for objective in scenario.objectives or ():
        for field_name in (
            "description",
            "priority",
            "related_criteria",
        ):
            references.append(
                EvidenceReference(
                    object_type="objective",
                    object_id=objective.id,
                    field_path=(field_name,),
                )
            )

    for assumption in scenario.assumptions or ():
        for field_name in (
            "statement",
            "status",
            "uncertainty",
        ):
            references.append(
                EvidenceReference(
                    object_type="assumption",
                    object_id=assumption.id,
                    field_path=(field_name,),
                )
            )

    for constraint in scenario.constraints or ():
        for field_name in (
            "description",
            "type",
            "parameter",
            "operator",
            "value",
            "unit",
            "verification_status",
            "scope",
        ):
            references.append(
                EvidenceReference(
                    object_type="constraint",
                    object_id=constraint.id,
                    field_path=(field_name,),
                )
            )
    for criterion in criteria:
        for field_name in (
            "name",
            "description",
            "direction",
            "unit",
        ):
            references.append(
                EvidenceReference(
                    object_type="criterion",
                    object_id=criterion.id,
                    field_path=(field_name,),
                )
            )

    for alternative in alternatives:
        for field_name in (
            "name",
            "description",
            "criterion_values",
        ):
            references.append(
                EvidenceReference(
                    object_type="alternative",
                    object_id=alternative.id,
                    field_path=(field_name,),
                )
            )

    for group in groups:
        for field_name in (
            "name",
            "description",
            "voting_power",
        ):
            references.append(
                EvidenceReference(
                    object_type="group",
                    object_id=group.id,
                    field_path=(field_name,),
                )
            )

    for field_name in (
        "weighting_method",
        "ranking_method",
        "aggregation_method",
        "consistency_method",
    ):
        references.append(
            EvidenceReference(
                object_type="processing",
                object_id=PROCESSING_EVIDENCE_ID,
                field_path=(field_name,),
            )
        )

    return tuple(references)

def _session_calculation(
    result: RunResult,
) -> SessionCalculationView | None:
    calculation = result.mcdm.session

    if calculation is None:
        return None

    if calculation.scope != "session":
        raise ValueError(
            "The session calculation has a "
            "non-session scope."
        )

    return SessionCalculationView.from_calculation(
        calculation
    )

def _session_group_contexts(
    *,
    result: RunResult,
    groups: tuple[StakeholderGroup, ...],
) -> tuple[GroupContext, ...]:
    calculations = _group_calculations(result)

    configured_ids = {
        group.id
        for group in groups
    }

    unknown_ids = (
        set(calculations)
        - configured_ids
    )

    if unknown_ids:
        raise ValueError(
            "Group calculations reference unknown "
            "configured groups."
        )

    return tuple(
        GroupContext(
            group=group,
            calculation=calculations.get(group.id),
        )
        for group in groups
    )

def _validate_session_diagnostics(
    *,
    diagnostics: tuple[
        DiagnosticContext, ...
    ],
    groups: tuple[GroupContext, ...],
    calculation: (
        SessionCalculationView | None
    ),
) -> None:
    diagnostic_ids = [
        diagnostic.diagnostic_id
        for diagnostic in diagnostics
    ]

    if len(diagnostic_ids) != len(
        set(diagnostic_ids)
    ):
        raise ValueError(
            "Session diagnostics must have "
            "unique identifiers."
        )

    known_entity_ids = {
        group.group.id
        for group in groups
    }

    if calculation is not None:
        known_entity_ids.add(
            calculation.entity_id
        )

    for diagnostic in diagnostics:
        if not diagnostic.entity_ids:
            raise ValueError(
                "A session diagnostic must identify "
                "the entities it covers."
            )

        unknown = (
            set(diagnostic.entity_ids)
            - known_entity_ids
        )

        if unknown:
            raise ValueError(
                "A session diagnostic references "
                "unknown entities."
            )


def _session_profile_evidence(
    *,
    scenario: ScenarioContext,
    criteria: tuple[Criterion, ...],
    alternatives: tuple[Alternative, ...],
    groups: tuple[GroupContext, ...],
    calculation: (
        SessionCalculationView | None
    ),
    diagnostics: tuple[
        DiagnosticContext, ...
    ],
) -> tuple[EvidenceReference, ...]:
    references = list(
        _session_understanding_evidence(
            scenario=scenario,
            criteria=criteria,
            alternatives=alternatives,
            groups=tuple(
                group.group
                for group in groups
            ),
        )
    )

    for group in groups:
        if group.calculation is not None:
            references.extend(
                _calculation_evidence(
                    group.calculation
                )
            )

    if calculation is not None:
        references.extend(
            _calculation_evidence(calculation)
        )

    for diagnostic in diagnostics:
        for field_name in (
            "method",
            "entity_ids",
            "status",
            "result",
            "limitations",
        ):
            references.append(
                EvidenceReference(
                    object_type="diagnostic",
                    object_id=(
                        diagnostic.diagnostic_id
                    ),
                    field_path=(field_name,),
                )
            )

    return tuple(references)


def _group_calculations(
    result: RunResult,
) -> dict[str, GroupCalculationView]:
    indexed = {}

    for calculation in result.mcdm.groups:
        if calculation.scope != "group":
            raise ValueError("The group calculation collection contains a non-group result.")

        if calculation.entity_id in indexed:
            raise ValueError("Duplicate group calculation identifier.")

        indexed[calculation.entity_id] = (
            GroupCalculationView.from_calculation(
                calculation
            )
        )

    return indexed


def _group_profile_evidence(
    *,
    scenario: ScenarioContext,
    criteria: tuple[Criterion, ...],
    alternatives: tuple[Alternative, ...],
    group: StakeholderGroup,
    calculation: GroupCalculationView | None,
    diagnostics: tuple[DiagnosticContext, ...],
) -> tuple[EvidenceReference, ...]:
    references = [
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("description",),
        ),
        EvidenceReference(
            object_type="scenario",
            object_id=scenario.scenario_id,
            field_path=("decision_question",),
        ),
        EvidenceReference(
            object_type="group",
            object_id=group.id,
            field_path=("description",),
        ),
        EvidenceReference(
            object_type="group",
            object_id=group.id,
            field_path=("voting_power",),
        ),
    ]

    for criterion in criteria:
        references.extend((
            EvidenceReference(
                object_type="criterion",
                object_id=criterion.id,
                field_path=("description",),
            ),
            EvidenceReference(
                object_type="criterion",
                object_id=criterion.id,
                field_path=("direction",),
            ),
            EvidenceReference(
                object_type="criterion",
                object_id=criterion.id,
                field_path=("unit",),
            ),
        ))

    for alternative in alternatives:
        references.extend((
            EvidenceReference(
                object_type="alternative",
                object_id=alternative.id,
                field_path=("description",),
            ),
            EvidenceReference(
                object_type="alternative",
                object_id=alternative.id,
                field_path=("criterion_values",),
            ),
        ))

    if calculation is not None:
        references.extend(
            _calculation_evidence(calculation)
        )

    for diagnostic in diagnostics:
        references.extend((
            EvidenceReference(
                object_type="diagnostic",
                object_id=diagnostic.diagnostic_id,
                field_path=("status",),
            ),
            EvidenceReference(
                object_type="diagnostic",
                object_id=diagnostic.diagnostic_id,
                field_path=("result",),
            ),
            EvidenceReference(
                object_type="diagnostic",
                object_id=diagnostic.diagnostic_id,
                field_path=("limitations",),
            ),
        ))

    return tuple(references)


def _validate_group_diagnostics(
    *,
    group: StakeholderGroup,
    diagnostics: tuple[DiagnosticContext, ...],
) -> None:
    diagnostic_ids = [
        diagnostic.diagnostic_id
        for diagnostic in diagnostics
    ]

    if len(diagnostic_ids) != len(set(diagnostic_ids)):
        raise ValueError(
            "Group diagnostics must have unique identifiers."
        )

    for diagnostic in diagnostics:
        if group.id not in diagnostic.entity_ids:
            raise ValueError(
                "A group-profile diagnostic must cover "
                "the target group."
            )


def _calculation_evidence(
    calculation: (
        GroupCalculationView
        | SessionCalculationView
    ),
) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = []

    if calculation.weighting is not None:
        for field_name in (
            "criterion_weights",
            "consistency_ratio",
            "consistency_threshold",
        ):
            references.append(
                EvidenceReference(
                    object_type="calculation",
                    object_id=calculation.entity_id,
                    field_path=(
                        "weighting",
                        field_name,
                    ),
                )
            )

    if calculation.ranking is not None:
        references.append(
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "ranking",
                    "rankings",
                ),
            )
        )

    for field_name in (
        "participant_count",
        "voting_power",
        "effective_voting_power",
        "ranking_criterion_ids",
        "applied_criterion_weights",
        "ranking_weight_source",
        "weighting_source",
        "stages",
    ):
        references.append(
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    field_name,
                ),
            )
        )

    return tuple(references)


def build_session_understanding_input(
    *,
    result: RunResult,
    required_limitations: tuple[str, ...] = (),
) -> SessionUnderstandingInput:
    require_completed_numerical_result(result)

    (
        scenario,
        criteria,
        alternatives,
        groups,
        processing,
    ) = _resolved_objects(result)

    group_ids = [group.id for group in groups]

    if len(group_ids) != len(set(group_ids)):
        raise ValueError(
            "Configured group identifiers must be unique."
        )

    available_evidence = (
        _session_understanding_evidence(
            scenario=scenario,
            criteria=criteria,
            alternatives=alternatives,
            groups=groups,
        )
    )

    return SessionUnderstandingInput(
        scenario=scenario,
        criteria=criteria,
        alternatives=alternatives,
        processing=processing,
        groups=groups,
        available_evidence=available_evidence,
        required_limitations=required_limitations,
        prior_interpretations=(),
    )


def build_session_profile_input(
    *,
    result: RunResult,
    state: DCLRunState,
    diagnostics: tuple[
        DiagnosticContext, ...
    ] = (),
    required_limitations: tuple[str, ...] = (),
) -> SessionProfileInput:
    if state.run_id != result.run_id:
        raise ValueError(
            "Session-profile state and numerical "
            "result must identify the same run."
        )

    if state.session_profile is not None:
        raise ValueError(
            "Session-profile input cannot be built "
            "from a state that already contains a "
            "session-profile artifact."
        )

    if state.session_understanding is None:
        raise ValueError(
            "Session-profile input requires successful "
            "session understanding."
        )

    require_completed_numerical_result(result)

    (
        scenario,
        criteria,
        alternatives,
        configured_groups,
        processing,
    ) = _resolved_objects(result)

    groups = _session_group_contexts(
        result=result,
        groups=configured_groups,
    )

    calculation = _session_calculation(result)

    _validate_session_diagnostics(
        diagnostics=diagnostics,
        groups=groups,
        calculation=calculation,
    )

    calculation_ids = [
        group.calculation.entity_id
        for group in groups
        if group.calculation is not None
    ]

    if calculation is not None:
        calculation_ids.append(
            calculation.entity_id
        )

    if len(calculation_ids) != len(
        set(calculation_ids)
    ):
        raise ValueError(
            "Group and session calculations must "
            "have unique identifiers."
        )

    limitations = list(required_limitations)

    if calculation is None:
        limitations.append(
            "No aggregate session calculation was "
            "produced; aggregate priorities and "
            "ranking are unavailable."
        )

    missing_group_ids = [
        group.group.id
        for group in groups
        if group.calculation is None
    ]

    if missing_group_ids:
        limitations.append(
            "No group calculation was produced for: "
            + ", ".join(missing_group_ids)
            + "."
        )

    available_evidence = (
        _session_profile_evidence(
            scenario=scenario,
            criteria=criteria,
            alternatives=alternatives,
            groups=groups,
            calculation=calculation,
            diagnostics=diagnostics,
        )
    )

    prior_interpretations = (
        _session_profile_priors(
            state=state,
            available_evidence=available_evidence,
        )
    )

    return SessionProfileInput(
        scenario=scenario,
        criteria=criteria,
        alternatives=alternatives,
        processing=processing,
        groups=groups,
        calculation=calculation,
        diagnostics=diagnostics,
        available_evidence=available_evidence,
        required_limitations=tuple(limitations),

        # Added only after safe downstream selection.
        prior_interpretations=(
            prior_interpretations
        ),
    )


def _session_profile_priors(
    *,
    state: DCLRunState,
    available_evidence: tuple[
        EvidenceReference, ...
    ],
) -> tuple[PriorInterpretation, ...]:
    session_artifact = (
        state.session_understanding
    )

    if session_artifact is None:
        raise ValueError(
            "Session-profile construction requires "
            "a session-understanding artifact."
        )

    artifacts = (
        session_artifact,
        *state.group_profiles,
    )

    available = {
        reference.identity()
        for reference in available_evidence
    }

    priors: list[PriorInterpretation] = []
    finding_ids: set[str] = set()

    for artifact in artifacts:
        if (
            artifact.request.run_id
            != state.run_id
            or artifact.request.ai_repetition
            != state.ai_repetition
        ):
            raise ValueError(
                "Prior artifacts cannot mix runs "
                "or AI repetitions."
            )

        if (
            artifact.content_review.status
            == "rejected"
        ):
            continue

        entity_ids = _artifact_entity_ids(
            artifact
        )

        for finding in artifact.findings:
            if (
                finding.reference_status
                != "validated"
            ):
                continue

            if (
                finding.content_review.status
                == "rejected"
            ):
                continue

            if finding.finding_id in finding_ids:
                raise ValueError(
                    "Prior finding identifiers "
                    "must be unique."
                )

            cited = {
                reference.identity()
                for reference
                in finding.draft.evidence
            }

            if not cited <= available:
                raise ValueError(
                    "A prior finding cites evidence "
                    "that is unavailable in the "
                    "session-profile snapshot."
                )

            finding_ids.add(
                finding.finding_id
            )

            priors.append(
                PriorInterpretation(
                    finding_id=(
                        finding.finding_id
                    ),
                    source_request_id=(
                        artifact.request.request_id
                    ),
                    source_run_id=(
                        artifact.request.run_id
                    ),
                    entity_ids=entity_ids,

                    # Preserve the model output exactly.
                    draft=finding.draft,

                    # Preserve review status.
                    content_status=(
                        finding.content_review.status
                    ),
                )
            )

    return tuple(priors)


def build_group_profile_input(
    *,
    result: RunResult,
    group_id: str,
    diagnostics: tuple[DiagnosticContext, ...] = (),
    prior_interpretations: tuple[
        PriorInterpretation,
        ...
    ] = (),
    required_limitations: tuple[str, ...] = (),
) -> GroupProfileInput:
    require_completed_numerical_result(result)

    (
        scenario,
        criteria,
        alternatives,
        groups,
        processing,
    ) = _resolved_objects(result)

    groups_by_id = {
        group.id: group
        for group in groups
    }

    if len(groups_by_id) != len(groups):
        raise ValueError(
            "Configured group identifiers must be unique."
        )

    try:
        group = groups_by_id[group_id]
    except KeyError:
        raise ValueError(
            f"Unknown configured group: {group_id}"
        ) from None

    calculations = _group_calculations(result)

    unknown_calculation_groups = (
        set(calculations)
        - set(groups_by_id)
    )

    if unknown_calculation_groups:
        raise ValueError(
            "Group calculations reference unknown "
            "configured groups."
        )

    calculation = calculations.get(group_id)

    _validate_group_diagnostics(
        group=group,
        diagnostics=diagnostics,
    )

    limitations = list(required_limitations)

    if calculation is None:
        limitations.append(
            "No group calculation was produced; numerical "
            "preferences and representation metadata are "
            "unavailable for this group."
        )
    
    available_evidence = _group_profile_evidence(
        scenario=scenario,
        criteria=criteria,
        alternatives=alternatives,
        group=group,
        calculation=calculation,
        diagnostics=diagnostics,
    )

    return GroupProfileInput(
        scenario=scenario,
        criteria=criteria,
        alternatives=alternatives,
        processing=processing,
        available_evidence=available_evidence,
        required_limitations=tuple(limitations),
        prior_interpretations=prior_interpretations,
        group=group,
        calculation=calculation,
        diagnostics=diagnostics,
    )


def build_group_profile_inputs(
    *,
    result: RunResult,
    diagnostics_by_group: dict[
        str,
        tuple[DiagnosticContext, ...],
    ] | None = None,
    required_limitations: tuple[str, ...] = (),
) -> tuple[GroupProfileInput, ...]:
    require_completed_numerical_result(result)

    session = result.inputs["session"]
    diagnostics_by_group = diagnostics_by_group or {}

    group_ids = tuple(
        group["id"]
        for group in session["stakeholder_groups"]
    )

    unknown_diagnostics = (set(diagnostics_by_group) - set(group_ids))

    if unknown_diagnostics:
        raise ValueError("Diagnostics were supplied for unknown groups.")

    return tuple(
        build_group_profile_input(
            result=result,
            group_id=group_id,
            diagnostics=diagnostics_by_group.get(
                group_id,
                (),
            ),
            required_limitations=required_limitations,
        )
        for group_id in group_ids
    )
