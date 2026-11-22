from __future__ import annotations

from pydantic import TypeAdapter

from src.ai.comprehension.models import (
    DiagnosticContext,
    EvidenceReference,
    GroupCalculationView,
    GroupProfileInput,
    PriorInterpretation,
    ProcessingContext,
    ScenarioContext,
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

        # TODO: These fields are not currently present in the experiment schema.
        objectives=None,
        assumptions=None,
        constraints=None,
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


def _session_understanding_evidence(
    *,
    scenario: ScenarioContext,
    criteria: tuple[Criterion, ...],
    alternatives: tuple[Alternative, ...],
    groups: tuple[StakeholderGroup, ...],
) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = [
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
        if calculation.weighting is not None:
            references.extend((
                EvidenceReference(
                    object_type="calculation",
                    object_id=calculation.entity_id,
                    field_path=(
                        "weighting",
                        "criterion_weights",
                    ),
                ),
                EvidenceReference(
                    object_type="calculation",
                    object_id=calculation.entity_id,
                    field_path=(
                        "weighting",
                        "consistency_ratio",
                    ),
                ),
                EvidenceReference(
                    object_type="calculation",
                    object_id=calculation.entity_id,
                    field_path=(
                        "weighting",
                        "consistency_threshold",
                    ),
                ),
            ))

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

        references.extend((
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "participant_count",
                ),
            ),
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "voting_power",
                ),
            ),
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "effective_voting_power",
                ),
            ),
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "ranking_criterion_ids",
                ),
            ),
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "applied_criterion_weights",
                ),
            ),
            EvidenceReference(
                object_type="calculation",
                object_id=calculation.entity_id,
                field_path=(
                    "metadata",
                    "ranking_weight_source",
                ),
            ),
        ))

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
        groups=groups,
        processing=processing,
        available_evidence=available_evidence,
        required_limitations=required_limitations,

        # Enforced by SessionUnderstandingInput as well.
        prior_interpretations=(),
    )


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