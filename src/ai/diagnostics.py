"""Adapters that expose deterministic diagnostics at the AI boundary."""

from src.ai.comprehension.models import DiagnosticContext
from src.experiments.results import RunResult
from src.mcdm.diagnostics import (
    DiagnosticNotEvaluableError,
    calculate_alternative_removal_effects,
    calculate_bounded_weight_perturbations,
    calculate_consistency_warnings,
    calculate_criterion_removal_effects,
    calculate_cross_group_rank_correlations,
    calculate_leave_one_group_out,
    calculate_score_rank_margins,
    calculate_winner_agreement,
    calculate_cross_group_weight_dispersion,
)

DEFAULT_ABSOLUTE_WEIGHT_DELTA = 0.05


def _ranking_context_target(session_id: str, group_id: str | None) -> tuple[str, str]:
    if group_id is None:
        return "session", session_id
    return f"group:{group_id}", group_id


def build_rank_diagnostic_contexts(
    result: RunResult,
) -> tuple[DiagnosticContext, ...]:
    correlations = calculate_cross_group_rank_correlations(
        result
    )
    winners = calculate_winner_agreement(result)
    group_ids = tuple(
        vector.group_id
        for vector in correlations.rank_vectors
    )
    session = result.mcdm.session

    if session is None:
        raise ValueError(
            "Winner agreement requires a session calculation."
        )

    winner_entity_ids = (*group_ids, session.entity_id)

    return (
        DiagnosticContext(
            diagnostic_id=(
                "session:cross_group_rank_correlations"
            ),
            kind="agreement",
            method=(
                "tie_aware_pairwise_spearman_"
                "kendall_tau_b_v1"
            ),
            entity_ids=group_ids,
            status="completed",
            result=correlations.model_dump(mode="json"),
            limitations=(
                "Rank correlation measures similarity between "
                "modeled group rankings; it does not establish "
                "participant agreement or population consensus.",
            ),
        ),
        DiagnosticContext(
            diagnostic_id=(
                "session:winner_agreement_counts"
            ),
            kind="agreement",
            method="modeled_group_winner_overlap_v1",
            entity_ids=winner_entity_ids,
            status="completed",
            result=winners.model_dump(mode="json"),
            limitations=(
                "Winner overlap counts modeled group-level top "
                "rankings and does not measure public support or "
                "stakeholder unanimity.",
            ),
        ),
    )


def build_weight_dispersion_context(
    result: RunResult,
) -> DiagnosticContext:
    dispersion = calculate_cross_group_weight_dispersion(
        result
    )

    return DiagnosticContext(
        diagnostic_id=(
            "session:cross_group_weight_dispersion"
        ),
        kind="agreement",
        method=(
            "criterion_weight_range_"
            "population_standard_deviation_v1"
        ),
        entity_ids=dispersion.evaluated_group_ids,
        status="completed",
        result=dispersion.model_dump(mode="json"),
        limitations=(
            "Weight dispersion compares modeled group-level "
            "criterion weights. It does not measure within-group "
            "participant agreement or population preference "
            "variation.",
            "The session weight is supplied as a contextual "
            "reference and is excluded from dispersion statistics.",
        ),
    )


def build_score_rank_margin_context(
    result: RunResult,
) -> DiagnosticContext:
    margins = calculate_score_rank_margins(result)

    return DiagnosticContext(
        diagnostic_id="session:score_rank_margins",
        kind="other",
        method=(
            "tolerance_aware_adjacent_"
            "score_rank_margins_v1"
        ),
        entity_ids=(
            *margins.evaluated_group_ids,
            margins.session_id,
        ),
        status="completed",
        result=margins.model_dump(mode="json"),
        limitations=(
            "Score and rank margins describe separation "
            "in the calculated TOPSIS rankings. They do "
            "not establish stability under changed inputs "
            "or weights.",
        ),
    )


def build_leave_one_group_out_context(
    result: RunResult,
) -> DiagnosticContext:
    analysis = calculate_leave_one_group_out(result)

    return DiagnosticContext(
        diagnostic_id="session:leave_one_group_out",
        kind="influence",
        method=(
            "one_at_a_time_group_omission_"
            "reaggregation_v1"
        ),
        entity_ids=(
            *analysis.evaluated_group_ids,
            analysis.session_id,
        ),
        status="completed",
        result=analysis.model_dump(mode="json"),
        limitations=(
            "Leave-one-group-out results describe a modeled "
            "counterfactual in which one represented group is "
            "excluded and remaining voting powers are "
            "renormalized.",
            "Observed changes do not establish causal influence, "
            "political power, or the consequences of actual "
            "stakeholder exclusion.",
            "One-at-a-time omissions do not measure interactions "
            "among multiple omitted groups.",
        ),
    )


def build_cross_group_diagnostic_contexts(
    result: RunResult,
) -> tuple[DiagnosticContext, ...]:
    try:
        return (
            *build_rank_diagnostic_contexts(result),
            build_weight_dispersion_context(result),
            build_leave_one_group_out_context(result),
        )
    except DiagnosticNotEvaluableError as error:
        group_ids = _evaluated_group_ids(result)
        session = result.mcdm.session

        if not group_ids:
            return ()

        if session is None:
            raise ValueError(
                "Winner agreement requires a session calculation."
            )

        winner_entity_ids = (
            *group_ids,
            session.entity_id,
        )

        return (
            DiagnosticContext(
                diagnostic_id=(
                    "session:cross_group_rank_correlations"
                ),
                kind="agreement",
                method=(
                    "tie_aware_pairwise_spearman_"
                    "kendall_tau_b_v1"
                ),
                entity_ids=group_ids,
                status="not_evaluable",
                result=None,
                limitations=(
                    str(error),
                    "Rank correlation measures similarity "
                    "between modeled group rankings; it does "
                    "not establish participant agreement or "
                    "population consensus.",
                ),
            ),
            DiagnosticContext(
                diagnostic_id=(
                    "session:winner_agreement_counts"
                ),
                kind="agreement",
                method="modeled_group_winner_overlap_v1",
                entity_ids=winner_entity_ids,
                status="not_evaluable",
                result=None,
                limitations=(
                    str(error),
                    "Winner overlap counts modeled group-level "
                    "top rankings and does not measure public "
                    "support or stakeholder unanimity.",
                ),
            ),
            DiagnosticContext(
                diagnostic_id=(
                    "session:cross_group_weight_dispersion"
                ),
                kind="agreement",
                method=(
                    "criterion_weight_range_"
                    "population_standard_deviation_v1"
                ),
                entity_ids=group_ids,
                status="not_evaluable",
                result=None,
                limitations=(
                    str(error),
                    "Weight dispersion compares modeled "
                    "group-level criterion weights. It does "
                    "not measure within-group participant "
                    "agreement or population preference "
                    "variation.",
                    "The session weight is supplied as a "
                    "contextual reference and is excluded "
                    "from dispersion statistics.",
                ),
            ),
            DiagnosticContext(
                diagnostic_id="session:leave_one_group_out",
                kind="influence",
                method=(
                    "one_at_a_time_group_omission_"
                    "reaggregation_v1"
                ),
                entity_ids=winner_entity_ids,
                status="not_evaluable",
                result=None,
                limitations=(
                    str(error),
                    "Leave-one-group-out analysis requires at "
                    "least two evaluated groups.",
                    "Omission effects do not establish causal "
                    "or real-world stakeholder influence.",
                ),
            ),
        )


def build_consistency_warning_context(
    result: RunResult,
) -> DiagnosticContext:
    warnings = calculate_consistency_warnings(result)

    return DiagnosticContext(
        diagnostic_id="session:consistency_warnings",
        kind="other",
        method="stored_ahp_consistency_threshold_v1",
        entity_ids=(
            *warnings.evaluated_group_ids,
            warnings.session_id,
        ),
        status="completed",
        result=warnings.model_dump(mode="json"),
        limitations=(
            "AHP consistency ratios measure internal "
            "judgment consistency; they do not establish "
            "preference validity, decision quality, or "
            "stakeholder representativeness.",
            "Participant-level results are summarized by "
            "group without participant identifiers.",
        ),
    )


def build_bounded_weight_perturbation_context(
    result: RunResult,
    *,
    absolute_weight_delta: float = DEFAULT_ABSOLUTE_WEIGHT_DELTA,
    group_id: str | None = None,
) -> DiagnosticContext:
    analysis = calculate_bounded_weight_perturbations(
        result,
        absolute_weight_delta=absolute_weight_delta,
        group_id=group_id,
    )
    namespace, entity_id = _ranking_context_target(analysis.session_id, analysis.group_id)
    completed_cases = tuple(
        case for case in analysis.cases if case.status == "completed"
    )
    limitations = (
        "Each scenario changes one applied ranking weight by "
        "the requested absolute amount and redistributes the remaining "
        "weight proportionally. The alternatives and their criterion "
        "values stay fixed.",
        "Conclusions apply only to the completed endpoints. Intermediate "
        "and simultaneous weight changes are outside this analysis.",
    )

    if not completed_cases:
        reasons = tuple(dict.fromkeys(
            case.reason for case in analysis.cases if case.reason is not None
        ))
        return DiagnosticContext(
            diagnostic_id=f"{namespace}:bounded_weight_perturbations",
            kind="sensitivity",
            method="one_at_a_time_proportional_weight_redistribution_v1",
            entity_ids=(entity_id,),
            status="not_evaluable",
            result=None,
            limitations=(
                *limitations,
                f"No endpoint completed at an absolute weight change of "
                f"{analysis.absolute_weight_delta:g}.",
                *reasons,
            ),
        )

    payload = analysis.model_dump(
        mode="json", exclude={"group_id"} if analysis.group_id is None else None,
    )
    payload["completed_case_count"] = len(completed_cases)
    payload["unavailable_case_count"] = len(analysis.cases) - len(completed_cases)
    payload["winner_changed_case_count"] = sum(
        case.winner_changed is True for case in completed_cases
    )

    for case, case_payload in zip(analysis.cases, payload["cases"], strict=True):
        if case.status != "completed":
            continue

        baseline_and_perturbed = tuple(zip(
            analysis.baseline_entries, case.entries, strict=True,
        ))
        case_payload["maximum_absolute_score_delta"] = max(
            abs(perturbed.score - baseline.score)
            for baseline, perturbed in baseline_and_perturbed
        )
        case_payload["maximum_absolute_rank_delta"] = max(
            abs(perturbed.statistical_rank - baseline.statistical_rank)
            for baseline, perturbed in baseline_and_perturbed
        )

    return DiagnosticContext(
        diagnostic_id=f"{namespace}:bounded_weight_perturbations",
        kind="sensitivity",
        method="one_at_a_time_proportional_weight_redistribution_v1",
        entity_ids=(entity_id,),
        status="completed",
        result=payload,
        limitations=limitations,
    )


def build_criterion_removal_context(
    result: RunResult, *, group_id: str | None = None,
) -> DiagnosticContext:
    analysis = calculate_criterion_removal_effects(result, group_id=group_id)
    namespace, entity_id = _ranking_context_target(analysis.session_id, analysis.group_id)
    completed = tuple(case for case in analysis.cases if case.status == "completed")
    limitations = (
        "Each case removes one criterion from the TOPSIS ranking "
        "and renormalizes the remaining applied ranking weights. "
        "AHP judgments, alternatives, and criterion values stay fixed.",
        "Scores describe each recalculated TOPSIS model. Conclusions "
        "cover the completed single-criterion removals.",
    )

    payload = None
    if completed:
        payload = analysis.model_dump(
            mode="json", exclude={"group_id"} if analysis.group_id is None else None,
        )
        payload["completed_case_count"] = len(completed)
        payload["unavailable_case_count"] = len(analysis.cases) - len(completed)
        payload["winner_changed_criterion_ids"] = [
            case.removed_criterion_id for case in completed if case.winner_changed
        ]
        for case, case_payload in zip(analysis.cases, payload["cases"], strict=True):
            if case.status == "completed":
                case_payload["maximum_absolute_score_delta"] = max(
                    abs(change.score_delta) for change in case.alternative_changes
                )
                case_payload["maximum_absolute_rank_delta"] = max(
                    abs(change.statistical_rank_delta) for change in case.alternative_changes
                )
    else:
        limitations += tuple(
            f"{case.removed_criterion_id}: {case.reason}" for case in analysis.cases
        )

    return DiagnosticContext(
        diagnostic_id=f"{namespace}:criterion_removal_effects",
        kind="sensitivity",
        method="one_at_a_time_ranking_criterion_removal_v1",
        entity_ids=(entity_id,),
        status="completed" if completed else "not_evaluable",
        result=payload,
        limitations=limitations,
    )


def build_alternative_removal_context(
    result: RunResult, *, group_id: str | None = None,
) -> DiagnosticContext:
    analysis = calculate_alternative_removal_effects(result, group_id=group_id)
    namespace, entity_id = _ranking_context_target(analysis.session_id, analysis.group_id)
    completed = tuple(case for case in analysis.cases if case.status == "completed")
    limitations = (
        "Each case removes one alternative and recalculates TOPSIS normalization, "
        "ideal points, and scores over the survivors with the same applied weights.",
        "Compare reranked winners with the baseline winner set among survivors. "
        "Rank deltas use baseline ranks recomputed over those same survivors.",
        "Score deltas describe a change in the TOPSIS comparison set. "
        "Conclusions cover the completed single-alternative removals.",
    )
    payload = None
    if completed:
        payload = analysis.model_dump(
            mode="json", exclude={"group_id"} if analysis.group_id is None else None,
        )
        payload["completed_case_count"] = len(completed)
        payload["unavailable_case_count"] = len(analysis.cases) - len(completed)
        payload["baseline_winner_removed_ids"] = [
            case.removed_alternative_id for case in analysis.cases if case.baseline_winner_removed
        ]
        payload["survivor_winner_changed_removal_ids"] = [
            case.removed_alternative_id for case in completed if case.survivor_winner_changed
        ]
        for case, case_payload in zip(analysis.cases, payload["cases"], strict=True):
            if case.status == "completed":
                case_payload["maximum_absolute_score_delta"] = max(
                    abs(change.score_delta) for change in case.alternative_changes
                )
                case_payload["maximum_absolute_rank_delta"] = max(
                    abs(change.statistical_rank_delta) for change in case.alternative_changes
                )
    else:
        limitations += tuple(
            f"{case.removed_alternative_id}: {case.reason}" for case in analysis.cases
        )

    return DiagnosticContext(
        diagnostic_id=f"{namespace}:alternative_removal_effects",
        kind="sensitivity",
        method="one_at_a_time_alternative_removal_v1",
        entity_ids=(entity_id,),
        status="completed" if completed else "not_evaluable",
        result=payload,
        limitations=limitations,
    )


def build_group_diagnostic_contexts(
    result: RunResult,
    *,
    absolute_weight_delta: float = DEFAULT_ABSOLUTE_WEIGHT_DELTA,
) -> dict[str, tuple[DiagnosticContext, ...]]:
    """Build diagnostics for each group with a completed numerical calculation."""
    consistency = calculate_consistency_warnings(result)
    assessments = {
        entry.entity_id: entry for entry in consistency.entity_assessments
        if entry.scope == "group"
    }
    summaries = {
        summary.group_id: summary for summary in consistency.participant_group_summaries
    }
    try:
        margins = calculate_score_rank_margins(result)
    except DiagnosticNotEvaluableError as error:
        margins_by_group = {}
        margin_gap = (str(error),)
    else:
        margins_by_group = {
            entry.entity_id: entry for entry in margins.entities if entry.scope == "group"
        }
        margin_gap = ()

    by_group = {}
    for group_id in consistency.evaluated_group_ids:
        score_margins = margins_by_group.get(group_id)
        by_group[group_id] = (
            DiagnosticContext(
                diagnostic_id=f"group:{group_id}:score_rank_margins",
                kind="other",
                method="tolerance_aware_adjacent_score_rank_margins_v1",
                entity_ids=(group_id,),
                status="completed" if score_margins is not None else "not_evaluable",
                result=(
                    score_margins.model_dump(mode="json")
                    if score_margins is not None else None
                ),
                limitations=(
                    "Margins describe separation in the group's baseline TOPSIS ranking.",
                    *margin_gap,
                ),
            ),
            DiagnosticContext(
                diagnostic_id=f"group:{group_id}:consistency_warnings",
                kind="other",
                method="stored_ahp_consistency_threshold_v1",
                entity_ids=(group_id,),
                status="completed",
                result={
                    "group_assessment": assessments[group_id].model_dump(mode="json"),
                    "participant_summary": summaries[group_id].model_dump(mode="json"),
                },
                limitations=(
                    "AHP consistency measures internal judgment consistency. "
                    "Participant evidence is summarized without participant identifiers.",
                ),
            ),
            build_bounded_weight_perturbation_context(
                result, absolute_weight_delta=absolute_weight_delta, group_id=group_id,
            ),
            build_criterion_removal_context(result, group_id=group_id),
            build_alternative_removal_context(result, group_id=group_id),
        )
    return by_group


def build_session_diagnostic_contexts(
    result: RunResult,
    *,
    absolute_weight_delta: float = DEFAULT_ABSOLUTE_WEIGHT_DELTA,
) -> tuple[DiagnosticContext, ...]:
    diagnostics = list(
        build_cross_group_diagnostic_contexts(result)
    )

    try:
        diagnostics.append(build_score_rank_margin_context(result))
    except DiagnosticNotEvaluableError as error:
        group_ids = _evaluated_group_ids(result)
        session = result.mcdm.session

        if session is not None:
            diagnostics.append(DiagnosticContext(
                diagnostic_id=(
                    "session:score_rank_margins"
                ),
                kind="other",
                method=(
                    "tolerance_aware_adjacent_"
                    "score_rank_margins_v1"
                ),
                entity_ids=(
                    *group_ids,
                    session.entity_id,
                ),
                status="not_evaluable",
                result=None,
                limitations=(
                    str(error),
                    "Score and rank margins require "
                    "multiple alternatives.",
                ),
            ))

    diagnostics.append(build_consistency_warning_context(result))
    diagnostics.append(build_bounded_weight_perturbation_context(
        result,
        absolute_weight_delta=absolute_weight_delta,
    ))
    diagnostics.append(build_criterion_removal_context(result))
    diagnostics.append(build_alternative_removal_context(result))

    return tuple(diagnostics)


def _evaluated_group_ids(
    result: RunResult,
) -> tuple[str, ...]:
    return tuple(
        calculation.entity_id
        for calculation in result.mcdm.groups
    )
