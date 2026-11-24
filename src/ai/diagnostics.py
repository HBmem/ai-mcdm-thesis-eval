"""Adapters that expose deterministic diagnostics at the AI boundary."""

from src.ai.comprehension.models import DiagnosticContext
from src.experiments.results import RunResult
from src.mcdm.diagnostics import (
    DiagnosticNotEvaluableError,
    calculate_consistency_warnings,
    calculate_cross_group_rank_correlations,
    calculate_score_rank_margins,
    calculate_winner_agreement,
    calculate_cross_group_weight_dispersion,
)


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


def build_cross_group_diagnostic_contexts(
    result: RunResult,
) -> tuple[DiagnosticContext, ...]:
    try:
        return (
            *build_rank_diagnostic_contexts(result),
            build_weight_dispersion_context(result),
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


def build_session_diagnostic_contexts(
    result: RunResult,
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

    return tuple(diagnostics)


def _evaluated_group_ids(
    result: RunResult,
) -> tuple[str, ...]:
    return tuple(
        calculation.entity_id
        for calculation in result.mcdm.groups
    )
