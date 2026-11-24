"""Shared deterministic helpers and calculations for MCDM diagnostics."""

from __future__ import annotations

import math
from itertools import combinations
from numbers import Real
from typing import Mapping, TYPE_CHECKING

import numpy as np
from pydantic import TypeAdapter
from scipy.stats import kendalltau, spearmanr
from statistics import fmean, pstdev

from src.mcdm.models.computation import EntityCalculation, RankingResult, WeightResult
from src.mcdm.models.diagnostics import (
    AdjacentScoreRankMargin,
    AlternativeWinnerSupport,
    ConsistencyWarningResult,
    CrossGroupRankCorrelationResult,
    DiagnosticRankEntry,
    EffectiveGroupPower,
    EntityConsistencyAssessment,
    EntityScoreRankMargins,
    GroupParticipantConsistencySummary,
    GroupRankCorrelation,
    GroupRankVector,
    GroupWinnerSet,
    LeaveOneOutAlternativeChange,
    LeaveOneOutCriterionWeightChange,
    LeaveOneGroupOutCase,
    LeaveOneGroupOutResult,
    ScoreRankMarginResult,
    WinnerAgreementResult,
    CriterionWeightDispersion,
    CrossGroupWeightDispersionResult,
    GroupCriterionWeight,
)
from src.mcdm.models.methods import AggregationMethodConfig, ConsistencyMethodConfig, RankingMethodConfig, WeightingMethodConfig
from src.mcdm.models.session import DecisionSession
from src.mcdm.service import MCDMService, build_ranking_weighting

if TYPE_CHECKING:
    from src.experiments.results import RunResult


TIE_ATOL = 1e-12
_UNDEFINED_CORRELATION_REASON = (
    "Correlation is undefined because at least one "
    "aligned ranking contains only ties."
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


class DiagnosticNotEvaluableError(ValueError):
    """The input is valid, but insufficient for this diagnostic."""


def _validated_tie_atol(tie_atol: float) -> float:
    if (
        isinstance(tie_atol, bool)
        or not isinstance(tie_atol, Real)
        or not math.isfinite(tie_atol)
        or tie_atol < 0
    ):
        raise ValueError(
            "tie_atol must be finite and nonnegative."
        )

    return float(tie_atol)


def _validated_scores(
    scores: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("Scores must be a nonempty mapping.")

    validated: dict[str, float] = {}

    for alternative_id, score in scores.items():
        if not isinstance(alternative_id, str) or not alternative_id:
            raise ValueError(
                "Score keys must be nonempty alternative IDs."
            )

        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
        ):
            raise ValueError("Scores must be finite real numbers.")

        validated[alternative_id] = float(score)

    return validated


def _weights_by_criterion(
    calculation: EntityCalculation,
    *,
    criterion_ids: tuple[str, ...],
) -> dict[str, float]:
    """Extract one complete criterion-weight vector."""

    weighting = calculation.weighting

    if weighting is None:
        raise ValueError(
            f"Entity '{calculation.entity_id}' has no weighting result."
        )

    weights: dict[str, float] = {}

    for item in weighting.criterion_weights:
        criterion_id = item.criterion_id

        if criterion_id in weights:
            raise ValueError(
                f"Weighting '{calculation.entity_id}' contains duplicate "
                f"criterion ID '{criterion_id}'."
            )

        weight = float(item.weight)

        if not math.isfinite(weight):
            raise ValueError(
                f"Weighting '{calculation.entity_id}' must contain "
                "finite criterion weights."
            )

        weights[criterion_id] = weight

    configured_ids = set(criterion_ids)
    actual_ids = set(weights)

    if actual_ids != configured_ids:
        missing_ids = tuple(
            criterion_id
            for criterion_id in criterion_ids
            if criterion_id not in actual_ids
        )
        extra_ids = tuple(sorted(actual_ids - configured_ids))
        parts = []

        if missing_ids:
            parts.append(f"missing: {', '.join(missing_ids)}")

        if extra_ids:
            parts.append(f"unexpected: {', '.join(extra_ids)}")

        raise ValueError(
            f"Weighting '{calculation.entity_id}' must cover the "
            f"configured criteria exactly once ({'; '.join(parts)})."
        )

    return {
        criterion_id: weights[criterion_id]
        for criterion_id in criterion_ids
    }

 
def statistical_ranks(
    scores: Mapping[str, float],
    *,
    tie_atol: float = TIE_ATOL,
) -> dict[str, float]:
    """Return descending, tie-aware average statistical ranks."""
    values = _validated_scores(scores)
    tolerance = _validated_tie_atol(tie_atol)
    ordered = sorted(
        values,
        key=lambda alternative_id: (
            -values[alternative_id],
            alternative_id,
        ),
    )

    ranks: dict[str, float] = {}
    start = 0

    while start < len(ordered):
        end = start + 1
        highest_score = values[ordered[start]]

        while (
            end < len(ordered)
            and highest_score - values[ordered[end]]
            <= tolerance
        ):
            end += 1

        average_rank = (start + 1 + end) / 2

        for alternative_id in ordered[start:end]:
            ranks[alternative_id] = float(average_rank)

        start = end

    return ranks


def winner_ids(
    scores: Mapping[str, float],
    *,
    tie_atol: float = TIE_ATOL,
) -> tuple[str, ...]:
    """Return every alternative tied with the maximum score."""
    values = _validated_scores(scores)
    tolerance = _validated_tie_atol(tie_atol)
    best_score = max(values.values())

    return tuple(sorted(
        alternative_id
        for alternative_id, score in values.items()
        if best_score - score <= tolerance
    ))


def _cross_group_diagnostic_inputs(
    result: RunResult,
) -> tuple[
    tuple[str, ...],
    tuple[EntityCalculation, ...],
    EntityCalculation,
]:
    inputs = _diagnostic_inputs(result)
    _, group_calculations, _ = inputs

    if len(group_calculations) < 2:
        raise DiagnosticNotEvaluableError(
            "Cross-group diagnostics require at least "
            "two evaluated groups."
        )

    return inputs


def _diagnostic_inputs(
    result: RunResult,
) -> tuple[
    tuple[str, ...],
    tuple[EntityCalculation, ...],
    EntityCalculation,
]:
    if result.execution.get("numerical_status") != "completed":
        raise ValueError(
            "Deterministic diagnostics require completed "
            "numerical results."
        )

    result.validate_numerical_complete()
    session = result.inputs["session"]
    alternative_ids = tuple(
        alternative["id"]
        for alternative in session["alternatives"]
    )
    calculations_by_id = {
        calculation.entity_id: calculation
        for calculation in result.mcdm.groups
    }
    configured_group_ids = tuple(
        group["id"]
        for group in session["stakeholder_groups"]
    )
    group_calculations = tuple(
        calculations_by_id[group_id]
        for group_id in configured_group_ids
        if group_id in calculations_by_id
    )

    session_calculation = result.mcdm.session

    if session_calculation is None:
        raise ValueError(
            "Deterministic diagnostics require a session "
            "calculation."
        )

    return (
        alternative_ids,
        group_calculations,
        session_calculation,
    )


def _scores_by_alternative(
    calculation: EntityCalculation,
    *,
    alternative_ids: tuple[str, ...],
) -> dict[str, float]:
    ranking = getattr(calculation, "ranking", None)

    if ranking is None:
        raise ValueError(
            f"Group '{calculation.entity_id}' has no completed ranking."
        )

    scores = {
        row.alternative_id: float(
            row.closeness_coefficient
        )
        for row in ranking.rankings
    }

    if (
        len(scores) != len(ranking.rankings)
        or set(scores) != set(alternative_ids)
    ):
        raise ValueError(
            f"Ranking '{calculation.entity_id}' must cover the "
            "configured alternatives exactly once."
        )

    return scores


def _rank_vectors(
    result: RunResult,
    *,
    tie_atol: float,
) -> tuple[
    tuple[str, ...],
    tuple[GroupRankVector, ...],
    EntityCalculation,
]:
    alternative_ids, calculations, session = (
        _cross_group_diagnostic_inputs(result)
    )
    vectors = []

    for calculation in calculations:
        scores = _scores_by_alternative(
            calculation,
            alternative_ids=alternative_ids,
        )
        ranks = statistical_ranks(
            scores,
            tie_atol=tie_atol,
        )
        vectors.append(GroupRankVector(
            group_id=calculation.entity_id,
            entries=tuple(
                DiagnosticRankEntry(
                    alternative_id=alternative_id,
                    score=scores[alternative_id],
                    statistical_rank=ranks[alternative_id],
                )
                for alternative_id in alternative_ids
            ),
            winner_ids=winner_ids(
                scores,
                tie_atol=tie_atol,
            ),
        ))

    return alternative_ids, tuple(vectors), session


def _correlation_value(value: object) -> float | None:
    coefficient = float(value)

    if not math.isfinite(coefficient):
        return None

    return max(-1.0, min(1.0, coefficient))


def calculate_cross_group_rank_correlations(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
) -> CrossGroupRankCorrelationResult:
    """Calculate pairwise rank similarity for evaluated groups."""
    tolerance = _validated_tie_atol(tie_atol)
    alternative_ids, vectors, _ = _rank_vectors(
        result,
        tie_atol=tolerance,
    )
    correlations = []

    for left, right in combinations(vectors, 2):
        left_by_id = {
            entry.alternative_id: entry.statistical_rank
            for entry in left.entries
        }
        right_by_id = {
            entry.alternative_id: entry.statistical_rank
            for entry in right.entries
        }
        left_ranks = tuple(
            left_by_id[alternative_id]
            for alternative_id in alternative_ids
        )
        right_ranks = tuple(
            right_by_id[alternative_id]
            for alternative_id in alternative_ids
        )

        if (
            len(alternative_ids) < 2
            or len(set(left_ranks)) < 2
            or len(set(right_ranks)) < 2
        ):
            rho = None
            tau = None
            undefined_reason = _UNDEFINED_CORRELATION_REASON
        else:
            rho = _correlation_value(
                spearmanr(left_ranks, right_ranks).statistic
            )
            tau = _correlation_value(
                kendalltau(
                    left_ranks,
                    right_ranks,
                    variant="b",
                ).statistic
            )
            undefined_reason = None

            if rho is None or tau is None:
                rho = None
                tau = None
                undefined_reason = _UNDEFINED_CORRELATION_REASON

        correlations.append(GroupRankCorrelation(
            left_group_id=left.group_id,
            right_group_id=right.group_id,
            alternative_ids=alternative_ids,
            spearman_rho=rho,
            kendall_tau_b=tau,
            undefined_reason=undefined_reason,
        ))

    return CrossGroupRankCorrelationResult(
        tie_atol=tolerance,
        rank_vectors=vectors,
        pairwise_correlations=tuple(correlations),
    )


def calculate_winner_agreement(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
) -> WinnerAgreementResult:
    """Count modeled group-level winner overlap by alternative."""
    tolerance = _validated_tie_atol(tie_atol)
    alternative_ids, vectors, session = _rank_vectors(
        result,
        tie_atol=tolerance,
    )
    evaluated_group_ids = tuple(
        vector.group_id for vector in vectors
    )
    group_winners = tuple(
        GroupWinnerSet(
            group_id=vector.group_id,
            winner_ids=vector.winner_ids,
        )
        for vector in vectors
    )
    winner_sets = {
        winners.group_id: set(winners.winner_ids)
        for winners in group_winners
    }
    support = tuple(
        AlternativeWinnerSupport(
            alternative_id=alternative_id,
            supporting_group_ids=tuple(
                group_id
                for group_id in evaluated_group_ids
                if alternative_id in winner_sets[group_id]
            ),
            group_count=sum(
                alternative_id in winner_sets[group_id]
                for group_id in evaluated_group_ids
            ),
            group_fraction=(
                sum(
                    alternative_id in winner_sets[group_id]
                    for group_id in evaluated_group_ids
                )
                / len(evaluated_group_ids)
            ),
        )
        for alternative_id in alternative_ids
    )
    shared = set(alternative_ids)

    for winners in winner_sets.values():
        shared &= winners

    highest_count = max(item.group_count for item in support)
    session_scores = _scores_by_alternative(
        session,
        alternative_ids=alternative_ids,
    )

    return WinnerAgreementResult(
        tie_atol=tolerance,
        evaluated_group_ids=evaluated_group_ids,
        group_winners=group_winners,
        support=support,
        shared_top_alternative_ids=tuple(
            alternative_id
            for alternative_id in alternative_ids
            if alternative_id in shared
        ),
        highest_support_alternative_ids=tuple(
            item.alternative_id
            for item in support
            if item.group_count == highest_count
        ),
        session_winner_ids=winner_ids(
            session_scores,
            tie_atol=tolerance,
        ),
    )


def calculate_cross_group_weight_dispersion(
    result: RunResult,
) -> CrossGroupWeightDispersionResult:
    """Measure criterion-weight variation across evaluated groups."""

    _, group_calculations, session_calculation = (
        _cross_group_diagnostic_inputs(result)
    )

    criterion_ids = tuple(
        criterion["id"]
        for criterion in result.inputs["session"]["criteria"]
    )

    group_ids = tuple(
        calculation.entity_id for calculation in group_calculations
    )
    group_weights = tuple(
        _weights_by_criterion(
            calculation,
            criterion_ids=criterion_ids,
        )
        for calculation in group_calculations
    )
    session_weights = _weights_by_criterion(
        session_calculation,
        criterion_ids=criterion_ids,
    )

    criterion_dispersions = []

    for criterion_id in criterion_ids:
        values = tuple(
            weights[criterion_id] for weights in group_weights
        )
        minimum = min(values)
        maximum = max(values)

        criterion_dispersions.append(
            CriterionWeightDispersion(
                criterion_id=criterion_id,
                group_weights=tuple(
                    GroupCriterionWeight(
                        group_id=group_id,
                        weight=weights[criterion_id],
                    )
                    for group_id, weights in zip(
                        group_ids,
                        group_weights,
                        strict=True,
                    )
                ),
                minimum_weight=minimum,
                maximum_weight=maximum,
                weight_range=maximum - minimum,
                mean_weight=fmean(values),
                population_standard_deviation=pstdev(values),
                minimum_group_ids=tuple(
                    group_id
                    for group_id, value in zip(
                        group_ids,
                        values,
                        strict=True,
                    )
                    if math.isclose(
                        value,
                        minimum,
                        rel_tol=0,
                        abs_tol=TIE_ATOL,
                    )
                ),
                maximum_group_ids=tuple(
                    group_id
                    for group_id, value in zip(
                        group_ids,
                        values,
                        strict=True,
                    )
                    if math.isclose(
                        value,
                        maximum,
                        rel_tol=0,
                        abs_tol=TIE_ATOL,
                    )
                ),
                session_weight=session_weights[criterion_id],
            )
        )

    return CrossGroupWeightDispersionResult(
        evaluated_group_ids=group_ids,
        criteria=tuple(criterion_dispersions),
    )


def _calculate_entity_score_rank_margins(
    calculation: EntityCalculation,
    *,
    alternative_ids: tuple[str, ...],
    tie_atol: float,
) -> EntityScoreRankMargins:
    scores = _scores_by_alternative(
        calculation,
        alternative_ids=alternative_ids,
    )
    ranks = statistical_ranks(
        scores,
        tie_atol=tie_atol,
    )

    canonical_index = {
        alternative_id: index
        for index, alternative_id
        in enumerate(alternative_ids)
    }

    ordered_ids = tuple(sorted(
        alternative_ids,
        key=lambda alternative_id: (
            -scores[alternative_id],
            canonical_index[alternative_id],
        ),
    ))

    entries = tuple(
        DiagnosticRankEntry(
            alternative_id=alternative_id,
            score=scores[alternative_id],
            statistical_rank=ranks[alternative_id],
        )
        for alternative_id in ordered_ids
    )

    adjacent_margins = []
    for higher, lower in zip(entries[:-1], entries[1:], strict=True):
        raw_margin = higher.score - lower.score
        is_tied = raw_margin <= tie_atol
        score_margin = max(0.0, raw_margin)
        rank_margin = (
            lower.statistical_rank
            - higher.statistical_rank
        )
        adjacent_margins.append(AdjacentScoreRankMargin(
            higher_alternative_id=higher.alternative_id,
            lower_alternative_id=lower.alternative_id,
            higher_score=higher.score,
            lower_score=lower.score,
            score_margin=score_margin,
            higher_statistical_rank=higher.statistical_rank,
            lower_statistical_rank=lower.statistical_rank,
            statistical_rank_margin=rank_margin,
            tied=is_tied,
        ))

    winners = tuple(
        alternative_id
        for alternative_id in ordered_ids
        if (
            scores[ordered_ids[0]]
            - scores[alternative_id]
            <= tie_atol
        )
    )

    if len(winners) > 1:
        top_score_margin = 0.0
        top_rank_margin = 0.0
    else:
        top_score_margin = (
            entries[0].score - entries[1].score
        )
        top_rank_margin = (
            entries[1].statistical_rank
            - entries[0].statistical_rank
        )

    positive_margins = tuple(
        margin.score_margin
        for margin in adjacent_margins
        if not margin.tied
    )

    smallest_positive_margin = (
        min(positive_margins)
        if positive_margins
        else None
    )

    return EntityScoreRankMargins(
        entity_id=calculation.entity_id,
        scope=calculation.scope,
        tie_atol=tie_atol,
        entries=entries,
        adjacent_margins=tuple(adjacent_margins),
        winner_ids=winners,
        top_score_margin=top_score_margin,
        top_statistical_rank_margin=top_rank_margin,
        smallest_positive_score_margin=(
            smallest_positive_margin
        ),
    )


def calculate_score_rank_margins(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
) -> ScoreRankMarginResult:
    tolerance = _validated_tie_atol(tie_atol)

    (
        alternative_ids,
        group_calculations,
        session_calculation,
    ) = _diagnostic_inputs(result)

    if len(alternative_ids) < 2:
        raise DiagnosticNotEvaluableError(
            "Score and rank margins require at least "
            "two alternatives."
        )

    group_ids = tuple(
        calculation.entity_id
        for calculation in group_calculations
    )

    entities = tuple(
        _calculate_entity_score_rank_margins(
            calculation,
            alternative_ids=alternative_ids,
            tie_atol=tolerance,
        )
        for calculation in (
            *group_calculations,
            session_calculation,
        )
    )

    return ScoreRankMarginResult(
        tie_atol=tolerance,
        evaluated_group_ids=group_ids,
        session_id=session_calculation.entity_id,
        entities=entities,
    )


def _consistency_assessment(
    calculation: EntityCalculation,
) -> EntityConsistencyAssessment:
    weighting = calculation.weighting

    if weighting is None:
        return EntityConsistencyAssessment(
            entity_id=calculation.entity_id,
            scope=calculation.scope,
            consistency_ratio=None,
            consistency_threshold=None,
            excess_amount=None,
            status="not_available",
            weighting_source_kind=None,
        )

    ratio = weighting.consistency_ratio
    threshold = weighting.consistency_threshold
    source = calculation.metadata.weighting_source

    if ratio is None:
        return EntityConsistencyAssessment(
            entity_id=calculation.entity_id,
            scope=calculation.scope,
            consistency_ratio=None,
            consistency_threshold=threshold,
            excess_amount=None,
            status="not_available",
            weighting_source_kind=(
                source.kind if source is not None else None
            ),
        )

    exceeds = ratio > threshold

    return EntityConsistencyAssessment(
        entity_id=calculation.entity_id,
        scope=calculation.scope,
        consistency_ratio=float(ratio),
        consistency_threshold=float(threshold),
        excess_amount=max(
            0.0,
            float(ratio) - float(threshold),
        ),
        status=(
            "exceeds_threshold"
            if exceeds
            else "within_threshold"
        ),
        weighting_source_kind=(
            source.kind if source is not None else None
        ),
    )


def _participant_consistency_summaries(
    result: RunResult,
    *,
    group_ids: tuple[str, ...],
) -> tuple[
    GroupParticipantConsistencySummary, ...
]:
    participants_by_group = {
        group_id: []
        for group_id in group_ids
    }

    for calculation in result.mcdm.participants:
        group_id = calculation.metadata.group_id

        if group_id not in participants_by_group:
            raise ValueError(
                "Participant calculation references an "
                "unknown evaluated group."
            )

        participants_by_group[group_id].append(
            calculation
        )

    summaries = []

    for group_id in group_ids:
        calculations = participants_by_group[group_id]
        ratios = []
        exceeding_count = 0
        unavailable_count = 0

        for calculation in calculations:
            weighting = calculation.weighting

            if (
                weighting is None
                or weighting.consistency_ratio is None
            ):
                unavailable_count += 1
                continue

            ratio = float(weighting.consistency_ratio)
            threshold = float(
                weighting.consistency_threshold
            )

            ratios.append(ratio)

            if ratio > threshold:
                exceeding_count += 1

        evaluated_count = len(ratios)

        summaries.append(
            GroupParticipantConsistencySummary(
                group_id=group_id,
                participant_count=len(calculations),
                evaluated_count=evaluated_count,
                unavailable_count=unavailable_count,
                exceeding_count=exceeding_count,
                exceeding_fraction_of_evaluated=(
                    exceeding_count / evaluated_count
                    if evaluated_count
                    else None
                ),
                maximum_consistency_ratio=(
                    max(ratios) if ratios else None
                ),
            )
        )

    return tuple(summaries)


def calculate_consistency_warnings(
    result: RunResult,
) -> ConsistencyWarningResult:
    (
        _,
        group_calculations,
        session_calculation,
    ) = _diagnostic_inputs(result)

    group_ids = tuple(
        calculation.entity_id
        for calculation in group_calculations
    )

    assessments = tuple(
        _consistency_assessment(calculation)
        for calculation in (
            *group_calculations,
            session_calculation,
        )
    )

    participant_summaries = (
        _participant_consistency_summaries(
            result,
            group_ids=group_ids,
        )
    )

    return ConsistencyWarningResult(
        evaluated_group_ids=group_ids,
        session_id=session_calculation.entity_id,
        entity_assessments=assessments,
        participant_group_summaries=(
            participant_summaries
        ),
        group_or_session_warning_ids=tuple(
            assessment.entity_id
            for assessment in assessments
            if assessment.status == "exceeds_threshold"
        ),
        participant_warning_group_ids=tuple(
            summary.group_id
            for summary in participant_summaries
            if summary.exceeding_count > 0
        ),
    )


def _resolved_recomputation_inputs(
    result: RunResult,
) -> tuple[
    DecisionSession,
    WeightingMethodConfig,
    RankingMethodConfig,
    AggregationMethodConfig,
    ConsistencyMethodConfig,
]:
    session_data = result.inputs.get("session")
    processing = result.inputs.get("processing")

    if not isinstance(session_data, dict):
        raise ValueError(
            "Recomputation requires the resolved session."
        )

    if not isinstance(processing, dict):
        raise ValueError(
            "Recomputation requires processing configuration."
        )

    session = DecisionSession.model_validate(
        session_data
    )

    return (
        session,
        _WEIGHTING_CONFIG_ADAPTER.validate_python(
            processing["weighting_method"]
        ),
        _RANKING_CONFIG_ADAPTER.validate_python(
            processing["ranking_method"]
        ),
        _AGGREGATION_CONFIG_ADAPTER.validate_python(
            processing["aggregation_method"]
        ),
        _CONSISTENCY_CONFIG_ADAPTER.validate_python(
            processing["consistency_method"]
        ),
    )


def _recomputed_session_ranking(
    *,
    session: DecisionSession,
    decision_matrix: np.ndarray,
    weighting: WeightResult,
    baseline: EntityCalculation,
    ranking_config: RankingMethodConfig,
) -> tuple[
    RankingResult,
    dict[str, float],
]:
    full_ids = tuple(
        criterion.id for criterion in session.criteria
    )

    selected_ids = tuple(
        baseline.metadata.ranking_criterion_ids
    )

    if not selected_ids:
        raise ValueError(
            "Baseline session is missing ranking criteria."
        )

    weight_source = (
        baseline.metadata.ranking_weight_source
    )

    override = (
        dict(baseline.metadata.applied_criterion_weights)
        if weight_source == "explicit"
        else None
    )

    (
        ranking_weighting,
        applied_weights,
        recomputed_source,
    ) = build_ranking_weighting(
        weighting,
        full_criterion_ids=full_ids,
        ranking_criterion_ids=selected_ids,
        override=override,
    )

    if recomputed_source != weight_source:
        raise ValueError(
            "Recomputed ranking weight policy does not "
            "match the baseline session."
        )

    positions = {
        criterion_id: index
        for index, criterion_id in enumerate(full_ids)
    }
    selected_positions = tuple(
        positions[criterion_id]
        for criterion_id in selected_ids
    )

    ranking_session = session.model_copy(update={
        "criteria": [
            session.criteria[index]
            for index in selected_positions
        ],
    })

    ranking = MCDMService.generate_ranking(
        session=ranking_session,
        decision_matrix=decision_matrix[
            :, selected_positions
        ],
        weighting=ranking_weighting,
        config=ranking_config,
    )

    return ranking, applied_weights


def _not_evaluable_omission_case(
    *,
    omitted_group_id: str,
    remaining_group_ids: tuple[str, ...],
    baseline_winner_ids: tuple[str, ...],
    reason: str,
) -> LeaveOneGroupOutCase:
    return LeaveOneGroupOutCase(
        omitted_group_id=omitted_group_id,
        remaining_group_ids=remaining_group_ids,
        status="not_evaluable",
        not_evaluable_reason=reason,
        baseline_winner_ids=baseline_winner_ids,
    )


def calculate_leave_one_group_out(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
) -> LeaveOneGroupOutResult:
    tolerance = _validated_tie_atol(tie_atol)

    (
        alternative_ids,
        group_calculations,
        baseline_session,
    ) = _cross_group_diagnostic_inputs(result)

    (
        session,
        weighting_config,
        ranking_config,
        aggregation_config,
        consistency_config,
    ) = _resolved_recomputation_inputs(result)

    decision_matrix = (
        MCDMService.create_decision_matrix(session)
    )

    group_ids = tuple(
        calculation.entity_id
        for calculation in group_calculations
    )

    group_matrices = {
        calculation.entity_id: np.asarray(
            calculation.pairwise_matrix,
            dtype=float,
        )
        for calculation in group_calculations
    }

    voting_powers = {}

    for calculation in group_calculations:
        power = calculation.metadata.voting_power

        if power is None:
            raise ValueError(
                "Leave-one-group-out requires recorded "
                "configured voting powers."
            )

        voting_powers[calculation.entity_id] = float(power)

    baseline_scores = _scores_by_alternative(
        baseline_session,
        alternative_ids=alternative_ids,
    )
    baseline_ranks = statistical_ranks(
        baseline_scores,
        tie_atol=tolerance,
    )
    baseline_winners = winner_ids(
        baseline_scores,
        tie_atol=tolerance,
    )
    baseline_applied_weights = dict(
        baseline_session.metadata.applied_criterion_weights
    )
    selected_criterion_ids = tuple(
        criterion.id
        for criterion in session.criteria
        if criterion.id in baseline_applied_weights
    )

    if set(selected_criterion_ids) != set(
        baseline_applied_weights
    ):
        raise ValueError(
            "Baseline applied weights must use configured criteria."
        )

    cases = []

    for omitted_group_id in group_ids:
        remaining_ids = tuple(
            group_id
            for group_id in group_ids
            if group_id != omitted_group_id
        )

        remaining_power = sum(
            voting_powers[group_id]
            for group_id in remaining_ids
        )

        if remaining_power <= 0:
            cases.append(_not_evaluable_omission_case(
                omitted_group_id=omitted_group_id,
                remaining_group_ids=remaining_ids,
                baseline_winner_ids=baseline_winners,
                reason=(
                    "Remaining groups have no positive total "
                    "configured voting power."
                ),
            ))
            continue

        aggregation = MCDMService.aggregate_group_matrices(
            group_ids=remaining_ids,
            group_matrices={
                group_id: group_matrices[group_id]
                for group_id in remaining_ids
            },
            voting_powers={
                group_id: voting_powers[group_id]
                for group_id in remaining_ids
            },
            config=aggregation_config,
        )

        weighting = MCDMService.generate_weights(
            session=session,
            pairwise_matrix=aggregation.matrix,
            config=weighting_config,
            consistency=consistency_config,
        )

        ranking, applied_weights = (
            _recomputed_session_ranking(
                session=session,
                decision_matrix=decision_matrix,
                weighting=weighting,
                baseline=baseline_session,
                ranking_config=ranking_config,
            )
        )

        recomputed_scores = {
            row.alternative_id: float(
                row.closeness_coefficient
            )
            for row in ranking.rankings
        }

        if set(recomputed_scores) != set(alternative_ids):
            raise ValueError(
                "Recomputed ranking does not cover the "
                "configured alternatives."
            )

        recomputed_ranks = statistical_ranks(
            recomputed_scores,
            tie_atol=tolerance,
        )
        recomputed_winners = winner_ids(
            recomputed_scores,
            tie_atol=tolerance,
        )

        criterion_weight_changes = tuple(
            LeaveOneOutCriterionWeightChange(
                criterion_id=criterion_id,
                baseline_weight=baseline_applied_weights[
                    criterion_id
                ],
                recomputed_weight=applied_weights[
                    criterion_id
                ],
                weight_delta=(
                    applied_weights[criterion_id]
                    - baseline_applied_weights[criterion_id]
                ),
            )
            for criterion_id in selected_criterion_ids
        )
        alternative_changes = tuple(
            LeaveOneOutAlternativeChange(
                alternative_id=alternative_id,
                baseline_score=baseline_scores[alternative_id],
                recomputed_score=recomputed_scores[
                    alternative_id
                ],
                score_delta=(
                    recomputed_scores[alternative_id]
                    - baseline_scores[alternative_id]
                ),
                baseline_statistical_rank=baseline_ranks[
                    alternative_id
                ],
                recomputed_statistical_rank=recomputed_ranks[
                    alternative_id
                ],
                statistical_rank_delta=(
                    recomputed_ranks[alternative_id]
                    - baseline_ranks[alternative_id]
                ),
            )
            for alternative_id in alternative_ids
        )

        cases.append(LeaveOneGroupOutCase(
            omitted_group_id=omitted_group_id,
            remaining_group_ids=remaining_ids,
            status="completed",
            effective_voting_powers=tuple(
                EffectiveGroupPower(
                    group_id=group_id,
                    effective_voting_power=(
                        aggregation.effective_voting_powers[
                            group_id
                        ]
                    ),
                )
                for group_id in remaining_ids
            ),
            baseline_winner_ids=baseline_winners,
            recomputed_winner_ids=recomputed_winners,
            winner_changed=(
                set(baseline_winners)
                != set(recomputed_winners)
            ),
            criterion_weight_changes=criterion_weight_changes,
            alternative_changes=alternative_changes,
            maximum_absolute_weight_delta=max(
                abs(change.weight_delta)
                for change in criterion_weight_changes
            ),
            maximum_absolute_score_delta=max(
                abs(change.score_delta)
                for change in alternative_changes
            ),
            maximum_absolute_rank_delta=max(
                abs(change.statistical_rank_delta)
                for change in alternative_changes
            ),
            recomputed_consistency_ratio=(
                weighting.consistency_ratio
            ),
        ))

    return LeaveOneGroupOutResult(
        tie_atol=tolerance,
        evaluated_group_ids=group_ids,
        session_id=baseline_session.entity_id,
        baseline_winner_ids=baseline_winners,
        cases=tuple(cases),
    )
