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
    AlternativeRemovalCase,
    AlternativeRemovalResult,
    BoundedWeightPerturbationCase,
    BoundedWeightPerturbationResult,
    ConsistencyWarningResult,
    CriterionRemovalCase,
    CriterionRemovalResult,
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
from src.mcdm.service import (
    MCDMService,
    UndefinedTopsisScoresError,
    build_ranking_weighting,
    validate_weight_vector,
)

if TYPE_CHECKING:
    from src.experiments.results import RunResult


TIE_ATOL = 1e-12
_UNDEFINED_CORRELATION_REASON = (
    "Correlation is undefined because at least one "
    "aligned ranking contains only ties."
)
_IDENTICAL_EFFECTIVE_ALTERNATIVES_REASON = (
    "All positively weighted criteria have identical values across "
    "alternatives, so TOPSIS is undefined."
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


def _ranking_diagnostic_baseline(
    result: RunResult,
    group_id: str | None,
) -> tuple[tuple[str, ...], EntityCalculation]:
    alternative_ids, groups, session = _diagnostic_inputs(result)
    if group_id is None:
        return alternative_ids, session

    for calculation in groups:
        if calculation.entity_id == group_id:
            return alternative_ids, calculation
    raise DiagnosticNotEvaluableError(
        f"No evaluated ranking is available for group {group_id!r}."
    )


def _scores_by_alternative(
    calculation: EntityCalculation,
    *,
    alternative_ids: tuple[str, ...],
) -> dict[str, float]:
    ranking = getattr(calculation, "ranking", None)

    if ranking is None:
        raise ValueError(
            f"Entity '{calculation.entity_id}' has no completed ranking."
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


def _rank_with_perturbed_weights(
    *,
    session: DecisionSession,
    decision_matrix: np.ndarray,
    baseline: EntityCalculation,
    ranking_config: RankingMethodConfig,
    perturbed_weights: dict[str, float],
) -> RankingResult:
    if baseline.weighting is None:
        raise ValueError("Session perturbations require baseline weighting.")

    return _rank_with_applied_weights(
        session=session,
        decision_matrix=decision_matrix,
        original_weighting=baseline.weighting,
        selected_ids=tuple(baseline.metadata.ranking_criterion_ids),
        ranking_config=ranking_config,
        applied_weights=perturbed_weights,
    )


def _rank_with_applied_weights(
    *,
    session: DecisionSession,
    decision_matrix: np.ndarray,
    original_weighting: WeightResult,
    selected_ids: tuple[str, ...],
    ranking_config: RankingMethodConfig,
    applied_weights: dict[str, float],
) -> RankingResult:
    full_ids = tuple(criterion.id for criterion in session.criteria)

    weighting, _, _ = build_ranking_weighting(
        original_weighting,
        full_criterion_ids=full_ids,
        ranking_criterion_ids=selected_ids,
        override=applied_weights,
    )

    positions_by_id = {cid: index for index, cid in enumerate(full_ids)}
    positions = tuple(positions_by_id[cid] for cid in selected_ids)
    ranking_session = session.model_copy(update={
        "criteria": [session.criteria[index] for index in positions],
    })

    return MCDMService.generate_ranking(
        session=ranking_session,
        decision_matrix=decision_matrix[:, positions],
        weighting=weighting,
        config=ranking_config,
    )


def _identical_effective_alternatives(
    decision_matrix: np.ndarray,
    criteria_positions: Mapping[str, int],
    applied_weights: Mapping[str, float],
) -> bool:
    if decision_matrix.ndim != 2 or decision_matrix.shape[0] == 0:
        return False

    active_positions = tuple(
        criteria_positions[criterion_id]
        for criterion_id, weight in applied_weights.items()
        if weight > 0
    )
    if not active_positions:
        return False

    active_values = decision_matrix[:, active_positions]
    return bool(np.all(active_values == active_values[0]))


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
        baseline_session.metadata.ranking_criterion_ids
    )

    if (
        not selected_criterion_ids
        or len(selected_criterion_ids)
        != len(set(selected_criterion_ids))
        or set(selected_criterion_ids)
        != set(baseline_applied_weights)
    ):
        raise ValueError(
            "Baseline ranking criteria and applied weights "
            "must cover the same criteria exactly once."
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


def perturb_applied_weights(
    baseline_weights: Mapping[str, float],
    *,
    criterion_id: str,
    weight_delta: float,
) -> dict[str, float] | None:
    """Move one weight by a signed amount and rescale its peers."""
    if criterion_id not in baseline_weights:
        raise ValueError(f"Unknown ranking criterion: {criterion_id!r}")

    if not math.isfinite(weight_delta) or not 0 < abs(weight_delta) < 1:
        raise ValueError("Weight delta must be finite and have magnitude in (0, 1).")

    criterion_ids = tuple(baseline_weights)
    validated = validate_weight_vector(
        [baseline_weights[cid] for cid in criterion_ids],
        expected_count=len(criterion_ids),
        label="Baseline applied ranking weights",
    )
    weights = dict(zip(criterion_ids, map(float, validated), strict=True))

    target_weight = weights[criterion_id] + weight_delta
    if not 0 <= target_weight <= 1:
        return None

    other_weight = math.fsum(
        weight for cid, weight in weights.items() if cid != criterion_id
    )
    if other_weight == 0:
        return None

    remaining_weight = 1.0 - target_weight
    perturbed = {
        cid: (
            target_weight
            if cid == criterion_id
            else (weight / other_weight) * remaining_weight
        )
        for cid, weight in weights.items()
    }

    validate_weight_vector(
        list(perturbed.values()),
        expected_count=len(criterion_ids),
        label="Perturbed applied ranking weights",
    )
    return perturbed


def calculate_bounded_weight_perturbations(
    result: RunResult,
    *,
    absolute_weight_delta: float,
    tie_atol: float = TIE_ATOL,
    group_id: str | None = None,
) -> BoundedWeightPerturbationResult:
    if (
        isinstance(absolute_weight_delta, bool)
        or not isinstance(absolute_weight_delta, Real)
        or not math.isfinite(absolute_weight_delta)
        or not 0 < absolute_weight_delta < 1
    ):
        raise ValueError(
            "absolute_weight_delta must be finite and in (0, 1)."
        )
    absolute_weight_delta = float(absolute_weight_delta)
    tolerance = _validated_tie_atol(tie_atol)
    alternative_ids, baseline = _ranking_diagnostic_baseline(result, group_id)
    (
        session,
        _,
        ranking_config,
        _,
        _,
    ) = _resolved_recomputation_inputs(result)

    decision_matrix = MCDMService.create_decision_matrix(session)
    baseline_scores = _scores_by_alternative(
        baseline,
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
    baseline_weights = dict(
        baseline.metadata.applied_criterion_weights
    )
    criterion_ids = tuple(
        baseline.metadata.ranking_criterion_ids
    )

    if (
        not criterion_ids
        or len(criterion_ids) != len(set(criterion_ids))
        or set(criterion_ids) != set(baseline_weights)
    ):
        raise ValueError(
            "Baseline ranking criteria and applied weights must "
            "cover the same criteria exactly once."
        )

    criteria_positions = {
        criterion.id: index
        for index, criterion in enumerate(session.criteria)
    }

    cases = []
    for criterion_id in criterion_ids:
        for requested_delta in (
            -absolute_weight_delta,
            absolute_weight_delta,
        ):
            applied_weights = perturb_applied_weights(
                baseline_weights,
                criterion_id=criterion_id,
                weight_delta=requested_delta,
            )
            if applied_weights is None:
                cases.append(BoundedWeightPerturbationCase(
                    criterion_id=criterion_id,
                    requested_delta=requested_delta,
                    status="not_evaluable",
                    reason=(
                        "The requested delta would make the applied "
                        "weight vector infeasible."
                    ),
                ))
                continue

            try:
                ranking = _rank_with_perturbed_weights(
                    session=session,
                    decision_matrix=decision_matrix,
                    baseline=baseline,
                    ranking_config=ranking_config,
                    perturbed_weights=applied_weights,
                )
            except UndefinedTopsisScoresError:
                if not _identical_effective_alternatives(
                    decision_matrix, criteria_positions, applied_weights,
                ):
                    raise
                cases.append(BoundedWeightPerturbationCase(
                    criterion_id=criterion_id,
                    requested_delta=requested_delta,
                    status="not_evaluable",
                    reason=_IDENTICAL_EFFECTIVE_ALTERNATIVES_REASON,
                ))
                continue

            scores = {
                row.alternative_id: float(row.closeness_coefficient)
                for row in ranking.rankings
            }
            if (
                len(scores) != len(ranking.rankings)
                or set(scores) != set(alternative_ids)
            ):
                raise ValueError(
                    "Perturbed ranking must cover the configured "
                    "alternatives exactly once."
                )

            ranks = statistical_ranks(
                scores,
                tie_atol=tolerance,
            )
            recomputed_winners = winner_ids(
                scores,
                tie_atol=tolerance,
            )
            cases.append(BoundedWeightPerturbationCase(
                criterion_id=criterion_id,
                requested_delta=requested_delta,
                status="completed",
                applied_weights=applied_weights,
                entries=tuple(
                    DiagnosticRankEntry(
                        alternative_id=alternative_id,
                        score=scores[alternative_id],
                        statistical_rank=ranks[alternative_id],
                    )
                    for alternative_id in alternative_ids
                ),
                winner_ids=recomputed_winners,
                winner_changed=(
                    set(baseline_winners)
                    != set(recomputed_winners)
                ),
            ))

    return BoundedWeightPerturbationResult(
        session_id=session.id,
        group_id=group_id,
        absolute_weight_delta=absolute_weight_delta,
        tie_atol=tolerance,
        ranking_criterion_ids=criterion_ids,
        baseline_applied_weights=baseline_weights,
        baseline_entries=tuple(
            DiagnosticRankEntry(
                alternative_id=alternative_id,
                score=baseline_scores[alternative_id],
                statistical_rank=baseline_ranks[alternative_id],
            )
            for alternative_id in alternative_ids
        ),
        baseline_winner_ids=baseline_winners,
        cases=tuple(cases),
    )


def calculate_criterion_removal_effects(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
    group_id: str | None = None,
) -> CriterionRemovalResult:
    """Rerank the same alternatives after dropping each applied criterion."""
    tolerance = _validated_tie_atol(tie_atol)
    alternative_ids, baseline = _ranking_diagnostic_baseline(result, group_id)
    session, _, ranking_config, _, _ = _resolved_recomputation_inputs(result)
    if baseline.weighting is None:
        raise ValueError("Criterion removal requires baseline weighting.")

    criterion_ids = tuple(baseline.metadata.ranking_criterion_ids)
    baseline_weights = {
        cid: baseline.metadata.applied_criterion_weights[cid] for cid in criterion_ids
    }
    baseline_scores = _scores_by_alternative(baseline, alternative_ids=alternative_ids)
    baseline_ranks = statistical_ranks(baseline_scores, tie_atol=tolerance)
    baseline_winners = winner_ids(baseline_scores, tie_atol=tolerance)
    decision_matrix = MCDMService.create_decision_matrix(session)
    criteria_positions = {
        criterion.id: index for index, criterion in enumerate(session.criteria)
    }

    cases = []
    for removed_id in criterion_ids:
        retained_ids = tuple(cid for cid in criterion_ids if cid != removed_id)
        retained_mass = math.fsum(baseline_weights[cid] for cid in retained_ids)
        if not retained_ids or retained_mass <= 0:
            cases.append(CriterionRemovalCase(
                removed_criterion_id=removed_id,
                retained_criterion_ids=retained_ids,
                status="not_evaluable",
                reason=(
                    "Removing this criterion leaves no ranking criteria."
                    if not retained_ids
                    else "The retained criteria have no positive applied ranking weight."
                ),
            ))
            continue

        applied_weights = {
            cid: baseline_weights[cid] / retained_mass for cid in retained_ids
        }
        try:
            ranking = _rank_with_applied_weights(
                session=session,
                decision_matrix=decision_matrix,
                original_weighting=baseline.weighting,
                selected_ids=retained_ids,
                ranking_config=ranking_config,
                applied_weights=applied_weights,
            )
        except UndefinedTopsisScoresError:
            if not _identical_effective_alternatives(
                decision_matrix, criteria_positions, applied_weights,
            ):
                raise
            cases.append(CriterionRemovalCase(
                removed_criterion_id=removed_id,
                retained_criterion_ids=retained_ids,
                status="not_evaluable",
                reason=_IDENTICAL_EFFECTIVE_ALTERNATIVES_REASON,
            ))
            continue

        scores = {
            row.alternative_id: float(row.closeness_coefficient)
            for row in ranking.rankings
        }
        if len(scores) != len(ranking.rankings) or set(scores) != set(alternative_ids):
            raise ValueError(
                "Criterion removal ranking must cover the baseline alternatives exactly once."
            )
        ranks = statistical_ranks(scores, tie_atol=tolerance)
        winners = winner_ids(scores, tie_atol=tolerance)
        cases.append(CriterionRemovalCase(
            removed_criterion_id=removed_id,
            retained_criterion_ids=retained_ids,
            status="completed",
            applied_weights=applied_weights,
            alternative_changes=tuple(
                LeaveOneOutAlternativeChange(
                    alternative_id=aid,
                    baseline_score=baseline_scores[aid],
                    recomputed_score=scores[aid],
                    score_delta=scores[aid] - baseline_scores[aid],
                    baseline_statistical_rank=baseline_ranks[aid],
                    recomputed_statistical_rank=ranks[aid],
                    statistical_rank_delta=ranks[aid] - baseline_ranks[aid],
                )
                for aid in alternative_ids
            ),
            winner_ids=winners,
            winner_changed=set(winners) != set(baseline_winners),
        ))

    return CriterionRemovalResult(
        session_id=session.id,
        group_id=group_id,
        tie_atol=tolerance,
        ranking_criterion_ids=criterion_ids,
        baseline_applied_weights=baseline_weights,
        alternative_ids=alternative_ids,
        baseline_winner_ids=baseline_winners,
        cases=tuple(cases),
    )


def calculate_alternative_removal_effects(
    result: RunResult,
    *,
    tie_atol: float = TIE_ATOL,
    group_id: str | None = None,
) -> AlternativeRemovalResult:
    """Hold applied weights fixed and rerank each set of surviving alternatives."""
    tolerance = _validated_tie_atol(tie_atol)
    alternative_ids, baseline = _ranking_diagnostic_baseline(result, group_id)
    session, _, ranking_config, _, _ = _resolved_recomputation_inputs(result)
    if baseline.weighting is None:
        raise ValueError("Alternative removal requires baseline weighting.")

    criterion_ids = tuple(baseline.metadata.ranking_criterion_ids)
    applied_weights = {
        cid: baseline.metadata.applied_criterion_weights[cid] for cid in criterion_ids
    }
    baseline_scores = _scores_by_alternative(baseline, alternative_ids=alternative_ids)
    baseline_ranks = statistical_ranks(baseline_scores, tie_atol=tolerance)
    baseline_winners = winner_ids(baseline_scores, tie_atol=tolerance)
    decision_matrix = MCDMService.create_decision_matrix(session)
    criteria_positions = {
        criterion.id: index for index, criterion in enumerate(session.criteria)
    }
    selected_positions = tuple(criteria_positions[cid] for cid in criterion_ids)

    cases = []
    for removed_index, removed_id in enumerate(alternative_ids):
        retained_positions = [
            index for index in range(len(alternative_ids)) if index != removed_index
        ]
        retained_ids = tuple(alternative_ids[index] for index in retained_positions)
        survivor_scores = {aid: baseline_scores[aid] for aid in retained_ids}
        survivor_winners = (
            winner_ids(survivor_scores, tie_atol=tolerance) if survivor_scores else ()
        )
        case_inputs = {
            "removed_alternative_id": removed_id,
            "retained_alternative_ids": retained_ids,
            "baseline_winner_removed": removed_id in baseline_winners,
            "baseline_survivor_winner_ids": survivor_winners,
        }

        if len(retained_ids) < 2:
            cases.append(AlternativeRemovalCase(
                **case_inputs,
                status="not_evaluable",
                reason="Alternative removal requires at least two surviving alternatives.",
            ))
            continue

        survivor_matrix = decision_matrix[retained_positions]
        zero_columns = np.all(survivor_matrix[:, selected_positions] == 0, axis=0)
        if np.any(zero_columns):
            zero_ids = [cid for cid, is_zero in zip(criterion_ids, zero_columns, strict=True) if is_zero]
            cases.append(AlternativeRemovalCase(
                **case_inputs,
                status="not_evaluable",
                reason=f"TOPSIS normalization is undefined for all-zero criteria: {', '.join(zero_ids)}.",
            ))
            continue

        survivor_session = session.model_copy(update={
            "alternatives": [session.alternatives[index] for index in retained_positions],
        })
        try:
            ranking = _rank_with_applied_weights(
                session=survivor_session,
                decision_matrix=survivor_matrix,
                original_weighting=baseline.weighting,
                selected_ids=criterion_ids,
                ranking_config=ranking_config,
                applied_weights=applied_weights,
            )
        except UndefinedTopsisScoresError:
            if not _identical_effective_alternatives(
                survivor_matrix, criteria_positions, applied_weights,
            ):
                raise
            cases.append(AlternativeRemovalCase(
                **case_inputs,
                status="not_evaluable",
                reason=_IDENTICAL_EFFECTIVE_ALTERNATIVES_REASON,
            ))
            continue

        scores = {
            row.alternative_id: float(row.closeness_coefficient)
            for row in ranking.rankings
        }
        if len(scores) != len(ranking.rankings) or set(scores) != set(retained_ids):
            raise ValueError("Alternative removal ranking must cover the survivors exactly once.")

        survivor_ranks = statistical_ranks(survivor_scores, tie_atol=tolerance)
        ranks = statistical_ranks(scores, tie_atol=tolerance)
        winners = winner_ids(scores, tie_atol=tolerance)
        cases.append(AlternativeRemovalCase(
            **case_inputs,
            status="completed",
            alternative_changes=tuple(
                LeaveOneOutAlternativeChange(
                    alternative_id=aid,
                    baseline_score=survivor_scores[aid],
                    recomputed_score=scores[aid],
                    score_delta=scores[aid] - survivor_scores[aid],
                    baseline_statistical_rank=survivor_ranks[aid],
                    recomputed_statistical_rank=ranks[aid],
                    statistical_rank_delta=ranks[aid] - survivor_ranks[aid],
                )
                for aid in retained_ids
            ),
            recomputed_winner_ids=winners,
            survivor_winner_changed=set(winners) != set(survivor_winners),
        ))

    return AlternativeRemovalResult(
        session_id=session.id,
        group_id=group_id,
        tie_atol=tolerance,
        ranking_criterion_ids=criterion_ids,
        applied_weights=applied_weights,
        baseline_entries=tuple(
            DiagnosticRankEntry(
                alternative_id=aid,
                score=baseline_scores[aid],
                statistical_rank=baseline_ranks[aid],
            )
            for aid in alternative_ids
        ),
        baseline_winner_ids=baseline_winners,
        cases=tuple(cases),
    )
