import math
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from src.mcdm.models.computation import WEIGHT_SUM_TOLERANCE


ConsistencyStatus = Literal[
    "within_threshold",
    "exceeds_threshold",
    "not_available",
]


class DiagnosticRankEntry(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    alternative_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    statistical_rank: float = Field(ge=1)


class GroupRankVector(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    group_id: str = Field(min_length=1)
    entries: tuple[
        DiagnosticRankEntry, ...
    ] = Field(min_length=2)

    winner_ids: tuple[str, ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        alternative_ids = tuple(
            entry.alternative_id
            for entry in self.entries
        )

        if len(set(alternative_ids)) != len(
            alternative_ids
        ):
            raise ValueError(
                "Rank entries must have unique "
                "alternative IDs."
            )

        if len(set(self.winner_ids)) != len(
            self.winner_ids
        ):
            raise ValueError(
                "Winner IDs must be unique."
            )

        unknown_winner_ids = (
            set(self.winner_ids)
            - set(alternative_ids)
        )

        if unknown_winner_ids:
            raise ValueError(
                "Winner IDs must reference alternatives "
                "present in the rank entries: "
                f"{sorted(unknown_winner_ids)}"
            )

        return self


def _validate_group_rank_vectors(
    vectors: tuple[GroupRankVector, ...],
) -> tuple[GroupRankVector, ...]:
    group_ids = tuple(
        vector.group_id for vector in vectors
    )

    if len(set(group_ids)) != len(group_ids):
        raise ValueError(
            "Group rank vectors must have unique group IDs."
        )

    expected_alternative_ids = {
        entry.alternative_id
        for entry in vectors[0].entries
    }

    for vector in vectors[1:]:
        alternative_ids = {
            entry.alternative_id
            for entry in vector.entries
        }

        if alternative_ids != expected_alternative_ids:
            missing_ids = sorted(
                expected_alternative_ids
                - alternative_ids
            )
            unexpected_ids = sorted(
                alternative_ids
                - expected_alternative_ids
            )
            raise ValueError(
                "Group rank vectors must cover the same "
                "alternatives. "
                f"Group '{vector.group_id}' is missing "
                f"{missing_ids} and has unexpected "
                f"alternatives {unexpected_ids}."
            )

    return vectors


GroupRankVectors = Annotated[
    tuple[GroupRankVector, ...],
    Field(min_length=2),
    AfterValidator(_validate_group_rank_vectors),
]


class GroupRankCorrelation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    left_group_id: str = Field(min_length=1)
    right_group_id: str = Field(min_length=1)
    alternative_ids: tuple[str, ...] = Field(
        min_length=2
    )
    spearman_rho: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    kendall_tau_b: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    undefined_reason: str | None = None

    @model_validator(mode="after")
    def validate_correlation(self) -> Self:
        if self.left_group_id == self.right_group_id:
            raise ValueError(
                "A rank correlation requires two distinct groups."
            )

        if len(set(self.alternative_ids)) != len(
            self.alternative_ids
        ):
            raise ValueError(
                "Correlation alternative IDs must be unique."
            )

        coefficients_defined = (
            self.spearman_rho is not None
            and self.kendall_tau_b is not None
        )
        coefficients_undefined = (
            self.spearman_rho is None
            and self.kendall_tau_b is None
        )

        if not (
            coefficients_defined
            or coefficients_undefined
        ):
            raise ValueError(
                "Spearman rho and Kendall tau-b must be "
                "defined or undefined together."
            )

        if coefficients_undefined:
            if (
                self.undefined_reason is None
                or not self.undefined_reason.strip()
            ):
                raise ValueError(
                    "Undefined correlations require a reason."
                )

        elif self.undefined_reason is not None:
            raise ValueError(
                "Defined correlations cannot have an "
                "undefined reason."
            )

        return self


class CrossGroupRankCorrelationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "cross-group-rank-correlation-v1"
    ] = "cross-group-rank-correlation-v1"
    tie_atol: float = Field(ge=0)
    rank_vectors: GroupRankVectors
    pairwise_correlations: tuple[
        GroupRankCorrelation, ...
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pairs(self) -> Self:
        group_ids = tuple(
            vector.group_id
            for vector in self.rank_vectors
        )
        expected_pairs = {
            frozenset(pair)
            for pair in combinations(group_ids, 2)
        }
        actual_pairs = [
            frozenset((
                correlation.left_group_id,
                correlation.right_group_id,
            ))
            for correlation in self.pairwise_correlations
        ]

        if (
            len(actual_pairs) != len(expected_pairs)
            or len(set(actual_pairs)) != len(actual_pairs)
            or set(actual_pairs) != expected_pairs
        ):
            raise ValueError(
                "Pairwise correlations must contain exactly one "
                "record for every unordered group pair."
            )

        expected_alternative_ids = tuple(
            entry.alternative_id
            for entry in self.rank_vectors[0].entries
        )

        for vector in self.rank_vectors[1:]:
            if tuple(
                entry.alternative_id
                for entry in vector.entries
            ) != expected_alternative_ids:
                raise ValueError(
                    "Rank vectors must use the same canonical "
                    "alternative order."
                )

        if any(
            correlation.alternative_ids
            != expected_alternative_ids
            for correlation in self.pairwise_correlations
        ):
            raise ValueError(
                "Pairwise correlations must use the canonical "
                "alternative order from the rank vectors."
            )

        return self


class GroupWinnerSet(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    group_id: str = Field(min_length=1)
    winner_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_winners(self) -> Self:
        if len(set(self.winner_ids)) != len(
            self.winner_ids
        ):
            raise ValueError("Winner IDs must be unique.")

        return self


class AlternativeWinnerSupport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    alternative_id: str = Field(min_length=1)
    supporting_group_ids: tuple[str, ...]
    group_count: int = Field(ge=0)
    group_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if len(set(self.supporting_group_ids)) != len(
            self.supporting_group_ids
        ):
            raise ValueError(
                "Supporting group IDs must be unique."
            )

        if self.group_count != len(
            self.supporting_group_ids
        ):
            raise ValueError(
                "Group count must equal the number of "
                "supporting groups."
            )

        return self


class WinnerAgreementResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "winner-agreement-v1"
    ] = "winner-agreement-v1"
    tie_atol: float = Field(ge=0)
    evaluated_group_ids: tuple[str, ...] = Field(
        min_length=2
    )
    group_winners: tuple[
        GroupWinnerSet, ...
    ] = Field(min_length=2)
    support: tuple[
        AlternativeWinnerSupport, ...
    ] = Field(min_length=1)
    shared_top_alternative_ids: tuple[str, ...]
    highest_support_alternative_ids: tuple[
        str, ...
    ] = Field(min_length=1)
    session_winner_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_agreement(self) -> Self:
        if len(set(self.evaluated_group_ids)) != len(
            self.evaluated_group_ids
        ):
            raise ValueError(
                "Evaluated group IDs must be unique."
            )

        winner_group_ids = tuple(
            item.group_id for item in self.group_winners
        )

        if winner_group_ids != self.evaluated_group_ids:
            raise ValueError(
                "Group winner sets must occur once in evaluated "
                "group order."
            )

        alternative_ids = tuple(
            item.alternative_id for item in self.support
        )

        if len(set(alternative_ids)) != len(alternative_ids):
            raise ValueError(
                "Winner support alternatives must be unique."
            )

        known_alternatives = set(alternative_ids)
        known_groups = set(self.evaluated_group_ids)
        winners_by_group = {
            item.group_id: set(item.winner_ids)
            for item in self.group_winners
        }

        if any(
            not set(item.winner_ids) <= known_alternatives
            for item in self.group_winners
        ):
            raise ValueError(
                "Group winners must reference support alternatives."
            )

        for item in self.support:
            expected_groups = tuple(
                group_id
                for group_id in self.evaluated_group_ids
                if item.alternative_id
                in winners_by_group[group_id]
            )
            expected_fraction = (
                len(expected_groups)
                / len(self.evaluated_group_ids)
            )

            if item.supporting_group_ids != expected_groups:
                raise ValueError(
                    "Supporting groups must match the modeled "
                    "group winner sets."
                )

            if not math.isclose(
                item.group_fraction,
                expected_fraction,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Group fraction must equal group count divided "
                    "by evaluated group count."
                )

            if not set(item.supporting_group_ids) <= known_groups:
                raise ValueError(
                    "Winner support references an unknown group."
                )

        shared = set(known_alternatives)

        for winners in winners_by_group.values():
            shared &= winners

        expected_shared = tuple(
            alternative_id
            for alternative_id in alternative_ids
            if alternative_id in shared
        )

        if self.shared_top_alternative_ids != expected_shared:
            raise ValueError(
                "Shared top alternatives must equal the "
                "intersection of group winner sets."
            )

        highest_count = max(
            item.group_count for item in self.support
        )
        expected_highest = tuple(
            item.alternative_id
            for item in self.support
            if item.group_count == highest_count
        )

        if self.highest_support_alternative_ids != expected_highest:
            raise ValueError(
                "Highest-support alternatives must contain every "
                "alternative with the maximum group count."
            )

        if (
            len(set(self.session_winner_ids))
            != len(self.session_winner_ids)
            or not set(self.session_winner_ids)
            <= known_alternatives
        ):
            raise ValueError(
                "Session winner IDs must be unique known "
                "alternatives."
            )

        return self


class GroupCriterionWeight(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    group_id: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)


class CriterionWeightDispersion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    criterion_id: str = Field(min_length=1)
    group_weights: tuple[
        GroupCriterionWeight, ...
    ] = Field(min_length=2)

    minimum_weight: float = Field(ge=0, le=1)
    maximum_weight: float = Field(ge=0, le=1)
    weight_range: float = Field(ge=0, le=1)

    mean_weight: float = Field(ge=0, le=1)
    population_standard_deviation: float = Field(ge=0)

    minimum_group_ids: tuple[str, ...] = Field(min_length=1)
    maximum_group_ids: tuple[str, ...] = Field(min_length=1)

    # Contextual reference only; exclude it from dispersion calculations.
    session_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_dispersion(self) -> Self:
        group_ids = tuple(
            item.group_id for item in self.group_weights
        )
        values = tuple(
            item.weight for item in self.group_weights
        )

        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "Criterion group weights require unique group IDs."
            )

        expected_minimum = min(values)
        expected_maximum = max(values)

        expected_range = expected_maximum - expected_minimum
        expected_mean = math.fsum(values) / len(values)
        expected_pstdev = math.sqrt(
            math.fsum(
                (value - expected_mean) ** 2
                for value in values
            )
            / len(values)
        )

        dispersion_values = (
            (
                self.minimum_weight,
                expected_minimum,
                "Minimum weight",
            ),
            (
                self.maximum_weight,
                expected_maximum,
                "Maximum weight",
            ),
            (
                self.weight_range,
                expected_range,
                "Weight range",
            ),
            (
                self.mean_weight,
                expected_mean,
                "Mean weight",
            ),
            (
                self.population_standard_deviation,
                expected_pstdev,
                "Population standard deviation",
            ),
        )

        for actual, expected, label in dispersion_values:
            if not math.isclose(
                actual,
                expected,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{label} must match the criterion group weights."
                )

        expected_minimum_groups = tuple(
            item.group_id
            for item in self.group_weights
            if math.isclose(
                item.weight,
                expected_minimum,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

        expected_maximum_groups = tuple(
            item.group_id
            for item in self.group_weights
            if math.isclose(
                item.weight,
                expected_maximum,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

        if self.minimum_group_ids != expected_minimum_groups:
            raise ValueError(
                "Minimum group IDs must match the criterion group "
                "weights in canonical group order."
            )

        if self.maximum_group_ids != expected_maximum_groups:
            raise ValueError(
                "Maximum group IDs must match the criterion group "
                "weights in canonical group order."
            )

        return self


class CrossGroupWeightDispersionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "cross-group-weight-dispersion-v1"
    ] = "cross-group-weight-dispersion-v1"

    evaluated_group_ids: tuple[str, ...] = Field(
        min_length=2
    )
    criteria: tuple[
        CriterionWeightDispersion, ...
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.evaluated_group_ids) != len(
            set(self.evaluated_group_ids)
        ):
            raise ValueError(
                "Evaluated group IDs must be unique."
            )

        criterion_ids = tuple(
            item.criterion_id for item in self.criteria
        )

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "Weight-dispersion criterion IDs must be unique."
            )

        for criterion in self.criteria:
            criterion_group_ids = tuple(
                item.group_id
                for item in criterion.group_weights
            )

            if criterion_group_ids != self.evaluated_group_ids:
                raise ValueError(
                    "Every criterion must contain exactly the "
                    "evaluated groups in canonical order."
                )

        return self


class AdjacentScoreRankMargin(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    higher_alternative_id: str = Field(min_length=1)
    lower_alternative_id: str = Field(min_length=1)

    higher_score: float = Field(ge=0, le=1)
    lower_score: float = Field(ge=0, le=1)
    score_margin: float = Field(ge=0, le=1)

    higher_statistical_rank: float = Field(ge=1)
    lower_statistical_rank: float = Field(ge=1)
    statistical_rank_margin: float = Field(ge=0)

    tied: bool

    @model_validator(mode="after")
    def validate_margins(self) -> Self:
        if self.higher_alternative_id == self.lower_alternative_id:
            raise ValueError(
                "Adjacent alternatives must have distinct IDs."
            )

        if self.higher_score < self.lower_score:
            raise ValueError(
                "The higher score must be at least the lower score."
            )

        expected_score_margin = (
            self.higher_score - self.lower_score
        )

        if not math.isclose(
            self.score_margin,
            expected_score_margin,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Score margin must equal the difference "
                "between scores."
            )

        if self.statistical_rank_margin != (
            self.lower_statistical_rank
            - self.higher_statistical_rank
        ):
            raise ValueError(
                "Statistical rank margin must equal the difference "
                "between ranks."
            )

        return self


class EntityScoreRankMargins(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    entity_id: str = Field(min_length=1)
    scope: Literal["group", "session"]
    tie_atol: float = Field(ge=0)

    entries: tuple[
        DiagnosticRankEntry, ...
    ] = Field(min_length=2)

    adjacent_margins: tuple[
        AdjacentScoreRankMargin, ...
    ] = Field(min_length=1)

    winner_ids: tuple[str, ...] = Field(min_length=1)

    top_score_margin: float = Field(ge=0, le=1)
    top_statistical_rank_margin: float = Field(ge=0)

    smallest_positive_score_margin: float | None = Field(
        default=None,
        gt=0,
        le=1,
    )

    @model_validator(mode="after")
    def validate_margins(self) -> Self:
        alternative_ids = tuple(
            entry.alternative_id for entry in self.entries
        )
        if len(set(alternative_ids)) != len(alternative_ids):
            raise ValueError(
                "Rank entries must have unique alternative IDs."
            )

        if any(
            higher.score < lower.score
            for higher, lower in zip(
                self.entries, self.entries[1:]
            )
        ):
            raise ValueError(
                "Rank entries must be ordered by descending score."
            )

        if len(self.adjacent_margins) != len(self.entries) - 1:
            raise ValueError(
                "Adjacent margins must contain one record for each "
                "consecutive entry pair."
            )

        for higher, lower, margin in zip(
            self.entries[:-1],
            self.entries[1:],
            self.adjacent_margins,
            strict=True,
        ):
            if (
                margin.higher_alternative_id != higher.alternative_id
                or margin.lower_alternative_id != lower.alternative_id
                or margin.higher_score != higher.score
                or margin.lower_score != lower.score
                or margin.score_margin != higher.score - lower.score
                or margin.higher_statistical_rank
                != higher.statistical_rank
                or margin.lower_statistical_rank
                != lower.statistical_rank
                or margin.statistical_rank_margin
                != lower.statistical_rank - higher.statistical_rank
            ):
                raise ValueError(
                    "Adjacent margins must match consecutive rank entries."
                )

            if margin.tied != (
                margin.score_margin <= self.tie_atol
            ):
                raise ValueError(
                    "Adjacent tied status must match the tie tolerance."
                )

        if len(set(self.winner_ids)) != len(self.winner_ids):
            raise ValueError("Winner IDs must be unique.")

        best_score = self.entries[0].score
        expected_winner_ids = {
            entry.alternative_id
            for entry in self.entries
            if best_score - entry.score <= self.tie_atol
        }
        if set(self.winner_ids) != expected_winner_ids:
            raise ValueError(
                "Winner IDs must contain every alternative within the "
                "tie tolerance of the best score."
            )

        first, second = self.entries[:2]

        if len(expected_winner_ids) > 1:
            expected_top_score_margin = 0.0
            expected_top_rank_margin = 0.0
        else:
            expected_top_score_margin = (
                first.score - second.score
            )
            expected_top_rank_margin = (
                second.statistical_rank
                - first.statistical_rank
            )

        if not math.isclose(
            self.top_score_margin,
            expected_top_score_margin,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Top score margin does not match the winning tier."
            )

        if not math.isclose(
            self.top_statistical_rank_margin,
            expected_top_rank_margin,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Top statistical rank margin does not match "
                "the winning tier."
            )

        positive_nontied_margins = tuple(
            margin.score_margin
            for margin in self.adjacent_margins
            if not margin.tied and margin.score_margin > 0
        )
        expected_smallest_positive_margin = (
            min(positive_nontied_margins)
            if positive_nontied_margins
            else None
        )
        if (
            self.smallest_positive_score_margin
            != expected_smallest_positive_margin
        ):
            raise ValueError(
                "Smallest positive score margin must match the "
                "smallest adjacent non-tied margin."
            )

        return self


class ScoreRankMarginResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "score-rank-margin-v1"
    ] = "score-rank-margin-v1"

    tie_atol: float = Field(ge=0)
    evaluated_group_ids: tuple[str, ...]
    session_id: str = Field(min_length=1)

    entities: tuple[
        EntityScoreRankMargins, ...
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entities(self) -> Self:
        if len(set(self.evaluated_group_ids)) != len(
            self.evaluated_group_ids
        ):
            raise ValueError("Evaluated group IDs must be unique.")

        expected_ids = self.evaluated_group_ids + (self.session_id,)
        actual_ids = tuple(entity.entity_id for entity in self.entities)
        if actual_ids != expected_ids:
            raise ValueError(
                "Entities must contain each configured group followed "
                "by exactly one session entity."
            )

        for entity, expected_scope in zip(
            self.entities,
            ("group",) * len(self.evaluated_group_ids) + ("session",),
            strict=True,
        ):
            if entity.scope != expected_scope:
                raise ValueError(
                    "Entity scopes must match configured group order "
                    "followed by the session."
                )
            if entity.tie_atol != self.tie_atol:
                raise ValueError(
                    "Every entity must use the result tie tolerance."
                )

        return self


class EntityConsistencyAssessment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    entity_id: str = Field(min_length=1)
    scope: Literal["group", "session"]

    consistency_ratio: float | None = Field(
        default=None,
        ge=0,
    )
    consistency_threshold: float | None = Field(
        default=None,
        ge=0,
    )
    excess_amount: float | None = Field(
        default=None,
        ge=0,
    )

    status: ConsistencyStatus
    weighting_source_kind: (
        Literal["calculated", "reused"] | None
    ) = None

    @model_validator(mode="after")
    def validate_consistency_assessment(self) -> Self:
        if self.status == "not_available":
            if (
                self.consistency_ratio is not None
                or self.excess_amount is not None
            ):
                raise ValueError(
                    "Unavailable consistency assessments must not "
                    "include a consistency ratio or excess amount."
                )
            return self

        if (
            self.consistency_ratio is None
            or self.consistency_threshold is None
        ):
            raise ValueError(
                "Evaluated consistency assessments require a ratio "
                "and threshold."
            )

        expected_exceeds = (
            self.consistency_ratio
            > self.consistency_threshold
        )
        expected_status = (
            "exceeds_threshold"
            if expected_exceeds
            else "within_threshold"
        )
        expected_excess = max(
            0.0,
            self.consistency_ratio
            - self.consistency_threshold,
        )

        if self.status != expected_status:
            raise ValueError(
                "Consistency status does not match the ratio "
                "and threshold."
            )
        if self.excess_amount is None or not math.isclose(
            self.excess_amount,
            expected_excess,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Excess amount does not match the ratio "
                "and threshold."
            )

        return self


class GroupParticipantConsistencySummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    group_id: str = Field(min_length=1)

    participant_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    exceeding_count: int = Field(ge=0)

    exceeding_fraction_of_evaluated: (
        float | None
    ) = Field(default=None, ge=0, le=1)

    maximum_consistency_ratio: float | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if (
            self.evaluated_count + self.unavailable_count
            != self.participant_count
        ):
            raise ValueError(
                "Evaluated and unavailable counts must sum "
                "to the participant count."
            )
        if self.exceeding_count > self.evaluated_count:
            raise ValueError(
                "Exceeding count cannot exceed evaluated count."
            )

        if self.evaluated_count == 0:
            if (
                self.exceeding_fraction_of_evaluated is not None
                or self.maximum_consistency_ratio is not None
            ):
                raise ValueError(
                    "Summaries without evaluated participants must "
                    "not include evaluated statistics."
                )
            return self

        expected_fraction = (
            self.exceeding_count / self.evaluated_count
        )
        if (
            self.exceeding_fraction_of_evaluated is None
            or not math.isclose(
                self.exceeding_fraction_of_evaluated,
                expected_fraction,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Exceeding fraction must match the evaluated counts."
            )
        if self.maximum_consistency_ratio is None:
            raise ValueError(
                "Summaries with evaluated participants require "
                "a maximum consistency ratio."
            )

        return self


class ConsistencyWarningResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "consistency-warning-v1"
    ] = "consistency-warning-v1"

    evaluated_group_ids: tuple[str, ...]
    session_id: str = Field(min_length=1)

    entity_assessments: tuple[
        EntityConsistencyAssessment, ...
    ] = Field(min_length=1)

    participant_group_summaries: tuple[
        GroupParticipantConsistencySummary, ...
    ]

    group_or_session_warning_ids: tuple[str, ...]
    participant_warning_group_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_consistency_warnings(self) -> Self:
        if len(set(self.evaluated_group_ids)) != len(
            self.evaluated_group_ids
        ):
            raise ValueError("Evaluated group IDs must be unique.")

        expected_entity_ids = (
            *self.evaluated_group_ids,
            self.session_id,
        )
        actual_entity_ids = tuple(
            assessment.entity_id
            for assessment in self.entity_assessments
        )
        if actual_entity_ids != expected_entity_ids:
            raise ValueError(
                "Entity assessments must follow evaluated group "
                "order and end with the session."
            )

        expected_scopes = (
            ("group",) * len(self.evaluated_group_ids)
            + ("session",)
        )
        actual_scopes = tuple(
            assessment.scope
            for assessment in self.entity_assessments
        )
        if actual_scopes != expected_scopes:
            raise ValueError(
                "Entity assessment scopes must match their "
                "group and session order."
            )

        summary_group_ids = tuple(
            summary.group_id
            for summary in self.participant_group_summaries
        )
        if summary_group_ids != self.evaluated_group_ids:
            raise ValueError(
                "Participant summaries must follow evaluated "
                "group order."
            )

        expected_entity_warnings = tuple(
            assessment.entity_id
            for assessment in self.entity_assessments
            if assessment.status == "exceeds_threshold"
        )
        if len(set(self.group_or_session_warning_ids)) != len(
            self.group_or_session_warning_ids
        ):
            raise ValueError("Warning IDs must be unique.")
        if (
            self.group_or_session_warning_ids
            != expected_entity_warnings
        ):
            raise ValueError(
                "Group and session warning IDs must match "
                "exceeding assessments in order."
            )

        expected_participant_warnings = tuple(
            summary.group_id
            for summary in self.participant_group_summaries
            if summary.exceeding_count > 0
        )
        if len(set(self.participant_warning_group_ids)) != len(
            self.participant_warning_group_ids
        ):
            raise ValueError("Warning IDs must be unique.")
        if (
            self.participant_warning_group_ids
            != expected_participant_warnings
        ):
            raise ValueError(
                "Participant warning group IDs must match "
                "summaries with exceeding participants in order."
            )

        return self


class EffectiveGroupPower(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    group_id: str = Field(min_length=1)
    effective_voting_power: float = Field(
        ge=0,
        le=1,
    )


class LeaveOneOutCriterionWeightChange(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    criterion_id: str = Field(min_length=1)
    baseline_weight: float = Field(ge=0, le=1)
    recomputed_weight: float = Field(ge=0, le=1)
    weight_delta: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def validate_weight_delta(self) -> Self:
        expected_delta = (
            self.recomputed_weight - self.baseline_weight
        )
        if not math.isclose(
            self.weight_delta,
            expected_delta,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Weight delta must equal the recomputed weight "
                "minus the baseline weight."
            )
        return self


class LeaveOneOutAlternativeChange(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    alternative_id: str = Field(min_length=1)

    baseline_score: float = Field(ge=0, le=1)
    recomputed_score: float = Field(ge=0, le=1)
    score_delta: float = Field(ge=-1, le=1)

    baseline_statistical_rank: float = Field(ge=1)
    recomputed_statistical_rank: float = Field(ge=1)
    statistical_rank_delta: float

    @model_validator(mode="after")
    def validate_deltas(self) -> Self:
        expected_score_delta = (
            self.recomputed_score - self.baseline_score
        )
        if not math.isclose(
            self.score_delta,
            expected_score_delta,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Score delta must equal the recomputed score "
                "minus the baseline score."
            )

        expected_rank_delta = (
            self.recomputed_statistical_rank
            - self.baseline_statistical_rank
        )
        if not math.isclose(
            self.statistical_rank_delta,
            expected_rank_delta,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Statistical rank delta must equal the recomputed "
                "rank minus the baseline rank."
            )

        return self


class LeaveOneGroupOutCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    omitted_group_id: str = Field(min_length=1)
    remaining_group_ids: tuple[str, ...] = Field(
        min_length=1
    )

    status: Literal["completed", "not_evaluable"]
    not_evaluable_reason: str | None = None

    effective_voting_powers: tuple[
        EffectiveGroupPower, ...
    ] = ()

    baseline_winner_ids: tuple[str, ...] = ()
    recomputed_winner_ids: tuple[str, ...] = ()
    winner_changed: bool | None = None

    criterion_weight_changes: tuple[
        LeaveOneOutCriterionWeightChange, ...
    ] = ()

    alternative_changes: tuple[
        LeaveOneOutAlternativeChange, ...
    ] = ()

    maximum_absolute_weight_delta: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    maximum_absolute_score_delta: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    maximum_absolute_rank_delta: float | None = Field(
        default=None,
        ge=0,
    )

    recomputed_consistency_ratio: float | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if (
            not self.baseline_winner_ids
            or len(self.baseline_winner_ids)
            != len(set(self.baseline_winner_ids))
        ):
            raise ValueError(
                "Baseline winner IDs must be nonempty and unique."
            )

        if self.status == "not_evaluable":
            if (
                not (self.not_evaluable_reason or "").strip()
                or self.effective_voting_powers
                or self.winner_changed is not None
                or self.recomputed_winner_ids
                or self.criterion_weight_changes
                or self.alternative_changes
                or self.maximum_absolute_weight_delta is not None
                or self.maximum_absolute_score_delta is not None
                or self.maximum_absolute_rank_delta is not None
                or self.recomputed_consistency_ratio is not None
            ):
                raise ValueError(
                    "Unavailable omissions require a reason and no "
                    "recomputed data."
                )
            return self

        if self.not_evaluable_reason is not None:
            raise ValueError(
                "Completed omission cases cannot have a "
                "not-evaluable reason."
            )

        effective_group_ids = tuple(
            item.group_id
            for item in self.effective_voting_powers
        )
        if effective_group_ids != self.remaining_group_ids:
            raise ValueError(
                "Effective voting powers must follow remaining group order."
            )

        total_effective_power = sum(
            item.effective_voting_power
            for item in self.effective_voting_powers
        )
        if not math.isclose(
            total_effective_power,
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Effective voting powers must sum to one."
            )

        if len(self.recomputed_winner_ids) != len(
            set(self.recomputed_winner_ids)
        ):
            raise ValueError(
                "Recomputed winner IDs must be unique."
            )

        if not self.baseline_winner_ids or not self.recomputed_winner_ids:
            raise ValueError(
                "Completed omission cases require baseline and "
                "recomputed winners."
            )

        expected_winner_changed = (
            set(self.baseline_winner_ids)
            != set(self.recomputed_winner_ids)
        )
        if self.winner_changed is not expected_winner_changed:
            raise ValueError(
                "Winner-changed status must match the winner sets."
            )

        if not self.criterion_weight_changes:
            raise ValueError(
                "Completed omission cases require criterion weight changes."
            )
        if not self.alternative_changes:
            raise ValueError(
                "Completed omission cases require alternative changes."
            )

        criterion_ids = tuple(
            change.criterion_id
            for change in self.criterion_weight_changes
        )
        alternative_ids = tuple(
            change.alternative_id
            for change in self.alternative_changes
        )

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "Criterion weight changes require unique IDs."
            )

        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError(
                "Alternative changes require unique IDs."
            )

        expected_maximum_weight_delta = max(
            abs(change.weight_delta)
            for change in self.criterion_weight_changes
        )
        expected_maximum_score_delta = max(
            abs(change.score_delta)
            for change in self.alternative_changes
        )
        expected_maximum_rank_delta = max(
            abs(change.statistical_rank_delta)
            for change in self.alternative_changes
        )

        for actual, expected, label in (
            (
                self.maximum_absolute_weight_delta,
                expected_maximum_weight_delta,
                "weight",
            ),
            (
                self.maximum_absolute_score_delta,
                expected_maximum_score_delta,
                "score",
            ),
            (
                self.maximum_absolute_rank_delta,
                expected_maximum_rank_delta,
                "rank",
            ),
        ):
            if actual is None or not math.isclose(
                actual,
                expected,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Maximum absolute {label} delta must match "
                    "the change records."
                )

        return self


class LeaveOneGroupOutResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[
        "leave-one-group-out-v1"
    ] = "leave-one-group-out-v1"

    tie_atol: float = Field(ge=0)
    evaluated_group_ids: tuple[str, ...] = Field(
        min_length=2
    )
    session_id: str = Field(min_length=1)

    baseline_winner_ids: tuple[str, ...] = Field(
        min_length=1
    )

    cases: tuple[
        LeaveOneGroupOutCase, ...
    ] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if len(set(self.evaluated_group_ids)) != len(
            self.evaluated_group_ids
        ):
            raise ValueError("Evaluated group IDs must be unique.")

        if len(self.cases) != len(self.evaluated_group_ids):
            raise ValueError(
                "There must be exactly one omission case per "
                "evaluated group."
            )

        case_omitted_group_ids = tuple(
            case.omitted_group_id for case in self.cases
        )
        if case_omitted_group_ids != self.evaluated_group_ids:
            raise ValueError(
                "Omission cases must follow evaluated group order."
            )

        for case, omitted_group_id in zip(
            self.cases,
            self.evaluated_group_ids,
            strict=True,
        ):
            expected_remaining_group_ids = tuple(
                group_id
                for group_id in self.evaluated_group_ids
                if group_id != omitted_group_id
            )
            if (
                case.omitted_group_id != omitted_group_id
                or case.remaining_group_ids
                != expected_remaining_group_ids
            ):
                raise ValueError(
                    "Each omission case must omit its corresponding "
                    "group and retain all other groups in order."
                )

        if len(self.baseline_winner_ids) != len(
            set(self.baseline_winner_ids)
        ):
            raise ValueError(
                "Baseline winner IDs must be unique."
            )

        if any(
            case.baseline_winner_ids
            != self.baseline_winner_ids
            for case in self.cases
        ):
            raise ValueError(
                "Every omission case must use the result's "
                "baseline winner IDs."
            )

        completed_cases = tuple(
            case
            for case in self.cases
            if case.status == "completed"
        )

        if completed_cases:
            expected_criterion_ids = tuple(
                change.criterion_id
                for change
                in completed_cases[0].criterion_weight_changes
            )
            expected_alternative_ids = tuple(
                change.alternative_id
                for change
                in completed_cases[0].alternative_changes
            )

            for case in completed_cases[1:]:
                if tuple(
                    change.criterion_id
                    for change in case.criterion_weight_changes
                ) != expected_criterion_ids:
                    raise ValueError(
                        "Completed omission cases must use the "
                        "same criterion order."
                    )

                if tuple(
                    change.alternative_id
                    for change in case.alternative_changes
                ) != expected_alternative_ids:
                    raise ValueError(
                        "Completed omission cases must use the "
                        "same alternative order."
                    )

        return self


class CriterionRemovalCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    removed_criterion_id: str = Field(min_length=1)
    retained_criterion_ids: tuple[str, ...]
    status: Literal["completed", "not_evaluable"]
    reason: str | None = None
    applied_weights: dict[str, float] | None = None
    alternative_changes: tuple[LeaveOneOutAlternativeChange, ...] = ()
    winner_ids: tuple[str, ...] = ()
    winner_changed: bool | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.status == "not_evaluable":
            if (
                not (self.reason or "").strip()
                or self.applied_weights is not None
                or self.alternative_changes
                or self.winner_ids
                or self.winner_changed is not None
            ):
                raise ValueError(
                    "Unavailable removals require a reason and no outcomes."
                )
            return self

        if (
            self.reason is not None
            or not self.retained_criterion_ids
            or self.applied_weights is None
            or set(self.applied_weights) != set(self.retained_criterion_ids)
            or not self.alternative_changes
            or not self.winner_ids
            or self.winner_changed is None
        ):
            raise ValueError("Completed removals require retained weights and outcomes.")
        if any(
            not 0 <= weight <= 1 for weight in self.applied_weights.values()
        ) or not math.isclose(
            math.fsum(self.applied_weights.values()), 1.0,
            rel_tol=0, abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError("Retained weights must be normalized.")
        return self


class CriterionRemovalResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    schema_version: Literal["criterion-removal-v1"] = "criterion-removal-v1"
    session_id: str = Field(min_length=1)
    group_id: str | None = Field(default=None, min_length=1)
    tie_atol: float = Field(ge=0)
    ranking_criterion_ids: tuple[str, ...] = Field(min_length=1)
    baseline_applied_weights: dict[str, float]
    alternative_ids: tuple[str, ...] = Field(min_length=1)
    baseline_winner_ids: tuple[str, ...] = Field(min_length=1)
    cases: tuple[CriterionRemovalCase, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        criterion_ids = self.ranking_criterion_ids
        if (
            len(set(criterion_ids)) != len(criterion_ids)
            or set(self.baseline_applied_weights) != set(criterion_ids)
            or tuple(case.removed_criterion_id for case in self.cases) != criterion_ids
        ):
            raise ValueError(
                "Baseline weights and removal cases must cover the ranking criteria."
            )
        if any(
            not 0 <= weight <= 1 for weight in self.baseline_applied_weights.values()
        ) or not math.isclose(
            math.fsum(self.baseline_applied_weights.values()), 1.0,
            rel_tol=0, abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError("Baseline applied weights must be normalized.")

        alternatives = set(self.alternative_ids)
        baseline_winners = set(self.baseline_winner_ids)
        if (
            len(alternatives) != len(self.alternative_ids)
            or len(baseline_winners) != len(self.baseline_winner_ids)
            or not baseline_winners <= alternatives
        ):
            raise ValueError("Baseline alternatives and winners must be unique and aligned.")

        baseline_outcomes: tuple[tuple[float, float], ...] | None = None
        for case in self.cases:
            expected_retained = tuple(
                cid for cid in criterion_ids if cid != case.removed_criterion_id
            )
            if case.retained_criterion_ids != expected_retained:
                raise ValueError("Each case must remove exactly its declared criterion.")
            if case.status != "completed":
                continue

            applied = case.applied_weights
            mass = math.fsum(self.baseline_applied_weights[cid] for cid in expected_retained)
            if applied is None or mass <= 0 or any(
                not math.isclose(
                    applied[cid], self.baseline_applied_weights[cid] / mass,
                    rel_tol=0, abs_tol=WEIGHT_SUM_TOLERANCE,
                )
                for cid in expected_retained
            ):
                raise ValueError("Retained weights must be proportional to baseline weights.")
            change_ids = tuple(change.alternative_id for change in case.alternative_changes)
            if change_ids != self.alternative_ids:
                raise ValueError("Removal outcomes must follow baseline alternative order.")

            case_baseline_outcomes = tuple(
                (change.baseline_score, change.baseline_statistical_rank)
                for change in case.alternative_changes
            )
            if baseline_outcomes is None:
                baseline_outcomes = case_baseline_outcomes
                baseline_best = max(change.baseline_score for change in case.alternative_changes)
                expected_baseline_winners = {
                    change.alternative_id
                    for change in case.alternative_changes
                    if baseline_best - change.baseline_score <= self.tie_atol
                }
                if baseline_winners != expected_baseline_winners:
                    raise ValueError("Baseline winner IDs must match the baseline scores.")
            elif case_baseline_outcomes != baseline_outcomes:
                raise ValueError("Completed removals must share baseline scores and ranks.")

            winners = set(case.winner_ids)
            best_score = max(change.recomputed_score for change in case.alternative_changes)
            expected_winners = {
                change.alternative_id for change in case.alternative_changes
                if best_score - change.recomputed_score <= self.tie_atol
            }
            if (
                len(winners) != len(case.winner_ids)
                or winners != expected_winners
                or case.winner_changed is not (winners != baseline_winners)
            ):
                raise ValueError("Removal winner evidence must match the scores and baseline.")
        return self


class AlternativeRemovalCase(BaseModel):
    """Compare a reranking with baseline scores and ranks among survivors."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    removed_alternative_id: str = Field(min_length=1)
    retained_alternative_ids: tuple[str, ...]
    baseline_winner_removed: bool
    baseline_survivor_winner_ids: tuple[str, ...]
    status: Literal["completed", "not_evaluable"]
    reason: str | None = None
    alternative_changes: tuple[LeaveOneOutAlternativeChange, ...] = ()
    recomputed_winner_ids: tuple[str, ...] = ()
    survivor_winner_changed: bool | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status == "not_evaluable":
            if (
                not (self.reason or "").strip()
                or self.alternative_changes
                or self.recomputed_winner_ids
                or self.survivor_winner_changed is not None
            ):
                raise ValueError("Unavailable alternative removals require a reason and no outcomes.")
        elif (
            self.reason is not None
            or len(self.retained_alternative_ids) < 2
            or not self.recomputed_winner_ids
            or self.survivor_winner_changed is None
            or tuple(change.alternative_id for change in self.alternative_changes)
            != self.retained_alternative_ids
        ):
            raise ValueError("Completed alternative removals require outcomes for all survivors.")
        return self


class AlternativeRemovalResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    schema_version: Literal["alternative-removal-v1"] = "alternative-removal-v1"
    session_id: str = Field(min_length=1)
    group_id: str | None = Field(default=None, min_length=1)
    tie_atol: float = Field(ge=0)
    ranking_criterion_ids: tuple[str, ...] = Field(min_length=1)
    applied_weights: dict[str, float]
    baseline_entries: tuple[DiagnosticRankEntry, ...] = Field(min_length=1)
    baseline_winner_ids: tuple[str, ...] = Field(min_length=1)
    cases: tuple[AlternativeRemovalCase, ...]

    def _winner_set(self, scores: dict[str, float]) -> set[str]:
        if not scores:
            return set()
        best = max(scores.values())
        return {aid for aid, score in scores.items() if best - score <= self.tie_atol}

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if (
            len(set(self.ranking_criterion_ids)) != len(self.ranking_criterion_ids)
            or set(self.applied_weights) != set(self.ranking_criterion_ids)
            or any(not 0 <= weight <= 1 for weight in self.applied_weights.values())
            or not math.isclose(
                math.fsum(self.applied_weights.values()), 1.0,
                rel_tol=0, abs_tol=WEIGHT_SUM_TOLERANCE,
            )
        ):
            raise ValueError("Applied weights must cover the ranking criteria and sum to one.")

        scores = {entry.alternative_id: entry.score for entry in self.baseline_entries}
        expected_winners = self._winner_set(scores)
        if (
            len(scores) != len(self.baseline_entries)
            or self.baseline_winner_ids != tuple(sorted(expected_winners))
            or tuple(case.removed_alternative_id for case in self.cases) != tuple(scores)
        ):
            raise ValueError("Baseline ranking and removal cases must cover the same alternatives.")

        for case in self.cases:
            removed_id = case.removed_alternative_id
            survivor_scores = {aid: score for aid, score in scores.items() if aid != removed_id}
            survivor_winners = self._winner_set(survivor_scores)
            if (
                case.retained_alternative_ids != tuple(survivor_scores)
                or case.baseline_winner_removed != (removed_id in expected_winners)
                or case.baseline_survivor_winner_ids != tuple(sorted(survivor_winners))
            ):
                raise ValueError("Removal case must match the baseline and its surviving alternatives.")
            if case.status == "completed":
                new_scores = {
                    change.alternative_id: change.recomputed_score
                    for change in case.alternative_changes
                }
                new_winners = self._winner_set(new_scores)
                if (
                    case.recomputed_winner_ids != tuple(sorted(new_winners))
                    or case.survivor_winner_changed != (new_winners != survivor_winners)
                    or any(
                        change.baseline_score != survivor_scores[change.alternative_id]
                        for change in case.alternative_changes
                    )
                ):
                    raise ValueError("Removal outcomes must agree with the scores and winner sets.")
        return self


class BoundedWeightPerturbationCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    criterion_id: str = Field(min_length=1)
    requested_delta: float = Field(gt=-1, lt=1)
    status: Literal["completed", "not_evaluable"]
    reason: str | None = None

    applied_weights: dict[str, float] | None = None
    entries: tuple[DiagnosticRankEntry, ...] = ()
    winner_ids: tuple[str, ...] = ()
    winner_changed: bool | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.status == "not_evaluable":
            if (
                not (self.reason or "").strip()
                or self.applied_weights is not None
                or self.entries
                or self.winner_ids
                or self.winner_changed is not None
            ):
                raise ValueError(
                    "Unavailable perturbations require a reason and "
                    "no outcome data."
                )
            return self

        if (
            self.reason is not None
            or self.applied_weights is None
            or not self.entries
            or not self.winner_ids
            or self.winner_changed is None
        ):
            raise ValueError(
                "Completed perturbations require outcome data and "
                "cannot have a reason."
            )

        if any(
            weight < 0 or weight > 1
            for weight in self.applied_weights.values()
        ) or not math.isclose(
            math.fsum(self.applied_weights.values()),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Applied weights must be a normalized weight vector."
            )

        alternative_ids = tuple(
            entry.alternative_id for entry in self.entries
        )
        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError(
                "Perturbation entries require unique alternative IDs."
            )

        if (
            len(self.winner_ids) != len(set(self.winner_ids))
            or not set(self.winner_ids) <= set(alternative_ids)
        ):
            raise ValueError(
                "Winner IDs must be unique alternatives in the entries."
            )

        return self


class BoundedWeightPerturbationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False,
    )

    schema_version: Literal["bounded-weight-perturbations-v1"] = (
        "bounded-weight-perturbations-v1"
    )
    session_id: str = Field(min_length=1)
    group_id: str | None = Field(default=None, min_length=1)
    absolute_weight_delta: float = Field(gt=0, lt=1)
    tie_atol: float = Field(ge=0)
    ranking_criterion_ids: tuple[str, ...] = Field(min_length=1)
    baseline_applied_weights: dict[str, float]
    baseline_entries: tuple[DiagnosticRankEntry, ...] = Field(min_length=1)
    baseline_winner_ids: tuple[str, ...] = Field(min_length=1)
    cases: tuple[BoundedWeightPerturbationCase, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        criterion_ids = self.ranking_criterion_ids
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Ranking criterion IDs must be unique.")

        if set(self.baseline_applied_weights) != set(criterion_ids):
            raise ValueError(
                "Baseline applied weights must cover exactly the "
                "ranking criteria."
            )

        if any(
            weight < 0 or weight > 1
            for weight in self.baseline_applied_weights.values()
        ) or not math.isclose(
            math.fsum(self.baseline_applied_weights.values()),
            1.0,
            rel_tol=0,
            abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise ValueError(
                "Baseline applied weights must be a normalized "
                "weight vector."
            )

        baseline_alternative_ids = tuple(
            entry.alternative_id for entry in self.baseline_entries
        )
        if len(baseline_alternative_ids) != len(
            set(baseline_alternative_ids)
        ):
            raise ValueError(
                "Baseline entries require unique alternative IDs."
            )

        if (
            len(self.baseline_winner_ids)
            != len(set(self.baseline_winner_ids))
            or not set(self.baseline_winner_ids)
            <= set(baseline_alternative_ids)
        ):
            raise ValueError(
                "Baseline winner IDs must be unique alternatives "
                "in the baseline entries."
            )

        if len(self.cases) != 2 * len(criterion_ids):
            raise ValueError(
                "Each ranking criterion requires exactly two "
                "perturbation cases."
            )

        baseline_criteria = set(self.baseline_applied_weights)
        expected_case_order = tuple(
            (criterion_id, requested_delta)
            for criterion_id in criterion_ids
            for requested_delta in (
                -self.absolute_weight_delta,
                self.absolute_weight_delta,
            )
        )
        for case, (criterion_id, requested_delta) in zip(
            self.cases,
            expected_case_order,
            strict=True,
        ):
            if (
                case.criterion_id != criterion_id
                or case.requested_delta != requested_delta
            ):
                raise ValueError(
                    "Perturbation cases must follow ranking criterion "
                    "order with negative then positive deltas."
                )

            if case.status == "completed":
                if (
                    set(case.applied_weights or {}) != baseline_criteria
                ):
                    raise ValueError(
                        "Completed perturbation weights must cover "
                        "exactly the baseline criteria."
                    )

                alternative_ids = tuple(
                    entry.alternative_id for entry in case.entries
                )
                if alternative_ids != baseline_alternative_ids:
                    raise ValueError(
                        "Completed perturbation entries must follow "
                        "baseline alternative order."
                    )

                expected_winner_changed = (
                    set(case.winner_ids)
                    != set(self.baseline_winner_ids)
                )
                if case.winner_changed is not expected_winner_changed:
                    raise ValueError(
                        "Winner-changed status must match the winner sets."
                    )

        return self
