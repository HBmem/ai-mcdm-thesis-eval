"""Service layer for AHP weight generation and TOPSIS ranking within a decision session."""
from dataclasses import dataclass
from typing import Literal
from itertools import combinations

import numpy as np

from pyDecision.algorithm import ahp_method, topsis_method

from src.mcdm.enums import (
    CriterionDirection,
    RankingMethod,
    WeightingMethod,
    PairwisePreference,
    MeanMethod,
    GroupPowerMethod,
    EmptyGroupPolicy,
)
from src.mcdm.models.domain import Participant
from src.mcdm.models.methods import RankingMethodConfig, WeightingMethodConfig, AggregationMethodConfig, ConsistencyMethodConfig
from src.mcdm.models.session import DecisionSession
from src.mcdm.models.computation import WeightResult, CriterionWeight, RankingResult, AlternativeRanking


ROUNDOFF_TOLERANCE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class SessionAggregationResult:
    matrix: np.ndarray
    effective_voting_powers: dict[str, float]


def validate_weight_vector(
    values,
    *,
    expected_count: int,
    label: str,
    allow_zero: bool = True,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: weights must form a numeric vector."
        ) from error

    # Reject booleans, strings, complex numbers, and object arrays.
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            f"{label}: weights must contain real numeric values."
        )

    if raw.shape != (expected_count,):
        raise ValueError(
            f"{label}: expected shape {(expected_count,)}, "
            f"received {raw.shape}."
        )

    weights = raw.astype(float, copy=True)

    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{label}: weights must be finite.")

    if np.any(weights < 0) or np.any(weights > 1):
        raise ValueError(
            f"{label}: weights must be between zero and one."
        )

    if not allow_zero and np.any(weights == 0):
        raise ValueError(
            f"{label}: weights must be strictly positive."
        )

    total = float(weights.sum())

    if not np.isclose(
        total,
        1.0,
        atol=WEIGHT_SUM_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(
            f"{label}: weights must already sum to one; "
            f"received {total!r}."
        )

    return weights


def build_ranking_weighting(
    weighting: WeightResult,
    *,
    full_criterion_ids: list[str] | tuple[str, ...],
    ranking_criterion_ids: list[str] | tuple[str, ...],
    override: dict[str, float] | None = None,
) -> tuple[
    WeightResult,
    dict[str, float],
    Literal[
        "original",
        "restricted_and_renormalized",
        "explicit",
    ],
]:
    full_ids = list(full_criterion_ids)
    selected_ids = list(ranking_criterion_ids)

    if (
        not full_ids
        or len(full_ids) != len(set(full_ids))
    ):
        raise ValueError(
            "Full criterion IDs must be nonempty and unique."
        )

    if (
        not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or not set(selected_ids) <= set(full_ids)
    ):
        raise ValueError(
            "Ranking criteria must be a nonempty, unique "
            "subset of the full criteria."
        )

    items = weighting.criterion_weights
    by_id = {
        item.criterion_id: item
        for item in items
    }

    if (
        len(by_id) != len(items)
        or set(by_id) != set(full_ids)
    ):
        raise ValueError(
            "Original weights must cover every full criterion "
            "exactly once."
        )

    original = validate_weight_vector(
        [by_id[cid].weight for cid in full_ids],
        expected_count=len(full_ids),
        label="Original weighting",
    )

    original_by_id = dict(zip(full_ids, original))

    if override is not None:
        if (
            not isinstance(override, dict)
            or set(override) != set(selected_ids)
        ):
            raise ValueError(
                "Explicit ranking weights must cover exactly "
                "the selected criteria."
            )

        applied = validate_weight_vector(
            [override[cid] for cid in selected_ids],
            expected_count=len(selected_ids),
            label="Explicit ranking weights",
        )

        source = "explicit"

    else:
        applied = np.asarray([
            original_by_id[cid]
            for cid in selected_ids
        ])

        if set(selected_ids) != set(full_ids):
            retained_mass = float(applied.sum())

            if retained_mass <= 0:
                raise ValueError(
                    "Retained criteria have no positive weight."
                )

            applied = applied / retained_mass
            source = "restricted_and_renormalized"

        else:
            # Reordering is allowed; repairing the total is not.
            source = "original"

    ranking_weighting = weighting.model_copy(deep=True)

    ranking_weighting.criterion_weights = [
        CriterionWeight(
            criterion_id=cid,
            criterion_name=by_id[cid].criterion_name,
            weight=float(value),
        )
        for cid, value in zip(
            selected_ids,
            applied,
            strict=True,
        )
    ]

    applied_by_id = dict(zip(
        selected_ids,
        applied.tolist(),
    ))

    return ranking_weighting, applied_by_id, source


class MCDMService:
    """
    Performs deterministic weight generation and ranking calculations for a DecisionSession.
    """

    @staticmethod
    def create_participant_matrix(
        session: DecisionSession,
        participant: Participant,
    ) -> np.ndarray:
        criterion_ids = [
            criterion.id
            for criterion in session.criteria
        ]

        if len(criterion_ids) < 2:
            raise ValueError("At least two criteria are required.")

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Duplicate criterion IDs.")

        criteria_index = {
            criterion_id: index
            for index, criterion_id in enumerate(criterion_ids)
        }

        questions = {
            question.id: question
            for question in session.questions
        }

        if len(questions) != len(session.questions):
            raise ValueError("Duplicate question IDs.")

        # Ensure every distinct criterion pair is defined exactly once.
        actual_pairs = []

        for question in session.questions:
            left = question.left_criterion_id
            right = question.right_criterion_id

            if (
                left not in criteria_index
                or right not in criteria_index
                or left == right
            ):
                raise ValueError(
                    f"Invalid criterion pair in question {question.id!r}."
                )

            actual_pairs.append(frozenset((left, right)))

        expected_pairs = {
            frozenset(pair)
            for pair in combinations(criterion_ids, 2)
        }

        if (
            len(actual_pairs) != len(set(actual_pairs))
            or set(actual_pairs) != expected_pairs
        ):
            raise ValueError(
                "Questions must cover every criterion pair exactly once."
            )

        response_ids = [
            response.question_id
            for response in participant.responses
        ]

        if (
            len(response_ids) != len(set(response_ids))
            or set(response_ids) != set(questions)
        ):
            raise ValueError(
                f"Participant {participant.id!r} must answer "
                "every question exactly once."
            )

        matrix = np.ones(
            (len(criterion_ids), len(criterion_ids)),
            dtype=float,
        )

        for response in participant.responses:
            if not isinstance(response.preference, PairwisePreference):
                raise ValueError(
                    "Judgments must be parsed PairwisePreference values."
                )

            ratio = float(response.preference.ahp_scale)

            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Preference ratios must be finite and positive.")

            question = questions[response.question_id]

            left = criteria_index[question.left_criterion_id]
            right = criteria_index[question.right_criterion_id]

            matrix[left, right] = ratio
            matrix[right, left] = 1.0 / ratio

        return matrix

    @classmethod
    def create_stakeholder_group_matrix(
        cls,
        session: DecisionSession,
        participants: list[Participant],
        *,
        config: AggregationMethodConfig,
    ) -> np.ndarray:
        if not participants:
            raise ValueError(
                "At least one participant is required for group aggregation."
            )

        participant_ids = [
            participant.id
            for participant in participants
        ]

        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Duplicate participants in group aggregation.")

        group_ids = {
            participant.stakeholder_group_id
            for participant in participants
        }

        if len(group_ids) != 1:
            raise ValueError(
                "Group aggregation requires participants from one group."
            )

        known_groups = {
            group.id
            for group in session.stakeholder_groups
        }

        if not group_ids <= known_groups:
            raise ValueError("Participants reference an unknown group.")

        matrices = np.stack([
            cls.create_participant_matrix(session, participant)
            for participant in participants
        ])

        if config.within_group == MeanMethod.GEOMETRIC_MEAN:
            return np.exp(
                np.mean(np.log(matrices), axis=0)
            )

        raise NotImplementedError(
            f"Within-group aggregation {config.within_group!r} "
            "is not implemented."
        )

    @classmethod
    def create_session_aggregation(
        cls,
        session: DecisionSession,
        *,
        config: AggregationMethodConfig,
    ) -> SessionAggregationResult:
        if config.empty_group_policy != EmptyGroupPolicy.EXCLUDED:
            raise NotImplementedError(
                "Only exclusion of empty groups is implemented."
            )

        if (
            config.group_power_normalization
            != GroupPowerMethod.REPRESENTED_GROUPS
        ):
            raise NotImplementedError(
                "Only represented-group power normalization is implemented."
            )

        if (
            config.between_groups
            != MeanMethod.WEIGHTED_GEOMETRIC_MEAN
        ):
            raise NotImplementedError(
                f"Between-group aggregation {config.between_groups!r} "
                "is not implemented."
            )

        group_ids = [
            group.id
            for group in session.stakeholder_groups
        ]

        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Duplicate stakeholder group IDs.")

        if any(
            participant.stakeholder_group_id not in set(group_ids)
            for participant in session.participants
        ):
            raise ValueError("A participant references an unknown group.")

        represented_groups = []
        group_matrices = []

        for group in session.stakeholder_groups:
            members = [
                participant
                for participant in session.participants
                if participant.stakeholder_group_id == group.id
            ]

            if not members:
                continue

            represented_groups.append(group)
            group_matrices.append(
                cls.create_stakeholder_group_matrix(
                    session,
                    members,
                    config=config,
                )
            )

        if not group_matrices:
            raise ValueError("No represented stakeholder groups.")

        return cls.aggregate_group_matrices(
            group_ids=tuple(
                group.id for group in represented_groups
            ),
            group_matrices={
                group.id: matrix
                for group, matrix in zip(
                    represented_groups,
                    group_matrices,
                    strict=True,
                )
            },
            voting_powers={
                group.id: float(group.voting_power)
                for group in represented_groups
            },
            config=config,
        )


    @classmethod
    def create_session_matrix(
        cls,
        session: DecisionSession,
        *,
        config: AggregationMethodConfig,
    ) -> np.ndarray:
        return cls.create_session_aggregation(
            session,
            config=config,
        ).matrix


    @staticmethod
    def create_decision_matrix(
        session: DecisionSession,
    ) -> np.ndarray:
        criterion_order = [
            criterion.id
            for criterion in session.criteria
        ]

        rows = []

        for alternative in session.alternatives:
            values = {
                item.criterion_id: item.value
                for item in alternative.criterion_values
            }

            missing = set(criterion_order) - set(values)

            if missing:
                raise ValueError(
                    f"Alternative '{alternative.name}' is missing "
                    f"criterion values for: {sorted(missing)}"
                )

            rows.append(
                [
                    values[criterion_id]
                    for criterion_id in criterion_order
                ]
            )

        return np.asarray(rows, dtype=float)

# ============================================================
# WEIGHT GENERATION
# ============================================================

    # @staticmethod
    def calculate_ahp(
        pairwise_matrix: np.ndarray,
        *,
        wd: Literal["mean", "m", "geometric", "g", "max_eigen", "me"] = "geometric",
    ) -> tuple[np.ndarray, float]:
        # Validate dimensions, reciprocity, positivity, and supported derivation.
        try:
            pairwise_matrix = np.asarray(pairwise_matrix, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("AHP pairwise matrix must contain numeric values.") from exc

        if (
            pairwise_matrix.ndim != 2
            or pairwise_matrix.shape[0] != pairwise_matrix.shape[1]
            or pairwise_matrix.shape[0] < 2
        ):
            raise ValueError("AHP pairwise matrix must be square and at least 2x2.")

        if not np.all(np.isfinite(pairwise_matrix)):
            raise ValueError("AHP pairwise matrix must contain only finite values.")

        if np.any(pairwise_matrix <= 0):
            raise ValueError("AHP pairwise matrix entries must be positive.")

        if not np.allclose(
            np.diag(pairwise_matrix), 1.0, rtol=0.0, atol=ROUNDOFF_TOLERANCE
        ):
            raise ValueError("AHP pairwise matrix diagonal entries must be 1.")

        if not np.allclose(
            np.log(pairwise_matrix) + np.log(pairwise_matrix.T),
            0.0,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("AHP pairwise matrix must be reciprocal.")

        supported_weight_derivations = {
            "mean",
            "m",
            "geometric",
            "g",
            "max_eigen",
            "me",
        }
        if not isinstance(wd, str) or wd not in supported_weight_derivations:
            raise ValueError(f"Unsupported AHP weight derivation method: {wd!r}.")

        criterion_count = pairwise_matrix.shape[0]

        # Apply your explicit two-criterion consistency policy.
        if criterion_count > 15:
            raise ValueError(
                "AHP consistency ratios are supported for at most 15 criteria."
            )

        if criterion_count == 2:
            ratio = pairwise_matrix[0, 1]

            weights = np.asarray([
                ratio / (1.0 + ratio),
                1.0 / (1.0 + ratio),
            ])

            consistency_ratio = 0.0

        else:
            # Perform the PyDecision calculation.
            weights, consistency_ratio = ahp_method(
                pairwise_matrix.copy(),
                wd=wd,
            )

        weights = validate_weight_vector(
            weights,
            expected_count=criterion_count,
            label="AHP output",
            allow_zero=False,
        )

        consistency_ratio = float(consistency_ratio)

        if not np.isfinite(consistency_ratio):
            raise ValueError(
                "AHP returned a nonfinite consistency ratio."
            )

        if consistency_ratio < -ROUNDOFF_TOLERANCE:
            raise ValueError(
                "AHP returned a materially negative consistency ratio."
            )

        consistency_ratio = max(0.0, consistency_ratio)

        return weights, consistency_ratio

# ============================================================
# RANKING GENERATION
# ============================================================

    @staticmethod
    def calculate_topsis(
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        directions: list[str],
        parameters: dict,
    ) -> np.ndarray:
        unknown = set(parameters) - {"normalization"}

        if unknown:
            raise ValueError(
                f"Unsupported TOPSIS parameters: {sorted(unknown)}"
            )

        if parameters.get("normalization", "vector") != "vector":
            raise ValueError(
                "Only vector TOPSIS normalization is implemented."
            )

        if any(direction not in {"min", "max"} for direction in directions):
            raise ValueError("Invalid criterion direction.")

        if np.any(np.all(decision_matrix == 0, axis=0)):
            raise ValueError(
                "An all-zero criterion column requires "
                "an explicit normalization policy."
            )

        try:
            with np.errstate(
                divide="raise",
                invalid="raise",
                over="raise",
            ):
                scores = topsis_method(
                    decision_matrix.copy(),
                    weights.copy(),
                    directions,
                    graph=False,
                    verbose=False,
                )
        except FloatingPointError as error:
            raise ValueError(
                "TOPSIS could not calculate finite scores. "
                "Check for numerically undefined inputs, "
                "including identical effective alternatives."
            ) from error

        scores = np.asarray(scores, dtype=float)

        if (
            not np.all(np.isfinite(scores))
            or np.any(scores < 0)
            or np.any(scores > 1)
        ):
            raise ValueError("Invalid TOPSIS closeness coefficients.")

        return scores


# ============================================================
# ENGINES
# ============================================================


    @classmethod
    def generate_weights(
        cls,
        session: DecisionSession,
        pairwise_matrix: np.ndarray,
        config: WeightingMethodConfig,
        consistency: ConsistencyMethodConfig
    ) -> WeightResult:
        if config is None:
            raise ValueError("A weighting configuration is required.")

        if config.method == WeightingMethod.AHP:
            unknown = set(config.parameters) - {"derivation"}
            if unknown:
                raise ValueError(
                    f"Unsupported AHP parameters: {sorted(unknown)}"
                )

            weights, consistency_ratio = cls.calculate_ahp(
                pairwise_matrix,
                wd=config.parameters.get("derivation", "geometric"),
            )

        else:
            raise NotImplementedError(
                f"Weighting method {config.method!r} is not implemented."
            )

        weights = validate_weight_vector(
            weights,
            expected_count=len(session.criteria),
            label=f"{config.method.value} weighting result",
            allow_zero=config.method != WeightingMethod.AHP,
        )
        
        return WeightResult(
            criterion_weights=[
                CriterionWeight(
                    criterion_id=criterion.id,
                    criterion_name=criterion.name,
                    weight=float(weight),
                )
                for criterion, weight in zip(
                    session.criteria,
                    weights,
                    strict=True,
                )
            ],
            consistency_ratio=(
                None
                if consistency_ratio is None
                else float(consistency_ratio)
            ),
            consistency_threshold=consistency.threshold
        )

    @classmethod
    def generate_ranking(
        cls,
        session: DecisionSession,
        decision_matrix: np.ndarray,
        weighting: WeightResult,
        config: RankingMethodConfig,
    ) -> RankingResult:
        matrix = np.asarray(decision_matrix, dtype=float)

        expected_shape = (
            len(session.alternatives),
            len(session.criteria),
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Decision matrix shape must be {expected_shape}; "
                f"received {matrix.shape}."
            )

        if not session.alternatives:
            raise ValueError("Ranking requires alternatives.")

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Decision matrix values must be finite."
            )

        # Associate weights with criteria before ordering them.
        weight_items = weighting.criterion_weights
        weights_by_id = {
            item.criterion_id: item.weight
            for item in weight_items
        }

        if len(weights_by_id) != len(weight_items):
            raise ValueError("Duplicate criterion weight IDs.")

        criterion_ids = [
            criterion.id
            for criterion in session.criteria
        ]

        if set(weights_by_id) != set(criterion_ids):
            raise ValueError(
                "Weights must cover exactly the session criteria."
            )

        weights = np.asarray([
            weights_by_id[criterion_id]
            for criterion_id in criterion_ids
        ])

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(
                weights.sum(),
                1.0,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise ValueError(
                "Weights must be finite, nonnegative, "
                "and sum to one."
            )

        directions = [
            criterion.direction.value
            for criterion in session.criteria
        ]

        # Dispatch to the configured numerical method.
        if config.method == RankingMethod.TOPSIS:
            scores = cls.calculate_topsis(
                matrix,
                weights,
                directions,
                config.parameters,
            )
        else:
            raise NotImplementedError(
                f"Ranking method {config.method!r} "
                "is not implemented."
            )

        scores = np.asarray(scores, dtype=float)

        if (
            scores.shape != (len(session.alternatives),)
            or not np.all(np.isfinite(scores))
        ):
            raise ValueError(
                "Ranking method returned invalid scores."
            )

        ordered_indices = sorted(
            range(len(session.alternatives)),
            key=lambda index: (-float(scores[index]), index),
        )

        rankings = []
        previous_score = None
        previous_rank = None

        for position, index in enumerate(ordered_indices, start=1):
            score = float(scores[index])
            rank = (
                previous_rank
                if previous_score is not None and score == previous_score
                else position
            )
            rankings.append(
                AlternativeRanking(
                    alternative_id=session.alternatives[index].id,
                    alternative_name=session.alternatives[index].name,
                    rank=rank,
                    closeness_coefficient=score,
                )
            )
            previous_score = score
            previous_rank = rank

        return RankingResult(
            rankings=rankings
        )


    @classmethod
    def aggregate_group_matrices(
        cls,
        *,
        group_ids: tuple[str, ...],
        group_matrices: dict[str, np.ndarray],
        voting_powers: dict[str, float],
        config: AggregationMethodConfig,
    ) -> SessionAggregationResult:
        """Aggregate represented group matrices in canonical order."""

        if config.empty_group_policy != EmptyGroupPolicy.EXCLUDED:
            raise NotImplementedError(
                "Only exclusion of empty groups is implemented."
            )

        if (
            config.group_power_normalization
            != GroupPowerMethod.REPRESENTED_GROUPS
        ):
            raise NotImplementedError(
                "Only represented-group power normalization is implemented."
            )

        if (
            config.between_groups
            != MeanMethod.WEIGHTED_GEOMETRIC_MEAN
        ):
            raise NotImplementedError(
                f"Between-group aggregation {config.between_groups!r} "
                "is not implemented."
            )

        if not group_ids:
            raise ValueError(
                "At least one represented group is required."
            )

        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "Represented group IDs must be unique."
            )

        if set(group_matrices) != set(group_ids):
            raise ValueError(
                "Group matrices must cover represented groups exactly."
            )

        if set(voting_powers) != set(group_ids):
            raise ValueError(
                "Voting powers must cover represented groups exactly."
            )

        matrices = tuple(
            np.asarray(
                group_matrices[group_id],
                dtype=float,
            )
            for group_id in group_ids
        )

        matrix_shape = matrices[0].shape

        if (
            matrices[0].ndim != 2
            or matrix_shape[0] != matrix_shape[1]
            or matrix_shape[0] < 2
        ):
            raise ValueError(
                "Group matrices must be square and at least 2x2."
            )

        for matrix in matrices:
            if (
                matrix.ndim != 2
                or matrix.shape[0] != matrix.shape[1]
                or matrix.shape != matrix_shape
                or matrix.shape[0] < 2
            ):
                raise ValueError(
                    "Group matrices must have the same square shape "
                    "and be at least 2x2."
                )

            if (
                not np.all(np.isfinite(matrix))
                or np.any(matrix <= 0)
            ):
                raise ValueError(
                    "Group matrices must contain only finite, "
                    "strictly positive values."
                )

            if not np.allclose(
                np.diag(matrix),
                1.0,
                rtol=0.0,
                atol=ROUNDOFF_TOLERANCE,
            ):
                raise ValueError(
                    "Group matrix diagonal entries must be 1."
                )

            if not np.allclose(
                np.log(matrix) + np.log(matrix.T),
                0.0,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Group matrices must be reciprocal."
                )

        ordered_powers = [
            voting_powers[group_id]
            for group_id in group_ids
        ]
        if any(
            isinstance(power, (bool, np.bool_))
            for power in ordered_powers
        ):
            raise ValueError(
                "Voting powers must be real numeric values, "
                "not booleans."
            )

        try:
            powers = np.asarray(
                ordered_powers,
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Voting powers must form a numeric vector."
            ) from error

        if (
            powers.shape != (len(group_ids),)
            or not np.all(np.isfinite(powers))
            or np.any(powers < 0)
        ):
            raise ValueError(
                "Voting powers must be finite and nonnegative."
            )

        total_power = float(powers.sum())
        if not np.isfinite(total_power) or total_power <= 0:
            raise ValueError(
                "Total voting power must be finite and positive."
            )

        effective_powers = powers / total_power
        matrix = np.exp(
            np.tensordot(
                effective_powers,
                np.log(np.stack(matrices)),
                axes=1,
            )
        )

        return SessionAggregationResult(
            matrix=matrix,
            effective_voting_powers={
                group_id: float(power)
                for group_id, power in zip(
                    group_ids,
                    effective_powers,
                    strict=True,
                )
            },
        )