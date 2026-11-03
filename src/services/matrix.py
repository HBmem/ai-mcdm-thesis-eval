from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyDecision.algorithm import ahp_method

from src.enum import CriterionDirection, RankingMethod, WeightingMethod
from src.models.domain import Participant
from src.models.methods import RankingMethodConfig, WeightingMethodConfig
from src.models.session import DecisionSession


@dataclass(frozen=True)
class AlternativeRanking:
    alternative_id: str
    score: float
    rank: int


class MCDMService:
    """
    Performs deterministic weight generation and ranking calculations for a DecisionSession.
    """

    @staticmethod
    def create_participant_matrix(
        session: DecisionSession,
        participant: Participant,
    ) -> np.ndarray:
        """
        Construct one participant's reciprocal pairwise comparison matrix.
        """

        criteria_index = {
            criterion.id: index
            for index, criterion in enumerate(session.criteria)
        }

        questions = {
            question.id: question
            for question in session.questions
        }

        n = len(session.criteria)

        matrix = np.ones((n, n), dtype=float)

        for response in participant.responses:
            question = questions[response.question_id]

            left_index = criteria_index[question.left_criterion_id]
            right_index = criteria_index[question.right_criterion_id]

            scale = (response.preference.ahp_scale
                     if hasattr(response.preference, "ahp_scale") else float(response.preference))

            matrix[left_index, right_index] = scale
            matrix[right_index, left_index] = 1.0 / scale

        return matrix

    @classmethod
    def create_stakeholder_group_matrix(
        cls,
        session: DecisionSession,
        participants: list[Participant],
    ) -> np.ndarray:
        """
        Aggregate participant matrices within one stakeholder group using geometric mean.
        """
        if not participants:
            raise ValueError(
                "At least one participant is required for group aggregation."
            )

        matrices = np.asarray(
            [
                cls.create_participant_matrix(session, participant)
                for participant in participants
            ],
            dtype=float,
        )

        return np.exp(
            np.mean(
                np.log(matrices),
                axis=0,
            )
        )

    @classmethod
    def create_session_matrix(
        cls,
        session: DecisionSession,
    ) -> np.ndarray:
        """
        Perform two-stage stakeholder aggregation.

        Stage 1:
            Aggregate participant within each stakeholder group.

        Stage 2:
            Aggregate stakeholder-group matrices according to normalized stakeholder voting power.
        """
        group_matrices: list[np.ndarray] = []
        group_weights: list[float] = []

        for stakeholder_group in session.stakeholder_groups:
            participants = [
                participant
                for participant in session.participants
                if participant.stakeholder_group_id
                == stakeholder_group.id
            ]

            if not participants:
                continue

            group_matrix = cls.create_stakeholder_group_matrix(
                session,
                participants,
            )

            group_matrices.append(group_matrix)
            group_weights.append(stakeholder_group.voting_power)

        if not group_matrices:
            raise ValueError(
                "No active stakeholder groups contain participants."
            )

        weights = np.asarray(group_weights, dtype=float)

        # Normalize only among groups represented in the session.
        weights /= weights.sum()

        matrices = np.asarray(group_matrices, dtype=float)

        session_matrix = np.exp(
            np.sum(
                weights[:, None, None]
                * np.log(matrices),
                axis=0,
            )
        )

        return session_matrix

    @staticmethod
    def create_decision_matrix(
        session: DecisionSession,
    ) -> np.ndarray:
        """
        Convert alternative criterion values into a TOPSIS decision matrix.

        Rows represent alternatives.
        Columns represent criteria in session.criteria order.
        """

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

    @classmethod
    def calculate_ahp(
        cls,
        session: DecisionSession,
        *,
        wd: Literal["mean", "m", "geometric", "g", "max_eigen", "me"] = "geometric",
    ) -> tuple[np.ndarray, float]:
        """
        Calculate aggregated criterion weights and consistency ratio.
        """
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

        session_matrix = cls.create_session_matrix(session)
        criterion_count = session_matrix.shape[0]

        if criterion_count == 2:
            left_to_right = session_matrix[0, 1]
            weights = np.asarray(
                [
                    left_to_right / (1.0 + left_to_right),
                    1.0 / (1.0 + left_to_right),
                ]
            )
            return weights, 0.0
        if criterion_count > 15:
            raise ValueError(
                "AHP consistency ratios are supported for at most 15 criteria."
            )

        weights, consistency_ratio = ahp_method(
            session_matrix,
            wd=wd,
        )

        return np.asarray(weights), float(consistency_ratio)

    @classmethod
    def generate_weights(
        cls,
        session: DecisionSession,
        config: WeightingMethodConfig,
    ) -> tuple[np.ndarray, float]:
        """Generate criterion weights using the configured weighting method."""
        if config.method is WeightingMethod.AHP:
            unknown = set(config.parameters) - {"wd"}
            if unknown:
                raise ValueError(
                    f"Unsupported AHP configuration options: {sorted(unknown)}."
                )
            return cls.calculate_ahp(
                session,
                wd=config.parameters.get("wd", "geometric"),
            )

        raise NotImplementedError(
            f"Weighting method {config.method.value!r} is not implemented."
        )

    @classmethod
    def generate_ranking(
        cls,
        session: DecisionSession,
        weights: np.ndarray,
        config: RankingMethodConfig,
    ) -> list[AlternativeRanking]:
        """Rank session alternatives using the configured ranking method."""
        if config.method is not RankingMethod.TOPSIS:
            raise NotImplementedError(
                f"Ranking method {config.method.value!r} is not implemented."
            )

        if config.parameters:
            raise ValueError(
                "Unsupported TOPSIS configuration options: "
                f"{sorted(config.parameters)}."
            )

        if not session.alternatives:
            raise ValueError("At least one alternative is required for ranking.")

        decision_matrix = cls.create_decision_matrix(session)
        criterion_weights = np.asarray(weights, dtype=float)
        criterion_count = len(session.criteria)
        if criterion_weights.shape != (criterion_count,):
            raise ValueError(
                "The weight vector must contain exactly one value for each "
                "session criterion."
            )
        if not np.all(np.isfinite(criterion_weights)):
            raise ValueError("Criterion weights must be finite.")
        if np.any(criterion_weights < 0) or criterion_weights.sum() <= 0:
            raise ValueError(
                "Criterion weights must be non-negative with a positive sum."
            )
        if not np.all(np.isfinite(decision_matrix)):
            raise ValueError("Alternative criterion values must be finite.")

        criterion_weights = criterion_weights / criterion_weights.sum()
        column_norms = np.linalg.norm(decision_matrix, axis=0)
        column_norms[column_norms == 0] = 1.0
        weighted_matrix = (decision_matrix / column_norms) * criterion_weights

        benefit_criteria = np.asarray(
            [
                criterion.direction is CriterionDirection.MAXIMIZE
                for criterion in session.criteria
            ],
            dtype=bool,
        )
        ideal_best = np.where(
            benefit_criteria,
            np.max(weighted_matrix, axis=0),
            np.min(weighted_matrix, axis=0),
        )
        ideal_worst = np.where(
            benefit_criteria,
            np.min(weighted_matrix, axis=0),
            np.max(weighted_matrix, axis=0),
        )
        distance_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
        distance_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
        denominator = distance_best + distance_worst
        scores = np.divide(
            distance_worst,
            denominator,
            out=np.full_like(distance_worst, 0.5),
            where=denominator != 0,
        )

        ordered_indices = sorted(
            range(len(session.alternatives)),
            key=lambda index: (-scores[index], index),
        )
        return [
            AlternativeRanking(
                alternative_id=session.alternatives[index].id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ordered_indices, start=1)
        ]