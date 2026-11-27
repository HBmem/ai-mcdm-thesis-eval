from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.mcdm.diagnostics import calculate_leave_one_group_out
from src.mcdm.models.computation import (
    AlternativeRanking,
    CriterionWeight,
    RankingResult,
    WeightResult,
)
from src.mcdm.service import MCDMService


class LeaveOneGroupOutDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(
            criteria=(
                SimpleNamespace(id="C1"),
                SimpleNamespace(id="C2"),
            ),
        )
        self.baseline_ranking = self._ranking(
            (("A", 0.8), ("B", 0.5))
        )
        self.baseline_session = SimpleNamespace(
            entity_id="SESSION",
            ranking=self.baseline_ranking,
            metadata=SimpleNamespace(
                applied_criterion_weights={
                    "C1": 0.4,
                    "C2": 0.6,
                },
                ranking_criterion_ids=["C1", "C2"],
                ranking_weight_source="original",
            ),
        )
        self.group_calculations = (
            self._group("G1", 1.0),
            self._group("G2", 3.0),
        )
        self.configs = (
            object(),
            object(),
            object(),
            object(),
        )

    @staticmethod
    def _ranking(scores):
        return RankingResult(
            rankings=[
                AlternativeRanking(
                    alternative_id=alternative_id,
                    alternative_name=alternative_id,
                    rank=index,
                    closeness_coefficient=score,
                )
                for index, (alternative_id, score) in enumerate(
                    scores,
                    start=1,
                )
            ]
        )

    @staticmethod
    def _group(group_id, power):
        return SimpleNamespace(
            entity_id=group_id,
            pairwise_matrix=[[1.0, 2.0], [0.5, 1.0]],
            metadata=SimpleNamespace(voting_power=power),
        )

    def _run(self, *, group_calculations=None, ranking_results=None):
        groups = (
            self.group_calculations
            if group_calculations is None
            else group_calculations
        )
        rankings = ranking_results or (
            (
                self._ranking((("A", 0.7), ("B", 0.6))),
                {"C1": 0.5, "C2": 0.5},
            ),
            (
                self._ranking((("A", 0.4), ("B", 0.9))),
                {"C1": 0.3, "C2": 0.7},
            ),
        )
        aggregations = (
            SimpleNamespace(
                matrix=np.eye(2),
                effective_voting_powers={"G2": 1.0},
            ),
            SimpleNamespace(
                matrix=np.eye(2),
                effective_voting_powers={"G1": 1.0},
            ),
        )
        weighting = WeightResult(
            criterion_weights=[
                CriterionWeight(
                    criterion_id="C1",
                    criterion_name="C1",
                    weight=0.5,
                ),
                CriterionWeight(
                    criterion_id="C2",
                    criterion_name="C2",
                    weight=0.5,
                ),
            ],
            consistency_ratio=0.03,
            consistency_threshold=0.1,
        )

        with (
            patch(
                "src.mcdm.diagnostics._cross_group_diagnostic_inputs",
                return_value=(
                    ("A", "B"),
                    groups,
                    self.baseline_session,
                ),
            ),
            patch(
                "src.mcdm.diagnostics._resolved_recomputation_inputs",
                return_value=(
                    self.session,
                    *self.configs,
                ),
            ),
            patch.object(
                MCDMService,
                "create_decision_matrix",
                return_value=np.ones((2, 2)),
            ),
            patch.object(
                MCDMService,
                "aggregate_group_matrices",
                side_effect=aggregations,
            ) as aggregate,
            patch.object(
                MCDMService,
                "generate_weights",
                return_value=weighting,
            ),
            patch(
                "src.mcdm.diagnostics._recomputed_session_ranking",
                side_effect=rankings,
            ),
        ):
            result = calculate_leave_one_group_out(object())

        return result, aggregate

    def test_completed_omissions_recompute_in_group_and_configured_order(self):
        result, aggregate = self._run()

        self.assertEqual(
            result.evaluated_group_ids,
            ("G1", "G2"),
        )
        self.assertEqual(
            tuple(case.omitted_group_id for case in result.cases),
            ("G1", "G2"),
        )
        self.assertEqual(
            tuple(change.criterion_id
                  for change in result.cases[0].criterion_weight_changes),
            ("C1", "C2"),
        )
        self.assertEqual(
            tuple(change.alternative_id
                  for change in result.cases[0].alternative_changes),
            ("A", "B"),
        )
        self.assertFalse(result.cases[0].winner_changed)
        self.assertTrue(result.cases[1].winner_changed)
        self.assertEqual(
            aggregate.call_args_list[0].kwargs["group_ids"],
            ("G2",),
        )
        self.assertEqual(
            aggregate.call_args_list[1].kwargs["group_ids"],
            ("G1",),
        )

    def test_criterion_weight_changes_preserve_baseline_ranking_order(self):
        self.baseline_session.metadata.ranking_criterion_ids = [
            "C2",
            "C1",
        ]

        result, _ = self._run()

        self.assertEqual(
            tuple(
                change.criterion_id
                for change in result.cases[0].criterion_weight_changes
            ),
            ("C2", "C1"),
        )

    def test_ranking_criteria_must_match_baseline_applied_weights(self):
        self.baseline_session.metadata.ranking_criterion_ids = ["C1"]

        with self.assertRaisesRegex(
            ValueError,
            "Baseline ranking criteria must match baseline applied weights",
        ):
            self._run()

    def test_zero_remaining_voting_power_produces_not_evaluable_cases(self):
        groups = (
            self._group("G1", 0.0),
            self._group("G2", 0.0),
        )

        result, aggregate = self._run(
            group_calculations=groups,
        )

        self.assertTrue(
            all(case.status == "not_evaluable" for case in result.cases)
        )
        self.assertTrue(
            all(case.baseline_winner_ids == ("A",) for case in result.cases)
        )
        aggregate.assert_not_called()

    def test_requires_recorded_configured_voting_powers(self):
        groups = (
            self._group("G1", 1.0),
            self._group("G2", None),
        )
        groups[1].metadata.voting_power = None

        with (
            patch(
                "src.mcdm.diagnostics._cross_group_diagnostic_inputs",
                return_value=(
                    ("A", "B"),
                    groups,
                    self.baseline_session,
                ),
            ),
            patch(
                "src.mcdm.diagnostics._resolved_recomputation_inputs",
                return_value=(self.session, *self.configs),
            ),
            patch.object(
                MCDMService,
                "create_decision_matrix",
                return_value=np.ones((2, 2)),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires recorded configured voting powers",
            ):
                calculate_leave_one_group_out(object())


if __name__ == "__main__":
    unittest.main()
