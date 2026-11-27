from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.mcdm.diagnostics import (
    calculate_bounded_weight_perturbations,
)
from src.mcdm.models.computation import (
    AlternativeRanking,
    RankingResult,
)
from src.mcdm.service import MCDMService
from src.mcdm.service import MCDMService


class BoundedWeightPerturbationDiagnosticsTests(unittest.TestCase):
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

    def _run(
        self,
        *,
        weights=None,
        delta=0.1,
        matrix=None,
        rank_side_effect=None,
    ):
        baseline_weights = (
            {"C1": 0.4, "C2": 0.6}
            if weights is None
            else weights
        )
        baseline = SimpleNamespace(
            entity_id="SESSION",
            ranking=self._ranking((("A", 0.8), ("B", 0.5))),
            metadata=SimpleNamespace(
                applied_criterion_weights=baseline_weights,
                ranking_criterion_ids=("C1", "C2"),
            ),
        )
        session = SimpleNamespace(
            criteria=(
                SimpleNamespace(id="C1"),
                SimpleNamespace(id="C2"),
            ),
        )
        decision_matrix = (
            np.asarray([[1.0, 0.1], [1.0, 0.9]])
            if matrix is None
            else matrix
        )
        rank_result = self._ranking((("B", 0.9), ("A", 0.4)))

        with (
            patch(
                "src.mcdm.diagnostics._diagnostic_inputs",
                return_value=(("A", "B"), (), baseline),
            ) as diagnostic_inputs,
            patch(
                "src.mcdm.diagnostics._resolved_recomputation_inputs",
                return_value=(
                    session,
                    object(),
                    object(),
                    object(),
                    object(),
                ),
            ) as resolved_inputs,
            patch.object(
                MCDMService,
                "create_decision_matrix",
                return_value=decision_matrix,
            ) as create_matrix,
            patch(
                "src.mcdm.diagnostics._rank_with_perturbed_weights",
                return_value=rank_result,
                side_effect=rank_side_effect,
            ) as rank_perturbed,
        ):
            result = calculate_bounded_weight_perturbations(
                object(),
                absolute_weight_delta=delta,
            )

        diagnostic_inputs.assert_called_once()
        resolved_inputs.assert_called_once()
        create_matrix.assert_called_once_with(session)
        return result, rank_perturbed, decision_matrix

    def test_completed_cases_use_baseline_order_and_perturbed_scores(self):
        result, rank_perturbed, decision_matrix = self._run()

        self.assertEqual(
            tuple(
                (case.criterion_id, case.requested_delta)
                for case in result.cases
            ),
            (
                ("C1", -0.1),
                ("C1", 0.1),
                ("C2", -0.1),
                ("C2", 0.1),
            ),
        )
        self.assertEqual(
            tuple(entry.alternative_id for entry in result.baseline_entries),
            ("A", "B"),
        )
        for case in result.cases:
            self.assertEqual(case.status, "completed")
            self.assertEqual(
                tuple(entry.alternative_id for entry in case.entries),
                ("A", "B"),
            )
            self.assertEqual(
                tuple(entry.statistical_rank for entry in case.entries),
                (2.0, 1.0),
            )
            self.assertEqual(case.winner_ids, ("B",))
            self.assertTrue(case.winner_changed)

        self.assertEqual(rank_perturbed.call_count, 4)
        self.assertTrue(all(
            call.kwargs["decision_matrix"] is decision_matrix
            for call in rank_perturbed.call_args_list
        ))

    def test_infeasible_perturbations_are_unavailable_with_requested_delta(
        self,
    ):
        result, rank_perturbed, _ = self._run(
            weights={"C1": 1.0, "C2": 0.0},
        )

        self.assertEqual(
            tuple(case.status for case in result.cases),
            (
                "not_evaluable",
                "not_evaluable",
                "not_evaluable",
                "completed",
            ),
        )
        for case in result.cases[:3]:
            self.assertIsNotNone(case.reason)
            self.assertEqual(case.entries, ())
            self.assertIsNone(case.applied_weights)
        self.assertEqual(
            result.cases[-1].applied_weights,
            {"C1": 0.9, "C2": 0.1},
        )
        rank_perturbed.assert_called_once()

    def test_identical_effective_alternatives_are_unavailable(self):
        def rank_side_effect(**kwargs):
            weights = kwargs["perturbed_weights"]
            scores = MCDMService.calculate_topsis(
                kwargs["decision_matrix"],
                np.asarray((weights["C1"], weights["C2"])),
                ["max", "max"],
                {},
            )
            return self._ranking((
                ("A", float(scores[0])),
                ("B", float(scores[1])),
            ))

        result, _, _ = self._run(
            weights={"C1": 0.5, "C2": 0.5},
            delta=0.5,
            matrix=np.asarray([[1.0, 0.1], [1.0, 0.9]]),
            rank_side_effect=rank_side_effect,
        )

        self.assertEqual(
            tuple(case.status for case in result.cases),
            (
                "completed",
                "not_evaluable",
                "not_evaluable",
                "completed",
            ),
        )
        self.assertTrue(all(
            "identical values" in case.reason
            for case in (result.cases[1], result.cases[2])
        ))

    def test_unrelated_ranking_errors_surface(self):
        with self.assertRaisesRegex(ValueError, "unrelated failure"):
            self._run(
                rank_side_effect=ValueError("unrelated failure"),
            )

    def test_absolute_weight_delta_must_be_finite_and_in_unit_interval(self):
        for delta in (0.0, -0.1, 1.0, float("nan"), True):
            with self.subTest(delta=delta):
                with self.assertRaisesRegex(
                    ValueError,
                    "absolute_weight_delta must be finite and in",
                ):
                    self._run(delta=delta)


if __name__ == "__main__":
    unittest.main()
