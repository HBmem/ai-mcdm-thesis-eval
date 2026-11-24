import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.mcdm.enums import (
    EmptyGroupPolicy,
    GroupPowerMethod,
    MeanMethod,
)
from src.mcdm.models.methods import AggregationMethodConfig
from src.mcdm.service import (
    MCDMService,
    SessionAggregationResult,
)


class GroupMatrixAggregationTests(unittest.TestCase):
    def setUp(self):
        self.config = AggregationMethodConfig(
            within_group=MeanMethod.GEOMETRIC_MEAN,
            between_groups=MeanMethod.WEIGHTED_GEOMETRIC_MEAN,
            group_power_normalization=(
                GroupPowerMethod.REPRESENTED_GROUPS
            ),
            empty_group_policy=EmptyGroupPolicy.EXCLUDED,
        )

    def aggregate(self, **overrides):
        arguments = {
            "group_ids": ("G1", "G2"),
            "group_matrices": {
                "G1": np.array([[1.0, 2.0], [0.5, 1.0]]),
                "G2": np.array([[1.0, 8.0], [0.125, 1.0]]),
            },
            "voting_powers": {"G1": 1.0, "G2": 3.0},
            "config": self.config,
        }
        arguments.update(overrides)
        return MCDMService.aggregate_group_matrices(**arguments)

    def test_aggregate_uses_canonical_group_order_and_normalized_power(self):
        matrices = {
            "G1": np.array([[1.0, 2.0], [0.5, 1.0]]),
            "G2": np.array([[1.0, 8.0], [0.125, 1.0]]),
        }
        originals = {
            group_id: matrix.copy()
            for group_id, matrix in matrices.items()
        }

        result = self.aggregate(
            group_ids=("G2", "G1"),
            group_matrices=matrices,
        )

        self.assertAlmostEqual(result.matrix[0, 1], 2 ** 2.5)
        self.assertAlmostEqual(result.matrix[1, 0], 1 / (2 ** 2.5))
        self.assertEqual(
            tuple(result.effective_voting_powers),
            ("G2", "G1"),
        )
        self.assertEqual(
            result.effective_voting_powers,
            {"G2": 0.75, "G1": 0.25},
        )
        for group_id, matrix in matrices.items():
            np.testing.assert_array_equal(matrix, originals[group_id])

    def test_identifier_coverage_is_exact_and_group_ids_are_unique(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least one represented group",
        ):
            self.aggregate(group_ids=())
        with self.assertRaisesRegex(ValueError, "must be unique"):
            self.aggregate(group_ids=("G1", "G1"))
        with self.assertRaisesRegex(ValueError, "Group matrices must cover"):
            self.aggregate(group_matrices={"G1": np.eye(2)})
        with self.assertRaisesRegex(ValueError, "Voting powers must cover"):
            self.aggregate(voting_powers={"G1": 1.0})

    def test_matrices_must_be_valid_same_sized_reciprocal_matrices(self):
        invalid_matrices = (
            (np.array([1.0, 2.0]), None),
            (np.eye(1), None),
            (
                np.eye(2),
                np.eye(3),
            ),
            (np.array([[1.0, 0.0], [1.0, 1.0]]), None),
            (np.array([[1.0, np.nan], [1.0, 1.0]]), None),
            (np.array([[2.0, 2.0], [0.5, 1.0]]), None),
            (np.array([[1.0, 2.0], [0.6, 1.0]]), None),
        )
        for first, second in invalid_matrices:
            with self.subTest(first=first, second=second):
                matrices = {
                    "G1": first,
                    "G2": (
                        second
                        if second is not None
                        else np.array([[1.0, 2.0], [0.5, 1.0]])
                    ),
                }
                with self.assertRaises(ValueError):
                    self.aggregate(group_matrices=matrices)

    def test_voting_powers_must_be_nonnegative_finite_and_nonboolean(self):
        for powers in (
            {"G1": True, "G2": 1.0},
            {"G1": np.bool_(False), "G2": 1.0},
            {"G1": np.inf, "G2": 1.0},
            {"G1": -1.0, "G2": 2.0},
            {"G1": 0.0, "G2": 0.0},
        ):
            with self.subTest(powers=powers):
                with self.assertRaises(ValueError):
                    self.aggregate(voting_powers=powers)

    def test_session_aggregation_delegates_represented_groups_in_order(self):
        group_g2 = SimpleNamespace(id="G2", voting_power=3.0)
        empty_group = SimpleNamespace(id="EMPTY", voting_power=10.0)
        group_g1 = SimpleNamespace(id="G1", voting_power=1.0)
        participant_g2 = SimpleNamespace(stakeholder_group_id="G2")
        participant_g1 = SimpleNamespace(stakeholder_group_id="G1")
        session = SimpleNamespace(
            stakeholder_groups=(
                group_g2,
                empty_group,
                group_g1,
            ),
            participants=(participant_g2, participant_g1),
        )
        matrix_g2 = np.array([[1.0, 8.0], [0.125, 1.0]])
        matrix_g1 = np.array([[1.0, 2.0], [0.5, 1.0]])
        expected = SessionAggregationResult(
            matrix=np.eye(2),
            effective_voting_powers={},
        )

        with (
            patch.object(
                MCDMService,
                "create_stakeholder_group_matrix",
                side_effect=(matrix_g2, matrix_g1),
            ),
            patch.object(
                MCDMService,
                "aggregate_group_matrices",
                return_value=expected,
            ) as aggregate,
        ):
            result = MCDMService.create_session_aggregation(
                session,
                config=self.config,
            )

        self.assertIs(result, expected)
        arguments = aggregate.call_args.kwargs
        self.assertEqual(
            arguments["group_ids"],
            ("G2", "G1"),
        )
        self.assertEqual(
            tuple(arguments["group_matrices"]),
            ("G2", "G1"),
        )
        np.testing.assert_array_equal(
            arguments["group_matrices"]["G2"],
            matrix_g2,
        )
        np.testing.assert_array_equal(
            arguments["group_matrices"]["G1"],
            matrix_g1,
        )
        self.assertEqual(
            arguments["voting_powers"],
            {"G2": 3.0, "G1": 1.0},
        )
        self.assertIs(arguments["config"], self.config)


if __name__ == "__main__":
    unittest.main()
