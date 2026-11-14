"""Execution planning checks against the real schema-3 experiment fixtures."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs
from src.mcdm.enums import MeanMethod, WeightingMethod


ROOT = Path(__file__).resolve().parents[1]


class PrepareRunsTests(unittest.TestCase):
    def setUp(self):
        self.experiments = load_experiments(ROOT / "experiments")
        self.cases = {e.session.id: e for e in self.experiments}

    def test_all_cases_have_unique_execution_keys(self):
        runs = prepare_runs(self.experiments)
        self.assertEqual(len(runs), 4)
        self.assertEqual(len({r.key for r in runs}), 4)
        self.assertEqual(sum(r.experiment.id == "S05" for r in runs), 3)
        self.assertTrue(all(r.requested for r in runs))

    def test_selection_includes_transitive_references(self):
        runs = prepare_runs(self.experiments, "S05/unequal_participation")
        self.assertEqual([r.case_key for r in runs], [
            "S01/control", "S05/control", "S05/unequal_participation",
        ])
        self.assertEqual([r.requested for r in runs], [False, False, True])
        self.assertEqual(runs[-1].reference_runs, {"S05/control": "S05/control#r001"})

    def test_family_aliases_and_multiple_selectors(self):
        for selector in ("S05", "S5", "s005"):
            runs = prepare_runs(self.experiments, selector)
            self.assertEqual({r.case_key for r in runs if r.requested}, {
                "S05/control", "S05/unequal_voting_power", "S05/unequal_participation",
            })
        runs = prepare_runs(self.experiments, ["S01", "S5/control"])
        self.assertEqual({r.case_key for r in runs}, {"S01/control", "S05/control"})

    def test_input_inheritance_does_not_require_source_execution(self):
        e = deepcopy(self.cases["S05/unequal_participation"])
        # The source case is intentionally unavailable. Its inputs are already
        # materialized; remove actual result references to isolate inheritance.
        e.baseline_case = e.session.id
        e.comparisons = []
        e.expected_assertions = []
        runs = prepare_runs([e])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].dependencies, ())
        self.assertEqual(len(runs[0].experiment.session.participants), 50)

    def test_repetitions_have_independent_inputs_and_seeds(self):
        for e in self.experiments:
            e.defaults.repetition = 3
        before = [e.to_dict() for e in self.experiments]
        runs = prepare_runs(self.experiments)
        self.assertEqual(len(runs), 12)
        group = [r for r in runs if r.case_key == "S05/unequal_participation"]
        self.assertEqual([r.repetition for r in group], [1, 2, 3])
        self.assertEqual([r.seed for r in group], [123456, 123457, 123458])
        for i, run in enumerate(group, 1):
            self.assertEqual(run.reference_runs["S05/control"], f"S05/control#r{i:03d}")
            self.assertEqual(run.experiment.execution.random_seed, run.seed)
            self.assertEqual(run.experiment.defaults.seed, run.seed)
        group[0].experiment.session.participants.clear()
        self.assertEqual(len(group[1].experiment.session.participants), 50)
        self.assertEqual([e.to_dict() for e in self.experiments], before)

    def test_single_reference_is_shared_across_repetitions(self):
        self.cases["S05/unequal_participation"].defaults.repetition = 3
        runs = prepare_runs(self.experiments, "S05/unequal_participation")
        group = [r for r in runs if r.requested]
        self.assertEqual(len(group), 3)
        self.assertEqual({r.reference_runs["S05/control"] for r in group}, {"S05/control#r001"})
        self.assertEqual(sum(r.case_key == "S05/control" for r in runs), 1)

    def test_ambiguous_repetition_pairing_is_rejected(self):
        self.cases["S05/control"].defaults.repetition = 2
        self.cases["S05/unequal_participation"].defaults.repetition = 3
        with self.assertRaisesRegex(ValueError, "cannot pair"):
            prepare_runs(self.experiments, "S05/unequal_participation")

    def test_assertion_only_reference_is_included(self):
        e = self.cases["S05/unequal_participation"]
        e.baseline_case = e.session.id
        e.comparisons = []
        runs = prepare_runs(self.experiments, e.session.id)
        self.assertIn("S05/control", {r.case_key for r in runs})

    def test_comparison_cycles_do_not_become_execution_cycles(self):
        self.cases["S01/control"].comparisons = [{
            "id": "reverse_comparison", "reference": "S05/control",
            "metrics": ["winner_changed"],
        }]
        runs = prepare_runs(self.experiments, "S01/control")
        self.assertEqual({r.case_key for r in runs}, {"S01/control", "S05/control"})
        self.assertTrue(all(not r.dependencies for r in runs))

    def test_result_dependencies_precede_their_consumers(self):
        self.cases["S01/control"].case_metadata["result_dependencies"] = ["S05/control"]
        for e in self.experiments:
            e.defaults.repetition = 2
        runs = prepare_runs(reversed(self.experiments), "S01/control")
        positions = {r.key: i for i, r in enumerate(runs)}
        for r in runs:
            for dependency in r.dependencies:
                self.assertLess(positions[dependency], positions[r.key])
        sources = [r for r in runs if r.requested]
        self.assertEqual(sources[1].dependencies, ("S05/control#r002",))

    def test_result_dependency_cycles_are_rejected(self):
        self.cases["S01/control"].case_metadata["result_dependencies"] = ["S05/control"]
        self.cases["S05/control"].case_metadata["result_dependencies"] = ["S01/control"]
        with self.assertRaisesRegex(ValueError, "Result dependency cycle"):
            prepare_runs(self.experiments)

    def test_input_order_does_not_change_plan(self):
        left = prepare_runs(self.experiments)
        right = prepare_runs(reversed(self.experiments))
        self.assertEqual([r.key for r in left], [r.key for r in right])
        self.assertEqual([r.reference_runs for r in left], [r.reference_runs for r in right])

    def test_invalid_selection_empty_input_and_duplicate_ids(self):
        for selection in ("S99", [], ["S01", "missing"], True):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                prepare_runs(self.experiments, selection)
        with self.assertRaisesRegex(ValueError, "No loaded"):
            prepare_runs([])
        with self.assertRaisesRegex(ValueError, "Duplicate case key"):
            prepare_runs(self.experiments + [self.experiments[0]])

    def test_missing_references_and_invalid_ids(self):
        self.cases["S05/control"].comparisons[0]["reference"] = "missing/control"
        with self.assertRaisesRegex(ValueError, "missing referenced case"):
            prepare_runs(self.experiments)
        self.cases["S05/control"].comparisons[0]["reference"] = "S01/control"
        self.cases["S05/control"].variant_id = "wrong/key"
        with self.assertRaisesRegex(ValueError, "same resolved case"):
            prepare_runs(self.experiments)

    def test_invalid_assertion_name_parameters_and_tolerance(self):
        original = deepcopy(self.cases["S01/control"].expected_assertions[0])
        for change in ({"check": "not_registered"}, {"parameters": {"wrong": 1}},
                       {"atol": float("nan")}, {"rtol": -1}, {"expected": float("inf")}):
            self.cases["S01/control"].expected_assertions[0] = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_runs(self.experiments)

    def test_invalid_engine_configuration_and_run_counts(self):
        original = deepcopy(self.cases["S01/control"])
        mutations = [
            lambda e: setattr(e.processing.weighting_method, "method", WeightingMethod.FUZZY_AHP),
            lambda e: e.processing.weighting_method.parameters.update(derivation="unknown"),
            lambda e: e.processing.ranking_method.parameters.update(normalization="unknown"),
            lambda e: setattr(e.processing.aggregation_method, "within_group", MeanMethod.MEAN),
            lambda e: setattr(e.defaults, "repetition", True),
            lambda e: setattr(e.defaults, "repetition", 0),
            lambda e: setattr(e.defaults, "seed", -1),
            lambda e: setattr(e.defaults.ai, "enabled", True),
        ]
        for mutate in mutations:
            e = deepcopy(original)
            mutate(e)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                prepare_runs([e])

    def test_incomplete_or_nonfinite_numerical_inputs(self):
        e = deepcopy(self.cases["S01/control"])
        e.session.participants[0].responses.pop()
        with self.assertRaisesRegex(ValueError, "every question"):
            prepare_runs([e])
        e = deepcopy(self.cases["S01/control"])
        e.session.alternatives[0].criterion_values[0].value = float("inf")
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            prepare_runs([e])

    def test_comparison_dimensions_and_invalid_policy_are_rejected(self):
        e = self.cases["S05/control"]
        e.session.alternatives.pop()
        with self.assertRaisesRegex(ValueError, "requires identical alternatives"):
            prepare_runs(self.experiments)
        self.experiments = load_experiments(ROOT / "experiments")
        e = next(e for e in self.experiments if e.session.id == "S05/control")
        e.comparisons[0]["alternative_alignment"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported comparison"):
            prepare_runs(self.experiments)

    def test_assertion_entity_ids_and_reference_stage_are_validated(self):
        e = self.cases["S05/unequal_participation"]
        e.expected_assertions.append({
            "id": "invalid_counts", "check": "participant_counts_equal",
            "parameters": {"counts": {"missing_group": 5}}, "expected": True,
        })
        with self.assertRaisesRegex(ValueError, "expected entity IDs"):
            prepare_runs(self.experiments)
        e.expected_assertions.pop()
        next(a for a in e.expected_assertions if a["check"] == "weights_equal")["stage"] = "calculation"
        with self.assertRaisesRegex(ValueError, "stage=comparison"):
            prepare_runs(self.experiments)

    def test_planning_does_not_calculate_or_regenerate_participants(self):
        with patch("src.mcdm.reference.evaluate", side_effect=AssertionError("calculation")), \
             patch("src.experiments.participants.generate_participants", side_effect=AssertionError("generation")):
            runs = prepare_runs(self.experiments)
        self.assertEqual(len(runs), 4)


if __name__ == "__main__":
    unittest.main()
