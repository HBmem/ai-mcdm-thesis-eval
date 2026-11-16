"""Numerical execution checks against installed PyDecision and schema-3 cases."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_experiments
from src.mcdm import execution
from src.mcdm.enums import WeightingMethod


ROOT = Path(__file__).resolve().parents[1]


class NumericalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.experiments = load_experiments(ROOT / "experiments")

    def test_all_entities_are_weighted_and_ranked_by_pydecision(self):
        runs = prepare_runs(self.experiments)
        with patch.object(execution, "ahp_method", wraps=execution.ahp_method) as ahp, \
             patch.object(execution, "topsis_method", wraps=execution.topsis_method) as topsis:
            path, results = execute_experiments(runs)
        self.assertIsNone(path)
        self.assertEqual([r["status"] for r in results], ["calculated"] * 4)
        # Three 5-person cases and one 50-person case, each with 5 groups + session.
        self.assertEqual(ahp.call_count, 89)
        self.assertEqual(topsis.call_count, 89)
        self.assertTrue(all(call.kwargs["graph"] is False and call.kwargs["verbose"] is False
                            for call in topsis.call_args_list))
        for result in results:
            trace = result["trace"]
            self.assertEqual(len(trace["group_rankings"]), 5)
            self.assertEqual(trace["engine"]["name"], "pyDecision")
            json.dumps(result, allow_nan=False)
            self.assertEqual(result["execution"]["steps"]["assertions"], "not_run")

    def test_derivation_reaches_every_level(self):
        experiment = self.experiments[0]
        for method in ("mean", "geometric", "max_eigen"):
            experiment.processing.weighting_method.parameters["derivation"] = method
            trace = execution.evaluate_experiment(experiment)
            for entry in [*trace["preferences"]["participants"],
                          *trace["preferences"]["groups"], trace["preferences"]["session"]]:
                self.assertEqual(entry["wd"], method)
                np.testing.assert_allclose(entry["weights"], [0.25] * 4, atol=1e-12)

    def test_each_ranking_uses_its_own_weights_and_library_scores(self):
        experiment = next(e for e in self.experiments if e.session.id == "S05/control")
        trace = execution.evaluate_experiment(experiment)
        for person in trace["preferences"]["participants"]:
            ranking = trace["participant_rankings"][person["id"]]
            np.testing.assert_allclose(list(ranking["applied_weights"].values()), person["weights"])
            scores = execution.topsis_method(
                np.array(ranking["decision_matrix"]["values"]), person["weights"],
                [c.direction.value for c in experiment.session.criteria], graph=False, verbose=False,
            )
            by_id = {r["alternative_id"]: r["score"] for r in ranking["rankings"]}
            np.testing.assert_allclose([by_id[a.id] for a in experiment.session.alternatives], scores)
            group = trace["group_rankings"][person["group_id"]]
            self.assertEqual([r["alternative_id"] for r in ranking["rankings"]],
                             [r["alternative_id"] for r in group["rankings"]])
            np.testing.assert_allclose([r["score"] for r in ranking["rankings"]],
                                       [r["score"] for r in group["rankings"]], atol=1e-12, rtol=0)
        self.assertFalse(np.allclose(trace["preferences"]["participants"][0]["weights"],
                                     trace["preferences"]["session"]["weights"]))

    def test_replication_invariance_and_voting_power_effect(self):
        _, results = execute_experiments(prepare_runs(self.experiments))
        traces = {r["case_key"]: r["trace"] for r in results}
        control = traces["S05/control"]
        expanded = traces["S05/unequal_participation"]
        np.testing.assert_allclose(list(control["applied_weights"].values()),
                                   list(expanded["applied_weights"].values()), atol=1e-12, rtol=0)
        for group in expanded["preferences"]["groups"]:
            self.assertEqual(group["effective_voting_power"], 0.2)
        self.assertEqual(len(expanded["participant_rankings"]), 50)
        self.assertFalse(np.allclose(list(control["applied_weights"].values()),
                                     list(traces["S05/unequal_voting_power"]["applied_weights"].values())))

    def test_id_alignment_with_permuted_value_order(self):
        experiment = self.experiments[0]
        baseline = execution.evaluate_experiment(experiment)
        for alternative in experiment.session.alternatives:
            alternative.criterion_values.reverse()
        for p in experiment.session.participants:
            p.responses.reverse()
        self.assertEqual(baseline, execution.evaluate_experiment(experiment))

    def test_two_criteria_keeps_pydecision_weights_and_zero_cr(self):
        weights = execution.calculate_ahp([[1, 3], [1 / 3, 1]], {"derivation": "mean"}, 0.1)
        np.testing.assert_allclose(weights["weights"], [0.75, 0.25])
        self.assertEqual(weights["consistency_ratio"], 0)
        self.assertEqual(weights["two_criterion_cr_convention"], "zero")

    def test_analytical_weights_and_directional_ranking(self):
        target = np.array([1, 3, 9]) / 13
        matrix = target[:, None] / target[None, :]
        for method in ("mean", "geometric", "max_eigen"):
            result = execution.calculate_ahp(matrix, {"derivation": method}, 0.1)
            np.testing.assert_allclose(result["weights"], target, atol=1e-12, rtol=0)
        result = execution.calculate_topsis([[1, 3], [2, 2], [3, 1]],
                                           [0.5, 0.5], ["max", "min"], {})
        np.testing.assert_allclose(result["scores"], [0, 0.5, 1], atol=1e-12, rtol=0)

    def test_new_registered_adapter_reaches_planner_and_executor(self):
        experiment = deepcopy(self.experiments[0])
        experiment.processing.weighting_method.method = WeightingMethod.FUZZY_AHP
        # Test-only stand-in demonstrates dispatch; it does not implement fuzzy AHP.
        with patch.dict(execution.WEIGHTING_METHODS, {
            WeightingMethod.FUZZY_AHP: execution.MethodAdapter(
                execution._ahp_parameters, execution.calculate_ahp,
            ),
        }):
            runs = prepare_runs([experiment])
            _, results = execute_experiments(runs)
        self.assertEqual(results[0]["status"], "calculated")

    def test_inconsistent_inputs_are_retained_and_flagged(self):
        result = execution.calculate_ahp(
            [[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]], {"derivation": "mean"}, 0.1,
        )
        self.assertTrue(result["flagged"])
        self.assertGreater(result["consistency_ratio"], 0.1)

    def test_undefined_topsis_inputs_fail_explicitly(self):
        for matrix in ([[0, 1], [0, 2]], [[1, 2], [1, 2]], [[1, 2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                execution.calculate_topsis(matrix, [0.5, 0.5], ["max", "min"], {})

    def test_invalid_library_output_is_rejected(self):
        with patch.object(execution, "ahp_method", return_value=([float("nan")] * 3, 0)):
            with self.assertRaisesRegex(ValueError, "invalid AHP weights"):
                execution.calculate_ahp(np.ones((3, 3)), {}, 0.1)
        with patch.object(execution, "topsis_method", return_value=np.array([float("nan"), 1])):
            with self.assertRaisesRegex(ValueError, "invalid TOPSIS scores"):
                execution.calculate_topsis([[1, 2], [2, 1]], [0.5, 0.5], ["max", "max"], {})

    def test_failure_does_not_block_a_comparison_only_reference(self):
        runs = prepare_runs(self.experiments)
        for a in runs[0].experiment.session.alternatives:
            a.criterion_values[0].value = 0
        _, results = execute_experiments(runs)
        self.assertEqual(results[0]["status"], "failed")
        self.assertTrue(all(r["status"] == "calculated" for r in results[1:]))
        self.assertIn("all-zero", results[0]["execution"]["messages"][0])

    def test_failed_hard_dependency_blocks_consumer(self):
        runs = prepare_runs(self.experiments)
        for a in runs[0].experiment.session.alternatives:
            a.criterion_values[0].value = 0
        runs[1] = replace(runs[1], dependencies=(runs[0].key,))
        _, results = execute_experiments(runs)
        self.assertEqual(results[1]["status"], "blocked")
        self.assertEqual(results[1]["trace"], {})

    def test_execution_is_immutable_and_assigns_fresh_attempt_ids(self):
        runs = prepare_runs(self.experiments, "S01")
        before = deepcopy(runs[0].experiment.to_dict())
        _, first = execute_experiments(runs)
        _, second = execute_experiments(runs)
        self.assertEqual(first[0]["trace"], second[0]["trace"])
        self.assertNotEqual(first[0]["run_id"], second[0]["run_id"])
        self.assertEqual(before, runs[0].experiment.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            execute_experiments(runs, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_ties_preserve_full_winner_set(self):
        # Two equally performing best options and a distinct worse option.
        experiment = self.experiments[0]
        best = experiment.session.alternatives[0]
        directions = {c.id: c.direction.value for c in experiment.session.criteria}
        for value in best.criterion_values:
            value.value = 100 if directions[value.criterion_id] == "max" else 1
        experiment.session.alternatives[1].criterion_values = deepcopy(best.criterion_values)
        trace = execution.evaluate_experiment(experiment)
        by_id = {r["alternative_id"]: r for r in trace["topsis"]["rankings"]}
        self.assertEqual(by_id["A1"]["score"], by_id["A2"]["score"])
        self.assertEqual(by_id["A1"]["statistical_rank"], by_id["A2"]["statistical_rank"])
        self.assertEqual(set(trace["topsis"]["winner_ids"]), {"A1", "A2"})

    def test_empty_groups_are_recorded_and_excluded_from_power_normalization(self):
        experiment = self.experiments[0]
        empty = experiment.session.stakeholder_groups[0].model_copy(deep=True)
        empty.id = "unrepresented_group"
        empty.voting_power = 0.9
        experiment.session.stakeholder_groups.append(empty)
        trace = execution.evaluate_experiment(experiment)
        self.assertEqual(trace["preferences"]["excluded_group_ids"], [empty.id])
        self.assertNotIn(empty.id, trace["group_rankings"])
        self.assertEqual([g["effective_voting_power"] for g in trace["preferences"]["groups"]], [0.2] * 5)

    def test_ai_requests_fail_instead_of_silently_skipping(self):
        runs = prepare_runs(self.experiments, "S01")
        with self.assertRaises(ValueError):
            execute_experiments(runs, ai_config=object())
        with self.assertRaises(NotImplementedError):
            execute_experiments(runs, ai_config=object(), ai_client=object())


if __name__ == "__main__":
    unittest.main()
