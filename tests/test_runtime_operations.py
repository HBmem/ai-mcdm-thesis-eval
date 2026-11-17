"""Runtime interventions preserve preference evidence and apply ranking changes."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from src.experiments.loader import load_experiments
from src.experiments.runner import (
    execute_experiments, prepare_runs, resolve_execution_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeOperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runs = prepare_runs(load_experiments(ROOT / "experiments"))
        cls.runs = {r.case_key: r for r in runs}
        _, results = execute_experiments(runs, output=None)
        cls.results = {r.case_key: r for r in results}

    def dependencies(self, run):
        return {key: self.results[key] for key in run.experiment.case_metadata["result_dependencies"]}

    def weights(self, entity):
        return {w.criterion_id: w.weight for w in entity.weighting.criterion_weights}

    def scores(self, entity):
        return {r.alternative_id: r.closeness_coefficient for r in entity.ranking.rankings}

    def test_all_cases_calculate_and_keep_scope_weights_normalized(self):
        self.assertEqual(len(self.results), 40)
        for key, result in self.results.items():
            with self.subTest(key=key):
                self.assertEqual(result.status, "calculated", result.execution["messages"])
                for entity in [*result.mcdm.participants, *result.mcdm.groups, result.mcdm.session]:
                    self.assertAlmostEqual(sum(entity.metadata["applied_criterion_weights"].values()), 1)

    def test_criterion_removal_keeps_full_evidence_and_restricts_all_scopes(self):
        baseline = self.results["S01/control"]
        for cid in ("C1", "C2", "C3", "C4"):
            result = self.results["S07/remove_" + cid.lower()]
            self.assertEqual(len(result.inputs["session"]["criteria"]), 4)
            self.assertEqual(len(result.inputs["session"]["questions"]), 6)
            sources = [*baseline.mcdm.participants, *baseline.mcdm.groups, baseline.mcdm.session]
            targets = [*result.mcdm.participants, *result.mcdm.groups, result.mcdm.session]
            for source, target in zip(sources, targets):
                np.testing.assert_allclose(target.pairwise_matrix, source.pairwise_matrix)
                self.assertEqual(self.weights(target), self.weights(source))
                self.assertEqual(target.weighting.consistency_ratio, source.weighting.consistency_ratio)
                applied = target.metadata["applied_criterion_weights"]
                self.assertNotIn(cid, applied)
                self.assertEqual(len(applied), 3)
                np.testing.assert_allclose(list(applied.values()), [1 / 3] * 3)

    def test_explicit_weights_change_only_session_ranking(self):
        baseline = self.results["S01/control"]
        for key, result in self.results.items():
            if not key.startswith("S08/"):
                continue
            operation = self.runs[key].experiment.case_metadata["runtime_operations"][0]
            applied = result.mcdm.session.metadata["applied_criterion_weights"]
            for cid, expected in operation["weights"].items():
                self.assertAlmostEqual(applied[cid], expected)
            self.assertEqual(self.weights(result.mcdm.session), self.weights(baseline.mcdm.session))
            for source, target in zip(baseline.mcdm.participants, result.mcdm.participants):
                self.assertEqual(self.weights(source), self.weights(target))
                np.testing.assert_allclose(list(self.scores(source).values()), list(self.scores(target).values()))
            # Independently compute session TOPSIS from the applied weight map.
            session = self.runs[key].experiment.session
            x = np.array([[next(v.value for v in a.criterion_values if v.criterion_id == c.id)
                           for c in session.criteria] for a in session.alternatives])
            v = x / np.linalg.norm(x, axis=0) * [applied[c.id] for c in session.criteria]
            benefit = np.array([c.direction == "max" for c in session.criteria])
            best = np.where(benefit, v.max(axis=0), v.min(axis=0))
            worst = np.where(benefit, v.min(axis=0), v.max(axis=0))
            db, dw = np.linalg.norm(v - best, axis=1), np.linalg.norm(v - worst, axis=1)
            actual = self.scores(result.mcdm.session)
            np.testing.assert_allclose([actual[a.id] for a in session.alternatives], dw / (db + dw))

    def test_addition_and_removal_restore_all_scope_scores(self):
        added = self.results["S09/add_dominated"]
        restored = self.results["S09/remove_dominated"]
        baseline = self.results["S01/control"]
        self.assertEqual(len(added.inputs["session"]["alternatives"]), 6)
        self.assertEqual(restored.inputs["session"]["alternatives"], baseline.inputs["session"]["alternatives"])
        for source, target in zip(
            [*baseline.mcdm.participants, *baseline.mcdm.groups, baseline.mcdm.session],
            [*restored.mcdm.participants, *restored.mcdm.groups, restored.mcdm.session],
        ):
            self.assertEqual(self.weights(source), self.weights(target))
            self.assertEqual(self.scores(source), self.scores(target))
        metadata = restored.execution["input_resolution"]["runtime_operations"]
        self.assertEqual([op["type"] for op in metadata], ["inherit_alternatives", "remove_alternatives"])
        self.assertEqual(metadata[0]["source_run_id"], added.run_id)

    def test_resolution_does_not_mutate_cases_or_dependency_results(self):
        for key in ("S06/remove_winner", "S07/remove_c1", "S08/c1_w085", "S09/remove_dominated"):
            run = self.runs[key]
            before = run.experiment.to_dict()
            dependencies = self.dependencies(run)
            source_snapshots = {k: deepcopy(v.inputs) for k, v in dependencies.items()}
            resolved = resolve_execution_inputs(run, dependency_results=dependencies)
            resolved.experiment.session.alternatives[0].name = "changed"
            for reuse in resolved.reused_weightings.values():
                reuse.weighting.criterion_weights[0].weight = 0
            self.assertEqual(run.experiment.to_dict(), before)
            for source, original in source_snapshots.items():
                self.assertEqual(dependencies[source].inputs, original)
            self.assertEqual(self.weights(self.results["S01/control"].mcdm.session)["C1"], 0.25)

    def test_invalid_operations_fail_during_resolution(self):
        run = deepcopy(self.runs["S08/c1_w085"])
        criteria = [c.id for c in run.experiment.session.criteria]
        invalid = [
            {"type": "remove_criteria", "criterion_ids": ["unknown"], "target": "ranking", "preserve_preference_evidence": True},
            {"type": "remove_criteria", "criterion_ids": criteria, "target": "ranking", "preserve_preference_evidence": True},
            {"type": "remove_criteria", "criterion_ids": ["C1"], "target": "ranking", "preserve_preference_evidence": False},
            {"type": "remove_alternatives", "alternative_ids": ["unknown"]},
            {"type": "remove_alternatives", "alternative_ids": [a.id for a in run.experiment.session.alternatives]},
            {"type": "remove_alternatives", "alternative_ids": ["A1", "A1"]},
            {"type": "inherit_alternatives", "reference": "S01/control"},
            {"type": "set_criterion_weights", "scopes": ["invalid"], "weights": dict.fromkeys(criteria, 0.25)},
            {"type": "set_criterion_weights", "scopes": ["session", "session"], "weights": dict.fromkeys(criteria, 0.25)},
        ]
        for value in (True, -1, float("nan"), float("inf"), 0.5):
            invalid.append({"type": "set_criterion_weights", "scopes": ["session"], "weights": {**dict.fromkeys(criteria, 0.25), "C1": value}})
        for op in invalid:
            with self.subTest(operation=op), self.assertRaises(ValueError):
                run.experiment.case_metadata["runtime_operations"] = [op]
                resolve_execution_inputs(run)

    def test_bad_added_alternatives_are_rejected(self):
        run = deepcopy(self.runs["S09/add_dominated"])
        valid = deepcopy(run.experiment.case_metadata["runtime_operations"][0]["alternatives"][0])
        invalid = [{**valid, "id": "A1"}, {**valid, "values": {"C1": 2}},
                   {**valid, "values": {**valid["values"], "C1": True}},
                   {**valid, "values": {**valid["values"], "C1": float("inf")}}]
        for alternative in invalid:
            with self.subTest(alternative=alternative), self.assertRaises(ValueError):
                run.experiment.case_metadata["runtime_operations"] = [{"type": "add_alternatives", "alternatives": [alternative]}]
                resolve_execution_inputs(run, dependency_results=self.dependencies(run))

    def test_conflicting_reuse_and_explicit_weights_are_rejected(self):
        run = deepcopy(self.runs["S06/remove_winner"])
        run.experiment.case_metadata["runtime_operations"] = [{
            "type": "set_criterion_weights", "scopes": ["session"],
            "weights": {f"C{i}": 0.25 for i in range(1, 5)},
        }]
        with self.assertRaisesRegex(ValueError, "both reuse and explicitly override"):
            resolve_execution_inputs(run, dependency_results=self.dependencies(run))

    def test_operation_order_and_all_tied_winner_removal(self):
        run = deepcopy(self.runs["S09/remove_dominated"])
        run.experiment.case_metadata["runtime_operations"].reverse()
        with self.assertRaisesRegex(ValueError, "unique known IDs"):
            resolve_execution_inputs(run, dependency_results=self.dependencies(run))
        winner_run = deepcopy(self.runs["S06/remove_winner"])
        source = deepcopy(self.results["S01/control"])
        rankings = source.mcdm.session.ranking.rankings
        rankings[1].closeness_coefficient = rankings[0].closeness_coefficient
        resolved = resolve_execution_inputs(winner_run, dependency_results={source.case_key: source})
        remaining = {a.id for a in resolved.experiment.session.alternatives}
        self.assertNotIn(rankings[0].alternative_id, remaining)
        self.assertNotIn(rankings[1].alternative_id, remaining)
        for ranking in rankings:
            ranking.closeness_coefficient = 0.5
        with self.assertRaisesRegex(ValueError, "no alternatives"):
            resolve_execution_inputs(winner_run, dependency_results={source.case_key: source})


if __name__ == "__main__":
    unittest.main()
