"""Comparison policies and reference-only planning for schema-3 cases."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.experiments.loader import load_experiments
from src.experiments.runner import _references, prepare_runs


ROOT = Path(__file__).resolve().parents[1]


class ComparisonReferenceTests(unittest.TestCase):
    def comparison(self, **overrides):
        return {
            "id": "against_baseline", "reference": "S01/control",
            "metrics": ["weight_delta", "spearman_rho"],
            **overrides,
        }

    def test_reference_deduplication_preserves_declarations(self):
        comparisons = [self.comparison(
            alternative_alignment="common_ids", criterion_alignment="common_ids",
            rerank_common_alternatives=True,
        )]
        original = deepcopy(comparisons)
        experiment = SimpleNamespace(baseline_case="S01/control", comparisons=comparisons)
        self.assertEqual(_references(experiment, ["S01/control", "S02/max"]),
                         ("S01/control", "S02/max"))
        self.assertEqual(comparisons, original)

    def test_reranking_requires_a_boolean_when_present(self):
        for value in (True, False):
            experiment = SimpleNamespace(baseline_case=None, comparisons=[
                self.comparison(alternative_alignment="common_ids", rerank_common_alternatives=value),
            ])
            self.assertEqual(_references(experiment, []), ("S01/control",))
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value):
                experiment = SimpleNamespace(baseline_case=None, comparisons=[
                    self.comparison(rerank_common_alternatives=value),
                ])
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    _references(experiment, [])

    def test_invalid_alignment_policies_are_rejected(self):
        for field in ("criterion_alignment", "alternative_alignment"):
            for value in ("unknown", None, True, []):
                with self.subTest(field=field, value=value):
                    experiment = SimpleNamespace(baseline_case=None, comparisons=[
                        self.comparison(**{field: value}),
                    ])
                    with self.assertRaisesRegex(ValueError, "Unsupported comparison"):
                        _references(experiment, [])

    def test_common_criteria_allow_different_dimensions_without_dependency(self):
        cases = {e.session.id: e for e in load_experiments(ROOT / "experiments")}
        baseline = cases["S01/control"]
        changed = cases["S05/control"]
        changed.expected_assertions = []
        changed.session.criteria = [c for c in changed.session.criteria if c.id != "C4"]
        changed.session.questions = [q for q in changed.session.questions
                                     if "C4" not in (q.left_criterion_id, q.right_criterion_id)]
        qids = {q.id for q in changed.session.questions}
        for participant in changed.session.participants:
            participant.responses = [r for r in participant.responses if r.question_id in qids]
        for alternative in changed.session.alternatives:
            alternative.criterion_values = [v for v in alternative.criterion_values
                                             if v.criterion_id != "C4"]
        changed.comparisons = [self.comparison()]
        with self.assertRaisesRegex(ValueError, "requires identical criteria"):
            prepare_runs([baseline, changed], changed.session.id)
        changed.comparisons[0]["criterion_alignment"] = "common_ids"
        runs = prepare_runs([baseline, changed], changed.session.id)
        self.assertEqual({r.case_key for r in runs}, {"S01/control", "S05/control"})
        self.assertTrue(all(not r.dependencies for r in runs))

    def test_all_real_comparisons_prepare_without_false_execution_edges(self):
        runs = prepare_runs(load_experiments(ROOT / "experiments"))
        self.assertEqual(len(runs), 40)
        siblings = [r for r in runs if r.experiment.id == "S02"]
        self.assertEqual(len(siblings), 2)
        self.assertTrue(all(not r.dependencies for r in siblings))
        for run in runs:
            for comparison in run.experiment.comparisons:
                self.assertIn(comparison["reference"], run.reference_runs)


if __name__ == "__main__":
    unittest.main()
