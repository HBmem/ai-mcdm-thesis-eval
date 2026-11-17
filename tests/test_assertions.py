"""Assertion contracts against current dataclass results and schema-3 cases."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.assertions import (
    ASSERTION_PARAMETERS, EVALUATORS, assertion_evaluator,
    evaluate_declared_assertions,
)
from src.experiments.loader import load_experiments
from src.experiments.runner import execute_experiments, execute_run, prepare_runs
from src.mcdm.reference import TIE_ATOL


ROOT = Path(__file__).resolve().parents[1]
SCOPES = ["participant", "group", "session"]
REFERENCE_CHECK = "computed_results_match_independent_reference_per_variant"


def spec(check, parameters=None, expected=True, **extra):
    return {"id": check, "check": check, "parameters": parameters or {},
            "expected": expected, "atol": 1e-10, "rtol": 0, **extra}


def evaluate(check, result, reference=None, parameters=None, expected=True, **extra):
    return EVALUATORS[check](result, reference, spec(check, parameters, expected, **extra))


class AssertionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = prepare_runs(load_experiments(ROOT / "experiments"))
        _, results = execute_experiments(cls.runs, output=None)
        cls.results = {r.case_key: r for r in results}
        cls.completed = {run.key: result for run, result in zip(cls.runs, results)}
        cls.baseline = cls.results["S01/control"]
        if any(r.status != "calculated" for r in results):
            raise AssertionError([(r.case_key, r.execution) for r in results if r.status != "calculated"])

    def setUp(self):
        self.current = deepcopy(self.baseline)
        self.reference = deepcopy(self.baseline)

    def test_registry_and_duplicate_registration(self):
        self.assertEqual(set(EVALUATORS), set(ASSERTION_PARAMETERS))
        original = dict(EVALUATORS)
        with self.assertRaisesRegex(ValueError, "Duplicate assertion evaluator"):
            assertion_evaluator("normalized_weights")(lambda *args: None)
        self.assertEqual(EVALUATORS, original)

    def test_real_declarations_at_both_stages(self):
        exercised = set()
        for run in self.runs:
            result = self.results[run.case_key]
            for stage in ("calculation", "comparison"):
                records = evaluate_declared_assertions(run, result, self.completed, stage=stage)
                json.dumps(records, allow_nan=False)
                for record in records:
                    with self.subTest(case=run.case_key, check=record["check"]):
                        self.assertIn(record["status"], ("passed", "deferred"), record)
                    exercised.add(record["check"])
        self.assertEqual(exercised, set(ASSERTION_PARAMETERS))

    def handler_examples(self):
        """A passing example for every handler, independent of deferred dispatch."""
        weights = dict(self.current.mcdm.session.metadata.applied_criterion_weights)
        session = self.current.inputs["session"]
        group = session["participants"][0]["stakeholder_group_id"]
        counts = {g["id"]: 1 for g in session["stakeholder_groups"]}
        powers = {g["id"]: g["voting_power"] for g in session["stakeholder_groups"]}
        for check in ASSERTION_PARAMETERS:
            current, reference = deepcopy(self.current), deepcopy(self.reference)
            parameters, expected = {}, True
            if "reference" in set().union(*ASSERTION_PARAMETERS[check]):
                parameters["reference"] = "S01/control"
            if check == "all_participant_consistency_ratios":
                expected = 0.0
            elif check == "inconsistent_participant_count":
                expected = 0
            elif check == "participant_counts_equal":
                parameters = {"counts": counts}
            elif check == "group_voting_powers_equal":
                parameters = {"expected_powers": powers}
            elif check in ("session_weights_equal", "group_weights_equal"):
                parameters = {"expected_weights": weights}
            elif check == "group_judgments_equal":
                parameters = {"expected_judgments": {group: {q["id"]: 0 for q in session["questions"]}}}
            elif check == "ranking_criterion_ids_equal":
                parameters = {"expected_ids": list(weights)}
            elif check in ("scoped_weights_equal", "scoped_scores_equal", "restricted_weights_equal"):
                parameters["scopes"] = SCOPES
            elif check == "alternative_ids_equal_reference_without_winners":
                parameters["tie_atol"] = TIE_ATOL
                rows = reference.mcdm.session.ranking.rankings
                best = max(r.closeness_coefficient for r in rows)
                winners = {r.alternative_id for r in rows if abs(r.closeness_coefficient - best) <= TIE_ATOL}
                current.inputs["session"]["alternatives"] = [a for a in session["alternatives"] if a["id"] not in winners]
            elif check == "alternative_score_not_greater":
                rows = current.mcdm.session.ranking.rankings
                parameters = {"alternative": rows[-1].alternative_id, "reference_alternative": rows[0].alternative_id}
            elif check == "criterion_weight_greater_than_reference":
                parameters["criterion_id"] = "C1"
                current.mcdm.session.metadata.applied_criterion_weights["C1"] += .1
            yield check, current, reference, parameters, expected

    def test_every_handler_positive_negative_and_serializable(self):
        for check, current, reference, parameters, expected in self.handler_examples():
            with self.subTest(check=check):
                observation = evaluate(check, current, reference, parameters, expected)
                self.assertTrue(observation.passed, observation)
                json.dumps(asdict(observation), allow_nan=False)
                opposite = not expected if type(expected) is bool else expected + 1
                self.assertFalse(evaluate(check, current, reference, parameters, opposite).passed)

    def test_numeric_invariants_detect_corrupted_evidence(self):
        mutations = {
            "normalized_weights": lambda r: setattr(r.mcdm.participants[0].weighting.criterion_weights[0], "weight", .9),
            "bounded_topsis_scores": lambda r: setattr(r.mcdm.groups[0].ranking.rankings[0], "closeness_coefficient", float("nan")),
            "submitted_values_in_saaty_domain": lambda r: r.inputs["session"]["participants"][0]["responses"][0].update(preference="9"),
            "reciprocal_matrices": lambda r: r.mcdm.groups[0].pairwise_matrix[0].__setitem__(1, 2),
            "all_participant_pairwise_matrices_equal": lambda r: r.mcdm.participants[0].pairwise_matrix[0].__setitem__(1, 2),
            "all_participant_consistency_ratios": lambda r: setattr(r.mcdm.participants[0].weighting, "consistency_ratio", .2),
            "session_aggregate_pairwise_matrix_all_ones": lambda r: r.mcdm.session.pairwise_matrix[0].__setitem__(1, 2),
            "group_weights_all_equal": lambda r: setattr(r.mcdm.groups[0].weighting.criterion_weights[0], "weight", .9),
            "inconsistent_participant_count": lambda r: setattr(r.mcdm.participants[0].weighting, "consistency_ratio", .2),
            "flagged_participants_retained": lambda r: r.mcdm.participants.pop(),
            "inconsistent_inputs_retained_and_flagged": lambda r: r.mcdm.participants[0].pairwise_matrix[0].__setitem__(1, 2),
            REFERENCE_CHECK: lambda r: setattr(r.mcdm.session.ranking.rankings[0], "closeness_coefficient", .123),
        }
        for check, mutate in mutations.items():
            with self.subTest(check=check):
                changed = deepcopy(self.current)
                mutate(changed)
                expected = 0 if check in ("all_participant_consistency_ratios", "inconsistent_participant_count") else True
                observation = evaluate(check, changed, expected=expected)
                self.assertFalse(observation.passed, observation)
                json.dumps(asdict(observation), allow_nan=False)

    def test_comparisons_detect_changed_evidence(self):
        mutations = {
            "session_weights_equal": lambda r: r.mcdm.session.metadata.applied_criterion_weights.update(C1=.3),
            "weights_equal": lambda r: r.mcdm.session.metadata.applied_criterion_weights.update(C1=.3),
            "scores_equal": lambda r: setattr(r.mcdm.session.ranking.rankings[0], "closeness_coefficient", .1),
            "scoped_scores_equal": lambda r: setattr(r.mcdm.participants[0].ranking.rankings[0], "closeness_coefficient", .1),
            "scoped_weights_equal": lambda r: r.mcdm.participants[0].metadata.applied_criterion_weights.update(C1=.3),
            "group_weights_equal": lambda r: setattr(r.mcdm.groups[0].weighting.criterion_weights[0], "weight", .3),
            "restricted_weights_equal": lambda r: r.mcdm.groups[0].metadata.applied_criterion_weights.update(C1=.3),
            "ranking_criterion_ids_equal": lambda r: r.mcdm.participants[0].metadata.ranking_criterion_ids.pop(),
            "participant_judgments_equal": lambda r: r.inputs["session"]["participants"][0]["responses"][0].update(preference="1"),
            "group_judgments_equal": lambda r: r.inputs["session"]["participants"][0]["responses"][0].update(preference="1"),
            "preference_evidence_equal": lambda r: r.inputs["session"]["questions"][0].update(text="changed"),
            "alternative_ids_equal": lambda r: r.inputs["session"]["alternatives"].pop(),
            "alternative_ids_equal_reference_without_winners": lambda r: r.inputs["session"]["alternatives"].pop(),
            "winner_sets_equal": lambda r: setattr(r.mcdm.session.ranking.rankings[-1], "closeness_coefficient", 1),
            "participant_counts_equal": lambda r: r.inputs["session"]["participants"].pop(),
            "group_voting_powers_equal": lambda r: r.inputs["session"]["stakeholder_groups"][0].update(voting_power=.9),
            "criterion_weight_greater_than_reference": lambda r: r.mcdm.session.metadata.applied_criterion_weights.update(C1=.25),
            "alternative_score_not_greater": lambda r: setattr(r.mcdm.session.ranking.rankings[-1], "closeness_coefficient", 1),
        }
        for check, current, reference, parameters, expected in self.handler_examples():
            if check not in mutations:
                continue
            with self.subTest(check=check):
                mutations[check](current)
                self.assertFalse(evaluate(check, current, reference, parameters, expected).passed)

    def test_mapping_and_matrix_alignment(self):
        session = self.current.inputs["session"]
        session["criteria"].reverse()
        for p in session["participants"]:
            p["responses"].reverse()
        session["questions"].reverse()
        session["participants"].reverse()
        session["stakeholder_groups"].reverse()
        session["alternatives"].reverse()
        self.current.mcdm.participants.reverse()
        self.current.mcdm.groups.reverse()
        for e in [*self.current.mcdm.participants, *self.current.mcdm.groups, self.current.mcdm.session]:
            e.pairwise_matrix = np.asarray(e.pairwise_matrix)[::-1, ::-1].tolist()
            e.weighting.criterion_weights.reverse()
            e.ranking.rankings.reverse()
        for check in ("scoped_weights_equal", "scoped_scores_equal", "participant_judgments_equal", "preference_evidence_equal"):
            with self.subTest(check=check):
                self.assertTrue(evaluate(check, self.current, self.reference, {"scopes": SCOPES}).passed)
        self.assertTrue(evaluate(REFERENCE_CHECK, self.current).passed)

    def test_missing_duplicate_and_incomplete_evidence(self):
        for collection, check, parameters in (
            ("participants", "scoped_weights_equal", {"scopes": ["participant"]}),
            ("groups", "group_weights_equal", {"reference": "S01/control"}),
        ):
            with self.subTest(collection=collection):
                current = deepcopy(self.current)
                entities = getattr(current.mcdm, collection)
                entities.pop()
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    evaluate(check, current, self.reference, parameters)
                entities.append(entities[0])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    evaluate(check, current, self.reference, parameters)
        self.current.mcdm.session.weighting.criterion_weights.append(self.current.mcdm.session.weighting.criterion_weights[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate("normalized_weights", self.current)
        self.current = deepcopy(self.baseline)
        self.current.mcdm.session.ranking.rankings.pop()
        self.assertFalse(evaluate("bounded_topsis_scores", self.current).passed)
        self.current.mcdm.session = None
        with self.assertRaisesRegex(ValueError, "Missing session"):
            evaluate("weights_equal", self.current, self.reference)

    def test_tolerances_and_negative_expectations(self):
        weights = self.current.mcdm.session.metadata.applied_criterion_weights
        weights["C1"] += 1e-5
        for atol, expected in ((1e-4, True), (1e-6, False)):
            self.assertTrue(evaluate("weights_equal", self.current, self.reference, expected=expected, atol=atol).passed)
        self.assertTrue(evaluate("weights_equal", self.current, self.reference, rtol=1e-3).passed)
        self.assertFalse(evaluate("criterion_weight_greater_than_reference", self.current, self.reference,
                                  {"criterion_id": "C1"}, atol=1e-4).passed)
        self.assertTrue(evaluate("criterion_weight_greater_than_reference", self.current, self.reference,
                                 {"criterion_id": "C1"}, atol=1e-6).passed)
        self.assertFalse(evaluate("inconsistent_participant_count", self.current, expected=False).passed)

    def test_tied_winners_use_scores_and_absolute_tolerance(self):
        for result in (self.current, self.reference):
            rows = result.mcdm.session.ranking.rankings
            for row in rows:
                row.closeness_coefficient = .1
            rows[0].closeness_coefficient = .9
        self.current.mcdm.session.ranking.rankings[1].closeness_coefficient = .9 - 5e-13
        self.assertFalse(evaluate("winner_sets_equal", self.current, self.reference).passed)
        self.reference.mcdm.session.ranking.rankings[1].closeness_coefficient = .9
        self.assertTrue(evaluate("winner_sets_equal", self.current, self.reference).passed)
        winner_ids = {r.alternative_id for r in self.reference.mcdm.session.ranking.rankings[:2]}
        self.current.inputs["session"]["alternatives"] = [a for a in self.current.inputs["session"]["alternatives"] if a["id"] not in winner_ids]
        self.assertTrue(evaluate("alternative_ids_equal_reference_without_winners", self.current, self.reference,
                                 {"tie_atol": 1e-12}).passed)

    def test_empty_groups_are_included_in_counts_and_configured_powers(self):
        session = self.current.inputs["session"]
        extra = {**session["stakeholder_groups"][0], "id": "empty", "voting_power": .5}
        session["stakeholder_groups"].append(extra)
        counts = {g["id"]: int(g["id"] != "empty") for g in session["stakeholder_groups"]}
        powers = {g["id"]: g["voting_power"] for g in session["stakeholder_groups"]}
        self.assertTrue(evaluate("participant_counts_equal", self.current, parameters={"counts": counts}).passed)
        self.assertTrue(evaluate("group_voting_powers_equal", self.current, parameters={"expected_powers": powers}).passed)
        self.assertTrue(evaluate(REFERENCE_CHECK, self.current).passed)

    def test_independent_reference_for_all_variants(self):
        for key, result in self.results.items():
            with self.subTest(case=key):
                self.assertTrue(evaluate(REFERENCE_CHECK, result).passed)

    def test_independent_reference_supports_all_ahp_derivations(self):
        source = next(r for r in self.runs if r.case_key == "S10/one_participant")
        for derivation in ("mean", "geometric", "max_eigen"):
            run = deepcopy(source)
            run.experiment.processing.weighting_method.parameters["derivation"] = derivation
            result = execute_run(run)
            with self.subTest(derivation=derivation):
                self.assertEqual(result.status, "calculated", result.execution)
                self.assertTrue(evaluate(REFERENCE_CHECK, result).passed)

    def test_numeric_domain_failures_and_serialization(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            current = deepcopy(self.current)
            current.mcdm.session.metadata.applied_criterion_weights["C1"] = value
            self.assertFalse(evaluate("normalized_weights", current).passed)
            observation = evaluate("session_weights_equal", current, parameters={
                "expected_weights": self.baseline.mcdm.session.metadata.applied_criterion_weights})
            self.assertFalse(observation.passed)
            json.dumps(asdict(observation), allow_nan=False)
        for value in (0, -1, float("nan")):
            current = deepcopy(self.current)
            current.mcdm.session.pairwise_matrix[0][0] = value
            observation = evaluate("reciprocal_matrices", current)
            self.assertFalse(observation.passed)
            json.dumps(asdict(observation), allow_nan=False)
        self.current.inputs["judgment_encoding"] = "saaty_ratio"
        for p in self.current.inputs["session"]["participants"]:
            for response in p["responses"]:
                response["preference"] = 1.0
        self.assertTrue(evaluate("submitted_values_in_saaty_domain", self.current).passed)
        self.current.inputs["session"]["participants"][0]["responses"][0]["preference"] = 1.5
        self.assertFalse(evaluate("submitted_values_in_saaty_domain", self.current).passed)

    def test_independent_reference_corruptions_and_unsupported_methods(self):
        mutations = (
            lambda r: setattr(r.mcdm.participants[0].weighting.criterion_weights[0], "weight", .9),
            lambda r: setattr(r.mcdm.groups[0].weighting, "consistency_ratio", .2),
            lambda r: r.mcdm.session.pairwise_matrix[0].__setitem__(1, 2),
            lambda r: r.mcdm.session.metadata.applied_criterion_weights.update(C1=.3),
            lambda r: setattr(r.mcdm.session.ranking.rankings[0], "rank", 99),
            lambda r: r.mcdm.session.metadata.ranking_criterion_ids.pop(),
        )
        for mutate in mutations:
            current = deepcopy(self.current)
            mutate(current)
            self.assertFalse(evaluate(REFERENCE_CHECK, current).passed)
        for config, field, value in (("weighting_method", "method", "fuzzy_ahp"),
                                      ("ranking_method", "method", "fuzzy_topsis"),
                                      ("aggregation_method", "within_group", "mean")):
            current = deepcopy(self.current)
            current.inputs["processing"][config][field] = value
            with self.assertRaisesRegex(NotImplementedError, "Independent reference"):
                evaluate(REFERENCE_CHECK, current)

    def test_restricted_and_explicit_weights(self):
        restricted = next(r for k, r in self.results.items() if k.startswith("S07/"))
        explicit = next(r for k, r in self.results.items() if k.startswith("S08/"))
        self.assertTrue(evaluate("restricted_weights_equal", restricted, self.reference, {"scopes": SCOPES}).passed)
        self.assertTrue(evaluate("preference_evidence_equal", restricted, self.reference).passed)
        for result in (restricted, explicit):
            self.assertTrue(evaluate(REFERENCE_CHECK, result).passed)
        for group in self.reference.mcdm.groups:
            group.metadata.applied_criterion_weights = {cid: 0.0 for cid in group.metadata.applied_criterion_weights}
        with self.assertRaisesRegex(ValueError, "positive retained mass"):
            evaluate("restricted_weights_equal", restricted, self.reference, {"scopes": ["group"]})

    def test_inconsistency_retention_checks_flags_and_original_evidence(self):
        original = self.results["S10/one_participant"]
        self.assertTrue(evaluate("flagged_participants_retained", original, atol=0).passed)
        for change in ("flag", "judgment", "matrix", "participant", "group"):
            current = deepcopy(original)
            entity = next(e for e in current.mcdm.participants if e.weighting.exceeds_consistency_threshold)
            if change == "flag":
                entity.weighting.consistency_ratio = 0
            elif change == "judgment":
                p = next(p for p in current.inputs["session"]["participants"] if p["id"] == entity.entity_id)
                p["responses"][0]["preference"] = "0"
            elif change == "matrix":
                entity.pairwise_matrix[0][1] *= 2
            elif change == "group":
                entity.metadata.group_id = "wrong"
            else:
                current.mcdm.participants.remove(entity)
            with self.subTest(change=change):
                self.assertFalse(evaluate("flagged_participants_retained", current).passed)

    def test_preference_evidence_includes_original_calculations(self):
        self.current.mcdm.session.weighting.criterion_weights[0].weight += .01
        self.assertFalse(evaluate("preference_evidence_equal", self.current, self.reference).passed)
        self.current = deepcopy(self.baseline)
        self.current.mcdm.groups[0].pairwise_matrix[0][1] = 2
        self.assertFalse(evaluate("preference_evidence_equal", self.current, self.reference).passed)
        self.current = deepcopy(self.baseline)
        self.current.mcdm.participants[0].weighting.consistency_ratio = .1
        self.assertFalse(evaluate("preference_evidence_equal", self.current, self.reference).passed)

    def test_group_judgments_check_every_member(self):
        result = deepcopy(self.results["S05/unequal_participation"])
        members = result.inputs["session"]["participants"]
        gid = members[0]["stakeholder_group_id"]
        group = [p for p in members if p["stakeholder_group_id"] == gid]
        expected = {r["question_id"]: r["preference"] for r in group[0]["responses"]}
        self.assertTrue(evaluate("group_judgments_equal", result, parameters={"expected_judgments": {gid: expected}}).passed)
        group[-1]["responses"][0]["preference"] = "4"
        self.assertFalse(evaluate("group_judgments_equal", result, parameters={"expected_judgments": {gid: expected}}).passed)

    def test_reference_and_stage_dispatch_with_isolated_errors(self):
        declarations = [spec("normalized_weights"), spec("weights_equal", {"reference": "base"}),
                        spec("unknown"), spec("scores_equal", {"reference": "missing"})]
        run = SimpleNamespace(experiment=SimpleNamespace(expected_assertions=declarations), reference_runs={"base": "base#1"})
        calculation = evaluate_declared_assertions(run, self.current, {"base#1": self.reference}, stage="calculation")
        self.assertEqual([r["status"] for r in calculation], ["passed", "error"])
        self.assertIn("LookupError", calculation[-1]["reason"])
        comparison = evaluate_declared_assertions(run, self.current, {"base#1": self.reference}, stage="comparison")
        self.assertEqual([r["status"] for r in comparison], ["passed", "error"])
        self.assertEqual(comparison[0]["reference_run_id"], self.reference.run_id)
        self.assertIn("KeyError", comparison[-1]["reason"])
        self.reference.execution["numerical_status"] = "failed"
        record = evaluate_declared_assertions(run, self.current, {"base#1": self.reference}, stage="comparison")[0]
        self.assertEqual(record["status"], "error")
        self.assertIn("no completed numerical results", record["reason"])
        json.dumps(comparison + calculation, allow_nan=False)

    def test_deferred_checks_bypass_dispatch_and_reference_lookup(self):
        declarations = [spec("unknown", deferred=True, reason="Awaiting evidence"),
                        spec("weights_equal", {"reference": "missing"}, deferred=True, reason="Missing source")]
        run = SimpleNamespace(experiment=SimpleNamespace(expected_assertions=declarations), reference_runs={})
        with patch.dict(EVALUATORS, {"unknown": lambda *a: self.fail("Dispatched deferred check")}):
            records = [*evaluate_declared_assertions(run, self.current, {}, stage="calculation"),
                       *evaluate_declared_assertions(run, self.current, {}, stage="comparison")]
        self.assertEqual([r["status"] for r in records], ["deferred", "deferred"])
        self.assertEqual([r["reason"] for r in records], ["Awaiting evidence", "Missing source"])
        self.assertTrue(all(r["observed"] is None for r in records))


if __name__ == "__main__":
    unittest.main()
