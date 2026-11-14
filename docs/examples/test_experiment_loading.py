"""Focused checks for the tutorial's generation and inheritance examples."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import tempfile

from src.experiments.loader import load_experiment_json, load_experiments
from src.experiments.participants import (
    _participants, generator_strategies, register_generator_strategy,
)
from src.experiments.variations import resolve_cases
from src.mcdm.enums import PairwisePreference
from src.mcdm.models.domain import Participant, PairwiseResponse


ROOT = Path(__file__).resolve().parents[2]


class LoadingExampleTests(unittest.TestCase):
    def setUp(self):
        self.documents = {p: json.loads(p.read_text())
                          for p in sorted((ROOT / "experiments").glob("*.json"))}
        self.s05 = next(d for d in self.documents.values() if d["experiment_id"] == "S05")
        self.predefined = self.s05["stakeholders"]["participants"]["predefined"]

    def parse(self, generators=None, predefined=None):
        return _participants(
            self.predefined if predefined is None else predefined,
            generators or [], "test",
            group_ids={g["id"] for g in self.s05["stakeholders"]["stakeholder_groups"]},
            question_ids={f"q{i}" for i in range(1, 7)},
        )

    def rule(self, **changes):
        return {"generator_id": "extra", "strategy": "repeat_templates",
                "group_id": "community_residents", "count": 3,
                "template_participant_ids": ["community_residents_p01"], **changes}

    def test_existing_cases_and_counts(self):
        result = {c.key: c for c in resolve_cases(self.documents)}
        self.assertEqual({k: len(c.participants) for k, c in result.items()}, {
            "S01/control": 5, "S05/control": 5,
            "S05/unequal_voting_power": 5, "S05/unequal_participation": 50,
        })
        expanded = result["S05/unequal_participation"]
        self.assertEqual(sum(p.group_id == "environmental_advocates"
                             for p in expanded.participants), 30)
        self.assertEqual([g.voting_power for g in expanded.groups], [0.2] * 5)
        changed = result["S05/unequal_voting_power"]
        self.assertEqual([g.voting_power for g in changed.groups], [0.1, 0.1, 0.2, 0.4, 0.2])

    def test_generator_adds_and_copies_templates(self):
        result = self.parse([self.rule()])
        self.assertEqual(len(result), 8)
        self.assertEqual(result[-1].judgments, result[0].judgments)
        result[-1].judgments["q1"] = PairwisePreference.EXTREME_IMPORTANCE
        self.assertEqual(result[0].judgments["q1"], PairwisePreference.STRONG_IMPORTANCE_RECIPROCAL)
        self.assertEqual(result[-1].participant_id, "community_residents_extra_0003")

    def test_codes_stay_enums_in_domain_conversion(self):
        p = self.parse()[0]
        domain = Participant(id=p.participant_id, name=p.participant_id,
                             stakeholder_group_id=p.group_id,
                             responses=[PairwiseResponse(question_id=q, preference=v)
                                        for q, v in p.judgments.items()])
        self.assertIs(domain.responses[3].preference, PairwisePreference.STRONG_IMPORTANCE)
        self.assertEqual(domain.responses[3].preference.ahp_scale, 5.0)

    def test_invalid_judgments(self):
        for value in (True, "4.0", 5, -5, 1.5, 1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                items = deepcopy(self.predefined)
                items[0]["judgments"]["q1"] = value
                self.parse(predefined=items)
        items = deepcopy(self.predefined)
        del items[0]["judgments"]["q1"]
        with self.assertRaises(ValueError):
            self.parse(predefined=items)

    def test_invalid_generators(self):
        for changes in ({"count": True}, {"count": -1}, {"count": 1.5},
                        {"strategy": "Repeat"}, {"group_id": "missing"},
                        {"template_participant_ids": []},
                        {"template_participant_ids": ["city_government_p01"]},
                        {"id_pattern": "community_residents_p01"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse([self.rule(**changes)])

    def test_cases_do_not_share_mutable_inputs(self):
        before = deepcopy(self.documents)
        result = {c.key: c for c in resolve_cases(self.documents)}
        result["S05/unequal_participation"].participants[0].judgments["q1"] = PairwisePreference.EXTREME_IMPORTANCE
        self.assertEqual(result["S05/control"].participants[0].judgments["q1"], PairwisePreference.STRONG_IMPORTANCE_RECIPROCAL)
        self.assertEqual(self.documents, before)

    def test_input_cycles_and_unknown_references(self):
        self.s05["cases"][0]["input_from"] = "S05/unequal_participation"
        with self.assertRaisesRegex(ValueError, "cycle"):
            resolve_cases(self.documents)
        del self.s05["cases"][0]["input_from"]
        self.s05["cases"][0]["comparisons"][0]["reference"] = "S99/control"
        with self.assertRaisesRegex(ValueError, "unknown reference"):
            resolve_cases(self.documents)

    def test_case_order_does_not_affect_resolution(self):
        self.s05["cases"].reverse()
        result = {c.key: c for c in resolve_cases(self.documents)}
        self.assertEqual(len(result["S05/unequal_participation"].participants), 50)

    def test_unsupported_operation_and_invalid_powers(self):
        operation = self.s05["cases"][1]["operations"][0]
        for power in (True, 0, -1, float("nan"), 2):
            operation["voting_powers"]["community_residents"] = power
            with self.subTest(power=power), self.assertRaises(ValueError):
                resolve_cases(self.documents)
        operation["type"] = "remove_winner"
        with self.assertRaisesRegex(ValueError, "Unsupported case operation"):
            resolve_cases(self.documents)

    def test_resize_rejects_shrink_and_nonunique_pattern(self):
        operation = self.s05["cases"][2]["operations"][0]
        operation["count"] = 0
        with self.assertRaisesRegex(ValueError, "Shrinking"):
            resolve_cases(self.documents)
        operation["count"] = 5
        operation["id_pattern"] = "constant"
        with self.assertRaisesRegex(ValueError, "unique participant IDs"):
            resolve_cases(self.documents)

    def test_all_legacy_codes_and_string_codes(self):
        for code in range(-4, 5):
            for value in (code, str(code)):
                items = deepcopy(self.predefined)
                items[0]["judgments"]["q1"] = value
                self.assertIs(self.parse(predefined=items)[0].judgments["q1"],
                              PairwisePreference(str(code)))

    def test_strategy_registration_dispatch_and_input_isolation(self):
        # A plugin registers a function once; _participants needs no changes.
        name = "test_custom_templates"
        if name not in generator_strategies():
            @register_generator_strategy(name)
            def custom(rule, templates):
                p = templates[0]
                p.participant_id = "custom_generated"
                p.judgments["q1"] = PairwisePreference.EXTREME_IMPORTANCE
                rule["count"] = 99
                return [p]
        rule = self.rule(strategy=name, count=1)
        before = deepcopy(self.predefined)
        result = self.parse([rule])
        self.assertEqual(result[-1].participant_id, "custom_generated")
        self.assertEqual(self.predefined, before)
        self.assertEqual(rule["count"], 1)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_generator_strategy(name)(lambda rule, templates: [])
        snapshot = generator_strategies()
        snapshot.clear()
        self.assertIn("repeat_templates", generator_strategies())

    def test_generators_do_not_collide_across_rules(self):
        rules = [self.rule(generator_id="first", id_pattern="shared_{index}"),
                 self.rule(generator_id="second", id_pattern="shared_{index}")]
        with self.assertRaisesRegex(ValueError, "Duplicate participant IDs"):
            self.parse(rules)

    def test_production_loader_and_serialization(self):
        experiments = load_experiments(ROOT / "experiments")
        self.assertEqual(len(experiments), 4)
        for experiment in experiments:
            serialized = experiment.to_dict()
            self.assertEqual(serialized["session"]["id"], experiment.session.id)
            json.dumps(serialized, allow_nan=False)
            for p in experiment.session.participants:
                self.assertTrue(all(isinstance(r.preference, PairwisePreference)
                                    for r in p.responses))

    def test_load_experiment_json_includes_generated_participants(self):
        data = deepcopy(self.s05)
        data["stakeholders"]["participants"]["generators"] = [self.rule()]
        source = ROOT / "experiments"
        for name in ("criteria", "alternatives", "dataset"):
            data["resources"][name] = str((source / data["resources"][name]).resolve())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "S05.json"
            path.write_text(json.dumps(data))
            result = {e.session.id: e for e in load_experiment_json(path)}
        self.assertEqual(len(result["S05/control"].session.participants), 8)
        self.assertEqual(len(result["S05/unequal_participation"].session.participants), 50)

    def test_scale_conversion_preserves_neutrality_and_replication_invariance(self):
        import numpy as np
        from src.mcdm.reference import ahp

        def weights(experiment):
            ids = [c.id for c in experiment.session.criteria]
            qs = {q.id: q for q in experiment.session.questions}
            matrices, powers = [], []
            for group in experiment.session.stakeholder_groups:
                individuals = []
                for p in experiment.session.participants:
                    if p.stakeholder_group_id != group.id:
                        continue
                    matrix = np.ones((len(ids), len(ids)))
                    for response in p.responses:
                        q = qs[response.question_id]
                        i, j = ids.index(q.left_criterion_id), ids.index(q.right_criterion_id)
                        matrix[i, j] = response.preference.ahp_scale
                        matrix[j, i] = 1 / matrix[i, j]
                    individuals.append(matrix)
                matrices.append(np.exp(np.log(individuals).mean(axis=0)))
                powers.append(group.voting_power)
            power = np.array(powers) / sum(powers)
            aggregate = np.exp(np.tensordot(power, np.log(matrices), axes=1))
            return ahp(aggregate, wd="mean")["weights"]

        result = {e.session.id: weights(e) for e in load_experiments(ROOT / "experiments")}
        np.testing.assert_allclose(result["S01/control"], [0.25] * 4, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result["S05/control"], result["S05/unequal_participation"],
                                   atol=1e-10, rtol=0)
        self.assertFalse(np.allclose(result["S05/control"], result["S05/unequal_voting_power"]))


if __name__ == "__main__":
    unittest.main()
