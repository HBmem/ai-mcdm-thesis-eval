"""Stakeholder intervention validation and inheritance isolation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.experiments.variations import resolve_cases
from src.mcdm.enums import PairwisePreference


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/S01_uniform_preferences.json"


class JudgmentOperationTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(SOURCE.read_text())
        self.document["assertions"] = []
        self.document["cases"] = [{
            "case_id": "control", "operations": [], "assertions": [],
        }]
        self.group = "community_residents"
        self.pid = self.group + "_p01"
        self.responses = {f"q{i}": 0 for i in range(1, 7)}
        second = deepcopy(self.document["stakeholders"]["participants"]["predefined"][0])
        second["participant_id"] = self.group + "_p02"
        self.document["stakeholders"]["participants"]["predefined"].append(second)

    def resolve(self):
        return {c.key: c for c in resolve_cases({SOURCE: self.document})}

    def participant(self, case, pid=None):
        return next(p for p in case.participants if p.participant_id == (pid or self.pid))

    def test_group_then_participant_override_and_isolated_inheritance(self):
        group_responses = {**self.responses, "q1": 4}
        individual = {**self.responses, "q1": -4}
        # A child can precede its source in the JSON file.
        self.document["cases"] = [
            {"case_id": "child", "input_from": "S01/control", "operations": [
                {"type": "set_group_judgments", "judgments": {self.group: group_responses}},
                {"type": "set_participant_judgments", "judgments": {self.pid: individual}},
            ]},
            {"case_id": "control", "operations": []},
            {"case_id": "sibling", "input_from": "S01/control", "operations": []},
        ]
        original = deepcopy(self.document)
        cases = self.resolve()
        child = cases["S01/child"]
        self.assertIs(self.participant(child).judgments["q1"], PairwisePreference("-4"))
        self.assertIs(self.participant(child, self.group + "_p02").judgments["q1"], PairwisePreference("4"))
        for name in ("control", "sibling"):
            self.assertIs(self.participant(cases["S01/" + name]).judgments["q1"], PairwisePreference("0"))
        outsider = self.participant(child, "city_government_p01")
        self.assertIs(outsider.judgments["q1"], PairwisePreference("0"))
        self.assertEqual(self.document, original)

    def test_group_replacements_have_independent_judgment_maps(self):
        self.document["cases"][0]["operations"] = [
            {"type": "set_group_judgments", "judgments": {self.group: self.responses}},
        ]
        participants = [p for p in self.resolve()["S01/control"].participants if p.group_id == self.group]
        participants[0].judgments["q1"] = PairwisePreference("4")
        self.assertIs(participants[1].judgments["q1"], PairwisePreference("0"))

    def test_replication_and_overrides_follow_declared_order(self):
        resize = {
            "type": "resize_group", "group_id": self.group, "count": 3,
            "strategy": "repeat_templates", "template_participant_ids": [self.pid],
            "preserve_existing": True, "id_pattern": "{group_id}_p{index:02d}",
        }
        target = self.group + "_p03"
        self.document["cases"][0]["operations"] = [resize, {
            "type": "set_participant_judgments",
            "judgments": {target: {**self.responses, "q1": "3"}},
        }]
        case = self.resolve()["S01/control"]
        self.assertIs(self.participant(case, target).judgments["q1"], PairwisePreference("3"))
        self.assertIs(self.participant(case).judgments["q1"], PairwisePreference("0"))
        self.document["cases"][0]["operations"].reverse()
        with self.assertRaisesRegex(ValueError, "known target IDs"):
            self.resolve()

    def test_invalid_judgment_operations(self):
        for kind, target in (("set_group_judgments", self.group),
                             ("set_participant_judgments", self.pid)):
            valid = {"type": kind, "judgments": {target: self.responses}}
            invalid = [
                {"type": kind},
                {"type": kind, "judgments": []},
                {"type": kind, "judgments": {"unknown": self.responses}},
                {**valid, "unexpected": True},
                {"type": kind, "judgments": {target: []}},
                {"type": kind, "judgments": {target: {"q1": 0}}},
                {"type": kind, "judgments": {target: {**self.responses, "q7": 0}}},
            ]
            invalid.extend({"type": kind, "judgments": {target: {**self.responses, "q1": v}}}
                           for v in (True, 4.0, 9, "1/3"))
            for operation in invalid:
                with self.subTest(operation=operation):
                    self.document["cases"][0]["operations"] = [operation]
                    with self.assertRaises(ValueError):
                        self.resolve()

    def test_malformed_operations_and_runtime_operation_rejected(self):
        for operations in (None, {}, [None], [{}], [{"type": 1}], [{"type": "remove_winner"}]):
            with self.subTest(operations=operations):
                self.document["cases"][0]["operations"] = operations
                with self.assertRaises(ValueError):
                    self.resolve()

    def test_all_experiment_cases_resolve(self):
        documents = {p: json.loads(p.read_text()) for p in (ROOT / "experiments").glob("*.json")}
        cases = {c.key: c for c in resolve_cases(documents)}
        self.assertEqual(len(cases), 40)
        self.assertTrue(all(isinstance(v, PairwisePreference) for c in cases.values()
                            for p in c.participants for v in p.judgments.values()))
        self.assertTrue(all(v is PairwisePreference("-4") for p in cases["S02/min"].participants
                            for v in p.judgments.values()))
        for key, changed_count in (("one_participant", 1), ("one_group", 2), ("all_groups", 10)):
            participants = cases["S10/" + key].participants
            self.assertEqual(len(participants), 10)
            self.assertEqual(sum(p.judgments["q1"] is PairwisePreference("2")
                                 for p in participants), changed_count)


if __name__ == "__main__":
    unittest.main()
