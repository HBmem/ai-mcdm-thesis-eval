"""AI configuration preservation and validation at the experiment loading boundary."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from src.experiments.loader import _ai, load_experiment_json, load_experiments
from src.experiments.models import ExperimentAI
from src.experiments.runner import prepare_runs


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class AILoaderTests(unittest.TestCase):
    def load_modified(self, document):
        document = deepcopy(document)
        for key in ("criteria", "alternatives", "dataset"):
            document["resources"][key] = str((EXPERIMENTS / document["resources"][key]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(document))
            return load_experiment_json(path)

    def test_all_specs_preserve_ai_settings_and_output(self):
        experiments = load_experiments(EXPERIMENTS)
        self.assertEqual(len(experiments), 45)
        sources = {d["experiment_id"]: d for path in EXPERIMENTS.glob("*.json")
                   for d in [json.loads(path.read_text())]}
        for experiment in experiments:
            with self.subTest(case=experiment.session.id):
                source = sources[experiment.id]
                self.assertIsInstance(experiment.defaults.ai, ExperimentAI)
                self.assertEqual(asdict(experiment.ai), source["defaults"]["ai"])
                self.assertEqual(asdict(experiment.defaults.ai), source["defaults"]["ai"])
                self.assertEqual(experiment.output, source["output"])
                plain = experiment.to_dict()
                self.assertEqual(plain["ai"], source["defaults"]["ai"])
                self.assertEqual(plain["defaults"]["ai"], source["defaults"]["ai"])
                json.dumps(plain, allow_nan=False)

    def test_nested_settings_are_independent_between_cases_and_defaults(self):
        document = json.loads((EXPERIMENTS / "S11_extreme_preference_ai_interpretation.json").read_text())
        document["defaults"]["ai"]["parameters"] = {"nested": {"values": [1]}}
        document["defaults"]["ai"]["prompt_overrides"] = {
            "data_comprehension": {"task": "Original task"},
        }
        first, second = self.load_modified(document)
        original = deepcopy(second.to_dict())
        first.ai.parameters["nested"]["values"].append(2)
        first.ai.layers.clear()
        first.ai.prompt_overrides["data_comprehension"]["task"] = "Changed"
        first.output["save_ai_calls"] = False
        self.assertEqual(second.to_dict(), original)
        self.assertEqual(first.defaults.ai.parameters["nested"]["values"], [1])
        self.assertEqual(first.defaults.ai.prompt_overrides["data_comprehension"]["task"], "Original task")
        self.assertEqual(document["defaults"]["ai"]["parameters"]["nested"]["values"], [1])
        treatment = self.load_modified(json.loads((EXPERIMENTS / "S13_inconsistent_reported_ranking.json").read_text()))[0]
        treatment.ai.reported_result_interventions[0]["score"] = 0.5
        self.assertEqual(treatment.defaults.ai.reported_result_interventions[0]["score"], 1.0)

    def test_minimal_legacy_ai_settings_get_independent_defaults(self):
        first = _ai({"enabled": False})
        second = _ai({"enabled": False})
        self.assertEqual(first.repetitions, 1)
        self.assertIsNone(first.provider)
        first.layers.clear()
        first.parameters["temperature"] = 0
        self.assertEqual(len(second.layers), 3)
        self.assertEqual(second.parameters, {})

    def test_invalid_ai_settings_are_rejected(self):
        invalid = [None, [], {"enabled": "true"}, {"unknown": 1}]
        invalid += [{"repetitions": value} for value in (True, 0, -1, 1.5)]
        invalid += [{"layers": value} for value in (None, "data_comprehension", ["unknown"], ["data_comprehension"] * 2, [{}])]
        invalid += [{"enabled": True, "layers": []}]
        invalid += [{key: value} for key in ("model", "provider") for value in ("", " ", 3, False)]
        invalid += [{"parameters": value} for value in ([], {"temperature": float("nan")}, {"x": object()})]
        invalid += [{"prompt_overrides": value} for value in (
            [], {"unknown": {}}, {"data_comprehension": []},
            {"data_comprehension": {"unknown": "text"}},
            {"data_comprehension": {"task": " "}},
        )]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                _ai(data)

    def test_reported_interventions_are_validated_without_being_applied(self):
        document = json.loads((EXPERIMENTS / "S13_inconsistent_reported_ranking.json").read_text())
        ai = document["defaults"]["ai"]
        intervention = ai["reported_result_interventions"][0]
        for field, value in (("scope", "group"), ("type", "unknown"),
                             ("alternative_id", ""), ("score", True),
                             ("score", 1.1), ("score", float("inf")),
                             ("rank", True), ("rank", 2), ("rerank_remaining", False)):
            bad = deepcopy(ai)
            bad["reported_result_interventions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                _ai(bad)
        loaded = self.load_modified(document)[0]
        a6 = next(a for a in loaded.session.alternatives if a.id == "A6")
        self.assertEqual({v.criterion_id: v.value for v in a6.criterion_values},
                         {"C1": 12, "C2": 20, "C3": 10, "C4": 36})
        self.assertEqual(loaded.ai.reported_result_interventions, [intervention])

    def test_pipeline_and_output_validation(self):
        document = json.loads((EXPERIMENTS / "S10_baseline_ai_interpretation.json").read_text())
        for pipeline in ("unknown", "mcdm"):
            bad = deepcopy(document)
            bad["defaults"]["pipeline"] = pipeline
            with self.subTest(pipeline=pipeline), self.assertRaises(ValueError):
                self.load_modified(bad)
        bad = deepcopy(document)
        bad["defaults"]["ai"]["enabled"] = False
        with self.assertRaises(ValueError):
            self.load_modified(bad)
        for output in ([], {"save_ai_calls": 1}, {"save_ai_responses": "false"}):
            bad = deepcopy(document)
            bad["output"] = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.load_modified(bad)

    def test_numerical_planning_still_works_and_ai_execution_is_not_enabled(self):
        experiments = load_experiments(EXPERIMENTS)
        numerical = [e for e in experiments if not e.ai.enabled]
        self.assertEqual(len(prepare_runs(numerical)), 40)
        baseline_ai = next(e for e in experiments if e.id == "S10")
        with self.assertRaisesRegex(ValueError, "Only the numerical mcdm pipeline"):
            prepare_runs([baseline_ai])


if __name__ == "__main__":
    unittest.main()
