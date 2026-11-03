import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import is_dataclass
from pathlib import Path
from shutil import copytree

import numpy as np

from src.enum import PairwisePreference, RankingMethod, WeightingMethod
from src.experiment import ExperimentExecution, ExperimentStakeholders
from src.load_experiments import load_experiments
from src.models.methods import RankingMethodConfig, WeightingMethodConfig
from src.models.session import DecisionSession
from src.services.matrix import MCDMService


ROOT = Path(__file__).resolve().parents[1]
S01 = "S01_uniform_preferences.json"
S02 = "S02_extreme_preferences.json"


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("experiments", "criteria", "alternative", "datasets"):
            copytree(ROOT / name, self.root / name)
        self.directory = self.root / "experiments"
        # These regression fixtures intentionally cover the original schema-1 inputs.
        for path in self.directory.glob("*.json"):
            if path.name not in (S01, S02):
                path.unlink()

    def read(self, filename=S01):
        return json.loads((self.directory / filename).read_text())

    def write(self, data, filename=S01):
        (self.directory / filename).write_text(json.dumps(data), encoding="utf-8")

    def load(self):
        return load_experiments(self.directory)

    def test_populated_runs_and_metadata(self):
        runs = self.load()
        self.assertEqual([run.session.id for run in runs], ["S01", "S02_MAX", "S02_MIN"])
        self.assertEqual([run.id for run in runs], ["S01", "S02", "S02"])
        self.assertEqual([run.variant_id for run in runs], [None, "S02_MAX", "S02_MIN"])
        for run in runs:
            self.assertTrue(is_dataclass(run))
            self.assertIsInstance(run.session, DecisionSession)
            self.assertEqual(
                [len(getattr(run.session, key)) for key in (
                    "criteria", "questions", "stakeholder_groups", "participants", "alternatives"
                )],
                [4, 6, 5, 5, 5],
            )
            self.assertIs(run.processing.weighting_method.method, WeightingMethod.AHP)
            self.assertIs(run.processing.ranking_method.method, RankingMethod.TOPSIS)
            self.assertEqual(run.processing.weighting_method.parameters, {"wd": "mean"})
            self.assertEqual(run.session.participants[0].name, run.session.participants[0].id)
        source = self.read()
        for key in ("schema_version", "name", "research_purpose", "expected_assertions", "ai_evaluation", "output"):
            self.assertEqual(getattr(runs[0], key), source[key])
        self.assertEqual(runs[0].scenario_id, source["scenario"]["id"])
        self.assertEqual(runs[1].variant_description, self.read(S02)["stakeholders"]["participant_variants"][0]["description"])
        self.assertEqual([item.id for item in runs[0].session.criteria], ["C1", "C2", "C3", "C4"])
        np.testing.assert_array_equal(MCDMService.create_decision_matrix(runs[0].session), [
            [8, 95, 75, 24], [3, 65, 90, 12], [6, 80, 40, 18],
            [2, 70, 65, 8], [5, 85, 80, 16],
        ])
        question = runs[0].session.questions[0]
        self.assertEqual((question.id, question.left_criterion_id, question.right_criterion_id), ("q1", "C1", "C2"))
        self.assertIn("Implementation Cost", question.text)
        self.assertIn("Accessibility Improvement", question.text)

    def test_preferences_and_services(self):
        runs = self.load()
        weights_by_run = []
        for run, code, ratio in zip(runs, ("0", "4", "-4"), (1, 9, 1 / 9)):
            response = run.session.participants[0].responses[0]
            self.assertIs(response.preference, PairwisePreference(code))
            matrix = MCDMService.create_session_matrix(run.session)
            self.assertAlmostEqual(matrix[0, 1], ratio)
            np.testing.assert_allclose(matrix * matrix.T, np.ones((4, 4)))
            weights, cr = MCDMService.generate_weights(run.session, run.processing.weighting_method)
            weights_by_run.append(weights)
            self.assertAlmostEqual(weights.sum(), 1)
            ranking = MCDMService.generate_ranking(run.session, weights, run.processing.ranking_method)
            self.assertEqual(len(ranking), 5)
            self.assertEqual(sorted(item.rank for item in ranking), [1, 2, 3, 4, 5])
            if run.variant_id:
                self.assertGreater(cr, 0.1)  # Inconsistent judgments remain executable.
            else:
                self.assertAlmostEqual(cr, 0)
        np.testing.assert_allclose(weights_by_run[0], [0.25] * 4)
        self.assertFalse(np.allclose(weights_by_run[1], weights_by_run[2]))

    def test_aliases_and_nested_method_configuration(self):
        data = self.read()
        data["id"] = data.pop("experiment_id")
        data["stakeholders"]["participants"] = data["stakeholders"].pop("participant")
        scenario = data.pop("scenario")
        data["scenario_id"] = scenario["id"]
        data["scenario_description"] = scenario["description"]
        for key, options in (("weighting_method", "weighting"), ("ranking_method", "ranking")):
            data["processing"][key] = {
                "method": data["processing"][key],
                "parameters": data["processing"].pop(options),
            }
        for item in data["stakeholders"]["participants"]:
            item["judgments"] = {key: str(value) for key, value in item["judgments"].items()}
        self.write(data)
        run = self.load()[0]
        self.assertEqual(run.id, "S01")
        self.assertEqual(run.processing.weighting_method.parameters, {"wd": "mean"})
        self.assertEqual(run.session.participants[0].responses[0].preference.ahp_scale, 1)

    def test_run_independence_and_serialization(self):
        _, maximum, minimum = self.load()
        maximum.session.criteria[0].name = "Changed"
        maximum.session.alternatives[0].criterion_values[0].value = 99
        maximum.session.stakeholder_groups[0].name = "Changed"
        maximum.processing.weighting_method.parameters["wd"] = "geometric"
        maximum.stakeholders.participant_variants[0].participants[0].judgments.clear()
        maximum.ai_evaluation["questions"].append("Changed")
        self.assertEqual(minimum.session.criteria[0].name, "Implementation Cost")
        self.assertEqual(minimum.session.alternatives[0].criterion_values[0].value, 8)
        self.assertEqual(minimum.session.stakeholder_groups[0].name, "Community Residents")
        self.assertEqual(minimum.processing.weighting_method.parameters, {"wd": "mean"})
        self.assertEqual(len(minimum.stakeholders.participant_variants[0].participants[0].judgments), 6)
        self.assertNotIn("Changed", minimum.ai_evaluation["questions"])
        plain = minimum.to_dict()
        self.assertEqual(json.loads(json.dumps(plain)), plain)
        self.assertEqual(plain["session"], minimum.session.model_dump(mode="json"))
        plain["ai_evaluation"]["questions"].clear()
        self.assertTrue(minimum.ai_evaluation["questions"])
        for factory, attribute in (
            (ExperimentStakeholders, "participants"),
            (lambda: WeightingMethodConfig(WeightingMethod.AHP), "parameters"),
            (lambda: RankingMethodConfig(RankingMethod.TOPSIS), "parameters"),
        ):
            self.assertIsNot(getattr(factory(), attribute), getattr(factory(), attribute))
        self.assertFalse(ExperimentExecution(123).variants)

    def test_invalid_participants_and_identities(self):
        original = self.read()
        cases = [
            ("Duplicate participant ID", lambda d: d["stakeholders"]["participant"].append(d["stakeholders"]["participant"][0])),
            ("Duplicate stakeholder group ID", lambda d: d["stakeholders"]["stakeholder_groups"].append(d["stakeholders"]["stakeholder_groups"][0])),
            ("unknown stakeholder group", lambda d: d["stakeholders"]["participant"][0].update(group_id="unknown")),
            ("unknown questions", lambda d: d["stakeholders"]["participant"][0]["judgments"].update(unknown=0)),
            ("invalid judgment", lambda d: d["stakeholders"]["participant"][0]["judgments"].update(q1=9)),
            ("invalid judgment", lambda d: d["stakeholders"]["participant"][0]["judgments"].update(q1=True)),
            ("nonempty participant set", lambda d: d["stakeholders"].update(participant=[])),
        ]
        for message, change in cases:
            with self.subTest(message=message):
                data = json.loads(json.dumps(original))
                change(data)
                self.write(data)
                with self.assertRaisesRegex(ValueError, message) as caught:
                    self.load()
                self.assertIn(S01, str(caught.exception))

    def test_invalid_variants(self):
        original = self.read(S02)
        for message, change in (
            ("Enabled variants", lambda d: d["stakeholders"].update(participant_variants=[])),
            ("Enabled variants", lambda d: d["stakeholders"]["participant_variants"][0].update(participants=[])),
            ("Duplicate participant variant ID", lambda d: d["stakeholders"]["participant_variants"].append(d["stakeholders"]["participant_variants"][0])),
            ("Duplicate participant ID", lambda d: d["stakeholders"]["participant_variants"][0]["participants"].append(d["stakeholders"]["participant_variants"][0]["participants"][0])),
        ):
            with self.subTest(message=message):
                data = json.loads(json.dumps(original))
                change(data)
                self.write(data, S02)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_dataset_failures_and_missing_file(self):
        dataset = self.root / "datasets" / "transport_core_v1.csv"
        original = dataset.read_text()
        for content, message in (
            (original.replace("accessibility_score", "missing"), "missing columns"),
            ("\n".join(original.splitlines()[:1] + original.splitlines()[2:]), "missing a row.*A1"),
            (original + original.splitlines()[1] + "\n", "Duplicate dataset alternative ID"),
            (original.replace("A1,8,", "A1,nan,"), "expected a finite number"),
        ):
            with self.subTest(message=message):
                dataset.write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
        data = self.read()
        data["resources"]["dataset"] = "missing.csv"
        self.write(data)
        with self.assertRaises(FileNotFoundError) as caught:
            self.load()
        self.assertIn("missing.csv", str(caught.exception))

    def test_resource_identity_failures(self):
        for relative, collection, message in (
            ("criteria/transport_v1.json", "criteria", "Duplicate criterion ID"),
            ("criteria/transport_v1.json", "pairwise_question_order", "Duplicate question ID"),
            ("alternative/transport_core_v1.json", "alternatives", "Duplicate alternative ID"),
        ):
            with self.subTest(collection=collection):
                path = self.root / relative
                original = path.read_text()
                data = json.loads(original)
                data[collection].append(data[collection][0])
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
                path.write_text(original)

    def test_base_execution_ignores_variants(self):
        data = self.read()
        data["stakeholders"]["participant_variants"] = self.read(S02)["stakeholders"]["participant_variants"]
        self.write(data)
        run = self.load()[0]
        self.assertIsNone(run.variant_id)
        self.assertEqual(run.session.participants[0].responses[0].preference.ahp_scale, 1)

    def test_runner_and_notebook(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_experiment.py"), "validate", "--experiments", str(self.directory)],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        self.assertIn("S02_MIN", result.stdout)
        notebook = json.loads((ROOT / "experiments.ipynb").read_text())
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("experiment1.session", sources)
        self.assertNotIn("experiment.model_dump", sources)
        from IPython.display import JSON
        JSON(self.load()[0].to_dict())


if __name__ == "__main__":
    unittest.main()
