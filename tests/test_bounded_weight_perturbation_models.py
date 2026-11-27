import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    BoundedWeightPerturbationCase,
    BoundedWeightPerturbationResult,
)


def rank_entry(alternative_id, score, statistical_rank):
    return {
        "alternative_id": alternative_id,
        "score": score,
        "statistical_rank": statistical_rank,
    }


def perturbation_case(criterion_id, requested_delta):
    return {
        "criterion_id": criterion_id,
        "requested_delta": requested_delta,
        "status": "completed",
        "applied_weights": {"C1": 0.4, "C2": 0.6},
        "entries": (
            rank_entry("A", 0.7, 1.0),
            rank_entry("B", 0.3, 2.0),
        ),
        "winner_ids": ("A",),
        "winner_changed": False,
    }


def result_data():
    return {
        "session_id": "SESSION",
        "absolute_weight_delta": 0.1,
        "tie_atol": 1e-12,
        "ranking_criterion_ids": ("C1", "C2"),
        "baseline_applied_weights": {"C1": 0.4, "C2": 0.6},
        "baseline_entries": (
            rank_entry("A", 0.7, 1.0),
            rank_entry("B", 0.3, 2.0),
        ),
        "baseline_winner_ids": ("A",),
        "cases": (
            perturbation_case("C1", -0.1),
            perturbation_case("C1", 0.1),
            perturbation_case("C2", -0.1),
            perturbation_case("C2", 0.1),
        ),
    }


class BoundedWeightPerturbationModelTests(unittest.TestCase):
    def test_unavailable_case_requires_reason_and_empty_outcomes(self):
        fields = {
            "criterion_id": "C1",
            "requested_delta": -0.1,
            "status": "not_evaluable",
            "reason": "The requested weight is out of bounds.",
        }
        case = BoundedWeightPerturbationCase.model_validate(fields)
        self.assertEqual(case.status, "not_evaluable")

        invalid_cases = (
            {"reason": "   "},
            {"reason": None},
            {"applied_weights": {"C1": 1.0}},
            {"entries": (rank_entry("A", 1.0, 1.0),)},
            {"winner_ids": ("A",)},
            {"winner_changed": False},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    BoundedWeightPerturbationCase.model_validate({
                        **fields,
                        **changes,
                    })

    def test_completed_case_requires_normalized_outcomes(self):
        fields = perturbation_case("C1", -0.1)
        self.assertEqual(
            BoundedWeightPerturbationCase.model_validate(fields).status,
            "completed",
        )

        for changes in (
            {"reason": "unexpected"},
            {"applied_weights": {"C1": 0.4, "C2": 0.5}},
            {"winner_ids": ("unknown",)},
            {"winner_ids": ("A", "A")},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    BoundedWeightPerturbationCase.model_validate({
                        **fields,
                        **changes,
                    })

    def test_result_accepts_two_ordered_cases_per_criterion(self):
        fields = result_data()
        fields["cases"] = (
            {
                "criterion_id": "C1",
                "requested_delta": -0.1,
                "status": "not_evaluable",
                "reason": "The requested weight is out of bounds.",
            },
            *fields["cases"][1:],
        )
        result = BoundedWeightPerturbationResult.model_validate(
            fields
        )
        self.assertEqual(len(result.cases), 4)

    def test_result_validates_cases_against_baseline(self):
        invalid_results = []

        wrong_weights = result_data()
        wrong_weights["cases"] = (
            {
                **wrong_weights["cases"][0],
                "applied_weights": {"C1": 0.5, "C3": 0.5},
            },
            *wrong_weights["cases"][1:],
        )
        invalid_results.append(wrong_weights)

        wrong_alternative_order = result_data()
        wrong_alternative_order["cases"] = (
            {
                **wrong_alternative_order["cases"][0],
                "entries": tuple(reversed(
                    wrong_alternative_order["cases"][0]["entries"]
                )),
            },
            *wrong_alternative_order["cases"][1:],
        )
        invalid_results.append(wrong_alternative_order)

        wrong_winner_changed = result_data()
        wrong_winner_changed["cases"] = (
            {
                **wrong_winner_changed["cases"][0],
                "winner_changed": True,
            },
            *wrong_winner_changed["cases"][1:],
        )
        invalid_results.append(wrong_winner_changed)

        for fields in invalid_results:
            with self.subTest(fields=fields):
                with self.assertRaises(ValidationError):
                    BoundedWeightPerturbationResult.model_validate(fields)

    def test_result_requires_negative_then_positive_cases_per_criterion(self):
        fields = result_data()
        fields["cases"] = tuple(reversed(fields["cases"]))
        with self.assertRaises(ValidationError):
            BoundedWeightPerturbationResult.model_validate(fields)

        fields = result_data()
        fields["cases"] = fields["cases"][:-1]
        with self.assertRaises(ValidationError):
            BoundedWeightPerturbationResult.model_validate(fields)


if __name__ == "__main__":
    unittest.main()
