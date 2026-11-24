import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    EntityConsistencyAssessment,
)


class EntityConsistencyAssessmentTests(unittest.TestCase):
    def test_not_available_allows_threshold_without_ratio(self):
        assessment = EntityConsistencyAssessment(
            entity_id="G1",
            scope="group",
            consistency_threshold=0.1,
            status="not_available",
        )

        self.assertEqual(assessment.consistency_threshold, 0.1)

    def test_not_available_rejects_ratio_or_excess(self):
        for changes in (
            {"consistency_ratio": 0.05},
            {"excess_amount": 0.0},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    EntityConsistencyAssessment(
                        entity_id="G1",
                        scope="group",
                        status="not_available",
                        **changes,
                    )

    def test_threshold_equality_is_within_threshold(self):
        assessment = EntityConsistencyAssessment(
            entity_id="G1",
            scope="group",
            consistency_ratio=0.1,
            consistency_threshold=0.1,
            excess_amount=0.0,
            status="within_threshold",
        )

        self.assertEqual(assessment.status, "within_threshold")

    def test_evaluated_assessment_requires_matching_values(self):
        valid = dict(
            entity_id="G1",
            scope="group",
            consistency_ratio=0.15,
            consistency_threshold=0.1,
            excess_amount=0.05,
            status="exceeds_threshold",
        )
        EntityConsistencyAssessment(**valid)

        for changes in (
            {"status": "within_threshold"},
            {"excess_amount": 0.04},
            {"consistency_ratio": None},
            {"consistency_threshold": None},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    EntityConsistencyAssessment(
                        **{**valid, **changes}
                    )


if __name__ == "__main__":
    unittest.main()
