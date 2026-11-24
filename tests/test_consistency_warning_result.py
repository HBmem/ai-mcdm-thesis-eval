import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    ConsistencyWarningResult,
)


class ConsistencyWarningResultTests(unittest.TestCase):
    def setUp(self):
        self.result = ConsistencyWarningResult(
            evaluated_group_ids=("G1", "G2"),
            session_id="SESSION",
            entity_assessments=(
                {
                    "entity_id": "G1",
                    "scope": "group",
                    "consistency_ratio": 0.05,
                    "consistency_threshold": 0.1,
                    "excess_amount": 0.0,
                    "status": "within_threshold",
                },
                {
                    "entity_id": "G2",
                    "scope": "group",
                    "consistency_ratio": 0.2,
                    "consistency_threshold": 0.1,
                    "excess_amount": 0.1,
                    "status": "exceeds_threshold",
                },
                {
                    "entity_id": "SESSION",
                    "scope": "session",
                    "consistency_ratio": 0.3,
                    "consistency_threshold": 0.1,
                    "excess_amount": 0.2,
                    "status": "exceeds_threshold",
                },
            ),
            participant_group_summaries=(
                {
                    "group_id": "G1",
                    "participant_count": 1,
                    "evaluated_count": 1,
                    "unavailable_count": 0,
                    "exceeding_count": 1,
                    "exceeding_fraction_of_evaluated": 1.0,
                    "maximum_consistency_ratio": 0.15,
                },
                {
                    "group_id": "G2",
                    "participant_count": 1,
                    "evaluated_count": 1,
                    "unavailable_count": 0,
                    "exceeding_count": 0,
                    "exceeding_fraction_of_evaluated": 0.0,
                    "maximum_consistency_ratio": 0.05,
                },
            ),
            group_or_session_warning_ids=("G2", "SESSION"),
            participant_warning_group_ids=("G1",),
        )

    def _assert_invalid(self, **changes):
        fields = self.result.model_dump()
        fields.update(changes)
        with self.assertRaises(ValidationError):
            ConsistencyWarningResult.model_validate(fields)

    def test_result_accepts_ordered_assessments_and_warnings(self):
        self.assertEqual(
            self.result.group_or_session_warning_ids,
            ("G2", "SESSION"),
        )
        self.assertEqual(
            self.result.participant_warning_group_ids,
            ("G1",),
        )

    def test_result_requires_unique_evaluated_groups(self):
        self._assert_invalid(
            evaluated_group_ids=("G1", "G1"),
        )

    def test_result_requires_ordered_assessment_entities_and_scopes(self):
        self._assert_invalid(
            entity_assessments=tuple(
                reversed(self.result.entity_assessments)
            ),
        )
        assessments = list(self.result.entity_assessments)
        assessments[0] = assessments[0].model_copy(
            update={"scope": "session"}
        )
        self._assert_invalid(entity_assessments=tuple(assessments))

    def test_result_requires_ordered_participant_summaries(self):
        self._assert_invalid(
            participant_group_summaries=tuple(
                reversed(self.result.participant_group_summaries)
            ),
        )

    def test_result_requires_warning_ids_to_match_derived_warnings(self):
        self._assert_invalid(
            group_or_session_warning_ids=("SESSION", "G2"),
        )
        self._assert_invalid(
            participant_warning_group_ids=("G2",),
        )

    def test_warning_id_lists_must_not_contain_duplicates(self):
        fields = self.result.model_dump()
        fields["session_id"] = "G2"
        fields["entity_assessments"] = (
            *fields["entity_assessments"][:2],
            {
                **fields["entity_assessments"][2],
                "entity_id": "G2",
            },
        )
        fields["group_or_session_warning_ids"] = ("G2", "G2")

        with self.assertRaises(ValidationError):
            ConsistencyWarningResult.model_validate(fields)


if __name__ == "__main__":
    unittest.main()
