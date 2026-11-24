import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    GroupParticipantConsistencySummary,
)


class GroupParticipantConsistencySummaryTests(unittest.TestCase):
    def test_summary_without_evaluated_participants_has_no_statistics(self):
        summary = GroupParticipantConsistencySummary(
            group_id="G1",
            participant_count=2,
            evaluated_count=0,
            unavailable_count=2,
            exceeding_count=0,
        )

        self.assertIsNone(summary.exceeding_fraction_of_evaluated)
        self.assertIsNone(summary.maximum_consistency_ratio)

    def test_summary_validates_counts_and_evaluated_statistics(self):
        summary = GroupParticipantConsistencySummary(
            group_id="G1",
            participant_count=3,
            evaluated_count=2,
            unavailable_count=1,
            exceeding_count=1,
            exceeding_fraction_of_evaluated=0.5,
            maximum_consistency_ratio=0.2,
        )

        self.assertEqual(summary.evaluated_count, 2)

        invalid_summaries = (
            dict(
                participant_count=3,
                evaluated_count=1,
                unavailable_count=1,
                exceeding_count=0,
                exceeding_fraction_of_evaluated=0.0,
                maximum_consistency_ratio=0.2,
            ),
            dict(
                participant_count=1,
                evaluated_count=1,
                unavailable_count=0,
                exceeding_count=2,
                exceeding_fraction_of_evaluated=2.0,
                maximum_consistency_ratio=0.2,
            ),
            dict(
                participant_count=1,
                evaluated_count=1,
                unavailable_count=0,
                exceeding_count=1,
                exceeding_fraction_of_evaluated=0.0,
                maximum_consistency_ratio=0.2,
            ),
            dict(
                participant_count=1,
                evaluated_count=1,
                unavailable_count=0,
                exceeding_count=1,
                exceeding_fraction_of_evaluated=1.0,
            ),
            dict(
                participant_count=1,
                evaluated_count=0,
                unavailable_count=1,
                exceeding_count=0,
                exceeding_fraction_of_evaluated=0.0,
            ),
            dict(
                participant_count=1,
                evaluated_count=0,
                unavailable_count=1,
                exceeding_count=0,
                maximum_consistency_ratio=0.2,
            ),
        )
        for fields in invalid_summaries:
            with self.subTest(fields=fields):
                with self.assertRaises(ValidationError):
                    GroupParticipantConsistencySummary(
                        group_id="G1",
                        **fields,
                    )


if __name__ == "__main__":
    unittest.main()
