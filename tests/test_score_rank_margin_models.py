import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    AdjacentScoreRankMargin,
    DiagnosticRankEntry,
    EntityScoreRankMargins,
    ScoreRankMarginResult,
)


class ScoreRankMarginModelTests(unittest.TestCase):
    def setUp(self):
        self.entries = (
            DiagnosticRankEntry(
                alternative_id="A",
                score=0.9,
                statistical_rank=1.0,
            ),
            DiagnosticRankEntry(
                alternative_id="B",
                score=0.8,
                statistical_rank=2.5,
            ),
            DiagnosticRankEntry(
                alternative_id="C",
                score=0.8,
                statistical_rank=2.5,
            ),
        )
        self.entity = EntityScoreRankMargins(
            entity_id="G1",
            scope="group",
            tie_atol=0.001,
            entries=self.entries,
            adjacent_margins=(
                AdjacentScoreRankMargin(
                    higher_alternative_id="A",
                    lower_alternative_id="B",
                    higher_score=0.9,
                    lower_score=0.8,
                    score_margin=0.9 - 0.8,
                    higher_statistical_rank=1.0,
                    lower_statistical_rank=2.5,
                    statistical_rank_margin=1.5,
                    tied=False,
                ),
                AdjacentScoreRankMargin(
                    higher_alternative_id="B",
                    lower_alternative_id="C",
                    higher_score=0.8,
                    lower_score=0.8,
                    score_margin=0.0,
                    higher_statistical_rank=2.5,
                    lower_statistical_rank=2.5,
                    statistical_rank_margin=0.0,
                    tied=True,
                ),
            ),
            winner_ids=("A",),
            top_score_margin=0.9 - 0.8,
            top_statistical_rank_margin=1.5,
            smallest_positive_score_margin=0.9 - 0.8,
        )

    def _assert_invalid_entity(self, **changes):
        payload = self.entity.model_dump()
        payload.update(changes)
        with self.assertRaises(ValidationError):
            EntityScoreRankMargins.model_validate(payload)

    def test_entity_accepts_consistent_rank_margins(self):
        self.assertEqual(
            EntityScoreRankMargins.model_validate(
                self.entity.model_dump()
            ),
            self.entity,
        )

    def test_entity_rejects_duplicate_or_unsorted_entries(self):
        duplicate_entries = list(self.entity.entries)
        duplicate_entries[2] = duplicate_entries[2].model_copy(
            update={"alternative_id": "B"}
        )
        self._assert_invalid_entity(entries=tuple(duplicate_entries))
        self._assert_invalid_entity(
            entries=tuple(reversed(self.entity.entries))
        )

    def test_entity_rejects_invalid_adjacent_records(self):
        self._assert_invalid_entity(
            adjacent_margins=self.entity.adjacent_margins[:1]
        )
        margins = list(self.entity.adjacent_margins)
        margins[0] = margins[0].model_copy(
            update={"higher_alternative_id": "C"}
        )
        self._assert_invalid_entity(adjacent_margins=tuple(margins))
        margins = list(self.entity.adjacent_margins)
        margins[1] = margins[1].model_copy(update={"tied": False})
        self._assert_invalid_entity(adjacent_margins=tuple(margins))

    def test_entity_rejects_incorrect_winners_and_summaries(self):
        self._assert_invalid_entity(winner_ids=("B",))
        self._assert_invalid_entity(top_score_margin=0.2)
        self._assert_invalid_entity(top_statistical_rank_margin=1.0)
        self._assert_invalid_entity(smallest_positive_score_margin=None)

    def test_result_requires_unique_groups_and_ordered_entity_records(self):
        group_two = self.entity.model_copy(update={"entity_id": "G2"})
        session = self.entity.model_copy(
            update={"entity_id": "SESSION", "scope": "session"}
        )
        result_fields = dict(
            tie_atol=0.001,
            evaluated_group_ids=("G1", "G2"),
            session_id="SESSION",
            entities=(self.entity, group_two, session),
        )
        result = ScoreRankMarginResult(**result_fields)
        self.assertEqual(
            ScoreRankMarginResult.model_validate(result.model_dump()),
            result,
        )

        with self.assertRaises(ValidationError):
            ScoreRankMarginResult(**{
                **result_fields,
                "evaluated_group_ids": ("G1", "G1"),
            })
        with self.assertRaises(ValidationError):
            ScoreRankMarginResult(**{
                **result_fields,
                "entities": (group_two, self.entity, session),
            })
        with self.assertRaises(ValidationError):
            ScoreRankMarginResult(**{
                **result_fields,
                "entities": (self.entity, self.entity, session),
            })
        with self.assertRaises(ValidationError):
            ScoreRankMarginResult(**{
                **result_fields,
                "entities": (self.entity, group_two),
            })
        with self.assertRaises(ValidationError):
            ScoreRankMarginResult(**{
                **result_fields,
                "entities": (
                    self.entity,
                    group_two,
                    session.model_copy(update={"tie_atol": 0.002}),
                ),
            })


if __name__ == "__main__":
    unittest.main()
