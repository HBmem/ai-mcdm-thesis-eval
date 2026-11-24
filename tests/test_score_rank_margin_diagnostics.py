import unittest

from src.mcdm.diagnostics import _calculate_entity_score_rank_margins
from src.mcdm.models.computation import (
    AlternativeRanking,
    EntityCalculation,
    RankingResult,
)


class ScoreRankMarginDiagnosticsTests(unittest.TestCase):
    def test_adjacent_margins_use_raw_score_difference_for_ties(self):
        calculation = EntityCalculation(
            entity_id="G1",
            scope="group",
            pairwise_matrix=[],
            ranking=RankingResult(rankings=[
                AlternativeRanking(
                    alternative_id="B",
                    alternative_name="B",
                    rank=2,
                    closeness_coefficient=0.8999999999995,
                ),
                AlternativeRanking(
                    alternative_id="A",
                    alternative_name="A",
                    rank=1,
                    closeness_coefficient=0.9,
                ),
                AlternativeRanking(
                    alternative_id="C",
                    alternative_name="C",
                    rank=3,
                    closeness_coefficient=0.7,
                ),
            ]),
        )

        result = _calculate_entity_score_rank_margins(
            calculation,
            alternative_ids=("C", "B", "A"),
            tie_atol=1e-12,
        )

        self.assertEqual(
            tuple(entry.alternative_id for entry in result.entries),
            ("A", "B", "C"),
        )
        first, second = result.adjacent_margins
        self.assertAlmostEqual(first.score_margin, 5e-13)
        self.assertTrue(first.tied)
        self.assertEqual(first.statistical_rank_margin, 0.0)
        self.assertFalse(second.tied)
        self.assertGreater(second.statistical_rank_margin, 0.0)
        self.assertEqual(result.winner_ids, ("A", "B"))
        self.assertEqual(result.top_score_margin, first.score_margin)
        self.assertEqual(
            result.smallest_positive_score_margin,
            second.score_margin,
        )


if __name__ == "__main__":
    unittest.main()
