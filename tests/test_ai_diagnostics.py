from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ai.diagnostics import build_rank_diagnostic_contexts
from src.experiments.results import RunResult
from src.mcdm.models.computation import EntityCalculation


class RankDiagnosticContextsTests(unittest.TestCase):
    def test_completed_winner_context_includes_session_only(self):
        result = RunResult(
            experiment_id="experiment",
            case_key="case",
            name="name",
            run_id="run",
            status="failed",
        )
        result.mcdm.session = EntityCalculation(
            entity_id="SESSION",
            scope="session",
            pairwise_matrix=[],
        )
        correlations = SimpleNamespace(
            rank_vectors=(SimpleNamespace(group_id="G1"),),
            model_dump=lambda **kwargs: {},
        )
        winners = SimpleNamespace(
            model_dump=lambda **kwargs: {},
        )

        with (
            patch(
                "src.ai.diagnostics.calculate_cross_group_rank_correlations",
                return_value=correlations,
            ),
            patch(
                "src.ai.diagnostics.calculate_winner_agreement",
                return_value=winners,
            ),
        ):
            rank_context, winner_context = (
                build_rank_diagnostic_contexts(result)
            )

        self.assertEqual(rank_context.entity_ids, ("G1",))
        self.assertEqual(
            winner_context.entity_ids,
            ("G1", "SESSION"),
        )


if __name__ == "__main__":
    unittest.main()
