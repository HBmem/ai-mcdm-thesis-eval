import unittest

from pydantic import ValidationError

from src.mcdm.models.diagnostics import (
    EffectiveGroupPower,
    LeaveOneGroupOutCase,
    LeaveOneGroupOutResult,
)


def completed_case(
    omitted_group_id="G1",
    remaining_group_ids=("G2",),
):
    remaining_power = EffectiveGroupPower(
        group_id=remaining_group_ids[0],
        effective_voting_power=1.0,
    )
    return LeaveOneGroupOutCase(
        omitted_group_id=omitted_group_id,
        remaining_group_ids=remaining_group_ids,
        status="completed",
        effective_voting_powers=(remaining_power,),
        baseline_winner_ids=("A",),
        recomputed_winner_ids=("B",),
        winner_changed=True,
        criterion_weight_changes=(
            {
                "criterion_id": "C1",
                "baseline_weight": 0.4,
                "recomputed_weight": 0.5,
                "weight_delta": 0.1,
            },
        ),
        alternative_changes=(
            {
                "alternative_id": "A",
                "baseline_score": 0.7,
                "recomputed_score": 0.6,
                "score_delta": -0.1,
                "baseline_statistical_rank": 1.0,
                "recomputed_statistical_rank": 2.0,
                "statistical_rank_delta": 1.0,
            },
        ),
        maximum_absolute_weight_delta=0.1,
        maximum_absolute_score_delta=0.1,
        maximum_absolute_rank_delta=1.0,
        recomputed_consistency_ratio=0.02,
    )


def not_evaluable_case():
    return LeaveOneGroupOutCase(
        omitted_group_id="G1",
        remaining_group_ids=("G2",),
        status="not_evaluable",
        not_evaluable_reason="No represented groups remain.",
        baseline_winner_ids=("A",),
    )


class LeaveOneGroupOutModelTests(unittest.TestCase):
    def setUp(self):
        self.case = completed_case()
        self.result = LeaveOneGroupOutResult(
            tie_atol=1e-12,
            evaluated_group_ids=("G1", "G2"),
            session_id="SESSION",
            baseline_winner_ids=("A",),
            cases=(
                self.case,
                completed_case("G2", ("G1",)),
            ),
        )

    def _assert_invalid_case(self, **changes):
        fields = self.case.model_dump()
        fields.update(changes)
        with self.assertRaises(ValidationError):
            LeaveOneGroupOutCase.model_validate(fields)

    def test_change_records_validate_their_deltas(self):
        fields = self.case.model_dump()
        criterion_change = dict(
            fields["criterion_weight_changes"][0]
        )
        criterion_change["weight_delta"] = 0.2
        with self.assertRaises(ValidationError):
            LeaveOneGroupOutCase.model_validate({
                **fields,
                "criterion_weight_changes": (criterion_change,),
            })

        alternative_change = dict(
            fields["alternative_changes"][0]
        )
        for field, value in (
            ("score_delta", -0.2),
            ("statistical_rank_delta", -1.0),
        ):
            changed = dict(alternative_change)
            changed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    LeaveOneGroupOutCase.model_validate({
                        **fields,
                        "alternative_changes": (changed,),
                    })

    def test_completed_case_accepts_consistent_aggregates(self):
        self.assertEqual(self.case.status, "completed")

    def test_completed_case_validates_reason_power_winners_and_deltas(self):
        invalid_cases = (
            {"not_evaluable_reason": "unexpected"},
            {
                "effective_voting_powers": (
                    {"group_id": "G1", "effective_voting_power": 1.0},
                ),
            },
            {
                "effective_voting_powers": (
                    {"group_id": "G2", "effective_voting_power": 0.9},
                ),
            },
            {"baseline_winner_ids": ()},
            {"recomputed_winner_ids": ()},
            {"winner_changed": False},
            {"criterion_weight_changes": ()},
            {"alternative_changes": ()},
            {"maximum_absolute_weight_delta": 0.2},
            {"maximum_absolute_score_delta": 0.2},
            {"maximum_absolute_rank_delta": 2.0},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                self._assert_invalid_case(**changes)

    def test_not_evaluable_case_allows_baseline_winners_only(self):
        case = not_evaluable_case()
        self.assertEqual(case.baseline_winner_ids, ("A",))
        self.assertEqual(case.recomputed_winner_ids, ())

    def test_not_evaluable_case_rejects_recomputed_results(self):
        fields = not_evaluable_case().model_dump()
        invalid_cases = (
            {"not_evaluable_reason": None},
            {"not_evaluable_reason": ""},
            {"not_evaluable_reason": "   "},
            {"winner_changed": False},
            {"recomputed_winner_ids": ("A",)},
            {
                "criterion_weight_changes": (
                    self.case.criterion_weight_changes[0].model_dump(),
                ),
            },
            {
                "alternative_changes": (
                    self.case.alternative_changes[0].model_dump(),
                ),
            },
            {"maximum_absolute_weight_delta": 0.1},
            {"maximum_absolute_score_delta": 0.1},
            {"maximum_absolute_rank_delta": 1.0},
            {"recomputed_consistency_ratio": 0.02},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    LeaveOneGroupOutCase.model_validate({
                        **fields,
                        **changes,
                    })

    def test_result_accepts_one_ordered_case_per_group(self):
        self.assertEqual(
            tuple(case.omitted_group_id for case in self.result.cases),
            ("G1", "G2"),
        )

    def test_result_validates_unique_groups_case_order_and_remaining_groups(
        self,
    ):
        fields = self.result.model_dump()
        invalid_results = (
            {"evaluated_group_ids": ("G1", "G1")},
            {"cases": (fields["cases"][0],)},
            {
                "cases": tuple(reversed(fields["cases"])),
            },
            {
                "cases": (
                    {
                        **fields["cases"][0],
                        "omitted_group_id": "G2",
                    },
                    fields["cases"][1],
                ),
            },
            {
                "cases": (
                    {
                        **fields["cases"][0],
                        "remaining_group_ids": ("G1",),
                        "effective_voting_powers": ({
                            "group_id": "G1",
                            "effective_voting_power": 1.0,
                        },),
                    },
                    fields["cases"][1],
                ),
            },
        )
        for changes in invalid_results:
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    LeaveOneGroupOutResult.model_validate({
                        **fields,
                        **changes,
                    })


if __name__ == "__main__":
    unittest.main()
