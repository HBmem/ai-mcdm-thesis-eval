from copy import deepcopy
from pathlib import Path
import unittest

from src.ai.comprehension.inputs import (
    build_group_profile_input,
    build_group_profile_inputs,
)
from src.ai.comprehension.models import DiagnosticContext
from src.ai.comprehension.prompts import (
    build_group_profile_messages,
)
from src.ai.comprehension.serialization import (
    hash_input_snapshot,
)
from src.ai.comprehension.validation import (
    validate_available_evidence,
)
from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_run


ROOT = Path(__file__).resolve().parents[1]


class GroupProfileInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        experiments = load_experiments(
            ROOT / "experiments"
        )

        runs = prepare_runs(
            experiments,
            "S00",
        )

        cls.result = execute_run(runs[0])

        if (
            cls.result.execution.get("numerical_status")
            != "completed"
        ):
            raise RuntimeError(
                "The S00 numerical fixture did not complete."
            )

    def test_real_result_projects_all_groups_in_order(self):
        profiles = build_group_profile_inputs(
            result=self.result,
        )

        expected_group_ids = [
            group["id"]
            for group in self.result.inputs[
                "session"
            ]["stakeholder_groups"]
        ]

        self.assertEqual(len(expected_group_ids), 5)
        self.assertEqual(len(profiles), 5)

        self.assertEqual(
            [profile.group.id for profile in profiles],
            expected_group_ids,
        )

        for profile in profiles:
            with self.subTest(group=profile.group.id):
                self.assertIsNotNone(profile.calculation)

                self.assertEqual(
                    profile.calculation.entity_id,
                    profile.group.id,
                )

                self.assertEqual(
                    validate_available_evidence(profile),
                    (),
                )

                messages = build_group_profile_messages(
                    profile
                )

                self.assertEqual(len(messages), 2)
                self.assertEqual(
                    [message.role for message in messages],
                    ["system", "user"],
                )

    def test_projection_hashes_are_stable(self):
        first = build_group_profile_inputs(
            result=self.result,
        )

        second = build_group_profile_inputs(
            result=self.result,
        )

        first_hashes = [
            hash_input_snapshot(profile)
            for profile in first
        ]

        second_hashes = [
            hash_input_snapshot(profile)
            for profile in second
        ]

        self.assertEqual(first_hashes, second_hashes)

        for fingerprint in first_hashes:
            self.assertRegex(
                fingerprint,
                r"^[0-9a-f]{64}$",
            )

    def test_unknown_calculation_group_is_rejected(self):
        result = deepcopy(self.result)

        unexpected = deepcopy(result.mcdm.groups[0])
        unexpected.entity_id = "unknown_group"

        result.mcdm.groups.append(unexpected)

        target_group = result.inputs[
            "session"
        ]["stakeholder_groups"][0]["id"]

        with self.assertRaisesRegex(
            ValueError,
            "calculations reference unknown",
        ):
            build_group_profile_input(
                result=result,
                group_id=target_group,
            )

    def test_duplicate_diagnostic_ids_are_rejected(self):
        target_group = self.result.inputs[
            "session"
        ]["stakeholder_groups"][0]["id"]

        diagnostic = DiagnosticContext(
            diagnostic_id="agreement",
            method="synthetic-test",
            entity_ids=(target_group,),
            status="not_run",
            result=None,
            limitations=("Test diagnostic.",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "unique identifiers",
        ):
            build_group_profile_input(
                result=self.result,
                group_id=target_group,
                diagnostics=(
                    diagnostic,
                    diagnostic,
                ),
            )

    def test_diagnostic_for_another_group_is_rejected(self):
        group_ids = [
            group["id"]
            for group in self.result.inputs[
                "session"
            ]["stakeholder_groups"]
        ]

        target_group = group_ids[0]
        other_group = group_ids[1]

        diagnostic = DiagnosticContext(
            diagnostic_id="agreement",
            method="synthetic-test",
            entity_ids=(other_group,),
            status="not_run",
            result=None,
            limitations=("Test diagnostic.",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "must cover the target group",
        ):
            build_group_profile_input(
                result=self.result,
                group_id=target_group,
                diagnostics=(diagnostic,),
            )
            