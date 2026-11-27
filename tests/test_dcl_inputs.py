from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from src.ai.backend import BackendReceipt, StructuredGeneration
from src.ai.comprehension.context import resolve_evidence
from src.ai.comprehension.inputs import (
    build_group_profile_input,
    build_group_profile_inputs,
    build_session_profile_input,
)
from src.ai.comprehension.models import (
    ContentReview,
    DiagnosticContext,
    GroupProfileResponse,
    InterpretationDraft,
    SessionProfileResponse,
)
from src.ai.comprehension.orchestration import run_dcl_repetition_for_result
from src.ai.comprehension.prompts import (
    build_group_profile_messages,
)
from src.ai.comprehension.serialization import (
    hash_input_snapshot,
)
from src.ai.comprehension.validation import (
    validate_available_evidence,
    validate_response_evidence,
)
from src.ai.diagnostics import build_group_diagnostic_contexts, build_session_diagnostic_contexts
from src.ai.reporting.context import validate_reporting_available_evidence
from src.ai.reporting.inputs import build_data_explanation_reporting_input
from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_run


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticCitingBackend:
    def generate_structured(self, *, messages, response_model):
        snapshot = json.loads(messages[1]["content"].split("INPUT_SNAPSHOT:\n", 1)[1])

        def finding(object_type, field_path):
            reference = next(
                ref for ref in snapshot["available_evidence"]
                if ref["object_type"] == object_type and ref["field_path"] == field_path
            )
            return InterpretationDraft.model_validate({
                "statement": "Interpretation for the evidence handoff test.",
                "assessment": "supported",
                "evidence": [reference],
                "assumptions": [],
                "uncertainties": [],
                "caveats": [],
            }, strict=False)

        scenario = finding("scenario", ["decision_question"])
        if response_model is GroupProfileResponse:
            response = response_model(
                priorities=finding("calculation", ["weighting", "criterion_weights"]),
                ranking=finding("calculation", ["ranking", "rankings"]),
                representation=scenario,
                diagnostics=(finding("diagnostic", ["result"]),),
                limitations=(),
            )
        elif response_model is SessionProfileResponse:
            response = response_model(
                priorities=scenario, ranking=scenario, group_comparisons=(scenario,),
                agreement=(), influence=(), robustness=(), limitations=(scenario,),
            )
        else:
            response = response_model(
                decision_frame=scenario,
                **{name: (scenario,) for name in response_model.model_fields if name != "decision_frame"},
            )

        return StructuredGeneration(
            value=response,
            raw_text=response.model_dump_json(),
            receipt=BackendReceipt(requested_model="test", received_at=datetime.now(timezone.utc)),
        )


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

    def test_group_diagnostic_findings_only_carry_forward_with_their_evidence(self):
        group_diagnostics = build_group_diagnostic_contexts(self.result)
        session_diagnostics = build_session_diagnostic_contexts(self.result)
        outcome = run_dcl_repetition_for_result(
            backend=DiagnosticCitingBackend(),
            result=self.result,
            ai_repetition=1,
            max_retries=0,
            retry_base_seconds=1,
            retry_max_wait_seconds=1,
            diagnostics_by_group=group_diagnostics,
            session_diagnostics=session_diagnostics,
        )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.calls[-1].request.stage, "session_profile")
        snapshot = outcome.calls[-1].input_snapshot
        priors = {prior.finding_id: prior.draft for prior in snapshot.prior_interpretations}
        artifacts = (outcome.state.session_understanding, *outcome.state.group_profiles)
        for artifact in artifacts:
            for finding in artifact.findings:
                self.assertEqual(priors[finding.finding_id], finding.draft)
                for reference in finding.draft.evidence:
                    self.assertIn(reference, snapshot.available_evidence)
                    self.assertEqual(
                        resolve_evidence(snapshot, reference),
                        resolve_evidence(artifact.input_snapshot, reference),
                    )

        cited_diagnostics = tuple(diagnostics[0] for diagnostics in group_diagnostics.values())
        self.assertEqual(snapshot.diagnostics, (*session_diagnostics, *cited_diagnostics))
        self.assertEqual(type(snapshot).model_validate_json(snapshot.model_dump_json()), snapshot)
        incomplete = snapshot.model_copy(update={
            "available_evidence": tuple(
                ref for ref in snapshot.available_evidence
                if ref.object_id != cited_diagnostics[0].diagnostic_id
            ),
        })
        with self.assertRaisesRegex(ValidationError, "cites evidence unavailable"):
            type(snapshot).model_validate_json(incomplete.model_dump_json())

        reporting = build_data_explanation_reporting_input(result=self.result, state=outcome.state)
        self.assertEqual(validate_reporting_available_evidence(reporting), ())

        state = outcome.state.model_copy(update={"session_profile": None})
        rebuild = build_session_profile_input(
            result=self.result, state=state,
            diagnostics=(*session_diagnostics, cited_diagnostics[0]),
        )
        self.assertEqual(rebuild, snapshot)

        conflict = cited_diagnostics[0].model_copy(update={"method": "different-method"})
        with self.assertRaisesRegex(ValueError, "Conflicting diagnostic"):
            build_session_profile_input(result=self.result, state=state, diagnostics=(conflict,))

        first_group, *other_groups = state.group_profiles
        rejected_group = first_group.model_copy(update={
            "content_review": ContentReview(status="rejected"),
        })
        rejected_state = state.model_copy(update={"group_profiles": (rejected_group, *other_groups)})
        rebuild = build_session_profile_input(
            result=self.result, state=rejected_state, diagnostics=session_diagnostics,
        )
        self.assertEqual(rebuild.diagnostics, (*session_diagnostics, *cited_diagnostics[1:]))
        self.assertFalse(any(
            prior.source_request_id == first_group.request.request_id
            for prior in rebuild.prior_interpretations
        ))

        group_call = next(call for call in outcome.calls if call.request.group_id is not None)
        response = group_call.parsed_response
        diagnostic_finding = response.diagnostics[0]
        invalid_reference = diagnostic_finding.evidence[0].model_copy(
            update={"object_id": "unknown-diagnostic"}
        )
        invalid_response = response.model_copy(update={
            "diagnostics": (diagnostic_finding.model_copy(update={"evidence": (invalid_reference,)}),),
        })
        validation = validate_response_evidence(group_call.input_snapshot, invalid_response)
        self.assertEqual([issue.code for issue in validation.issues], ["evidence_not_offered"])

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
            kind="agreement",
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
            kind="agreement",
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
