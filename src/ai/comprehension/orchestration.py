from dataclasses import dataclass
from typing import Literal

from src.ai.backend import StructuredAIBackend
from src.ai.comprehension.inputs import build_group_profile_inputs
from src.ai.comprehension.models import DCLCallRecord, DCLRunState, DiagnosticContext
from src.ai.comprehension.pipeline import build_group_profile_artifact, run_group_profile_stage
from src.ai.comprehension.requests import create_group_profile_request
from src.experiments.results import RunResult


@dataclass(frozen=True, slots=True)
class GroupProfilesOutcome:
    state: DCLRunState
    calls: tuple[DCLCallRecord, ...]
    failed_group_ids: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failed_group_ids

    def __post_init__(self) -> None:
        artifact_group_ids = {
            artifact.request.group_id
            for artifact in self.state.group_profiles
        }

        failed = set(self.failed_group_ids)

        if artifact_group_ids & failed:
            raise ValueError("A group cannot be both successful and failed.")

        if len(self.failed_group_ids) != len(failed):
            raise ValueError("Failed group identifiers must be unique.")

        if any(
            call.request.run_id != self.state.run_id
            or call.request.ai_repetition
            != self.state.ai_repetition
            for call in self.calls
        ):
            raise ValueError("Group-profile calls cannot mix runs or AI repetitions.")

        calls_by_group: dict[str, list[DCLCallRecord]] = {}

        for call in self.calls:
            group_id = call.request.group_id

            if group_id is None:
                raise ValueError( "Group-profile calls require a group identifier.")

            calls_by_group.setdefault(
                group_id,
                [],
            ).append(call)

        expected_group_ids = (
            artifact_group_ids
            | failed
        )

        if set(calls_by_group) != expected_group_ids:
            raise ValueError("Group-profile calls must exactly cover successful and failed groups.")

        for group_id, group_calls in calls_by_group.items():
            attempts = [
                call.attempt
                for call in group_calls
            ]

            expected_attempts = list(
                range(1, len(group_calls) + 1)
            )

            if attempts != expected_attempts:
                raise ValueError("Group-profile call attempts must be consecutive and ordered.")

        for group_id in artifact_group_ids:
            if calls_by_group[group_id][-1].status != "completed":
                raise ValueError("Successful groups must end with a completed call.")

        for group_id in failed:
            if calls_by_group[group_id][-1].status != "failed":
                raise ValueError("Failed groups must end with a failed call.")


def run_group_profiles_for_result(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    ai_repetition: int,
    max_retries: int,
    diagnostics_by_group: dict[
        str,
        tuple[DiagnosticContext, ...],
    ] | None = None,
    required_limitations: tuple[str, ...] = (),
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> GroupProfilesOutcome:
    if ai_repetition < 1:
        raise ValueError(
            "AI repetition must be at least one."
        )

    if max_retries < 0:
        raise ValueError(
            "Retries must not be negative."
        )

    inputs = build_group_profile_inputs(
        result=result,
        diagnostics_by_group=diagnostics_by_group,
        required_limitations=required_limitations,
    )

    artifacts = []
    calls = []
    failed_group_ids = []

    for input_snapshot in inputs:
        request = create_group_profile_request(
            run_id=result.run_id,
            ai_repetition=ai_repetition,
            input_snapshot=input_snapshot,
            input_origin=input_origin,
        )

        outcome = run_group_profile_stage(
            backend=backend,
            request=request,
            input_snapshot=input_snapshot,
            max_retries=max_retries,
        )

        calls.extend(outcome.calls)

        if not outcome.succeeded:
            failed_group_ids.append(
                input_snapshot.group.id
            )
            continue

        artifact = build_group_profile_artifact(
            request=request,
            input_snapshot=input_snapshot,
            outcome=outcome,
        )

        artifacts.append(artifact)

    state = DCLRunState(
        run_id=result.run_id,
        ai_repetition=ai_repetition,
        group_profiles=tuple(artifacts),
    )

    return GroupProfilesOutcome(
        state=state,
        calls=tuple(calls),
        failed_group_ids=tuple(failed_group_ids),
    )
