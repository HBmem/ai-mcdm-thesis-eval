from dataclasses import dataclass
from typing import Literal

from src.ai.backend import StructuredAIBackend
from src.ai.comprehension.inputs import (
    build_session_profile_input,
    build_session_understanding_input,
    build_group_profile_inputs,
)
from src.ai.comprehension.models import DCLCallRecord, DCLRunState, DiagnosticContext
from src.ai.comprehension.pipeline import (
    build_session_profile_artifact,
    build_session_understanding_artifact,
    run_session_profile_stage,
    run_session_understanding_stage,
    build_group_profile_artifact,
    run_group_profile_stage,
)
from src.ai.comprehension.requests import create_session_profile_request, create_session_understanding_request, create_group_profile_request
from src.experiments.results import RunResult


SessionUnderstandingStatus = Literal[
    "completed",
    "failed",
]

GroupProfilesStatus = Literal[
    "completed",
    "partial",
    "failed",
    "blocked",
]

SessionProfileStatus = Literal[
    "completed",
    "failed",
    "blocked",
]

DCLRepetitionStatus = Literal[
    "completed",
    "partial",
    "failed",
]


@dataclass(frozen=True, slots=True)
class DCLRepetitionOutcome:
    state: DCLRunState
    calls: tuple[DCLCallRecord, ...]

    expected_group_ids: tuple[str, ...]
    failed_group_ids: tuple[str, ...]

    session_understanding_status: (
        SessionUnderstandingStatus
    )
    group_profiles_status: GroupProfilesStatus
    session_profile_status: SessionProfileStatus

    @property
    def status(self) -> DCLRepetitionStatus:
        if (
            self.session_understanding_status
            == "failed"
        ):
            return "failed"

        if (
            self.group_profiles_status == "completed"
            and self.session_profile_status
            == "completed"
        ):
            return "completed"

        return "partial"

    def __post_init__(self) -> None:
        expected = set(self.expected_group_ids)
        failed = set(self.failed_group_ids)

        if len(expected) != len(
            self.expected_group_ids
        ):
            raise ValueError(
                "Expected group identifiers must "
                "be unique."
            )

        if len(failed) != len(
            self.failed_group_ids
        ):
            raise ValueError(
                "Failed group identifiers must "
                "be unique."
            )

        if not failed <= expected:
            raise ValueError(
                "Failed group identifiers must be "
                "configured groups."
            )

        if any(
            call.request.run_id != self.state.run_id
            or call.request.ai_repetition
            != self.state.ai_repetition
            for call in self.calls
        ):
            raise ValueError(
                "DCL repetition calls cannot mix "
                "runs or AI repetitions."
            )

        successful = {
            artifact.request.group_id
            for artifact in self.state.group_profiles
        }

        if successful & failed:
            raise ValueError(
                "A group cannot be both successful "
                "and failed."
            )

        stage_order = {
            "session_understanding": 0,
            "stakeholder_group_profile": 1,
            "session_profile": 2,
        }

        call_stage_order = [
            stage_order[call.request.stage]
            for call in self.calls
        ]

        if call_stage_order != sorted(
            call_stage_order
        ):
            raise ValueError(
                "DCL calls must preserve stage order."
            )

        has_understanding = (
            self.state.session_understanding is not None
        )

        if (
            self.session_understanding_status
            == "completed"
        ) != has_understanding:
            raise ValueError(
                "Session-understanding status does not "
                "match the stored artifact."
            )

        if self.group_profiles_status == "blocked":
            if successful or failed:
                raise ValueError(
                    "Blocked group profiles cannot have "
                    "successful or failed groups."
                )

        elif successful | failed != expected:
            raise ValueError(
                "Attempted group profiles must cover "
                "every configured group."
            )

        elif self.group_profiles_status == "completed":
            if failed:
                raise ValueError(
                    "Completed group profiles cannot "
                    "contain failures."
                )

        elif self.group_profiles_status == "partial":
            if not successful or not failed:
                raise ValueError(
                    "Partial group profiles require both "
                    "successes and failures."
                )

        elif self.group_profiles_status == "failed":
            if successful or not failed:
                raise ValueError(
                    "Failed group profiles require every "
                    "attempted group to fail."
                )

        has_profile = (
            self.state.session_profile is not None
        )

        if self.session_profile_status == "completed":
            if not has_profile:
                raise ValueError(
                    "Completed session profile requires "
                    "an artifact."
                )

        elif self.session_profile_status == "failed":
            if has_profile:
                raise ValueError(
                    "Failed session profile cannot contain "
                    "an artifact."
                )

        elif self.session_profile_status == "blocked":
            if has_profile:
                raise ValueError(
                    "Blocked session profile cannot contain "
                    "an artifact."
                )

        if (
            self.session_understanding_status == "failed"
            and (
                self.group_profiles_status != "blocked"
                or self.session_profile_status != "blocked"
            )
        ):
            raise ValueError(
                "Failed session understanding must block "
                "all downstream DCL stages."
            )

        calls_by_stage = {
            stage: tuple(
                call
                for call in self.calls
                if call.request.stage == stage
            )
            for stage in (
                "session_understanding",
                "stakeholder_group_profile",
                "session_profile",
            )
        }

        if not calls_by_stage["session_understanding"]:
            raise ValueError(
                "A DCL repetition requires a "
                "session-understanding call."
            )

        if self.group_profiles_status == "blocked":
            if calls_by_stage["stakeholder_group_profile"]:
                raise ValueError(
                    "Blocked group profiles cannot "
                    "contain calls."
                )
        elif not calls_by_stage["stakeholder_group_profile"]:
            raise ValueError(
                "Attempted group profiles require calls."
            )

        if self.session_profile_status == "blocked":
            if calls_by_stage["session_profile"]:
                raise ValueError(
                    "A blocked session profile cannot "
                    "contain calls."
                )
        elif not calls_by_stage["session_profile"]:
            raise ValueError(
                "An attempted session profile requires calls."
            )


def _configured_group_ids(
    result: RunResult,
) -> tuple[str, ...]:
    session = result.inputs.get("session")

    if not isinstance(session, dict):
        raise ValueError(
            "Resolved session inputs are missing."
        )

    groups = session.get("stakeholder_groups")

    if not isinstance(groups, list):
        raise ValueError(
            "Resolved stakeholder groups are missing."
        )

    identifiers = []

    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(
                "Resolved stakeholder groups must "
                "be objects."
            )

        group_id = group.get("id")

        if (
            not isinstance(group_id, str)
            or not group_id
        ):
            raise ValueError(
                "Configured groups require "
                "nonempty identifiers."
            )

        identifiers.append(group_id)

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "Configured group identifiers must "
            "be unique."
        )

    return tuple(identifiers)


def run_dcl_repetition_for_result(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    ai_repetition: int,
    max_retries: int,
    diagnostics_by_group: dict[
        str,
        tuple[DiagnosticContext, ...],
    ] | None = None,
    session_diagnostics: tuple[
        DiagnosticContext, ...
    ] = (),
    session_understanding_limitations: (
        tuple[str, ...]
    ) = (),
    group_profile_limitations: (
        tuple[str, ...]
    ) = (),
    session_profile_limitations: (
        tuple[str, ...]
    ) = (),
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> DCLRepetitionOutcome:
    expected_group_ids = _configured_group_ids(
        result
    )

    understanding = (
        run_session_understanding_for_result(
            backend=backend,
            result=result,
            ai_repetition=ai_repetition,
            max_retries=max_retries,
            required_limitations=(
                session_understanding_limitations
            ),
            input_origin=input_origin,
        )
    )

    if not understanding.succeeded:
        return DCLRepetitionOutcome(
            state=understanding.state,
            calls=understanding.calls,
            expected_group_ids=expected_group_ids,
            failed_group_ids=(),
            session_understanding_status="failed",
            group_profiles_status="blocked",
            session_profile_status="blocked",
        )

    groups = run_group_profiles_for_result(
        backend=backend,
        result=result,
        ai_repetition=ai_repetition,
        max_retries=max_retries,
        diagnostics_by_group=(
            diagnostics_by_group
        ),
        required_limitations=(
            group_profile_limitations
        ),
        input_origin=input_origin,
    )

    intermediate_state = DCLRunState(
        run_id=result.run_id,
        ai_repetition=ai_repetition,
        session_understanding=(
            understanding.state
            .session_understanding
        ),
        group_profiles=(
            groups.state.group_profiles
        ),
    )

    if groups.succeeded:
        group_status = "completed"
    elif groups.state.group_profiles:
        group_status = "partial"
    else:
        group_status = "failed"

    profile = run_session_profile_for_result(
        backend=backend,
        result=result,
        state=intermediate_state,
        max_retries=max_retries,
        diagnostics=session_diagnostics,
        failed_group_ids=(
            groups.failed_group_ids
        ),
        required_limitations=(
            session_profile_limitations
        ),
        input_origin=input_origin,
    )

    return DCLRepetitionOutcome(
        state=profile.state,
        calls=(
            *understanding.calls,
            *groups.calls,
            *profile.calls,
        ),
        expected_group_ids=expected_group_ids,
        failed_group_ids=(
            groups.failed_group_ids
        ),
        session_understanding_status="completed",
        group_profiles_status=group_status,
        session_profile_status=(
            "completed"
            if profile.succeeded
            else "failed"
        ),
    )


@dataclass(frozen=True, slots=True)
class SessionUnderstandingOutcome:
    state: DCLRunState
    calls: tuple[DCLCallRecord, ...]

    @property
    def succeeded(self) -> bool:
        return (
            self.state.session_understanding
            is not None
        )

    def __post_init__(self) -> None:
        artifact = self.state.session_understanding

        if self.state.group_profiles:
            raise ValueError(
                "A session-understanding outcome must "
                "not contain group profiles."
            )

        if self.state.session_profile is not None:
            raise ValueError(
                "A session-understanding outcome must "
                "not contain a session profile."
            )

        if not self.calls:
            raise ValueError(
                "A session-understanding outcome "
                "requires at least one call."
            )

        if any(
            call.request.stage
            != "session_understanding"
            for call in self.calls
        ):
            raise ValueError(
                "Session-understanding calls must use "
                "the session-understanding stage."
            )

        if any(
            call.request.run_id != self.state.run_id
            or call.request.ai_repetition
            != self.state.ai_repetition
            for call in self.calls
        ):
            raise ValueError(
                "Session-understanding calls cannot "
                "mix runs or AI repetitions."
            )

        request_ids = {
            call.request.request_id
            for call in self.calls
        }

        if len(request_ids) != 1:
            raise ValueError(
                "Session-understanding retries must "
                "belong to one logical request."
            )

        attempts = [
            call.attempt
            for call in self.calls
        ]

        if attempts != list(
            range(1, len(self.calls) + 1)
        ):
            raise ValueError(
                "Session-understanding attempts must "
                "be consecutive and ordered."
            )

        final_call = self.calls[-1]

        if artifact is None:
            if final_call.status != "failed":
                raise ValueError(
                    "A failed session-understanding "
                    "outcome must end with a failed call."
                )

            return

        if final_call.status != "completed":
            raise ValueError(
                "A successful session-understanding "
                "outcome must end with a completed call."
            )

        if artifact is not None:
            if final_call.parsed_response != artifact.response:
                raise ValueError(
                    "The completed call and artifact must "
                    "contain the same response."
                )

            if final_call.receipt != artifact.receipt:
                raise ValueError(
                    "The completed call and artifact must "
                    "contain the same receipt."
                )

        canonical_request = self.calls[0].request

        if any(
            call.request != canonical_request
            for call in self.calls
        ):
            raise ValueError(
                "Session-understanding retries must "
                "retain identical request metadata."
            )

        if (
            artifact is not None
            and artifact.request != canonical_request
        ):
            raise ValueError(
                "The artifact and calls must belong "
                "to the same request."
            )


def run_session_understanding_for_result(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    ai_repetition: int,
    max_retries: int,
    required_limitations: tuple[str, ...] = (),
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> SessionUnderstandingOutcome:
    if ai_repetition < 1:
        raise ValueError(
            "AI repetition must be at least one."
        )

    if max_retries < 0:
        raise ValueError(
            "Retries must not be negative."
        )

    input_snapshot = (
        build_session_understanding_input(
            result=result,
            required_limitations=(
                required_limitations
            ),
        )
    )

    request = create_session_understanding_request(
        run_id=result.run_id,
        ai_repetition=ai_repetition,
        input_snapshot=input_snapshot,
        input_origin=input_origin,
    )

    stage_outcome = (
        run_session_understanding_stage(
            backend=backend,
            request=request,
            input_snapshot=input_snapshot,
            max_retries=max_retries,
        )
    )

    artifact = None

    if stage_outcome.succeeded:
        artifact = (
            build_session_understanding_artifact(
                request=request,
                input_snapshot=input_snapshot,
                outcome=stage_outcome,
            )
        )

    state = DCLRunState(
        run_id=result.run_id,
        ai_repetition=ai_repetition,
        session_understanding=artifact,
    )

    return SessionUnderstandingOutcome(
        state=state,
        calls=stage_outcome.calls,
    )


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

@dataclass(frozen=True, slots=True)
class SessionProfileOutcome:
    state: DCLRunState
    calls: tuple[DCLCallRecord, ...]

    @property
    def succeeded(self) -> bool:
        return self.state.session_profile is not None

    def __post_init__(self) -> None:
        if self.state.session_understanding is None:
            raise ValueError(
                "A session-profile outcome requires "
                "session understanding."
            )

        if not self.calls:
            raise ValueError(
                "A session-profile outcome requires "
                "at least one call."
            )

        if any(
            call.request.stage != "session_profile"
            for call in self.calls
        ):
            raise ValueError(
                "Session-profile calls must use the "
                "session-profile stage."
            )

        if any(
            call.request.run_id != self.state.run_id
            or call.request.ai_repetition
            != self.state.ai_repetition
            for call in self.calls
        ):
            raise ValueError(
                "Session-profile calls cannot mix "
                "runs or AI repetitions."
            )

        canonical_request = self.calls[0].request

        if any(
            call.request != canonical_request
            for call in self.calls
        ):
            raise ValueError(
                "Session-profile retries must retain "
                "identical request metadata."
            )

        attempts = [
            call.attempt
            for call in self.calls
        ]

        if attempts != list(
            range(1, len(self.calls) + 1)
        ):
            raise ValueError(
                "Session-profile attempts must be "
                "consecutive and ordered."
            )

        artifact = self.state.session_profile
        final_call = self.calls[-1]

        if artifact is None:
            if final_call.status != "failed":
                raise ValueError(
                    "A failed session-profile outcome "
                    "must end with a failed call."
                )

            return

        if final_call.status != "completed":
            raise ValueError(
                "A successful session-profile outcome "
                "must end with a completed call."
            )

        if artifact.request != canonical_request:
            raise ValueError(
                "The session-profile artifact and calls "
                "must use the same request."
            )

        if final_call.parsed_response != artifact.response:
            raise ValueError(
                "The completed call and artifact must "
                "contain the same response."
            )

        if final_call.receipt != artifact.receipt:
            raise ValueError(
                "The completed call and artifact must "
                "contain the same receipt."
            )

def run_session_profile_for_result(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    state: DCLRunState,
    max_retries: int,
    diagnostics: tuple[
        DiagnosticContext, ...
    ] = (),
    failed_group_ids: tuple[str, ...] = (),
    required_limitations: tuple[str, ...] = (),
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> SessionProfileOutcome:
    if max_retries < 0:
        raise ValueError(
            "Retries must not be negative."
        )

    if state.run_id != result.run_id:
        raise ValueError(
            "DCL state and numerical result must "
            "identify the same run."
        )

    if state.session_understanding is None:
        raise ValueError(
            "Session-profile execution requires "
            "successful session understanding."
        )

    if state.session_profile is not None:
        raise ValueError(
            "The supplied state already contains "
            "a session-profile artifact."
        )

    if len(failed_group_ids) != len(
        set(failed_group_ids)
    ):
        raise ValueError(
            "Failed group identifiers must be unique."
        )

    successful_group_ids = {
        artifact.request.group_id
        for artifact in state.group_profiles
    }

    if successful_group_ids & set(
        failed_group_ids
    ):
        raise ValueError(
            "A group cannot be both successful "
            "and failed."
        )

    limitations = list(required_limitations)

    if failed_group_ids:
        limitations.append(
            "No validated group-profile artifact was "
            "available for: "
            + ", ".join(failed_group_ids)
            + ". The session profile uses the supplied "
            "raw group calculations instead."
        )

    input_snapshot = build_session_profile_input(
        result=result,
        state=state,
        diagnostics=diagnostics,
        required_limitations=tuple(limitations),
    )

    request = create_session_profile_request(
        run_id=result.run_id,
        ai_repetition=state.ai_repetition,
        input_snapshot=input_snapshot,
        input_origin=input_origin,
    )

    stage_outcome = run_session_profile_stage(
        backend=backend,
        request=request,
        input_snapshot=input_snapshot,
        max_retries=max_retries,
    )

    artifact = None

    if stage_outcome.succeeded:
        artifact = build_session_profile_artifact(
            request=request,
            input_snapshot=input_snapshot,
            outcome=stage_outcome,
        )

    combined_state = DCLRunState(
        run_id=state.run_id,
        ai_repetition=state.ai_repetition,
        session_understanding=(
            state.session_understanding
        ),
        group_profiles=state.group_profiles,
        session_profile=artifact,
    )

    return SessionProfileOutcome(
        state=combined_state,
        calls=stage_outcome.calls,
    )
