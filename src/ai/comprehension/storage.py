from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.ai.comprehension.models import DCLCallRecord
from src.common.privacy import sanitize
from src.experiments.storage import (
    ArtifactWriteError,
    directory_component,
    write_json_atomic,
)


if TYPE_CHECKING:
    from src.ai.comprehension.orchestration import (
        GroupProfilesOutcome,
        DCLRepetitionOutcome,
    )


def _call_payload(
    call: DCLCallRecord,
) -> dict[str, object]:
    payload = call.model_dump(mode="json")

    payload.pop("raw_response", None)
    payload.pop("parsed_response", None)

    return payload

def _response_payload(
    call: DCLCallRecord,
) -> dict[str, object] | None:
    if (
        call.receipt is None
        and call.raw_response is None
        and call.parsed_response is None
    ):
        return None
    
    return {
        "schema_version": "dcl-response-v1",
        "call_id": call.call_id,
        "request_id": call.request.request_id,
        "run_id": call.request.run_id,
        "ai_repetition": call.request.ai_repetition,
        "stage": call.request.stage,
        "group_id": call.request.group_id,
        "attempt": call.attempt,
        "status": call.status,
        "receipt": (
            call.receipt.model_dump(mode="json")
            if call.receipt is not None
            else None
        ),
        "raw_response": call.raw_response,
        "parsed_response": (
            call.parsed_response.model_dump(mode="json")
            if call.parsed_response is not None
            else None
        ),
    }

def _safe_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    clean = sanitize(payload)

    if not isinstance(clean, dict):
        raise TypeError("Sanitized DCL payload must remain an object.")

    return clean

def _call_filename(
    call: DCLCallRecord,
) -> str:
    call_id = directory_component(call.call_id)

    return (
        f"a{call.attempt:03d}-"
        f"{call_id}.json"
    )

def _call_stage_directory(
    *,
    root: Path,
    call: DCLCallRecord,
) -> Path:
    stage = call.request.stage

    if stage == "session_understanding":
        return root / "session_understanding"

    if stage == "session_profile":
        return root / "session_profile"

    if stage == "stakeholder_group_profile":
        group_id = call.request.group_id

        if group_id is None:
            raise ValueError(
                "Group-profile calls require "
                "a group identifier."
            )

        return (
            root
            / "group_profiles"
            / "groups"
            / directory_component(group_id)
        )

    raise ValueError(
        f"Unsupported DCL stage: {stage!r}"
    )

def _index_payload(
    outcome: GroupProfilesOutcome,
    *,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> dict[str, object]:
    successful = [
        artifact.request.group_id
        for artifact in outcome.state.group_profiles
    ]

    failed = list(outcome.failed_group_ids)

    if not failed:
        status = "completed"
    elif successful:
        status = "partial"
    else:
        status = "failed"

    return {
        "schema_version": "dcl-group-profiles-v1",
        "stage": "stakeholder_group_profile",
        "run_id": outcome.state.run_id,
        "ai_repetition": outcome.state.ai_repetition,
        "status": status,
        "successful_group_ids": successful,
        "failed_group_ids": failed,
        "call_count": len(outcome.calls),
        "artifact_count": len(
            outcome.state.group_profiles
        ),
        "saved": {
            "calls": save_ai_calls,
            "responses": save_ai_responses,
        },
    }

def _repetition_index_payload(
    outcome: DCLRepetitionOutcome,
    *,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> dict[str, object]:
    successful_group_ids = [
        artifact.request.group_id
        for artifact
        in outcome.state.group_profiles
    ]

    call_counts = {
        stage: sum(
            call.request.stage == stage
            for call in outcome.calls
        )
        for stage in (
            "session_understanding",
            "stakeholder_group_profile",
            "session_profile",
        )
    }

    return {
        "schema_version": "dcl-repetition-v1",
        "run_id": outcome.state.run_id,
        "ai_repetition": (
            outcome.state.ai_repetition
        ),
        "status": outcome.status,
        "stages": {
            "session_understanding": (
                outcome.session_understanding_status
            ),
            "stakeholder_group_profiles": (
                outcome.group_profiles_status
            ),
            "session_profile": (
                outcome.session_profile_status
            ),
        },
        "expected_group_ids": list(
            outcome.expected_group_ids
        ),
        "successful_group_ids": (
            successful_group_ids
        ),
        "failed_group_ids": list(
            outcome.failed_group_ids
        ),
        "call_count": len(outcome.calls),
        "call_counts_by_stage": call_counts,
        "artifacts": {
            "session_understanding": (
                outcome.state.session_understanding
                is not None
            ),
            "group_profile_count": len(
                outcome.state.group_profiles
            ),
            "session_profile": (
                outcome.state.session_profile
                is not None
            ),
        },
        "saved": {
            "calls": save_ai_calls,
            "responses": save_ai_responses,
        },
    }

def _write_artifact_export(
    *,
    path: Path,
    artifact: object,
) -> None:
    original = artifact.model_dump(
        mode="json",
        round_trip=True,
    )

    clean = _safe_payload(original)

    write_json_atomic(
        path,
        {
            "schema_version": (
                "dcl-artifact-export-v1"
            ),
            "redacted": clean != original,
            "artifact": clean,
        },
    )

# Deprecated function 
# def save_group_profiles_outcome(
#     *,
#     run_directory: Path,
#     outcome: GroupProfilesOutcome,
#     save_ai_calls: bool,
#     save_ai_responses: bool,
# ) -> Path:
#     """Persist one repetition of stakeholder-group DCL outcomes.

#     Call records and raw/parsed responses are controlled independently
#     by their output flags. Successful artifacts are considered parsed AI
#     responses and are therefore controlled by save_ai_responses.

#     index.json is the authoritative completion marker and is written
#     only after all requested files have been saved.
#     """
#     if not isinstance(save_ai_calls, bool):
#         raise TypeError(
#             "save_ai_calls must be boolean."
#         )

#     if not isinstance(save_ai_responses, bool):
#         raise TypeError(
#             "save_ai_responses must be boolean."
#         )

#     run_directory = Path(run_directory)

#     root = (
#         run_directory
#         / "ai"
#         / "repetitions"
#         / f"r{outcome.state.ai_repetition:03d}"
#         / "data_comprehension"
#         / "group_profiles"
#     )

#     if root.exists():
#         raise ArtifactWriteError(f"DCL repetition artifacts already exist: {root}")

#     artifacts_by_group = {}

#     for artifact in outcome.state.group_profiles:
#         group_id = artifact.request.group_id

#         if group_id is None:
#             raise ValueError(
#                 "Group-profile artifacts require a group identifier."
#             )

#         if group_id in artifacts_by_group:
#             raise ValueError(
#                 "Group-profile artifacts must have unique group "
#                 "identifiers."
#             )

#         artifacts_by_group[group_id] = artifact

#     call_ids = [
#         call.call_id
#         for call in outcome.calls
#     ]

#     if len(call_ids) != len(set(call_ids)):
#         raise ValueError(
#             "Group-profile call identifiers must be unique."
#         )

#     # Write each provider attempt under its target group.
#     for call in outcome.calls:
#         group_id = call.request.group_id

#         if group_id is None:
#             raise ValueError("Group-profile calls require a group identifier.")

#         group_directory = (
#             root
#             / "groups"
#             / directory_component(group_id)
#         )

#         filename = _call_filename(call)

#         if save_ai_calls:
#             original = _call_payload(call)
#             clean = _safe_payload(original)

#             clean["redacted"] = bool(
#                 call.redacted
#                 or clean != original
#             )

#             write_json_atomic(
#                 group_directory
#                 / "calls"
#                 / filename,
#                 clean,
#             )

#         if save_ai_responses:
#             original = _response_payload(call)

#             # Input validation failures, connection failures, and timeouts may not have received a provider response.
#             if original is not None:
#                 clean = _safe_payload(original)

#                 clean["redacted"] = bool(
#                     call.redacted
#                     or clean != original
#                 )

#                 write_json_atomic(
#                     group_directory
#                     / "responses"
#                     / filename,
#                     clean,
#                 )

#     # Only successful groups have artifacts. Failed groups retain their call records and appear in index.json, but do not receive
#     # a fabricated artifact.
#     if save_ai_responses:
#         for group_id, artifact in artifacts_by_group.items():
#             group_directory = (
#                 root
#                 / "groups"
#                 / directory_component(group_id)
#             )

#             original = artifact.model_dump(
#                 mode="json",
#                 round_trip=True,
#             )

#             clean = _safe_payload(original)

#             artifact_export = {
#                 "schema_version": "dcl-artifact-export-v1",
#                 "redacted": clean != original,
#                 "artifact": clean,
#             }

#             write_json_atomic(
#                 group_directory / "artifact.json",
#                 artifact_export,
#             )

#     # This is the authoritative completion marker. It must remain the final write: its presence means every requested preceding file
#     # was written successfully.
#     index_payload = _index_payload(
#         outcome,
#         save_ai_calls=save_ai_calls,
#         save_ai_responses=save_ai_responses,
#     )

#     write_json_atomic(
#         root / "index.json",
#         _safe_payload(index_payload),
#     )

#     return root

def save_dcl_repetition_outcome(
    *,
    run_directory: Path,
    outcome: DCLRepetitionOutcome,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> Path:
    if not isinstance(save_ai_calls, bool):
        raise TypeError(
            "save_ai_calls must be boolean."
        )

    if not isinstance(save_ai_responses, bool):
        raise TypeError(
            "save_ai_responses must be boolean."
        )

    root = (
        Path(run_directory)
        / "ai"
        / "repetitions"
        / f"r{outcome.state.ai_repetition:03d}"
        / "data_comprehension"
    )

    if root.exists():
        raise ArtifactWriteError(
            "DCL repetition artifacts already "
            f"exist: {root}"
        )

    call_ids = [
        call.call_id
        for call in outcome.calls
    ]

    if len(call_ids) != len(set(call_ids)):
        raise ValueError(
            "DCL call identifiers must be unique."
        )

    for call in outcome.calls:
        stage_directory = (
            _call_stage_directory(
                root=root,
                call=call,
            )
        )

        filename = _call_filename(call)

        if save_ai_calls:
            original = _call_payload(call)
            clean = _safe_payload(original)

            clean["redacted"] = bool(
                call.redacted
                or clean != original
            )

            write_json_atomic(
                stage_directory
                / "calls"
                / filename,
                clean,
            )

        if save_ai_responses:
            original = _response_payload(call)

            if original is not None:
                clean = _safe_payload(original)

                clean["redacted"] = bool(
                    call.redacted
                    or clean != original
                )

                write_json_atomic(
                    stage_directory
                    / "responses"
                    / filename,
                    clean,
                )

    if save_ai_responses:
        understanding = (
            outcome.state.session_understanding
        )

        if understanding is not None:
            _write_artifact_export(
                path=(
                    root
                    / "session_understanding"
                    / "artifact.json"
                ),
                artifact=understanding,
            )

        for artifact in (
            outcome.state.group_profiles
        ):
            group_id = artifact.request.group_id

            if group_id is None:
                raise ValueError(
                    "Group-profile artifacts require "
                    "a group identifier."
                )

            _write_artifact_export(
                path=(
                    root
                    / "group_profiles"
                    / "groups"
                    / directory_component(group_id)
                    / "artifact.json"
                ),
                artifact=artifact,
            )

        profile = outcome.state.session_profile

        if profile is not None:
            _write_artifact_export(
                path=(
                    root
                    / "session_profile"
                    / "artifact.json"
                ),
                artifact=profile,
            )

    index = _repetition_index_payload(
        outcome,
        save_ai_calls=save_ai_calls,
        save_ai_responses=save_ai_responses,
    )

    write_json_atomic(
        root / "index.json",
        _safe_payload(index),
    )

    return root
