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

def save_group_profiles_outcome(
    *,
    run_directory: Path,
    outcome: GroupProfilesOutcome,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> Path:
    """Persist one repetition of stakeholder-group DCL outcomes.

    Call records and raw/parsed responses are controlled independently
    by their output flags. Successful artifacts are considered parsed AI
    responses and are therefore controlled by save_ai_responses.

    index.json is the authoritative completion marker and is written
    only after all requested files have been saved.
    """
    if not isinstance(save_ai_calls, bool):
        raise TypeError(
            "save_ai_calls must be boolean."
        )

    if not isinstance(save_ai_responses, bool):
        raise TypeError(
            "save_ai_responses must be boolean."
        )

    run_directory = Path(run_directory)

    root = (
        run_directory
        / "ai"
        / "repetitions"
        / f"r{outcome.state.ai_repetition:03d}"
        / "data_comprehension"
        / "group_profiles"
    )

    if root.exists():
        raise ArtifactWriteError(f"DCL repetition artifacts already exist: {root}")

    artifacts_by_group = {}

    for artifact in outcome.state.group_profiles:
        group_id = artifact.request.group_id

        if group_id is None:
            raise ValueError(
                "Group-profile artifacts require a group identifier."
            )

        if group_id in artifacts_by_group:
            raise ValueError(
                "Group-profile artifacts must have unique group "
                "identifiers."
            )

        artifacts_by_group[group_id] = artifact

    call_ids = [
        call.call_id
        for call in outcome.calls
    ]

    if len(call_ids) != len(set(call_ids)):
        raise ValueError(
            "Group-profile call identifiers must be unique."
        )

    # Write each provider attempt under its target group.
    for call in outcome.calls:
        group_id = call.request.group_id

        if group_id is None:
            raise ValueError("Group-profile calls require a group identifier.")

        group_directory = (
            root
            / "groups"
            / directory_component(group_id)
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
                group_directory
                / "calls"
                / filename,
                clean,
            )

        if save_ai_responses:
            original = _response_payload(call)

            # Input validation failures, connection failures, and timeouts may not have received a provider response.
            if original is not None:
                clean = _safe_payload(original)

                clean["redacted"] = bool(
                    call.redacted
                    or clean != original
                )

                write_json_atomic(
                    group_directory
                    / "responses"
                    / filename,
                    clean,
                )

    # Only successful groups have artifacts. Failed groups retain their call records and appear in index.json, but do not receive
    # a fabricated artifact.
    if save_ai_responses:
        for group_id, artifact in artifacts_by_group.items():
            group_directory = (
                root
                / "groups"
                / directory_component(group_id)
            )

            original = artifact.model_dump(
                mode="json",
                round_trip=True,
            )

            clean = _safe_payload(original)

            artifact_export = {
                "schema_version": "dcl-artifact-export-v1",
                "redacted": clean != original,
                "artifact": clean,
            }

            write_json_atomic(
                group_directory / "artifact.json",
                artifact_export,
            )

    # This is the authoritative completion marker. It must remain the final write: its presence means every requested preceding file
    # was written successfully.
    index_payload = _index_payload(
        outcome,
        save_ai_calls=save_ai_calls,
        save_ai_responses=save_ai_responses,
    )

    write_json_atomic(
        root / "index.json",
        _safe_payload(index_payload),
    )

    return root