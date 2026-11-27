from __future__ import annotations

from pathlib import Path

from src.ai.advisory.orchestration import AdvisoryRepetitionOutcome
from src.ai.advisory.records import (
    AdvisoryCallRecord,
    PolicyDecisionAdvisoryArtifact,
)
from src.common.privacy import sanitize
from src.experiments.storage import (
    ArtifactWriteError,
    directory_component,
    write_json_atomic,
)


def _safe_payload(payload: dict[str, object]) -> dict[str, object]:
    clean = sanitize(payload)

    if not isinstance(clean, dict):
        raise TypeError("Sanitized advisory payload must remain an object.")

    return clean


def _call_payload(call: AdvisoryCallRecord) -> dict[str, object]:
    payload = call.model_dump(mode="json", round_trip=True)
    payload.pop("raw_response", None)
    payload.pop("parsed_response", None)
    return payload


def _response_payload(call: AdvisoryCallRecord) -> dict[str, object] | None:
    if (
        call.receipt is None
        and call.raw_response is None
        and call.parsed_response is None
    ):
        return None

    return {
        "schema_version": "advisory-response-export-v1",
        "call_id": call.call_id,
        "request_id": call.request.request_id,
        "run_id": call.request.run_id,
        "ai_repetition": call.request.ai_repetition,
        "stage": call.request.stage,
        "attempt": call.attempt,
        "status": call.status,
        "receipt": (
            call.receipt.model_dump(mode="json")
            if call.receipt is not None
            else None
        ),
        "raw_response": call.raw_response,
        "parsed_response": (
            call.parsed_response.model_dump(
                mode="json",
                round_trip=True,
            )
            if call.parsed_response is not None
            else None
        ),
    }


def _call_filename(call: AdvisoryCallRecord) -> str:
    call_id = directory_component(call.call_id)
    return f"a{call.attempt:03d}-{call_id}.json"


def _artifact_payload(
    artifact: PolicyDecisionAdvisoryArtifact,
) -> dict[str, object]:
    original = artifact.model_dump(mode="json", round_trip=True)
    clean = _safe_payload(original)

    return {
        "schema_version": "advisory-artifact-export-v1",
        "redacted": clean != original,
        "artifact": clean,
    }


def _index_payload(
    *,
    outcome: AdvisoryRepetitionOutcome,
    artifact: PolicyDecisionAdvisoryArtifact | None,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> dict[str, object]:
    if outcome.stage_outcome is None:
        raise ValueError("An attempted advisory outcome requires call records.")

    request = outcome.stage_outcome.calls[0].request

    return {
        "schema_version": "advisory-repetition-v1",
        "run_id": outcome.run_id,
        "ai_repetition": outcome.ai_repetition,
        "stage": request.stage,
        "status": outcome.status,
        "request_id": request.request_id,
        "source_report_artifact_id": request.source_report_artifact_id,
        "call_count": len(outcome.stage_outcome.calls),
        "artifact_produced": artifact is not None,
        "artifact_saved": artifact is not None and save_ai_responses,
        "saved": {
            "calls": save_ai_calls,
            "responses": save_ai_responses,
        },
    }


def save_policy_decision_advisory_outcome(
    *,
    run_directory: Path,
    outcome: AdvisoryRepetitionOutcome,
    save_ai_calls: bool,
    save_ai_responses: bool,
) -> Path:
    if not isinstance(save_ai_calls, bool):
        raise TypeError("save_ai_calls must be boolean.")

    if not isinstance(save_ai_responses, bool):
        raise TypeError("save_ai_responses must be boolean.")

    if outcome.status == "blocked" or outcome.stage_outcome is None:
        raise ValueError(
            "Only attempted advisory outcomes can be saved."
        )

    stage_outcome = outcome.stage_outcome
    artifact = outcome.artifact

    if stage_outcome.succeeded and artifact is None:
        raise ValueError("A successful advisory outcome requires an artifact.")

    if not stage_outcome.succeeded and artifact is not None:
        raise ValueError("A failed advisory outcome cannot have an artifact.")

    final_call = stage_outcome.calls[-1]

    if artifact is not None:
        if artifact.source_call_id != final_call.call_id:
            raise ValueError(
                "Artifact source call does not match the completed outcome call."
            )

        if artifact.request != final_call.request:
            raise ValueError("Artifact request does not match the advisory outcome.")

        if artifact.input_snapshot != final_call.input_snapshot:
            raise ValueError("Artifact input does not match the final call.")

        if artifact.response != final_call.parsed_response:
            raise ValueError("Artifact response does not match the final call.")

        if artifact.receipt != final_call.receipt:
            raise ValueError("Artifact receipt does not match the final call.")

    calls = stage_outcome.calls
    call_ids = [call.call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("Advisory call identifiers must be unique.")

    request = calls[0].request
    root = (
        Path(run_directory)
        / "ai"
        / "repetitions"
        / f"r{outcome.ai_repetition:03d}"
        / "policy_decision_advisory"
    )

    if root.exists():
        raise ArtifactWriteError(
            f"Advisory repetition artifacts already exist: {root}"
        )

    for call in calls:
        filename = _call_filename(call)

        if save_ai_calls:
            original = _call_payload(call)
            clean = _safe_payload(original)

            # Keep request.input_hash unchanged: it fingerprints the original
            # provider input, even when the exported snapshot is redacted.
            clean["redacted"] = bool(call.redacted or clean != original)

            write_json_atomic(root / "calls" / filename, clean)

        if save_ai_responses:
            original_response = _response_payload(call)

            if original_response is not None:
                clean_response = _safe_payload(original_response)
                clean_response["redacted"] = bool(
                    call.redacted or clean_response != original_response
                )
                write_json_atomic(
                    root / "responses" / filename,
                    clean_response,
                )

    if save_ai_responses and artifact is not None:
        write_json_atomic(
            root / "artifact.json",
            _artifact_payload(artifact),
        )

    index = _index_payload(
        outcome=outcome,
        artifact=artifact,
        save_ai_calls=save_ai_calls,
        save_ai_responses=save_ai_responses,
    )
    write_json_atomic(root / "index.json", _safe_payload(index))

    return root
