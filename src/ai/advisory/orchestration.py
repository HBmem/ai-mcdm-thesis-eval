from dataclasses import dataclass
from typing import Literal

from src.ai.advisory.inputs import build_policy_decision_advisory_input
from src.ai.advisory.pipeline import (
    AdvisoryOutcome,
    build_policy_decision_advisory_artifact,
    run_policy_decision_advisory,
)
from src.ai.advisory.progress import AdvisoryProgressCallback
from src.ai.advisory.records import PolicyDecisionAdvisoryArtifact
from src.ai.advisory.requests import create_policy_decision_advisory_request
from src.ai.backend import StructuredAIBackend
from src.ai.reporting.orchestration import ReportingRepetitionOutcome
from src.experiments.results import RunResult


AdvisoryRepetitionStatus = Literal[
    "completed",
    "failed",
    "blocked",
]


@dataclass(frozen=True, slots=True)
class AdvisoryRepetitionOutcome:
    run_id: str
    ai_repetition: int
    stage_outcome: AdvisoryOutcome | None
    artifact: PolicyDecisionAdvisoryArtifact | None
    blocked_reason: str | None = None

    @property
    def status(self) -> AdvisoryRepetitionStatus:
        if self.blocked_reason is not None:
            return "blocked"

        if self.stage_outcome is not None and self.stage_outcome.succeeded:
            return "completed"

        return "failed"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Advisory repetition requires a run ID.")

        if self.ai_repetition < 1:
            raise ValueError("AI repetition must be at least one.")

        if self.blocked_reason is not None:
            if not self.blocked_reason.strip():
                raise ValueError("A blocked advisory repetition requires a reason.")

            if self.stage_outcome is not None or self.artifact is not None:
                raise ValueError(
                    "A blocked advisory repetition cannot contain "
                    "calls or an artifact."
                )
            return

        if self.stage_outcome is None:
            raise ValueError(
                "An attempted advisory repetition requires a stage outcome."
            )

        final_call = self.stage_outcome.calls[-1]
        request = final_call.request

        if (self.run_id, self.ai_repetition) != (
            request.run_id,
            request.ai_repetition,
        ):
            raise ValueError(
                "Advisory repetition and call must identify "
                "the same run and repetition."
            )

        if self.stage_outcome.succeeded:
            if self.artifact is None:
                raise ValueError("Successful advisory requires an artifact.")

            if final_call.status != "completed":
                raise ValueError(
                    "A completed advisory outcome must end with "
                    "a completed call."
                )

            if self.artifact.source_call_id != final_call.call_id:
                raise ValueError(
                    "Advisory artifact must originate from "
                    "the completed final call."
                )

            if self.artifact.request != final_call.request:
                raise ValueError(
                    "Advisory artifact and final call must use "
                    "the same request."
                )

            if self.artifact.input_snapshot != final_call.input_snapshot:
                raise ValueError(
                    "Advisory artifact and final call must use "
                    "the same input snapshot."
                )

            if self.artifact.response != final_call.parsed_response:
                raise ValueError(
                    "Advisory artifact and final call must contain "
                    "the same response."
                )

            if self.artifact.receipt != final_call.receipt:
                raise ValueError(
                    "Advisory artifact and final call must contain "
                    "the same receipt."
                )

        elif self.artifact is not None:
            raise ValueError("Failed advisory cannot contain an artifact.")


def run_advisory_for_reporting_repetition(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    reporting_outcome: ReportingRepetitionOutcome,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_wait_seconds: float,
    on_progress: AdvisoryProgressCallback | None = None,
) -> AdvisoryRepetitionOutcome:
    if reporting_outcome.run_id != result.run_id:
        raise ValueError(
            "Reporting repetition and numerical result must identify "
            "the same run."
        )

    def require_matching_ids(
        source: str,
        source_run_id: str,
        source_repetition: int,
    ) -> None:
        if source_run_id != reporting_outcome.run_id:
            raise ValueError(
                f"{source} and reporting repetition must identify "
                "the same run."
            )

        if source_repetition != reporting_outcome.ai_repetition:
            raise ValueError(
                f"{source} and reporting repetition must identify "
                "the same AI repetition."
            )

    if reporting_outcome.input_snapshot is not None:
        require_matching_ids(
            "Reporting input",
            reporting_outcome.input_snapshot.run_id,
            reporting_outcome.input_snapshot.ai_repetition,
        )

    if reporting_outcome.artifact is not None:
        artifact = reporting_outcome.artifact
        require_matching_ids(
            "Reporting artifact input",
            artifact.input_snapshot.run_id,
            artifact.input_snapshot.ai_repetition,
        )
        require_matching_ids(
            "Reporting artifact request",
            artifact.request.run_id,
            artifact.request.ai_repetition,
        )

    if reporting_outcome.stage_outcome is not None:
        for call in reporting_outcome.stage_outcome.calls:
            require_matching_ids(
                "Reporting call input",
                call.input_snapshot.run_id,
                call.input_snapshot.ai_repetition,
            )
            require_matching_ids(
                "Reporting call request",
                call.request.run_id,
                call.request.ai_repetition,
            )

    if reporting_outcome.status != "completed":
        return AdvisoryRepetitionOutcome(
            run_id=result.run_id,
            ai_repetition=reporting_outcome.ai_repetition,
            stage_outcome=None,
            artifact=None,
            blocked_reason=(
                "Advisory requires a completed reporting repetition."
            ),
        )

    report_artifact = reporting_outcome.artifact
    if report_artifact is None:
        raise ValueError(
            "A completed reporting repetition must contain an artifact."
        )

    try:
        input_snapshot = build_policy_decision_advisory_input(
            result=result,
            artifact=report_artifact,
        )
    except ValueError as error:
        return AdvisoryRepetitionOutcome(
            run_id=result.run_id,
            ai_repetition=reporting_outcome.ai_repetition,
            stage_outcome=None,
            artifact=None,
            blocked_reason=f"Reporting artifact is ineligible for advisory: {error}",
        )

    request = create_policy_decision_advisory_request(
        snapshot=input_snapshot,
    )

    stage_outcome = run_policy_decision_advisory(
        backend=backend,
        request=request,
        input_snapshot=input_snapshot,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_wait_seconds=retry_max_wait_seconds,
        on_progress=on_progress,
    )

    artifact = (
        build_policy_decision_advisory_artifact(outcome=stage_outcome)
        if stage_outcome.succeeded
        else None
    )

    return AdvisoryRepetitionOutcome(
        run_id=result.run_id,
        ai_repetition=reporting_outcome.ai_repetition,
        stage_outcome=stage_outcome,
        artifact=artifact,
    )
