from dataclasses import dataclass
from typing import Literal

from src.ai.backend import StructuredAIBackend
from src.ai.comprehension.orchestration import (
    DCLRepetitionOutcome,
)
from src.ai.reporting.inputs import (
    build_data_explanation_reporting_input,
)
from src.ai.reporting.models import (
    DataExplanationReportingInput,
)
from src.ai.reporting.pipeline import (
    ReportingOutcome,
    build_data_explanation_report_artifact,
    run_data_explanation_reporting,
)
from src.ai.reporting.progress import (
    ReportingProgressCallback,
)
from src.ai.reporting.records import (
    DataExplanationReportArtifact,
)
from src.ai.reporting.requests import (
    create_data_explanation_reporting_request,
)
from src.experiments.results import RunResult


ReportingRepetitionStatus = Literal[
    "completed",
    "failed",
    "blocked",
]


@dataclass(frozen=True, slots=True)
class ReportingRepetitionOutcome:
    run_id: str
    ai_repetition: int

    input_snapshot: (
        DataExplanationReportingInput | None
    )
    stage_outcome: ReportingOutcome | None
    artifact: DataExplanationReportArtifact | None

    blocked_reason: str | None = None

    @property
    def status(self) -> ReportingRepetitionStatus:
        if self.blocked_reason is not None:
            return "blocked"

        if self.artifact is not None:
            return "completed"

        return "failed"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError(
                "Reporting repetition requires a run ID."
            )

        if self.ai_repetition < 1:
            raise ValueError(
                "AI repetition must be at least one."
            )

        if self.blocked_reason is not None:
            if (
                self.input_snapshot is not None
                or self.stage_outcome is not None
                or self.artifact is not None
            ):
                raise ValueError(
                    "A blocked reporting repetition cannot "
                    "contain input, calls, or an artifact."
                )

            return

        if (
            self.input_snapshot is None
            or self.stage_outcome is None
        ):
            raise ValueError(
                "An attempted reporting repetition requires "
                "its input and stage outcome."
            )

        if self.input_snapshot.run_id != self.run_id:
            raise ValueError(
                "Reporting input and repetition must "
                "identify the same run."
            )

        if (
            self.input_snapshot.ai_repetition
            != self.ai_repetition
        ):
            raise ValueError(
                "Reporting input and repetition must "
                "identify the same AI repetition."
            )

        if self.stage_outcome.succeeded:
            if self.artifact is None:
                raise ValueError(
                    "Successful reporting requires an artifact."
                )

            final_call = self.stage_outcome.calls[-1]

            if (
                self.artifact.source_call_id
                != final_call.call_id
            ):
                raise ValueError(
                    "Reporting artifact must originate from "
                    "the completed final call."
                )

            if self.artifact.request != final_call.request:
                raise ValueError(
                    "Reporting artifact and final call must "
                    "use the same request."
                )

        elif self.artifact is not None:
            raise ValueError(
                "Failed reporting cannot contain an artifact."
            )


def run_reporting_for_dcl_repetition(
    *,
    backend: StructuredAIBackend,
    result: RunResult,
    dcl_outcome: DCLRepetitionOutcome,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_wait_seconds: float,
    on_progress: ReportingProgressCallback | None = None,
) -> ReportingRepetitionOutcome:
    if dcl_outcome.state.run_id != result.run_id:
        raise ValueError(
            "DCL outcome and numerical result must identify "
            "the same run."
        )

    ai_repetition = dcl_outcome.state.ai_repetition

    if dcl_outcome.status != "completed":
        return ReportingRepetitionOutcome(
            run_id=result.run_id,
            ai_repetition=ai_repetition,
            input_snapshot=None,
            stage_outcome=None,
            artifact=None,
            blocked_reason=(
                "Data explanation reporting requires a "
                "completed DCL repetition."
            ),
        )

    input_snapshot = (
        build_data_explanation_reporting_input(
            result=result,
            state=dcl_outcome.state,
        )
    )

    request = (
        create_data_explanation_reporting_request(
            input_snapshot=input_snapshot,
        )
    )

    stage_outcome = run_data_explanation_reporting(
        backend=backend,
        request=request,
        input_snapshot=input_snapshot,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_wait_seconds=retry_max_wait_seconds,
        on_progress=on_progress,
    )

    artifact = None

    if stage_outcome.succeeded:
        artifact = (
            build_data_explanation_report_artifact(
                request=request,
                input_snapshot=input_snapshot,
                outcome=stage_outcome,
            )
        )

    return ReportingRepetitionOutcome(
        run_id=result.run_id,
        ai_repetition=ai_repetition,
        input_snapshot=input_snapshot,
        stage_outcome=stage_outcome,
        artifact=artifact,
    )