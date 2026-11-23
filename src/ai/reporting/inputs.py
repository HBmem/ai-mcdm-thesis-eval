from __future__ import annotations

from src.ai.comprehension.inputs import (
    require_completed_numerical_result,
)
from src.ai.comprehension.models import (
    DCLRunState,
    DiagnosticContext,
    EvidenceReference,
    GroupProfileArtifact,
    GroupProfileInput,
    SessionProfileArtifact,
    SessionProfileInput,
    SessionUnderstandingArtifact,
)
from src.ai.reporting.models import (
    DataExplanationReportingInput,
    ReportingFindingContext,
)
from src.experiments.results import RunResult


ReportingSourceArtifact = (
    SessionUnderstandingArtifact
    | GroupProfileArtifact
    | SessionProfileArtifact
)


def _ordered_artifacts(
    state: DCLRunState,
) -> tuple[ReportingSourceArtifact, ...]:
    if state.session_understanding is None:
        raise ValueError(
            "Reporting input requires completed "
            "session understanding."
        )

    if state.session_profile is None:
        raise ValueError(
            "Reporting input requires a completed "
            "session profile."
        )

    profile_snapshot = (
        state.session_profile.input_snapshot
    )

    expected_group_ids = tuple(
        group.group.id
        for group in profile_snapshot.groups
    )

    actual_group_ids = tuple(
        artifact.request.group_id
        for artifact in state.group_profiles
    )

    if actual_group_ids != expected_group_ids:
        raise ValueError(
            "Reporting requires one group-profile "
            "artifact for every configured group, "
            "in configured order."
        )

    return (
        state.session_understanding,
        *state.group_profiles,
        state.session_profile,
    )


def build_reporting_evidence(
    *,
    state: DCLRunState,
) -> tuple[EvidenceReference, ...]:
    references: list[EvidenceReference] = []
    seen: set[tuple[str, str, tuple[str | int, ...]]] = set()

    for artifact in _ordered_artifacts(state):
        for reference in (
            artifact
            .input_snapshot
            .available_evidence
        ):
            identity = reference.identity()

            if identity in seen:
                continue

            seen.add(identity)
            references.append(reference)

    if not references:
        raise ValueError(
            "Reporting input requires raw evidence."
        )

    return tuple(references)


def select_reporting_findings(
    *,
    state: DCLRunState,
    available_evidence: tuple[
        EvidenceReference, ...
    ],
) -> tuple[ReportingFindingContext, ...]:
    available = {
        reference.identity()
        for reference in available_evidence
    }

    selected: list[
        ReportingFindingContext
    ] = []
    finding_ids: set[str] = set()

    for artifact in _ordered_artifacts(state):
        if (
            artifact.content_review.status
            == "rejected"
        ):
            continue

        for finding in artifact.findings:
            if finding.reference_status != "validated":
                continue

            if (
                finding.content_review.status
                == "rejected"
            ):
                continue

            if finding.finding_id in finding_ids:
                raise ValueError(
                    "DCL finding identifiers must "
                    "be unique across reporting input."
                )

            cited = {
                reference.identity()
                for reference
                in finding.draft.evidence
            }

            if not cited <= available:
                raise ValueError(
                    "A selected DCL finding cites "
                    "evidence unavailable to reporting."
                )

            selected.append(
                ReportingFindingContext(
                    finding_id=finding.finding_id,
                    stage=artifact.request.stage,
                    group_id=(
                        artifact.request.group_id
                    ),
                    draft=finding.draft,
                    reference_status="validated",
                    content_review=(
                        finding.content_review
                    ),
                )
            )

            finding_ids.add(finding.finding_id)

    if not selected:
        raise ValueError(
            "Reporting input requires at least one "
            "eligible DCL finding."
        )

    return tuple(selected)


def _collect_diagnostics(
    state: DCLRunState,
) -> tuple[DiagnosticContext, ...]:
    diagnostics: list[DiagnosticContext] = []
    by_id: dict[str, DiagnosticContext] = {}

    for artifact in _ordered_artifacts(state):
        snapshot = artifact.input_snapshot

        if isinstance(
            snapshot,
            (GroupProfileInput, SessionProfileInput),
        ):
            candidates = snapshot.diagnostics
        else:
            candidates = ()

        for diagnostic in candidates:
            existing = by_id.get(
                diagnostic.diagnostic_id
            )

            if existing is not None:
                if existing != diagnostic:
                    raise ValueError(
                        "DCL snapshots contain conflicting "
                        "diagnostics with the same ID."
                    )

                continue

            by_id[diagnostic.diagnostic_id] = (
                diagnostic
            )
            diagnostics.append(diagnostic)

    return tuple(diagnostics)


def _collect_required_disclosures(
    state: DCLRunState,
) -> tuple[str, ...]:
    disclosures: list[str] = []
    seen: set[str] = set()

    for artifact in _ordered_artifacts(state):
        for disclosure in (
            artifact.required_limitations
        ):
            if disclosure in seen:
                continue

            seen.add(disclosure)
            disclosures.append(disclosure)

    return tuple(disclosures)


def build_data_explanation_reporting_input(
    *,
    result: RunResult,
    state: DCLRunState,
) -> DataExplanationReportingInput:
    require_completed_numerical_result(result)

    if state.run_id != result.run_id:
        raise ValueError(
            "Reporting state and numerical result "
            "must identify the same run."
        )

    artifacts = _ordered_artifacts(state)

    input_origins = {
        artifact.request.input_origin
        for artifact in artifacts
    }

    if len(input_origins) != 1:
        raise ValueError(
            "Reporting cannot mix calculated and "
            "reported DCL evidence."
        )

    profile_artifact = state.session_profile

    if profile_artifact is None:
        raise RuntimeError(
            "Validated reporting state lost its "
            "session-profile artifact."
        )

    snapshot = profile_artifact.input_snapshot

    available_evidence = (
        build_reporting_evidence(
            state=state,
        )
    )

    findings = select_reporting_findings(
        state=state,
        available_evidence=available_evidence,
    )

    return DataExplanationReportingInput(
        run_id=result.run_id,
        ai_repetition=state.ai_repetition,
        input_origin=artifacts[0].request.input_origin,
        scenario=snapshot.scenario,
        criteria=snapshot.criteria,
        alternatives=snapshot.alternatives,
        groups=snapshot.groups,
        processing=snapshot.processing,
        session_calculation=(
            snapshot.calculation
        ),
        diagnostics=_collect_diagnostics(state),
        comprehension_findings=findings,
        available_evidence=available_evidence,
        required_disclosures=(
            _collect_required_disclosures(state)
        ),
    )
