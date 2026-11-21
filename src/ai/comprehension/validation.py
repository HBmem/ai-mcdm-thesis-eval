from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from src.ai.models.analysis import Record
from src.ai.comprehension.context import (
    EvidenceResolutionError,
    resolve_evidence,
)
from src.ai.comprehension.models import (
    DCLStageInput,
    DCLStageResponse,
    EvidenceReference,
    InterpretationDraft,
    StoredInterpretation,
    ValidationIssue,
)


def _walk_interpretations(
    value: object,
    path: tuple[str | int, ...],
):
    if isinstance(value, InterpretationDraft):
        yield FindingCandidate(
            response_path=path,
            draft=value,
        )
        return

    if isinstance(value, Record):
        for field_name in type(value).model_fields:
            field_value = getattr(value, field_name)

            yield from _walk_interpretations(
                field_value,
                (*path, field_name),
            )

        return

    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk_interpretations(
                item,
                (*path, index),
            )


def make_finding_id(
    request_id: str,
    response_path: tuple[str | int, ...],
) -> str:
    encoded_path = "/".join(
        str(part)
        for part in response_path
    )

    return uuid5(
        NAMESPACE_URL,
        f"dcl:{request_id}:{encoded_path}",
    ).hex


def materialize_validated_findings(
    *,
    request_id: str,
    validation: ReferenceValidationResult,
) -> tuple[StoredInterpretation, ...]:
    if not validation.valid:
        raise ValueError(
            "Cannot materialize findings with invalid references."
        )

    return tuple(
        StoredInterpretation(
            finding_id=make_finding_id(
                request_id,
                candidate.response_path,
            ),
            response_path=candidate.response_path,
            draft=candidate.draft,
            reference_status="validated",
        )
        for candidate in validation.findings
    )


@dataclass(frozen=True, slots=True)
class FindingCandidate:
    """One interpretation and its location in a stage response."""

    response_path: tuple[str | int, ...]
    draft: InterpretationDraft


@dataclass(frozen=True, slots=True)
class ReferenceValidationResult:
    """Runtime result of response-reference validation."""

    findings: tuple[FindingCandidate, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_available_evidence(
    snapshot: DCLStageInput,
) -> tuple[ValidationIssue, ...]:
    """Validate the application-created evidence allowlist."""
    issues: list[ValidationIssue] = []

    for index, reference in enumerate(
        snapshot.available_evidence
    ):
        try:
            resolve_evidence(snapshot, reference)
        except EvidenceResolutionError:
            issues.append(
                ValidationIssue(
                    code="unresolvable_available_evidence",
                    message=("An evidence reference does not resolve against the input snapshot."),
                    response_path=(
                        "available_evidence",
                        index,
                    ),
                )
            )

    return tuple(issues)


def validate_response_evidence(
    snapshot: DCLStageInput,
    response: DCLStageResponse,
) -> ReferenceValidationResult:
    """Validate model citations against the allowlist and snapshot."""
    findings = extract_findings(response)
    issues: list[ValidationIssue] = []

    available = {
        reference.identity()
        for reference in snapshot.available_evidence
    }

    for finding in findings:
        for index, reference in enumerate(
            finding.draft.evidence
        ):
            citation_path = (
                *finding.response_path,
                "evidence",
                index,
            )

            if reference.identity() not in available:
                issues.append(
                    ValidationIssue(
                        code="evidence_not_offered",
                        message=("The response cited evidence that was not supplied in the allowlist."),
                        response_path=citation_path,
                    )
                )
                continue

            try:
                resolve_evidence(snapshot, reference)
            except EvidenceResolutionError:
                issues.append(
                    ValidationIssue(
                        code="unresolvable_response_evidence",
                        message=("The cited evidence does not resolve against the input snapshot."),
                        response_path=citation_path,
                    )
                )

    return ReferenceValidationResult(
        findings=findings,
        issues=tuple(issues),
    )


def extract_findings(
    response: DCLStageResponse,
) -> tuple[FindingCandidate, ...]:
    """Traverse a concrete response and locate every interpretation."""
    return tuple(
        _walk_interpretations(
            response,
            (),
        )
    )