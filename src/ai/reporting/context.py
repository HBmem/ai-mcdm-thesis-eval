from src.ai.comprehension.context import (
    EvidenceResolutionError,
    resolve_session_profile_evidence,
)
from src.ai.comprehension.models import (
    EvidenceReference,
    SessionProfileInput,
    ValidationIssue,
)
from src.ai.reporting.models import (
    DataExplanationReportingInput,
)


def _as_resolution_snapshot(
    reporting_input: DataExplanationReportingInput,
) -> SessionProfileInput:
    """Adapt reporting input to the existing evidence resolver."""

    return SessionProfileInput(
        scenario=reporting_input.scenario,
        criteria=reporting_input.criteria,
        alternatives=reporting_input.alternatives,
        processing=reporting_input.processing,
        available_evidence=(
            reporting_input.available_evidence
        ),
        required_limitations=(
            reporting_input.required_disclosures
        ),
        prior_interpretations=(),
        groups=reporting_input.groups,
        calculation=(
            reporting_input.session_calculation
        ),
        diagnostics=reporting_input.diagnostics,
    )


def resolve_reporting_evidence(
    reporting_input: DataExplanationReportingInput,
    reference: EvidenceReference,
) -> object:
    snapshot = _as_resolution_snapshot(
        reporting_input
    )

    return resolve_session_profile_evidence(
        snapshot,
        reference,
    )


def validate_reporting_available_evidence(
    reporting_input: DataExplanationReportingInput,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    snapshot = _as_resolution_snapshot(
        reporting_input
    )

    for index, reference in enumerate(
        reporting_input.available_evidence
    ):
        try:
            resolve_session_profile_evidence(
                snapshot,
                reference,
            )
        except EvidenceResolutionError:
            issues.append(
                ValidationIssue(
                    code=(
                        "unresolvable_reporting_evidence"
                    ),
                    message=(
                        "Reporting evidence does not "
                        "resolve against its input snapshot."
                    ),
                    response_path=(
                        "available_evidence",
                        index,
                    ),
                )
            )

    return tuple(issues)
