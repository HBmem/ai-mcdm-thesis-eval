from dataclasses import dataclass
from typing import Literal

from src.ai.comprehension.context import (
    EvidenceResolutionError,
)
from src.ai.comprehension.models import (
    DiagnosticKind,
    ValidationIssue,
)
from src.ai.reporting.context import (
    resolve_reporting_evidence,
)
from src.ai.reporting.models import (
    DataExplanationReport,
    DataExplanationReportingInput,
    ReportClaim,
    ReportSection,
)


ReportSectionName = Literal[
    "executive_summary",
    "decision_context",
    "aggregate_results",
    "stakeholder_patterns",
    "methodology",
    "limitations",
    "agreement",
    "influence",
    "robustness",
]
DiagnosticSectionName = Literal[
    "agreement",
    "influence",
    "robustness",
]


_OPTIONAL_SECTION_KINDS: dict[
    DiagnosticSectionName,
    frozenset[DiagnosticKind],
] = {
    "agreement": frozenset({"agreement"}),
    "influence": frozenset({"influence"}),
    "robustness": frozenset({
        "robustness",
        "sensitivity",
    }),
}


def _completed_diagnostic_ids(
    reporting_input: DataExplanationReportingInput,
    section_name: DiagnosticSectionName,
) -> set[str]:
    allowed_kinds = _OPTIONAL_SECTION_KINDS[
        section_name
    ]

    return {
        diagnostic.diagnostic_id
        for diagnostic in reporting_input.diagnostics
        if (
            diagnostic.status == "completed"
            and diagnostic.kind in allowed_kinds
        )
    }


@dataclass(frozen=True, slots=True)
class ReportClaimCandidate:
    section: ReportSectionName
    claim_index: int
    claim: ReportClaim

    @property
    def response_path(
        self,
    ) -> tuple[str | int, ...]:
        return (
            self.section,
            "claims",
            self.claim_index,
        )


@dataclass(frozen=True, slots=True)
class ReportValidationResult:
    claims: tuple[ReportClaimCandidate, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def extract_report_claims(
    report: DataExplanationReport,
) -> tuple[ReportClaimCandidate, ...]:
    section_names: tuple[ReportSectionName, ...] = (
        "executive_summary",
        "decision_context",
        "aggregate_results",
        "stakeholder_patterns",
        "methodology",
        "limitations",
        "agreement",
        "influence",
        "robustness",
    )

    candidates: list[ReportClaimCandidate] = []

    for section_name in section_names:
        section: ReportSection | None = getattr(
            report,
            section_name,
        )

        if section is None:
            continue

        for index, claim in enumerate(section.claims):
            candidates.append(
                ReportClaimCandidate(
                    section=section_name,
                    claim_index=index,
                    claim=claim,
                )
            )

    return tuple(candidates)


def validate_report_response(
    *,
    reporting_input: DataExplanationReportingInput,
    report: DataExplanationReport,
) -> ReportValidationResult:
    claims = extract_report_claims(report)
    issues: list[ValidationIssue] = []

    available_evidence = {
        reference.identity()
        for reference
        in reporting_input.available_evidence
    }

    known_finding_ids = {
        finding.finding_id
        for finding
        in reporting_input.comprehension_findings
    }

    for candidate in claims:
        for evidence_index, reference in enumerate(
            candidate.claim.evidence
        ):
            evidence_path = (
                *candidate.response_path,
                "evidence",
                evidence_index,
            )

            if (
                reference.identity()
                not in available_evidence
            ):
                issues.append(
                    ValidationIssue(
                        code="report_evidence_not_offered",
                        message=(
                            "The report claim cited evidence "
                            "that was not supplied."
                        ),
                        response_path=evidence_path,
                    )
                )
                continue

            try:
                resolve_reporting_evidence(
                    reporting_input,
                    reference,
                )
            except EvidenceResolutionError:
                issues.append(
                    ValidationIssue(
                        code=(
                            "unresolvable_report_evidence"
                        ),
                        message=(
                            "The report claim cited evidence "
                            "that does not resolve against "
                            "the reporting input."
                        ),
                        response_path=evidence_path,
                    )
                )

        for finding_index, finding_id in enumerate(
            candidate.claim.source_finding_ids
        ):
            if finding_id not in known_finding_ids:
                issues.append(
                    ValidationIssue(
                        code="unknown_source_finding",
                        message=(
                            "The report claim references an "
                            "unknown DCL finding."
                        ),
                        response_path=(
                            *candidate.response_path,
                            "source_finding_ids",
                            finding_index,
                        ),
                    )
                )

    claims_by_section = {
        section_name: tuple(
            candidate
            for candidate in claims
            if candidate.section == section_name
        )
        for section_name in _OPTIONAL_SECTION_KINDS
    }

    for section_name in _OPTIONAL_SECTION_KINDS:
        section = getattr(report, section_name)

        diagnostic_ids = _completed_diagnostic_ids(
            reporting_input,
            section_name,
        )

        if section is None:
            if diagnostic_ids:
                issues.append(
                    ValidationIssue(
                        code="missing_diagnostic_section",
                        message=(
                            "A completed diagnostic is "
                            "available, but its report section "
                            "was omitted."
                        ),
                        response_path=(section_name,),
                    )
                )

            continue

        if not diagnostic_ids:
            issues.append(
                ValidationIssue(
                    code="unsupported_diagnostic_section",
                    message=(
                        "The report includes an analytical "
                        "section without an applicable "
                        "completed diagnostic."
                    ),
                    response_path=(section_name,),
                )
            )
            continue

        for candidate in claims_by_section[
            section_name
        ]:
            cited_diagnostic_ids = {
                reference.object_id
                for reference in candidate.claim.evidence
                if reference.object_type == "diagnostic"
            }

            if cited_diagnostic_ids.isdisjoint(
                diagnostic_ids
            ):
                issues.append(
                    ValidationIssue(
                        code=(
                            "diagnostic_claim_without_"
                            "diagnostic_evidence"
                        ),
                        message=(
                            "A diagnostic-section claim must "
                            "cite an applicable completed "
                            "diagnostic."
                        ),
                        response_path=(
                            *candidate.response_path,
                            "evidence",
                        ),
                    )
                )

    return ReportValidationResult(
        claims=claims,
        issues=tuple(issues),
    )
