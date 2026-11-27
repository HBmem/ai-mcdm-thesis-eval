from src.ai.advisory.models import (
    PolicyDecisionAdvisoryInput,
    PolicyDecisionAdvisoryResponse,
)
from src.ai.comprehension.context import EvidenceResolutionError
from src.ai.comprehension.models import ValidationIssue
from src.ai.reporting.context import resolve_reporting_evidence


def validate_policy_decision_advisory_response(
    *,
    advisory_input: PolicyDecisionAdvisoryInput,
    response: PolicyDecisionAdvisoryResponse,
) -> tuple[ValidationIssue, ...]:
    snapshot = advisory_input.evidence_snapshot
    offered = {ref.identity() for ref in snapshot.available_evidence}
    diagnostics = {item.diagnostic_id: item for item in snapshot.diagnostics}
    issues: list[ValidationIssue] = []

    def check_citations(evidence, path: tuple[str | int, ...]) -> None:
        for index, ref in enumerate(evidence):
            ref_path = (*path, index)
            if ref.identity() not in offered:
                issues.append(ValidationIssue(
                    code="unoffered_advisory_evidence",
                    message="Evidence reference was not offered.",
                    response_path=ref_path,
                ))
                continue

            try:
                resolve_reporting_evidence(snapshot, ref)
            except EvidenceResolutionError:
                issues.append(ValidationIssue(
                    code="unresolvable_advisory_evidence",
                    message="Evidence reference could not be resolved.",
                    response_path=ref_path,
                ))

    for index, improvement in enumerate(response.improvements):
        check_citations(improvement.evidence, ("improvements", index, "evidence"))

    for index, alternative in enumerate(response.conditional_alternatives):
        base_path = ("conditional_alternatives", index)
        diagnostic_path = (*base_path, "diagnostic_id")
        check_citations(alternative.evidence, (*base_path, "evidence"))

        if alternative.status != "evaluated":
            continue

        diagnostic = diagnostics.get(alternative.diagnostic_id)
        if diagnostic is None:
            issues.append(ValidationIssue(
                code="advisory_diagnostic_not_found",
                message="Diagnostic was not found in the evidence snapshot.",
                response_path=diagnostic_path,
            ))
            continue

        if not any(
            ref.object_type == "diagnostic"
            and ref.object_id == alternative.diagnostic_id
            and ref.field_path == ("result",)
            for ref in alternative.evidence
        ):
            issues.append(ValidationIssue(
                code="advisory_result_citation_missing",
                message="Evaluated configuration must cite the diagnostic result.",
                response_path=(*base_path, "evidence"),
            ))

        guards_passed = True
        if diagnostic.status != "completed":
            issues.append(ValidationIssue(
                code="advisory_diagnostic_not_completed",
                message="Selected diagnostic is not completed.",
                response_path=diagnostic_path,
            ))
            guards_passed = False

        if diagnostic.method != "one_at_a_time_ranking_criterion_removal_v1":
            issues.append(ValidationIssue(
                code="unsupported_advisory_diagnostic_method",
                message="Diagnostic method is not supported for evaluated configurations.",
                response_path=diagnostic_path,
            ))
            guards_passed = False

        if alternative.entity_id not in diagnostic.entity_ids:
            issues.append(ValidationIssue(
                code="advisory_entity_not_in_diagnostic",
                message="Selected entity is not part of the diagnostic.",
                response_path=(*base_path, "entity_id"),
            ))
            guards_passed = False

        if not guards_passed:
            continue

        result = diagnostic.result
        if not isinstance(result, dict) or not isinstance(result.get("cases"), list):
            issues.append(ValidationIssue(
                code="invalid_advisory_diagnostic_result",
                message="Diagnostic result does not contain a valid cases list.",
                response_path=diagnostic_path,
            ))
            continue

        cases = result["cases"]
        if not all(isinstance(case, dict) for case in cases):
            issues.append(ValidationIssue(
                code="invalid_advisory_diagnostic_result",
                message="Diagnostic result contains an invalid case.",
                response_path=diagnostic_path,
            ))
            continue

        matching_cases = [
            case for case in cases
            if case.get("removed_criterion_id") == alternative.removed_criterion_id
        ]
        criterion_path = (*base_path, "removed_criterion_id")

        if not matching_cases:
            issues.append(ValidationIssue(
                code="advisory_case_not_found",
                message="Selected criterion-removal case was not found.",
                response_path=criterion_path,
            ))
            continue

        if len(matching_cases) != 1:
            issues.append(ValidationIssue(
                code="advisory_case_not_unique",
                message="Selected criterion-removal case is not unique.",
                response_path=criterion_path,
            ))
            continue

        case = matching_cases[0]
        if case.get("status") != "completed":
            issues.append(ValidationIssue(
                code="advisory_case_not_completed",
                message="Selected criterion-removal case is not completed.",
                response_path=criterion_path,
            ))
            continue

        winner_ids = case.get("winner_ids")
        if not isinstance(winner_ids, list) or not all(
            isinstance(winner_id, str) for winner_id in winner_ids
        ):
            issues.append(ValidationIssue(
                code="invalid_advisory_case_winners",
                message="Selected diagnostic case has invalid winner IDs.",
                response_path=criterion_path,
            ))
            continue

        observed = alternative.observed_winner_ids
        if (
            len(observed) != len(set(observed))
            or set(observed) != set(winner_ids)
        ):
            issues.append(ValidationIssue(
                code="advisory_winners_mismatch",
                message="Observed winner IDs do not match the selected diagnostic case.",
                response_path=(*base_path, "observed_winner_ids"),
            ))

    return tuple(issues)