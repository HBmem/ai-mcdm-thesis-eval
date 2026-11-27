from src.ai.advisory.models import PolicyDecisionAdvisoryInput
from src.ai.reporting.records import DataExplanationReportArtifact
from src.experiments.results import RunResult


def build_policy_decision_advisory_input(
    *,
    result: RunResult,
    artifact: DataExplanationReportArtifact,
) -> PolicyDecisionAdvisoryInput:
    if result.status != "passed":
        raise ValueError("Advisory input requires a passed numerical run.")
    
    snapshot = artifact.input_snapshot

    if result.run_id != snapshot.run_id:
        raise ValueError("Run ID does not match the report artifact.")
    if snapshot.input_origin != "calculated":
        raise ValueError("Advisory input requires calculated evidence.")
    if artifact.content_review.status == "rejected":
        raise ValueError("Advisory input requires a non-rejected report.")

    return PolicyDecisionAdvisoryInput(
        source_report_artifact_id=artifact.artifact_id,
        evidence_snapshot=snapshot,
        explanatory_report=artifact.report,
        report_content_review=artifact.content_review,
    )
