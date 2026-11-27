from uuid import uuid4

from src.ai.advisory.models import PolicyDecisionAdvisoryInput
from src.ai.advisory.prompts import POLICY_DECISION_ADVISORY_PROMPT_VERSION
from src.ai.advisory.records import AdvisoryRequestMetadata
from src.ai.advisory.serialization import hash_advisory_input


def create_policy_decision_advisory_request(
    *,
    snapshot: PolicyDecisionAdvisoryInput,
) -> AdvisoryRequestMetadata:
    source = snapshot.evidence_snapshot

    return AdvisoryRequestMetadata(
        request_id=uuid4().hex,
        run_id=source.run_id,
        ai_repetition=source.ai_repetition,
        source_report_artifact_id=snapshot.source_report_artifact_id,
        prompt_version=POLICY_DECISION_ADVISORY_PROMPT_VERSION,
        input_hash=hash_advisory_input(snapshot),
    )
