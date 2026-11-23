from uuid import uuid4

from src.ai.reporting.models import (
    DataExplanationReportingInput,
)
from src.ai.reporting.prompts import (
    DATA_EXPLANATION_REPORTING_PROMPT_VERSION,
)
from src.ai.reporting.records import (
    ReportingRequestMetadata,
)
from src.ai.reporting.serialization import (
    hash_reporting_input,
)


def create_data_explanation_reporting_request(
    *,
    input_snapshot: DataExplanationReportingInput,
) -> ReportingRequestMetadata:
    return ReportingRequestMetadata(
        request_id=uuid4().hex,
        run_id=input_snapshot.run_id,
        ai_repetition=input_snapshot.ai_repetition,
        stage="data_explanation_reporting",
        prompt_version=(
            DATA_EXPLANATION_REPORTING_PROMPT_VERSION
        ),
        input_hash=hash_reporting_input(
            input_snapshot
        ),
        input_origin=input_snapshot.input_origin,
    )
