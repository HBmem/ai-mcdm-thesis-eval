from __future__ import annotations

import hashlib
import json

from src.ai.reporting.models import (
    DataExplanationReportingInput,
)


def canonical_reporting_input_json(
    snapshot: DataExplanationReportingInput,
) -> str:
    payload = snapshot.model_dump(
        mode="json",
        round_trip=True,
    )

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_reporting_input(
    snapshot: DataExplanationReportingInput,
) -> str:
    canonical = canonical_reporting_input_json(
        snapshot
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def render_reporting_input_json(
    snapshot: DataExplanationReportingInput,
) -> str:
    """Render the compact snapshot sent to the provider."""

    return canonical_reporting_input_json(snapshot)
