from __future__ import annotations

import hashlib
import json

from src.ai.advisory.models import PolicyDecisionAdvisoryInput


def canonical_advisory_input_json(
    snapshot: PolicyDecisionAdvisoryInput,
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


def hash_advisory_input(
    snapshot: PolicyDecisionAdvisoryInput,
) -> str:
    canonical = canonical_advisory_input_json(snapshot)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
