from __future__ import annotations

import hashlib
import json

from src.ai.comprehension.models import DCLStageInput


def canonical_input_json(
    snapshot: DCLStageInput,
) -> str:
    """Return the canonical representation used for fingerprinting."""

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


def hash_input_snapshot(
    snapshot: DCLStageInput,
) -> str:
    canonical = canonical_input_json(snapshot)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def render_input_json(
    snapshot: DCLStageInput,
) -> str:
    payload = snapshot.model_dump(
        mode="json",
        round_trip=True,
    )

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
