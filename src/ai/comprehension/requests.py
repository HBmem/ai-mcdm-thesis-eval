from __future__ import annotations

from typing import Literal
from uuid import uuid4

from src.ai.comprehension.models import (
    DCLRequestMetadata,
    GroupProfileInput,
    SessionProfileInput,
    SessionUnderstandingInput,
)
from src.ai.comprehension.prompts import (
    SESSION_UNDERSTANDING_PROMPT_VERSION,
    GROUP_PROFILE_PROMPT_VERSION,
    SESSION_PROFILE_PROMPT_VERSION,
)
from src.ai.comprehension.serialization import (
    hash_input_snapshot,
)


def create_session_understanding_request(
    *,
    run_id: str,
    ai_repetition: int,
    input_snapshot: SessionUnderstandingInput,
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> DCLRequestMetadata:
    return DCLRequestMetadata(
        request_id=uuid4().hex,
        run_id=run_id,
        ai_repetition=ai_repetition,
        stage="session_understanding",

        # Session-level requests must not claim a group.
        group_id=None,

        prompt_version=(
            SESSION_UNDERSTANDING_PROMPT_VERSION
        ),
        input_hash=hash_input_snapshot(
            input_snapshot
        ),
        input_origin=input_origin,
    )

def create_group_profile_request(
    *,
    run_id: str,
    ai_repetition: int,
    input_snapshot: GroupProfileInput,
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> DCLRequestMetadata:
    """Create metadata for one logical group-profile request."""

    return DCLRequestMetadata(
        request_id=uuid4().hex,
        run_id=run_id,
        ai_repetition=ai_repetition,
        stage="stakeholder_group_profile",
        group_id=input_snapshot.group.id,
        prompt_version=GROUP_PROFILE_PROMPT_VERSION,
        input_hash=hash_input_snapshot(input_snapshot),
        input_origin=input_origin,
    )


def create_session_profile_request(
    *,
    run_id: str,
    ai_repetition: int,
    input_snapshot: SessionProfileInput,
    input_origin: Literal[
        "calculated",
        "reported",
    ] = "calculated",
) -> DCLRequestMetadata:
    return DCLRequestMetadata(
        request_id=uuid4().hex,
        run_id=run_id,
        ai_repetition=ai_repetition,
        stage="session_profile",
        group_id=None,
        prompt_version=(
            SESSION_PROFILE_PROMPT_VERSION
        ),
        input_hash=hash_input_snapshot(
            input_snapshot
        ),
        input_origin=input_origin,
    )
