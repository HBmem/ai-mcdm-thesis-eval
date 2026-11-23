from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias


DCLStage = Literal[
    "session_understanding",
    "stakeholder_group_profile",
    "session_profile",
]


@dataclass(frozen=True, slots=True)
class DCLCallProgress:
    stage: DCLStage
    group_id: str | None
    attempt: int
    max_attempts: int
    state: Literal[
        "request_started",
        "retry_wait",
    ]
    delay_seconds: float | None = None
    code: str | None = None
    limit_source: str | None = None


DCLProgressCallback: TypeAlias = Callable[
    [DCLCallProgress],
    None,
]
