from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class AdvisoryCallProgress:
    stage: Literal[
        "policy_decision_advisory"
    ]
    attempt: int
    max_attempts: int

    state: Literal[
        "request_started",
        "retry_wait",
    ]

    delay_seconds: float | None = None
    code: str | None = None
    limit_source: str | None = None


AdvisoryProgressCallback: TypeAlias = Callable[
    [AdvisoryCallProgress],
    None,
]
